"""kernel_thr_warp: GEMV with explicit 2-level reduction (thread → warp).

Same operation as kernel_warp (scores = q @ K.T), but with the reduction
hierarchy made explicit using CuteDSL APIs — mirroring quack / v5v2 style.

Reduction hierarchy:
  Level 1 — Thread (register tile) : each lane issues NUM_VEC-wide vectorized
                                      loads via zipped_divide + .load(), stores
                                      partial dot products in make_rmem_tensor
  Level 2 — Warp (shuffle)         : warp_reduce(scalar, lambda a,b: a+b)
  Level 3 — Block / Cluster        : not needed

Vectorized load:
  NUM_VEC=4 → LDG.64 (4×BF16 = 8 bytes) per load
  ITERS_PER_LANE = K_PER_LANE // NUM_VEC = 16 // 4 = 4 iterations per warp round
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math

D              : cutlass.Constexpr = 512
N              : cutlass.Constexpr = 64
BLOCK_SIZE     : cutlass.Constexpr = 1024
NUM_WARPS      : cutlass.Constexpr = BLOCK_SIZE // 32   # 32
K_PER_LANE     : cutlass.Constexpr = D // 32            # 16
NUM_VEC        : cutlass.Constexpr = 8                  # BF16 elements per vectorized load (LDG.64)
ITERS_PER_LANE : cutlass.Constexpr = K_PER_LANE // NUM_VEC  # 4


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def kernel_thr_warp_fn(
    q:      cute.Tensor,   # [D]    bf16
    K:      cute.Tensor,   # [N, D] bf16
    scores: cute.Tensor,   # [N]    fp32
    stream,
):
    kernel_thr_warp_kernel(q, K, scores).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_thr_warp_kernel(
    q:      cute.Tensor,   # [D]    bf16
    K:      cute.Tensor,   # [N, D] bf16
    scores: cute.Tensor,   # [N]    fp32
):
    d              : cutlass.Constexpr = D
    n              : cutlass.Constexpr = N
    num_warps      : cutlass.Constexpr = NUM_WARPS
    num_vec        : cutlass.Constexpr = NUM_VEC
    iters_per_lane : cutlass.Constexpr = ITERS_PER_LANE
    num_threads    : cutlass.Constexpr = BLOCK_SIZE

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx   = cute.arch.warp_idx()
    warp_idx   = cute.arch.make_warp_uniform(warp_idx)
    lane_idx   = cute.arch.lane_idx()
    wsize      = cute.arch.WARP_SIZE   # 32

    # ── Smem allocation ───────────────────────────────────────────────────
    allocator   = cutlass.utils.SmemAllocator()
    smem_q      = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((d,), stride=(1,)), 16, None)
    smem_scores = allocator.allocate_tensor(
        cutlass.Float32,  cute.make_layout((n,), stride=(1,)), 16, None)

    # ── Load q → smem ─────────────────────────────────────────────────────
    for i in range(tidx, d, num_threads):
        smem_q[i] = q[i]
    cute.arch.sync_threads()

    # ── Score phase ───────────────────────────────────────────────────────
    num_rounds: cutlass.Constexpr = (n + num_warps - 1) // num_warps  # 64

    # zipped_divide smem_q once outside the loop: ((NUM_VEC,), (D//NUM_VEC,))
    q_z = cute.zipped_divide(smem_q, (num_vec,))

    for round_idx in range(num_rounds):
        token_idx = round_idx * num_warps + warp_idx

        # Get row K[token_idx, :] as a 1D (D,) view, then tile it
        K_row = K[token_idx, None]                    # (D,) BF16 row
        K_z   = cute.zipped_divide(K_row, (num_vec,)) # ((NUM_VEC,), (D//NUM_VEC,))

        # ── Level 1: thread reduction — TensorSSA multiply + reduce ──────────
        sum_partial = cutlass.Float32(0)
        for it in range(iters_per_lane):
            group  = it * wsize + lane_idx            # tile group index [0, 64)
            q_frag = q_z[(None, (group,))].load()  # (NUM_VEC,) F32 TensorSSA
            K_frag = K_z[(None, (group,))].load()  # (NUM_VEC,) F32 TensorSSA
            sumSSA = q_frag * K_frag                  # element-wise → TensorSSA
            partial = cutlass.Float32(
                sumSSA.reduce(cute.ReductionOp.ADD, init_val=float(0), reduction_profile=0)
            )
            sum_partial = sum_partial + partial

        # ── Level 2: warp reduction ─────────────────────────────────────────
        s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_scores[token_idx] = s

    cute.arch.sync_threads()

    # ── Writeback smem_scores → global ────────────────────────────────────
    for i in range(tidx, n, num_threads):
        scores[i] = smem_scores[i]


# ── Compilation ────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_thr_warp():
    q      = _fake(cute.BFloat16, (D,),    (0,),    16)
    K      = _fake(cute.BFloat16, (N, D),  (1, 0),  16)
    scores = _fake(cute.Float32,  (N,),    (0,),    16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_thr_warp_fn,
        q, K, scores, stream,
        options="--enable-tvm-ffi",
    )


kernel_thr_warp_compiled = compile_kernel_thr_warp()


def run(q, K, scores):
    """q: [D] bf16, K: [N,D] bf16, scores: [N] fp32 (pre-allocated)."""
    kernel_thr_warp_compiled(q, K, scores)

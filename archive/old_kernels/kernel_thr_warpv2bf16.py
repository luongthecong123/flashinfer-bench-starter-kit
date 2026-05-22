"""kernel_thr_warpv2: GEMV with interleaved multi-row batching (FastGEMV-inspired).

Based on kernel_thr_warp, but batches 4 K-row dot products per warp per round.
Within each inner iteration, loads from 4 different K rows are interleaved,
giving the memory pipeline 4× more outstanding requests for better latency
hiding. The q vector fragment is loaded once and reused across all 4 rows.
Reductions are batched at the end of each round.

Key changes from v1:
  - ROWS_PER_WARP=4 → 4 interleaved K row loads per inner iteration
  - 16 outer rounds (was 64), 4× fewer loop iterations
  - q_frag shared across 4 rows → 4× fewer smem_q reads
  - Reductions batched after all loads → better instruction scheduling

Ref: https://github.com/wangsiping97/FastGEMV/blob/main/method_and_result.md
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math

D               : cutlass.Constexpr = 512
N               : cutlass.Constexpr = 2048
BLOCK_SIZE      : cutlass.Constexpr = 1024
NUM_WARPS       : cutlass.Constexpr = BLOCK_SIZE // 32   # 32
ROWS_PER_WARP   : cutlass.Constexpr = 4                  # rows batched per warp per round
NUM_VEC         : cutlass.Constexpr = 8                   # BF16 elements per vectorized load
K_PER_LANE      : cutlass.Constexpr = D // 32             # 16 elements per lane (full warp)
ITERS_PER_LANE  : cutlass.Constexpr = K_PER_LANE // NUM_VEC  # 2
ROWS_PER_ROUND  : cutlass.Constexpr = NUM_WARPS * ROWS_PER_WARP  # 128


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def kernel_thr_warpv2_fn(
    q:      cute.Tensor,   # [D]    bf16
    K:      cute.Tensor,   # [N, D] bf16
    scores: cute.Tensor,   # [N]    fp32
    stream,
):
    kernel_thr_warpv2_kernel(q, K, scores).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_thr_warpv2_kernel(
    q:      cute.Tensor,   # [D]    bf16
    K:      cute.Tensor,   # [N, D] bf16
    scores: cute.Tensor,   # [N]    fp32
):
    d               : cutlass.Constexpr = D
    n               : cutlass.Constexpr = N
    num_warps       : cutlass.Constexpr = NUM_WARPS
    num_vec         : cutlass.Constexpr = NUM_VEC
    rows_per_warp   : cutlass.Constexpr = ROWS_PER_WARP
    iters_per_lane  : cutlass.Constexpr = ITERS_PER_LANE
    rows_per_round  : cutlass.Constexpr = ROWS_PER_ROUND
    num_threads     : cutlass.Constexpr = BLOCK_SIZE

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
    num_rounds: cutlass.Constexpr = (n + rows_per_round - 1) // rows_per_round  # 16

    # zipped_divide smem_q once outside the loop: ((NUM_VEC,), (D//NUM_VEC,))
    q_z = cute.zipped_divide(smem_q, (num_vec,))

    for round_idx in range(num_rounds):
        # Base token for this round + warp
        base_token = round_idx * rows_per_round + warp_idx * rows_per_warp

        # Prepare zipped_divide views for 4 K rows (uniform across warp)
        K_z0 = cute.zipped_divide(K[base_token + 0, None], (num_vec,))
        K_z1 = cute.zipped_divide(K[base_token + 1, None], (num_vec,))
        K_z2 = cute.zipped_divide(K[base_token + 2, None], (num_vec,))
        K_z3 = cute.zipped_divide(K[base_token + 3, None], (num_vec,))

        # Partial sums for 4 rows
        sp0 = cutlass.Float32(0)
        sp1 = cutlass.Float32(0)
        sp2 = cutlass.Float32(0)
        sp3 = cutlass.Float32(0)

        # ── Level 1: interleaved loads from 4 rows ──────────────────────
        for it in range(iters_per_lane):
            group = it * wsize + lane_idx    # tile group index [0, 64)
            q_frag = q_z[(None, (group,))].load()     # shared across 4 rows

            K_frag0 = K_z0[(None, (group,))].load()
            K_frag1 = K_z1[(None, (group,))].load()
            K_frag2 = K_z2[(None, (group,))].load()
            K_frag3 = K_z3[(None, (group,))].load()

            sp0 = sp0 + cutlass.Float32(
                (q_frag * K_frag0).reduce(cute.ReductionOp.ADD, init_val=float(0), reduction_profile=0))
            sp1 = sp1 + cutlass.Float32(
                (q_frag * K_frag1).reduce(cute.ReductionOp.ADD, init_val=float(0), reduction_profile=0))
            sp2 = sp2 + cutlass.Float32(
                (q_frag * K_frag2).reduce(cute.ReductionOp.ADD, init_val=float(0), reduction_profile=0))
            sp3 = sp3 + cutlass.Float32(
                (q_frag * K_frag3).reduce(cute.ReductionOp.ADD, init_val=float(0), reduction_profile=0))

        # ── Level 2: batched warp reduction ─────────────────────────────
        s0 = warp_reduce(sp0, lambda a, b: a + b, width=32)
        s1 = warp_reduce(sp1, lambda a, b: a + b, width=32)
        s2 = warp_reduce(sp2, lambda a, b: a + b, width=32)
        s3 = warp_reduce(sp3, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_scores[base_token + 0] = s0
            smem_scores[base_token + 1] = s1
            smem_scores[base_token + 2] = s2
            smem_scores[base_token + 3] = s3

    cute.arch.sync_threads()

    # ── Writeback smem_scores → global ────────────────────────────────────
    for i in range(tidx, n, num_threads):
        scores[i] = smem_scores[i]


# ── Compilation ────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_thr_warpv2():
    q      = _fake(cute.BFloat16, (D,),    (0,),    16)
    K      = _fake(cute.BFloat16, (N, D),  (1, 0),  16)
    scores = _fake(cute.Float32,  (N,),    (0,),    16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_thr_warpv2_fn,
        q, K, scores, stream,
        options="--enable-tvm-ffi",
    )


kernel_thr_warpv2_compiled = compile_kernel_thr_warpv2()


def run(q, K, scores):
    """q: [D] bf16, K: [N,D] bf16, scores: [N] fp32 (pre-allocated)."""
    kernel_thr_warpv2_compiled(q, K, scores)

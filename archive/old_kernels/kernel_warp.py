"""kernel_warp: GEMV baseline mirroring the v5v2 score phase.

Operation:   scores = q @ K.T
  q       [D]       fp32 (query vector, pre-loaded to smem)
  K       [N, D]    bf16 (key matrix, read from GMEM)
  scores  [N]       fp32 (output)

  D = 512, N = 2048   (matches fused_tiny5v2 head_dim_ckv / top_k_len)

Reduction hierarchy (exactly as in v5v2 score phase):
  Level 1 — Register : each lane accumulates D/WSIZE = 16 products
  Level 2 — Warp     : warp_reduce (butterfly shuffle) → scalar dot per warp
  Level 3 — Block    : not needed (each warp owns one output row exclusively)
  Level 4 — Cluster  : not needed (single-block kernel)

Mapping:
  Block:   1024 threads = 32 warps
  Each warp computes dot(q[D], K[token, D]) for one token per round.
  num_rounds = ceil(N / num_warps) = ceil(2048/32) = 64
  Lane lane_idx handles dims: lane_idx, lane_idx+32, ..., lane_idx+480

  Per warp per round:
    sum_partial = sum_{k=0}^{D//WSIZE - 1} q[k*32 + lane] * K[token, k*32 + lane]
    s = warp_reduce_sum(sum_partial)
    if lane == 0: scores[token] = s
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math

D         : cutlass.Constexpr = 512
N         : cutlass.Constexpr = 64
BLOCK_SIZE: cutlass.Constexpr = 1024
NUM_WARPS : cutlass.Constexpr = BLOCK_SIZE // 32   # 32
K_PER_LANE: cutlass.Constexpr = D // 32            # 16  (512 / warp_size)


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def kernel_warp_fn(
    q:      cute.Tensor,   # [D]    bf16
    K:      cute.Tensor,   # [N, D] bf16
    scores: cute.Tensor,   # [N]    fp32
    stream,
):
    kernel_warp_kernel(q, K, scores).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_warp_kernel(
    q:      cute.Tensor,   # [D]    bf16
    K:      cute.Tensor,   # [N, D] bf16
    scores: cute.Tensor,   # [N]    fp32
):
    d         : cutlass.Constexpr = D
    n         : cutlass.Constexpr = N
    num_warps : cutlass.Constexpr = NUM_WARPS
    k_per_lane: cutlass.Constexpr = K_PER_LANE
    num_threads: cutlass.Constexpr = BLOCK_SIZE

    tidx, _, _  = cute.arch.thread_idx()
    warp_idx    = cute.arch.warp_idx()
    warp_idx    = cute.arch.make_warp_uniform(warp_idx)
    lane_idx    = cute.arch.lane_idx()
    wsize       = cute.arch.WARP_SIZE   # 32

    # ── Load q into smem (all 1024 threads collaborate) ───────────────────
    allocator  = cutlass.utils.SmemAllocator()
    smem_q     = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((d,), stride=(1,)), 16, None
    )
    smem_scores = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None
    )
    for i in range(tidx, d, num_threads):
        smem_q[i] = q[i]
    cute.arch.sync_threads()

    # ── Score phase: 32 warps × num_rounds ────────────────────────────────
    # Round robin: warp w handles token (round * num_warps + w) each round.
    # num_rounds = ceil(N / num_warps) = 64
    num_rounds: cutlass.Constexpr = (n + num_warps - 1) // num_warps  # 64

    for round_idx in range(num_rounds):
        token_idx = round_idx * num_warps + warp_idx   # which K row this warp scores
        # No bounds guard: N=2048, num_warps=32, num_rounds=64 → max token_idx=2047
        sum_partial = cutlass.Float32(0)
        for k in range(k_per_lane):
            dim         = k * wsize + lane_idx          # 0..511 strided by warp
            q_n         = cutlass.Float32(smem_q[dim])  # bf16→fp32, matches v5v2
            kv          = cutlass.Float32(K[token_idx, dim])
            sum_partial = sum_partial + q_n * kv

        # Warp reduce: sum across 32 lanes → scalar on lane 0
        s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_scores[token_idx] = s   # store to smem like v5v2

    cute.arch.sync_threads()

    # ── Writeback smem_scores → global scores ────────────────────────────
    for i in range(tidx, n, num_threads):
        scores[i] = smem_scores[i]


# ── Compilation ────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_warp():
    q      = _fake(cute.BFloat16, (D,),    (0,),    16)
    K      = _fake(cute.BFloat16, (N, D),  (1, 0),  16)
    scores = _fake(cute.Float32,  (N,),    (0,),    16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_warp_fn,
        q, K, scores, stream,
        options="--enable-tvm-ffi",
    )


kernel_warp_compiled = compile_kernel_warp()


def run(q, K, scores):
    """q: [D] bf16, K: [N,D] bf16, scores: [N] fp32 (pre-allocated)."""
    kernel_warp_compiled(q, K, scores)

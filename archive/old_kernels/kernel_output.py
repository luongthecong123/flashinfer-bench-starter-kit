"""kernel_output: standalone output-phase GEMV baseline.

Operation: output = scores @ V
  scores : [N]       fp32  (attention weights, already softmax'd)
  V      : [N, D]    fp32  (value cache)
  output : [D]       fp32

Exact pattern from fused_tiny5v2 output phase:
  - Register accumulator tile: each lane holds DIMS_PER_LANE fp32 regs
  - num_rounds = N // num_warps rounds; warp w handles token (round*num_warps + w)
  - out_regs[k] += weight * V[j, k*wsize + lane_idx]
  - Write partials to smem_partial[32, 512], cross-warp reduce → smem_output
  - Epilogue: smem_output → global output (fp32)

Grid: [1, 1, 1]   Block: [1024, 1, 1]
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math

N              : cutlass.Constexpr = 64
D              : cutlass.Constexpr = 512
BLOCK_SIZE     : cutlass.Constexpr = 1024
NUM_WARPS      : cutlass.Constexpr = BLOCK_SIZE // 32   # 32
DIMS_PER_LANE  : cutlass.Constexpr = D // 32            # 16
NUM_ROUNDS     : cutlass.Constexpr = N // NUM_WARPS     # 64


@cute.jit
def kernel_output_fn(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
    stream,
):
    kernel_output_kernel(scores, V, output).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_output_kernel(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
):
    n              : cutlass.Constexpr = N
    d              : cutlass.Constexpr = D
    num_warps      : cutlass.Constexpr = NUM_WARPS
    dims_per_lane  : cutlass.Constexpr = DIMS_PER_LANE
    num_rounds     : cutlass.Constexpr = NUM_ROUNDS
    num_threads    : cutlass.Constexpr = BLOCK_SIZE
    wsize          = cute.arch.WARP_SIZE   # 32

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx   = cute.arch.warp_idx()
    warp_idx   = cute.arch.make_warp_uniform(warp_idx)
    lane_idx   = cute.arch.lane_idx()

    # ── Smem allocation ───────────────────────────────────────────────────
    allocator      = cutlass.utils.SmemAllocator()
    smem_scores    = allocator.allocate_tensor(
        cutlass.Float32,  cute.make_layout((n,), stride=(1,)), 16, None)
    smem_partial   = allocator.allocate_tensor(
        cutlass.Float32,  cute.make_layout((num_warps, d), stride=(d, 1)), 16, None)
    smem_output    = allocator.allocate_tensor(
        cutlass.Float32,  cute.make_layout((d,), stride=(1,)), 16, None)

    # ── Load scores → smem ────────────────────────────────────────────────
    for i in range(tidx, n, num_threads):
        smem_scores[i] = scores[i]
    cute.arch.sync_threads()

    # ── Output phase: per-warp register accumulation ──────────────────────
    # Lane lane_idx of warp warp_idx owns output dims: k*wsize + lane_idx
    out_regs = cute.make_rmem_tensor(
        cute.make_layout((dims_per_lane,), stride=(1,)),
        cutlass.Float32,
    )
    for k in range(dims_per_lane):
        out_regs[k] = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        j      = round_idx * num_warps + warp_idx   # token index [0, N)
        weight = smem_scores[j]
        for k in range(dims_per_lane):
            out_regs[k] += weight * V[j, k * wsize + lane_idx]

    # Write per-warp partial sums to smem_partial[warp_idx, :]
    for k in range(dims_per_lane):
        smem_partial[warp_idx, k * wsize + lane_idx] = out_regs[k]

    cute.arch.sync_threads()

    # Cross-warp reduce: each of 1024 threads sums over 32 warps for its dim
    for i in range(tidx, d, num_threads):
        acc = cutlass.Float32(0)
        for w in range(num_warps):
            acc += smem_partial[w, i]
        smem_output[i] = acc

    cute.arch.sync_threads()

    # ── Epilogue: smem_output → global output (fp32) ───────────────────────
    for i in range(tidx, d, num_threads):
        output[i] = smem_output[i]


# ── Compilation ────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_output():
    scores = _fake(cute.Float32,  (N,),    (0,),    16)
    V      = _fake(cute.Float32,  (N, D),  (1, 0),  16)
    output = _fake(cute.Float32,  (D,),    (0,),    16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_output_fn,
        scores, V, output, stream,
        options="--enable-tvm-ffi",
    )


kernel_output_compiled = compile_kernel_output()


def run(scores, V, output):
    """scores: [N] fp32, V: [N,D] bf16, output: [D] bf16 (pre-allocated)."""
    kernel_output_compiled(scores, V, output)

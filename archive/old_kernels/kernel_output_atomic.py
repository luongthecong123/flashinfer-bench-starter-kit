"""kernel_output_atomic: output-phase GEMV with atomicAdd epilogue.

Same accumulation as baseline (kernel_output.py), but replaces the
smem_partial + cross-warp reduce + smem_output epilogue with a direct
atomicAdd from registers to global memory.

Benefits:
  - Eliminates 64KB smem_partial + 2KB smem_output
  - Eliminates 2 sync_threads barriers
  - Eliminates 32×LDS serial FADD reduction chain per output dim
  
Cost:
  - 32 warps contend on 512 output addresses
  - Each dim gets 32 atomicAdds, staggered by warp scheduling

Grid: [1, 1, 1]   Block: [1024, 1, 1]
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

N              : cutlass.Constexpr = 2048
D              : cutlass.Constexpr = 512
BLOCK_SIZE     : cutlass.Constexpr = 1024
NUM_WARPS      : cutlass.Constexpr = BLOCK_SIZE // 32   # 32
DIMS_PER_LANE  : cutlass.Constexpr = D // 32            # 16
NUM_ROUNDS     : cutlass.Constexpr = N // NUM_WARPS     # 64


@cute.jit
def kernel_output_atomic_fn(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
    stream,
):
    kernel_output_atomic_kernel(scores, V, output).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_output_atomic_kernel(
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

    # ── Smem allocation (scores only — no smem_partial or smem_output) ───
    allocator    = cutlass.utils.SmemAllocator()
    smem_scores  = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None)

    # ── Load scores → smem ────────────────────────────────────────────────
    for i in range(tidx, n, num_threads):
        smem_scores[i] = scores[i]
    cute.arch.sync_threads()

    # ── Output phase: per-warp register accumulation ──────────────────────
    out_regs = cute.make_rmem_tensor(
        cute.make_layout((dims_per_lane,), stride=(1,)),
        cutlass.Float32,
    )
    for k in range(dims_per_lane):
        out_regs[k] = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        j      = round_idx * num_warps + warp_idx
        weight = smem_scores[j]
        for k in range(dims_per_lane):
            out_regs[k] += weight * V[j, k * wsize + lane_idx]

    # ── Epilogue: atomicAdd from registers directly to global output ─────
    # Get base pointer of output tensor as Int64 for PTX addressing
    output_ptr = output.iterator
    for k in range(dims_per_lane):
        dim  = k * wsize + lane_idx
        cute.arch.atomic_add(output_ptr + dim, out_regs[k], sem="relaxed", scope="gpu")


# ── Compilation ────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_output_atomic():
    scores = _fake(cute.Float32, (N,),    (0,),   16)
    V      = _fake(cute.Float32, (N, D),  (1, 0), 16)
    output = _fake(cute.Float32, (D,),    (0,),   16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_output_atomic_fn,
        scores, V, output, stream,
        options="--enable-tvm-ffi",
    )


kernel_output_atomic_compiled = compile_kernel_output_atomic()


def run(scores, V, output):
    """scores: [N] fp32, V: [N,D] fp32, output: [D] fp32 (pre-allocated, MUST be zeroed)."""
    kernel_output_atomic_compiled(scores, V, output)

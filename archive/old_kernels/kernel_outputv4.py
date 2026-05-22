"""kernel_outputv4: output-phase GEMV with smem-staged bf16 V reads.

Built on baseline (kernel_output.py).

Key change:
  Baseline: each warp reads V[j, k*32+lane_idx] directly from fp32 global each k-step.
  v4:       before the k-loop, each warp loads V[j, :] bf16 → fp32 into
            smem_partial[warp_idx, :], then sync_warp(), then the k-loop reads from smem.

  smem_partial is reused across two phases:
    Phase 1 (round loop):  smem_partial[warp_idx, :] = staged V row (temp)
    Phase 2 (after loop):  smem_partial[warp_idx, :] = final partial sums → reduce

  Benefits:
    - V loads: bf16 → 2× bandwidth vs fp32 baseline
    - FFMA reads from smem (LDS, ~32 cycles) instead of gmem (LDG, ~400 cycles)
    - Compiler can better pipeline LDG (next token's V) and FFMA (current smem regs)

  Cost:
    - 16 extra STS per round (write to smem_partial before compute)
    - sync_warp() per round (lightweight vs sync_threads)

Grid: [1, 1, 1]   Block: [1024, 1, 1]
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math

N              : cutlass.Constexpr = 2048
D              : cutlass.Constexpr = 512
BLOCK_SIZE     : cutlass.Constexpr = 1024
NUM_WARPS      : cutlass.Constexpr = BLOCK_SIZE // 32   # 32
DIMS_PER_LANE  : cutlass.Constexpr = D // 32            # 16
NUM_ROUNDS     : cutlass.Constexpr = N // NUM_WARPS     # 64


@cute.jit
def kernel_outputv4_fn(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] bf16
    output: cute.Tensor,   # [D]    fp32
    stream,
):
    kernel_outputv4_kernel(scores, V, output).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_outputv4_kernel(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] bf16
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
    allocator    = cutlass.utils.SmemAllocator()
    smem_scores  = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None)
    # smem_partial serves dual purpose:
    #   [round loop]  smem_partial[warp_idx, :] = staged bf16→fp32 V row (per-warp temp)
    #   [after loop]  smem_partial[warp_idx, :] = final out_regs (for cross-warp reduce)
    # Row-major (stride d,1): conflict-free writes/reads (stride-1 across lanes per k).
    smem_partial = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((num_warps, d), stride=(d, 1)), 16, None)
    smem_output  = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((d,), stride=(1,)), 16, None)

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
        j      = round_idx * num_warps + warp_idx   # token index [0, N)
        weight = smem_scores[j]

        # Stage V[j, :] bf16 → fp32 → smem_partial[warp_idx, :]
        # Each lane loads dims_per_lane=16 bf16 values (2 LDG.128 per lane)
        # Conflict-free STS: stride 1 across lanes for fixed k.
        for k in range(dims_per_lane):
            smem_partial[warp_idx, k * wsize + lane_idx] = cutlass.Float32(V[j, k * wsize + lane_idx])

        # Ensure all smem writes from this warp are visible before reading
        cute.arch.sync_warp()

        # FFMA from smem (LDS) instead of gmem (LDG)
        for k in range(dims_per_lane):
            out_regs[k] += weight * smem_partial[warp_idx, k * wsize + lane_idx]

    # ── Write final partial sums → smem_partial (overwrite staging data) ──
    for k in range(dims_per_lane):
        smem_partial[warp_idx, k * wsize + lane_idx] = out_regs[k]

    cute.arch.sync_threads()

    # ── Cross-warp reduce ─────────────────────────────────────────────────
    for i in range(tidx, d, num_threads):
        acc = cutlass.Float32(0)
        for w in range(num_warps):
            acc += smem_partial[w, i]
        smem_output[i] = acc

    cute.arch.sync_threads()

    # ── Epilogue ──────────────────────────────────────────────────────────
    for i in range(tidx, d, num_threads):
        output[i] = smem_output[i]


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_outputv4():
    scores = _fake(cute.Float32,   (N,),    (0,),    16)
    V      = _fake(cute.BFloat16,  (N, D),  (1, 0),  16)
    output = _fake(cute.Float32,   (D,),    (0,),    16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_outputv4_fn,
        scores, V, output, stream,
        options="--enable-tvm-ffi",
    )


kernel_outputv4_compiled = compile_kernel_outputv4()


def run(scores, V, output):
    """scores: [N] fp32, V: [N,D] bf16, output: [D] fp32 (pre-allocated)."""
    kernel_outputv4_compiled(scores, V, output)

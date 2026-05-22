"""kernel_output_atomicv2: output-phase GEMV with LDG.128 + atomicAdd.

Same operation as kernel_output_atomic, but changes the V-access layout
so that each lane owns CONTIGUOUS dims, enabling vectorized LDG.128:
  - Lane l owns dims [l*16 .. l*16+15]
  - 4 fp32 per LDG.128 → 4 iterations per round, perfectly coalesced

Benefits over kernel_output_atomic (interleaved layout):
  - 4× fewer load instructions (4 LDG.128 vs 16 LDG.E per round)
  - Same atomicAdd epilogue (no smem reduce)

Grid: [1, 1, 1]   Block: [1024, 1, 1]
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

N               : cutlass.Constexpr = 2048
D               : cutlass.Constexpr = 512
BLOCK_SIZE      : cutlass.Constexpr = 1024
NUM_WARPS       : cutlass.Constexpr = BLOCK_SIZE // 32   # 32
DIMS_PER_LANE   : cutlass.Constexpr = D // 32            # 16
NUM_VEC         : cutlass.Constexpr = 4                   # fp32 per LDG.128
ITERS_PER_LANE  : cutlass.Constexpr = DIMS_PER_LANE // NUM_VEC  # 4
NUM_ROUNDS      : cutlass.Constexpr = N // NUM_WARPS     # 64


@cute.jit
def kernel_output_atomicv2_fn(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
    stream,
):
    kernel_output_atomicv2_kernel(scores, V, output).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_output_atomicv2_kernel(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
):
    n               : cutlass.Constexpr = N
    d               : cutlass.Constexpr = D
    num_warps       : cutlass.Constexpr = NUM_WARPS
    dims_per_lane   : cutlass.Constexpr = DIMS_PER_LANE
    num_vec         : cutlass.Constexpr = NUM_VEC
    iters_per_lane  : cutlass.Constexpr = ITERS_PER_LANE
    num_rounds      : cutlass.Constexpr = NUM_ROUNDS
    num_threads     : cutlass.Constexpr = BLOCK_SIZE
    wsize           = cute.arch.WARP_SIZE   # 32

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx   = cute.arch.warp_idx()
    warp_idx   = cute.arch.make_warp_uniform(warp_idx)
    lane_idx   = cute.arch.lane_idx()

    # ── Smem allocation (scores only) ─────────────────────────────────────
    allocator    = cutlass.utils.SmemAllocator()
    smem_scores  = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None)

    # ── Load scores → smem ────────────────────────────────────────────────
    for i in range(tidx, n, num_threads):
        smem_scores[i] = scores[i]
    cute.arch.sync_threads()

    # ── Output phase: per-warp register accumulation with LDG.128 ─────────
    out_regs = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane, num_vec), stride=(num_vec, 1)),
        cutlass.Float32,
    )
    for it in range(iters_per_lane):
        for v in range(num_vec):
            out_regs[it, v] = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        j      = round_idx * num_warps + warp_idx
        weight = smem_scores[j]

        # zipped_divide V row: ((NUM_VEC,), (D//NUM_VEC,))
        V_row_z = cute.zipped_divide(V[j, None], (num_vec,))

        for it in range(iters_per_lane):
            # Each lane loads 4 contiguous fp32 from its owned dims
            group = lane_idx * iters_per_lane + it
            frag  = V_row_z[(None, (group,))].load()

            for v in range(num_vec):
                out_regs[it, v] += weight * frag[v]

    # ── Epilogue: atomicAdd from registers directly to global output ──────
    output_ptr = output.iterator
    for it in range(iters_per_lane):
        for v in range(num_vec):
            dim  = lane_idx * dims_per_lane + it * num_vec + v
            cute.arch.atomic_add(output_ptr + dim, out_regs[it, v], sem="relaxed", scope="gpu")


# ── Compilation ────────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_output_atomicv2():
    scores = _fake(cute.Float32, (N,),    (0,),   16)
    V      = _fake(cute.Float32, (N, D),  (1, 0), 16)
    output = _fake(cute.Float32, (D,),    (0,),   16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_output_atomicv2_fn,
        scores, V, output, stream,
        options="--enable-tvm-ffi",
    )


kernel_output_atomicv2_compiled = compile_kernel_output_atomicv2()


def run(scores, V, output):
    """scores: [N] fp32, V: [N,D] fp32, output: [D] fp32 (pre-allocated, MUST be zeroed)."""
    kernel_output_atomicv2_compiled(scores, V, output)

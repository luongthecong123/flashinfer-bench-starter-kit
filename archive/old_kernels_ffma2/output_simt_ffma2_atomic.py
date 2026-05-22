"""
output_simt_ffma2_atomic.py — Output GEMV with FFMA2 + atomic add to global output.

Same GEMV as output_simt_ffma2.py but eliminates:
  - smem_partial  (16 × 2 × 512 × 4B = 16 KB)
  - smem_partial write loop + sync
  - cross-warp serial reduce loop

Instead, after accumulating in registers, each warp atomically adds its
partial sums directly into the global output tensor.

Output must be zero-initialised by the caller (torch.zeros handles this).
"""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments

M = 2
K = 128
N = 512

NUM_THREADS     = 512
NUM_WARPS       = NUM_THREADS // 32    # 16
VEC_SIZE        = 8
ITERS_PER_LANE  = N // (32 * VEC_SIZE) # 2
NUM_ROUNDS      = K // NUM_WARPS       # 8


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


@cute.jit
def output_gemv_ffma2_atomic_jit(
    weights: cute.Tensor,
    ckv:     cute.Tensor,
    output:  cute.Tensor,
):
    output_gemv_ffma2_atomic_kernel(weights, ckv, output).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1])


@cute.kernel
def output_gemv_ffma2_atomic_kernel(
    weights: cute.Tensor,   # (M, K)   float32
    ckv:     cute.Tensor,   # (K, N)   BF16
    output:  cute.Tensor,   # (M, N)   float32  — must be zeroed before launch
):
    M_:             cutlass.Constexpr = M
    K_:             cutlass.Constexpr = K
    N_:             cutlass.Constexpr = N
    num_threads:    cutlass.Constexpr = NUM_THREADS
    num_warps:      cutlass.Constexpr = NUM_WARPS
    vec_size:       cutlass.Constexpr = VEC_SIZE
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE
    num_rounds:     cutlass.Constexpr = NUM_ROUNDS

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    alloc = cutlass.utils.SmemAllocator()
    # Only smem_weight needed — smem_partial eliminated entirely
    smem_weight = _smem(alloc, cutlass.Float32, (M_ * K_,), (1,), 16)

    # ── Load weights → smem_weight (interleaved) ──────────────────────────────
    for col in range(tidx, K_, num_threads):
        smem_weight[col * 2 + 0] = weights[0, col]
        smem_weight[col * 2 + 1] = weights[1, col]
    cute.arch.sync_threads()

    smem_w_vec2 = cute.zipped_divide(smem_weight, (2,))
    ckv_        = cute.zipped_divide(ckv, (1, vec_size))

    # ── Registers ─────────────────────────────────────────────────────────────
    out_regs_r0 = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane * vec_size,), stride=(1,)), cutlass.Float32)
    out_regs_r1 = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane * vec_size,), stride=(1,)), cutlass.Float32)
    for i in range(iters_per_lane * vec_size):
        out_regs_r0[i] = cutlass.Float32(0)
        out_regs_r1[i] = cutlass.Float32(0)

    # ── GEMV with FFMA2 ───────────────────────────────────────────────────────
    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < K_:
            w_frag = smem_w_vec2[(None,), (sparse_idx,)].load()
            w0 = w_frag[0]
            w1 = w_frag[1]
            ckv_row = ckv_[(0, None), (sparse_idx, None)]

            for it in range(iters_per_lane):
                rest_idx = it * wsize + lane_idx
                ckv_vec = ckv_row[None, rest_idx].load()
                for v in range(vec_size):
                    ckv_f32 = cutlass.Float32(ckv_vec[v])
                    reg_idx = it * vec_size + v
                    out_regs_r0[reg_idx], out_regs_r1[reg_idx] = \
                        cute.arch.fma_packed_f32x2(
                            (w0, w1), (ckv_f32, ckv_f32),
                            (out_regs_r0[reg_idx], out_regs_r1[reg_idx]),
                        )

    # ── Atomic add partial sums → global output (replaces smem write + reduce) ─
    output_ptr = output.iterator
    for it in range(iters_per_lane):
        for v in range(vec_size):
            n_col = (it * wsize + lane_idx) * vec_size + v
            cute.arch.atomic_add(
                output_ptr + (0 * N_ + n_col),
                out_regs_r0[it * vec_size + v],
                sem="relaxed", scope="cta")
            cute.arch.atomic_add(
                output_ptr + (1 * N_ + n_col),
                out_regs_r1[it * vec_size + v],
                sem="relaxed", scope="cta")


def main():
    torch.manual_seed(42)
    weights = torch.rand((M, K), device="cuda", dtype=torch.float32)
    ckv     = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    output  = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    ref = weights.float() @ ckv.float()

    weights_ = from_dlpack(weights, assumed_align=16)
    ckv_     = from_dlpack(ckv,     assumed_align=16)
    output_  = from_dlpack(output,  assumed_align=16)

    compiled = cute.compile(output_gemv_ffma2_atomic_jit, weights_, ckv_, output_)
    compiled(weights_, ckv_, output_)
    torch.cuda.synchronize()

    ok       = torch.allclose(output, ref, atol=1e-2, rtol=1e-2)
    max_diff = (output - ref).abs().max().item()
    print(f"Correctness: {'PASS' if ok else 'FAIL'}  max_diff={max_diff:.6f}")

    # Benchmark: zero output before each pass manually
    # (atomic_add accumulates, so benchmark loop would skew correctness but timing is valid)
    output.zero_()
    time = benchmark(compiled, kernel_arguments=JitArguments(weights_, ckv_, output_))
    print(f"DURATION: {time:>5.4f} µs")


if __name__ == "__main__":
    main()

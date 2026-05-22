"""
output_simt_ffma2_less.py — Minimal output GEMV using FFMA2.

Same as output_simt_ffma2.py but replaces the per-warp smem_partial
(num_warps × M × N = 16KB) + serial reduce with a single shared
smem_output (M × N interleaved = 4KB). All 16 warps accumulate
directly into smem_output — intentional data races, no barriers
between rounds.

smem_output layout: 1D interleaved [r0[n], r1[n], r0[n+1], r1[n+1], ...]
  → LDS.64 loads both row values for column n in one instruction.
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
def output_gemv_ffma2_less_jit(
    weights: cute.Tensor,
    ckv:     cute.Tensor,
    output:  cute.Tensor,
):
    output_gemv_ffma2_less_kernel(weights, ckv, output).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1])


@cute.kernel
def output_gemv_ffma2_less_kernel(
    weights: cute.Tensor,   # (M, K)   float32
    ckv:     cute.Tensor,   # (K, N)   BF16
    output:  cute.Tensor,   # (M, N)   float32
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
    # smem_weight: 1D interleaved [w0[k], w1[k], ...] — LDS.64 per warp per round
    smem_weight  = _smem(alloc, cutlass.Float32, (M_ * K_,),  (1,), 16)
    # smem_output: 1D interleaved [r0[n], r1[n], ...] — LDS.64 / scalar STS per round
    # 2 × 512 = 1024 floats = 4 KB  (vs 16 × 2 × 512 = 16 KB for smem_partial)
    smem_output  = _smem(alloc, cutlass.Float32, (M_ * N_,),  (1,), 16)

    # ── Init: zero smem_output + load weights into smem_weight ───────────────
    for col in range(tidx, K_, num_threads):
        smem_weight[col * 2 + 0] = weights[0, col]
        smem_weight[col * 2 + 1] = weights[1, col]
    for i in range(tidx, N_, num_threads):
        smem_output[i * 2 + 0] = cutlass.Float32(0)
        smem_output[i * 2 + 1] = cutlass.Float32(0)
    cute.arch.sync_threads()

    # vec2 views for LDS.64
    smem_w_vec2   = cute.zipped_divide(smem_weight, (2,))
    smem_out_vec2 = cute.zipped_divide(smem_output, (2,))

    ckv_ = cute.zipped_divide(ckv, (1, vec_size))

    # ── GEMV: all 16 warps accumulate directly into smem_output (racy) ───────
    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < K_:
            # LDS.64: load (w0, w1) pair from interleaved smem_weight
            w_frag = smem_w_vec2[(None,), (sparse_idx,)].load()
            w0 = w_frag[0]
            w1 = w_frag[1]
            ckv_row = ckv_[(0, None), (sparse_idx, None)]

            for it in range(iters_per_lane):
                rest_idx = it * wsize + lane_idx
                ckv_vec  = ckv_row[None, rest_idx].load()   # LDG.128: 8 × BF16
                for v in range(vec_size):
                    n_col   = (it * wsize + lane_idx) * vec_size + v
                    ckv_f32 = cutlass.Float32(ckv_vec[v])
                    # LDS.64: load (r0[n_col], r1[n_col]) in one instruction
                    out_pair = smem_out_vec2[(None,), (n_col,)].load()
                    r0 = out_pair[0]
                    r1 = out_pair[1]
                    # FFMA2: both rows in one instruction
                    r0, r1 = cute.arch.fma_packed_f32x2(
                        (w0, w1), (ckv_f32, ckv_f32), (r0, r1))
                    # Store back (adjacent → STS.64)
                    smem_output[n_col * 2 + 0] = r0
                    smem_output[n_col * 2 + 1] = r1

    cute.arch.sync_threads()

    # ── Copy smem_output → global output ─────────────────────────────────────
    for i in range(tidx, N_, num_threads):
        output[0, i] = smem_output[i * 2 + 0]
        output[1, i] = smem_output[i * 2 + 1]


def main():
    torch.manual_seed(42)
    weights = torch.rand((M, K), device="cuda", dtype=torch.float32)
    ckv     = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    output  = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    ref = weights.float() @ ckv.float()

    weights_ = from_dlpack(weights, assumed_align=16)
    ckv_     = from_dlpack(ckv,     assumed_align=16)
    output_  = from_dlpack(output,  assumed_align=16)

    compiled = cute.compile(output_gemv_ffma2_less_jit, weights_, ckv_, output_)
    compiled(weights_, ckv_, output_)
    torch.cuda.synchronize()

    ok       = torch.allclose(output, ref, atol=1e-2, rtol=1e-2)
    max_diff = (output - ref).abs().max().item()
    print(f"Correctness: {'PASS' if ok else 'FAIL'}  max_diff={max_diff:.6f}")

    time = benchmark(compiled, kernel_arguments=JitArguments(weights_, ckv_, output_))
    print(f"DURATION: {time:>5.4f} µs")


if __name__ == "__main__":
    main()

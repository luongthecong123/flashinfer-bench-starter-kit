"""
output_simt_ffma2.py — Output GEMV kernel using packed FFMA2 instruction.

Computes: output = weights @ ckv
  weights: (M, K)  float32  — softmax probabilities, 2 query rows
  ckv:     (K, N)  BF16     — cached key-value matrix
  output:  (M, N)  float32  — output vectors

Dimensions: M=2, K=128, N=512.
Thread block: 512 threads (16 warps) for more registers per thread.

Design:
  - smem_weight: (2*K,) float32, interleaved: [w0[k], w1[k], w0[k+1], w1[k+1], ...]
      No padding needed — adjacent f32 pairs → single LDS.64 per warp per round.
  - Each warp owns one K-row per round (NUM_ROUNDS=8).
  - Each lane accumulates ITERS_PER_LANE × VEC_SIZE = 2×8 = 16 f32 per output row.
  - M=2 rows → 32 f32 registers per lane (out_regs_r0 + out_regs_r1).
  - fma_packed_f32x2: for each ckv element, compute both rows simultaneously.
      (acc_r0, acc_r1) = fma_packed_f32x2((w0, w1), (ckv_v, ckv_v), (acc_r0, acc_r1))
  - smem_partial: (num_warps, M, N) float32 — per-warp partial sums.
  - Final reduce: for each output row, sum over warps.
"""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments

M = 2       # number of query rows
K = 128     # reduction dimension (token rows)
N = 512     # CKV head dimension (output columns)

NUM_THREADS     = 512
NUM_WARPS       = NUM_THREADS // 32    # 16
VEC_SIZE        = 8                    # 8 × BF16 = 128-bit load
ITERS_PER_LANE  = N // (32 * VEC_SIZE) # 512 / 256 = 2   → 16 regs per row per lane
NUM_ROUNDS      = K // NUM_WARPS       # 128 / 16 = 8


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


@cute.jit
def output_gemv_ffma2_jit(weights: cute.Tensor, ckv: cute.Tensor, output: cute.Tensor):
    output_gemv_ffma2_kernel(weights, ckv, output).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1])


@cute.kernel
def output_gemv_ffma2_kernel(
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

    # ── SMEM allocation ───────────────────────────────────────────────────────
    alloc = cutlass.utils.SmemAllocator()
    # smem_weight: 1D interleaved (2*K,) — pairs [w0[k], w1[k]] are adjacent.
    # LDS.64 fetches both scalars in one instruction (no padding needed).
    smem_weight  = _smem(alloc, cutlass.Float32, (M_ * K_,), (1,), 16)
    # smem_partial: (num_warps, M, N), stride (M*N, N, 1)
    smem_partial = _smem(alloc, cutlass.Float32,
                         (num_warps, M_, N_), (M_ * N_, N_, 1), 16)

    # ── Load weights → smem_weight (interleaved) ──────────────────────────────
    # 512 threads, K=128: each thread writes at most one (w0,w1) pair
    for col in range(tidx, K_, num_threads):
        smem_weight[col * 2 + 0] = weights[0, col]
        smem_weight[col * 2 + 1] = weights[1, col]
    cute.arch.sync_threads()

    # vec2 view: ((2,), (K,)) — each slice of 2 adjacent f32 → LDS.64
    smem_w_vec2 = cute.zipped_divide(smem_weight, (2,))

    # ── Vectorized view of CKV ────────────────────────────────────────────────
    ckv_ = cute.zipped_divide(ckv, (1, vec_size))

    # ── Registers: 16 accumulators per row × 2 rows = 32 f32 per lane ────────
    out_regs_r0 = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane * vec_size,), stride=(1,)), cutlass.Float32)
    out_regs_r1 = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane * vec_size,), stride=(1,)), cutlass.Float32)
    for i in range(iters_per_lane * vec_size):
        out_regs_r0[i] = cutlass.Float32(0)
        out_regs_r1[i] = cutlass.Float32(0)

    # ── GEMV with FFMA2 ───────────────────────────────────────────────────────
    # Each warp handles one K-row per round.
    # fma_packed_f32x2((w0,w1),(ckv,ckv),(acc0,acc1))
    #   → (acc0 + w0*ckv, acc1 + w1*ckv)  — one instruction for both output rows.
    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < K_:
            # LDS.64: load (w0, w1) pair from interleaved smem in one instruction
            w_frag = smem_w_vec2[(None,), (sparse_idx,)].load()
            w0 = w_frag[0]   # weight for row 0
            w1 = w_frag[1]   # weight for row 1
            ckv_row = ckv_[(0, None), (sparse_idx, None)]

            for it in range(iters_per_lane):
                rest_idx = it * wsize + lane_idx
                ckv_vec = ckv_row[None, rest_idx].load()   # 8 × BF16
                for v in range(vec_size):
                    ckv_f32 = cutlass.Float32(ckv_vec[v])
                    reg_idx = it * vec_size + v
                    out_regs_r0[reg_idx], out_regs_r1[reg_idx] = \
                        cute.arch.fma_packed_f32x2(
                            (w0, w1), (ckv_f32, ckv_f32),
                            (out_regs_r0[reg_idx], out_regs_r1[reg_idx]),
                        )

    # ── Write warp partial sums → smem_partial ────────────────────────────────
    for it in range(iters_per_lane):
        for v in range(vec_size):
            n_col = (it * wsize + lane_idx) * vec_size + v
            smem_partial[warp_idx, 0, n_col] = out_regs_r0[it * vec_size + v]
            smem_partial[warp_idx, 1, n_col] = out_regs_r1[it * vec_size + v]

    cute.arch.sync_threads()

    # ── Reduce across warps → output ─────────────────────────────────────────
    for m in range(M_):
        for i in range(tidx, N_, num_threads):
            acc = cutlass.Float32(0)
            for w in range(num_warps):
                acc += smem_partial[w, m, i]
            output[m, i] = acc


def main():
    torch.manual_seed(42)
    weights = torch.rand((M, K), device="cuda", dtype=torch.float32)
    ckv     = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    output  = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    # Reference: weights @ ckv  (M,K) @ (K,N) = (M,N)
    ref = weights.float() @ ckv.float()

    weights_ = from_dlpack(weights, assumed_align=16)
    ckv_     = from_dlpack(ckv,     assumed_align=16)
    output_  = from_dlpack(output,  assumed_align=16)

    compiled = cute.compile(output_gemv_ffma2_jit, weights_, ckv_, output_)
    compiled(weights_, ckv_, output_)
    torch.cuda.synchronize()

    assert torch.allclose(output, ref, atol=1e-2, rtol=1e-2), \
        f"CORRECTNESS FAILED — max diff: {(output - ref).abs().max():.6f}"
    print("CORRECTNESS PASS")

    time = benchmark(compiled, kernel_arguments=JitArguments(weights_, ckv_, output_))
    print(f"DURATION: {time:>5.4f} µs")


if __name__ == "__main__":
    main()

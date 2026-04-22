"""
output_simt.py — Standalone output GEMV kernel (SIMT FMA).

Computes: output = weights @ ckv
  weights: (K,)    float32  — softmax probabilities for each CKV row
  ckv:     (K, N)  BF16    — cached key-value matrix
  output:  (N,)    float32  — output vector

Dimensions: K=256, N=512, 1024 threads (32 warps).
Follows the output phase of kv_split_xor_pdl_v3_pro_v2.py exactly:
  - Each warp handles K/NUM_WARPS = 8 rows of ckv per round (NUM_ROUNDS=8)
  - Each lane accumulates ITERS_PER_LANE × VEC_SIZE = 16 output elements
  - Partial sums written to smem_partial (32×512), then reduced across warps
"""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments

K = 256        # number of token rows (reduction dimension)
N = 512        # CKV head dimension (output dimension)

NUM_THREADS     = 1024
NUM_WARPS       = NUM_THREADS // 32   # 32
VEC_SIZE        = 8                   # 8 × BF16 = 128-bit load
ITERS_PER_LANE  = N // (32 * VEC_SIZE)  # 512 // 256 = 2
NUM_ROUNDS      = (K + NUM_WARPS - 1) // NUM_WARPS  # 8


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


@cute.jit
def output_gemv_jit(weights: cute.Tensor, ckv: cute.Tensor, output: cute.Tensor):
    output_gemv_kernel(weights, ckv, output).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1])


@cute.kernel
def output_gemv_kernel(
    weights: cute.Tensor,   # (K,)    float32 — softmax weights
    ckv:     cute.Tensor,   # (K, N)  BF16
    output:  cute.Tensor,   # (N,)    float32
):
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

    # SMEM allocations
    alloc = cutlass.utils.SmemAllocator()
    smem_weight  = _smem(alloc, cutlass.Float32, (K_,),           (1,),    16)
    smem_partial = _smem(alloc, cutlass.Float32, (num_warps, N_), (N_, 1), 16)

    # ── Load weights into smem ────────────────────────────────────────────────
    for i in range(tidx, K_, num_threads):
        smem_weight[i] = weights[i]
    cute.arch.sync_threads()

    # Vectorized view: chunks of vec_size BF16 along the N dimension
    ckv_ = cute.zipped_divide(ckv, (1, vec_size))

    # Registers: iters_per_lane × vec_size = 16 accumulators per lane
    out_regs = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane * vec_size,), stride=(1,)),
        cutlass.Float32,
    )
    for i in range(iters_per_lane * vec_size):
        out_regs[i] = cutlass.Float32(0)

    # ── GEMV: each warp processes one row of ckv per round ───────────────────
    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < K_:
            e = smem_weight[sparse_idx]
            ckv_row = ckv_[(0, None), (sparse_idx, None)]
            for it in range(iters_per_lane):
                rest_idx = it * wsize + lane_idx
                ckv_vec = ckv_row[None, rest_idx].load()
                for v in range(vec_size):
                    out_regs[it * vec_size + v] += e * cutlass.Float32(ckv_vec[v])

    # ── Write warp partial sums to smem ──────────────────────────────────────
    for it in range(iters_per_lane):
        for v in range(vec_size):
            smem_partial[warp_idx, (it * wsize + lane_idx) * vec_size + v] = \
                out_regs[it * vec_size + v]

    cute.arch.sync_threads()

    # ── Reduce across all warps → final output (each thread handles ≤1 elem) ─
    for i in range(tidx, N_, num_threads):
        acc = cutlass.Float32(0)
        for w in range(num_warps):
            acc += smem_partial[w, i]
        output[i] = acc


def main():
    torch.manual_seed(42)
    weights = torch.rand((K,), device="cuda", dtype=torch.float32)
    ckv     = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    output  = torch.zeros((N,), device="cuda", dtype=torch.float32)

    # Reference: (1,K) @ (K,N) → (1,N), squeezed to (N,)
    ref = (weights.float().unsqueeze(0) @ ckv.float()).squeeze(0)

    weights_ = from_dlpack(weights, assumed_align=16)
    ckv_     = from_dlpack(ckv,     assumed_align=16)
    output_  = from_dlpack(output,  assumed_align=16)

    compiled = cute.compile(output_gemv_jit, weights_, ckv_, output_)
    compiled(weights_, ckv_, output_)
    torch.cuda.synchronize()

    assert torch.allclose(output, ref, atol=1e-2, rtol=1e-2), \
        f"CORRECTNESS FAILED — max diff: {(output - ref).abs().max():.6f}"
    print("CORRECTNESS PASS")

    time = benchmark(compiled, kernel_arguments=JitArguments(weights_, ckv_, output_))
    print(f"DURATION: {time:>5.4f} µs")


if __name__ == "__main__":
    main()

"""
Standalone fp8 SIMT GEMM test.
Mimics the indexer score kernel: C[m,n] = sum_k K_fp8[m,k] * Q_fp8[n,k]

Data is passed as int8 (byte-level) and recast to fp8 inside the kernel,
matching how the contest passes k_index_cache_fp8 and q_index_fp8.

Uses naive CUDA-like SIMT (no tensor cores) to validate correctness.

Run locally:
    python test_fp8_simt_gemm.py
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cute.testing import benchmark, JitArguments


# ── Constants matching the contest ──
PAGE_SIZE = 64
HEAD_DIM  = 128
NUM_HEADS = 64
BM = 128   # M tile = 2 pages = 128 tokens
BN = 64    # N tile = 64 heads

# Thread block: 128 threads. Each thread computes one M-row.
THREADS = 128


# ──────────────────────────────────────────────────────────────────────────────
# Kernel: SIMT fp8 GEMM  C[M,N] += A_fp8[M,K] * B_fp8[N,K]
# A and B are passed as Int8, recast to Float8E4M3FN inside.
# Each of the 128 threads handles one M-row, iterating over all N heads.
# ──────────────────────────────────────────────────────────────────────────────
@cute.jit
def simt_fp8_gemm_jit(
    mA_i8: cute.Tensor,   # (BM, BK) int8 — K tile (fp8 bits)
    mB_i8: cute.Tensor,   # (BN, BK) int8 — Q tile (fp8 bits)
    mC: cute.Tensor,      # (BM, BN) float32 — output
):
    M = mA_i8.shape[0]
    N = mB_i8.shape[0]
    K = mA_i8.shape[1]

    simt_fp8_gemm_kernel(mA_i8, mB_i8, mC, M, N, K).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
    )


@cute.kernel
def simt_fp8_gemm_kernel(
    gA_i8: cute.Tensor,   # (M, K) int8
    gB_i8: cute.Tensor,   # (N, K) int8
    gC: cute.Tensor,      # (M, N) f32
    M: int, N: int, K: int,
):
    tidx, _, _ = cute.arch.thread_idx()

    # Recast int8 → fp8 (same bits, different type interpretation)
    fp8_A_ptr = cute.recast_ptr(gA_i8.iterator, dtype=cutlass.Float8E4M3FN)
    gA = cute.make_tensor(fp8_A_ptr, cute.make_layout((M, K), stride=(K, 1)))

    fp8_B_ptr = cute.recast_ptr(gB_i8.iterator, dtype=cutlass.Float8E4M3FN)
    gB = cute.make_tensor(fp8_B_ptr, cute.make_layout((N, K), stride=(K, 1)))

    num_vec: cutlass.Constexpr = 4   # 4 fp8 = 32 bits (aligned)
    K_iters: cutlass.Constexpr = 32  # K=128 / num_vec=4 = 32

    m = tidx
    if m < M:
        A_row = gA[m, None]
        A_z   = cute.zipped_divide(A_row, (num_vec,))

        for n in range(N):
            B_row = gB[n, None]
            B_z   = cute.zipped_divide(B_row, (num_vec,))

            acc = cutlass.Float32(0)
            for k4 in range(K_iters):
                a_frag = A_z[(None, (k4,))].load()
                b_frag = B_z[(None, (k4,))].load()
                a_f32 = a_frag.to(cutlass.Float32)
                b_f32 = b_frag.to(cutlass.Float32)
                for v in cutlass.range_constexpr(num_vec):
                    acc += a_f32[v] * b_f32[v]
            gC[m, n] = acc


# ──────────────────────────────────────────────────────────────────────────────
# Test harness
# ──────────────────────────────────────────────────────────────────────────────
def make_test_data():
    """Create fp8 A (K-cache tile) and B (Q tile) with known values."""
    torch.manual_seed(42)

    # Random fp8 data
    A_f32 = torch.randn(BM, HEAD_DIM, device="cuda", dtype=torch.float32).clamp(-240, 240)
    B_f32 = torch.randn(BN, HEAD_DIM, device="cuda", dtype=torch.float32).clamp(-240, 240)

    A_fp8 = A_f32.to(torch.float8_e4m3fn)
    B_fp8 = B_f32.to(torch.float8_e4m3fn)

    # Reference in f32
    A_ref = A_fp8.to(torch.float32)
    B_ref = B_fp8.to(torch.float32)
    C_ref = A_ref @ B_ref.T   # (BM, BN)

    C_out = torch.zeros(BM, BN, device="cuda", dtype=torch.float32)

    # View fp8 as int8 for dlpack transport
    A_i8 = A_fp8.view(torch.int8)
    B_i8 = B_fp8.view(torch.int8)

    return A_i8, B_i8, C_out, C_ref


def main():
    A_i8, B_i8, C_out, C_ref = make_test_data()

    A_ = from_dlpack(A_i8, assumed_align=16)
    B_ = from_dlpack(B_i8, assumed_align=16)
    C_ = from_dlpack(C_out, assumed_align=16)

    print(f"A shape: {A_i8.shape}, dtype: {A_i8.dtype} (fp8 bits)")
    print(f"B shape: {B_i8.shape}, dtype: {B_i8.dtype} (fp8 bits)")
    print(f"C shape: {C_out.shape}, dtype: {C_out.dtype}")

    compiled = cute.compile(simt_fp8_gemm_jit, A_, B_, C_)
    compiled(A_, B_, C_)

    # Validate
    diff = (C_out - C_ref).abs()
    max_diff = diff.max().item()
    rel_diff = (diff / (C_ref.abs() + 1e-6)).max().item()

    print(f"\nmax_abs_diff = {max_diff:.6f}")
    print(f"max_rel_diff = {rel_diff:.6f}")

    # fp8 has limited precision, so tolerance is generous
    if max_diff < 2.0:
        print("CORRECTNESS PASS ✓")
    else:
        print("CORRECTNESS FAIL ✗")
        # Print first few mismatches
        bad = (diff > 2.0).nonzero(as_tuple=False)[:5]
        for idx in bad:
            m, n = idx[0].item(), idx[1].item()
            print(f"  [{m},{n}] kern={C_out[m,n].item():.4f} ref={C_ref[m,n].item():.4f} diff={diff[m,n].item():.4f}")

    # Benchmark
    time_us = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    flops = 2 * BM * BN * HEAD_DIM
    print(f"\nDURATION: {time_us:>5.2f} µs")
    print(f"GFLOPS:   {flops / (time_us * 1e3):>5.2f}")


if __name__ == "__main__":
    main()

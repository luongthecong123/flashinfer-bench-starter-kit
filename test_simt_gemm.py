"""
Standalone SIMT GEMM test (no tensor cores).
Step 1: f32 GEMM to prove the CuTeDSL SIMT pattern.
Step 2: fp8 GEMM with manual dequant.

C[m,n] = sum_k A[m,k] * B[n,k]   (B is transposed)

Run locally:
    python test_simt_gemm.py
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ── Dimensions ──
BM = 128   # M tile = 128 tokens
BN = 64    # N tile = 64 heads
BK = 128   # K = head_dim
THREADS = 128


# ──────────────────────────────────────────────────────────────────────────────
# Kernel: f32 SIMT GEMM
# ──────────────────────────────────────────────────────────────────────────────
@cute.jit
def simt_gemm_jit(
    mA: cute.Tensor,   # (BM, BK) f32
    mB: cute.Tensor,   # (BN, BK) f32
    mC: cute.Tensor,   # (BM, BN) f32
):
    M = mA.shape[0]
    N = mB.shape[0]
    K = mA.shape[1]
    simt_gemm_kernel(mA, mB, mC, M, N, K).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
    )


@cute.kernel
def simt_gemm_kernel(
    gA: cute.Tensor,   # (M, K) f32
    gB: cute.Tensor,   # (N, K) f32
    gC: cute.Tensor,   # (M, N) f32
    M: int, N: int, K: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    m = tidx
    if m < M:
        for n in range(N):
            acc = cutlass.Float32(0)
            for k in range(K):
                acc = acc + gA[m, k] * gB[n, k]
            gC[m, n] = acc


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(42)

    A = torch.randn(BM, BK, device="cuda", dtype=torch.float32)
    B = torch.randn(BN, BK, device="cuda", dtype=torch.float32)
    C_ref = A @ B.T
    C_out = torch.zeros(BM, BN, device="cuda", dtype=torch.float32)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C_out, assumed_align=16)

    print(f"A: {A.shape} f32, B: {B.shape} f32, C: {C_out.shape} f32")

    compiled = cute.compile(simt_gemm_jit, A_, B_, C_)
    compiled(A_, B_, C_)

    diff = (C_out - C_ref).abs()
    max_diff = diff.max().item()
    print(f"max_abs_diff = {max_diff:.6f}")

    if max_diff < 1e-2:
        print("CORRECTNESS PASS ✓")
    else:
        print("CORRECTNESS FAIL ✗")
        bad = (diff > 1e-2).nonzero(as_tuple=False)[:5]
        for idx in bad:
            m, n = idx[0].item(), idx[1].item()
            print(f"  [{m},{n}] kern={C_out[m,n]:.4f} ref={C_ref[m,n]:.4f}")


if __name__ == "__main__":
    main()

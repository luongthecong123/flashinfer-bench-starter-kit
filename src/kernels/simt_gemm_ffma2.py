import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch

@cute.jit
def naive_ffma2(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[0]
    K = mA.shape[1]

    BM, BN = 16, 16

    naive_ffma2_kernel(mA, mB, mC, M, N, K).launch(
        grid=[N // BN, M // BM, 1],
        block=[BM, BN // 2, 1])  # BN//2 threads: each thread computes 2 N-outputs via ffma2

@cute.kernel
def naive_ffma2_kernel(
    gA: cute.Tensor,  # [M, K]
    gB: cute.Tensor,  # [N, K]
    gC: cute.Tensor,  # [M, N]
    M: int,
    N: int,
    K: int,
):
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    tidx, tidy, _ = cute.arch.thread_idx()

    m = bidy * 16 + tidy
    # Each thread handles two adjacent N columns: n0 and n1
    n0 = bidx * 16 + tidx * 2
    n1 = n0 + 1

    acc0 = cutlass.Float32(0.0)
    acc1 = cutlass.Float32(0.0)

    for k in range(K):
        a_val = gA[m, k]
        b0    = gB[n0, k]
        b1    = gB[n1, k]
        # fma_packed_f32x2: (d0, d1) = a * b + c  (packed 2x f32)
        acc0, acc1 = cute.arch.fma_packed_f32x2(
            (a_val, a_val), (b0, b1), (acc0, acc1)
        )

    gC[m, n0] = acc0
    gC[m, n1] = acc1

def main():
    M, N, K = 1024, 1024, 1024

    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((N, K), device="cuda", dtype=torch.float32)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    compiled = cute.compile(naive_ffma2, A_, B_, C_)
    compiled(A_, B_, C_)

    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")

if __name__ == "__main__":
    main()

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments

NUM_THR_X = 16
NUM_THR_Y = 16
NUM_THR = NUM_THR_X * NUM_THR_Y
THR_TILING_M = 8
THR_TILING_N = 8
BM = NUM_THR_X * THR_TILING_M
BN = NUM_THR_Y * THR_TILING_N
BK = 16

@cute.jit
def smem_cute_launcher(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    grid_m = (mC.shape[0] + BM - 1) // BM
    grid_n = (mC.shape[1] + BN - 1) // BN

    smem_cute_kernel(mA, mB, mC).launch(
        grid=[grid_m, grid_n, 1],
        block=[NUM_THR, 1, 1]
    )

@cute.kernel
def smem_cute_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    bidx, bidy, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    tidx = tid % NUM_THR_X
    tidy = tid // NUM_THR_X
    
    gA_ = cute.zipped_divide(mA, (BK, BM)) # ((BK, BM), (K//BK, M//BM))
    gB_ = cute.zipped_divide(mB, (BK, BN)) # ((BK, BN), (K//BK, N//BN))
    gC_ = cute.zipped_divide(mC, (BM, BN)) # ((BM, BN), (M//BM, N//BN))
    
    gA = gA_[(None, None), (None, bidx)] # (BK, BM, K//BK)
    gB = gB_[(None, None), (None, bidy)] # (BK, BN, K//BK)
    gC = gC_[(None, None), (bidx, bidy)] # (BM, BN)
    
    print("gA: ", gA)
    print("gB: ", gB)
    print("gC: ", gC)

    # Allocate shared memory
    allocator = cutlass.utils.SmemAllocator()
    sA_layout = cute.make_layout((BK, BM), stride=(BM, 1))
    sB_layout = cute.make_layout((BK, BN), stride=(BN, 1))
    sA = allocator.allocate_tensor(cutlass.Float16, sA_layout, 16, None)
    sB = allocator.allocate_tensor(cutlass.Float16, sB_layout, 16, None)


    # # Tiled MMA: universal (non-TC) 1x1 atom tiled to 16x16 threads
    # # permutation_mnk repeats the 16x16 atom to cover the full BM x BN tile
    # atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    # mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    # tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout, permutation_mnk=(2, 2, 1))

    # # Thread partitioning for MMA
    # thr_mma = tiled_mma.get_slice(tidx)
    # tCgC = thr_mma.partition_C(gC)
    # tCrC = tiled_mma.make_fragment_C(tCgC)
    # tCrC.fill(0)

    # # Partition smem for MMA (these are reused every k-iteration)
    # tCsA = thr_mma.partition_A(sA)
    # tCsB = thr_mma.partition_B(sB)
    # tCrA = tiled_mma.make_fragment_A(tCsA)
    # tCrB = tiled_mma.make_fragment_B(tCsB)

    # print("tCsA: ", tCsA)
    # print("tCsB: ", tCsB)
    # print("tCrA: ", tCrA)
    # print("tCrB: ", tCrB)

    K_tiles = gA.shape[2]
    
    accum_layout = cute.make_layout((THR_TILING_M, THR_TILING_N), stride=(THR_TILING_N, 1))
    accum = cute.make_rmem_tensor(accum_layout, cutlass.Float32)
    accum.fill(0)
    for k in range(K_tiles):
        # GMEM -> SMEM
        print("gA[None, None, k]: ", gA[None, None, k])
        print("gB[None, None, k]: ", gB[None, None, k])
        
        cute.autovec_copy(gA[None, None, k], sA)
        cute.autovec_copy(gB[None, None, k], sB)

        cute.arch.sync_threads()

        # # SMEM -> RMEM (cast fp16 -> fp32)
        # for i in range(cute.size(tCrA)):
        #     tCrA[i] = cutlass.Float32(tCsA[i])
        # for i in range(cute.size(tCrB)):
        #     tCrB[i] = cutlass.Float32(tCsB[i])

        # # GEMM on register fragments
        # cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
        

        
        for kidx in range(BK):
            for mma_m in cutlass.range(THR_TILING_M):
                for mma_n in cutlass.range(THR_TILING_N):
                    a_val = sA[kidx, tidy * THR_TILING_M + mma_m] # [BK, BM]
                    b_val = sB[kidx, tidx * THR_TILING_N + mma_n] # [BK, BN]
                    accum[mma_m, mma_n] += cutlass.Float32(a_val) * cutlass.Float32(b_val)
        
        cute.arch.sync_threads()
        

    # Store results back to GMEM (cast fp32 -> fp16)
    for write_m in cutlass.range(THR_TILING_M):
        for write_n in cutlass.range(THR_TILING_N):
            gC[tidy * THR_TILING_M + write_m, tidx * THR_TILING_N + write_n] = cutlass.Float16(accum[write_m, write_n])
                

def main():
    M, N, K = 1024, 1024, 1024

    A = torch.randn((K, M), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    compiled = cute.compile(smem_cute_launcher, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, torch.matmul(A.T, B), atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")
    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"DURATION: {time:>5.4f} µs\nTFLOPS: {(2 * M * N * K) / (time * 1e6):>5.4f}")

if __name__ == "__main__":
    main()

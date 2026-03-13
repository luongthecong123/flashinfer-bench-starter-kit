import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
from typing import Tuple

import torch

"""
Batched WMMA GEMM with SMEM (all static shapes).
Based on b2_wmma_smem.py — adds a batch dimension sliced via bidz.
"""

class Gemm_TC_Batched:
    def __init__(
        self,
        cta_tiler = (16, 64, 64)
    ):
        self._cta_tiler = cta_tiler
        self._bM, self._bN, self._bK = self._cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (1, 4, 1)
        self._num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self._smem_padding = 8
        self._num_vectorized = 4   
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor, 
        mC: cute.Tensor,
    ):
        # ====== MMA Layout ======
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)

        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )     
        
        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)
        
        # ====== SMEM layout ======
        padding = self._smem_padding
        sA_layout = cute.make_layout(shape=(self._bM, self._bK), stride=(self._bK + padding, 1))
        sB_layout = cute.make_layout(shape=(self._bN, self._bK), stride=(self._bK + padding, 1))
        
        # ====== COPY layout ======
        num_vectorized = self._num_vectorized
        atom_copy_A = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vectorized
        )
        atom_copy_B = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mB.element_type,
            num_bits_per_copy=mB.element_type.width * num_vectorized
        )
        # K-major
        major_mode_size = self._bK // num_vectorized
        tA = cute.make_layout(
            shape=(self._num_threads // major_mode_size, major_mode_size),
            stride=(major_mode_size, 1)
        )
        vA = cute.make_layout(shape=(1, num_vectorized), stride=(0, 1))

        tiled_copy_A = cute.make_tiled_copy_tv(atom_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_copy_B, tA, vA)
        
        # grid_dim: (ceil(M/BM), ceil(N/BN), BS)
        # mC shape is (BS, M, N) — mode 0 is batch, modes 1,2 are spatial
        BS = mC.shape[0]
        grid_dim = *cute.ceil_div((mC.shape[1], mC.shape[2]), (self._bM, self._bN)), BS

        self.kernel(
            mA, mB, mC,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B,
            tiled_mma, permutation_mnk
            ).launch(
            grid=grid_dim, 
            block=(self._num_threads, 1, 1)
        )
    
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        permutation_mnk: Tuple
    ):
        # ====== Thread, Block setup =======
        bidx, bidy, bidz = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        
        # ===== Smem allocation ======
        allocator = cutlass.utils.SmemAllocator()
        sA = allocator.allocate_tensor(cutlass.Float16, sA_layout, 16, None)
        sB = allocator.allocate_tensor(cutlass.Float16, sB_layout, 16, None)
        
        # Slice batch dimension first, then tile the 2D matrices
        gA_batch = mA[bidz, None, None]  # (M, K)
        gB_batch = mB[bidz, None, None]  # (N, K)
        gC_batch = mC[bidz, None, None]  # (M, N)

        gA = cute.local_tile(
            input=gA_batch, 
            tiler=self._cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, None, 1))
        
        gB = cute.local_tile(
            input=gB_batch, 
            tiler=self._cta_tiler,
            coord=(bidx, bidy, None),
            proj=(None, 1, 1))
        
        gC = cute.local_tile(
            input=gC_batch, 
            tiler=self._cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, 1, None))
        
        thr_copyA = tiled_copy_A.get_slice(tid)
        thr_copyB = tiled_copy_B.get_slice(tid)
        tAgA = thr_copyA.partition_S(gA)
        tAsA = thr_copyA.partition_D(sA)
        tBgB = thr_copyB.partition_S(gB)
        tBsB = thr_copyB.partition_D(sB)
        
        # ===== mma thread partitioning memory spaces =====
        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        
        # ====== Shared memory to register copy ======
        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mA.element_type,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mB.element_type,
        )

        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tid)
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tid)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)   
        
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
        
        # ====== Main loop ======
        tCrC.fill(0.0)
        
        # K is mode 1 of gA_batch (M, K), so total K tiles = K / bK
        for kidx in range(gA_batch.shape[1] // self._bK):
            # Load gmem -> smem
            cute.copy(
                atom=tiled_copy_A,
                src=tAgA[None, None, None, kidx],
                dst=tAsA[None, None, None]
            )
            
            cute.copy(
                atom=tiled_copy_B,
                src=tBgB[None, None, None, kidx],
                dst=tBsB[None, None, None]
            )
            
            cute.arch.sync_threads()
            
            # Load smem -> register
            cute.copy(
                atom=tiled_copy_s2r_A,
                src=tCsA_copy_view,
                dst=tCrA_copy_view
            )
            
            cute.copy(
                atom=tiled_copy_s2r_B,
                src=tCsB_copy_view,
                dst=tCrB_copy_view
            )
            
            # GEMM on register fragments
            cute.gemm(
                atom=tiled_mma,
                d=tCrC,
                a=tCrA,
                b=tCrB,
                c=tCrC
            )
            
            cute.arch.sync_threads()
        
        # ====== Store results ======
        atom_universal = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC.element_type 
        )
        
        # tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        
        # for reg_idx in range(cute.size(tCrC_out)):
        #     tCrC_out[reg_idx] = cutlass.Float16(tCrC[reg_idx])
            
        cute.copy(
            atom=atom_universal,
            src=tCrC,
            dst=tCgC
        )


def main():
    BS, M, N, K = 8, 16, 2048, 512

    A = torch.randn((BS, M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((BS, N, K), device="cuda", dtype=torch.bfloat16)
    C = torch.empty((BS, M, N), device="cuda", dtype=torch.float32)

    A_ = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=A.dim_order()).mark_compact_shape_dynamic(mode=2, divisibility=64)
    B_ = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=B.dim_order()).mark_compact_shape_dynamic(mode=2, divisibility=64)
    C_ = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=C.dim_order())

    gemm = Gemm_TC_Batched()
    compiled = cute.compile(gemm, A_, B_, C_)
    compiled(A_, B_, C_)

    ref = torch.bmm(A.to(torch.float32), B.mT.to(torch.float32))
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED"
    print("CORRECTNESS PASS")

    time = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    tflops = (2 * BS * M * N * K) / (time * 1e6)
    print(f"DURATION: {time:>5.4f} µs | TFLOPS: {tflops:.4f}")
    
    BS, K = 8, 256
    A = torch.randn((BS, M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((BS, N, K), device="cuda", dtype=torch.bfloat16)
    C = torch.empty((BS, M, N), device="cuda", dtype=torch.float32)
    
    A_ = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=A.dim_order()).mark_compact_shape_dynamic(mode=2, divisibility=64)
    B_ = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=B.dim_order()).mark_compact_shape_dynamic(mode=2, divisibility=64)
    C_ = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=C.dim_order())
    
    compiled(A_, B_, C_)
    
    ref = torch.bmm(A.to(torch.float32), B.mT.to(torch.float32))
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "CORRECTNESS FAILED with different K"
    print("CORRECTNESS PASS with different K")
    
    time2 = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    tflops2 = (2 * BS * M * N * K) / (time2 * 1e6)
    print(f"  Reused for K={K}:")
    print(f"  DURATION: {time2:>5.4f} µs | TFLOPS: {tflops2:.4f}")


if __name__ == "__main__":
    main()

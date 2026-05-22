import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor
from cutlass.cute.testing import benchmark, JitArguments

from typing import Tuple
import math
import torch

"""
Tensor core version of letmecook.py
Here we do B @ A.T = C.T to leverage tensor core large M dimension and small N dimension.

"""


# https://www.youtube.com/watch?v=5qSN-R_E3w0
@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        # cute.arch.shuffle_sync_bfly will read from another thread's registers
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

@cute.jit
class Fused_DSA:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int, int, int] = (16, 64, 64, 64, 16)
    ):
        self.tile_shape_mnk = cta_tiler
        self.BM, self.BN, self.Bdkc, self.Bdkp, self.Bdv = self.tile_shape_mnk
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (1, 4, 1)
        self.num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1] # 128 threads
        self.smem_padding = 8
        self.num_vectorized = 4   
    
    @cute.jit
    def __call__(
        self,
    q_nope: cute.Tensor, 
    q_pe: cute.Tensor, 
    ckv_cache: cute.Tensor, 
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    output: cute.Tensor,
    lse: cute.Tensor):
        """Fused DSA sparse attention kernel.

        Inputs:
            q_nope:              [T, 16, 512]    bf16
            q_pe:                [T, 16, 64]     bf16
            ckv_cache:           [N, 512]        bf16   (flat page pool)
            kpe_cache:           [N, 64]         bf16   (flat page pool)
            sparse_indices:      [T, 2048]       int32  (flat token indices, -1 = end sentinel)
            sm_scale:            Constexpr               (baked in at compile time)
            output:              [T, 16, 512]    bf16
            lse:                 [T, 16]         float
        """
        T, num_heads, dkc = q_nope.shape
        T, num_heads, dv = output.shape
        
        # ====== MMA Layout ======
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)

        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )     
        
        tiled_mma_logits = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)
        
        mma_op_output = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        mma_output_layout = cute.make_layout((self.BM, self.Bdv), stride=(self.Bdv, 1))
        tiled_mma_output = cute.make_tiled_mma(mma_op_output, mma_output_layout)
        
        
        
        # # ====== SMEM layout ======
        # padding = self.smem_padding
        # sA_layout = cute.make_layout(shape=(self.BM, self.BK), stride=(self.BK + padding, 1))
        # sB_layout = cute.make_layout(shape=(self.BN, self.BK), stride=(self.BK + padding, 1))
        
        # # ====== COPY layout ======
        # num_vectorized = self.num_vectorized
        # atom_copy_A = cute.make_copy_atom(
        #     cute.nvgpu.CopyUniversalOp(),
        #     mA.element_type,
        #     num_bits_per_copy=mA.element_type.width * num_vectorized
        # )
        # atom_copy_B = cute.make_copy_atom(
        #     cute.nvgpu.CopyUniversalOp(),
        #     mB.element_type,
        #     num_bits_per_copy=mB.element_type.width * num_vectorized
        # )
        # # K-major
        # major_mode_size = self.BK // num_vectorized
        # tA = cute.make_layout(
        #     shape=(self.num_threads // major_mode_size, major_mode_size),
        #     stride=(major_mode_size, 1)
        # )
        # vA = cute.make_layout(shape=(1, num_vectorized), stride=(0, 1))

        # tiled_copy_A = cute.make_tiled_copy_tv(atom_copy_A, tA, vA)
        # tiled_copy_B = cute.make_tiled_copy_tv(atom_copy_B, tA, vA)
        
        # # grid_dim: (ceil(M/BM), ceil(N/BN), BS)
        # # mC shape is (BS, M, N) — mode 0 is batch, modes 1,2 are spatial
        # BS = mC.shape[0]
        # grid_dim = *cute.ceil_div((mC.shape[1], mC.shape[2]), (self.BM, self.BN)), BS

        self.kernel(
            q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse,
            tiled_mma_logits
            ).launch(
            grid=[num_heads // self.BM, dv // self.Bdv, T],
            block=(self.num_threads, 1, 1)
        )
    
    @cute.kernel
    def kernel(
        self,
        q_nope: cute.Tensor, q_pe: cute.Tensor, 
        ckv_cache: cute.Tensor, kpe_cache: cute.Tensor,
        sparse_indices: cute.Tensor, sm_scale: cutlass.Constexpr,
        output: cute.Tensor, lse: cute.Tensor,
        tiled_mma_logits: cute.TiledMMA,
    ):
        """Fused DSA sparse attention kernel.

        Inputs:
            q_nope:              [T, 16, 512]    bf16
            q_pe:                [T, 16, 64]     bf16
            ckv_cache:           [N, 512]        bf16   (flat page pool)
            kpe_cache:           [N, 64]         bf16   (flat page pool)
            sparse_indices:      [T, 2048]       int32  (flat token indices, -1 = end sentinel)
            sm_scale:            Constexpr               (baked in at compile time)
            output:              [T, 16, 512]    bf16
            lse:                 [T, 16]         float
        """
        _, topk = sparse_indices.shape
        _, _, dkc = q_nope.shape
        _, _, dkp = output.shape
                
        # ====== Thread, Block setup =======
        bidx, bidy, bidz = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        
        # ===== Smem allocation ======
        allocator = cutlass.utils.SmemAllocator()
        
        # Logits tiles
        sQn_layout = cute.make_layout((self.BM, self.Bdkc), stride=(self.Bdkc, 1))
        sK1_layout = cute.make_layout((self.BN, self.Bdkc), stride=(self.Bdkc, 1))
        sQn = allocator.allocate_tensor(cutlass.Float16, sQn_layout, 16, None)
        sK1 = allocator.allocate_tensor(cutlass.Float16, sK1_layout, 16, None)
        smem_sparse_idx = allocator.allocate_tensor(cutlass.Int32, cute.make_layout((topk), stride=(1)), 16, None)
        
        # Softmax and output tiles
        sL = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((self.BM, self.BN), stride=(self.BN, 1)), 16, None)
        sK2_layout = cute.make_layout((self.BN, self.Bdkp), stride=(1, self.Bdkp)) # Column major
        sK2 = allocator.allocate_tensor(cutlass.Float16, sK2_layout, 16, None)

        # smem_max_idx = allocator.allocate_tensor(cutlass.Int32, cute.make_layout((1), stride=(1)), 4, None)
        # smem_max_idx[0] = topk + 9999
        

        
        # Load sparse indices to SMEM for faster access
        cute.autovec_copy(sparse_indices[bidz], smem_sparse_idx)
        cute.arch.sync_threads()
        
# ============================== GMEM partitioning ===============================        
                               
        qn = q_nope[bidz, None, None] # (16, 512)
        gQn_ = cute.zipped_divide(qn, (self.BM, self.Bdkc)) # ((BM, Bdkc), (M//BM, dkc//Bdkc))
        gQn = gQn_[(None, None), (bidx, None)] # (BM, Bdkc, dkc//Bdkc) # matA
        gKc1_ = cute.zipped_divide(ckv_cache, (self.BN, self.Bdkc)) # ((BN, Bdkc), (N//BN, dkc//Bdkc)) # matB
        
        qp = q_pe[bidz, None, None] # (16, 64)
        gQp_ = cute.zipped_divide(qp, (self.BM, self.Bdkp)) # ((BM, Bdkp), (M//BM, dkp//Bdkp))
        gQp = gQp_[(None, None), (bidx, None)] # (BM, Bdkp, dkp//Bdkp) # matA
        gKp_ = cute.zipped_divide(kpe_cache, (self.BN, self.Bdkp)) # ((BN, Bdkp), (N//BN, dkp//Bdkp)) # matB
        
# ============================== Logits MMA setup =============================== 
        
        # MMA thread partitioning:
        thr_mma = tiled_mma_logits.get_slice(tid)
        tCsA = thr_mma.partition_A(sQn)
        tCsB = thr_mma.partition_B(sK1)        

        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            q_nope.element_type,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            ckv_cache.element_type,
        )

        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma_logits)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma_logits)
        
        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tid)
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tid)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sQn) 
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sK1)

        tCgC = thr_mma.partition_C(gL)
        frag_rowsum_partial = tiled_mma_logits.make_fragment_C(tCgC)
        frag_rowsum_partial.fill(0.0)
                
        # Create placeholder for matC/logits as we don't materialize it
        gL = cute.make_layout((self.BM, self.BN, 1))
        
        # Loop over all 2048 entries, skip if -1, track max_valid_idx as we go
        max_valid_idx = topk + 9999
        for nidx in range(topk // self.BN):
            
            if nidx * self.BN < max_valid_idx: # Early exit condition
                
                # Find max valid index in this BN tile
                for idx in range(self.BN * nidx, self.BN * (nidx + 1)):
                    if smem_sparse_idx[idx] == -1:
                        max_valid_idx = idx
                
                gKc1 = gKc1_[(None, None), (nidx, None)] # (BN, Bdkc, dkc//Bdkc) # matB
                gKp = gKp_[(None, None), (nidx, None)] # (BN, Bdkp, dkp//Bdkp) # matB
                
                # Fill sK1 with zero to allow mma on partial tiles
                sK1.fill(0)
                
                cute.arch.sync_threads() # Might consider removing this if it doesn't cause race condition

                
                tCrA = tiled_mma_logits.make_fragment_A(tCsA)
                tCrB = tiled_mma_logits.make_fragment_B(tCsB)
                tCrC = tiled_mma_logits.make_fragment_C(tCgC)                    
            
                tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)   
                tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)                
            
                tCrC.fill(0.0)

# ============================== Step 1: Calculate logits ===============================
                
                # Main loop score_nope
                for kidx in range(dkc//self.Bdkc):
                    # Load Q to SMEM
                    cute.autovec_copy(gQn[None, None, kidx], sQn)
                    
                    # Load K to SMEM conditionally
                    for idx in range(self.BN):
                        if idx + nidx * self.BN < max_valid_idx:
                            cute.autovec_copy(gKc1[idx, None, kidx], sK1[idx])
                            
                    cute.arch.sync_threads()
                    
                    ## WMMA on the tile
                    cute.copy(atom=tiled_copy_s2r_A, src=tCsA_copy_view, dst=tCrA_copy_view)
                    cute.copy(atom=tiled_copy_s2r_B, src=tCsB_copy_view, dst=tCrB_copy_view)
                    cute.gemm(atom=tiled_mma_logits, d=tCrC, a=tCrA, b=tCrB, c=tCrC)
                    cute.arch.sync_threads()                    
                # End for kidx in score_nope loop

                # Main loop score_pe
                for kidx in range(dkp//self.Bdkp): # Should exit in one iteration since dkp=Bdkp=64
                    # Load Q to SMEM
                    cute.autovec_copy(gQp[None, None, kidx], sQn)
                    
                    # Load K to SMEM conditionally
                    for idx in range(self.BN):
                        if idx + nidx * self.BN < max_valid_idx:
                            cute.autovec_copy(gKp[idx, None, kidx], sK1[idx])
                            
                    cute.arch.sync_threads()
                    
                    ## WMMA on the tile
                    cute.copy(atom=tiled_copy_s2r_A, src=tCsA_copy_view, dst=tCrA_copy_view)
                    cute.copy(atom=tiled_copy_s2r_B, src=tCsB_copy_view, dst=tCrB_copy_view)
                    cute.gemm(atom=tiled_mma_logits, d=tCrC, a=tCrA, b=tCrB, c=tCrC)
                    cute.arch.sync_threads()
                # End for kidx in score_pe loop
                    
# ============================== Step 2: Softmax =============================== 
            # Scale and take exponential on numerator
            for idx in range(cute.size(tCrC)):
                if tCrC[idx] != 0.0:
                    tCrC[idx] = math.exp(tCrC[idx] * sm_scale)
            
            # Accumulate partial denominator
            for idx in range(cute.size(tCrC)):
                frag_rowsum_partial[idx] += tCrC[idx]
            
            
            cute.autovec_copy(tCrC, sL)
            cute.arch.sync_threads()
            
            
            
            
            
            # End if nidx * self.BN < max_valid_idx # Early exit condition   
        
        # End for nidx in range(topk // self.BN)
        
        
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
        for kidx in range(gA_batch.shape[1] // self.BK):
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



def fake_wrapper(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=assumed_align)
        
def compile_fused_dsa_kernel():
    T = cute.sym_int()
    N = cute.sym_int()  # total tokens in flat cache (num_pages * page_size)
    num_heads, dkc, dkp, topk = 16, 512, 64, 2048
    
    q_nope = fake_wrapper(cute.BFloat16, (T, num_heads, dkc), (2, 1, 0), 16)
    q_pe = fake_wrapper(cute.BFloat16, (T, num_heads, dkp), (2, 1, 0), 16)
    ckv_cache = fake_wrapper(cute.BFloat16, (N, dkc), (1, 0), 16)
    kpe_cache = fake_wrapper(cute.BFloat16, (N, dkp), (1, 0), 16)
    sparse_indices = fake_wrapper(cute.Int32, (T, topk), (1, 0), 4)
    sm_scale = 0.1352337788608801
    output = fake_wrapper(cute.BFloat16, (T, num_heads, dkc), (2, 1, 0), 16)
    lse = fake_wrapper(cute.Float32, (T, num_heads), (1, 0), 4)
    
    fused_dsa = Fused_DSA()
    
    return cute.compile(
        fused_dsa,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse,
        options="--enable-tvm-ffi"
    )

fused_dsa_compiled = compile_fused_dsa_kernel()
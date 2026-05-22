import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cute.testing import benchmark, JitArguments

from typing import Tuple
import math
import torch

# https://www.youtube.com/watch?v=5qSN-R_E3w0
@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        # cute.arch.shuffle_sync_bfly will read from another thread's registers
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

class Fused_DSA:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int, int, int] = (16, 64, 64, 64, 8)
    ):
        self.tile_shape_mnk = cta_tiler
        self.BM, self.BN, self.Bdkc, self.Bdkp, self.Bdv = self.tile_shape_mnk
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (1, 4, 1)
        self.num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1] # 128 threads
        self.warp_size = cute.arch.WARP_SIZE
        
        self.smem_padding = 8
        self.num_vectorized = 4   

        assert self.BN >= 32, "BN must be at least 32 to utilize full warp for reduction"
        # NOTE: BMxBdv can be smaller than num threads, where we can add out of bound guards
        assert self.BM * self.Bdv == self.num_threads, "Naive gemm output: num threads must be equal BMxBdv"   
    
    @cute.jit
    def __call__(
        self,
        q_nope: cute.Tensor,            # [T, 16, 512] bf16
        q_pe: cute.Tensor,              # [T, 16, 64]  bf16    
        kc: cute.Tensor,                # [T, 2048, 512] bf16   (pre-gathered, zero-padded)
        kp: cute.Tensor,                # [T, 2048, 64]  bf16   (pre-gathered, zero-padded)
        sparse_indices: cute.Tensor,    # [T, 2048]   int32  (flat token indices, -1 = end sentinel)
        max_valid: cute.Tensor,         # [T] int32  — count of valid (!=-1) per token
        sm_scale: cute.Tensor,          # [1] float32 — scalar softmax scale
        output: cute.Tensor,            # [T, 16, 512] bf16
        lse: cute.Tensor,               # [T, 16]     float
        stream):                        # CUDA stream
        T, num_heads, dkc = q_nope.shape
        T, num_heads, dv = output.shape
        
        # ====== Tiled MMA logits setup ======
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
        
        self.kernel(
            q_nope, q_pe, kc, kp, sparse_indices, max_valid, sm_scale, output, lse,
            tiled_mma_logits
            ).launch(
            grid=[num_heads // self.BM, dv // self.Bdv, T],
            block=(self.num_threads, 1, 1),
            stream=stream
        )
    
    @cute.kernel
    def kernel(
        self,
        q_nope: cute.Tensor,                        # [T, 16, 512] bf16 
        q_pe: cute.Tensor,                          # [T, 16, 64]  bf16
        kc: cute.Tensor,                            # [T, 2048, 512] bf16   (pre-gathered)
        kp: cute.Tensor,                            # [T, 2048, 64]  bf16   (pre-gathered)
        sparse_indices: cute.Tensor,                # [T, 2048]   int32  (flat token indices, -1 = end sentinel)
        max_valid: cute.Tensor,                     # [T] int32  — output: count of valid (!=-1) per token
        sm_scale: cute.Tensor,                      # [1] float32 — scalar softmax scale
        output: cute.Tensor,                        # [T, 16, 512] bf16, 
        lse: cute.Tensor,                           # [T, 16] float
        tiled_mma_logits: cute.TiledMma
    ):
        T, topk = sparse_indices.shape
        _, num_heads, dkc = q_nope.shape
        _, _, dkp = q_pe.shape
        _, _, dv = output.shape
                
        # ====== Thread, Block setup =======
        
        bidx, bidy, batch_idx = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
                
        # ===== Smem allocation ======
        
        allocator = cutlass.utils.SmemAllocator()
        
        # Logits tiles
        sQn_layout = cute.make_layout((self.BM, self.Bdkc), stride=(self.Bdkc, 1))
        sK1_layout = cute.make_layout((self.BN, self.Bdkc), stride=(self.Bdkc, 1))
        sQn = allocator.allocate_tensor(cutlass.Float16, sQn_layout, 16, None)
        sK1 = allocator.allocate_tensor(cutlass.Float16, sK1_layout, 16, None)
        
        # Softmax and output tiles
        sL = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((self.BM, self.BN), stride=(self.BN, 1)), 16, None)
        sK2_layout = cute.make_layout((self.Bdv, self.BN), stride=(self.BN + 4, 1))
        sK2 = allocator.allocate_tensor(cutlass.Float16, sK2_layout, 4, None)
        sLSE = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((self.BM,), stride=(1,)), 4, None)
        sLSE.fill(0.0)

        # ============================== GMEM partitioning ===============================        
                              
        qn = q_nope[batch_idx, None, None] # (16, 512)
        gQn_ = cute.zipped_divide(qn, (self.BM, self.Bdkc)) # ((BM, Bdkc), (M//BM, dkc//Bdkc))
        gQn = gQn_[(None, None), (bidx, None)] # (BM, Bdkc, dkc//Bdkc) # matA
        kc_batch = kc[batch_idx, None, None] # (2048, 512)
        gKc1_ = cute.zipped_divide(kc_batch, (self.BN, self.Bdkc)) # ((BN, Bdkc), (topk//BN, dkc//Bdkc)) # matB
        
        qp = q_pe[batch_idx, None, None] # (16, 64)
        gQp_ = cute.zipped_divide(qp, (self.BM, self.Bdkp)) # ((BM, Bdkp), (M//BM, dkp//Bdkp))
        gQp = gQp_[(None, None), (bidx, None)] # (BM, Bdkp, dkp//Bdkp) # matA
        kp_batch = kp[batch_idx, None, None] # (2048, 64)
        gKp_ = cute.zipped_divide(kp_batch, (self.BN, self.Bdkp)) # ((BN, Bdkp), (topk//BN, dkp//Bdkp)) # matB
        
        gKc2__ = cute.zipped_divide(kc_batch, (self.BN, self.Bdv)) # ((BN, Bdv), (topk//BN, dv//Bdv)) # matB for output MMA
        gKc2_ = gKc2__[(None, None), (None, bidy)]
        
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
            kc.element_type,
        )

        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma_logits)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma_logits)
        
        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tid)
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tid)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sQn) 
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sK1)

        # Get accumulator shape without materializing a C tensor
        acc_shape = thr_mma.partition_shape_C((self.BM, self.BN))
                                        
        accum_out = cutlass.Float32(0)
        local_max_valid = max_valid[batch_idx]
        num_BN_tiles = (local_max_valid + cutlass.Int32(self.BN - 1)) // cutlass.Int32(self.BN)
        for nidx in range(num_BN_tiles):
                       
            # ============================== Step 1: Calculate logits ===============================            
            gKc1 = gKc1_[(None, None), (nidx, None)] # (BN, Bdkc, dkc//Bdkc) # matB
                        
            tCrA = tiled_mma_logits.make_fragment_A(tCsA)
            tCrB = tiled_mma_logits.make_fragment_B(tCsB)
            tCrC = tiled_mma_logits.make_fragment_C(acc_shape)
        
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)   
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)                
        
            tCrC.fill(0.0)
            sL.fill(0.0)
                        
            # Main loop score_nope
            for kidx in range(dkc//self.Bdkc):
                cute.autovec_copy(gQn[None, None, kidx], sQn)
                cute.autovec_copy(gKc1[None, None, kidx], sK1)                        
                cute.arch.sync_threads()
                
                ## WMMA on the tile
                cute.copy(atom=tiled_copy_s2r_A, src=tCsA_copy_view, dst=tCrA_copy_view)
                cute.copy(atom=tiled_copy_s2r_B, src=tCsB_copy_view, dst=tCrB_copy_view)
                cute.gemm(atom=tiled_mma_logits, d=tCrC, a=tCrA, b=tCrB, c=tCrC)
                cute.arch.sync_threads()                    
            # End for kidx in score_nope loop
            
            gKp = gKp_[(None, None), (nidx, None)] # (BN, Bdkp, dkp//Bdkp) # matB
            # Main loop score_pe
            for kidx in range(dkp//self.Bdkp): # Should exit in one iteration since dkp=Bdkp=64
                cute.autovec_copy(gQp[None, None, kidx], sQn)
                cute.autovec_copy(gKp[None, None, kidx], sK1)                        
                cute.arch.sync_threads()
                
                ## WMMA on the tile
                cute.copy(atom=tiled_copy_s2r_A, src=tCsA_copy_view, dst=tCrA_copy_view)
                cute.copy(atom=tiled_copy_s2r_B, src=tCsB_copy_view, dst=tCrB_copy_view)
                cute.gemm(atom=tiled_mma_logits, d=tCrC, a=tCrA, b=tCrB, c=tCrC)
                cute.arch.sync_threads()
            # End for kidx in score_pe loop
                    
            # ============================== Step 2: Softmax =============================== 
            # Scatter accumulator registers to smem using TV-to-MN mapping
            # Apply scale + exp only for valid positions; invalid stay 0.0
            tv_layout_C = tiled_mma_logits.tv_layout_C_tiled
            sL_shape = cute.make_layout((self.BM, self.BN), stride=(self.BN, 1)).shape
            for reg_idx in range(cute.size(tCrC)):
                coord = cute.idx2crd((tid, reg_idx), tv_layout_C.shape)
                mn_flat = cute.crd2idx(coord, tv_layout_C)
                m, n = cute.idx2crd(mn_flat, sL_shape)
                global_n = nidx * self.BN + n
                if global_n < local_max_valid:
                    sL[m, n] = cute.math.exp(tCrC[reg_idx] * sm_scale[0])
            cute.arch.sync_threads()
            
            # Reduction to get rowsum/denominator
            lane_idx = cute.arch.lane_idx()
            for row_idx in range(warp_idx, self.BM, self.num_threads // self.warp_size):
                local_sum = cutlass.Float32(0.0)
                for i in range(self.BN // self.warp_size):
                    local_sum += sL[row_idx, lane_idx + i * self.warp_size]
                
                # Intra-warp reduction to get total sum for the row
                total_sum = warp_reduce(local_sum, lambda a, b: a + b)
                # Lane 0 writes the total sum (rowsum) to shared memory
                if lane_idx == 0:
                    sLSE[row_idx] += total_sum
            
            # ============================== Step 3: Calculate output =============================== 
            gKc2 = gKc2_[None, None, nidx] # (BN, Bdv) # matB for output MMA
            
            # Transposed load to sK2
            num_loads_B = self.BN * self.Bdv
            for i in range(tid, num_loads_B, self.num_threads):
                k = i // self.Bdv
                n = i % self.Bdv
                if k < local_max_valid:
                    sK2[n, k] = gKc2[k, n] 
            cute.arch.sync_threads()
            
            # accum_out
            tidx = tid % self.Bdv
            tidy = tid // self.Bdv
            
            for mmak in range(self.BN):
                if mmak < local_max_valid:
                    accum_out += sL[tidy, mmak] * cutlass.Float32(sK2[tidx, mmak])
        
        # End for nidx in range(num_BN_tiles)
        if bidy == 0:
            if tid < self.BM:
                lse[batch_idx, tid] = cute.math.log(sLSE[tid]) / cutlass.Float32(0.6931471805599453)
            
        tidx = tid % self.Bdv
        tidy = tid // self.Bdv
        
        gOut_ = cute.zipped_divide(output[batch_idx, None, None], (self.BM, self.Bdv)) # ((BM, Bdv), (M//BM, dv//Bdv))
        gOut = gOut_[(None, None), (bidx, bidy)]
        gOut[tidy, tidx] = cutlass.BFloat16(accum_out / sLSE[tidy])


def fake_wrapper(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=assumed_align)
        
def compile_fused_dsa_kernel():
    T = cute.sym_int()
    num_heads, dkc, dkp, topk = 16, 512, 64, 2048
    
    q_nope = fake_wrapper(cute.BFloat16, (T, num_heads, dkc), (2, 1, 0), 16)
    q_pe = fake_wrapper(cute.BFloat16, (T, num_heads, dkp), (2, 1, 0), 16)
    kc = fake_wrapper(cute.BFloat16, (T, topk, dkc), (2, 1, 0), 16)
    kp = fake_wrapper(cute.BFloat16, (T, topk, dkp), (2, 1, 0), 16)
    sparse_indices = fake_wrapper(cute.Int32, (T, topk), (1, 0), 4)
    max_valid = fake_wrapper(cute.Int32, (T,), (0,), 4)
    sm_scale = fake_wrapper(cute.Float32, (1,), (0,), 4)
    output = fake_wrapper(cute.BFloat16, (T, num_heads, dkc), (2, 1, 0), 16)
    lse = fake_wrapper(cute.Float32, (T, num_heads), (1, 0), 4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    
    fused_dsa = Fused_DSA()
    
    return cute.compile(
        fused_dsa,
        q_nope, q_pe, kc, kp, sparse_indices, max_valid, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )

fused_dsa_compiled = compile_fused_dsa_kernel()
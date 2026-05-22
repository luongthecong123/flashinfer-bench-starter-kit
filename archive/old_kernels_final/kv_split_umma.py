import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
import math
import torch

# Input constants
NUM_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
TOP_K = 2048
NUM_PAGES = 8462
PAGE_SIZE = 64
FLAT_CACHE = NUM_PAGES * PAGE_SIZE
LN2 = 0.6931471805599453
SM_SCALE: cutlass.Constexpr = 0.1352337788608801
LIMIT_REQUEST = 8
# TODO: Might need to move this to runtime assert under __call__
# assert T <= LIMIT_REQUEST, f"Got {T} requests, max is {LIMIT_REQUEST}"
assert LIMIT_REQUEST <= 8, "EXCEPTION: Impl is hard-coded to max 8 requests, for more requests, add another for loop outside"
# KV split constant
DIM_CHUNK = 8
NUM_SPLITS = 16
DIM_SPLIT = (TOP_K + NUM_SPLITS - 1) // NUM_SPLITS  # 128
HEADS_PER_SPLIT = 2


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )

@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

class Dsa():
    def __init__(self):
        self.wsize = cute.arch.WARP_SIZE
        
        # Fully fused kernel (small workloads)
        self.fused_threads = 1024
        self.fused_warps = self.fused_threads // 32   # 32
        self.dims_per_lane = HEAD_DIM_CKV // 32  # 16
        self.fused_num_vec = 8
        self.fused_iters   = self.dims_per_lane // self.fused_num_vec  # 2
        
        # KV split compute kernel
        # Prologue
        self.swz_rot_shift = 7
        self.sp_vec_size_i32 = 4
        self.out_stages = 4
        self.out_vec = HEAD_DIM_CKV // (self.out_stages * self.wsize) # 4
                
        # UMMA: KV @ Q.T = SCORE.T
        self.umma_threads = 256
        self.num_umma_warps = self.umma_threads // self.wsize
        self.umma_inst = (DIM_SPLIT, DIM_CHUNK * HEADS_PER_SPLIT, 16) # (128, 16, 16)
        self.tmem_ld_rep = self.umma_inst[1]
        self.cta_tiler = (DIM_SPLIT, DIM_CHUNK * HEADS_PER_SPLIT, HEAD_DIM_CKV + HEAD_DIM_KPE) # (128, 16, 576)
        self.umma_bar_id = 2
        self.ab_dtype  = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.ckv_vec = 8
        self.kpe_vec = 2
        self.k_tile = 64
        
        # SGEMM: SM(Q @ KV.T) @ V
        self.sgemm_threads = 512
        self.num_sgemm_warps = self.sgemm_threads // self.wsize
        self.sgemm_ckv_vec = 4
        self.sgemm_kpe_vec = 2
        self.sgemm_bar_id = 3
        self.sgemm_max_red_bar_id = 4
        
        
        # Reduce kernel
        self.reduce_threads = 256
        self.reduce_warps = self.reduce_threads // self.wsize  # 8
        self.vec_reduce = 2
        
        # ── Workspace: allocated once, reused across calls ────────────────────
        self.partial_out = torch.empty(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.float32, device="cuda")
        self.partial_lse = torch.empty(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, 2, dtype=torch.float32, device="cuda")        
    
    @cute.jit
    def __call__(
        self,
        q_nope:         cute.Tensor,
        q_pe:           cute.Tensor,
        ckv_cache:      cute.Tensor,
        kpe_cache:      cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale:       cutlass.Constexpr,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor,
        stream):
        
        T, _, _ = q_nope.shape
        ckv_flat = cute.make_tensor(
            ckv_cache.iterator,
            cute.make_layout((FLAT_CACHE, HEAD_DIM_CKV), stride=(HEAD_DIM_CKV, 1)))
        kpe_flat = cute.make_tensor(
            kpe_cache.iterator,
            cute.make_layout((FLAT_CACHE, HEAD_DIM_KPE), stride=(HEAD_DIM_KPE, 1)))

        op = tcgen05.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.umma_inst,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.cta_tiler, self.ab_dtype, 1,)
        b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, self.cta_tiler, self.ab_dtype, 1,)

        @cute.struct
        class SharedStorage:
            umma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.compute_kernel(
            tiled_mma, a_smem_layout, b_smem_layout,
            q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
            partial_out, partial_lse, output, lse,
        ).launch(grid=[NUM_HEADS // HEADS_PER_SPLIT, NUM_SPLITS, 1],
                 block=[self.umma_threads + self.sgemm_threads, 1, 1], stream=stream)

        self.reduce_kernel(
            sparse_indices, partial_out, partial_lse, output, lse,
        ).launch(grid=[T, NUM_HEADS, 1],
                 block=[self.reduce_threads, 1, 1], stream=stream)
        
    @staticmethod
    def _smem(allocator, dtype, shape, stride, byte_alignment=16, swizzle=None):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), byte_alignment, swizzle)
    
    @cute.kernel
    def compute_kernel(
        self,
        tiled_mma, a_smem_layout, b_smem_layout,
        q_nope:         cute.Tensor,
        q_pe:           cute.Tensor,
        ckv_flat:       cute.Tensor,
        kpe_flat:       cute.Tensor,
        sparse_indices: cute.Tensor,
        sm_scale:       cutlass.Constexpr,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor):
        
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()
        
        umma_tidx = tidx # 0 -> 255
        sgemm_tidx = tidx - self.umma_threads # 0 -> 511
        
        # ========= SMEM setup =========
        alloc = cutlass.utils.SmemAllocator()
        sA = alloc.allocate_tensor(self.ab_dtype, a_smem_layout.outer, 128, a_smem_layout.inner)
        sB = alloc.allocate_tensor(self.ab_dtype, b_smem_layout.outer, 128, b_smem_layout.inner)
        smem_qr_sgemm = self._smem(alloc, self.ab_dtype, (LIMIT_REQUEST, HEADS_PER_SPLIT, HEAD_DIM_CKV),(HEADS_PER_SPLIT * HEAD_DIM_CKV, HEAD_DIM_CKV, 1))
        smem_qn_sgemm = self._smem(alloc, self.ab_dtype, (LIMIT_REQUEST, HEADS_PER_SPLIT, HEAD_DIM_KPE),(HEADS_PER_SPLIT * HEAD_DIM_KPE, HEAD_DIM_KPE, 1))
        smem_sp_indices = self._smem(alloc, cutlass.Int32, (DIM_CHUNK, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_assign = self._smem(alloc, cutlass.Int32, (DIM_CHUNK, 2),(2,1))
        smem_score_umma = self._smem(alloc, cutlass.Float32, (DIM_SPLIT * HEADS_PER_SPLIT,), (1,))     
        smem_score_sgemm = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT, DIM_SPLIT), (DIM_SPLIT, 1))
        # TODO: Might needs swizzle 128b here to avoid bank conflict on 2 heads
        smem_partial_umma = self._smem(alloc, cutlass.Float32, (self.num_umma_warps, HEADS_PER_SPLIT, HEAD_DIM_CKV // self.out_stages),
                           (HEADS_PER_SPLIT * (HEAD_DIM_CKV // self.out_stages), HEAD_DIM_CKV // self.out_stages, 1))
        # TODO: Might needs swizzle 128b here to avoid bank conflict on 2 heads
        smem_partial_sgemm = self._smem(alloc, cutlass.Float32, (self.num_sgemm_warps, HEADS_PER_SPLIT, DIM_SPLIT),
                           (HEADS_PER_SPLIT * DIM_SPLIT, DIM_SPLIT, 1))
        # Per-warp intermediate for UMMA cross-warp max/sum reduction (shape: num_umma_warps × HEADS_PER_SPLIT)
        smem_sm_red_umma = self._smem(alloc, cutlass.Float32, (self.num_umma_warps, HEADS_PER_SPLIT), (HEADS_PER_SPLIT, 1))
        smem_max = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT,), (1,))
        smem_sum = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT,), (1,))
        smem_logits_flat = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT * DIM_SPLIT,), (1,))
        
        storage = alloc.allocate(self.shared_storage)
        umma_mbar = storage.umma_mbar_ptr.data_ptr()
        
        # ========= cp.async setup =========
        atom_cpa_ckv = cute.make_copy_atom(cpasync.CopyG2SOp(), self.ab_dtype, num_bits_per_copy=128)
        thr_layout_per_warp_ckv = cute.make_layout((1, (8, 4)), stride=(32, (1, 8)))
        val_layout_per_warp_ckv = cute.make_layout((1, (self.ckv_vec, 1)), stride=(0, (1,0)))
        tiled_copy_per_warp_ckv = cute.make_tiled_copy_tv(atom_cpa_ckv, thr_layout_per_warp_ckv, val_layout_per_warp_ckv)
        lane_copy_ckv = tiled_copy_per_warp_ckv.get_slice(lane_idx)
        
        atom_cpa_kpe = cute.make_copy_atom(cpasync.CopyG2SOp(), self.ab_dtype, num_bits_per_copy=32)
        thr_layout_per_warp_kpe = cute.make_layout((1, 32), stride=(32, 1))
        val_layout_per_warp_kpe = cute.make_layout((1, self.kpe_vec), stride=(0, 1))
        tiled_copy_per_warp_kpe = cute.make_tiled_copy_tv(atom_cpa_kpe, thr_layout_per_warp_kpe, val_layout_per_warp_kpe)
        lane_copy_kpe = tiled_copy_per_warp_kpe.get_slice(lane_idx)

        # Dedicated SGEMM query preloads: match test_score's known-good
        # 1D (thread, vec) mapping to avoid UMMA copy-layout remap issues.
        atom_cpa_qn_score = cute.make_copy_atom(cpasync.CopyG2SOp(), self.ab_dtype, num_bits_per_copy=128)
        tiled_copy_qn_score = cute.make_tiled_copy_tv(
            atom_cpa_qn_score,
            cute.make_layout((32,), stride=(1,)),
            cute.make_layout((self.ckv_vec,), stride=(1,)),
        )
        lane_copy_qn_score = tiled_copy_qn_score.get_slice(lane_idx)

        atom_cpa_qr_score = cute.make_copy_atom(cpasync.CopyG2SOp(), self.ab_dtype, num_bits_per_copy=32)
        tiled_copy_qr_score = cute.make_tiled_copy_tv(
            atom_cpa_qr_score,
            cute.make_layout((32,), stride=(1,)),
            cute.make_layout((self.kpe_vec,), stride=(1,)),
        )
        lane_copy_qr_score = tiled_copy_qr_score.get_slice(lane_idx)
        
        # ========= Reshape tensor =========
        ckv_full = cute.make_tensor(
            ckv_flat.iterator, 
            cute.make_layout(
                (1, FLAT_CACHE, (self.k_tile, HEAD_DIM_CKV // self.k_tile)), 
                stride=(0, HEAD_DIM_CKV, (1, self.k_tile))
            )
        )
        
        gB_full = cute.make_tensor(
            q_nope.iterator, 
            cute.make_layout(
                (1, DIM_CHUNK, NUM_HEADS, (self.k_tile, HEAD_DIM_CKV // self.k_tile)), 
                stride=(0, NUM_HEADS * HEAD_DIM_CKV, HEAD_DIM_CKV, (1, self.k_tile))
            )
        )
        
        kpe_full = cute.make_tensor(
            kpe_flat.iterator,
            cute.make_layout((1, FLAT_CACHE, self.k_tile), stride=(0, HEAD_DIM_KPE, 1)),
        )
        q_nope_full = cute.make_tensor(
            q_pe.iterator,
            cute.make_layout(
                (1, DIM_CHUNK, NUM_HEADS, self.k_tile),
                stride=(0, NUM_HEADS * HEAD_DIM_KPE, HEAD_DIM_KPE, 1),
            ),
        )

        sA_ckv = cute.make_tensor(
            sA.iterator,
            cute.make_layout(
                (1, DIM_SPLIT, (self.k_tile, HEAD_DIM_CKV // self.k_tile)),
                stride=(0, self.k_tile, (1, DIM_SPLIT * self.k_tile)),
            ),
        )
        
        # Shift 128x512
        sA_kpe = cute.make_tensor(
            sA.iterator + (DIM_SPLIT * HEAD_DIM_CKV),
            cute.make_layout((1, DIM_SPLIT, self.k_tile), stride=(0, self.k_tile, 1)),
        )
        sB_qr = cute.make_tensor(
            sB.iterator,
            cute.make_layout(
                (1, self.umma_inst[1], (self.k_tile, HEAD_DIM_CKV // self.k_tile)),
                stride=(0, self.k_tile, (1, self.umma_inst[1] * self.k_tile)),
            ),
        )
        
        # Shift by UMMA_Nx512
        sB_qn = cute.make_tensor(
            sB.iterator + (self.umma_inst[1] * HEAD_DIM_CKV),
            cute.make_layout((1, self.umma_inst[1], self.k_tile), stride=(0, self.k_tile, 1)),
        )
        
        # ========= TMEM alloc + mbarrier init =========
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape       = tiled_mma.partition_shape_C(self.cta_tiler[:2])
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)
        
        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        
        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_init(umma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        num_k_blocks = cute.size(tCrA, mode=[2])
        umma_phase = cutlass.Int32(0)
        
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        
        # ========= UMMA score epilogue setup =========
        M_acc          = cute.size(tCtAcc, mode=[0, 0])
        ld_op          = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler      = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi     = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r  = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy  = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc       = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc       = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)
        
        # ========= Prologue - Swizzle split with wrapped-around rotation =========                  
        # TODO: Bring this to after work assignment to avoid duplicated loading
        
        # 1. Regardless of split, we load q ckv and q kpe
        
        # Each warp loads 2 head, i.e. first block holds the first 2 heads of 8 requests.
        # Head maps to bidx, request_idx maps to warp_idx
        head_base_idx, split_idx_old, _ = cute.arch.block_idx() # 0 -> 7, 0 -> 15, _
        
        T, _, _ = q_nope.shape
        # 2D thr/val copy atoms need (1, DIM) shaped destinations; wrap smem as 4D views
        smem_qr_full = cute.make_tensor(
            smem_qr_sgemm.iterator,
            cute.make_layout(
                (1, LIMIT_REQUEST, HEADS_PER_SPLIT, HEAD_DIM_CKV),
                stride=(0, HEADS_PER_SPLIT * HEAD_DIM_CKV, HEAD_DIM_CKV, 1),
            ),
        )
        smem_qn_full = cute.make_tensor(
            smem_qn_sgemm.iterator,
            cute.make_layout(
                (1, LIMIT_REQUEST, HEADS_PER_SPLIT, HEAD_DIM_KPE),
                stride=(0, HEADS_PER_SPLIT * HEAD_DIM_KPE, HEAD_DIM_KPE, 1),
            ),
        )
        # UMMA query preload: 16 rows (token-major, 2 heads each), matching reference mapping.
        if warp_idx < self.umma_inst[1]:
            t_idx_w = warp_idx >> cutlass.Int32(1)
            h_local_w = warp_idx & cutlass.Int32(1)
            safe_t = t_idx_w
            if t_idx_w >= T:
                safe_t = cutlass.Int32(0)

            head_idx_w = head_base_idx * HEADS_PER_SPLIT + h_local_w
            sB_row_w = warp_idx

            gB_row = gB_full[None, safe_t, head_idx_w, None]
            sB_qr_row = sB_qr[None, sB_row_w, None]
            cute.copy(atom_cpa_ckv, lane_copy_ckv.partition_S(gB_row), lane_copy_ckv.partition_D(sB_qr_row))

            qn_row = q_nope_full[None, safe_t, head_idx_w, None]
            sB_qn_row = sB_qn[None, sB_row_w, None]
            cute.copy(atom_cpa_kpe, lane_copy_kpe.partition_S(qn_row), lane_copy_kpe.partition_D(sB_qn_row))

            cute.arch.cp_async_commit_group()

        # SGEMM query preload: only for real tokens.
        if warp_idx < T:
            T_idx = warp_idx
            head_idx0 = head_base_idx * 2
            head_idx1 = head_base_idx * 2 + 1

            cute.copy(atom_cpa_qn_score, lane_copy_qn_score.partition_S(q_nope[T_idx, head_idx0, None]), lane_copy_qn_score.partition_D(smem_qr_sgemm[T_idx, 0, None]))
            cute.copy(atom_cpa_qn_score, lane_copy_qn_score.partition_S(q_nope[T_idx, head_idx1, None]), lane_copy_qn_score.partition_D(smem_qr_sgemm[T_idx, 1, None]))
            cute.copy(atom_cpa_qr_score, lane_copy_qr_score.partition_S(q_pe[T_idx, head_idx0, None]), lane_copy_qr_score.partition_D(smem_qn_sgemm[T_idx, 0, None]))
            cute.copy(atom_cpa_qr_score, lane_copy_qr_score.partition_S(q_pe[T_idx, head_idx1, None]), lane_copy_qr_score.partition_D(smem_qn_sgemm[T_idx, 1, None]))

            cute.arch.cp_async_commit_group()
            
        # 2. UMMA & SGEMM worker split assignment
        # Vectorized loads from sparse_indices to smem_sp_indices, swizzled to reduce split 0 compute pressure
        # Calculate num_valid, if num_valid = 0 -> OOB, skip, if num_valid = 128 -> UMMA, else -> SGEMM
        sparse_indices_ = cute.zipped_divide(sparse_indices, (1, self.sp_vec_size_i32)) # ((1,4), (T,2048//4))
        smem_sp_indices_ = cute.zipped_divide(smem_sp_indices, (1, self.sp_vec_size_i32)) # ((1,4), (8, 128//32)) -> 1 warp
        if DIM_CHUNK <= warp_idx < DIM_CHUNK + T:
            warp_idx_sgemm = warp_idx - DIM_CHUNK 
            split_idx_new = (split_idx_old + self.swz_rot_shift) % NUM_SPLITS
            si_vec = sparse_indices_[(0, None), (warp_idx_sgemm, split_idx_new * DIM_SPLIT + lane_idx)].load()

            num_valid_partial = 0
            for v in range(self.sp_vec_size_i32):
                val = si_vec[v]
                if 0 <= val < FLAT_CACHE:
                    num_valid_partial += 1
                else:
                    val = 0
                smem_sp_indices_[(0, v), (warp_idx_sgemm, lane_idx)] = val
            
            num_valid = warp_reduce(num_valid_partial, lambda a, b: a + b, width=self.wsize)
            smem_assign[warp_idx_sgemm, 0] = split_idx_new
            smem_assign[warp_idx_sgemm, 1] = num_valid
                
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # SGEMM score vector views (same idea as test_score)
        # All zipped_divide must be hoisted above any dynamic if-branch (CuTe DSL constraint)
        smem_qr_sgemm_z = cute.zipped_divide(smem_qr_sgemm, (1, 1, self.sgemm_ckv_vec))
        smem_qn_sgemm_z = cute.zipped_divide(smem_qn_sgemm, (1, 1, self.sgemm_kpe_vec))
        ckv_flat_z = cute.zipped_divide(ckv_flat, (1, self.sgemm_ckv_vec))
        kpe_flat_z = cute.zipped_divide(kpe_flat, (1, self.sgemm_kpe_vec))
        ckv_flat_out = cute.zipped_divide(ckv_flat, (1, self.out_vec))  # ((1,out_vec), (FLAT_CACHE, 512//out_vec))
        smem_logits_flat_ = cute.zipped_divide(smem_logits_flat, (HEADS_PER_SPLIT,))
        smem_score_sgemm_ = cute.zipped_divide(smem_score_sgemm, (1, DIM_SPLIT // self.wsize))
        smem_partial_sgemm_ = cute.zipped_divide(smem_partial_sgemm, (1, 1, self.out_vec))

        # UMMA output GEMV: load 2 bf16 per thread per k-block from smem sA_ckv
        atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.ab_dtype, num_bits_per_copy=32)
        OUT_VEC_PER_KO:  cutlass.Constexpr = 2
        N_KO_PER_STAGE:  cutlass.Constexpr = (HEAD_DIM_CKV // self.k_tile) // self.out_stages  # = 2
        OUT_VEC_TOTAL_UMMA: cutlass.Constexpr = OUT_VEC_PER_KO * N_KO_PER_STAGE               # = 4
        thr_layout_out = cute.make_layout((32,), stride=(1,))
        val_layout_out = cute.make_layout((OUT_VEC_PER_KO,), stride=(1,))
        tiled_copy_out = cute.make_tiled_copy_tv(atom_s2r, thr_layout_out, val_layout_out)
        lane_copy_out  = tiled_copy_out.get_slice(lane_idx)

        # 3. Warp specialization: 
        # First 256 threads -> UMMA on num_valid = 128
        # Last 512 threads -> SGEMM on 0 < num_valid < 128
        
        # UMMA workers
        if warp_idx < self.num_umma_warps:
            umma_warp_idx = warp_idx   # 0..num_umma_warps-1
            umma_tidx     = tidx       # 0..umma_threads-1

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    split_idx_new = smem_assign[T_idx, 0]
                    num_valid     = smem_assign[T_idx, 1]
                    if num_valid == DIM_SPLIT:
                        # ── Load sA: each warp loads (DIM_SPLIT // num_umma_warps) rows ──
                        num_rounds_load: cutlass.Constexpr = DIM_SPLIT // self.num_umma_warps  # 16


        # SGEMM workers
        else:
            warp_idx_sgemm = warp_idx - self.num_umma_warps
            for T_idx in range(T):
                split_idx_new = smem_assign[T_idx, 0]
                num_valid = smem_assign[T_idx, 1]
                if 0 < num_valid < DIM_SPLIT:
                    
                    # Phase 1: Score computation
                    
                    # 1 warp handles 1 column, round robin
                    num_rounds  = (num_valid + self.num_sgemm_warps - 1) // self.num_sgemm_warps
                    for round_idx in range(num_rounds):
                        col_idx = round_idx * self.num_sgemm_warps + warp_idx_sgemm
                        
                        if col_idx < num_valid:
                            flat_cache_idx = smem_sp_indices[T_idx, col_idx]

                            acc0 = cutlass.Float32(0)
                            acc1 = cutlass.Float32(0)

                            for i in range(HEAD_DIM_CKV // (self.sgemm_ckv_vec * self.wsize)):
                                row_idx = i * self.wsize + lane_idx
                                qr0_frag = smem_qr_sgemm_z[(0, 0, None), (T_idx, 0, row_idx)].load().to(cutlass.Float32)
                                qr1_frag = smem_qr_sgemm_z[(0, 0, None), (T_idx, 1, row_idx)].load().to(cutlass.Float32)
                                ckv_frag = ckv_flat_z[(0, None), (flat_cache_idx, row_idx)].load().to(cutlass.Float32)
                                for v in range(self.sgemm_ckv_vec):
                                    acc0, acc1 = cute.arch.fma_packed_f32x2(
                                        (qr0_frag[v], qr1_frag[v]), 
                                        (ckv_frag[v], ckv_frag[v]), 
                                        (acc0, acc1))

                            for i in range(HEAD_DIM_KPE // (self.sgemm_kpe_vec * self.wsize)):
                                row_idx = i * self.wsize + lane_idx
                                qn0_frag = smem_qn_sgemm_z[(0, 0, None), (T_idx, 0, row_idx)].load().to(cutlass.Float32)
                                qn1_frag = smem_qn_sgemm_z[(0, 0, None), (T_idx, 1, row_idx)].load().to(cutlass.Float32)
                                kpe_frag = kpe_flat_z[(0, None), (flat_cache_idx, row_idx)].load().to(cutlass.Float32)
                                for v in range(self.sgemm_kpe_vec):
                                    acc0, acc1 = cute.arch.fma_packed_f32x2(
                                        (qn0_frag[v], qn1_frag[v]), 
                                        (kpe_frag[v], kpe_frag[v]), 
                                        (acc0, acc1))

                            acc0 = warp_reduce(acc0, lambda a, b: a + b)
                            acc1 = warp_reduce(acc1, lambda a, b: a + b)

                            if lane_idx == 0:
                                smem_score_sgemm[0, col_idx] = acc0 * cutlass.Float32(sm_scale)
                                smem_score_sgemm[1, col_idx] = acc1 * cutlass.Float32(sm_scale)
                    cute.arch.barrier(barrier_id=self.sgemm_bar_id, 
                                      number_of_threads=self.sgemm_threads)            
                    
                    # Phase 2: Softmax
                    
                    # 1 warp handles 1 head                    
                    if warp_idx_sgemm < HEADS_PER_SPLIT:
                        num_elems = DIM_SPLIT // self.wsize  # = 4
                        vec = smem_score_sgemm_[(0, None), (warp_idx_sgemm, lane_idx)].load()
                        
                        # Build masked vec in rmem (invalid entries = -inf) to avoid
                        # scalar mutation inside dynamic if (breaks CuTe DSL SSA dominance)
                        vec_masked = cute.make_rmem_tensor(
                            cute.make_layout((num_elems,), stride=(1,)), cutlass.Float32)
                        for v_idx in range(num_elems):
                            vec_masked[v_idx] = -cutlass.Float32(math.inf)  # tensor store: OK
                        for v_idx in range(num_elems):
                            col_idx = lane_idx * num_elems + v_idx
                            if col_idx < num_valid:
                                vec_masked[v_idx] = vec[v_idx]  # conditional tensor store: OK
                        
                        # Max reduction - no inner conditional, scalar mutation only in for body
                        row_max = -cutlass.Float32(math.inf)
                        for v_idx in range(num_elems):
                            row_max = cute.arch.fmax(row_max, vec_masked[v_idx])

                        row_max = warp_reduce(row_max, cute.arch.fmax)                                    
                        if lane_idx == 0:
                            smem_max[warp_idx_sgemm] = row_max
                        cute.arch.barrier(barrier_id=self.sgemm_max_red_bar_id,
                                          number_of_threads=HEADS_PER_SPLIT * self.wsize)
                        # Find row sum - no inner conditional, invalid entries: exp(-inf) = 0
                        row_sum = cutlass.Float32(0)
                        row_max = smem_max[warp_idx_sgemm]
                        for v_idx in range(num_elems):
                            col_idx = lane_idx * num_elems + v_idx
                            e = cute.math.exp(vec_masked[v_idx] - row_max)
                            row_sum += e  # scalar mutation in for body only: OK
                            smem_logits_flat[col_idx * 2 + warp_idx_sgemm] = e  # tensor store: OK
                        row_sum = warp_reduce(row_sum, lambda a, b: a + b)        
                        if lane_idx == 0:
                            smem_sum[warp_idx_sgemm] = row_sum
                    
                        # Store LSE to GMEM
                        if lane_idx == 0:
                            head_base_idx, split_idx_old, _ = cute.arch.block_idx()
                            head_idx_global = head_base_idx * 2 + warp_idx_sgemm
                            partial_lse[T_idx, split_idx_new, head_idx_global, 0] = row_max
                            partial_lse[T_idx, split_idx_new, head_idx_global, 1] = row_sum
                    
                    cute.arch.barrier(barrier_id=self.sgemm_bar_id, 
                                      number_of_threads=self.sgemm_threads)
                    
                    # Phase 3: Output
                    
                    # 1 warp handles 1/num_warps partial sum of the output (2,512)
                    out0 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                    out1 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)

                    # 16 warps, round robin partial sum
                    
                    num_rounds  = (num_valid + self.num_sgemm_warps - 1) // self.num_sgemm_warps
                    
                    for stage_idx in range(self.out_stages):
                        out0.fill(cutlass.Float32(0))
                        out1.fill(cutlass.Float32(0))
                        
                        # GEMV round robin warp per row                      
                        for round_idx in range(num_rounds):
                            k = round_idx * self.num_sgemm_warps + warp_idx_sgemm
                            if k < num_valid:
                                flat_cache_idx = smem_sp_indices[T_idx, k]
                                # stage_idx * wsize + lane_idx → chunk index within row (512//out_vec total chunks)
                                gmem_ckv_vec = ckv_flat_out[(0, None), (flat_cache_idx, stage_idx * self.wsize + lane_idx)].load().to(cutlass.Float32) # (out_vec,)
                                smem_logits_vec = smem_logits_flat_[(None), (k)].load() # (2,)
                                
                                for v_idx in range(self.out_vec):
                                    out0[v_idx], out1[v_idx] = (
                                        cute.arch.fma_packed_f32x2(
                                            (smem_logits_vec[0], smem_logits_vec[1]),
                                            (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                            (out0[v_idx], out1[v_idx])
                                        )
                                    )

                        # Store partial result to stage-size smem
                        smem_partial_sgemm_[(0, 0, None), (warp_idx_sgemm, 0, lane_idx)].store(out0.load())
                        smem_partial_sgemm_[(0, 0, None), (warp_idx_sgemm, 1, lane_idx)].store(out1.load())
                        
                        cute.arch.barrier(barrier_id=self.sgemm_bar_id, 
                                        number_of_threads=self.sgemm_threads)
                        
                        # Final reduction and store to gmem
                        # TODO: after swizzle, we can consider do warp reduction (width=num_warps) for final stage, still hit 4 way bank conflicts tho, ROI is low
                        
                        # Use 2 thread groups (128 threads/group), one group per head.
                        final_sum = cutlass.Float32(0)
                        tidx_sgemm = tidx - self.umma_threads
                        thr_group_idx = tidx_sgemm // DIM_SPLIT
                        thr_group_lane = tidx_sgemm % DIM_SPLIT
                        
                        if thr_group_idx < HEADS_PER_SPLIT:
                            head_base_idx, split_idx_old, _ = cute.arch.block_idx()
                            head_idx_global = head_base_idx * 2 + thr_group_idx
                            out_col = stage_idx * DIM_SPLIT + thr_group_lane
                            for i in range(self.num_sgemm_warps):
                                final_sum += smem_partial_sgemm[i, thr_group_idx, thr_group_lane]

                            # Store results to gmem
                            partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum
                                
                        cute.arch.barrier(barrier_id=self.sgemm_bar_id, 
                                        number_of_threads=self.sgemm_threads)
        
        # Epilogue: all threads converge before releasing/deallocating TMEM.
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

    @cute.kernel
    def reduce_kernel(
        self,
        sparse_indices: cute.Tensor,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        T_idx, head_idx, _ = cute.arch.block_idx()

        # Count valid indices for this token to determine active split count.
        alloc = cutlass.utils.SmemAllocator()
        smem_red_i32 = self._smem(alloc, cutlass.Int32, (32,), (1,))
        smem_reduce_f32 = self._smem(alloc, cutlass.Float32, (2,), (1,))

        partial_cnt = cutlass.Int32(0)
        for i in range(tidx, TOP_K, self.reduce_threads):
            idx = sparse_indices[T_idx, i]
            if idx >= cutlass.Int32(0):
                partial_cnt += cutlass.Int32(1)

        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = cnt_sum
        cute.arch.sync_threads()

        if warp_idx == 0:
            val = cutlass.Int32(0)
            if lane_idx < self.reduce_warps:
                val = smem_red_i32[lane_idx]
            val = warp_reduce(val, lambda a, b: a + b, width=self.reduce_warps)
            if lane_idx == 0:
                smem_red_i32[0] = val
        cute.arch.sync_threads()

        num_valid = smem_red_i32[0]
        if num_valid <= 0:
            if tidx < HEAD_DIM_CKV:
                output[T_idx, head_idx, tidx] = cutlass.BFloat16(0)
            if tidx == 0:
                lse[T_idx, head_idx] = -cutlass.Float32(math.inf)
        else:
            if tidx == 0:
                # Exact active-split detection from sparse indices to avoid
                # assumptions about contiguous split occupancy.
                for s in range(NUM_SPLITS):
                    cnt = cutlass.Int32(0)
                    base = s * DIM_SPLIT
                    for i in range(DIM_SPLIT):
                        if sparse_indices[T_idx, base + i] >= cutlass.Int32(0):
                            cnt += cutlass.Int32(1)
                    smem_red_i32[s] = cnt

                g_max = -cutlass.Float32(math.inf)
                for s in range(NUM_SPLITS):
                    if smem_red_i32[s] > cutlass.Int32(0):
                        local_max = partial_lse[T_idx, s, head_idx, 0]
                        if local_max > g_max:
                            g_max = local_max

                g_lse_sum = cutlass.Float32(0)
                for s in range(NUM_SPLITS):
                    if smem_red_i32[s] > cutlass.Int32(0):
                        l_max = partial_lse[T_idx, s, head_idx, 0]
                        l_sum = partial_lse[T_idx, s, head_idx, 1]
                        g_lse_sum += l_sum * cute.math.exp(l_max - g_max)

                smem_reduce_f32[0] = g_max
                smem_reduce_f32[1] = g_lse_sum
                lse[T_idx, head_idx] = (g_max + cute.math.log(g_lse_sum)) / cutlass.Float32(LN2)

            cute.arch.sync_threads()

            g_max = smem_reduce_f32[0]
            g_lse_sum = smem_reduce_f32[1]

            partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, self.vec_reduce))
            output_v = cute.zipped_divide(output, (1, 1, self.vec_reduce))

            if tidx < (HEAD_DIM_CKV // self.vec_reduce):
                acc_rmem = cute.make_rmem_tensor(cute.make_layout((self.vec_reduce,), stride=(1,)), cutlass.Float32)
                acc_rmem[0] = cutlass.Float32(0)
                acc_rmem[1] = cutlass.Float32(0)
                acc = acc_rmem.load()

                for s in range(NUM_SPLITS):
                    if smem_red_i32[s] > cutlass.Int32(0):
                        l_max = partial_lse[T_idx, s, head_idx, 0]
                        scale = cute.math.exp(l_max - g_max)
                        p = partial_out_v[(0, 0, 0, None), (T_idx, s, head_idx, tidx)].load()
                        acc = acc + scale * p

                out_val = acc / g_lse_sum
                output_v[(0, 0, None), (T_idx, head_idx, tidx)].store(out_val.to(cutlass.BFloat16))
                                                                        
                            
                            
                                                                                                                    
                                        
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_hybrid():
    T = cute.sym_int()
    q_nope         = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_KPE), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (NUM_PAGES, PAGE_SIZE, HEAD_DIM_CKV), (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (NUM_PAGES, PAGE_SIZE, HEAD_DIM_KPE), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K), (1, 0), 4)
    sm_scale       = SM_SCALE
    partial_out    = _fake(cute.Float32,  (LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, HEAD_DIM_CKV), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, NUM_HEADS, HEAD_DIM_CKV), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, NUM_HEADS), (1, 0), 4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    hybrid = Dsa()

    compiled = cute.compile(
        hybrid,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, stream,
        options="--enable-tvm-ffi"
    )
    return hybrid, compiled


_hybrid, _compiled = compile_hybrid()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    _hybrid.partial_out.zero_()
    _hybrid.partial_lse[:, :, :, 0].fill_(-float("inf"))
    _hybrid.partial_lse[:, :, :, 1].zero_()
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
              _hybrid.partial_out, _hybrid.partial_lse, output, lse)    
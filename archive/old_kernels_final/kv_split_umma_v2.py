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
def _panel_copy_layout(num_rows: int, k_packed: int, k_tiles: int):
    return cute.make_layout((num_rows, (k_packed, k_tiles)),
                            stride=(k_packed, (1, num_rows * k_packed)),)


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

class Dsa():
    def __init__(self):
        self.wsize = cute.arch.WARP_SIZE
                
        # KV split compute kernel
        # Prologue
        self.swz_rot_shift = 7
        self.sp_vec_size_i32 = 4
        self.out_stages = 4
        self.out_vec = HEAD_DIM_CKV // (self.out_stages * self.wsize) # 4
                
        # UMMA: KV @ Q.T = SCORE.T
        self.umma_threads = 256
        self.num_umma_warps = self.umma_threads // self.wsize
        self.umma_inst = (DIM_SPLIT, 8, 16) # (128, 8, 16) - Opt4: N=8, only 2 cols used
        self.tmem_ld_rep = HEADS_PER_SPLIT
        self.ab_dtype  = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.sgemm_threads = DIM_CHUNK * self.wsize  # 8 warps * 32 = 256
        self.umma_bar_id = 2          # named barrier: all UMMA threads sync
        self.umma_max_red_bar_id = 3  # named barrier: 2 head-warps sync after max write

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

        @cute.struct
        class SharedStorage:
            umma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.compute_kernel(
            tiled_mma,
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
        tiled_mma,
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
        
        # ========= SMEM setup =========
        alloc = cutlass.utils.SmemAllocator()
        
        # SMEM for prologue
        smem_sp_indices = self._smem(alloc, cutlass.Int32, (DIM_CHUNK, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_assign = self._smem(alloc, cutlass.Int32, (DIM_CHUNK, 2),(2,1))
        # Score output: (HEADS_PER_SPLIT, DIM_SPLIT) f32 — written per T_idx iteration
        smem_score = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT, DIM_SPLIT), (DIM_SPLIT, 1))
        # Softmax intermediates
        smem_max          = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT,), (1,))
        smem_sum          = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT,), (1,))
        smem_logits_flat  = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT * DIM_SPLIT,), (1,))
        # Output partial sums: (num_umma_warps, HEADS_PER_SPLIT, HEAD_DIM_CKV // out_stages) f32
        smem_partial_umma = self._smem(alloc, cutlass.Float32,
                              (self.num_umma_warps, HEADS_PER_SPLIT, HEAD_DIM_CKV // self.out_stages),
                              (HEADS_PER_SPLIT * (HEAD_DIM_CKV // self.out_stages), HEAD_DIM_CKV // self.out_stages, 1))

        # ── UMMA smem: 9-panel layout (panels 0-7=CKV/Q_nope, panel 8=KPE/Q_pe) ──
        swizzle    = cute.make_swizzle(3, 4, 3)
        _MK_PACK   = 4
        _MK_PACKED = 64                               # MMA_K(16) * 4
        _MK_TILES     = HEAD_DIM_CKV // _MK_PACKED   # = 8
        _MK_TILES_PE  = HEAD_DIM_KPE  // _MK_PACKED  # = 1
        _MK_TILES_FULL = _MK_TILES + _MK_TILES_PE    # = 9
        _MMA_M = DIM_SPLIT                            # = 128
        _MMA_N = 8                                    # Opt4: was HEADS_PER_SPLIT*DIM_CHUNK=16
        _MMA_K = 16
        _MMA_M_PACK, _MMA_N_PACK = 1, 1
        a_outer = cute.make_layout(
            ((_MMA_M, _MMA_K), _MMA_M_PACK, (_MK_PACK, _MK_TILES_FULL)),
            stride=((_MK_PACKED, 1), 0, (_MMA_K, _MMA_M * _MK_PACKED)),
        )
        b_outer = cute.make_layout(
            ((_MMA_N, _MMA_K), _MMA_N_PACK, (_MK_PACK, _MK_TILES_FULL)),
            stride=((_MK_PACKED, 1), 0, (_MMA_K, _MMA_N * _MK_PACKED)),
        )
        sA = alloc.allocate_tensor(
            element_type=cutlass.BFloat16, layout=a_outer,
            byte_alignment=16, swizzle=swizzle,
        )
        sB = alloc.allocate_tensor(
            element_type=cutlass.BFloat16, layout=b_outer,
            byte_alignment=16, swizzle=swizzle,
        )
        # Copy views: panels 0-7 (base pointer) and panel 8 (offset pointer)
        sA_ckv_copy = cute.make_tensor(sA.iterator, _panel_copy_layout(_MMA_M, _MK_PACKED, _MK_TILES))
        sB_ckv_copy = cute.make_tensor(sB.iterator, _panel_copy_layout(_MMA_N, _MK_PACKED, _MK_TILES))
        panel_stride_A = _MMA_M * _MK_PACKED * _MK_TILES  # = 65536 elems
        panel_stride_B = _MMA_N * _MK_PACKED * _MK_TILES  # = 8192  elems
        sA_kpe_copy = cute.make_tensor(sA.iterator + panel_stride_A, _panel_copy_layout(_MMA_M, _MK_PACKED, _MK_TILES_PE))
        sB_kpe_copy = cute.make_tensor(sB.iterator + panel_stride_B, _panel_copy_layout(_MMA_N, _MK_PACKED, _MK_TILES_PE))
        # composition shapes for rank-1 row slices
        k_split_shape    = cute.make_layout(((_MK_PACKED, _MK_TILES),))
        k_split_shape_pe = cute.make_layout(((_MK_PACKED, _MK_TILES_PE),))
        # cp.async tiled copy — CKV/Q_nope: 128-bit, 32×8=256 elems/step
        atom_cpa   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout = cute.make_layout(((8, 4),), stride=((1, 8),))
        val_layout = cute.make_layout(((8, 1),), stride=((1, 0),))
        tiled_copy = cute.make_tiled_copy_tv(atom_cpa, thr_layout, val_layout)
        lane_copy  = tiled_copy.get_slice(lane_idx)
        # cp.async tiled copy — KPE/Q_pe: 32-bit, 32×2=64 elems/step
        atom_cpa_pe   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=32)
        val_layout_pe = cute.make_layout(((2, 1),), stride=((1, 0),))
        tiled_copy_pe = cute.make_tiled_copy_tv(atom_cpa_pe, thr_layout, val_layout_pe)
        lane_copy_pe  = tiled_copy_pe.get_slice(lane_idx)
        # SharedStorage (mbarrier + tmem holding buffer)
        storage  = alloc.allocate(self.shared_storage)
        mma_mbar = storage.umma_mbar_ptr.data_ptr()

        # ========= Prologue - Swizzle split with wrapped-around rotation =========                  
        head_base_idx, split_idx_old, _ = cute.arch.block_idx() # 0 -> 7, 0 -> 15, _
        
        T, _, _ = q_nope.shape
            
        # Work split assignment
        # Vectorized loads from sparse_indices to smem_sp_indices, swizzled to reduce split 0 compute pressure
        sparse_indices_ = cute.zipped_divide(sparse_indices, (1, self.sp_vec_size_i32)) # ((1,4), (T,2048//4))
        smem_sp_indices_ = cute.zipped_divide(smem_sp_indices, (1, self.sp_vec_size_i32)) # ((1,4), (8, 128//32)) -> 1 warp
        if DIM_CHUNK <= warp_idx < DIM_CHUNK + T:
            warp_idx_sgemm = warp_idx - DIM_CHUNK
            # Per-request rotation: each T_idx sees a different split assignment.
            split_idx_new = (split_idx_old + warp_idx_sgemm * self.swz_rot_shift) % cutlass.Int32(NUM_SPLITS)

            # Formula 2, emprically better for these workloads
            # _t = warp_idx_sgemm  # T_idx in [0, 7]
            # split_idx_new = ((split_idx_old ^ _t) + _t * cutlass.Int32(14) % cutlass.Int32(NUM_SPLITS)) % cutlass.Int32(NUM_SPLITS)
            # sparse_indices_ second-mode coord indexes vec-chunks (each = sp_vec_size_i32 source elems).
            # A split spans DIM_SPLIT source elems = (DIM_SPLIT // sp_vec_size_i32) vec-chunks.
            split_vec_stride = DIM_SPLIT // self.sp_vec_size_i32  # = 32
            si_vec = sparse_indices_[(0, None), (warp_idx_sgemm, split_idx_new * split_vec_stride + lane_idx)].load()

            num_valid_partial = 0
            for v in range(self.sp_vec_size_i32):
                val = si_vec[v]
                if 0 <= val < FLAT_CACHE:
                    num_valid_partial += 1
                else:
                    val = 0
                smem_sp_indices_[(0, v), (warp_idx_sgemm, lane_idx)] = val
            
            num_valid = warp_reduce(num_valid_partial, lambda a, b: a + b, width=self.wsize)
            if lane_idx == 0:
                smem_assign[warp_idx_sgemm, 0] = split_idx_new
                smem_assign[warp_idx_sgemm, 1] = num_valid

        cute.arch.sync_threads()

        # ── tmem / mbarrier setup (all threads participate before UMMA guard) ──
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape       = tiled_mma.partition_shape_C((_MMA_M, _MMA_N))
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)
        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        M_acc           = cute.size(tCtAcc, mode=[0, 0])
        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler       = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi      = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, cutlass.Float32)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, cutlass.Float32)

        # Hoist zipped_divide views above dynamic branches (CuTe DSL constraint)
        smem_score_        = cute.zipped_divide(smem_score,        (1, DIM_SPLIT // self.wsize))
        smem_logits_flat_  = cute.zipped_divide(smem_logits_flat,  (HEADS_PER_SPLIT,))
        smem_partial_umma_ = cute.zipped_divide(smem_partial_umma, (1, 1, self.out_vec))
        ckv_flat_out       = cute.zipped_divide(ckv_flat,          (1, self.out_vec))  # ((1,out_vec), (FLAT_CACHE, 512//out_vec))
        sA_ckv_out         = cute.zipped_divide(sA_ckv_copy,       (1, self.out_vec))  # ((1,out_vec), (DIM_SPLIT, 512//out_vec))

        # UMMA workers
        if warp_idx < self.num_umma_warps:
            umma_warp_idx = warp_idx
            umma_tidx     = tidx
            num_rounds    = DIM_SPLIT // self.num_umma_warps  # = 16 rows per warp
            # Track mbarrier phase across only the iterations that actually arrive.
            # Skipping commits (when num_valid==0) must NOT advance the phase, otherwise
            # subsequent waits would observe a stale/wrong phase and either hang or
            # race with the next MMA.
            mma_phase = cutlass.Int32(0)

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    split_idx_new = smem_assign[T_idx, 0]
                    num_valid     = smem_assign[T_idx, 1]

                    if num_valid > 0:
                        # Per-iter Q load: 2 heads → sB rows 0,1
                        if umma_warp_idx < HEADS_PER_SPLIT:
                            head_h = head_base_idx * HEADS_PER_SPLIT + umma_warp_idx
                            cute.copy(atom_cpa,
                                      lane_copy.partition_S(cute.composition(q_nope[T_idx, head_h, None], k_split_shape)),
                                      lane_copy.partition_D(sB_ckv_copy[umma_warp_idx, None]))
                            cute.copy(atom_cpa_pe,
                                      lane_copy_pe.partition_S(cute.composition(q_pe[T_idx, head_h, None], k_split_shape_pe)),
                                      lane_copy_pe.partition_D(sB_kpe_copy[umma_warp_idx, None]))

                        # ckv_flat → sA panels 0-7,  kpe_flat → sA panel 8
                        # Opt3: unconditional cp.async — sentinels already zeroed in prologue,
                        # so invalid rows hit flat page idx 0 (L2-cached). Garbage scores get
                        # masked to -inf in softmax (col_idx >= num_valid), so safe.
                        for round_idx in range(num_rounds):
                            sp_idx  = round_idx * self.num_umma_warps + umma_warp_idx
                            row_idx = smem_sp_indices[T_idx, sp_idx]
                            cute.copy(atom_cpa,
                                    lane_copy.partition_S(cute.composition(ckv_flat[row_idx, None], k_split_shape)),
                                    lane_copy.partition_D(sA_ckv_copy[sp_idx, None]))
                            cute.copy(atom_cpa_pe,
                                    lane_copy_pe.partition_S(cute.composition(kpe_flat[row_idx, None], k_split_shape_pe)),
                                    lane_copy_pe.partition_D(sA_kpe_copy[sp_idx, None]))

                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        cute.arch.barrier(barrier_id=self.umma_bar_id,
                                        number_of_threads=self.umma_threads)

                        # ── MMA: single loop over all 9 k-blocks ──────────────────
                        tcgen05_fence()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        if umma_warp_idx == 0:
                            num_k_blocks = cute.size(tCrA, mode=[2])  # = 9
                            for k_block_idx in range(num_k_blocks):
                                k_block_coord = (None, None, k_block_idx)
                                cute.gemm(tiled_mma, tCtAcc,
                                        tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            if umma_tidx == 0:
                                tcgen05.commit(mma_mbar)
                        cute.arch.mbarrier_wait(mma_mbar, mma_phase)
                        mma_phase = mma_phase ^ cutlass.Int32(1)

                        # ── tmem → rmem → smem_score ──────────────────────────────
                        if tidx < DIM_SPLIT:
                            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)
                            smem_score[0, tidx] = tTR_rAcc[0] * cutlass.Float32(sm_scale)
                            smem_score[1, tidx] = tTR_rAcc[1] * cutlass.Float32(sm_scale)

                        # ── Softmax (Phase 2) — 2 warps, one head each, lanes hold 4 regs ──
                        cute.arch.barrier(barrier_id=self.umma_bar_id,
                                        number_of_threads=self.umma_threads)

                        if umma_warp_idx < HEADS_PER_SPLIT:
                            num_elems: cutlass.Constexpr = DIM_SPLIT // self.wsize  # = 4
                            head_idx_global = head_base_idx * HEADS_PER_SPLIT + umma_warp_idx

                            # Load lane's 4 score elements (kept alive across max + sum)
                            vec = smem_score_[(0, None), (umma_warp_idx, lane_idx)].load()

                            # Mask invalid entries with -inf in rmem (no scalar mutation in if)
                            vec_masked = cute.make_rmem_tensor(
                                cute.make_layout((num_elems,), stride=(1,)), cutlass.Float32)
                            for v_idx in range(num_elems):
                                vec_masked[v_idx] = -cutlass.Float32(math.inf)
                            for v_idx in range(num_elems):
                                col_idx = lane_idx * num_elems + v_idx
                                if col_idx < num_valid:
                                    vec_masked[v_idx] = vec[v_idx]

                            # Pass 1: per-lane max over 4 regs, then warp-reduce
                            row_max = -cutlass.Float32(math.inf)
                            for v_idx in range(num_elems):
                                row_max = cute.arch.fmax(row_max, vec_masked[v_idx])
                            row_max = warp_reduce(row_max, cute.arch.fmax)

                            # Pass 2: exp into the same 4 regs, per-lane sum, then warp-reduce
                            row_sum = cutlass.Float32(0)
                            for v_idx in range(num_elems):
                                e = cute.math.exp(vec_masked[v_idx] - row_max)
                                vec_masked[v_idx] = e
                                row_sum += e
                            row_sum = warp_reduce(row_sum, lambda a, b: a + b)

                            # Scatter logits to smem in [pos, head] layout (head fastest)
                            for v_idx in range(num_elems):
                                col_idx = lane_idx * num_elems + v_idx
                                smem_logits_flat[col_idx * HEADS_PER_SPLIT + umma_warp_idx] = vec_masked[v_idx]

                            if lane_idx == 0:
                                partial_lse[T_idx, split_idx_new, head_idx_global, 0] = row_max
                                partial_lse[T_idx, split_idx_new, head_idx_global, 1] = row_sum

                        cute.arch.barrier(barrier_id=self.umma_bar_id,
                                        number_of_threads=self.umma_threads)

                        # ── Output (Phase 3) ──────────────────────────────────
                        num_rounds_out: cutlass.Constexpr = DIM_SPLIT // self.num_umma_warps  # = 16
                        out0 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        out1 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        for stage_idx in range(self.out_stages):
                            out0.fill(cutlass.Float32(0))
                            out1.fill(cutlass.Float32(0))
                            for round_idx in range(num_rounds_out):
                                k = round_idx * self.num_umma_warps + umma_warp_idx
                                if k < num_valid:
                                    # Opt2: read CKV from sA (already loaded for MMA) instead of GMEM
                                    gmem_ckv_vec = sA_ckv_out[(0, None), (k, stage_idx * self.wsize + lane_idx)].load().to(cutlass.Float32)
                                    smem_logits_vec = smem_logits_flat_[(None), (k)].load()
                                    for v_idx in range(self.out_vec):
                                        out0[v_idx], out1[v_idx] = cute.arch.fma_packed_f32x2(
                                            (smem_logits_vec[0], smem_logits_vec[1]),
                                            (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                            (out0[v_idx], out1[v_idx]),
                                        )
                            smem_partial_umma_[(0, 0, None), (umma_warp_idx, 0, lane_idx)].store(out0.load())
                            smem_partial_umma_[(0, 0, None), (umma_warp_idx, 1, lane_idx)].store(out1.load())
                            cute.arch.barrier(barrier_id=self.umma_bar_id, number_of_threads=self.umma_threads)
                            thr_group_idx  = tidx // DIM_SPLIT
                            thr_group_lane = tidx % DIM_SPLIT
                            if thr_group_idx < HEADS_PER_SPLIT:
                                head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                out_col = stage_idx * DIM_SPLIT + thr_group_lane
                                final_sum = cutlass.Float32(0)
                                for i in range(self.num_umma_warps):
                                    final_sum += smem_partial_umma[i, thr_group_idx, thr_group_lane]
                                
                                partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum
                                  
                            cute.arch.barrier(barrier_id=self.umma_bar_id, number_of_threads=self.umma_threads)

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

        alloc = cutlass.utils.SmemAllocator()
        smem_red_i32 = self._smem(alloc, cutlass.Int32,   (32,),          (1,))
        smem_max_sum = self._smem(alloc, cutlass.Float32, (NUM_SPLITS, 2), (2, 1))

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

        num_active_splits = (num_valid + DIM_SPLIT - 1) // DIM_SPLIT

        if tidx < num_active_splits:
            smem_max_sum[tidx, 0] = partial_lse[T_idx, tidx, head_idx, 0]
            smem_max_sum[tidx, 1] = partial_lse[T_idx, tidx, head_idx, 1]

        cute.arch.sync_threads()

        partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, self.vec_reduce))
        output_v      = cute.zipped_divide(output,      (1, 1, self.vec_reduce))

        g_max = -cutlass.Float32(math.inf)
        for s in range(num_active_splits):
            local_max = smem_max_sum[s, 0]
            if local_max > g_max:
                g_max = local_max

        g_lse_sum = cutlass.Float32(0)
        acc_rmem = cute.make_rmem_tensor(cute.make_layout((self.vec_reduce,), stride=(1,)), cutlass.Float32)
        acc_rmem[0] = cutlass.Float32(0)
        acc_rmem[1] = cutlass.Float32(0)
        acc = acc_rmem.load()

        for s in range(num_active_splits):
            l_max = smem_max_sum[s, 0]
            l_sum = smem_max_sum[s, 1]
            scale = cute.math.exp(l_max - g_max)
            g_lse_sum += l_sum * scale

            a = partial_out_v[(0, 0, 0, None), (T_idx, s, head_idx, tidx)].load()
            acc = acc + scale * a

        if tidx == 0:
            lse[T_idx, head_idx] = (g_max + cute.math.log(g_lse_sum)) / cutlass.Float32(LN2)

        output_v[(0, 0, None), (T_idx, head_idx, tidx)].store((acc / g_lse_sum).to(cutlass.BFloat16))


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
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
              _hybrid.partial_out, _hybrid.partial_lse, output, lse)    
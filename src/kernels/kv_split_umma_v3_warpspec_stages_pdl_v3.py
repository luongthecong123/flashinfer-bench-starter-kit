"""kv_split_umma_v3_warpspec_stages_pdl_intra_v6.py

3-actor decoupled pipeline with 6 sA/sB stages.

  - 24 warps (768 threads) per block.
  - LOAD warps:  0..7   (256 threads)  — cp.async loads for PE + ckv chunks.
  - MMA  warps:  8..11  (128 threads)  — only warp 8 issues tcgen05.mma;
      warps 9..11 align with the issuer for warpgroup-style barriers.
  - CONS warps: 12..23  (384 threads)  — score read + softmax + output GEMV.

Slot ring: NUM_CKV_STAGES=6. Token T's chunk c lives in slot (T*4+c)%6.
PE prologue writes 8 token panels into K-tiles 0..7 (= slots 0..3). After
pe_done, the LOAD warps may overwrite slots 0..5 with ckv chunk data.

Mbars:
  - score_mbars[8]            : mma→cons, fired after token T's 4 chunk MMAs.
  - pe_loaded_mbar (cnt=1)    : load→mma, all PE cp.async data visible.
  - pe_done_mbar              : mma tcgen05.commit; load waits before reusing slot.
  - chunk_loaded_mbars[6]     : load→mma, slot s filled.
  - chunk_free_mbars[6]       : cons→load, slot s consumed (bootstrap-arrived).

Trace layout (role): 0=LOAD, 1=CONS, 2=PROLOGUE/MMA.
"""
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
import math, json, torch



@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,)


# ── Constants (mirror kv_split_umma_v3_warpspec_stages_pdl_v2.py) ─────────────
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
DIM_CHUNK = 8
NUM_SPLITS = 16
DIM_SPLIT = (TOP_K + NUM_SPLITS - 1) // NUM_SPLITS
HEADS_PER_SPLIT = 2

_MMA_M, _MMA_N, _MMA_K = DIM_SPLIT, 8, 16
_MMA_K_PACK   = 4
_MMA_K_PACKED = _MMA_K * _MMA_K_PACK
_MMA_K_TILES  = HEAD_DIM_CKV // _MMA_K_PACKED
_MMA_K_TILES_FULL = _MMA_K_TILES

PANELS_PER_CHUNK: cutlass.Constexpr = 2
NUM_CKV_CHUNKS:   cutlass.Constexpr = _MMA_K_TILES // PANELS_PER_CHUNK
CHUNK_PACKED:     cutlass.Constexpr = _MMA_K_PACKED * PANELS_PER_CHUNK
CKV_KBLOCKS_PER_CHUNK: cutlass.Constexpr = _MMA_K_PACK * PANELS_PER_CHUNK
TMEM_COLS_PER_TOKEN = _MMA_N

PROLOGUE_BAR_ID    = 1
LOAD_BAR_ID        = 2
CONS_BAR_ID        = 3
MMA_BAR_ID         = 4

NUM_CKV_STAGES: cutlass.Constexpr = 6
_MMA_K_TILES_RING: cutlass.Constexpr = NUM_CKV_STAGES * PANELS_PER_CHUNK  # 12



@cute.jit
def _panel_copy_layout(num_rows: int, k_packed: int, k_tiles: int):
    return cute.make_layout((num_rows, (k_packed, k_tiles)),
                            stride=(k_packed, (1, num_rows * k_packed)))


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class Dsa():
    def __init__(self):
        self.wsize = cute.arch.WARP_SIZE
        self.swz_rot_shift = 7
        self.sp_vec_size_i32 = 4
        self.out_stages = 4
        self.out_vec = HEAD_DIM_CKV // (self.out_stages * self.wsize)

        self.umma_threads      = 768
        self.num_umma_warps    = self.umma_threads // self.wsize
        self.num_load_warps    = 8
        self.num_mma_warps     = 4
        self.num_cons_warps    = 12
        self.load_threads      = self.num_load_warps * self.wsize  # 256
        self.mma_threads       = self.num_mma_warps  * self.wsize  # 128
        self.cons_threads      = self.num_cons_warps * self.wsize  # 384
        self.umma_inst         = (DIM_SPLIT, 8, 16)
        self.tmem_ld_rep       = HEADS_PER_SPLIT
        self.ab_dtype          = cutlass.BFloat16
        self.acc_dtype         = cutlass.Float32

        self.reduce_threads = 256
        self.reduce_warps   = self.reduce_threads // self.wsize
        self.vec_reduce     = 2

        self.partial_out = torch.zeros(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.float32, device="cuda")
        self.partial_lse = torch.zeros(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, 2,            dtype=torch.float32, device="cuda")

    @cute.jit
    def __call__(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
                 sm_scale: cutlass.Constexpr,
                 partial_out, partial_lse, output, lse, stream):
        T, _, _ = q_nope.shape
        ckv_flat = cute.make_tensor(ckv_cache.iterator,
            cute.make_layout((FLAT_CACHE, HEAD_DIM_CKV), stride=(HEAD_DIM_CKV, 1)))
        kpe_flat = cute.make_tensor(kpe_cache.iterator,
            cute.make_layout((FLAT_CACHE, HEAD_DIM_KPE), stride=(HEAD_DIM_KPE, 1)))

        op = tcgen05.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.umma_inst,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        @cute.struct
        class SharedStorage:
            score_mbars:        cute.struct.MemRange[cutlass.Int64, LIMIT_REQUEST]
            pe_loaded_mbars:    cute.struct.MemRange[cutlass.Int64, 1]
            pe_done_mbars:      cute.struct.MemRange[cutlass.Int64, 1]
            chunk_loaded_mbars: cute.struct.MemRange[cutlass.Int64, NUM_CKV_STAGES]
            chunk_free_mbars:   cute.struct.MemRange[cutlass.Int64, NUM_CKV_STAGES]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.compute_kernel(
            tiled_mma, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
            partial_out, partial_lse, output, lse,
        ).launch(grid=[NUM_HEADS // HEADS_PER_SPLIT, NUM_SPLITS, 1],
                 block=[self.umma_threads, 1, 1],
                 stream=stream, use_pdl=True)

        self.reduce_kernel(
            sparse_indices, partial_out, partial_lse, output, lse,
        ).launch(grid=[T, NUM_HEADS, 1],
                 block=[self.reduce_threads, 1, 1],
                 stream=stream, use_pdl=True)

    @staticmethod
    def _smem(allocator, dtype, shape, stride, byte_alignment=16, swizzle=None):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), byte_alignment, swizzle)

    @cute.kernel
    def compute_kernel(
        self, tiled_mma, q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()

        head_base_idx, split_idx_old, _ = cute.arch.block_idx()

        # ========= SMEM setup =========
        alloc = cutlass.utils.SmemAllocator()

        smem_sp_indices = self._smem(alloc, cutlass.Int32,   (DIM_CHUNK, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_assign     = self._smem(alloc, cutlass.Int32,   (DIM_CHUNK, 2),         (2, 1))
        smem_score        = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_logits_flat       = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT * DIM_SPLIT,), (1,))

        smem_partial_umma = self._smem(alloc, cutlass.Float32,
            (self.num_cons_warps, HEADS_PER_SPLIT, HEAD_DIM_CKV // self.out_stages),
            (HEADS_PER_SPLIT * (HEAD_DIM_CKV // self.out_stages), HEAD_DIM_CKV // self.out_stages, 1))

        swizzle = cute.make_swizzle(3, 4, 3)
        a_outer = cute.make_layout(
            ((_MMA_M, _MMA_K), 1, (_MMA_K_PACK, _MMA_K_TILES_RING)),
            stride=((_MMA_K_PACKED, 1), 0, (_MMA_K, _MMA_M * _MMA_K_PACKED)))
        b_outer = cute.make_layout(
            ((_MMA_N, _MMA_K), 1, (_MMA_K_PACK, _MMA_K_TILES_RING)),
            stride=((_MMA_K_PACKED, 1), 0, (_MMA_K, _MMA_N * _MMA_K_PACKED)))
        sA = alloc.allocate_tensor(cutlass.BFloat16, a_outer, byte_alignment=16, swizzle=swizzle)
        sB = alloc.allocate_tensor(cutlass.BFloat16, b_outer, byte_alignment=16, swizzle=swizzle)
        sA_ckv_copy = cute.make_tensor(sA.iterator, _panel_copy_layout(_MMA_M, _MMA_K_PACKED, _MMA_K_TILES_RING))
        sB_ckv_copy = cute.make_tensor(sB.iterator, _panel_copy_layout(_MMA_N, _MMA_K_PACKED, _MMA_K_TILES_RING))

        panel_stride_A: cutlass.Constexpr = _MMA_M * _MMA_K_PACKED
        panel_stride_B: cutlass.Constexpr = _MMA_N * _MMA_K_PACKED
        chunk_stride_A: cutlass.Constexpr = panel_stride_A * PANELS_PER_CHUNK
        chunk_stride_B: cutlass.Constexpr = panel_stride_B * PANELS_PER_CHUNK

        k_split_shape_chunk = cute.make_layout(((_MMA_K_PACKED, PANELS_PER_CHUNK),))
        k_split_shape_pe    = cute.make_layout(((_MMA_K_PACKED, 1),))

        atom_cpa_chunk128   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout_chunk128 = cute.make_layout(((16, 1),), stride=((1, 16),))
        val_layout_chunk128 = cute.make_layout(((8, 1),),  stride=((1, 0),))
        tiled_copy_chunk128 = cute.make_tiled_copy_tv(atom_cpa_chunk128, thr_layout_chunk128, val_layout_chunk128)
        lane_copy_chunk128  = tiled_copy_chunk128.get_slice(lane_idx % 16)

        atom_cpa_pe128   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout_pe128 = cute.make_layout(((8, 1),), stride=((1, 8),))
        val_layout_pe128 = cute.make_layout(((8, 1),), stride=((1, 0),))
        tiled_copy_pe128 = cute.make_tiled_copy_tv(atom_cpa_pe128, thr_layout_pe128, val_layout_pe128)
        lane_copy_pe128  = tiled_copy_pe128.get_slice(lane_idx % 8)

        sA_ckv_out = cute.zipped_divide(sA_ckv_copy, (1, self.out_vec))

        # Per-slot output view (1 chunk = 2 K-tiles per slot). Used by consumer
        # to read the slot corresponding to (T_idx, stage_idx).
        chunk_out_layout = _panel_copy_layout(_MMA_M, _MMA_K_PACKED, PANELS_PER_CHUNK)

        storage             = alloc.allocate(self.shared_storage)
        score_mbar_base     = storage.score_mbars.data_ptr()
        pe_loaded_mbar      = storage.pe_loaded_mbars.data_ptr()
        pe_done_mbar        = storage.pe_done_mbars.data_ptr()
        chunk_loaded_base   = storage.chunk_loaded_mbars.data_ptr()
        chunk_free_base     = storage.chunk_free_mbars.data_ptr()

        head_base_idx, split_idx_old, _ = cute.arch.block_idx()
        T, _, _ = q_nope.shape

        # ── tmem alloc + mbar init (warp 0) ──
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape       = tiled_mma.partition_shape_C((_MMA_M, _MMA_N))
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        per_token_cols  = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(per_token_cols * LIMIT_REQUEST)

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                for i in range(LIMIT_REQUEST):
                    cute.arch.mbarrier_init(score_mbar_base + i, cnt=1)
                # pe_loaded: load warps fence → lane 0 of warp 0 arrives once.
                cute.arch.mbarrier_init(pe_loaded_mbar, cnt=1)
                # pe_done: tcgen05.commit by mma warp 8.
                cute.arch.mbarrier_init(pe_done_mbar, cnt=1)
                for s in range(NUM_CKV_STAGES):
                    cute.arch.mbarrier_init(chunk_loaded_base + s, cnt=1)
                    cute.arch.mbarrier_init(chunk_free_base + s, cnt=1)
                cute.arch.mbarrier_init_fence()
                # Bootstrap chunk_free phase 0 so producer's first iteration is
                # unblocked for every slot.
                for s in range(NUM_CKV_STAGES):
                    cute.arch.mbarrier_arrive(chunk_free_base + s)
        cute.arch.sync_threads()

        # ── work assignment: cons warps 12..23 (one per token) ──
        sparse_indices_  = cute.zipped_divide(sparse_indices, (1, self.sp_vec_size_i32))
        smem_sp_indices_ = cute.zipped_divide(smem_sp_indices, (1, self.sp_vec_size_i32))
        ASSIGN_BASE_WARP: cutlass.Constexpr = 12
        if ASSIGN_BASE_WARP <= warp_idx < ASSIGN_BASE_WARP + T:
            warp_idx_assign = warp_idx - ASSIGN_BASE_WARP
            split_idx_new = (split_idx_old + warp_idx_assign * self.swz_rot_shift) % cutlass.Int32(NUM_SPLITS)
            split_vec_stride = DIM_SPLIT // self.sp_vec_size_i32
            si_vec = sparse_indices_[(0, None), (warp_idx_assign, split_idx_new * split_vec_stride + lane_idx)].load()
            num_valid_partial = 0
            for v in range(self.sp_vec_size_i32):
                val = si_vec[v]
                if 0 <= val < FLAT_CACHE:
                    num_valid_partial += 1
                smem_sp_indices_[(0, v), (warp_idx_assign, lane_idx)] = val
            num_valid = warp_reduce(num_valid_partial, lambda a, b: a + b, width=self.wsize)
            if lane_idx == 0:
                smem_assign[warp_idx_assign, 0] = split_idx_new
                smem_assign[warp_idx_assign, 1] = num_valid
        cute.arch.sync_threads()

        # ── PDL: assignment is committed; let the reduce kernel begin its
        #     launch + sparse_indices count overlapped with the rest of compute.
        cute.arch.griddepcontrol_launch_dependents()
        first_valid_T = cutlass.Int32(-1)
        total_valid   = cutlass.Int32(0)
        for i in range(LIMIT_REQUEST):
            if i < T:
                nv = smem_assign[i, 1]
                total_valid += nv
                if (first_valid_T < cutlass.Int32(0)) & (nv > cutlass.Int32(0)):
                    first_valid_T = cutlass.Int32(i)
        has_valid = total_valid > cutlass.Int32(0)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf)

        # Hoist: tmem region for first_valid_T is known immediately after assignment.
        tCtAcc_ff = cute.make_tensor(
            tmem_ptr + first_valid_T * cutlass.Int32(TMEM_COLS_PER_TOKEN),
            tCtAcc_tmpl.layout)

        tCtAcc_base     = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        M_acc           = cute.size(tCtAcc_base, mode=[0, 0])
        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler       = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi_base = cute.zipped_divide(tCtAcc_base, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, cutlass.Float32)
        cons_base_thr   = self.load_threads + self.mma_threads  # 384
        cons_tidx       = tidx - cons_base_thr
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi_base[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(cons_tidx)
        tTR_tAcc_base   = tmem_thr_copy.partition_S(tCtAcc_epi_base)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc_base[None, None, 0].shape, cutlass.Float32)

        smem_score_              = cute.zipped_divide(smem_score,              (1, DIM_SPLIT // self.wsize))
        smem_logits_flat_        = cute.zipped_divide(smem_logits_flat,        (HEADS_PER_SPLIT,))
        smem_partial_umma_  = cute.zipped_divide(smem_partial_umma,  (1, 1, self.out_vec))

        cons_base_warp: cutlass.Constexpr = 12
        is_load = warp_idx < self.num_load_warps
        is_mma  = (warp_idx >= self.num_load_warps) & (warp_idx < self.num_load_warps + self.num_mma_warps)
        is_consumer = warp_idx >= cons_base_warp

        mma_tidx     = tidx - self.load_threads          # 0..127 within mma group
        mma_warp_idx = warp_idx - self.num_load_warps    # 0..3

        # ============================================================
        # LOAD warps (warps 0..7, 256 thr)
        #   PE cp.async loads (q_pe, kpe) → signal pe_loaded → wait pe_done →
        #   per-(T,c) chunk cp.async loads into sA/sB ring slots.
        # ============================================================
        if is_load and has_valid:
            kpe_rows_per_group:  cutlass.Constexpr = _MMA_M // 4   # 32
            pe_row_group = lane_idx // 8                            # 0..3

            for i in cutlass.range_constexpr(LIMIT_REQUEST):
                if i < T:
                    num_valid_pe = smem_assign[i, 1]
                    if num_valid_pe > 0:
                        sA_pe_i = cute.make_tensor(
                            sA.iterator + i * panel_stride_A,
                            _panel_copy_layout(_MMA_M, _MMA_K_PACKED, 1))
                        sB_pe_i = cute.make_tensor(
                            sB.iterator + i * panel_stride_B,
                            _panel_copy_layout(_MMA_N, _MMA_K_PACKED, 1))

                        if warp_idx == i:
                            qpe_row = lane_idx // 8
                            if qpe_row < HEADS_PER_SPLIT:
                                head_h = head_base_idx * HEADS_PER_SPLIT + qpe_row
                                cute.copy(atom_cpa_pe128,
                                          lane_copy_pe128.partition_S(cute.composition(q_pe[i, head_h, None], k_split_shape_pe)),
                                          lane_copy_pe128.partition_D(sB_pe_i[qpe_row, None]))

                            for r in range(kpe_rows_per_group):
                                row_idx  = r * 4 + pe_row_group
                                if row_idx < num_valid_pe:
                                    flat_row = smem_sp_indices[i, row_idx]
                                    cute.copy(atom_cpa_pe128,
                                              lane_copy_pe128.partition_S(cute.composition(kpe_flat[flat_row, None], k_split_shape_pe)),
                                              lane_copy_pe128.partition_D(sA_pe_i[row_idx, None]))

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=PROLOGUE_BAR_ID, number_of_threads=self.load_threads)

            if tidx == 0:
                cute.arch.mbarrier_arrive(pe_loaded_mbar)

            # Wait for mma warp to retire PE UMMAs before reusing sA slots 0..3.
            cute.arch.mbarrier_wait(pe_done_mbar, cutlass.Int32(0))

            # ---- Steady-state: per-(T,c) chunk loads into ring slots. ----
            load_row_group = warp_idx * 2 + (lane_idx // 16)         # 0..15
            LOAD_ROW_GROUPS: cutlass.Constexpr = 16
            num_rounds = DIM_SPLIT // LOAD_ROW_GROUPS                 # 8

            free_p0 = cutlass.Int32(0); free_p1 = cutlass.Int32(0)
            free_p2 = cutlass.Int32(0); free_p3 = cutlass.Int32(0)
            free_p4 = cutlass.Int32(0); free_p5 = cutlass.Int32(0)

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    num_valid = smem_assign[T_idx, 1]
                    if num_valid > 0:
                        for c in cutlass.range_constexpr(NUM_CKV_CHUNKS):
                            slot: cutlass.Constexpr = (T_idx * NUM_CKV_CHUNKS + c) % NUM_CKV_STAGES

                            if slot == 0:
                                cute.arch.mbarrier_wait(chunk_free_base + 0, free_p0)
                                free_p0 = free_p0 ^ cutlass.Int32(1)
                            elif slot == 1:
                                cute.arch.mbarrier_wait(chunk_free_base + 1, free_p1)
                                free_p1 = free_p1 ^ cutlass.Int32(1)
                            elif slot == 2:
                                cute.arch.mbarrier_wait(chunk_free_base + 2, free_p2)
                                free_p2 = free_p2 ^ cutlass.Int32(1)
                            elif slot == 3:
                                cute.arch.mbarrier_wait(chunk_free_base + 3, free_p3)
                                free_p3 = free_p3 ^ cutlass.Int32(1)
                            elif slot == 4:
                                cute.arch.mbarrier_wait(chunk_free_base + 4, free_p4)
                                free_p4 = free_p4 ^ cutlass.Int32(1)
                            else:
                                cute.arch.mbarrier_wait(chunk_free_base + 5, free_p5)
                                free_p5 = free_p5 ^ cutlass.Int32(1)

                            sA_slot = cute.make_tensor(
                                sA.iterator + slot * chunk_stride_A,
                                _panel_copy_layout(_MMA_M, _MMA_K_PACKED, PANELS_PER_CHUNK))
                            sB_slot = cute.make_tensor(
                                sB.iterator + slot * chunk_stride_B,
                                _panel_copy_layout(_MMA_N, _MMA_K_PACKED, PANELS_PER_CHUNK))

                            if load_row_group < HEADS_PER_SPLIT:
                                head_h = head_base_idx * HEADS_PER_SPLIT + load_row_group
                                q_nope_chunk = cute.make_tensor(
                                    q_nope[T_idx, head_h, None].iterator + c * CHUNK_PACKED,
                                    cute.make_layout((CHUNK_PACKED,), stride=(1,)))
                                cute.copy(atom_cpa_chunk128,
                                          lane_copy_chunk128.partition_S(cute.composition(q_nope_chunk, k_split_shape_chunk)),
                                          lane_copy_chunk128.partition_D(sB_slot[load_row_group, None]))

                            for round_idx in range(num_rounds):
                                row_idx  = round_idx * LOAD_ROW_GROUPS + load_row_group
                                if row_idx < num_valid:
                                    flat_row = smem_sp_indices[T_idx, row_idx]
                                    ckv_chunk = cute.make_tensor(
                                        ckv_flat[flat_row, None].iterator + c * CHUNK_PACKED,
                                        cute.make_layout((CHUNK_PACKED,), stride=(1,)))
                                    cute.copy(atom_cpa_chunk128,
                                              lane_copy_chunk128.partition_S(cute.composition(ckv_chunk, k_split_shape_chunk)),
                                              lane_copy_chunk128.partition_D(sA_slot[row_idx, None]))

                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)
                            cute.arch.fence_view_async_shared()
                            cute.arch.barrier(barrier_id=LOAD_BAR_ID, number_of_threads=self.load_threads)

                            if tidx == 0:
                                cute.arch.mbarrier_arrive(chunk_loaded_base + slot)

        if is_mma and has_valid:
            cute.arch.mbarrier_wait(pe_loaded_mbar, cutlass.Int32(0))

            tcgen05_fence()
            if mma_warp_idx == 0:
                for g in cutlass.range_constexpr(LIMIT_REQUEST):
                    if g < T:
                        num_valid_g = smem_assign[g, 1]
                        if num_valid_g > 0:
                            tCtAcc_g = cute.make_tensor(
                                tmem_ptr + cutlass.Int32(g * TMEM_COLS_PER_TOKEN),
                                tCtAcc_tmpl.layout)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for kb in range(_MMA_K_PACK):
                                k_flat = g * _MMA_K_PACK + kb
                                coord  = (None, None, k_flat)
                                cute.gemm(tiled_mma, tCtAcc_g,
                                          tCrA[coord], tCrB[coord], tCtAcc_g)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if mma_warp_idx == 0 and lane_idx == 0:
                tcgen05.commit(pe_done_mbar)

            # Steady-state UMMA loop.
            loaded_p0 = cutlass.Int32(0); loaded_p1 = cutlass.Int32(0)
            loaded_p2 = cutlass.Int32(0); loaded_p3 = cutlass.Int32(0)
            loaded_p4 = cutlass.Int32(0); loaded_p5 = cutlass.Int32(0)

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    num_valid = smem_assign[T_idx, 1]
                    if num_valid > 0:
                        tCtAcc_i = cute.make_tensor(
                            tmem_ptr + cutlass.Int32(T_idx * TMEM_COLS_PER_TOKEN),
                            tCtAcc_tmpl.layout)

                        for c in cutlass.range_constexpr(NUM_CKV_CHUNKS):
                            slot: cutlass.Constexpr = (T_idx * NUM_CKV_CHUNKS + c) % NUM_CKV_STAGES

                            if slot == 0:
                                cute.arch.mbarrier_wait(chunk_loaded_base + 0, loaded_p0)
                                loaded_p0 = loaded_p0 ^ cutlass.Int32(1)
                            elif slot == 1:
                                cute.arch.mbarrier_wait(chunk_loaded_base + 1, loaded_p1)
                                loaded_p1 = loaded_p1 ^ cutlass.Int32(1)
                            elif slot == 2:
                                cute.arch.mbarrier_wait(chunk_loaded_base + 2, loaded_p2)
                                loaded_p2 = loaded_p2 ^ cutlass.Int32(1)
                            elif slot == 3:
                                cute.arch.mbarrier_wait(chunk_loaded_base + 3, loaded_p3)
                                loaded_p3 = loaded_p3 ^ cutlass.Int32(1)
                            elif slot == 4:
                                cute.arch.mbarrier_wait(chunk_loaded_base + 4, loaded_p4)
                                loaded_p4 = loaded_p4 ^ cutlass.Int32(1)
                            else:
                                cute.arch.mbarrier_wait(chunk_loaded_base + 5, loaded_p5)
                                loaded_p5 = loaded_p5 ^ cutlass.Int32(1)

                            tcgen05_fence()
                            if mma_warp_idx == 0:
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                for kb in range(CKV_KBLOCKS_PER_CHUNK):
                                    k_flat = slot * CKV_KBLOCKS_PER_CHUNK + kb
                                    coord  = (None, None, k_flat)
                                    cute.gemm(tiled_mma, tCtAcc_i,
                                              tCrA[coord], tCrB[coord], tCtAcc_i)

                        if mma_warp_idx == 0 and lane_idx == 0:
                            tcgen05.commit(score_mbar_base + T_idx)

        if is_consumer and has_valid:
            cons_warp_idx = warp_idx - cons_base_warp

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    split_idx_new = smem_assign[T_idx, 0]
                    num_valid     = smem_assign[T_idx, 1]

                    if num_valid > 0:
                        cute.arch.mbarrier_wait(score_mbar_base + T_idx, cutlass.Int32(0))

                        tTR_tAcc_i = cute.make_tensor(
                            tTR_tAcc_base.iterator + cutlass.Int32(T_idx * TMEM_COLS_PER_TOKEN),
                            tTR_tAcc_base.layout)

                        if cons_tidx < DIM_SPLIT:
                            cute.copy(tmem_tiled_copy, tTR_tAcc_i[None, None, 0], tTR_rAcc)
                            smem_score[0, cons_tidx] = tTR_rAcc[0] * cutlass.Float32(sm_scale)
                            smem_score[1, cons_tidx] = tTR_rAcc[1] * cutlass.Float32(sm_scale)

                        cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)

                        if cons_warp_idx < HEADS_PER_SPLIT:
                            num_elems: cutlass.Constexpr = DIM_SPLIT // self.wsize
                            head_idx_global = head_base_idx * HEADS_PER_SPLIT + cons_warp_idx
                            vec = smem_score_[(0, None), (cons_warp_idx, lane_idx)].load()
                            vec_masked = cute.make_rmem_tensor(
                                cute.make_layout((num_elems,), stride=(1,)), cutlass.Float32)
                            for v_idx in range(num_elems):
                                vec_masked[v_idx] = -cutlass.Float32(math.inf)
                            for v_idx in range(num_elems):
                                col_idx = lane_idx * num_elems + v_idx
                                if col_idx < num_valid:
                                    vec_masked[v_idx] = vec[v_idx]
                            row_max = -cutlass.Float32(math.inf)
                            for v_idx in range(num_elems):
                                row_max = cute.arch.fmax(row_max, vec_masked[v_idx])
                            row_max = warp_reduce(row_max, cute.arch.fmax)
                            row_sum = cutlass.Float32(0)
                            for v_idx in range(num_elems):
                                e = cute.math.exp(vec_masked[v_idx] - row_max)
                                vec_masked[v_idx] = e
                                row_sum += e
                            row_sum = warp_reduce(row_sum, lambda a, b: a + b)
                            for v_idx in range(num_elems):
                                col_idx = lane_idx * num_elems + v_idx
                                smem_logits_flat[col_idx * HEADS_PER_SPLIT + cons_warp_idx] = vec_masked[v_idx]
                            if lane_idx == 0:
                                partial_lse[T_idx, split_idx_new, head_idx_global, 0] = row_max
                                partial_lse[T_idx, split_idx_new, head_idx_global, 1] = row_sum

                        cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)

                        num_rounds_out: cutlass.Constexpr = (DIM_SPLIT + self.num_cons_warps - 1) // self.num_cons_warps
                        out0 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        out1 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        for stage_idx in cutlass.range_constexpr(self.out_stages):
                            slot: cutlass.Constexpr = (T_idx * NUM_CKV_CHUNKS + stage_idx) % NUM_CKV_STAGES
                            sA_slot_panel = cute.make_tensor(
                                sA.iterator + slot * chunk_stride_A,
                                chunk_out_layout)
                            sA_slot_out = cute.zipped_divide(sA_slot_panel, (1, self.out_vec))
                            out0.fill(cutlass.Float32(0))
                            out1.fill(cutlass.Float32(0))
                            for round_idx in range(num_rounds_out):
                                k = round_idx * self.num_cons_warps + cons_warp_idx
                                if k < num_valid:
                                    gmem_ckv_vec = sA_slot_out[(0, None), (k, lane_idx)].load().to(cutlass.Float32)
                                    smem_logits_vec = smem_logits_flat_[(None), (k)].load()
                                    for v_idx in range(self.out_vec):
                                        out0[v_idx], out1[v_idx] = cute.arch.fma_packed_f32x2(
                                            (smem_logits_vec[0], smem_logits_vec[1]),
                                            (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                            (out0[v_idx], out1[v_idx]))
                            smem_partial_umma_[(0, 0, None), (cons_warp_idx, 0, lane_idx)].store(out0.load())
                            smem_partial_umma_[(0, 0, None), (cons_warp_idx, 1, lane_idx)].store(out1.load())
                            cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)
                            if cons_tidx == 0:
                                cute.arch.mbarrier_arrive(chunk_free_base + cutlass.Int32(slot))
                            thr_group_idx  = cons_tidx // DIM_SPLIT
                            thr_group_lane = cons_tidx %  DIM_SPLIT
                            if thr_group_idx < HEADS_PER_SPLIT:
                                head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                out_col = stage_idx * DIM_SPLIT + thr_group_lane
                                final_sum = cutlass.Float32(0)
                                for i in range(self.num_cons_warps):
                                    final_sum += smem_partial_umma[i, thr_group_idx, thr_group_lane]
                                partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum
                            cute.arch.barrier(barrier_id=CONS_BAR_ID, number_of_threads=self.cons_threads)
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

    @cute.kernel
    def reduce_kernel(
        self, sparse_indices, partial_out, partial_lse, output, lse,
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

        # ── PDL: stall here until compute has finished writing partial_out /
        #     partial_lse for this (T_idx, head_idx).
        cute.arch.griddepcontrol_wait()
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


def compile_kernel():
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


_hybrid, _compiled = compile_kernel()


# ─────────────────────────────────────────────────────────────────────────────
# submit-compatible entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
              _hybrid.partial_out, _hybrid.partial_lse, output, lse)

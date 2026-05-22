"""kv_split_umma_v6_clean_opt_v1.py

Builds on v6 by introducing a `PipelineAsyncUmma` for the score-stage:
producer = 96 threads (3 warps) doing cp.async per K-panel,
consumer = 32 threads (1 warp) issuing tcgen05 UMMA per panel.

Key changes vs v6:
  * sA / sB are now PANEL-staged with `umma_mma_stages = 3` (down from a
    single buffer of all 9 K-panels).  SMEM for sA goes 128*576*2 = 144 KB
    -> 3 * 128*64*2 = 48 KB.
  * Producer↔Consumer overlap of cp.async + UMMA across panels via
    `cutlass.pipeline.PipelineAsyncUmma`.
  * After all 9 panels of a token, the MMA warp issues `tcgen05.commit`
    on the per-token `mma_full_mbars[T_idx]` (kept manual) so the 12
    softmax warps can `mbarrier_wait` on it.
  * All 4 of the original producer warps (cp.async + MMA together) get
    `setmaxregister_decrease(32)`.
"""
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync_mod
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass import pipeline as cute_pipeline
from cutlass.pipeline import NamedBarrier, PipelineAsyncUmma, CooperativeGroup, Agent
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
import math
import torch

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
DIM_SPLIT = (TOP_K + NUM_SPLITS - 1) // NUM_SPLITS  # 128
HEADS_PER_SPLIT = 2

_MK_PACKED = 64       # K elements per panel
_MK_TILES_CKV = HEAD_DIM_CKV // _MK_PACKED   # 8
_MK_TILES_PE  = HEAD_DIM_KPE  // _MK_PACKED  # 1
_MK_TILES_FULL = _MK_TILES_CKV + _MK_TILES_PE  # 9

_MMA_M = DIM_SPLIT  # 128
_MMA_N = 8
_MMA_K = 16
_MK_PACK = _MK_PACKED // _MMA_K  # 4 k-blocks per panel


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

        self.swz_rot_shift = 7
        self.sp_vec_size_i32 = 4
        self.out_stages = 4
        self.out_vec = HEAD_DIM_CKV // (self.out_stages * self.wsize)  # 4

        # ── UMMA workers (full splits) ──
        # 12 softmax cons (384) + 3 cpasync prod (96) + 1 mma cons (32) = 16 warps = 512 t
        self.num_cons_warps     = 12
        self.num_cpasync_warps  = 3
        self.num_mma_warps      = 1
        self.cons_threads       = self.num_cons_warps    * self.wsize  # 384
        self.cpasync_threads    = self.num_cpasync_warps * self.wsize  # 96
        self.mma_threads        = self.num_mma_warps     * self.wsize  # 32
        self.umma_threads       = self.cons_threads + self.cpasync_threads + self.mma_threads  # 512
        self.num_umma_warps     = self.umma_threads // self.wsize       # 16

        self.umma_inst = (DIM_SPLIT, 8, 16)
        self.tmem_cols_per_token = self.umma_inst[1]  # 8
        self.tmem_ld_rep = HEADS_PER_SPLIT
        self.ab_dtype  = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

        # ── Score pipeline ──
        self.umma_mma_stages = 3

        # Named barrier IDs (avoid 0 which sync_threads uses)
        self.cons_bar_id  = 2
        self.sgemm_bar_id = 3

        self.cons_bar  = NamedBarrier(barrier_id=self.cons_bar_id,  num_threads=self.cons_threads)
        self.sgemm_bar = NamedBarrier(barrier_id=self.sgemm_bar_id, num_threads=512)

        self.num_regs_producer = 32

        # ── SGEMM workers ──
        self.sgemm_threads = 512
        self.num_sgemm_warps = self.sgemm_threads // self.wsize
        self.sgemm_ckv_vec = 4
        self.sgemm_kpe_vec = 2

        # ── Reduce kernel ──
        self.reduce_threads = 256
        self.reduce_warps = self.reduce_threads // self.wsize
        self.vec_reduce = 2

        self.partial_out = torch.empty(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.float32, device="cuda")
        self.partial_lse = torch.empty(LIMIT_REQUEST, NUM_SPLITS, NUM_HEADS, 2,            dtype=torch.float32, device="cuda")

    @cute.jit
    def __call__(
        self,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        sm_scale: cutlass.Constexpr,
        partial_out, partial_lse, output, lse, stream,
    ):
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
            score_pipe_mbars: cute.struct.MemRange[cutlass.Int64, 2 * 3]   # umma_mma_stages*2
            mma_full_mbars:   cute.struct.MemRange[cutlass.Int64, LIMIT_REQUEST]
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
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
        sm_scale: cutlass.Constexpr,
        partial_out, partial_lse, output, lse,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()

        # ========= SMEM setup =========
        alloc = cutlass.utils.SmemAllocator()

        smem_sp_indices = self._smem(alloc, cutlass.Int32,   (DIM_CHUNK, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_assign     = self._smem(alloc, cutlass.Int32,   (DIM_CHUNK, 2),         (2, 1))
        smem_score        = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_score_sgemm  = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT, DIM_SPLIT), (DIM_SPLIT, 1))
        smem_logits_flat       = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT * DIM_SPLIT,), (1,))
        smem_logits_flat_sgemm = self._smem(alloc, cutlass.Float32, (HEADS_PER_SPLIT * DIM_SPLIT,), (1,))

        _PARTIAL_DIM = HEAD_DIM_CKV // self.out_stages  # 128
        smem_partial_umma_out = self._smem(alloc, cutlass.Float32,
            (self.num_cons_warps, HEADS_PER_SPLIT, _PARTIAL_DIM),
            (HEADS_PER_SPLIT * _PARTIAL_DIM, _PARTIAL_DIM, 1))
        smem_partial_sgemm = self._smem(alloc, cutlass.Float32,
            (self.num_sgemm_warps, HEADS_PER_SPLIT, HEAD_DIM_CKV // self.out_stages),
            (HEADS_PER_SPLIT * (HEAD_DIM_CKV // self.out_stages), HEAD_DIM_CKV // self.out_stages, 1))

        # ── Panel-staged sA / sB ──
        # sA layout for MMA fragment: ((128,16), 1, (4, num_stages))
        #   inner stride ((64,1), 0, (16, 128*64))   # last mode is stage
        swizzle = cute.make_swizzle(3, 4, 3)
        S = self.umma_mma_stages
        a_outer = cute.make_layout(
            ((_MMA_M, _MMA_K), 1, (_MK_PACK,), S),
            stride=((_MK_PACKED, 1), 0, (_MMA_K,), _MMA_M * _MK_PACKED))
        b_outer = cute.make_layout(
            ((_MMA_N, _MMA_K), 1, (_MK_PACK,), S),
            stride=((_MK_PACKED, 1), 0, (_MMA_K,), _MMA_N * _MK_PACKED))
        sA = alloc.allocate_tensor(cutlass.BFloat16, a_outer, byte_alignment=16, swizzle=swizzle)
        sB = alloc.allocate_tensor(cutlass.BFloat16, b_outer, byte_alignment=16, swizzle=swizzle)
        # Linear views for cp.async (per-thread pointer arithmetic)
        # sA stage stride = _MMA_M * _MK_PACKED = 8192 elements
        # within stage: row-major (row, k) with row stride _MK_PACKED=64
        sA_iter = sA.iterator
        sB_iter = sB.iterator
        SA_STAGE_STRIDE: cutlass.Constexpr = _MMA_M * _MK_PACKED   # 8192
        SB_STAGE_STRIDE: cutlass.Constexpr = _MMA_N * _MK_PACKED   # 512

        storage  = alloc.allocate(self.shared_storage)
        mma_full_mbars = storage.mma_full_mbars.data_ptr()

        cons_bar  = self.cons_bar
        sgemm_bar = self.sgemm_bar

        # ========= Prologue: gather sparse indices =========
        head_base_idx, split_idx_old, _ = cute.arch.block_idx()
        T, _, _ = q_nope.shape

        sparse_indices_  = cute.zipped_divide(sparse_indices, (1, self.sp_vec_size_i32))
        smem_sp_indices_ = cute.zipped_divide(smem_sp_indices, (1, self.sp_vec_size_i32))
        if self.num_umma_warps <= warp_idx < self.num_umma_warps + T:
            warp_idx_sgemm = warp_idx - self.num_umma_warps
            split_idx_new = (split_idx_old + warp_idx_sgemm * self.swz_rot_shift) % cutlass.Int32(NUM_SPLITS)
            split_vec_stride = DIM_SPLIT // self.sp_vec_size_i32
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

        # ── tmem alloc + per-token mma_full_mbars init ──
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape       = tiled_mma.partition_shape_C((_MMA_M, _MMA_N))
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols * LIMIT_REQUEST)
        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                for i in range(LIMIT_REQUEST):
                    cute.arch.mbarrier_init(mma_full_mbars + i, cnt=1)
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf)
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

        # ── Score pipeline: PipelineAsyncUmma (cp.async producer + UMMA consumer) ──
        producer_grp = CooperativeGroup(Agent.Thread, self.cpasync_threads)
        consumer_grp = CooperativeGroup(Agent.Thread, self.mma_threads)
        score_producer, score_consumer = PipelineAsyncUmma.create(
            num_stages=self.umma_mma_stages,
            producer_group=producer_grp,
            consumer_group=consumer_grp,
            barrier_storage=storage.score_pipe_mbars.data_ptr(),
        ).make_participants()

        # Hoisted views
        smem_score_              = cute.zipped_divide(smem_score,              (1, DIM_SPLIT // self.wsize))
        smem_score_sgemm_        = cute.zipped_divide(smem_score_sgemm,        (1, DIM_SPLIT // self.wsize))
        smem_logits_flat_        = cute.zipped_divide(smem_logits_flat,        (HEADS_PER_SPLIT,))
        smem_logits_flat_sgemm_  = cute.zipped_divide(smem_logits_flat_sgemm,  (HEADS_PER_SPLIT,))
        smem_partial_umma_out_ = cute.zipped_divide(smem_partial_umma_out, (1, 1, self.out_vec))
        smem_partial_sgemm_ = cute.zipped_divide(smem_partial_sgemm, (1, 1, self.out_vec))
        ckv_flat_out        = cute.zipped_divide(ckv_flat,           (1, self.out_vec))
        q_nope_z   = cute.zipped_divide(q_nope,   (1, 1, self.sgemm_ckv_vec))
        q_pe_z     = cute.zipped_divide(q_pe,     (1, 1, self.sgemm_kpe_vec))
        ckv_flat_z = cute.zipped_divide(ckv_flat, (1, self.sgemm_ckv_vec))
        kpe_flat_z = cute.zipped_divide(kpe_flat, (1, self.sgemm_kpe_vec))

        atom_cpa = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)

        # ============================================================
        # Role assignment
        # ============================================================
        is_softmax_cons   = warp_idx < self.num_cons_warps
        is_cpasync_prod   = self.num_cons_warps <= warp_idx < self.num_cons_warps + self.num_cpasync_warps
        is_mma_cons       = warp_idx == self.num_cons_warps + self.num_cpasync_warps   # warp 15
        is_sgemm_warp     = warp_idx >= self.num_umma_warps

        # Reduce regs on the producer side (cp.async + mma warps)
        if is_cpasync_prod or is_mma_cons:
            cute.arch.setmaxregister_decrease(self.num_regs_producer)

        # ============================================================
        # CP.ASYNC PRODUCER (3 warps = 96 threads)
        # ============================================================
        if is_cpasync_prod:
            cpa_warp_idx = warp_idx - self.num_cons_warps              # 0..2
            cpa_tid      = cpa_warp_idx * self.wsize + lane_idx        # 0..95

            # 1024 vec-ops per panel (128 rows × 8 K-vec).  96 threads → 11 rounds.
            VEC_OPS_PER_PANEL: cutlass.Constexpr = _MMA_M * (_MK_PACKED // 8)  # 1024
            ROUNDS_PER_PANEL: cutlass.Constexpr  = (VEC_OPS_PER_PANEL + 96 - 1) // 96  # 11

            # Q vec-ops per panel: HEADS_PER_SPLIT * 8 = 16  (warp 0, lanes 0..15)
            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    num_valid = smem_assign[T_idx, 1]
                    if num_valid == DIM_SPLIT:
                        for panel_idx in cutlass.range_constexpr(_MK_TILES_FULL):
                            handle = score_producer.acquire_and_advance()
                            stage  = handle.index
                            sa_stage_off = stage * SA_STAGE_STRIDE
                            sb_stage_off = stage * SB_STAGE_STRIDE

                            if cutlass.const_expr(panel_idx < _MK_TILES_CKV):
                                ckv_panel_k_base: cutlass.Constexpr = panel_idx * _MK_PACKED
                                # ── Load Q (sB[stage]) for CKV panel
                                if cpa_warp_idx == 0 and lane_idx < HEADS_PER_SPLIT * 8:
                                    head_local = lane_idx // 8
                                    k_vec      = lane_idx % 8
                                    head_g     = head_base_idx * HEADS_PER_SPLIT + head_local
                                    src_off = (T_idx * NUM_HEADS * HEAD_DIM_CKV
                                                + head_g * HEAD_DIM_CKV
                                                + ckv_panel_k_base + k_vec * 8)
                                    src_p = cute.make_ptr(cutlass.BFloat16,
                                        (q_nope.iterator + src_off).toint(),
                                        mem_space=cute.AddressSpace.gmem, assumed_align=16)
                                    dst_p = cute.make_ptr(cutlass.BFloat16,
                                        (sB_iter + (sb_stage_off + head_local * _MK_PACKED + k_vec * 8)).toint(),
                                        mem_space=cute.AddressSpace.smem, assumed_align=16)
                                    src_vec = cute.make_tensor(src_p, cute.make_layout((8,), stride=(1,)))
                                    dst_vec = cute.make_tensor(dst_p, cute.make_layout((8,), stride=(1,)))
                                    cute.copy(atom_cpa, src_vec, dst_vec)
                                # ── Load K (sA[stage]) for CKV panel
                                for round_idx in cutlass.range_constexpr(ROUNDS_PER_PANEL):
                                    pos = round_idx * 96 + cpa_tid
                                    if pos < VEC_OPS_PER_PANEL:
                                        row   = pos // 8
                                        k_vec = pos %  8
                                        row_global = smem_sp_indices[T_idx, row]
                                        src_p = cute.make_ptr(cutlass.BFloat16,
                                            (ckv_flat.iterator + (row_global * HEAD_DIM_CKV
                                                                  + ckv_panel_k_base + k_vec * 8)).toint(),
                                            mem_space=cute.AddressSpace.gmem, assumed_align=16)
                                        dst_p = cute.make_ptr(cutlass.BFloat16,
                                            (sA_iter + (sa_stage_off + row * _MK_PACKED + k_vec * 8)).toint(),
                                            mem_space=cute.AddressSpace.smem, assumed_align=16)
                                        src_vec = cute.make_tensor(src_p, cute.make_layout((8,), stride=(1,)))
                                        dst_vec = cute.make_tensor(dst_p, cute.make_layout((8,), stride=(1,)))
                                        cute.copy(atom_cpa, src_vec, dst_vec)
                            else:
                                # ── Load Q (sB[stage]) for KPE panel
                                if cpa_warp_idx == 0 and lane_idx < HEADS_PER_SPLIT * 8:
                                    head_local = lane_idx // 8
                                    k_vec      = lane_idx % 8
                                    head_g     = head_base_idx * HEADS_PER_SPLIT + head_local
                                    src_p = cute.make_ptr(cutlass.BFloat16,
                                        (q_pe.iterator + (T_idx * NUM_HEADS * HEAD_DIM_KPE
                                                          + head_g * HEAD_DIM_KPE + k_vec * 8)).toint(),
                                        mem_space=cute.AddressSpace.gmem, assumed_align=16)
                                    dst_p = cute.make_ptr(cutlass.BFloat16,
                                        (sB_iter + (sb_stage_off + head_local * _MK_PACKED + k_vec * 8)).toint(),
                                        mem_space=cute.AddressSpace.smem, assumed_align=16)
                                    src_vec = cute.make_tensor(src_p, cute.make_layout((8,), stride=(1,)))
                                    dst_vec = cute.make_tensor(dst_p, cute.make_layout((8,), stride=(1,)))
                                    cute.copy(atom_cpa, src_vec, dst_vec)
                                # ── Load K (sA[stage]) for KPE panel
                                for round_idx in cutlass.range_constexpr(ROUNDS_PER_PANEL):
                                    pos = round_idx * 96 + cpa_tid
                                    if pos < VEC_OPS_PER_PANEL:
                                        row   = pos // 8
                                        k_vec = pos %  8
                                        row_global = smem_sp_indices[T_idx, row]
                                        src_p = cute.make_ptr(cutlass.BFloat16,
                                            (kpe_flat.iterator + (row_global * HEAD_DIM_KPE + k_vec * 8)).toint(),
                                            mem_space=cute.AddressSpace.gmem, assumed_align=16)
                                        dst_p = cute.make_ptr(cutlass.BFloat16,
                                            (sA_iter + (sa_stage_off + row * _MK_PACKED + k_vec * 8)).toint(),
                                            mem_space=cute.AddressSpace.smem, assumed_align=16)
                                        src_vec = cute.make_tensor(src_p, cute.make_layout((8,), stride=(1,)))
                                        dst_vec = cute.make_tensor(dst_p, cute.make_layout((8,), stride=(1,)))
                                        cute.copy(atom_cpa, src_vec, dst_vec)

                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)
                            cute.arch.fence_view_async_shared()
                            handle.commit()
            score_producer.tail()

        # ============================================================
        # MMA CONSUMER (1 warp = 32 threads)
        # ============================================================
        if is_mma_cons:
            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    num_valid = smem_assign[T_idx, 1]
                    if num_valid == DIM_SPLIT:
                        tmem_slot_offset = cutlass.Int32(T_idx * self.tmem_cols_per_token)
                        tCtAcc_i = cute.make_tensor(tmem_ptr + tmem_slot_offset, tCtAcc_tmpl.layout)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                        for panel_idx in cutlass.range_constexpr(_MK_TILES_FULL):
                            handle = score_consumer.wait_and_advance()
                            stage  = handle.index

                            tcgen05_fence()
                            if lane_idx == 0:
                                # 4 k-blocks per panel
                                for kb in cutlass.range_constexpr(_MK_PACK):
                                    coord = (None, None, kb, stage)
                                    cute.gemm(tiled_mma, tCtAcc_i,
                                              tCrA[coord], tCrB[coord], tCtAcc_i)
                                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                            handle.release()

                        # All 9 panels for this token done — signal softmax warps
                        if lane_idx == 0:
                            tcgen05.commit(mma_full_mbars + T_idx)

        # ============================================================
        # SOFTMAX / OUTPUT CONSUMERS (12 warps = 384 threads, unchanged)
        # ============================================================
        if is_softmax_cons:
            cons_warp_idx = warp_idx
            num_rounds_out  = (DIM_SPLIT + self.num_cons_warps - 1) // self.num_cons_warps  # 11

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    split_idx_new = smem_assign[T_idx, 0]
                    num_valid     = smem_assign[T_idx, 1]
                    if num_valid == DIM_SPLIT:
                        cute.arch.mbarrier_wait(mma_full_mbars + T_idx, cutlass.Int32(0))

                        tmem_slot_offset = cutlass.Int32(T_idx * self.tmem_cols_per_token)
                        tTR_tAcc_i = cute.make_tensor(
                            tTR_tAcc.iterator + tmem_slot_offset, tTR_tAcc.layout)

                        if tidx < DIM_SPLIT:
                            cute.copy(tmem_tiled_copy, tTR_tAcc_i[None, None, 0], tTR_rAcc)
                            smem_score[0, tidx] = tTR_rAcc[0] * cutlass.Float32(sm_scale)
                            smem_score[1, tidx] = tTR_rAcc[1] * cutlass.Float32(sm_scale)

                        cons_bar.arrive_and_wait()

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

                        cons_bar.arrive_and_wait()

                        out0 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        out1 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        for stage_idx in range(self.out_stages):
                            out0.fill(cutlass.Float32(0))
                            out1.fill(cutlass.Float32(0))
                            for round_idx in range(num_rounds_out):
                                k = round_idx * self.num_cons_warps + cons_warp_idx
                                if k < num_valid:
                                    flat_cache_idx = smem_sp_indices[T_idx, k]
                                    gmem_ckv_vec = ckv_flat_out[(0, None), (flat_cache_idx, stage_idx * self.wsize + lane_idx)].load().to(cutlass.Float32)
                                    smem_logits_vec = smem_logits_flat_[(None), (k)].load()
                                    for v_idx in range(self.out_vec):
                                        out0[v_idx], out1[v_idx] = cute.arch.fma_packed_f32x2(
                                            (smem_logits_vec[0], smem_logits_vec[1]),
                                            (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                            (out0[v_idx], out1[v_idx]))
                            smem_partial_umma_out_[(0, 0, None), (cons_warp_idx, 0, lane_idx)].store(out0.load())
                            smem_partial_umma_out_[(0, 0, None), (cons_warp_idx, 1, lane_idx)].store(out1.load())
                            cons_bar.arrive_and_wait()
                            thr_group_idx  = tidx // DIM_SPLIT
                            thr_group_lane = tidx %  DIM_SPLIT
                            if thr_group_idx < HEADS_PER_SPLIT:
                                head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                out_col = stage_idx * DIM_SPLIT + thr_group_lane
                                final_sum = cutlass.Float32(0)
                                for i in range(self.num_cons_warps):
                                    final_sum += smem_partial_umma_out[i, thr_group_idx, thr_group_lane]
                                partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum
                            cons_bar.arrive_and_wait()

        # ============================================================
        # SGEMM workers (partial splits) — unchanged from v6_clean
        # ============================================================
        if is_sgemm_warp:
            sgemm_warp_idx = warp_idx - self.num_umma_warps
            sgemm_tidx     = tidx - self.umma_threads

            for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                if T_idx < T:
                    split_idx_new = smem_assign[T_idx, 0]
                    num_valid     = smem_assign[T_idx, 1]
                    if 0 < num_valid < DIM_SPLIT:
                        head_idx0 = head_base_idx * HEADS_PER_SPLIT
                        head_idx1 = head_base_idx * HEADS_PER_SPLIT + 1
                        num_rounds_score = (num_valid + self.num_sgemm_warps - 1) // self.num_sgemm_warps
                        for round_idx in range(num_rounds_score):
                            col_idx = round_idx * self.num_sgemm_warps + sgemm_warp_idx
                            if col_idx < num_valid:
                                flat_cache_idx = smem_sp_indices[T_idx, col_idx]
                                acc0 = cutlass.Float32(0)
                                acc1 = cutlass.Float32(0)
                                for i in range(HEAD_DIM_CKV // (self.sgemm_ckv_vec * self.wsize)):
                                    row_idx = i * self.wsize + lane_idx
                                    qn0_frag = q_nope_z[(0, 0, None), (T_idx, head_idx0, row_idx)].load().to(cutlass.Float32)
                                    qn1_frag = q_nope_z[(0, 0, None), (T_idx, head_idx1, row_idx)].load().to(cutlass.Float32)
                                    ckv_frag = ckv_flat_z[(0, None), (flat_cache_idx, row_idx)].load().to(cutlass.Float32)
                                    for v in range(self.sgemm_ckv_vec):
                                        acc0, acc1 = cute.arch.fma_packed_f32x2(
                                            (qn0_frag[v], qn1_frag[v]),
                                            (ckv_frag[v], ckv_frag[v]),
                                            (acc0, acc1))
                                for i in range(HEAD_DIM_KPE // (self.sgemm_kpe_vec * self.wsize)):
                                    row_idx = i * self.wsize + lane_idx
                                    qp0_frag = q_pe_z[(0, 0, None), (T_idx, head_idx0, row_idx)].load().to(cutlass.Float32)
                                    qp1_frag = q_pe_z[(0, 0, None), (T_idx, head_idx1, row_idx)].load().to(cutlass.Float32)
                                    kpe_frag = kpe_flat_z[(0, None), (flat_cache_idx, row_idx)].load().to(cutlass.Float32)
                                    for v in range(self.sgemm_kpe_vec):
                                        acc0, acc1 = cute.arch.fma_packed_f32x2(
                                            (qp0_frag[v], qp1_frag[v]),
                                            (kpe_frag[v], kpe_frag[v]),
                                            (acc0, acc1))
                                acc0 = warp_reduce(acc0, lambda a, b: a + b)
                                acc1 = warp_reduce(acc1, lambda a, b: a + b)
                                if lane_idx == 0:
                                    smem_score_sgemm[0, col_idx] = acc0 * cutlass.Float32(sm_scale)
                                    smem_score_sgemm[1, col_idx] = acc1 * cutlass.Float32(sm_scale)
                        sgemm_bar.arrive_and_wait()
                        if sgemm_warp_idx < HEADS_PER_SPLIT:
                            num_elems: cutlass.Constexpr = DIM_SPLIT // self.wsize
                            head_idx_global = head_base_idx * HEADS_PER_SPLIT + sgemm_warp_idx
                            vec = smem_score_sgemm_[(0, None), (sgemm_warp_idx, lane_idx)].load()
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
                                smem_logits_flat_sgemm[col_idx * HEADS_PER_SPLIT + sgemm_warp_idx] = vec_masked[v_idx]
                            if lane_idx == 0:
                                partial_lse[T_idx, split_idx_new, head_idx_global, 0] = row_max
                                partial_lse[T_idx, split_idx_new, head_idx_global, 1] = row_sum
                        sgemm_bar.arrive_and_wait()

                        num_rounds_out_s = (num_valid + self.num_sgemm_warps - 1) // self.num_sgemm_warps
                        out0_s = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        out1_s = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                        for stage_idx in range(self.out_stages):
                            out0_s.fill(cutlass.Float32(0))
                            out1_s.fill(cutlass.Float32(0))
                            for round_idx in range(num_rounds_out_s):
                                k = round_idx * self.num_sgemm_warps + sgemm_warp_idx
                                if k < num_valid:
                                    flat_cache_idx = smem_sp_indices[T_idx, k]
                                    gmem_ckv_vec = ckv_flat_out[(0, None), (flat_cache_idx, stage_idx * self.wsize + lane_idx)].load().to(cutlass.Float32)
                                    smem_logits_vec = smem_logits_flat_sgemm_[(None), (k)].load()
                                    for v_idx in range(self.out_vec):
                                        out0_s[v_idx], out1_s[v_idx] = cute.arch.fma_packed_f32x2(
                                            (smem_logits_vec[0], smem_logits_vec[1]),
                                            (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                            (out0_s[v_idx], out1_s[v_idx]))
                            smem_partial_sgemm_[(0, 0, None), (sgemm_warp_idx, 0, lane_idx)].store(out0_s.load())
                            smem_partial_sgemm_[(0, 0, None), (sgemm_warp_idx, 1, lane_idx)].store(out1_s.load())
                            sgemm_bar.arrive_and_wait()
                            thr_group_idx  = sgemm_tidx // DIM_SPLIT
                            thr_group_lane = sgemm_tidx %  DIM_SPLIT
                            if thr_group_idx < HEADS_PER_SPLIT:
                                head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                out_col = stage_idx * DIM_SPLIT + thr_group_lane
                                final_sum = cutlass.Float32(0)
                                for i in range(self.num_sgemm_warps):
                                    final_sum += smem_partial_sgemm[i, thr_group_idx, thr_group_lane]
                                partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum
                            sgemm_bar.arrive_and_wait()

        # Epilogue
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

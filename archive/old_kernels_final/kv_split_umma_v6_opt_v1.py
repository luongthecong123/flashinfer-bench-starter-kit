"""kv_split_umma_v6_opt_v1.py

v6 + 3-stage panel pipeline using DIRTY manual SMEM mbarriers.

Layout (UMMA pool = 16 warps = 512 t):
  warps  0..11  (384t) softmax/output consumers
  warps 12..14  (96t)  cp.async producers (panel ring)
  warp     15   (32t)  tcgen05 UMMA issuer
SGEMM pool: 16 warps × 32 = 512 threads (unchanged from v6).

Score pipeline: 9 panels per token (8 CKV + 1 KPE), staged in `umma_mma_stages = 3`.
  ab_full_mbars [stage]  cnt = 96 (one arrive per producer thread after cp.async drain)
  ab_empty_mbars[stage]  cnt = 1  (single tcgen05.commit by MMA warp lane 0)
  mma_full_mbars[T_idx]  cnt = 1  (per-token, signals softmax cons after last panel)

SMEM for sA: 3 × 128 × 64 × 2B = 48 KB (down from v6's 144 KB).
"""
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync_mod
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass.pipeline import NamedBarrier
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


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@cute.jit
def _panel_copy_layout(num_rows: int, k_packed: int, num_stages: int):
    """Layout (rows, (k_packed, num_stages)) row-major within panel; stages are outer K-mode."""
    return cute.make_layout((num_rows, (k_packed, num_stages)),
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
        self.out_vec = HEAD_DIM_CKV // (self.out_stages * self.wsize)  # 4

        # ── UMMA workers (full splits) ──
        # 12 cons (384) + 4 prod (128) = 16 warps = 512 threads
        # Producer warps do cp.async + tcgen05.mma in a 3-stage panel pipeline.
        self.num_cons_warps = 12
        self.num_prod_warps = 4
        self.cons_threads   = self.num_cons_warps * self.wsize  # 384
        self.prod_threads   = self.num_prod_warps * self.wsize  # 128
        self.umma_threads   = self.cons_threads + self.prod_threads  # 512
        self.num_umma_warps = self.umma_threads // self.wsize  # 16

        self.umma_inst = (DIM_SPLIT, 8, 16)
        self.tmem_cols_per_token = self.umma_inst[1]  # 8
        self.tmem_ld_rep = HEADS_PER_SPLIT
        self.ab_dtype  = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

        self.cons_bar_id  = 2
        self.prod_bar_id  = 4
        self.sgemm_bar_id = 3

        # ── Pipeline staging ──
        self.umma_mma_stages = 3
        self.num_panels      = (HEAD_DIM_CKV + HEAD_DIM_KPE) // 64  # 9

        self.num_regs_producer = 32

        # ── SGEMM workers (partial splits) ──
        self.sgemm_threads = 512
        self.num_sgemm_warps = self.sgemm_threads // self.wsize   # 16
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
        stream,
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
            mma_full_mbars:   cute.struct.MemRange[cutlass.Int64, LIMIT_REQUEST]
            ab_empty_mbars:   cute.struct.MemRange[cutlass.Int64, self.umma_mma_stages]
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
        lse:            cute.Tensor,
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

        # ── UMMA smem: NUM_STAGES-panel ring (3 panels × 64 cols) using canonical sm100 layouts ──
        _MK_PACKED = 64
        _NUM_STAGES = self.umma_mma_stages   # 3
        _MMA_M = DIM_SPLIT   # 128
        _MMA_N = 8
        _MMA_K = 16
        cta_tile_mnk = (_MMA_M, _MMA_N, _MK_PACKED)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, cta_tile_mnk, cutlass.BFloat16, _NUM_STAGES)
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, cta_tile_mnk, cutlass.BFloat16, _NUM_STAGES)
        sA = alloc.allocate_tensor(
            element_type=cutlass.BFloat16, layout=a_smem_layout.outer,
            byte_alignment=128, swizzle=a_smem_layout.inner)
        sB = alloc.allocate_tensor(
            element_type=cutlass.BFloat16, layout=b_smem_layout.outer,
            byte_alignment=128, swizzle=b_smem_layout.inner)
        sA_copy = cute.make_tensor(sA.iterator, _panel_copy_layout(_MMA_M, _MK_PACKED, _NUM_STAGES))  # (128, (64, 3))
        sB_copy = cute.make_tensor(sB.iterator, _panel_copy_layout(_MMA_N, _MK_PACKED, _NUM_STAGES))  # (8,   (64, 3))

        # cp.async atom — 128b per lane, one warp covers (4 rows × 8 K-vecs) per call
        atom_cpa   = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
        thr_layout = cute.make_layout(((8, 4),), stride=((1, 8),))
        val_layout = cute.make_layout(((8, 1),), stride=((1, 0),))
        tiled_copy = cute.make_tiled_copy_tv(atom_cpa, thr_layout, val_layout)
        lane_copy  = tiled_copy.get_slice(lane_idx)

        storage          = alloc.allocate(self.shared_storage)
        mma_full_mbars   = storage.mma_full_mbars.data_ptr()
        ab_empty_mbars   = storage.ab_empty_mbars.data_ptr()

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

        # ── tmem / mbarrier setup ──
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
                # ab_empty: cnt = 1 (single tcgen05.commit by prod warp 0 lane 0)
                for i in range(self.umma_mma_stages):
                    cute.arch.mbarrier_init(ab_empty_mbars + i, cnt=1)
                cute.arch.mbarrier_init_fence()
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

        # Per-row source views split into 64-col panels along K
        ckv_flat_panels = cute.zipped_divide(ckv_flat, (1, _MK_PACKED))   # → outer (1, 8)
        kpe_flat_panels = cute.zipped_divide(kpe_flat, (1, _MK_PACKED))   # → outer (1, 1)
        q_nope_panels   = cute.zipped_divide(q_nope,   (1, 1, _MK_PACKED))
        q_pe_panels     = cute.zipped_divide(q_pe,     (1, 1, _MK_PACKED))

        # ============================================================
        # UMMA workers: warp-specialized (12 cons + 4 prod)
        # ============================================================
        prod_bar = NamedBarrier(barrier_id=self.prod_bar_id, num_threads=self.prod_threads)

        if warp_idx < self.num_umma_warps:
            is_consumer   = warp_idx <  self.num_cons_warps
            is_producer   = warp_idx >= self.num_cons_warps
            cons_warp_idx = warp_idx
            prod_warp_idx = warp_idx - self.num_cons_warps      # 0..3

            num_rounds_out     = (DIM_SPLIT + self.num_cons_warps - 1) // self.num_cons_warps  # 11
            num_kblk_per_panel = _MK_PACKED // _MMA_K            # 4
            num_panels         = self.num_panels                 # 9
            NS                 = self.umma_mma_stages            # 3

            if is_producer:
                cute.arch.setmaxregister_decrease(self.num_regs_producer)

            # =====================================================
            # PRODUCER (warps 12..15): cp.async + tcgen05.mma with
            # 3-stage panel pipeline. Each panel:
            #   - all 4 prod warps cp.async into sA/sB[stage]
            #   - prod_bar.arrive_and_wait sync
            #   - prod warp 0 lane 0 issues 4 cute.gemm on the
            #     completed (NS-1)-stages-ago panel
            # `pcycle` is GLOBAL across tokens (increments only for processed
            # tokens) so empty-mbar phase tracking stays consistent.
            # =====================================================
            if is_producer:
                pcycle = cutlass.Int32(0)
                for T_idx in cutlass.range_constexpr(LIMIT_REQUEST):
                    if T_idx < T:
                        num_valid = smem_assign[T_idx, 1]
                        if num_valid == DIM_SPLIT:
                            tmem_slot_offset = cutlass.Int32(T_idx * self.tmem_cols_per_token)
                            tCtAcc_i = cute.make_tensor(tmem_ptr + tmem_slot_offset, tCtAcc_tmpl.layout)
                            head_h0 = head_base_idx * HEADS_PER_SPLIT + 0
                            head_h1 = head_base_idx * HEADS_PER_SPLIT + 1
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                            # ===== Steady-state pipeline: load panel `panel_idx`,
                            # MMA the panel issued NS-1 ago. =====
                            for panel_idx in cutlass.range_constexpr(num_panels):
                                stage_idx = pcycle % cutlass.Int32(NS)
                                cycle     = pcycle // cutlass.Int32(NS)

                                # Wait empty[stage] before overwriting (skip first NS panels of the kernel).
                                if pcycle >= cutlass.Int32(NS):
                                    wait_phase = (cycle & cutlass.Int32(1)) ^ cutlass.Int32(1)
                                    cute.arch.mbarrier_wait(ab_empty_mbars + stage_idx, wait_phase)

                                # ── Load K (sA): 128 rows × 64 cols, 4-warp row round-robin ──
                                if cutlass.const_expr(panel_idx < 8):
                                    for row_idx in range(_MMA_M):
                                        if (row_idx % self.num_prod_warps) == prod_warp_idx:
                                            flat_row = smem_sp_indices[T_idx, row_idx]
                                            cute.copy(atom_cpa,
                                                      lane_copy.partition_S(ckv_flat_panels[(0, None), (flat_row, panel_idx)]),
                                                      lane_copy.partition_D(sA_copy[row_idx, (None, stage_idx)]))
                                else:
                                    for row_idx in range(_MMA_M):
                                        if (row_idx % self.num_prod_warps) == prod_warp_idx:
                                            flat_row = smem_sp_indices[T_idx, row_idx]
                                            cute.copy(atom_cpa,
                                                      lane_copy.partition_S(kpe_flat_panels[(0, None), (flat_row, 0)]),
                                                      lane_copy.partition_D(sA_copy[row_idx, (None, stage_idx)]))

                                # ── Load Q (sB): 2 rows × 64 cols, prod warp 0 only ──
                                if prod_warp_idx == 0:
                                    if cutlass.const_expr(panel_idx < 8):
                                        cute.copy(atom_cpa,
                                                  lane_copy.partition_S(q_nope_panels[(0, 0, None), (T_idx, head_h0, panel_idx)]),
                                                  lane_copy.partition_D(sB_copy[0, (None, stage_idx)]))
                                        cute.copy(atom_cpa,
                                                  lane_copy.partition_S(q_nope_panels[(0, 0, None), (T_idx, head_h1, panel_idx)]),
                                                  lane_copy.partition_D(sB_copy[1, (None, stage_idx)]))
                                    else:
                                        cute.copy(atom_cpa,
                                                  lane_copy.partition_S(q_pe_panels[(0, 0, None), (T_idx, head_h0, 0)]),
                                                  lane_copy.partition_D(sB_copy[0, (None, stage_idx)]))
                                        cute.copy(atom_cpa,
                                                  lane_copy.partition_S(q_pe_panels[(0, 0, None), (T_idx, head_h1, 0)]),
                                                  lane_copy.partition_D(sB_copy[1, (None, stage_idx)]))

                                cute.arch.cp_async_commit_group()

                                # When we've issued >= NS groups, the oldest (NS-1 ago) is ready: drain & MMA.
                                if pcycle >= cutlass.Int32(NS - 1):
                                    completed       = pcycle - cutlass.Int32(NS - 1)
                                    completed_stage = completed % cutlass.Int32(NS)
                                    cute.arch.cp_async_wait_group(NS - 1)
                                    cute.arch.fence_view_async_shared()
                                    prod_bar.arrive_and_wait()
                                    tcgen05_fence()
                                    if prod_warp_idx == 0 and lane_idx == 0:
                                        for kb in range(num_kblk_per_panel):
                                            k_block_coord = (None, None, kb, completed_stage)
                                            cute.gemm(tiled_mma, tCtAcc_i,
                                                      tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc_i)
                                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                        # Release stage when MMA reads complete.
                                        tcgen05.commit(ab_empty_mbars + completed_stage)

                                pcycle = pcycle + cutlass.Int32(1)

                            # ===== Tail: drain remaining NS-1 in-flight panels =====
                            for tail_i in cutlass.range_constexpr(NS - 1):
                                completed       = pcycle - cutlass.Int32(NS - 1 - tail_i)
                                completed_stage = completed % cutlass.Int32(NS)
                                is_last_panel: cutlass.Constexpr = (tail_i == (NS - 2))
                                cute.arch.cp_async_wait_group(NS - 2 - tail_i)
                                cute.arch.fence_view_async_shared()
                                prod_bar.arrive_and_wait()
                                tcgen05_fence()
                                if prod_warp_idx == 0 and lane_idx == 0:
                                    for kb in range(num_kblk_per_panel):
                                        k_block_coord = (None, None, kb, completed_stage)
                                        cute.gemm(tiled_mma, tCtAcc_i,
                                                  tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc_i)
                                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                    # Release stage AND, on the very last panel of the token, also signal the consumer.
                                    tcgen05.commit(ab_empty_mbars + completed_stage)
                                    if cutlass.const_expr(is_last_panel):
                                        tcgen05.commit(mma_full_mbars + T_idx)

            # =====================================================
            # SOFTMAX / OUTPUT CONSUMERS (warps 0..11) — verbatim from v6
            # =====================================================
            if is_consumer:
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

                            cute.arch.barrier(barrier_id=self.cons_bar_id,
                                              number_of_threads=self.cons_threads)

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

                            cute.arch.barrier(barrier_id=self.cons_bar_id,
                                              number_of_threads=self.cons_threads)

                            out0 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                            out1 = cute.make_rmem_tensor((self.out_vec,), cutlass.Float32)
                            for stage_idx_o in range(self.out_stages):
                                out0.fill(cutlass.Float32(0))
                                out1.fill(cutlass.Float32(0))
                                for round_idx in range(num_rounds_out):
                                    k = round_idx * self.num_cons_warps + cons_warp_idx
                                    if k < num_valid:
                                        flat_cache_idx = smem_sp_indices[T_idx, k]
                                        gmem_ckv_vec = ckv_flat_out[(0, None), (flat_cache_idx, stage_idx_o * self.wsize + lane_idx)].load().to(cutlass.Float32)
                                        smem_logits_vec = smem_logits_flat_[(None), (k)].load()
                                        for v_idx in range(self.out_vec):
                                            out0[v_idx], out1[v_idx] = cute.arch.fma_packed_f32x2(
                                                (smem_logits_vec[0], smem_logits_vec[1]),
                                                (gmem_ckv_vec[v_idx], gmem_ckv_vec[v_idx]),
                                                (out0[v_idx], out1[v_idx]))

                                smem_partial_umma_out_[(0, 0, None), (cons_warp_idx, 0, lane_idx)].store(out0.load())
                                smem_partial_umma_out_[(0, 0, None), (cons_warp_idx, 1, lane_idx)].store(out1.load())

                                cute.arch.barrier(barrier_id=self.cons_bar_id,
                                                  number_of_threads=self.cons_threads)

                                thr_group_idx  = tidx // DIM_SPLIT
                                thr_group_lane = tidx %  DIM_SPLIT
                                if thr_group_idx < HEADS_PER_SPLIT:
                                    head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                    out_col = stage_idx_o * DIM_SPLIT + thr_group_lane
                                    final_sum = cutlass.Float32(0)
                                    for i in range(self.num_cons_warps):
                                        final_sum += smem_partial_umma_out[i, thr_group_idx, thr_group_lane]
                                    partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum

                                cute.arch.barrier(barrier_id=self.cons_bar_id,
                                                  number_of_threads=self.cons_threads)

        # ============================================================
        # SGEMM workers: partial splits — verbatim from v6
        # ============================================================
        else:
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

                        cute.arch.barrier(barrier_id=self.sgemm_bar_id,
                                          number_of_threads=self.sgemm_threads)

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

                        cute.arch.barrier(barrier_id=self.sgemm_bar_id,
                                          number_of_threads=self.sgemm_threads)

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

                            cute.arch.barrier(barrier_id=self.sgemm_bar_id,
                                              number_of_threads=self.sgemm_threads)

                            thr_group_idx  = sgemm_tidx // DIM_SPLIT
                            thr_group_lane = sgemm_tidx %  DIM_SPLIT
                            if thr_group_idx < HEADS_PER_SPLIT:
                                head_idx_global = head_base_idx * HEADS_PER_SPLIT + thr_group_idx
                                out_col = stage_idx * DIM_SPLIT + thr_group_lane
                                final_sum = cutlass.Float32(0)
                                for i in range(self.num_sgemm_warps):
                                    final_sum += smem_partial_sgemm[i, thr_group_idx, thr_group_lane]
                                partial_out[T_idx, split_idx_new, head_idx_global, out_col] = final_sum

                            cute.arch.barrier(barrier_id=self.sgemm_bar_id,
                                              number_of_threads=self.sgemm_threads)

        # Epilogue
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

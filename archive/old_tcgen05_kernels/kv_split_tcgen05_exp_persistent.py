"""kv_split_tcgen05_exp_persistent.py — Persistent across T.

Same compute body as kv_split_tcgen05_exp.py (N_REAL=2, full M=128 work,
no skip logic) but the grid drops the T dimension and each CTA loops over
the T tokens internally.  This mirrors the production kv_split kernel's
T-inner-loop structure so we can isolate the cost of that loop vs launching
one CTA per token.

Grid = (HEAD_GROUPS=8, NUM_SPLITS=16) = 128 CTAs.
Per-CTA work: T iterations of (cp.async load → MMA → score → softmax → output).
Output: partial_out shape (T, NUM_HEADS=16, NUM_SPLITS=16, DIM_SPLIT=128).
"""

import json
import math
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T


# ── Problem dims (mirrors production kv_split kernel) ───────────────────────
M               = 128                    # rows per (split, token, head_group)
N_REAL          = 2                      # heads per CTA (h0, h1)
N_MMA           = 8
K_CKV           = 512                    # head_dim_ckv (q_nope side)
K_KPE           = 64                     # head_dim_kpe (q_pe   side)
K_FULL          = K_CKV + K_KPE          # 576
DIM_SPLIT       = 128                    # output dim chunk (1 of 4 over K_CKV)

NUM_HEADS       = 16
HEAD_GROUPS     = NUM_HEADS // N_REAL    # 8
TOP_K_LEN       = 2048
NUM_SPLITS      = TOP_K_LEN // M         # 16
PS              = 64                     # page_size (only used to view ckv flat)

THREADS_PER_CTA = 512
NUM_WARPS       = THREADS_PER_CTA // 32  # 16
NUM_ROUNDS_MAX  = M // NUM_WARPS         # 8
MMA_INST_MNK    = (128, N_MMA, 16)
CTA_TILE_MNK    = (M, N_MMA, K_FULL)

# Output GEMV (FFMA2) layout
OUT_VEC         = 4                      # dims per lane per inner step
OUT_INNER_LANES = 16                     # lanes per K_TILE (16 × 4 = 64)
# Softmax
SM_WARPS        = M // 32                # 4 warps cover 128 m-rows


# ── Probe infra ───────────────────────────────────────────────────────────────

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(MLIR_T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int64(t)


@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(MLIR_T.i32(), [], "mov.u32 $0, %smid;", "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int32(t)


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES  = 64
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY

TAGS = {
    "total":          2,
    "prologue":       4,
    "load_ab":        6,
    "mma":            8,
    "score_epi":     10,
    "softmax":       12,
    "output":        14,
    "token":         16,
}
TAG_NAMES = {v: k for k, v in TAGS.items()}


def range_start(probe, row, cnt, sm_val, tag_val):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 0] = cutlass.Int64(sm_val)
    probe[row, off + 1] = cutlass.Int64(tag_val)
    probe[row, off + 2] = globaltimer_u64()


def range_stop(probe, row, cnt):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 3] = globaltimer_u64() - probe[row, off + 2]
    return cnt + cutlass.Int32(1)


def range_finalize(probe, row, cnt):
    probe[row, 0] = cutlass.Int64(cnt)


@cute.jit
def warp_reduce_add(val: cutlass.Int32, width: cutlass.Constexpr = 32) -> cutlass.Int32:
    for i in range(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def warp_reduce_add_f32(val: cutlass.Float32, width: cutlass.Constexpr = 32) -> cutlass.Float32:
    for i in range(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def warp_reduce_max_f32(val: cutlass.Float32, width: cutlass.Constexpr = 32) -> cutlass.Float32:
    for i in range(int(math.log2(width))):
        other = cute.arch.shuffle_sync_bfly(val, offset=1 << i)
        if other > val:
            val = other
    return val


# ══════════════════════════════════════════════════════════════════════════════
class KvSplitTcgen05ExpPersistent:
    def __init__(self, sm_scale: float = 0.1352337788608801,
                 T: int = 8, num_pages: int = 8462):
        self.num_stages  = 1
        self.tmem_ld_rep = N_REAL
        self.sm_scale    = sm_scale
        self.T           = T   # token count (grid.x)
        self.num_pages   = num_pages

    @cute.jit
    def __call__(
        self,
        ckv_flat:       cute.Tensor,   # (POOL, K_CKV)            bf16
        kpe_flat:       cute.Tensor,   # (POOL, K_KPE)            bf16
        q_rope:         cute.Tensor,   # (T, NUM_HEADS, K_CKV)    bf16  (= q_nope in workload)
        q_nope:         cute.Tensor,   # (T, NUM_HEADS, K_KPE)    bf16  (= q_pe   in workload)
        sparse_indices: cute.Tensor,   # (T, TOP_K_LEN)           int32
        partial_out:    cute.Tensor,   # (T, NUM_HEADS, NUM_SPLITS, DIM_SPLIT) f32
        probe:          cute.Tensor,
    ):
        ab_dtype  = cutlass.BFloat16
        acc_dtype = cutlass.Float32

        op = tcgen05.MmaF16BF16Op(
            ab_dtype, acc_dtype, MMA_INST_MNK,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE_MNK, ab_dtype, self.num_stages,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, CTA_TILE_MNK, ab_dtype, self.num_stages,
        )

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma, a_smem_layout, b_smem_layout, ab_dtype, acc_dtype,
            ckv_flat, kpe_flat, q_rope, q_nope, sparse_indices, partial_out, probe,
        ).launch(grid=[HEAD_GROUPS, NUM_SPLITS, 1],
                 block=[THREADS_PER_CTA, 1, 1])

    @cute.kernel
    def kernel(
        self,
        tiled_mma, a_smem_layout, b_smem_layout, ab_dtype, acc_dtype,
        ckv_flat:       cute.Tensor,
        kpe_flat:       cute.Tensor,
        q_rope:         cute.Tensor,
        q_nope:         cute.Tensor,
        sparse_indices: cute.Tensor,
        partial_out:    cute.Tensor,
        probe:          cute.Tensor,
    ):
        N_real:      cutlass.Constexpr = N_REAL
        tmem_ld_rep: cutlass.Constexpr = self.tmem_ld_rep
        sm_scale:    cutlass.Constexpr = self.sm_scale

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()
        hg_idx, split_idx, _ = cute.arch.block_idx()

        # Flat probe row (one slot per CTA): hg_idx * NUM_SPLITS + split_idx
        probe_row = hg_idx * cutlass.Int32(NUM_SPLITS) + split_idx

        # Token-local sparse-indices base & head_base
        si_base   = split_idx * cutlass.Int32(M)
        head_base = hg_idx    * cutlass.Int32(N_REAL)
        T_const:  cutlass.Constexpr = self.T

        # ── SMEM layout ─────────────────────────────────────────────────────
        alloc = cutlass.utils.SmemAllocator()
        sA = alloc.allocate_tensor(
            element_type=ab_dtype, layout=a_smem_layout.outer,
            byte_alignment=1024, swizzle=a_smem_layout.inner,
        )
        sB = alloc.allocate_tensor(
            element_type=ab_dtype, layout=b_smem_layout.outer,
            byte_alignment=1024, swizzle=b_smem_layout.inner,
        )
        smem_sparse = alloc.allocate_tensor(
            cutlass.Int32, cute.make_layout((M,), stride=(1,)), 4, None,
        )
        smem_red = alloc.allocate_tensor(
            cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None,
        )
        # softmax: stores per-(m, h) f32 logits / probs.  sts.64 friendly.
        smem_score = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((M, N_REAL), stride=(N_REAL, 1)), 16, None,
        )
        # output partials: (warp, head, dim)
        smem_partial = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((NUM_WARPS, N_REAL, DIM_SPLIT),
                             stride=(N_REAL * DIM_SPLIT, DIM_SPLIT, 1)),
            16, None,
        )
        # softmax cross-warp reduction (4 warps × 2 heads)
        smem_sm_red = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((NUM_WARPS, N_REAL), stride=(N_REAL, 1)), 16, None,
        )
        storage = alloc.allocate(self.shared_storage)
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        sm_val = smid_u32()
        if tidx == cutlass.Int32(0):
            range_start(probe, probe_row, cutlass.Int32(0), sm_val, TAGS["total"])

        mma_phase = cutlass.Int32(0)
        # ══════════════════════════════════════════════════════════════════
        # T-LOOP  (persistent across tokens)
        # ══════════════════════════════════════════════════════════════════
        for t_idx in cutlass.range(T_const, unroll=1):
            # Per-token probe slot base: token bracket + 6 phase slots.
            slot_base = cutlass.Int32(1) + t_idx * cutlass.Int32(7)

            if tidx == cutlass.Int32(0):
                range_start(probe, probe_row, slot_base, sm_val, TAGS["token"])

            # ══════════════════════════════════════════════════════════════════
            # 1. Prologue: load M=128 sparse_indices for this (T_idx, split_idx)
            #              and clamp negatives to 0.  No valid-count, no
            #              early-exit — every block always does the M=128 work.
            # ══════════════════════════════════════════════════════════════════
            if tidx == cutlass.Int32(0):
                range_start(probe, probe_row, slot_base + cutlass.Int32(1), sm_val, TAGS["prologue"])

            for m in cutlass.range_constexpr(M // THREADS_PER_CTA + 1):
                idx_lin = cutlass.Int32(m) * cutlass.Int32(THREADS_PER_CTA) + tidx
                if idx_lin < cutlass.Int32(M):
                    idx = sparse_indices[t_idx, si_base + idx_lin]
                    # Clamp negatives to 0 silently (output on those rows is garbage).
                    if idx < cutlass.Int32(0):
                        idx = cutlass.Int32(0)
                    smem_sparse[idx_lin] = idx
            cute.arch.sync_threads()

            # No num_valid: we always run the full M=128 body.  Re-bind for
            # clarity with the rest of the body.
            num_valid   = cutlass.Int32(M)
            round_limit = cutlass.Int32(M)

            if tidx == cutlass.Int32(0):
                range_stop(probe, probe_row, slot_base + cutlass.Int32(1))

            # ══════════════════════════════════════════════════════════════════
            # 2. cp.async A (ckv+kpe) + B (q_rope+q_nope)
            # ══════════════════════════════════════════════════════════════════
            if tidx == cutlass.Int32(0):
                range_start(probe, probe_row, slot_base + cutlass.Int32(2), sm_val, TAGS["load_ab"])

            K_TILE:        cutlass.Constexpr = 64
            K_OUTER_CKV:   cutlass.Constexpr = K_CKV  // K_TILE      # 8
            K_OUTER_FULL:  cutlass.Constexpr = K_FULL // K_TILE      # 9
            K_OUTER_KPE_IDX: cutlass.Constexpr = K_OUTER_CKV         # 8
            VEC_BF16:      cutlass.Constexpr = 8                     # 128b atom (ckv)
            K_OUTER_HALF:  cutlass.Constexpr = K_OUTER_CKV // 2      # 4
            VEC_BF16_KPE:  cutlass.Constexpr = 2                     # 32b atom (kpe)

            # ckv/q_rope tv (val=8 bf16, 256 bf16 tile)
            atom_cpa = cute.make_copy_atom(
                cpasync.CopyG2SOp(), ab_dtype, num_bits_per_copy=128,
            )
            thr_layout_warp = cute.make_layout(
                (1, (8, K_OUTER_HALF)), stride=(32, (1, 8)),
            )
            val_layout_warp = cute.make_layout(
                (1, (VEC_BF16, 1)), stride=(0, (1, 0)),
            )
            tiled_copy_warp = cute.make_tiled_copy_tv(
                atom_cpa, thr_layout_warp, val_layout_warp,
            )
            lane_copy = tiled_copy_warp.get_slice(lane_idx)

            # kpe/q_nope tv (val=2 bf16, 64 bf16 tile = single K_TILE)
            atom_cpa_kpe = cute.make_copy_atom(
                cpasync.CopyG2SOp(), ab_dtype, num_bits_per_copy=32,
            )
            thr_layout_kpe = cute.make_layout((1, 32), stride=(32, 1))
            val_layout_kpe = cute.make_layout((1, VEC_BF16_KPE), stride=(0, 1))
            tiled_copy_kpe = cute.make_tiled_copy_tv(
                atom_cpa_kpe, thr_layout_kpe, val_layout_kpe,
            )
            lane_copy_kpe = tiled_copy_kpe.get_slice(lane_idx)

            # gmem source views — pool flattened to (NUM_PAGES * PAGE_SIZE)
            N_pool: cutlass.Constexpr = self.num_pages * PS
            T_const: cutlass.Constexpr = self.T
            ckv_full = cute.make_tensor(
                ckv_flat.iterator,
                cute.make_layout(
                    (1, N_pool, (K_TILE, K_OUTER_CKV)),
                    stride=(0, K_CKV, (1, K_TILE)),
                ),
            )
            # q_rope (= workload q_nope) view: (1, T, NUM_HEADS, (K_TILE, K_OUTER_CKV))
            gB_full = cute.make_tensor(
                q_rope.iterator,
                cute.make_layout(
                    (1, T_const, NUM_HEADS, (K_TILE, K_OUTER_CKV)),
                    stride=(0, NUM_HEADS * K_CKV, K_CKV, (1, K_TILE)),
                ),
            )
            kpe_full = cute.make_tensor(
                kpe_flat.iterator,
                cute.make_layout((1, N_pool, K_TILE), stride=(0, K_KPE, 1)),
            )
            # q_nope (= workload q_pe) view
            q_nope_full = cute.make_tensor(
                q_nope.iterator,
                cute.make_layout(
                    (1, T_const, NUM_HEADS, K_TILE),
                    stride=(0, NUM_HEADS * K_KPE, K_KPE, 1),
                ),
            )

            # SMEM destination views — preserve swizzle iterator (constexpr offsets)
            sA_ckv = cute.make_tensor(
                sA.iterator,
                cute.make_layout(
                    (1, M, (K_TILE, K_OUTER_CKV)),
                    stride=(0, K_TILE, (1, M * K_TILE)),
                ),
            )
            sA_kpe = cute.make_tensor(
                sA.iterator + (K_OUTER_KPE_IDX * M * K_TILE),
                cute.make_layout((1, M, K_TILE), stride=(0, K_TILE, 1)),
            )
            sB_qr = cute.make_tensor(
                sB.iterator,
                cute.make_layout(
                    (1, N_MMA, (K_TILE, K_OUTER_CKV)),
                    stride=(0, K_TILE, (1, N_MMA * K_TILE)),
                ),
            )
            sB_qn = cute.make_tensor(
                sB.iterator + (K_OUTER_KPE_IDX * N_MMA * K_TILE),
                cute.make_layout((1, N_MMA, K_TILE), stride=(0, K_TILE, 1)),
            )

            # ---- ISSUE A (ckv + kpe per-row) ----
            for rnd in cutlass.range_constexpr(NUM_ROUNDS_MAX):
                m_local = cutlass.Int32(rnd) * cutlass.Int32(NUM_WARPS) + warp_idx
                if m_local < round_limit:
                    pool_idx = smem_sparse[m_local]

                    gA_row     = ckv_full[None, pool_idx, None]
                    sA_ckv_row = sA_ckv  [None, m_local,  None]
                    cute.copy(atom_cpa,
                              lane_copy.partition_S(gA_row),
                              lane_copy.partition_D(sA_ckv_row))

                    gA_kpe_row = kpe_full[None, pool_idx, None]
                    sA_kpe_row = sA_kpe  [None, m_local,  None]
                    cute.copy(atom_cpa_kpe,
                              lane_copy_kpe.partition_S(gA_kpe_row),
                              lane_copy_kpe.partition_D(sA_kpe_row))

            # ---- ISSUE B (q_rope + q_nope, warps 0..N_REAL-1) ----
            # Pull q for this token + (head_base + warp_idx).  warps >= N_REAL idle.
            if warp_idx < cutlass.Int32(N_REAL):
                head_idx  = head_base + warp_idx
                gB_row    = gB_full     [None, t_idx, head_idx, None]
                sB_qr_row = sB_qr       [None, warp_idx, None]
                cute.copy(atom_cpa,
                          lane_copy.partition_S(gB_row),
                          lane_copy.partition_D(sB_qr_row))

                gB_qn_row = q_nope_full[None, t_idx, head_idx, None]
                sB_qn_row = sB_qn      [None, warp_idx, None]
                cute.copy(atom_cpa_kpe,
                          lane_copy_kpe.partition_S(gB_qn_row),
                          lane_copy_kpe.partition_D(sB_qn_row))

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            if tidx == cutlass.Int32(0):
                range_stop(probe, probe_row, slot_base + cutlass.Int32(2))

            # ══════════════════════════════════════════════════════════════════
            # 3. tcgen05 MMA (K=576 → 36 k-blocks)
            # ══════════════════════════════════════════════════════════════════
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            acc_shape       = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
            tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
            num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
            tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

            if warp_idx == 0 and t_idx == cutlass.Int32(0):
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)

            tmem_barrier_id = 1
            cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)

            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                acc_dtype, alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)

            if warp_idx == 0 and t_idx == cutlass.Int32(0):
                if tidx == 0:
                    cute.arch.mbarrier_init(mma_mbar, cnt=1)
                    cute.arch.mbarrier_init_fence()
            cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)

            num_k_blocks = cute.size(tCrA, mode=[2])

            if tidx == cutlass.Int32(0):
                range_start(probe, probe_row, slot_base + cutlass.Int32(3), sm_val, TAGS["mma"])

            tcgen05_fence()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            if warp_idx == 0:
                for k_block_idx in range(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, 0)
                    cute.gemm(
                        tiled_mma, tCtAcc,
                        tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                if tidx == 0:
                    tcgen05.commit(mma_mbar)

            cute.arch.mbarrier_wait(mma_mbar, mma_phase)
            mma_phase = mma_phase ^ cutlass.Int32(1)

            if tidx == cutlass.Int32(0):
                range_stop(probe, probe_row, slot_base + cutlass.Int32(3))

            # ══════════════════════════════════════════════════════════════════
            # 4. Score epilogue: tmem → registers → smem_score (sts.64)
            # ══════════════════════════════════════════════════════════════════
            if tidx == cutlass.Int32(0):
                range_start(probe, probe_row, slot_base + cutlass.Int32(4), sm_val, TAGS["score_epi"])

            M_acc          = cute.size(tCtAcc, mode=[0, 0])
            ld_op          = tcgen05.Ld32x32bOp(tcgen05.Repetition(tmem_ld_rep))
            epi_tiler      = ((M_acc, tmem_ld_rep),)
            tCtAcc_epi     = cute.zipped_divide(tCtAcc, epi_tiler)
            copy_atom_t2r  = cute.make_copy_atom(ld_op, acc_dtype)
            tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
            tmem_thr_copy  = tmem_tiled_copy.get_slice(tidx)
            tTR_tAcc       = tmem_thr_copy.partition_S(tCtAcc_epi)
            tTR_rAcc       = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, acc_dtype)

            if tidx < cutlass.Int32(M):
                cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)
                # apply scale + sts.64 to smem_score[tidx, :]
                for n_idx in cutlass.range_constexpr(N_REAL):
                    smem_score[tidx, n_idx] = tTR_rAcc[n_idx] * cutlass.Float32(sm_scale)

            cute.arch.sync_threads()

            if tidx == cutlass.Int32(0):
                range_stop(probe, probe_row, slot_base + cutlass.Int32(4))

            # ── Free tmem only on last T iteration ─────────────────────────
            if t_idx == T_const - cutlass.Int32(1):
                if warp_idx == 0:
                    cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.barrier(barrier_id=tmem_barrier_id)
                if warp_idx == 0:
                    cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

            # ══════════════════════════════════════════════════════════════════
            # 5. Softmax over m, per head (4 active warps × 32 lanes = 128 m rows)
            # ══════════════════════════════════════════════════════════════════
            if tidx == cutlass.Int32(0):
                range_start(probe, probe_row, slot_base + cutlass.Int32(5), sm_val, TAGS["softmax"])

            NEG_INF: cutlass.Constexpr = -1.0e30

            s0 = cutlass.Float32(NEG_INF)
            s1 = cutlass.Float32(NEG_INF)
            if tidx < cutlass.Int32(M) and tidx < num_valid:
                s0 = smem_score[tidx, 0]
                s1 = smem_score[tidx, 1]

            # Warp-level max over 32 lanes (each warp covers 32 m rows)
            m0 = warp_reduce_max_f32(s0, width=32)
            m1 = warp_reduce_max_f32(s1, width=32)

            # Warps 0..SM_WARPS-1 hold valid data; rest write -inf which won't matter.
            if lane_idx == cutlass.Int32(0):
                smem_sm_red[warp_idx, 0] = m0
                smem_sm_red[warp_idx, 1] = m1
            cute.arch.sync_threads()

            # Block-level max across SM_WARPS warps (warp 0 only)
            if warp_idx == cutlass.Int32(0):
                v0 = cutlass.Float32(NEG_INF)
                v1 = cutlass.Float32(NEG_INF)
                if lane_idx < cutlass.Int32(SM_WARPS):
                    v0 = smem_sm_red[lane_idx, 0]
                    v1 = smem_sm_red[lane_idx, 1]
                v0 = warp_reduce_max_f32(v0, width=SM_WARPS)
                v1 = warp_reduce_max_f32(v1, width=SM_WARPS)
                if lane_idx == cutlass.Int32(0):
                    smem_sm_red[0, 0] = v0
                    smem_sm_red[0, 1] = v1
            cute.arch.sync_threads()

            row_max_0 = smem_sm_red[0, 0]
            row_max_1 = smem_sm_red[0, 1]

            # exp(score - row_max) and write back to smem_score
            e0 = cutlass.Float32(0)
            e1 = cutlass.Float32(0)
            if tidx < cutlass.Int32(M) and tidx < num_valid:
                e0 = cute.math.exp(s0 - row_max_0)
                e1 = cute.math.exp(s1 - row_max_1)
                smem_score[tidx, 0] = e0
                smem_score[tidx, 1] = e1

            # Sum reductions
            sum0 = warp_reduce_add_f32(e0, width=32)
            sum1 = warp_reduce_add_f32(e1, width=32)
            if lane_idx == cutlass.Int32(0):
                smem_sm_red[warp_idx, 0] = sum0
                smem_sm_red[warp_idx, 1] = sum1
            cute.arch.sync_threads()

            if warp_idx == cutlass.Int32(0):
                v0 = cutlass.Float32(0)
                v1 = cutlass.Float32(0)
                if lane_idx < cutlass.Int32(SM_WARPS):
                    v0 = smem_sm_red[lane_idx, 0]
                    v1 = smem_sm_red[lane_idx, 1]
                v0 = warp_reduce_add_f32(v0, width=SM_WARPS)
                v1 = warp_reduce_add_f32(v1, width=SM_WARPS)
                if lane_idx == cutlass.Int32(0):
                    smem_sm_red[0, 0] = v0
                    smem_sm_red[0, 1] = v1
            cute.arch.sync_threads()

            row_sum_0 = smem_sm_red[0, 0]
            row_sum_1 = smem_sm_red[0, 1]

            # Normalize: smem_score[m, h] /= row_sum[h]
            inv0 = cutlass.Float32(1.0) / row_sum_0
            inv1 = cutlass.Float32(1.0) / row_sum_1
            if tidx < cutlass.Int32(M) and tidx < num_valid:
                smem_score[tidx, 0] = e0 * inv0
                smem_score[tidx, 1] = e1 * inv1
            cute.arch.sync_threads()

            if tidx == cutlass.Int32(0):
                range_stop(probe, probe_row, slot_base + cutlass.Int32(5))

            # ══════════════════════════════════════════════════════════════════
            # 6. Output GEMV (FFMA2) — reuse sA for ckv access
            #    Build a tiled_copy that loads, per warp & per m row, exactly
            #    DIM_SPLIT=128 bf16 = 2 × K_TILE chunks via 32 lanes × vec=4.
            #    Lane mapping: lane → (k_outer = lane>>4, inner_lane = lane&15)
            #    so that dim_local = k_outer*64 + inner_lane*4 + v = lane*4 + v.
            # ══════════════════════════════════════════════════════════════════
            if tidx == cutlass.Int32(0):
                range_start(probe, probe_row, slot_base + cutlass.Int32(6), sm_val, TAGS["output"])

            # SMEM → RMEM atom: 2 bf16 = 32 bits per lane per copy.
            atom_s2r = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), ab_dtype, num_bits_per_copy=32,
            )

            # tv layout: 32 lanes × 2 vec bf16 = 64 bf16 = 1 K_TILE per call.
            # We iterate k_outer ∈ {0, 1} explicitly to cover DIM_SPLIT = 128.
            OUT_VEC_PER_KO: cutlass.Constexpr = 2          # dims per lane per k_outer
            N_KO_OUT:       cutlass.Constexpr = DIM_SPLIT // K_TILE   # 2
            OUT_VEC_TOTAL:  cutlass.Constexpr = OUT_VEC_PER_KO * N_KO_OUT   # 4

            thr_layout_out = cute.make_layout((32,), stride=(1,))
            val_layout_out = cute.make_layout((OUT_VEC_PER_KO,), stride=(1,))
            tiled_copy_out = cute.make_tiled_copy_tv(
                atom_s2r, thr_layout_out, val_layout_out,
            )
            lane_copy_out = tiled_copy_out.get_slice(lane_idx)

            out0 = cute.make_rmem_tensor(
                cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)), cutlass.Float32)
            out1 = cute.make_rmem_tensor(
                cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)), cutlass.Float32)
            for v in cutlass.range_constexpr(OUT_VEC_TOTAL):
                out0[v] = cutlass.Float32(0)
                out1[v] = cutlass.Float32(0)

            for round_idx in cutlass.range_constexpr(NUM_ROUNDS_MAX):
                m_local = cutlass.Int32(round_idx) * cutlass.Int32(NUM_WARPS) + warp_idx
                if m_local < num_valid:
                    p0 = smem_score[m_local, 0]
                    p1 = smem_score[m_local, 1]

                    for ko in cutlass.range_constexpr(N_KO_OUT):
                        # sA_ckv shape: (1, M, (K_TILE, K_OUTER_CKV=8)).  Slice
                        # to (K_TILE,) for this row & K_OUTER chunk.
                        sA_chunk = sA_ckv[0, m_local, (None, ko)]
                        src_part = lane_copy_out.partition_S(sA_chunk)
                        ckv_rmem = cute.make_rmem_tensor(src_part.shape, ab_dtype)
                        cute.copy(atom_s2r, src_part, ckv_rmem)
                        for v in cutlass.range_constexpr(OUT_VEC_PER_KO):
                            ckv_f = cutlass.Float32(ckv_rmem[v])
                            idx = ko * OUT_VEC_PER_KO + v
                            out0[idx], out1[idx] = cute.arch.fma_packed_f32x2(
                                (p0, p1), (ckv_f, ckv_f),
                                (out0[idx], out1[idx]),
                            )

            # Store partials into smem_partial[warp, head, dim_local]
            # dim_local = ko * K_TILE + lane_idx * OUT_VEC_PER_KO + v
            for ko in cutlass.range_constexpr(N_KO_OUT):
                for v in cutlass.range_constexpr(OUT_VEC_PER_KO):
                    d = (cutlass.Int32(ko) * cutlass.Int32(K_TILE)
                         + lane_idx * cutlass.Int32(OUT_VEC_PER_KO)
                         + cutlass.Int32(v))
                    idx = ko * OUT_VEC_PER_KO + v
                    smem_partial[warp_idx, 0, d] = out0[idx]
                    smem_partial[warp_idx, 1, d] = out1[idx]
            cute.arch.sync_threads()

            # Cross-warp reduction: 256 threads handle (h, dim) pairs.
            # Write into partial_out[t_idx, head_base + h, split_idx, d].
            if tidx < cutlass.Int32(N_REAL * DIM_SPLIT):
                h = tidx // cutlass.Int32(DIM_SPLIT)
                d = tidx %  cutlass.Int32(DIM_SPLIT)
                acc = cutlass.Float32(0)
                for w in cutlass.range_constexpr(NUM_WARPS):
                    acc = acc + smem_partial[w, h, d]
                partial_out[t_idx, head_base + h, split_idx, d] = acc

            cute.arch.sync_threads()

            if tidx == cutlass.Int32(0):
                range_stop(probe, probe_row, slot_base + cutlass.Int32(6))
            # Close per-token bracket
            if tidx == cutlass.Int32(0):
                range_stop(probe, probe_row, slot_base)
        # ── END OF T-LOOP BODY ─────────────────────────────────────────────

        if tidx == cutlass.Int32(0):
            range_stop(probe, probe_row, cutlass.Int32(0))
            # Total entries written: 1 (whole-CTA total) + T * 7 (per-token).
            range_finalize(probe, probe_row, cutlass.Int32(1) + T_const * cutlass.Int32(7))


# ══════════════════════════════════════════════════════════════════════════════
# Per-CTA probe aggregation (mirrors kv_split_v3_thr_warpv3_intra.py style)
# ══════════════════════════════════════════════════════════════════════════════
def _probe_events(probe_cpu, num_blocks, pid_offset=0):
    """Collect raw per-CTA events; return (events_list, global_base_ns)."""
    events = []
    base = None
    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (base is None or s < base):
                base = s
    base = base or 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt = int(data[0])
        if cnt == 0:
            continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1])
            t0  = int(data[off + 2])
            dur = int(data[off + 3])
            if t0 == 0 and dur == 0:
                continue
            events.append(dict(
                name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(t0 - base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id + pid_offset, tid=bid,
            ))
    return events, base


PHASE_ORDER = ["total", "token", "prologue", "load_ab", "mma",
               "score_epi", "softmax", "output"]


def dump_probe(probe: torch.Tensor, num_blocks: int, label: str):
    """Print per-phase aggregate stats across all CTAs and dump slowest block."""
    probe_cpu = probe.cpu().contiguous().tolist()

    # Find slowest block by total duration (tag=='total')
    max_total, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt  = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1])
            dur = int(data[off + 3])
            if tag == TAGS["total"] and dur > max_total:
                max_total, max_bid = dur, bid

    data = probe_cpu[max_bid]
    cnt  = int(data[0])
    print(f"\n--- [{label}] Slowest block {max_bid} "
          f"(total={max_total/1000:.3f}µs): {cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off])
        tag   = int(data[off + 1])
        dur   = int(data[off + 3])
        name  = TAG_NAMES.get(tag, f"tag_{tag}")
        print(f"  sm={sm_id:>3} {name:>12s}  dur={dur:>10} ns  ({dur/1000:.3f} µs)")

    # Aggregate per phase across all blocks
    tag_durs: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]
        cnt  = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1])
            dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_durs.setdefault(name, []).append(dur)

    print(f"\n[{label}] grid={num_blocks} CTAs   per-phase across blocks (ns):")
    print(f"{'phase':>12s} {'min µs':>9s} {'avg µs':>9s} {'max µs':>9s} {'count':>7s}")
    print("-" * 52)
    agg = {}
    for name in PHASE_ORDER:
        if name not in tag_durs:
            continue
        ds = tag_durs[name]
        mn, mx = min(ds), max(ds)
        av = sum(ds) / len(ds)
        print(f"{name:>12s} {mn/1000:>9.3f} {av/1000:>9.3f} "
              f"{mx/1000:>9.3f} {len(ds):>7d}")
        agg[name] = {"min_us": mn / 1000.0, "avg_us": av / 1000.0,
                     "max_us": mx / 1000.0, "count": len(ds)}

    return agg, _probe_events(probe_cpu, num_blocks)


def run_workload(workload_idx: int) -> tuple:
    """Load real workload data, compile, profile, return (summary_json, trace_json)."""
    import os, json as _json
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [_json.loads(l) for l in open(JSONL)]
    w   = workloads[workload_idx]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]
    _uuid, _T, max_valid = WORKLOAD_INFO[workload_idx]
    sm_scale = 0.1352337788608801

    print(f"Workload {workload_idx + 1}: uuid={_uuid}  T={T}  P={P}  MaxValid={max_valid}")
    print(f"Grid = (HEAD_GROUPS={HEAD_GROUPS}, NUM_SPLITS={NUM_SPLITS}) "
          f"= {HEAD_GROUPS * NUM_SPLITS} CTAs   (each CTA loops T={T})")

    # ── Build inputs ────────────────────────────────────────────────────────
    q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)            # randn
    sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()       # (T, TOPK) int32

    # Flatten ckv/kpe pool to (P*PS, K_*)
    ckv_flat = ckv.view(P * PS, K_CKV).contiguous()
    kpe_flat = kpe.view(P * PS, K_KPE).contiguous()

    partial_out = torch.zeros((T, NUM_HEADS, NUM_SPLITS, DIM_SPLIT),
                              dtype=torch.float32, device="cuda")
    num_blocks  = HEAD_GROUPS * NUM_SPLITS
    probe = torch.zeros((num_blocks, PROBE_COLS), dtype=torch.int64, device="cuda")

    # ── Compile ─────────────────────────────────────────────────────────────
    ckv_  = from_dlpack(ckv_flat,    assumed_align=128)
    kpe_  = from_dlpack(kpe_flat,    assumed_align=128)
    qn_   = from_dlpack(q_nope,      assumed_align=128)   # (T, H, K_CKV) → q_rope side
    qp_   = from_dlpack(q_pe,        assumed_align=128)   # (T, H, K_KPE) → q_nope side
    si_   = from_dlpack(si,          assumed_align=16)
    pout_ = from_dlpack(partial_out, assumed_align=16)
    probe_= from_dlpack(probe,       assumed_align=8)

    print("Compiling kv_split_tcgen05_exp_persistent...")
    kernel = KvSplitTcgen05ExpPersistent(sm_scale=sm_scale, T=T, num_pages=P)
    compiled = cute.compile(kernel, ckv_, kpe_, qn_, qp_, si_, pout_, probe_)

    # ── Warmup + profile launch ─────────────────────────────────────────────
    for _ in range(3):
        partial_out.zero_(); probe.zero_()
        compiled(ckv_, kpe_, qn_, qp_, si_, pout_, probe_)
    torch.cuda.synchronize()
    partial_out.zero_(); probe.zero_()
    compiled(ckv_, kpe_, qn_, qp_, si_, pout_, probe_)
    torch.cuda.synchronize()

    # ── Correctness on FULL TILES ONLY ─────────────────────────────────────
    # A "full tile" is (t, split) where every sparse_indices[t, split*M:(split+1)*M]
    # is non-negative (no clamped slots).  We verify partial_out at those slots.
    si_cpu = si.cpu()
    full_tiles = []
    for t in range(T):
        for split in range(NUM_SPLITS):
            slab = si_cpu[t, split * M:(split + 1) * M]
            if (slab >= 0).all().item():
                full_tiles.append((t, split))

    print(f"Full tiles: {len(full_tiles)} / {T * NUM_SPLITS}")

    pass_cnt, fail_cnt, max_diff = 0, 0, 0.0
    if full_tiles:
        ckv_f = ckv_flat.float()
        kpe_f = kpe_flat.float()
        qn_f  = q_nope.float()        # q_rope side  (T, H, K_CKV)
        qp_f  = q_pe.float()          # q_nope side  (T, H, K_KPE)
        for t, split in full_tiles[:8]:                        # cap to keep host work small
            slab = si_cpu[t, split * M:(split + 1) * M].long()
            ckv_v = ckv_f[slab]            # (M, K_CKV)
            kpe_v = kpe_f[slab]            # (M, K_KPE)
            for hg in range(HEAD_GROUPS):
                head_lo = hg * N_REAL
                qr_h = qn_f[t, head_lo:head_lo + N_REAL]      # (2, K_CKV)
                qn_h = qp_f[t, head_lo:head_lo + N_REAL]      # (2, K_KPE)
                score = (ckv_v @ qr_h.T + kpe_v @ qn_h.T) * sm_scale  # (M, 2)
                row_max = score.max(dim=0, keepdim=True).values
                e = torch.exp(score - row_max)
                p = e / e.sum(dim=0, keepdim=True)            # (M, 2)
                ref = p.T @ ckv_v[:, :DIM_SPLIT]              # (2, DIM_SPLIT)
                got = partial_out[t, head_lo:head_lo + N_REAL, split, :].float().cpu()
                diff = (got - ref.cpu()).abs().max().item()
                max_diff = max(max_diff, diff)
                if diff < 1e-2:
                    pass_cnt += 1
                else:
                    fail_cnt += 1
    print(f"Correctness on first {pass_cnt + fail_cnt} (tile, head_grp) pairs: "
          f"{pass_cnt} PASS / {fail_cnt} FAIL  max_diff={max_diff:.5f}")

    # ── Aggregate intra probes ──────────────────────────────────────────────
    agg, (events, base) = dump_probe(
        probe, num_blocks, label=f"WL{workload_idx + 1} grid={num_blocks}",
    )

    summary = {
        "workload_idx": workload_idx,
        "uuid": _uuid, "T": T, "P": P,
        "num_blocks": num_blocks,
        "full_tiles_total": len(full_tiles),
        "correct_pass": pass_cnt, "correct_fail": fail_cnt,
        "max_diff": max_diff,
        "per_phase": agg,
    }
    trace = json.dumps({
        "traceEvents": [
            {"name": "process_name", "ph": "M", "pid": 0, "tid": 0,
             "args": {"name": f"WL{workload_idx + 1} T={T} grid={num_blocks}"}},
        ] + events,
        "displayTimeUnit": "ns",
    })
    return json.dumps(summary, indent=2), trace


def run_intra(workload_idx: int = 22) -> tuple:
    return run_workload(workload_idx)


if __name__ == "__main__":
    summary, trace = run_intra()
    print(summary)


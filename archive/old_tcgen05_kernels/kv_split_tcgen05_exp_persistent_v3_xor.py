"""kv_split_tcgen05_exp_persistent_v3_xor.py — v3 + XOR split-swizzle.

Same two-branch routing as v3 (mma_path / fastgemv_path / skip), but each
token applies an XOR over the CTA's split index:

    split_idx_xor = t_idx ^ split_idx

This is a bijective permutation of [0, NUM_SPLITS) for every t_idx (T=7
<= NUM_SPLITS=16), so each split is still written exactly once per token.
The goal is to spread per-split workload imbalance — heavily skewed first
splits in particular — across all CTAs over the T-loop.

All measurement probes are kept identical to v3 (no extra syncs).
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
T_CHUNK         = 4                      # tokens processed per chunk (share q + sparse)

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

# Output cross-warp reduction staging (4 stages × 4 warps each = 16 warps).
NUM_STAGES_RED  = 4
WARPS_PER_STAGE = NUM_WARPS // NUM_STAGES_RED   # 4


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
    "chunk_prologue": 4,
    "mma_path":       6,
    "fastgemv_path":  8,
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
class KvSplitTcgen05ExpPersistentV3Xor:
    def __init__(self, sm_scale: float = 0.1352337788608801,
                 T: int = 8, num_pages: int = 8462):
        self.num_stages  = 1
        # Score-epi loads all N=8 MMA cols at once; per-token softmax slices 2.
        self.tmem_ld_rep = N_MMA
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
        # Per-chunk: 4 requests' full TOP_K_LEN sparse indices.
        # Each warpgroup (128 thr) loads one row of 2048 i32 = 8 KB.
        smem_sparse_indices = alloc.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((T_CHUNK, TOP_K_LEN), stride=(TOP_K_LEN, 1)),
            4, None,
        )
        # Per-chunk per-split num_valid (i32, T_CHUNK*NUM_SPLITS = 256 B).
        smem_num_valid_per_split = alloc.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((T_CHUNK, NUM_SPLITS), stride=(NUM_SPLITS, 1)),
            4, None,
        )
        # softmax: stores per-(m, h) f32 logits / probs.  sts.64 friendly.
        smem_score = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((M, N_REAL), stride=(N_REAL, 1)), 16, None,
        )
        # output partials, 4-stage cross-warp reduce buffer (4 KB).
        # Both branches accumulate per-warp output in registers, then write
        # 4 warps at a time → smem → cross-stage accumulate in rmem.
        smem_partial_st = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((NUM_STAGES_RED, N_REAL, DIM_SPLIT),
                             stride=(N_REAL * DIM_SPLIT, DIM_SPLIT, 1)),
            16, None,
        )
        # softmax cross-warp reduction (warps × 2 heads)
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

        # ── Hoisted: cp.async setup (constexpr) ────────────────────────────
        K_TILE:        cutlass.Constexpr = 64
        K_OUTER_CKV:   cutlass.Constexpr = K_CKV  // K_TILE      # 8
        K_OUTER_FULL:  cutlass.Constexpr = K_FULL // K_TILE      # 9
        K_OUTER_KPE_IDX: cutlass.Constexpr = K_OUTER_CKV         # 8
        VEC_BF16:      cutlass.Constexpr = 8
        K_OUTER_HALF:  cutlass.Constexpr = K_OUTER_CKV // 2      # 4
        VEC_BF16_KPE:  cutlass.Constexpr = 2

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

        atom_cpa_kpe = cute.make_copy_atom(
            cpasync.CopyG2SOp(), ab_dtype, num_bits_per_copy=32,
        )
        thr_layout_kpe = cute.make_layout((1, 32), stride=(32, 1))
        val_layout_kpe = cute.make_layout((1, VEC_BF16_KPE), stride=(0, 1))
        tiled_copy_kpe = cute.make_tiled_copy_tv(
            atom_cpa_kpe, thr_layout_kpe, val_layout_kpe,
        )
        lane_copy_kpe = tiled_copy_kpe.get_slice(lane_idx)

        N_pool: cutlass.Constexpr = self.num_pages * PS

        ckv_full = cute.make_tensor(
            ckv_flat.iterator,
            cute.make_layout(
                (1, N_pool, (K_TILE, K_OUTER_CKV)),
                stride=(0, K_CKV, (1, K_TILE)),
            ),
        )
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
        q_nope_full = cute.make_tensor(
            q_nope.iterator,
            cute.make_layout(
                (1, T_const, NUM_HEADS, K_TILE),
                stride=(0, NUM_HEADS * K_KPE, K_KPE, 1),
            ),
        )

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

        # ── Hoisted: tmem alloc + mbarrier init ───────────────────────────
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape       = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
        tmem_barrier_id = 1
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)
        num_k_blocks = cute.size(tCrA, mode=[2])

        # ── Hoisted: score-epi + output-GEMV setup (constexpr) ────────────
        M_acc          = cute.size(tCtAcc, mode=[0, 0])
        ld_op          = tcgen05.Ld32x32bOp(tcgen05.Repetition(tmem_ld_rep))
        epi_tiler      = ((M_acc, tmem_ld_rep),)
        tCtAcc_epi     = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r  = cute.make_copy_atom(ld_op, acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy  = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc       = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc       = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, acc_dtype)

        atom_s2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), ab_dtype, num_bits_per_copy=32,
        )
        OUT_VEC_PER_KO: cutlass.Constexpr = 2
        N_KO_OUT:       cutlass.Constexpr = DIM_SPLIT // K_TILE   # 2
        OUT_VEC_TOTAL:  cutlass.Constexpr = OUT_VEC_PER_KO * N_KO_OUT   # 4
        thr_layout_out = cute.make_layout((32,), stride=(1,))
        val_layout_out = cute.make_layout((OUT_VEC_PER_KO,), stride=(1,))
        tiled_copy_out = cute.make_tiled_copy_tv(
            atom_s2r, thr_layout_out, val_layout_out,
        )
        lane_copy_out = tiled_copy_out.get_slice(lane_idx)

        # ── FastGEMV warp-reduce score helpers (constexpr) ────────────────
        # Each warp loads K_TILE elements via 32 lanes × 2 elements (s2r).
        SCORE_VEC_PER_KO:  cutlass.Constexpr = 2
        ROWS_PER_WARP:     cutlass.Constexpr = 4
        ROWS_PER_ROUND_S:  cutlass.Constexpr = NUM_WARPS * ROWS_PER_WARP   # 64
        NUM_SCORE_ROUNDS:  cutlass.Constexpr = M // ROWS_PER_ROUND_S       # 2
        thr_layout_sc = cute.make_layout((32,), stride=(1,))
        val_layout_sc = cute.make_layout((SCORE_VEC_PER_KO,), stride=(1,))
        tiled_copy_sc = cute.make_tiled_copy_tv(
            atom_s2r, thr_layout_sc, val_layout_sc,
        )
        lane_copy_sc = tiled_copy_sc.get_slice(lane_idx)

        # ── Sparse-load helpers ───────────────────────────────────────────
        # Vectorized direct LDG (no cp.async): each warpgroup (128 thr) walks
        # one row of TOP_K_LEN=2048 i32 with vec4 (128-bit) loads.  Per pass
        # 128*4 = 512 i32 covered → 4 passes per WG.  During the same pass we
        # count non-negatives and clamp -1→0 in smem.  Per-WG barrier lets
        # each warpgroup finish its row without waiting for the others.
        wg_idx        = warp_idx // cutlass.Int32(4)               # 0..3
        warp_in_wg    = warp_idx &  cutlass.Int32(3)               # 0..3
        thr_idx_in_wg = warp_in_wg * cutlass.Int32(32) + lane_idx  # 0..127
        VEC_SPARSE:    cutlass.Constexpr = 4
        TOP_K_CHUNKS:  cutlass.Constexpr = TOP_K_LEN // VEC_SPARSE  # 512
        SPARSE_PASSES: cutlass.Constexpr = TOP_K_CHUNKS // 128      # 4
        si_vec      = cute.zipped_divide(sparse_indices,      (1, VEC_SPARSE))
        smem_si_vec = cute.zipped_divide(smem_sparse_indices, (1, VEC_SPARSE))

        # ══════════════════════════════════════════════════════════════════
        # CHUNK LOOP  (NUM_CHUNKS = ceil(T/T_CHUNK))
        # ══════════════════════════════════════════════════════════════════
        NUM_CHUNKS:  cutlass.Constexpr = (T_const + T_CHUNK - 1) // T_CHUNK
        CHUNK_SLOTS: cutlass.Constexpr = 1 + T_CHUNK    # chunk_prologue + 1 path slot per token

        for chunk_idx in cutlass.range(NUM_CHUNKS, unroll=1):
            chunk_slot_base = cutlass.Int32(1) + chunk_idx * cutlass.Int32(CHUNK_SLOTS)
            chunk_start     = chunk_idx * cutlass.Int32(T_CHUNK)

            # ── chunk_prologue: q (cp.async) + sparse (vec LDG + count + clamp) ──
            if tidx == cutlass.Int32(0):
                range_start(probe, probe_row, chunk_slot_base, sm_val,
                            TAGS["chunk_prologue"])

            # 1. q (qrope+qnope): 4 tokens × 2 heads = 8 sB rows (warps 0..7).
            #    Fire cp.async first so it overlaps with the sparse scan below.
            if warp_idx < cutlass.Int32(8):
                t_in_chunk_w = warp_idx >> cutlass.Int32(1)        # 0..3
                h_in_grp_w   = warp_idx &  cutlass.Int32(1)        # 0..1
                t_global_w   = chunk_start + t_in_chunk_w
                safe_tw      = t_global_w
                if t_global_w >= T_const:
                    safe_tw = cutlass.Int32(0)
                head_idx_w   = head_base + h_in_grp_w
                sB_row_w     = warp_idx                             # 0..7

                gB_row    = gB_full     [None, safe_tw, head_idx_w, None]
                sB_qr_row = sB_qr       [None, sB_row_w, None]
                cute.copy(atom_cpa,
                          lane_copy.partition_S(gB_row),
                          lane_copy.partition_D(sB_qr_row))

                gB_qn_row = q_nope_full[None, safe_tw, head_idx_w, None]
                sB_qn_row = sB_qn      [None, sB_row_w, None]
                cute.copy(atom_cpa_kpe,
                          lane_copy_kpe.partition_S(gB_qn_row),
                          lane_copy_kpe.partition_D(sB_qn_row))

            cute.arch.cp_async_commit_group()

            # 2. sparse_indices: 4 rows × 2048 i32 (one row per WG).
            #    Vec4 LDG straight to register, count non-neg, clamp -1→0,
            #    write to smem.  No cp.async overhead.
            row_t  = chunk_start + wg_idx
            safe_t = row_t
            if row_t >= T_const:
                safe_t = cutlass.Int32(0)

            per_pass_count = cute.make_rmem_tensor(
                cute.make_layout((SPARSE_PASSES,), stride=(1,)), cutlass.Int32)
            for p in cutlass.range_constexpr(SPARSE_PASSES):
                per_pass_count[p] = cutlass.Int32(0)

            for p in cutlass.range_constexpr(SPARSE_PASSES):
                chunk = cutlass.Int32(p) * cutlass.Int32(128) + thr_idx_in_wg
                vec = si_vec[(0, None), (safe_t, chunk)].load()
                for v in cutlass.range_constexpr(VEC_SPARSE):
                    val = vec[v]
                    if val < cutlass.Int32(0):
                        val = cutlass.Int32(0)
                    else:
                        per_pass_count[p] = per_pass_count[p] + cutlass.Int32(1)
                    smem_si_vec[(0, v), (wg_idx, chunk)] = val

            # Per-pass intra-warp reduction → per-split count.
            # Warp `warp_in_wg` in pass p owns split (p*4 + warp_in_wg) of row wg_idx.
            for p in cutlass.range_constexpr(SPARSE_PASSES):
                pc = warp_reduce_add(per_pass_count[p], width=32)
                if lane_idx == cutlass.Int32(0):
                    smem_num_valid_per_split[
                        wg_idx, cutlass.Int32(p * 4) + warp_in_wg] = pc

            # 3. Wait for q cp.async (only this stalls the whole CTA).
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            if tidx == cutlass.Int32(0):
                range_stop(probe, probe_row, chunk_slot_base)

            round_limit = cutlass.Int32(M)

            # ══════════════════════════════════════════════════════════════
            # PER-TOKEN LOOP within chunk (T_CHUNK iterations, unrolled)
            #
            #   nv_split == 0   → no-op (no gmem write)
            #   0 < nv < M      → FastGEMV path  (probe: fastgemv_path)
            #   nv == M         → tcgen05  path  (probe: mma_path)
            #
            # Each branch has ONE probe placed AFTER the branch's existing
            # final sync (no extra syncs introduced by measurement).
            # ══════════════════════════════════════════════════════════════
            for t_in_chunk in cutlass.range_constexpr(T_CHUNK):
                t_idx          = chunk_start + cutlass.Int32(t_in_chunk)
                # Stride-7 rotation swizzle: each token rotates which split
                # this CTA owns by 7*t_idx mod NUM_SPLITS.  GCD(7,16)=1 so it
                # is a bijection per t_idx.  Compared to XOR (coset trap when
                # T<=NS/2) and +1 rotation (window stays in heavy contiguous
                # region), stride=7 scatters visits non-contiguously across
                # the 16-split ring — see scripts/sim_split_swizzle.py.
                split_idx_xor  = (split_idx + t_idx * cutlass.Int32(7)) % cutlass.Int32(NUM_SPLITS)
                si_base_xor    = split_idx_xor * cutlass.Int32(M)
                slot_base = chunk_slot_base + cutlass.Int32(1 + t_in_chunk)
                nv_split  = smem_num_valid_per_split[t_in_chunk, split_idx_xor]

                # Skip ALL work and probe when nv_split == 0.  Correctness
                # validator only checks full-tile slots (nv == M); production
                # reduce kernel masks via max_seq_len.
                if nv_split != cutlass.Int32(0):

                    if nv_split == cutlass.Int32(M):
                        # ══════════════════════════════════════════════════
                        # ── tcgen05 path (full 128 rows, no bounds checks)
                        # ══════════════════════════════════════════════════
                        if tidx == cutlass.Int32(0):
                            range_start(probe, probe_row, slot_base,
                                        sm_val, TAGS["mma_path"])

                        # cp.async A (ckv+kpe) — all M rows, unrolled
                        for rnd in cutlass.range_constexpr(NUM_ROUNDS_MAX):
                            m_local = cutlass.Int32(rnd) * cutlass.Int32(NUM_WARPS) + warp_idx
                            pool_idx = smem_sparse_indices[t_in_chunk, si_base_xor + m_local]
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
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_view_async_shared()
                        cute.arch.sync_threads()

                        # MMA
                        tcgen05_fence()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        if warp_idx == 0:
                            for k_block_idx in range(num_k_blocks):
                                k_block_coord = (None, None, k_block_idx, 0)
                                cute.gemm(tiled_mma, tCtAcc,
                                          tCrA[k_block_coord],
                                          tCrB[k_block_coord], tCtAcc)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            if tidx == 0:
                                tcgen05.commit(mma_mbar)
                        cute.arch.mbarrier_wait(mma_mbar, mma_phase)
                        mma_phase = mma_phase ^ cutlass.Int32(1)

                        # Score epi: tmem → smem_score
                        if tidx < cutlass.Int32(M):
                            cute.copy(tmem_tiled_copy,
                                      tTR_tAcc[None, None, 0], tTR_rAcc)
                            col_start = t_in_chunk * 2  # constexpr
                            for n_idx in cutlass.range_constexpr(N_REAL):
                                smem_score[tidx, n_idx] = (
                                    tTR_rAcc[col_start + n_idx]
                                    * cutlass.Float32(sm_scale)
                                )
                        cute.arch.sync_threads()

                        # Softmax (no num_valid mask: nv == M)
                        NEG_INF: cutlass.Constexpr = -1.0e30
                        s0 = cutlass.Float32(NEG_INF)
                        s1 = cutlass.Float32(NEG_INF)
                        if tidx < cutlass.Int32(M):
                            s0 = smem_score[tidx, 0]
                            s1 = smem_score[tidx, 1]
                        m0 = warp_reduce_max_f32(s0, width=32)
                        m1 = warp_reduce_max_f32(s1, width=32)
                        if lane_idx == cutlass.Int32(0):
                            smem_sm_red[warp_idx, 0] = m0
                            smem_sm_red[warp_idx, 1] = m1
                        cute.arch.sync_threads()
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

                        e0 = cutlass.Float32(0)
                        e1 = cutlass.Float32(0)
                        if tidx < cutlass.Int32(M):
                            e0 = cute.math.exp(s0 - row_max_0)
                            e1 = cute.math.exp(s1 - row_max_1)
                            smem_score[tidx, 0] = e0
                            smem_score[tidx, 1] = e1
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
                        inv0 = cutlass.Float32(1.0) / row_sum_0
                        inv1 = cutlass.Float32(1.0) / row_sum_1
                        if tidx < cutlass.Int32(M):
                            smem_score[tidx, 0] = e0 * inv0
                            smem_score[tidx, 1] = e1 * inv1
                        cute.arch.sync_threads()

                        # Output GEMV — per-warp accumulator across NUM_ROUNDS rows
                        # Sources from sA_ckv (already in smem from cp.async).
                        out0 = cute.make_rmem_tensor(
                            cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)),
                            cutlass.Float32)
                        out1 = cute.make_rmem_tensor(
                            cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)),
                            cutlass.Float32)
                        for v in cutlass.range_constexpr(OUT_VEC_TOTAL):
                            out0[v] = cutlass.Float32(0)
                            out1[v] = cutlass.Float32(0)
                        for round_idx in cutlass.range_constexpr(NUM_ROUNDS_MAX):
                            m_local = (cutlass.Int32(round_idx)
                                       * cutlass.Int32(NUM_WARPS) + warp_idx)
                            p0 = smem_score[m_local, 0]
                            p1 = smem_score[m_local, 1]
                            for ko in cutlass.range_constexpr(N_KO_OUT):
                                sA_chunk = sA_ckv[0, m_local, (None, ko)]
                                src_part = lane_copy_out.partition_S(sA_chunk)
                                ckv_rmem = cute.make_rmem_tensor(src_part.shape,
                                                                  ab_dtype)
                                cute.copy(atom_s2r, src_part, ckv_rmem)
                                for v in cutlass.range_constexpr(OUT_VEC_PER_KO):
                                    ckv_f = cutlass.Float32(ckv_rmem[v])
                                    idx = ko * OUT_VEC_PER_KO + v
                                    out0[idx], out1[idx] = (
                                        cute.arch.fma_packed_f32x2(
                                            (p0, p1), (ckv_f, ckv_f),
                                            (out0[idx], out1[idx]))
                                    )

                        # 4-stage cross-warp reduction → partial_out (gmem)
                        my_h = tidx // cutlass.Int32(DIM_SPLIT)   # 0 or 1
                        my_d = tidx %  cutlass.Int32(DIM_SPLIT)
                        acc  = cutlass.Float32(0)
                        for stage in cutlass.range_constexpr(NUM_STAGES_RED):
                            warp_lo = cutlass.Int32(stage * WARPS_PER_STAGE)
                            warp_hi = cutlass.Int32((stage + 1) * WARPS_PER_STAGE)
                            if warp_idx >= warp_lo and warp_idx < warp_hi:
                                w_in_st = warp_idx - warp_lo
                                for ko in cutlass.range_constexpr(N_KO_OUT):
                                    for v in cutlass.range_constexpr(OUT_VEC_PER_KO):
                                        d = (cutlass.Int32(ko) * cutlass.Int32(K_TILE)
                                             + lane_idx * cutlass.Int32(OUT_VEC_PER_KO)
                                             + cutlass.Int32(v))
                                        idx = ko * OUT_VEC_PER_KO + v
                                        smem_partial_st[w_in_st, 0, d] = out0[idx]
                                        smem_partial_st[w_in_st, 1, d] = out1[idx]
                            cute.arch.sync_threads()
                            if tidx < cutlass.Int32(N_REAL * DIM_SPLIT):
                                for w in cutlass.range_constexpr(WARPS_PER_STAGE):
                                    acc = acc + smem_partial_st[w, my_h, my_d]
                            cute.arch.sync_threads()

                        if t_idx < T_const:
                            if tidx < cutlass.Int32(N_REAL * DIM_SPLIT):
                                partial_out[t_idx, head_base + my_h,
                                            split_idx_xor, my_d] = acc

                        # Final sync of branch (already covered by last reduce sync).
                        if tidx == cutlass.Int32(0):
                            range_stop(probe, probe_row, slot_base)

                    else:
                        # ══════════════════════════════════════════════════
                        # ── FastGEMV path (0 < nv < M)
                        # q from sB (already loaded in chunk_prologue)
                        # CKV+KPE loaded straight from gmem (no cp.async A)
                        # Early-exit on warp-level via num_valid mask
                        # ══════════════════════════════════════════════════
                        if tidx == cutlass.Int32(0):
                            range_start(probe, probe_row, slot_base,
                                        sm_val, TAGS["fastgemv_path"])

                        num_valid = nv_split

                        # ── Score: 4-row interleaved per warp ──────────────
                        for round_idx in cutlass.range_constexpr(NUM_SCORE_ROUNDS):
                            base_row = (cutlass.Int32(round_idx)
                                        * cutlass.Int32(ROWS_PER_ROUND_S)
                                        + warp_idx * cutlass.Int32(ROWS_PER_WARP))
                            if base_row < num_valid:
                                # 4 sparse pool indices for this warp's 4 rows.
                                # Negatives already clamped → 0 in chunk_prologue
                                # so OOB rows alias pool[0] (harmless reads).
                                pidx0 = smem_sparse_indices[t_in_chunk,
                                            si_base_xor + base_row + cutlass.Int32(0)]
                                pidx1 = smem_sparse_indices[t_in_chunk,
                                            si_base_xor + base_row + cutlass.Int32(1)]
                                pidx2 = smem_sparse_indices[t_in_chunk,
                                            si_base_xor + base_row + cutlass.Int32(2)]
                                pidx3 = smem_sparse_indices[t_in_chunk,
                                            si_base_xor + base_row + cutlass.Int32(3)]

                                for h_local in cutlass.range_constexpr(N_REAL):
                                    h_in_sB = t_in_chunk * 2 + h_local  # constexpr
                                    sums = cute.make_rmem_tensor(
                                        cute.make_layout((ROWS_PER_WARP,),
                                                         stride=(1,)),
                                        cutlass.Float32,
                                    )
                                    for r in cutlass.range_constexpr(ROWS_PER_WARP):
                                        sums[r] = cutlass.Float32(0)

                                    # CKV: walk K_OUTER_CKV chunks of K_TILE
                                    for ko in cutlass.range_constexpr(K_OUTER_CKV):
                                        # q from sB (smem)
                                        q_chunk = sB_qr[0, h_in_sB, (None, ko)]
                                        q_part  = lane_copy_sc.partition_S(q_chunk)
                                        q_rmem  = cute.make_rmem_tensor(
                                            q_part.shape, ab_dtype)
                                        cute.copy(atom_s2r, q_part, q_rmem)
                                        # 4 rows of ckv from gmem (LDG.32 vec=2)
                                        a0_chunk = ckv_full[0, pidx0, (None, ko)]
                                        a1_chunk = ckv_full[0, pidx1, (None, ko)]
                                        a2_chunk = ckv_full[0, pidx2, (None, ko)]
                                        a3_chunk = ckv_full[0, pidx3, (None, ko)]
                                        a0p = lane_copy_sc.partition_S(a0_chunk)
                                        a1p = lane_copy_sc.partition_S(a1_chunk)
                                        a2p = lane_copy_sc.partition_S(a2_chunk)
                                        a3p = lane_copy_sc.partition_S(a3_chunk)
                                        a0r = a0p.load()
                                        a1r = a1p.load()
                                        a2r = a2p.load()
                                        a3r = a3p.load()
                                        for v in cutlass.range_constexpr(SCORE_VEC_PER_KO):
                                            qv = cutlass.Float32(q_rmem[v])
                                            sums[0] = sums[0] + qv * cutlass.Float32(a0r[v])
                                            sums[1] = sums[1] + qv * cutlass.Float32(a1r[v])
                                            sums[2] = sums[2] + qv * cutlass.Float32(a2r[v])
                                            sums[3] = sums[3] + qv * cutlass.Float32(a3r[v])

                                    # KPE: 1 chunk of K_TILE
                                    qn_chunk = sB_qn[0, h_in_sB, None]
                                    qn_part  = lane_copy_sc.partition_S(qn_chunk)
                                    qn_rmem  = cute.make_rmem_tensor(qn_part.shape,
                                                                      ab_dtype)
                                    cute.copy(atom_s2r, qn_part, qn_rmem)
                                    k0_chunk = kpe_full[0, pidx0, None]
                                    k1_chunk = kpe_full[0, pidx1, None]
                                    k2_chunk = kpe_full[0, pidx2, None]
                                    k3_chunk = kpe_full[0, pidx3, None]
                                    k0r = lane_copy_sc.partition_S(k0_chunk).load()
                                    k1r = lane_copy_sc.partition_S(k1_chunk).load()
                                    k2r = lane_copy_sc.partition_S(k2_chunk).load()
                                    k3r = lane_copy_sc.partition_S(k3_chunk).load()
                                    for v in cutlass.range_constexpr(SCORE_VEC_PER_KO):
                                        qv = cutlass.Float32(qn_rmem[v])
                                        sums[0] = sums[0] + qv * cutlass.Float32(k0r[v])
                                        sums[1] = sums[1] + qv * cutlass.Float32(k1r[v])
                                        sums[2] = sums[2] + qv * cutlass.Float32(k2r[v])
                                        sums[3] = sums[3] + qv * cutlass.Float32(k3r[v])

                                    # Warp reduce + write smem_score (mask)
                                    for r in cutlass.range_constexpr(ROWS_PER_WARP):
                                        sums[r] = warp_reduce_add_f32(sums[r],
                                                                       width=32)
                                        row = base_row + cutlass.Int32(r)
                                        if lane_idx == cutlass.Int32(0) and row < num_valid:
                                            smem_score[row, h_local] = (
                                                sums[r] * cutlass.Float32(sm_scale)
                                            )
                        cute.arch.sync_threads()

                        # ── Softmax (with num_valid mask) ──────────────────
                        NEG_INF: cutlass.Constexpr = -1.0e30
                        s0 = cutlass.Float32(NEG_INF)
                        s1 = cutlass.Float32(NEG_INF)
                        if tidx < cutlass.Int32(M) and tidx < num_valid:
                            s0 = smem_score[tidx, 0]
                            s1 = smem_score[tidx, 1]
                        m0 = warp_reduce_max_f32(s0, width=32)
                        m1 = warp_reduce_max_f32(s1, width=32)
                        if lane_idx == cutlass.Int32(0):
                            smem_sm_red[warp_idx, 0] = m0
                            smem_sm_red[warp_idx, 1] = m1
                        cute.arch.sync_threads()
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
                        e0 = cutlass.Float32(0)
                        e1 = cutlass.Float32(0)
                        if tidx < cutlass.Int32(M) and tidx < num_valid:
                            e0 = cute.math.exp(s0 - row_max_0)
                            e1 = cute.math.exp(s1 - row_max_1)
                            smem_score[tidx, 0] = e0
                            smem_score[tidx, 1] = e1
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
                        inv0 = cutlass.Float32(1.0) / row_sum_0
                        inv1 = cutlass.Float32(1.0) / row_sum_1
                        if tidx < cutlass.Int32(M) and tidx < num_valid:
                            smem_score[tidx, 0] = e0 * inv0
                            smem_score[tidx, 1] = e1 * inv1
                        cute.arch.sync_threads()

                        # ── Output GEMV (CKV from gmem; early exit) ────────
                        out0 = cute.make_rmem_tensor(
                            cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)),
                            cutlass.Float32)
                        out1 = cute.make_rmem_tensor(
                            cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)),
                            cutlass.Float32)
                        for v in cutlass.range_constexpr(OUT_VEC_TOTAL):
                            out0[v] = cutlass.Float32(0)
                            out1[v] = cutlass.Float32(0)
                        for round_idx in cutlass.range_constexpr(NUM_ROUNDS_MAX):
                            m_local = (cutlass.Int32(round_idx)
                                       * cutlass.Int32(NUM_WARPS) + warp_idx)
                            if m_local < num_valid:
                                pool_idx = smem_sparse_indices[t_in_chunk,
                                                               si_base_xor + m_local]
                                p0 = smem_score[m_local, 0]
                                p1 = smem_score[m_local, 1]
                                for ko in cutlass.range_constexpr(N_KO_OUT):
                                    g_chunk = ckv_full[0, pool_idx, (None, ko)]
                                    src_part = lane_copy_out.partition_S(g_chunk)
                                    ckv_rmem = src_part.load()
                                    for v in cutlass.range_constexpr(OUT_VEC_PER_KO):
                                        ckv_f = cutlass.Float32(ckv_rmem[v])
                                        idx = ko * OUT_VEC_PER_KO + v
                                        out0[idx], out1[idx] = (
                                            cute.arch.fma_packed_f32x2(
                                                (p0, p1), (ckv_f, ckv_f),
                                                (out0[idx], out1[idx]))
                                        )

                        # 4-stage cross-warp reduction → partial_out (gmem)
                        my_h = tidx // cutlass.Int32(DIM_SPLIT)
                        my_d = tidx %  cutlass.Int32(DIM_SPLIT)
                        acc  = cutlass.Float32(0)
                        for stage in cutlass.range_constexpr(NUM_STAGES_RED):
                            warp_lo = cutlass.Int32(stage * WARPS_PER_STAGE)
                            warp_hi = cutlass.Int32((stage + 1) * WARPS_PER_STAGE)
                            if warp_idx >= warp_lo and warp_idx < warp_hi:
                                w_in_st = warp_idx - warp_lo
                                for ko in cutlass.range_constexpr(N_KO_OUT):
                                    for v in cutlass.range_constexpr(OUT_VEC_PER_KO):
                                        d = (cutlass.Int32(ko) * cutlass.Int32(K_TILE)
                                             + lane_idx * cutlass.Int32(OUT_VEC_PER_KO)
                                             + cutlass.Int32(v))
                                        idx = ko * OUT_VEC_PER_KO + v
                                        smem_partial_st[w_in_st, 0, d] = out0[idx]
                                        smem_partial_st[w_in_st, 1, d] = out1[idx]
                            cute.arch.sync_threads()
                            if tidx < cutlass.Int32(N_REAL * DIM_SPLIT):
                                for w in cutlass.range_constexpr(WARPS_PER_STAGE):
                                    acc = acc + smem_partial_st[w, my_h, my_d]
                            cute.arch.sync_threads()

                        if t_idx < T_const:
                            if tidx < cutlass.Int32(N_REAL * DIM_SPLIT):
                                partial_out[t_idx, head_base + my_h,
                                            split_idx_xor, my_d] = acc

                        if tidx == cutlass.Int32(0):
                            range_stop(probe, probe_row, slot_base)
            # ── END per-token loop ────────────────────────────────────────
        # ── END CHUNK LOOP ───────────────────────────────────────────────

        # Free tmem after all chunks done.
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier(barrier_id=tmem_barrier_id)
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        if tidx == cutlass.Int32(0):
            range_stop(probe, probe_row, cutlass.Int32(0))
            # Total entries: 1 (whole-CTA) + NUM_CHUNKS × CHUNK_SLOTS.
            range_finalize(probe, probe_row,
                           cutlass.Int32(1) + cutlass.Int32(NUM_CHUNKS * CHUNK_SLOTS))


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


PHASE_ORDER = ["total", "chunk_prologue", "mma_path", "fastgemv_path"]


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

    print("Compiling kv_split_tcgen05_exp_persistent_v3_xor...")
    kernel = KvSplitTcgen05ExpPersistentV3Xor(sm_scale=sm_scale, T=T, num_pages=P)
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


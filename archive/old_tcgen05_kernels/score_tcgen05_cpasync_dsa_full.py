"""score_tcgen05_cpasync_dsa_full.py — full DSA score+softmax+output (1 block).

Combines `score_tcgen05_cpasync_dsa_kpe.py` (score) with the `_ffma2_dsa`
output GEMV pattern.  Single block computes:

    score[m, h]   = (ckv[idx[m]] · q_rope[h]) + (kpe[idx[m]] · q_nope[h])  (* sm_scale)
    p[m, h]       = softmax_m(score[m, h])
    output[h, d]  = sum_{m valid} p[m, h] * ckv[idx[m], d]                 d ∈ [0, DIM_SPLIT=128)

For 2 consecutive heads (N_REAL=2) and a single dim split (128 of 512 ckv dims).

Phases
------
1. Score (identical to dsa_kpe):
     prologue → cp.async ckv+kpe(A) and q_rope+q_nope(B) → tcgen05 K=576 mma
     → tmem epilogue stores (h0,h1) per row to `smem_score` via sts.64.
2. Softmax (4 active warps, 128 threads, 1 thread per m row, 2 vals/thread):
     row_max[h]  = warp+block reduce of smem_score[m, h] over m
     smem_score[m, h] = exp(score - row_max[h])
     row_sum[h]  = warp+block reduce
     smem_score[m, h] /= row_sum[h]
3. Output GEMV (FFMA2, reuse `sA` from score for ckv access):
     16 warps × 8 rounds → m_local = round*16+warp
     32 lanes × vec=4    → 128 dims = DIM_SPLIT, packed as (k_outer, inner_lane)
     accumulate (h0,h1) via fma_packed_f32x2((p0,p1), (ckv,ckv), acc)
     store partials → cross-warp reduce → output[h, d].

We rely on cute reading sA via its swizzled layout: dim 0..63 → K_OUTER=0,
dim 64..127 → K_OUTER=1.  Lanes pack lane_idx → (k_outer=lane>>4, inner=lane&15)
so that 32 lanes × 4 vec covers exactly DIM_SPLIT=128.
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


# ── Problem dims ──────────────────────────────────────────────────────────────
M               = 128
N_REAL          = 2
N_MMA           = 8
K_CKV           = 512
K_KPE           = 64
K_FULL          = K_CKV + K_KPE          # 576
POOL            = 256
DIM_SPLIT       = 128                    # output dim per block (1 of 4 splits)

THREADS_PER_CTA = 512
NUM_WARPS       = THREADS_PER_CTA // 32  # 16
NUM_ROUNDS_MAX  = M // NUM_WARPS         # 8 (also output rounds)
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
MAX_ENTRIES  = 16
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY

TAGS = {
    "total":          2,
    "prologue":       4,
    "load_ab":        6,
    "mma":            8,
    "score_epi":     10,
    "softmax":       12,
    "output":        14,
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
class ScoreTcgen05CpAsyncDSAFull:
    def __init__(self, sm_scale: float = 0.1352337788608801,
                 num_blocks: int = 1, pool_stride: int = 0, pool_size: int = POOL):
        self.num_stages  = 1
        self.tmem_ld_rep = N_REAL
        self.sm_scale    = sm_scale
        self.num_blocks  = num_blocks
        # Per-CTA pool offset (in rows). Each CTA gathers a distinct slice
        # to mimic production's scattered HBM access. 0 = all CTAs hit the
        # same pool rows (best L2 case).
        self.pool_stride = pool_stride
        self.pool_size   = pool_size

    @cute.jit
    def __call__(
        self,
        ckv_flat:       cute.Tensor,   # (POOL, K_CKV)    bf16
        kpe_flat:       cute.Tensor,   # (POOL, K_KPE)    bf16
        q_rope:         cute.Tensor,   # (N_MMA, K_CKV)   bf16
        q_nope:         cute.Tensor,   # (N_MMA, K_KPE)   bf16
        sparse_indices: cute.Tensor,   # (M,)             int32
        output:         cute.Tensor,   # (N_REAL, DIM_SPLIT) f32
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
            ckv_flat, kpe_flat, q_rope, q_nope, sparse_indices, output, probe,
        ).launch(grid=[self.num_blocks, 1, 1], block=[THREADS_PER_CTA, 1, 1])

    @cute.kernel
    def kernel(
        self,
        tiled_mma, a_smem_layout, b_smem_layout, ab_dtype, acc_dtype,
        ckv_flat:       cute.Tensor,
        kpe_flat:       cute.Tensor,
        q_rope:         cute.Tensor,
        q_nope:         cute.Tensor,
        sparse_indices: cute.Tensor,
        output:         cute.Tensor,
        probe:          cute.Tensor,
    ):
        N_real:      cutlass.Constexpr = N_REAL
        tmem_ld_rep: cutlass.Constexpr = self.tmem_ld_rep
        sm_scale:    cutlass.Constexpr = self.sm_scale

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()
        bidx, _, _ = cute.arch.block_idx()
        probe_row  = bidx

        # Per-CTA pool offset to spread HBM gathers across the pool.
        # When pool_stride==0 this is a no-op (all CTAs hit the same rows).
        pool_stride: cutlass.Constexpr = self.pool_stride
        pool_size:   cutlass.Constexpr = self.pool_size
        pool_offset = (bidx * cutlass.Int32(pool_stride)) % cutlass.Int32(pool_size)

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

        # ══════════════════════════════════════════════════════════════════
        # 1. Prologue (sparse_indices cache + valid count + clamp)
        # ══════════════════════════════════════════════════════════════════
        if tidx == cutlass.Int32(0):
            range_start(probe, probe_row, cutlass.Int32(1), sm_val, TAGS["prologue"])

        partial_valid = cutlass.Int32(0)
        for m in cutlass.range_constexpr(M // THREADS_PER_CTA + 1):
            idx_lin = cutlass.Int32(m) * cutlass.Int32(THREADS_PER_CTA) + tidx
            if idx_lin < cutlass.Int32(M):
                idx = sparse_indices[idx_lin]
                smem_sparse[idx_lin] = idx
                if idx >= cutlass.Int32(0):
                    partial_valid = partial_valid + cutlass.Int32(1)

        warp_sum = warp_reduce_add(partial_valid, width=32)
        if lane_idx == cutlass.Int32(0):
            smem_red[warp_idx] = warp_sum
        cute.arch.sync_threads()

        if warp_idx == cutlass.Int32(0):
            val = cutlass.Int32(0)
            if lane_idx < cutlass.Int32(NUM_WARPS):
                val = smem_red[lane_idx]
            block_sum = warp_reduce_add(val, width=NUM_WARPS)
            if lane_idx == cutlass.Int32(0):
                smem_red[0] = block_sum
        cute.arch.sync_threads()

        num_valid  = smem_red[0]
        num_rounds = (num_valid + cutlass.Int32(NUM_WARPS - 1)) // cutlass.Int32(NUM_WARPS)
        round_limit = num_rounds * cutlass.Int32(NUM_WARPS)

        for m in cutlass.range_constexpr(M // THREADS_PER_CTA + 1):
            idx_lin = cutlass.Int32(m) * cutlass.Int32(THREADS_PER_CTA) + tidx
            if idx_lin < cutlass.Int32(M):
                if smem_sparse[idx_lin] < cutlass.Int32(0):
                    smem_sparse[idx_lin] = cutlass.Int32(0)
        cute.arch.sync_threads()

        if tidx == cutlass.Int32(0):
            range_stop(probe, probe_row, cutlass.Int32(1))

        # ══════════════════════════════════════════════════════════════════
        # 2. cp.async A (ckv+kpe) + B (q_rope+q_nope)
        # ══════════════════════════════════════════════════════════════════
        if tidx == cutlass.Int32(0):
            range_start(probe, probe_row, cutlass.Int32(2), sm_val, TAGS["load_ab"])

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

        # gmem source views
        ckv_full = cute.make_tensor(
            ckv_flat.iterator,
            cute.make_layout(
                (1, POOL, (K_TILE, K_OUTER_CKV)),
                stride=(0, K_CKV, (1, K_TILE)),
            ),
        )
        gB_full = cute.make_tensor(
            q_rope.iterator,
            cute.make_layout(
                (1, N_MMA, (K_TILE, K_OUTER_CKV)),
                stride=(0, K_CKV, (1, K_TILE)),
            ),
        )
        kpe_full = cute.make_tensor(
            kpe_flat.iterator,
            cute.make_layout((1, POOL, K_TILE), stride=(0, K_KPE, 1)),
        )
        q_nope_full = cute.make_tensor(
            q_nope.iterator,
            cute.make_layout((1, N_MMA, K_TILE), stride=(0, K_KPE, 1)),
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
                pool_idx_raw = smem_sparse[m_local]
                pool_idx = (pool_idx_raw + pool_offset) % cutlass.Int32(pool_size)

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
        if warp_idx < cutlass.Int32(N_REAL):
            gB_row    = gB_full[None, warp_idx, None]
            sB_qr_row = sB_qr [None, warp_idx, None]
            cute.copy(atom_cpa,
                      lane_copy.partition_S(gB_row),
                      lane_copy.partition_D(sB_qr_row))

            gB_qn_row = q_nope_full[None, warp_idx, None]
            sB_qn_row = sB_qn      [None, warp_idx, None]
            cute.copy(atom_cpa_kpe,
                      lane_copy_kpe.partition_S(gB_qn_row),
                      lane_copy_kpe.partition_D(sB_qn_row))

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        if tidx == cutlass.Int32(0):
            range_stop(probe, probe_row, cutlass.Int32(2))

        # ══════════════════════════════════════════════════════════════════
        # 3. tcgen05 MMA (K=576 → 36 k-blocks)
        # ══════════════════════════════════════════════════════════════════
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

        if tidx == cutlass.Int32(0):
            range_start(probe, probe_row, cutlass.Int32(3), sm_val, TAGS["mma"])

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

        cute.arch.mbarrier_wait(mma_mbar, cutlass.Int32(0))

        if tidx == cutlass.Int32(0):
            range_stop(probe, probe_row, cutlass.Int32(3))

        # ══════════════════════════════════════════════════════════════════
        # 4. Score epilogue: tmem → registers → smem_score (sts.64)
        # ══════════════════════════════════════════════════════════════════
        if tidx == cutlass.Int32(0):
            range_start(probe, probe_row, cutlass.Int32(4), sm_val, TAGS["score_epi"])

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
            range_stop(probe, probe_row, cutlass.Int32(4))

        # ── Free tmem (no longer needed) ───────────────────────────────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier(barrier_id=tmem_barrier_id)
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ══════════════════════════════════════════════════════════════════
        # 5. Softmax over m, per head (4 active warps × 32 lanes = 128 m rows)
        # ══════════════════════════════════════════════════════════════════
        if tidx == cutlass.Int32(0):
            range_start(probe, probe_row, cutlass.Int32(5), sm_val, TAGS["softmax"])

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
            range_stop(probe, probe_row, cutlass.Int32(5))

        # ══════════════════════════════════════════════════════════════════
        # 6. Output GEMV (FFMA2) — reuse sA for ckv access
        #    Build a tiled_copy that loads, per warp & per m row, exactly
        #    DIM_SPLIT=128 bf16 = 2 × K_TILE chunks via 32 lanes × vec=4.
        #    Lane mapping: lane → (k_outer = lane>>4, inner_lane = lane&15)
        #    so that dim_local = k_outer*64 + inner_lane*4 + v = lane*4 + v.
        # ══════════════════════════════════════════════════════════════════
        if tidx == cutlass.Int32(0):
            range_start(probe, probe_row, cutlass.Int32(6), sm_val, TAGS["output"])

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

        # Cross-warp reduction: 256 threads handle (h, dim) pairs
        # tidx 0..255 → (h = tidx // DIM_SPLIT, d = tidx % DIM_SPLIT)
        if tidx < cutlass.Int32(N_REAL * DIM_SPLIT):
            h = tidx // cutlass.Int32(DIM_SPLIT)
            d = tidx %  cutlass.Int32(DIM_SPLIT)
            acc = cutlass.Float32(0)
            for w in cutlass.range_constexpr(NUM_WARPS):
                acc = acc + smem_partial[w, h, d]
            output[h, d] = acc

        cute.arch.sync_threads()

        if tidx == cutlass.Int32(0):
            range_stop(probe, probe_row, cutlass.Int32(6))
            range_stop(probe, probe_row, cutlass.Int32(0))
            range_finalize(probe, probe_row, cutlass.Int32(7))


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


PHASE_ORDER = ["total", "prologue", "load_ab", "mma",
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


def run_dsa_full_cases(num_blocks_list=(1, 128), pool_size: int = POOL,
                       pool_stride: int = 0) -> dict:
    label = "score_tcgen05_cpasync_dsa_full"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {label}  M={M}  K_CKV={K_CKV}  K_KPE={K_KPE}  K_FULL={K_FULL}  "
          f"DIM_SPLIT={DIM_SPLIT}  POOL={pool_size}  N_real={N_REAL}  threads={THREADS_PER_CTA}")
    print(f"        pool_stride={pool_stride}  configs={list(num_blocks_list)} CTAs")

    sm_scale = 0.1352337788608801

    torch.manual_seed(42)
    ckv_flat = torch.randn((pool_size, K_CKV),  device="cuda", dtype=torch.bfloat16) * 0.1
    kpe_flat = torch.randn((pool_size, K_KPE),  device="cuda", dtype=torch.bfloat16) * 0.1
    q_rope   = torch.randn((N_MMA, K_CKV), device="cuda", dtype=torch.bfloat16) * 0.1
    q_nope   = torch.randn((N_MMA, K_KPE), device="cuda", dtype=torch.bfloat16) * 0.1

    si_full  = torch.arange(M, device="cuda", dtype=torch.int32)

    SEQ_SHORT = 64
    si_short = torch.full((M,), -1, device="cuda", dtype=torch.int32)
    si_short[:SEQ_SHORT] = torch.arange(SEQ_SHORT, dtype=torch.int32)

    output = torch.zeros((N_REAL, DIM_SPLIT), device="cuda", dtype=torch.float32)
    si_full_ = from_dlpack(si_full, assumed_align=16)
    ckv_ = from_dlpack(ckv_flat, assumed_align=128)
    kpe_ = from_dlpack(kpe_flat, assumed_align=128)
    qr_  = from_dlpack(q_rope,   assumed_align=128)
    qn_  = from_dlpack(q_nope,   assumed_align=128)
    out_ = from_dlpack(output,   assumed_align=16)

    # Per (num_blocks): allocate a (num_blocks, PROBE_COLS) probe and
    # compile a kernel instance.  All CTAs write to row=bidx.
    compiled_by_blocks = {}
    probe_by_blocks = {}
    for nblk in num_blocks_list:
        probe = torch.zeros((nblk, PROBE_COLS), dtype=torch.int64, device="cuda")
        probe_by_blocks[nblk] = probe
        probe_ = from_dlpack(probe, assumed_align=8)
        k = ScoreTcgen05CpAsyncDSAFull(
            sm_scale=sm_scale, num_blocks=nblk,
            pool_stride=pool_stride, pool_size=pool_size,
        )
        compiled_by_blocks[nblk] = (
            cute.compile(k, ckv_, kpe_, qr_, qn_, si_full_, out_, probe_),
            probe_,
        )

    results = {}

    for case_name, si_tensor, seq_len in [
        ("full",  si_full,  M),
        ("short", si_short, SEQ_SHORT),
    ]:
        si_ = from_dlpack(si_tensor, assumed_align=16)

        # Correctness vs reference (use 1-CTA build if available, else first)
        ref_nblk = 1 if 1 in compiled_by_blocks else num_blocks_list[0]
        ref_compiled, ref_probe_ = compiled_by_blocks[ref_nblk]
        ref_probe = probe_by_blocks[ref_nblk]
        for _ in range(3):
            ref_probe.zero_(); output.zero_()
            ref_compiled(ckv_, kpe_, qr_, qn_, si_, out_, ref_probe_)
        torch.cuda.synchronize()

        valid_mask = (si_tensor >= 0)
        valid_idx  = si_tensor[valid_mask].long()
        ckv_v = ckv_flat[valid_idx].float()
        kpe_v = kpe_flat[valid_idx].float()
        qr_h  = q_rope[:N_REAL].float()
        qn_h  = q_nope[:N_REAL].float()
        score = (ckv_v @ qr_h.T + kpe_v @ qn_h.T) * sm_scale
        score_max = score.max(dim=0, keepdim=True).values
        e = torch.exp(score - score_max)
        p = e / e.sum(dim=0, keepdim=True)
        ref = p.T @ ckv_v[:, :DIM_SPLIT]
        ok = torch.allclose(output, ref, atol=1e-2, rtol=1e-2)
        max_diff = (output - ref).abs().max().item()
        print(f"\n[{case_name:5s}]  seq_len={seq_len:3d}  "
              f"{'PASS' if ok else f'FAIL(max_diff={max_diff:.4f})'}")

        # Intra profile per grid config
        per_grid = {}
        per_grid_events = {}
        for nblk in num_blocks_list:
            compiled, probe_ = compiled_by_blocks[nblk]
            probe = probe_by_blocks[nblk]
            # Warmup so caches/SM-resident state are realistic
            for _ in range(3):
                probe.zero_(); output.zero_()
                compiled(ckv_, kpe_, qr_, qn_, si_, out_, probe_)
            torch.cuda.synchronize()
            # Single profiled launch
            probe.zero_(); output.zero_()
            compiled(ckv_, kpe_, qr_, qn_, si_, out_, probe_)
            torch.cuda.synchronize()
            agg, (events, base) = dump_probe(
                probe, nblk, label=f"{case_name} grid={nblk}",
            )
            per_grid[str(nblk)] = agg
            per_grid_events[str(nblk)] = {"events": events, "base": base}

        results[case_name] = {
            "seq_len": seq_len,
            "correct": ok,
            "max_diff": float(max_diff),
            "per_grid": per_grid,
            "per_grid_events": per_grid_events,
        }

    return results


def build_perfetto_trace(results: dict) -> str:
    """Combined trace: each (case, grid) gets its own pid-block."""
    all_events = []
    pid_block = 0
    for case_name, d in results.items():
        for nblk, ev in d.get("per_grid_events", {}).items():
            events = ev["events"]
            if not events:
                continue
            offset_pid = pid_block * 1000  # large gap between case/grid lanes
            all_events.append({
                "name": "process_name", "ph": "M",
                "pid": offset_pid, "tid": 0,
                "args": {"name": f"{case_name} grid={nblk}"},
            })
            for e in events:
                all_events.append({
                    "name": e["name"], "ph": "X",
                    "ts": e["ts"], "dur": e["dur"],
                    "pid": e["pid"] + offset_pid, "tid": e["tid"],
                })
            pid_block += 1
    return json.dumps({"traceEvents": all_events, "displayTimeUnit": "ns"})


def run_intra() -> str:
    res = run_dsa_full_cases()
    # Strip per_grid_events from JSON summary (it's huge); keep for trace only
    trimmed = {k: {kk: vv for kk, vv in v.items() if kk != "per_grid_events"}
               for k, v in res.items()}
    return json.dumps(trimmed, indent=2), res


if __name__ == "__main__":
    summary, _ = run_intra()
    print(summary)

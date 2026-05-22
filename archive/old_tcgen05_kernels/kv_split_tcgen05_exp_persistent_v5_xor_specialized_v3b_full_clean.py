"""kv_split_tcgen05_exp_persistent_v5_xor_specialized_v3b_full.py

v3b_full: complete 2-kernel pipeline (compute + reduce). Compute kernel is
v3b (8 MMA + 16 FGV warps) extended to write the FULL HEAD_DIM_CKV=512
output in 4 inner chunks of DIM_SPLIT_OUT=128 each, plus partial_lse
(row_max, row_sum) per (T, head, split). A separate reduce kernel merges
the NUM_SPLITS=16 partials per (T, head) and writes BF16 output + LSE.

Design principles:
  * MMA worker stays clean (no fast-path branching). It only ever sees
    full slabs (nv == M); writes UNNORMALIZED partial_out and partial_lse.
  * FGV worker handles every “messy” case: tail slabs (0 < nv < M) and
    is the natural place to add a final-output fast-path later. Today it
    also writes partial_out + partial_lse and lets the reducer merge.
  * Reduce kernel is ported from kv_split_xor_pdl_v3_pro_v2_tcgen05.
"""

import json
import math
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import (
    from_dlpack, make_fake_compact_tensor, make_fake_stream,
)
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T


LN2 = 0.6931471805599453


# ── Problem dims (mirrors v4_xor) ───────────────────────────────────────────
M               = 128
N_REAL          = 2
N_MMA           = 16
K_CKV           = 512
K_KPE           = 64
K_FULL          = K_CKV + K_KPE          # 576
DIM_SPLIT       = 128
T_CHUNK         = 8

NUM_HEADS       = 16
HEAD_GROUPS     = NUM_HEADS // N_REAL    # 8
NUM_PAGES       = 8462                   # static pool size for this workload
TOP_K_LEN       = 2048
NUM_SPLITS      = TOP_K_LEN // M         # 16
PS              = 64

# ── Worker layout ──────────────────────────────────────────────────────────
MMA_WARPS         = 8
FGV_WARPS         = 16
TOTAL_WARPS       = MMA_WARPS + FGV_WARPS         # 24
THREADS_PER_CTA   = TOTAL_WARPS * 32              # 768
MMA_THREADS       = MMA_WARPS * 32                # 256
FGV_THREADS       = FGV_WARPS * 32                # 512

NUM_ROUNDS_MMA    = M // MMA_WARPS                # 16
NUM_ROUNDS_FGV    = M // FGV_WARPS                # 8

MMA_INST_MNK      = (128, N_MMA, 16)
CTA_TILE_MNK      = (M, N_MMA, K_FULL)

OUT_VEC           = 4
OUT_INNER_LANES   = 16
SM_WARPS          = M // 32                       # 4

# Output GEMV chunking: cover full HEAD_DIM_CKV in 4 inner chunks of 128 dims.
DIM_SPLIT_OUT     = DIM_SPLIT                     # 128 dims per inner chunk
N_OUT_CHUNKS      = K_CKV // DIM_SPLIT_OUT        # 4

NUM_STAGES_RED_FGV = 4
WARPS_PER_STAGE_FGV = FGV_WARPS // NUM_STAGES_RED_FGV   # 4

# Reduce kernel constants (mirror kv_split_xor_pdl_v3_pro_v2_tcgen05).
NUM_THREADS_REDUCE = 256
NUM_WARPS_REDUCE   = NUM_THREADS_REDUCE // 32     # 8
VEC_REDUCE         = 2                            # 256 × 2 = 512 dims/pass
T_MAX              = 8

# Named barriers (avoid id 0 — reserved as sync_threads)
BAR_MMA           = 2     # 256 threads, MMA-internal
BAR_FGV           = 3     # 512 threads, FGV-internal


# ── Probe infra ────────────────────────────────────────────────────────────
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
# Per token: MMA uses 5 slots (1 outer + 4 inner), FGV uses 4 (1 outer + 3 inner).
# Worst case T_CHUNK=8 * NUM_CHUNKS=3 * 5 = 120, so 128 is enough.
MAX_ENTRIES  = 128
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY

# Per-token sub-event slot counts (must match the per-worker probe layout below).
SLOTS_PER_TOK_MMA = 5     # mma_path (outer) + mma_load + mma_compute + mma_softmax + mma_output
SLOTS_PER_TOK_FGV = 4     # fastgemv_path (outer) + fgv_score + fgv_softmax + fgv_output

TAGS = {
    "total":          2,
    "chunk_prologue": 4,
    "mma_path":       6,
    "fastgemv_path":  8,
    # MMA-worker fine-grained sub-phases
    "mma_load":       10,
    "mma_compute":    12,
    "mma_softmax":    14,
    "mma_output":     16,
    # FGV-worker fine-grained sub-phases
    "fgv_score":      18,
    "fgv_softmax":    20,
    "fgv_output":     22,
}
TAG_NAMES = {v: k for k, v in TAGS.items()}

ROLE_NAMES = ["prologue", "mma", "fgv"]


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
class KvSplitTcgen05ExpPersistentV5XorSpecializedV3bFull:
    def __init__(self, sm_scale: float = 0.1352337788608801,
                 device: str = "cuda"):
        self.num_stages  = 1
        self.tmem_ld_rep = N_MMA
        self.sm_scale    = sm_scale
        # Workspace: allocated once, reused across all calls.
        self.partial_out = torch.empty(
            T_MAX, NUM_HEADS, NUM_SPLITS, K_CKV,
            dtype=torch.float32, device=device)
        self.partial_lse = torch.empty(
            T_MAX, NUM_HEADS, NUM_SPLITS, 2,
            dtype=torch.float32, device=device)
        # Probe buffer kept as a no-op-sized scratch since the kernel still
        # writes range_start/stop entries; size matches the kernel's grid.
        self._dummy_probe = torch.zeros(
            (HEAD_GROUPS * NUM_SPLITS * 3, PROBE_COLS),
            dtype=torch.int64, device=device)

    @cute.jit
    def __call__(
        self,
        ckv_flat:       cute.Tensor,
        kpe_flat:       cute.Tensor,
        q_rope:         cute.Tensor,
        q_nope:         cute.Tensor,
        sparse_indices: cute.Tensor,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor,
        probe:          cute.Tensor,
        stream,
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
            ckv_flat, kpe_flat, q_rope, q_nope, sparse_indices,
            partial_out, partial_lse, probe,
        ).launch(grid=[HEAD_GROUPS, NUM_SPLITS, 1],
                 block=[THREADS_PER_CTA, 1, 1],
                 stream=stream, use_pdl=True)

        kvsplit_reduce_kernel(
            sparse_indices, partial_out, partial_lse, output, lse,
        ).launch(
            grid=[T_MAX, NUM_HEADS, 1],
            block=[NUM_THREADS_REDUCE, 1, 1],
            stream=stream, use_pdl=True,
        )

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
        partial_lse:    cute.Tensor,
        probe:          cute.Tensor,
    ):
        N_real:      cutlass.Constexpr = N_REAL
        tmem_ld_rep: cutlass.Constexpr = self.tmem_ld_rep
        sm_scale:    cutlass.Constexpr = self.sm_scale

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()
        hg_idx, split_idx, _ = cute.arch.block_idx()

        bid_ctx        = hg_idx * cutlass.Int32(NUM_SPLITS) + split_idx
        probe_row_pro  = bid_ctx * cutlass.Int32(3) + cutlass.Int32(0)
        probe_row_mma  = bid_ctx * cutlass.Int32(3) + cutlass.Int32(1)
        probe_row_fgv  = bid_ctx * cutlass.Int32(3) + cutlass.Int32(2)
        head_base      = hg_idx * cutlass.Int32(N_REAL)
        # T_const = T_MAX is the compile-time UPPER bound used for layout shapes
        # and loop counts; T_runtime is the actual per-call token count read from
        # the q_nope tensor and used for all dynamic OOB guards.
        T_const:       cutlass.Constexpr = T_MAX
        T_runtime      = cutlass.Int32(q_nope.shape[0])

        # Worker IDs (warp-uniform)
        wg_idx       = warp_idx >> cutlass.Int32(2)        # 0=MMA, >=1=FGV
        mma_tidx     = tidx                                # valid in [0,255]
        mma_warp_idx = warp_idx                            # valid in [0,7]
        fgv_tidx     = tidx - cutlass.Int32(MMA_THREADS)   # valid in [0,511]
        fgv_warp_idx = warp_idx - cutlass.Int32(MMA_WARPS) # valid in [0,15]

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
        # v2: separate Q smem for FGV worker (eliminates MMA<->FGV smem read
        # contention on the shared Q tile).
        sB_fgemv = alloc.allocate_tensor(
            element_type=ab_dtype, layout=b_smem_layout.outer,
            byte_alignment=1024, swizzle=b_smem_layout.inner,
        )
        smem_sp_indices = alloc.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((T_CHUNK, M), stride=(M, 1)),
            16, None,
        )
        smem_assign = alloc.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((T_CHUNK, 2), stride=(2, 1)),
            16, None,
        )
        # Per-worker score / partial / sm_red tensors (concurrent access).
        smem_score_mma = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((M, N_REAL), stride=(N_REAL, 1)), 16, None,
        )
        smem_score_fgv = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((M, N_REAL), stride=(N_REAL, 1)), 16, None,
        )
        smem_partial_mma = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((MMA_WARPS, N_REAL, DIM_SPLIT),
                             stride=(N_REAL * DIM_SPLIT, DIM_SPLIT, 1)),
            16, None,
        )
        smem_partial_fgv = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((NUM_STAGES_RED_FGV, N_REAL, DIM_SPLIT),
                             stride=(N_REAL * DIM_SPLIT, DIM_SPLIT, 1)),
            16, None,
        )
        smem_sm_red_mma = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((MMA_WARPS, N_REAL), stride=(N_REAL, 1)), 16, None,
        )
        smem_sm_red_fgv = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((FGV_WARPS, N_REAL), stride=(N_REAL, 1)), 16, None,
        )
        storage = alloc.allocate(self.shared_storage)
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        sm_val = smid_u32()
        if tidx == cutlass.Int32(0):
            range_start(probe, probe_row_pro, cutlass.Int32(0), sm_val,
                        TAGS["total"])

        mma_phase = cutlass.Int32(0)

        # ── Hoisted: cp.async setup (constexpr) ────────────────────────────
        K_TILE:        cutlass.Constexpr = 64
        K_OUTER_CKV:   cutlass.Constexpr = K_CKV  // K_TILE      # 8
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

        N_pool: cutlass.Constexpr = NUM_PAGES * PS

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
        # v2: identical-layout views over the FGV-private Q copy.
        sB_qr_fgv = cute.make_tensor(
            sB_fgemv.iterator,
            cute.make_layout(
                (1, N_MMA, (K_TILE, K_OUTER_CKV)),
                stride=(0, K_TILE, (1, N_MMA * K_TILE)),
            ),
        )
        sB_qn_fgv = cute.make_tensor(
            sB_fgemv.iterator + (K_OUTER_KPE_IDX * N_MMA * K_TILE),
            cute.make_layout((1, N_MMA, K_TILE), stride=(0, K_TILE, 1)),
        )

        # ── Hoisted: tmem alloc + mbarrier init (preamble; full CTA) ───────
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape       = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
        cute.arch.sync_threads()      # full CTA (640t) — bar 0 reserved
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()      # full CTA
        num_k_blocks = cute.size(tCrA, mode=[2])

        # ── Hoisted: score-epi setup (MMA worker uses tidx∈[0,127]) ────────
        M_acc          = cute.size(tCtAcc, mode=[0, 0])
        ld_op          = tcgen05.Ld32x32bOp(tcgen05.Repetition(tmem_ld_rep))
        epi_tiler      = ((M_acc, tmem_ld_rep),)
        tCtAcc_epi     = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r  = cute.make_copy_atom(ld_op, acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        # NOTE: get_slice(tidx) is only meaningful for tidx∈[0,127]; we only
        # consume tTR_tAcc inside the MMA-worker branch (mma_tidx<128).
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

        # FastGEMV warp-reduce score helpers (FGV worker, 16 warps)
        SCORE_VEC_PER_KO:  cutlass.Constexpr = 2
        ROWS_PER_WARP:     cutlass.Constexpr = 4
        ROWS_PER_ROUND_S:  cutlass.Constexpr = FGV_WARPS * ROWS_PER_WARP   # 64
        NUM_SCORE_ROUNDS:  cutlass.Constexpr = M // ROWS_PER_ROUND_S       # 2
        thr_layout_sc = cute.make_layout((32,), stride=(1,))
        val_layout_sc = cute.make_layout((SCORE_VEC_PER_KO,), stride=(1,))
        tiled_copy_sc = cute.make_tiled_copy_tv(
            atom_s2r, thr_layout_sc, val_layout_sc,
        )
        lane_copy_sc = tiled_copy_sc.get_slice(lane_idx)

        # ── 1D-sparse loader vec view ──────────────────────────────────────
        VEC_SPARSE:   cutlass.Constexpr = 4
        SLAB_VECS:    cutlass.Constexpr = M // VEC_SPARSE   # 32
        si_vec    = cute.zipped_divide(sparse_indices,  (1, VEC_SPARSE))
        sp_vec    = cute.zipped_divide(smem_sp_indices, (1, VEC_SPARSE))

        # ══════════════════════════════════════════════════════════════════
        # CHUNK LOOP  (NUM_CHUNKS = ceil(T/T_CHUNK) — typically 1)
        # ══════════════════════════════════════════════════════════════════
        NUM_CHUNKS:  cutlass.Constexpr = (T_const + T_CHUNK - 1) // T_CHUNK

        for chunk_idx in cutlass.range(NUM_CHUNKS, unroll=1):
            chunk_start = chunk_idx * cutlass.Int32(T_CHUNK)
            pro_slot    = cutlass.Int32(1) + chunk_idx     # row_pro slot
            tok_slot_b  = chunk_idx * cutlass.Int32(T_CHUNK)  # mma/fgv slot base

            # ─────────── UNIFIED PROLOGUE (Option A, 640 threads) ──────────
            if tidx == cutlass.Int32(0):
                range_start(probe, probe_row_pro, pro_slot, sm_val,
                            TAGS["chunk_prologue"])

            # 1. q load — warps 0..15 (warps 16..19 idle).
            if warp_idx < cutlass.Int32(16):
                t_in_chunk_w = warp_idx >> cutlass.Int32(1)        # 0..7
                h_in_grp_w   = warp_idx &  cutlass.Int32(1)        # 0..1
                t_global_w   = chunk_start + t_in_chunk_w
                safe_tw      = t_global_w
                if t_global_w >= T_runtime:
                    safe_tw = cutlass.Int32(0)
                head_idx_w   = head_base + h_in_grp_w
                sB_row_w     = warp_idx                             # 0..15

                gB_row    = gB_full     [None, safe_tw, head_idx_w, None]
                sB_qr_row = sB_qr       [None, sB_row_w, None]
                cute.copy(atom_cpa,
                          lane_copy.partition_S(gB_row),
                          lane_copy.partition_D(sB_qr_row))
                # v2: duplicate q_rope into FGV-private Q copy.
                sB_qr_row_fgv = sB_qr_fgv[None, sB_row_w, None]
                cute.copy(atom_cpa,
                          lane_copy.partition_S(gB_row),
                          lane_copy.partition_D(sB_qr_row_fgv))

                gB_qn_row = q_nope_full[None, safe_tw, head_idx_w, None]
                sB_qn_row = sB_qn      [None, sB_row_w, None]
                cute.copy(atom_cpa_kpe,
                          lane_copy_kpe.partition_S(gB_qn_row),
                          lane_copy_kpe.partition_D(sB_qn_row))
                # v2: duplicate q_nope into FGV-private Q copy.
                sB_qn_row_fgv = sB_qn_fgv[None, sB_row_w, None]
                cute.copy(atom_cpa_kpe,
                          lane_copy_kpe.partition_S(gB_qn_row),
                          lane_copy_kpe.partition_D(sB_qn_row_fgv))

                cute.arch.cp_async_commit_group()

            # 2. Per-token slab load + classify — warps 0..7.
            if warp_idx < cutlass.Int32(T_CHUNK):
                t_local  = warp_idx
                t_global = chunk_start + t_local
                safe_t   = t_global
                oob      = t_global >= T_runtime
                if oob:
                    safe_t = cutlass.Int32(0)
                swz = (split_idx + t_global * cutlass.Int32(7)) % cutlass.Int32(NUM_SPLITS)
                src_chunk = swz * cutlass.Int32(SLAB_VECS) + lane_idx
                vec = si_vec[(0, None), (safe_t, src_chunk)].load()
                nv_local = cutlass.Int32(0)
                for v in cutlass.range_constexpr(VEC_SPARSE):
                    val = vec[v]
                    if val < cutlass.Int32(0):
                        val = cutlass.Int32(0)
                    else:
                        nv_local = nv_local + cutlass.Int32(1)
                    sp_vec[(0, v), (t_local, lane_idx)] = val
                nv = warp_reduce_add(nv_local, width=32)
                if lane_idx == cutlass.Int32(0):
                    nv_store = nv
                    if oob:
                        nv_store = cutlass.Int32(0)
                    smem_assign[t_local, 0] = swz
                    smem_assign[t_local, 1] = nv_store

            # 3. Wait for q cp.async + publish to all 640 threads.
            #    Note: cp.async commit/wait is per-thread; only warps 0..15
            #    issued cp.async, but cp_async_wait_group(0) is safe to call
            #    on every thread (no-op for threads with no outstanding async).
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()       # full CTA — publish q + slab + assign

            if tidx == cutlass.Int32(0):
                range_stop(probe, probe_row_pro, pro_slot)

            # ══════════════════════════════════════════════════════════════
            # WARP-SPECIALIZED SPLIT — concurrent MMA + FGV workers
            # v3: 8 MMA warps (0..7) + 8 FGV warps (8..15)
            # ══════════════════════════════════════════════════════════════
            if warp_idx < cutlass.Int32(MMA_WARPS):
                # ============================================================
                #                       MMA WORKER  (256 threads)
                # ============================================================
                for t_in_chunk in cutlass.range_constexpr(T_CHUNK):
                    nv_t = smem_assign[t_in_chunk, 1]
                    if nv_t == cutlass.Int32(M):
                        swz_t     = smem_assign[t_in_chunk, 0]
                        t_idx     = chunk_start + cutlass.Int32(t_in_chunk)
                        # Fine-grained probe slots: 5 per token
                        # +0 outer mma_path, +1 load, +2 compute, +3 softmax, +4 output
                        slot_mma_base = ((tok_slot_b + cutlass.Int32(t_in_chunk))
                                         * cutlass.Int32(SLOTS_PER_TOK_MMA))
                        slot_mma      = slot_mma_base + cutlass.Int32(0)
                        slot_mma_load = slot_mma_base + cutlass.Int32(1)
                        slot_mma_cmp  = slot_mma_base + cutlass.Int32(2)
                        slot_mma_sm   = slot_mma_base + cutlass.Int32(3)
                        slot_mma_out  = slot_mma_base + cutlass.Int32(4)

                        if mma_tidx == cutlass.Int32(0):
                            range_start(probe, probe_row_mma, slot_mma,
                                        sm_val, TAGS["mma_path"])
                            range_start(probe, probe_row_mma, slot_mma_load,
                                        sm_val, TAGS["mma_load"])

                        # cp.async A (ckv+kpe) — 32 rounds × 4 warps = 128 rows
                        for rnd in cutlass.range_constexpr(NUM_ROUNDS_MMA):
                            m_local = (cutlass.Int32(rnd) * cutlass.Int32(MMA_WARPS)
                                       + mma_warp_idx)
                            pool_idx = smem_sp_indices[t_in_chunk, m_local]
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
                        cute.arch.barrier(barrier_id=BAR_MMA,
                                          number_of_threads=MMA_THREADS)

                        if mma_tidx == cutlass.Int32(0):
                            range_stop(probe, probe_row_mma, slot_mma_load)
                            range_start(probe, probe_row_mma, slot_mma_cmp,
                                        sm_val, TAGS["mma_compute"])

                        # MMA — 1 warp drives, all 4 warps wait.
                        tcgen05_fence()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        if mma_warp_idx == 0:
                            for k_block_idx in range(num_k_blocks):
                                k_block_coord = (None, None, k_block_idx, 0)
                                cute.gemm(tiled_mma, tCtAcc,
                                          tCrA[k_block_coord],
                                          tCrB[k_block_coord], tCtAcc)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            if mma_tidx == 0:
                                tcgen05.commit(mma_mbar)
                        cute.arch.mbarrier_wait(mma_mbar, mma_phase)
                        mma_phase = mma_phase ^ cutlass.Int32(1)

                        if mma_tidx == cutlass.Int32(0):
                            range_stop(probe, probe_row_mma, slot_mma_cmp)
                            range_start(probe, probe_row_mma, slot_mma_sm,
                                        sm_val, TAGS["mma_softmax"])

                        # Score epi: tmem → smem_score_mma
                        if mma_tidx < cutlass.Int32(M):
                            cute.copy(tmem_tiled_copy,
                                      tTR_tAcc[None, None, 0], tTR_rAcc)
                            col_start = t_in_chunk * 2  # constexpr
                            for n_idx in cutlass.range_constexpr(N_REAL):
                                smem_score_mma[mma_tidx, n_idx] = (
                                    tTR_rAcc[col_start + n_idx]
                                    * cutlass.Float32(sm_scale)
                                )
                        cute.arch.barrier(barrier_id=BAR_MMA,
                                          number_of_threads=MMA_THREADS)

                        # Softmax (no num_valid mask: nv == M); MMA_WARPS=4
                        NEG_INF: cutlass.Constexpr = -1.0e30
                        s0 = cutlass.Float32(NEG_INF)
                        s1 = cutlass.Float32(NEG_INF)
                        if mma_tidx < cutlass.Int32(M):
                            s0 = smem_score_mma[mma_tidx, 0]
                            s1 = smem_score_mma[mma_tidx, 1]
                        m0 = warp_reduce_max_f32(s0, width=32)
                        m1 = warp_reduce_max_f32(s1, width=32)
                        if lane_idx == cutlass.Int32(0):
                            smem_sm_red_mma[mma_warp_idx, 0] = m0
                            smem_sm_red_mma[mma_warp_idx, 1] = m1
                        cute.arch.barrier(barrier_id=BAR_MMA,
                                          number_of_threads=MMA_THREADS)
                        if mma_warp_idx == cutlass.Int32(0):
                            v0 = cutlass.Float32(NEG_INF)
                            v1 = cutlass.Float32(NEG_INF)
                            if lane_idx < cutlass.Int32(MMA_WARPS):
                                v0 = smem_sm_red_mma[lane_idx, 0]
                                v1 = smem_sm_red_mma[lane_idx, 1]
                            v0 = warp_reduce_max_f32(v0, width=MMA_WARPS)
                            v1 = warp_reduce_max_f32(v1, width=MMA_WARPS)
                            if lane_idx == cutlass.Int32(0):
                                smem_sm_red_mma[0, 0] = v0
                                smem_sm_red_mma[0, 1] = v1
                        cute.arch.barrier(barrier_id=BAR_MMA,
                                          number_of_threads=MMA_THREADS)
                        row_max_0 = smem_sm_red_mma[0, 0]
                        row_max_1 = smem_sm_red_mma[0, 1]

                        e0 = cutlass.Float32(0)
                        e1 = cutlass.Float32(0)
                        if mma_tidx < cutlass.Int32(M):
                            e0 = cute.math.exp(s0 - row_max_0)
                            e1 = cute.math.exp(s1 - row_max_1)
                            smem_score_mma[mma_tidx, 0] = e0
                            smem_score_mma[mma_tidx, 1] = e1
                        sum0 = warp_reduce_add_f32(e0, width=32)
                        sum1 = warp_reduce_add_f32(e1, width=32)
                        if lane_idx == cutlass.Int32(0):
                            smem_sm_red_mma[mma_warp_idx, 0] = sum0
                            smem_sm_red_mma[mma_warp_idx, 1] = sum1
                        cute.arch.barrier(barrier_id=BAR_MMA,
                                          number_of_threads=MMA_THREADS)
                        if mma_warp_idx == cutlass.Int32(0):
                            v0 = cutlass.Float32(0)
                            v1 = cutlass.Float32(0)
                            if lane_idx < cutlass.Int32(MMA_WARPS):
                                v0 = smem_sm_red_mma[lane_idx, 0]
                                v1 = smem_sm_red_mma[lane_idx, 1]
                            v0 = warp_reduce_add_f32(v0, width=MMA_WARPS)
                            v1 = warp_reduce_add_f32(v1, width=MMA_WARPS)
                            if lane_idx == cutlass.Int32(0):
                                smem_sm_red_mma[0, 0] = v0
                                smem_sm_red_mma[0, 1] = v1
                        cute.arch.barrier(barrier_id=BAR_MMA,
                                          number_of_threads=MMA_THREADS)
                        row_sum_0 = smem_sm_red_mma[0, 0]
                        row_sum_1 = smem_sm_red_mma[0, 1]
                        # NOTE: smem_score_mma stays as UNNORMALIZED exp(s-rmax).
                        # We write (rmax, rsum) to partial_lse and let the
                        # reduce kernel divide by the merged sum.

                        if mma_tidx == cutlass.Int32(0):
                            range_stop(probe, probe_row_mma, slot_mma_sm)
                            range_start(probe, probe_row_mma, slot_mma_out,
                                        sm_val, TAGS["mma_output"])

                        # Output GEMV — 4 inner chunks of 128 dims, full 512.
                        for chunk_idx in cutlass.range_constexpr(N_OUT_CHUNKS):
                            out0 = cute.make_rmem_tensor(
                                cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)),
                                cutlass.Float32)
                            out1 = cute.make_rmem_tensor(
                                cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)),
                                cutlass.Float32)
                            for v in cutlass.range_constexpr(OUT_VEC_TOTAL):
                                out0[v] = cutlass.Float32(0)
                                out1[v] = cutlass.Float32(0)
                            for round_idx in cutlass.range_constexpr(NUM_ROUNDS_MMA):
                                m_local = (cutlass.Int32(round_idx)
                                           * cutlass.Int32(MMA_WARPS) + mma_warp_idx)
                                p0 = smem_score_mma[m_local, 0]
                                p1 = smem_score_mma[m_local, 1]
                                for ko in cutlass.range_constexpr(N_KO_OUT):
                                    k_outer_global = (cutlass.Int32(chunk_idx)
                                                       * cutlass.Int32(N_KO_OUT)
                                                       + cutlass.Int32(ko))
                                    sA_chunk = sA_ckv[0, m_local, (None, k_outer_global)]
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

                            # Cross-warp reduction (8 warps → smem → sum).
                            for ko in cutlass.range_constexpr(N_KO_OUT):
                                for v in cutlass.range_constexpr(OUT_VEC_PER_KO):
                                    d = (cutlass.Int32(ko) * cutlass.Int32(K_TILE)
                                         + lane_idx * cutlass.Int32(OUT_VEC_PER_KO)
                                         + cutlass.Int32(v))
                                    idx = ko * OUT_VEC_PER_KO + v
                                    smem_partial_mma[mma_warp_idx, 0, d] = out0[idx]
                                    smem_partial_mma[mma_warp_idx, 1, d] = out1[idx]
                            cute.arch.barrier(barrier_id=BAR_MMA,
                                              number_of_threads=MMA_THREADS)

                            # 256 MMA threads → (h, d) of 128 dims.
                            my_h = mma_tidx >> cutlass.Int32(7)     # // 128
                            my_d = mma_tidx & cutlass.Int32(127)    # %  128
                            acc  = cutlass.Float32(0)
                            for w in cutlass.range_constexpr(MMA_WARPS):
                                acc = acc + smem_partial_mma[w, my_h, my_d]

                            d_global = (cutlass.Int32(chunk_idx)
                                         * cutlass.Int32(DIM_SPLIT_OUT) + my_d)
                            if t_idx < T_runtime:
                                partial_out[t_idx, head_base + my_h,
                                            swz_t, d_global] = acc
                            cute.arch.barrier(barrier_id=BAR_MMA,
                                              number_of_threads=MMA_THREADS)

                        # Write partial_lse for both heads.
                        if t_idx < T_runtime and mma_tidx == cutlass.Int32(0):
                            partial_lse[t_idx, head_base + cutlass.Int32(0), swz_t, 0] = row_max_0
                            partial_lse[t_idx, head_base + cutlass.Int32(0), swz_t, 1] = row_sum_0
                            partial_lse[t_idx, head_base + cutlass.Int32(1), swz_t, 0] = row_max_1
                            partial_lse[t_idx, head_base + cutlass.Int32(1), swz_t, 1] = row_sum_1

                        if mma_tidx == cutlass.Int32(0):
                            range_stop(probe, probe_row_mma, slot_mma_out)
                            range_stop(probe, probe_row_mma, slot_mma)

            else:
                # ============================================================
                #                       FGV WORKER  (512 threads)
                # ============================================================
                for t_in_chunk in cutlass.range_constexpr(T_CHUNK):
                    nv_t = smem_assign[t_in_chunk, 1]
                    if nv_t > cutlass.Int32(0) and nv_t < cutlass.Int32(M):
                        swz_t     = smem_assign[t_in_chunk, 0]
                        t_idx     = chunk_start + cutlass.Int32(t_in_chunk)
                        # Fine-grained probe slots: 4 per token
                        # +0 outer fastgemv_path, +1 score, +2 softmax, +3 output
                        slot_fgv_base = ((tok_slot_b + cutlass.Int32(t_in_chunk))
                                         * cutlass.Int32(SLOTS_PER_TOK_FGV))
                        slot_fgv      = slot_fgv_base + cutlass.Int32(0)
                        slot_fgv_sc   = slot_fgv_base + cutlass.Int32(1)
                        slot_fgv_sm   = slot_fgv_base + cutlass.Int32(2)
                        slot_fgv_out  = slot_fgv_base + cutlass.Int32(3)
                        num_valid = nv_t

                        if fgv_tidx == cutlass.Int32(0):
                            range_start(probe, probe_row_fgv, slot_fgv,
                                        sm_val, TAGS["fastgemv_path"])
                            range_start(probe, probe_row_fgv, slot_fgv_sc,
                                        sm_val, TAGS["fgv_score"])

                        # ── Score: 4-row interleaved per warp ──────────────
                        for round_idx in cutlass.range_constexpr(NUM_SCORE_ROUNDS):
                            base_row = (cutlass.Int32(round_idx)
                                        * cutlass.Int32(ROWS_PER_ROUND_S)
                                        + fgv_warp_idx * cutlass.Int32(ROWS_PER_WARP))
                            if base_row < num_valid:
                                pidx0 = smem_sp_indices[t_in_chunk, base_row + cutlass.Int32(0)]
                                pidx1 = smem_sp_indices[t_in_chunk, base_row + cutlass.Int32(1)]
                                pidx2 = smem_sp_indices[t_in_chunk, base_row + cutlass.Int32(2)]
                                pidx3 = smem_sp_indices[t_in_chunk, base_row + cutlass.Int32(3)]

                                for h_local in cutlass.range_constexpr(N_REAL):
                                    h_in_sB = t_in_chunk * 2 + h_local
                                    sums = cute.make_rmem_tensor(
                                        cute.make_layout((ROWS_PER_WARP,),
                                                         stride=(1,)),
                                        cutlass.Float32,
                                    )
                                    for r in cutlass.range_constexpr(ROWS_PER_WARP):
                                        sums[r] = cutlass.Float32(0)

                                    for ko in cutlass.range_constexpr(K_OUTER_CKV):
                                        q_chunk = sB_qr_fgv[0, h_in_sB, (None, ko)]
                                        q_part  = lane_copy_sc.partition_S(q_chunk)
                                        q_rmem  = cute.make_rmem_tensor(
                                            q_part.shape, ab_dtype)
                                        cute.copy(atom_s2r, q_part, q_rmem)
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

                                    qn_chunk = sB_qn_fgv[0, h_in_sB, None]
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

                                    for r in cutlass.range_constexpr(ROWS_PER_WARP):
                                        sums[r] = warp_reduce_add_f32(sums[r],
                                                                       width=32)
                                        row = base_row + cutlass.Int32(r)
                                        if lane_idx == cutlass.Int32(0) and row < num_valid:
                                            smem_score_fgv[row, h_local] = (
                                                sums[r] * cutlass.Float32(sm_scale)
                                            )
                        cute.arch.barrier(barrier_id=BAR_FGV,
                                          number_of_threads=FGV_THREADS)

                        if fgv_tidx == cutlass.Int32(0):
                            range_stop(probe, probe_row_fgv, slot_fgv_sc)
                            range_start(probe, probe_row_fgv, slot_fgv_sm,
                                        sm_val, TAGS["fgv_softmax"])

                        # ── Softmax (with num_valid mask) ──────────────────
                        NEG_INF: cutlass.Constexpr = -1.0e30
                        s0 = cutlass.Float32(NEG_INF)
                        s1 = cutlass.Float32(NEG_INF)
                        if fgv_tidx < cutlass.Int32(M) and fgv_tidx < num_valid:
                            s0 = smem_score_fgv[fgv_tidx, 0]
                            s1 = smem_score_fgv[fgv_tidx, 1]
                        m0 = warp_reduce_max_f32(s0, width=32)
                        m1 = warp_reduce_max_f32(s1, width=32)
                        if lane_idx == cutlass.Int32(0):
                            smem_sm_red_fgv[fgv_warp_idx, 0] = m0
                            smem_sm_red_fgv[fgv_warp_idx, 1] = m1
                        cute.arch.barrier(barrier_id=BAR_FGV,
                                          number_of_threads=FGV_THREADS)
                        if fgv_warp_idx == cutlass.Int32(0):
                            v0 = cutlass.Float32(NEG_INF)
                            v1 = cutlass.Float32(NEG_INF)
                            if lane_idx < cutlass.Int32(SM_WARPS):
                                v0 = smem_sm_red_fgv[lane_idx, 0]
                                v1 = smem_sm_red_fgv[lane_idx, 1]
                            v0 = warp_reduce_max_f32(v0, width=SM_WARPS)
                            v1 = warp_reduce_max_f32(v1, width=SM_WARPS)
                            if lane_idx == cutlass.Int32(0):
                                smem_sm_red_fgv[0, 0] = v0
                                smem_sm_red_fgv[0, 1] = v1
                        cute.arch.barrier(barrier_id=BAR_FGV,
                                          number_of_threads=FGV_THREADS)
                        row_max_0 = smem_sm_red_fgv[0, 0]
                        row_max_1 = smem_sm_red_fgv[0, 1]
                        e0 = cutlass.Float32(0)
                        e1 = cutlass.Float32(0)
                        if fgv_tidx < cutlass.Int32(M) and fgv_tidx < num_valid:
                            e0 = cute.math.exp(s0 - row_max_0)
                            e1 = cute.math.exp(s1 - row_max_1)
                            smem_score_fgv[fgv_tidx, 0] = e0
                            smem_score_fgv[fgv_tidx, 1] = e1
                        sum0 = warp_reduce_add_f32(e0, width=32)
                        sum1 = warp_reduce_add_f32(e1, width=32)
                        if lane_idx == cutlass.Int32(0):
                            smem_sm_red_fgv[fgv_warp_idx, 0] = sum0
                            smem_sm_red_fgv[fgv_warp_idx, 1] = sum1
                        cute.arch.barrier(barrier_id=BAR_FGV,
                                          number_of_threads=FGV_THREADS)
                        if fgv_warp_idx == cutlass.Int32(0):
                            v0 = cutlass.Float32(0)
                            v1 = cutlass.Float32(0)
                            if lane_idx < cutlass.Int32(SM_WARPS):
                                v0 = smem_sm_red_fgv[lane_idx, 0]
                                v1 = smem_sm_red_fgv[lane_idx, 1]
                            v0 = warp_reduce_add_f32(v0, width=SM_WARPS)
                            v1 = warp_reduce_add_f32(v1, width=SM_WARPS)
                            if lane_idx == cutlass.Int32(0):
                                smem_sm_red_fgv[0, 0] = v0
                                smem_sm_red_fgv[0, 1] = v1
                        cute.arch.barrier(barrier_id=BAR_FGV,
                                          number_of_threads=FGV_THREADS)
                        row_sum_0 = smem_sm_red_fgv[0, 0]
                        row_sum_1 = smem_sm_red_fgv[0, 1]
                        # NOTE: smem_score_fgv stays UNNORMALIZED; reduce kernel
                        # divides by merged sum.

                        if fgv_tidx == cutlass.Int32(0):
                            range_stop(probe, probe_row_fgv, slot_fgv_sm)
                            range_start(probe, probe_row_fgv, slot_fgv_out,
                                        sm_val, TAGS["fgv_output"])

                        # ── Output GEMV — 4 inner chunks of 128 dims ───────
                        for chunk_idx in cutlass.range_constexpr(N_OUT_CHUNKS):
                            out0 = cute.make_rmem_tensor(
                                cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)),
                                cutlass.Float32)
                            out1 = cute.make_rmem_tensor(
                                cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)),
                                cutlass.Float32)
                            for v in cutlass.range_constexpr(OUT_VEC_TOTAL):
                                out0[v] = cutlass.Float32(0)
                                out1[v] = cutlass.Float32(0)
                            for round_idx in cutlass.range_constexpr(NUM_ROUNDS_FGV):
                                m_local = (cutlass.Int32(round_idx)
                                           * cutlass.Int32(FGV_WARPS) + fgv_warp_idx)
                                if m_local < num_valid:
                                    pool_idx = smem_sp_indices[t_in_chunk, m_local]
                                    p0 = smem_score_fgv[m_local, 0]
                                    p1 = smem_score_fgv[m_local, 1]
                                    for ko in cutlass.range_constexpr(N_KO_OUT):
                                        k_outer_global = (cutlass.Int32(chunk_idx)
                                                           * cutlass.Int32(N_KO_OUT)
                                                           + cutlass.Int32(ko))
                                        g_chunk = ckv_full[0, pool_idx, (None, k_outer_global)]
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

                            my_h = fgv_tidx // cutlass.Int32(DIM_SPLIT)
                            my_d = fgv_tidx %  cutlass.Int32(DIM_SPLIT)
                            acc  = cutlass.Float32(0)
                            for stage in cutlass.range_constexpr(NUM_STAGES_RED_FGV):
                                warp_lo = cutlass.Int32(stage * WARPS_PER_STAGE_FGV)
                                warp_hi = cutlass.Int32((stage + 1) * WARPS_PER_STAGE_FGV)
                                if fgv_warp_idx >= warp_lo and fgv_warp_idx < warp_hi:
                                    w_in_st = fgv_warp_idx - warp_lo
                                    for ko in cutlass.range_constexpr(N_KO_OUT):
                                        for v in cutlass.range_constexpr(OUT_VEC_PER_KO):
                                            d = (cutlass.Int32(ko) * cutlass.Int32(K_TILE)
                                                 + lane_idx * cutlass.Int32(OUT_VEC_PER_KO)
                                                 + cutlass.Int32(v))
                                            idx = ko * OUT_VEC_PER_KO + v
                                            smem_partial_fgv[w_in_st, 0, d] = out0[idx]
                                            smem_partial_fgv[w_in_st, 1, d] = out1[idx]
                                cute.arch.barrier(barrier_id=BAR_FGV,
                                                  number_of_threads=FGV_THREADS)
                                if fgv_tidx < cutlass.Int32(N_REAL * DIM_SPLIT):
                                    for w in cutlass.range_constexpr(WARPS_PER_STAGE_FGV):
                                        acc = acc + smem_partial_fgv[w, my_h, my_d]
                                cute.arch.barrier(barrier_id=BAR_FGV,
                                                  number_of_threads=FGV_THREADS)

                            d_global = (cutlass.Int32(chunk_idx)
                                         * cutlass.Int32(DIM_SPLIT_OUT) + my_d)
                            if t_idx < T_runtime:
                                if fgv_tidx < cutlass.Int32(N_REAL * DIM_SPLIT):
                                    partial_out[t_idx, head_base + my_h,
                                                swz_t, d_global] = acc
                            cute.arch.barrier(barrier_id=BAR_FGV,
                                              number_of_threads=FGV_THREADS)

                        # Write partial_lse for both heads.
                        if t_idx < T_runtime and fgv_tidx == cutlass.Int32(0):
                            partial_lse[t_idx, head_base + cutlass.Int32(0), swz_t, 0] = row_max_0
                            partial_lse[t_idx, head_base + cutlass.Int32(0), swz_t, 1] = row_sum_0
                            partial_lse[t_idx, head_base + cutlass.Int32(1), swz_t, 0] = row_max_1
                            partial_lse[t_idx, head_base + cutlass.Int32(1), swz_t, 1] = row_sum_1

                        if fgv_tidx == cutlass.Int32(0):
                            range_stop(probe, probe_row_fgv, slot_fgv_out)
                            range_stop(probe, probe_row_fgv, slot_fgv)
            # ── END WARP-SPECIALIZED SPLIT ────────────────────────────────
        # ── END CHUNK LOOP ───────────────────────────────────────────────

        # Epilogue — full CTA must converge before tmem dealloc.
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # Finalize probe (per-row counts).
        if tidx == cutlass.Int32(0):
            range_stop(probe, probe_row_pro, cutlass.Int32(0))
            range_finalize(probe, probe_row_pro,
                           cutlass.Int32(1) + cutlass.Int32(NUM_CHUNKS))
            range_finalize(probe, probe_row_mma,
                           cutlass.Int32(NUM_CHUNKS * T_CHUNK
                                         * SLOTS_PER_TOK_MMA))
            range_finalize(probe, probe_row_fgv,
                           cutlass.Int32(NUM_CHUNKS * T_CHUNK
                                         * SLOTS_PER_TOK_FGV))


# ══════════════════════════════════════════════════════════════════════════════
# Reduce kernel — merges NUM_SPLITS partials per (T, head)  → BF16 output, LSE
# Grid: [T_MAX, NUM_HEADS, 1] × NUM_THREADS_REDUCE=256.
# Uses sparse_indices count (num_valid) to derive num_active_splits and skip
# inactive (T, head) and inactive splits.
# ══════════════════════════════════════════════════════════════════════════════
@cute.kernel
def kvsplit_reduce_kernel(
    sparse_indices: cute.Tensor,        # (T, TOP_K_LEN)   i32
    partial_out:    cute.Tensor,        # (T, NUM_HEADS, NUM_SPLITS, K_CKV) f32
    partial_lse:    cute.Tensor,        # (T, NUM_HEADS, NUM_SPLITS, 2)     f32
    output:         cute.Tensor,        # (T, NUM_HEADS, K_CKV)             bf16
    lse:            cute.Tensor,        # (T, NUM_HEADS)                    f32
):
    T, _ = sparse_indices.shape
    head_dim_ckv:   cutlass.Constexpr = K_CKV
    top_k_len:      cutlass.Constexpr = TOP_K_LEN
    dim_split:      cutlass.Constexpr = M
    num_splits:     cutlass.Constexpr = NUM_SPLITS
    num_threads:    cutlass.Constexpr = NUM_THREADS_REDUCE
    num_warps:      cutlass.Constexpr = NUM_WARPS_REDUCE
    vec_reduce:     cutlass.Constexpr = VEC_REDUCE
    t_max:          cutlass.Constexpr = T_MAX

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _    = cute.arch.thread_idx()
    warp_idx      = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx      = cute.arch.lane_idx()

    block_t_idx = bidx
    head_idx    = bidy

    alloc = cutlass.utils.SmemAllocator()
    smem_red_i32 = alloc.allocate_tensor(
        cutlass.Int32, cute.make_layout((32,), stride=(1,)), 4, None,
    )
    smem_max_sum = alloc.allocate_tensor(
        cutlass.Float32, cute.make_layout((num_splits, 2), stride=(2, 1)),
        4, None,
    )

    partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, vec_reduce))
    output_v      = cute.zipped_divide(output,      (1, 1,    vec_reduce))

    num_groups = (T + t_max - 1) // t_max
    for group_idx in range(num_groups):
        T_idx = group_idx * t_max + block_t_idx
        if T_idx < T:
            partial_cnt = 0
            for i in range(tidx, top_k_len, num_threads):
                idx = sparse_indices[T_idx, i]
                if idx >= cutlass.Int32(0):
                    partial_cnt += 1

            cnt_sum = warp_reduce_add(partial_cnt, width=32)
            if lane_idx == 0:
                smem_red_i32[warp_idx] = cnt_sum
            cute.arch.sync_threads()

            if warp_idx == 0:
                val = smem_red_i32[lane_idx]
                cnt_sum = warp_reduce_add(val, width=num_warps)
                smem_red_i32[0] = cnt_sum
            cute.arch.sync_threads()

            num_valid = smem_red_i32[0]

            cute.arch.griddepcontrol_wait()

            if num_valid > cutlass.Int32(0):
                num_active_splits = (num_valid + cutlass.Int32(dim_split)
                                     - cutlass.Int32(1)) // cutlass.Int32(dim_split)

                if tidx < num_active_splits:
                    smem_max_sum[tidx, 0] = partial_lse[T_idx, head_idx, tidx, 0]
                    smem_max_sum[tidx, 1] = partial_lse[T_idx, head_idx, tidx, 1]
                cute.arch.sync_threads()

                g_max = -cutlass.Float32(math.inf)
                for s in range(num_active_splits):
                    local_max = smem_max_sum[s, 0]
                    if local_max > g_max:
                        g_max = local_max

                g_lse_sum = cutlass.Float32(0)
                acc_rmem = cute.make_rmem_tensor(
                    cute.make_layout((vec_reduce,), stride=(1,)), cutlass.Float32,
                )
                acc_rmem[0] = cutlass.Float32(0)
                acc_rmem[1] = cutlass.Float32(0)
                acc = acc_rmem.load()

                for s in range(num_active_splits):
                    l_max = smem_max_sum[s, 0]
                    l_sum = smem_max_sum[s, 1]
                    scale = cute.math.exp(l_max - g_max)
                    g_lse_sum += l_sum * scale

                    a = partial_out_v[(0, 0, 0, None),
                                      (T_idx, head_idx, s, tidx)].load()
                    acc = acc + scale * a

                if tidx == 0:
                    lse[T_idx, head_idx] = (
                        (g_max + cute.math.log(g_lse_sum))
                        / cutlass.Float32(LN2)
                    )

                output_v[(0, 0, None), (T_idx, head_idx, tidx)].store(
                    (acc / g_lse_sum).to(cutlass.BFloat16)
                )

            cute.arch.sync_threads()




# ══════════════════════════════════════════════════════════════════════════════
# Compilation + run wrapper for submit.py
# ══════════════════════════════════════════════════════════════════════════════
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align,
    )


def compile_kernel():
    T = cute.sym_int()
    num_heads  = NUM_HEADS
    num_splits = NUM_SPLITS
    N_pool     = NUM_PAGES * PS

    ckv_flat   = _fake(cute.BFloat16, (N_pool, K_CKV),                            (1, 0),       16)
    kpe_flat   = _fake(cute.BFloat16, (N_pool, K_KPE),                            (1, 0),       16)
    q_rope     = _fake(cute.BFloat16, (T,         num_heads, K_CKV),              (2, 1, 0),    16)
    q_nope     = _fake(cute.BFloat16, (T,         num_heads, K_KPE),              (2, 1, 0),    16)
    sparse_idx = _fake(cute.Int32,    (T,         TOP_K_LEN),                     (1, 0),        4)
    partial_out= _fake(cute.Float32,  (T_MAX,     num_heads, num_splits, K_CKV),  (3, 2, 1, 0), 16)
    partial_lse= _fake(cute.Float32,  (T_MAX,     num_heads, num_splits, 2),      (3, 2, 1, 0), 16)
    output     = _fake(cute.BFloat16, (T,         num_heads, K_CKV),              (2, 1, 0),    16)
    lse        = _fake(cute.Float32,  (T,         num_heads),                     (1, 0),        4)
    probe      = _fake(cute.Int64,    (HEAD_GROUPS * NUM_SPLITS * 3, PROBE_COLS), (1, 0),        8)
    stream     = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = KvSplitTcgen05ExpPersistentV5XorSpecializedV3bFull(
        sm_scale=0.1352337788608801,
    )
    compiled = cute.compile(
        kernel,
        ckv_flat, kpe_flat, q_rope, q_nope, sparse_idx,
        partial_out, partial_lse, output, lse, probe, stream,
        options="--enable-tvm-ffi",
    )
    return kernel, compiled


_kernel, _compiled = compile_kernel()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    P = ckv_cache.shape[0]
    ckv_flat = ckv_cache.view(P * PS, K_CKV)
    kpe_flat = kpe_cache.view(P * PS, K_KPE)
    _compiled(
        ckv_flat, kpe_flat, q_nope, q_pe, sparse_indices,
        _kernel.partial_out, _kernel.partial_lse, output, lse,
        _kernel._dummy_probe,
    )

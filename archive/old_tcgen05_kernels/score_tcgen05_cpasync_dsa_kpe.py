"""score_tcgen05_cpasync_dsa_kpe.py — DSA + KPE/Q-NOPE concat.

Extension of `score_tcgen05_cpasync_dsa.py` that handles the full DSA score:
    score[m, n] = sum_k(ckv[idx[m], k] * q_rope[n, k])     # K = 512
                + sum_k(kpe[idx[m], k] * q_nope[n, k])     # K_KPE = 64

We concat into a single MMA with K=576 by appending KPE/Q-NOPE in SMEM at
`K_OUTER=8` (the 9th K_TILE chunk):

  sA layout: (M, (K_TILE=64, K_OUTER=9))  — ckv[..,:512] at K_OUTER 0..7,
                                            kpe[..,:64]  at K_OUTER 8
  sB layout: (N_MMA, (K_TILE=64, K_OUTER=9)) — q_rope[..,:512] at 0..7,
                                               q_nope[..,:64]  at 8

A-load (per warp, per round):
  * ckv  : same per-warp tv (val=8 bf16, 32 lanes × 4 K_OUTER halves) — 256 bf16
           issued twice per row (= one full K=512 row).
  * kpe  : narrower tv (val=2 bf16, 32 lanes × 1 K_TILE) — 64 bf16, one chunk.
           Issued in the SAME for-rnd loop as ckv, writing to K_OUTER=8.

B-load (warps 0..N_REAL-1 only):
  * q_rope : same as v3_ab.
  * q_nope : narrow tv, single K_TILE chunk → K_OUTER=8.

MMA: tcgen05 sees K=576 → 36 k-blocks (was 32). Same code path otherwise.
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

THREADS_PER_CTA = 512
NUM_WARPS       = THREADS_PER_CTA // 32  # 16
NUM_ROUNDS_MAX  = M // NUM_WARPS         # 8
MMA_INST_MNK    = (128, N_MMA, 16)
CTA_TILE_MNK    = (M, N_MMA, K_FULL)


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

TAGS = {"total": 2, "load_ab": 4, "mma": 6, "epilogue": 8,
        "load_a": 10, "load_b": 12,
        "issue_a": 14, "commit_a": 16, "issue_b": 18,
        "prologue": 20}
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


# ══════════════════════════════════════════════════════════════════════════════
class ScoreTcgen05CpAsyncDSAKpe:
    def __init__(self):
        self.num_stages  = 1
        self.tmem_ld_rep = N_MMA

    @cute.jit
    def __call__(
        self,
        ckv_flat:       cute.Tensor,   # (POOL, K_CKV)   bf16
        kpe_flat:       cute.Tensor,   # (POOL, K_KPE)   bf16
        q_rope:         cute.Tensor,   # (N_MMA, K_CKV)  bf16
        q_nope:         cute.Tensor,   # (N_MMA, K_KPE)  bf16
        sparse_indices: cute.Tensor,   # (M,)            int32
        C:              cute.Tensor,   # (M, N_REAL)     float32
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
            ckv_flat, kpe_flat, q_rope, q_nope, sparse_indices, C, probe,
        ).launch(grid=[1, 1, 1], block=[THREADS_PER_CTA, 1, 1])

    @cute.kernel
    def kernel(
        self,
        tiled_mma, a_smem_layout, b_smem_layout, ab_dtype, acc_dtype,
        ckv_flat:       cute.Tensor,
        kpe_flat:       cute.Tensor,
        q_rope:         cute.Tensor,
        q_nope:         cute.Tensor,
        sparse_indices: cute.Tensor,
        C:              cute.Tensor,
        probe:          cute.Tensor,
    ):
        N_real:      cutlass.Constexpr = N_REAL
        tmem_ld_rep: cutlass.Constexpr = self.tmem_ld_rep

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()

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
        storage = alloc.allocate(self.shared_storage)
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        sm_val = smid_u32()
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])

        # ── Prologue: cache sparse_indices + count valid ──────────────────────
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(9), sm_val, TAGS["prologue"])

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
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(9))

        # ── load_ab phase ────────────────────────────────────────────────────
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["load_ab"])

        # ── tile / atom params ───────────────────────────────────────────────
        K_TILE:        cutlass.Constexpr = 64
        K_OUTER_CKV:   cutlass.Constexpr = K_CKV  // K_TILE      # 8
        K_OUTER_FULL:  cutlass.Constexpr = K_FULL // K_TILE      # 9
        K_OUTER_KPE_IDX: cutlass.Constexpr = K_OUTER_CKV         # 8 (last slot)
        VEC_BF16:      cutlass.Constexpr = 8                     # 128b atom (ckv)
        K_OUTER_HALF:  cutlass.Constexpr = K_OUTER_CKV // 2      # 4
        VEC_BF16_KPE:  cutlass.Constexpr = 2                     # 32b atom (kpe)

        # ── ckv/q_rope tv (val=8 bf16, 256 bf16 tile) ────────────────────────
        atom_cpa = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            ab_dtype, num_bits_per_copy=128,
        )
        thr_layout_warp = cute.make_layout(
            (1, (8, K_OUTER_HALF)),
            stride=(32, (1, 8)),
        )
        val_layout_warp = cute.make_layout(
            (1, (VEC_BF16, 1)),
            stride=(0, (1, 0)),
        )
        tiled_copy_warp = cute.make_tiled_copy_tv(
            atom_cpa, thr_layout_warp, val_layout_warp,
        )
        lane_copy = tiled_copy_warp.get_slice(lane_idx)

        # ── kpe/q_nope tv (val=2 bf16, 64 bf16 tile = single K_TILE) ─────────
        atom_cpa_kpe = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            ab_dtype, num_bits_per_copy=32,
        )
        # 32 lanes mapped over (K_inner_atoms=32) within K_TILE=64:
        thr_layout_kpe = cute.make_layout(
            (1, 32),
            stride=(32, 1),
        )
        val_layout_kpe = cute.make_layout(
            (1, VEC_BF16_KPE),
            stride=(0, 1),
        )
        tiled_copy_kpe = cute.make_tiled_copy_tv(
            atom_cpa_kpe, thr_layout_kpe, val_layout_kpe,
        )
        lane_copy_kpe = tiled_copy_kpe.get_slice(lane_idx)

        # ── Hierarchical 3-mode views ────────────────────────────────────────
        # gmem source views (unchanged from v3_ab style)
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
            cute.make_layout(
                (1, POOL, K_TILE),
                stride=(0, K_KPE, 1),
            ),
        )
        q_nope_full = cute.make_tensor(
            q_nope.iterator,
            cute.make_layout(
                (1, N_MMA, K_TILE),
                stride=(0, K_KPE, 1),
            ),
        )

        # SMEM destination views — preserve swizzled iterator (no dynamic
        # pointer arithmetic, only constexpr offsets via the iterator's
        # `+ const` operator which keeps swizzle alignment annotations).
        # sA layout in helper: (M, (K_TILE, K_OUTER_FULL=9)) stride
        # (K_TILE, (1, M*K_TILE)).  K_OUTER chunk `o` lives at offset
        # o*M*K_TILE bf16 elements.
        sA_ckv = cute.make_tensor(
            sA.iterator,
            cute.make_layout(
                (1, M, (K_TILE, K_OUTER_CKV)),
                stride=(0, K_TILE, (1, M * K_TILE)),
            ),
        )
        sA_kpe = cute.make_tensor(
            sA.iterator + (K_OUTER_KPE_IDX * M * K_TILE),
            cute.make_layout(
                (1, M, K_TILE),
                stride=(0, K_TILE, 1),
            ),
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
            cute.make_layout(
                (1, N_MMA, K_TILE),
                stride=(0, K_TILE, 1),
            ),
        )

        # ---- ISSUE A (ckv + kpe in same per-row loop) ----
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(4), sm_val, TAGS["issue_a"])

        for rnd in cutlass.range_constexpr(NUM_ROUNDS_MAX):
            m_local = cutlass.Int32(rnd) * cutlass.Int32(NUM_WARPS) + warp_idx
            if m_local < round_limit:
                pool_idx = smem_sparse[m_local]

                # ckv -> sA[m_local, :, 0..7]
                gA_row     = ckv_full[None, pool_idx, None]
                sA_ckv_row = sA_ckv  [None, m_local,  None]
                lAgA = lane_copy.partition_S(gA_row)
                lAsA = lane_copy.partition_D(sA_ckv_row)
                cute.copy(atom_cpa, lAgA, lAsA)

                # kpe -> sA[m_local, :, 8]
                gA_kpe_row = kpe_full[None, pool_idx, None]
                sA_kpe_row = sA_kpe  [None, m_local,  None]
                lAgA_kpe = lane_copy_kpe.partition_S(gA_kpe_row)
                lAsA_kpe = lane_copy_kpe.partition_D(sA_kpe_row)
                cute.copy(atom_cpa_kpe, lAgA_kpe, lAsA_kpe)

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(4))

        # ---- ISSUE B (q_rope + q_nope, warps 0..N_REAL-1) ----
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(6), sm_val, TAGS["issue_b"])

        if warp_idx < cutlass.Int32(N_REAL):
            # q_rope -> sB_qr[warp_idx, :, 0..7]
            gB_row    = gB_full[None, warp_idx, None]
            sB_qr_row = sB_qr [None, warp_idx, None]
            lBgB = lane_copy.partition_S(gB_row)
            lBsB = lane_copy.partition_D(sB_qr_row)
            cute.copy(atom_cpa, lBgB, lBsB)

            # q_nope -> sB_qn[warp_idx, :]
            gB_qn_row = q_nope_full[None, warp_idx, None]
            sB_qn_row = sB_qn      [None, warp_idx, None]
            lBgB_qn = lane_copy_kpe.partition_S(gB_qn_row)
            lBsB_qn = lane_copy_kpe.partition_D(sB_qn_row)
            cute.copy(atom_cpa_kpe, lBgB_qn, lBsB_qn)

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(6))

        # ---- COMMIT A+B ----
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(5), sm_val, TAGS["commit_a"])
        cute.arch.cp_async_commit_group()
        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(5))

        # ---- WAIT ----
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(7), sm_val, TAGS["load_a"])
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(7))

        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(8), sm_val, TAGS["load_b"])
            range_stop(probe,  cutlass.Int32(0), cutlass.Int32(8))

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(1))

        # ── MMA setup (K=576 → 36 k-blocks) ───────────────────────────────────
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
            range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["mma"])

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
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))

        # ---- EPILOGUE ----
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(3), sm_val, TAGS["epilogue"])

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
            for n_idx in cutlass.range_constexpr(N_REAL):
                C[tidx, n_idx] = tTR_rAcc[n_idx]

        cute.arch.sync_threads()

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(3))

        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier(barrier_id=tmem_barrier_id)
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(0))
            range_finalize(probe, cutlass.Int32(0), cutlass.Int32(10))


# ══════════════════════════════════════════════════════════════════════════════
def run_dsa_cases() -> dict:
    label = "score_tcgen05_cpasync_dsa_kpe"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {label}  M={M}  K_CKV={K_CKV}  K_KPE={K_KPE}  K_FULL={K_FULL}  "
          f"POOL={POOL}  N_real={N_REAL}  N_mma={N_MMA}  threads={THREADS_PER_CTA}")

    kernel = ScoreTcgen05CpAsyncDSAKpe()

    torch.manual_seed(42)
    ckv_flat = torch.randn((POOL, K_CKV),  device="cuda", dtype=torch.bfloat16) * 0.1
    kpe_flat = torch.randn((POOL, K_KPE),  device="cuda", dtype=torch.bfloat16) * 0.1
    q_rope   = torch.randn((N_MMA, K_CKV), device="cuda", dtype=torch.bfloat16) * 0.1
    q_nope   = torch.randn((N_MMA, K_KPE), device="cuda", dtype=torch.bfloat16) * 0.1

    si_full = torch.arange(M, device="cuda", dtype=torch.int32)

    SEQ_SHORT = 64
    si_short = torch.full((M,), -1, device="cuda", dtype=torch.int32)
    si_short[:SEQ_SHORT] = torch.arange(SEQ_SHORT, dtype=torch.int32)

    C     = torch.zeros((M, N_REAL), device="cuda", dtype=torch.float32)
    probe = torch.zeros((1, PROBE_COLS), dtype=torch.int64, device="cuda")

    ckv_   = from_dlpack(ckv_flat, assumed_align=128)
    kpe_   = from_dlpack(kpe_flat, assumed_align=128)
    qr_    = from_dlpack(q_rope,   assumed_align=128)
    qn_    = from_dlpack(q_nope,   assumed_align=128)
    C_     = from_dlpack(C,        assumed_align=16)
    probe_ = from_dlpack(probe,    assumed_align=8)

    si_full_ = from_dlpack(si_full, assumed_align=16)
    compiled = cute.compile(kernel, ckv_, kpe_, qr_, qn_, si_full_, C_, probe_)

    results = {}
    for case_name, si_tensor, seq_len in [
        ("full",  si_full,  M),
        ("short", si_short, SEQ_SHORT),
    ]:
        si_ = from_dlpack(si_tensor, assumed_align=16)

        for _ in range(3):
            probe.zero_(); C.zero_()
            compiled(ckv_, kpe_, qr_, qn_, si_, C_, probe_)
        torch.cuda.synchronize()

        valid_mask = (si_tensor >= 0)
        valid_idx  = si_tensor[valid_mask].long()
        ref_ckv = ckv_flat[valid_idx].float() @ q_rope[:N_REAL].float().T
        ref_kpe = kpe_flat[valid_idx].float() @ q_nope[:N_REAL].float().T
        ref     = ref_ckv + ref_kpe

        nv = int(valid_mask.sum().item())
        ok = torch.allclose(C[:nv], ref, atol=1e-2, rtol=1e-2)
        max_diff = (C[:nv] - ref).abs().max().item()

        probe.zero_(); C.zero_()
        compiled(ckv_, kpe_, qr_, qn_, si_, C_, probe_)
        torch.cuda.synchronize()

        p   = probe[0].cpu().tolist()
        cnt = int(p[0])
        probes = []
        for i in range(cnt):
            off    = PROBE_HEADER + i * PROBE_ENTRY
            tag_v  = int(p[off + 1])
            dur_ns = int(p[off + 3])
            name   = TAG_NAMES.get(tag_v, f"tag{tag_v}")
            us     = dur_ns / 1000.0
            probes.append({"phase": name, "us": us})

        total_us = next((x["us"] for x in probes if x["phase"] == "total"), 0)
        print(f"\n[{case_name:5s}]  seq_len={seq_len:3d}  "
              f"{'PASS' if ok else f'FAIL(max_diff={max_diff:.4f})'}  "
              f"total={total_us:.3f} us")
        for p_ in probes:
            print(f"  {p_['phase']:10s}: {p_['us']:7.3f} µs")

        results[case_name] = {
            "seq_len": seq_len, "correct": ok, "max_diff": float(max_diff),
            "total_us": total_us, "probes": probes,
        }

    return results


def run_intra() -> str:
    return json.dumps(run_dsa_cases(), indent=2)


if __name__ == "__main__":
    print(run_intra())

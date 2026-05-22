"""
kv_split_xor_pdl_v3_pro_v2_tcgen05.py — kv_split with tcgen05-MMA score phase.

Adapts `score_tcgen05_cpasync_dsa_full.py` (single-block DSA score + softmax +
output GEMV with packed f32x2 FFMA2) to a full kv_split compute kernel.

Layout (vs v3_pro_v2):
  * v3_pro_v2:  DIM_SPLIT(sparse) = 256, num_head_per_block = 1
                grid = [NUM_HEADS=16, NUM_SPLITS=8, 1] = 128 blocks
                Score = FastGEMV 4-row interleaved CUDA-core dot products

  * v2_tcgen05: DIM_SPLIT(sparse) = 128, num_head_per_block = 2 (N_REAL)
                grid = [HEAD_GROUPS=8, NUM_SPLITS=16, 1] = 128 blocks
                Score = single tcgen05.MmaF16BF16Op (M=128, N=8 padded, K=576)
                Output = same FFMA2 packed-(h0,h1) pattern as dsa_full,
                         covering full HEAD_DIM_CKV=512 in 4 inner chunks
                         of DIM_SPLIT_OUT=128 each.

Persistent over T with XOR swizzle on split index (matches v3_pro_v2):
  split_idx_new = T_idx ^ split_idx_old
"""

import json
import math
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.cute.nvgpu.cpasync as cpasync_mod
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import (
    from_dlpack, make_fake_compact_tensor, make_fake_stream,
)
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T


# ── Problem dims (match v3_pro_v2 + dsa_full) ────────────────────────────────
NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN = 16, 512, 64, 2048
NUM_PAGES, PAGE_SIZE = 8462, 64
T_MAX = 8
LN2 = 0.6931471805599453

# tcgen05 score-phase tile
N_REAL          = 2
N_MMA           = 8
M               = 128
K_CKV           = HEAD_DIM_CKV
K_KPE           = HEAD_DIM_KPE
K_FULL          = K_CKV + K_KPE          # 576
DIM_SPLIT       = M                      # sparse rows per block (== M)
HEAD_GROUPS     = NUM_HEADS // N_REAL    # 8
NUM_SPLITS      = TOP_K_LEN // DIM_SPLIT # 16

THREADS_PER_CTA = 512
NUM_WARPS       = THREADS_PER_CTA // 32  # 16
NUM_ROUNDS_MAX  = M // NUM_WARPS         # 8 (also output rounds)
MMA_INST_MNK    = (128, N_MMA, 16)
CTA_TILE_MNK    = (M, N_MMA, K_FULL)

# Output GEMV (FFMA2)
DIM_SPLIT_OUT   = 128                    # dims per inner chunk
N_OUT_CHUNKS    = HEAD_DIM_CKV // DIM_SPLIT_OUT  # 4
K_TILE          = 64                     # cp.async/MMA K tile (fixed)
N_KO_OUT        = DIM_SPLIT_OUT // K_TILE        # 2 K-tiles per chunk

# Softmax
SM_WARPS        = M // 32                # 4 warps cover 128 m-rows

# Reduce kernel constants (mirror v3_pro_v2's reduce)
NUM_THREADS_REDUCE = 256
NUM_WARPS_REDUCE   = NUM_THREADS_REDUCE // 32  # 8
VEC_REDUCE         = 2                          # 256×2 = 512 dims per pass

SENTINEL_SKIP = float("inf")


# ── Helpers ──────────────────────────────────────────────────────────────────

@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@cute.jit
def warp_reduce_add_i32(val: cutlass.Int32, width: cutlass.Constexpr = 32) -> cutlass.Int32:
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


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(
        dtype, cute.make_layout(shape, stride=stride), align, None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Host JIT: launch compute then reduce with PDL
# ══════════════════════════════════════════════════════════════════════════════

class KVSplitTcgen05:
    def __init__(self, sm_scale: float = 0.1352337788608801):
        self.num_stages = 1
        self.tmem_ld_rep = N_REAL
        self.sm_scale = sm_scale

    @cute.jit
    def __call__(
        self,
        q_nope:         cute.Tensor,   # (T, NUM_HEADS, K_CKV)  bf16
        q_pe:           cute.Tensor,   # (T, NUM_HEADS, K_KPE)  bf16
        ckv_cache:      cute.Tensor,   # (NUM_PAGES, PAGE_SIZE, K_CKV) bf16
        kpe_cache:      cute.Tensor,   # (NUM_PAGES, PAGE_SIZE, K_KPE) bf16
        sparse_indices: cute.Tensor,   # (T, TOP_K_LEN)         i32
        partial_out:    cute.Tensor,   # (T, NUM_HEADS, NUM_SPLITS, K_CKV) f32
        partial_lse:    cute.Tensor,   # (T, NUM_HEADS, NUM_SPLITS, 2)     f32
        output:         cute.Tensor,   # (T, NUM_HEADS, K_CKV)  bf16
        lse:            cute.Tensor,   # (T, NUM_HEADS)         f32
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

        # Flatten ckv/kpe paged caches to (N, K)
        N: cutlass.Constexpr = NUM_PAGES * PAGE_SIZE
        ckv_flat = cute.make_tensor(
            ckv_cache.iterator,
            cute.make_layout((N, K_CKV), stride=(K_CKV, 1)),
        )
        kpe_flat = cute.make_tensor(
            kpe_cache.iterator,
            cute.make_layout((N, K_KPE), stride=(K_KPE, 1)),
        )

        self.compute_kernel(
            tiled_mma, a_smem_layout, b_smem_layout, ab_dtype, acc_dtype,
            q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
            partial_out, partial_lse, output, lse,
        ).launch(
            grid=[HEAD_GROUPS, NUM_SPLITS, 1],
            block=[THREADS_PER_CTA, 1, 1],
            stream=stream, use_pdl=True,
        )

        kvsplit_reduce_kernel(
            sparse_indices, partial_out, partial_lse, output, lse,
        ).launch(
            grid=[T_MAX, NUM_HEADS, 1],
            block=[NUM_THREADS_REDUCE, 1, 1],
            stream=stream, use_pdl=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Compute kernel — persistent T-loop, XOR split swizzle
    # ══════════════════════════════════════════════════════════════════════
    @cute.kernel
    def compute_kernel(
        self,
        tiled_mma, a_smem_layout, b_smem_layout, ab_dtype, acc_dtype,
        q_nope:         cute.Tensor,
        q_pe:           cute.Tensor,
        ckv_flat:       cute.Tensor,
        kpe_flat:       cute.Tensor,
        sparse_indices: cute.Tensor,
        partial_out:    cute.Tensor,
        partial_lse:    cute.Tensor,
        output:         cute.Tensor,
        lse:            cute.Tensor,
    ):
        T, _, _ = q_nope.shape

        N_real:      cutlass.Constexpr = N_REAL
        tmem_ld_rep: cutlass.Constexpr = self.tmem_ld_rep
        sm_scale:    cutlass.Constexpr = self.sm_scale
        m_const:     cutlass.Constexpr = M
        dim_split:   cutlass.Constexpr = DIM_SPLIT
        n_out_chunks:cutlass.Constexpr = N_OUT_CHUNKS
        dim_split_out: cutlass.Constexpr = DIM_SPLIT_OUT
        head_dim_ckv:  cutlass.Constexpr = HEAD_DIM_CKV
        num_warps:   cutlass.Constexpr = NUM_WARPS
        num_rounds:  cutlass.Constexpr = NUM_ROUNDS_MAX
        sm_warps:    cutlass.Constexpr = SM_WARPS

        bidx, bidy, _ = cute.arch.block_idx()  # head_group_idx, split_idx_old
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()

        head_group_idx = bidx
        split_idx_old  = bidy
        head_base      = head_group_idx * N_real     # global head index of h0

        # ── SMEM allocation ─────────────────────────────────────────────────
        alloc = cutlass.utils.SmemAllocator()
        sA = alloc.allocate_tensor(
            element_type=ab_dtype, layout=a_smem_layout.outer,
            byte_alignment=1024, swizzle=a_smem_layout.inner,
        )
        sB = alloc.allocate_tensor(
            element_type=ab_dtype, layout=b_smem_layout.outer,
            byte_alignment=1024, swizzle=b_smem_layout.inner,
        )
        # Per-T cached sparse slice + valid count (hoisted upfront load).
        # smem_local_sparse[t, m] = clamped sparse_indices for this CTA's split
        # at token t (split_idx_new = t ^ split_idx_old). 8*128*4 = 4 KB.
        smem_local_sparse = alloc.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((T_MAX, m_const), stride=(m_const, 1)), 4, None,
        )
        smem_local_valid = alloc.allocate_tensor(
            cutlass.Int32, cute.make_layout((T_MAX,), stride=(1,)), 4, None,
        )
        smem_red_i32 = alloc.allocate_tensor(
            cutlass.Int32, cute.make_layout((num_warps,), stride=(1,)), 4, None,
        )
        smem_score = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((m_const, N_real), stride=(N_real, 1)), 16, None,
        )
        # Dedicated smem_partial (16 KB). NOTE: cannot overlay onto sA because
        # sA is swizzled — the first 16 KB of sA's physical bytes are scattered
        # across many logical (m, k, ko) positions, so a flat overlay would
        # corrupt sA chunks 1..3 read in subsequent output passes.
        smem_partial = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((num_warps, N_real, dim_split_out),
                             stride=(N_real * dim_split_out, dim_split_out, 1)),
            16, None,
        )
        smem_sm_red = alloc.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((num_warps, N_real), stride=(N_real, 1)), 16, None,
        )
        storage = alloc.allocate(self.shared_storage)
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        # ── tmem alloc + mbar init (once) ───────────────────────────────────
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

        # ── Constexpr cp.async copy atoms / tiled copies ────────────────────
        K_OUTER_CKV:   cutlass.Constexpr = K_CKV  // K_TILE      # 8
        K_OUTER_KPE_IDX: cutlass.Constexpr = K_OUTER_CKV         # 8
        VEC_BF16:      cutlass.Constexpr = 8                     # 128b atom (ckv)
        K_OUTER_HALF:  cutlass.Constexpr = K_OUTER_CKV // 2      # 4
        VEC_BF16_KPE:  cutlass.Constexpr = 2                     # 32b atom (kpe)

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

        # gmem source views  (constexpr layouts, slice per row inside loop)
        ckv_full = cute.make_tensor(
            ckv_flat.iterator,
            cute.make_layout(
                (1, NUM_PAGES * PAGE_SIZE, (K_TILE, K_OUTER_CKV)),
                stride=(0, K_CKV, (1, K_TILE)),
            ),
        )
        kpe_full = cute.make_tensor(
            kpe_flat.iterator,
            cute.make_layout(
                (1, NUM_PAGES * PAGE_SIZE, K_TILE),
                stride=(0, K_KPE, 1),
            ),
        )
        # q_nope/q_pe gmem views: (1, T, NUM_HEADS, (K_TILE, K_OUTER_CKV))
        qnope_full = cute.make_tensor(
            q_nope.iterator,
            cute.make_layout(
                (1, T, NUM_HEADS, (K_TILE, K_OUTER_CKV)),
                stride=(0, NUM_HEADS * K_CKV, K_CKV, (1, K_TILE)),
            ),
        )
        qpe_full = cute.make_tensor(
            q_pe.iterator,
            cute.make_layout(
                (1, T, NUM_HEADS, K_TILE),
                stride=(0, NUM_HEADS * K_KPE, K_KPE, 1),
            ),
        )

        # SMEM destination views for sA/sB (preserve swizzle iterator)
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

        # Output-phase tiled copy (smem → rmem, 32b per lane = 2 bf16)
        atom_s2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), ab_dtype, num_bits_per_copy=32,
        )
        OUT_VEC_PER_KO: cutlass.Constexpr = 2          # dims per lane per k_outer
        OUT_VEC_TOTAL:  cutlass.Constexpr = OUT_VEC_PER_KO * N_KO_OUT  # 4
        thr_layout_out = cute.make_layout((32,), stride=(1,))
        val_layout_out = cute.make_layout((OUT_VEC_PER_KO,), stride=(1,))
        tiled_copy_out = cute.make_tiled_copy_tv(
            atom_s2r, thr_layout_out, val_layout_out,
        )
        lane_copy_out = tiled_copy_out.get_slice(lane_idx)

        # ══════════════════════════════════════════════════════════════════
        # UPFRONT prologue (runs once, before T-loop):
        # For each token t, load this CTA's sparse slice for split_idx_new=t^split_idx_old
        # into smem_local_sparse[t,:] (clamped to ≥0) and reduce valid count
        # into smem_local_valid[t]. T-loop then reads from smem only and skips
        # entire iters with local_valid==0.
        # ══════════════════════════════════════════════════════════════════
        for t in range(T):
            split_start_t = (cutlass.Int32(t) ^ split_idx_old) * cutlass.Int32(dim_split)
            partial_valid = cutlass.Int32(0)
            if tidx < cutlass.Int32(M):
                raw = sparse_indices[t, split_start_t + tidx]
                if raw >= cutlass.Int32(0):
                    smem_local_sparse[t, tidx] = raw
                    partial_valid = cutlass.Int32(1)
                else:
                    smem_local_sparse[t, tidx] = cutlass.Int32(0)

            warp_sum = warp_reduce_add_i32(partial_valid, width=32)
            if lane_idx == cutlass.Int32(0):
                smem_red_i32[warp_idx] = warp_sum
            cute.arch.sync_threads()
            if warp_idx == cutlass.Int32(0):
                v = cutlass.Int32(0)
                if lane_idx < cutlass.Int32(NUM_WARPS):
                    v = smem_red_i32[lane_idx]
                blk = warp_reduce_add_i32(v, width=NUM_WARPS)
                if lane_idx == cutlass.Int32(0):
                    smem_local_valid[t] = blk
            cute.arch.sync_threads()

        # Signal dependents now that upfront slice is cached
        cute.arch.griddepcontrol_launch_dependents()

        # MMA-completion mbarrier phase tracker — flips ONLY on iters we issue
        mma_phase = cutlass.Int32(0)

        # ── Persistent T-loop with XOR swizzle ──────────────────────────────
        for T_idx in range(T):
            local_valid = smem_local_valid[T_idx]

            # ── OOB / empty-split skip: bypass cp.async + MMA + everything ──
            if local_valid > cutlass.Int32(0):
                split_idx_new = T_idx ^ split_idx_old

                num_rounds_act = (local_valid + cutlass.Int32(NUM_WARPS - 1)) // cutlass.Int32(NUM_WARPS)
                round_limit    = num_rounds_act * cutlass.Int32(NUM_WARPS)

                # ── cp.async A (ckv + kpe per active row) ───────────────
                for rnd in cutlass.range_constexpr(NUM_ROUNDS_MAX):
                    m_local = cutlass.Int32(rnd) * cutlass.Int32(NUM_WARPS) + warp_idx
                    if m_local < round_limit:
                        pool_idx = smem_local_sparse[T_idx, m_local]

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

                if warp_idx < cutlass.Int32(N_REAL):
                    head_g    = head_base + warp_idx
                    gB_row    = qnope_full[None, T_idx, head_g, None]
                    sB_qr_row = sB_qr     [None, warp_idx, None]
                    cute.copy(atom_cpa,
                              lane_copy.partition_S(gB_row),
                              lane_copy.partition_D(sB_qr_row))

                    gB_qn_row = qpe_full[None, T_idx, head_g, None]
                    sB_qn_row = sB_qn   [None, warp_idx,    None]
                    cute.copy(atom_cpa_kpe,
                              lane_copy_kpe.partition_S(gB_qn_row),
                              lane_copy_kpe.partition_D(sB_qn_row))

                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                cute.arch.sync_threads()

                # ── tcgen05 MMA (K=576 → 36 k-blocks) ────────────────────
                tCrA = tiled_mma.make_fragment_A(sA)
                tCrB = tiled_mma.make_fragment_B(sB)
                num_k_blocks = cute.size(tCrA, mode=[2])

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

                # ── Score epilogue: tmem → rmem → smem_score (sts.64) ────
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
                        smem_score[tidx, n_idx] = tTR_rAcc[n_idx] * cutlass.Float32(sm_scale)

                cute.arch.sync_threads()

                # ══════════════════════════════════════════════════════════
                # Softmax (4 active warps × 32 lanes = 128 m-rows)
                # ══════════════════════════════════════════════════════════
                NEG_INF: cutlass.Constexpr = -1.0e30
                s0 = cutlass.Float32(NEG_INF)
                s1 = cutlass.Float32(NEG_INF)
                if tidx < cutlass.Int32(M) and tidx < local_valid:
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
                if tidx < cutlass.Int32(M) and tidx < local_valid:
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

                # Note: keep smem_score as exp(score - max), NOT normalized.
                # The reduce kernel re-normalizes using (row_max, row_sum).
                # So partial_out[h, d] = sum_m exp(s-rmax) * ckv[m, d]
                #                      = unnormalized weighted sum.

                # ══════════════════════════════════════════════════════════
                # Output GEMV — 4 chunks of 128 dims, full 512 covered
                # ══════════════════════════════════════════════════════════
                for chunk_idx in cutlass.range_constexpr(N_OUT_CHUNKS):
                    out0 = cute.make_rmem_tensor(
                        cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)),
                        cutlass.Float32,
                    )
                    out1 = cute.make_rmem_tensor(
                        cute.make_layout((OUT_VEC_TOTAL,), stride=(1,)),
                        cutlass.Float32,
                    )
                    for v in cutlass.range_constexpr(OUT_VEC_TOTAL):
                        out0[v] = cutlass.Float32(0)
                        out1[v] = cutlass.Float32(0)

                    for round_idx in cutlass.range_constexpr(NUM_ROUNDS_MAX):
                        m_local = cutlass.Int32(round_idx) * cutlass.Int32(NUM_WARPS) + warp_idx
                        if m_local < local_valid:
                            p0 = smem_score[m_local, 0]
                            p1 = smem_score[m_local, 1]

                            for ko in cutlass.range_constexpr(N_KO_OUT):
                                k_outer_global = cutlass.Int32(chunk_idx * N_KO_OUT + ko)
                                sA_chunk = sA_ckv[0, m_local, (None, k_outer_global)]
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

                    # Store this chunk's per-warp partials
                    for ko in cutlass.range_constexpr(N_KO_OUT):
                        for v in cutlass.range_constexpr(OUT_VEC_PER_KO):
                            d_in_chunk = (cutlass.Int32(ko) * cutlass.Int32(K_TILE)
                                          + lane_idx * cutlass.Int32(OUT_VEC_PER_KO)
                                          + cutlass.Int32(v))
                            idx = ko * OUT_VEC_PER_KO + v
                            smem_partial[warp_idx, 0, d_in_chunk] = out0[idx]
                            smem_partial[warp_idx, 1, d_in_chunk] = out1[idx]
                    cute.arch.sync_threads()

                    # Cross-warp reduce + write to partial_out
                    if tidx < cutlass.Int32(N_REAL * dim_split_out):
                        h = tidx // cutlass.Int32(dim_split_out)
                        d = tidx %  cutlass.Int32(dim_split_out)
                        acc = cutlass.Float32(0)
                        for w in cutlass.range_constexpr(NUM_WARPS):
                            acc = acc + smem_partial[w, h, d]
                        d_global = cutlass.Int32(chunk_idx * dim_split_out) + d
                        partial_out[T_idx, head_base + h, split_idx_new, d_global] = acc

                    cute.arch.sync_threads()

                # Write partial_lse for both heads
                if tidx == cutlass.Int32(0):
                    partial_lse[T_idx, head_base + 0, split_idx_new, 0] = row_max_0
                    partial_lse[T_idx, head_base + 0, split_idx_new, 1] = row_sum_0
                    partial_lse[T_idx, head_base + 1, split_idx_new, 0] = row_max_1
                    partial_lse[T_idx, head_base + 1, split_idx_new, 1] = row_sum_1

                cute.arch.sync_threads()

        # ── Free tmem ────────────────────────────────────────────────────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier(barrier_id=tmem_barrier_id)
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ══════════════════════════════════════════════════════════════════════════════
# Reduce kernel — copy of v3_pro_v2 reduce, retargeted for NUM_SPLITS=16
# Each block reduces (T_idx, head_idx) over all NUM_SPLITS partials.
# ══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_reduce_kernel(
    sparse_indices: cute.Tensor,        # (T, TOP_K_LEN)
    partial_out:    cute.Tensor,        # (T, NUM_HEADS, NUM_SPLITS, K_CKV)
    partial_lse:    cute.Tensor,        # (T, NUM_HEADS, NUM_SPLITS, 2)
    output:         cute.Tensor,        # (T, NUM_HEADS, K_CKV)
    lse:            cute.Tensor,        # (T, NUM_HEADS)
):
    T, _ = sparse_indices.shape
    head_dim_ckv:   cutlass.Constexpr = HEAD_DIM_CKV
    top_k_len:      cutlass.Constexpr = TOP_K_LEN
    dim_split:      cutlass.Constexpr = DIM_SPLIT
    num_splits:     cutlass.Constexpr = NUM_SPLITS
    num_threads:    cutlass.Constexpr = NUM_THREADS_REDUCE
    num_warps:      cutlass.Constexpr = NUM_WARPS_REDUCE
    vec_reduce:     cutlass.Constexpr = VEC_REDUCE
    t_max:          cutlass.Constexpr = T_MAX

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()

    block_t_idx = bidx
    head_idx    = bidy

    alloc = cutlass.utils.SmemAllocator()
    smem_red_i32 = alloc.allocate_tensor(
        cutlass.Int32, cute.make_layout((32,), stride=(1,)), 4, None,
    )
    smem_max_sum = alloc.allocate_tensor(
        cutlass.Float32, cute.make_layout((num_splits, 2), stride=(2, 1)), 4, None,
    )

    partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, vec_reduce))
    output_v      = cute.zipped_divide(output,      (1, 1, vec_reduce))

    num_groups = (T + t_max - 1) // t_max
    for group_idx in range(num_groups):
        T_idx = group_idx * t_max + block_t_idx
        if T_idx < T:
            partial_cnt = 0
            for i in range(tidx, top_k_len, num_threads):
                idx = sparse_indices[T_idx, i]
                if idx >= cutlass.Int32(0):
                    partial_cnt += 1

            cnt_sum = warp_reduce_add_i32(partial_cnt, width=32)
            if lane_idx == 0:
                smem_red_i32[warp_idx] = cnt_sum
            cute.arch.sync_threads()

            if warp_idx == 0:
                val = smem_red_i32[lane_idx]
                cnt_sum = warp_reduce_add_i32(val, width=num_warps)
                smem_red_i32[0] = cnt_sum
            cute.arch.sync_threads()

            num_valid = smem_red_i32[0]

            cute.arch.griddepcontrol_wait()

            if num_valid > cutlass.Int32(0):
                num_active_splits = (num_valid + cutlass.Int32(dim_split) - cutlass.Int32(1)) // cutlass.Int32(dim_split)

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

                    a = partial_out_v[(0, 0, 0, None), (T_idx, head_idx, s, tidx)].load()
                    acc = acc + scale * a

                if tidx == 0:
                    lse[T_idx, head_idx] = (g_max + cute.math.log(g_lse_sum)) / cutlass.Float32(LN2)

                output_v[(0, 0, None), (T_idx, head_idx, tidx)].store(
                    (acc / g_lse_sum).to(cutlass.BFloat16)
                )

            cute.arch.sync_threads()


# ══════════════════════════════════════════════════════════════════════════════
# Compilation + run wrapper
# ══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align,
    )


def compile_kernel():
    T = cute.sym_int()
    num_heads      = NUM_HEADS
    num_splits     = NUM_SPLITS

    q_nope         = _fake(cute.BFloat16, (T, num_heads, K_CKV),                      (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, K_KPE),                      (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (NUM_PAGES, PAGE_SIZE, K_CKV),              (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (NUM_PAGES, PAGE_SIZE, K_KPE),              (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K_LEN),                             (1, 0),     4)
    partial_out    = _fake(cute.Float32,  (T, num_heads, num_splits, K_CKV),          (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (T, num_heads, num_splits, 2),              (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, num_heads, K_CKV),                      (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),                             (1, 0),     4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = KVSplitTcgen05()
    return cute.compile(
        kernel,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        partial_out, partial_lse, output, lse, stream,
        options="--enable-tvm-ffi",
    )


_compiled = compile_kernel()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    T = q_nope.shape[0]
    partial_out = torch.empty(T, NUM_HEADS, NUM_SPLITS, HEAD_DIM_CKV,
                              dtype=torch.float32, device=output.device)
    partial_lse = torch.empty(T, NUM_HEADS, NUM_SPLITS, 2,
                              dtype=torch.float32, device=output.device)
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
              partial_out, partial_lse, output, lse)

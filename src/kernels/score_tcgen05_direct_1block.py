"""
score_tcgen05_direct_1block.py — Single-block for-loop, NO TMA.

Same single-CTA design as score_tcgen05_1block.py but replaces TMA
(GMEM→SMEM) with cooperative GMEM→SMEM copy via:
  1. thr_mma.partition_A/B  — reshape GMEM to match MMA tile layout
  2. local_partition         — divide elements across 128 threads
  3. autovec_copy            — each thread copies its share
  4. sync_threads            — ensure SMEM is globally visible
  5. tcgen05_fence           — order sync before MMA reads SMEM

Key contrasts vs score_tcgen05_1block.py:
  - No TMA atom, no tma_mbar, no prefetch_descriptor
  - Receives raw GMEM tensors mA / mB instead of TMA descriptor tensors
  - thr_mma is used for BOTH GMEM partitioning (copy) and MMA descriptor setup

GEMM formulation (K-major A and B):
  C[M=256, N=8] = A[M, K] × B[N, K].T
  scores[256]   = C[:, 0]   (only TMEM column 0 stored)

Where:
  A = KV tokens  [256, 512] BF16 — one split's worth of cached KV
  B = query      [  8, 512] BF16 — row 0: actual query, rows 1-7: zeros

MMA instruction: (M_inst=128, N_inst=8, K_inst=16)
  → 2 M-tiles over 256 KV tokens, 1 N-tile, 8 BK tiles × 4 K-steps
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.testing import benchmark, JitArguments

# ── Dimensions ───────────────────────────────────────────────────────────────
M = 256    # KV tokens per split
N = 8      # padded query rows (1 actual + 7 zeros)
K = 512    # head_dim_ckv
BK = 64    # K-tile per CTA (K/BK = 8 outer tiles)

MMA_INST_MNK    = (128, 8, 16)
CTA_TILE_MNK    = (128, N, BK)
NUM_M_TILES     = M // CTA_TILE_MNK[0]   # 2
NUM_BK_TILES    = K // BK                 # 8

THREADS_PER_CTA = 128
TMEM_LD_REP     = 1              # 1 → 1 reg/thread = TMEM col 0
TMEM_ALLOC_COLS = 32             # hardware minimum (only 8 cols used)


# ── Helper ────────────────────────────────────────────────────────────────────
@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )


# ── Kernel class ──────────────────────────────────────────────────────────────
class ScoreGEMM_Direct_1Block:
    """
    Single-CTA tcgen05 GEMM — no TMA. GMEM→SMEM via cooperative copy.
    One block iterates over 2 M-tiles via cutlass.range_constexpr.
    Output: scores[M=256] written to c_out[M, 1].
    """

    def __init__(self):
        self.BM, self.BN, self.BK   = CTA_TILE_MNK
        self.mma_inst_shape_mnk     = MMA_INST_MNK
        self.threads_per_cta        = THREADS_PER_CTA
        self.num_stages             = 1
        self.tmem_ld_rep            = TMEM_LD_REP

    # ----------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        kv:    cute.Tensor,   # A: [M=256, K=512] BF16
        q_pad: cute.Tensor,   # B: [N=8,   K=512] BF16
        c_out: cute.Tensor,   # output [M=256, 1] Float32
    ):
        self.kv_dtype  = kv.element_type
        self.q_dtype   = q_pad.element_type
        self.c_dtype   = c_out.element_type
        self.acc_dtype = cutlass.Float32

        op = tcgen05.MmaF16BF16Op(
            self.kv_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        print("tiled_mma:", self.tiled_mma)

        # Swizzled SMEM layouts required for MMA SMEM descriptors
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, CTA_TILE_MNK, kv.element_type, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, CTA_TILE_MNK, q_pad.element_type, self.num_stages,
        )

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        # Single block: for loop over M-tiles internally
        self.kernel(
            self.tiled_mma,
            kv,
            q_pad,
            c_out,
            self.a_smem_layout,
            self.b_smem_layout,
        ).launch(
            grid=(1, 1, 1),
            block=(self.threads_per_cta, 1, 1),
        )

    # ----------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma:     cute.TiledMma,
        mA:            cute.Tensor,   # GMEM kv    [M, K]
        mB:            cute.Tensor,   # GMEM q_pad [N, K]
        mC:            cute.Tensor,   # output     [M, 1]
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)

        # ── SMEM allocation ──────────────────────────────────────────
        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = smem.allocate_tensor(
            element_type=self.kv_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        # ── MMA SMEM descriptors (invariant: same sA/sB buffer each M-tile) ─
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # ── TMEM allocation ──────────────────────────────────────────
        acc_shape       = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc          = tiled_mma.make_fragment_C(acc_shape)
        tmem_alloc_cols = cutlass.Int32(TMEM_ALLOC_COLS)

        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        tmem_barrier_id = 1
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=self.threads_per_cta)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # ── Epilogue atoms (invariant) ────────────────────────────────
        M_acc = cute.size(tCtAcc, mode=[0, 0])
        print(f"M_acc={M_acc}")

        # Ld32x32bOp Repetition(1): 1 reg/thread = TMEM col 0
        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        subtile_n       = self.tmem_ld_rep
        epi_tiler       = ((M_acc, subtile_n),)
        tCtAcc_epi      = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        # ── CTA-level MMA slice (T=1 for tcgen05) ─────────────────────
        # thr_mma serves double duty:
        #   1. partition_A/B(gA/gB): reshape GMEM into MMA-tiled view for copy
        #   2. make_fragment_A/B(sA/sB): build SMEM descriptors for MMA
        thr_mma = tiled_mma.get_slice(thr_idx=0)

        # ── Invariant per-thread SMEM partitions for cooperative copy ──
        # sA/sB layout: ((M_mma,K_mma), 1, K_blocks, stages)
        # [None,None,None,0] selects stage=0, yielding a 3-D tensor.
        # local_partition divides all logical elements among 128 threads.
        thr_layout = cute.make_layout(self.threads_per_cta)
        sA_thr = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        sB_thr = cute.local_partition(sB[None, None, None, 0], thr_layout, tidx)

        num_k_blocks = cute.size(tCrA, mode=[2])

        mma_phase = 0

        # ── For loop over M-tiles ([0:128], [128:256]) ────────────────
        # cutlass.range_constexpr forces Python-level unrolling — no scf.for,
        # no loop-carried tiled_mma state (which would cause a compile error).
        for m_tile_idx in cutlass.range_constexpr(NUM_M_TILES):   # unrolled: 0, 1
            mma_coord = (m_tile_idx, 0, None)

            # ─ GMEM tile for this M-tile ───────────────────────────
            gA = cute.local_tile(mA, CTA_TILE_MNK, mma_coord, proj=(1, None, 1))
            gB = cute.local_tile(mB, CTA_TILE_MNK, mma_coord, proj=(None, 1, 1))

            # ─ Reshape GMEM to match MMA-partitioned SMEM structure ─
            # For tcgen05 (T=1), partition_A/B gives:
            #   tCgA: ((128,16), 1, K_blocks, BK_tiles=8)
            #   tCgB: ((8,16),   1, K_blocks, BK_tiles=8)
            tCgA = thr_mma.partition_A(gA)
            tCgB = thr_mma.partition_B(gB)

            # ─ Outer loop over BK tiles (BK=64, 8 tiles) ──────────
            for bk_idx in cutlass.range_constexpr(NUM_BK_TILES):

                # ─ Cooperative GMEM→SMEM for this BK tile ──────────
                gA_thr = cute.local_partition(tCgA[None, None, None, bk_idx], thr_layout, tidx)
                gB_thr = cute.local_partition(tCgB[None, None, None, bk_idx], thr_layout, tidx)
                cute.autovec_copy(gA_thr, sA_thr)
                cute.autovec_copy(gB_thr, sB_thr)

                cute.arch.sync_threads()
                tcgen05_fence()

                # ─ MMA over K-blocks within this BK tile ───────────
                if warp_idx == 0:
                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        if bk_idx == 0:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_block_idx != 0)
                        else:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        k_block_coord = (None, None, k_block_idx, 0)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            tCtAcc,
                        )

                    if tidx == 0:
                        tcgen05.commit(mma_mbar)

                cute.arch.mbarrier_wait(mma_mbar, mma_phase)
                mma_phase ^= 1

            # ─ Epilogue: TMEM col-0 → 1 reg/thread → GMEM[M,1] ───
            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)
            m_base = m_tile_idx * M_acc
            mC[m_base + tidx, 0] = tTR_rAcc[0]

        # ── TMEM dealloc (after all M-tiles done) ─────────────────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── Host: compile + correctness test ──────────────────────────────────────────
def main():
    kv    = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    q     = torch.randn((1, K), device="cuda", dtype=torch.bfloat16)
    q_pad = torch.empty((N, K), device="cuda", dtype=torch.bfloat16)
    q_pad[0] = q[0]

    c_out = torch.zeros((M, 1), device="cuda", dtype=torch.float32)

    kv_    = from_dlpack(kv,    assumed_align=16)
    q_pad_ = from_dlpack(q_pad, assumed_align=16)
    c_out_ = from_dlpack(c_out, assumed_align=16)

    gemm     = ScoreGEMM_Direct_1Block()
    compiled = cute.compile(gemm, kv_, q_pad_, c_out_)
    compiled(kv_, q_pad_, c_out_)

    scores     = c_out[:, 0]
    ref_scores = (kv.float() @ q.float().T).squeeze(1)

    atol, rtol = 1e-1, 1e-1
    match   = torch.allclose(scores, ref_scores, atol=atol, rtol=rtol)
    max_err = (scores - ref_scores).abs().max().item()
    print(f"\nCORRECTNESS {'PASS' if match else 'FAIL'}  (max_err={max_err:.4f})")
    if not match:
        print("  scores[:8]:", scores[:8].tolist())
        print("  ref   [:8]:", ref_scores[:8].tolist())

    t = benchmark(compiled, kernel_arguments=JitArguments(kv_, q_pad_, c_out_))
    print(f"DURATION: {t:.4f} µs")


if __name__ == "__main__":
    main()

"""
score_direct_tcgen05.py — Score phase using tcgen05.mma WITHOUT TMA.

Same GEMM as score_tcgen05.py but replaces TMA (GMEM→SMEM) with
cooperative GMEM→SMEM copies via partition_A/B + local_partition + autovec_copy.

GMEM is reshaped to match the MMA-partitioned SMEM layout using partition_A/B,
then elements are divided among threads via local_partition for cooperative copy.

GEMM formulation (K-major A and B):
  C[M=256, N=8] = A[M, K] × B[N, K].T
  scores[256]   = C[:, 0]

Where:
  A = KV tokens    [256, 512] BF16 — one split's worth of cached KV
  B = padded query [  8, 512] BF16 — row 0: actual query, rows 1–7: zeros

MMA instruction: (M_inst=128, N_inst=8, K_inst=16)
  → 2 M-tiles over 256 KV tokens, 1 N-tile (query padded to 8), 32 K-steps
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.testing import benchmark, JitArguments

# ── Dimensions ───────────────────────────────────────────────────────────────
M = 256    # KV tokens per split
N = 8      # padded query rows (1 actual query + 7 zero rows)
K = 512    # head_dim_ckv

MMA_INST_MNK    = (128, 8, 16)   # tcgen05 instruction shape M×N×K
CTA_TILE_MNK    = (128, N, K)    # one CTA covers 128 M-rows, all N, all K

THREADS_PER_CTA = 128
TMEM_LD_REP     = 4              # tcgen05.ld repetition factor


# ── Helper: tcgen05.fence::after_thread_sync ─────────────────────────────────
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


# ── Kernel class ─────────────────────────────────────────────────────────────
class ScoreGEMM_Direct:
    """
    tcgen05 GEMM that computes attention scores — NO TMA.
    Uses partition_A/B + local_partition + autovec_copy for GMEM→SMEM.

    A[256, 512] (KV tokens, K-major) × B[8, 512].T (padded query, K-major)
      = C[256, 8]   →   scores = C[:, 0]
    """

    def __init__(self):
        self.BM, self.BN, self.BK   = CTA_TILE_MNK
        self.mma_inst_shape_mnk     = MMA_INST_MNK
        self.threads_per_cta        = THREADS_PER_CTA
        self.num_stages             = 1
        self.tmem_ld_rep            = TMEM_LD_REP

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        kv:     cute.Tensor,   # A: [M=256, K=512] BF16  K-major
        q_pad:  cute.Tensor,   # B: [N=8,   K=512] BF16  K-major
        c_out:  cute.Tensor,   # output  [M=256, N=8] Float32
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
            tcgen05.OperandMajorMode.K,   # A: [M, K] row-major
            tcgen05.OperandMajorMode.K,   # B: [N, K] row-major → transposed in MMA
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        print("tiled_mma:", self.tiled_mma)

        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, CTA_TILE_MNK, kv.element_type, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, CTA_TILE_MNK, q_pad.element_type, self.num_stages,
        )
        print("a_smem_layout:", self.a_smem_layout)
        print("b_smem_layout:", self.b_smem_layout)

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        # 2 blocks: each covers one 128-row M-tile of the 256-row KV matrix
        grid_m = kv.shape[0] // self.BM   # = M // 128 = 2
        self.kernel(
            self.tiled_mma,
            kv,
            q_pad,
            c_out,
            self.a_smem_layout,
            self.b_smem_layout,
        ).launch(
            grid=(grid_m, 1, 1),
            block=(self.threads_per_cta, 1, 1),
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma:       cute.TiledMma,
        mA:              cute.Tensor,   # GMEM kv    [M, K]
        mB:              cute.Tensor,   # GMEM q_pad [N, K]
        mC:              cute.Tensor,   # output     [M, N]
        a_smem_layout:   cute.ComposedLayout,
        b_smem_layout:   cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)
        bidx, bidy, _ = cute.arch.block_idx()   # bidx = M-tile index (0 or 1)

        # ── SMEM allocation ───────────────────────────────────────────
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

        # ── GMEM tile views: each block handles one 128-row M-tile ────
        mma_coord_mnk = (bidx, bidy, None)
        gA = cute.local_tile(mA, CTA_TILE_MNK, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB, CTA_TILE_MNK, mma_coord_mnk, proj=(None, 1, 1))
        gC = cute.local_tile(mC, CTA_TILE_MNK, mma_coord_mnk, proj=(1, 1, None))

        # ── MMA partitioning ──────────────────────────────────────────
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgC    = thr_mma.partition_C(gC)

        # Reshape GMEM to MMA-partitioned shape (matches SMEM structure)
        tCgA = thr_mma.partition_A(gA)  # ((128,16), 1, 32, K_tiles=1)
        tCgB = thr_mma.partition_B(gB)  # ((8,16),   1, 32, K_tiles=1)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc    = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols  = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        print("tCgA:", tCgA)
        print("sA:", sA)
        print("tCgB:", tCgB)
        print("sB:", sB)

        # ── Barriers ─────────────────────────────────────────────────
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

        # ── Epilogue setup: TMEM → RMEM → GMEM ───────────────────────
        M_acc = cute.size(tCtAcc, mode=[0, 0])
        N_acc = cute.size(tCtAcc, mode=[0, 1])

        num_dp = M_acc // 4
        if cutlass.const_expr(num_dp == 32):
            ld_op = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
            fp32_cols_per_rep = 1
        elif cutlass.const_expr(num_dp == 16):
            ld_op = tcgen05.Ld16x256bOp(tcgen05.Repetition(self.tmem_ld_rep))
            fp32_cols_per_rep = 8

        subtile_n   = self.tmem_ld_rep * fp32_cols_per_rep
        epi_tiler   = ((M_acc, subtile_n),)

        tCtAcc_epi  = cute.zipped_divide(tCtAcc, epi_tiler)
        gC_epi      = cute.zipped_divide(tCgC, epi_tiler)

        copy_atom_t2r    = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy  = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy    = tmem_tiled_copy.get_slice(tidx)

        tTR_tAcc = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_gC   = tmem_thr_copy.partition_D(gC_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[None, None, 0].shape, self.acc_dtype)
        tTR_rC   = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)

        simt_atom   = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        subtile_cnt = cute.size(tTR_tAcc, mode=[2])

        # ── Cooperative GMEM → SMEM copy via local_partition + autovec ──
        # Select single K-tile (K=BK=512) from GMEM partitions
        tCgA_sel = tCgA[None, None, None, 0]   # remove K-tiles dim
        tCgB_sel = tCgB[None, None, None, 0]

        # Remove stage dim from SMEM (num_stages=1)
        sA_sel = sA[None, None, None, 0]
        sB_sel = sB[None, None, None, 0]

        # Divide elements among 128 threads via local_partition.
        # Both GMEM and SMEM have matching MMA-partitioned shape — the first
        # sub-mode (M=128 or N=8) gets divided among threads, giving each
        # thread its own row(s) of K elements.
        thr_layout = cute.make_layout(self.threads_per_cta)

        gA_thr = cute.local_partition(tCgA_sel, thr_layout, tidx)
        sA_thr = cute.local_partition(sA_sel, thr_layout, tidx)
        cute.autovec_copy(gA_thr, sA_thr)

        gB_thr = cute.local_partition(tCgB_sel, thr_layout, tidx)
        sB_thr = cute.local_partition(sB_sel, thr_layout, tidx)
        cute.autovec_copy(gB_thr, sB_thr)

        cute.arch.sync_threads()    # ensure SMEM visible to all threads

        tcgen05_fence()

        # ── MMA: iterate over K-blocks within the tile ────────────────
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        mma_phase = 0
        num_k_blocks = cute.size(tCrA, mode=[2])

        if warp_idx == 0:
            for k_block_idx in range(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, 0)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            if tidx == 0:
                tcgen05.commit(mma_mbar)

        cute.arch.mbarrier_wait(mma_mbar, mma_phase)

        # ── Epilogue: TMEM → RMEM → GMEM ─────────────────────────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        for subtile_idx in range(subtile_cnt):
            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, subtile_idx], tTR_rAcc)
            tTR_rC.store(tTR_rAcc.load().to(self.c_dtype))
            cute.copy(simt_atom, tTR_rC, tTR_gC[None, None, subtile_idx])

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── Host: compile + correctness test ─────────────────────────────────────────
def main():
    # ── Build test tensors ────────────────────────────────────────────────────
    kv    = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    q     = torch.randn((1, K), device="cuda", dtype=torch.bfloat16)

    # Pad query with 7 zero rows → [8, 512]
    q_pad = torch.zeros((N, K), device="cuda", dtype=torch.bfloat16)
    q_pad[0] = q[0]

    c_out = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    kv_    = from_dlpack(kv,    assumed_align=16)
    q_pad_ = from_dlpack(q_pad, assumed_align=16)
    c_out_ = from_dlpack(c_out, assumed_align=16)

    # ── Compile & run ─────────────────────────────────────────────────────────
    gemm     = ScoreGEMM_Direct()
    compiled = cute.compile(gemm, kv_, q_pad_, c_out_)
    compiled(kv_, q_pad_, c_out_)

    # ── Extract scores (first column = actual query) ──────────────────────────
    scores = c_out[:, 0]   # [256]

    # ── Reference: scores[i] = dot(kv[i], q[0]) ──────────────────────────────
    ref_scores = (kv.float() @ q.float().T).squeeze(1)  # [256]

    atol, rtol = 1e-1, 1e-1
    match = torch.allclose(scores, ref_scores, atol=atol, rtol=rtol)
    max_err = (scores - ref_scores).abs().max().item()
    print(f"\nCORRECTNESS {'PASS' if match else 'FAIL'}  (max_err={max_err:.4f})")
    if not match:
        print("  scores[:8]:", scores[:8].tolist())
        print("  ref   [:8]:", ref_scores[:8].tolist())

    # ── Quick latency ─────────────────────────────────────────────────────────
    t = benchmark(compiled, kernel_arguments=JitArguments(kv_, q_pad_, c_out_))
    print(f"DURATION: {t:.4f} µs")


if __name__ == "__main__":
    main()

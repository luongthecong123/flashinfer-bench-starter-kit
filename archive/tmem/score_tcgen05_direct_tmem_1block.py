"""
score_tcgen05_direct_tmem_1block.py — Single-block, NO TMA, UMMA TS mode.

Same single-CTA design as score_tcgen05_direct_1block.py but uses
UMMA TS mode (tcgen05.OperandSource.TMEM) where operand A is loaded
from TMEM instead of SMEM.

Data flow for A:
  GMEM[BF16]  →  recast to F32  →  RMEM  →  TMEM  (via St32x32bOp)
               ─────────────────────────────────────┘  (per BK tile)

Data flow for B (unchanged):
  GMEM[BF16]  →  SMEM  (autovec_copy, cooperative)

MMA: tcgen05 TS mode — A read from TMEM, B read from SMEM desc.

TMEM layout:
  cols  0 .. TMEM_A_COLS-1  : A operand  (BK//2 Float32-wide = BK BF16)
  cols  TMEM_A_COLS .. +31   : C accumulator  (hardware minimum 32)
  Total allocation: TMEM_A_COLS + TMEM_C_COLS = 64 columns

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

ROWS            = 128             # TMEM has exactly 128 lanes
MMA_INST_MNK    = (128, 8, 16)
CTA_TILE_MNK    = (128, N, BK)
NUM_M_TILES     = M // CTA_TILE_MNK[0]   # 2
NUM_BK_TILES    = K // BK                 # 8

THREADS_PER_CTA = 128
TMEM_LD_REP     = 1              # 1 → 1 reg/thread = TMEM col 0 (epilogue)
# TMEM A:  BK BF16 per row = BK//2 Float32 columns
TMEM_A_COLS     = BK // 2        # = 32
TMEM_C_COLS     = 32             # hardware minimum (N=8 needs 8, min alloc = 32)
TMEM_ALLOC_COLS = TMEM_A_COLS + TMEM_C_COLS   # = 64


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
class ScoreGEMM_Direct_TMEM_1Block:
    """
    Single-CTA tcgen05 GEMM with UMMA TS mode.
    A (KV tokens) is loaded GMEM → RMEM → TMEM each BK tile via St32x32bOp.
    B (query) is loaded GMEM → SMEM via cooperative autovec_copy.
    MMA reads A from TMEM (TS mode) and B from SMEM (descriptors).
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

        # ── TS mode MMA atom (A from TMEM, B from SMEM) ──────────────
        op = tcgen05.MmaF16BF16Op(
            self.kv_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,    # <-- TS mode
            tcgen05.OperandMajorMode.K,    # A: [M, K] row-major
            tcgen05.OperandMajorMode.K,    # B: [N, K] row-major
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        print("tiled_mma:", self.tiled_mma)

        # ── TMEM A layout (following FMHA pattern: make_smem_layout_a
        #    with TS-mode tiled_mma gives a TMEM-compatible layout) ─────
        self.a_ts_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, CTA_TILE_MNK, self.kv_dtype, self.num_stages,
        )

        # ── B still goes through SMEM ─────────────────────────────────
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, CTA_TILE_MNK, self.q_dtype, self.num_stages,
        )

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma,
            kv,
            q_pad,
            c_out,
            self.a_ts_layout,
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
        mA:            cute.Tensor,   # GMEM kv    [M, K] BF16
        mB:            cute.Tensor,   # GMEM q_pad [N, K] BF16
        mC:            cute.Tensor,   # output     [M, 1] Float32
        a_ts_layout:   cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)

        # ── SMEM allocation (B only; A goes to TMEM) ─────────────────
        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sB = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        # ── B SMEM descriptor (invariant across all tiles) ────────────
        tCrB = tiled_mma.make_fragment_B(sB)

        # ── TMEM allocation: A (cols 0..TMEM_A_COLS-1) + ─────────────
        #    C accumulator (cols TMEM_A_COLS..TMEM_A_COLS+31) ──────────
        acc_shape          = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc_proto       = tiled_mma.make_fragment_C(acc_shape)
        tmem_alloc_cols    = cutlass.Int32(TMEM_ALLOC_COLS)

        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        tmem_barrier_id = 1
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=self.threads_per_cta)

        # Float32 TMEM pointer for both A store and C accumulator
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tmem_ptr_bf16 = cute.arch.retrieve_tmem_ptr(
            cutlass.BFloat16, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )

        # ── TMEM A tensor: structured layout from make_smem_layout_a ──
        # a_ts_layout.outer gives ((M_mma, K_inst), K_warp, K_blocks, stages)
        # Following FMHA: make_tensor(float32_tmem_ptr, a_ts_layout.outer)
        # works for make_fragment_A because TMEM ptr unit = stride unit.
        tA_tmem = cute.make_tensor(tmem_ptr, a_ts_layout.outer)
        print(f"tA_tmem shape: {tA_tmem.shape},  layout: {tA_tmem.layout}")

        # ── TMEM C (accumulator) starts after A ──────────────────────
        tCtAcc = cute.make_tensor(tmem_ptr + TMEM_A_COLS, tCtAcc_proto.layout)

        # ── TS A fragment: TMEM-backed, indexed by K-block ────────────
        tCrA = tiled_mma.make_fragment_A(tA_tmem)
        print(f"tCrA shape: {tCrA.shape}")

        # ── TMEM A store setup: Float32 (128 rows × 32 cols = BK BF16) ──────
        # Using F32 ptr for St32x32bOp — each col is 32b = 2 BF16.
        tA_store = cute.make_tensor(
            tmem_ptr,
            cute.make_layout((ROWS, TMEM_A_COLS), stride=(65536, 1)),
        )
        cT_A = cute.make_identity_tensor((ROWS, TMEM_A_COLS))
        rep_A    = tcgen05.Repetition(TMEM_A_COLS)
        ld_atom_A = cute.make_copy_atom(tcgen05.Ld32x32bOp(rep_A), self.acc_dtype)
        st_atom  = cute.make_copy_atom(tcgen05.St32x32bOp(rep_A), self.acc_dtype)
        tiled_ld_A = tcgen05.make_tmem_copy(ld_atom_A, tA_store)
        tiled_st = tcgen05.make_tmem_copy(st_atom, tA_store)
        thr_ld_A = tiled_ld_A.get_slice(tidx)
        thr_st   = tiled_st.get_slice(tidx)
        tA_dst   = thr_st.partition_D(tA_store)
        tA_coord = thr_ld_A.partition_D(cT_A)  # gives per-thread shape

        # RMEM buffer: shape from Ld atom's coord partition (like z2_tmem_lower.py)
        mA_f32 = cute.make_tensor(
            cute.recast_ptr(mA.iterator, dtype=cutlass.Float32),
            cute.make_layout((M, K // 2), stride=(K // 2, 1)),
        )
        rA_f32 = cute.make_rmem_tensor(tA_coord.shape, self.acc_dtype)
        print(f"rA_f32 shape: {rA_f32.shape}")

        simt_f32 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype)

        # ── Epilogue atoms (invariant) ────────────────────────────────
        M_acc = cute.size(tCtAcc, mode=[0, 0])
        print(f"M_acc={M_acc}")

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
        thr_mma = tiled_mma.get_slice(thr_idx=0)

        # ── Invariant B SMEM partitions for cooperative copy ──────────
        thr_layout = cute.make_layout(self.threads_per_cta)
        sB_thr = cute.local_partition(sB[None, None, None, 0], thr_layout, tidx)

        num_k_blocks = cute.size(tCrA, mode=[2])
        mma_phase    = 0

        # ── For loop over M-tiles ([0:128], [128:256]) ────────────────
        for m_tile_idx in cutlass.range_constexpr(NUM_M_TILES):
            mma_coord = (m_tile_idx, 0, None)

            gB = cute.local_tile(mB, CTA_TILE_MNK, mma_coord, proj=(None, 1, 1))
            tCgB = thr_mma.partition_B(gB)

            # ── Outer loop over BK tiles ───────────────────────────────
            for bk_idx in cutlass.range_constexpr(NUM_BK_TILES):

                # ─ GMEM(F32-recast)→RMEM(F32)→TMEM for A ────────────────
                gA_f32_tile = cute.local_tile(mA_f32, (ROWS, TMEM_A_COLS), (m_tile_idx, bk_idx))
                gA_f32_thr = thr_ld_A.partition_D(gA_f32_tile)  # per-thread GMEM slice
                cute.copy(simt_f32, gA_f32_thr, rA_f32)   # GMEM F32 → RMEM F32
                cute.copy(tiled_st, rA_f32, tA_dst)        # RMEM F32 → TMEM
                cute.arch.fence_view_async_tmem_store()   # TMEM visible to MMA

                # ─ Cooperative GMEM→SMEM for B ────────────────────────
                gB_thr = cute.local_partition(tCgB[None, None, None, bk_idx], thr_layout, tidx)
                cute.autovec_copy(gB_thr, sB_thr)

                cute.arch.sync_threads()
                tcgen05_fence()

                # ─ MMA over K-blocks (warp 0 only) ────────────────────
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

            # ─ Epilogue: TMEM col-0 → 1 reg/thread → GMEM[M,1] ──────
            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)
            m_base = m_tile_idx * M_acc
            mC[m_base + tidx, 0] = tTR_rAcc[0]

        # ── TMEM dealloc ──────────────────────────────────────────────
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

    gemm     = ScoreGEMM_Direct_TMEM_1Block()
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

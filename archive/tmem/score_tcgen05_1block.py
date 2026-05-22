"""
score_tcgen05_1block.py — Single-block for-loop design (WIP / experimental).

One CTA iterates over M-tiles with a for loop.
TMEM is allocated once (minimum 32 columns; N=8 only needs 8 but hardware
requires a power-of-2 >= 32).

GEMM formulation (K-major A and B):
  C[M=256, N=8] = A[M, K] × B[N, K].T
  scores[256]   = C[:, 0]

Where:
  A = KV tokens  [256, 512] BF16 — one split's worth of cached KV
  B = query      [  8, 512] BF16 — row 0: actual query, rows 1-7: garbage
                                    (only column 0 of C is read back)

MMA instruction: (M_inst=128, N_inst=8, K_inst=16)
  → 2 M-tiles over 256 KV tokens, 1 N-tile, 32 K-steps
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
M = 256    # KV tokens per split  (dim_split = TOP_K_LEN // NUM_SPLITS = 2048//8)
N = 8      # padded query rows (1 actual query + 7 zero rows)
K = 512    # head_dim_ckv

MMA_INST_MNK    = (128, 8, 16)   # tcgen05 instruction shape M×N×K
# sA=128KB, sB=8KB → 136KB (fits in ~228KB SMEM). Two M-tiles in a for loop.
CTA_TILE_MNK    = (128, N, K)    # one CTA covers 128 M-rows, all N, all K
NUM_M_TILES     = M // CTA_TILE_MNK[0]   # 2

THREADS_PER_CTA = 128
TMEM_LD_REP     = 1              # tcgen05.ld repetition factor (1 → 1 reg/thread = col 0)
# N=8 needs only 8 TMEM columns, but hardware minimum allocation is 32 (power-of-2 >= 32).
TMEM_ALLOC_COLS = 32


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
class ScoreGEMM:
    """
    Single-CTA tcgen05 GEMM that computes attention scores for one split.

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

        a_smem_layout_one_stage = cute.select(self.a_smem_layout, mode=[0, 1, 2])
        b_smem_layout_one_stage = cute.select(self.b_smem_layout, mode=[0, 1, 2])

        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op_g2s, kv, a_smem_layout_one_stage, CTA_TILE_MNK, self.tiled_mma,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, q_pad, b_smem_layout_one_stage, CTA_TILE_MNK, self.tiled_mma,
        )

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        # Single block: iterates over M-tiles internally via for loop.
        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            c_out,
            self.a_smem_layout,
            self.b_smem_layout,
        ).launch(
            grid=(1, 1, 1),
            block=(self.threads_per_cta, 1, 1),
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma:       cute.TiledMma,
        tma_atom_a:      cute.CopyAtom,
        mA_tma_tensor:   cute.Tensor,   # TMA view of kv   [M, K]
        tma_atom_b:      cute.CopyAtom,
        mB_tma_tensor:   cute.Tensor,   # TMA view of q_pad [N, K]
        mC:              cute.Tensor,   # output            [M, N]
        a_smem_layout:   cute.ComposedLayout,
        b_smem_layout:   cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

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

        # ── MMA fragments referencing SMEM (invariant across M-tiles) ─
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # ── TMEM allocation ──────────────────────────────────────────
        # N=8 → 8 TMEM columns needed, but hardware minimum is 32.
        # tcgen05.alloc requires a power-of-2 >= 32 (see Blackwell docs).
        acc_shape       = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc          = tiled_mma.make_fragment_C(acc_shape)
        tmem_alloc_cols = cutlass.Int32(TMEM_ALLOC_COLS)   # 32 (min; only 8 used)
        print(f"tmem_alloc_cols={TMEM_ALLOC_COLS} (need 8, must alloc >=32)")

        # ── Barriers ─────────────────────────────────────────────────
        tma_mbar = storage.tma_mbar_ptr.data_ptr()
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        tma_transaction_bytes = (
            cute.size_in_bytes(self.kv_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
            + cute.size_in_bytes(self.q_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
        )

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(tma_mbar, cnt=1)
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        tmem_barrier_id = 1
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=self.threads_per_cta)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # ── Epilogue atoms (invariant across M-tiles) ─────────────────
        M_acc = cute.size(tCtAcc, mode=[0, 0])
        print(f"M_acc={M_acc}")

        # Ld32x32bOp: 32 DP lanes × 32b = 1 fp32 col per repetition.
        # M_acc=128 → 128/4=32 DP lanes, so this is the only valid op here.
        ld_op     = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        subtile_n = self.tmem_ld_rep   # 1 fp32 col per rep
        epi_tiler       = ((M_acc, subtile_n),)
        tCtAcc_epi      = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)

        # Epilogue RMEM setup: Repetition(1) → 1 reg/thread = TMEM col 0.
        # Size RMEM from the TMEM source slice (same rank as copy src).
        thr_mma  = tiled_mma.get_slice(thr_idx=0)
        tTR_rAcc = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)
        print(f"tTR_rAcc shape: {tTR_rAcc.shape}")

        num_k_blocks = cute.size(tCrA, mode=[2])

        # ── For loop over M-tiles ([0:128], [128:256]) ────────────────
        # cutlass.range_constexpr forces Python-level unrolling so the loop
        # body is emitted sequentially (no MLIR scf.for, no loop-carried
        # tiled_mma state that would cause a compile-time pickle error).
        tma_phase = 0
        mma_phase = 0

        for m_tile_idx in cutlass.range_constexpr(NUM_M_TILES):   # unrolled: 0, 1
            mma_coord = (m_tile_idx, 0, None)

            gA = cute.local_tile(mA_tma_tensor, CTA_TILE_MNK, mma_coord, proj=(1, None, 1))
            gB = cute.local_tile(mB_tma_tensor, CTA_TILE_MNK, mma_coord, proj=(None, 1, 1))

            tCgA = thr_mma.partition_A(gA)
            tCgB = thr_mma.partition_B(gB)
            
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a, 0, cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b, 0, cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )

            # ─ TMA load (single K-tile: K==BK) ──────────────────────
            if warp_idx == 0:
                cute.copy(tma_atom_a, tAgA[None, 0], tAsA[None, 0], tma_bar_ptr=tma_mbar)
                cute.copy(tma_atom_b, tBgB[None, 0], tBsB[None, 0], tma_bar_ptr=tma_mbar)
                if tidx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_transaction_bytes)

            cute.arch.mbarrier_wait(tma_mbar, tma_phase)
            tma_phase ^= 1

            tcgen05_fence()

            # ─ MMA: ACCUMULATE=False for k_block 0 (resets TMEM),
            #        ACCUMULATE=True  for k_block 1+ (accumulates).
            #   This is the same pattern as fmha.py kphase loop.
            if warp_idx == 0:
                for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_block_idx != 0)
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

            # ─ Epilogue: TMEM col-0 → RMEM (1 reg/thread) → GMEM directly ─
            # Repetition(1): thread t holds TMEM[t, 0] = score[m_base+t].
            # Literal index [0] is static — no scf.for, no dynamic SSA.
            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)
            m_base = m_tile_idx * M_acc
            mC[m_base + tidx, 0] = tTR_rAcc[0]

        # ── TMEM dealloc (after all M-tiles done) ─────────────────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── Host: compile + correctness test ─────────────────────────────────────────
def main():
    # ── Build test tensors ────────────────────────────────────────────────────
    kv    = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    q     = torch.randn((1, K), device="cuda", dtype=torch.bfloat16)

    # Only row 0 matters; rows 1-7 are garbage (we only read c_out[:, 0]).
    q_pad = torch.empty((N, K), device="cuda", dtype=torch.bfloat16)
    q_pad[0] = q[0]

    c_out = torch.zeros((M, 1), device="cuda", dtype=torch.float32)

    kv_    = from_dlpack(kv,    assumed_align=16)
    q_pad_ = from_dlpack(q_pad, assumed_align=16)
    c_out_ = from_dlpack(c_out, assumed_align=16)

    # ── Compile & run ─────────────────────────────────────────────────────────
    gemm     = ScoreGEMM()
    compiled = cute.compile(gemm, kv_, q_pad_, c_out_)
    compiled(kv_, q_pad_, c_out_)

    # ── Extract scores (column 0 = actual query) ──────────────────────────────
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

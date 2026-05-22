"""
score_tcgen05_fp8_direct.py — FP8 tcgen05.mma with direct Ld32x32b TMEM epilogue.

Adapts score_tcgen05_fp8.py's GEMM with the Ld32x32bOp epilogue pattern from
archive/tmem/score_tcgen05_direct_1block.py:

  TMEM_LD_REP = N = 64
  Ld32x32bOp(Repetition(64)) → each thread holds 64 registers after one copy.

  Thread mapping (128-thread CTA, UMMA_M=128):
    tidx (0..127)  → UMMA_M row within the CTA tile
    reg[n] (0..63) → UMMA_N = head n score

  With subtile_n = TMEM_LD_REP = 64 = N_acc → subtile_cnt = 1.
  One cute.copy drains the entire [128 × 64] TMEM accumulator.

GEMM: C[M=2048, N=64] = K_fp8[M,K] @ q_fp8[N,K].T   (fp8 → float32 acc)
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
M = 2048   # KV tokens
N = 64     # query heads (UMMA_N)
K = 128    # head dim

MMA_INST_MNK    = (128, 64, 32)   # tcgen05 FP8 instruction shape
CTA_TILE_MNK    = (128, N, K)     # one CTA: 128 M-rows × all 64 heads × all K

THREADS_PER_CTA = 128
TMEM_LD_REP     = N               # = 64 → Ld32x32b reads all N cols in one shot


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
class ScoreGEMMFP8Direct:
    """
    FP8 tcgen05 GEMM with direct Ld32x32b(Repetition=64) epilogue.

    Each thread in the 128-thread CTA owns one UMMA_M row.
    After a single tmem copy, it holds 64 float32 regs — one per query head.
    Direct write: mC[bidx*128 + tidx, n] = tTR_rAcc[n]  for n in 0..63
    """

    def __init__(self):
        self.BM, self.BN, self.BK   = CTA_TILE_MNK
        self.mma_inst_shape_mnk     = MMA_INST_MNK
        self.threads_per_cta        = THREADS_PER_CTA
        self.num_stages             = 1
        self.tmem_ld_rep            = TMEM_LD_REP   # = N = 64

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        kv:    cute.Tensor,   # A: [M=2048, K=128] float8_e4m3fn  K-major
        q:     cute.Tensor,   # B: [N=64,   K=128] float8_e4m3fn  K-major
        c_out: cute.Tensor,   # output [M=2048, N=64] Float32
    ):
        self.ab_dtype  = kv.element_type   # Float8E4M3FN
        self.c_dtype   = c_out.element_type
        self.acc_dtype = cutlass.Float32
        
        op = tcgen05.MmaFP8Op(
            self.ab_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        print("tiled_mma:", self.tiled_mma)

        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, CTA_TILE_MNK, kv.element_type, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, CTA_TILE_MNK, q.element_type, self.num_stages,
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
            op_g2s, q, b_smem_layout_one_stage, CTA_TILE_MNK, self.tiled_mma,
        )

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        grid_m = kv.shape[0] // self.BM   # = 2048 // 128 = 16
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
            grid=(grid_m, 1, 1),
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
        mB_tma_tensor:   cute.Tensor,   # TMA view of q    [N, K]
        mC:              cute.Tensor,   # output            [M, N]
        a_smem_layout:   cute.ComposedLayout,
        b_smem_layout:   cute.ComposedLayout,
    ):
        tidx, _, _    = cute.arch.thread_idx()
        warp_idx      = cute.arch.warp_idx()
        warp_idx      = cute.arch.make_warp_uniform(warp_idx)
        bidx, _, _    = cute.arch.block_idx()   # M-tile index (0..15)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        # ── SMEM allocation ───────────────────────────────────────────
        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        # ── MMA tensor views ──────────────────────────────────────────
        mma_coord_mnk = (bidx, 0, None)
        gA = cute.local_tile(mA_tma_tensor, CTA_TILE_MNK, mma_coord_mnk, proj=(1, None, 1))
        gB = cute.local_tile(mB_tma_tensor, CTA_TILE_MNK, mma_coord_mnk, proj=(None, 1, 1))

        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA    = thr_mma.partition_A(gA)
        tCgB    = thr_mma.partition_B(gB)

        tCrA    = tiled_mma.make_fragment_A(sA)
        tCrB    = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc    = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)
        print(f"num_tmem_cols: {num_tmem_cols}")

        # ── TMA partitioning ──────────────────────────────────────────
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

        # ── Barriers ─────────────────────────────────────────────────
        tma_mbar = storage.tma_mbar_ptr.data_ptr()
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        tma_transaction_bytes = (
            cute.size_in_bytes(self.ab_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
            + cute.size_in_bytes(self.ab_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
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

        # ── TMEM epilogue setup ───────────────────────────────────────
        # M_acc = 128, num_dp = 128 // 4 = 32 → Ld32x32bOp
        # Repetition(N=64): subtile_n = 64 = N_acc → subtile_cnt = 1
        # One copy call drains all 64 head-columns; each thread gets 64 regs.
        M_acc = cute.size(tCtAcc, mode=[0, 0])
        print(f"M_acc={M_acc}")

        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        subtile_n       = self.tmem_ld_rep   # = N = 64
        epi_tiler       = ((M_acc, subtile_n),)
        tCtAcc_epi      = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)
        print(f"tTR_rAcc shape: {tTR_rAcc.shape}  (expect 64 regs per thread)")

        # ── MMA main loop ─────────────────────────────────────────────
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        tma_phase = 0
        mma_phase = 0

        for kidx in range(mA_tma_tensor.shape[1] // self.BK):
            if warp_idx == 0:
                cute.copy(tma_atom_a, tAgA[None, kidx], tAsA[None, 0], tma_bar_ptr=tma_mbar)
                cute.copy(tma_atom_b, tBgB[None, kidx], tBsB[None, 0], tma_bar_ptr=tma_mbar)
                if tidx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_transaction_bytes)

            cute.arch.mbarrier_wait(tma_mbar, tma_phase)
            tma_phase ^= 1

            tcgen05_fence()

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
            mma_phase ^= 1

        # ── Epilogue: TMEM → 64 regs/thread → direct GMEM write ──────
        # subtile_cnt = 1: single copy covers all 64 N columns.
        # tTR_rAcc[n] = C[bidx*128 + tidx, n]  for n in 0..63
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

        m_base = bidx * M_acc
        for n_idx in cutlass.range_constexpr(N):
            mC[m_base + tidx, n_idx] = tTR_rAcc[n_idx]

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── Host: compile + correctness test ─────────────────────────────────────────
def main():
    device = "cuda"

    K_fp8 = torch.randn(M, K, device=device).to(torch.float8_e4m3fn)
    q_fp8 = torch.randn(N, K, device=device).to(torch.float8_e4m3fn)
    c_out = torch.zeros((M, N), device=device, dtype=torch.float32)

    kv_ = from_dlpack(K_fp8, assumed_align=16)
    q_  = from_dlpack(q_fp8, assumed_align=16)
    c_  = from_dlpack(c_out, assumed_align=16)

    gemm     = ScoreGEMMFP8Direct()
    compiled = cute.compile(gemm, kv_, q_, c_)
    compiled(kv_, q_, c_)

    # Reference: C[M,N] = K_fp8 @ q_fp8.T
    ref_c = K_fp8.float() @ q_fp8.float().T   # [2048, 64]

    atol, rtol = 1.0, 0.5
    match   = torch.allclose(c_out, ref_c, atol=atol, rtol=rtol)
    max_err = (c_out - ref_c).abs().max().item()
    print(f"\nCORRECTNESS {'PASS' if match else 'FAIL'}  (max_err={max_err:.4f})")
    if not match:
        print("  c_out[0,:8]:", c_out[0, :8].tolist())
        print("  ref_c[0,:8]:", ref_c[0, :8].tolist())

    t = benchmark(compiled, kernel_arguments=JitArguments(kv_, q_, c_))
    print(f"DURATION: {t:.4f} µs")


if __name__ == "__main__":
    main()

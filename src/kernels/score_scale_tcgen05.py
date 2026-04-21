"""score_scale_tcgen05.py — tcgen05 fp8 GEMM using autovec_copy for G→S loads.

Same GEMM as score_tcgen05_fp8.py but replaces TMA bulk-copy with
cute.autovec_copy (thread-parallel G→S).

  C[M, N] = A[M, K] × B[N, K].T   (fp8 → float32 accumulation)

  A = kv  [2048, 128] Float8E4M3FN  — KV tokens (K-major)
  B = q   [  64, 128] Float8E4M3FN  — query heads (K-major)
  C       [2048,  64] Float32       — attention scores

MMA instruction: tcgen05, (M_inst=128, N_inst=64, K_inst=32)
  → 16 M-tiles, 1 N-tile, 4 K-inst blocks per tile (K=128 = 4×32)
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass.cute.testing import benchmark, JitArguments

# ── Dimensions ────────────────────────────────────────────────────────────────
M = 2048
N = 64
K = 128

MMA_INST_MNK    = (128, 64, 32)
CTA_TILE_MNK    = (128, N, K)    # 1 M-tile = 128 rows, all N heads, all K

THREADS_PER_CTA = 128


# ── tcgen05.fence::after_thread_sync ─────────────────────────────────────────
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
class ScoreGEMMFP8AutoVec:
    """
    tcgen05 FP8 GEMM: C[2048, 64] = kv[2048, 128] × q[64, 128].T

    G→S load via cute.autovec_copy (all THREADS_PER_CTA threads participate).
    MMA and epilogue (TMEM→RMEM→GMEM) identical to TMA version.
    """

    def __init__(self):
        self.num_stages  = 1
        self.tmem_ld_rep = N   # all N cols fit in one TMEM load

    @cute.jit
    def __call__(
        self,
        kv:    cute.Tensor,   # [M=2048, K=128] Float8E4M3FN  K-major
        q:     cute.Tensor,   # [N=64,   K=128] Float8E4M3FN  K-major
        c_out: cute.Tensor,   # [M=2048, N=64]  Float32
    ):
        self.ab_dtype  = kv.element_type
        self.c_dtype   = c_out.element_type
        self.acc_dtype = cutlass.Float32

        op = tcgen05.MmaFP8Op(
            self.ab_dtype,
            self.acc_dtype,
            MMA_INST_MNK,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)

        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, CTA_TILE_MNK, kv.element_type, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, CTA_TILE_MNK, q.element_type, self.num_stages,
        )

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        grid_m = kv.shape[0] // CTA_TILE_MNK[0]   # 2048 // 128 = 16
        self.kernel(
            self.tiled_mma,
            kv,
            q,
            c_out,
            self.a_smem_layout,
            self.b_smem_layout,
        ).launch(
            grid=(grid_m, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma:     cute.TiledMma,
        mA:            cute.Tensor,          # [M, K] fp8
        mB:            cute.Tensor,          # [N, K] fp8
        mC:            cute.Tensor,          # [M, N] float32
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _    = cute.arch.thread_idx()
        warp_idx      = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _    = cute.arch.block_idx()   # M-tile index

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

        # ── MMA tensor fragments ───────────────────────────────────────
        thr_mma = tiled_mma.get_slice(thr_idx=0)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc    = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols  = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        # ── TMEM alloc + barrier init ─────────────────────────────────
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        tmem_barrier_id = 1
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # ── autovec G→S: flat 2D views, no K-tiling (rank matches SMEM) ─
        thr_layout = cute.make_layout(THREADS_PER_CTA)

        # A: [CTA_M, K] block at row offset bidx*CTA_M
        gA_2d    = cute.make_tensor(mA.iterator,
                                    cute.make_layout((mA.shape[0], K), stride=(K, 1)))
        gA_tile  = cute.local_tile(gA_2d, (CTA_TILE_MNK[0], K), (bidx, 0))
        tCgA     = thr_mma.partition_A(gA_tile)
        sA_thr   = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr   = cute.local_partition(tCgA, thr_layout, tidx)
        cute.autovec_copy(gA_thr, sA_thr)

        # B: full [N, K] view (no K-tiling)
        gB_2d    = cute.make_tensor(mB.iterator,
                                    cute.make_layout((N, K), stride=(K, 1)))
        tCgB     = thr_mma.partition_B(gB_2d)
        sB_thr   = cute.local_partition(sB[None, None, None, 0], thr_layout, tidx)
        gB_thr   = cute.local_partition(tCgB, thr_layout, tidx)
        cute.autovec_copy(gB_thr, sB_thr)

        cute.arch.sync_threads()

        # ── Epilogue setup: TMEM → RMEM ──────────────────────────────
        M_acc = cute.size(tCtAcc, mode=[0, 0])

        ld_op          = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler      = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi     = cute.zipped_divide(tCtAcc, epi_tiler)

        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        # ── tcgen05 MMA (warp 0) ──────────────────────────────────────
        tcgen05_fence()

        num_k_blocks = cute.size(tCrA, mode=[2])

        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
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

        cute.arch.mbarrier_wait(mma_mbar, 0)

        # ── Epilogue: TMEM → RMEM → GMEM (128 threads, direct write) ─
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

        m_out = bidx * cutlass.Int32(CTA_TILE_MNK[0]) + tidx
        for n_idx in cutlass.range_constexpr(N):
            mC[m_out, n_idx] = tTR_rAcc[n_idx].to(self.c_dtype)

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── Python wrapper ────────────────────────────────────────────────────────────
def run_gemm(
    q_fp8: torch.Tensor,   # [64, 128]   Float8E4M3FN
    kv_fp8: torch.Tensor,  # [2048, 128] Float8E4M3FN
    c_out: torch.Tensor,   # [2048, 64]  Float32  (pre-allocated)
):
    c_out.zero_()
    kv_ = from_dlpack(kv_fp8, assumed_align=16)
    q_  = from_dlpack(q_fp8, assumed_align=16)
    c_  = from_dlpack(c_out, assumed_align=16)

    gemm = ScoreGEMMFP8AutoVec()
    compiled = cute.compile(gemm, kv_, q_, c_)
    compiled(kv_, q_, c_)


# ── Host: correctness test ────────────────────────────────────────────────────
def main():
    device = "cuda"

    kv_fp8 = torch.randn(M, K, device=device).clamp(-240, 240).to(torch.float8_e4m3fn)
    q_fp8  = torch.randn(N, K, device=device).clamp(-240, 240).to(torch.float8_e4m3fn)
    c_out  = torch.zeros(M, N, device=device, dtype=torch.float32)

    kv_ = from_dlpack(kv_fp8, assumed_align=16)
    q_  = from_dlpack(q_fp8,  assumed_align=16)
    c_  = from_dlpack(c_out,  assumed_align=16)

    gemm     = ScoreGEMMFP8AutoVec()
    compiled = cute.compile(gemm, kv_, q_, c_)
    compiled(kv_, q_, c_)

    ref_c   = kv_fp8.float() @ q_fp8.float().T   # [2048, 64]
    max_err = (c_out - ref_c).abs().max().item()
    match   = max_err < 1.0

    print(f"\nCORRECTNESS {'PASS' if match else 'FAIL'}  (max_err={max_err:.4f})")
    # if not match:
    print("  c_out[0,:8]:", c_out[0, :8].tolist())
    print("  ref_c[0,:8]:", ref_c[0, :8].tolist())

    t = benchmark(compiled, kernel_arguments=JitArguments(kv_, q_, c_))
    print(f"DURATION: {t:.4f} µs")


if __name__ == "__main__":
    main()

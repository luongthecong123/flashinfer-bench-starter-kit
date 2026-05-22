"""
score_scale.py — FP8 tcgen05.mma with per-token scale applied in epilogue.

Input layout: kv_raw [M, 132] uint8 (real format, no padding)
  Each row = 128 bytes fp8 (float8_e4m3fn) + 4 bytes float32 scale.

Strategy: cooperative GMEM→SMEM via partition_A + local_partition + autovec_copy (no TMA for A).
  kv_fp8   = make_tensor(recast_ptr(kv_raw.iterator, Float8E4M3FN), layout=(M,128), stride=(132,1))
  k_scales_packed = make_tensor(recast_ptr(kv_raw.iterator, Float32), layout=(M,33), stride=(33,1))
    → scale for token m = k_scales_packed[m, 32]   (col 32 = byte offset 128)

A (kv_fp8): cooperative gmem→smem copy using thr_mma.partition_A + local_partition + autovec_copy.
  ROW_STRIDE=132 is fine for simple loads (only TMA required 16-byte stride alignment).
B (q_fp8): TMA as before (q is contiguous, stride=128 ✓).
Scales: 128 float32 in SMEM (sScales[128]), one per thread, loaded before MMA.

GEMM: C[M=2048, N=64] = kv_fp8[M,128] @ q_fp8[64,128].T  (fp8 → float32 acc)

Epilogue (in kernel, after Ld32x32b into 64 regs/thread):
  m = bidx * 128 + tidx
  mC[m, n] = tTR_rAcc[n] * sScales[tidx]   for n in 0..63
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
M        = 2048   # KV tokens
N        = 64     # query heads (UMMA_N)
HEAD_DIM = 128    # fp8 head dim (K)
ROW_STRIDE = 132  # bytes per kv row: 128 fp8 + 4 scale (real format, no padding)

MMA_INST_MNK    = (128, 64, 32)
CTA_TILE_MNK    = (128, N, HEAD_DIM)

THREADS_PER_CTA = 128
TMEM_LD_REP     = N    # = 64 → Ld32x32b(rep=64) reads all N cols in one shot


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
class ScoreScale:
    """
    FP8 tcgen05 GEMM + per-token scale in epilogue.

    kv_raw  [M, 132] uint8   — packed fp8 + scale (real format)
    q_fp8   [N,  128] float8_e4m3fn
    c_out   [M,   64] float32

    A loaded via cooperative autovec_copy (no TMA); B via TMA.
    Scales: 128×float32 SMEM buffer (sScales), loaded before MMA.
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
        kv_raw: cute.Tensor,   # [M, 132] UInt8 — packed fp8 + scale
        q:      cute.Tensor,   # [N, 128] Float8E4M3FN
        c_out:  cute.Tensor,   # [M,  64] Float32
    ):
        self.fp8_dtype  = cutlass.Float8E4M3FN
        self.c_dtype    = c_out.element_type
        self.acc_dtype  = cutlass.Float32

        # ── fp8 view: [M, HEAD_DIM] fp8, row stride = ROW_STRIDE = 132 ─────
        # Simple recast: uint8→fp8 (1:1), stride (132, 1) in fp8 units.
        # ROW_STRIDE=132 is NOT TMA-alignable (not multiple of 16 × elem_size for
        # 128-bit swizzle), so we use cooperative autovec_copy instead of TMA for A.
        kv_fp8 = cute.make_tensor(
            cute.recast_ptr(kv_raw.iterator, dtype=cutlass.Float8E4M3FN),
            cute.make_layout((M, HEAD_DIM), stride=(ROW_STRIDE, 1)),
        )

        # ── Scale view: [M, 33] float32; scale for row m at col 32 ───────────
        # ROW_STRIDE // 4 = 33 float32 per row; col 32 × 4 bytes = byte offset 128. ✓
        k_scales_packed = cute.make_tensor(
            cute.recast_ptr(kv_raw.iterator, dtype=cutlass.Float32),
            cute.make_layout((M, ROW_STRIDE // 4), stride=(ROW_STRIDE // 4, 1)),
        )

        # ── MMA + SMEM layouts ────────────────────────────────────────
        op = tcgen05.MmaFP8Op(
            self.fp8_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)

        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, CTA_TILE_MNK, self.fp8_dtype, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, CTA_TILE_MNK, q.element_type, self.num_stages,
        )

        # ── TMA for B only (q is contiguous, stride=128 ✓) ───────────
        b_smem_layout_one_stage = cute.select(self.b_smem_layout, mode=[0, 1, 2])
        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, q, b_smem_layout_one_stage, CTA_TILE_MNK, self.tiled_mma,
        )

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        grid_m = M // self.BM   # = 16
        self.kernel(
            self.tiled_mma,
            kv_fp8,
            tma_atom_b,
            tma_tensor_b,
            k_scales_packed,
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
        mA_fp8:          cute.Tensor,   # GMEM kv_fp8 [M, HEAD_DIM] fp8, stride (132,1)
        tma_atom_b:      cute.CopyAtom,
        mB_tma_tensor:   cute.Tensor,   # TMA view of q [N, HEAD_DIM]
        k_scales_packed: cute.Tensor,   # [M, 33] float32 — scale at col 32
        mC:              cute.Tensor,   # output [M, N] float32
        a_smem_layout:   cute.ComposedLayout,
        b_smem_layout:   cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)
        bidx, _, _  = cute.arch.block_idx()   # M-tile index (0..15)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_b)

        # ── SMEM allocation ───────────────────────────────────────────
        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = smem.allocate_tensor(
            element_type=self.fp8_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.fp8_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )
        # 128 float32 scale values, one per thread
        sScales = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(self.threads_per_cta),
            byte_alignment=16,
            swizzle=None,
        )

        # ── MMA tensor views ──────────────────────────────────────────
        m_base        = bidx * self.BM
        mma_coord_mnk = (bidx, 0, None)
        gB = cute.local_tile(mB_tma_tensor, CTA_TILE_MNK, mma_coord_mnk, proj=(None, 1, 1))

        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgB    = thr_mma.partition_B(gB)

        tCrA    = tiled_mma.make_fragment_A(sA)
        tCrB    = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc    = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        # ── TMA partition for B only ──────────────────────────────────
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # ── Barriers ─────────────────────────────────────────────────
        tma_mbar = storage.tma_mbar_ptr.data_ptr()
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        # Only B goes through TMA now
        tma_transaction_bytes = cute.size_in_bytes(
            self.fp8_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
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
        M_acc = cute.size(tCtAcc, mode=[0, 0])   # = 128 = BM

        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        subtile_n       = self.tmem_ld_rep         # = 64
        epi_tiler       = ((M_acc, subtile_n),)
        tCtAcc_epi      = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        # ── Load A upfront: cooperative GMEM→SMEM ─────────────────────
        # A has stride (ROW_STRIDE=132, 1) in fp8 — fine for simple ld, not for TMA.
        # Use thr_mma.partition_A + local_partition + autovec_copy (same as
        # archive/tmem/score_tcgen05_direct_1block.py).
        gA_local   = cute.local_tile(mA_fp8, CTA_TILE_MNK, mma_coord_mnk, proj=(1, None, 1))
        tCgA       = thr_mma.partition_A(gA_local)
        thr_layout = cute.make_layout(self.threads_per_cta)
        sA_thr     = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr     = cute.local_partition(tCgA[None, None, None, 0], thr_layout, tidx)
        cute.autovec_copy(gA_thr, sA_thr)

        # ── Load scale for this thread into SMEM ──────────────────────
        sScales[tidx] = k_scales_packed[m_base + tidx, HEAD_DIM // 4]

        # ── MMA main loop ─────────────────────────────────────────────
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        tma_phase = 0
        mma_phase = 0

        cute.arch.sync_threads()   # wait for A copy and scale load to complete

        for kidx in range(HEAD_DIM // self.BK):
            if warp_idx == 0:
                cute.copy(tma_atom_b, tBgB[None, kidx], tBsB[None, 0], tma_bar_ptr=tma_mbar)
                if tidx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_transaction_bytes)

            cute.arch.mbarrier_wait(tma_mbar, tma_phase)
            tma_phase ^= 1

            tcgen05_fence()   # order sA (sync_threads) and sB (mbarrier_wait) for MMA

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

        # ── Epilogue: TMEM → 64 regs → scale → GMEM ──────────────────
        # Each thread holds 1 token row; scale pre-loaded in sScales[tidx].
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

        scale = sScales[tidx]
        for n_idx in cutlass.range_constexpr(N):
            mC[m_base + tidx, n_idx] = tTR_rAcc[n_idx] * scale

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── Host: compile + correctness test ─────────────────────────────────────────
def main():
    device = "cuda"

    # Clamp to fp8 safe range (±100 << ±448) to avoid 0x7F NaN pattern
    K_fp8    = torch.randn(M, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
    K_scales = torch.rand(M, device=device, dtype=torch.float32) + 0.5
    q_fp8    = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)

    # Pack into kv_raw [M, 132] uint8 (real format: 128 fp8 + 4 scale bytes)
    kv_raw = torch.zeros(M, ROW_STRIDE, device=device, dtype=torch.uint8)
    kv_raw[:, :HEAD_DIM] = K_fp8.view(torch.uint8)
    kv_raw[:, HEAD_DIM:HEAD_DIM + 4] = K_scales.view(torch.uint8).reshape(M, 4)

    c_out = torch.zeros((M, N), device=device, dtype=torch.float32)

    kv_raw_ = from_dlpack(kv_raw, assumed_align=16)
    q_      = from_dlpack(q_fp8,  assumed_align=16)
    c_      = from_dlpack(c_out,  assumed_align=16)

    # ── Compile & run ─────────────────────────────────────────────────────────
    kernel   = ScoreScale()
    compiled = cute.compile(kernel, kv_raw_, q_, c_)
    compiled(kv_raw_, q_, c_)

    # ── Reference: C[m,n] = dot(K_fp8[m], q_fp8[n]) * K_scales[m] ───────────
    ref_c = (K_fp8.float() @ q_fp8.float().T) * K_scales[:, None]   # [M, N]

    atol, rtol = 1.0, 0.5
    match   = torch.allclose(c_out, ref_c, atol=atol, rtol=rtol)
    max_err = (c_out - ref_c).abs().max().item()
    print(f"\nCORRECTNESS {'PASS' if match else 'FAIL'}  (max_err={max_err:.4f})")
    if not match:
        print("  c_out[0,:8]:", c_out[0, :8].tolist())
        print("  ref_c[0,:8]:", ref_c[0, :8].tolist())

    t = benchmark(compiled, kernel_arguments=JitArguments(kv_raw_, q_, c_))
    print(f"DURATION: {t:.4f} µs")


if __name__ == "__main__":
    main()

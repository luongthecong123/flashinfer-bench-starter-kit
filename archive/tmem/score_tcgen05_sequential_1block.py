"""
score_tcgen05_sequential_1block.py — Single-block, per-row TMA via inline PTX.

Fires 128 individual cp.async.bulk.tensor.2d TMAs (one per KV row) using a
dynamic cutlass.range(128) loop + inline PTX.  The TMA descriptor is copied to
SMEM so the PTX instruction can use the sm_100a .shared::cta tensormap path.
B uses a single standard TMA per BK-tile (no IR explosion).

This is the stepping stone toward gather-based sparse attention: changing the
per-row GMEM coordinate is all that is needed to gather from arbitrary rows.

Key design:
  - TMA A: cpasync.make_tiled_tma_atom with (1, BK) box, swizzled SMEM layout.
           Descriptor copied to SMEM.  128 TMAs fired in a dynamic for-loop
           via tma_g2s_2d dsl_user_op (inline PTX).
  - TMA B: cpasync.make_tiled_tma_atom with (N, BK) box.
           1 TMA per BK-step via standard cute.copy (no unroll explosion).
  - SMEM:  Swizzled (S<3,4,3>), required by tcgen05 MMA SMEM descriptors.
  - MMA:   Standard SS-mode tcgen05 (128,8,16), both operands from SMEM.

GEMM:
  C[M=256, N=8] = A[M, K] × B[N, K].T
  scores[256]   = C[:, 0]
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm, arith
from cutlass._mlir import ir as mlir_ir
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.testing import benchmark, JitArguments

# ── Dimensions ───────────────────────────────────────────────────────────────
M = 256
N = 8
K = 512
BK = 64                      # K-tile per stage

MMA_INST_MNK    = (128, 8, 16)
CTA_TILE_MNK    = (128, N, BK)
NUM_M_TILES     = M // 128            # 2
NUM_BK_TILES    = K // BK             # 8
NUM_ROWS        = CTA_TILE_MNK[0]     # 128
ROW_BYTES       = BK * 2              # bytes per row of BF16 = 128

THREADS_PER_CTA = 128
TMEM_LD_REP     = 1
TMEM_ALLOC_COLS = 32


# ── Helpers ───────────────────────────────────────────────────────────────────
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


@dsl_user_op
def tma_g2s_2d(
    desc_smem_ptr,     # Pointer — SMEM-resident TMA descriptor
    dest_smem_base,    # Pointer — sA base address in SMEM
    row,               # dynamic Int — row index 0..127
    row_bytes_const,   # Python int — BK * 2 (compile-time)
    gmem_m_base,       # Python int — m_tile * NUM_ROWS (compile-time)
    gmem_k_offset,     # Python int — bk_tile * BK (compile-time)
    mbar_ptr,          # Pointer — mbarrier in SMEM
    *, loc=None, ip=None,
):
    """Fire a single cp.async.bulk.tensor.2d G2S for a (1, BK) box via PTX."""
    i32_ty = mlir_ir.IntegerType.get_signless(32)

    desc  = desc_smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    base  = dest_smem_base.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    row_v = row.ir_value(loc=loc, ip=ip)
    mbar  = mbar_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)

    # dest = base + row * row_bytes
    rb  = arith.ConstantOp(i32_ty, row_bytes_const, loc=loc, ip=ip).result
    off = arith.MulIOp(row_v, rb, loc=loc, ip=ip).result
    dst = arith.AddIOp(base, off, loc=loc, ip=ip).result

    # GMEM coordinates: (m_base + row, k_offset)
    mb  = arith.ConstantOp(i32_ty, gmem_m_base, loc=loc, ip=ip).result
    c0  = arith.AddIOp(mb, row_v, loc=loc, ip=ip).result
    c1  = arith.ConstantOp(i32_ty, gmem_k_offset, loc=loc, ip=ip).result

    llvm.inline_asm(
        None,
        [dst, desc, c0, c1, mbar],
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes [$0], [$1], {$2, $3}, [$4];",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )


# ── Kernel class ──────────────────────────────────────────────────────────────
class ScoreGEMM_Sequential_1Block:
    """
    Single-CTA tcgen05 GEMM with 128 per-row TMAs for A (via inline PTX)
    and 1 standard TMA per BK-step for B.
    Output: scores[M=256] written to c_out[M, 1].
    """

    def __init__(self):
        self.BM, self.BN, self.BK = CTA_TILE_MNK
        self.mma_inst_shape_mnk   = MMA_INST_MNK
        self.threads_per_cta      = THREADS_PER_CTA
        self.num_stages           = 1
        self.tmem_ld_rep          = TMEM_LD_REP

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

        # ── SS-mode MMA (both operands from SMEM) ────────────────────
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

        # ── SMEM layouts (with swizzle, needed by MMA descriptors) ────
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, CTA_TILE_MNK, kv.element_type, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, CTA_TILE_MNK, q_pad.element_type, self.num_stages,
        )
        print("a_smem_layout:", self.a_smem_layout)
        print("b_smem_layout:", self.b_smem_layout)

        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        # ── Per-row TMA for A: (1, BK) box with swizzle ─────────────
        row_smem_a = cute.make_composed_layout(
            self.a_smem_layout.inner,      # Swizzle S<3,4,3>
            0,
            cute.make_layout((1, BK), stride=(BK, 1)),
        )
        tma_atom_a, _ = cpasync.make_tiled_tma_atom(
            op_g2s, kv, row_smem_a, (1, BK),
        )

        # ── Full-tile TMA for B: (N, BK) box with swizzle ───────────
        box_smem_b = cute.make_composed_layout(
            self.b_smem_layout.inner,
            0,
            cute.make_layout((N, BK), stride=(BK, 1)),
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            op_g2s, q_pad, box_smem_b, (N, BK),
        )

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_atom_b, tma_tensor_b,
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
        tma_atom_a:    cute.CopyAtom,
        tma_atom_b:    cute.CopyAtom,
        tma_tensor_b:  cute.Tensor,
        mC:            cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)

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
        # 128 bytes for A TMA descriptor in SMEM (sm_100a allows .shared::cta)
        desc_a_smem = smem.allocate_tensor(
            element_type=cutlass.Uint8,
            layout=cute.make_layout(128),
            byte_alignment=128,
        )

        print("sA:", sA)
        print("sB:", sB)

        # ── MMA SMEM descriptors ──────────────────────────────────────
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # ── TMEM allocation ───────────────────────────────────────────
        acc_shape       = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc          = tiled_mma.make_fragment_C(acc_shape)
        tmem_alloc_cols = cutlass.Int32(TMEM_ALLOC_COLS)

        tma_mbar = storage.tma_mbar_ptr.data_ptr()
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        # transaction bytes per step: 128 rows × BK × 2  +  N × BK × 2
        tma_transaction_bytes = NUM_ROWS * BK * 2 + N * BK * 2

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            # Copy A TMA descriptor to SMEM
            if tidx == 0:
                cpasync.copy_tensormap(tma_atom_a, desc_a_smem.iterator)
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

        # ── Epilogue atoms (invariant) ────────────────────────────────
        M_acc = cute.size(tCtAcc, mode=[0, 0])

        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        subtile_n       = self.tmem_ld_rep
        epi_tiler       = ((M_acc, subtile_n),)
        tCtAcc_epi      = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        # ── B TMA: single-element SMEM view for tma_partition ─────────
        sB_for_tma = cute.make_tensor(sB.iterator, cute.make_layout(1))

        # ── Prefetch B descriptor (A is already in SMEM) ──────────────
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_b)

        num_k_blocks = cute.size(tCrA, mode=[2])

        tma_phase = 0
        mma_phase = 0

        # ── Main loop: M-tiles × BK-tiles ────────────────────────────
        for m_tile in cutlass.range_constexpr(NUM_M_TILES):
            for bk_tile in cutlass.range_constexpr(NUM_BK_TILES):

                # ─ B: full-tile TMA (1 copy per step) ────────────────
                gB = cute.local_tile(tma_tensor_b, (1, 1), (0, bk_tile))
                gB_grouped = cute.group_modes(gB, 0, 2)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b, 0, cute.make_layout(1),
                    sB_for_tma, gB_grouped,
                )

                if warp_idx == 0:
                    # ─ A: 128 per-row TMAs in dynamic loop ───────────
                    if tidx == 0:
                        for row in cutlass.range(NUM_ROWS):
                            tma_g2s_2d(
                                desc_a_smem.iterator,   # SMEM descriptor
                                sA.iterator,            # sA base
                                row,                    # dynamic row 0..127
                                ROW_BYTES,              # const: BK*2
                                m_tile * NUM_ROWS,      # const: GMEM m-base
                                bk_tile * BK,           # const: GMEM k-offset
                                tma_mbar,               # mbarrier
                            )

                    # ─ B: 1 full-tile TMA ─────────────────────────────
                    cute.copy(tma_atom_b, tBgB, tBsB,
                              tma_bar_ptr=tma_mbar)
                    if tidx == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            tma_mbar, tma_transaction_bytes)

                cute.arch.mbarrier_wait(tma_mbar, tma_phase)
                tma_phase ^= 1

                tcgen05_fence()

                # ─ MMA over K-blocks within BK tile ───────────────────
                if warp_idx == 0:
                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        if bk_tile == 0:
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

            # ─ Epilogue: TMEM col-0 → RMEM → GMEM ────────────────────
            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)
            m_base = m_tile * M_acc
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

    gemm     = ScoreGEMM_Sequential_1Block()
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

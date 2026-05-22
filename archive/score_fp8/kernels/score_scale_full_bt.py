"""
score_scale_full_bt.py — score_scale_full_page.py extended with block_table.

Inputs:
  kv_pool      [NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE] uint8 — global KV cache
  block_table  [NUM_PG] int32 — page indices for ONE request (single-request kernel)
  q            [N, HEAD_DIM] fp8
  w            [N] fp32
  c_out        [M] fp32 where M = NUM_PG * PAGE_SIZE

Each CTA covers BM=128 rows = 2 pages. For tile bidx, it loads pages
block_table[2*bidx] and block_table[2*bidx+1] from kv_pool. These pages
are non-contiguous, so the per-CTA gA view uses a dynamic stride between
the two pages: ((PAGE_SIZE, 2), HEAD_DIM) : ((ROW_STRIDE, jump), 1).

Assumes NUM_PG is even — host rounds odd up and pads block_table; the extra
garbage rows are filtered by the seq_len mask in the topk stage downstream.
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

# ── Fixed dimensions ─────────────────────────────────────────────────────────
PAGE_SIZE  = 64
N          = 64
HEAD_DIM   = 128
ROW_STRIDE = 132            # bytes per kv row: 128 fp8 + 4 scale
PAGES_PER_TILE = 2          # BM = PAGE_SIZE * PAGES_PER_TILE = 128

MMA_INST_MNK = (128, 64, 32)
BM, BN, BK   = 128, N, HEAD_DIM

THREADS_PER_CTA = 128
TMEM_LD_REP     = N         # = 64 → Ld32x32b reads all N cols in one shot


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
class ScoreScaleFullBT:
    """
    FP8 tcgen05 GEMM with block_table indirection.

    NUM_PG is fixed at construction time (compile per shape).
    """

    def __init__(self, num_pg: int):
        assert num_pg % 2 == 0, "NUM_PG must be even (host should pad)"
        self.num_pg          = num_pg
        self.M               = num_pg * PAGE_SIZE
        self.grid_m          = num_pg // PAGES_PER_TILE   # = M // BM
        self.threads_per_cta = THREADS_PER_CTA
        self.num_stages      = 1
        self.tmem_ld_rep     = TMEM_LD_REP
        self.cta_tile_mnk    = (BM, BN, BK)
        self.mma_inst_shape_mnk = MMA_INST_MNK

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        kv_pool:     cute.Tensor,   # [NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE] UInt8
        block_table: cute.Tensor,   # [NUM_PG] Int32
        q:           cute.Tensor,   # [N, HEAD_DIM] Float8E4M3FN
        w:           cute.Tensor,   # [N] Float32
        c_out:       cute.Tensor,   # [M] Float32
    ):
        self.fp8_dtype  = cutlass.Float8E4M3FN
        self.acc_dtype  = cutlass.Float32

        # ── MMA op ───────────────────────────────────────────────────
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
            self.tiled_mma, self.cta_tile_mnk, self.fp8_dtype, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.cta_tile_mnk, q.element_type, self.num_stages,
        )

        # ── TMA for B (q is contiguous [N, HEAD_DIM] fp8) ────────────
        b_smem_layout_one_stage = cute.select(self.b_smem_layout, mode=[0, 1, 2])
        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, q, b_smem_layout_one_stage, self.cta_tile_mnk, self.tiled_mma,
        )

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma,
            kv_pool,
            block_table,
            tma_atom_b,
            tma_tensor_b,
            w,
            c_out,
            self.a_smem_layout,
            self.b_smem_layout,
        ).launch(
            grid=(self.grid_m, 1, 1),
            block=(self.threads_per_cta, 1, 1),
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma:       cute.TiledMma,
        kv_pool:         cute.Tensor,   # [NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE] UInt8
        block_table:     cute.Tensor,   # [NUM_PG] Int32
        tma_atom_b:      cute.CopyAtom,
        mB_tma_tensor:   cute.Tensor,   # TMA view of q  [N, HEAD_DIM]
        w:               cute.Tensor,   # [N] float32
        mC:              cute.Tensor,   # [M] float32
        a_smem_layout:   cute.ComposedLayout,
        b_smem_layout:   cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)
        bidx, _, _  = cute.arch.block_idx()

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
        sScales = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(self.threads_per_cta),
            byte_alignment=16,
            swizzle=None,
        )
        sWeights = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(N),
            byte_alignment=16,
            swizzle=None,
        )

        # ── Per-CTA page indices and dynamic strides ─────────────────
        # Each tile = 2 pages: page0 and page1 from block_table.
        page0_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 0])
        page1_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 1])

        # Byte offsets into kv_pool (uint8). Page stride = PAGE_SIZE * ROW_STRIDE.
        page_stride_bytes = PAGE_SIZE * ROW_STRIDE
        page0_off_b = page0_id * page_stride_bytes
        jump_b      = (page1_id - page0_id) * page_stride_bytes

        # fp8 element = 1 byte → element stride = byte stride
        fp8_base = cute.recast_ptr(kv_pool.iterator, dtype=self.fp8_dtype) + page0_off_b

        # gA view shape: ((PAGE_SIZE, 2 pages), HEAD_DIM) flattens to (BM=128, HEAD_DIM)
        # Stride: rows within a page = ROW_STRIDE, between pages = jump_b, K = 1
        gA_layout = cute.make_layout(
            ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM),
            stride=((ROW_STRIDE, jump_b), 1),
        )
        gA = cute.make_tensor(fp8_base, gA_layout)

        # ── MMA tensor views ──────────────────────────────────────────
        mma_coord_mnk = (0, 0, None)   # no M tiling here — gA already CTA-local
        gB = cute.local_tile(mB_tma_tensor, self.cta_tile_mnk, (bidx, 0, None), proj=(None, 1, 1))

        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA    = thr_mma.partition_A(gA)
        tCgB    = thr_mma.partition_B(gB)

        tCrA    = tiled_mma.make_fragment_A(sA)
        tCrB    = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(self.cta_tile_mnk[:2])
        tCtAcc    = tiled_mma.make_fragment_C(acc_shape)

        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        # ── TMA partition for B ───────────────────────────────────────
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tma_mbar = storage.tma_mbar_ptr.data_ptr()
        mma_mbar = storage.mma_mbar_ptr.data_ptr()

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
        M_acc = cute.size(tCtAcc, mode=[0, 0])  # = 128 = BM

        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        subtile_n       = self.tmem_ld_rep
        epi_tiler       = ((M_acc, subtile_n),)
        tCtAcc_epi      = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        # ── Cooperative GMEM→SMEM for A (uses dynamic-stride gA) ─────
        # gA was built as 2D (BM, BK), so tCgA has only 3 modes (atom, 1_M, K_reps).
        # sA is 4D (..., stages); we still index stage=0.
        thr_layout = cute.make_layout(self.threads_per_cta)
        sA_thr     = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr     = cute.local_partition(tCgA, thr_layout, tidx)
        cute.autovec_copy(gA_thr, sA_thr)

        # ── Per-token scale: gather from kv_pool via the 2-page layout ─
        # Scales live at byte offset 128 within each row (4 bytes = 1 fp32).
        # View kv_pool as fp32 with stride (33, ...) per row, advance by 32 fp32 = 128 bytes.
        SCALE_ROW_STRIDE_F32 = ROW_STRIDE // 4        # = 33
        page_stride_f32      = PAGE_SIZE * SCALE_ROW_STRIDE_F32
        page0_off_f32        = page0_id * page_stride_f32
        jump_f32             = (page1_id - page0_id) * page_stride_f32

        fp32_base = cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Float32) + page0_off_f32
        # Scale column lives at fp32 col 32 (= byte 128 from row start)
        scale_ptr = fp32_base + (HEAD_DIM // 4)
        scale_layout = cute.make_layout(
            ((PAGE_SIZE, PAGES_PER_TILE),),
            stride=((SCALE_ROW_STRIDE_F32, jump_f32),),
        )
        gScale = cute.make_tensor(scale_ptr, scale_layout)
        # Each thread loads its own scale (tidx flat-indexes into the 2-page layout)
        sScales[tidx] = gScale[tidx]

        # Reduction weights (threads 0..N-1 each load one)
        if tidx < N:
            sWeights[tidx] = w[tidx]

        cute.arch.sync_threads()

        # ── MMA main loop ─────────────────────────────────────────────
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        tma_phase = 0
        mma_phase = 0

        for kidx in range(HEAD_DIM // BK):
            if warp_idx == 0:
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

        # ── Epilogue: TMEM→regs → max(·,0) → weighted sum → GMEM ─────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

        scale    = sScales[tidx]
        out_val  = cutlass.Float32(0)
        for n_idx in cutlass.range_constexpr(N):
            val      = tTR_rAcc[n_idx] * scale
            out_val  = out_val + max(val, cutlass.Float32(0)) * sWeights[n_idx]

        # Output: c_out is [M = NUM_PG * PAGE_SIZE], CTA writes 128 contiguous rows
        m_out = bidx * BM + tidx
        mC[m_out] = out_val

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── Host: build packed kv_pool from per-page contributions ───────────────────
def pack_kv_pool(K_fp8_per_page, K_scales_per_page, num_pages_pool, device):
    """
    K_fp8_per_page:    [num_pages_used, PAGE_SIZE, HEAD_DIM] float8_e4m3fn
    K_scales_per_page: [num_pages_used, PAGE_SIZE]            float32
    Returns: kv_pool [num_pages_pool, PAGE_SIZE, ROW_STRIDE] uint8 with the
    used pages packed (the rest is garbage zeros).
    """
    kv_pool = torch.zeros(num_pages_pool, PAGE_SIZE, ROW_STRIDE, device=device, dtype=torch.uint8)
    n_used  = K_fp8_per_page.shape[0]
    kv_pool[:n_used, :, :HEAD_DIM] = K_fp8_per_page.view(torch.uint8)
    kv_pool[:n_used, :, HEAD_DIM:HEAD_DIM + 4] = (
        K_scales_per_page.view(torch.uint8).reshape(n_used, PAGE_SIZE, 4)
    )
    return kv_pool


def main():
    """Quick local sanity test (no Modal): num_pg=4 (M=256), random pages."""
    device = "cuda"
    torch.manual_seed(0)

    NUM_PG = 4
    M = NUM_PG * PAGE_SIZE
    NUM_PAGES_POOL = 32

    # Random per-page data
    K_fp8_used    = torch.randn(NUM_PG, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
    K_scales_used = (torch.rand(NUM_PG, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5)

    # Choose scattered page IDs in [1, NUM_PAGES_POOL)
    block_table = torch.tensor([5, 12, 3, 27], dtype=torch.int32, device=device)

    # Pack into pool at the chosen page IDs
    kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE, device=device, dtype=torch.uint8)
    for i in range(NUM_PG):
        pid = block_table[i].item()
        kv_pool[pid, :, :HEAD_DIM] = K_fp8_used[i].view(torch.uint8)
        kv_pool[pid, :, HEAD_DIM:HEAD_DIM + 4] = (
            K_scales_used[i].view(torch.uint8).reshape(PAGE_SIZE, 4)
        )

    q_fp8 = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
    w     = torch.randn(N, device=device, dtype=torch.float32)
    c_out = torch.zeros(M, device=device, dtype=torch.float32)

    kv_pool_ = from_dlpack(kv_pool, assumed_align=16)
    bt_      = from_dlpack(block_table, assumed_align=4)
    q_       = from_dlpack(q_fp8, assumed_align=16)
    w_       = from_dlpack(w, assumed_align=16)
    c_       = from_dlpack(c_out, assumed_align=16)

    kernel   = ScoreScaleFullBT(num_pg=NUM_PG)
    compiled = cute.compile(kernel, kv_pool_, bt_, q_, w_, c_)
    compiled(kv_pool_, bt_, q_, w_, c_)

    # Reference: K is just the per-page data flattened in block_table order
    K_ref = K_fp8_used.reshape(M, HEAD_DIM)
    K_scales_ref = K_scales_used.reshape(M)
    scores  = (K_ref.float() @ q_fp8.float().T) * K_scales_ref[:, None]
    ref_out = (torch.relu(scores) @ w)

    match   = torch.allclose(c_out, ref_out, atol=1.0, rtol=0.5)
    max_err = (c_out - ref_out).abs().max().item()
    print(f"\nCORRECTNESS {'PASS' if match else 'FAIL'}  (max_err={max_err:.4f})")
    if not match:
        print("  c_out[:8]:  ", c_out[:8].tolist())
        print("  ref_out[:8]:", ref_out[:8].tolist())

    t = benchmark(compiled, kernel_arguments=JitArguments(kv_pool_, bt_, q_, w_, c_))
    print(f"DURATION: {t:.4f} us  (M={M}, NUM_PG={NUM_PG})")


if __name__ == "__main__":
    main()

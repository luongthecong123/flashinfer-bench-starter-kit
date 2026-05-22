"""score_scale_tcgen05_page_faithful.py — tcgen05 fp8 GEMM with paged KV cache + per-token scale.

Faithful paged interface with block_table indirection (pages are NOT contiguous).

  kv_pool     : (num_pool_pages, PAGE_SIZE, ROW_STRIDE) int8  — global page pool
  block_table : (num_tiles * PAGES_PER_TILE,) int32           — page-ID indirection
  q_fp8       : (N, HEAD_DIM) Float8E4M3FN
  c_out       : (M, N) Float32

Page layout (per-row, PAGE_BYTES = 8448 bytes = 64 tokens × 132 bytes/token):
  token t, fp8 data : row[t, 0:128]   (HEAD_DIM bytes)
  token t, scale    : row[t, 128:132] (4 bytes, float32)

The page-jump stride between the two pages is computed at runtime from
block_table[bidx*2] and block_table[bidx*2+1], allowing non-contiguous pages.

C[m, n] = (sum_k fp8_A[m,k] * fp8_B[n,k]) * scale[m]
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

# ── Dimensions ────────────────────────────────────────────────────────────────
M              = 2048
N              = 64
HEAD_DIM       = 128
PAGE_SIZE      = 64
ROW_STRIDE     = HEAD_DIM + 4            # 132 bytes/token
PAGE_BYTES     = PAGE_SIZE * ROW_STRIDE  # 8448
FP8_REGION     = PAGE_SIZE * HEAD_DIM   # 8192 — fp8 bytes per page
PAGES_PER_TILE = 2
BM             = PAGE_SIZE * PAGES_PER_TILE  # 128

MMA_INST_MNK    = (128, N, 32)
CTA_TILE_MNK    = (BM, N, HEAD_DIM)

THREADS_PER_CTA = 128


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


class ScoreScaleTcgen05Page:
    """
    tcgen05 fp8 GEMM with non-contiguous paged KV cache via block_table.

    Each CTA tile handles PAGES_PER_TILE=2 pages whose indices are read from
    block_table[bidx*2] and block_table[bidx*2+1].  The page-jump stride is
    computed at runtime so a single autovec_copy covers both arbitrary pages.
    """

    def __init__(self):
        self.num_stages  = 1
        self.tmem_ld_rep = N   # load all N columns in one shot

    @cute.jit
    def __call__(
        self,
        kv_pool:     cute.Tensor,  # (num_pool_pages, PAGE_SIZE, ROW_STRIDE) int8
        block_table: cute.Tensor,  # (num_tiles * PAGES_PER_TILE,) int32
        q:           cute.Tensor,  # (N, HEAD_DIM) Float8E4M3FN
        c_out:       cute.Tensor,  # (M, N) Float32
    ):
        self.ab_dtype  = cutlass.Float8E4M3FN
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
            self.tiled_mma, CTA_TILE_MNK, self.ab_dtype, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, CTA_TILE_MNK, q.element_type, self.num_stages,
        )

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage

        num_tiles = block_table.shape[0] // PAGES_PER_TILE

        self.kernel(
            self.tiled_mma,
            kv_pool,
            block_table,
            q,
            c_out,
            self.a_smem_layout,
            self.b_smem_layout,
        ).launch(
            grid=(num_tiles, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma:     cute.TiledMma,
        kv_pool:       cute.Tensor,          # (num_pool_pages, PAGE_SIZE, ROW_STRIDE) int8
        block_table:   cute.Tensor,          # (num_tiles * PAGES_PER_TILE,) int32
        mB:            cute.Tensor,          # (N, HEAD_DIM) Float8E4M3FN
        mC:            cute.Tensor,          # (M, N) Float32
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _  = cute.arch.block_idx()

        # ── Block-table indirection ──────────────────────────────────
        page0_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 0])
        page1_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 1])

        # ── SMEM allocation ──────────────────────────────────────────
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

        # ── MMA fragments ────────────────────────────────────────────
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCrA    = tiled_mma.make_fragment_A(sA)
        tCrB    = tiled_mma.make_fragment_B(sB)

        acc_shape       = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc          = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        # ── TMEM alloc + mbarrier init ───────────────────────────────
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

        # ── autovec G→S for sA: runtime page-jump stride ─────────────
        # page0_id and page1_id come from block_table (non-contiguous pages).
        # jump_bytes = (page1_id - page0_id) * PAGE_BYTES is a runtime value;
        # CUTE layout supports runtime strides so we embed it as the outer stride
        # of the page dimension in gA_paged.
        thr_layout = cute.make_layout(THREADS_PER_CTA)

        fp8_base   = cute.recast_ptr(kv_pool.iterator, dtype=self.ab_dtype)
        page0_off  = page0_id * cutlass.Int32(PAGE_BYTES)
        jump_bytes = (page1_id - page0_id) * cutlass.Int32(PAGE_BYTES)

        fp8_ptr = cute.make_ptr(
            cutlass.Float8E4M3FN,
            (fp8_base + page0_off).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=1,
        )
        # Layout: ((PAGE_SIZE=64, PAGES_PER_TILE=2), HEAD_DIM=128)
        #   inner row stride = ROW_STRIDE=132 (each token row is 128 fp8 + 4 scale bytes)
        #   page stride      = jump_bytes (runtime jump to next page's fp8 base)
        #   dim stride       = 1
        gA_paged = cute.make_tensor(
            fp8_ptr,
            cute.make_layout(
                ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM),
                stride=((ROW_STRIDE, jump_bytes), 1),
            ),
        )
        tCgA   = thr_mma.partition_A(gA_paged)
        sA_thr = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr = cute.local_partition(tCgA, thr_layout, tidx)
        cute.autovec_copy(gA_thr, sA_thr)

        # ── autovec G→S for sB ───────────────────────────────────────
        gB_2d  = cute.make_tensor(mB.iterator,
                                  cute.make_layout((N, HEAD_DIM), stride=(HEAD_DIM, 1)))
        tCgB   = thr_mma.partition_B(gB_2d)
        sB_thr = cute.local_partition(sB[None, None, None, 0], thr_layout, tidx)
        gB_thr = cute.local_partition(tCgB, thr_layout, tidx)
        cute.autovec_copy(gB_thr, sB_thr)

        cute.arch.sync_threads()

        # ── Epilogue setup: TMEM → RMEM ──────────────────────────────
        M_acc          = cute.size(tCtAcc, mode=[0, 0])
        ld_op          = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler      = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi     = cute.zipped_divide(tCtAcc, epi_tiler)

        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        # ── tcgen05 MMA (warp 0) ─────────────────────────────────────
        tcgen05_fence()

        num_k_blocks = cute.size(tCrA, mode=[2])
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        if warp_idx == 0:
            for k_block_idx in range(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, 0)
                cute.gemm(tiled_mma, tCtAcc,
                          tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if tidx == 0:
                tcgen05.commit(mma_mbar)

        cute.arch.mbarrier_wait(mma_mbar, 0)

        # ── Epilogue: TMEM → RMEM → GMEM with per-token scale ────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

        # Each thread owns one token row (tidx) within the 128-token CTA tile.
        # Use block_table to get the actual page ID for scale indirection.
        # Per-row layout: scale for token t is at byte t*ROW_STRIDE + HEAD_DIM within a page.
        # As float32 offset from pool base:
        #   page_id * (PAGE_BYTES//4) + token_in_page * (ROW_STRIDE//4) + (HEAD_DIM//4)
        page_sel      = tidx // cutlass.Int32(PAGE_SIZE)         # 0 or 1
        token_in_page = tidx - page_sel * cutlass.Int32(PAGE_SIZE)
        page_id_t     = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + page_sel])
        scale_f32_off = (page_id_t * cutlass.Int32(PAGE_BYTES // 4)
                         + token_in_page * cutlass.Int32(ROW_STRIDE // 4)
                         + cutlass.Int32(HEAD_DIM // 4))
        fp32_base  = cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Float32)
        scale_ptr  = cute.make_ptr(
            cutlass.Float32,
            (fp32_base + scale_f32_off).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=1,
        )
        scale = cute.make_tensor(scale_ptr, cute.make_layout((1,), stride=(1,)))[0]

        m_out = bidx * cutlass.Int32(BM) + tidx
        for n_idx in cutlass.range_constexpr(N):
            mC[m_out, n_idx] = (tTR_rAcc[n_idx] * scale).to(self.c_dtype)

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── Python wrapper ────────────────────────────────────────────────────────────
def run_gemm(
    kv_pool:     torch.Tensor,  # (num_pool_pages, PAGE_SIZE, ROW_STRIDE) int8
    block_table: torch.Tensor,  # (num_tiles * PAGES_PER_TILE,) int32
    q_fp8:       torch.Tensor,  # (N, HEAD_DIM) Float8E4M3FN
    c_out:       torch.Tensor,  # (M, N) Float32
):
    c_out.zero_()
    kv_ = from_dlpack(kv_pool,     assumed_align=16)
    bt_ = from_dlpack(block_table, assumed_align=4)
    q_  = from_dlpack(q_fp8,       assumed_align=16)
    c_  = from_dlpack(c_out,       assumed_align=16)
    gemm     = ScoreScaleTcgen05Page()
    compiled = cute.compile(gemm, kv_, bt_, q_, c_)
    compiled(kv_, bt_, q_, c_)


# ── Reference dequant via block_table ───────────────────────────────────────
def dequant_flat_with_bt(kv_pool: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
    """Extract fp8 + scale via block_table indirection (per-row layout)."""
    k_u8 = kv_pool.view(torch.uint8)                               # (num_pool, 64, 132)
    # Gather pages in block_table order
    pages = block_table.cpu()
    fp8_data   = k_u8[pages, :, :HEAD_DIM].view(torch.float8_e4m3fn).to(torch.float32)  # (num_pages, 64, 128)
    scale_data = k_u8[pages, :, HEAD_DIM:].view(torch.float32)    # (num_pages, 64, 1)
    return fp8_data * scale_data                                    # (num_pages, 64, 128)


# ── Main: correctness test ────────────────────────────────────────────────────
def main():
    device    = "cuda"
    num_tiles = M // BM                    # 2048 / 128 = 16 CTAs
    num_pages = num_tiles * PAGES_PER_TILE  # 32 pages consumed

    # Build a pool larger than strictly needed to allow non-contiguous access.
    pool_size = num_pages + 16

    q_fp8 = torch.randn(N, HEAD_DIM, dtype=torch.float32, device=device).clamp(-4, 4).to(
        torch.float8_e4m3fn
    )

    # ── Build kv_pool: (pool_size, 64, 132) int8, per-row layout ───
    # Each token row: bytes [0:128] = fp8 K data, bytes [128:132] = float32 scale
    kv_pool_u8 = torch.zeros(
        pool_size, PAGE_SIZE, ROW_STRIDE, dtype=torch.uint8, device=device,
    )
    fp8_vals = (
        torch.randn(pool_size * PAGE_SIZE * HEAD_DIM, device=device)
        .clamp(-4, 4).to(torch.float8_e4m3fn)
    )
    kv_pool_u8[:, :, :HEAD_DIM].copy_(
        fp8_vals.view(torch.uint8).reshape(pool_size, PAGE_SIZE, HEAD_DIM)
    )
    scales_f32 = torch.rand(pool_size * PAGE_SIZE, device=device) * 0.1 + 0.01
    kv_pool_u8[:, :, HEAD_DIM:].copy_(
        scales_f32.view(torch.uint8).reshape(pool_size, PAGE_SIZE, 4)
    )
    kv_pool = kv_pool_u8.view(torch.int8)

    # ── Build non-contiguous block_table ─────────────────────────────
    # Shuffle page assignments to exercise non-contiguous access.
    perm = torch.randperm(pool_size, device=device)[:num_pages]
    block_table = perm.to(torch.int32)

    c_out = torch.zeros((M, N), device=device, dtype=torch.float32)

    run_gemm(kv_pool, block_table, q_fp8, c_out)
    torch.cuda.synchronize()

    # ── Reference ────────────────────────────────────────────────────
    K_scaled = dequant_flat_with_bt(kv_pool, block_table)  # (num_pages, 64, 128) scaled
    K_flat   = K_scaled.reshape(M, HEAD_DIM).to(device)
    ref_c    = K_flat @ q_fp8.float().T                    # (M, N)

    diff     = (c_out - ref_c).abs()
    max_err  = diff.max().item()
    mean_err = diff.mean().item()
    passed   = max_err < 1e-3

    print(f"\nCORRECTNESS {'PASS' if passed else 'FAIL'}  "
          f"(max_err={max_err:.6f}  mean_err={mean_err:.8f})")
    if not passed:
        worst = diff.argmax()
        m_w, n_w = divmod(worst.item(), N)
        print(f"  worst at (m={m_w}, n={n_w}): "
              f"kernel={c_out[m_w, n_w].item():.6f}  ref={ref_c[m_w, n_w].item():.6f}")
        print("  c_out[0,:8]:", c_out[0, :8].tolist())
        print("  ref_c[0,:8]:", ref_c[0, :8].tolist())

    kv_ = from_dlpack(kv_pool,     assumed_align=16)
    bt_ = from_dlpack(block_table, assumed_align=4)
    q_  = from_dlpack(q_fp8,       assumed_align=16)
    c_  = from_dlpack(c_out,       assumed_align=16)
    gemm     = ScoreScaleTcgen05Page()
    compiled = cute.compile(gemm, kv_, bt_, q_, c_)
    t = benchmark(compiled, kernel_arguments=JitArguments(kv_, bt_, q_, c_))
    print(f"DURATION: {t:.4f} µs")


if __name__ == "__main__":
    main()

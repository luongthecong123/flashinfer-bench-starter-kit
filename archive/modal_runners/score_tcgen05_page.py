"""score_tcgen05_page.py — Faithful paged tcgen05 fp8 MMA diagnostic.

k_cache shape: (num_pool_pages, PAGE_SIZE=64, ROW_STRIDE=132) int8
  - each row is 132 bytes: first 128 = fp8 K data, last 4 = scale (ignored here)
  - page0 and page1 for a tile are non-contiguous (random block_table)

Kernel: pure MMA accumulator — scale bytes are present but ignored.
Reference: block_table gather of fp8 bytes → float32 matmul, no scale.
Expected: max_err = 0.0000

The key correctness detail: the G→S layout token stride must be ROW_STRIDE=132
(not HEAD_DIM=128), because each token row in the page has 4 trailing scale bytes.

Usage:
    modal run src/modal/score_tcgen05_page.py
"""

import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

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

from src.modal.modal_utils import app, image

# ── Dimensions ─────────────────────────────────────────────────────────────
M              = 2048
N              = 64
HEAD_DIM       = 128
PAGE_SIZE      = 64
ROW_STRIDE     = HEAD_DIM + 4            # 132 bytes/token (128 fp8 + 4 scale)
PAGE_BYTES     = PAGE_SIZE * ROW_STRIDE  # 8448
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


class ScorePageMmaOnly:
    """
    Paged tcgen05 fp8 MMA — pure accumulator output, no scale.

    k_cache : (num_pool_pages, PAGE_SIZE, ROW_STRIDE) int8
    block_table : (num_tiles * PAGES_PER_TILE,) int32  — non-contiguous page IDs
    q           : (N, HEAD_DIM) Float8E4M3FN
    c_out       : (M, N) Float32

    The two pages for each CTA tile are looked up via block_table and may
    reside at arbitrary, non-adjacent positions in k_cache.
    """

    def __init__(self):
        self.num_stages  = 1
        self.tmem_ld_rep = N

    @cute.jit
    def __call__(
        self,
        k_cache:     cute.Tensor,  # (num_pool_pages, PAGE_SIZE, ROW_STRIDE) int8
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
            k_cache,
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
        k_cache:       cute.Tensor,          # (num_pool_pages, PAGE_SIZE, ROW_STRIDE) int8
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

        # ── autovec G→S for sA: paged, non-contiguous ────────────────
        #
        # k_cache layout (as int8/fp8 bytes):
        #   page pid starts at byte  pid * PAGE_BYTES  from k_cache base
        #   token t within a page is at byte  t * ROW_STRIDE  from the page base
        #   dim d within a token  is at byte  d  from the token base
        #
        # CRITICAL: inner row stride = ROW_STRIDE = 132, NOT HEAD_DIM = 128.
        # Each token row has 4 trailing scale bytes; skipping them requires
        # stride 132 to reach the fp8 data of the next token.
        #
        thr_layout  = cute.make_layout(THREADS_PER_CTA)
        fp8_base    = cute.recast_ptr(k_cache.iterator, dtype=self.ab_dtype)
        page0_off   = page0_id * cutlass.Int32(PAGE_BYTES)
        jump_bytes  = (page1_id - page0_id) * cutlass.Int32(PAGE_BYTES)

        fp8_ptr = cute.make_ptr(
            cutlass.Float8E4M3FN,
            (fp8_base + page0_off).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=1,
        )
        gA_paged = cute.make_tensor(
            fp8_ptr,
            cute.make_layout(
                ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM),
                stride=((ROW_STRIDE, jump_bytes), 1),   # ← ROW_STRIDE=132, not HEAD_DIM=128
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

        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

        # ── Epilogue: raw accumulator, NO scale ──────────────────────
        m_out = bidx * cutlass.Int32(BM) + tidx
        for n_idx in cutlass.range_constexpr(N):
            mC[m_out, n_idx] = tTR_rAcc[n_idx].to(self.c_dtype)

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


def main():
    device    = "cuda"
    num_tiles = M // BM                       # 16 CTAs
    num_pages = num_tiles * PAGES_PER_TILE    # 32 pages used

    # Pool is larger than needed so block_table permutation is non-trivial.
    pool_size = num_pages + 16   # 48 pages in pool

    q_fp8 = (
        torch.randn(N, HEAD_DIM, device=device)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )

    # ── Build k_cache: (pool_size, 64, 132) int8 ─────────────────────
    # Rows [  :128] = fp8 K data (HEAD_DIM bytes per token)
    # Rows [128:132] = float32 scale (present but NOT used by this kernel)
    k_u8 = torch.zeros(pool_size, PAGE_SIZE, ROW_STRIDE, dtype=torch.uint8, device=device)

    fp8_vals = (
        torch.randn(pool_size * PAGE_SIZE * HEAD_DIM, device=device)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    # Write fp8 data into the first HEAD_DIM bytes of each row
    k_u8[:, :, :HEAD_DIM].copy_(
        fp8_vals.view(torch.uint8).reshape(pool_size, PAGE_SIZE, HEAD_DIM)
    )
    # Write dummy scale bytes (values irrelevant — ignored by kernel)
    k_u8[:, :, HEAD_DIM:].fill_(0)

    k_cache = k_u8.view(torch.int8)   # (pool_size, 64, 132) int8

    # ── Non-contiguous block_table ────────────────────────────────────
    perm        = torch.randperm(pool_size, device=device)[:num_pages]
    block_table = perm.to(torch.int32)

    c_out = torch.zeros(M, N, device=device, dtype=torch.float32)

    # ── Compile & run kernel ─────────────────────────────────────────
    kc_ = from_dlpack(k_cache,    assumed_align=16)
    bt_ = from_dlpack(block_table, assumed_align=4)
    q_  = from_dlpack(q_fp8,      assumed_align=16)
    c_  = from_dlpack(c_out,      assumed_align=16)

    gemm     = ScorePageMmaOnly()
    compiled = cute.compile(gemm, kc_, bt_, q_, c_)
    compiled(kc_, bt_, q_, c_)
    torch.cuda.synchronize()

    # ── Reference: gather fp8 rows via block_table, no scale ─────────
    # k_u8[:, :, :HEAD_DIM] has shape (pool_size, PAGE_SIZE, HEAD_DIM)
    fp8_pool = k_u8[:, :, :HEAD_DIM].view(torch.float8_e4m3fn)  # (pool, 64, 128)
    # Gather pages in block_table order → (num_pages, 64, 128)
    K_gathered = fp8_pool[block_table.cpu()]                      # CPU gather is fine
    K_flat     = K_gathered.reshape(M, HEAD_DIM).to(torch.float32).to(device)
    ref_c      = K_flat @ q_fp8.float().T                         # (M, N)

    diff     = (c_out - ref_c).abs()
    max_err  = diff.max().item()
    mean_err = diff.mean().item()
    passed   = max_err == 0.0

    print(f"\nCORRECTNESS {'PASS' if passed else 'FAIL'}  "
          f"max_err={max_err:.4f}  mean_err={mean_err:.6f}")
    if not passed:
        worst = diff.argmax()
        m_w, n_w = divmod(worst.item(), N)
        print(f"  worst at (m={m_w}, n={n_w}): "
              f"kernel={c_out[m_w, n_w].item():.4f}  ref={ref_c[m_w, n_w].item():.4f}")
        print(f"  c_out[0,:8]: {c_out[0, :8].tolist()}")
        print(f"  ref_c[0,:8]: {ref_c[0, :8].tolist()}")

    t = benchmark(compiled, kernel_arguments=JitArguments(kc_, bt_, q_, c_))
    print(f"DURATION: {t:.4f} µs")


# ── Modal entrypoint ──────────────────────────────────────────────────────────
@app.function(image=image, gpu="B200:1", timeout=300)
def run_test():
    sys.path.insert(0, "/app")
    main()


@app.local_entrypoint()
def go():
    run_test.remote()

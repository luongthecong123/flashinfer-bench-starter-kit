"""
score_scale.py — FP8 tcgen05.mma with per-token scale applied in epilogue.

Uses FLAT byte layout (matching idxer_ref / idxer_tc reference):
  Per page (8448 bytes): first 8192 bytes = fp8 data (64 tokens × 128 dims),
  last 256 bytes = scales (64 tokens × 4 bytes float32).

A (kv_fp8): cooperative gmem→smem copy (contiguous stride 128).
B (q_fp8): TMA (q is contiguous, stride=128 ✓).
Scales: 128 float32 in SMEM (sScales[128]), one per thread, loaded before MMA.

GEMM: C[M, N=64] = kv_fp8[M,128] @ q_fp8[64,128].T  (fp8 → float32 acc)

Epilogue (in kernel, after Ld32x32b into 64 regs/thread):
  m = bidx * 128 + tidx
  mC[m, n] = tTR_rAcc[n] * sScales[tidx]   for n in 0..63
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05

# ── Dimensions ───────────────────────────────────────────────────────────────
TOP_K       = 2048
LIMIT_REQUEST = 4
LIMIT_SEQ_LEN = 2752
PAGE_SIZE   = 64
NUM_HEADS   = 64
N           = NUM_HEADS       # query heads (UMMA_N)
HEAD_DIM    = 128             # fp8 head dim (K)

ROW_STRIDE  = HEAD_DIM + 4    # 132 bytes per row in raw storage
PAGE_BYTES  = PAGE_SIZE * ROW_STRIDE   # 8448 bytes per page
FP8_REGION  = PAGE_SIZE * HEAD_DIM     # 8192 bytes of fp8 per page

MMA_INST_MNK    = (128, 64, 32)
CTA_TILE_MNK    = (128, N, HEAD_DIM)
BM              = 128

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

    kv_fp8   [M, 128] Float8E4M3FN — contiguous (flat-extracted)
    q_fp8    [N, 128] Float8E4M3FN
    k_scales [M]      Float32
    c_out    [M,  64] Float32

    A loaded via cooperative autovec_copy; B via TMA.
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
        kv_fp8:   cute.Tensor,   # [M, HEAD_DIM] Float8E4M3FN — contiguous
        q:        cute.Tensor,   # [N, HEAD_DIM] Float8E4M3FN
        k_scales: cute.Tensor,   # [M] Float32
        c_out:    cute.Tensor,   # [M, N] Float32
    ):
        self.fp8_dtype  = cutlass.Float8E4M3FN
        self.c_dtype    = c_out.element_type
        self.acc_dtype  = cutlass.Float32

        # kv_fp8 already has layout [M, HEAD_DIM] stride (HEAD_DIM, 1) — contiguous
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

        M_dim = kv_fp8.shape[0]
        grid_m = (M_dim + self.BM - 1) // self.BM
        self.kernel(
            self.tiled_mma,
            kv_fp8,
            tma_atom_b,
            tma_tensor_b,
            k_scales,
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
        mA_fp8:          cute.Tensor,   # GMEM kv_fp8 [M, HEAD_DIM] fp8, stride (128,1)
        tma_atom_b:      cute.CopyAtom,
        mB_tma_tensor:   cute.Tensor,   # TMA view of q [N, HEAD_DIM]
        k_scales:        cute.Tensor,   # [M] float32 — per-token scale
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
        # A has stride (HEAD_DIM=128, 1) — contiguous in flat layout.
        gA_local   = cute.local_tile(mA_fp8, CTA_TILE_MNK, mma_coord_mnk, proj=(1, None, 1))
        tCgA       = thr_mma.partition_A(gA_local)
        thr_layout = cute.make_layout(self.threads_per_cta)
        sA_thr     = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr     = cute.local_partition(tCgA[None, None, None, 0], thr_layout, tidx)
        cute.autovec_copy(gA_thr, sA_thr)

        # ── Load scale for this thread into SMEM ──────────────────────
        sScales[tidx] = k_scales[m_base + tidx]

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


# ── Compile with symbolic M ──────────────────────────────────────────────────
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_score_scale():
    M_sym = cute.sym_int()

    kv_fp8_fake   = _fake(cute.Float8E4M3FN, (M_sym, HEAD_DIM), (1, 0), 16)
    q_fake        = _fake(cute.Float8E4M3FN, (N, HEAD_DIM), (1, 0), 16)
    k_scales_fake = _fake(cute.Float32, (M_sym,), (0,), 4)
    c_out_fake    = _fake(cute.Float32, (M_sym, N), (1, 0), 16)

    kernel = ScoreScale()
    compiled = cute.compile(kernel, kv_fp8_fake, q_fake, k_scales_fake, c_out_fake)
    return kernel, compiled


_kernel_ss, _compiled_ss = compile_score_scale()


def _extract_flat(k_index_cache_fp8, block_table, max_sl):
    """Extract fp8 and scales from paged kv cache using flat byte layout."""
    B = block_table.shape[0]
    max_num_pages = block_table.shape[1]

    k_u8 = k_index_cache_fp8.view(torch.uint8)
    flat_bt = block_table.long().reshape(-1)
    gathered = k_u8[flat_bt]
    gathered_flat = gathered.reshape(B * max_num_pages, PAGE_BYTES)

    fp8_data = (gathered_flat[:, :FP8_REGION]
                .contiguous()
                .reshape(B, max_sl, HEAD_DIM)
                .view(torch.float8_e4m3fn))
    scale_data = (gathered_flat[:, FP8_REGION:]
                  .contiguous()
                  .view(torch.float32)
                  .reshape(B, max_sl))
    return fp8_data, scale_data


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    B = q_index_fp8.shape[0]
    max_num_pages = block_table.shape[1]
    max_sl = max_num_pages * PAGE_SIZE
    device = q_index_fp8.device

    # ── Extract fp8 + scales from paged kv cache (flat layout) ────────
    fp8_mat, scale_mat = _extract_flat(k_index_cache_fp8, block_table, max_sl)

    # ── Run MMA kernel per request ────────────────────────────────────
    scores_list = []
    for b in range(B):
        c_b = torch.zeros(max_sl, NUM_HEADS, dtype=torch.float32, device=device)
        _compiled_ss(fp8_mat[b], q_index_fp8[b], scale_mat[b], c_b)
        scores_list.append(c_b)

    torch.cuda.synchronize()

    # scores: (B, max_sl, NUM_HEADS) → (B, NUM_HEADS, max_sl)
    scores = torch.stack(scores_list, dim=0).permute(0, 2, 1)
    scores = torch.relu(scores)

    # Weighted sum across heads
    final = torch.einsum("bhs,bh->bs", scores, weights)

    # Mask padding positions
    positions = torch.arange(max_sl, device=device).unsqueeze(0)
    mask = positions >= seq_lens.unsqueeze(1)
    final.masked_fill_(mask, float("-inf"))

    # Top-K
    actual_k = min(TOP_K, max_sl)
    _, topk_idx = torch.topk(final, actual_k, dim=1)

    topk_page = topk_idx // PAGE_SIZE
    topk_off  = topk_idx % PAGE_SIZE
    global_pages = torch.gather(block_table.long(), 1, topk_page)
    global_tokens = (global_pages * PAGE_SIZE + topk_off).to(torch.int32)
    invalid = torch.gather(mask, 1, topk_idx)
    global_tokens[invalid] = -1

    topk_indices.fill_(-1)
    topk_indices[:, :actual_k] = global_tokens


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda"
    M_test = 2048

    num_pages = M_test // PAGE_SIZE

    q_fp8 = torch.randn(N, HEAD_DIM, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)

    k_index_cache_fp8 = torch.randint(0, 256,
                                       (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
                                       dtype=torch.uint8, device=device).view(torch.int8)

    # ── Extract flat fp8 + scales ──
    k_u8 = k_index_cache_fp8.view(torch.uint8)
    kv_flat = k_u8.reshape(num_pages, PAGE_BYTES)
    kv_fp8 = kv_flat[:, :FP8_REGION].contiguous().reshape(M_test, HEAD_DIM).view(torch.float8_e4m3fn)
    k_scales = kv_flat[:, FP8_REGION:].contiguous().view(torch.float32).reshape(M_test)

    # ── Reference (flat, matches idxer_tc.py) ──
    ref_flat = (kv_fp8.float() @ q_fp8.float().T) * k_scales[:, None]

    # ── Run kernel ──
    c_out = torch.zeros((M_test, N), device=device, dtype=torch.float32)
    _compiled_ss(kv_fp8, q_fp8, k_scales, c_out)

    # ── Diagnostics ──
    c_total = c_out.numel()
    c_nan = c_out.isnan().sum().item()
    ref_nan = ref_flat.isnan().sum().item()
    print(f"\nNaN counts (of {c_total}): kernel={c_nan}, ref_flat={ref_nan}")

    both_fin = torch.isfinite(c_out) & torch.isfinite(ref_flat)
    if both_fin.any():
        diff = (c_out[both_fin] - ref_flat[both_fin]).abs()
        print(f"Kernel vs FLAT ref: max_err={diff.max().item():.6f}  mean_err={diff.mean().item():.6f}")
        print("PASS" if diff.max().item() < 0.01 else "FAIL")
    else:
        print("No finite values to compare")

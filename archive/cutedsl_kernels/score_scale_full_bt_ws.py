"""
score_scale_full_bt_ws.py — score_scale_full_bt with workspace + tvm-ffi.

Differences from score_scale_full_bt.py:
  1. Workspace c_out: a single fixed [128, 640000] fp32 tensor allocated once
     in __init__. Each call writes only to row 0 (experimental — multi-batch
     dispatch will use rows for different requests later).
  2. tvm-ffi compile (make_fake_compact_tensor + make_fake_stream).
  3. NUM_PG is a runtime sym_int — kernel handles arbitrary even page counts.
  4. Benchmarking via torch.cuda.Event with L2 flush + arg clone (matches
     scripts/run_modal.py style — flashinfer-bench methodology).

Input layout (real contest):
  kv_pool      [NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE] uint8
  block_table  [NUM_PG] int32 (one request)
  q            [N=64, HEAD_DIM=128] fp8
  w            [N=64] fp32
  Workspace    [128, 640000] fp32 — kernel writes to row 0, cols [0, M)
"""

import math
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05

# ── Fixed dimensions ─────────────────────────────────────────────────────────
PAGE_SIZE  = 64
N          = 64
HEAD_DIM   = 128
ROW_STRIDE = 132
PAGES_PER_TILE = 2

MMA_INST_MNK = (128, 64, 32)
BM, BN, BK   = 128, N, HEAD_DIM

THREADS_PER_CTA = 128
TMEM_LD_REP     = N

# ── Workspace dims ───────────────────────────────────────────────────────────
WS_ROWS = 128
WS_COLS = 640000   # ≥ max possible M for any request (12K pages × 64 = 768K, but
                   # real contest max is 91 pages → 5824, this is plenty for now)


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
class ScoreScaleFullBTWS:
    """FP8 tcgen05 GEMM with block_table indirection + workspace output."""

    def __init__(self):
        self.threads_per_cta    = THREADS_PER_CTA
        self.num_stages         = 1
        self.tmem_ld_rep        = TMEM_LD_REP
        self.cta_tile_mnk       = (BM, BN, BK)
        self.mma_inst_shape_mnk = MMA_INST_MNK

        # ── Workspace: [128, 640000] fp32 — allocated once, reused ─────
        self.workspace = torch.empty(WS_ROWS, WS_COLS, dtype=torch.float32, device="cuda")

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        kv_pool:     cute.Tensor,   # [NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE] UInt8
        block_table: cute.Tensor,   # [NUM_PG] Int32 — sym_int leading dim
        q:           cute.Tensor,   # [N, HEAD_DIM] Float8E4M3FN
        w:           cute.Tensor,   # [N] Float32
        workspace:   cute.Tensor,   # [WS_ROWS, WS_COLS] Float32
        stream,
    ):
        self.fp8_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32

        # Grid M = NUM_PG // PAGES_PER_TILE (dynamic, from block_table extent)
        num_pg  = cute.size(block_table, mode=[0])
        grid_m  = num_pg // PAGES_PER_TILE

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
            workspace,
            self.a_smem_layout,
            self.b_smem_layout,
        ).launch(
            grid=(grid_m, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma:       cute.TiledMma,
        kv_pool:         cute.Tensor,
        block_table:     cute.Tensor,
        tma_atom_b:      cute.CopyAtom,
        mB_tma_tensor:   cute.Tensor,
        w:               cute.Tensor,
        workspace:       cute.Tensor,   # [WS_ROWS, WS_COLS]
        a_smem_layout:   cute.ComposedLayout,
        b_smem_layout:   cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)
        bidx, _, _  = cute.arch.block_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_b)

        # ── SMEM ─────────────────────────────────────────────────────
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

        # ── Per-CTA page indices ─────────────────────────────────────
        page0_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 0])
        page1_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 1])

        page_stride_bytes = PAGE_SIZE * ROW_STRIDE
        page0_off_b = page0_id * page_stride_bytes
        jump_b      = (page1_id - page0_id) * page_stride_bytes

        fp8_base = cute.recast_ptr(kv_pool.iterator, dtype=self.fp8_dtype) + page0_off_b
        gA_layout = cute.make_layout(
            ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM),
            stride=((ROW_STRIDE, jump_b), 1),
        )
        gA = cute.make_tensor(fp8_base, gA_layout)

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

        # ── Cooperative GMEM→SMEM for A ──────────────────────────────
        thr_layout = cute.make_layout(self.threads_per_cta)
        sA_thr     = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr     = cute.local_partition(tCgA, thr_layout, tidx)
        cute.autovec_copy(gA_thr, sA_thr)

        # ── Per-token scales (gather via 2-page layout) ──────────────
        SCALE_ROW_STRIDE_F32 = ROW_STRIDE // 4
        page_stride_f32      = PAGE_SIZE * SCALE_ROW_STRIDE_F32
        page0_off_f32        = page0_id * page_stride_f32
        jump_f32             = (page1_id - page0_id) * page_stride_f32

        fp32_base = cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Float32) + page0_off_f32
        scale_ptr = fp32_base + (HEAD_DIM // 4)
        scale_layout = cute.make_layout(
            ((PAGE_SIZE, PAGES_PER_TILE),),
            stride=((SCALE_ROW_STRIDE_F32, jump_f32),),
        )
        gScale = cute.make_tensor(scale_ptr, scale_layout)
        sScales[tidx] = gScale[tidx]

        if tidx < N:
            sWeights[tidx] = w[tidx]

        cute.arch.sync_threads()

        # ── MMA main loop ────────────────────────────────────────────
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

        # ── Epilogue → workspace[0, m_out] ───────────────────────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

        scale    = sScales[tidx]
        out_val  = cutlass.Float32(0)
        for n_idx in cutlass.range_constexpr(N):
            val      = tTR_rAcc[n_idx] * scale
            out_val  = out_val + max(val, cutlass.Float32(0)) * sWeights[n_idx]

        # Experimental: write only to workspace row 0, col = bidx*BM + tidx
        m_out = bidx * BM + tidx
        workspace[0, m_out] = out_val

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── tvm-ffi compile ──────────────────────────────────────────────────────────
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_score_scale_full_bt_ws():
    """Compile with NUM_PG as sym_int (must be even, divisibility=2)."""
    # NUM_PG ranges 33–91 in real workloads, but is dynamic in compiled kernel.
    NUM_PG = cute.sym_int(divisibility=2)
    NUM_PAGES_POOL = cute.sym_int()

    # kv_pool: [NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE] uint8 (matches existing kernel)
    kv_pool     = _fake(cute.Uint8,  (NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE),    (2, 1, 0),    16)
    block_table = _fake(cute.Int32,  (NUM_PG,),                                  (0,),         4)
    q           = _fake(cute.Float8E4M3FN, (N, HEAD_DIM),                        (1, 0),       16)
    w           = _fake(cute.Float32,(N,),                                       (0,),         16)
    workspace   = _fake(cute.Float32,(WS_ROWS, WS_COLS),                         (1, 0),       16)
    stream      = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = ScoreScaleFullBTWS()
    compiled = cute.compile(
        kernel,
        kv_pool, block_table, q, w, workspace, stream,
        options="--enable-tvm-ffi",
    )
    return kernel, compiled


# Singleton instance + compiled kernel (compile-once)
_kernel, _compiled = None, None


def get_compiled():
    global _kernel, _compiled
    if _compiled is None:
        _kernel, _compiled = compile_score_scale_full_bt_ws()
    return _kernel, _compiled


def run(kv_pool, block_table, q, w):
    """Convenience wrapper: returns the workspace's row 0 view of length M."""
    kernel, compiled = get_compiled()
    M = block_table.shape[0] * PAGE_SIZE
    compiled(kv_pool, block_table, q, w, kernel.workspace)
    return kernel.workspace[0, :M]


# ── Local sanity test ────────────────────────────────────────────────────────
def main():
    """Quick local test: NUM_PG=4, M=256."""
    device = "cuda"
    torch.manual_seed(0)

    NUM_PG = 4
    M = NUM_PG * PAGE_SIZE
    NUM_PAGES_POOL = 32

    K_fp8_used    = torch.randn(NUM_PG, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
    K_scales_used = torch.rand(NUM_PG, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5

    block_table = torch.tensor([5, 12, 3, 27], dtype=torch.int32, device=device)

    # Real layout: [pages, PAGE_SIZE, ROW_STRIDE]
    kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE, device=device, dtype=torch.uint8)
    for i in range(NUM_PG):
        pid = block_table[i].item()
        kv_pool[pid, :, :HEAD_DIM] = K_fp8_used[i].view(torch.uint8)
        kv_pool[pid, :, HEAD_DIM:HEAD_DIM + 4] = (
            K_scales_used[i].view(torch.uint8).reshape(PAGE_SIZE, 4)
        )

    q_fp8 = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
    w     = torch.randn(N, device=device, dtype=torch.float32)

    c_view = run(kv_pool, block_table, q_fp8, w)

    K_ref = K_fp8_used.reshape(M, HEAD_DIM)
    K_sc  = K_scales_used.reshape(M)
    scores  = (K_ref.float() @ q_fp8.float().T) * K_sc[:, None]
    ref_out = (torch.relu(scores) @ w)

    match   = torch.allclose(c_view, ref_out, atol=1.0, rtol=0.5)
    max_err = (c_view - ref_out).abs().max().item()
    print(f"\nCORRECTNESS {'PASS' if match else 'FAIL'}  (max_err={max_err:.4f})")
    if not match:
        print("  c_view[:8]:  ", c_view[:8].tolist())
        print("  ref_out[:8]:", ref_out[:8].tolist())


if __name__ == "__main__":
    main()

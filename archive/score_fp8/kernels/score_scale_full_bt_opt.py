"""
score_scale_full_bt_opt.py — optimized score_scale_full_bt_ws.

Optimizations vs score_scale_full_bt_ws:
  (1) A load uses **cp.async** (128-bit vectorized) instead of synchronous
      `cute.autovec_copy`. cp.async returns immediately, letting the B TMA
      wait + mbarrier wait overlap with the A-load latency.
  (2) sScales / sWeights loads are **issued before cp.async wait** so the
      LDG latency overlaps with cp.async + TMA latency. Single sync_threads
      barrier covers all three (A, scales, weights).
  (3) **PDL** (programmatic dependent launch):
        - launch with use_pdl=True
        - griddepcontrol_wait() before reading kv_pool/block_table/q (waits
          for prior kernel to publish them)
        - griddepcontrol_launch_dependents() right after TMA-B + cp.async-A
          are issued — lets the next kernel start launching while we do MMA
          + epilogue. Hides ~1-2 µs of grid-launch latency in fused pipelines.
  (4) Bench module supports both **warm-cache** and **L2-flush** modes,
      matching the cute `benchmark()` methodology for fair comparison.

All other shapes / layouts identical to score_scale_full_bt_ws. NUM_PG and
NUM_PAGES_POOL are runtime sym_int. Workspace [128, 640000] fp32 row 0.
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

WS_ROWS = 128
WS_COLS = 640000

# 128-bit cp.async: 16 fp8 elts per thread per copy
CPASYNC_BITS = 128
VEC_FP8      = CPASYNC_BITS // 8   # = 16


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


class ScoreScaleFullBTOpt:
    def __init__(self):
        self.threads_per_cta    = THREADS_PER_CTA
        self.num_stages         = 1
        self.tmem_ld_rep        = TMEM_LD_REP
        self.cta_tile_mnk       = (BM, BN, BK)
        self.mma_inst_shape_mnk = MMA_INST_MNK
        self.workspace = torch.empty(WS_ROWS, WS_COLS, dtype=torch.float32, device="cuda")

    @cute.jit
    def __call__(
        self,
        kv_pool:     cute.Tensor,
        block_table: cute.Tensor,
        q:           cute.Tensor,
        w:           cute.Tensor,
        workspace:   cute.Tensor,
        stream,
    ):
        self.fp8_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32

        num_pg  = cute.size(block_table, mode=[0])
        grid_m  = num_pg // PAGES_PER_TILE

        op = tcgen05.MmaFP8Op(
            self.fp8_dtype, self.acc_dtype, self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
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
            kv_pool, block_table,
            tma_atom_b, tma_tensor_b,
            w, workspace,
            self.a_smem_layout, self.b_smem_layout,
        ).launch(
            grid=(grid_m, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma:       cute.TiledMma,
        kv_pool:         cute.Tensor,
        block_table:     cute.Tensor,
        tma_atom_b:      cute.CopyAtom,
        mB_tma_tensor:   cute.Tensor,
        w:               cute.Tensor,
        workspace:       cute.Tensor,
        a_smem_layout:   cute.ComposedLayout,
        b_smem_layout:   cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)
        bidx, _, _  = cute.arch.block_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_b)

        # ── PDL: wait for prior kernel to publish kv_pool/block_table/q ──
        cute.arch.griddepcontrol_wait()

        smem    = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = smem.allocate_tensor(
            element_type=self.fp8_dtype, layout=a_smem_layout.outer,
            byte_alignment=128, swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.fp8_dtype, layout=b_smem_layout.outer,
            byte_alignment=128, swizzle=b_smem_layout.inner,
        )
        sScales = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(self.threads_per_cta),
            byte_alignment=16, swizzle=None,
        )
        sWeights = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(N),
            byte_alignment=16, swizzle=None,
        )

        # ── Per-CTA page indices and dynamic strides ────────────────
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

        # ── TMEM epilogue setup ──────────────────────────────────────
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

        # ════════════════════════════════════════════════════════════
        # OPT 1+2: fire TMA-B, then cp.async-A, then per-thread scalar
        # loads — wait once at the end. Maximizes overlap.
        # ════════════════════════════════════════════════════════════

        # ── Fire TMA for B (warp 0) ──────────────────────────────────
        if warp_idx == 0:
            cute.copy(tma_atom_b, tBgB[None, 0], tBsB[None, 0], tma_bar_ptr=tma_mbar)
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_transaction_bytes)

        # ── Fire cp.async for A (vectorized 128-bit, all threads) ────
        # Layout: gA is ((PAGE_SIZE, 2), HEAD_DIM) = (128, 128). HEAD_DIM
        # is contiguous (stride 1) → vectorizable along col dim.
        # 128 threads × 16 fp8 = 2048 elts per round. 128*128/2048 = 8 rounds.
        copy_atom_cp = cute.make_copy_atom(
            cpasync.CopyG2SOp(), self.fp8_dtype, num_bits_per_copy=CPASYNC_BITS,
        )

        thr_layout = cute.make_layout(self.threads_per_cta)
        sA_thr_full = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr_full = cute.local_partition(tCgA, thr_layout, tidx)
        cute.copy(copy_atom_cp, gA_thr_full, sA_thr_full)
        cute.arch.cp_async_commit_group()

        # ── PDL: signal next kernel can start launching now ─────────
        # All GMEM reads have been issued; remaining work is on-chip
        # (waits + MMA + epilogue + workspace store).
        cute.arch.griddepcontrol_launch_dependents()

        # ── Per-thread scale gather (overlaps with cp.async) ─────────
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

        # ── Wait for everything ──────────────────────────────────────
        cute.arch.cp_async_wait_group(0)
        cute.arch.mbarrier_wait(tma_mbar, 0)
        cute.arch.sync_threads()

        # ── MMA (single K-iter since BK == HEAD_DIM == 128) ──────────
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        tcgen05_fence()

        num_k_blocks = cute.size(tCrA, mode=[2])

        if warp_idx == 0:
            for k_block_idx in range(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, 0)
                cute.gemm(
                    tiled_mma, tCtAcc,
                    tCrA[k_block_coord], tCrB[k_block_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if tidx == 0:
                tcgen05.commit(mma_mbar)

        cute.arch.mbarrier_wait(mma_mbar, 0)

        # ── Epilogue ─────────────────────────────────────────────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()

        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

        scale    = sScales[tidx]
        out_val  = cutlass.Float32(0)
        for n_idx in cutlass.range_constexpr(N):
            val      = tTR_rAcc[n_idx] * scale
            out_val  = out_val + max(val, cutlass.Float32(0)) * sWeights[n_idx]

        m_out = bidx * BM + tidx
        workspace[0, m_out] = out_val

        cute.arch.barrier(barrier_id=tmem_barrier_id)

        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── tvm-ffi compile ──────────────────────────────────────────────────────────
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_score_scale_full_bt_opt():
    NUM_PG = cute.sym_int(divisibility=2)
    NUM_PAGES_POOL = cute.sym_int()

    kv_pool     = _fake(cute.Uint8,  (NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE), (2, 1, 0), 16)
    block_table = _fake(cute.Int32,  (NUM_PG,),                                (0,),      4)
    q           = _fake(cute.Float8E4M3FN, (N, HEAD_DIM),                     (1, 0),    16)
    w           = _fake(cute.Float32,(N,),                                    (0,),      16)
    workspace   = _fake(cute.Float32,(WS_ROWS, WS_COLS),                      (1, 0),    16)
    stream      = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = ScoreScaleFullBTOpt()
    compiled = cute.compile(
        kernel, kv_pool, block_table, q, w, workspace, stream,
        options="--enable-tvm-ffi",
    )
    return kernel, compiled


_kernel, _compiled = None, None


def get_compiled():
    global _kernel, _compiled
    if _compiled is None:
        _kernel, _compiled = compile_score_scale_full_bt_opt()
    return _kernel, _compiled

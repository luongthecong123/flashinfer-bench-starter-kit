"""score_scale_full_bt_ws_cpasync — production cp.async A-load kernel.

Strips intra-CTA probing from score_scale_full_bt_ws_cpasync_intra.py.
Carries the SOLVED A-load pattern:
  - 512-thread CTA, 128 compute threads
  - cp.async A-load via TV-layout PISL i32 + Sw<3,2,3>∘row_major
  - sA allocated FIRST in SMEM (offset 0 — required for swizzle correctness;
    see /memories/repo/cpasync-mma-A-load-SOLVED.md)
  - fence_view_async_shared between cp.async wait and MMA
  - Named barriers: INIT_BAR_ID=1 (512-wide), EPI_BAR_ID=2 (128-wide compute)

Measured on B200 (intra profiling):
  total ≈ 1.89 µs  (was 6.07 µs with autovec A-load — 3.2× speedup)
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05

# ── Constants ────────────────────────────────────────────────────────────────
PAGE_SIZE  = 64
N          = 64
HEAD_DIM   = 128
ROW_STRIDE = 132
PAGES_PER_TILE = 2

MMA_INST_MNK = (128, 64, 32)
BM, BN, BK   = 128, N, HEAD_DIM

THREADS_PER_CTA = 512        # all 512 issue cp.async A
COMPUTE_THREADS = 128        # TMA / MMA / epilogue
TMEM_LD_REP     = N

INIT_BAR_ID = 1   # 512-wide: tmem alloc + mbarrier init visible to all
EPI_BAR_ID  = 2   # 128-wide: epilogue compute group only

WS_ROWS = 128
WS_COLS = 640000

ROW_STRIDE_I32 = ROW_STRIDE // 4    # 33
HEAD_DIM_I32   = HEAD_DIM   // 4    # 32


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(None, [],
        "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


class ScoreScaleFullBTWSCpAsync:
    def __init__(self):
        self.threads_per_cta    = THREADS_PER_CTA
        self.num_stages         = 1
        self.tmem_ld_rep        = TMEM_LD_REP
        self.cta_tile_mnk       = (BM, BN, BK)
        self.mma_inst_shape_mnk = MMA_INST_MNK
        self.workspace = torch.empty(WS_ROWS, WS_COLS, dtype=torch.float32, device="cuda")

    @cute.jit
    def __call__(self, kv_pool, block_table, q, w, workspace, stream):
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
            self.tiled_mma, self.cta_tile_mnk, self.fp8_dtype, self.num_stages)
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.cta_tile_mnk, q.element_type, self.num_stages)

        b_smem_layout_one_stage = cute.select(self.b_smem_layout, mode=[0, 1, 2])
        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, q, b_smem_layout_one_stage, self.cta_tile_mnk, self.tiled_mma)

        @cute.struct
        class SharedStorage:
            tma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma, kv_pool, block_table,
            tma_atom_b, tma_tensor_b, w, workspace,
            self.a_smem_layout, self.b_smem_layout,
        ).launch(grid=(grid_m, 1, 1), block=(self.threads_per_cta, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self, tiled_mma, kv_pool, block_table, tma_atom_b, mB_tma_tensor,
        w, workspace, a_smem_layout, b_smem_layout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.warp_idx()
        warp_idx    = cute.arch.make_warp_uniform(warp_idx)
        bidx, _, _  = cute.arch.block_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_b)

        smem    = cutlass.utils.SmemAllocator()
        # CRITICAL: sA MUST be allocated at SMEM offset 0. Sw<3,2,3>∘row_major
        # on the i32 view of sA does NOT subtract the SMEM base offset; if sA
        # is offset>0 the swizzle XORs against absolute address bits → wrong
        # byte placement (max_err≈347). See cpasync-mma-A-load-SOLVED note.
        sA = smem.allocate_tensor(
            element_type=self.fp8_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sA_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BM, HEAD_DIM_I32),
                             stride=(HEAD_DIM_I32, 1)),
        )
        sA_i32_ptr = cute.recast_ptr(sA.iterator, dtype=cutlass.Int32)
        sA_load    = cute.make_tensor(sA_i32_ptr, sA_load_layout)
        storage = smem.allocate(self.shared_storage)
        sB = smem.allocate_tensor(self.fp8_dtype, b_smem_layout.outer, 128, b_smem_layout.inner)
        sScales  = smem.allocate_tensor(cutlass.Float32,
                       cute.make_layout(self.threads_per_cta), 16, None)
        sWeights = smem.allocate_tensor(cutlass.Float32,
                       cute.make_layout(N), 16, None)

        # ── Per-CTA pages ────────────────────────────────────────────
        page0_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 0])
        page1_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 1])
        page_stride_b   = PAGE_SIZE * ROW_STRIDE
        page_stride_i32 = page_stride_b // 4
        page0_off_i32   = page0_id * page_stride_i32
        jump_i32        = (page1_id - page0_id) * page_stride_i32
        page0_off_b     = page0_id * page_stride_b
        jump_b          = (page1_id - page0_id) * page_stride_b

        # ── GMEM B view (TMA) ────────────────────────────────────────
        gB = cute.local_tile(mB_tma_tensor, self.cta_tile_mnk, (bidx, 0, None), proj=(None, 1, 1))

        thr_mma = tiled_mma.get_slice(thr_idx=0)
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
            self.fp8_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(tma_mbar, cnt=1)
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        # 512-wide barrier: tmem alloc + mbarrier init visible to all threads.
        cute.arch.barrier(barrier_id=INIT_BAR_ID, number_of_threads=self.threads_per_cta)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # TMEM epilogue setup
        M_acc = cute.size(tCtAcc, mode=[0, 0])
        ld_op = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler  = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        # ════════════════════════════════════════════════════════════
        # FIRE: A-load — 512-thread cp.async via TV-layout PISL i32 view
        # ════════════════════════════════════════════════════════════
        i32_base = cute.make_ptr(
            cutlass.Int32,
            (cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Int32) + page0_off_i32).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=4,
        )
        gA_i32 = cute.make_tensor(i32_base, cute.make_layout(
            ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM_I32),
            stride=((ROW_STRIDE_I32, jump_i32), 1),
        ))
        atom_cpa = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32, num_bits_per_copy=cutlass.Int32.width,
        )
        N_PER_THREAD_I32 = (BM * HEAD_DIM_I32) // THREADS_PER_CTA   # 8
        thr_layout_load  = cute.make_layout((16, HEAD_DIM_I32),
                                            stride=(HEAD_DIM_I32, 1))
        val_layout_load  = cute.make_layout((N_PER_THREAD_I32, 1),
                                            stride=(1, 1))
        tiled_copy_a = cute.make_tiled_copy_tv(atom_cpa, thr_layout_load, val_layout_load)
        thr_copy_a   = tiled_copy_a.get_slice(tidx)
        tAgA = thr_copy_a.partition_S(gA_i32)
        tAsA = thr_copy_a.partition_D(sA_load)
        cute.copy(atom_cpa, tAgA, tAsA)

        # ── TMA-B fire (warp 0 elect — no COMPUTE_THREADS gate needed) ──
        if warp_idx == 0:
            cute.copy(tma_atom_b, tBgB[None, 0], tBsB[None, 0], tma_bar_ptr=tma_mbar)
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_transaction_bytes)

        # ── Per-token scales + weights (data-parallel; gScale has BM=128 elems) ──
        SCALE_ROW_STRIDE_F32 = ROW_STRIDE // 4
        page_stride_f32      = PAGE_SIZE * SCALE_ROW_STRIDE_F32
        page0_off_f32        = page0_id * page_stride_f32
        jump_f32             = (page1_id - page0_id) * page_stride_f32
        fp32_base = cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Float32) + page0_off_f32
        scale_ptr = fp32_base + (HEAD_DIM // 4)
        scale_layout = cute.make_layout(((PAGE_SIZE, PAGES_PER_TILE),),
                                        stride=((SCALE_ROW_STRIDE_F32, jump_f32),))
        gScale = cute.make_tensor(scale_ptr, scale_layout)
        if tidx < BM:
            sScales[tidx] = gScale[tidx]
        if tidx < N:
            sWeights[tidx] = w[tidx]

        # ── Consolidated sync: TMA wait + cp.async commit/wait + proxy fence ──
        cute.arch.mbarrier_wait(tma_mbar, 0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        cute.arch.fence_view_async_shared()

        # ── MMA fire (warp 0 elect — no COMPUTE_THREADS gate needed) ──
        tcgen05_fence()
        if warp_idx == 0:
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in range(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, 0)
                cute.gemm(tiled_mma, tCtAcc,
                          tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if tidx == 0:
                tcgen05.commit(mma_mbar)
            cute.arch.relinquish_tmem_alloc_permit()

        cute.arch.mbarrier_wait(mma_mbar, 0)

        # ── Epilogue (tmem→rmem→workspace): genuinely needs tidx<COMPUTE_THREADS
        # because tmem partition is per-thread for 128 threads and workspace
        # store at bidx*BM+tidx would OOB for tidx≥128.
        if tidx < COMPUTE_THREADS:
            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

            scale   = sScales[tidx]
            out_val = cutlass.Float32(0)
            for n_idx in cutlass.range_constexpr(N):
                val      = tTR_rAcc[n_idx] * scale
                out_val  = out_val + max(val, cutlass.Float32(0)) * sWeights[n_idx]

            m_out = bidx * BM + tidx
            workspace[0, m_out] = out_val

            cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=COMPUTE_THREADS)

            if warp_idx == 0:
                cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── tvm-ffi compile ──────────────────────────────────────────────────────────
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=align)


def compile_score_scale_full_bt_ws_cpasync():
    NUM_PG = cute.sym_int(divisibility=2)
    NUM_PAGES_POOL = cute.sym_int()

    kv_pool     = _fake(cute.Uint8,  (NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE), (2, 1, 0), 16)
    block_table = _fake(cute.Int32,  (NUM_PG,),                               (0,),      4)
    q           = _fake(cute.Float8E4M3FN, (N, HEAD_DIM),                     (1, 0),    16)
    w           = _fake(cute.Float32,(N,),                                    (0,),      16)
    workspace   = _fake(cute.Float32,(WS_ROWS, WS_COLS),                      (1, 0),    16)
    stream      = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = ScoreScaleFullBTWSCpAsync()
    compiled = cute.compile(
        kernel, kv_pool, block_table, q, w, workspace, stream,
        options="--enable-tvm-ffi",
    )
    return kernel, compiled


_kernel, _compiled = None, None


def get_compiled():
    global _kernel, _compiled
    if _compiled is None:
        _kernel, _compiled = compile_score_scale_full_bt_ws_cpasync()
    return _kernel, _compiled


def run(kv_pool, block_table, q, w):
    kernel, compiled = get_compiled()
    M = block_table.shape[0] * PAGE_SIZE
    compiled(kv_pool, block_table, q, w, kernel.workspace)
    return kernel.workspace[0, :M]

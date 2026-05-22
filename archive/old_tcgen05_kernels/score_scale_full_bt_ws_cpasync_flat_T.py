"""score_scale_full_bt_ws_cpasync_flat_T — T-dim persistent FLAT kernel.

NOTE: despite the file name, this version uses cp.async for BOTH A and B
(TMA + dynamic-T_idx hit several CUTLASS-DSL op-creation walls — see the
exploration history). B is only 8 KB so cp.async is essentially free.

Inputs:
  kv_pool      : (NUM_PAGES_POOL, PAGE_SIZE, 1, ROW_STRIDE) uint8  — FLAT layout
  block_table  : (T_actual, max_num_pages) int32
  seq_lens     : (T_actual,) int32
  q            : (T_actual, NUM_HEADS, HEAD_DIM) fp8
  w            : (T_actual, NUM_HEADS) fp32
  workspace    : (WS_ROWS, WS_COLS) fp32
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
MAX_T          = 32                 # ← static cap for TMA descriptor
PAGE_SIZE      = 64
NUM_HEADS      = 64
HEAD_DIM       = 128
ROW_STRIDE     = HEAD_DIM + 4
PAGES_PER_TILE = 2

PAGE_BYTES     = PAGE_SIZE * ROW_STRIDE     # 8448
FP8_REGION     = PAGE_SIZE * HEAD_DIM       # 8192

MMA_INST_MNK = (128, 64, 32)
BM, BN, BK   = 128, NUM_HEADS, HEAD_DIM

THREADS_PER_CTA = 512
COMPUTE_THREADS = 128
TMEM_LD_REP     = NUM_HEADS

INIT_BAR_ID = 1
EPI_BAR_ID  = 2

WS_ROWS = 32
WS_COLS = 320000

HEAD_DIM_I32   = HEAD_DIM // 4
PAGE_BYTES_I32 = PAGE_BYTES // 4
PAGE_BYTES_F32 = PAGE_BYTES // 4
FP8_REGION_F32 = FP8_REGION // 4


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(None, [],
        "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


class ScoreScaleFullBTWSCpAsyncFlatT:
    def __init__(self):
        self.threads_per_cta    = THREADS_PER_CTA
        self.num_stages         = 1
        self.tmem_ld_rep        = TMEM_LD_REP
        self.cta_tile_mnk       = (BM, BN, BK)
        self.mma_inst_shape_mnk = MMA_INST_MNK
        self.workspace = torch.empty(WS_ROWS, WS_COLS, dtype=torch.float32, device="cuda")

    @cute.jit
    def __call__(self, kv_pool, block_table, seq_lens, q, w, workspace, stream):
        self.fp8_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32

        T_actual, max_num_pages = block_table.shape
        num_splits = (max_num_pages + PAGES_PER_TILE - 1) // PAGES_PER_TILE

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

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma, kv_pool, block_table, seq_lens,
            q, w, workspace, T_actual,
            self.a_smem_layout, self.b_smem_layout,
        ).launch(grid=(num_splits, 1, 1), block=(self.threads_per_cta, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self, tiled_mma, kv_pool, block_table, seq_lens,
        q, w, workspace, T_actual,
        a_smem_layout, b_smem_layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        # Allocate sA first at offset 0 (its swizzle requires alignment).
        sA = smem.allocate_tensor(
            element_type=self.fp8_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sA_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BM, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1)),
        )
        sA_i32_ptr = cute.recast_ptr(sA.iterator, dtype=cutlass.Int32)
        sA_load    = cute.make_tensor(sA_i32_ptr, sA_load_layout)
        # sB right after sA — both are 16KB / 8KB SW128-swizzled buffers; sA is
        # 16384 B (mult of 1024 swizzle period), so sB ends up at an aligned
        # offset for SW128 too.
        sB = smem.allocate_tensor(self.fp8_dtype, b_smem_layout.outer, 1024, b_smem_layout.inner)
        storage = smem.allocate(self.shared_storage)
        sScales  = smem.allocate_tensor(cutlass.Float32,
                       cute.make_layout(self.threads_per_cta), 16, None)
        sWeights = smem.allocate_tensor(cutlass.Float32,
                       cute.make_layout(BN), 16, None)

        # ── MMA fragments (constant across iters) ────────────────────
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCrA    = tiled_mma.make_fragment_A(sA)
        tCrB    = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(self.cta_tile_mnk[:2])
        tCtAcc    = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        # ── One-time TMEM alloc + mbar init ──────────────────────────
        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        cute.arch.barrier(barrier_id=INIT_BAR_ID, number_of_threads=self.threads_per_cta)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # TMEM epilogue setup (constant)
        M_acc = cute.size(tCtAcc, mode=[0, 0])
        ld_op = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler  = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        # cp.async A static atom (constant)
        atom_cpa = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32, num_bits_per_copy=cutlass.Int32.width,
        )
        N_PER_THREAD_I32 = (BM * HEAD_DIM_I32) // THREADS_PER_CTA
        thr_layout_load = cute.make_layout((16, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1))
        val_layout_load = cute.make_layout((N_PER_THREAD_I32, 1), stride=(1, 1))
        tiled_copy_a = cute.make_tiled_copy_tv(atom_cpa, thr_layout_load, val_layout_load)
        thr_copy_a   = tiled_copy_a.get_slice(tidx)

        # ── cp.async B: 8 KB total (BN*BK fp8 = 64*128 = 8192 B) ──
        # Same SW128 swizzle as A in i32 view — both are K-major fp8 BK=128.
        BK_I32 = HEAD_DIM_I32   # 32
        sB_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BN, BK_I32), stride=(BK_I32, 1)),
        )
        sB_i32_ptr = cute.recast_ptr(sB.iterator, dtype=cutlass.Int32)
        sB_load    = cute.make_tensor(sB_i32_ptr, sB_load_layout)
        N_PER_THREAD_B_I32 = (BN * BK_I32) // THREADS_PER_CTA   # 4
        thr_layout_load_b  = cute.make_layout((THREADS_PER_CTA // BK_I32, BK_I32),
                                              stride=(BK_I32, 1))   # (16, 32)
        val_layout_load_b  = cute.make_layout((N_PER_THREAD_B_I32, 1), stride=(1, 1))
        tiled_copy_b = cute.make_tiled_copy_tv(atom_cpa, thr_layout_load_b, val_layout_load_b)
        thr_copy_b   = tiled_copy_b.get_slice(tidx)

        # q recast to i32; per-t base offset.
        q_i32_base_full = cute.recast_ptr(q.iterator, dtype=cutlass.Int32)
        Q_T_STRIDE_I32  = BN * BK_I32   # 64 * 32 = 2048 i32 per request


        # ── Persistent loop over T_actual requests ─────────────────────
        mma_phase = cutlass.Int32(0)

        for t_idx in range(T_actual):
            seq = seq_lens[t_idx]
            req_tiles = (seq + cutlass.Int32(BM - 1)) // cutlass.Int32(BM)

            if bidx < req_tiles:
                page0_id = cutlass.Int32(block_table[t_idx, bidx * PAGES_PER_TILE + 0])
                page1_id = cutlass.Int32(block_table[t_idx, bidx * PAGES_PER_TILE + 1])

                # ── B-load: 512-thread cp.async on q[t_idx, :, :] (i32 view) ──
                q_off_i32 = t_idx * Q_T_STRIDE_I32
                qB_base = cute.make_ptr(
                    cutlass.Int32,
                    (q_i32_base_full + q_off_i32).toint(),
                    mem_space=cute.AddressSpace.gmem, assumed_align=16,
                )
                gB_i32 = cute.make_tensor(qB_base, cute.make_layout(
                    (BN, BK_I32), stride=(BK_I32, 1),
                ))
                tBgB = thr_copy_b.partition_S(gB_i32)
                tBsB = thr_copy_b.partition_D(sB_load)
                cute.copy(atom_cpa, tBgB, tBsB)

                # ── A-load: 512-thread cp.async ──
                page_stride_i32 = PAGE_BYTES_I32
                page0_off_i32   = page0_id * page_stride_i32
                jump_i32        = (page1_id - page0_id) * page_stride_i32
                i32_base = cute.make_ptr(
                    cutlass.Int32,
                    (cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Int32) + page0_off_i32).toint(),
                    mem_space=cute.AddressSpace.gmem, assumed_align=4,
                )
                gA_i32 = cute.make_tensor(i32_base, cute.make_layout(
                    ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM_I32),
                    stride=((HEAD_DIM_I32, jump_i32), 1),
                ))
                tAgA = thr_copy_a.partition_S(gA_i32)
                tAsA = thr_copy_a.partition_D(sA_load)
                cute.copy(atom_cpa, tAgA, tAsA)

                # ── Per-token scales (FLAT) ──
                page_stride_f32 = PAGE_BYTES_F32
                page0_off_f32   = page0_id * page_stride_f32
                jump_f32        = (page1_id - page0_id) * page_stride_f32
                fp32_base = cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Float32) + page0_off_f32
                scale_ptr = fp32_base + FP8_REGION_F32
                scale_layout = cute.make_layout(((PAGE_SIZE, PAGES_PER_TILE),),
                                                stride=((1, jump_f32),))
                gScale = cute.make_tensor(scale_ptr, scale_layout)
                if tidx < BM:
                    sScales[tidx] = gScale[tidx]
                if tidx < BN:
                    sWeights[tidx] = w[t_idx, tidx]

                # ── Sync (cp.async only — no TMA mbar) ──
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                cute.arch.fence_view_async_shared()

                # ── MMA ──
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

                cute.arch.mbarrier_wait(mma_mbar, mma_phase)
                mma_phase = mma_phase ^ cutlass.Int32(1)

                # ── Epilogue ──
                if tidx < COMPUTE_THREADS:
                    cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

                    scale   = sScales[tidx]
                    out_val = cutlass.Float32(0)
                    for n_idx in cutlass.range_constexpr(BN):
                        val      = tTR_rAcc[n_idx] * scale
                        out_val  = out_val + max(val, cutlass.Float32(0)) * sWeights[n_idx]

                    m_out = bidx * BM + tidx
                    workspace[t_idx, m_out] = out_val

                cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=self.threads_per_cta)

        # ── One-time TMEM dealloc ────────────────────────────────────
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── tvm-ffi compile ──────────────────────────────────────────────────────────
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=align)


def compile_score_scale_full_bt_ws_cpasync_flat_T():
    T_ACTUAL       = cute.sym_int()
    MAX_NUM_PAGES  = cute.sym_int()
    NUM_PAGES_POOL = cute.sym_int()

    kv_pool     = _fake(cute.Uint8,  (NUM_PAGES_POOL, PAGE_SIZE, 1, ROW_STRIDE), (3, 2, 1, 0), 16)
    block_table = _fake(cute.Int32,  (T_ACTUAL, MAX_NUM_PAGES),                  (1, 0),       4)
    seq_lens    = _fake(cute.Int32,  (T_ACTUAL,),                                (0,),         4)
    q           = _fake(cute.Float8E4M3FN, (T_ACTUAL, NUM_HEADS, HEAD_DIM),      (2, 1, 0),    16)
    w           = _fake(cute.Float32,(T_ACTUAL, NUM_HEADS),                      (1, 0),       16)
    workspace   = _fake(cute.Float32,(WS_ROWS, WS_COLS),                         (1, 0),       16)
    stream      = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = ScoreScaleFullBTWSCpAsyncFlatT()
    compiled = cute.compile(
        kernel, kv_pool, block_table, seq_lens, q, w, workspace, stream,
        options="--enable-tvm-ffi",
    )
    return kernel, compiled


_kernel, _compiled = None, None


def get_compiled():
    global _kernel, _compiled
    if _compiled is None:
        _kernel, _compiled = compile_score_scale_full_bt_ws_cpasync_flat_T()
    return _kernel, _compiled

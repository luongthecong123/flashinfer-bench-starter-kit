"""score_scale_full_bt_ws_cpasync_persist — persistent variant of
score_scale_full_bt_ws_cpasync.py.

Persistence axis: REQUESTS (not tiles).
  - grid       = (num_tiles, 1, 1)            # one block per tile
  - inner loop = for req_idx in range(num_requests): compute this tile for that request
  - per-request runtime TMA-B slicing on q_3d[req_idx, :, :]
  - per-request block_table row, weights row, workspace row

This mirrors the indexer's `indexer_ksplit_kernel` persistent score path:
  - bidx = tile_idx (split_idx in indexer terminology)
  - inner loop over indexer_requests (T_idx values with seq > 2048)

Reuses the SOLVED A-load pattern from score_scale_full_bt_ws_cpasync.py:
  - 512-thread CTA, 128 compute threads
  - cp.async A via TV-layout PISL i32 + Sw<3,2,3>∘row_major
  - sA at SMEM offset 0 (CRITICAL — see cpasync-mma-A-load-SOLVED note)

Mbarrier reuse via phase = req_idx & 1.
TMEM alloc once before loop, dealloc once after.
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

THREADS_PER_CTA = 512
COMPUTE_THREADS = 128
TMEM_LD_REP     = N

INIT_BAR_ID = 1
EPI_BAR_ID  = 2

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


class ScoreScaleFullBTWSCpAsyncPersist:
    def __init__(self):
        self.threads_per_cta    = THREADS_PER_CTA
        self.num_stages         = 1
        self.tmem_ld_rep        = TMEM_LD_REP
        self.cta_tile_mnk       = (BM, BN, BK)
        self.mma_inst_shape_mnk = MMA_INST_MNK
        self.workspace = torch.empty(WS_ROWS, WS_COLS, dtype=torch.float32, device="cuda")

    @cute.jit
    def __call__(self, kv_pool, block_table_2d, q_3d, w_2d, workspace, stream):
        """
        kv_pool        : (NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE) u8  — global pool
        block_table_2d : (num_requests, num_pg) Int32              — per-request
        q_3d           : (num_requests, N, HEAD_DIM) fp8           — per-request
        w_2d           : (num_requests, N) f32                     — per-request
        workspace      : (WS_ROWS, WS_COLS) f32                    — output, row=req_idx
        """
        self.fp8_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32

        num_requests = cute.size(block_table_2d, mode=[0])
        num_pg       = cute.size(block_table_2d, mode=[1])
        num_tiles    = num_pg // PAGES_PER_TILE

        op = tcgen05.MmaFP8Op(
            self.fp8_dtype, self.acc_dtype, self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.cta_tile_mnk, self.fp8_dtype, self.num_stages)
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.cta_tile_mnk, q_3d.element_type, self.num_stages)

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        # Grid = num_tiles; each block persists over num_requests.
        # NOTE: q is loaded via cp.async (not TMA) because per-request runtime
        # slicing on the outer (req_idx) mode of a 3D TMA source isn't supported
        # by local_tile/proj. q is only 8 KB/req so cp.async is fine.
        self.kernel(
            self.tiled_mma, kv_pool, block_table_2d, num_requests,
            q_3d, w_2d, workspace,
            self.a_smem_layout, self.b_smem_layout,
        ).launch(grid=(num_tiles, 1, 1), block=(self.threads_per_cta, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self, tiled_mma, kv_pool, block_table_2d, num_requests,
        q_3d, w_2d, workspace, a_smem_layout, b_smem_layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()  # = tile_idx
        warp_idx   = cute.arch.warp_idx()
        warp_idx   = cute.arch.make_warp_uniform(warp_idx)

        smem = cutlass.utils.SmemAllocator()
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
        sB = smem.allocate_tensor(self.fp8_dtype, b_smem_layout.outer, 1024, b_smem_layout.inner)
        sScales  = smem.allocate_tensor(cutlass.Float32,
                       cute.make_layout(self.threads_per_cta), 16, None)
        sWeights = smem.allocate_tensor(cutlass.Float32,
                       cute.make_layout(N), 16, None)

        tCrA    = tiled_mma.make_fragment_A(sA)
        tCrB    = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(self.cta_tile_mnk[:2])
        tCtAcc    = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        mma_mbar = storage.mma_mbar_ptr.data_ptr()

        # ── One-time setup: TMEM alloc + mbarrier init ──
        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        cute.arch.barrier(barrier_id=INIT_BAR_ID, number_of_threads=self.threads_per_cta)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # TMEM epilogue plumbing (constant across iters)
        M_acc = cute.size(tCtAcc, mode=[0, 0])
        ld_op = tcgen05.Ld32x32bOp(tcgen05.Repetition(self.tmem_ld_rep))
        epi_tiler  = ((M_acc, self.tmem_ld_rep),)
        tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, self.acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, self.acc_dtype)

        # cp.async A copy plumbing (constant across iters)
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
        tAsA = thr_copy_a.partition_D(sA_load)

        # cp.async B (q) plumbing — q tile = (BN=64, HEAD_DIM=128) fp8 = 8 KB.
        # Mirror A path: 32b atom + manual Sw<3,4,3>∘row_major i32 view of sB.
        # sB allocation alignment was bumped to 1024 (= swizzle period for
        # Sw<3,4,3>: 2^(B+M+S) = 2^10) so absolute address bits 0..9 of sB's
        # base equal relative offset bits → manual swizzle composes correctly
        # even though sB is NOT at SMEM offset 0 (avoids the SOLVED A-load bug).
        # 64 rows × 32 i32 = 2048 entries; 512 threads × 4 i32/thread = 16B/thread.
        N_PER_THREAD_I32_B = (BN * HEAD_DIM_I32) // THREADS_PER_CTA   # 4
        atom_cpb = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32, num_bits_per_copy=cutlass.Int32.width,
        )
        # 512 threads laid (BN=64 rows, HEAD_DIM_I32/N_PER_THREAD_I32_B=8 cols)
        thr_layout_load_b = cute.make_layout(
            (BN, HEAD_DIM_I32 // N_PER_THREAD_I32_B),
            stride=(HEAD_DIM_I32 // N_PER_THREAD_I32_B, 1),
        )
        val_layout_load_b = cute.make_layout(
            (1, N_PER_THREAD_I32_B), stride=(1, 1),
        )
        tiled_copy_b = cute.make_tiled_copy_tv(atom_cpb, thr_layout_load_b, val_layout_load_b)
        thr_copy_b   = tiled_copy_b.get_slice(tidx)
        sB_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BN, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1)),
        )
        sB_load = cute.make_tensor(
            cute.recast_ptr(sB.iterator, dtype=cutlass.Int32),
            sB_load_layout,
        )
        tBsB_dst = thr_copy_b.partition_D(sB_load)

        # ════════════════════════════════════════════════════════════
        # Persistent loop over requests (each block = one tile = bidx)
        # ════════════════════════════════════════════════════════════
        for req_idx in range(num_requests):
            phase = req_idx & 1

            # ── Per-request page IDs for this tile (bidx) ──
            page0_id = cutlass.Int32(block_table_2d[req_idx, bidx * PAGES_PER_TILE + 0])
            page1_id = cutlass.Int32(block_table_2d[req_idx, bidx * PAGES_PER_TILE + 1])
            page_stride_b   = PAGE_SIZE * ROW_STRIDE
            page_stride_i32 = page_stride_b // 4
            page0_off_i32   = page0_id * page_stride_i32
            jump_i32        = (page1_id - page0_id) * page_stride_i32

            # ── A-load (cp.async i32 view) ──
            i32_base = cute.make_ptr(
                cutlass.Int32,
                (cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Int32) + page0_off_i32).toint(),
                mem_space=cute.AddressSpace.gmem, assumed_align=4,
            )
            gA_i32 = cute.make_tensor(i32_base, cute.make_layout(
                ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM_I32),
                stride=((ROW_STRIDE_I32, jump_i32), 1),
            ))
            tAgA = thr_copy_a.partition_S(gA_i32)
            cute.copy(atom_cpa, tAgA, tAsA)

            # ── B-load: q[req_idx, :, :] via 32b cp.async, swizzled sB dest ──
            q_req_off_i32 = req_idx * (N * HEAD_DIM_I32)
            gB_i32_ptr = cute.make_ptr(
                cutlass.Int32,
                (cute.recast_ptr(q_3d.iterator, dtype=cutlass.Int32) + q_req_off_i32).toint(),
                mem_space=cute.AddressSpace.gmem, assumed_align=4,
            )
            gB_i32 = cute.make_tensor(gB_i32_ptr, cute.make_layout(
                (N, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1),
            ))
            tBgB = thr_copy_b.partition_S(gB_i32)
            cute.copy(atom_cpb, tBgB, tBsB_dst)

            # ── Per-tile scales ──
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

            # ── Per-request weights ──
            if tidx < N:
                sWeights[tidx] = w_2d[req_idx, tidx]

            # ── Sync: cp.async commit + wait + proxy fence ──
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            cute.arch.fence_view_async_shared()

            # ── MMA fire ──
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

            cute.arch.mbarrier_wait(mma_mbar, phase)

            # ── Epilogue ──
            if tidx < COMPUTE_THREADS:
                cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

                scale   = sScales[tidx]
                out_val = cutlass.Float32(0)
                for n_idx in cutlass.range_constexpr(N):
                    val      = tTR_rAcc[n_idx] * scale
                    out_val  = out_val + max(val, cutlass.Float32(0)) * sWeights[n_idx]

                m_out = bidx * BM + tidx
                workspace[req_idx, m_out] = out_val

            # All threads must end the iter together before the next phase.
            cute.arch.sync_threads()

        # ── One-time teardown ──
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ── tvm-ffi compile ──────────────────────────────────────────────────────────
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=align)


def compile_score_scale_full_bt_ws_cpasync_persist():
    NUM_REQ = cute.sym_int()
    NUM_PG  = cute.sym_int(divisibility=2)
    NUM_PAGES_POOL = cute.sym_int()

    kv_pool        = _fake(cute.Uint8,        (NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE), (2, 1, 0), 16)
    block_table_2d = _fake(cute.Int32,        (NUM_REQ, NUM_PG),                       (1, 0),    4)
    q_3d           = _fake(cute.Float8E4M3FN, (NUM_REQ, N, HEAD_DIM),                  (2, 1, 0), 16)
    w_2d           = _fake(cute.Float32,      (NUM_REQ, N),                            (1, 0),    16)
    workspace      = _fake(cute.Float32,      (WS_ROWS, WS_COLS),                      (1, 0),    16)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = ScoreScaleFullBTWSCpAsyncPersist()
    compiled = cute.compile(
        kernel, kv_pool, block_table_2d, q_3d, w_2d, workspace, stream,
        options="--enable-tvm-ffi",
    )
    return kernel, compiled


_kernel, _compiled = None, None


def get_compiled():
    global _kernel, _compiled
    if _compiled is None:
        _kernel, _compiled = compile_score_scale_full_bt_ws_cpasync_persist()
    return _kernel, _compiled


def run(kv_pool, block_table_2d, q_3d, w_2d):
    """
    block_table_2d : (num_requests, num_pg) Int32
    q_3d           : (num_requests, N, HEAD_DIM) fp8
    w_2d           : (num_requests, N) f32
    Returns: workspace[:num_requests, :num_pg*PAGE_SIZE]
    """
    kernel, compiled = get_compiled()
    num_requests = block_table_2d.shape[0]
    M = block_table_2d.shape[1] * PAGE_SIZE
    compiled(kv_pool, block_table_2d, q_3d, w_2d, kernel.workspace)
    return kernel.workspace[:num_requests, :M]

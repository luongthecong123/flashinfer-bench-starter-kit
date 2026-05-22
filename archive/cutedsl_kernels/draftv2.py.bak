import math
import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op, T

TOP_K = 2048
LIMIT_REQUEST = 128
LIMIT_SEQ_LEN = 640000
DIM_SPLIT = 128
PAGE_SIZE = 64
NUM_HEADS  = 64
HEAD_DIM   = 128

# Score-kernel constants (mirrored from score_scale_full_bt_ws_cpasync_persist.py)
ROW_STRIDE     = HEAD_DIM + 4   # 132 bytes per row (HEAD_DIM fp8 + 4B scale)
PAGES_PER_TILE = DIM_SPLIT // PAGE_SIZE   # 2
BM             = DIM_SPLIT       # 128 — M tile (tokens per split)
BN             = NUM_HEADS       # 64  — N tile (one q tile)
BK             = HEAD_DIM        # 128 — K tile
MMA_INST_MNK   = (128, 64, 32)
TMEM_LD_REP    = BN
HEAD_DIM_I32   = HEAD_DIM // 4   # 32
ROW_STRIDE_I32 = ROW_STRIDE // 4 # 33
INIT_BAR_ID    = 1


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(None, [],
        "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

@dsl_user_op
def float_to_radix(v: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    r = llvm.inline_asm(
        T.i32(), [v.ir_value()],
        "{"
        ".reg .u32 x; .reg .u32 mask; .reg .pred pneg; .reg .pred pnan;"
        "mov.b32 x, $1;"
        "setp.lt.f32 pneg, $1, 0f00000000;"
        "setp.neu.f32 pnan, $1, $1;"
        "selp.u32 mask, 0xFFFFFFFF, 0x80000000, pneg;"
        "xor.b32 x, x, mask;"
        "selp.u32 $0, 0xFFFFFFFF, x, pnan;"
        "}",
        "=r,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return cutlass.Uint32(r)


@cute.jit
def warp_sum_i32(val: cutlass.Int32) -> cutlass.Int32:
    for i in cutlass.range_constexpr(5):
        peer = cute.arch.shuffle_sync_bfly(val, 1 << i)
        val = val + peer
    return val


@cute.jit
def count_element(bits, desired, desired_mask, digit_pos_u, c0, c1, c2, c3):
    if (bits & desired_mask) == (desired & desired_mask):
        digit = (bits >> digit_pos_u) & cutlass.Uint32(3)
        if digit == cutlass.Uint32(0):
            c0 = c0 + cutlass.Int32(1)
        if digit == cutlass.Uint32(1):
            c1 = c1 + cutlass.Int32(1)
        if digit == cutlass.Uint32(2):
            c2 = c2 + cutlass.Int32(1)
        if digit == cutlass.Uint32(3):
            c3 = c3 + cutlass.Int32(1)
    return c0, c1, c2, c3


@cute.struct
class ScoreSharedStorage:
    mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 2]
    tmem_holding_buf: cutlass.Int32


class Indexer_kvsplit:    
    def __init__(self):
        self.top_k = TOP_K
        self.dim_split = DIM_SPLIT
        self.page_size = PAGE_SIZE
        
        self.indexer_threads = 512
        self.pass_through_threads = 1024
        self.topk_threads = 1024
        
        self.wsize = cute.arch.WARP_SIZE
        
        self.limit_request = LIMIT_REQUEST
        self.limit_seq_len = LIMIT_SEQ_LEN

        # Score-kernel MMA params
        self.fp8_dtype          = cutlass.Float8E4M3FN
        self.acc_dtype          = cutlass.Float32
        self.num_stages         = 1
        self.tmem_ld_rep        = TMEM_LD_REP
        self.cta_tile_mnk       = (BM, BN, BK)
        self.mma_inst_shape_mnk = MMA_INST_MNK
        
        # Workspace
        self.ws_score_output = torch.empty(LIMIT_REQUEST, LIMIT_SEQ_LEN, dtype=torch.float32, device="cuda")
        
    @cute.jit
    def __call__(
        self,
        q_index_fp8,        # (T, NUM_HEADS, HEAD_DIM)
        k_index_cache_fp8,  # (NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM+4)
        weights,            # (T, NUM_HEADS)
        seq_lens,           # (T)
        block_table,        # (T, max_num_pages)
        score_output,       # (MAX_REQUEST, MAX_SEQ_LEN)
        top_k_indices,      # (T, 2048)
        stream
        ):
        
        T, max_num_pages = block_table.shape
        pages_per_split = self.dim_split // self.page_size
        num_splits = (max_num_pages + pages_per_split - 1) // pages_per_split

        # Build score-kernel MMA + smem layouts (compile-time, hoisted out of
        # dynamic if-else so self isn't mutated within a dynamic branch).
        op = tcgen05.MmaFP8Op(
            self.fp8_dtype, self.acc_dtype, self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma     = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.cta_tile_mnk, self.fp8_dtype, self.num_stages)
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.cta_tile_mnk, q_index_fp8.element_type, self.num_stages)

        if max_num_pages <= 32:
            self.pass_through_kernel(seq_lens, block_table, top_k_indices).launch(
                grid=[T, 1, 1], block=[1024, 1, 1], stream=stream
            )
        else:
            self.indexer_ksplit_kernel(
                q_index_fp8, k_index_cache_fp8, weights,
                seq_lens, block_table, num_splits, score_output, top_k_indices,
                tiled_mma, a_smem_layout, b_smem_layout,
            ).launch(
                grid=[T + num_splits, 1, 1], block=[self.indexer_threads, 1, 1], stream=stream
            )
            self.topk_kernel(seq_lens, block_table, num_splits, score_output, top_k_indices).launch(
                grid=[T, 1, 1], block=[self.topk_threads, 1, 1], stream=stream
            )
    
    @staticmethod
    def _smem(allocator, dtype, shape, stride, align):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)
        
    @cute.kernel
    def pass_through_kernel(
        self,
        seq_lens,           # (T)
        block_table,        # (T, max_num_pages)
        topk_indices        # (T, 2048) Int32
        ):
        # This kernel gather the indices to topk_indices, the right most is padded with -1 sentinals
        
        top_k_len:      cutlass.Constexpr = self.top_k
        T, max_num_pages = block_table.shape
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()        
        
        max_seq_len = seq_lens[bidx]
        
        alloc = cutlass.utils.SmemAllocator()
        smem_sparse = self._smem(alloc, cutlass.Int32, (top_k_len, ),      (1,), 4)
        smem_page   = self._smem(alloc, cutlass.Int32, (top_k_len // 64,), (1,), 4)        
        for i in range(tidx, top_k_len, self.pass_through_threads):
            smem_sparse[i] = - 1
        
        for j in range(tidx, top_k_len // 64, self.pass_through_threads):
            smem_page[j] = block_table[bidx, j]
            
        cute.arch.sync_threads()
        
        # Each warp handles one page; guard against warps beyond max_num_pages
        if warp_idx < max_num_pages:
            page_idx = smem_page[warp_idx]
            # Compute the token range for this page
            page_start = warp_idx * cutlass.Int32(PAGE_SIZE)
            page_end   = page_start + cutlass.Int32(PAGE_SIZE)
            if page_end > max_seq_len:
                page_end = max_seq_len
            for i in range(lane_idx, page_end - page_start, self.wsize):
                token_idx = page_start + i
                if token_idx < max_seq_len:
                    smem_sparse[token_idx] = page_idx * cutlass.Int32(PAGE_SIZE) + i
                
        cute.arch.sync_threads()
        
        for i in range(tidx, top_k_len, self.pass_through_threads):
            topk_indices[bidx, i] = smem_sparse[i] 
    
    @cute.kernel
    def indexer_ksplit_kernel(
        self,
        q_index_fp8,        # (T, NUM_HEADS, HEAD_DIM) fp8
        k_index_cache_fp8,  # (NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM+4) i8 — flat byte pool
        weights,            # (T, NUM_HEADS) f32
        seq_lens,           # (T,) i32
        block_table,        # (T, max_num_pages) i32
        num_splits,         # int32
        score_output,       # (MAX_REQUEST, MAX_SEQ_LEN) f32
        topk_indices,       # (T, 2048) Int32
        tiled_mma,
        a_smem_layout,
        b_smem_layout,
    ):
        top_k_len:      cutlass.Constexpr = self.top_k
        limit_request:      cutlass.Constexpr = self.limit_request
        
        T, max_num_pages = block_table.shape

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_blocks, _, _ = cute.arch.grid_dim()
        
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()        
        
        # ── SMEM allocation (order matters for sB swizzle alignment) ──
        # sA  : at offset 0       — Sw<3,2,3>∘row_major i32 view safe (offset 0)
        # storage : mbar+tmem buf
        # sB  : 1024-aligned      — Sw<3,2,3> i32 view safe (multiple of swizzle period)
        # sScales, sWeights       — score-path scratch
        # smem_indexer_T_idx, smem_num_idxer — score-path metadata
        # smem_sparse, smem_page  — pass-through scratch
        alloc = cutlass.utils.SmemAllocator()
        sA = alloc.allocate_tensor(
            element_type=self.fp8_dtype, layout=a_smem_layout.outer,
            byte_alignment=128, swizzle=a_smem_layout.inner,
        )
        sA_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BM, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1)),
        )
        sA_load = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=cutlass.Int32), sA_load_layout)

        sMmaMbar = alloc.allocate_tensor(
            cutlass.Int64, cute.make_layout(2 * self.num_stages), 16, None)
        sTmemHold = alloc.allocate_tensor(
            cutlass.Int32, cute.make_layout(1), 16, None)

        sB = alloc.allocate_tensor(
            self.fp8_dtype, b_smem_layout.outer, 1024, b_smem_layout.inner)
        sB_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BN, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1)),
        )
        sB_load = cute.make_tensor(
            cute.recast_ptr(sB.iterator, dtype=cutlass.Int32), sB_load_layout)

        sScales  = alloc.allocate_tensor(cutlass.Float32,
                       cute.make_layout(self.indexer_threads), 16, None)
        sWeights = alloc.allocate_tensor(cutlass.Float32,
                       cute.make_layout(BN), 16, None)

        smem_indexer_T_idx = self._smem(alloc, cutlass.Int32, (limit_request,), (1,), 4)
        smem_num_idxer     = self._smem(alloc, cutlass.Int32, (1,),             (1,), 4)
        smem_sparse        = self._smem(alloc, cutlass.Int32, (top_k_len, ),      (1,), 4)
        smem_page          = self._smem(alloc, cutlass.Int32, (top_k_len // 64,), (1,), 4)
                
        # Example: If num requests = 128, max seq len = 4096, dim split = 128 -> 32 splits
        # Then we launch 32 + 128 blocks
        # The first 32 blocks will persistently compute indexer score across all requests with seq len > 2048
        # The last 128 blocks will write pass through indices to requests with seq len < 2048
        
        # pass through blocks
        # Here, the block indices will map to request indices
        if bidx >= num_splits:
            bidx_pass = bidx - num_splits
            max_seq_len = seq_lens[bidx_pass]
            
            if max_seq_len <= 2048:
                for i in range(tidx, top_k_len, self.indexer_threads):
                    smem_sparse[i] = - 1
                
                for j in range(tidx, top_k_len // 64, self.indexer_threads):
                    smem_page[j] = block_table[bidx_pass, j]
                    
                cute.arch.sync_threads()
                
                # Each warp handles one page; guard against warps beyond max_num_pages
                if warp_idx < max_num_pages:
                    page_idx   = smem_page[warp_idx]
                    page_start = warp_idx * cutlass.Int32(PAGE_SIZE)
                    page_end   = page_start + cutlass.Int32(PAGE_SIZE)
                    if page_end > max_seq_len:
                        page_end = max_seq_len
                    for i in range(lane_idx, page_end - page_start, self.wsize):
                        token_idx = page_start + i
                        if token_idx < max_seq_len:
                            smem_sparse[token_idx] = page_idx * cutlass.Int32(PAGE_SIZE) + i
                cute.arch.sync_threads()
                
                for i in range(tidx, top_k_len, self.indexer_threads):
                    topk_indices[bidx_pass, i] = smem_sparse[i]
        
        # indexer score calculation blocks
        # Here, the block indices will map to split_idx
        else:            
            # Step 1: warp 0 scans seq_lens, compacts indices > 2048 into
            # smem_indexer_T_idx[], publishes total count to smem_num_idxer[0].
            # T ≤ limit_request = 128 → at most 4 chunks of 32 lanes.
            if warp_idx == 0:
                base = cutlass.Int32(0)
                for chunk_start in cutlass.range_constexpr(0, limit_request, 32):
                    i = cutlass.Int32(chunk_start) + lane_idx
                    is_idx = cutlass.Int32(0)
                    if i < T:
                        if seq_lens[i] > cutlass.Int32(2048):
                            is_idx = cutlass.Int32(1)
                    # Inclusive scan within warp via shuffle-up
                    scan = is_idx
                    for s in cutlass.range_constexpr(5):
                        peer = cute.arch.shuffle_sync_up(scan, 1 << s, mask_and_clamp=0)
                        if lane_idx >= cutlass.Int32(1 << s):
                            scan = scan + peer
                    excl = scan - is_idx
                    if is_idx != cutlass.Int32(0):
                        smem_indexer_T_idx[base + excl] = i
                    base = base + cute.arch.shuffle_sync(scan, 31)
                if lane_idx == cutlass.Int32(0):
                    smem_num_idxer[0] = base
            cute.arch.sync_threads()
            num_idxer_requests = smem_num_idxer[0]

            # ── Score path one-time setup (mirrors score_scale_full_bt_ws_cpasync_persist.py) ──
            tCrA   = tiled_mma.make_fragment_A(sA)
            tCrB   = tiled_mma.make_fragment_B(sB)
            acc_shape       = tiled_mma.partition_shape_C(self.cta_tile_mnk[:2])
            tCtAcc          = tiled_mma.make_fragment_C(acc_shape)
            num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc)
            tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

            mma_mbar = sMmaMbar.iterator

            if warp_idx == 0:
                cute.arch.alloc_tmem(tmem_alloc_cols, sTmemHold.iterator)
                if tidx == 0:
                    cute.arch.mbarrier_init(mma_mbar, cnt=1)
                    cute.arch.mbarrier_init_fence()

            cute.arch.barrier(barrier_id=INIT_BAR_ID,
                              number_of_threads=self.indexer_threads)

            tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, alignment=16,
                ptr_to_buffer_holding_addr=sTmemHold.iterator)
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
            N_PER_THREAD_I32 = (BM * HEAD_DIM_I32) // self.indexer_threads   # 8
            thr_layout_load  = cute.make_layout((16, HEAD_DIM_I32),
                                                stride=(HEAD_DIM_I32, 1))
            val_layout_load  = cute.make_layout((N_PER_THREAD_I32, 1),
                                                stride=(1, 1))
            tiled_copy_a = cute.make_tiled_copy_tv(atom_cpa, thr_layout_load, val_layout_load)
            thr_copy_a   = tiled_copy_a.get_slice(tidx)
            tAsA = thr_copy_a.partition_D(sA_load)

            # cp.async B (q) plumbing
            N_PER_THREAD_I32_B = (BN * HEAD_DIM_I32) // self.indexer_threads   # 4
            atom_cpb = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
                cutlass.Int32, num_bits_per_copy=cutlass.Int32.width,
            )
            thr_layout_load_b = cute.make_layout(
                (BN, HEAD_DIM_I32 // N_PER_THREAD_I32_B),
                stride=(HEAD_DIM_I32 // N_PER_THREAD_I32_B, 1),
            )
            val_layout_load_b = cute.make_layout(
                (1, N_PER_THREAD_I32_B), stride=(1, 1),
            )
            tiled_copy_b = cute.make_tiled_copy_tv(atom_cpb, thr_layout_load_b, val_layout_load_b)
            thr_copy_b   = tiled_copy_b.get_slice(tidx)
            tBsB_dst     = thr_copy_b.partition_D(sB_load)

            # ── Persistent loop over indexer requests ──
            # phase counter only advances when MMA actually fires (when bidx is
            # within the request's split range). Skipped iters do nothing.
            phase = cutlass.Int32(0)
            for indexer_request in range(num_idxer_requests):
                T_idx = smem_indexer_T_idx[indexer_request]
                req_seq_len = seq_lens[T_idx]
                # request_num_tiles = ceil(seq_len / BM); only run if this split
                # has any tokens for this request.
                request_num_tiles = (req_seq_len + cutlass.Int32(BM - 1)) // cutlass.Int32(BM)
                if bidx < request_num_tiles:
                    # ── Per-request page IDs for this tile (bidx) ──
                    page0_id = cutlass.Int32(block_table[T_idx, bidx * PAGES_PER_TILE + 0])
                    page1_id = cutlass.Int32(block_table[T_idx, bidx * PAGES_PER_TILE + 1])
                    page_stride_b   = PAGE_SIZE * ROW_STRIDE
                    page_stride_i32 = page_stride_b // 4
                    page0_off_i32   = page0_id * page_stride_i32
                    jump_i32        = (page1_id - page0_id) * page_stride_i32

                    # ── A-load (cp.async i32 view) ──
                    i32_base = cute.make_ptr(
                        cutlass.Int32,
                        (cute.recast_ptr(k_index_cache_fp8.iterator, dtype=cutlass.Int32) + page0_off_i32).toint(),
                        mem_space=cute.AddressSpace.gmem, assumed_align=4,
                    )
                    gA_i32 = cute.make_tensor(i32_base, cute.make_layout(
                        ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM_I32),
                        stride=((ROW_STRIDE_I32, jump_i32), 1),
                    ))
                    tAgA = thr_copy_a.partition_S(gA_i32)
                    cute.copy(atom_cpa, tAgA, tAsA)

                    # ── B-load: q[T_idx, :, :] via 32b cp.async, swizzled sB dest ──
                    q_req_off_i32 = T_idx * (BN * HEAD_DIM_I32)
                    gB_i32_ptr = cute.make_ptr(
                        cutlass.Int32,
                        (cute.recast_ptr(q_index_fp8.iterator, dtype=cutlass.Int32) + q_req_off_i32).toint(),
                        mem_space=cute.AddressSpace.gmem, assumed_align=4,
                    )
                    gB_i32 = cute.make_tensor(gB_i32_ptr, cute.make_layout(
                        (BN, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1),
                    ))
                    tBgB = thr_copy_b.partition_S(gB_i32)
                    cute.copy(atom_cpb, tBgB, tBsB_dst)

                    # ── Per-tile scales (last 4B of each row in kv_pool) ──
                    SCALE_ROW_STRIDE_F32 = ROW_STRIDE // 4
                    page_stride_f32      = PAGE_SIZE * SCALE_ROW_STRIDE_F32
                    page0_off_f32        = page0_id * page_stride_f32
                    jump_f32             = (page1_id - page0_id) * page_stride_f32
                    fp32_base = cute.recast_ptr(k_index_cache_fp8.iterator, dtype=cutlass.Float32) + page0_off_f32
                    scale_ptr = fp32_base + (HEAD_DIM // 4)
                    scale_layout = cute.make_layout(((PAGE_SIZE, PAGES_PER_TILE),),
                                                    stride=((SCALE_ROW_STRIDE_F32, jump_f32),))
                    gScale = cute.make_tensor(scale_ptr, scale_layout)
                    if tidx < BM:
                        sScales[tidx] = gScale[tidx]

                    # ── Per-request weights ──
                    if tidx < BN:
                        sWeights[tidx] = weights[T_idx, tidx]

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
                    if tidx < self.indexer_threads // 4:   # COMPUTE_THREADS = 128
                        cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)

                        scale   = sScales[tidx]
                        out_val = cutlass.Float32(0)
                        for n_idx in cutlass.range_constexpr(BN):
                            val      = tTR_rAcc[n_idx] * scale
                            out_val  = out_val + max(val, cutlass.Float32(0)) * sWeights[n_idx]

                        m_out = bidx * BM + tidx
                        score_output[T_idx, m_out] = out_val

                    cute.arch.sync_threads()
                    phase = phase ^ cutlass.Int32(1)

            # ── One-time teardown ──
            if warp_idx == 0:
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)
            
    @cute.kernel
    def topk_kernel(
        self,
        seq_lens,           # (T)
        block_table,        # (T, max_num_pages)
        num_splits,         # int32
        score_output,       # (MAX_REQUEST, MAX_SEQ_LEN)
        topk_indices        # (T, 2048) Int32
    ):
        top_k_len:    cutlass.Constexpr = self.top_k
        topk_threads: cutlass.Constexpr = self.topk_threads
        num_warps:    cutlass.Constexpr = self.topk_threads // 32

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        if seq_lens[bidx] > cutlass.Int32(2048):
            sl      = seq_lens[bidx]
            max_col = score_output.shape[1]

            allocator      = cutlass.utils.SmemAllocator()
            smem_warp_bins = allocator.allocate_tensor(
                cutlass.Int32, cute.make_layout((num_warps * 4,), stride=(1,)), 4, None)
            smem_bins      = allocator.allocate_tensor(
                cutlass.Int32, cute.make_layout((4,), stride=(1,)), 4, None)
            smem_warp_above = allocator.allocate_tensor(
                cutlass.Int32, cute.make_layout((num_warps,), stride=(1,)), 4, None)
            smem_warp_tie   = allocator.allocate_tensor(
                cutlass.Int32, cute.make_layout((num_warps,), stride=(1,)), 4, None)
            smem_above_round = allocator.allocate_tensor(
                cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)
            smem_tie_round   = allocator.allocate_tensor(
                cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)

            # ── Phase 1: radix select with early exit ────────────────────
            desired      = cutlass.Uint32(0)
            desired_mask = cutlass.Uint32(0)
            k_to_find    = cutlass.Int32(top_k_len)

            pass_idx = cutlass.Int32(0)
            while pass_idx < cutlass.Int32(16):
                digit_pos   = cutlass.Int32(30) - pass_idx * cutlass.Int32(2)
                digit_pos_u = cutlass.Uint32(digit_pos)

                if tidx < cutlass.Int32(4):
                    smem_bins[tidx] = cutlass.Int32(0)
                cute.arch.sync_threads()

                c0 = cutlass.Int32(0); c1 = cutlass.Int32(0)
                c2 = cutlass.Int32(0); c3 = cutlass.Int32(0)

                base = tidx * cutlass.Int32(4)
                while base + cutlass.Int32(3) < sl:
                    bits0 = float_to_radix(score_output[bidx, base])
                    bits1 = float_to_radix(score_output[bidx, base + cutlass.Int32(1)])
                    bits2 = float_to_radix(score_output[bidx, base + cutlass.Int32(2)])
                    bits3 = float_to_radix(score_output[bidx, base + cutlass.Int32(3)])
                    c0, c1, c2, c3 = count_element(bits0, desired, desired_mask, digit_pos_u, c0, c1, c2, c3)
                    c0, c1, c2, c3 = count_element(bits1, desired, desired_mask, digit_pos_u, c0, c1, c2, c3)
                    c0, c1, c2, c3 = count_element(bits2, desired, desired_mask, digit_pos_u, c0, c1, c2, c3)
                    c0, c1, c2, c3 = count_element(bits3, desired, desired_mask, digit_pos_u, c0, c1, c2, c3)
                    base = base + cutlass.Int32(topk_threads * 4)

                while base < sl:
                    bits = float_to_radix(score_output[bidx, base])
                    c0, c1, c2, c3 = count_element(bits, desired, desired_mask, digit_pos_u, c0, c1, c2, c3)
                    base = base + cutlass.Int32(1)

                c0 = warp_sum_i32(c0); c1 = warp_sum_i32(c1)
                c2 = warp_sum_i32(c2); c3 = warp_sum_i32(c3)

                if lane_idx == cutlass.Int32(0):
                    smem_warp_bins[warp_idx * cutlass.Int32(4) + 0] = c0
                    smem_warp_bins[warp_idx * cutlass.Int32(4) + 1] = c1
                    smem_warp_bins[warp_idx * cutlass.Int32(4) + 2] = c2
                    smem_warp_bins[warp_idx * cutlass.Int32(4) + 3] = c3
                cute.arch.sync_threads()

                if warp_idx == cutlass.Int32(0):
                    g0 = smem_warp_bins[lane_idx * cutlass.Int32(4) + 0]
                    g1 = smem_warp_bins[lane_idx * cutlass.Int32(4) + 1]
                    g2 = smem_warp_bins[lane_idx * cutlass.Int32(4) + 2]
                    g3 = smem_warp_bins[lane_idx * cutlass.Int32(4) + 3]
                    g0 = warp_sum_i32(g0); g1 = warp_sum_i32(g1)
                    g2 = warp_sum_i32(g2); g3 = warp_sum_i32(g3)
                    if lane_idx == cutlass.Int32(0):
                        smem_bins[0] = g0; smem_bins[1] = g1
                        smem_bins[2] = g2; smem_bins[3] = g3
                cute.arch.sync_threads()

                g0 = smem_bins[0]; g1 = smem_bins[1]
                g2 = smem_bins[2]; g3 = smem_bins[3]
                cute.arch.sync_threads()

                dp_u    = cutlass.Uint32(digit_pos)
                shifted = cutlass.Uint32(3) << dp_u
                inv_sh  = shifted ^ cutlass.Uint32(0xFFFFFFFF)

                chosen_count = cutlass.Int32(0)
                if g3 >= k_to_find:
                    desired      = (desired & inv_sh) | (cutlass.Uint32(3) << dp_u)
                    desired_mask = desired_mask | shifted
                    chosen_count = g3
                else:
                    k_to_find = k_to_find - g3
                    if g2 >= k_to_find:
                        desired      = (desired & inv_sh) | (cutlass.Uint32(2) << dp_u)
                        desired_mask = desired_mask | shifted
                        chosen_count = g2
                    else:
                        k_to_find = k_to_find - g2
                        if g1 >= k_to_find:
                            desired      = (desired & inv_sh) | (cutlass.Uint32(1) << dp_u)
                            desired_mask = desired_mask | shifted
                            chosen_count = g1
                        else:
                            k_to_find = k_to_find - g1
                            desired      = desired & inv_sh
                            desired_mask = desired_mask | shifted
                            chosen_count = g0

                if chosen_count == k_to_find:
                    pass_idx = cutlass.Int32(16)
                else:
                    pass_idx = pass_idx + cutlass.Int32(1)

            above_total = cutlass.Int32(top_k_len) - k_to_find
            need_ties   = k_to_find
            desired_pin = desired & desired_mask

            # ── Phase 2 (fused, mask-aware): single scatter pass ──────────
            above_cursor = cutlass.Int32(0)
            tie_cursor   = cutlass.Int32(0)

            col = cutlass.Int32(0)
            while col < max_col:
                cur_col  = col + tidx
                is_valid = cur_col < sl

                bits = cutlass.Uint32(0)
                if is_valid:
                    bits = float_to_radix(score_output[bidx, cur_col])

                is_b = cutlass.Int32(0)
                is_t = cutlass.Int32(0)
                if is_valid:
                    masked = bits & desired_mask
                    if masked > desired_pin:
                        is_b = cutlass.Int32(1)
                    if masked == desired_pin:
                        is_t = cutlass.Int32(1)

                scan_b = is_b
                for s in cutlass.range_constexpr(5):
                    peer = cute.arch.shuffle_sync_up(scan_b, 1 << s, mask_and_clamp=0)
                    if lane_idx >= cutlass.Int32(1 << s):
                        scan_b = scan_b + peer
                my_b_excl  = scan_b - is_b
                warp_b_tot = cute.arch.shuffle_sync(scan_b, 31)

                scan_t = is_t
                for s in cutlass.range_constexpr(5):
                    peer2 = cute.arch.shuffle_sync_up(scan_t, 1 << s, mask_and_clamp=0)
                    if lane_idx >= cutlass.Int32(1 << s):
                        scan_t = scan_t + peer2
                my_t_excl  = scan_t - is_t
                warp_t_tot = cute.arch.shuffle_sync(scan_t, 31)

                if lane_idx == cutlass.Int32(31):
                    smem_warp_above[warp_idx] = warp_b_tot
                    smem_warp_tie[warp_idx]   = warp_t_tot
                cute.arch.sync_threads()

                if warp_idx == cutlass.Int32(0):
                    wta      = smem_warp_above[lane_idx]
                    orig_wta = wta
                    for s in cutlass.range_constexpr(5):
                        p = cute.arch.shuffle_sync_up(wta, 1 << s, mask_and_clamp=0)
                        if lane_idx >= cutlass.Int32(1 << s):
                            wta = wta + p
                    smem_warp_above[lane_idx] = wta - orig_wta
                    above_round_tot = warp_sum_i32(orig_wta)
                    if lane_idx == cutlass.Int32(0):
                        smem_above_round[0] = above_round_tot

                    wtt      = smem_warp_tie[lane_idx]
                    orig_wtt = wtt
                    for s in cutlass.range_constexpr(5):
                        p2 = cute.arch.shuffle_sync_up(wtt, 1 << s, mask_and_clamp=0)
                        if lane_idx >= cutlass.Int32(1 << s):
                            wtt = wtt + p2
                    smem_warp_tie[lane_idx] = wtt - orig_wtt
                    tie_round_tot = warp_sum_i32(orig_wtt)
                    if lane_idx == cutlass.Int32(0):
                        smem_tie_round[0] = tie_round_tot
                cute.arch.sync_threads()

                warp_b_off = smem_warp_above[warp_idx]
                warp_t_off = smem_warp_tie[warp_idx]

                if is_b > cutlass.Int32(0):
                    goff = above_cursor + warp_b_off + my_b_excl
                    if goff < above_total:
                        topk_indices[bidx, goff] = cur_col

                if is_t > cutlass.Int32(0):
                    toff    = tie_cursor + warp_t_off + my_t_excl
                    wrt_pos = above_total + toff
                    if toff < need_ties:
                        if wrt_pos < cutlass.Int32(top_k_len):
                            topk_indices[bidx, wrt_pos] = cur_col

                above_round  = smem_above_round[0]
                tie_round    = smem_tie_round[0]
                cute.arch.sync_threads()

                above_cursor = above_cursor + above_round
                tie_cursor   = tie_cursor   + tie_round
                col          = col + cutlass.Int32(topk_threads)

# TODO
# tvm ffi compile code
def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)

def compile_hybrid():
    T             = cute.sym_int()
    max_num_pages = cute.sym_int()
    num_pages     = cute.sym_int()

    q_index_fp8       = _fake(cute.Float8E4M3FN, (T, NUM_HEADS, HEAD_DIM),          (2, 1, 0), 16)
    k_index_cache_fp8 = _fake(cute.Int8,         (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4), (3, 2, 1, 0), 16)
    weights           = _fake(cute.Float32,      (T, NUM_HEADS),                    (1, 0),    4)
    seq_lens          = _fake(cute.Int32,        (T,),                              (0,),      4)
    block_table       = _fake(cute.Int32,        (T, max_num_pages),                (1, 0),    4)
    score_output      = _fake(cute.Float32,      (LIMIT_REQUEST, LIMIT_SEQ_LEN),    (1, 0),    16)
    top_k_indices     = _fake(cute.Int32,        (T, TOP_K),                        (1, 0),    4)
    stream            = make_fake_stream(use_tvm_ffi_env_stream=True)

    indexer = Indexer_kvsplit()

    compiled = cute.compile(
        indexer,
        q_index_fp8, k_index_cache_fp8, weights,
        seq_lens, block_table, score_output, top_k_indices, stream,
        options="--enable-tvm-ffi"
    )
    return indexer, compiled


_indexer, _compiled = compile_hybrid()

def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    _compiled(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, _indexer.ws_score_output, topk_indices)
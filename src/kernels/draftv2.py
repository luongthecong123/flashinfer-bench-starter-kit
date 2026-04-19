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
        if max_num_pages <= 32:
            self.pass_through_kernel(seq_lens, block_table, top_k_indices).launch(
                grid=[T, 1, 1], block=[1024, 1, 1], stream=stream
            )
        else:
            
            self.indexer_ksplit_kernel(seq_lens, block_table, num_splits, top_k_indices, score_output).launch(
                grid=[T + num_splits, 1, 1], block=[self.indexer_threads, 1, 1], stream=stream
            )
            self.topk_kernel(seq_lens, block_table, num_splits, top_k_indices, score_output).launch(
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
        
        # Each warp writes a page, 32 warps -> 32 pages
        if smem_page[warp_idx] != 0:
            page_idx = smem_page[warp_idx]
            for i in range(lane_idx, max_seq_len, self.wsize):
                smem_sparse[i] = page_idx * 64 + i
                
        cute.arch.sync_threads()
        
        for i in range(tidx, top_k_len, self.pass_through_threads):
            topk_indices[bidx, i] = smem_sparse[i] 
    
    @cute.kernel
    def indexer_ksplit_kernel(
        self,
        seq_lens,           # (T)
        block_table,        # (T, max_num_pages)
        num_splits,         # int32
        score_output,       # (MAX_REQUEST, MAX_SEQ_LEN)
        topk_indices        # (T, 2048) Int32
    ):
        top_k_len:      cutlass.Constexpr = self.top_k
        limit_request:      cutlass.Constexpr = self.limit_request
        
        T, max_num_pages = block_table.shape

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_blocks, _, _ = cute.arch.grid_dim()
        
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()        
        
        
        
        alloc = cutlass.utils.SmemAllocator()
        smem_sparse = self._smem(alloc, cutlass.Int32, (top_k_len, ),      (1,), 4)
        smem_page   = self._smem(alloc, cutlass.Int32, (top_k_len // 64,), (1,), 4)  
        smem_indexer_T_idx = self._smem(alloc, cutlass.Int32, (limit_request),      (1,), 4)
                
        # Example: If num requests = 128, max seq len = 4096, dim split = 128 -> 32 splits
        # Then we launch 32 + 128 blocks
        # The first 32 blocks will persistently compute indexer score across all requests with seq len > 2048
        # The last 128 blocks will write pass through indices to requests with seq len < 2048
        
        # pass through blocks
        # Here, the block indices will map to request indices
        if bidx > num_splits:
            bidx_pass -= num_blocks - num_splits
            max_seq_len = seq_lens[bidx_pass]
            
            if max_seq_len <= 2048:
                for i in range(tidx, top_k_len, self.topk_threads):
                    smem_sparse[i] = - 1
                
                for j in range(tidx, top_k_len // 64, self.topk_threads):
                    smem_page[j] = block_table[bidx_pass, j]
                    
                cute.arch.sync_threads()
                
                # Each warp writes a page, 16 warps -> 32 pages
                num_rounds = ((top_k_len // 64) // (self.topk_threads // self.wsize))
                for round_idx in range(num_rounds):
                    if smem_page[round_idx * num_rounds + warp_idx] != 0:
                        page_idx = smem_page[round_idx * num_rounds + warp_idx]
                        for i in range(lane_idx, max_seq_len, self.wsize):
                            smem_sparse[i] = page_idx * 64 + i       
                cute.arch.sync_threads()
                
                for i in range(tidx, top_k_len, self.topk_threads):
                    topk_indices[bidx_pass, i] = smem_sparse[i]
        
        # indexer score calculation blocks
        # Here, the block indices will map to split_idx
        else:            
            # Step 1: Find num requests that has seq len > 2048 and their respective indices in the block_table
            
            # If first request is 0, store the request indices to position num_idxer_requests in smem_indexer_T_idx
            num_idxer_requests = 0
            for i in range(tidx, T, self.topk_threads):
                if seq_lens[tidx] > 2048:
                    smem_indexer_T_idx[num_idxer_requests] = i
                    num_idxer_requests += 1                    
            cute.arch.sync_threads()
            
            # TODO
            # Step 2: compute score persisently across num_idxer_requests
            # src/kernels/score_scale_full_bt_ws_cpasync.py
            for indexer_request in range(num_idxer_requests):
                T_idx = smem_indexer_T_idx[indexer_request]
                split_idx_old = bidx
                split_idx_new = (T_idx + split_idx_old) % num_splits

                # TODO
                # Step 3: Store results to appropriate split indices
            
    @cute.kernel
    def topk_kernel(
        self,
        seq_lens,           # (T)
        block_table,        # (T, max_num_pages)
        num_splits,         # int32
        score_output,       # (MAX_REQUEST, MAX_SEQ_LEN)
        topk_indices        # (T, 2048) Int32
    ):
        top_k_len:      cutlass.Constexpr = self.top_k
        limit_request:      cutlass.Constexpr = self.limit_request
        
        T, max_num_pages = block_table.shape

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        num_blocks, _, _ = cute.arch.grid_dim()            
        
        # TODO
        # find top k on score_output
        
        # Write results back to topk_indices
        # src/kernels/topk_aten_cutedsl_v4_fuse_earlyexit.py

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
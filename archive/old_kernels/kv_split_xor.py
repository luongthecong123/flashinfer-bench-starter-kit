import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN = 16, 512, 64, 2048
NUM_PAGES, PAGE_SIZE = 8462, 64
T_MAX = 8
NUM_SPLITS = 8
DIM_SPLIT = (TOP_K_LEN + NUM_SPLITS - 1) // NUM_SPLITS
# NUM_SPLITS = 9
# DIM_SPLIT = 228
LN2 = 0.6931471805599453

# https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md
@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        # cute.arch.shuffle_sync_bfly will read from another thread's registers
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val

@cute.jit
def count_valid_indices(
    sparse_indices:   cute.Tensor,        # (T, top_k_len) i32  — global
    smem_sparse:      cute.Tensor,        # (T_max, top_k_len) i32 — smem cache
    smem_red_i32:     cute.Tensor,        # (T_max, 32) i32     — smem scratch
    smem_num_valid:   cute.Tensor,        # (T_max,) i32        — smem output
    T:                cute.Numeric,
    tidx:             cute.Numeric,       # flat thread index (threadIdx.x)
    warp_idx:         cute.Numeric,       # warp index (warp-uniform)
    top_k_len:        cutlass.Constexpr,
    sparse_thr_per_T: cutlass.Constexpr,
    num_warps_per_T:  cutlass.Constexpr,
) -> None:
    """
    Load sparse_indices into smem_sparse and count non-negative entries
    """
    thr_idx_per_T  = tidx % sparse_thr_per_T
    lane_idx_per_T = thr_idx_per_T % cute.arch.WARP_SIZE
    wg_per_T_idx   = tidx // sparse_thr_per_T
    warp_per_T_idx = warp_idx % num_warps_per_T

    partial_cnt = 0
    if wg_per_T_idx < T:
        for i in range(thr_idx_per_T, top_k_len, sparse_thr_per_T):
            idx = sparse_indices[wg_per_T_idx, i]
            smem_sparse[wg_per_T_idx, i] = idx
            if idx >= cutlass.Int32(0):
                partial_cnt += 1

        # Intra warp reduction
        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
        if lane_idx_per_T == 0:
            smem_red_i32[wg_per_T_idx, warp_per_T_idx] = cnt_sum

        cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                          number_of_threads=sparse_thr_per_T)

        # Inter warp reduction
        if warp_per_T_idx == 0:
            val     = smem_red_i32[wg_per_T_idx, lane_idx_per_T]
            cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps_per_T)
            smem_red_i32[wg_per_T_idx, 0] = cnt_sum

        cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                          number_of_threads=sparse_thr_per_T)

        smem_num_valid[wg_per_T_idx] = smem_red_i32[wg_per_T_idx, 0]

class Kv_split_xor():
    def __init__(
        self):
        self.num_head, self.head_dim_ckv, self.head_dim_kpe, self.top_k_len = NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN
        self.num_pages, self.page_size  = NUM_PAGES, PAGE_SIZE
        self.T_max = T_MAX
        self.num_splits = NUM_SPLITS
        self.dim_split = (self.top_k_len + self.num_splits - 1) // self.num_splits
        self.num_threads = 1024
        self.wsize = cute.arch.WARP_SIZE
        self.num_warps = self.num_threads // self.wsize
        self.vec_size_ckv = 8
        self.vec_size_kpe = 2
        self.vec_size_out = 16
        self.iters_per_lane_ckv = self.head_dim_ckv // (self.wsize * self.vec_size_ckv)
        
        self.sparse_thr_per_T = 128
        self.num_warps_per_T = self.sparse_thr_per_T // self.wsize
        
        assert self.head_dim_ckv // (self.wsize * self.vec_size_ckv) > 0, "head_dim_ckv can't be partitioned to a warp given current vec_size"
        # assert self.top_k_len % self.num_splits == 0, "top_k_len must be divisible by num_splits"
            
    @staticmethod
    def _smem(allocator, dtype, shape, stride, align):
        return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)

    @cute.jit
    def __call__(
        self,
        q_nope:         cute.Tensor,        # (T,16,512)    = (T, num_heads, head_dim_ckv)              bf16
        q_pe:           cute.Tensor,        # (T,16, 64)    = (T, num_heads, head_dim_kpe)              bf16
        ckv_cache:      cute.Tensor,        # (8462,64,512) = (num_pages, page_size, head_dim_ckv)      bf16
        kpe_cache:      cute.Tensor,        # (8462,64, 64) = (num_pages, page_size, head_dim_kpe)      bf16
        sparse_indices: cute.Tensor,        # (T,2048)      = (T, top_k)                                i32
        sm_scale:       cutlass.Constexpr,  # scalar                                                    f32
        partial_out:    cute.Tensor,        # (8,16,8,512)  = (T_MAX, num_heads, num_splits, head_dim)  f32
        partial_lse:    cute.Tensor,        # (8,16,8,  2)  = (T_MAX, num_heads, num_splits, [max,sum]) f32
        output:         cute.Tensor,        # (T,16,512)    = (T, num_heads, head_dim_ckv)              bf16
        lse:            cute.Tensor,        # (T,16)        = (T, num_heads)                            f32
        stream):                            # CUDA stream

        T, _, _ = q_nope.shape
        N = self.num_pages * self.page_size
        ckv_flat = cute.make_tensor(
            ckv_cache.iterator, cute.make_layout((N, self.head_dim_ckv), stride=(self.head_dim_ckv, 1)))
        kpe_flat = cute.make_tensor(
            kpe_cache.iterator, cute.make_layout((N, self.head_dim_kpe), stride=(self.head_dim_kpe, 1)))
                
        self.compute_kernel(
            q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
            partial_out, partial_lse, output, lse
        ).launch(grid=[self.num_head, self.num_splits, 1], block=[self.num_threads, 1, 1], stream=stream)
        
        self.reduce_kernel(
            sparse_indices, partial_out, partial_lse, output, lse
        ).launch(grid=[T, self.num_head, 1], block=[self.num_threads, 1, 1], stream=stream)
    
    @cute.kernel
    def reduce_kernel(
        self,
        sparse_indices: cute.Tensor,        # (T,2048)         = (T, top_k)                                i32
        partial_out:    cute.Tensor,        # (8,16,8,512)     = (T_MAX, num_heads, num_splits, head_dim)  f32
        partial_lse:    cute.Tensor,        # (8,16,8,  2)     = (T_MAX, num_heads, num_splits, [max,sum]) f32
        output:         cute.Tensor,        # (T,16,512)       = (T, num_heads, head_dim_ckv)              bf16
        lse:            cute.Tensor,        # (T,16)           = (T, num_heads)                            f32
    ):
        T, _, _ = output.shape
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()
        
        alloc = cutlass.utils.SmemAllocator()
        smem_red_i32 = self._smem(alloc, cutlass.Int32, (32,), (1,), 4)  #  1 KB
        smem_max_sum = self._smem(alloc, cutlass.Float32, (self.num_splits, 2), (2, 1), 4)  #  1 KB
        
        if tidx < self.num_splits:
            smem_max_sum[tidx, 0] = partial_lse[bidx, bidy, tidx, 0]
            smem_max_sum[tidx, 1] = partial_lse[bidx, bidy, tidx, 1]
        
        partial_cnt = 0
        for i in range(tidx, self.top_k_len, self.num_threads):
            idx = sparse_indices[bidx, i]
            if idx >= cutlass.Int32(0):
                partial_cnt += 1        
        
        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = cnt_sum
        cute.arch.sync_threads()
        
        if warp_idx == 0:
            val = smem_red_i32[lane_idx]
            cnt_sum = warp_reduce(val, lambda a, b: a + b, width=self.num_warps)
            smem_red_i32[0] = cnt_sum
        cute.arch.sync_threads()

        num_valid = smem_red_i32[0]
        is_single_split_request = num_valid < self.dim_split
        
        if not is_single_split_request:
            # Normalize and reduce partial results
            num_active_splits = (num_valid + self.dim_split - 1) // self.dim_split
            
            # Find global max and sum
            g_max = -cutlass.Float32(math.inf)
            for split_idx in range(num_active_splits):
                l_max = smem_max_sum[split_idx, 0]
                if l_max > g_max:
                    g_max = l_max
            
            # g_sum = 0
            # for split_idx in range(num_active_splits):
            #     l_max = smem_max_sum[split_idx, 0]
            #     l_sum = smem_max_sum[split_idx, 1]
            #     g_sum += l_sum * cute.math.exp(l_max - g_max)
            
            # if tidx == 0:
            #     lse[bidx, bidy] = (g_max + cute.math.log(g_sum)) / cutlass.Float32(LN2)
            
            if tidx < self.head_dim_ckv:
                g_lse_sum = cutlass.Float32(0)
                acc = cutlass.Float32(0)
                for split_idx in range(num_active_splits):
                    l_max = smem_max_sum[split_idx, 0]
                    l_sum = smem_max_sum[split_idx, 1]      
                    g_lse_sum += l_sum * cute.math.exp(l_max - g_max)
                    acc += cute.math.exp(l_max - g_max) * partial_out[bidx, bidy, split_idx, tidx]
                
                output[bidx, bidy, tidx] = cutlass.BFloat16(acc / g_lse_sum)
                
                if tidx == 0:
                    lse[bidx, bidy] = (g_max + cute.math.log(g_lse_sum)) / cutlass.Float32(LN2)

                
    @cute.kernel
    def compute_kernel(
        self,
        q_nope:         cute.Tensor,        # (T,16,512)       = (T, num_heads, head_dim_ckv)              bf16
        q_pe:           cute.Tensor,        # (T,16, 64)       = (T, num_heads, head_dim_kpe)              bf16
        ckv_flat:       cute.Tensor,        # (541568,512)     = (num_pages*page_size, head_dim_ckv)       bf16  [flat]
        kpe_flat:       cute.Tensor,        # (541568, 64)     = (num_pages*page_size, head_dim_kpe)       bf16  [flat]
        sparse_indices: cute.Tensor,        # (T,2048)         = (T, top_k)                                i32
        sm_scale:       cutlass.Constexpr,  # scalar                                                       f32
        partial_out:    cute.Tensor,        # (8,16,8,512)     = (T_MAX, num_heads, num_splits, head_dim)  f32
        partial_lse:    cute.Tensor,        # (8,16,8,  2)     = (T_MAX, num_heads, num_splits, [max,sum]) f32
        output:         cute.Tensor,        # (T,16,512)       = (T, num_heads, head_dim_ckv)              bf16
        lse:            cute.Tensor,        # (T,16)           = (T, num_heads)                            f32
    ):
        T, _, _ = q_nope.shape
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        # Cooperative group equivalent — only needed for post-upfront compute
        wg_per_T_idx = tidx // self.sparse_thr_per_T

        # SMEM allocation  (smem_sparse=64KB, smem_q_nope=8KB, smem_partial=64KB, rest~3KB → total~139KB)
        alloc = cutlass.utils.SmemAllocator()
        smem_sparse     = self._smem(alloc, cutlass.Int32,    (self.T_max, self.top_k_len),        (self.top_k_len,    1),  4)  # 64 KB
        smem_num_valid  = self._smem(alloc, cutlass.Int32,    (self.T_max,),                       (1,),                    4)  # 32  B
        smem_logits     = self._smem(alloc, cutlass.Float32,  (self.dim_split,),                   (1,),                   16)  #  1 KB
        smem_red_i32    = self._smem(alloc, cutlass.Int32,    (self.T_max, 32),                    (32,                1),  4)  #  1 KB
        smem_max_red_f32= self._smem(alloc, cutlass.Float32,  (32,),                               (1,),                   16)  # 128 B
        smem_sum_red_f32= self._smem(alloc, cutlass.Float32,  (32,),                               (1,),                   16)  # 128 B
        smem_q_nope     = self._smem(alloc, cutlass.BFloat16, (self.T_max, self.head_dim_ckv),     (self.head_dim_ckv, 1), 16)  #  8 KB
        smem_q_pe       = self._smem(alloc, cutlass.BFloat16, (self.T_max, self.head_dim_kpe),     (self.head_dim_kpe, 1), 16)  #  1 KB
        smem_partial    = self._smem(alloc, cutlass.Float32,  (self.num_warps, self.head_dim_ckv), (self.head_dim_ckv, 1), 16)  # 64 KB
        smem_out        = self._smem(alloc, cutlass.Float32,  (self.head_dim_ckv,),                (1,),                   16)  #  2 KB
        
        # For no bank conflict
        # smem_partial = self._smem(alloc, cutlass.Float32,
        #     (self.num_warps, rest, self.vec_size_ckv),
        #     (rest * (self.vec_size_ckv + 1), self.vec_size_ckv + 1, 1), 16)
                                
        # Load sparse_indices and calculate OOB tiles up front
        head_idx = bidx
        thr_idx_per_T = tidx % self.sparse_thr_per_T
        wg_per_T_idx  = tidx // self.sparse_thr_per_T

        if wg_per_T_idx < T:
            for i in range(thr_idx_per_T, self.head_dim_ckv, self.sparse_thr_per_T):
                smem_q_nope[wg_per_T_idx, i] = q_nope[wg_per_T_idx, head_idx, i]
            for i in range(thr_idx_per_T, self.head_dim_kpe, self.sparse_thr_per_T):
                smem_q_pe[wg_per_T_idx, i] = q_pe[wg_per_T_idx, head_idx, i]

        # Can optimize further by early exit sum = - 16, stop reading, stop writing
        count_valid_indices(
            sparse_indices, smem_sparse, smem_red_i32, smem_num_valid,
            T, tidx, warp_idx,
            self.top_k_len, self.sparse_thr_per_T, self.num_warps_per_T,
        )
        
        cute.arch.sync_threads()
        
        split_idx_old = bidy
        smem_q_nope_ = cute.zipped_divide(smem_q_nope, (1, self.vec_size_ckv)) # ((1, 8), (T_max, 512 // 8))
        ckv_flat_    = cute.zipped_divide(ckv_flat,    (1, self.vec_size_ckv)) # ((1, 8), (541568, 512  // 8))
        kpe_flat_    = cute.zipped_divide(kpe_flat,    (1, self.vec_size_kpe)) # ((1, 2), (541568, 64  // 2)) 
        smem_q_pe_   = cute.zipped_divide(smem_q_pe,   (1, self.vec_size_kpe)) # ((1, 2), (T_max, 64  // 2))
        
        # Persistent kernel along T dimension        
        for T_idx in range(T):
            # Swizzle the split index via rotation (works for any NUM_SPLITS)
            split_idx_new = (T_idx + split_idx_old) % self.num_splits
            
            num_valid_T = smem_num_valid[T_idx]
            split_start = split_idx_new * self.dim_split
            is_OOB = split_start >= num_valid_T            
            
            if not is_OOB:
                # Stage 1: Compute Score
                local_valid = min(num_valid_T - split_start, self.dim_split)
                    
                num_rounds = (local_valid + self.num_warps - 1) // self.num_warps
                
                for round_idx in range(num_rounds):
                    sparse_idx = round_idx * self.num_warps + warp_idx
                    if sparse_idx < local_valid:
                        cur_idx = smem_sparse[T_idx, split_start + sparse_idx]
                        ckv_row_ = ckv_flat_[(0, None), (cur_idx, None)] # (8, 512 // 8)
                        kpe_row_ = kpe_flat_[(0, None), (cur_idx, None)] # (2, 64 // 2)
                        
                        sum_partial = cutlass.Float32(0)
                        
                        for it in range(self.iters_per_lane_ckv):
                            rest_idx = it * self.wsize + lane_idx
                            qn_vec = smem_q_nope_[(0, None), (T_idx, rest_idx)].load()
                            ckv_vec = ckv_row_[None, rest_idx].load()
                            for i in range(self.vec_size_ckv):
                                sum_partial += cutlass.Float32(qn_vec[i]) * cutlass.Float32(ckv_vec[i])
                            
                        qp_vec = smem_q_pe_[(0, None), (T_idx, lane_idx)].load()
                        kpe_vec = kpe_row_[None, lane_idx].load()
                        for i in range(self.vec_size_kpe):
                            sum_partial += cutlass.Float32(qp_vec[i]) * cutlass.Float32(kpe_vec[i])
                        
                        s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                        if lane_idx == 0:
                            smem_logits[sparse_idx] = s * sm_scale
                
                cute.arch.sync_threads()
                
                # Stage 2: Compute softmax
                # 2.1: Find local row max
                partial_max = -cutlass.Float32(math.inf)
                for idx in range(tidx, local_valid, self.num_threads):
                    v = smem_logits[idx]
                    if v > partial_max:
                        partial_max = v

                max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
                if lane_idx == 0:
                    smem_max_red_f32[warp_idx] = max_val
                cute.arch.sync_threads()
                if warp_idx == 0:
                    val = smem_max_red_f32[lane_idx]
                    max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=self.num_warps)
                    smem_max_red_f32[0] = max_val
                cute.arch.sync_threads()

                row_max = smem_max_red_f32[0]
                
                # 2.2: Find local row sum
                local_sum = cutlass.Float32(0)
                for idx in range(tidx, local_valid, self.num_threads):
                    e = cute.math.exp(smem_logits[idx] - row_max)
                    smem_logits[idx] = e
                    local_sum += e

                sum_val = warp_reduce(local_sum, lambda a, b: a + b, width=32)
                if lane_idx == 0:
                    smem_sum_red_f32[warp_idx] = sum_val
                cute.arch.sync_threads()
                if warp_idx == 0:
                    val = smem_sum_red_f32[lane_idx]
                    sum_val = warp_reduce(val, lambda a, b: a + b, width=self.num_warps)
                    smem_sum_red_f32[0] = sum_val
                cute.arch.sync_threads()

                row_sum = smem_sum_red_f32[0]
                
                # Stage 3: Compute output
                out_regs = cute.make_rmem_tensor(cute.make_layout((self.vec_size_out,), stride=(1,)), cutlass.Float32)
                for i in range(self.vec_size_out):
                    out_regs[i] = cutlass.Float32(0)
                    
                for round_idx in range(num_rounds):
                    sparse_idx = round_idx * self.num_warps + warp_idx
                    if sparse_idx < local_valid:
                        cur_idx = smem_sparse[T_idx, split_start + sparse_idx]
                        ckv_row_ = ckv_flat_[(0, None), (cur_idx, None)] # (8, 512 // 8)
                        e = smem_logits[sparse_idx]
                        
                        for it in range(self.iters_per_lane_ckv):
                            rest_idx = it * self.wsize + lane_idx
                            ckv_vec = ckv_row_[None, rest_idx].load()
                            for i in range(self.vec_size_ckv):
                                out_regs[it * self.vec_size_ckv + i] += e * cutlass.Float32(ckv_vec[i])
                
                if warp_idx < local_valid:
                    for it in range(self.iters_per_lane_ckv):
                        for v in range(self.vec_size_ckv):
                            smem_partial[warp_idx, (it * self.wsize + lane_idx) * self.vec_size_ckv + v] = out_regs[it * self.vec_size_ckv + v]
                
                # For no bank conflict
                # --- Store: no bank conflict
                
                # for it in range(self.iters_per_lane_ckv):       # 0..1
                #     for v in range(self.vec_size_ckv):            # 0..7
                #         smem_partial[warp_idx, it * self.wsize + lane_idx, v] = out_regs[it * self.vec_size_ckv + v]
                
                # --- Load: 2-way bank conflict
                
                # for i in range(tidx, self.head_dim_ckv, self.num_threads):
                #     acc = cutlass.Float32(0)
                #     for w in range(self.num_warps):
                #         acc += smem_partial[w, i // self.vec_size_ckv, i % self.vec_size_ckv]
                #     partial_out[T_idx, head_idx, split_idx_new, i] = acc
                
                cute.arch.sync_threads()

                # use 512 threads to accumulate results
                num_active_warps = local_valid if local_valid < self.num_warps else self.num_warps
                for i in range(tidx, self.head_dim_ckv, self.num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_active_warps):
                        acc += smem_partial[w, i]
                    smem_out[i] = acc
                cute.arch.sync_threads()

                is_single_split_request = num_valid_T < self.dim_split
                
                if is_single_split_request and split_idx_new == 0:
                    # If request occupies just a single split, normalize and store directly to gmem
                    for i in range(tidx, self.head_dim_ckv, self.num_threads):
                        output[T_idx, head_idx, i] = cutlass.BFloat16(smem_out[i] / row_sum)
                    
                    if tidx == 0:
                        lse[T_idx, head_idx] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)

                else:
                    # normalize in reduction kernel
                    for i in range(tidx, self.head_dim_ckv, self.num_threads):
                        partial_out[T_idx, head_idx, split_idx_new, i] = smem_out[i]
                    if tidx == 0:
                        partial_lse[T_idx, head_idx, split_idx_new, 0] = row_max
                        partial_lse[T_idx, head_idx, split_idx_new, 1] = row_sum                  


        
# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_kvsplit():
    T = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN
    num_pages, page_size = NUM_PAGES, PAGE_SIZE
    num_splits = NUM_SPLITS

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_ckv), (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_kpe), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, top_k_len),               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    T_MAX = 8
    partial_out    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, head_dim_ckv), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),               (1, 0),     4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    kvsplit_xor = Kv_split_xor()

    return cute.compile(
        kvsplit_xor,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, stream,
        options="--enable-tvm-ffi"
    )


_compiled = compile_kvsplit()

def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    partial_out = torch.empty(T_MAX, NUM_HEADS, NUM_SPLITS, HEAD_DIM_CKV, dtype=torch.float32, device=output.device)
    partial_lse = torch.empty(T_MAX, NUM_HEADS, NUM_SPLITS, 2,             dtype=torch.float32, device=output.device)
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, partial_out, partial_lse, output, lse)

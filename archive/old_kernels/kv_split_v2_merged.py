"""
kv_split_v2_merged.py — Single-compile wrapper: launches compute + reduce
back-to-back from one @cute.jit so NCU captures the real end-to-end latency
including any inter-kernel gap.

Kernel code is identical to kv_split_v2.py.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

import math
import torch

# ── Configuration (same as kv_split_v2) ───────────────────────────────────────
DIM_SPLIT  = 256
TOP_K      = 2048
NUM_SPLITS = (TOP_K + DIM_SPLIT - 1) // DIM_SPLIT  # 8

BLOCK_SIZE_COMPUTE = 1024
NUM_WARPS_COMPUTE  = BLOCK_SIZE_COMPUTE // 32  # 32
DIMS_PER_LANE: cutlass.Constexpr = 512 // 32   # 16

NUM_VEC_SCORE      : cutlass.Constexpr = 8
ITERS_PER_LANE_CKV : cutlass.Constexpr = (512 // 32) // 8   # 2

BLOCK_SIZE_REDUCE = 512

SENTINEL_SKIP = float("inf")
LN2 = 0.6931471805599453


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1: compute_partial (identical to kv_split_v2)
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_compute_partial_kernel(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor):

    T, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = kpe_cache.shape[1]
    dim_split: cutlass.Constexpr = DIM_SPLIT
    num_splits: cutlass.Constexpr = NUM_SPLITS
    num_vec_score: cutlass.Constexpr = NUM_VEC_SCORE
    iters_per_lane_ckv: cutlass.Constexpr = ITERS_PER_LANE_CKV
    dims_per_lane: cutlass.Constexpr = DIMS_PER_LANE

    bidx, bidy, bidz = cute.arch.block_idx()
    num_threads: cutlass.Constexpr = BLOCK_SIZE_COMPUTE
    num_warps:   cutlass.Constexpr = NUM_WARPS_COMPUTE
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    allocator = cutlass.utils.SmemAllocator()

    smem_logits  = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((dim_split,), stride=(1,)), 16, None)
    smem_sparse  = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((dim_split,), stride=(1,)),  4, None)
    smem_red_i32 = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((32,),        stride=(1,)),  4, None)
    smem_red_f32 = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((32,),        stride=(1,)), 16, None)
    smem_q_nope  = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv,), stride=(1,)), 16, None)
    smem_q_pe    = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe,), stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout((num_warps, head_dim_ckv), stride=(head_dim_ckv, 1)), 16, None)

    kv_start = bidz * dim_split
    partial_cnt_valid = 0
    for i in range(tidx, dim_split, num_threads):
        global_idx = kv_start + i
        idx = sparse_indices[bidx, global_idx]
        smem_sparse[i] = idx
        if idx >= cutlass.Int32(0):
            partial_cnt_valid += 1

    for i in range(tidx, head_dim_ckv, num_threads):
        smem_q_nope[i] = q_nope[bidx, bidy, i]
    for i in range(tidx, head_dim_kpe, num_threads):
        smem_q_pe[i] = q_pe[bidx, bidy, i]

    sum_valid = warp_reduce(partial_cnt_valid, lambda a, b: a + b, width=32)
    if lane_idx == 0:
        smem_red_i32[warp_idx] = sum_valid
    cute.arch.sync_threads()

    if warp_idx == 0:
        val = smem_red_i32[lane_idx]
        sum_valid = warp_reduce(val, lambda a, b: a + b, width=num_warps)
        smem_red_i32[0] = sum_valid
    cute.arch.sync_threads()

    valid_count = smem_red_i32[0]

    if tidx == 0:
        next_split_first = sparse_indices[bidx, dim_split]
        if next_split_first < cutlass.Int32(0):
            smem_red_i32[1] = cutlass.Int32(1)
        else:
            smem_red_i32[1] = cutlass.Int32(0)
    cute.arch.sync_threads()

    is_single_split = smem_red_i32[1]

    if valid_count == cutlass.Int32(0):
        for i in range(tidx, head_dim_ckv, num_threads):
            partial_out[bidx, bidy, bidz, i] = cutlass.Float32(0)
        if tidx == 0:
            partial_lse[bidx, bidy, bidz, 0] = -cutlass.Float32(math.inf)
            partial_lse[bidx, bidy, bidz, 1] = cutlass.Float32(0)
    else:
        num_rounds = (valid_count + num_warps - 1) // num_warps
        q_nope_z = cute.zipped_divide(smem_q_nope, (num_vec_score,))

        for round_idx in range(num_rounds):
            sparse_idx = round_idx * num_warps + warp_idx
            if sparse_idx < valid_count:
                cur_idx = smem_sparse[sparse_idx]
                ckv_row = ckv_cache[cur_idx, None]
                ckv_z   = cute.zipped_divide(ckv_row, (num_vec_score,))

                sum_partial = cutlass.Float32(0)
                for it in range(iters_per_lane_ckv):
                    group  = it * wsize + lane_idx
                    q_frag = q_nope_z[(None, (group,))].load()
                    K_frag = ckv_z[(None, (group,))].load()
                    sumSSA = q_frag * K_frag
                    partial = cutlass.Float32(
                        sumSSA.reduce(cute.ReductionOp.ADD, init_val=float(0), reduction_profile=0)
                    )
                    sum_partial = sum_partial + partial

                for k_idx in range(head_dim_kpe // wsize):
                    q_p = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                    kv  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                    sum_partial += q_p * kv

                s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                if lane_idx == 0:
                    smem_logits[sparse_idx] = s * sm_scale

        cute.arch.sync_threads()

        partial_max = -cutlass.Float32(math.inf)
        for idx in range(tidx, valid_count, num_threads):
            v = smem_logits[idx]
            if v > partial_max:
                partial_max = v

        max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
        if lane_idx == 0:
            smem_red_f32[warp_idx] = max_val
        cute.arch.sync_threads()
        if warp_idx == 0:
            val = smem_red_f32[lane_idx]
            max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
            smem_red_f32[0] = max_val
        cute.arch.sync_threads()

        row_max = smem_red_f32[0]

        local_sum = cutlass.Float32(0)
        for idx in range(tidx, valid_count, num_threads):
            smem_logits[idx] = cute.math.exp(smem_logits[idx] - row_max)
            local_sum += smem_logits[idx]

        sum_val = warp_reduce(local_sum, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_f32[warp_idx] = sum_val
        cute.arch.sync_threads()
        if warp_idx == 0:
            val = smem_red_f32[lane_idx]
            sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
            smem_red_f32[0] = sum_val
        cute.arch.sync_threads()

        row_sum = smem_red_f32[0]

        if bidz == cutlass.Int32(0):
            if is_single_split == cutlass.Int32(1):
                for i in range(tidx, valid_count, num_threads):
                    smem_logits[i] = smem_logits[i] / row_sum
                cute.arch.sync_threads()

        out_regs = cute.make_rmem_tensor(
            cute.make_layout((dims_per_lane,), stride=(1,)),
            cutlass.Float32,
        )
        for k in range(dims_per_lane):
            out_regs[k] = cutlass.Float32(0)

        for round_idx in range(num_rounds):
            j = round_idx * num_warps + warp_idx
            if j < valid_count:
                kv_idx = smem_sparse[j]
                weight = smem_logits[j]
                for k in range(dims_per_lane):
                    out_regs[k] += weight * cutlass.Float32(ckv_cache[kv_idx, k * wsize + lane_idx])

        for k in range(dims_per_lane):
            smem_partial[warp_idx, k * wsize + lane_idx] = out_regs[k]

        cute.arch.sync_threads()

        if bidz == cutlass.Int32(0):
            if is_single_split == cutlass.Int32(1):
                for i in range(tidx, head_dim_ckv, num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_warps):
                        acc += smem_partial[w, i]
                    output[bidx, bidy, i] = cutlass.BFloat16(acc)
                if tidx == 0:
                    partial_lse[bidx, bidy, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                    lse[bidx, bidy] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
            else:
                for i in range(tidx, head_dim_ckv, num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_warps):
                        acc += smem_partial[w, i]
                    partial_out[bidx, bidy, bidz, i] = acc
                if tidx == 0:
                    partial_lse[bidx, bidy, bidz, 0] = row_max
                    partial_lse[bidx, bidy, bidz, 1] = row_sum
        else:
            for i in range(tidx, head_dim_ckv, num_threads):
                acc = cutlass.Float32(0)
                for w in range(num_warps):
                    acc += smem_partial[w, i]
                partial_out[bidx, bidy, bidz, i] = acc
            if tidx == 0:
                partial_lse[bidx, bidy, bidz, 0] = row_max
                partial_lse[bidx, bidy, bidz, 1] = row_sum


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2: reduce_splits (identical to kv_split_v2)
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_reduce_kernel(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor):

    head_dim_ckv = partial_out.shape[3]
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    allocator = cutlass.utils.SmemAllocator()
    smem_sentinel = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

    if tidx == 0:
        smem_sentinel[0] = partial_lse[bidx, bidy, 0, 0]
    cute.arch.sync_threads()

    sentinel_val = smem_sentinel[0]

    if sentinel_val < cutlass.Float32(1e30):
        allocator2 = cutlass.utils.SmemAllocator()
        smem_global_max   = allocator2.allocate_tensor(cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_global_denom = allocator2.allocate_tensor(cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

        if tidx == 0:
            g_max = -cutlass.Float32(math.inf)
            for s in range(num_splits):
                local_max = partial_lse[bidx, bidy, s, 0]
                if local_max > g_max:
                    g_max = local_max
            smem_global_max[0] = g_max

            g_denom = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                g_denom += local_denom * cute.math.exp(local_max - g_max)
            smem_global_denom[0] = g_denom

        cute.arch.sync_threads()

        g_max   = smem_global_max[0]
        g_denom = smem_global_denom[0]

        if tidx == 0:
            lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

        if tidx < head_dim_ckv:
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                scale = local_denom * cute.math.exp(local_max - g_max) / g_denom
                acc += partial_out[bidx, bidy, s, tidx] * scale
            output[bidx, bidy, tidx] = cutlass.BFloat16(acc)


# ═══════════════════════════════════════════════════════════════════════════════
# Merged host function: single @cute.jit launching both kernels
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def kvsplit_merged(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
    stream):
    T, num_heads, head_dim_ckv = q_nope.shape
    kvsplit_compute_partial_kernel(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse
    ).launch(grid=[T, num_heads, NUM_SPLITS], block=[BLOCK_SIZE_COMPUTE, 1, 1], stream=stream)
    T2 = partial_out.shape[0]
    H2 = partial_out.shape[1]
    kvsplit_reduce_kernel(
        partial_out, partial_lse, output, lse
    ).launch(grid=[T2, H2, 1], block=[BLOCK_SIZE_REDUCE, 1, 1], stream=stream)


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation — single compile for the merged host function
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_kvsplit_merged():
    T = cute.sym_int()
    N = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048
    num_splits = NUM_SPLITS

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (N, head_dim_ckv),            (1, 0),    16)
    kpe_cache      = _fake(cute.BFloat16, (N, head_dim_kpe),            (1, 0),    16)
    sparse_indices = _fake(cute.Int32,    (T, top_k_len),               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    partial_out    = _fake(cute.Float32,  (T, num_heads, num_splits, head_dim_ckv), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (T, num_heads, num_splits, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),               (1, 0),     4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kvsplit_merged,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, stream,
        options="--enable-tvm-ffi"
    )


merged_compiled = compile_kvsplit_merged()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    T, H, D = q_nope.shape
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])

    partial_out = torch.zeros(T, H, NUM_SPLITS, D, dtype=torch.float32, device=output.device)
    partial_lse = torch.zeros(T, H, NUM_SPLITS, 2, dtype=torch.float32, device=output.device)

    merged_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, partial_out, partial_lse, output, lse)

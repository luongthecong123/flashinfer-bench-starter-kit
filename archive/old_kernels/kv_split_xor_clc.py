"""
kv_split_xor_clc.py — CLC (Cluster Launch Control) variant of kv_split_xor.

Architecture: Blackwell SM100a only.

Differences from kv_split_xor:
  - Uses ClcDynamicPersistentTileScheduler instead of manual T-persistent + XOR
  - Each block processes one (tok, head, split) tile at a time
  - OOB splits finish quickly and steal pending tiles via CLC
  - CLC query fired EARLY (after tile decode) so response overlaps tile compute
  - Requires 2 extra SMEM tensors: smem_mbar (8B) + smem_clc_rsp (16B)
  - smem_sparse reduced from (T_max, top_k) to (top_k,) since per-tile loading

Grid (compute): CLC-managed over (T, num_heads, num_splits)
Grid (reduce):  [T, num_heads, 1]  Block: 1024 threads
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils

import math
import torch

NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN = 16, 512, 64, 2048
NUM_PAGES, PAGE_SIZE = 8462, 64
T_MAX = 8
NUM_SPLITS = 8
DIM_SPLIT = TOP_K_LEN // NUM_SPLITS
LN2 = 0.6931471805599453

NUM_THREADS = 1024
NUM_WARPS = NUM_THREADS // 32
VEC_SIZE_CKV = 8
VEC_SIZE_KPE = 2
VEC_SIZE_OUT = 16
ITERS_PER_LANE_CKV = HEAD_DIM_CKV // (32 * VEC_SIZE_CKV)

NUM_SMS_COMPUTE = 128          # 128 out of 144 SMs for compute
NUM_THREADS_REDUCE = 256       # 16 SMs × 8 blocks/SM = 128 blocks
NUM_WARPS_REDUCE = NUM_THREADS_REDUCE // 32  # 8

SENTINEL_SKIP = float("inf")


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Host JIT: build CLC params, launch compute + reduce
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def kvsplit_xor_clc(
    q_nope:         cute.Tensor,
    q_pe:           cute.Tensor,
    ckv_cache:      cute.Tensor,
    kpe_cache:      cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale:       cutlass.Constexpr,
    partial_out:    cute.Tensor,
    partial_lse:    cute.Tensor,
    output:         cute.Tensor,
    lse:            cute.Tensor,
    stream):
    T, num_heads, head_dim_ckv = q_nope.shape

    N: cutlass.Constexpr = NUM_PAGES * PAGE_SIZE
    ckv_flat = cute.make_tensor(
        ckv_cache.iterator,
        cute.make_layout((N, head_dim_ckv), stride=(head_dim_ckv, 1)))
    kpe_flat = cute.make_tensor(
        kpe_cache.iterator,
        cute.make_layout((N, q_pe.shape[2]), stride=(q_pe.shape[2], 1)))

    num_splits_c: cutlass.Constexpr = NUM_SPLITS

    clc_params = utils.ClcDynamicPersistentTileSchedulerParams(
        (T, num_heads, num_splits_c),
        (1, 1, 1),
    )
    grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(clc_params)
    # Cap compute grid to NUM_SMS_COMPUTE so reduce can occupy the rest
    num_sms_compute: cutlass.Constexpr = NUM_SMS_COMPUTE
    capped_grid = [min(grid[0], num_sms_compute), grid[1], grid[2]]

    kvsplit_compute_clc_kernel(
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
        clc_params,
    ).launch(grid=capped_grid, block=[NUM_THREADS, 1, 1], stream=stream)

    kvsplit_reduce_kernel_clc(
        sparse_indices, partial_out, partial_lse, output, lse,
    ).launch(grid=[T, num_heads, 1], block=[NUM_THREADS_REDUCE, 1, 1], stream=stream)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1: CLC compute — work-stealing via ClcDynamicPersistentTileScheduler
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_compute_clc_kernel(
    q_nope:         cute.Tensor,        # (T,16,512)
    q_pe:           cute.Tensor,        # (T,16, 64)
    ckv_flat:       cute.Tensor,        # (N, 512)
    kpe_flat:       cute.Tensor,        # (N,  64)
    sparse_indices: cute.Tensor,        # (T, 2048)
    sm_scale:       cutlass.Constexpr,
    partial_out:    cute.Tensor,        # (T_MAX, 16, 8, 512)
    partial_lse:    cute.Tensor,        # (T_MAX, 16, 8, 2)
    output:         cute.Tensor,        # (T, 16, 512)
    lse:            cute.Tensor,        # (T, 16)
    clc_params:     utils.ClcDynamicPersistentTileSchedulerParams,
):
    T, _, _ = q_nope.shape
    head_dim_ckv:   cutlass.Constexpr = HEAD_DIM_CKV
    head_dim_kpe:   cutlass.Constexpr = HEAD_DIM_KPE
    top_k_len:      cutlass.Constexpr = TOP_K_LEN
    dim_split:      cutlass.Constexpr = DIM_SPLIT
    num_splits:     cutlass.Constexpr = NUM_SPLITS
    num_threads:    cutlass.Constexpr = NUM_THREADS
    num_warps:      cutlass.Constexpr = NUM_WARPS
    vec_size_ckv:   cutlass.Constexpr = VEC_SIZE_CKV
    vec_size_kpe:   cutlass.Constexpr = VEC_SIZE_KPE
    vec_size_out:   cutlass.Constexpr = VEC_SIZE_OUT
    iters_per_lane_ckv: cutlass.Constexpr = ITERS_PER_LANE_CKV

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    wsize = cute.arch.WARP_SIZE

    # ── SMEM allocation ──────────────────────────────────────────────────────
    alloc = cutlass.utils.SmemAllocator()
    smem_sparse      = alloc.allocate_tensor(cutlass.Int32,    cute.make_layout((top_k_len,),              stride=(1,)),               4, None)  # 8 KB
    smem_logits      = alloc.allocate_tensor(cutlass.Float32,  cute.make_layout((dim_split,),              stride=(1,)),              16, None)  # 1 KB
    smem_red_i32     = alloc.allocate_tensor(cutlass.Int32,    cute.make_layout((32,),                     stride=(1,)),               4, None)
    smem_max_red_f32 = alloc.allocate_tensor(cutlass.Float32,  cute.make_layout((32,),                     stride=(1,)),              16, None)
    smem_sum_red_f32 = alloc.allocate_tensor(cutlass.Float32,  cute.make_layout((32,),                     stride=(1,)),              16, None)
    smem_q_nope      = alloc.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv,),           stride=(1,)),              16, None)  # 1 KB
    smem_q_pe        = alloc.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe,),           stride=(1,)),              16, None)
    smem_partial     = alloc.allocate_tensor(cutlass.Float32,  cute.make_layout((num_warps, head_dim_ckv), stride=(head_dim_ckv, 1)), 16, None)  # 64 KB
    smem_out         = alloc.allocate_tensor(cutlass.Float32,  cute.make_layout((head_dim_ckv,),           stride=(1,)),              16, None)  # 2 KB
    # CLC sync primitives
    smem_mbar        = alloc.allocate_tensor(cutlass.Int64,    cute.make_layout((1,), stride=(1,)),  8, None)
    smem_clc_rsp     = alloc.allocate_tensor(cutlass.Int32,    cute.make_layout((4,), stride=(1,)), 16, None)

    # ── One-time mbarrier init ───────────────────────────────────────────────
    if tidx == 0:
        cute.arch.mbarrier_init(smem_mbar.iterator, 1)
    cute.arch.sync_threads()

    # ── Create CLC scheduler ────────────────────────────────────────────────
    scheduler = utils.ClcDynamicPersistentTileScheduler.create(
        clc_params,
        cute.arch.block_idx(),
        cute.arch.grid_dim(),
        smem_clc_rsp.iterator,
    )

    phase = cutlass.Int32(0)
    work_tile = scheduler.initial_work_tile_info()

    smem_q_nope_ = cute.zipped_divide(smem_q_nope, (vec_size_ckv,))
    ckv_flat_    = cute.zipped_divide(ckv_flat,     (1, vec_size_ckv))
    kpe_flat_    = cute.zipped_divide(kpe_flat,     (1, vec_size_kpe))
    smem_q_pe_   = cute.zipped_divide(smem_q_pe,   (vec_size_kpe,))

    # ── Work-stealing tile loop ──────────────────────────────────────────────
    while work_tile.is_valid_tile:

        tok       = work_tile.tile_idx[0]
        head_idx  = work_tile.tile_idx[1]
        split_idx = work_tile.tile_idx[2]
        split_start = split_idx * dim_split

        # ── Fire CLC query for NEXT tile early ───────────────────────────────
        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(smem_mbar.iterator, 16)
            scheduler.advance_to_next_work(smem_mbar.iterator)

        # ── Phase 1: load sparse_indices + Q into smem ───────────────────────
        partial_cnt = 0
        for i in range(tidx, top_k_len, num_threads):
            idx = sparse_indices[tok, i]
            smem_sparse[i] = idx
            if idx >= cutlass.Int32(0):
                partial_cnt += 1

        for i in range(tidx, head_dim_ckv, num_threads):
            smem_q_nope[i] = q_nope[tok, head_idx, i]
        for i in range(tidx, head_dim_kpe, num_threads):
            smem_q_pe[i] = q_pe[tok, head_idx, i]

        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = cnt_sum
        cute.arch.sync_threads()

        if warp_idx == 0:
            val = smem_red_i32[lane_idx]
            cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps)
            smem_red_i32[0] = cnt_sum
        cute.arch.sync_threads()

        num_valid = smem_red_i32[0]

        local_valid = num_valid - split_start
        if local_valid > dim_split:
            local_valid = dim_split
        if local_valid < cutlass.Int32(0):
            local_valid = cutlass.Int32(0)

        # ── Phase 2: compute ─────────────────────────────────────────────────
        if local_valid == cutlass.Int32(0):
            # OOB split: write sentinel and bail quickly
            for i in range(tidx, head_dim_ckv, num_threads):
                partial_out[tok, head_idx, split_idx, i] = cutlass.Float32(0)
            if tidx == 0:
                partial_lse[tok, head_idx, split_idx, 0] = -cutlass.Float32(math.inf)
                partial_lse[tok, head_idx, split_idx, 1] = cutlass.Float32(0)
        else:
            num_rounds = (local_valid + num_warps - 1) // num_warps

            # ── Score ────────────────────────────────────────────────────────
            for round_idx in range(num_rounds):
                sparse_idx = round_idx * num_warps + warp_idx
                if sparse_idx < local_valid:
                    cur_idx = smem_sparse[split_start + sparse_idx]
                    ckv_row_ = ckv_flat_[(0, None), (cur_idx, None)]
                    kpe_row_ = kpe_flat_[(0, None), (cur_idx, None)]

                    sum_partial = cutlass.Float32(0)

                    for it in range(iters_per_lane_ckv):
                        rest_idx = it * wsize + lane_idx
                        qn_vec = smem_q_nope_[(None, (rest_idx,))].load()
                        ckv_vec = ckv_row_[None, rest_idx].load()
                        for i in range(vec_size_ckv):
                            sum_partial += cutlass.Float32(qn_vec[i]) * cutlass.Float32(ckv_vec[i])

                    qp_vec = smem_q_pe_[(None, (lane_idx,))].load()
                    kpe_vec = kpe_row_[None, lane_idx].load()
                    for i in range(vec_size_kpe):
                        sum_partial += cutlass.Float32(qp_vec[i]) * cutlass.Float32(kpe_vec[i])

                    s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                    if lane_idx == 0:
                        smem_logits[sparse_idx] = s * sm_scale

            cute.arch.sync_threads()

            # ── Softmax: max ─────────────────────────────────────────────────
            partial_max = -cutlass.Float32(math.inf)
            for idx in range(tidx, local_valid, num_threads):
                v = smem_logits[idx]
                if v > partial_max:
                    partial_max = v

            max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
            if lane_idx == 0:
                smem_max_red_f32[warp_idx] = max_val
            cute.arch.sync_threads()
            if warp_idx == 0:
                val = smem_max_red_f32[lane_idx]
                max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
                smem_max_red_f32[0] = max_val
            cute.arch.sync_threads()

            row_max = smem_max_red_f32[0]

            # ── Softmax: exp + sum ───────────────────────────────────────────
            local_sum = cutlass.Float32(0)
            for idx in range(tidx, local_valid, num_threads):
                e = cute.math.exp(smem_logits[idx] - row_max)
                smem_logits[idx] = e
                local_sum += e

            sum_val = warp_reduce(local_sum, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_sum_red_f32[warp_idx] = sum_val
            cute.arch.sync_threads()
            if warp_idx == 0:
                val = smem_sum_red_f32[lane_idx]
                sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
                smem_sum_red_f32[0] = sum_val
            cute.arch.sync_threads()

            row_sum = smem_sum_red_f32[0]

            # ── Output ───────────────────────────────────────────────────────
            out_regs = cute.make_rmem_tensor(cute.make_layout((vec_size_out,), stride=(1,)), cutlass.Float32)
            for i in range(vec_size_out):
                out_regs[i] = cutlass.Float32(0)

            for round_idx in range(num_rounds):
                sparse_idx = round_idx * num_warps + warp_idx
                if sparse_idx < local_valid:
                    cur_idx = smem_sparse[split_start + sparse_idx]
                    ckv_row_ = ckv_flat_[(0, None), (cur_idx, None)]
                    e = smem_logits[sparse_idx]

                    for it in range(iters_per_lane_ckv):
                        rest_idx = it * wsize + lane_idx
                        ckv_vec = ckv_row_[None, rest_idx].load()
                        for i in range(vec_size_ckv):
                            out_regs[it * vec_size_ckv + i] += e * cutlass.Float32(ckv_vec[i])

            if warp_idx < local_valid:
                for it in range(iters_per_lane_ckv):
                    for v in range(vec_size_ckv):
                        smem_partial[warp_idx, (it * wsize + lane_idx) * vec_size_ckv + v] = out_regs[it * vec_size_ckv + v]

            cute.arch.sync_threads()

            num_active_warps = local_valid if local_valid < num_warps else num_warps
            for i in range(tidx, head_dim_ckv, num_threads):
                acc = cutlass.Float32(0)
                for w in range(num_active_warps):
                    acc += smem_partial[w, i]
                smem_out[i] = acc
            cute.arch.sync_threads()

            is_single_split_request = num_valid < dim_split

            if is_single_split_request and split_idx == cutlass.Int32(0):
                for i in range(tidx, head_dim_ckv, num_threads):
                    output[tok, head_idx, i] = cutlass.BFloat16(smem_out[i] / row_sum)
                if tidx == 0:
                    partial_lse[tok, head_idx, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                    lse[tok, head_idx] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
            else:
                for i in range(tidx, head_dim_ckv, num_threads):
                    partial_out[tok, head_idx, split_idx, i] = smem_out[i]
                if tidx == 0:
                    partial_lse[tok, head_idx, split_idx, 0] = row_max
                    partial_lse[tok, head_idx, split_idx, 1] = row_sum

        # ── Wait for CLC response, then fetch next tile ──────────────────────
        cute.arch.mbarrier_wait(smem_mbar.iterator, phase)
        phase = phase ^ cutlass.Int32(1)
        work_tile = scheduler.get_current_work()


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2: reduce — sentinel pattern to skip single-split requests
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_reduce_kernel_clc(
    sparse_indices: cute.Tensor,        # (T, 2048)
    partial_out:    cute.Tensor,        # (T_MAX, 16, 8, 512)
    partial_lse:    cute.Tensor,        # (T_MAX, 16, 8, 2)
    output:         cute.Tensor,        # (T, 16, 512)
    lse:            cute.Tensor,        # (T, 16)
):
    head_dim_ckv = partial_out.shape[3]
    num_splits:    cutlass.Constexpr = NUM_SPLITS
    dim_split:     cutlass.Constexpr = DIM_SPLIT
    top_k_len:     cutlass.Constexpr = TOP_K_LEN
    num_threads:   cutlass.Constexpr = NUM_THREADS_REDUCE
    num_warps:     cutlass.Constexpr = NUM_WARPS_REDUCE

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _    = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()

    sentinel_val = partial_lse[bidx, bidy, 0, 0]

    # If sentinel is SENTINEL_SKIP, single-split output was already written
    if sentinel_val < cutlass.Float32(1e30):
        alloc2 = cutlass.utils.SmemAllocator()
        smem_red_i32 = alloc2.allocate_tensor(
            cutlass.Int32, cute.make_layout((32,), stride=(1,)), 4, None)
        smem_global_max = alloc2.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_global_denom = alloc2.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

        # Count valid to know active splits
        partial_cnt = 0
        for i in range(tidx, top_k_len, num_threads):
            idx = sparse_indices[bidx, i]
            if idx >= cutlass.Int32(0):
                partial_cnt += 1

        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = cnt_sum
        cute.arch.sync_threads()
        if warp_idx == 0:
            val = smem_red_i32[lane_idx]
            cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps)
            smem_red_i32[0] = cnt_sum
        cute.arch.sync_threads()

        num_valid = smem_red_i32[0]
        num_active_splits = (num_valid + dim_split - 1) // dim_split

        if tidx == 0:
            g_max = -cutlass.Float32(math.inf)
            for s in range(num_active_splits):
                local_max = partial_lse[bidx, bidy, s, 0]
                if local_max > g_max:
                    g_max = local_max
            smem_global_max[0] = g_max

            g_denom = cutlass.Float32(0)
            for s in range(num_active_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                g_denom += local_denom * cute.math.exp(local_max - g_max)
            smem_global_denom[0] = g_denom

        cute.arch.sync_threads()

        g_max   = smem_global_max[0]
        g_denom = smem_global_denom[0]

        if tidx == 0:
            lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

        # Loop: 256 threads cover 512 elements (2 iterations each)
        for i in range(tidx, head_dim_ckv, num_threads):
            acc = cutlass.Float32(0)
            for s in range(num_active_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                scale = cute.math.exp(local_max - g_max) / g_denom
                acc += partial_out[bidx, bidy, s, i] * scale
            output[bidx, bidy, i] = cutlass.BFloat16(acc)


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_kvsplit_xor_clc():
    T = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe = NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE
    num_pages, page_size = NUM_PAGES, PAGE_SIZE
    num_splits = NUM_SPLITS

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_ckv), (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_kpe), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K_LEN),               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    partial_out    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, head_dim_ckv), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),               (1, 0),     4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kvsplit_xor_clc,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, stream,
        options="--enable-tvm-ffi"
    )


_compiled = compile_kvsplit_xor_clc()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    partial_out = torch.empty(T_MAX, NUM_HEADS, NUM_SPLITS, HEAD_DIM_CKV, dtype=torch.float32, device=output.device)
    partial_lse = torch.empty(T_MAX, NUM_HEADS, NUM_SPLITS, 2,            dtype=torch.float32, device=output.device)
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, partial_out, partial_lse, output, lse)

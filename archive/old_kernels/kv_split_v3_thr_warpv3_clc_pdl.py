"""
kv_split_v3_thr_warpv3_clc_pdl.py — CLC compute + PDL-enabled reduce.

Architecture: Blackwell SM100a only.

PDL pattern (Programmatic Dependent Launch):
  Kernel 1 (compute CLC):
    - Fires griddepcontrol_launch_dependents() at the START of every tile
      iteration (first call counts, subsequent are NOPs).  This lets the
      reduce kernel start its prolog — smem init, sentinel read from L2 —
      while compute is still processing tiles.
  Kernel 2 (reduce):
    - Launched with use_pdl=True.
    - Begins with griddepcontrol_wait() which blocks the mainloop until ALL
      compute blocks have issued their grid-ending membar, guaranteeing that
      partial_lse / partial_out are visible in global memory.
    - Everything BEFORE griddepcontrol_wait() is the prolog (free overlap).

Expected benefit:
  T_reduce_launch + T_reduce_prolog overlap with T_compute_tail + T_membar.
  For T=8 with heavy imbalance, the last few active tiles are long; reduce
  can start its sentinel read and smem setup during that time.

References:
  https://yang-yifan.github.io/blogs/pdl/pdl.html
  src/kernels/test_grid_sync_pdl.py
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils

import math
import torch

NUM_HEADS = 16
DV = 512
ROW_MAX_SUM_PAIR = 2

DIM_SPLIT  = 256
TOP_K      = 2048
NUM_SPLITS = (TOP_K + DIM_SPLIT - 1) // DIM_SPLIT  # 8

BLOCK_SIZE_COMPUTE = 1024
NUM_WARPS_COMPUTE  = BLOCK_SIZE_COMPUTE // 32  # 32
DIMS_PER_LANE: cutlass.Constexpr = 512 // 32   # 16

NUM_VEC          : cutlass.Constexpr = 8
ITERS_PER_LANE   : cutlass.Constexpr = (512 // 32) // 8   # 2

BLOCK_SIZE_REDUCE = 512

SENTINEL_SKIP = float("inf")
LN2 = 0.6931471805599453


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Host JIT: CLC compute + PDL reduce
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def kvsplit_fused_clc_pdl(
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

    N: cutlass.Constexpr = 8462 * 64
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

    # Compute kernel: fires launch_dependents early in first iteration
    kvsplit_compute_clc_pdl_kernel(
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
        clc_params,
    ).launch(grid=list(grid), block=[BLOCK_SIZE_COMPUTE, 1, 1],
             stream=stream, use_pdl=True)

    # Reduce kernel: starts prolog immediately, blocks at griddepcontrol_wait()
    kvsplit_reduce_pdl_kernel(
        partial_out, partial_lse, output, lse,
    ).launch(grid=[T, num_heads, 1], block=[BLOCK_SIZE_REDUCE, 1, 1],
             stream=stream, use_pdl=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1: CLC compute with PDL launch_dependents
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_compute_clc_pdl_kernel(
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
    clc_params:     utils.ClcDynamicPersistentTileSchedulerParams,
):
    T, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = kpe_cache.shape[1]
    dim_split:      cutlass.Constexpr = DIM_SPLIT
    num_splits:     cutlass.Constexpr = NUM_SPLITS
    top_k:          cutlass.Constexpr = TOP_K
    num_vec:        cutlass.Constexpr = NUM_VEC
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE
    dims_per_lane:  cutlass.Constexpr = DIMS_PER_LANE
    num_threads:    cutlass.Constexpr = BLOCK_SIZE_COMPUTE
    num_warps:      cutlass.Constexpr = NUM_WARPS_COMPUTE

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    allocator = cutlass.utils.SmemAllocator()
    smem_sparse  = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((top_k,),        stride=(1,)),  4, None)
    smem_logits  = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((dim_split,),    stride=(1,)), 16, None)
    smem_red_i32 = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((32,),           stride=(1,)),  4, None)
    smem_red_f32 = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((32,),           stride=(1,)), 16, None)
    smem_q_nope  = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv,), stride=(1,)), 16, None)
    smem_q_pe    = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe,), stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout((num_warps, head_dim_ckv), stride=(head_dim_ckv, 1)), 16, None)
    smem_mbar    = allocator.allocate_tensor(cutlass.Int64,
        cute.make_layout((1,), stride=(1,)), 8, None)
    smem_clc_rsp = allocator.allocate_tensor(cutlass.Int32,
        cute.make_layout((4,), stride=(1,)), 16, None)

    if tidx == 0:
        cute.arch.mbarrier_init(smem_mbar.iterator, 1)
    cute.arch.sync_threads()

    scheduler = utils.ClcDynamicPersistentTileScheduler.create(
        clc_params,
        cute.arch.block_idx(),
        cute.arch.grid_dim(),
        smem_clc_rsp.iterator,
    )

    phase = cutlass.Int32(0)
    work_tile = scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:

        tok   = work_tile.tile_idx[0]
        head  = work_tile.tile_idx[1]
        split = work_tile.tile_idx[2]
        split_start = split * dim_split

        # ── PDL: signal reduce kernel to start its prolog.
        # First call counts; subsequent calls on the same block are NOPs.
        # The reduce kernel's griddepcontrol_wait() blocks its mainloop until
        # ALL compute blocks have finished (grid-ending membar), so correctness
        # is NOT affected by how early we fire this.
        cute.arch.griddepcontrol_launch_dependents()

        # ── Fire CLC query for NEXT tile ──────────────────────────────────────
        # advance_to_next_work uses elect_one() (warp-level), must be called from single warp
        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(smem_mbar.iterator, 16)
            scheduler.advance_to_next_work(smem_mbar.iterator)

        # ── Phase 1: load sparse_indices + Q ─────────────────────────────────
        partial_cnt = 0
        for i in range(tidx, top_k, num_threads):
            idx = sparse_indices[tok, i]
            smem_sparse[i] = idx
            if idx >= cutlass.Int32(0):
                partial_cnt += 1
        for i in range(tidx, head_dim_ckv, num_threads):
            smem_q_nope[i] = q_nope[tok, head, i]
        for i in range(tidx, head_dim_kpe, num_threads):
            smem_q_pe[i] = q_pe[tok, head, i]

        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = cnt_sum
        cute.arch.sync_threads()
        if warp_idx == 0:
            val = smem_red_i32[lane_idx]
            cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps)
            smem_red_i32[0] = cnt_sum
        cute.arch.sync_threads()

        global_num_valid = smem_red_i32[0]
        local_valid = global_num_valid - split_start
        if local_valid > dim_split:
            local_valid = dim_split
        if local_valid < cutlass.Int32(0):
            local_valid = cutlass.Int32(0)
        active_splits = (global_num_valid + dim_split - 1) // dim_split

        row_max = -cutlass.Float32(math.inf)
        row_sum = cutlass.Float32(0)

        if local_valid == cutlass.Int32(0):
            for i in range(tidx, head_dim_ckv, num_threads):
                partial_out[tok, head, split, i] = cutlass.Float32(0)
            if tidx == 0:
                partial_lse[tok, head, split, 0] = -cutlass.Float32(math.inf)
                partial_lse[tok, head, split, 1] = cutlass.Float32(0)
        else:
            num_rounds = (local_valid + num_warps - 1) // num_warps

            q_nope_z = cute.zipped_divide(smem_q_nope, (num_vec,))
            for round_idx in range(num_rounds):
                sparse_idx = round_idx * num_warps + warp_idx
                if sparse_idx < local_valid:
                    cur_idx = smem_sparse[split_start + sparse_idx]
                    ckv_row = ckv_cache[cur_idx, None]
                    ckv_z   = cute.zipped_divide(ckv_row, (num_vec,))
                    sum_partial = cutlass.Float32(0)
                    for it in range(iters_per_lane):
                        group  = it * wsize + lane_idx
                        q_frag = q_nope_z[(None, (group,))].load()
                        K_frag = ckv_z[(None, (group,))].load()
                        for v in range(num_vec):
                            sum_partial += cutlass.Float32(q_frag[v]) * cutlass.Float32(K_frag[v])
                    for k_idx in range(head_dim_kpe // wsize):
                        q_p = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                        kv  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                        sum_partial += q_p * kv
                    s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                    if lane_idx == 0:
                        smem_logits[sparse_idx] = s * sm_scale
            cute.arch.sync_threads()

            partial_max = -cutlass.Float32(math.inf)
            for idx in range(tidx, local_valid, num_threads):
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
            for idx in range(tidx, local_valid, num_threads):
                e = cute.math.exp(smem_logits[idx] - row_max)
                smem_logits[idx] = e
                local_sum += e
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

            if active_splits == cutlass.Int32(1):
                if split == cutlass.Int32(0):
                    for i in range(tidx, local_valid, num_threads):
                        smem_logits[i] = smem_logits[i] / row_sum
                    cute.arch.sync_threads()

            out_regs = cute.make_rmem_tensor(
                cute.make_layout((dims_per_lane,), stride=(1,)), cutlass.Float32)
            for k in range(dims_per_lane):
                out_regs[k] = cutlass.Float32(0)
            for round_idx in range(num_rounds):
                j = round_idx * num_warps + warp_idx
                if j < local_valid:
                    kv_idx = smem_sparse[split_start + j]
                    weight = smem_logits[j]
                    V_row  = ckv_cache[kv_idx, None]
                    V_z    = cute.zipped_divide(V_row, (num_vec,))
                    for it in range(iters_per_lane):
                        group = it * wsize + lane_idx
                        frag  = V_z[(None, (group,))].load()
                        for v in range(num_vec):
                            out_regs[it * num_vec + v] += weight * cutlass.Float32(frag[v])
            for it in range(iters_per_lane):
                for v in range(num_vec):
                    smem_partial[warp_idx, (it * wsize + lane_idx) * num_vec + v] = out_regs[it * num_vec + v]
            cute.arch.sync_threads()

        # ── Wait for CLC response + fetch next tile ───────────────────────────
        cute.arch.mbarrier_wait(smem_mbar.iterator, phase)
        phase = phase ^ cutlass.Int32(1)
        work_tile = scheduler.get_current_work()

        # ── Epilogue: write smem_partial → GMEM (overlaps reduce prolog) ─────
        if local_valid != cutlass.Int32(0):
            if active_splits == cutlass.Int32(1):
                if split == cutlass.Int32(0):
                    for i in range(tidx, head_dim_ckv, num_threads):
                        acc = cutlass.Float32(0)
                        for w in range(num_warps):
                            acc += smem_partial[w, i]
                        output[tok, head, i] = cutlass.BFloat16(acc)
                    if tidx == 0:
                        partial_lse[tok, head, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                        lse[tok, head] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
            else:
                for i in range(tidx, head_dim_ckv, num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_warps):
                        acc += smem_partial[w, i]
                    partial_out[tok, head, split, i] = acc
                if tidx == 0:
                    partial_lse[tok, head, split, 0] = row_max
                    partial_lse[tok, head, split, 1] = row_sum


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2: PDL-enabled reduce
#
# Prolog (before griddepcontrol_wait):
#   - smem allocation (zero overhead — compile-time)
#   - sentinel read from L2: partial_lse[bidx, bidy, 0, 0]  ← overlaps compute
#
# griddepcontrol_wait():
#   - Blocks until ALL compute blocks have issued grid-ending membar.
#   - After this point, partial_out and partial_lse are globally visible.
#
# Mainloop (after griddepcontrol_wait):
#   - Same reduce logic as kvsplit_reduce_kernel_clc
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_reduce_pdl_kernel(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output:      cute.Tensor,
    lse:         cute.Tensor,
):
    head_dim_ckv = partial_out.shape[3]
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _    = cute.arch.thread_idx()

    # ── Prolog: read sentinel — overlaps compute kernel's remaining tiles ─────
    # Safe because griddepcontrol_wait() below ensures we can't enter the
    # mainloop until all compute GMEM writes are globally visible.
    sentinel_val = partial_lse[bidx, bidy, 0, 0]

    # ── griddepcontrol_wait: block until compute is done ─────────────────────
    cute.arch.griddepcontrol_wait()

    # ── Mainloop: all partial_out / partial_lse reads are now safe ───────────
    if sentinel_val < cutlass.Float32(1e30):
        allocator = cutlass.utils.SmemAllocator()
        smem_global_max   = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_global_denom = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

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
                scale = cute.math.exp(local_max - g_max) / g_denom
                acc += partial_out[bidx, bidy, s, tidx] * scale
            output[bidx, bidy, tidx] = cutlass.BFloat16(acc)


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation + run
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_kvsplit_clc_pdl():
    T          = cute.sym_int()
    T_MAX      = 8
    num_heads, head_dim_ckv, head_dim_kpe = 16, 512, 64
    num_pages, page_size = 8462, 64
    num_splits = NUM_SPLITS

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_ckv), (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_kpe), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K),                   (1, 0),     4)
    sm_scale       = 0.1352337788608801
    partial_out    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, head_dim_ckv), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),               (1, 0),     4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kvsplit_fused_clc_pdl,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, stream,
        options="--enable-tvm-ffi"
    )


_compiled = compile_kvsplit_clc_pdl()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    T = q_nope.shape[0]
    partial_out = torch.empty(8, NUM_HEADS, NUM_SPLITS, DV,               dtype=torch.float32, device=output.device)
    partial_lse = torch.empty(8, NUM_HEADS, NUM_SPLITS, ROW_MAX_SUM_PAIR, dtype=torch.float32, device=output.device)
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
              partial_out, partial_lse, output, lse)

"""fused_persistent.py — kv_split with persistent scheduler + T>S>H task swizzle.

Motivation (WL20 lopsided workload: T=8, valid=[8,11,11,16,1641,73,1,1]):
  Default kv_split grid [T,H,S] = 1024 blocks. B200 C=296 concurrent slots,
  tok4's 112 heavy splits cluster in waves 2-3 → ~42µs observed.

  Persistent G=128 workers + T>S>H flat ordering:
    flat_idx = tok*(S*H) + split*H + head = tok*128 + split*16 + head
    Worker b: split = b//H (fixed), head = b%H (fixed), tok loops 0..T_MAX-1

  T>S>H property:
    tok4's heavy spans flat_idx [512..639] covering all 128 residues mod 128.
    Each worker gets at most 1 tok4-heavy task regardless of workload.

  Workers 0..15   (split=0): 7 light-heavy + 1 tok4-heavy ≈ 14µs
  Workers 16..111 (split=1-6): 7 OOB + 1 tok4-heavy ≈ 12.8µs
  Workers 112..127 (split=7): all OOB ≈ 2.4µs
  Critical path: ~14µs vs ~42µs measured → ~3× improvement.

  Pre-pass: computes global_valid_count[T] → OOB tokens skip the 8KB
  sparse_indices load entirely (7/8 rounds for workers 16..111).

Grid (pre-pass):  [T, 1, 1]  Block: 1024 threads
Grid (compute):   persistent, 128 CTAs, Block: 1024 threads
Grid (reduce):    [T, H, 1]  Block: 512 threads

Ordering:  T>S>H  (split=outermost after token, breaks C%S resonance)
API:       StaticPersistentTileScheduler from cutlass.utils
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils

import math
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_HEADS  = 16
D_CKV      = 512
D_KPE      = 64
DIM_SPLIT  = 256
TOP_K      = 2048
NUM_SPLITS = TOP_K // DIM_SPLIT   # 8
T_MAX      = 8
TOTAL_TASKS: cutlass.Constexpr = T_MAX * NUM_SPLITS * NUM_HEADS  # 1024

BLOCK_SIZE_COMPUTE = 1024
NUM_WARPS_COMPUTE  = BLOCK_SIZE_COMPUTE // 32   # 32
DIMS_PER_LANE: cutlass.Constexpr = D_CKV // 32  # 16
NUM_VEC:       cutlass.Constexpr = 8
ITERS_PER_LANE: cutlass.Constexpr = (D_CKV // 32) // 8  # 2

BLOCK_SIZE_REDUCE = 512
MAX_ACTIVE_CLUSTERS = 128   # persistent workers

LN2 = 0.6931471805599453


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-pass: count valid (non-(-1)) sparse_indices per token
# One CTA per token, 1024 threads each scan TOP_K=2048 entries.
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def valid_count_kernel(
    sparse_indices: cute.Tensor,
    global_valid_count: cute.Tensor,
):
    tok = cute.arch.block_idx()[0]
    tidx, _, _ = cute.arch.thread_idx()
    num_threads: cutlass.Constexpr = 1024
    num_warps_k: cutlass.Constexpr = 32
    top_k: cutlass.Constexpr = TOP_K
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    allocator = cutlass.utils.SmemAllocator()
    smem_red = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((32,), stride=(1,)), 4, None)

    cnt = 0
    for i in range(tidx, top_k, num_threads):
        if sparse_indices[tok, i] >= cutlass.Int32(0):
            cnt += 1

    s = warp_reduce(cnt, lambda a, b: a + b, width=32)
    if lane_idx == 0:
        smem_red[warp_idx] = s
    cute.arch.sync_threads()

    if warp_idx == 0:
        val = smem_red[lane_idx]
        s = warp_reduce(val, lambda a, b: a + b, width=num_warps_k)
        if lane_idx == 0:
            global_valid_count[tok] = s


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent compute kernel
# T>S>H task ordering: flat_idx = tok*(S*H) + split*H + head = tok*128 + split*16 + head
# Each worker b: fixed split = b//H, fixed head = b%H, loops tok = 0..T_MAX-1
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def persistent_compute_kernel(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    global_valid_count: cute.Tensor,
    tile_sched_params: utils.PersistentTileSchedulerParams,
):
    dim_split:     cutlass.Constexpr = DIM_SPLIT
    num_splits:    cutlass.Constexpr = NUM_SPLITS
    top_k:         cutlass.Constexpr = TOP_K
    num_vec:       cutlass.Constexpr = NUM_VEC
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE
    dims_per_lane:  cutlass.Constexpr = DIMS_PER_LANE
    num_threads:   cutlass.Constexpr = BLOCK_SIZE_COMPUTE
    num_warps:     cutlass.Constexpr = NUM_WARPS_COMPUTE
    n_sh:          cutlass.Constexpr = NUM_SPLITS * NUM_HEADS   # 128

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    actual_T = q_nope.shape[0]

    # ── Smem (reused across tiles in the persistent loop) ────────────────────
    allocator  = cutlass.utils.SmemAllocator()
    smem_sparse  = allocator.allocate_tensor(
        cutlass.Int32,    cute.make_layout((top_k,),     stride=(1,)),  4, None)
    smem_logits  = allocator.allocate_tensor(
        cutlass.Float32,  cute.make_layout((dim_split,), stride=(1,)), 16, None)
    smem_red_i32 = allocator.allocate_tensor(
        cutlass.Int32,    cute.make_layout((32,),        stride=(1,)),  4, None)
    smem_red_f32 = allocator.allocate_tensor(
        cutlass.Float32,  cute.make_layout((32,),        stride=(1,)), 16, None)
    smem_q_nope  = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((D_CKV,),     stride=(1,)), 16, None)
    smem_q_pe    = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((D_KPE,),     stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((num_warps, D_CKV), stride=(D_CKV, 1)), 16, None)

    # ── Persistent tile scheduler ─────────────────────────────────────────────
    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    while work_tile.is_valid_tile:
        flat_idx = work_tile.tile_idx[0]

        # T>S>H decode: flat_idx = tok*(S*H) + split*H + head
        tok   =  flat_idx // n_sh                          # 0..T_MAX-1
        split = (flat_idx // NUM_HEADS) % num_splits       # 0..S-1
        head  =  flat_idx % NUM_HEADS                      # 0..H-1
        split_start = split * dim_split

        # Skip ghost tokens beyond actual batch size
        if tok < actual_T:
            # ── OOB check using pre-computed global_valid_count ──────────────
            # (avoids 8KB sparse_indices load for non-participating splits)
            valid_cnt   = global_valid_count[tok]
            local_valid = valid_cnt - split_start
            if local_valid > dim_split:
                local_valid = dim_split
            if local_valid < cutlass.Int32(0):
                local_valid = cutlass.Int32(0)

            if local_valid == cutlass.Int32(0):
                # OOB split: write sentinel partials for reduce kernel
                for i in range(tidx, D_CKV, num_threads):
                    partial_out[tok, head, split, i] = cutlass.Float32(0)
                if tidx == 0:
                    partial_lse[tok, head, split, 0] = -cutlass.Float32(math.inf)
                    partial_lse[tok, head, split, 1] = cutlass.Float32(0)
            else:
                # ── Load sparse_indices (only for active splits) ─────────────
                for i in range(tidx, top_k, num_threads):
                    smem_sparse[i] = sparse_indices[tok, i]

                for i in range(tidx, D_CKV, num_threads):
                    smem_q_nope[i] = q_nope[tok, head, i]
                for i in range(tidx, D_KPE, num_threads):
                    smem_q_pe[i] = q_pe[tok, head, i]

                cute.arch.sync_threads()

                # ── Score: LDG.128 + fp32 multiply ──────────────────────────
                q_nope_z   = cute.zipped_divide(smem_q_nope, (num_vec,))
                num_rounds = (local_valid + num_warps - 1) // num_warps

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
                                sum_partial += (cutlass.Float32(q_frag[v])
                                                * cutlass.Float32(K_frag[v]))

                        for k_idx in range(D_KPE // wsize):
                            q_p = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                            kv  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                            sum_partial += q_p * kv

                        s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                        if lane_idx == 0:
                            smem_logits[sparse_idx] = s * sm_scale

                cute.arch.sync_threads()

                # ── Softmax max ──────────────────────────────────────────────
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

                # ── Softmax exp + sum (write back to smem_logits) ────────────
                partial_sum = cutlass.Float32(0)
                for idx in range(tidx, local_valid, num_threads):
                    e = cute.math.exp(smem_logits[idx] - row_max)
                    smem_logits[idx] = e
                    partial_sum += e

                sum_val = warp_reduce(partial_sum, lambda a, b: a + b, width=32)
                if lane_idx == 0:
                    smem_red_f32[warp_idx] = sum_val
                cute.arch.sync_threads()
                if warp_idx == 0:
                    val = smem_red_f32[lane_idx]
                    sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
                    smem_red_f32[0] = sum_val
                cute.arch.sync_threads()
                row_sum = smem_red_f32[0]

                # ── Output: vectorized LDG.128 reads (unnormalised) ──────────
                out_regs = cute.make_rmem_tensor(
                    cute.make_layout((dims_per_lane,), stride=(1,)),
                    cutlass.Float32,
                )
                for k in range(dims_per_lane):
                    out_regs[k] = cutlass.Float32(0)

                for round_idx in range(num_rounds):
                    j = round_idx * num_warps + warp_idx
                    if j < local_valid:
                        kv_idx = smem_sparse[split_start + j]
                        weight = smem_logits[j]

                        V_row = ckv_cache[kv_idx, None]
                        V_z   = cute.zipped_divide(V_row, (num_vec,))

                        for it in range(iters_per_lane):
                            group = it * wsize + lane_idx
                            frag  = V_z[(None, (group,))].load()
                            for v in range(num_vec):
                                out_regs[it * num_vec + v] += (
                                    weight * cutlass.Float32(frag[v]))

                # Write per-warp accumulators to smem
                for it in range(iters_per_lane):
                    for v in range(num_vec):
                        smem_partial[warp_idx, (it * wsize + lane_idx) * num_vec + v] = (
                            out_regs[it * num_vec + v])

                cute.arch.sync_threads()

                # Cross-warp reduce → partial_out (unnormalised)
                for i in range(tidx, D_CKV, num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_warps):
                        acc += smem_partial[w, i]
                    partial_out[tok, head, split, i] = acc

                if tidx == 0:
                    partial_lse[tok, head, split, 0] = row_max
                    partial_lse[tok, head, split, 1] = row_sum

        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()


# ═══════════════════════════════════════════════════════════════════════════════
# Reduce kernel: merge S partial outputs into final (tok, head) output
# Same logic as kv_split_v3_thr_warpv3's reduce kernel.
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def persistent_reduce_kernel(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
):
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()   # tok, head
    tidx, _, _ = cute.arch.thread_idx()

    # Find global max across all splits
    g_max = -cutlass.Float32(math.inf)
    for s in range(num_splits):
        local_max = partial_lse[bidx, bidy, s, 0]
        if local_max > g_max:
            g_max = local_max

    # Compute global denominator (sum of rescaled per-split sums)
    g_denom = cutlass.Float32(0)
    for s in range(num_splits):
        local_max   = partial_lse[bidx, bidy, s, 0]
        local_denom = partial_lse[bidx, bidy, s, 1]
        g_denom += local_denom * cute.math.exp(local_max - g_max)

    if tidx == 0:
        lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

    if tidx < D_CKV:
        acc = cutlass.Float32(0)
        for s in range(num_splits):
            local_max   = partial_lse[bidx, bidy, s, 0]
            local_denom = partial_lse[bidx, bidy, s, 1]
            scale = cute.math.exp(local_max - g_max) / g_denom
            acc += partial_out[bidx, bidy, s, tidx] * scale
        output[bidx, bidy, tidx] = cutlass.BFloat16(acc)


# ═══════════════════════════════════════════════════════════════════════════════
# JIT launcher: pre-pass → persistent compute → reduce
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def fused_persistent_launcher(
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
    global_valid_count: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream,
):
    T, num_heads, head_dim_ckv = q_nope.shape

    # Flatten paged caches [P, PS, D] → [P*PS, D]
    N: cutlass.Constexpr = 8462 * 64
    ckv_flat = cute.make_tensor(
        ckv_cache.iterator,
        cute.make_layout((N, head_dim_ckv), stride=(head_dim_ckv, 1)))
    kpe_flat = cute.make_tensor(
        kpe_cache.iterator,
        cute.make_layout((N, D_KPE), stride=(D_KPE, 1)))

    # ── 1. Pre-pass: count valid indices per token ────────────────────────────
    valid_count_kernel(sparse_indices, global_valid_count).launch(
        grid=[T, 1, 1], block=[1024, 1, 1], stream=stream,
    )

    # ── 2. Persistent compute (T>S>H swizzle, 128 workers) ───────────────────
    total_tasks: cutlass.Constexpr = TOTAL_TASKS
    cluster_shape_mnl = (1, 1, 1)
    num_ctas_mnl = (total_tasks, 1, 1)

    tile_sched_params = utils.PersistentTileSchedulerParams(
        num_ctas_mnl, cluster_shape_mnl, swizzle_size=1, raster_along_m=True,
    )
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )

    persistent_compute_kernel(
        q_nope, q_pe, ckv_flat, kpe_flat,
        sparse_indices, sm_scale,
        partial_out, partial_lse,
        global_valid_count, tile_sched_params,
    ).launch(grid=grid, block=[BLOCK_SIZE_COMPUTE, 1, 1], stream=stream)

    # ── 3. Reduce: merge S splits ─────────────────────────────────────────────
    persistent_reduce_kernel(partial_out, partial_lse, output, lse).launch(
        grid=[T, num_heads, 1], block=[BLOCK_SIZE_REDUCE, 1, 1], stream=stream,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_fused_persistent():
    T = cute.sym_int()
    num_heads, head_dim_ckv = NUM_HEADS, D_CKV
    head_dim_kpe = D_KPE
    num_pages, page_size = 8462, 64

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv),             (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe),             (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_ckv),     (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_kpe),     (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K),                               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    partial_out    = _fake(cute.Float32,  (T_MAX, num_heads, NUM_SPLITS, head_dim_ckv), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (T_MAX, num_heads, NUM_SPLITS, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv),             (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),                           (1, 0),     4)
    global_valid_count = _fake(cute.Int32, (T_MAX,),                                (0,),       4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fused_persistent_launcher,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, global_valid_count,
        MAX_ACTIVE_CLUSTERS, stream,
        options="--enable-tvm-ffi",
    )


_compiled = compile_fused_persistent()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    device = output.device
    partial_out = torch.empty(
        T_MAX, NUM_HEADS, NUM_SPLITS, D_CKV, dtype=torch.float32, device=device)
    partial_lse = torch.empty(
        T_MAX, NUM_HEADS, NUM_SPLITS, 2, dtype=torch.float32, device=device)
    global_valid_count = torch.zeros(T_MAX, dtype=torch.int32, device=device)
    _compiled(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        partial_out, partial_lse, output, lse, global_valid_count,
    )

"""fused_persistent_v3b.py — v3 with MAX_ACTIVE_CLUSTERS=148 (all B200 SMs).

Identical to v3 except launches 148 CTAs instead of 128.
Key insight: with T>S>H ordering and MAX_ACTIVE_CLUSTERS=148 workers,
each worker b has a FIXED (head_b, split_b) across ALL rounds:
    head_b  = b % NUM_HEADS         (fixed for CTA's lifetime)
    split_b = (b // NUM_HEADS) % NUM_SPLITS  (fixed)
    tok     = flat_idx // n_sh      (changes each round)

Startup amortizes (per-CTA, once):
    smem_q_nope_all [T_MAX × D_CKV]      bf16  8KB  ← q_nope[:, head_b, :]
    smem_q_pe_all   [T_MAX × D_KPE]      bf16  1KB  ← q_pe[:, head_b, :]
    smem_sparse     [T_MAX × DIM_SPLIT]  i32   8KB  ← sparse[:, split_b*256:(split_b+1)*256]

Valid counts per token come from a pre-pass kernel (global_valid_count[T]).

Smem layout per block (~82KB, within B200's 228KB/SM, fits 2 blocks/SM):
  smem_sparse     [T_MAX * DIM_SPLIT]      i32    8KB   (current-split window)
  smem_q_nope_all [T_MAX * D_CKV]          bf16   8KB
  smem_q_pe_all   [T_MAX * D_KPE]          bf16   1KB
  smem_partial    [NUM_WARPS * D_CKV]      f32   64KB   (cross-warp accumulator)
  smem_logits     [DIM_SPLIT]              f32    1KB
  smem_red_f32    [32]                     f32  128B
  Total                                         ~82KB

Grid (pre-pass):  [T, 1, 1]              Block: 1024 threads
Grid (compute):   persistent, 148 CTAs,  Block: 1024 threads
Grid (reduce):    [T, H, 1]              Block: 512 threads
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
MAX_ACTIVE_CLUSTERS = 148   # persistent workers — all B200 SMs

LN2 = 0.6931471805599453
SENTINEL_SKIP = float("inf")   # written to partial_lse[tok,head,0,0] to signal direct-write done

N_PAGES_FLAT: cutlass.Constexpr = 8462 * 64


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-pass: count valid (non-(-1)) sparse_indices per token.
# One CTA per token, 1024 threads scan TOP_K=2048 entries.
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def valid_count_kernel_v3b(
    sparse_indices: cute.Tensor,
    global_valid_count: cute.Tensor,
):
    tok = cute.arch.block_idx()[0]
    tidx, _, _ = cute.arch.thread_idx()
    num_threads_k: cutlass.Constexpr = 1024
    num_warps_k:   cutlass.Constexpr = 32
    top_k:         cutlass.Constexpr = TOP_K
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    allocator = cutlass.utils.SmemAllocator()
    smem_red = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((32,), stride=(1,)), 4, None)

    cnt = 0
    for i in range(tidx, top_k, num_threads_k):
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
# Persistent compute kernel v3
#
# Startup (amortized, runs once per CTA before tile scheduler):
#   1. Decode head_b, split_b from block_idx — fixed for this CTA's lifetime
#   2. Load smem_q_nope_all  ← q_nope[:, head_b, :]               (8KB)
#   3. Load smem_q_pe_all    ← q_pe[:, head_b, :]                 (1KB)
#   4. Load smem_sparse      ← sparse[:, split_b*256:(split_b+1)*256] (8KB)
#
# Task loop (T>S>H, head/split from startup, only tok decoded at task time):
#   - Uses global_valid_count[tok] from pre-pass for per-token valid count
#   - Active tasks: q/sparse all from smem — zero HBM reads for q/k data
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def persistent_compute_kernel_v3b(
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
        tok   =  flat_idx // n_sh
        split = (flat_idx // NUM_HEADS) % num_splits
        head  =  flat_idx % NUM_HEADS
        split_start = split * dim_split

        if tok < actual_T:
            valid_cnt   = global_valid_count[tok]
            local_valid = valid_cnt - split_start
            if local_valid > dim_split:
                local_valid = dim_split
            if local_valid < cutlass.Int32(0):
                local_valid = cutlass.Int32(0)

            active_splits = (valid_cnt + dim_split - 1) // dim_split

            if local_valid == cutlass.Int32(0):
                # OOB split: write sentinel partials for reduce kernel
                for i in range(tidx, D_CKV, num_threads):
                    partial_out[tok, head, split, i] = cutlass.Float32(0)
                if tidx == 0:
                    partial_lse[tok, head, split, 0] = -cutlass.Float32(math.inf)
                    partial_lse[tok, head, split, 1] = cutlass.Float32(0)
            else:
                # ── Load sparse indices + q into smem ────────────────────────
                for i in range(tidx, top_k, num_threads):
                    smem_sparse[i] = sparse_indices[tok, i]

                for i in range(tidx, D_CKV, num_threads):
                    smem_q_nope[i] = q_nope[tok, head, i]
                for i in range(tidx, D_KPE, num_threads):
                    smem_q_pe[i] = q_pe[tok, head, i]

                cute.arch.sync_threads()

                # ── Score: LDG.128 + fp32 multiply ───────────────────────────
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

                # ── Softmax max ───────────────────────────────────────────────
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

                # ── Softmax exp + sum ─────────────────────────────────────────
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

                # ── Write results (sentinel or partial) ───────────────────────
                if active_splits == cutlass.Int32(1):
                    if split == cutlass.Int32(0):
                        # Single-split fast path: write directly + sentinel
                        for i in range(tidx, D_CKV, num_threads):
                            acc = cutlass.Float32(0)
                            for w in range(num_warps):
                                acc += smem_partial[w, i]
                            output[tok, head, i] = cutlass.BFloat16(acc / row_sum)
                        if tidx == 0:
                            lse[tok, head] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
                            partial_lse[tok, head, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                else:
                    # Multi-split: write to partial buffer (unnormalised)
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
# Reduce kernel — matches kv_split_v3_thr_warpv3 sentinel pattern exactly
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def persistent_reduce_kernel_v3b(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
):
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()   # tok, head
    tidx, _, _    = cute.arch.thread_idx()

    allocator = cutlass.utils.SmemAllocator()
    smem_sentinel = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

    if tidx == 0:
        smem_sentinel[0] = partial_lse[bidx, bidy, 0, 0]
    cute.arch.sync_threads()

    # Read sentinel into register (same as kv_split_v3_thr_warpv3 pattern)
    sentinel_val = smem_sentinel[0]

    if sentinel_val < cutlass.Float32(1e30):
        # Multi-split path: merge all splits
        allocator2 = cutlass.utils.SmemAllocator()
        smem_g_max   = allocator2.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_g_denom = allocator2.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

        if tidx == 0:
            g_max = -cutlass.Float32(math.inf)
            for s in range(num_splits):
                local_max = partial_lse[bidx, bidy, s, 0]
                if local_max > g_max:
                    g_max = local_max
            smem_g_max[0] = g_max

            g_denom = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                g_denom += local_denom * cute.math.exp(local_max - g_max)
            smem_g_denom[0] = g_denom
        cute.arch.sync_threads()

        g_max   = smem_g_max[0]
        g_denom = smem_g_denom[0]

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
def fused_persistent_v3b_launcher(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    global_valid_count: cute.Tensor,
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream,
):
    T, num_heads, _ = q_nope.shape

    ckv_flat = cute.make_tensor(
        ckv_cache.iterator,
        cute.make_layout((N_PAGES_FLAT, D_CKV), stride=(D_CKV, 1)))
    kpe_flat = cute.make_tensor(
        kpe_cache.iterator,
        cute.make_layout((N_PAGES_FLAT, D_KPE), stride=(D_KPE, 1)))

    # ── 0. Pre-pass: count valid KV entries per token ─────────────────────────
    valid_count_kernel_v3b(sparse_indices, global_valid_count).launch(
        grid=[T, 1, 1], block=[1024, 1, 1], stream=stream)

    # ── 1. Persistent compute (amortized startup + T>S>H task loop) ──────────
    total_tasks: cutlass.Constexpr = TOTAL_TASKS
    cluster_shape_mnl = (1, 1, 1)
    num_ctas_mnl = (total_tasks, 1, 1)

    # utils.gemm.sm100.StaticPersistentTileScheduler

    tile_sched_params = utils.PersistentTileSchedulerParams(
        num_ctas_mnl, cluster_shape_mnl, swizzle_size=1, raster_along_m=True,
    )
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )

    persistent_compute_kernel_v3b(
        q_nope, q_pe, ckv_flat, kpe_flat,
        sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
        global_valid_count,
        tile_sched_params,
    ).launch(grid=grid, block=[BLOCK_SIZE_COMPUTE, 1, 1], stream=stream)

    # ── 2. Reduce: merge S splits ─────────────────────────────────────────────
    persistent_reduce_kernel_v3b(partial_out, partial_lse, output, lse).launch(
        grid=[T, num_heads, 1], block=[BLOCK_SIZE_REDUCE, 1, 1], stream=stream,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_fused_persistent_v3b():
    T = cute.sym_int()
    num_pages, page_size = 8462, 64

    q_nope             = _fake(cute.BFloat16, (T, NUM_HEADS, D_CKV),                  (2, 1, 0), 16)
    q_pe               = _fake(cute.BFloat16, (T, NUM_HEADS, D_KPE),                  (2, 1, 0), 16)
    ckv_cache          = _fake(cute.BFloat16, (num_pages, page_size, D_CKV),          (2, 1, 0), 16)
    kpe_cache          = _fake(cute.BFloat16, (num_pages, page_size, D_KPE),          (2, 1, 0), 16)
    sparse_indices     = _fake(cute.Int32,    (T, TOP_K),                             (1, 0),     4)
    sm_scale           = 0.1352337788608801
    global_valid_count = _fake(cute.Int32,    (T_MAX,),                               (0,),       4)
    partial_out        = _fake(cute.Float32,  (T_MAX, NUM_HEADS, NUM_SPLITS, D_CKV),  (3, 2, 1, 0), 16)
    partial_lse        = _fake(cute.Float32,  (T_MAX, NUM_HEADS, NUM_SPLITS, 2),      (3, 2, 1, 0), 16)
    output             = _fake(cute.BFloat16, (T, NUM_HEADS, D_CKV),                  (2, 1, 0), 16)
    lse                = _fake(cute.Float32,  (T, NUM_HEADS),                         (1, 0),      4)
    stream             = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fused_persistent_v3b_launcher,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        global_valid_count, partial_out, partial_lse, output, lse,
        MAX_ACTIVE_CLUSTERS, stream,
        options="--enable-tvm-ffi",
    )


_compiled = compile_fused_persistent_v3b()

_global_valid_count = None
_partial_out = None
_partial_lse = None


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    global _global_valid_count, _partial_out, _partial_lse
    if _global_valid_count is None:
        _global_valid_count = torch.empty(T_MAX, dtype=torch.int32, device=output.device)
        _partial_out = torch.empty(
            T_MAX, NUM_HEADS, NUM_SPLITS, D_CKV, dtype=torch.float32, device=output.device)
        _partial_lse = torch.empty(
            T_MAX, NUM_HEADS, NUM_SPLITS, 2, dtype=torch.float32, device=output.device)
    _compiled(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        _global_valid_count, _partial_out, _partial_lse, output, lse,
    )


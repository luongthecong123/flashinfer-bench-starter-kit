"""
kv_split_dsmem_v3.py — DSMEM KV-split v3: push model + global num_valid.

Design:
  1. ALL blocks load entire sparse_indices[token, 0:2048] → smem (8 KB).
     Parallel reduction gives global_num_valid.  local_valid and
     active_splits are derived arithmetically.
  2. Each block allocates smem_all_out[num_splits * 512] and
     smem_all_lse[num_splits * 2].  Only block 0's copy is the master.
  3. After compute, each block PUSHES its partial to block 0's smem:
     - Block 0 (split 0): writes to row 0 locally (ss='cta')
     - Blocks 1-3: write to their row via mapa(ptr, 0) + store(ss='cluster')
  4. Single cluster barrier ensures all pushes land.
  5. Non-0 blocks exit.  Block 0 merges from its LOCAL smem — no DSMEM
     reads needed.  **No 2nd barrier.**
  6. OOB blocks: skip compute, skip write — block 0 knows active_splits
     and only merges active rows.

Grid:    [T*H, num_splits, 1]
Cluster: [1, num_splits, 1]   (num_splits = 4)
Block:   [1024, 1, 1]
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

import math
import torch

# ── Configuration ─────────────────────────────────────────────────────────────
DIM_SPLIT  = 512
TOP_K      = 2048
NUM_SPLITS = (TOP_K + DIM_SPLIT - 1) // DIM_SPLIT  # 4

BLOCK_SIZE = 1024
NUM_WARPS  = BLOCK_SIZE // 32  # 32
DIMS_PER_LANE: cutlass.Constexpr = 512 // 32  # 16

NUM_VEC_SCORE      : cutlass.Constexpr = 8
ITERS_PER_LANE_CKV : cutlass.Constexpr = (512 // 32) // 8  # 2

LN2 = 0.6931471805599453


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Single fused kernel with DSMEM cluster reduction — v3
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def kvsplit_dsmem(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    output: cute.Tensor,
    lse: cute.Tensor,
    stream):
    T, num_heads, head_dim_ckv = q_nope.shape
    num_splits: cutlass.Constexpr = NUM_SPLITS
    kvsplit_dsmem_v3_kernel(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
    ).launch(
        grid=[T * num_heads, num_splits, 1],
        block=[BLOCK_SIZE, 1, 1],
        cluster=[1, num_splits, 1],
        stream=stream,
    )


@cute.kernel
def kvsplit_dsmem_v3_kernel(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    output: cute.Tensor,
    lse: cute.Tensor):

    T, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = kpe_cache.shape[1]
    dim_split: cutlass.Constexpr = DIM_SPLIT
    num_splits: cutlass.Constexpr = NUM_SPLITS
    top_k: cutlass.Constexpr = TOP_K
    num_vec_score: cutlass.Constexpr = NUM_VEC_SCORE
    iters_per_lane_ckv: cutlass.Constexpr = ITERS_PER_LANE_CKV
    dims_per_lane: cutlass.Constexpr = DIMS_PER_LANE

    bidx_linear, _, _ = cute.arch.block_idx()
    token_idx = bidx_linear // num_heads
    head_idx  = bidx_linear % num_heads
    split_id  = cute.arch.block_idx_in_cluster()

    num_threads: cutlass.Constexpr = BLOCK_SIZE
    num_warps:   cutlass.Constexpr = NUM_WARPS
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    allocator = cutlass.utils.SmemAllocator()

    # 8 KB: ALL sparse indices for this request (global visibility)
    smem_sparse  = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((top_k,),      stride=(1,)),  4, None)
    smem_logits  = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((dim_split,),  stride=(1,)), 16, None)
    smem_red_i32 = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((32,),         stride=(1,)),  4, None)
    smem_red_f32 = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((32,),         stride=(1,)), 16, None)
    smem_q_nope  = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv,), stride=(1,)), 16, None)
    smem_q_pe    = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe,), stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout((num_warps, head_dim_ckv), stride=(head_dim_ckv, 1)), 16, None)

    # DSMEM push target: block 0's copy is the master destination
    # Each block allocates; peers write to block 0's copy via mapa
    smem_all_out = allocator.allocate_array(cutlass.Float32, num_elems=num_splits * 512)
    smem_all_lse = allocator.allocate_array(cutlass.Float32, num_elems=num_splits * 2)

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 1: ALL blocks load full sparse_indices → global_num_valid
    # ═══════════════════════════════════════════════════════════════════════════
    partial_cnt = 0
    for i in range(tidx, top_k, num_threads):
        idx = sparse_indices[token_idx, i]
        smem_sparse[i] = idx
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

    global_num_valid = smem_red_i32[0]

    # Derive this split's valid count arithmetically
    split_start = split_id * dim_split
    local_valid = global_num_valid - split_start
    if local_valid > dim_split:
        local_valid = dim_split
    if local_valid < cutlass.Int32(0):
        local_valid = cutlass.Int32(0)

    active_splits = (global_num_valid + dim_split - 1) // dim_split

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 2: Compute partials (block-uniform if/else)
    # ═══════════════════════════════════════════════════════════════════════════
    if local_valid == cutlass.Int32(0):
        # OOB block: no compute, no write — block 0 knows via active_splits
        pass
    else:
        # ── Valid block: full compute pipeline ────────────────────────────
        for i in range(tidx, head_dim_ckv, num_threads):
            smem_q_nope[i] = q_nope[token_idx, head_idx, i]
        for i in range(tidx, head_dim_kpe, num_threads):
            smem_q_pe[i] = q_pe[token_idx, head_idx, i]
        cute.arch.sync_threads()

        num_rounds = (local_valid + num_warps - 1) // num_warps
        q_nope_z = cute.zipped_divide(smem_q_nope, (num_vec_score,))

        # ── Score phase ───────────────────────────────────────────────────
        for round_idx in range(num_rounds):
            sparse_idx = round_idx * num_warps + warp_idx
            if sparse_idx < local_valid:
                cur_idx = smem_sparse[split_start + sparse_idx]

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

        # ── Softmax: max ──────────────────────────────────────────────────
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

        # ── Softmax: exp + sum + normalise ────────────────────────────────
        local_sum = cutlass.Float32(0)
        for idx in range(tidx, local_valid, num_threads):
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

        for i in range(tidx, local_valid, num_threads):
            smem_logits[i] = smem_logits[i] / row_sum
        cute.arch.sync_threads()

        # ── Output GEMV ───────────────────────────────────────────────────
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
                for k in range(dims_per_lane):
                    out_regs[k] += weight * cutlass.Float32(ckv_cache[kv_idx, k * wsize + lane_idx])

        for k in range(dims_per_lane):
            smem_partial[warp_idx, k * wsize + lane_idx] = out_regs[k]

        cute.arch.sync_threads()

        # ── Cross-warp reduce + PUSH to block 0's smem ───────────────────
        # Each block writes its 512-dim output to row split_id in block 0
        if split_id == cutlass.Int32(0):
            # Block 0: local store (ss='cta')
            for i in range(tidx, head_dim_ckv, num_threads):
                acc = cutlass.Float32(0)
                for w in range(num_warps):
                    acc += smem_partial[w, i]
                cute.arch.store(smem_all_out + i, acc, ss='cta')
            if tidx == 0:
                cute.arch.store(smem_all_lse, row_max, ss='cta')
                cute.arch.store(smem_all_lse + 1, row_sum, ss='cta')
        else:
            # Peers: push to block 0's smem via DSMEM write
            for i in range(tidx, head_dim_ckv, num_threads):
                acc = cutlass.Float32(0)
                for w in range(num_warps):
                    acc += smem_partial[w, i]
                dst = cute.arch.mapa(smem_all_out + split_id * dim_split + i, 0)
                cute.arch.store(dst, acc, ss='cluster')
            if tidx == 0:
                dst_max = cute.arch.mapa(smem_all_lse + split_id * 2, 0)
                dst_sum = cute.arch.mapa(smem_all_lse + split_id * 2 + 1, 0)
                cute.arch.store(dst_max, row_max, ss='cluster')
                cute.arch.store(dst_sum, row_sum, ss='cluster')

    # ═══════════════════════════════════════════════════════════════════════════
    # Single cluster barrier: ensures all pushes land in block 0's smem
    # ═══════════════════════════════════════════════════════════════════════════
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    # Non-0 blocks: exit (block 0 has all data locally, no 2nd barrier)
    # Block 0: merge from local smem_all_out / smem_all_lse
    if split_id == cutlass.Int32(0):
        if active_splits == cutlass.Int32(1):
            # Single split: output already normalised → write directly
            for d in range(tidx, head_dim_ckv, num_threads):
                val = cute.arch.load(smem_all_out + d, cutlass.Float32, ss='cta')
                output[token_idx, head_idx, d] = cutlass.BFloat16(val)
            if tidx == 0:
                s_max = cute.arch.load(smem_all_lse, cutlass.Float32, ss='cta')
                s_sum = cute.arch.load(smem_all_lse + 1, cutlass.Float32, ss='cta')
                lse[token_idx, head_idx] = (s_max + cute.math.log(s_sum)) / cutlass.Float32(LN2)
        else:
            # Multi-split: thread 0 reads lse + precomputes per-split scale
            if tidx == 0:
                g_max = -cutlass.Float32(math.inf)
                for s in range(num_splits):
                    if s < active_splits:
                        s_max = cute.arch.load(smem_all_lse + s * 2, cutlass.Float32, ss='cta')
                        if s_max > g_max:
                            g_max = s_max

                g_denom = cutlass.Float32(0)
                for s in range(num_splits):
                    if s < active_splits:
                        s_max = cute.arch.load(smem_all_lse + s * 2, cutlass.Float32, ss='cta')
                        s_sum = cute.arch.load(smem_all_lse + s * 2 + 1, cutlass.Float32, ss='cta')
                        g_denom += s_sum * cute.math.exp(s_max - g_max)

                for s in range(num_splits):
                    if s < active_splits:
                        s_max = cute.arch.load(smem_all_lse + s * 2, cutlass.Float32, ss='cta')
                        s_sum = cute.arch.load(smem_all_lse + s * 2 + 1, cutlass.Float32, ss='cta')
                        smem_red_f32[s] = s_sum * cute.math.exp(s_max - g_max) / g_denom
                    else:
                        smem_red_f32[s] = cutlass.Float32(0)

                lse[token_idx, head_idx] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

            cute.arch.sync_threads()

            # All threads merge output dims from local smem (no DSMEM!)
            for d in range(tidx, head_dim_ckv, num_threads):
                acc = cutlass.Float32(0)
                for s in range(num_splits):
                    if s < active_splits:
                        s_out = cute.arch.load(smem_all_out + s * dim_split + d, cutlass.Float32, ss='cta')
                        acc += s_out * smem_red_f32[s]
                output[token_idx, head_idx, d] = cutlass.BFloat16(acc)


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_kvsplit_dsmem():
    T = cute.sym_int()
    N = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (N, head_dim_ckv),            (1, 0),    16)
    kpe_cache      = _fake(cute.BFloat16, (N, head_dim_kpe),            (1, 0),    16)
    sparse_indices = _fake(cute.Int32,    (T, top_k_len),               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),               (1, 0),     4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kvsplit_dsmem,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


kvsplit_dsmem_compiled = compile_kvsplit_dsmem()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    kvsplit_dsmem_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)

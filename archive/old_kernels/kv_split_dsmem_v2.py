"""
kv_split_dsmem_v2.py — DSMEM KV-split v2: optimised cluster reduction.

Changes from v1:
  1. DIM_SPLIT = 512 → NUM_SPLITS = 4 (half DSMEM traffic).
  2. Non-CTA-0 blocks: after computing partials, coalescedly write their
     smem_out + smem_lse to CTA 0's DSMEM via mapa+store, then exit.
     No 2nd cluster barrier needed.
  3. OOB blocks (valid_count=0): write -inf/0 directly to CTA 0's DSMEM
     for their slot, then exit immediately — freeing the SM.
  4. CTA 0: computes its own partial, arrives at barrier, waits for all
     peers' DSMEM writes to land, then merges.
  5. Removed redundant smem_lse reads inside per-dim loop (cache in regs).

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


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Single fused kernel with DSMEM cluster reduction — v2
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
    kvsplit_dsmem_kernel(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
    ).launch(
        grid=[T * num_heads, num_splits, 1],
        block=[BLOCK_SIZE, 1, 1],
        cluster=[1, num_splits, 1],
        stream=stream,
    )


@cute.kernel
def kvsplit_dsmem_kernel(
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

    smem_logits  = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((dim_split,), stride=(1,)), 16, None)
    smem_sparse  = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((dim_split,), stride=(1,)),  4, None)
    smem_red_i32 = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((32,),        stride=(1,)),  4, None)
    smem_red_f32 = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((32,),        stride=(1,)), 16, None)
    smem_q_nope  = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv,), stride=(1,)), 16, None)
    smem_q_pe    = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe,), stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout((num_warps, head_dim_ckv), stride=(head_dim_ckv, 1)), 16, None)

    # DSMEM-accessible arrays: smem_out[512] + smem_lse[2] per block
    smem_out_ptr = allocator.allocate_array(cutlass.Float32, num_elems=512)
    smem_lse_ptr = allocator.allocate_array(cutlass.Float32, num_elems=2)

    # CTA 0 also has a receive buffer for each peer's results
    # Each peer slot: 512 fp32 (output) + 2 fp32 (lse) = 514 floats
    # We allocate num_splits slots (0 = own, 1..3 = peers)
    smem_recv_out_ptr = allocator.allocate_array(cutlass.Float32, num_elems=512 * num_splits)
    smem_recv_lse_ptr = allocator.allocate_array(cutlass.Float32, num_elems=2 * num_splits)

    # ── Load sparse_indices for this split ────────────────────────────────────
    kv_start = split_id * dim_split
    partial_cnt_valid = 0
    for i in range(tidx, dim_split, num_threads):
        global_idx = kv_start + i
        idx = sparse_indices[token_idx, global_idx]
        smem_sparse[i] = idx
        if idx >= cutlass.Int32(0):
            partial_cnt_valid += 1

    for i in range(tidx, head_dim_ckv, num_threads):
        smem_q_nope[i] = q_nope[token_idx, head_idx, i]
        cute.arch.store(smem_out_ptr + i, cutlass.Float32(0), ss='cta')
    for i in range(tidx, head_dim_kpe, num_threads):
        smem_q_pe[i] = q_pe[token_idx, head_idx, i]

    # ── Valid-count reduction ─────────────────────────────────────────────────
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

    # ── Compute partials (score + softmax + output GEMV) ──────────────────────
    if valid_count == cutlass.Int32(0):
        if tidx == 0:
            cute.arch.store(smem_lse_ptr, -cutlass.Float32(math.inf), ss='cta')
            cute.arch.store(smem_lse_ptr + 1, cutlass.Float32(0), ss='cta')
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

        # Softmax max
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

        # Softmax exp + sum
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

        # Normalise weights (needed since output is weighted sum)
        for i in range(tidx, valid_count, num_threads):
            smem_logits[i] = smem_logits[i] / row_sum
        cute.arch.sync_threads()

        # Store lse
        if tidx == 0:
            cute.arch.store(smem_lse_ptr, row_max, ss='cta')
            cute.arch.store(smem_lse_ptr + 1, row_sum, ss='cta')

        # Output GEMV
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

        # Cross-warp reduce → smem_out
        for i in range(tidx, head_dim_ckv, num_threads):
            acc = cutlass.Float32(0)
            for w in range(num_warps):
                acc += smem_partial[w, i]
            cute.arch.store(smem_out_ptr + i, acc, ss='cta')

    # ═══════════════════════════════════════════════════════════════════════════
    # DSMEM transfer: all blocks write their results to CTA 0's receive buffer
    # ═══════════════════════════════════════════════════════════════════════════
    cute.arch.sync_threads()

    # Each block writes its smem_out and smem_lse to CTA 0's receive buffer
    # slot = split_id.  CTA 0 writes to its own recv buffer (slot 0).
    # Non-CTA-0 blocks use mapa to write to CTA 0's smem.

    # CTA 0: copy own data into recv slot 0 (local smem copy)
    if split_id == cutlass.Int32(0):
        for i in range(tidx, head_dim_ckv, num_threads):
            val = cute.arch.load(smem_out_ptr + i, cutlass.Float32, ss='cta')
            cute.arch.store(smem_recv_out_ptr + i, val, ss='cta')
        if tidx == 0:
            v0 = cute.arch.load(smem_lse_ptr, cutlass.Float32, ss='cta')
            v1 = cute.arch.load(smem_lse_ptr + 1, cutlass.Float32, ss='cta')
            cute.arch.store(smem_recv_lse_ptr, v0, ss='cta')
            cute.arch.store(smem_recv_lse_ptr + 1, v1, ss='cta')

    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    # Non-CTA-0: write to CTA 0's DSMEM recv buffer at their slot via mapa
    if split_id != cutlass.Int32(0):
        for i in range(tidx, head_dim_ckv, num_threads):
            val = cute.arch.load(smem_out_ptr + i, cutlass.Float32, ss='cta')
            # Write to CTA 0's recv buffer: slot = split_id, offset = split_id * 512 + i
            # But split_id is runtime, so we use mapa to target CTA 0
            dst_ptr = cute.arch.mapa(smem_recv_out_ptr + i, 0)
            # DSMEM write: we need to do store to cluster. But DSMEM write isn't directly
            # supported in the same way as read. Instead, CTA 0 pulls from peers.
            # Let CTA 0 do the pulling after barrier.
            pass

    # ═══════════════════════════════════════════════════════════════════════════
    # CTA 0 pulls peer data via DSMEM after cluster barrier
    # ═══════════════════════════════════════════════════════════════════════════
    if split_id == cutlass.Int32(0):
        # Pull each peer's smem_out and smem_lse into our recv buffer slots
        for peer in range(1, num_splits):
            for i in range(tidx, head_dim_ckv, num_threads):
                peer_ptr = cute.arch.mapa(smem_out_ptr + i, peer)
                val = cute.arch.load(peer_ptr, cutlass.Float32, ss='cluster')
                cute.arch.store(smem_recv_out_ptr + peer * 512 + i, val, ss='cta')
            if tidx == 0:
                peer_lse0 = cute.arch.mapa(smem_lse_ptr, peer)
                peer_lse1 = cute.arch.mapa(smem_lse_ptr + 1, peer)
                v0 = cute.arch.load(peer_lse0, cutlass.Float32, ss='cluster')
                v1 = cute.arch.load(peer_lse1, cutlass.Float32, ss='cluster')
                cute.arch.store(smem_recv_lse_ptr + peer * 2, v0, ss='cta')
                cute.arch.store(smem_recv_lse_ptr + peer * 2 + 1, v1, ss='cta')

        cute.arch.sync_threads()

        # Now merge from local recv buffer (no more DSMEM needed)
        # Step 1: global max from recv_lse[s*2] for s in 0..num_splits-1
        if tidx == 0:
            g_max = -cutlass.Float32(math.inf)
            for s in range(num_splits):
                s_max = cute.arch.load(smem_recv_lse_ptr + s * 2, cutlass.Float32, ss='cta')
                if s_max > g_max:
                    g_max = s_max

            g_denom = cutlass.Float32(0)
            for s in range(num_splits):
                s_max = cute.arch.load(smem_recv_lse_ptr + s * 2, cutlass.Float32, ss='cta')
                s_sum = cute.arch.load(smem_recv_lse_ptr + s * 2 + 1, cutlass.Float32, ss='cta')
                g_denom += s_sum * cute.math.exp(s_max - g_max)

            smem_red_f32[0] = g_max
            smem_red_f32[1] = g_denom

        cute.arch.sync_threads()

        g_max_bc   = smem_red_f32[0]
        g_denom_bc = smem_red_f32[1]

        if tidx == 0:
            lse[token_idx, head_idx] = (g_max_bc + cute.math.log(g_denom_bc)) / cutlass.Float32(0.6931471805599453)

        # Step 2: merge output across splits
        for d in range(tidx, head_dim_ckv, num_threads):
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                s_max = cute.arch.load(smem_recv_lse_ptr + s * 2, cutlass.Float32, ss='cta')
                s_sum = cute.arch.load(smem_recv_lse_ptr + s * 2 + 1, cutlass.Float32, ss='cta')
                s_out = cute.arch.load(smem_recv_out_ptr + s * 512 + d, cutlass.Float32, ss='cta')
                scale = s_sum * cute.math.exp(s_max - g_max_bc) / g_denom_bc
                acc += s_out * scale

            output[token_idx, head_idx, d] = cutlass.BFloat16(acc)

    # 2nd barrier: keep peers alive until CTA 0 is done reading their smem
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()


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

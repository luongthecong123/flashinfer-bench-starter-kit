"""
kv_split_dsmem.py — Single-kernel KV-split using DSMEM for cross-split reduction.

Instead of two separate kernels, all splits within a (token, head) group are
launched as a thread block cluster. After computing local partial results, all
blocks synchronize via cluster_arrive/wait, then block 0 in the cluster pulls
peer blocks' smem results via DSMEM (mapa + load ss='cluster') and performs
the online softmax merge in-place.

Grid:    [T * H, num_splits, 1]     ← or [T, H, num_splits] if cluster dim=Z
Cluster: [1, num_splits, 1]         ← splits within same (token,head) co-scheduled
Block:   [1024, 1, 1]               ← 32 warps

Each block:
  1. Count valid tokens in its split range → early exit if 0
  2. Score phase (vectorized thr_warp)
  3. Local softmax → local max, local sum_exp, local weights
  4. Output GEMV → partial weighted sum in smem_out[D] fp32
  5. Store local max + sum_exp in smem_lse[2] fp32
  6. cluster_arrive / cluster_wait
  7. Block 0 (split_id=0) pulls all peers' smem_out + smem_lse via DSMEM,
     does online softmax merge, writes final output + lse

Key DSMEM APIs:
  - cute.arch.block_idx_in_cluster()    → split_id within cluster
  - cute.arch.cluster_arrive()          → release fence, signal peers
  - cute.arch.cluster_wait()            → wait for all cluster peers
  - cute.arch.mapa(smem_ptr, peer_idx)  → get DSMEM address of peer's smem
  - cute.arch.load(ptr, dtype, ss='cluster')  → read from peer's smem via DSMEM

DIM_SPLIT = 256, NUM_SPLITS = 8
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

import math
import torch

# ── Configuration ─────────────────────────────────────────────────────────────
DIM_SPLIT  = 256
TOP_K      = 2048
NUM_SPLITS = (TOP_K + DIM_SPLIT - 1) // DIM_SPLIT  # 8

BLOCK_SIZE = 1024
NUM_WARPS  = BLOCK_SIZE // 32  # 32
DIMS_PER_LANE: cutlass.Constexpr = 512 // 32  # 16

# Vectorized score (from thr_warp)
NUM_VEC_SCORE      : cutlass.Constexpr = 8
ITERS_PER_LANE_CKV : cutlass.Constexpr = (512 // 32) // 8  # 2


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Single fused kernel with DSMEM cluster reduction
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def kvsplit_dsmem(
    q_nope: cute.Tensor,        # [T, H, 512]  bf16
    q_pe: cute.Tensor,          # [T, H, 64]   bf16
    ckv_cache: cute.Tensor,     # [N, 512]     bf16 (flattened)
    kpe_cache: cute.Tensor,     # [N, 64]      bf16 (flattened)
    sparse_indices: cute.Tensor,# [T, 2048]    int32
    sm_scale: cutlass.Constexpr,
    output: cute.Tensor,        # [T, H, 512]  bf16
    lse: cute.Tensor,           # [T, H]       fp32
    stream):
    T, num_heads, head_dim_ckv = q_nope.shape
    num_splits: cutlass.Constexpr = NUM_SPLITS
    # Grid: [T*H, num_splits, 1] — linearize token×head into X dim
    # Cluster: [1, num_splits, 1] — splits co-scheduled
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

    # Grid [T*H, num_splits, 1] — decode token/head from X
    bidx_linear, _, _ = cute.arch.block_idx()
    token_idx = bidx_linear // num_heads
    head_idx  = bidx_linear % num_heads
    split_id  = cute.arch.block_idx_in_cluster()  # 0..num_splits-1

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
    # smem_out_ptr: final reduced output per block (512 fp32) — used for DSMEM pull
    # Must be allocate_array (returns cute.ptr) so we can use mapa() on it.
    smem_out_ptr = allocator.allocate_array(cutlass.Float32, num_elems=512)
    # smem_lse_ptr: [max, sum_exp] for this split — used for DSMEM pull
    smem_lse_ptr = allocator.allocate_array(cutlass.Float32, num_elems=2)

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

    # ── If no valid tokens in this split, set lse to (-inf, 0) ────────────────
    if valid_count == cutlass.Int32(0):
        if tidx == 0:
            cute.arch.store(smem_lse_ptr, -cutlass.Float32(math.inf), ss='cta')
            cute.arch.store(smem_lse_ptr + 1, cutlass.Float32(0), ss='cta')
        # smem_out already zeroed above
    else:
        num_rounds = (valid_count + num_warps - 1) // num_warps

        # ── Score phase: vectorized BF16 loads ────────────────────────────────
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

        # ── Local softmax: max ────────────────────────────────────────────────
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

        # ── Local softmax: exp + sum ──────────────────────────────────────────
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

        # Normalize weights
        for i in range(tidx, valid_count, num_threads):
            smem_logits[i] = smem_logits[i] / row_sum

        cute.arch.sync_threads()

        # Write local max + sum_exp to smem for DSMEM pull
        if tidx == 0:
            cute.arch.store(smem_lse_ptr, row_max, ss='cta')
            cute.arch.store(smem_lse_ptr + 1, row_sum, ss='cta')

        # ── Output GEMV: per-warp register accumulation ──────────────────────
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

        # Cross-warp reduce → smem_out_ptr
        for i in range(tidx, head_dim_ckv, num_threads):
            acc = cutlass.Float32(0)
            for w in range(num_warps):
                acc += smem_partial[w, i]
            cute.arch.store(smem_out_ptr + i, acc, ss='cta')

    # ═══════════════════════════════════════════════════════════════════════════
    # DSMEM cluster reduction
    # ═══════════════════════════════════════════════════════════════════════════
    cute.arch.sync_threads()       # all threads done writing smem_out + smem_lse
    cute.arch.cluster_arrive()     # fence (release) + signal cluster peers
    cute.arch.cluster_wait()       # wait for all peers

    # Block 0 in the cluster does the merge
    if split_id == cutlass.Int32(0):
        # Step 1: Find global max across all splits' smem_lse_ptr[0]
        g_max = cute.arch.load(smem_lse_ptr, cutlass.Float32, ss='cta')  # own max
        for peer in range(1, num_splits):
            peer_lse_ptr = cute.arch.mapa(smem_lse_ptr, peer)
            peer_max     = cute.arch.load(peer_lse_ptr, cutlass.Float32, ss='cluster')
            if peer_max > g_max:
                g_max = peer_max

        # Step 2: Compute global denominator (single-threaded, num_splits=8 is tiny)
        # Only thread 0 needs to do this, but broadcast via smem
        if tidx == 0:
            g_denom = cutlass.Float32(0)
            own_sum = cute.arch.load(smem_lse_ptr + 1, cutlass.Float32, ss='cta')
            own_max_val = cute.arch.load(smem_lse_ptr, cutlass.Float32, ss='cta')
            g_denom = own_sum * cute.math.exp(own_max_val - g_max)
            for peer in range(1, num_splits):
                peer_lse_ptr_p = cute.arch.mapa(smem_lse_ptr, peer)
                p_max  = cute.arch.load(peer_lse_ptr_p, cutlass.Float32, ss='cluster')
                p_sum_ptr = cute.arch.mapa(smem_lse_ptr + 1, peer)
                p_sum  = cute.arch.load(p_sum_ptr, cutlass.Float32, ss='cluster')
                g_denom += p_sum * cute.math.exp(p_max - g_max)
            smem_red_f32[0] = g_max
            smem_red_f32[1] = g_denom

        cute.arch.sync_threads()

        g_max_bc   = smem_red_f32[0]
        g_denom_bc = smem_red_f32[1]

        # Write LSE
        if tidx == 0:
            lse[token_idx, head_idx] = (g_max_bc + cute.math.log(g_denom_bc)) / cutlass.Float32(0.6931471805599453)

        # Step 3: Merge output across splits (parallel over D, each thread handles 1 dim)
        # Thread tidx handles dim tidx (and tidx+512 if block>512, but we use 1024 threads, D=512)
        for d in range(tidx, head_dim_ckv, num_threads):
            # Own contribution
            own_max_val = cute.arch.load(smem_lse_ptr, cutlass.Float32, ss='cta')
            own_sum_val = cute.arch.load(smem_lse_ptr + 1, cutlass.Float32, ss='cta')
            own_out_val = cute.arch.load(smem_out_ptr + d, cutlass.Float32, ss='cta')
            scale = own_sum_val * cute.math.exp(own_max_val - g_max_bc) / g_denom_bc
            acc = own_out_val * scale

            # Pull peer contributions via DSMEM
            for peer in range(1, num_splits):
                peer_out_ptr = cute.arch.mapa(smem_out_ptr + d, peer)
                peer_out_val = cute.arch.load(peer_out_ptr, cutlass.Float32, ss='cluster')
                peer_lse_ptr_p = cute.arch.mapa(smem_lse_ptr, peer)
                p_max    = cute.arch.load(peer_lse_ptr_p, cutlass.Float32, ss='cluster')
                p_sum_ptr = cute.arch.mapa(smem_lse_ptr + 1, peer)
                p_sum    = cute.arch.load(p_sum_ptr, cutlass.Float32, ss='cluster')
                p_scale  = p_sum * cute.math.exp(p_max - g_max_bc) / g_denom_bc
                acc += peer_out_val * p_scale

            output[token_idx, head_idx, d] = cutlass.BFloat16(acc)
    
    # ── 2nd cluster sync: keep all blocks alive until block 0 is done ─────────
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

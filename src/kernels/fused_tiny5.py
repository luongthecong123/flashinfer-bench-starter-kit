"""
fused_tiny5: parallel output accumulation via per-warp register accumulators.

Key change vs fused_tiny3:
  - Output phase: instead of `for j in range(valid_count): ... (serial HBM chain)`,
    each warp accumulates its own keys in registers across num_rounds rounds,
    then writes partial sums into smem_partial[32, 512], which is reduced by all 1024
    threads into smem_output.

  Why this helps:
    - fused_tiny3 output: valid_count serial L2 reads, dependency chain ~173 ns each
    - fused_tiny5 output: num_rounds independent L2 reads per warp (parallel),
      final cross-warp reduce is pure smem reads (no L2 dependency chain)
    - WL3 (valid=52): 52 serial iterations → 2 rounds × parallel, then 1 smem-only reduce

Smem budget (per block):
  Existing (~19 KB):  logits_scaled(8) + sparse_idx(8) + reductions(0.25) + output(2) + q_nope(1) + q_pe(0.125)
  New:                smem_partial = 32 × 512 × 4B = 64 KB
  Total: ~83 KB < 228 KB (B200)

Grid: [T, 16, 1]  Block: 1024 threads = 32 warps
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

import math
import torch


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


BLOCK_SIZE = 1024
NUM_WARPS  = BLOCK_SIZE // 32   # 32
# Each lane of a warp owns 512/32 = 16 output dimensions
DIMS_PER_LANE: cutlass.Constexpr = 512 // 32   # 16


@cute.jit
def fused_dsa_v5(
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
    fused_dsa_kernel_v5(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
    ).launch(grid=[T, num_heads, 1], block=[BLOCK_SIZE, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_v5(
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
    top_k_len    = 2048
    dims_per_lane: cutlass.Constexpr = DIMS_PER_LANE  # 16

    bidx, bidy, _ = cute.arch.block_idx()
    num_threads: cutlass.Constexpr = BLOCK_SIZE
    num_warps:   cutlass.Constexpr = NUM_WARPS
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE  # 32

    allocator = cutlass.utils.SmemAllocator()

    smem_logits_scaled   = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((top_k_len),    stride=(1)), 16, None)
    smem_sparse_idx      = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((top_k_len),    stride=(1)),  4, None)
    smem_reduction_int32 = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((32),           stride=(1)),  4, None)
    smem_reduction_fp32  = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((32),           stride=(1)), 16, None)
    smem_output          = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((head_dim_ckv), stride=(1)), 16, None)
    smem_q_nope          = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv), stride=(1)), 16, None)
    smem_q_pe            = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe), stride=(1)), 16, None)
    # Cross-warp partial output sums: warp w owns smem_partial[w*512 .. (w+1)*512-1]
    # 32 × 512 × fp32 = 64 KB
    smem_partial         = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout((num_warps * head_dim_ckv), stride=(1)), 16, None)

    # ── Load phase ────────────────────────────────────────────────────────────
    partial_cnt_valid = 0
    for i in range(tidx, top_k_len, num_threads):
        idx = sparse_indices[bidx, i]
        smem_sparse_idx[i] = idx
        if idx >= cutlass.Int32(0):
            partial_cnt_valid += 1

    for i in range(tidx, head_dim_ckv, num_threads):
        smem_q_nope[i] = q_nope[bidx, bidy, i]
        smem_output[i] = cutlass.Float32(0)
    for i in range(tidx, head_dim_kpe, num_threads):
        smem_q_pe[i] = q_pe[bidx, bidy, i]

    # ── Valid-count reduction ─────────────────────────────────────────────────
    sum_valid = warp_reduce(partial_cnt_valid, lambda a, b: a + b, width=32)
    if lane_idx == 0:
        smem_reduction_int32[warp_idx] = sum_valid
    cute.arch.sync_threads()

    if warp_idx == 0:
        val = smem_reduction_int32[lane_idx]
        sum_valid = warp_reduce(val, lambda a, b: a + b, width=num_warps)
        smem_reduction_int32[0] = sum_valid
    cute.arch.sync_threads()

    valid_count = smem_reduction_int32[0]
    num_rounds  = (valid_count + num_warps - 1) // num_warps

    # ── Score phase (unchanged from fused_tiny3) ──────────────────────────────
    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < valid_count:
            cur_idx = smem_sparse_idx[sparse_idx]

            sum_partial = cutlass.Float32(0)
            for k_idx in range(head_dim_ckv // wsize):
                q_n = cutlass.Float32(smem_q_nope[k_idx * wsize + lane_idx])
                cv  = cutlass.Float32(ckv_cache[cur_idx, k_idx * wsize + lane_idx])
                sum_partial += q_n * cv
            for k_idx in range(head_dim_kpe // wsize):
                q_p = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                kv  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                sum_partial += q_p * kv

            s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_logits_scaled[sparse_idx] = s * sm_scale

    cute.arch.sync_threads()

    # ── Softmax: pass 1 — block-wide max ──────────────────────────────────────
    partial_max = -cutlass.Float32(math.inf)
    for idx in range(tidx, valid_count, num_threads):
        v = smem_logits_scaled[idx]
        if v > partial_max:
            partial_max = v

    max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
    if lane_idx == 0:
        smem_reduction_fp32[warp_idx] = max_val
    cute.arch.sync_threads()

    if warp_idx == 0:
        val = smem_reduction_fp32[lane_idx]
        max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
        smem_reduction_fp32[0] = max_val
    cute.arch.sync_threads()

    row_max = smem_reduction_fp32[0]

    # ── Softmax: pass 2 — block-wide exp+sum ─────────────────────────────────
    partial_sum = cutlass.Float32(0)
    for idx in range(tidx, valid_count, num_threads):
        partial_sum += cute.math.exp(smem_logits_scaled[idx] - row_max)

    sum_val = warp_reduce(partial_sum, lambda a, b: a + b, width=32)
    if lane_idx == 0:
        smem_reduction_fp32[warp_idx] = sum_val
    cute.arch.sync_threads()

    if warp_idx == 0:
        val = smem_reduction_fp32[lane_idx]
        sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
        smem_reduction_fp32[0] = sum_val
    cute.arch.sync_threads()

    row_sum = smem_reduction_fp32[0]

    if tidx == 0:
        lse[bidx, bidy] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(0.6931471805599453)

    for i in range(tidx, valid_count, num_threads):
        smem_logits_scaled[i] = cute.math.exp(smem_logits_scaled[i] - row_max) / row_sum

    cute.arch.sync_threads()

    # ── Output phase: per-warp register accumulation ──────────────────────────
    # Each warp accumulates its own partial weighted sum in registers.
    # Lane lane_idx of warp warp_idx owns dims: k*wsize + lane_idx for k in 0..dims_per_lane-1
    # i.e. dims: lane_idx, lane_idx+32, lane_idx+64, ..., lane_idx+480
    #
    # For num_rounds rounds, warp w processes key (round*num_warps + w).
    # No dependency between rounds — purely parallel within each warp.
    # Final cross-warp reduce: smem_partial[w, :] → smem_output via 1024 threads.

    out0  = cutlass.Float32(0)
    out1  = cutlass.Float32(0)
    out2  = cutlass.Float32(0)
    out3  = cutlass.Float32(0)
    out4  = cutlass.Float32(0)
    out5  = cutlass.Float32(0)
    out6  = cutlass.Float32(0)
    out7  = cutlass.Float32(0)
    out8  = cutlass.Float32(0)
    out9  = cutlass.Float32(0)
    out10 = cutlass.Float32(0)
    out11 = cutlass.Float32(0)
    out12 = cutlass.Float32(0)
    out13 = cutlass.Float32(0)
    out14 = cutlass.Float32(0)
    out15 = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        j = round_idx * num_warps + warp_idx
        if j < valid_count:
            kv_idx = smem_sparse_idx[j]
            weight = smem_logits_scaled[j]
            out0  += weight * cutlass.Float32(ckv_cache[kv_idx,  0 * wsize + lane_idx])
            out1  += weight * cutlass.Float32(ckv_cache[kv_idx,  1 * wsize + lane_idx])
            out2  += weight * cutlass.Float32(ckv_cache[kv_idx,  2 * wsize + lane_idx])
            out3  += weight * cutlass.Float32(ckv_cache[kv_idx,  3 * wsize + lane_idx])
            out4  += weight * cutlass.Float32(ckv_cache[kv_idx,  4 * wsize + lane_idx])
            out5  += weight * cutlass.Float32(ckv_cache[kv_idx,  5 * wsize + lane_idx])
            out6  += weight * cutlass.Float32(ckv_cache[kv_idx,  6 * wsize + lane_idx])
            out7  += weight * cutlass.Float32(ckv_cache[kv_idx,  7 * wsize + lane_idx])
            out8  += weight * cutlass.Float32(ckv_cache[kv_idx,  8 * wsize + lane_idx])
            out9  += weight * cutlass.Float32(ckv_cache[kv_idx,  9 * wsize + lane_idx])
            out10 += weight * cutlass.Float32(ckv_cache[kv_idx, 10 * wsize + lane_idx])
            out11 += weight * cutlass.Float32(ckv_cache[kv_idx, 11 * wsize + lane_idx])
            out12 += weight * cutlass.Float32(ckv_cache[kv_idx, 12 * wsize + lane_idx])
            out13 += weight * cutlass.Float32(ckv_cache[kv_idx, 13 * wsize + lane_idx])
            out14 += weight * cutlass.Float32(ckv_cache[kv_idx, 14 * wsize + lane_idx])
            out15 += weight * cutlass.Float32(ckv_cache[kv_idx, 15 * wsize + lane_idx])

    # Write per-warp partial sums into smem_partial[warp_idx, :]
    smem_partial[warp_idx * head_dim_ckv +  0 * wsize + lane_idx] = out0
    smem_partial[warp_idx * head_dim_ckv +  1 * wsize + lane_idx] = out1
    smem_partial[warp_idx * head_dim_ckv +  2 * wsize + lane_idx] = out2
    smem_partial[warp_idx * head_dim_ckv +  3 * wsize + lane_idx] = out3
    smem_partial[warp_idx * head_dim_ckv +  4 * wsize + lane_idx] = out4
    smem_partial[warp_idx * head_dim_ckv +  5 * wsize + lane_idx] = out5
    smem_partial[warp_idx * head_dim_ckv +  6 * wsize + lane_idx] = out6
    smem_partial[warp_idx * head_dim_ckv +  7 * wsize + lane_idx] = out7
    smem_partial[warp_idx * head_dim_ckv +  8 * wsize + lane_idx] = out8
    smem_partial[warp_idx * head_dim_ckv +  9 * wsize + lane_idx] = out9
    smem_partial[warp_idx * head_dim_ckv + 10 * wsize + lane_idx] = out10
    smem_partial[warp_idx * head_dim_ckv + 11 * wsize + lane_idx] = out11
    smem_partial[warp_idx * head_dim_ckv + 12 * wsize + lane_idx] = out12
    smem_partial[warp_idx * head_dim_ckv + 13 * wsize + lane_idx] = out13
    smem_partial[warp_idx * head_dim_ckv + 14 * wsize + lane_idx] = out14
    smem_partial[warp_idx * head_dim_ckv + 15 * wsize + lane_idx] = out15

    cute.arch.sync_threads()

    # Cross-warp reduce: each of 1024 threads sums over 32 warps for its dim
    # threads 0-511 handle dims 0-511; threads 512-1023 are parallel duplicates
    # Use num_threads to cover all 512 dims with stride num_threads
    for i in range(tidx, head_dim_ckv, num_threads):
        acc = cutlass.Float32(0)
        for w in range(num_warps):
            acc += smem_partial[w * head_dim_ckv + i]
        smem_output[i] = acc

    cute.arch.sync_threads()

    # ── Epilogue ──────────────────────────────────────────────────────────────
    for i in range(tidx, head_dim_ckv, num_threads):
        output[bidx, bidy, i] = cutlass.BFloat16(smem_output[i])


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_fused_dsa_v5():
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
        fused_dsa_v5,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


fused_dsa_v5_compiled = compile_fused_dsa_v5()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    fused_dsa_v5_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)

"""
impl2: 32-warp parallel-keys letmecook kernel.

Same fused DSA as letmecook.py but with 1024 threads (32 warps).
Each warp independently scores 1 key at a time → 32 keys scored in parallel.
The outer loop over 2048 entries is divided by 32 → only 64 rounds.

Grid: [T, 16, 1]  — one block per (token, head)
Block: 1024 threads = 32 warps
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cute.testing import benchmark, JitArguments

from typing import Tuple
import math
import torch


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def fused_dsa_v2(
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

    fused_dsa_kernel_v2(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
    ).launch(grid=[T, num_heads, 1], block=[1024, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_v2(
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
    top_k_len = 2048

    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    num_threads = bdimx
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    wsize = cute.arch.WARP_SIZE
    num_warps = bdimx // wsize

    allocator = cutlass.utils.SmemAllocator()
    smem_score_nope = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_score_pe = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_logits_scaled = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_sparse_idx = allocator.allocate_tensor(cutlass.Int32, cute.make_layout((top_k_len), stride=(1)), 4, None)
    smem_valid_count = allocator.allocate_tensor(cutlass.Int32, cute.make_layout((1), stride=(1)), 4, None)

    # Cooperatively load sparse_indices into smem — 1024 threads, 2 loads each
    for i in range(tidx, top_k_len, num_threads):
        smem_sparse_idx[i] = sparse_indices[bidx, i]

    cute.arch.sync_threads()

    q_nope_local = q_nope[bidx, bidy, None]
    q_pe_local = q_pe[bidx, bidy, None]

    # ── Score phase: 32 warps process 32 keys in parallel ──
    # Each warp handles keys at indices: warp_idx, warp_idx+32, warp_idx+64, ...
    # Each warp does full 512-dim dot (nope) + 64-dim dot (pe) for its key
    num_valid_indices = cutlass.Int32(0)
    for round_idx in range(top_k_len // num_warps):
        sparse_idx = round_idx * num_warps + warp_idx
        cur_idx = smem_sparse_idx[sparse_idx]

        if cur_idx >= cutlass.Int32(0):
            lane_idx = cute.arch.lane_idx()

            # nope dot product: 512 dims, 32 lanes → 16 iterations
            sum_partial_nope = cutlass.Float32(0)
            for k_idx in range(head_dim_ckv // wsize):
                q_nope_val = cutlass.Float32(q_nope_local[k_idx * wsize + lane_idx])
                ckv_val = cutlass.Float32(ckv_cache[cur_idx, k_idx * wsize + lane_idx])
                sum_partial_nope += q_nope_val * ckv_val
            sum_nope = warp_reduce(sum_partial_nope, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_score_nope[sparse_idx] = sum_nope

            # pe dot product: 64 dims, 32 lanes → 2 iterations
            sum_partial_pe = cutlass.Float32(0)
            for k_idx in range(head_dim_kpe // wsize):
                q_pe_val = cutlass.Float32(q_pe_local[k_idx * wsize + lane_idx])
                kpe_val = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                sum_partial_pe += q_pe_val * kpe_val
            sum_pe = warp_reduce(sum_partial_pe, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_score_pe[sparse_idx] = sum_pe

    cute.arch.sync_threads()

    # ── Compute num_valid: thread 0 scans backwards to find last valid ──
    if tidx == 0:
        num_valid = cutlass.Int32(0)
        for i in range(top_k_len):
            if smem_sparse_idx[i] >= cutlass.Int32(0):
                num_valid = cutlass.Int32(i + 1)
        smem_valid_count[0] = num_valid
    cute.arch.sync_threads()
    valid_count = smem_valid_count[0]

    # ── Scale logits ──
    for i in range(tidx, valid_count, num_threads):
        logits_scaled = sm_scale * (smem_score_nope[i] + smem_score_pe[i])
        smem_logits_scaled[i] = logits_scaled

    cute.arch.sync_threads()

    # ── Softmax (serial on thread 0) ──
    if tidx == 0:
        row_max = smem_logits_scaled[0]
        for i in range(valid_count):
            if smem_logits_scaled[i] > row_max:
                row_max = smem_logits_scaled[i]

        row_sum = cutlass.Float32(0)
        for i in range(valid_count):
            row_sum += cute.math.exp(smem_logits_scaled[i] - row_max)

        lse[bidx, bidy] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(0.6931471805599453)

        for i in range(valid_count):
            smem_logits_scaled[i] = cute.math.exp(smem_logits_scaled[i] - row_max) / row_sum

    cute.arch.sync_threads()

    # ── Output accumulation: 1024 threads, 512 dims → 2 elements/thread ──
    smem_output = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((head_dim_ckv), stride=(1)), 16, None)

    for i in range(tidx, head_dim_ckv, num_threads):
        smem_output[i] = cutlass.Float32(0)
    cute.arch.sync_threads()

    for j in range(valid_count):
        kv_idx = smem_sparse_idx[j]
        attn_weight = smem_logits_scaled[j]
        for i in range(tidx, head_dim_ckv, num_threads):
            smem_output[i] += attn_weight * cutlass.Float32(ckv_cache[kv_idx, i])

    cute.arch.sync_threads()

    for i in range(tidx, head_dim_ckv, num_threads):
        output[bidx, bidy, i] = cutlass.BFloat16(smem_output[i])


# ── Compilation ─────────────────────────────────────────────────────

def fake_wrapper(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=assumed_align)


def compile_fused_dsa_v2():
    T = cute.sym_int()
    N = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048

    q_nope = fake_wrapper(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe = fake_wrapper(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache = fake_wrapper(cute.BFloat16, (N, head_dim_ckv), (1, 0), 16)
    kpe_cache = fake_wrapper(cute.BFloat16, (N, head_dim_kpe), (1, 0), 16)
    sparse_indices = fake_wrapper(cute.Int32, (T, top_k_len), (1, 0), 4)
    sm_scale = 0.1352337788608801
    output = fake_wrapper(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse = fake_wrapper(cute.Float32, (T, num_heads), (1, 0), 4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fused_dsa_v2,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


fused_dsa_v2_compiled = compile_fused_dsa_v2()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    fused_dsa_v2_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)

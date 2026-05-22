"""impl3: 32-warp parallel-keys letmecook using gathered dense buffers.

Uses kc[T,2048,512], Kp[T,2048,64], max_valid[T] from the gather kernel.
No sparse_indices needed — data is already gathered into dense layout.

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

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'zen'))
from gather import gather_compiled


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def fused_dsa_v3(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    kc: cute.Tensor,
    Kp: cute.Tensor,
    max_valid: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    output: cute.Tensor,
    lse: cute.Tensor,
    stream):
    T, num_heads, head_dim_ckv = q_nope.shape

    fused_dsa_kernel_v3(
        q_nope, q_pe, kc, Kp, max_valid, sm_scale, output, lse
    ).launch(grid=[T, num_heads, 1], block=[1024, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_v3(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    kc: cute.Tensor,
    Kp: cute.Tensor,
    max_valid: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    output: cute.Tensor,
    lse: cute.Tensor):

    T, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = Kp.shape[2]
    top_k_len = 2048
    num_warps = 32

    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    num_threads = bdimx
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    wsize = cute.arch.WARP_SIZE

    allocator = cutlass.utils.SmemAllocator()
    smem_score_nope = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_score_pe = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_logits_scaled = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)

    # Load valid_count from precomputed max_valid
    valid_count = max_valid[bidx]

    q_nope_local = q_nope[bidx, bidy, None]
    q_pe_local = q_pe[bidx, bidy, None]

    # ── Score phase: 32 warps process 32 keys in parallel ──
    # Only loop over valid entries (ceil(valid_count / 32) rounds)
    for round_idx in range(top_k_len // num_warps):
        key_idx = round_idx * num_warps + warp_idx
        if key_idx < valid_count:
            lane_idx = cute.arch.lane_idx()

            # nope dot product: 512 dims, 32 lanes → 16 iterations
            sum_partial_nope = cutlass.Float32(0)
            for k_idx in range(head_dim_ckv // wsize):
                q_nope_val = cutlass.Float32(q_nope_local[k_idx * wsize + lane_idx])
                kc_val = cutlass.Float32(kc[bidx, key_idx, k_idx * wsize + lane_idx])
                sum_partial_nope += q_nope_val * kc_val
            sum_nope = warp_reduce(sum_partial_nope, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_score_nope[key_idx] = sum_nope

            # pe dot product: 64 dims, 32 lanes → 2 iterations
            sum_partial_pe = cutlass.Float32(0)
            for k_idx in range(head_dim_kpe // wsize):
                q_pe_val = cutlass.Float32(q_pe_local[k_idx * wsize + lane_idx])
                kp_val = cutlass.Float32(Kp[bidx, key_idx, k_idx * wsize + lane_idx])
                sum_partial_pe += q_pe_val * kp_val
            sum_pe = warp_reduce(sum_partial_pe, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_score_pe[key_idx] = sum_pe

    cute.arch.sync_threads()

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

    # ── Output accumulation: 1024 threads, 512 dims ──
    smem_output = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((head_dim_ckv), stride=(1)), 16, None)

    for i in range(tidx, head_dim_ckv, num_threads):
        smem_output[i] = cutlass.Float32(0)
    cute.arch.sync_threads()

    for j in range(valid_count):
        attn_weight = smem_logits_scaled[j]
        for i in range(tidx, head_dim_ckv, num_threads):
            smem_output[i] += attn_weight * cutlass.Float32(kc[bidx, j, i])

    cute.arch.sync_threads()

    for i in range(tidx, head_dim_ckv, num_threads):
        output[bidx, bidy, i] = cutlass.BFloat16(smem_output[i])


# ── Compilation ─────────────────────────────────────────────────────

def fake_wrapper(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=assumed_align)


def compile_fused_dsa_v3():
    T = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048

    q_nope = fake_wrapper(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe = fake_wrapper(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    kc = fake_wrapper(cute.BFloat16, (T, top_k_len, head_dim_ckv), (2, 1, 0), 16)
    Kp = fake_wrapper(cute.BFloat16, (T, top_k_len, head_dim_kpe), (2, 1, 0), 16)
    max_valid = fake_wrapper(cute.Int32, (T,), (0,), 4)
    sm_scale = 0.1352337788608801
    output = fake_wrapper(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse = fake_wrapper(cute.Float32, (T, num_heads), (1, 0), 4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fused_dsa_v3,
        q_nope, q_pe, kc, Kp, max_valid, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


fused_dsa_v3_compiled = compile_fused_dsa_v3()

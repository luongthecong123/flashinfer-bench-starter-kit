"""
fused_tiny4: per-round smem CKV staging for score + output phases.

Key change vs fused_tiny3:
  - Allocate smem_ckv_stage[32, 512] BF16 (32 KB) and smem_kpe_stage[32, 64] BF16 (4 KB)
  - Score phase: all 1024 threads cooperatively stage 32 CKV/KPE rows into smem,
    then each warp dots against its own smem row (avoids scattered L2 reads in compute).
  - Output phase: same cooperative staging per round; serial inner accumulation
    reads from smem (2-4 ns latency) instead of L2 (≥50 ns per access), turning
    the 288-iteration serial L2-latency chain into a 9-round smem-latency chain.

Expected improvement: output phase from ~51 µs → ~3-5 µs (smem vs L2 latency × serial depth).

Grid: [T, 16, 1]  Block: 1024 threads = 32 warps
Smem budget: ~19 KB existing + 32 KB CKV stage + 4 KB KPE stage ≈ 55 KB < 228 KB (B200)
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream

import math
import torch


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


BLOCK_SIZE = 1024
NUM_WARPS  = BLOCK_SIZE // 32   # 32


@cute.jit
def fused_dsa_v4(
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

    fused_dsa_kernel_v4(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
    ).launch(grid=[T, num_heads, 1], block=[BLOCK_SIZE, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_v4(
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
    num_threads: cutlass.Constexpr = BLOCK_SIZE
    num_warps:   cutlass.Constexpr = NUM_WARPS
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE   # 32

    allocator = cutlass.utils.SmemAllocator()

    # Persistent across phases
    smem_logits_scaled    = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((top_k_len), stride=(1)), 16, None)
    smem_sparse_idx       = allocator.allocate_tensor(cutlass.Int32,   cute.make_layout((top_k_len), stride=(1)),  4, None)
    smem_reduction_int32  = allocator.allocate_tensor(cutlass.Int32,   cute.make_layout((32),        stride=(1)),  4, None)
    smem_reduction_fp32   = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((32),        stride=(1)), 16, None)
    smem_output           = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((head_dim_ckv), stride=(1)), 16, None)
    smem_q_nope           = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv), stride=(1)), 16, None)
    smem_q_pe             = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe), stride=(1)), 16, None)

    # Per-round staging buffers: flat layout, row w at offset w*head_dim_ckv / w*head_dim_kpe
    # smem_ckv_stage: 32 × 512 BF16 = 32 KB
    # smem_kpe_stage: 32 × 64  BF16 =  4 KB
    smem_ckv_stage = allocator.allocate_tensor(cutlass.BFloat16,
        cute.make_layout((num_warps * head_dim_ckv), stride=(1)), 16, None)
    smem_kpe_stage = allocator.allocate_tensor(cutlass.BFloat16,
        cute.make_layout((num_warps * head_dim_kpe), stride=(1)), 16, None)

    # ── Load phase ────────────────────────────────────────────────────────────
    # 1. Cooperatively load sparse_indices + count valid entries
    partial_cnt_valid = 0
    for i in range(tidx, top_k_len, num_threads):
        idx = sparse_indices[bidx, i]
        smem_sparse_idx[i] = idx
        if idx >= cutlass.Int32(0):
            partial_cnt_valid += 1

    # 2. Load q_nope, q_pe, init output accumulator
    for i in range(tidx, head_dim_ckv, num_threads):
        smem_q_nope[i] = q_nope[bidx, bidy, i]
        smem_output[i] = cutlass.Float32(0)
    for i in range(tidx, head_dim_kpe, num_threads):
        smem_q_pe[i] = q_pe[bidx, bidy, i]

    # ── Valid-count reduction ──────────────────────────────────────────────────
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

    # ── Score phase: per-round cooperative CKV/KPE staging ───────────────────
    # Each round:
    #   1. All 1024 threads cooperatively stage 32 CKV rows + 32 KPE rows into smem.
    #   2. Valid warps compute dot product from smem (smem latency, not L2).
    #   Result: dot products use fast smem reads; HBM loads are hidden behind cooperative staging.
    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx   # which key this warp scores

        # Safe fallback: warps with sparse_idx >= valid_count load from key 0 (harmless).
        raw_kv = smem_sparse_idx[sparse_idx]
        kv_idx = raw_kv if raw_kv >= cutlass.Int32(0) else cutlass.Int32(0)

        # Stage CKV row for warp_idx into smem_ckv_stage[warp_idx, :]
        for k_idx in range(head_dim_ckv // wsize):
            smem_ckv_stage[warp_idx * head_dim_ckv + k_idx * wsize + lane_idx] = \
                ckv_cache[kv_idx, k_idx * wsize + lane_idx]

        # Stage KPE row for warp_idx into smem_kpe_stage[warp_idx, :]
        for k_idx in range(head_dim_kpe // wsize):
            smem_kpe_stage[warp_idx * head_dim_kpe + k_idx * wsize + lane_idx] = \
                kpe_cache[kv_idx, k_idx * wsize + lane_idx]

        cute.arch.sync_threads()

        # Valid warps: dot product from smem
        if sparse_idx < valid_count:
            sum_partial = cutlass.Float32(0)

            for k_idx in range(head_dim_ckv // wsize):
                q  = cutlass.Float32(smem_q_nope[k_idx * wsize + lane_idx])
                cv = cutlass.Float32(smem_ckv_stage[warp_idx * head_dim_ckv + k_idx * wsize + lane_idx])
                sum_partial += q * cv

            for k_idx in range(head_dim_kpe // wsize):
                q  = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                kv = cutlass.Float32(smem_kpe_stage[warp_idx * head_dim_kpe + k_idx * wsize + lane_idx])
                sum_partial += q * kv

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

    # ── Output phase: per-round cooperative CKV staging + smem accumulation ───
    # Same round structure as score.  Each round:
    #   1. All 1024 threads cooperatively stage 32 CKV rows into smem_ckv_stage
    #      (same bandwidth as current serial HBM reads, but 32× more parallelism).
    #   2. Serial inner loop over 32 staged rows reads from smem (2-4 ns latency)
    #      instead of L2 (≥50 ns per scattered access).
    # This replaces the 288-deep serial L2-latency chain with 9 smem-latency rounds.
    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx

        raw_kv = smem_sparse_idx[sparse_idx]
        kv_idx = raw_kv if raw_kv >= cutlass.Int32(0) else cutlass.Int32(0)

        # Cooperative stage (CKV only — KPE not needed for output)
        for k_idx in range(head_dim_ckv // wsize):
            smem_ckv_stage[warp_idx * head_dim_ckv + k_idx * wsize + lane_idx] = \
                ckv_cache[kv_idx, k_idx * wsize + lane_idx]

        cute.arch.sync_threads()

        # Serial inner: 32 staged rows, read from smem → no serial L2-latency chain
        for inner in range(num_warps):
            j = round_idx * num_warps + inner
            if j < valid_count:
                weight = smem_logits_scaled[j]
                for i in range(tidx, head_dim_ckv, num_threads):
                    smem_output[i] += weight * cutlass.Float32(smem_ckv_stage[inner * head_dim_ckv + i])

        cute.arch.sync_threads()

    # ── Epilogue: write output to gmem ────────────────────────────────────────
    for i in range(tidx, head_dim_ckv, num_threads):
        output[bidx, bidy, i] = cutlass.BFloat16(smem_output[i])


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_fused_dsa_v4():
    T = cute.sym_int()
    N = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048

    q_nope        = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe          = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache     = _fake(cute.BFloat16, (N, head_dim_ckv),            (1, 0),    16)
    kpe_cache     = _fake(cute.BFloat16, (N, head_dim_kpe),            (1, 0),    16)
    sparse_indices = _fake(cute.Int32,   (T, top_k_len),               (1, 0),     4)
    sm_scale      = 0.1352337788608801
    output        = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse           = _fake(cute.Float32,  (T, num_heads),               (1, 0),     4)
    stream        = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fused_dsa_v4,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


fused_dsa_v4_compiled = compile_fused_dsa_v4()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    fused_dsa_v4_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)

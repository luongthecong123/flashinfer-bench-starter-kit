"""
fused_tinyv8: fused_tiny5v2 + interleaved multi-row score phase (FastGEMV-inspired).

Score phase change vs v5v2:
  - 4 rows per warp per round (was 1), reusing q_nope/q_pe from smem
  - Interleaved ckv/kpe loads from 4 different sparse indices per inner iteration
  - 4× fewer outer rounds, 4× fewer smem q reads
  - Warp reductions batched after all dot-product work

Everything else (valid_count, softmax, output phase) identical to v5v2.

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
DIMS_PER_LANE: cutlass.Constexpr = 512 // 32   # 16
ROWS_PER_WARP: cutlass.Constexpr = 4           # interleaved rows per warp per round


@cute.jit
def fused_dsa_v8(
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
    fused_dsa_kernel_v8(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
    ).launch(grid=[T, num_heads, 1], block=[BLOCK_SIZE, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_v8(
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
    rows_per_warp: cutlass.Constexpr = ROWS_PER_WARP  # 4

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
    smem_partial         = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout((num_warps, head_dim_ckv), stride=(head_dim_ckv, 1)), 16, None)

    # ── Load phase ────────────────────────────────────────────────────────────
    partial_cnt_valid = 0
    for i in range(tidx, top_k_len, num_threads):
        idx = sparse_indices[bidx, i]
        smem_sparse_idx[i] = idx if idx >= cutlass.Int32(0) else cutlass.Int32(0)
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
    # 4 rows per warp per round → effective stride = num_warps * rows_per_warp = 128
    rows_per_round: cutlass.Constexpr = num_warps * rows_per_warp  # 128
    num_rounds  = (valid_count + rows_per_round - 1) // rows_per_round

    # ── Score phase: interleaved 4 rows per warp ─────────────────────────────
    for round_idx in range(num_rounds):
        base_sparse = round_idx * rows_per_round + warp_idx * rows_per_warp

        # Load 4 sparse cache indices (already clamped ≥0 in load phase)
        ci0 = smem_sparse_idx[base_sparse + 0]
        ci1 = smem_sparse_idx[base_sparse + 1]
        ci2 = smem_sparse_idx[base_sparse + 2]
        ci3 = smem_sparse_idx[base_sparse + 3]

        # Register accumulators for 4 rows
        sums = cute.make_rmem_tensor(
            cute.make_layout((rows_per_warp,), stride=(1,)),
            cutlass.Float32,
        )
        for r in range(rows_per_warp):
            sums[r] = cutlass.Float32(0)

        # Interleaved dot product: ckv (512 dims)
        for k_idx in range(head_dim_ckv // wsize):
            dim = k_idx * wsize + lane_idx
            qn = cutlass.Float32(smem_q_nope[dim])
            sums[0] = sums[0] + qn * cutlass.Float32(ckv_cache[ci0, dim])
            sums[1] = sums[1] + qn * cutlass.Float32(ckv_cache[ci1, dim])
            sums[2] = sums[2] + qn * cutlass.Float32(ckv_cache[ci2, dim])
            sums[3] = sums[3] + qn * cutlass.Float32(ckv_cache[ci3, dim])

        # Interleaved dot product: kpe (64 dims)
        for k_idx in range(head_dim_kpe // wsize):
            dim = k_idx * wsize + lane_idx
            qp = cutlass.Float32(smem_q_pe[dim])
            sums[0] = sums[0] + qp * cutlass.Float32(kpe_cache[ci0, dim])
            sums[1] = sums[1] + qp * cutlass.Float32(kpe_cache[ci1, dim])
            sums[2] = sums[2] + qp * cutlass.Float32(kpe_cache[ci2, dim])
            sums[3] = sums[3] + qp * cutlass.Float32(kpe_cache[ci3, dim])

        # Batched warp reduction + unguarded writeback
        # (writes beyond valid_count land in valid smem, never read by softmax)
        for r in range(rows_per_warp):
            sums[r] = warp_reduce(sums[r], lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_logits_scaled[base_sparse + 0] = sums[0] * sm_scale
            smem_logits_scaled[base_sparse + 1] = sums[1] * sm_scale
            smem_logits_scaled[base_sparse + 2] = sums[2] * sm_scale
            smem_logits_scaled[base_sparse + 3] = sums[3] * sm_scale

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

    # ── Output phase: per-warp register accumulation (same as v5v2) ───────────
    num_rounds_out = (valid_count + num_warps - 1) // num_warps

    out_regs = cute.make_rmem_tensor(
        cute.make_layout((dims_per_lane,), stride=(1,)),
        cutlass.Float32,
    )
    for k in range(dims_per_lane):
        out_regs[k] = cutlass.Float32(0)

    for round_idx in range(num_rounds_out):
        j = round_idx * num_warps + warp_idx
        if j < valid_count:
            kv_idx = smem_sparse_idx[j]
            weight = smem_logits_scaled[j]
            for k in range(dims_per_lane):
                out_regs[k] += weight * cutlass.Float32(ckv_cache[kv_idx, k * wsize + lane_idx])

    # Write per-warp partial sums into smem_partial[warp_idx, :]
    for k in range(dims_per_lane):
        smem_partial[warp_idx, k * wsize + lane_idx] = out_regs[k]

    cute.arch.sync_threads()

    # Cross-warp reduce
    for i in range(tidx, head_dim_ckv, num_threads):
        acc = cutlass.Float32(0)
        for w in range(num_warps):
            acc += smem_partial[w, i]
        smem_output[i] = acc

    cute.arch.sync_threads()

    # ── Epilogue ──────────────────────────────────────────────────────────────
    for i in range(tidx, head_dim_ckv, num_threads):
        output[bidx, bidy, i] = cutlass.BFloat16(smem_output[i])


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_fused_dsa_v8():
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
        fused_dsa_v8,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


fused_dsa_v8_compiled = compile_fused_dsa_v8()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    fused_dsa_v8_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)

"""
fused_tiny5v3: each warp owns a tile of 16 output dims; warp-reduce replaces cross-warp smem.

Key change vs fused_tiny5v2:
  - Output phase: each warp w is responsible for dims [w*16 .. w*16+15].
    Lane l handles all keys at index (round*32 + l).
    After accumulation, warp_reduce sums contributions across 32 lanes (keys).
    Lane 0 stores 16 bf16 values directly to gmem.

  Why this helps vs v2:
    - Eliminates smem_partial (64 KB) and smem_output (2 KB)
    - Eliminates cross-warp reduce loop + sync
    - Lower smem pressure → better occupancy

Smem budget (per block):
  logits_scaled(8) + sparse_idx(8) + reductions(0.5) + q_nope(1) + q_pe(0.125) ≈ 18 KB
  (vs ~83 KB in v2)

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
# Each warp owns 512/32 = 16 consecutive output dimensions
DIMS_PER_WARP: cutlass.Constexpr = 512 // 32   # 16


@cute.jit
def fused_dsa_v5v3(
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
    fused_dsa_kernel_v5v3(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
    ).launch(grid=[T, num_heads, 1], block=[BLOCK_SIZE, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_v5v3(
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
    dims_per_warp: cutlass.Constexpr = DIMS_PER_WARP  # 16

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
    smem_q_nope          = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv), stride=(1)), 16, None)
    smem_q_pe            = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe), stride=(1)), 16, None)
    # ckv staging buffer: 32 keys × 512 dims × bf16 = 32 KB, reused each round
    smem_ckv             = allocator.allocate_tensor(cutlass.BFloat16,
        cute.make_layout((wsize * head_dim_ckv), stride=(1)), 16, None)

    # ── Load phase ────────────────────────────────────────────────────────────
    partial_cnt_valid = 0
    for i in range(tidx, top_k_len, num_threads):
        idx = sparse_indices[bidx, i]
        smem_sparse_idx[i] = idx
        if idx >= cutlass.Int32(0):
            partial_cnt_valid += 1

    for i in range(tidx, head_dim_ckv, num_threads):
        smem_q_nope[i] = q_nope[bidx, bidy, i]
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

    # ── Output phase v3: each warp owns dims_per_warp consecutive output dims ──
    # Warp w → dims [w*dims_per_warp .. (w+1)*dims_per_warp - 1]
    # Lane l → processes key (round*wsize + l) for all rounds
    # smem_ckv[32, 512] staged cooperatively each round (coalesced gmem loads)
    # After accumulation: warp_reduce sums 32 lanes' contributions per dim
    # Lane 0 stores dims_per_warp values directly to gmem

    out_regs = cute.make_rmem_tensor(
        cute.make_layout((dims_per_warp,), stride=(1,)),
        cutlass.Float32,
    )
    for k in range(dims_per_warp):
        out_regs[k] = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        base_j = round_idx * wsize
        # Cooperatively stage 32 ckv rows into smem: 1024 threads × 16 elements each
        # Index i → key_slot = i // head_dim_ckv, dim = i % head_dim_ckv
        # Thread layout: tidx=0..511 → key0 dims 0..511 (coalesced), tidx=512..1023 → key1, etc.
        cute.arch.sync_threads()
        for i in range(tidx, wsize * head_dim_ckv, num_threads):
            key_slot = i // head_dim_ckv
            dim = i % head_dim_ckv
            j = base_j + key_slot
            if j < valid_count:
                smem_ckv[i] = ckv_cache[smem_sparse_idx[j], dim]
        cute.arch.sync_threads()

        j = base_j + lane_idx
        if j < valid_count:
            weight = smem_logits_scaled[j]
            for k in range(dims_per_warp):
                out_regs[k] += weight * cutlass.Float32(smem_ckv[lane_idx * head_dim_ckv + warp_idx * dims_per_warp + k])

    # Warp reduce: sum contributions across all 32 lanes (different keys)
    for k in range(dims_per_warp):
        out_regs[k] = warp_reduce(out_regs[k], lambda a, b: a + b)

    # Lane 0 writes directly to gmem
    if lane_idx == 0:
        for k in range(dims_per_warp):
            output[bidx, bidy, warp_idx * dims_per_warp + k] = cutlass.BFloat16(out_regs[k])


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_fused_dsa_v5v3():
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
        fused_dsa_v5v3,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


fused_dsa_v5v3_compiled = compile_fused_dsa_v5v3()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    fused_dsa_v5v3_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)

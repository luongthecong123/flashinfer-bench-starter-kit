"""
fused_tiny_thr_warp_ldgv1b: fused_tiny5v2 + vectorized score (thr_warp) + vectorized output (ldgv1b).

Score phase change vs v5v2:
  - Uses cute.zipped_divide + .load() for vectorized BF16 loads (LDG.128: 8×bf16 = 16B)
  - TensorSSA multiply + reduce for thread-level partial sums
  - 512-dim nope part: 2 vectorized iterations (was 16 scalar)
  - 64-dim PE part: kept scalar (only 2 iterations per lane)

Output phase change vs v5v2:
  - LDG.128 vectorized loads (8×bf16 = 16B per load) via cute.zipped_divide
  - 3D smem_partial layout [vec_group, warp, vec] with padding to avoid bank conflicts
  - Warp butterfly reduce for cross-warp reduction
  - Coalesced smem_output → global epilogue

Everything else (load, valid_count, softmax) identical to v5v2.

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

# Vectorized score constants
NUM_VEC_SCORE      : cutlass.Constexpr = 8       # 8 BF16 per load = LDG.128
ITERS_PER_LANE_CKV : cutlass.Constexpr = (512 // 32) // 8   # 2

# Output vectorization constants (BF16 cache → LDG.128)
VEC_OUT             : cutlass.Constexpr = 8                                     # 8 × bf16 = 128 bits
NUM_VEC_GROUPS_OUT  : cutlass.Constexpr = 512 // 8                              # 64
GROUPS_PER_PASS_OUT : cutlass.Constexpr = 32                                    # one group per lane
NUM_DIM_PASSES_OUT  : cutlass.Constexpr = NUM_VEC_GROUPS_OUT // GROUPS_PER_PASS_OUT  # 2
VG_PER_WARP_OUT     : cutlass.Constexpr = NUM_VEC_GROUPS_OUT // NUM_WARPS       # 2

# 3D smem strides: [VG, WARP, VEC] — padded to avoid bank conflicts
SMEM_S_VEC_OUT  : cutlass.Constexpr = 1
SMEM_S_WARP_OUT : cutlass.Constexpr = VEC_OUT + 1                              # 9
SMEM_S_VG_OUT   : cutlass.Constexpr = (NUM_WARPS + 1) * SMEM_S_WARP_OUT        # 297


@cute.jit
def fused_dsa_thr_warp_ldgv1b(
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
    fused_dsa_kernel_thr_warp_ldgv1b(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse
    ).launch(grid=[T, num_heads, 1], block=[BLOCK_SIZE, 1, 1], stream=stream)


@cute.kernel
def fused_dsa_kernel_thr_warp_ldgv1b(
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

    # Score vectorization constexprs
    num_vec_score      : cutlass.Constexpr = NUM_VEC_SCORE       # 8
    iters_per_lane_ckv : cutlass.Constexpr = ITERS_PER_LANE_CKV  # 2

    # Output vectorization constexprs
    vec_out            : cutlass.Constexpr = VEC_OUT              # 8
    num_vec_groups_out : cutlass.Constexpr = NUM_VEC_GROUPS_OUT   # 64
    groups_per_pass_out: cutlass.Constexpr = GROUPS_PER_PASS_OUT  # 32
    num_dim_passes_out : cutlass.Constexpr = NUM_DIM_PASSES_OUT   # 2
    vg_per_warp_out    : cutlass.Constexpr = VG_PER_WARP_OUT      # 2
    s_vg_out           : cutlass.Constexpr = SMEM_S_VG_OUT        # 297
    s_warp_out         : cutlass.Constexpr = SMEM_S_WARP_OUT      # 9
    s_vec_out          : cutlass.Constexpr = SMEM_S_VEC_OUT       # 1

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
    # 3D smem_partial with bank-conflict padding: [vec_group, warp, vec]
    smem_partial         = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout(
            (num_vec_groups_out, num_warps, vec_out),
            stride=(s_vg_out, s_warp_out, s_vec_out),
        ), 16, None)

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

    # ── Score phase: vectorized BF16 loads (LDG.128) ─────────────────────────
    q_nope_z = cute.zipped_divide(smem_q_nope, (num_vec_score,))

    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < valid_count:
            cur_idx = smem_sparse_idx[sparse_idx]

            # Vectorized nope dot product (512 dims, 2 iterations)
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

            # Scalar PE dot product (64 dims, 2 iterations)
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

    # ── Output phase: LDG.128 vectorized loads + 3D smem ─────────────────────
    out_regs = cute.make_rmem_tensor(
        cute.make_layout((num_dim_passes_out, vec_out), stride=(vec_out, 1)),
        cutlass.Float32,
    )
    for dp in range(num_dim_passes_out):
        for e in range(vec_out):
            out_regs[dp, e] = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        j = round_idx * num_warps + warp_idx
        if j < valid_count:
            kv_idx = smem_sparse_idx[j]
            weight = smem_logits_scaled[j]

            V_row = ckv_cache[kv_idx, None]
            V_z   = cute.zipped_divide(V_row, (vec_out,))

            for dp in range(num_dim_passes_out):
                group = dp * groups_per_pass_out + lane_idx
                frag  = V_z[(None, (group,))].load()   # LDG.128: 8 BF16

                for v in range(vec_out):
                    out_regs[dp, v] += weight * cutlass.Float32(frag[v])

    # Write partial sums → 3D smem_partial
    for dp in range(num_dim_passes_out):
        vg = dp * groups_per_pass_out + lane_idx
        for e in range(vec_out):
            smem_partial[vg, warp_idx, e] = out_regs[dp, e]

    cute.arch.sync_threads()

    # Cross-warp reduce via butterfly → smem_output
    for vg_off in range(vg_per_warp_out):
        vg = warp_idx * vg_per_warp_out + vg_off
        for e in range(vec_out):
            val = smem_partial[vg, lane_idx, e]
            val = warp_reduce(val, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_output[vg * vec_out + e] = val

    cute.arch.sync_threads()

    # ── Coalesced epilogue: smem_output → global output ───────────────────────
    for i in range(tidx, head_dim_ckv, num_threads):
        output[bidx, bidy, i] = cutlass.BFloat16(smem_output[i])


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_fused_dsa_thr_warp_ldgv1b():
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
        fused_dsa_thr_warp_ldgv1b,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, stream,
        options="--enable-tvm-ffi"
    )


fused_dsa_thr_warp_ldgv1b_compiled = compile_fused_dsa_thr_warp_ldgv1b()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    fused_dsa_thr_warp_ldgv1b_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)

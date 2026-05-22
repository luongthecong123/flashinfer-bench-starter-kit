"""kv_split_xor_pdl_v3_SSA_v2_gmem_static.py — v3 + SSA_v2 + direct GMEM, no single-split branch.

Always writes to partial_out, always reduces. No branching.
256 threads × vec_size=2 = 512 dims exactly.

Design:
  Grid (compute): [num_heads, num_splits, 1] × 1024 threads → 128 SMs
    - Upfront smem_sparse load + valid_count as prologue (every block)
    - griddepcontrol_launch_dependents() fired after prologue
    - Persistent T-loop with XOR swizzle

  Grid (reduce):  [T, num_heads, 1] × 256 threads → 16 SMs (8 blocks/SM)
    - Prologue: count valid indices for this T_idx (overlaps compute writes)
    - griddepcontrol_wait() after prologue
    - Each block reduces one (T_idx, head_idx) pair

  Both launched with use_pdl=True.
  Compute's epilogue (partial_out/partial_lse writes) overlaps with
  reduce's prologue (sparse_indices valid count).
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN = 16, 512, 64, 2048
NUM_PAGES, PAGE_SIZE = 8462, 64
T_MAX = 8
NUM_SPLITS = 8
DIM_SPLIT = (TOP_K_LEN + NUM_SPLITS - 1) // NUM_SPLITS
LN2 = 0.6931471805599453

NUM_THREADS = 1024
NUM_WARPS = NUM_THREADS // 32
VEC_SIZE_CKV = 8
VEC_SIZE_KPE = 2
VEC_SIZE_OUT = 16
ITERS_PER_LANE_CKV = HEAD_DIM_CKV // (32 * VEC_SIZE_CKV)

SPARSE_THR_PER_T = 128
NUM_WARPS_PER_T = SPARSE_THR_PER_T // 32

NUM_THREADS_REDUCE = 256
NUM_WARPS_REDUCE = NUM_THREADS_REDUCE // 32  # 8

VEC_REDUCE = 2  # 256 threads × 2 = 512 dims, no loop needed

SENTINEL_SKIP = float("inf")


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def count_valid_indices(
    sparse_indices:   cute.Tensor,        # (T, top_k_len) i32  — global
    smem_sparse:      cute.Tensor,        # (T_max, top_k_len) i32 — smem cache
    smem_red_i32:     cute.Tensor,        # (T_max, 32) i32     — smem scratch
    smem_num_valid:   cute.Tensor,        # (T_max,) i32        — smem output
    T:                cute.Numeric,
    tidx:             cute.Numeric,
    warp_idx:         cute.Numeric,
    top_k_len:        cutlass.Constexpr,
    sparse_thr_per_T: cutlass.Constexpr,
    num_warps_per_T:  cutlass.Constexpr,
) -> None:
    """Load sparse_indices into smem_sparse and count non-negative entries."""
    thr_idx_per_T  = tidx % sparse_thr_per_T
    lane_idx_per_T = thr_idx_per_T % cute.arch.WARP_SIZE
    wg_per_T_idx   = tidx // sparse_thr_per_T
    warp_per_T_idx = warp_idx % num_warps_per_T

    partial_cnt = 0
    if wg_per_T_idx < T:
        for i in range(thr_idx_per_T, top_k_len, sparse_thr_per_T):
            idx = sparse_indices[wg_per_T_idx, i]
            smem_sparse[wg_per_T_idx, i] = idx
            if idx >= cutlass.Int32(0):
                partial_cnt += 1

        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
        if lane_idx_per_T == 0:
            smem_red_i32[wg_per_T_idx, warp_per_T_idx] = cnt_sum

        cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                          number_of_threads=sparse_thr_per_T)

        if warp_per_T_idx == 0:
            val     = smem_red_i32[wg_per_T_idx, lane_idx_per_T]
            cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps_per_T)
            smem_red_i32[wg_per_T_idx, 0] = cnt_sum

        cute.arch.barrier(barrier_id=wg_per_T_idx + 1,
                          number_of_threads=sparse_thr_per_T)

        smem_num_valid[wg_per_T_idx] = smem_red_i32[wg_per_T_idx, 0]


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Host JIT: launch compute (128 SMs) then reduce (16 SMs) with PDL
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def kvsplit_xor_pdl(
    q_nope:         cute.Tensor,
    q_pe:           cute.Tensor,
    ckv_cache:      cute.Tensor,
    kpe_cache:      cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale:       cutlass.Constexpr,
    partial_out:    cute.Tensor,
    partial_lse:    cute.Tensor,
    output:         cute.Tensor,
    lse:            cute.Tensor,
    stream):
    T, num_heads, head_dim_ckv = q_nope.shape

    N: cutlass.Constexpr = NUM_PAGES * PAGE_SIZE
    ckv_flat = cute.make_tensor(
        ckv_cache.iterator,
        cute.make_layout((N, head_dim_ckv), stride=(head_dim_ckv, 1)))
    kpe_flat = cute.make_tensor(
        kpe_cache.iterator,
        cute.make_layout((N, q_pe.shape[2]), stride=(q_pe.shape[2], 1)))

    # Compute: 128 blocks = num_heads × num_splits = 16 × 8
    kvsplit_compute_kernel(
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
    ).launch(grid=[NUM_HEADS, NUM_SPLITS, 1], block=[NUM_THREADS, 1, 1],
             stream=stream, use_pdl=True)

    # Reduce: T×16 blocks with 256 threads each → 8 blocks/SM on 16 SMs
    kvsplit_reduce_kernel(
        sparse_indices, partial_out, partial_lse, output, lse,
    ).launch(grid=[T, NUM_HEADS, 1], block=[NUM_THREADS_REDUCE, 1, 1],
             stream=stream, use_pdl=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1: Compute — XOR-persistent over T, shared prologue
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_compute_kernel(
    q_nope:         cute.Tensor,        # (T,16,512)
    q_pe:           cute.Tensor,        # (T,16, 64)
    ckv_flat:       cute.Tensor,        # (N, 512)
    kpe_flat:       cute.Tensor,        # (N,  64)
    sparse_indices: cute.Tensor,        # (T, 2048)
    sm_scale:       cutlass.Constexpr,
    partial_out:    cute.Tensor,        # (T_MAX, 16, 8, 512)
    partial_lse:    cute.Tensor,        # (T_MAX, 16, 8, 2)
    output:         cute.Tensor,        # (T, 16, 512)
    lse:            cute.Tensor,        # (T, 16)
):
    T, _, _ = q_nope.shape
    head_dim_ckv:   cutlass.Constexpr = HEAD_DIM_CKV
    head_dim_kpe:   cutlass.Constexpr = HEAD_DIM_KPE
    top_k_len:      cutlass.Constexpr = TOP_K_LEN
    dim_split:      cutlass.Constexpr = DIM_SPLIT
    num_splits:     cutlass.Constexpr = NUM_SPLITS
    num_threads:    cutlass.Constexpr = NUM_THREADS
    num_warps:      cutlass.Constexpr = NUM_WARPS
    vec_size_ckv:   cutlass.Constexpr = VEC_SIZE_CKV
    vec_size_kpe:   cutlass.Constexpr = VEC_SIZE_KPE
    vec_size_out:   cutlass.Constexpr = VEC_SIZE_OUT
    iters_per_lane_ckv: cutlass.Constexpr = ITERS_PER_LANE_CKV
    sparse_thr_per_T:   cutlass.Constexpr = SPARSE_THR_PER_T
    num_warps_per_T:    cutlass.Constexpr = NUM_WARPS_PER_T
    t_max:          cutlass.Constexpr = T_MAX

    bidx, bidy, _ = cute.arch.block_idx()  # head_idx, split_idx
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    wsize = cute.arch.WARP_SIZE

    head_idx = bidx
    split_idx_old = bidy

    # ── SMEM allocation (shared prologue layout) ─────────────────────────────
    alloc = cutlass.utils.SmemAllocator()
    smem_sparse      = _smem(alloc, cutlass.Int32,    (t_max, top_k_len),           (top_k_len, 1),     4)  # 64 KB
    smem_num_valid   = _smem(alloc, cutlass.Int32,    (t_max,),                     (1,),               4)
    smem_logits      = _smem(alloc, cutlass.Float32,  (dim_split,),                 (1,),              16)
    smem_red_i32     = _smem(alloc, cutlass.Int32,    (t_max, 32),                  (32, 1),            4)
    smem_max_red_f32 = _smem(alloc, cutlass.Float32,  (32,),                        (1,),              16)
    smem_sum_red_f32 = _smem(alloc, cutlass.Float32,  (32,),                        (1,),              16)
    smem_q_nope      = _smem(alloc, cutlass.BFloat16, (t_max, head_dim_ckv),        (head_dim_ckv, 1), 16)
    smem_q_pe        = _smem(alloc, cutlass.BFloat16, (t_max, head_dim_kpe),        (head_dim_kpe, 1), 16)
    smem_partial     = _smem(alloc, cutlass.Float32,  (num_warps, head_dim_ckv),    (head_dim_ckv, 1), 16)
    # No smem_out — write directly to gmem from smem_partial

    # ── Prologue: load sparse_indices + Q + valid count ──────────────────────
    wg_per_T_idx = tidx // sparse_thr_per_T
    thr_idx_per_T = tidx % sparse_thr_per_T

    if wg_per_T_idx < T:
        for i in range(thr_idx_per_T, head_dim_ckv, sparse_thr_per_T):
            smem_q_nope[wg_per_T_idx, i] = q_nope[wg_per_T_idx, head_idx, i]
        for i in range(thr_idx_per_T, head_dim_kpe, sparse_thr_per_T):
            smem_q_pe[wg_per_T_idx, i] = q_pe[wg_per_T_idx, head_idx, i]

    count_valid_indices(
        sparse_indices, smem_sparse, smem_red_i32, smem_num_valid,
        T, tidx, warp_idx,
        top_k_len, sparse_thr_per_T, num_warps_per_T,
    )

    cute.arch.sync_threads()

    # ── PDL: fire dependent launch after prologue ────────────────────────────
    cute.arch.griddepcontrol_launch_dependents()

    # ── Vectorized views ─────────────────────────────────────────────────────
    smem_q_nope_ = cute.zipped_divide(smem_q_nope, (1, vec_size_ckv))
    ckv_flat_    = cute.zipped_divide(ckv_flat,     (1, vec_size_ckv))
    kpe_flat_    = cute.zipped_divide(kpe_flat,     (1, vec_size_kpe))
    smem_q_pe_   = cute.zipped_divide(smem_q_pe,   (1, vec_size_kpe))

    # ── Persistent T-loop with XOR swizzle ───────────────────────────────────
    for T_idx in range(T):
        split_idx_new = (T_idx + split_idx_old) % num_splits

        num_valid_T = smem_num_valid[T_idx]
        split_start = split_idx_new * dim_split
        is_OOB = split_start >= num_valid_T

        if not is_OOB:
            local_valid = min(num_valid_T - split_start, dim_split)
            num_rounds = (local_valid + num_warps - 1) // num_warps

            # ── Score ────────────────────────────────────────────────────────
            for round_idx in range(num_rounds):
                sparse_idx = round_idx * num_warps + warp_idx
                if sparse_idx < local_valid:
                    cur_idx = smem_sparse[T_idx, split_start + sparse_idx]
                    ckv_row_ = ckv_flat_[(0, None), (cur_idx, None)]
                    kpe_row_ = kpe_flat_[(0, None), (cur_idx, None)]

                    sum_partial = cutlass.Float32(0)

                    for it in range(iters_per_lane_ckv):
                        rest_idx = it * wsize + lane_idx
                        qn_vec = smem_q_nope_[(0, None), (T_idx, rest_idx)].load()
                        ckv_vec = ckv_row_[None, rest_idx].load()
                        for i in range(vec_size_ckv):
                            sum_partial += cutlass.Float32(qn_vec[i]) * cutlass.Float32(ckv_vec[i])

                    qp_vec = smem_q_pe_[(0, None), (T_idx, lane_idx)].load()
                    kpe_vec = kpe_row_[None, lane_idx].load()
                    for i in range(vec_size_kpe):
                        sum_partial += cutlass.Float32(qp_vec[i]) * cutlass.Float32(kpe_vec[i])

                    s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                    if lane_idx == 0:
                        smem_logits[sparse_idx] = s * sm_scale

            cute.arch.sync_threads()

            # ── Softmax: max ─────────────────────────────────────────────────
            partial_max = -cutlass.Float32(math.inf)
            for idx in range(tidx, local_valid, num_threads):
                v = smem_logits[idx]
                if v > partial_max:
                    partial_max = v

            max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
            if lane_idx == 0:
                smem_max_red_f32[warp_idx] = max_val
            cute.arch.sync_threads()
            if warp_idx == 0:
                val = smem_max_red_f32[lane_idx]
                max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
                smem_max_red_f32[0] = max_val
            cute.arch.sync_threads()

            row_max = smem_max_red_f32[0]

            # ── Softmax: exp + sum ───────────────────────────────────────────
            local_sum = cutlass.Float32(0)
            for idx in range(tidx, local_valid, num_threads):
                e = cute.math.exp(smem_logits[idx] - row_max)
                smem_logits[idx] = e
                local_sum += e

            sum_val = warp_reduce(local_sum, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_sum_red_f32[warp_idx] = sum_val
            cute.arch.sync_threads()
            if warp_idx == 0:
                val = smem_sum_red_f32[lane_idx]
                sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
                smem_sum_red_f32[0] = sum_val
            cute.arch.sync_threads()

            row_sum = smem_sum_red_f32[0]

            # ── Output (SSA_v2) ────────────────────────────────────────────────
            out_rmem0 = cute.make_rmem_tensor(cute.make_layout((vec_size_ckv,), stride=(1,)), cutlass.Float32)
            out_rmem1 = cute.make_rmem_tensor(cute.make_layout((vec_size_ckv,), stride=(1,)), cutlass.Float32)
            for i in range(vec_size_ckv):
                out_rmem0[i] = cutlass.Float32(0)
                out_rmem1[i] = cutlass.Float32(0)
            acc0 = out_rmem0.load()
            acc1 = out_rmem1.load()

            for round_idx in range(num_rounds):
                sparse_idx = round_idx * num_warps + warp_idx
                if sparse_idx < local_valid:
                    cur_idx = smem_sparse[T_idx, split_start + sparse_idx]
                    ckv_row_ = ckv_flat_[(0, None), (cur_idx, None)]
                    e = smem_logits[sparse_idx]

                    ckv_vec0 = ckv_row_[None, 0 * wsize + lane_idx].load()
                    acc0 = acc0 + e * ckv_vec0.to(cutlass.Float32)
                    ckv_vec1 = ckv_row_[None, 1 * wsize + lane_idx].load()
                    acc1 = acc1 + e * ckv_vec1.to(cutlass.Float32)

            if warp_idx < local_valid:
                for v in range(vec_size_ckv):
                    smem_partial[warp_idx, (0 * wsize + lane_idx) * vec_size_ckv + v] = acc0[v]
                    smem_partial[warp_idx, (1 * wsize + lane_idx) * vec_size_ckv + v] = acc1[v]

            cute.arch.sync_threads()

            num_active_warps = local_valid if local_valid < num_warps else num_warps

            # Always write to partial_out (no single-split special case)
            for i in range(tidx, head_dim_ckv, num_threads):
                acc = cutlass.Float32(0)
                for w in range(num_active_warps):
                    acc += smem_partial[w, i]
                partial_out[T_idx, head_idx, split_idx_new, i] = acc
            if tidx == 0:
                partial_lse[T_idx, head_idx, split_idx_new, 0] = row_max
                partial_lse[T_idx, head_idx, split_idx_new, 1] = row_sum


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2: Reduce — v3: tensorSSA + vectorized load/store (no dim loop)
#
# Grid: [T, num_heads, 1] × 256 threads → 8 blocks/SM on 16 SMs
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_reduce_kernel(
    sparse_indices: cute.Tensor,        # (T, 2048)
    partial_out:    cute.Tensor,        # (T_MAX, 16, 8, 512)
    partial_lse:    cute.Tensor,        # (T_MAX, 16, 8, 2)
    output:         cute.Tensor,        # (T, 16, 512)
    lse:            cute.Tensor,        # (T, 16)
):
    head_dim_ckv:   cutlass.Constexpr = HEAD_DIM_CKV
    top_k_len:      cutlass.Constexpr = TOP_K_LEN
    dim_split:      cutlass.Constexpr = DIM_SPLIT
    num_splits:     cutlass.Constexpr = NUM_SPLITS
    num_threads:    cutlass.Constexpr = NUM_THREADS_REDUCE
    num_warps:      cutlass.Constexpr = NUM_WARPS_REDUCE
    vec_reduce:     cutlass.Constexpr = VEC_REDUCE

    bidx, bidy, _ = cute.arch.block_idx()  # T_idx, head_idx
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()

    T_idx    = bidx
    head_idx = bidy

    # ── Prologue: count valid for this T_idx (overlaps compute writes) ───────
    alloc = cutlass.utils.SmemAllocator()
    smem_red_i32     = _smem(alloc, cutlass.Int32,    (32,),            (1,),   4)
    smem_max_sum     = _smem(alloc, cutlass.Float32,  (num_splits, 2),  (2, 1), 4)

    partial_cnt = 0
    for i in range(tidx, top_k_len, num_threads):
        idx = sparse_indices[T_idx, i]
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

    num_valid = smem_red_i32[0]

    # ── griddepcontrol_wait: stall until all compute blocks are done ──────────
    cute.arch.griddepcontrol_wait()

    # ── Reduce this (T_idx, head_idx) — always reduce (no single-split skip) ─
    num_active_splits = (num_valid + dim_split - 1) // dim_split
    if num_active_splits < 1:
        num_active_splits = 1

    if tidx < num_active_splits:
        smem_max_sum[tidx, 0] = partial_lse[T_idx, head_idx, tidx, 0]
        smem_max_sum[tidx, 1] = partial_lse[T_idx, head_idx, tidx, 1]

    cute.arch.sync_threads()

    # zipped_divide views for vectorized access
    partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, vec_reduce))
    output_v      = cute.zipped_divide(output, (1, 1, vec_reduce))

    g_max = -cutlass.Float32(math.inf)
    for s in range(num_active_splits):
        local_max = smem_max_sum[s, 0]
        if local_max > g_max:
            g_max = local_max

    # Fused vectorized reduction — tensorSSA arithmetic
    g_lse_sum = cutlass.Float32(0)
    acc_rmem = cute.make_rmem_tensor(cute.make_layout((vec_reduce,), stride=(1,)), cutlass.Float32)
    acc_rmem[0] = cutlass.Float32(0)
    acc_rmem[1] = cutlass.Float32(0)
    acc = acc_rmem.load()

    for s in range(num_active_splits):
        l_max = smem_max_sum[s, 0]
        l_sum = smem_max_sum[s, 1]
        scale = cute.math.exp(l_max - g_max)
        g_lse_sum += l_sum * scale

        a = partial_out_v[(0, 0, 0, None), (T_idx, head_idx, s, tidx)].load()
        acc = acc + scale * a

    if tidx == 0:
        lse[T_idx, head_idx] = (g_max + cute.math.log(g_lse_sum)) / cutlass.Float32(LN2)

    output_v[(0, 0, None), (T_idx, head_idx, tidx)].store((acc / g_lse_sum).to(cutlass.BFloat16))


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_kvsplit_xor_pdl():
    T = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe = NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE
    num_pages, page_size = NUM_PAGES, PAGE_SIZE
    num_splits = NUM_SPLITS

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_ckv), (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_kpe), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K_LEN),               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    partial_out    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, head_dim_ckv), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),               (1, 0),     4)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kvsplit_xor_pdl,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, stream,
        options="--enable-tvm-ffi"
    )


_compiled = compile_kvsplit_xor_pdl()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    partial_out = torch.empty(T_MAX, NUM_HEADS, NUM_SPLITS, HEAD_DIM_CKV, dtype=torch.float32, device=output.device)
    partial_lse = torch.empty(T_MAX, NUM_HEADS, NUM_SPLITS, 2,            dtype=torch.float32, device=output.device)
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, partial_out, partial_lse, output, lse)

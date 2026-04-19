"""
kv_split_xor_pdl_v3_pro_v2_1024T.py — v2 + arbitrary T support (up to 1024).

Based on v3_pro_v2 but adds:
  - Outer group loop (groups of T_MAX=8) in compute kernel for arbitrary T
  - Outer group loop in reduce kernel: grid is [T_MAX, NUM_HEADS, 1] fixed
  - NUM_REQ_CONCURR=1024 for partial buffers (static allocation)
  - Assertion in run() to guard T < NUM_REQ_CONCURR

Ref: https://github.com/wangsiping97/FastGEMV/blob/main/method_and_result.md
"""
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, TOP_K_LEN = 16, 512, 64, 2048
NUM_PAGES, PAGE_SIZE = 8462, 64
T_MAX = 8
NUM_REQ_CONCURR = 1024  # max concurrent requests; partial buffers are pre-allocated to this size
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

# cp.async + early-exit constants
VEC_SPARSE = 4                          # 4 × i32 = 128-bit LDG for sparse_indices
VEC_Q = 8                               # 8 × BF16 = 128-bit cp.async for q
TOP_K_CHUNKS = TOP_K_LEN // VEC_SPARSE  # 512
Q_NOPE_CHUNKS = HEAD_DIM_CKV // VEC_Q   # 64
Q_PE_CHUNKS = HEAD_DIM_KPE // VEC_Q     # 8

SENTINEL_SKIP = float("inf")

# FastGEMV score constants
ROWS_PER_WARP = 4                               # rows batched per warp per round
ROWS_PER_ROUND_SCORE = NUM_WARPS * ROWS_PER_WARP  # 128


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Host JIT: launch compute (128 SMs) then reduce (128 SMs) with PDL
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

    # Reduce: fixed T_MAX×16 blocks — each block loops over groups of T_MAX
    kvsplit_reduce_kernel(
        sparse_indices, partial_out, partial_lse, output, lse,
    ).launch(grid=[T_MAX, NUM_HEADS, 1], block=[NUM_THREADS_REDUCE, 1, 1],
             stream=stream, use_pdl=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1: Compute — XOR-persistent over T, cp.async + early-exit prologue
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_compute_kernel(
    q_nope:         cute.Tensor,        # (T,16,512)
    q_pe:           cute.Tensor,        # (T,16, 64)
    ckv_flat:       cute.Tensor,        # (N, 512)
    kpe_flat:       cute.Tensor,        # (N,  64)
    sparse_indices: cute.Tensor,        # (T, 2048)
    sm_scale:       cutlass.Constexpr,
    partial_out:    cute.Tensor,        # (NUM_REQ_CONCURR, 16, 8, 512)
    partial_lse:    cute.Tensor,        # (NUM_REQ_CONCURR, 16, 8, 2)
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
    vec_sparse:     cutlass.Constexpr = VEC_SPARSE
    vec_q:          cutlass.Constexpr = VEC_Q
    top_k_chunks:   cutlass.Constexpr = TOP_K_CHUNKS
    q_nope_chunks:  cutlass.Constexpr = Q_NOPE_CHUNKS
    q_pe_chunks:    cutlass.Constexpr = Q_PE_CHUNKS
    rows_per_warp:  cutlass.Constexpr = ROWS_PER_WARP
    rows_per_round_score: cutlass.Constexpr = ROWS_PER_ROUND_SCORE

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
    smem_out         = _smem(alloc, cutlass.Float32,  (head_dim_ckv,),              (1,),              16)

    # ── Thread-group indices ─────────────────────────────────────────────────
    wg_per_T_idx   = tidx // sparse_thr_per_T
    thr_idx_per_T  = tidx % sparse_thr_per_T
    lane_idx_per_T = thr_idx_per_T % wsize
    warp_per_T_idx = warp_idx % num_warps_per_T

    # ── cp.async copy atom: 8 × BF16 = 128 bits per transfer ────────────────
    copy_atom_q = cute.make_copy_atom(
        cpasync.CopyG2SOp(),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )

    # ── Vec views for cp.async ───────────────────────────────────────────────
    q_nope_vec      = cute.zipped_divide(q_nope,      (1, 1, vec_q))
    smem_q_nope_vec = cute.zipped_divide(smem_q_nope, (1, vec_q))
    q_pe_vec        = cute.zipped_divide(q_pe,        (1, 1, vec_q))
    smem_q_pe_vec   = cute.zipped_divide(smem_q_pe,   (1, vec_q))

    # ── Vec view for sparse_load (early-exit pattern) ────────────────────────
    si_vec = cute.zipped_divide(sparse_indices, (1, vec_sparse))

    # ── Vectorized views (independent of T, computed once) ───────────────────
    smem_q_nope_ = cute.zipped_divide(smem_q_nope, (1, vec_size_ckv))
    ckv_flat_    = cute.zipped_divide(ckv_flat,     (1, vec_size_ckv))
    kpe_flat_    = cute.zipped_divide(kpe_flat,     (1, vec_size_kpe))
    smem_q_pe_   = cute.zipped_divide(smem_q_pe,   (1, vec_size_kpe))

    # ── Outer group loop: process up to T_MAX tokens per group ───────────────
    num_groups = (T + t_max - 1) // t_max
    for group_idx in range(num_groups):
        t_group_start = group_idx * t_max
        group_size = T - t_group_start
        if group_size > t_max:
            group_size = t_max
        T_global_wg = t_group_start + wg_per_T_idx

        # ══════════════════════════════════════════════════════════════════
        # Phase 1: cp.async fire q_nope + q_pe (non-blocking gmem → smem)
        # ══════════════════════════════════════════════════════════════════
        if T_global_wg < T:
            for chunk in range(thr_idx_per_T, q_nope_chunks, sparse_thr_per_T):
                cute.copy(
                    copy_atom_q,
                    q_nope_vec[(0, 0, None), (T_global_wg, head_idx, chunk)],
                    smem_q_nope_vec[(0, None), (wg_per_T_idx, chunk)],
                )
            for chunk in range(thr_idx_per_T, q_pe_chunks, sparse_thr_per_T):
                cute.copy(
                    copy_atom_q,
                    q_pe_vec[(0, 0, None), (T_global_wg, head_idx, chunk)],
                    smem_q_pe_vec[(0, None), (wg_per_T_idx, chunk)],
                )

        cute.arch.cp_async_commit_group()

        # ══════════════════════════════════════════════════════════════════
        # Phase 2: sparse_load — vec4 LDG.128 + early-exit + valid count
        # ══════════════════════════════════════════════════════════════════
        partial_cnt = 0
        if T_global_wg < T:
            chunk = cutlass.Int32(thr_idx_per_T)
            while chunk < cutlass.Int32(top_k_chunks):
                vec = si_vec[(0, None), (T_global_wg, chunk)].load()
                v0 = vec[0]
                for v in range(vec_sparse):
                    smem_sparse[wg_per_T_idx, chunk * vec_sparse + v] = vec[v]
                    if vec[v] >= cutlass.Int32(0):
                        partial_cnt += 1
                if v0 < cutlass.Int32(0):
                    chunk = cutlass.Int32(top_k_chunks)   # exit while
                else:
                    chunk = chunk + cutlass.Int32(sparse_thr_per_T)

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

        # ══════════════════════════════════════════════════════════════════
        # Phase 3: cp_async_wait — stall until q lands in smem, then sync
        # ══════════════════════════════════════════════════════════════════
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # ── Clamp negative sparse indices → 0 for safe OOB-free FastGEMV ─
        for t_fix in range(t_max):
            if t_fix < group_size:
                for i in range(tidx, top_k_len, num_threads):
                    if smem_sparse[t_fix, i] < cutlass.Int32(0):
                        smem_sparse[t_fix, i] = cutlass.Int32(0)
        cute.arch.sync_threads()

        # ── PDL: signal dependents after group 0 prologue ────────────────
        if group_idx == 0:
            cute.arch.griddepcontrol_launch_dependents()

        # ── Inner T-loop with XOR swizzle (within current group) ─────────
        for T_local in range(group_size):
            T_global = t_group_start + T_local
            split_idx_new = T_local ^ split_idx_old

            num_valid_T = smem_num_valid[T_local]
            split_start = split_idx_new * dim_split
            is_OOB = split_start >= num_valid_T

            if not is_OOB:
                local_valid = min(num_valid_T - split_start, dim_split)
                num_rounds = (local_valid + num_warps - 1) // num_warps

                # ── Score (FastGEMV: 4-row interleaved per warp) ─────────
                num_rounds_score = (local_valid + rows_per_round_score - 1) // rows_per_round_score

                for round_idx in range(num_rounds_score):
                    base_sparse = round_idx * rows_per_round_score + warp_idx * rows_per_warp

                    cur_idx0 = smem_sparse[T_local, split_start + base_sparse + 0]
                    cur_idx1 = smem_sparse[T_local, split_start + base_sparse + 1]
                    cur_idx2 = smem_sparse[T_local, split_start + base_sparse + 2]
                    cur_idx3 = smem_sparse[T_local, split_start + base_sparse + 3]

                    ckv_row0 = ckv_flat_[(0, None), (cur_idx0, None)]
                    ckv_row1 = ckv_flat_[(0, None), (cur_idx1, None)]
                    ckv_row2 = ckv_flat_[(0, None), (cur_idx2, None)]
                    ckv_row3 = ckv_flat_[(0, None), (cur_idx3, None)]

                    kpe_row0 = kpe_flat_[(0, None), (cur_idx0, None)]
                    kpe_row1 = kpe_flat_[(0, None), (cur_idx1, None)]
                    kpe_row2 = kpe_flat_[(0, None), (cur_idx2, None)]
                    kpe_row3 = kpe_flat_[(0, None), (cur_idx3, None)]

                    sums = cute.make_rmem_tensor(
                        cute.make_layout((rows_per_warp,), stride=(1,)),
                        cutlass.Float32,
                    )
                    for r in range(rows_per_warp):
                        sums[r] = cutlass.Float32(0)

                    # CKV dot products — interleaved 4-row loads, shared q_frag
                    for it in range(iters_per_lane_ckv):
                        rest_idx = it * wsize + lane_idx
                        qn_frag = smem_q_nope_[(0, None), (T_local, rest_idx)].load()

                        ckv_f0 = ckv_row0[None, rest_idx].load()
                        ckv_f1 = ckv_row1[None, rest_idx].load()
                        ckv_f2 = ckv_row2[None, rest_idx].load()
                        ckv_f3 = ckv_row3[None, rest_idx].load()

                        for v in range(vec_size_ckv):
                            qv = cutlass.Float32(qn_frag[v])
                            sums[0] = sums[0] + qv * cutlass.Float32(ckv_f0[v])
                            sums[1] = sums[1] + qv * cutlass.Float32(ckv_f1[v])
                            sums[2] = sums[2] + qv * cutlass.Float32(ckv_f2[v])
                            sums[3] = sums[3] + qv * cutlass.Float32(ckv_f3[v])

                    # KPE dot products — interleaved
                    qp_frag = smem_q_pe_[(0, None), (T_local, lane_idx)].load()
                    kpe_f0 = kpe_row0[None, lane_idx].load()
                    kpe_f1 = kpe_row1[None, lane_idx].load()
                    kpe_f2 = kpe_row2[None, lane_idx].load()
                    kpe_f3 = kpe_row3[None, lane_idx].load()
                    for v in range(vec_size_kpe):
                        qv = cutlass.Float32(qp_frag[v])
                        sums[0] = sums[0] + qv * cutlass.Float32(kpe_f0[v])
                        sums[1] = sums[1] + qv * cutlass.Float32(kpe_f1[v])
                        sums[2] = sums[2] + qv * cutlass.Float32(kpe_f2[v])
                        sums[3] = sums[3] + qv * cutlass.Float32(kpe_f3[v])

                    # Batched warp reduction
                    for r in range(rows_per_warp):
                        sums[r] = warp_reduce(sums[r], lambda a, b: a + b, width=32)
                    if lane_idx == 0:
                        for r in range(rows_per_warp):
                            smem_logits[base_sparse + r] = sums[r] * sm_scale

                cute.arch.sync_threads()

                # ── Softmax: max ─────────────────────────────────────────
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

                # ── Softmax: exp + sum ───────────────────────────────────
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

                # ── Output ───────────────────────────────────────────────
                out_regs = cute.make_rmem_tensor(cute.make_layout((vec_size_out,), stride=(1,)), cutlass.Float32)
                for i in range(vec_size_out):
                    out_regs[i] = cutlass.Float32(0)

                for round_idx in range(num_rounds):
                    sparse_idx = round_idx * num_warps + warp_idx
                    if sparse_idx < local_valid:
                        cur_idx = smem_sparse[T_local, split_start + sparse_idx]
                        ckv_row_ = ckv_flat_[(0, None), (cur_idx, None)]
                        e = smem_logits[sparse_idx]

                        for it in range(iters_per_lane_ckv):
                            rest_idx = it * wsize + lane_idx
                            ckv_vec = ckv_row_[None, rest_idx].load()
                            for i in range(vec_size_ckv):
                                out_regs[it * vec_size_ckv + i] += e * cutlass.Float32(ckv_vec[i])

                if warp_idx < local_valid:
                    for it in range(iters_per_lane_ckv):
                        for v in range(vec_size_ckv):
                            smem_partial[warp_idx, (it * wsize + lane_idx) * vec_size_ckv + v] = out_regs[it * vec_size_ckv + v]

                cute.arch.sync_threads()

                num_active_warps = local_valid if local_valid < num_warps else num_warps
                for i in range(tidx, head_dim_ckv, num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_active_warps):
                        acc += smem_partial[w, i]
                    smem_out[i] = acc
                cute.arch.sync_threads()

                is_single_split_request = num_valid_T < dim_split

                if is_single_split_request and split_idx_new == 0:
                    # Single split: normalize and write directly to output
                    for i in range(tidx, head_dim_ckv, num_threads):
                        output[T_global, head_idx, i] = cutlass.BFloat16(smem_out[i] / row_sum)
                    if tidx == 0:
                        partial_lse[T_global, head_idx, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                        lse[T_global, head_idx] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
                else:
                    # Multi-split: write to partial buffers for reduce kernel
                    for i in range(tidx, head_dim_ckv, num_threads):
                        partial_out[T_global, head_idx, split_idx_new, i] = smem_out[i]
                    if tidx == 0:
                        partial_lse[T_global, head_idx, split_idx_new, 0] = row_max
                        partial_lse[T_global, head_idx, split_idx_new, 1] = row_sum




# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2: Reduce — v3: tensorSSA + vectorized load/store (no dim loop)
#
# Grid: [T_MAX, num_heads, 1] × 256 threads — each block loops over T groups
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_reduce_kernel(
    sparse_indices: cute.Tensor,        # (T, 2048)
    partial_out:    cute.Tensor,        # (NUM_REQ_CONCURR, 16, 8, 512)
    partial_lse:    cute.Tensor,        # (NUM_REQ_CONCURR, 16, 8, 2)
    output:         cute.Tensor,        # (T, 16, 512)
    lse:            cute.Tensor,        # (T, 16)
):
    T, _ = sparse_indices.shape
    head_dim_ckv:   cutlass.Constexpr = HEAD_DIM_CKV
    top_k_len:      cutlass.Constexpr = TOP_K_LEN
    dim_split:      cutlass.Constexpr = DIM_SPLIT
    num_splits:     cutlass.Constexpr = NUM_SPLITS
    num_threads:    cutlass.Constexpr = NUM_THREADS_REDUCE
    num_warps:      cutlass.Constexpr = NUM_WARPS_REDUCE
    vec_reduce:     cutlass.Constexpr = VEC_REDUCE
    t_max:          cutlass.Constexpr = T_MAX

    bidx, bidy, _ = cute.arch.block_idx()  # block_t_idx, head_idx
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()

    block_t_idx = bidx   # 0..T_MAX-1
    head_idx    = bidy

    alloc = cutlass.utils.SmemAllocator()
    smem_red_i32     = _smem(alloc, cutlass.Int32,    (32,),            (1,),   4)
    smem_max_sum     = _smem(alloc, cutlass.Float32,  (num_splits, 2),  (2, 1), 4)

    # zipped_divide views for vectorized access (computed once)
    partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, vec_reduce))
    output_v      = cute.zipped_divide(output, (1, 1, vec_reduce))

    # ── Outer loop over groups of T_MAX ──────────────────────────────────────
    num_groups = (T + t_max - 1) // t_max
    for group_idx in range(num_groups):
        T_idx = group_idx * t_max + block_t_idx
        if T_idx < T:
            # ── Count valid for this T_idx ───────────────────────────────
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

            # ── PDL: wait for compute after valid counting ───────────────
            cute.arch.griddepcontrol_wait()

            # ── Reduce this (T_idx, head_idx) ────────────────────────────
            is_single_split = num_valid < dim_split

            if not is_single_split:
                num_active_splits = (num_valid + dim_split - 1) // dim_split

                if tidx < num_active_splits:
                    smem_max_sum[tidx, 0] = partial_lse[T_idx, head_idx, tidx, 0]
                    smem_max_sum[tidx, 1] = partial_lse[T_idx, head_idx, tidx, 1]

                cute.arch.sync_threads()

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

            cute.arch.sync_threads()


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
    num_req_concurr = NUM_REQ_CONCURR

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv), (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe), (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_ckv), (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_kpe), (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K_LEN),               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    partial_out    = _fake(cute.Float32,  (num_req_concurr, num_heads, num_splits, head_dim_ckv), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (num_req_concurr, num_heads, num_splits, 2),            (3, 2, 1, 0), 16)
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
    T = q_nope.shape[0]
    assert T < NUM_REQ_CONCURR, f"T={T} exceeds NUM_REQ_CONCURR={NUM_REQ_CONCURR}"
    partial_out = torch.empty(NUM_REQ_CONCURR, NUM_HEADS, NUM_SPLITS, HEAD_DIM_CKV, dtype=torch.float32, device=output.device)
    partial_lse = torch.empty(NUM_REQ_CONCURR, NUM_HEADS, NUM_SPLITS, 2,            dtype=torch.float32, device=output.device)
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, partial_out, partial_lse, output, lse)

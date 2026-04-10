"""
kv_split_148_v3_pro.py — 148-block persistent compute (one block per SM).

Design:
  - Launch exactly 148 blocks (= B200 SM count), 1024 threads each.
  - Prologue (fixed ~2µs): load ALL T tokens' sparse_indices into smem,
    count valid per token.  Same cost for all SMs.
  - Main loop: round-robin over T × 16 × 8 tiles:
      for tile_id in range(bidx, T * 16 * 8, 148):
          T_idx     = tile_id // (16 * 8)
          head_idx  = (tile_id // 8) % 16
          split_idx = tile_id % 8
          ... vanilla q load, score, softmax, output ...
  - No PDL, no XOR swizzle.  Separate reduce kernel.
  - q_nope / q_pe reloaded per tile (vanilla gmem → smem).
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

NUM_BLOCKS = 148  # one per B200 SM

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

VEC_REDUCE = 2

# Early-exit sparse_load constants
VEC_SPARSE = 4
TOP_K_CHUNKS = TOP_K_LEN // VEC_SPARSE  # 512

SENTINEL_SKIP = float("inf")


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Host JIT: launch compute (148 SMs) then reduce
# ═══════════════════════════════════════════════════════════════════════════════

@cute.jit
def kvsplit_148(
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

    # Compute: 148 blocks = one per SM
    kvsplit_compute_kernel(
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
    ).launch(grid=[NUM_BLOCKS, 1, 1], block=[NUM_THREADS, 1, 1],
             stream=stream)

    # Reduce: T×16 blocks with 256 threads
    kvsplit_reduce_kernel(
        sparse_indices, partial_out, partial_lse, output, lse,
    ).launch(grid=[T, NUM_HEADS, 1], block=[NUM_THREADS_REDUCE, 1, 1],
             stream=stream)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1: Compute — 148-block persistent, round-robin over tiles
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
    num_blocks:     cutlass.Constexpr = NUM_BLOCKS
    vec_size_ckv:   cutlass.Constexpr = VEC_SIZE_CKV
    vec_size_kpe:   cutlass.Constexpr = VEC_SIZE_KPE
    vec_size_out:   cutlass.Constexpr = VEC_SIZE_OUT
    iters_per_lane_ckv: cutlass.Constexpr = ITERS_PER_LANE_CKV
    sparse_thr_per_T:   cutlass.Constexpr = SPARSE_THR_PER_T
    num_warps_per_T:    cutlass.Constexpr = NUM_WARPS_PER_T
    t_max:          cutlass.Constexpr = T_MAX
    vec_sparse:     cutlass.Constexpr = VEC_SPARSE
    top_k_chunks:   cutlass.Constexpr = TOP_K_CHUNKS

    bidx, _, _ = cute.arch.block_idx()  # 0..147
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    wsize = cute.arch.WARP_SIZE

    # ── SMEM allocation ──────────────────────────────────────────────────────
    alloc = cutlass.utils.SmemAllocator()
    smem_sparse      = _smem(alloc, cutlass.Int32,    (t_max, top_k_len),           (top_k_len, 1),     4)  # 64 KB
    smem_num_valid   = _smem(alloc, cutlass.Int32,    (t_max,),                     (1,),               4)
    smem_logits      = _smem(alloc, cutlass.Float32,  (dim_split,),                 (1,),              16)
    smem_red_i32     = _smem(alloc, cutlass.Int32,    (t_max, 32),                  (32, 1),            4)
    smem_max_red_f32 = _smem(alloc, cutlass.Float32,  (32,),                        (1,),              16)
    smem_sum_red_f32 = _smem(alloc, cutlass.Float32,  (32,),                        (1,),              16)
    # q buffers: single-token (reloaded per tile)
    smem_q_nope      = _smem(alloc, cutlass.BFloat16, (head_dim_ckv,),              (1,),              16)
    smem_q_pe        = _smem(alloc, cutlass.BFloat16, (head_dim_kpe,),              (1,),              16)
    smem_partial     = _smem(alloc, cutlass.Float32,  (num_warps, head_dim_ckv),    (head_dim_ckv, 1), 16)
    smem_out         = _smem(alloc, cutlass.Float32,  (head_dim_ckv,),              (1,),              16)

    # ── Thread-group indices for prologue sparse_load ────────────────────────
    wg_per_T_idx   = tidx // sparse_thr_per_T
    thr_idx_per_T  = tidx % sparse_thr_per_T
    lane_idx_per_T = thr_idx_per_T % wsize
    warp_per_T_idx = warp_idx % num_warps_per_T

    # ── Vec view for sparse_load (early-exit pattern) ────────────────────────
    si_vec = cute.zipped_divide(sparse_indices, (1, vec_sparse))

    # ══════════════════════════════════════════════════════════════════════════
    # PROLOGUE: load ALL T tokens' sparse_indices + count valid per token
    # ══════════════════════════════════════════════════════════════════════════
    partial_cnt = 0
    if wg_per_T_idx < T:
        chunk = cutlass.Int32(thr_idx_per_T)
        while chunk < cutlass.Int32(top_k_chunks):
            vec = si_vec[(0, None), (wg_per_T_idx, chunk)].load()
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

    cute.arch.sync_threads()

    # ── Vectorized views for score GEMV ──────────────────────────────────────
    smem_q_nope_ = cute.zipped_divide(smem_q_nope, (vec_size_ckv,))
    ckv_flat_    = cute.zipped_divide(ckv_flat,     (1, vec_size_ckv))
    kpe_flat_    = cute.zipped_divide(kpe_flat,     (1, vec_size_kpe))
    smem_q_pe_   = cute.zipped_divide(smem_q_pe,   (vec_size_kpe,))

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN LOOP: round-robin over T × 16 × 8 tiles
    # ══════════════════════════════════════════════════════════════════════════
    total_tiles = T * NUM_HEADS * NUM_SPLITS
    tile_id = cutlass.Int32(bidx)

    while tile_id < total_tiles:
        # Decode tile coordinates
        T_idx     = tile_id // (NUM_HEADS * NUM_SPLITS)
        head_idx  = (tile_id // NUM_SPLITS) % NUM_HEADS
        split_idx = tile_id % NUM_SPLITS

        num_valid_T = smem_num_valid[T_idx]
        split_start = split_idx * dim_split
        is_OOB = split_start >= num_valid_T

        if not is_OOB:
            # ── Load q_nope, q_pe for this (T_idx, head_idx) ─────────────
            for i in range(tidx, head_dim_ckv, num_threads):
                smem_q_nope[i] = q_nope[T_idx, head_idx, i]
            for i in range(tidx, head_dim_kpe, num_threads):
                smem_q_pe[i] = q_pe[T_idx, head_idx, i]
            cute.arch.sync_threads()

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
                        qn_vec = smem_q_nope_[(None, (rest_idx,))].load()
                        ckv_vec = ckv_row_[None, rest_idx].load()
                        for i in range(vec_size_ckv):
                            sum_partial += cutlass.Float32(qn_vec[i]) * cutlass.Float32(ckv_vec[i])

                    qp_vec = smem_q_pe_[(None, (lane_idx,))].load()
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

            # ── Output ───────────────────────────────────────────────────────
            out_regs = cute.make_rmem_tensor(cute.make_layout((vec_size_out,), stride=(1,)), cutlass.Float32)
            for i in range(vec_size_out):
                out_regs[i] = cutlass.Float32(0)

            for round_idx in range(num_rounds):
                sparse_idx = round_idx * num_warps + warp_idx
                if sparse_idx < local_valid:
                    cur_idx = smem_sparse[T_idx, split_start + sparse_idx]
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

            if is_single_split_request and split_idx == cutlass.Int32(0):
                # Single split: normalize and write directly to output
                for i in range(tidx, head_dim_ckv, num_threads):
                    output[T_idx, head_idx, i] = cutlass.BFloat16(smem_out[i] / row_sum)
                if tidx == 0:
                    partial_lse[T_idx, head_idx, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                    lse[T_idx, head_idx] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
            else:
                # Multi-split: write to partial buffers for reduce kernel
                for i in range(tidx, head_dim_ckv, num_threads):
                    partial_out[T_idx, head_idx, split_idx, i] = smem_out[i]
                if tidx == 0:
                    partial_lse[T_idx, head_idx, split_idx, 0] = row_max
                    partial_lse[T_idx, head_idx, split_idx, 1] = row_sum

        # Advance to next tile for this block
        tile_id = tile_id + cutlass.Int32(num_blocks)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2: Reduce — same as xor_pdl_v3_pro (without PDL)
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

    alloc = cutlass.utils.SmemAllocator()
    smem_red_i32     = _smem(alloc, cutlass.Int32,    (32,),            (1,),   4)
    smem_max_sum     = _smem(alloc, cutlass.Float32,  (num_splits, 2),  (2, 1), 4)

    # Count valid for this T_idx
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

    is_single_split = num_valid < dim_split

    if not is_single_split:
        num_active_splits = (num_valid + dim_split - 1) // dim_split

        if tidx < num_active_splits:
            smem_max_sum[tidx, 0] = partial_lse[T_idx, head_idx, tidx, 0]
            smem_max_sum[tidx, 1] = partial_lse[T_idx, head_idx, tidx, 1]

        cute.arch.sync_threads()

        partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, vec_reduce))
        output_v      = cute.zipped_divide(output, (1, 1, vec_reduce))

        g_max = -cutlass.Float32(math.inf)
        for s in range(num_active_splits):
            local_max = smem_max_sum[s, 0]
            if local_max > g_max:
                g_max = local_max

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


def compile_kvsplit_148():
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
        kvsplit_148,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, stream,
        options="--enable-tvm-ffi"
    )


_compiled = compile_kvsplit_148()


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    partial_out = torch.empty(T_MAX, NUM_HEADS, NUM_SPLITS, HEAD_DIM_CKV, dtype=torch.float32, device=output.device)
    partial_lse = torch.empty(T_MAX, NUM_HEADS, NUM_SPLITS, 2,            dtype=torch.float32, device=output.device)
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, partial_out, partial_lse, output, lse)

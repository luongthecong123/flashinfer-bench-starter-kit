"""Intra-phase profiling for kv_split_xor_pdl — v3 SSA_v1.

SSA_v1: tensorSSA for score dot-product (`.to()` + `*` + `.reduce()`).
Replaces manual `for i in range(vec_size)` element loops in score phase.

Perfetto layout (pid bands):
  0..N       → compute phases   (pid = sm_id)
               phases: upfront | score | softmax_max | softmax_exp_sum | output | write
  200..N+200 → reduce phases    (pid = sm_id + 200)
               sub-phases: pdl_wait | reduce

Grid (compute): [num_heads, num_splits, 1] = [16, 8, 1] = 128 blocks, persistent over T
Grid (reduce):  [T, num_heads, 1] × 256 threads → 16 SMs (8 blocks/SM)
  Each block reduces one (T_idx, head_idx) pair.
Both launched with use_pdl=True.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm
import math, json, torch


# ── Timer helpers ─────────────────────────────────────────────────────────────

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(MLIR_T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int64(t)

@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(MLIR_T.i32(), [], "mov.u32 $0, %smid;", "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int32(t)


PROBE_HEADER = 1
PROBE_ENTRY  = 4
# Compute: 1 upfront + T_MAX(8) * 5 phases = 41 → round to 48
MAX_ENTRIES_COMPUTE = 48
# Reduce: pdl_wait(1) + reduce(1) = 2 per block (each block handles one T)
MAX_ENTRIES_REDUCE  = 4

PROBE_COLS_COMPUTE = PROBE_HEADER + MAX_ENTRIES_COMPUTE * PROBE_ENTRY   # 193
PROBE_COLS_REDUCE  = PROBE_HEADER + MAX_ENTRIES_REDUCE  * PROBE_ENTRY   # 17

TAGS_COMPUTE = {
    "upfront":         0,
    "score":           2,
    "softmax_max":     4,
    "softmax_exp_sum": 6,
    "output":          8,
    "write":           10,
}
TAG_NAMES_COMPUTE = {v: k for k, v in TAGS_COMPUTE.items()}
PHASE_ORDER_COMPUTE = ["upfront", "score", "softmax_max", "softmax_exp_sum", "output", "write"]

TAGS_REDUCE      = {"pdl_wait": 0, "reduce": 2}
TAG_NAMES_REDUCE = {v: k for k, v in TAGS_REDUCE.items()}
PHASE_ORDER_REDUCE = ["pdl_wait", "reduce"]

SENTINEL_SKIP = float("inf")


def range_start(probe, row, cnt, sm_val, tag_val):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 0] = cutlass.Int64(sm_val)
    probe[row, off + 1] = cutlass.Int64(tag_val)
    probe[row, off + 2] = globaltimer_u64()

def range_stop(probe, row, cnt):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 3] = globaltimer_u64() - probe[row, off + 2]
    return cnt + cutlass.Int32(1)

def range_finalize(probe, row, cnt):
    probe[row, 0] = cutlass.Int64(cnt)


# ── Dump helpers ──────────────────────────────────────────────────────────────

def _probe_events(probe_cpu, num_blocks, tag_names, pid_offset=0):
    events = []
    base = None
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (base is None or s < base):
                base = s
    base = base or 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        if cnt == 0: continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off  = PROBE_HEADER + i * PROBE_ENTRY
            tag  = int(data[off + 1])
            t0   = int(data[off + 2])
            dur  = int(data[off + 3])
            if t0 == 0 and dur == 0: continue
            events.append(dict(
                name=tag_names.get(tag, f"tag_{tag}"), ph="X",
                ts=(t0 - base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id + pid_offset, tid=bid))
    return events, base


def dump_compute(probe: torch.Tensor, num_blocks: int, num_head: int, num_splits: int):
    probe_cpu = probe.cpu().contiguous().tolist()

    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur:
            max_dur, max_bid = total, bid

    data = probe_cpu[max_bid]; cnt = int(data[0])
    head     = max_bid // num_splits
    split_old = max_bid % num_splits
    print(f"\n--- Compute: Slowest block {max_bid} "
          f"(head={head}, split_old={split_old}, total={max_dur/1000:.1f}µs): {cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES_COMPUTE.get(tag, f'tag_{tag}'):>16s}"
              f"  dur={dur:>10} ns  ({dur/1000:.1f} µs)")

    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off  = PROBE_HEADER + i * PROBE_ENTRY
            tag  = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES_COMPUTE.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"\n{'='*65}")
    print(f"{'Phase (all blocks)':>24s} {'Total (ms)':>12s} {'Count':>6s}"
          f" {'Avg (µs)':>10s} {'%':>6s}")
    print(f"{'='*65}")
    grand = sum(tag_totals.values())
    for name in PHASE_ORDER_COMPUTE:
        if name in tag_totals:
            tot = tag_totals[name]; cnt_ = tag_counts[name]
            print(f"{name:>24s} {tot/1e6:>12.3f} {cnt_:>6d}"
                  f" {tot/cnt_/1000:>10.1f} {100*tot/grand:>5.1f}%")
    print(f"{'TOTAL':>24s} {grand/1e6:>12.3f}")

    return _probe_events(probe_cpu, num_blocks, TAG_NAMES_COMPUTE, pid_offset=0)


def dump_reduce(probe: torch.Tensor, num_blocks: int):
    probe_cpu = probe.cpu().contiguous().tolist()

    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur:
            max_dur, max_bid = total, bid

    data = probe_cpu[max_bid]; cnt = int(data[0])
    T_slow = max_bid // NUM_HEADS
    H_slow = max_bid % NUM_HEADS
    print(f"\n--- Reduce: Slowest block {max_bid} (T={T_slow}, head={H_slow}, "
          f"total={max_dur/1000:.1f}µs): {cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES_REDUCE.get(tag, f'tag_{tag}'):>12s}"
              f"  dur={dur:>10} ns  ({dur/1000:.1f} µs)")

    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off  = PROBE_HEADER + i * PROBE_ENTRY
            tag  = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES_REDUCE.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"\n{'='*52}")
    print(f"  pdl_wait: prologue (valid count) + griddepcontrol_wait stall")
    print(f"  reduce:   actual reduction for this (T, head)")
    print(f"{'='*52}")
    for name in PHASE_ORDER_REDUCE:
        if name in tag_totals:
            n = tag_counts[name]
            print(f"  {name:>10}: avg={tag_totals[name]/n/1000:.1f}µs  "
                  f"total={tag_totals[name]/1e6:.3f}ms  n={n}")

    return _probe_events(probe_cpu, num_blocks, TAG_NAMES_REDUCE, pid_offset=200)


def build_combined_trace(compute_events, compute_base,
                         reduce_events, reduce_base) -> str:
    shared_base = min(b for b in [compute_base, reduce_base] if b)
    all_events = []
    for ev in compute_events:
        all_events.append(dict(ev, ts=ev["ts"] + (compute_base - shared_base) / 1000.0))
    for ev in reduce_events:
        all_events.append(dict(ev, ts=ev["ts"] + (reduce_base - shared_base) / 1000.0))
    return json.dumps({"traceEvents": all_events})


# ── Kernel constants ──────────────────────────────────────────────────────────

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


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def count_valid_indices(
    sparse_indices:   cute.Tensor,
    smem_sparse:      cute.Tensor,
    smem_red_i32:     cute.Tensor,
    smem_num_valid:   cute.Tensor,
    T:                cute.Numeric,
    tidx:             cute.Numeric,
    warp_idx:         cute.Numeric,
    top_k_len:        cutlass.Constexpr,
    sparse_thr_per_T: cutlass.Constexpr,
    num_warps_per_T:  cutlass.Constexpr,
) -> None:
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
def kvsplit_xor_pdl_intra(
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
    probe_compute:  cute.Tensor,
    probe_reduce:   cute.Tensor,
    stream):
    T, num_heads, head_dim_ckv = q_nope.shape

    N: cutlass.Constexpr = NUM_PAGES * PAGE_SIZE
    ckv_flat = cute.make_tensor(
        ckv_cache.iterator,
        cute.make_layout((N, head_dim_ckv), stride=(head_dim_ckv, 1)))
    kpe_flat = cute.make_tensor(
        kpe_cache.iterator,
        cute.make_layout((N, q_pe.shape[2]), stride=(q_pe.shape[2], 1)))

    # Compute: 128 blocks = num_heads × num_splits
    kvsplit_compute_kernel(
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, probe_compute,
    ).launch(grid=[NUM_HEADS, NUM_SPLITS, 1], block=[NUM_THREADS, 1, 1],
             stream=stream, use_pdl=True)

    # Reduce: T×16 blocks with 256 threads each → 8 blocks/SM on 16 SMs
    kvsplit_reduce_kernel(
        sparse_indices, partial_out, partial_lse, output, lse, probe_reduce,
    ).launch(grid=[T, NUM_HEADS, 1], block=[NUM_THREADS_REDUCE, 1, 1],
             stream=stream, use_pdl=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1: Compute — XOR-persistent over T, instrumented
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
    probe_compute:  cute.Tensor,        # (128, PROBE_COLS_COMPUTE)
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
    probe_row = bidx * num_splits + bidy
    sm = cutlass.Int64(smid_u32())
    probe_cnt = cutlass.Int32(0)

    # ── SMEM allocation ──────────────────────────────────────────────────────
    alloc = cutlass.utils.SmemAllocator()
    smem_sparse      = _smem(alloc, cutlass.Int32,    (t_max, top_k_len),           (top_k_len, 1),     4)
    smem_num_valid   = _smem(alloc, cutlass.Int32,    (t_max,),                     (1,),               4)
    smem_logits      = _smem(alloc, cutlass.Float32,  (dim_split,),                 (1,),              16)
    smem_red_i32     = _smem(alloc, cutlass.Int32,    (t_max, 32),                  (32, 1),            4)
    smem_max_red_f32 = _smem(alloc, cutlass.Float32,  (32,),                        (1,),              16)
    smem_sum_red_f32 = _smem(alloc, cutlass.Float32,  (32,),                        (1,),              16)
    smem_q_nope      = _smem(alloc, cutlass.BFloat16, (t_max, head_dim_ckv),        (head_dim_ckv, 1), 16)
    smem_q_pe        = _smem(alloc, cutlass.BFloat16, (t_max, head_dim_kpe),        (head_dim_kpe, 1), 16)
    smem_partial     = _smem(alloc, cutlass.Float32,  (num_warps, head_dim_ckv),    (head_dim_ckv, 1), 16)
    smem_out         = _smem(alloc, cutlass.Float32,  (head_dim_ckv,),              (1,),              16)

    # ── Upfront phase: load sparse + Q + valid count ─────────────────────────
    if tidx == 0:
        range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["upfront"])

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

    if tidx == 0:
        probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

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
            if tidx == 0:
                range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["score"])

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
                        prod = qn_vec.to(cutlass.Float32) * ckv_vec.to(cutlass.Float32)
                        sum_partial += prod.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)

                    qp_vec = smem_q_pe_[(0, None), (T_idx, lane_idx)].load()
                    kpe_vec = kpe_row_[None, lane_idx].load()
                    prod_pe = qp_vec.to(cutlass.Float32) * kpe_vec.to(cutlass.Float32)
                    sum_partial += prod_pe.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)

                    s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                    if lane_idx == 0:
                        smem_logits[sparse_idx] = s * sm_scale

            cute.arch.sync_threads()
            if tidx == 0:
                probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

            # ── Softmax: max ─────────────────────────────────────────────────
            if tidx == 0:
                range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["softmax_max"])

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

            if tidx == 0:
                probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

            # ── Softmax: exp + sum ───────────────────────────────────────────
            if tidx == 0:
                range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["softmax_exp_sum"])

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

            if tidx == 0:
                probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

            # ── Output GEMV ──────────────────────────────────────────────────
            if tidx == 0:
                range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["output"])

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

            if tidx == 0:
                probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

            # ── Write ────────────────────────────────────────────────────────
            if tidx == 0:
                range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["write"])

            is_single_split_request = num_valid_T < dim_split

            if is_single_split_request and split_idx_new == 0:
                for i in range(tidx, head_dim_ckv, num_threads):
                    output[T_idx, head_idx, i] = cutlass.BFloat16(smem_out[i] / row_sum)
                if tidx == 0:
                    partial_lse[T_idx, head_idx, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                    lse[T_idx, head_idx] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
            else:
                for i in range(tidx, head_dim_ckv, num_threads):
                    partial_out[T_idx, head_idx, split_idx_new, i] = smem_out[i]
                if tidx == 0:
                    partial_lse[T_idx, head_idx, split_idx_new, 0] = row_max
                    partial_lse[T_idx, head_idx, split_idx_new, 1] = row_sum

            cute.arch.sync_threads()
            if tidx == 0:
                probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

    if tidx == 0:
        range_finalize(probe_compute, probe_row, probe_cnt)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2: Reduce — v3: tensorSSA + vectorized load/store (no dim loop)
# Grid: [T, num_heads, 1] × 256 threads → 8 blocks/SM on 16 SMs
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def kvsplit_reduce_kernel(
    sparse_indices: cute.Tensor,        # (T, 2048)
    partial_out:    cute.Tensor,        # (T_MAX, 16, 8, 512)
    partial_lse:    cute.Tensor,        # (T_MAX, 16, 8, 2)
    output:         cute.Tensor,        # (T, 16, 512)
    lse:            cute.Tensor,        # (T, 16)
    probe_reduce:   cute.Tensor,        # (T_MAX*16, PROBE_COLS_REDUCE)
):
    head_dim_ckv:   cutlass.Constexpr = HEAD_DIM_CKV
    top_k_len:      cutlass.Constexpr = TOP_K_LEN
    dim_split:      cutlass.Constexpr = DIM_SPLIT
    num_splits:     cutlass.Constexpr = NUM_SPLITS
    num_threads:    cutlass.Constexpr = NUM_THREADS_REDUCE
    num_warps:      cutlass.Constexpr = NUM_WARPS_REDUCE
    num_heads:      cutlass.Constexpr = NUM_HEADS
    vec_reduce:     cutlass.Constexpr = VEC_REDUCE

    bidx, bidy, _ = cute.arch.block_idx()  # T_idx, head_idx
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()

    T_idx    = bidx
    head_idx = bidy
    probe_row = T_idx * num_heads + head_idx
    sm = cutlass.Int64(smid_u32())
    probe_cnt = cutlass.Int32(0)

    # ── Prologue: count valid for this T_idx (overlaps compute writes) ───────
    if tidx == 0:
        range_start(probe_reduce, probe_row, probe_cnt, sm, TAGS_REDUCE["pdl_wait"])

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

    if tidx == 0:
        probe_cnt = range_stop(probe_reduce, probe_row, probe_cnt)

    # ── Reduce this (T_idx, head_idx) — vectorized tensorSSA ─────────────────
    is_single_split = num_valid < dim_split

    if not is_single_split:
        if tidx == 0:
            range_start(probe_reduce, probe_row, probe_cnt, sm, TAGS_REDUCE["reduce"])

        num_active_splits = (num_valid + dim_split - 1) // dim_split

        if tidx < num_active_splits:
            smem_max_sum[tidx, 0] = partial_lse[T_idx, head_idx, tidx, 0]
            smem_max_sum[tidx, 1] = partial_lse[T_idx, head_idx, tidx, 1]

        cute.arch.sync_threads()

        # zipped_divide views for vectorized access
        partial_out_v = cute.zipped_divide(partial_out, (1, 1, 1, vec_reduce))
        output_v      = cute.zipped_divide(output, (1, 1, vec_reduce))

        # Every thread computes g_max redundantly (≤8 iterations, trivial)
        g_max = -cutlass.Float32(math.inf)
        for s in range(num_active_splits):
            local_max = smem_max_sum[s, 0]
            if local_max > g_max:
                g_max = local_max

        # Fused vectorized reduction — tensorSSA arithmetic
        # 256 threads × vec_size=2 = 512 dims exactly
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

        if tidx == 0:
            probe_cnt = range_stop(probe_reduce, probe_row, probe_cnt)

    if tidx == 0:
        range_finalize(probe_reduce, probe_row, probe_cnt)


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def compile_kernel():
    T  = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe = NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE
    num_pages, page_size = NUM_PAGES, PAGE_SIZE
    num_splits = NUM_SPLITS
    Bc = num_heads * num_splits   # 128

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv),                  (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe),                  (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_ckv),          (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_kpe),          (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, TOP_K_LEN),                                (1, 0),     4)
    sm_scale       = 0.1352337788608801
    partial_out    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, head_dim_ckv),  (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, 2),             (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv),                  (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),                                (1, 0),     4)
    probe_compute  = _fake(cute.Int64,    (Bc, PROBE_COLS_COMPUTE),                      (1, 0),     8)
    probe_reduce   = _fake(cute.Int64,    (T_MAX * num_heads, PROBE_COLS_REDUCE),        (1, 0),     8)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kvsplit_xor_pdl_intra,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
        probe_compute, probe_reduce, stream,
        options="--enable-tvm-ffi"
    )


_compiled = compile_kernel()


def run_single(workload_idx: int) -> str:
    import os, json as _json
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors

    H, D_ckv = NUM_HEADS, HEAD_DIM_CKV
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling profiled kv_split_xor_pdl kernel...")

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [_json.loads(l) for l in open(JSONL)]
    w   = workloads[workload_idx]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]
    _uuid, _T, max_valid = WORKLOAD_INFO[workload_idx]

    num_compute_blocks = NUM_HEADS * NUM_SPLITS   # 128 (fixed)
    num_reduce_blocks  = T * H                    # T×16 (one per (T_idx, head))
    print(f"\nWorkload {workload_idx + 1}: MaxValid={max_valid}  T={T}  "
          f"ComputeBlocks={num_compute_blocks}  ReduceBlocks={num_reduce_blocks}")

    q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)
    sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

    output_t    = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse_t       = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
    partial_out = torch.empty(T_MAX, H, NUM_SPLITS, D_ckv, dtype=torch.float32, device="cuda")
    partial_lse = torch.empty(T_MAX, H, NUM_SPLITS, 2, dtype=torch.float32, device="cuda")
    probe_compute = torch.zeros((num_compute_blocks, PROBE_COLS_COMPUTE), dtype=torch.int64, device="cuda")
    max_reduce_blocks = T_MAX * H
    probe_reduce  = torch.zeros((max_reduce_blocks,  PROBE_COLS_REDUCE),  dtype=torch.int64, device="cuda")

    # Warmup
    for _ in range(3):
        output_t.zero_(); lse_t.fill_(-float("inf"))
        probe_compute.zero_(); probe_reduce.zero_()
        _compiled(q_nope, q_pe, ckv, kpe, si, partial_out, partial_lse,
                  output_t, lse_t, probe_compute, probe_reduce)
        torch.cuda.synchronize()

    # Profile run
    probe_compute.zero_(); probe_reduce.zero_()
    output_t.zero_(); lse_t.fill_(-float("inf"))
    _compiled(q_nope, q_pe, ckv, kpe, si, partial_out, partial_lse,
              output_t, lse_t, probe_compute, probe_reduce)
    torch.cuda.synchronize()

    compute_events, compute_base = dump_compute(
        probe_compute, num_compute_blocks, NUM_HEADS, NUM_SPLITS)
    reduce_events,  reduce_base  = dump_reduce(probe_reduce, num_reduce_blocks)
    return build_combined_trace(compute_events, compute_base, reduce_events, reduce_base)

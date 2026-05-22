"""Intra-kernel profiling for kv_split_v3_thr_warpv3.

Compute phases (per split-block, bidz=split index):
  load / valid_count / score / softmax_max / softmax_exp_sum / output / write
  OOB splits (local_valid==0) terminate after valid_count.

Reduce phases (per token-head block):
  reduce

Goals:
  - Load imbalance: active vs OOB splits, per-token valid_count variation
  - Kernel launch overhead: gap between compute and reduce on globaltimer axis
  - Reduction tax: cost of cross-split merge kernel

Output is a combined Perfetto-compatible trace JSON.
  compute events: pid = sm_id
  reduce  events: pid = sm_id + 200   (visually separated in Perfetto)
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
MAX_ENTRIES  = 14
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY  # 57

# Compute kernel phase tags
TAGS_COMPUTE = {"load": 0, "valid_count": 2, "score": 4,
                "softmax_max": 6, "softmax_exp_sum": 8, "output": 10, "write": 12}
TAG_NAMES_COMPUTE = {v: k for k, v in TAGS_COMPUTE.items()}
PHASE_ORDER_COMPUTE = ["load", "valid_count", "score", "softmax_max",
                       "softmax_exp_sum", "output", "write"]

# Reduce kernel phase tags
TAGS_REDUCE      = {"reduce": 0}
TAG_NAMES_REDUCE = {v: k for k, v in TAGS_REDUCE.items()}
PHASE_ORDER_REDUCE = ["reduce"]

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
    """Collect raw events; return (events_list, global_base_ns)."""
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


def dump_compute(probe: torch.Tensor, num_blocks: int, T: int,
                 num_heads: int, num_splits: int):
    """Print compute phase breakdown; return (events_list, global_base_ns)."""
    probe_cpu = probe.cpu().contiguous().tolist()

    active, oob = [], []
    for bid in range(num_blocks):
        cnt = int(probe_cpu[bid][0])
        token = bid // (num_heads * num_splits)
        split = bid % num_splits
        (active if cnt >= 5 else oob).append((bid, token, split, cnt))

    # Find slowest active block
    max_dur, max_bid = -1, (active[0][0] if active else 0)
    for bid, *_ in active:
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur:
            max_dur, max_bid = total, bid

    data = probe_cpu[max_bid]; cnt = int(data[0])
    token = max_bid // (num_heads * num_splits)
    split = max_bid % num_splits
    print(f"\n--- Compute: Slowest active block {max_bid} "
          f"(token={token}, split={split}, total={max_dur/1000:.1f}µs): {cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES_COMPUTE.get(tag, f'tag_{tag}'):>15s}"
              f"  dur={dur:>10} ns  ({dur/1000:.1f} µs)")

    print(f"\n  Active blocks: {len(active)}, OOB blocks: {len(oob)}")

    tag_totals: dict = {}; tag_counts: dict = {}
    for bid, *_ in active:
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off  = PROBE_HEADER + i * PROBE_ENTRY
            tag  = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES_COMPUTE.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"\n{'='*62}")
    print(f"{'Phase (active blks only)':>24s} {'Total (ms)':>12s} {'Count':>6s}"
          f" {'Avg (µs)':>10s} {'%':>6s}")
    print(f"{'='*62}")
    grand = sum(tag_totals.values())
    for name in PHASE_ORDER_COMPUTE:
        if name in tag_totals:
            tot = tag_totals[name]; cnt_ = tag_counts[name]
            print(f"{name:>24s} {tot/1e6:>12.3f} {cnt_:>6d}"
                  f" {tot/cnt_/1000:>10.1f} {100*tot/grand:>5.1f}%")
    print(f"{'TOTAL':>24s} {grand/1e6:>12.3f}")

    return _probe_events(probe_cpu, num_blocks, TAG_NAMES_COMPUTE, pid_offset=0)


def dump_reduce(probe: torch.Tensor, num_blocks: int):
    """Print reduce phase breakdown; return (events_list, global_base_ns)."""
    probe_cpu = probe.cpu().contiguous().tolist()

    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur:
            max_dur, max_bid = total, bid

    data = probe_cpu[max_bid]; cnt = int(data[0])
    print(f"\n--- Reduce: Slowest block {max_bid} (total={max_dur/1000:.1f}µs): {cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES_REDUCE.get(tag, f'tag_{tag}'):>10s}"
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

    print(f"\n{'='*50}")
    print(f"{'Phase':>10s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>10s}")
    print(f"{'='*50}")
    for name in PHASE_ORDER_REDUCE:
        if name in tag_totals:
            tot = tag_totals[name]; cnt_ = tag_counts[name]
            print(f"{name:>10s} {tot/1e6:>12.3f} {cnt_:>6d} {tot/cnt_/1000:>10.1f}")

    return _probe_events(probe_cpu, num_blocks, TAG_NAMES_REDUCE, pid_offset=200)


def build_combined_trace(compute_events, compute_base,
                         reduce_events, reduce_base) -> str:
    """Shift all events to a shared timeline and serialize to Perfetto JSON."""
    # Both probes use the absolute globaltimer, so just use the earlier base
    shared_base = min(b for b in [compute_base, reduce_base] if b)
    all_events = []
    for ev in compute_events:
        all_events.append(dict(ev, ts=ev["ts"] + (compute_base - shared_base) / 1000.0))
    for ev in reduce_events:
        all_events.append(dict(ev, ts=ev["ts"] + (reduce_base - shared_base) / 1000.0))
    return json.dumps({"traceEvents": all_events})


# ── Kernel constants ──────────────────────────────────────────────────────────

NUM_HEADS = 16
DV = 512
ROW_MAX_SUM_PAIR = 2

DIM_SPLIT  = 256
TOP_K      = 2048
NUM_SPLITS = (TOP_K + DIM_SPLIT - 1) // DIM_SPLIT  # 8

BLOCK_SIZE_COMPUTE = 1024
NUM_WARPS_COMPUTE  = BLOCK_SIZE_COMPUTE // 32  # 32
DIMS_PER_LANE: cutlass.Constexpr = 512 // 32   # 16

NUM_VEC        : cutlass.Constexpr = 8
ITERS_PER_LANE : cutlass.Constexpr = (512 // 32) // 8  # 2

BLOCK_SIZE_REDUCE = 512

SENTINEL_SKIP = float("inf")
LN2 = 0.6931471805599453


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


# ── Profiled JIT wrapper ──────────────────────────────────────────────────────

@cute.jit
def kvsplit_fused_profiled(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
    probe_compute: cute.Tensor,
    probe_reduce: cute.Tensor,
    stream):
    T, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[2]

    N: cutlass.Constexpr = 8462 * 64
    ckv_flat = cute.make_tensor(
        ckv_cache.iterator,
        cute.make_layout((N, head_dim_ckv), stride=(head_dim_ckv, 1)))
    kpe_flat = cute.make_tensor(
        kpe_cache.iterator,
        cute.make_layout((N, head_dim_kpe), stride=(head_dim_kpe, 1)))

    kvsplit_compute_partial_profiled(
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, probe_compute
    ).launch(grid=[T, num_heads, NUM_SPLITS], block=[BLOCK_SIZE_COMPUTE, 1, 1], stream=stream)

    kvsplit_reduce_profiled(
        partial_out, partial_lse, output, lse, probe_reduce
    ).launch(grid=[T, num_heads, 1], block=[BLOCK_SIZE_REDUCE, 1, 1], stream=stream)


# ── Compute kernel ────────────────────────────────────────────────────────────

@cute.kernel
def kvsplit_compute_partial_profiled(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
    probe_compute: cute.Tensor):

    T, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = kpe_cache.shape[1]
    dim_split: cutlass.Constexpr = DIM_SPLIT
    num_splits: cutlass.Constexpr = NUM_SPLITS
    top_k: cutlass.Constexpr = TOP_K
    num_vec: cutlass.Constexpr = NUM_VEC
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE
    dims_per_lane: cutlass.Constexpr = DIMS_PER_LANE

    bidx, bidy, bidz = cute.arch.block_idx()
    num_threads: cutlass.Constexpr = BLOCK_SIZE_COMPUTE
    num_warps:   cutlass.Constexpr = NUM_WARPS_COMPUTE
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    probe_row = bidx * num_heads * num_splits + bidy * num_splits + bidz
    sm = cutlass.Int64(smid_u32())
    probe_cnt = cutlass.Int32(0)

    allocator = cutlass.utils.SmemAllocator()
    smem_sparse  = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((top_k,),       stride=(1,)),  4, None)
    smem_logits  = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((dim_split,),   stride=(1,)), 16, None)
    smem_red_i32 = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((32,),          stride=(1,)),  4, None)
    smem_red_f32 = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((32,),          stride=(1,)), 16, None)
    smem_q_nope  = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv,), stride=(1,)), 16, None)
    smem_q_pe    = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe,), stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout((num_warps, head_dim_ckv), stride=(head_dim_ckv, 1)), 16, None)

    # ── Phase 1: Load ────────────────────────────────────────────────────────
    if tidx == 0:
        range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["load"])

    partial_cnt = 0
    for i in range(tidx, top_k, num_threads):
        idx = sparse_indices[bidx, i]
        smem_sparse[i] = idx
        if idx >= cutlass.Int32(0):
            partial_cnt += 1
    for i in range(tidx, head_dim_ckv, num_threads):
        smem_q_nope[i] = q_nope[bidx, bidy, i]
    for i in range(tidx, head_dim_kpe, num_threads):
        smem_q_pe[i] = q_pe[bidx, bidy, i]

    # ── Phase 2: Valid count ──────────────────────────────────────────────────
    cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
    if lane_idx == 0:
        smem_red_i32[warp_idx] = cnt_sum
    cute.arch.sync_threads()
    if tidx == 0:
        probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)
        range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["valid_count"])

    if warp_idx == 0:
        val = smem_red_i32[lane_idx]
        cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps)
        smem_red_i32[0] = cnt_sum
    cute.arch.sync_threads()

    global_num_valid = smem_red_i32[0]
    split_start = bidz * dim_split
    local_valid  = global_num_valid - split_start
    if local_valid > dim_split:
        local_valid = dim_split
    if local_valid < cutlass.Int32(0):
        local_valid = cutlass.Int32(0)
    active_splits = (global_num_valid + dim_split - 1) // dim_split

    if tidx == 0:
        probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

    # ── OOB vs Active branch ──────────────────────────────────────────────────
    if local_valid == cutlass.Int32(0):
        # OOB split: write sentinels and finalize probe
        for i in range(tidx, head_dim_ckv, num_threads):
            partial_out[bidx, bidy, bidz, i] = cutlass.Float32(0)
        if tidx == 0:
            partial_lse[bidx, bidy, bidz, 0] = -cutlass.Float32(math.inf)
            partial_lse[bidx, bidy, bidz, 1] = cutlass.Float32(0)
            range_finalize(probe_compute, probe_row, probe_cnt)
    else:
        num_rounds = (local_valid + num_warps - 1) // num_warps

        # ── Phase 3: Score ───────────────────────────────────────────────────
        if tidx == 0:
            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["score"])

        q_nope_z = cute.zipped_divide(smem_q_nope, (num_vec,))
        for round_idx in range(num_rounds):
            sparse_idx = round_idx * num_warps + warp_idx
            if sparse_idx < local_valid:
                cur_idx = smem_sparse[split_start + sparse_idx]

                ckv_row = ckv_cache[cur_idx, None]
                ckv_z   = cute.zipped_divide(ckv_row, (num_vec,))

                sum_partial = cutlass.Float32(0)
                for it in range(iters_per_lane):
                    group  = it * wsize + lane_idx
                    q_frag = q_nope_z[(None, (group,))].load()
                    K_frag = ckv_z[(None, (group,))].load()
                    for v in range(num_vec):
                        sum_partial += cutlass.Float32(q_frag[v]) * cutlass.Float32(K_frag[v])
                for k_idx in range(head_dim_kpe // wsize):
                    q_p = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                    kv  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                    sum_partial += q_p * kv
                s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                if lane_idx == 0:
                    smem_logits[sparse_idx] = s * sm_scale

        cute.arch.sync_threads()
        if tidx == 0:
            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

        # ── Phase 4: Softmax max ─────────────────────────────────────────────
        if tidx == 0:
            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["softmax_max"])

        partial_max = -cutlass.Float32(math.inf)
        for idx in range(tidx, local_valid, num_threads):
            v = smem_logits[idx]
            if v > partial_max:
                partial_max = v
        max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
        if lane_idx == 0:
            smem_red_f32[warp_idx] = max_val
        cute.arch.sync_threads()
        if warp_idx == 0:
            val = smem_red_f32[lane_idx]
            max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
            smem_red_f32[0] = max_val
        cute.arch.sync_threads()
        row_max = smem_red_f32[0]
        if tidx == 0:
            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

        # ── Phase 5: Softmax exp+sum+writeback ───────────────────────────────
        if tidx == 0:
            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["softmax_exp_sum"])

        local_sum = cutlass.Float32(0)
        for idx in range(tidx, local_valid, num_threads):
            e = cute.math.exp(smem_logits[idx] - row_max)
            smem_logits[idx] = e
            local_sum += e
        sum_val = warp_reduce(local_sum, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_f32[warp_idx] = sum_val
        cute.arch.sync_threads()
        if warp_idx == 0:
            val = smem_red_f32[lane_idx]
            sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
            smem_red_f32[0] = sum_val
        cute.arch.sync_threads()
        row_sum = smem_red_f32[0]
        if tidx == 0:
            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

        # ── Phase 6: Output GEMV ─────────────────────────────────────────────
        if tidx == 0:
            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["output"])

        # Single-split fast path: normalise weights in-place before GEMV
        if active_splits == cutlass.Int32(1):
            if bidz == cutlass.Int32(0):
                for i in range(tidx, local_valid, num_threads):
                    smem_logits[i] = smem_logits[i] / row_sum
                cute.arch.sync_threads()

        out_regs = cute.make_rmem_tensor(
            cute.make_layout((dims_per_lane,), stride=(1,)),
            cutlass.Float32,
        )
        for k in range(dims_per_lane):
            out_regs[k] = cutlass.Float32(0)

        for round_idx in range(num_rounds):
            j = round_idx * num_warps + warp_idx
            if j < local_valid:
                kv_idx = smem_sparse[split_start + j]
                weight = smem_logits[j]
                V_row  = ckv_cache[kv_idx, None]
                V_z    = cute.zipped_divide(V_row, (num_vec,))
                for it in range(iters_per_lane):
                    group = it * wsize + lane_idx
                    frag  = V_z[(None, (group,))].load()
                    for v in range(num_vec):
                        out_regs[it * num_vec + v] += weight * cutlass.Float32(frag[v])

        for it in range(iters_per_lane):
            for v in range(num_vec):
                smem_partial[warp_idx, (it * wsize + lane_idx) * num_vec + v] = out_regs[it * num_vec + v]

        cute.arch.sync_threads()
        if tidx == 0:
            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)

        # ── Phase 7: Write (cross-warp reduce → partial_out or output) ───────
        if tidx == 0:
            range_start(probe_compute, probe_row, probe_cnt, sm, TAGS_COMPUTE["write"])

        if active_splits == cutlass.Int32(1):
            if bidz == cutlass.Int32(0):
                for i in range(tidx, head_dim_ckv, num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_warps):
                        acc += smem_partial[w, i]
                    output[bidx, bidy, i] = cutlass.BFloat16(acc)
                if tidx == 0:
                    partial_lse[bidx, bidy, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                    lse[bidx, bidy] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
        else:
            for i in range(tidx, head_dim_ckv, num_threads):
                acc = cutlass.Float32(0)
                for w in range(num_warps):
                    acc += smem_partial[w, i]
                partial_out[bidx, bidy, bidz, i] = acc
            if tidx == 0:
                partial_lse[bidx, bidy, bidz, 0] = row_max
                partial_lse[bidx, bidy, bidz, 1] = row_sum

        if tidx == 0:
            probe_cnt = range_stop(probe_compute, probe_row, probe_cnt)
            range_finalize(probe_compute, probe_row, probe_cnt)


# ── Reduce kernel ─────────────────────────────────────────────────────────────

@cute.kernel
def kvsplit_reduce_profiled(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
    probe_reduce: cute.Tensor):

    head_dim_ckv = partial_out.shape[3]
    num_heads    = partial_out.shape[1]
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    reduce_row = bidx * num_heads + bidy
    sm = cutlass.Int64(smid_u32())
    probe_cnt = cutlass.Int32(0)

    if tidx == 0:
        range_start(probe_reduce, reduce_row, probe_cnt, sm, TAGS_REDUCE["reduce"])

    allocator = cutlass.utils.SmemAllocator()
    smem_sentinel = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout((1,), stride=(1,)), 16, None)

    if tidx == 0:
        smem_sentinel[0] = partial_lse[bidx, bidy, 0, 0]
    cute.arch.sync_threads()

    sentinel_val = smem_sentinel[0]

    if sentinel_val < cutlass.Float32(1e30):
        allocator2 = cutlass.utils.SmemAllocator()
        smem_global_max   = allocator2.allocate_tensor(cutlass.Float32,
            cute.make_layout((1,), stride=(1,)), 16, None)
        smem_global_denom = allocator2.allocate_tensor(cutlass.Float32,
            cute.make_layout((1,), stride=(1,)), 16, None)

        if tidx == 0:
            g_max = -cutlass.Float32(math.inf)
            for s in range(num_splits):
                local_max = partial_lse[bidx, bidy, s, 0]
                if local_max > g_max:
                    g_max = local_max
            smem_global_max[0] = g_max

            g_denom = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                g_denom += local_denom * cute.math.exp(local_max - g_max)
            smem_global_denom[0] = g_denom

        cute.arch.sync_threads()

        g_max   = smem_global_max[0]
        g_denom = smem_global_denom[0]

        if tidx == 0:
            lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

        if tidx < head_dim_ckv:
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                scale = cute.math.exp(local_max - g_max) / g_denom
                acc += partial_out[bidx, bidy, s, tidx] * scale
            output[bidx, bidy, tidx] = cutlass.BFloat16(acc)

    if tidx == 0:
        probe_cnt = range_stop(probe_reduce, reduce_row, probe_cnt)
        range_finalize(probe_reduce, reduce_row, probe_cnt)


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=align)

def compile_kernel():
    T   = cute.sym_int()
    N   = cute.sym_int()
    Bc  = cute.sym_int()  # T * NUM_HEADS * NUM_SPLITS (compute probe rows)
    Br  = cute.sym_int()  # T * NUM_HEADS              (reduce  probe rows)

    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048
    num_pages, page_size, num_splits = 8462, 64, NUM_SPLITS
    T_MAX = 8

    q_nope         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv),                 (2, 1, 0), 16)
    q_pe           = _fake(cute.BFloat16, (T, num_heads, head_dim_kpe),                 (2, 1, 0), 16)
    ckv_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_ckv),         (2, 1, 0), 16)
    kpe_cache      = _fake(cute.BFloat16, (num_pages, page_size, head_dim_kpe),         (2, 1, 0), 16)
    sparse_indices = _fake(cute.Int32,    (T, top_k_len),                               (1, 0),     4)
    sm_scale       = 0.1352337788608801
    partial_out    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, head_dim_ckv), (3, 2, 1, 0), 16)
    partial_lse    = _fake(cute.Float32,  (T_MAX, num_heads, num_splits, 2),            (3, 2, 1, 0), 16)
    output         = _fake(cute.BFloat16, (T, num_heads, head_dim_ckv),                 (2, 1, 0), 16)
    lse            = _fake(cute.Float32,  (T, num_heads),                               (1, 0),     4)
    probe_compute  = _fake(cute.Int64,    (Bc, PROBE_COLS),                             (1, 0),     8)
    probe_reduce   = _fake(cute.Int64,    (Br, PROBE_COLS),                             (1, 0),     8)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kvsplit_fused_profiled,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, probe_compute, probe_reduce, stream,
        options="--enable-tvm-ffi"
    )


def run_single(workload_idx: int) -> str:
    import os
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors

    H, D_ckv = NUM_HEADS, DV
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling profiled kv_split_v3_thr_warpv3 kernel...")
    compiled = compile_kernel()

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [json.loads(l) for l in open(JSONL)]
    w   = workloads[workload_idx]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]
    _uuid, _T, max_valid = WORKLOAD_INFO[workload_idx]

    num_compute_blocks = T * H * NUM_SPLITS
    num_reduce_blocks  = T * H
    print(f"\nWorkload {workload_idx + 1}: MaxValid={max_valid}  T={T}  "
          f"ComputeBlocks={num_compute_blocks}  ReduceBlocks={num_reduce_blocks}")

    q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)
    sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

    output      = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse         = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
    partial_out = torch.empty(8, H, NUM_SPLITS, D_ckv, dtype=torch.float32, device="cuda")
    partial_lse = torch.empty(8, H, NUM_SPLITS, ROW_MAX_SUM_PAIR, dtype=torch.float32, device="cuda")
    probe_compute = torch.zeros((num_compute_blocks, PROBE_COLS), dtype=torch.int64, device="cuda")
    probe_reduce  = torch.zeros((num_reduce_blocks,  PROBE_COLS), dtype=torch.int64, device="cuda")

    # Warmup
    for _ in range(3):
        output.zero_(); lse.fill_(-float("inf"))
        probe_compute.zero_(); probe_reduce.zero_()
        compiled(q_nope, q_pe, ckv, kpe, si, partial_out, partial_lse,
                 output, lse, probe_compute, probe_reduce)
        torch.cuda.synchronize()

    # Profile run
    probe_compute.zero_(); probe_reduce.zero_()
    output.zero_(); lse.fill_(-float("inf"))
    compiled(q_nope, q_pe, ckv, kpe, si, partial_out, partial_lse,
             output, lse, probe_compute, probe_reduce)
    torch.cuda.synchronize()

    compute_events, compute_base = dump_compute(
        probe_compute, num_compute_blocks, T, H, NUM_SPLITS)
    reduce_events, reduce_base = dump_reduce(probe_reduce, num_reduce_blocks)

    return build_combined_trace(compute_events, compute_base,
                                reduce_events, reduce_base)

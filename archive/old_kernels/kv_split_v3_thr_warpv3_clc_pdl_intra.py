"""Intra-phase profiling for kv_split_v3_thr_warpv3_clc_pdl.

Two specific measurements on top of the standard phase breakdown:
  1. sparse_indices load overhead — isolated within "load" phase.
  2. pdl_wait overhead — how long reduce blocks stall at griddepcontrol_wait().

Perfetto layout (pid bands):
  0..N       → compute phases   (pid = sm_id)
               phases: load | valid_count | score | softmax_max |
                       softmax_exp_sum | output | clc_wait
  100..N+100 → epilogue write   (pid = sm_id + 100)
  200..N+200 → reduce phases    (pid = sm_id + 200)
               sub-phases: pdl_wait | reduce
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm
import cutlass.utils as utils
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
MAX_ENTRIES  = 10
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY  # 41

TAGS_COMPUTE = {
    "load": 0, "valid_count": 2, "score": 4,
    "softmax_max": 6, "softmax_exp_sum": 8, "output": 10, "clc_wait": 12,
}
TAG_NAMES_COMPUTE = {v: k for k, v in TAGS_COMPUTE.items()}
PHASE_ORDER_COMPUTE = ["load", "valid_count", "score", "softmax_max",
                        "softmax_exp_sum", "output", "clc_wait"]

TAGS_EPI = {"write": 0}
TAG_NAMES_EPI = {0: "write"}

# Reduce phases: pdl_wait comes BEFORE the griddepcontrol_wait returns;
# reduce is the actual computation after.
TAGS_REDUCE = {"pdl_wait": 0, "reduce": 2}
TAG_NAMES_REDUCE = {v: k for k, v in TAGS_REDUCE.items()}
PHASE_ORDER_REDUCE = ["pdl_wait", "reduce"]


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


def _probe_events(probe_cpu, num_rows, tag_names, pid_offset=0):
    events = []
    base = None
    for row in range(num_rows):
        data = probe_cpu[row]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (base is None or s < base):
                base = s
    base = base or 0
    for row in range(num_rows):
        data = probe_cpu[row]; cnt = int(data[0])
        if cnt == 0: continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); t0 = int(data[off + 2]); dur = int(data[off + 3])
            if t0 == 0 and dur == 0: continue
            events.append(dict(
                name=tag_names.get(tag, f"tag_{tag}"), ph="X",
                ts=(t0 - base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id + pid_offset, tid=row))
    return events, base


def dump_compute(probe, num_tiles, T, num_heads, num_splits):
    probe_cpu = probe.cpu().contiguous().tolist()
    active, oob = [], []
    for tid in range(num_tiles):
        cnt = int(probe_cpu[tid][0])
        (active if cnt >= 5 else oob).append(tid)

    max_dur, max_tid = -1, (active[0] if active else 0)
    for tid in active:
        data = probe_cpu[tid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur:
            max_dur, max_tid = total, tid

    data = probe_cpu[max_tid]; cnt = int(data[0])
    tok = max_tid // (num_heads * num_splits); split = max_tid % num_splits
    print(f"\n--- Compute: Slowest tile {max_tid} (token={tok}, split={split}, "
          f"total={max_dur/1000:.1f}µs): {cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off]); tag = int(data[off+1]); dur = int(data[off+3])
        print(f"  sm={sm_id:>3} {TAG_NAMES_COMPUTE.get(tag,f'tag_{tag}'):>16s}"
              f"  dur={dur:>10} ns  ({dur/1000:.1f} µs)")
    print(f"  Active tiles: {len(active)}, OOB: {len(oob)}")

    tag_totals: dict = {}; tag_counts: dict = {}
    for tid in active:
        data = probe_cpu[tid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off+1]); dur = int(data[off+3])
            name = TAG_NAMES_COMPUTE.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1
    grand = sum(tag_totals.values())
    print(f"\n{'='*64}")
    print(f"{'Phase':>26s} {'Total(ms)':>12s} {'N':>6s} {'Avg(µs)':>10s} {'%':>6s}")
    print(f"{'='*64}")
    for name in PHASE_ORDER_COMPUTE:
        if name in tag_totals:
            tot = tag_totals[name]; n = tag_counts[name]
            print(f"{name:>26s} {tot/1e6:>12.3f} {n:>6d} {tot/n/1000:>10.1f} {100*tot/grand:>5.1f}%")
    print(f"{'TOTAL':>26s} {grand/1e6:>12.3f}")
    return _probe_events(probe_cpu, num_tiles, TAG_NAMES_COMPUTE, pid_offset=0)


def dump_epilogue(probe, num_tiles, T, num_heads, num_splits):
    probe_cpu = probe.cpu().contiguous().tolist()
    non_empty = [(t, probe_cpu[t]) for t in range(num_tiles) if int(probe_cpu[t][0]) > 0]
    tag_totals: dict = {}; tag_counts: dict = {}
    for tid, data in non_empty:
        cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off+1]); dur = int(data[off+3])
            name = TAG_NAMES_EPI.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1
    print(f"\n--- Epilogue: {len(non_empty)} active tiles ---")
    for name in ["write"]:
        if name in tag_totals:
            print(f"  {name}: avg={tag_totals[name]/tag_counts[name]/1000:.1f}µs  "
                  f"total={tag_totals[name]/1e6:.3f}ms  n={tag_counts[name]}")
    return _probe_events(probe_cpu, num_tiles, TAG_NAMES_EPI, pid_offset=100)


def dump_reduce(probe, num_blocks):
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
        sm_id = int(data[off]); tag = int(data[off+1]); dur = int(data[off+3])
        print(f"  sm={sm_id:>3} {TAG_NAMES_REDUCE.get(tag,f'tag_{tag}'):>10s}"
              f"  dur={dur:>10} ns  ({dur/1000:.1f} µs)")

    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off+1]); dur = int(data[off+3])
            name = TAG_NAMES_REDUCE.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1
    print(f"\n{'='*50}")
    print(f"  pdl_wait: how long reduce blocks stalled waiting for compute")
    print(f"  reduce:   actual reduction computation")
    print(f"{'='*50}")
    for name in PHASE_ORDER_REDUCE:
        if name in tag_totals:
            n = tag_counts[name]
            print(f"  {name:>10}: avg={tag_totals[name]/n/1000:.1f}µs  "
                  f"total={tag_totals[name]/1e6:.3f}ms  n={n}")
    return _probe_events(probe_cpu, num_blocks, TAG_NAMES_REDUCE, pid_offset=200)


def build_combined_trace(ce, cb, ee, eb, re, rb) -> str:
    shared_base = min(b for b in [cb, eb, rb] if b)
    all_events = []
    for ev in ce: all_events.append(dict(ev, ts=ev["ts"] + (cb - shared_base) / 1000.0))
    for ev in ee: all_events.append(dict(ev, ts=ev["ts"] + (eb - shared_base) / 1000.0))
    for ev in re: all_events.append(dict(ev, ts=ev["ts"] + (rb - shared_base) / 1000.0))
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
DIMS_PER_LANE: cutlass.Constexpr = 512 // 32
NUM_VEC: cutlass.Constexpr = 8
ITERS_PER_LANE: cutlass.Constexpr = (512 // 32) // 8
BLOCK_SIZE_REDUCE = 512
SENTINEL_SKIP = float("inf")
LN2 = 0.6931471805599453


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def kvsplit_fused_clc_pdl_intra(
    q_nope: cute.Tensor, q_pe: cute.Tensor,
    ckv_cache: cute.Tensor, kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor, sm_scale: cutlass.Constexpr,
    partial_out: cute.Tensor, partial_lse: cute.Tensor,
    output: cute.Tensor, lse: cute.Tensor,
    probe_compute: cute.Tensor, probe_epilogue: cute.Tensor,
    probe_reduce: cute.Tensor, stream):

    T, num_heads, head_dim_ckv = q_nope.shape
    N: cutlass.Constexpr = 8462 * 64
    ckv_flat = cute.make_tensor(ckv_cache.iterator,
        cute.make_layout((N, head_dim_ckv), stride=(head_dim_ckv, 1)))
    kpe_flat = cute.make_tensor(kpe_cache.iterator,
        cute.make_layout((N, q_pe.shape[2]), stride=(q_pe.shape[2], 1)))

    num_splits_c: cutlass.Constexpr = NUM_SPLITS
    clc_params = utils.ClcDynamicPersistentTileSchedulerParams(
        (T, num_heads, num_splits_c), (1, 1, 1))
    grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(clc_params)

    kvsplit_compute_clc_pdl_intra_kernel(
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse, clc_params,
        probe_compute, probe_epilogue,
    ).launch(grid=list(grid), block=[BLOCK_SIZE_COMPUTE, 1, 1],
             stream=stream, use_pdl=True)

    kvsplit_reduce_pdl_intra_kernel(
        partial_out, partial_lse, output, lse, probe_reduce,
    ).launch(grid=[T, num_heads, 1], block=[BLOCK_SIZE_REDUCE, 1, 1],
             stream=stream, use_pdl=True)


@cute.kernel
def kvsplit_compute_clc_pdl_intra_kernel(
    q_nope: cute.Tensor, q_pe: cute.Tensor,
    ckv_cache: cute.Tensor, kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor, sm_scale: cutlass.Constexpr,
    partial_out: cute.Tensor, partial_lse: cute.Tensor,
    output: cute.Tensor, lse: cute.Tensor,
    clc_params: utils.ClcDynamicPersistentTileSchedulerParams,
    probe_compute: cute.Tensor, probe_epilogue: cute.Tensor,
):
    T, num_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = kpe_cache.shape[1]
    dim_split:      cutlass.Constexpr = DIM_SPLIT
    num_splits:     cutlass.Constexpr = NUM_SPLITS
    top_k:          cutlass.Constexpr = TOP_K
    num_vec:        cutlass.Constexpr = NUM_VEC
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE
    dims_per_lane:  cutlass.Constexpr = DIMS_PER_LANE
    num_threads:    cutlass.Constexpr = BLOCK_SIZE_COMPUTE
    num_warps:      cutlass.Constexpr = NUM_WARPS_COMPUTE

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    allocator = cutlass.utils.SmemAllocator()
    smem_sparse  = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((top_k,),        stride=(1,)),  4, None)
    smem_logits  = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((dim_split,),    stride=(1,)), 16, None)
    smem_red_i32 = allocator.allocate_tensor(cutlass.Int32,    cute.make_layout((32,),           stride=(1,)),  4, None)
    smem_red_f32 = allocator.allocate_tensor(cutlass.Float32,  cute.make_layout((32,),           stride=(1,)), 16, None)
    smem_q_nope  = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_ckv,), stride=(1,)), 16, None)
    smem_q_pe    = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((head_dim_kpe,), stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(cutlass.Float32,
        cute.make_layout((num_warps, head_dim_ckv), stride=(head_dim_ckv, 1)), 16, None)
    smem_mbar    = allocator.allocate_tensor(cutlass.Int64,  cute.make_layout((1,), stride=(1,)), 8, None)
    smem_clc_rsp = allocator.allocate_tensor(cutlass.Int32,  cute.make_layout((4,), stride=(1,)), 16, None)

    if tidx == 0:
        cute.arch.mbarrier_init(smem_mbar.iterator, 1)
    cute.arch.sync_threads()

    scheduler = utils.ClcDynamicPersistentTileScheduler.create(
        clc_params, cute.arch.block_idx(), cute.arch.grid_dim(), smem_clc_rsp.iterator)

    phase = cutlass.Int32(0)
    work_tile = scheduler.initial_work_tile_info()

    while work_tile.is_valid_tile:
        tok   = work_tile.tile_idx[0]
        head  = work_tile.tile_idx[1]
        split = work_tile.tile_idx[2]
        split_start = split * dim_split
        tile_id = tok * num_heads * num_splits + head * num_splits + split
        sm = cutlass.Int64(smid_u32())
        cnt_c = cutlass.Int32(0)

        # PDL: fire as early as possible (first call per block counts)
        cute.arch.griddepcontrol_launch_dependents()

        # advance_to_next_work uses elect_one() (warp-level), must be called from single warp
        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(smem_mbar.iterator, 16)
            scheduler.advance_to_next_work(smem_mbar.iterator)

        # ── Load phase ────────────────────────────────────────────────────────
        if tidx == 0:
            range_start(probe_compute, tile_id, cnt_c, sm, TAGS_COMPUTE["load"])

        partial_cnt = 0
        for i in range(tidx, top_k, num_threads):
            idx = sparse_indices[tok, i]
            smem_sparse[i] = idx
            if idx >= cutlass.Int32(0):
                partial_cnt += 1
        for i in range(tidx, head_dim_ckv, num_threads):
            smem_q_nope[i] = q_nope[tok, head, i]
        for i in range(tidx, head_dim_kpe, num_threads):
            smem_q_pe[i] = q_pe[tok, head, i]

        cnt_sum = warp_reduce(partial_cnt, lambda a, b: a + b, width=32)
        if lane_idx == 0:
            smem_red_i32[warp_idx] = cnt_sum
        cute.arch.sync_threads()

        if tidx == 0:
            cnt_c = range_stop(probe_compute, tile_id, cnt_c)
            range_start(probe_compute, tile_id, cnt_c, sm, TAGS_COMPUTE["valid_count"])

        if warp_idx == 0:
            val = smem_red_i32[lane_idx]
            cnt_sum = warp_reduce(val, lambda a, b: a + b, width=num_warps)
            smem_red_i32[0] = cnt_sum
        cute.arch.sync_threads()

        global_num_valid = smem_red_i32[0]
        local_valid = global_num_valid - split_start
        if local_valid > dim_split: local_valid = dim_split
        if local_valid < cutlass.Int32(0): local_valid = cutlass.Int32(0)
        active_splits = (global_num_valid + dim_split - 1) // dim_split

        if tidx == 0:
            cnt_c = range_stop(probe_compute, tile_id, cnt_c)

        row_max = -cutlass.Float32(math.inf)
        row_sum = cutlass.Float32(0)

        if local_valid == cutlass.Int32(0):
            for i in range(tidx, head_dim_ckv, num_threads):
                partial_out[tok, head, split, i] = cutlass.Float32(0)
            if tidx == 0:
                partial_lse[tok, head, split, 0] = -cutlass.Float32(math.inf)
                partial_lse[tok, head, split, 1] = cutlass.Float32(0)
        else:
            num_rounds = (local_valid + num_warps - 1) // num_warps

            if tidx == 0:
                range_start(probe_compute, tile_id, cnt_c, sm, TAGS_COMPUTE["score"])
            q_nope_z = cute.zipped_divide(smem_q_nope, (num_vec,))
            for round_idx in range(num_rounds):
                sparse_idx = round_idx * num_warps + warp_idx
                if sparse_idx < local_valid:
                    cur_idx = smem_sparse[split_start + sparse_idx]
                    ckv_row = ckv_cache[cur_idx, None]
                    ckv_z   = cute.zipped_divide(ckv_row, (num_vec,))
                    sum_partial = cutlass.Float32(0)
                    for it in range(iters_per_lane):
                        group = it * wsize + lane_idx
                        q_frag = q_nope_z[(None, (group,))].load()
                        K_frag = ckv_z[(None, (group,))].load()
                        for v in range(num_vec):
                            sum_partial += cutlass.Float32(q_frag[v]) * cutlass.Float32(K_frag[v])
                    for k_idx in range(head_dim_kpe // wsize):
                        sum_partial += (cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx]) *
                                        cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx]))
                    s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                    if lane_idx == 0:
                        smem_logits[sparse_idx] = s * sm_scale
            cute.arch.sync_threads()
            if tidx == 0:
                cnt_c = range_stop(probe_compute, tile_id, cnt_c)
                range_start(probe_compute, tile_id, cnt_c, sm, TAGS_COMPUTE["softmax_max"])

            partial_max = -cutlass.Float32(math.inf)
            for idx in range(tidx, local_valid, num_threads):
                v = smem_logits[idx]
                if v > partial_max: partial_max = v
            max_val = warp_reduce(partial_max, lambda a, b: a if a > b else b, width=32)
            if lane_idx == 0: smem_red_f32[warp_idx] = max_val
            cute.arch.sync_threads()
            if warp_idx == 0:
                val = smem_red_f32[lane_idx]
                max_val = warp_reduce(val, lambda a, b: a if a > b else b, width=num_warps)
                smem_red_f32[0] = max_val
            cute.arch.sync_threads()
            row_max = smem_red_f32[0]
            if tidx == 0:
                cnt_c = range_stop(probe_compute, tile_id, cnt_c)
                range_start(probe_compute, tile_id, cnt_c, sm, TAGS_COMPUTE["softmax_exp_sum"])

            local_sum = cutlass.Float32(0)
            for idx in range(tidx, local_valid, num_threads):
                e = cute.math.exp(smem_logits[idx] - row_max)
                smem_logits[idx] = e; local_sum += e
            sum_val = warp_reduce(local_sum, lambda a, b: a + b, width=32)
            if lane_idx == 0: smem_red_f32[warp_idx] = sum_val
            cute.arch.sync_threads()
            if warp_idx == 0:
                val = smem_red_f32[lane_idx]
                sum_val = warp_reduce(val, lambda a, b: a + b, width=num_warps)
                smem_red_f32[0] = sum_val
            cute.arch.sync_threads()
            row_sum = smem_red_f32[0]
            if tidx == 0:
                cnt_c = range_stop(probe_compute, tile_id, cnt_c)
                range_start(probe_compute, tile_id, cnt_c, sm, TAGS_COMPUTE["output"])

            if active_splits == cutlass.Int32(1):
                if split == cutlass.Int32(0):
                    for i in range(tidx, local_valid, num_threads):
                        smem_logits[i] = smem_logits[i] / row_sum
                    cute.arch.sync_threads()

            out_regs = cute.make_rmem_tensor(cute.make_layout((dims_per_lane,), stride=(1,)), cutlass.Float32)
            for k in range(dims_per_lane): out_regs[k] = cutlass.Float32(0)
            for round_idx in range(num_rounds):
                j = round_idx * num_warps + warp_idx
                if j < local_valid:
                    kv_idx = smem_sparse[split_start + j]; weight = smem_logits[j]
                    V_row = ckv_cache[kv_idx, None]; V_z = cute.zipped_divide(V_row, (num_vec,))
                    for it in range(iters_per_lane):
                        group = it * wsize + lane_idx; frag = V_z[(None, (group,))].load()
                        for v in range(num_vec):
                            out_regs[it * num_vec + v] += weight * cutlass.Float32(frag[v])
            for it in range(iters_per_lane):
                for v in range(num_vec):
                    smem_partial[warp_idx, (it * wsize + lane_idx) * num_vec + v] = out_regs[it * num_vec + v]
            cute.arch.sync_threads()
            if tidx == 0:
                cnt_c = range_stop(probe_compute, tile_id, cnt_c)

        # ── CLC wait phase ────────────────────────────────────────────────────
        if tidx == 0:
            range_start(probe_compute, tile_id, cnt_c, sm, TAGS_COMPUTE["clc_wait"])
        cute.arch.mbarrier_wait(smem_mbar.iterator, phase)
        phase = phase ^ cutlass.Int32(1)
        work_tile = scheduler.get_current_work()
        if tidx == 0:
            cnt_c = range_stop(probe_compute, tile_id, cnt_c)
            range_finalize(probe_compute, tile_id, cnt_c)

        # ── Epilogue ──────────────────────────────────────────────────────────
        if local_valid != cutlass.Int32(0):
            cnt_e = cutlass.Int32(0)
            if tidx == 0:
                range_start(probe_epilogue, tile_id, cnt_e, sm, TAGS_EPI["write"])
            if active_splits == cutlass.Int32(1):
                if split == cutlass.Int32(0):
                    for i in range(tidx, head_dim_ckv, num_threads):
                        acc = cutlass.Float32(0)
                        for w in range(num_warps): acc += smem_partial[w, i]
                        output[tok, head, i] = cutlass.BFloat16(acc)
                    if tidx == 0:
                        partial_lse[tok, head, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                        lse[tok, head] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
            else:
                for i in range(tidx, head_dim_ckv, num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_warps): acc += smem_partial[w, i]
                    partial_out[tok, head, split, i] = acc
                if tidx == 0:
                    partial_lse[tok, head, split, 0] = row_max
                    partial_lse[tok, head, split, 1] = row_sum
            if tidx == 0:
                cnt_e = range_stop(probe_epilogue, tile_id, cnt_e)
                range_finalize(probe_epilogue, tile_id, cnt_e)


@cute.kernel
def kvsplit_reduce_pdl_intra_kernel(
    partial_out: cute.Tensor, partial_lse: cute.Tensor,
    output: cute.Tensor, lse: cute.Tensor,
    probe_reduce: cute.Tensor,
):
    head_dim_ckv = partial_out.shape[3]
    num_heads    = partial_out.shape[1]
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _    = cute.arch.thread_idx()

    reduce_row = bidx * num_heads + bidy
    sm = cutlass.Int64(smid_u32())
    cnt_r = cutlass.Int32(0)

    # ── Prolog (overlaps compute kernel): read sentinel from L2 ──────────────
    # Timed as "pdl_wait" — this starts BEFORE griddepcontrol_wait, so it
    # measures both the L2 read overhead AND the stall at griddepcontrol_wait.
    if tidx == 0:
        range_start(probe_reduce, reduce_row, cnt_r, sm, TAGS_REDUCE["pdl_wait"])

    sentinel_val = partial_lse[bidx, bidy, 0, 0]

    # ── griddepcontrol_wait: stall until all compute tiles are done ───────────
    cute.arch.griddepcontrol_wait()

    if tidx == 0:
        cnt_r = range_stop(probe_reduce, reduce_row, cnt_r)
        range_start(probe_reduce, reduce_row, cnt_r, sm, TAGS_REDUCE["reduce"])

    if sentinel_val < cutlass.Float32(1e30):
        allocator = cutlass.utils.SmemAllocator()
        smem_global_max   = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_global_denom = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

        if tidx == 0:
            g_max = -cutlass.Float32(math.inf)
            for s in range(num_splits):
                local_max = partial_lse[bidx, bidy, s, 0]
                if local_max > g_max: g_max = local_max
            smem_global_max[0] = g_max

            g_denom = cutlass.Float32(0)
            for s in range(num_splits):
                g_denom += (partial_lse[bidx, bidy, s, 1] *
                            cute.math.exp(partial_lse[bidx, bidy, s, 0] - g_max))
            smem_global_denom[0] = g_denom

        cute.arch.sync_threads()
        g_max = smem_global_max[0]; g_denom = smem_global_denom[0]

        if tidx == 0:
            lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)
        if tidx < head_dim_ckv:
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                scale = cute.math.exp(partial_lse[bidx, bidy, s, 0] - g_max) / g_denom
                acc += partial_out[bidx, bidy, s, tidx] * scale
            output[bidx, bidy, tidx] = cutlass.BFloat16(acc)

    if tidx == 0:
        cnt_r = range_stop(probe_reduce, reduce_row, cnt_r)
        range_finalize(probe_reduce, reduce_row, cnt_r)


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)

def compile_kernel():
    T  = cute.sym_int()
    Bt = cute.sym_int()
    Br = cute.sym_int()
    num_heads, head_dim_ckv, head_dim_kpe, top_k_len = 16, 512, 64, 2048
    num_pages, page_size, num_splits, T_MAX = 8462, 64, NUM_SPLITS, 8

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
    probe_compute  = _fake(cute.Int64,    (Bt, PROBE_COLS),                             (1, 0),     8)
    probe_epilogue = _fake(cute.Int64,    (Bt, PROBE_COLS),                             (1, 0),     8)
    probe_reduce   = _fake(cute.Int64,    (Br, PROBE_COLS),                             (1, 0),     8)
    stream         = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kvsplit_fused_clc_pdl_intra,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
        probe_compute, probe_epilogue, probe_reduce, stream,
        options="--enable-tvm-ffi"
    )

_compiled = compile_kernel()


def run_single(workload_idx: int) -> str:
    import os
    from pathlib import Path
    from safetensors.torch import load_file
    from src.utils import WORKLOAD_INFO, make_tensors

    H, D_ckv = NUM_HEADS, DV
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling kv_split_v3_thr_warpv3_clc_pdl_intra kernel...")

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [json.loads(l) for l in open(JSONL)]
    w  = workloads[workload_idx]
    ax = w["workload"]["axes"]; inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]
    _uuid, _T, max_valid = WORKLOAD_INFO[workload_idx]

    num_tiles       = T * H * NUM_SPLITS
    num_reduce_rows = T * H
    print(f"\nWL{workload_idx+1}: uuid={_uuid}  T={T}  MaxValid={max_valid}")
    print(f"  Tiles={num_tiles}  ReduceRows={num_reduce_rows}")

    q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)
    sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

    output      = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse         = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
    partial_out = torch.empty(8, H, NUM_SPLITS, D_ckv, dtype=torch.float32, device="cuda")
    partial_lse = torch.empty(8, H, NUM_SPLITS, ROW_MAX_SUM_PAIR, dtype=torch.float32, device="cuda")
    probe_compute  = torch.zeros((num_tiles,       PROBE_COLS), dtype=torch.int64, device="cuda")
    probe_epilogue = torch.zeros((num_tiles,       PROBE_COLS), dtype=torch.int64, device="cuda")
    probe_reduce   = torch.zeros((num_reduce_rows, PROBE_COLS), dtype=torch.int64, device="cuda")

    for _ in range(3):
        output.zero_(); lse.fill_(-float("inf"))
        probe_compute.zero_(); probe_epilogue.zero_(); probe_reduce.zero_()
        _compiled(q_nope, q_pe, ckv, kpe, si, partial_out, partial_lse,
                  output, lse, probe_compute, probe_epilogue, probe_reduce)
        torch.cuda.synchronize()

    probe_compute.zero_(); probe_epilogue.zero_(); probe_reduce.zero_()
    output.zero_(); lse.fill_(-float("inf"))
    _compiled(q_nope, q_pe, ckv, kpe, si, partial_out, partial_lse,
              output, lse, probe_compute, probe_epilogue, probe_reduce)
    torch.cuda.synchronize()

    print("\n" + "="*68 + "\nCOMPUTE PROBE\n" + "="*68)
    ce, cb = dump_compute(probe_compute, num_tiles, T, H, NUM_SPLITS)
    print("\n" + "="*68 + "\nEPILOGUE PROBE (pid=sm_id+100)\n" + "="*68)
    ee, eb = dump_epilogue(probe_epilogue, num_tiles, T, H, NUM_SPLITS)
    print("\n" + "="*68 + "\nREDUCE PROBE (pid=sm_id+200)  [pdl_wait = stall at griddepcontrol_wait]\n" + "="*68)
    re, rb = dump_reduce(probe_reduce, num_reduce_rows)
    return build_combined_trace(ce, cb, ee, eb, re, rb)

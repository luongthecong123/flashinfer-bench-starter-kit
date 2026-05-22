"""fused_persistent_v3_intra.py — per-phase timing for the persistent compute kernel.

Matches fused_persistent_v3.py: per-tile loads (no startup amortization),
global_valid_count from pre-pass, OOB splits write sentinel partials.

Phases logged per CTA per tile:
  score    — K·q dot-product inner loop
  softmax  — max + exp + sum reductions (fused)
  output   — weighted V accumulation → smem_partial
  reduce   — cross-warp smem_partial → partial_out / direct output

Probe tensor shape: (MAX_ACTIVE_CLUSTERS, PROBE_COLS) = (128, 1 + 40*4) = (128, 161)
Each CTA: up to T_MAX×4 task phases = 32 entries max; 40 gives buffer.

Usage (via modal runner):
    modal run src/modal/persistent_v3_intra.py
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm
import cutlass.utils as utils
import math, json, torch

# ── Probe utilities ───────────────────────────────────────────────────────────

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
MAX_ENTRIES  = 40   # 1 startup + T_MAX(8) × 4 phases = 33 max; 40 gives buffer
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY   # 161

TAGS = {"score": 2, "softmax": 4, "output": 6, "reduce": 8, "oob": 10}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["score", "softmax", "output", "reduce", "oob"]

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

def dump_probe(probe: torch.Tensor, num_blocks: int) -> str:
    probe_cpu = probe.cpu().contiguous().tolist()
    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        total = sum(int(data[PROBE_HEADER + i * PROBE_ENTRY + 3]) for i in range(cnt))
        if total > max_dur:
            max_dur, max_bid = total, bid
    for bid in [max_bid]:
        data = probe_cpu[bid]; cnt = int(data[0])
        print(f"\n--- CTA {bid} (longest, total={max_dur/1000:.1f}µs): {cnt} entries ---")
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id, tag = int(data[off]), int(data[off + 1])
            dur = int(data[off + 3])
            print(f"  sm={sm_id:>3} {TAG_NAMES.get(tag, f'tag_{tag}'):>10s}  dur={dur:>10} ns  ({dur/1000:.1f} µs)")
    tag_totals: dict = {}; tag_counts: dict = {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1
    print(f"\n{'='*60}")
    print(f"{'Phase':>10s} {'Total (ms)':>12s} {'Count':>6s} {'Avg (µs)':>12s} {'%':>8s}")
    print(f"{'='*60}")
    grand_total = sum(tag_totals.values())
    for name in PHASE_ORDER:
        if name in tag_totals:
            total_ns = tag_totals[name]; count = tag_counts[name]
            pct = 100.0 * total_ns / grand_total if grand_total > 0 else 0
            print(f"{name:>10s} {total_ns/1e6:>12.3f} {count:>6d} {total_ns/count/1000:>12.1f} {pct:>7.1f}%")
    print(f"{'TOTAL':>10s} {grand_total/1e6:>12.3f}")
    events = []; global_base = None
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (global_base is None or s < global_base):
                global_base = s
    global_base = global_base or 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        if cnt == 0: continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); start = int(data[off + 2]); dur = int(data[off + 3])
            if start == 0 and dur == 0: continue
            events.append(dict(name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(start - global_base) / 1000.0, dur=dur / 1000.0,
                pid=bid, tid=0))
    return json.dumps({"traceEvents": events})


# ── Constants (must match fused_persistent_v3) ────────────────────────────────
NUM_HEADS  = 16
D_CKV      = 512
D_KPE      = 64
DIM_SPLIT  = 256
TOP_K      = 2048
NUM_SPLITS = TOP_K // DIM_SPLIT   # 8
T_MAX      = 8
TOTAL_TASKS: cutlass.Constexpr = T_MAX * NUM_SPLITS * NUM_HEADS   # 1024

BLOCK_SIZE_COMPUTE = 1024
NUM_WARPS_COMPUTE  = BLOCK_SIZE_COMPUTE // 32   # 32
DIMS_PER_LANE: cutlass.Constexpr = D_CKV // 32  # 16
NUM_VEC:       cutlass.Constexpr = 8
ITERS_PER_LANE: cutlass.Constexpr = (D_CKV // 32) // 8  # 2

BLOCK_SIZE_REDUCE = 512
MAX_ACTIVE_CLUSTERS = 128

LN2 = 0.6931471805599453
SENTINEL_SKIP = float("inf")
N_PAGES_FLAT: cutlass.Constexpr = 8462 * 64


@cute.jit
def warp_reduce(val: cute.Numeric, op: callable, width = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.kernel
def valid_count_kernel_v3_intra(
    sparse_indices: cute.Tensor,
    global_valid_count: cute.Tensor,
):
    tok = cute.arch.block_idx()[0]
    tidx, _, _ = cute.arch.thread_idx()
    num_threads_k: cutlass.Constexpr = 1024
    num_warps_k:   cutlass.Constexpr = 32
    top_k:         cutlass.Constexpr = TOP_K
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    allocator = cutlass.utils.SmemAllocator()
    smem_red = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((32,), stride=(1,)), 4, None)

    cnt = 0
    for i in range(tidx, top_k, num_threads_k):
        if sparse_indices[tok, i] >= cutlass.Int32(0):
            cnt += 1

    s = warp_reduce(cnt, lambda a, b: a + b, width=32)
    if lane_idx == 0:
        smem_red[warp_idx] = s
    cute.arch.sync_threads()

    if warp_idx == 0:
        val = smem_red[lane_idx]
        s = warp_reduce(val, lambda a, b: a + b, width=num_warps_k)
        if lane_idx == 0:
            global_valid_count[tok] = s


# ═══════════════════════════════════════════════════════════════════════════════
# Instrumented compute kernel
# ═══════════════════════════════════════════════════════════════════════════════

@cute.kernel
def persistent_compute_kernel_v3_intra(
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
    global_valid_count: cute.Tensor,
    tile_sched_params: utils.PersistentTileSchedulerParams,
    probe: cute.Tensor,
):
    dim_split:     cutlass.Constexpr = DIM_SPLIT
    num_splits:    cutlass.Constexpr = NUM_SPLITS
    top_k:         cutlass.Constexpr = TOP_K
    num_vec:       cutlass.Constexpr = NUM_VEC
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE
    dims_per_lane:  cutlass.Constexpr = DIMS_PER_LANE
    num_threads:   cutlass.Constexpr = BLOCK_SIZE_COMPUTE
    num_warps:     cutlass.Constexpr = NUM_WARPS_COMPUTE
    n_sh:          cutlass.Constexpr = NUM_SPLITS * NUM_HEADS

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    bidx_x, _, bidx_z = cute.arch.block_idx()
    actual_T = q_nope.shape[0]

    probe_row = bidx_z
    sm        = cutlass.Int64(smid_u32())

    allocator  = cutlass.utils.SmemAllocator()
    smem_sparse  = allocator.allocate_tensor(
        cutlass.Int32,    cute.make_layout((top_k,),     stride=(1,)),  4, None)
    smem_logits  = allocator.allocate_tensor(
        cutlass.Float32,  cute.make_layout((dim_split,), stride=(1,)), 16, None)
    smem_red_f32 = allocator.allocate_tensor(
        cutlass.Float32,  cute.make_layout((32,),        stride=(1,)), 16, None)
    smem_q_nope  = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((D_CKV,),     stride=(1,)), 16, None)
    smem_q_pe    = allocator.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((D_KPE,),     stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((num_warps, D_CKV), stride=(D_CKV, 1)), 16, None)

    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    while work_tile.is_valid_tile:
        flat_idx = work_tile.tile_idx[0]
        tok   =  flat_idx // n_sh
        split = (flat_idx // NUM_HEADS) % num_splits
        head  =  flat_idx % NUM_HEADS
        split_start = split * dim_split

        if tok < actual_T:
            valid_cnt   = global_valid_count[tok]
            local_valid = valid_cnt - split_start
            if local_valid > dim_split:
                local_valid = dim_split
            if local_valid < cutlass.Int32(0):
                local_valid = cutlass.Int32(0)

            active_splits = (valid_cnt + dim_split - 1) // dim_split

            if local_valid == cutlass.Int32(0):
                if tidx == 0:
                    _off = PROBE_HEADER + (tok * 4 + 0) * PROBE_ENTRY
                    probe[probe_row, _off]     = sm
                    probe[probe_row, _off + 1] = cutlass.Int64(TAGS["oob"])
                    probe[probe_row, _off + 2] = globaltimer_u64()
                for i in range(tidx, D_CKV, num_threads):
                    partial_out[tok, head, split, i] = cutlass.Float32(0)
                if tidx == 0:
                    partial_lse[tok, head, split, 0] = -cutlass.Float32(math.inf)
                    partial_lse[tok, head, split, 1] = cutlass.Float32(0)
                    _off = PROBE_HEADER + (tok * 4 + 0) * PROBE_ENTRY
                    probe[probe_row, _off + 3] = globaltimer_u64() - probe[probe_row, _off + 2]
            else:
                for i in range(tidx, top_k, num_threads):
                    smem_sparse[i] = sparse_indices[tok, i]
                for i in range(tidx, D_CKV, num_threads):
                    smem_q_nope[i] = q_nope[tok, head, i]
                for i in range(tidx, D_KPE, num_threads):
                    smem_q_pe[i] = q_pe[tok, head, i]
                cute.arch.sync_threads()

                q_nope_z   = cute.zipped_divide(smem_q_nope, (num_vec,))
                num_rounds = (local_valid + num_warps - 1) // num_warps

                # ── [score] ───────────────────────────────────────────────────
                if tidx == 0:
                    _off = PROBE_HEADER + (tok * 4 + 0) * PROBE_ENTRY
                    probe[probe_row, _off]     = sm
                    probe[probe_row, _off + 1] = cutlass.Int64(TAGS["score"])
                    probe[probe_row, _off + 2] = globaltimer_u64()

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
                                sum_partial += (cutlass.Float32(q_frag[v])
                                                * cutlass.Float32(K_frag[v]))

                        for k_idx in range(D_KPE // wsize):
                            q_p = cutlass.Float32(smem_q_pe[k_idx * wsize + lane_idx])
                            kv  = cutlass.Float32(kpe_cache[cur_idx, k_idx * wsize + lane_idx])
                            sum_partial += q_p * kv

                        s = warp_reduce(sum_partial, lambda a, b: a + b, width=32)
                        if lane_idx == 0:
                            smem_logits[sparse_idx] = s * sm_scale

                cute.arch.sync_threads()

                if tidx == 0:
                    _off = PROBE_HEADER + (tok * 4 + 0) * PROBE_ENTRY
                    probe[probe_row, _off + 3] = globaltimer_u64() - probe[probe_row, _off + 2]

                # ── [softmax] ─────────────────────────────────────────────────
                if tidx == 0:
                    _off = PROBE_HEADER + (tok * 4 + 1) * PROBE_ENTRY
                    probe[probe_row, _off]     = sm
                    probe[probe_row, _off + 1] = cutlass.Int64(TAGS["softmax"])
                    probe[probe_row, _off + 2] = globaltimer_u64()

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

                partial_sum = cutlass.Float32(0)
                for idx in range(tidx, local_valid, num_threads):
                    e = cute.math.exp(smem_logits[idx] - row_max)
                    smem_logits[idx] = e
                    partial_sum += e
                sum_val = warp_reduce(partial_sum, lambda a, b: a + b, width=32)
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
                    _off = PROBE_HEADER + (tok * 4 + 1) * PROBE_ENTRY
                    probe[probe_row, _off + 3] = globaltimer_u64() - probe[probe_row, _off + 2]

                # ── [output] ──────────────────────────────────────────────────
                if tidx == 0:
                    _off = PROBE_HEADER + (tok * 4 + 2) * PROBE_ENTRY
                    probe[probe_row, _off]     = sm
                    probe[probe_row, _off + 1] = cutlass.Int64(TAGS["output"])
                    probe[probe_row, _off + 2] = globaltimer_u64()

                out_regs = cute.make_rmem_tensor(
                    cute.make_layout((dims_per_lane,), stride=(1,)), cutlass.Float32)
                for k in range(dims_per_lane):
                    out_regs[k] = cutlass.Float32(0)

                for round_idx in range(num_rounds):
                    j = round_idx * num_warps + warp_idx
                    if j < local_valid:
                        kv_idx = smem_sparse[split_start + j]
                        weight = smem_logits[j]
                        V_row = ckv_cache[kv_idx, None]
                        V_z   = cute.zipped_divide(V_row, (num_vec,))
                        for it in range(iters_per_lane):
                            group = it * wsize + lane_idx
                            frag  = V_z[(None, (group,))].load()
                            for v in range(num_vec):
                                out_regs[it * num_vec + v] += (
                                    weight * cutlass.Float32(frag[v]))

                for it in range(iters_per_lane):
                    for v in range(num_vec):
                        smem_partial[warp_idx, (it * wsize + lane_idx) * num_vec + v] = (
                            out_regs[it * num_vec + v])

                cute.arch.sync_threads()

                if tidx == 0:
                    _off = PROBE_HEADER + (tok * 4 + 2) * PROBE_ENTRY
                    probe[probe_row, _off + 3] = globaltimer_u64() - probe[probe_row, _off + 2]

                # ── [reduce] ──────────────────────────────────────────────────
                if tidx == 0:
                    _off = PROBE_HEADER + (tok * 4 + 3) * PROBE_ENTRY
                    probe[probe_row, _off]     = sm
                    probe[probe_row, _off + 1] = cutlass.Int64(TAGS["reduce"])
                    probe[probe_row, _off + 2] = globaltimer_u64()

                if active_splits == cutlass.Int32(1):
                    if split == cutlass.Int32(0):
                        for i in range(tidx, D_CKV, num_threads):
                            acc = cutlass.Float32(0)
                            for w in range(num_warps):
                                acc += smem_partial[w, i]
                            output[tok, head, i] = cutlass.BFloat16(acc / row_sum)
                        if tidx == 0:
                            lse[tok, head] = (row_max + cute.math.log(row_sum)) / cutlass.Float32(LN2)
                            partial_lse[tok, head, 0, 0] = cutlass.Float32(SENTINEL_SKIP)
                else:
                    for i in range(tidx, D_CKV, num_threads):
                        acc = cutlass.Float32(0)
                        for w in range(num_warps):
                            acc += smem_partial[w, i]
                        partial_out[tok, head, split, i] = acc
                    if tidx == 0:
                        partial_lse[tok, head, split, 0] = row_max
                        partial_lse[tok, head, split, 1] = row_sum

                if tidx == 0:
                    _off = PROBE_HEADER + (tok * 4 + 3) * PROBE_ENTRY
                    probe[probe_row, _off + 3] = globaltimer_u64() - probe[probe_row, _off + 2]

        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()

    if tidx == 0:
        probe[probe_row, 0] = cutlass.Int64(T_MAX * 4)


# ── Reduce kernel (copy from v3 — unchanged) ─────────────────────────────────

@cute.kernel
def persistent_reduce_kernel_v3_intra(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
):
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _    = cute.arch.thread_idx()

    allocator = cutlass.utils.SmemAllocator()
    smem_sentinel = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

    if tidx == 0:
        smem_sentinel[0] = partial_lse[bidx, bidy, 0, 0]
    cute.arch.sync_threads()

    sentinel_val = smem_sentinel[0]

    if sentinel_val < cutlass.Float32(1e30):
        allocator2 = cutlass.utils.SmemAllocator()
        smem_g_max   = allocator2.allocate_tensor(cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_g_denom = allocator2.allocate_tensor(cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

        if tidx == 0:
            g_max = -cutlass.Float32(math.inf)
            for s in range(num_splits):
                local_max = partial_lse[bidx, bidy, s, 0]
                if local_max > g_max:
                    g_max = local_max
            smem_g_max[0] = g_max

            g_denom = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                g_denom += local_denom * cute.math.exp(local_max - g_max)
            smem_g_denom[0] = g_denom
        cute.arch.sync_threads()

        g_max   = smem_g_max[0]
        g_denom = smem_g_denom[0]

        if tidx == 0:
            lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

        if tidx < D_CKV:
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                scale = cute.math.exp(local_max - g_max) / g_denom
                acc += partial_out[bidx, bidy, s, tidx] * scale
            output[bidx, bidy, tidx] = cutlass.BFloat16(acc)


# ── JIT launcher ─────────────────────────────────────────────────────────────

@cute.jit
def fused_persistent_v3_intra_launcher(
    q_nope: cute.Tensor,
    q_pe: cute.Tensor,
    ckv_cache: cute.Tensor,
    kpe_cache: cute.Tensor,
    sparse_indices: cute.Tensor,
    sm_scale: cutlass.Constexpr,
    global_valid_count: cute.Tensor,
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output: cute.Tensor,
    lse: cute.Tensor,
    probe: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream,
):
    T, num_heads, _ = q_nope.shape

    ckv_flat = cute.make_tensor(ckv_cache.iterator,
        cute.make_layout((N_PAGES_FLAT, D_CKV), stride=(D_CKV, 1)))
    kpe_flat = cute.make_tensor(kpe_cache.iterator,
        cute.make_layout((N_PAGES_FLAT, D_KPE), stride=(D_KPE, 1)))

    # Pre-pass: count valid entries per token
    valid_count_kernel_v3_intra(sparse_indices, global_valid_count).launch(
        grid=[T, 1, 1], block=[1024, 1, 1], stream=stream)

    total_tasks: cutlass.Constexpr = TOTAL_TASKS
    cluster_shape_mnl = (1, 1, 1)
    num_ctas_mnl = (total_tasks, 1, 1)
    tile_sched_params = utils.PersistentTileSchedulerParams(
        num_ctas_mnl, cluster_shape_mnl, swizzle_size=1, raster_along_m=True)
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters)

    persistent_compute_kernel_v3_intra(
        q_nope, q_pe, ckv_flat, kpe_flat,
        sparse_indices, sm_scale,
        partial_out, partial_lse, output, lse,
        global_valid_count,
        tile_sched_params, probe,
    ).launch(grid=grid, block=[BLOCK_SIZE_COMPUTE, 1, 1], stream=stream)

    persistent_reduce_kernel_v3_intra(partial_out, partial_lse, output, lse).launch(
        grid=[T, num_heads, 1], block=[BLOCK_SIZE_REDUCE, 1, 1], stream=stream)


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=align)


def compile_intra():
    T = cute.sym_int()
    num_pages, page_size = 8462, 64

    q_nope             = _fake(cute.BFloat16, (T, NUM_HEADS, D_CKV),                 (2, 1, 0), 16)
    q_pe               = _fake(cute.BFloat16, (T, NUM_HEADS, D_KPE),                 (2, 1, 0), 16)
    ckv_cache          = _fake(cute.BFloat16, (num_pages, page_size, D_CKV),          (2, 1, 0), 16)
    kpe_cache          = _fake(cute.BFloat16, (num_pages, page_size, D_KPE),          (2, 1, 0), 16)
    sparse_indices     = _fake(cute.Int32,    (T, TOP_K),                             (1, 0),     4)
    sm_scale           = 0.1352337788608801
    global_valid_count = _fake(cute.Int32,    (T_MAX,),                               (0,),       4)
    partial_out        = _fake(cute.Float32,  (T_MAX, NUM_HEADS, NUM_SPLITS, D_CKV),  (3, 2, 1, 0), 16)
    partial_lse        = _fake(cute.Float32,  (T_MAX, NUM_HEADS, NUM_SPLITS, 2),      (3, 2, 1, 0), 16)
    output             = _fake(cute.BFloat16, (T, NUM_HEADS, D_CKV),                 (2, 1, 0), 16)
    lse                = _fake(cute.Float32,  (T, NUM_HEADS),                         (1, 0),      4)
    probe              = _fake(cute.Int64,    (MAX_ACTIVE_CLUSTERS, PROBE_COLS),      (1, 0),      8)
    stream             = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fused_persistent_v3_intra_launcher,
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale,
        global_valid_count, partial_out, partial_lse, output, lse,
        probe, MAX_ACTIVE_CLUSTERS, stream,
        options="--enable-tvm-ffi",
    )


# ── run_single ────────────────────────────────────────────────────────────────

def run_single(workload_idx: int) -> str:
    import os
    from pathlib import Path
    from safetensors.torch import load_file

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling persistent_v3_intra kernel...")
    compiled = compile_intra()

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [json.loads(l) for l in open(JSONL)]
    w   = workloads[workload_idx]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]

    print(f"\nWorkload {workload_idx + 1}: T={T}  MAX_ACTIVE_CLUSTERS={MAX_ACTIVE_CLUSTERS}")

    q_nope = torch.randn(T, NUM_HEADS, D_CKV, dtype=torch.bfloat16, device="cuda")
    q_pe   = torch.randn(T, NUM_HEADS, D_KPE, dtype=torch.bfloat16, device="cuda")
    ckv    = torch.randn(P, 64, D_CKV,        dtype=torch.bfloat16, device="cuda")
    kpe    = torch.randn(P, 64, D_KPE,        dtype=torch.bfloat16, device="cuda")
    sf     = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si     = sf[inp["sparse_indices"]["tensor_key"]].cuda()
    output = torch.zeros(T, NUM_HEADS, D_CKV,  dtype=torch.bfloat16, device="cuda")
    lse    = torch.full((T, NUM_HEADS), -float("inf"), dtype=torch.float32, device="cuda")
    global_valid_count = torch.empty(T_MAX, dtype=torch.int32, device="cuda")
    partial_out = torch.empty(T_MAX, NUM_HEADS, NUM_SPLITS, D_CKV,
                              dtype=torch.float32, device="cuda")
    partial_lse = torch.empty(T_MAX, NUM_HEADS, NUM_SPLITS, 2,
                              dtype=torch.float32, device="cuda")
    probe = torch.zeros(MAX_ACTIVE_CLUSTERS, PROBE_COLS, dtype=torch.int64, device="cuda")

    # Warmup (triggers JIT + 3 warm iterations)
    for _ in range(3):
        output.zero_(); lse.fill_(-float("inf"))
        compiled(q_nope, q_pe, ckv, kpe, si, global_valid_count, partial_out, partial_lse, output, lse, probe)
        torch.cuda.synchronize()

    # Profiled run
    probe.zero_(); output.zero_(); lse.fill_(-float("inf"))
    compiled(q_nope, q_pe, ckv, kpe, si, global_valid_count, partial_out, partial_lse, output, lse, probe)
    torch.cuda.synchronize()

    return dump_probe(probe, num_blocks=MAX_ACTIVE_CLUSTERS)

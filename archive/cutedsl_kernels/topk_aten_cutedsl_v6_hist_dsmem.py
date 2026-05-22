"""
topk_aten_cutedsl_v6_hist_dsmem.py — 4-CTA cluster variant of v5_hist.

Strategy:
  * Cluster = 4 CTAs (rank 0..3), each CTA handles a contiguous 1/4-shard of
    cols: rank k owns [k * SHARD_CAP, k * SHARD_CAP + own_shard_len).
  * NUM_THREADS=256/CTA, 8 warps/CTA → 32 warps across the cluster (matches v5).
  * Phase 1 (4-pass 8-bit radix, same algorithm as v5 but distributed):
      - Each CTA bins own shard into 8 per-warp sub-hists.
      - 4-way DSMEM broadcast: every CTA's merged count for each bin is
        `red.add`-ed into **every rank's** local `merged[256]` (4 writes per
        bin per CTA), so after a cluster barrier ALL ranks hold the same
        256-bin histogram. Each rank then does τ-find locally — no broadcast
        needed for τ-state (deterministic, identical inputs).
  * Phase 2 (mask-aware scatter):
      - Pre-count: each CTA computes own_above, own_tie on its shard;
        broadcasts via the same 4-way `red.add` trick → all ranks hold
        per-rank counts. Each rank computes above_prefix/tie_prefix locally.
      - Main scatter: each CTA runs v5's warp-scan scatter on ITS shard with
        above_cursor initialised to above_prefix[rank] (and similarly tie).

SMEM per CTA (~20 KB, comfortably fits in B200's 228 KB):
  smem_bits          [SHARD_CAP]              = 10 KB
  smem_warp_hist     [NUM_WARPS * HIST_BINS]  =  8 KB
  smem_merged        [HIST_BINS]              =  1 KB
  smem_tau           [5]                      + small scratch ≈  1 KB
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm, nvvm
from src.idx_utils import check_topk_indices

# ── Shape ──
TOPK           = 2048
CLUSTER_SIZE   = 4
NUM_THREADS    = 256
NUM_WARPS      = NUM_THREADS // 32           # 8
VEC            = 4
LIMIT_TOPK_SEQ_LEN = 10240
SHARD_CAP      = LIMIT_TOPK_SEQ_LEN // CLUSTER_SIZE  # 2560
HIST_BINS      = 256
NUM_PASSES     = 4


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


# ── Probe infra ──
PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES  = 8
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY
TAGS = {"total": 0, "setup": 2, "phase1": 4, "phase2": 6}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["total", "setup", "phase1", "phase2"]


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


@dsl_user_op
def float_to_radix(v: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    r = llvm.inline_asm(MLIR_T.i32(), [v.ir_value()],
        "{"
        ".reg .u32 x; .reg .u32 mask; .reg .pred pneg; .reg .pred pnan;"
        "mov.b32 x, $1;"
        "setp.lt.f32 pneg, $1, 0f00000000;"
        "setp.neu.f32 pnan, $1, $1;"
        "selp.u32 mask, 0xFFFFFFFF, 0x80000000, pneg;"
        "xor.b32 x, x, mask;"
        "selp.u32 $0, 0xFFFFFFFF, x, pnan;"
        "}",
        "=r,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Uint32(r)


@dsl_user_op
def red_shared_cluster_add_i32(ptr_ir, val: cutlass.Int32, *, loc=None, ip=None) -> None:
    """`red.relaxed.cluster.shared::cluster.add.s32 [ptr], val` (fire-and-forget)."""
    nvvm.red(
        op=nvvm.ReductionOp.ADD,
        type_=nvvm.ReductionType.S32,
        a=ptr_ir,
        b=val.ir_value(loc=loc, ip=ip),
        mem_order=nvvm.MemOrderKind.RELAXED,
        shared_space=nvvm.SharedSpace.shared_cluster,
        mem_scope=nvvm.MemScopeKind.CLUSTER,
        loc=loc, ip=ip,
    )


@cute.jit
def warp_sum_i32(val: cutlass.Int32) -> cutlass.Int32:
    for i in cutlass.range_constexpr(5):
        peer = cute.arch.shuffle_sync_bfly(val, 1 << i)
        val = val + peer
    return val


@cute.kernel
def topk_hist_dsmem_kernel(
    scores:   cute.Tensor,
    out_idx:  cute.Tensor,
    seq_lens: cute.Tensor,
    probe:    cute.Tensor,
):
    # Launch: grid=[CLUSTER_SIZE, B, 1], cluster=[CLUSTER_SIZE,1,1].
    # block_idx_in_cluster() returns rank in the cluster (0..3).
    rank = cute.arch.block_idx_in_cluster()
    b    = cute.arch.block_idx()[1]
    tid  = cute.arch.thread_idx()[0]
    lane = tid % cutlass.Int32(32)
    warp = tid // cutlass.Int32(32)

    sl      = seq_lens[b]
    max_col = scores.shape[1]

    # Own shard: rank k handles cols in [k*SHARD_CAP, min(sl, (k+1)*SHARD_CAP)).
    shard_start = rank * cutlass.Int32(SHARD_CAP)
    shard_stop  = shard_start + cutlass.Int32(SHARD_CAP)
    if shard_stop > sl:
        shard_stop = sl
    if shard_start > sl:
        shard_start = sl
    own_shard_len = shard_stop - shard_start

    probe_row = b * cutlass.Int32(CLUSTER_SIZE) + rank
    probe_cnt = cutlass.Int32(0)
    sm        = smid_u32()

    if tid == cutlass.Int32(0):
        range_start(probe, probe_row, probe_cnt, sm, TAGS["total"])
        probe_cnt = cutlass.Int32(1)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["setup"])

    # ── SMEM alloc ──
    allocator = cutlass.utils.SmemAllocator()
    smem_bits = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((SHARD_CAP,), stride=(1,)), 4, None)
    smem_warp_hist = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS * HIST_BINS,), stride=(1,)), 4, None)
    smem_merged = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((HIST_BINS,), stride=(1,)), 4, None)
    # τ-state + cluster scratch
    smem_tau = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((5,), stride=(1,)), 4, None)
    smem_counts_above = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((CLUSTER_SIZE,), stride=(1,)), 4, None)
    smem_counts_tie = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((CLUSTER_SIZE,), stride=(1,)), 4, None)
    # Phase 2 warp scan scratch
    smem_warp_above = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_warp_tie = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_above_round = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)
    smem_tie_round = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)

    merged_ptr    = smem_merged.iterator
    warp_hist_ptr = smem_warp_hist.iterator
    counts_above_ptr = smem_counts_above.iterator
    counts_tie_ptr   = smem_counts_tie.iterator
    warp_base     = warp * cutlass.Int32(HIST_BINS)

    # Init τ-state (identical on all ranks).
    if tid == cutlass.Int32(0):
        smem_tau[0] = cutlass.Int32(0)          # desired
        smem_tau[1] = cutlass.Int32(0)          # desired_mask
        smem_tau[2] = cutlass.Int32(0)          # above_total
        smem_tau[3] = cutlass.Int32(TOPK)       # k_to_find
        smem_tau[4] = cutlass.Int32(0)          # early_exit

    # ── Phase 0 (setup): load own shard float → radix ──
    i = tid
    while i < own_shard_len:
        bits = float_to_radix(scores[b, shard_start + i])
        smem_bits[i] = cutlass.Int32(bits)
        i = i + cutlass.Int32(NUM_THREADS)
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["phase1"])

    # ── Phase 1: 4-pass 8-bit radix select, distributed ──
    for pass_c in cutlass.range_constexpr(NUM_PASSES):
        digit_pos   = 24 - pass_c * 8
        digit_pos_u = cutlass.Uint32(digit_pos)

        early = smem_tau[4]
        if early == cutlass.Int32(0):
            desired_s      = cutlass.Uint32(smem_tau[0])
            desired_mask_s = cutlass.Uint32(smem_tau[1])
            desired_pin_s  = desired_s & desired_mask_s

            # Clear sub-hist AND merged (merged receives DSMEM red from 4 peers).
            ci = tid
            while ci < cutlass.Int32(NUM_WARPS * HIST_BINS):
                smem_warp_hist[ci] = cutlass.Int32(0)
                ci = ci + cutlass.Int32(NUM_THREADS)
            ci = tid
            while ci < cutlass.Int32(HIST_BINS):
                smem_merged[ci] = cutlass.Int32(0)
                ci = ci + cutlass.Int32(NUM_THREADS)
            cute.arch.sync_threads()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # Bin own shard.
            p = tid
            while p < own_shard_len:
                bits = cutlass.Uint32(smem_bits[p])
                if (bits & desired_mask_s) == desired_pin_s:
                    bin = cutlass.Int32((bits >> digit_pos_u) & cutlass.Uint32(0xFF))
                    cute.arch.atomic_add(warp_hist_ptr + (warp_base + bin),
                                         cutlass.Int32(1),
                                         sem="relaxed", scope="cta")
                p = p + cutlass.Int32(NUM_THREADS)
            cute.arch.sync_threads()

            # Merge own 8 sub-hists into a per-thread bin sum, then 4-way
            # DSMEM broadcast-add into EVERY peer's merged[bin] — after the
            # cluster barrier every rank holds the same 256-bin histogram.
            if tid < cutlass.Int32(HIST_BINS):
                s = cutlass.Int32(0)
                for w in cutlass.range_constexpr(NUM_WARPS):
                    s = s + smem_warp_hist[w * HIST_BINS + tid]
                for r_c in cutlass.range_constexpr(CLUSTER_SIZE):
                    dst = cute.arch.mapa(merged_ptr + tid, cutlass.Int32(r_c))
                    red_shared_cluster_add_i32(dst, s)
            cute.arch.sync_threads()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # τ-find — every rank computes the same answer from the same merged.
            if tid == cutlass.Int32(0):
                k_need = smem_tau[3]
                acc    = cutlass.Int32(0)
                tau_b  = cutlass.Int32(0)
                done   = cutlass.Int32(0)
                for i_c in cutlass.range_constexpr(HIST_BINS):
                    bi_c = HIST_BINS - 1 - i_c
                    if done == cutlass.Int32(0):
                        cnt_b = smem_merged[bi_c]
                        if acc + cnt_b >= k_need:
                            tau_b = cutlass.Int32(bi_c)
                            done  = cutlass.Int32(1)
                        else:
                            acc = acc + cnt_b
                new_desired_mask = desired_mask_s | (cutlass.Uint32(0xFF) << digit_pos_u)
                new_desired      = desired_s | ((cutlass.Uint32(tau_b) & cutlass.Uint32(0xFF)) << digit_pos_u)
                chosen_cnt       = smem_merged[tau_b]
                new_above_total  = smem_tau[2] + acc
                new_k_to_find    = k_need - acc

                smem_tau[0] = cutlass.Int32(new_desired)
                smem_tau[1] = cutlass.Int32(new_desired_mask)
                smem_tau[2] = new_above_total
                smem_tau[3] = new_k_to_find
                if chosen_cnt == new_k_to_find:
                    smem_tau[4] = cutlass.Int32(1)
            cute.arch.sync_threads()

    desired      = cutlass.Uint32(smem_tau[0])
    desired_mask = cutlass.Uint32(smem_tau[1])
    above_total  = smem_tau[2]
    need_ties    = smem_tau[3]
    desired_pin  = desired & desired_mask

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["phase2"])

    # ── Phase 2a: count own_above / own_tie across own shard ──
    # Clear cluster scratch (each rank clears its own counts buffers).
    if tid < cutlass.Int32(CLUSTER_SIZE):
        smem_counts_above[tid] = cutlass.Int32(0)
        smem_counts_tie[tid]   = cutlass.Int32(0)
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    my_a = cutlass.Int32(0)
    my_t = cutlass.Int32(0)
    p = tid
    while p < own_shard_len:
        bits   = cutlass.Uint32(smem_bits[p])
        masked = bits & desired_mask
        if masked > desired_pin:
            my_a = my_a + cutlass.Int32(1)
        if masked == desired_pin:
            my_t = my_t + cutlass.Int32(1)
        p = p + cutlass.Int32(NUM_THREADS)
    # warp reduce
    wa = warp_sum_i32(my_a)
    wt = warp_sum_i32(my_t)
    if lane == cutlass.Int32(0):
        smem_warp_above[warp] = wa
        smem_warp_tie[warp]   = wt
    cute.arch.sync_threads()

    own_above = cutlass.Int32(0)
    own_tie   = cutlass.Int32(0)
    if tid == cutlass.Int32(0):
        for w in cutlass.range_constexpr(NUM_WARPS):
            own_above = own_above + smem_warp_above[w]
            own_tie   = own_tie   + smem_warp_tie[w]
        # 4-way DSMEM broadcast of this rank's counts.
        for r_c in cutlass.range_constexpr(CLUSTER_SIZE):
            da = cute.arch.mapa(counts_above_ptr + rank, cutlass.Int32(r_c))
            dt = cute.arch.mapa(counts_tie_ptr   + rank, cutlass.Int32(r_c))
            red_shared_cluster_add_i32(da, own_above)
            red_shared_cluster_add_i32(dt, own_tie)
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    # Compute own prefixes (all ranks compute identical prefixes locally).
    above_prefix = cutlass.Int32(0)
    tie_prefix   = cutlass.Int32(0)
    for r_c in cutlass.range_constexpr(CLUSTER_SIZE):
        if rank > cutlass.Int32(r_c):
            above_prefix = above_prefix + smem_counts_above[r_c]
            tie_prefix   = tie_prefix   + smem_counts_tie[r_c]
        elif rank == cutlass.Int32(r_c):
            # no-op; prefix excludes self
            pass

    # ── Phase 2b: warp-scan scatter over OWN shard ──
    above_cursor = above_prefix
    tie_cursor   = tie_prefix

    col = cutlass.Int32(0)
    while col < cutlass.Int32(SHARD_CAP):
        cur_local = col + tid
        is_valid  = cur_local < own_shard_len
        cur_col   = shard_start + cur_local       # absolute column index

        bits = cutlass.Uint32(0)
        if is_valid:
            bits = cutlass.Uint32(smem_bits[cur_local])

        is_b = cutlass.Int32(0)
        is_t = cutlass.Int32(0)
        if is_valid:
            masked = bits & desired_mask
            if masked > desired_pin:
                is_b = cutlass.Int32(1)
            if masked == desired_pin:
                is_t = cutlass.Int32(1)

        scan_b = is_b
        for s in cutlass.range_constexpr(5):
            peer = cute.arch.shuffle_sync_up(scan_b, 1 << s, mask_and_clamp=0)
            if lane >= cutlass.Int32(1 << s):
                scan_b = scan_b + peer
        my_b_excl  = scan_b - is_b
        warp_b_tot = cute.arch.shuffle_sync(scan_b, 31)

        scan_t = is_t
        for s in cutlass.range_constexpr(5):
            peer2 = cute.arch.shuffle_sync_up(scan_t, 1 << s, mask_and_clamp=0)
            if lane >= cutlass.Int32(1 << s):
                scan_t = scan_t + peer2
        my_t_excl  = scan_t - is_t
        warp_t_tot = cute.arch.shuffle_sync(scan_t, 31)

        if lane == cutlass.Int32(31):
            smem_warp_above[warp] = warp_b_tot
            smem_warp_tie[warp]   = warp_t_tot
        cute.arch.sync_threads()

        # 8 warp totals → inclusive scan on warp 0.
        if warp == cutlass.Int32(0) and lane < cutlass.Int32(NUM_WARPS):
            wta = smem_warp_above[lane]
            wtt = smem_warp_tie[lane]
            orig_wta = wta
            orig_wtt = wtt
            for s in cutlass.range_constexpr(3):    # NUM_WARPS=8 → 3 steps
                pa = cute.arch.shuffle_sync_up(wta, 1 << s, mask_and_clamp=0)
                pt = cute.arch.shuffle_sync_up(wtt, 1 << s, mask_and_clamp=0)
                if lane >= cutlass.Int32(1 << s):
                    wta = wta + pa
                    wtt = wtt + pt
            smem_warp_above[lane] = wta - orig_wta
            smem_warp_tie[lane]   = wtt - orig_wtt
            if lane == cutlass.Int32(NUM_WARPS - 1):
                smem_above_round[0] = wta
                smem_tie_round[0]   = wtt
        cute.arch.sync_threads()

        warp_b_off = smem_warp_above[warp]
        warp_t_off = smem_warp_tie[warp]

        if is_b > cutlass.Int32(0):
            goff = above_cursor + warp_b_off + my_b_excl
            if goff < above_total:
                out_idx[b, goff] = cur_col

        if is_t > cutlass.Int32(0):
            toff    = tie_cursor + warp_t_off + my_t_excl
            wrt_pos = above_total + toff
            if toff < need_ties:
                if wrt_pos < cutlass.Int32(TOPK):
                    out_idx[b, wrt_pos] = cur_col

        above_round = smem_above_round[0]
        tie_round   = smem_tie_round[0]
        cute.arch.sync_threads()

        above_cursor = above_cursor + above_round
        tie_cursor   = tie_cursor   + tie_round
        col          = col + cutlass.Int32(NUM_THREADS)

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        off = PROBE_HEADER + 0 * PROBE_ENTRY
        probe[probe_row, off + 3] = globaltimer_u64() - probe[probe_row, off + 2]
        range_finalize(probe, probe_row, probe_cnt)


@cute.jit
def topk_hist_dsmem_cutedsl(scores, out_idx, seq_lens, probe):
    B = scores.shape[0]
    topk_hist_dsmem_kernel(scores, out_idx, seq_lens, probe).launch(
        grid=[CLUSTER_SIZE, B, 1],
        block=[NUM_THREADS, 1, 1],
        cluster=[CLUSTER_SIZE, 1, 1],
    )


def dump_probe(probe: torch.Tensor, B: int, label: str = "") -> str:
    probe_cpu = probe.cpu().contiguous().tolist()
    # Find slowest (rank, batch) by total.
    max_dur = -1
    max_row = 0
    for row in range(B * CLUSTER_SIZE):
        data = probe_cpu[row]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            if int(data[off + 1]) == TAGS["total"]:
                if int(data[off + 3]) > max_dur:
                    max_dur = int(data[off + 3])
                    max_row = row
                break

    data = probe_cpu[max_row]; cnt = int(data[0])
    b_of = max_row // CLUSTER_SIZE
    r_of = max_row %  CLUSTER_SIZE
    print(f"\n--- {label}  slowest row (b={b_of}, rank={r_of})  total={max_dur/1000:.2f} µs ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id = int(data[off]); tag = int(data[off + 1]); dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES.get(tag, f'tag_{tag}'):>10s}  dur={dur:>10} ns  ({dur/1000:.2f} µs)")

    tag_totals, tag_counts = {}, {}
    for row in range(B * CLUSTER_SIZE):
        data = probe_cpu[row]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"\n{'='*64}")
    print(f"{'Phase':>10s} {'Sum (µs)':>12s} {'N':>6s} {'Avg (µs)':>12s}")
    print(f"{'='*64}")
    for name in PHASE_ORDER:
        if name in tag_totals:
            t = tag_totals[name]; c = tag_counts[name]
            print(f"{name:>10s} {t/1000:>12.2f} {c:>6d} {t/c/1000:>12.2f}")

    # chrome trace
    events = []
    global_base = None
    for row in range(B * CLUSTER_SIZE):
        data = probe_cpu[row]; cnt = int(data[0])
        for i in range(cnt):
            s = int(data[PROBE_HEADER + i * PROBE_ENTRY + 2])
            if s > 0 and (global_base is None or s < global_base):
                global_base = s
    global_base = global_base or 0
    for row in range(B * CLUSTER_SIZE):
        data = probe_cpu[row]; cnt = int(data[0])
        if cnt == 0:
            continue
        sm_id = int(data[PROBE_HEADER])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); start = int(data[off + 2]); dur = int(data[off + 3])
            if start == 0 and dur == 0:
                continue
            events.append(dict(name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(start - global_base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id, tid=row))
    return json.dumps({"traceEvents": events})


WORKLOAD_CASES = [
    ("WL0  B=1 sl=2049", 1, 2049),
    ("WL1  B=1 sl=3000", 1, 3000),
    ("WL2  B=1 sl=4096", 1, 4096),
    ("WL3  B=1 sl=6000", 1, 6000),
    ("WL4  B=1 sl=8192", 1, 8192),
    ("WL5  B=1 sl=10000", 1, 10000),
]


def make_fakes():
    return (
        make_fake_compact_tensor(dtype=cute.Float32, shape=(cute.sym_int(), cute.sym_int()),
                                 stride_order=(1, 0), assumed_align=16),
        make_fake_compact_tensor(dtype=cute.Int32,   shape=(cute.sym_int(), cute.sym_int()),
                                 stride_order=(1, 0), assumed_align=16),
        make_fake_compact_tensor(dtype=cute.Int32,   shape=(cute.sym_int(),),
                                 stride_order=(0,),   assumed_align=4),
        make_fake_compact_tensor(dtype=cute.Int64,   shape=(cute.sym_int(), cute.sym_int()),
                                 stride_order=(1, 0), assumed_align=8),
    )


def run_single(workload_idx: int) -> str:
    label, B, sl = WORKLOAD_CASES[workload_idx]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n── {label} ──")
    print("Compiling v6_hist_dsmem kernel...")
    compiled = cute.compile(topk_hist_dsmem_cutedsl, *make_fakes())
    print("Done.")

    device = "cuda"
    torch.manual_seed(0)
    scores   = torch.randn(B, sl, dtype=torch.float32, device=device)
    seq_lens = torch.full((B,), sl, dtype=torch.int32, device=device)
    out_idx  = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
    probe    = torch.zeros((B * CLUSTER_SIZE, PROBE_COLS), dtype=torch.int64, device=device)

    for _ in range(5):
        probe.zero_()
        compiled(scores, out_idx, seq_lens, probe)
        torch.cuda.synchronize()

    ref_idx = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
    for bb in range(B):
        _, idx = torch.topk(scores[bb, :sl], min(TOPK, sl))
        ref_idx[bb, :min(TOPK, sl)] = idx.int()
    ok, miss = check_topk_indices(ref_idx, out_idx, seq_lens)
    print(f"  CORRECTNESS {'PASS' if ok else 'FAIL'}  worst_miss={miss:.6f}")

    probe.zero_()
    compiled(scores, out_idx, seq_lens, probe)
    torch.cuda.synchronize()
    return dump_probe(probe, B=B, label=label)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl", type=int, default=0)
    args = ap.parse_args()
    run_single(args.wl)

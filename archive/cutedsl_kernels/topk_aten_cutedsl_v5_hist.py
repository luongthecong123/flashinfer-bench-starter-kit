"""
v5_hist — 8-bit-digit radix select (4-pass histogram) replacing v4's 2-bit 16-pass.

vs v4_fuse_earlyexit_intra_smem_v1:
  * Phase 1 digit size = 8 bits (256 bins) instead of 2 bits (4 bins).
  * 4 passes max instead of 16 — each pass:
      1. histogram the current digit over elements still matching `desired_mask`
      2. walk hist from bin 255 → 0, find τ-bucket b*
      3. extend `desired / desired_mask` with b* at this digit position
      4. early-exit if chosen bucket count exactly equals `k_to_find`
  * Phase 2 is verbatim from v4 (mask-aware, since our final desired_mask may
    be any of 0xFF000000 / 0xFFFF0000 / 0xFFFFFF00 / 0xFFFFFFFF).
  * τ-find is unrolled with `range_constexpr(256)` — single-thread scan.

SMEM budget (per CTA):
  smem_bits       [10240]  int32 = 40 KB
  smem_warp_hist  [32*256] int32 = 32 KB   (32 warps × 256 bins, cta-scope atomics)
  smem_hist       [256]    int32 =  1 KB
  ... scan scratch               ≈  1 KB
  Total                          ≈ 74 KB   (B200 has 228 KB)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm
from src.idx_utils import check_topk_indices

TOPK        = 2048
NUM_THREADS = 1024
NUM_WARPS   = NUM_THREADS // 32
VEC         = 4

LIMIT_TOPK_SEQ_LEN = 10240
SMEM_BITS_CAP      = LIMIT_TOPK_SEQ_LEN

HIST_BINS  = 256
NUM_PASSES = 4


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


@cute.jit
def warp_sum_i32(val: cutlass.Int32) -> cutlass.Int32:
    for i in cutlass.range_constexpr(5):
        peer = cute.arch.shuffle_sync_bfly(val, 1 << i)
        val = val + peer
    return val


@cute.kernel
def topk_hist_kernel(
    scores:   cute.Tensor,
    out_idx:  cute.Tensor,
    seq_lens: cute.Tensor,
    probe:    cute.Tensor,
):
    b    = cute.arch.block_idx()[0]
    tid  = cute.arch.thread_idx()[0]
    lane = tid % cutlass.Int32(32)
    warp = tid // cutlass.Int32(32)

    sl        = seq_lens[b]
    max_col   = scores.shape[1]
    probe_row = b
    probe_cnt = cutlass.Int32(0)
    sm        = smid_u32()

    if tid == cutlass.Int32(0):
        range_start(probe, probe_row, probe_cnt, sm, TAGS["total"])
        probe_cnt = cutlass.Int32(1)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["setup"])

    # ── SMEM allocation ──
    allocator = cutlass.utils.SmemAllocator()
    smem_bits = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((SMEM_BITS_CAP,), stride=(1,)), 4, None)
    smem_warp_hist = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS * HIST_BINS,), stride=(1,)), 4, None)
    smem_hist = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((HIST_BINS,), stride=(1,)), 4, None)
    # [desired, desired_mask, above_total, k_to_find, early_exit]
    smem_tau = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((5,), stride=(1,)), 4, None)
    smem_warp_above = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_warp_tie = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_above_round = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)
    smem_tie_round = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)

    warp_hist_ptr = smem_warp_hist.iterator
    warp_base     = warp * cutlass.Int32(HIST_BINS)

    # Init τ-state.
    if tid == cutlass.Int32(0):
        smem_tau[0] = cutlass.Int32(0)              # desired
        smem_tau[1] = cutlass.Int32(0)              # desired_mask
        smem_tau[2] = cutlass.Int32(0)              # above_total
        smem_tau[3] = cutlass.Int32(TOPK)           # k_to_find
        smem_tau[4] = cutlass.Int32(0)              # early_exit

    # ── Setup: float → radix into smem_bits ──
    setup_base = tid * cutlass.Int32(VEC)
    while setup_base + cutlass.Int32(VEC - 1) < sl:
        bits0 = float_to_radix(scores[b, setup_base])
        bits1 = float_to_radix(scores[b, setup_base + cutlass.Int32(1)])
        bits2 = float_to_radix(scores[b, setup_base + cutlass.Int32(2)])
        bits3 = float_to_radix(scores[b, setup_base + cutlass.Int32(3)])
        smem_bits[setup_base + cutlass.Int32(0)] = cutlass.Int32(bits0)
        smem_bits[setup_base + cutlass.Int32(1)] = cutlass.Int32(bits1)
        smem_bits[setup_base + cutlass.Int32(2)] = cutlass.Int32(bits2)
        smem_bits[setup_base + cutlass.Int32(3)] = cutlass.Int32(bits3)
        setup_base = setup_base + cutlass.Int32(NUM_THREADS * VEC)
    while setup_base < sl:
        bits = float_to_radix(scores[b, setup_base])
        smem_bits[setup_base] = cutlass.Int32(bits)
        setup_base = setup_base + cutlass.Int32(1)
    cute.arch.sync_threads()

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["phase1"])

    # ── Phase 1: 4-pass 8-bit histogram radix select ──
    for pass_c in cutlass.range_constexpr(NUM_PASSES):
        digit_pos   = 24 - pass_c * 8           # 24, 16, 8, 0
        digit_pos_u = cutlass.Uint32(digit_pos)

        early = smem_tau[4]
        if early == cutlass.Int32(0):
            desired_s      = cutlass.Uint32(smem_tau[0])
            desired_mask_s = cutlass.Uint32(smem_tau[1])
            desired_pin_s  = desired_s & desired_mask_s

            # Clear warp sub-hists.
            clear_idx = tid
            while clear_idx < cutlass.Int32(NUM_WARPS * HIST_BINS):
                smem_warp_hist[clear_idx] = cutlass.Int32(0)
                clear_idx = clear_idx + cutlass.Int32(NUM_THREADS)
            cute.arch.sync_threads()

            # Bin elements matching desired_mask.
            pos = tid
            while pos < sl:
                bits = cutlass.Uint32(smem_bits[pos])
                if (bits & desired_mask_s) == desired_pin_s:
                    bin = cutlass.Int32((bits >> digit_pos_u) & cutlass.Uint32(0xFF))
                    cute.arch.atomic_add(warp_hist_ptr + (warp_base + bin),
                                         cutlass.Int32(1),
                                         sem="relaxed", scope="cta")
                pos = pos + cutlass.Int32(NUM_THREADS)
            cute.arch.sync_threads()

            # Merge 32 sub-hists → smem_hist.
            if tid < cutlass.Int32(HIST_BINS):
                s = cutlass.Int32(0)
                for w in cutlass.range_constexpr(NUM_WARPS):
                    s = s + smem_warp_hist[w * HIST_BINS + tid]
                smem_hist[tid] = s
            cute.arch.sync_threads()

            # τ-find on tid 0, fully unrolled over 256 bins.
            if tid == cutlass.Int32(0):
                k_need = smem_tau[3]
                acc    = cutlass.Int32(0)
                tau_b  = cutlass.Int32(0)
                done   = cutlass.Int32(0)
                for i_c in cutlass.range_constexpr(HIST_BINS):
                    bi_c = HIST_BINS - 1 - i_c
                    if done == cutlass.Int32(0):
                        cnt_b = smem_hist[bi_c]
                        if acc + cnt_b >= k_need:
                            tau_b = cutlass.Int32(bi_c)
                            done  = cutlass.Int32(1)
                        else:
                            acc = acc + cnt_b
                new_desired_mask = desired_mask_s | (cutlass.Uint32(0xFF) << digit_pos_u)
                new_desired      = desired_s | ((cutlass.Uint32(tau_b) & cutlass.Uint32(0xFF)) << digit_pos_u)
                chosen_cnt       = smem_hist[tau_b]
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

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["phase2"])

    # ── Phase 2 (fused, MASK-AWARE) — verbatim from v4 ──
    above_cursor = cutlass.Int32(0)
    tie_cursor   = cutlass.Int32(0)
    desired_pin  = desired & desired_mask

    col = cutlass.Int32(0)
    while col < max_col:
        cur_col  = col + tid
        is_valid = cur_col < sl

        bits = cutlass.Uint32(0)
        if is_valid:
            bits = cutlass.Uint32(smem_bits[cur_col])

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

        if warp == cutlass.Int32(0):
            wta = smem_warp_above[lane]
            orig_wta = wta
            for s in cutlass.range_constexpr(5):
                p = cute.arch.shuffle_sync_up(wta, 1 << s, mask_and_clamp=0)
                if lane >= cutlass.Int32(1 << s):
                    wta = wta + p
            smem_warp_above[lane] = wta - orig_wta
            above_round_tot = warp_sum_i32(orig_wta)
            if lane == cutlass.Int32(0):
                smem_above_round[0] = above_round_tot

            wtt = smem_warp_tie[lane]
            orig_wtt = wtt
            for s in cutlass.range_constexpr(5):
                p2 = cute.arch.shuffle_sync_up(wtt, 1 << s, mask_and_clamp=0)
                if lane >= cutlass.Int32(1 << s):
                    wtt = wtt + p2
            smem_warp_tie[lane] = wtt - orig_wtt
            tie_round_tot = warp_sum_i32(orig_wtt)
            if lane == cutlass.Int32(0):
                smem_tie_round[0] = tie_round_tot
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
def topk_hist_cutedsl(scores, out_idx, seq_lens, probe):
    B = scores.shape[0]
    topk_hist_kernel(scores, out_idx, seq_lens, probe).launch(
        grid=[B, 1, 1], block=[NUM_THREADS, 1, 1],
    )


def dump_probe(probe: torch.Tensor, num_blocks: int, label: str = "") -> str:
    probe_cpu = probe.cpu().contiguous().tolist()
    max_dur, max_bid = -1, 0
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        block_total = 0
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            if int(data[off + 1]) == TAGS["total"]:
                block_total = int(data[off + 3])
                break
        if block_total > max_dur:
            max_dur, max_bid = block_total, bid

    data = probe_cpu[max_bid]; cnt = int(data[0])
    print(f"\n--- {label}  Block {max_bid} (longest total={max_dur/1000:.2f} µs): {cnt} entries ---")
    for i in range(cnt):
        off = PROBE_HEADER + i * PROBE_ENTRY
        sm_id, tag = int(data[off]), int(data[off + 1])
        dur = int(data[off + 3])
        print(f"  sm={sm_id:>3} {TAG_NAMES.get(tag, f'tag_{tag}'):>10s}  dur={dur:>10} ns  ({dur/1000:.2f} µs)")

    tag_totals, tag_counts = {}, {}
    for bid in range(num_blocks):
        data = probe_cpu[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"\n{'='*64}")
    print(f"{'Phase':>10s} {'Total (µs)':>12s} {'Count':>6s} {'Avg (µs)':>12s} {'% of total':>12s}")
    print(f"{'='*64}")
    total_ref = tag_totals.get("total", 0)
    for name in PHASE_ORDER:
        if name in tag_totals:
            total_ns = tag_totals[name]; count = tag_counts[name]
            pct = 100.0 * total_ns / total_ref if total_ref > 0 else 0
            print(f"{name:>10s} {total_ns/1000:>12.2f} {count:>6d} {total_ns/count/1000:>12.2f} {pct:>11.1f}%")

    events, global_base = [], None
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
                pid=sm_id, tid=bid))
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
    print("Compiling v5_hist kernel...")
    compiled = cute.compile(topk_hist_cutedsl, *make_fakes())
    print("Done.")

    device = "cuda"
    torch.manual_seed(0)
    scores   = torch.randn(B, sl, dtype=torch.float32, device=device)
    seq_lens = torch.full((B,), sl, dtype=torch.int32, device=device)
    out_idx  = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
    probe    = torch.zeros((B, PROBE_COLS), dtype=torch.int64, device=device)

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
    return dump_probe(probe, num_blocks=B, label=label)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl", type=int, default=0)
    args = ap.parse_args()
    run_single(args.wl)

"""
histogram_dsmem.py — single-cluster 256-bin histogram using DSMEM atomics,
with intra-kernel globaltimer profiling (the definitive GPU-time measurement).

Kernel shape:
  * `grid == cluster == [CLUSTER_SIZE, 1, 1]` (4 CTAs, one cluster).
  * 192 threads / CTA (6 warps × 32).
  * Phase 1 (`phase1_bin`):    per-warp sub-histograms in SMEM via cta-local atomicAdd.
  * Phase 2 (`phase2_reduce`): each CTA reduces 6 sub-hists → 256-bin vector,
                               DSMEM reduce-add into CTA 0's `merged[256]` via
                               `nvvm.red` → PTX `red.relaxed.cluster.shared::cluster.add.s32`.
                               (NVVM rejects `atomicrmw syncscope=cluster` on ptr<3>.)
  * Phase 3 (`phase3_write`):  CTA 0 plain-stores the merged histogram to gmem.

Intra profiling:
  Per-CTA probe buffer records phase start + duration via %globaltimer.
  We use this (not torch.cuda.Event) as the definitive GPU-time measurement —
  wall-clock timing includes launch overhead which dominates for small N.

Note on `red.async`:
  PTX 8.1+ supports `red.async.relaxed.cluster.shared::cluster.
  mbarrier::complete_tx::bytes.add.s32 [addr], b, [mbar]`. Non-blocking,
  completion tracked via mbarrier. Useful when there is independent work
  to overlap between issue and completion. Here, Phase 2 is immediately
  followed by a cluster barrier (peer atomics must land before Phase 3),
  and Phase 2 *is* the hot step — the synchronous `nvvm.red` is sufficient.
  If we ever want to overlap Phase 2 with an unrelated producer, switching
  to `red.async` + mbarrier `complete_tx` is straightforward.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm, nvvm

# ── Cluster / CTA shape ──
CLUSTER_SIZE    = 4                       # 4 CTAs per cluster
NUM_WARPS       = 6                       # 6 warps per CTA
THREADS_PER_CTA = NUM_WARPS * 32          # = 192
HIST_BINS       = 256
SUBHIST_SIZE    = NUM_WARPS * HIST_BINS   # per-CTA sub-histogram ints


# ── Probe / intra-kernel timing ──
PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES  = 8
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY
TAGS = {
    "total": 0,
    "clear": 2,
    "phase1_bin": 4,
    "phase2_reduce": 6,
    "phase3_write": 8,
}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["total", "clear", "phase1_bin", "phase2_reduce", "phase3_write"]


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
    """DSMEM reduce-add: `red.relaxed.cluster.shared::cluster.add.s32 [ptr], val`.

    `ptr_ir` must be `!llvm.ptr<3>` (shared addr-space) — obtain via `cute.arch.mapa`.
    Fire-and-forget (no return value). Matches our nvvm.red dict enum discovery:
      op=ADD, type_=S32, shared_space=shared_cluster, mem_scope=CLUSTER, mem_order=RELAXED.
    """
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


@cute.kernel
def histogram_dsmem_top8_kernel(
    data:  cute.Tensor,    # [N] float32
    hist:  cute.Tensor,    # [256] int32
    probe: cute.Tensor,    # [CLUSTER_SIZE, PROBE_COLS] int64
    n:     cutlass.Int32,
):
    tid  = cute.arch.thread_idx()[0]
    warp = tid // cutlass.Int32(32)
    rank = cute.arch.block_idx_in_cluster()                # 0..CLUSTER_SIZE-1
    cluster_threads = cutlass.Int32(CLUSTER_SIZE * THREADS_PER_CTA)
    gtid = rank * cutlass.Int32(THREADS_PER_CTA) + tid

    probe_row = rank
    probe_cnt = cutlass.Int32(0)
    sm        = smid_u32()

    if tid == cutlass.Int32(0):
        range_start(probe, probe_row, probe_cnt, sm, TAGS["total"])
        probe_cnt = cutlass.Int32(1)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["clear"])

    allocator = cutlass.utils.SmemAllocator()
    sub = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((SUBHIST_SIZE,), stride=(1,)), 4, None,
    )
    # `merged` must be allocated in every CTA so the SMEM layout is identical
    # across the cluster — mapa(ptr, 0) remaps CTA k's local offset into CTA 0's
    # SMEM, which requires equal layouts.
    merged = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((HIST_BINS,), stride=(1,)), 4, None,
    )
    sub_ptr    = sub.iterator
    merged_ptr = merged.iterator

    # ── Clear sub-hist + merged (CTA 0's merged is the real destination) ──
    i = tid
    while i < cutlass.Int32(SUBHIST_SIZE):
        sub[i] = cutlass.Int32(0)
        i = i + cutlass.Int32(THREADS_PER_CTA)
    i = tid
    while i < cutlass.Int32(HIST_BINS):
        merged[i] = cutlass.Int32(0)
        i = i + cutlass.Int32(THREADS_PER_CTA)
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["phase1_bin"])

    # ── Phase 1: bin into per-warp sub-hist; data split across whole cluster ──
    warp_base = warp * cutlass.Int32(HIST_BINS)
    pos = gtid
    while pos < n:
        bits = float_to_radix(data[pos])
        bin  = cutlass.Int32((bits >> cutlass.Uint32(24)) & cutlass.Uint32(0xFF))
        cute.arch.atomic_add(sub_ptr + (warp_base + bin), cutlass.Int32(1),
                             sem="relaxed", scope="cta")
        pos = pos + cluster_threads
    cute.arch.sync_threads()

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["phase2_reduce"])

    # ── Phase 2: reduce 6 sub-hists → DSMEM-add into CTA 0's merged ──
    bin_idx = tid
    while bin_idx < cutlass.Int32(HIST_BINS):
        s = cutlass.Int32(0)
        for w in cutlass.range_constexpr(NUM_WARPS):
            s = s + sub[w * HIST_BINS + bin_idx]
        dst = cute.arch.mapa(merged_ptr + bin_idx, cutlass.Int32(0))
        red_shared_cluster_add_i32(dst, s)
        bin_idx = bin_idx + cutlass.Int32(THREADS_PER_CTA)
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["phase3_write"])

    # ── Phase 3: CTA 0 writes merged histogram to global ──
    if rank == cutlass.Int32(0):
        bin_idx = tid
        while bin_idx < cutlass.Int32(HIST_BINS):
            hist[bin_idx] = merged[bin_idx]
            bin_idx = bin_idx + cutlass.Int32(THREADS_PER_CTA)

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        # close `total`
        off_total = PROBE_HEADER + 0 * PROBE_ENTRY
        probe[probe_row, off_total + 3] = globaltimer_u64() - probe[probe_row, off_total + 2]
        range_finalize(probe, probe_row, probe_cnt)


@cute.jit
def histogram_dsmem_launch(data, hist, probe, n: cutlass.Int32):
    histogram_dsmem_top8_kernel(data, hist, probe, n).launch(
        grid=[CLUSTER_SIZE, 1, 1],
        block=[THREADS_PER_CTA, 1, 1],
        cluster=[CLUSTER_SIZE, 1, 1],
    )


# ── Reference + probe dump ────────────────────────────────────────────────────
def reference_top8(data_f32: torch.Tensor) -> torch.Tensor:
    bits = data_f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign_neg = (data_f32 < 0)
    nan_mask = torch.isnan(data_f32)
    mask = torch.where(sign_neg, torch.tensor(0xFFFFFFFF, dtype=torch.int64),
                                  torch.tensor(0x80000000, dtype=torch.int64))
    radix = bits ^ mask
    radix = torch.where(nan_mask, torch.tensor(0xFFFFFFFF, dtype=torch.int64), radix)
    top8 = (radix >> 24) & 0xFF
    return torch.bincount(top8, minlength=256).to(torch.int32)


def dump_probe(probe: torch.Tensor, label: str = ""):
    p = probe.cpu().contiguous().tolist()
    num_ctas = len(p)
    print(f"\n── Intra-kernel probe: {label} ({num_ctas} CTAs in cluster) ──")

    # per-CTA totals
    totals = []
    for bid in range(num_ctas):
        data = p[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            if int(data[off + 1]) == TAGS["total"]:
                totals.append((bid, int(data[off + 3])))
                break
    if totals:
        max_bid, max_dur = max(totals, key=lambda x: x[1])
        min_bid, min_dur = min(totals, key=lambda x: x[1])
        print(f"  total per CTA: min={min_dur/1000:.3f} µs (rank {min_bid})  "
              f"max={max_dur/1000:.3f} µs (rank {max_bid})")

    # slowest CTA detailed breakdown
    if totals:
        max_bid, max_dur = max(totals, key=lambda x: x[1])
        data = p[max_bid]; cnt = int(data[0])
        print(f"  slowest CTA: rank={max_bid}  total={max_dur/1000:.3f} µs")
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            sm_id, tag, dur = int(data[off]), int(data[off + 1]), int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            print(f"    sm={sm_id:>3} {name:>14s}  dur={dur:>9} ns  ({dur/1000:.3f} µs)")

    # phase averages across cluster
    tag_totals, tag_counts = {}, {}
    for bid in range(num_ctas):
        data = p[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1

    print(f"  {'Phase':>14s} {'Sum (µs)':>10s} {'N':>4s} {'Avg (µs)':>10s}")
    for name in PHASE_ORDER:
        if name in tag_totals:
            t, c = tag_totals[name], tag_counts[name]
            print(f"  {name:>14s} {t/1000:>10.3f} {c:>4d} {t/c/1000:>10.3f}")


def make_fakes():
    return (
        make_fake_compact_tensor(dtype=cute.Float32, shape=(cute.sym_int(),),
                                 stride_order=(0,), assumed_align=16),
        make_fake_compact_tensor(dtype=cute.Int32, shape=(HIST_BINS,),
                                 stride_order=(0,), assumed_align=16),
        make_fake_compact_tensor(dtype=cute.Int64, shape=(cute.sym_int(), cute.sym_int()),
                                 stride_order=(1, 0), assumed_align=8),
    )


def run_case(n: int):
    device = "cuda"
    torch.manual_seed(0)
    label = f"N={n:,}"
    print(f"\n── {label} ──")

    data  = torch.randn(n, dtype=torch.float32, device=device)
    hist  = torch.zeros(HIST_BINS, dtype=torch.int32, device=device)
    probe = torch.zeros((CLUSTER_SIZE, PROBE_COLS), dtype=torch.int64, device=device)

    compiled = cute.compile(histogram_dsmem_launch, *make_fakes(), cutlass.Int32(n))

    # warmup
    for _ in range(5):
        hist.zero_(); probe.zero_()
        compiled(data, hist, probe, cutlass.Int32(n))
    torch.cuda.synchronize()

    # correctness
    hist.zero_(); probe.zero_()
    compiled(data, hist, probe, cutlass.Int32(n))
    torch.cuda.synchronize()
    ref = reference_top8(data.cpu())
    got = hist.cpu()
    diff = (got - ref).abs()
    ok = bool((diff == 0).all())
    print(f"  CORRECTNESS {'PASS' if ok else 'FAIL'}  max|diff|={int(diff.max())}  sum={int(got.sum())}")

    # definitive measurement: intra-kernel probe
    dump_probe(probe, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=65536)
    args = ap.parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    run_case(args.n)


if __name__ == "__main__":
    main()

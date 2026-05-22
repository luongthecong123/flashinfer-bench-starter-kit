"""
histogram.py — 256-bin histogram POC in CuTe DSL.

Adapted from NVIDIA cuda-samples/Samples/2_Concepts_and_Techniques/histogram
(histogram256.cu). The core idea is per-warp sub-histograms in shared memory
to avoid block-wide atomic contention:

    sub_hist[NUM_WARPS][256]   in SMEM
    each warp has its own slice → atomicAdd inside a warp's slice contends
    only with that warp's own threads (32-way), not block-wide (192-way).

After binning, a final merge pass sums each bin across warps and writes the
block-partial histogram into the global output (atomically, so multiple CTAs
can cooperate on a single dataset).

Usage:
    python src/kernels/histogram.py [--n N] [--blocks G] [--bytes-mode {byte,top8}]

Modes:
    byte   — treat input as uint32 and histogram its 4 bytes (matches NVIDIA sample).
    top8   — treat input as float32, convert to monotone radix bits, histogram top 8 bits
             (the building block we want for the topk threshold kernel).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass._mlir.dialects import llvm

# ── Kernel constants ──────────────────────────────────────────────────────────
HIST_BINS       = 256
NUM_WARPS       = 6                        # NVIDIA sample uses 6 warps / CTA
THREADS_PER_CTA = NUM_WARPS * 32           # = 192
SUBHIST_SIZE    = NUM_WARPS * HIST_BINS    # ints in SMEM (= 6 KB)


# ── Intra-kernel probe (single-CTA top8 path only) ──
PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES  = 8
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY
TAGS = {
    "total": 0,
    "clear": 2,
    "phase1_bin": 4,
    "phase2_merge_write": 6,
}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["total", "clear", "phase1_bin", "phase2_merge_write"]


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
    """float32 → monotone uint32 (NaN → 0xFFFFFFFF, sits at the top)."""
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


# ── Byte-mode kernel: histograms 4 bytes per uint32 (matches NVIDIA sample) ───
@cute.kernel
def histogram256_byte_kernel(
    data: cute.Tensor,    # [N] uint32
    hist: cute.Tensor,    # [256] uint32 (global, accumulated atomically)
    n:    cutlass.Int32,
):
    tid  = cute.arch.thread_idx()[0]
    bid  = cute.arch.block_idx()[0]
    grid = cute.arch.grid_dim()[0]
    warp = tid // cutlass.Int32(32)

    allocator = cutlass.utils.SmemAllocator()
    smem = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((SUBHIST_SIZE,), stride=(1,)), 4, None,
    )
    smem_ptr = smem.iterator

    # ── Clear sub-histograms ──
    i = tid
    while i < cutlass.Int32(SUBHIST_SIZE):
        smem[i] = cutlass.Int32(0)
        i = i + cutlass.Int32(THREADS_PER_CTA)
    cute.arch.sync_threads()

    warp_base = warp * cutlass.Int32(HIST_BINS)

    # ── Per-warp binning of 4 bytes per uint32 word ──
    pos    = bid * cutlass.Int32(THREADS_PER_CTA) + tid
    stride = grid * cutlass.Int32(THREADS_PER_CTA)
    while pos < n:
        word = cutlass.Uint32(data[pos])
        b0 = cutlass.Int32( word                          & cutlass.Uint32(0xFF))
        b1 = cutlass.Int32((word >> cutlass.Uint32(8))    & cutlass.Uint32(0xFF))
        b2 = cutlass.Int32((word >> cutlass.Uint32(16))   & cutlass.Uint32(0xFF))
        b3 = cutlass.Int32((word >> cutlass.Uint32(24))   & cutlass.Uint32(0xFF))
        cute.arch.atomic_add(smem_ptr + (warp_base + b0), cutlass.Int32(1), sem="relaxed", scope="cta")
        cute.arch.atomic_add(smem_ptr + (warp_base + b1), cutlass.Int32(1), sem="relaxed", scope="cta")
        cute.arch.atomic_add(smem_ptr + (warp_base + b2), cutlass.Int32(1), sem="relaxed", scope="cta")
        cute.arch.atomic_add(smem_ptr + (warp_base + b3), cutlass.Int32(1), sem="relaxed", scope="cta")
        pos = pos + stride
    cute.arch.sync_threads()

    # ── Merge per-warp → block partial → global atomic ──
    hist_ptr = hist.iterator
    bin_idx = tid
    while bin_idx < cutlass.Int32(HIST_BINS):
        s = cutlass.Int32(0)
        for w in cutlass.range_constexpr(NUM_WARPS):
            s = s + smem[w * HIST_BINS + bin_idx]
        cute.arch.atomic_add(hist_ptr + bin_idx, s, sem="relaxed", scope="gpu")
        bin_idx = bin_idx + cutlass.Int32(THREADS_PER_CTA)


# ── Top-8 mode: float32 → radix bits → histogram top 8 bits (1 bin per word) ──
@cute.kernel
def histogram256_top8_kernel(
    data: cute.Tensor,    # [N] float32
    hist: cute.Tensor,    # [256] uint32
    n:    cutlass.Int32,
):
    tid  = cute.arch.thread_idx()[0]
    bid  = cute.arch.block_idx()[0]
    grid = cute.arch.grid_dim()[0]
    warp = tid // cutlass.Int32(32)

    allocator = cutlass.utils.SmemAllocator()
    smem = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((SUBHIST_SIZE,), stride=(1,)), 4, None,
    )
    smem_ptr = smem.iterator

    i = tid
    while i < cutlass.Int32(SUBHIST_SIZE):
        smem[i] = cutlass.Int32(0)
        i = i + cutlass.Int32(THREADS_PER_CTA)
    cute.arch.sync_threads()

    warp_base = warp * cutlass.Int32(HIST_BINS)

    pos    = bid * cutlass.Int32(THREADS_PER_CTA) + tid
    stride = grid * cutlass.Int32(THREADS_PER_CTA)
    while pos < n:
        bits = float_to_radix(data[pos])
        bin  = cutlass.Int32((bits >> cutlass.Uint32(24)) & cutlass.Uint32(0xFF))
        cute.arch.atomic_add(smem_ptr + (warp_base + bin), cutlass.Int32(1), sem="relaxed", scope="cta")
        pos = pos + stride
    cute.arch.sync_threads()

    hist_ptr = hist.iterator
    bin_idx = tid
    while bin_idx < cutlass.Int32(HIST_BINS):
        s = cutlass.Int32(0)
        for w in cutlass.range_constexpr(NUM_WARPS):
            s = s + smem[w * HIST_BINS + bin_idx]
        cute.arch.atomic_add(hist_ptr + bin_idx, s, sem="relaxed", scope="gpu")
        bin_idx = bin_idx + cutlass.Int32(THREADS_PER_CTA)


@cute.jit
def histogram_byte_launch(data, hist, n: cutlass.Int32, grid: cutlass.Constexpr):
    histogram256_byte_kernel(data, hist, n).launch(
        grid=[grid, 1, 1], block=[THREADS_PER_CTA, 1, 1],
    )


@cute.jit
def histogram_top8_launch(data, hist, n: cutlass.Int32, grid: cutlass.Constexpr):
    histogram256_top8_kernel(data, hist, n).launch(
        grid=[grid, 1, 1], block=[THREADS_PER_CTA, 1, 1],
    )


# ── Intra-probed single-CTA top8 kernel (direct SMEM→gmem plain store) ──
# Used for fair comparison vs histogram_dsmem: this is the shape we'd drop
# into the topk slow-path (1 CTA per batch element, no cross-CTA merge).
@cute.kernel
def histogram256_top8_probed_kernel(
    data:  cute.Tensor,    # [N] float32
    hist:  cute.Tensor,    # [256] int32
    probe: cute.Tensor,    # [1, PROBE_COLS] int64
    n:     cutlass.Int32,
):
    tid  = cute.arch.thread_idx()[0]
    warp = tid // cutlass.Int32(32)

    probe_row = cutlass.Int32(0)
    probe_cnt = cutlass.Int32(0)
    sm        = smid_u32()

    if tid == cutlass.Int32(0):
        range_start(probe, probe_row, probe_cnt, sm, TAGS["total"])
        probe_cnt = cutlass.Int32(1)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["clear"])

    allocator = cutlass.utils.SmemAllocator()
    smem = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((SUBHIST_SIZE,), stride=(1,)), 4, None,
    )
    smem_ptr = smem.iterator

    i = tid
    while i < cutlass.Int32(SUBHIST_SIZE):
        smem[i] = cutlass.Int32(0)
        i = i + cutlass.Int32(THREADS_PER_CTA)
    cute.arch.sync_threads()

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["phase1_bin"])

    warp_base = warp * cutlass.Int32(HIST_BINS)
    pos    = tid
    stride = cutlass.Int32(THREADS_PER_CTA)
    while pos < n:
        bits = float_to_radix(data[pos])
        bin  = cutlass.Int32((bits >> cutlass.Uint32(24)) & cutlass.Uint32(0xFF))
        cute.arch.atomic_add(smem_ptr + (warp_base + bin), cutlass.Int32(1),
                             sem="relaxed", scope="cta")
        pos = pos + stride
    cute.arch.sync_threads()

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        range_start(probe, probe_row, probe_cnt, sm, TAGS["phase2_merge_write"])

    # Single-CTA merge: sum across warps, plain-store to gmem (no global atomic).
    bin_idx = tid
    while bin_idx < cutlass.Int32(HIST_BINS):
        s = cutlass.Int32(0)
        for w in cutlass.range_constexpr(NUM_WARPS):
            s = s + smem[w * HIST_BINS + bin_idx]
        hist[bin_idx] = s
        bin_idx = bin_idx + cutlass.Int32(THREADS_PER_CTA)

    if tid == cutlass.Int32(0):
        probe_cnt = range_stop(probe, probe_row, probe_cnt)
        off_total = PROBE_HEADER + 0 * PROBE_ENTRY
        probe[probe_row, off_total + 3] = globaltimer_u64() - probe[probe_row, off_total + 2]
        range_finalize(probe, probe_row, probe_cnt)


@cute.jit
def histogram_top8_probed_launch(data, hist, probe, n: cutlass.Int32):
    histogram256_top8_probed_kernel(data, hist, probe, n).launch(
        grid=[1, 1, 1], block=[THREADS_PER_CTA, 1, 1],
    )


# ── Reference + tester ────────────────────────────────────────────────────────
def reference_byte(data_u32: torch.Tensor) -> torch.Tensor:
    """4 bytes per word → 256-bin histogram."""
    bytes_view = data_u32.view(torch.uint8)            # length = 4N
    return torch.bincount(bytes_view.to(torch.int64), minlength=256).to(torch.int32)


def reference_top8(data_f32: torch.Tensor) -> torch.Tensor:
    """float32 → radix bits → top 8 bits → 256-bin histogram."""
    bits = data_f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign_neg = (data_f32 < 0)
    nan_mask = torch.isnan(data_f32)
    mask = torch.where(sign_neg, torch.tensor(0xFFFFFFFF, dtype=torch.int64),
                                  torch.tensor(0x80000000, dtype=torch.int64))
    radix = bits ^ mask
    radix = torch.where(nan_mask, torch.tensor(0xFFFFFFFF, dtype=torch.int64), radix)
    top8 = (radix >> 24) & 0xFF
    return torch.bincount(top8, minlength=256).to(torch.int32)


def make_fakes_byte():
    return (
        make_fake_compact_tensor(dtype=cute.Uint32, shape=(cute.sym_int(),),
                                 stride_order=(0,), assumed_align=16),
        make_fake_compact_tensor(dtype=cute.Uint32, shape=(HIST_BINS,),
                                 stride_order=(0,), assumed_align=16),
    )


def make_fakes_top8():
    return (
        make_fake_compact_tensor(dtype=cute.Float32, shape=(cute.sym_int(),),
                                 stride_order=(0,), assumed_align=16),
        make_fake_compact_tensor(dtype=cute.Uint32, shape=(HIST_BINS,),
                                 stride_order=(0,), assumed_align=16),
    )


def run(n: int, grid: int, mode: str):
    device = "cuda"
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"mode={mode}  N={n:,}  grid={grid}  threads/CTA={THREADS_PER_CTA}")

    if mode == "byte":
        data = torch.randint(0, 2**31 - 1, (n,), dtype=torch.int32, device=device)
        hist = torch.zeros(HIST_BINS, dtype=torch.int32, device=device)
        ref  = torch.bincount(data.cpu().view(torch.uint8).to(torch.int64), minlength=256).to(torch.int32)
        compiled = cute.compile(histogram_byte_launch, *make_fakes_byte(), cutlass.Int32(n), grid)
        compiled(data, hist, cutlass.Int32(n))
        torch.cuda.synchronize()
    elif mode == "top8":
        data = torch.randn(n, dtype=torch.float32, device=device)
        hist = torch.zeros(HIST_BINS, dtype=torch.int32, device=device)
        ref  = reference_top8(data.cpu())
        compiled = cute.compile(histogram_top8_launch, *make_fakes_top8(), cutlass.Int32(n), grid)
        compiled(data, hist, cutlass.Int32(n))
        torch.cuda.synchronize()
    else:
        raise ValueError(mode)

    got = hist.cpu().to(torch.int32)
    diff = (got - ref).abs()
    ok   = bool((diff == 0).all())
    print(f"  sum(got)={int(got.sum())}  sum(ref)={int(ref.sum())}  max|diff|={int(diff.max())}")
    print(f"  CORRECTNESS {'PASS' if ok else 'FAIL'}")

    # quick latency
    iters = 50
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        hist.zero_()
        compiled(data, hist, cutlass.Int32(n))
    stop.record(); torch.cuda.synchronize()
    avg_us = start.elapsed_time(stop) * 1000.0 / iters
    print(f"  latency: {avg_us:.2f} µs  ({n / (avg_us * 1e-6) / 1e9:.2f} G elements/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--mode", choices=["byte", "top8"], default="top8")
    args = ap.parse_args()
    run(args.n, args.blocks, args.mode)


# ── Intra-kernel probe variant: single-CTA top8, for fair comparison vs dsmem ──
def make_fakes_top8_probed():
    return (
        make_fake_compact_tensor(dtype=cute.Float32, shape=(cute.sym_int(),),
                                 stride_order=(0,), assumed_align=16),
        make_fake_compact_tensor(dtype=cute.Int32, shape=(HIST_BINS,),
                                 stride_order=(0,), assumed_align=16),
        make_fake_compact_tensor(dtype=cute.Int64, shape=(cute.sym_int(), cute.sym_int()),
                                 stride_order=(1, 0), assumed_align=8),
    )


def dump_probe(probe: torch.Tensor, label: str = ""):
    p = probe.cpu().contiguous().tolist()
    num_ctas = len(p)
    print(f"\n── Intra-kernel probe: {label} ({num_ctas} CTA) ──")
    tag_totals, tag_counts = {}, {}
    for bid in range(num_ctas):
        data = p[bid]; cnt = int(data[0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(data[off + 1]); dur = int(data[off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            tag_totals[name] = tag_totals.get(name, 0) + dur
            tag_counts[name] = tag_counts.get(name, 0) + 1
    print(f"  {'Phase':>20s} {'Avg (µs)':>10s}")
    for name in PHASE_ORDER:
        if name in tag_totals:
            t, c = tag_totals[name], tag_counts[name]
            print(f"  {name:>20s} {t/c/1000:>10.3f}")


def run_case(n: int):
    """Single-CTA top8 histogram with intra-kernel probe timing (match dsmem runner)."""
    device = "cuda"
    torch.manual_seed(0)
    label = f"N={n:,}"
    print(f"\n── {label} ──")

    data  = torch.randn(n, dtype=torch.float32, device=device)
    hist  = torch.zeros(HIST_BINS, dtype=torch.int32, device=device)
    probe = torch.zeros((1, PROBE_COLS), dtype=torch.int64, device=device)

    compiled = cute.compile(histogram_top8_probed_launch,
                            *make_fakes_top8_probed(), cutlass.Int32(n))

    for _ in range(5):
        hist.zero_(); probe.zero_()
        compiled(data, hist, probe, cutlass.Int32(n))
    torch.cuda.synchronize()

    hist.zero_(); probe.zero_()
    compiled(data, hist, probe, cutlass.Int32(n))
    torch.cuda.synchronize()
    ref = reference_top8(data.cpu())
    got = hist.cpu()
    diff = (got - ref).abs()
    ok = bool((diff == 0).all())
    print(f"  CORRECTNESS {'PASS' if ok else 'FAIL'}  max|diff|={int(diff.max())}  sum={int(got.sum())}")
    dump_probe(probe, label=label)


if __name__ == "__main__":
    main()

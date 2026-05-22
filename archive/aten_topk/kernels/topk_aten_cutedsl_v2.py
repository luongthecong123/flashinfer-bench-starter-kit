"""
Exact top-K via CuTe DSL — sbtopk radix select (RADIX_BITS=4, 8 passes).
v2: wider radix (4-bit digits, 16 bins, 8 passes instead of 16).

Type discipline:
- Int32  for all counts, cursors, indices, smem tensors
- Uint32 for bit-pattern registers only (desired, desired_mask, kth_bits, 'bits')
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.testing import benchmark, JitArguments
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm
from src.idx_utils import check_topk_indices

TOPK        = 2048
NUM_THREADS = 1024
NUM_WARPS   = NUM_THREADS // 32
RADIX_BITS  = 4
NUM_BINS    = 1 << RADIX_BITS   # 16
NUM_PASSES  = 32 // RADIX_BITS  # 8
DIGIT_MASK  = NUM_BINS - 1      # 0xF

# ────────────────────────────────────────────────────────────────────────────
# PTX helpers (Uint32 bit-pattern operations)
# ────────────────────────────────────────────────────────────────────────────

@dsl_user_op
def float_to_radix(v: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    r = llvm.inline_asm(
        T.i32(), [v.ir_value()],
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
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return cutlass.Uint32(r)

@dsl_user_op
def u_and(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    r = llvm.inline_asm(T.i32(), [a.ir_value(), b.ir_value()],
        "and.b32 $0,$1,$2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def u_or(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    r = llvm.inline_asm(T.i32(), [a.ir_value(), b.ir_value()],
        "or.b32 $0,$1,$2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def u_xor(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    r = llvm.inline_asm(T.i32(), [a.ir_value(), b.ir_value()],
        "xor.b32 $0,$1,$2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def u_shr(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    r = llvm.inline_asm(T.i32(), [a.ir_value(), b.ir_value()],
        "shr.u32 $0,$1,$2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def u_shl(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Uint32:
    r = llvm.inline_asm(T.i32(), [a.ir_value(), b.ir_value()],
        "shl.b32 $0,$1,$2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Uint32(r)

@dsl_user_op
def u_gt(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Boolean:
    r = llvm.inline_asm(T.i(1), [a.ir_value(), b.ir_value()],
        "setp.gt.u32 $0,$1,$2;", "=b,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Boolean(r)

@dsl_user_op
def u_eq(a: cutlass.Uint32, b: cutlass.Uint32, *, loc=None, ip=None) -> cutlass.Boolean:
    r = llvm.inline_asm(T.i(1), [a.ir_value(), b.ir_value()],
        "setp.eq.u32 $0,$1,$2;", "=b,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Boolean(r)


# ────────────────────────────────────────────────────────────────────────────
# Warp-level primitives (Int32 counts)
# ────────────────────────────────────────────────────────────────────────────

@cute.jit
def warp_sum_i32(val: cutlass.Int32) -> cutlass.Int32:
    for i in cutlass.range_constexpr(5):
        peer = cute.arch.shuffle_sync_bfly(val, 1 << i)
        val = val + peer
    return val


# ────────────────────────────────────────────────────────────────────────────
# Main kernel — RADIX_BITS=4, 8 passes, 16 bins
# ────────────────────────────────────────────────────────────────────────────

@cute.kernel
def topk_radix_kernel(
    scores:   cute.Tensor,
    out_idx:  cute.Tensor,
    seq_lens: cute.Tensor,
):
    b   = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    lane = tid % cutlass.Int32(32)
    warp = tid // cutlass.Int32(32)

    sl      = seq_lens[b]
    max_col = scores.shape[1]

    # ── Shared memory ─────────────────────────────────────────────────────
    allocator      = cutlass.utils.SmemAllocator()
    smem_warp_bins = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS * NUM_BINS,), stride=(1,)), 4, None)
    smem_bins = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_BINS,), stride=(1,)), 4, None)
    smem_warp_tot = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_round = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)

    # ── Phase 1: 8-pass radix select (4-bit digits) ──────────────────────
    desired      = cutlass.Uint32(0)
    desired_mask = cutlass.Uint32(0)
    k_to_find    = cutlass.Int32(TOPK)

    for pass_idx in cutlass.range_constexpr(NUM_PASSES):
        digit_pos = (32 - RADIX_BITS) - pass_idx * RADIX_BITS  # 28,24,20,16,12,8,4,0

        if tid < cutlass.Int32(NUM_BINS):
            smem_bins[tid] = cutlass.Int32(0)
        cute.arch.sync_threads()

        # Per-thread register counters (16 bins)
        c0  = cutlass.Int32(0); c1  = cutlass.Int32(0)
        c2  = cutlass.Int32(0); c3  = cutlass.Int32(0)
        c4  = cutlass.Int32(0); c5  = cutlass.Int32(0)
        c6  = cutlass.Int32(0); c7  = cutlass.Int32(0)
        c8  = cutlass.Int32(0); c9  = cutlass.Int32(0)
        c10 = cutlass.Int32(0); c11 = cutlass.Int32(0)
        c12 = cutlass.Int32(0); c13 = cutlass.Int32(0)
        c14 = cutlass.Int32(0); c15 = cutlass.Int32(0)

        i = tid
        while i < sl:
            bits = float_to_radix(scores[b, i])
            if u_eq(u_and(bits, desired_mask), u_and(desired, desired_mask)):
                digit = u_and(u_shr(bits, cutlass.Uint32(digit_pos)), cutlass.Uint32(DIGIT_MASK))
                if u_eq(digit, cutlass.Uint32(0)):  c0  = c0  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(1)):  c1  = c1  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(2)):  c2  = c2  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(3)):  c3  = c3  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(4)):  c4  = c4  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(5)):  c5  = c5  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(6)):  c6  = c6  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(7)):  c7  = c7  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(8)):  c8  = c8  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(9)):  c9  = c9  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(10)): c10 = c10 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(11)): c11 = c11 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(12)): c12 = c12 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(13)): c13 = c13 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(14)): c14 = c14 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(15)): c15 = c15 + cutlass.Int32(1)
            i = i + cutlass.Int32(NUM_THREADS)

        # Warp-reduce each bin
        c0  = warp_sum_i32(c0);  c1  = warp_sum_i32(c1)
        c2  = warp_sum_i32(c2);  c3  = warp_sum_i32(c3)
        c4  = warp_sum_i32(c4);  c5  = warp_sum_i32(c5)
        c6  = warp_sum_i32(c6);  c7  = warp_sum_i32(c7)
        c8  = warp_sum_i32(c8);  c9  = warp_sum_i32(c9)
        c10 = warp_sum_i32(c10); c11 = warp_sum_i32(c11)
        c12 = warp_sum_i32(c12); c13 = warp_sum_i32(c13)
        c14 = warp_sum_i32(c14); c15 = warp_sum_i32(c15)

        if lane == cutlass.Int32(0):
            base = warp * cutlass.Int32(NUM_BINS)
            smem_warp_bins[base + 0]  = c0;  smem_warp_bins[base + 1]  = c1
            smem_warp_bins[base + 2]  = c2;  smem_warp_bins[base + 3]  = c3
            smem_warp_bins[base + 4]  = c4;  smem_warp_bins[base + 5]  = c5
            smem_warp_bins[base + 6]  = c6;  smem_warp_bins[base + 7]  = c7
            smem_warp_bins[base + 8]  = c8;  smem_warp_bins[base + 9]  = c9
            smem_warp_bins[base + 10] = c10; smem_warp_bins[base + 11] = c11
            smem_warp_bins[base + 12] = c12; smem_warp_bins[base + 13] = c13
            smem_warp_bins[base + 14] = c14; smem_warp_bins[base + 15] = c15
        cute.arch.sync_threads()

        # Warp 0: cross-warp reduction for each bin
        if warp == cutlass.Int32(0):
            for bin_k in cutlass.range_constexpr(NUM_BINS):
                val = smem_warp_bins[lane * cutlass.Int32(NUM_BINS) + cutlass.Int32(bin_k)]
                val = warp_sum_i32(val)
                if lane == cutlass.Int32(0):
                    smem_bins[bin_k] = val
        cute.arch.sync_threads()

        # Read global bin counts
        gvals = [smem_bins[k] for k in range(NUM_BINS)]
        cute.arch.sync_threads()

        # Greedy descent: from bin 15 down to bin 0
        dp_u    = cutlass.Uint32(digit_pos)
        shifted = u_shl(cutlass.Uint32(DIGIT_MASK), dp_u)
        inv_sh  = u_xor(shifted, cutlass.Uint32(0xFFFFFFFF))

        found = cutlass.Int32(0)
        for bin_k in cutlass.range_constexpr(NUM_BINS):
            actual_bin = NUM_BINS - 1 - bin_k
            gval = gvals[actual_bin]
            if found == cutlass.Int32(0):
                if gval >= k_to_find:
                    desired      = u_or(u_and(desired, inv_sh), u_shl(cutlass.Uint32(actual_bin), dp_u))
                    desired_mask = u_or(desired_mask, shifted)
                    found = cutlass.Int32(1)
                else:
                    k_to_find = k_to_find - gval

    kth_bits = desired

    # ── Phase 2a: gather elements strictly > kth ─────────────────────────
    write_cursor = cutlass.Int32(0)

    col = cutlass.Int32(0)
    while col < max_col:
        cur_col  = col + tid
        is_valid = cur_col < sl
        bits     = cutlass.Uint32(0)
        if is_valid:
            bits = float_to_radix(scores[b, cur_col])
        is_b = cutlass.Int32(0)
        if is_valid:
            if u_gt(bits, kth_bits):
                is_b = cutlass.Int32(1)

        scan_val = is_b
        for s in cutlass.range_constexpr(5):
            peer = cute.arch.shuffle_sync_up(scan_val, 1 << s, mask_and_clamp=0)
            if lane >= cutlass.Int32(1 << s):
                scan_val = scan_val + peer
        my_excl  = scan_val - is_b
        warp_tot = cute.arch.shuffle_sync(scan_val, 31)

        if lane == cutlass.Int32(31):
            smem_warp_tot[warp] = warp_tot
        cute.arch.sync_threads()

        if warp == cutlass.Int32(0):
            wt      = smem_warp_tot[lane]
            orig_wt = wt
            for s in cutlass.range_constexpr(5):
                p2 = cute.arch.shuffle_sync_up(wt, 1 << s, mask_and_clamp=0)
                if lane >= cutlass.Int32(1 << s):
                    wt = wt + p2
            smem_warp_tot[lane] = wt - orig_wt
            round_tot = warp_sum_i32(orig_wt)
            if lane == cutlass.Int32(0):
                smem_round[0] = round_tot
        cute.arch.sync_threads()

        warp_off = smem_warp_tot[warp]
        goff     = write_cursor + warp_off + my_excl

        if is_b > cutlass.Int32(0):
            if goff < cutlass.Int32(TOPK):
                out_idx[b, goff] = cur_col

        round_tot    = smem_round[0]
        cute.arch.sync_threads()
        write_cursor = write_cursor + round_tot

        col = col + cutlass.Int32(NUM_THREADS)

    # ── Phase 2b: fill exact ties up to TOPK ─────────────────────────────
    tie_cursor = cutlass.Int32(0)

    col2 = cutlass.Int32(0)
    while col2 < max_col:
        cur_col  = col2 + tid
        is_valid = cur_col < sl
        bits     = cutlass.Uint32(0)
        if is_valid:
            bits = float_to_radix(scores[b, cur_col])
        is_t = cutlass.Int32(0)
        if is_valid:
            if u_eq(bits, kth_bits):
                is_t = cutlass.Int32(1)

        scan_val = is_t
        for s in cutlass.range_constexpr(5):
            peer = cute.arch.shuffle_sync_up(scan_val, 1 << s, mask_and_clamp=0)
            if lane >= cutlass.Int32(1 << s):
                scan_val = scan_val + peer
        my_excl  = scan_val - is_t
        warp_tot = cute.arch.shuffle_sync(scan_val, 31)

        if lane == cutlass.Int32(31):
            smem_warp_tot[warp] = warp_tot
        cute.arch.sync_threads()

        if warp == cutlass.Int32(0):
            wt      = smem_warp_tot[lane]
            orig_wt = wt
            for s in cutlass.range_constexpr(5):
                p2 = cute.arch.shuffle_sync_up(wt, 1 << s, mask_and_clamp=0)
                if lane >= cutlass.Int32(1 << s):
                    wt = wt + p2
            smem_warp_tot[lane] = wt - orig_wt
            round_tot = warp_sum_i32(orig_wt)
            if lane == cutlass.Int32(0):
                smem_round[0] = round_tot
        cute.arch.sync_threads()

        warp_off = smem_warp_tot[warp]
        toff     = tie_cursor + warp_off + my_excl
        wrt_pos  = write_cursor + toff
        need     = cutlass.Int32(TOPK) - write_cursor

        if is_t > cutlass.Int32(0):
            if toff < need:
                if wrt_pos < cutlass.Int32(TOPK):
                    out_idx[b, wrt_pos] = cur_col

        round_tot  = smem_round[0]
        cute.arch.sync_threads()
        tie_cursor = tie_cursor + round_tot
        col2       = col2 + cutlass.Int32(NUM_THREADS)


# ────────────────────────────────────────────────────────────────────────────
# JIT wrapper
# ────────────────────────────────────────────────────────────────────────────

@cute.jit
def topk_radix_cutedsl(scores: cute.Tensor, out_idx: cute.Tensor, seq_lens: cute.Tensor):
    B = scores.shape[0]
    topk_radix_kernel(scores, out_idx, seq_lens).launch(
        grid=[B, 1, 1],
        block=[NUM_THREADS, 1, 1],
    )


# ────────────────────────────────────────────────────────────────────────────
# Compile + test
# ────────────────────────────────────────────────────────────────────────────

def make_fakes():
    return (
        make_fake_compact_tensor(dtype=cute.Float32, shape=(cute.sym_int(), cute.sym_int()),
                                 stride_order=(1, 0), assumed_align=16),
        make_fake_compact_tensor(dtype=cute.Int32,   shape=(cute.sym_int(), cute.sym_int()),
                                 stride_order=(1, 0), assumed_align=16),
        make_fake_compact_tensor(dtype=cute.Int32,   shape=(cute.sym_int(),),
                                 stride_order=(0,),   assumed_align=4),
    )


def topk_radix(scores: torch.Tensor, seq_lens: torch.Tensor, topk: int = TOPK) -> torch.Tensor:
    B = scores.shape[0]
    out = torch.full((B, topk), -1, dtype=torch.int32, device=scores.device)
    compiled = cute.compile(topk_radix_cutedsl, *make_fakes())
    compiled(scores, out, seq_lens)
    return out


def test_correctness():
    torch.manual_seed(0)
    device = torch.device("cuda")

    print("Compiling CuTe DSL sbtopk v2 (RADIX_BITS=4) kernel...")
    compiled = cute.compile(topk_radix_cutedsl, *make_fakes())
    print("Done.\n")

    print("=== Correctness test (v2: RADIX_BITS=4) ===")
    for B, max_sl in [(1, 4096), (4, 6000), (8, 3000), (1, 2049), (2, 5806)]:
        scores = torch.full((B, max_sl), float("-inf"), dtype=torch.float32, device=device)
        sl_list = [max(TOPK + 1, max_sl - i * (max_sl // max(B, 2))) for i in range(B)]
        seq_lens = torch.tensor(sl_list, dtype=torch.int32, device=device)
        for b, sl in enumerate(sl_list):
            scores[b, :sl] = torch.randn(sl, device=device)

        ref_idx = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
        for b, sl in enumerate(sl_list):
            k = min(TOPK, sl)
            _, idx = torch.topk(scores[b, :sl], k)
            ref_idx[b, :k] = idx.int()

        out = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
        compiled(scores, out, seq_lens)
        torch.cuda.synchronize()

        ok, miss = check_topk_indices(ref_idx, out, seq_lens)
        print(f"  B={B:2d}  max_sl={max_sl:5d}  worst_miss={miss:.6f}  [{'PASS' if ok else 'FAIL'}]")


def benchmark_vs_torch():
    import time
    torch.manual_seed(1)
    device = torch.device("cuda")

    B, max_sl = 8, 6000
    scores   = torch.randn(B, max_sl, device=device, dtype=torch.float32)
    seq_lens = torch.full((B,), max_sl, dtype=torch.int32, device=device)
    out      = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)

    compiled = cute.compile(topk_radix_cutedsl, *make_fakes())

    for _ in range(20):
        compiled(scores, out, seq_lens)
        torch.topk(scores, TOPK, dim=1)
    torch.cuda.synchronize()

    args   = JitArguments(scores, out, seq_lens)
    our_us = benchmark(compiled, kernel_arguments=args)

    t0 = time.perf_counter()
    N = 500
    for _ in range(N):
        torch.topk(scores, TOPK, dim=1)
    torch.cuda.synchronize()
    ref_us = (time.perf_counter() - t0) / N * 1e6

    print(f"\n=== Benchmark v2 RADIX_BITS=4 (B={B}, max_sl={max_sl}) ===")
    print(f"  cutedsl sbtopk v2: {our_us:.1f} µs")
    print(f"  torch.topk       : {ref_us:.1f} µs")
    print(f"  speedup          : {ref_us / our_us:.2f}x")


if __name__ == "__main__":
    test_correctness()
    benchmark_vs_torch()

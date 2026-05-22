"""
Exact top-K via CuTe DSL — sbtopk radix select (RADIX_BITS=2, 16 passes).

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

# ────────────────────────────────────────────────────────────────────────────
# PTX helpers (Uint32 bit-pattern operations)
# ────────────────────────────────────────────────────────────────────────────

@dsl_user_op
def float_to_radix(v: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    """Order-preserving float32→uint32 for radix sort."""
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
    """Butterfly warp-reduce sum of Int32. All 32 lanes get the block sum."""
    for i in cutlass.range_constexpr(5):
        peer = cute.arch.shuffle_sync_bfly(val, 1 << i)
        val = val + peer
    return val


# ────────────────────────────────────────────────────────────────────────────
# Main kernel
# ────────────────────────────────────────────────────────────────────────────

@cute.kernel
def topk_radix_kernel(
    scores:   cute.Tensor,   # [B, max_sl]  Float32
    out_idx:  cute.Tensor,   # [B, TOPK]    Int32
    seq_lens: cute.Tensor,   # [B]          Int32
):
    b   = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    lane = tid % cutlass.Int32(32)
    warp = tid // cutlass.Int32(32)

    sl      = seq_lens[b]
    max_col = scores.shape[1]

    # ── Shared memory (all Int32) ─────────────────────────────────────────
    allocator      = cutlass.utils.SmemAllocator()
    smem_warp_bins = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS * 4,), stride=(1,)), 4, None)
    smem_bins = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((4,), stride=(1,)), 4, None)
    smem_warp_tot = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_round = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)

    # ── Phase 1: 16-pass radix select ────────────────────────────────────
    desired      = cutlass.Uint32(0)
    desired_mask = cutlass.Uint32(0)
    k_to_find    = cutlass.Int32(TOPK)

    for pass_idx in cutlass.range_constexpr(16):
        digit_pos = 30 - pass_idx * 2   # Python int, constexpr per unrolled iter

        if tid < cutlass.Int32(4):
            smem_bins[tid] = cutlass.Int32(0)
        cute.arch.sync_threads()

        c0 = cutlass.Int32(0); c1 = cutlass.Int32(0)
        c2 = cutlass.Int32(0); c3 = cutlass.Int32(0)

        i = tid
        while i < sl:
            bits = float_to_radix(scores[b, i])
            if u_eq(u_and(bits, desired_mask), u_and(desired, desired_mask)):
                digit = u_and(u_shr(bits, cutlass.Uint32(digit_pos)), cutlass.Uint32(3))
                if u_eq(digit, cutlass.Uint32(0)):
                    c0 = c0 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(1)):
                    c1 = c1 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(2)):
                    c2 = c2 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(3)):
                    c3 = c3 + cutlass.Int32(1)
            i = i + cutlass.Int32(NUM_THREADS)

        c0 = warp_sum_i32(c0); c1 = warp_sum_i32(c1)
        c2 = warp_sum_i32(c2); c3 = warp_sum_i32(c3)

        if lane == cutlass.Int32(0):
            smem_warp_bins[warp * cutlass.Int32(4) + 0] = c0
            smem_warp_bins[warp * cutlass.Int32(4) + 1] = c1
            smem_warp_bins[warp * cutlass.Int32(4) + 2] = c2
            smem_warp_bins[warp * cutlass.Int32(4) + 3] = c3
        cute.arch.sync_threads()

        if warp == cutlass.Int32(0):
            g0 = smem_warp_bins[lane * cutlass.Int32(4) + 0]
            g1 = smem_warp_bins[lane * cutlass.Int32(4) + 1]
            g2 = smem_warp_bins[lane * cutlass.Int32(4) + 2]
            g3 = smem_warp_bins[lane * cutlass.Int32(4) + 3]
            g0 = warp_sum_i32(g0); g1 = warp_sum_i32(g1)
            g2 = warp_sum_i32(g2); g3 = warp_sum_i32(g3)
            if lane == cutlass.Int32(0):
                smem_bins[0] = g0; smem_bins[1] = g1
                smem_bins[2] = g2; smem_bins[3] = g3
        cute.arch.sync_threads()

        g0 = smem_bins[0]; g1 = smem_bins[1]
        g2 = smem_bins[2]; g3 = smem_bins[3]
        cute.arch.sync_threads()

        # Greedy descent: Int32 comparisons, Uint32 bit ops
        dp_u    = cutlass.Uint32(digit_pos)
        shifted = u_shl(cutlass.Uint32(3), dp_u)
        inv_sh  = u_xor(shifted, cutlass.Uint32(0xFFFFFFFF))

        if g3 >= k_to_find:
            desired      = u_or(u_and(desired, inv_sh), u_shl(cutlass.Uint32(3), dp_u))
            desired_mask = u_or(desired_mask, shifted)
        else:
            k_to_find = k_to_find - g3
            if g2 >= k_to_find:
                desired      = u_or(u_and(desired, inv_sh), u_shl(cutlass.Uint32(2), dp_u))
                desired_mask = u_or(desired_mask, shifted)
            else:
                k_to_find = k_to_find - g2
                if g1 >= k_to_find:
                    desired      = u_or(u_and(desired, inv_sh), u_shl(cutlass.Uint32(1), dp_u))
                    desired_mask = u_or(desired_mask, shifted)
                else:
                    k_to_find = k_to_find - g1
                    desired      = u_and(desired, inv_sh)
                    desired_mask = u_or(desired_mask, shifted)

    kth_bits = desired  # Uint32

    # ── Phase 2a: gather elements strictly > kth ─────────────────────────
    write_cursor = cutlass.Int32(0)   # loop-carried

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

        # Kogge-Stone inclusive prefix sum within warp
        scan_val = is_b
        for s in cutlass.range_constexpr(5):
            peer = cute.arch.shuffle_sync_up(scan_val, 1 << s, mask_and_clamp=0)
            if lane >= cutlass.Int32(1 << s):
                scan_val = scan_val + peer
        my_excl  = scan_val - is_b
        warp_tot = cute.arch.shuffle_sync(scan_val, 31)  # lane 31 total → all

        if lane == cutlass.Int32(31):
            smem_warp_tot[warp] = warp_tot
        cute.arch.sync_threads()

        # Warp 0: exclusive prefix of warp totals + block-wide total
        if warp == cutlass.Int32(0):
            wt      = smem_warp_tot[lane]
            orig_wt = wt
            for s in cutlass.range_constexpr(5):
                p2 = cute.arch.shuffle_sync_up(wt, 1 << s, mask_and_clamp=0)
                if lane >= cutlass.Int32(1 << s):
                    wt = wt + p2
            smem_warp_tot[lane] = wt - orig_wt           # exclusive prefix
            round_tot = warp_sum_i32(orig_wt)            # block-wide total
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
    tie_cursor = cutlass.Int32(0)   # loop-carried

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
    """Drop-in replacement for torch.topk that returns sorted indices."""
    B = scores.shape[0]
    out = torch.full((B, topk), -1, dtype=torch.int32, device=scores.device)
    compiled = cute.compile(topk_radix_cutedsl, *make_fakes())
    compiled(scores, out, seq_lens)
    return out


def test_correctness():
    torch.manual_seed(0)
    device = torch.device("cuda")

    print("Compiling CuTe DSL sbtopk kernel...")
    compiled = cute.compile(topk_radix_cutedsl, *make_fakes())
    print("Done.\n")

    print("=== Correctness test ===")
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

    print(f"\n=== Benchmark (B={B}, max_sl={max_sl}) ===")
    print(f"  cutedsl sbtopk : {our_us:.1f} µs")
    print(f"  torch.topk     : {ref_us:.1f} µs")
    print(f"  speedup        : {ref_us / our_us:.2f}x")


if __name__ == "__main__":
    test_correctness()
    benchmark_vs_torch()

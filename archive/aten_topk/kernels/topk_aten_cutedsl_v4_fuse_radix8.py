"""
Exact top-K via CuTe DSL — sbtopk radix select (RADIX_BITS=4, 8 passes).
v4_fuse_radix8: combines both optimizations from v4_radix8 and v4_fuse:
  - 4 bits per pass → 8 passes, 16 bins (half the sequential barriers of v4)
  - Fused Phase 2: single scatter sweep using above_total = TOPK - k_to_find

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
VEC         = 4   # elements per thread per iteration
RADIX_BITS  = 4   # bits per pass
NUM_PASSES  = 32 // RADIX_BITS   # 8
NUM_BINS    = 1 << RADIX_BITS    # 16

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
# Main kernel — RADIX_BITS=4, 8 passes, fused Phase 2
# ────────────────────────────────────────────────────────────────────────────

@cute.kernel
def topk_radix_kernel(
    scores:   cute.Tensor,
    out_idx:  cute.Tensor,
    seq_lens: cute.Tensor,
):
    b    = cute.arch.block_idx()[0]
    tid  = cute.arch.thread_idx()[0]
    lane = tid % cutlass.Int32(32)
    warp = tid // cutlass.Int32(32)

    sl      = seq_lens[b]
    max_col = scores.shape[1]

    # ── Shared memory ─────────────────────────────────────────────────────
    allocator      = cutlass.utils.SmemAllocator()
    # Phase 1: 16 bins × NUM_WARPS partial counts + 16 global bins
    smem_warp_bins = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS * NUM_BINS,), stride=(1,)), 4, None)
    smem_bins = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_BINS,), stride=(1,)), 4, None)
    # Phase 2 (fused): per-warp above/tie totals + per-round totals
    smem_warp_above  = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_warp_tie    = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_above_round = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)
    smem_tie_round   = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)

    # ── Phase 1: 8-pass radix select (4 bits/pass, 16 bins) ──────────────
    desired      = cutlass.Uint32(0)
    desired_mask = cutlass.Uint32(0)
    k_to_find    = cutlass.Int32(TOPK)

    for pass_idx in cutlass.range_constexpr(NUM_PASSES):
        digit_pos   = 32 - (pass_idx + 1) * RADIX_BITS   # 28, 24, 20, ..., 0
        digit_pos_u  = cutlass.Uint32(digit_pos)
        digit_mask_u = cutlass.Uint32(NUM_BINS - 1)        # 0xF

        # Zero 16 global bins
        if tid < cutlass.Int32(NUM_BINS):
            smem_bins[tid] = cutlass.Int32(0)
        cute.arch.sync_threads()

        # Per-thread counts — 16 scalars
        cnt0  = cutlass.Int32(0); cnt1  = cutlass.Int32(0)
        cnt2  = cutlass.Int32(0); cnt3  = cutlass.Int32(0)
        cnt4  = cutlass.Int32(0); cnt5  = cutlass.Int32(0)
        cnt6  = cutlass.Int32(0); cnt7  = cutlass.Int32(0)
        cnt8  = cutlass.Int32(0); cnt9  = cutlass.Int32(0)
        cnt10 = cutlass.Int32(0); cnt11 = cutlass.Int32(0)
        cnt12 = cutlass.Int32(0); cnt13 = cutlass.Int32(0)
        cnt14 = cutlass.Int32(0); cnt15 = cutlass.Int32(0)

        # Vectorized loop (VEC=4 consecutive elements per thread)
        base = tid * cutlass.Int32(VEC)
        while base + cutlass.Int32(VEC - 1) < sl:
            for vi in cutlass.range_constexpr(VEC):
                bits = float_to_radix(scores[b, base + cutlass.Int32(vi)])
                if u_eq(u_and(bits, desired_mask), u_and(desired, desired_mask)):
                    digit = u_and(u_shr(bits, digit_pos_u), digit_mask_u)
                    if u_eq(digit, cutlass.Uint32( 0)): cnt0  = cnt0  + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32( 1)): cnt1  = cnt1  + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32( 2)): cnt2  = cnt2  + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32( 3)): cnt3  = cnt3  + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32( 4)): cnt4  = cnt4  + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32( 5)): cnt5  = cnt5  + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32( 6)): cnt6  = cnt6  + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32( 7)): cnt7  = cnt7  + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32( 8)): cnt8  = cnt8  + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32( 9)): cnt9  = cnt9  + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32(10)): cnt10 = cnt10 + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32(11)): cnt11 = cnt11 + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32(12)): cnt12 = cnt12 + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32(13)): cnt13 = cnt13 + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32(14)): cnt14 = cnt14 + cutlass.Int32(1)
                    if u_eq(digit, cutlass.Uint32(15)): cnt15 = cnt15 + cutlass.Int32(1)
            base = base + cutlass.Int32(NUM_THREADS * VEC)

        # Scalar tail
        while base < sl:
            bits = float_to_radix(scores[b, base])
            if u_eq(u_and(bits, desired_mask), u_and(desired, desired_mask)):
                digit = u_and(u_shr(bits, digit_pos_u), digit_mask_u)
                if u_eq(digit, cutlass.Uint32( 0)): cnt0  = cnt0  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32( 1)): cnt1  = cnt1  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32( 2)): cnt2  = cnt2  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32( 3)): cnt3  = cnt3  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32( 4)): cnt4  = cnt4  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32( 5)): cnt5  = cnt5  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32( 6)): cnt6  = cnt6  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32( 7)): cnt7  = cnt7  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32( 8)): cnt8  = cnt8  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32( 9)): cnt9  = cnt9  + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(10)): cnt10 = cnt10 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(11)): cnt11 = cnt11 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(12)): cnt12 = cnt12 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(13)): cnt13 = cnt13 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(14)): cnt14 = cnt14 + cutlass.Int32(1)
                if u_eq(digit, cutlass.Uint32(15)): cnt15 = cnt15 + cutlass.Int32(1)
            base = base + cutlass.Int32(1)

        # Warp-reduce each bin
        cnt0  = warp_sum_i32(cnt0);  cnt1  = warp_sum_i32(cnt1)
        cnt2  = warp_sum_i32(cnt2);  cnt3  = warp_sum_i32(cnt3)
        cnt4  = warp_sum_i32(cnt4);  cnt5  = warp_sum_i32(cnt5)
        cnt6  = warp_sum_i32(cnt6);  cnt7  = warp_sum_i32(cnt7)
        cnt8  = warp_sum_i32(cnt8);  cnt9  = warp_sum_i32(cnt9)
        cnt10 = warp_sum_i32(cnt10); cnt11 = warp_sum_i32(cnt11)
        cnt12 = warp_sum_i32(cnt12); cnt13 = warp_sum_i32(cnt13)
        cnt14 = warp_sum_i32(cnt14); cnt15 = warp_sum_i32(cnt15)

        if lane == cutlass.Int32(0):
            wb = warp * cutlass.Int32(NUM_BINS)
            smem_warp_bins[wb +  0] = cnt0;  smem_warp_bins[wb +  1] = cnt1
            smem_warp_bins[wb +  2] = cnt2;  smem_warp_bins[wb +  3] = cnt3
            smem_warp_bins[wb +  4] = cnt4;  smem_warp_bins[wb +  5] = cnt5
            smem_warp_bins[wb +  6] = cnt6;  smem_warp_bins[wb +  7] = cnt7
            smem_warp_bins[wb +  8] = cnt8;  smem_warp_bins[wb +  9] = cnt9
            smem_warp_bins[wb + 10] = cnt10; smem_warp_bins[wb + 11] = cnt11
            smem_warp_bins[wb + 12] = cnt12; smem_warp_bins[wb + 13] = cnt13
            smem_warp_bins[wb + 14] = cnt14; smem_warp_bins[wb + 15] = cnt15
        cute.arch.sync_threads()

        # Warp 0 reduces: lane = bin index (16 bins, 32 lanes → fits)
        if warp == cutlass.Int32(0):
            if lane < cutlass.Int32(NUM_BINS):
                acc = cutlass.Int32(0)
                for wi in cutlass.range_constexpr(NUM_WARPS):
                    acc = acc + smem_warp_bins[cutlass.Int32(wi) * cutlass.Int32(NUM_BINS) + lane]
                smem_bins[lane] = acc
        cute.arch.sync_threads()

        # Greedy descent over bins 15 → 0
        dp_u    = cutlass.Uint32(digit_pos)
        shifted = u_shl(cutlass.Uint32(NUM_BINS - 1), dp_u)
        inv_sh  = u_xor(shifted, cutlass.Uint32(0xFFFFFFFF))

        g15 = smem_bins[15]; g14 = smem_bins[14]; g13 = smem_bins[13]; g12 = smem_bins[12]
        g11 = smem_bins[11]; g10 = smem_bins[10]; g9  = smem_bins[ 9]; g8  = smem_bins[ 8]
        g7  = smem_bins[ 7]; g6  = smem_bins[ 6]; g5  = smem_bins[ 5]; g4  = smem_bins[ 4]
        g3  = smem_bins[ 3]; g2  = smem_bins[ 2]; g1  = smem_bins[ 1]; g0  = smem_bins[ 0]
        cute.arch.sync_threads()

        found_digit = cutlass.Uint32(0)

        if g15 >= k_to_find:
            found_digit = cutlass.Uint32(15)
        else:
            k_to_find = k_to_find - g15
            if g14 >= k_to_find:
                found_digit = cutlass.Uint32(14)
            else:
                k_to_find = k_to_find - g14
                if g13 >= k_to_find:
                    found_digit = cutlass.Uint32(13)
                else:
                    k_to_find = k_to_find - g13
                    if g12 >= k_to_find:
                        found_digit = cutlass.Uint32(12)
                    else:
                        k_to_find = k_to_find - g12
                        if g11 >= k_to_find:
                            found_digit = cutlass.Uint32(11)
                        else:
                            k_to_find = k_to_find - g11
                            if g10 >= k_to_find:
                                found_digit = cutlass.Uint32(10)
                            else:
                                k_to_find = k_to_find - g10
                                if g9 >= k_to_find:
                                    found_digit = cutlass.Uint32(9)
                                else:
                                    k_to_find = k_to_find - g9
                                    if g8 >= k_to_find:
                                        found_digit = cutlass.Uint32(8)
                                    else:
                                        k_to_find = k_to_find - g8
                                        if g7 >= k_to_find:
                                            found_digit = cutlass.Uint32(7)
                                        else:
                                            k_to_find = k_to_find - g7
                                            if g6 >= k_to_find:
                                                found_digit = cutlass.Uint32(6)
                                            else:
                                                k_to_find = k_to_find - g6
                                                if g5 >= k_to_find:
                                                    found_digit = cutlass.Uint32(5)
                                                else:
                                                    k_to_find = k_to_find - g5
                                                    if g4 >= k_to_find:
                                                        found_digit = cutlass.Uint32(4)
                                                    else:
                                                        k_to_find = k_to_find - g4
                                                        if g3 >= k_to_find:
                                                            found_digit = cutlass.Uint32(3)
                                                        else:
                                                            k_to_find = k_to_find - g3
                                                            if g2 >= k_to_find:
                                                                found_digit = cutlass.Uint32(2)
                                                            else:
                                                                k_to_find = k_to_find - g2
                                                                if g1 >= k_to_find:
                                                                    found_digit = cutlass.Uint32(1)
                                                                else:
                                                                    k_to_find = k_to_find - g1
                                                                    found_digit = cutlass.Uint32(0)

        desired      = u_or(u_and(desired, inv_sh), u_shl(found_digit, dp_u))
        desired_mask = u_or(desired_mask, shifted)

    kth_bits = desired

    # above_total: elements strictly above kth (uniform across all threads)
    above_total = cutlass.Int32(TOPK) - k_to_find
    need_ties   = k_to_find

    # ── Phase 2 (fused): single sweep — scatter above AND tie together ────
    above_cursor = cutlass.Int32(0)
    tie_cursor   = cutlass.Int32(0)

    col = cutlass.Int32(0)
    while col < max_col:
        cur_col  = col + tid
        is_valid = cur_col < sl

        bits = cutlass.Uint32(0)
        if is_valid:
            bits = float_to_radix(scores[b, cur_col])

        is_b = cutlass.Int32(0)
        is_t = cutlass.Int32(0)
        if is_valid:
            if u_gt(bits, kth_bits):
                is_b = cutlass.Int32(1)
            if u_eq(bits, kth_bits):
                is_t = cutlass.Int32(1)

        # Warp prefix scan — above
        scan_b = is_b
        for s in cutlass.range_constexpr(5):
            peer = cute.arch.shuffle_sync_up(scan_b, 1 << s, mask_and_clamp=0)
            if lane >= cutlass.Int32(1 << s):
                scan_b = scan_b + peer
        my_b_excl  = scan_b - is_b
        warp_b_tot = cute.arch.shuffle_sync(scan_b, 31)

        # Warp prefix scan — tie
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

        # Warp 0: build block-level exclusive prefix for both above and tie
        if warp == cutlass.Int32(0):
            wta      = smem_warp_above[lane]
            orig_wta = wta
            for s in cutlass.range_constexpr(5):
                p = cute.arch.shuffle_sync_up(wta, 1 << s, mask_and_clamp=0)
                if lane >= cutlass.Int32(1 << s):
                    wta = wta + p
            smem_warp_above[lane] = wta - orig_wta
            above_round_tot = warp_sum_i32(orig_wta)
            if lane == cutlass.Int32(0):
                smem_above_round[0] = above_round_tot

            wtt      = smem_warp_tie[lane]
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

    print("Compiling CuTe DSL sbtopk v4_fuse_radix8 (4-bit/pass, 8 passes, fused Phase 2)...")
    compiled = cute.compile(topk_radix_cutedsl, *make_fakes())
    print("Done.\n")

    print("=== Correctness test (v4_fuse_radix8) ===")
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

    compiled = cute.compile(topk_radix_cutedsl, *make_fakes())

    configs = [
        (1, 2049), (1, 3000), (1, 4096), (1, 6000), (1, 8192), (1, 16384),
        (4, 3000), (4, 6000), (4, 8192), (4, 16384),
        (8, 3000), (8, 6000), (8, 8192), (8, 16384),
    ]

    print(f"\n{'B':>3s}  {'max_sl':>7s}  {'ours_µs':>8s}  {'torch_µs':>9s}  {'speedup':>7s}")
    print("-" * 42)

    for B, max_sl in configs:
        scores   = torch.randn(B, max_sl, device=device, dtype=torch.float32)
        seq_lens = torch.full((B,), max_sl, dtype=torch.int32, device=device)
        out      = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)

        for _ in range(10):
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

        print(f"{B:3d}  {max_sl:7d}  {our_us:8.1f}  {ref_us:9.1f}  {ref_us / our_us:7.2f}x")


if __name__ == "__main__":
    test_correctness()
    benchmark_vs_torch()

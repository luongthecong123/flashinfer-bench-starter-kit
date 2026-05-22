"""
Exact top-K via CuTe DSL — sbtopk radix select (RADIX_BITS=2, 16 passes).
v4_fuse_t512: same as v4_fuse but uses 512 threads instead of 1024.

Motivation: for seq_len ≤ 4096, 1024-thread launch wastes half its threads.
  - sl=2049 w/ 1024T: threads 512-1023 do at most 1 scalar element and wait
  - sl=2049 w/ 512T:  all 512 threads do exactly 1 VEC=4 iter → 100% util
  - warp0 reduction: 16 warp bins instead of 32 → fewer smem reads

Phase 2 has 2× more rounds but scatter cost is minor vs Phase 1.

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
NUM_THREADS = 512
NUM_WARPS   = NUM_THREADS // 32   # 16
VEC         = 4   # elements per thread per iteration

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
# Helper: count one element into bins
# ────────────────────────────────────────────────────────────────────────────

@cute.jit
def count_element(
    bits: cutlass.Uint32,
    desired: cutlass.Uint32, desired_mask: cutlass.Uint32,
    digit_pos_u: cutlass.Uint32,
    c0: cutlass.Int32, c1: cutlass.Int32, c2: cutlass.Int32, c3: cutlass.Int32,
) -> tuple:
    if u_eq(u_and(bits, desired_mask), u_and(desired, desired_mask)):
        digit = u_and(u_shr(bits, digit_pos_u), cutlass.Uint32(3))
        if u_eq(digit, cutlass.Uint32(0)):
            c0 = c0 + cutlass.Int32(1)
        if u_eq(digit, cutlass.Uint32(1)):
            c1 = c1 + cutlass.Int32(1)
        if u_eq(digit, cutlass.Uint32(2)):
            c2 = c2 + cutlass.Int32(1)
        if u_eq(digit, cutlass.Uint32(3)):
            c3 = c3 + cutlass.Int32(1)
    return c0, c1, c2, c3


# ────────────────────────────────────────────────────────────────────────────
# Main kernel — 512 threads, fused Phase 2
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
    # Only NUM_WARPS=16 warp slots needed
    smem_warp_bins = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS * 4,), stride=(1,)), 4, None)
    smem_bins = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((4,), stride=(1,)), 4, None)
    smem_warp_above  = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_warp_tie    = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_above_round = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)
    smem_tie_round   = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)

    # ── Phase 1: 16-pass radix select ────────────────────────────────────
    desired      = cutlass.Uint32(0)
    desired_mask = cutlass.Uint32(0)
    k_to_find    = cutlass.Int32(TOPK)

    for pass_idx in cutlass.range_constexpr(16):
        digit_pos = 30 - pass_idx * 2
        digit_pos_u = cutlass.Uint32(digit_pos)

        if tid < cutlass.Int32(4):
            smem_bins[tid] = cutlass.Int32(0)
        cute.arch.sync_threads()

        c0 = cutlass.Int32(0); c1 = cutlass.Int32(0)
        c2 = cutlass.Int32(0); c3 = cutlass.Int32(0)

        # Vectorized loop: 4 consecutive elements per thread
        base = tid * cutlass.Int32(VEC)
        while base + cutlass.Int32(VEC - 1) < sl:
            bits0 = float_to_radix(scores[b, base])
            bits1 = float_to_radix(scores[b, base + cutlass.Int32(1)])
            bits2 = float_to_radix(scores[b, base + cutlass.Int32(2)])
            bits3 = float_to_radix(scores[b, base + cutlass.Int32(3)])

            c0, c1, c2, c3 = count_element(bits0, desired, desired_mask, digit_pos_u, c0, c1, c2, c3)
            c0, c1, c2, c3 = count_element(bits1, desired, desired_mask, digit_pos_u, c0, c1, c2, c3)
            c0, c1, c2, c3 = count_element(bits2, desired, desired_mask, digit_pos_u, c0, c1, c2, c3)
            c0, c1, c2, c3 = count_element(bits3, desired, desired_mask, digit_pos_u, c0, c1, c2, c3)

            base = base + cutlass.Int32(NUM_THREADS * VEC)

        # Scalar tail
        while base < sl:
            bits = float_to_radix(scores[b, base])
            c0, c1, c2, c3 = count_element(bits, desired, desired_mask, digit_pos_u, c0, c1, c2, c3)
            base = base + cutlass.Int32(1)

        c0 = warp_sum_i32(c0); c1 = warp_sum_i32(c1)
        c2 = warp_sum_i32(c2); c3 = warp_sum_i32(c3)

        if lane == cutlass.Int32(0):
            smem_warp_bins[warp * cutlass.Int32(4) + 0] = c0
            smem_warp_bins[warp * cutlass.Int32(4) + 1] = c1
            smem_warp_bins[warp * cutlass.Int32(4) + 2] = c2
            smem_warp_bins[warp * cutlass.Int32(4) + 3] = c3
        cute.arch.sync_threads()

        # Warp 0 reduces: NUM_WARPS=16 warps, so lanes 0..15 read valid entries,
        # lanes 16..31 contribute 0 to the warp_sum.
        if warp == cutlass.Int32(0):
            g0 = cutlass.Int32(0); g1 = cutlass.Int32(0)
            g2 = cutlass.Int32(0); g3 = cutlass.Int32(0)
            if lane < cutlass.Int32(NUM_WARPS):
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

    kth_bits    = desired
    above_total = cutlass.Int32(TOPK) - k_to_find
    need_ties   = k_to_find

    # ── Phase 2 (fused): single sweep — scatter above AND tie ────────────
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

        # Warp 0: block-level exclusive prefix for above and tie.
        # NUM_WARPS=16: lanes 0..15 hold valid warp totals, lanes 16..31 contribute 0.
        if warp == cutlass.Int32(0):
            wta = cutlass.Int32(0)
            if lane < cutlass.Int32(NUM_WARPS):
                wta = smem_warp_above[lane]
            orig_wta = wta
            for s in cutlass.range_constexpr(5):
                p = cute.arch.shuffle_sync_up(wta, 1 << s, mask_and_clamp=0)
                if lane >= cutlass.Int32(1 << s):
                    wta = wta + p
            if lane < cutlass.Int32(NUM_WARPS):
                smem_warp_above[lane] = wta - orig_wta
            above_round_tot = warp_sum_i32(orig_wta)
            if lane == cutlass.Int32(0):
                smem_above_round[0] = above_round_tot

            wtt = cutlass.Int32(0)
            if lane < cutlass.Int32(NUM_WARPS):
                wtt = smem_warp_tie[lane]
            orig_wtt = wtt
            for s in cutlass.range_constexpr(5):
                p2 = cute.arch.shuffle_sync_up(wtt, 1 << s, mask_and_clamp=0)
                if lane >= cutlass.Int32(1 << s):
                    wtt = wtt + p2
            if lane < cutlass.Int32(NUM_WARPS):
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

    print("Compiling CuTe DSL sbtopk v4_fuse_t512 (512 threads, fused Phase 2)...")
    compiled = cute.compile(topk_radix_cutedsl, *make_fakes())
    print("Done.\n")

    print("=== Correctness test (v4_fuse_t512: 512 threads) ===")
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

    print(f"\n{'B':>3s}  {'max_sl':>7s}  {'t512_µs':>8s}  {'fuse_µs':>8s}  {'torch_µs':>9s}  {'vs_fuse':>7s}")
    print("-" * 52)

    # Reference numbers from v4_fuse run
    fuse_ref = {
        (1, 2049): 29.5, (1, 3000): 28.5, (1, 4096): 28.0, (1, 6000): 30.7,
        (1, 8192): 35.1, (1, 16384): 56.9,
        (4, 3000): 28.5, (4, 6000): 30.7, (4, 8192): 35.4, (4, 16384): 57.4,
        (8, 3000): 28.1, (8, 6000): 31.1, (8, 8192): 35.5, (8, 16384): 57.4,
    }

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

        fuse_us = fuse_ref.get((B, max_sl), 0.0)
        delta   = f"{fuse_us / our_us:+.2f}x" if fuse_us > 0 else "  n/a"
        print(f"{B:3d}  {max_sl:7d}  {our_us:8.1f}  {fuse_us:8.1f}  {ref_us:9.1f}  {delta:>7s}")


if __name__ == "__main__":
    test_correctness()
    benchmark_vs_torch()

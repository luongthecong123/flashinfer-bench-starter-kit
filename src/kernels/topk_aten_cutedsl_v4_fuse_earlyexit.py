"""
Exact top-K via CuTe DSL — v4_fuse + EARLY EXIT.

Same as v4_fuse (single fused scatter pass after radix Phase 1) but the
16-pass constexpr radix loop becomes a runtime while-loop that breaks as
soon as the chosen bin's count equals k_to_find exactly.  At that point
all elements sharing the high-bit prefix are tied for the kth position —
no further refinement is needed.

Phase 2 uses mask-aware comparisons (bits & desired_mask vs desired & desired_mask)
so partial masks from early-exit classify correctly.  When all 16 passes
run, desired_mask == 0xFFFFFFFF and the mask is a no-op.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm
from src.idx_utils import check_topk_indices

TOPK        = 2048
NUM_THREADS = 1024
NUM_WARPS   = NUM_THREADS // 32
VEC         = 4


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


@cute.jit
def warp_sum_i32(val: cutlass.Int32) -> cutlass.Int32:
    for i in cutlass.range_constexpr(5):
        peer = cute.arch.shuffle_sync_bfly(val, 1 << i)
        val = val + peer
    return val


@cute.jit
def count_element(bits, desired, desired_mask, digit_pos_u, c0, c1, c2, c3):
    if (bits & desired_mask) == (desired & desired_mask):
        digit = (bits >> digit_pos_u) & cutlass.Uint32(3)
        if digit == cutlass.Uint32(0):
            c0 = c0 + cutlass.Int32(1)
        if digit == cutlass.Uint32(1):
            c1 = c1 + cutlass.Int32(1)
        if digit == cutlass.Uint32(2):
            c2 = c2 + cutlass.Int32(1)
        if digit == cutlass.Uint32(3):
            c3 = c3 + cutlass.Int32(1)
    return c0, c1, c2, c3


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

    allocator = cutlass.utils.SmemAllocator()
    smem_warp_bins = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS * 4,), stride=(1,)), 4, None)
    smem_bins = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((4,), stride=(1,)), 4, None)
    smem_warp_above = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_warp_tie   = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None)
    smem_above_round = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)
    smem_tie_round   = allocator.allocate_tensor(
        cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4, None)

    # ── Phase 1: radix select with early exit ────────────────────────────
    desired      = cutlass.Uint32(0)
    desired_mask = cutlass.Uint32(0)
    k_to_find    = cutlass.Int32(TOPK)

    pass_idx = cutlass.Int32(0)
    while pass_idx < cutlass.Int32(16):
        digit_pos   = cutlass.Int32(30) - pass_idx * cutlass.Int32(2)
        digit_pos_u = cutlass.Uint32(digit_pos)

        if tid < cutlass.Int32(4):
            smem_bins[tid] = cutlass.Int32(0)
        cute.arch.sync_threads()

        c0 = cutlass.Int32(0); c1 = cutlass.Int32(0)
        c2 = cutlass.Int32(0); c3 = cutlass.Int32(0)

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

        dp_u    = cutlass.Uint32(digit_pos)
        shifted = cutlass.Uint32(3) << dp_u
        inv_sh  = shifted ^ cutlass.Uint32(0xFFFFFFFF)

        chosen_count = cutlass.Int32(0)
        if g3 >= k_to_find:
            desired      = (desired & inv_sh) | (cutlass.Uint32(3) << dp_u)
            desired_mask = desired_mask | shifted
            chosen_count = g3
        else:
            k_to_find = k_to_find - g3
            if g2 >= k_to_find:
                desired      = (desired & inv_sh) | (cutlass.Uint32(2) << dp_u)
                desired_mask = desired_mask | shifted
                chosen_count = g2
            else:
                k_to_find = k_to_find - g2
                if g1 >= k_to_find:
                    desired      = (desired & inv_sh) | (cutlass.Uint32(1) << dp_u)
                    desired_mask = desired_mask | shifted
                    chosen_count = g1
                else:
                    k_to_find = k_to_find - g1
                    desired      = desired & inv_sh
                    desired_mask = desired_mask | shifted
                    chosen_count = g0

        if chosen_count == k_to_find:
            pass_idx = cutlass.Int32(16)
        else:
            pass_idx = pass_idx + cutlass.Int32(1)

    above_total = cutlass.Int32(TOPK) - k_to_find
    need_ties   = k_to_find
    desired_pin = desired & desired_mask

    # ── Phase 2 (fused, mask-aware): single scatter pass ─────────────────
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


@cute.jit
def topk_radix_cutedsl(scores: cute.Tensor, out_idx: cute.Tensor, seq_lens: cute.Tensor):
    B = scores.shape[0]
    topk_radix_kernel(scores, out_idx, seq_lens).launch(
        grid=[B, 1, 1],
        block=[NUM_THREADS, 1, 1],
    )


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

    print("Compiling CuTe DSL sbtopk v4_fuse_earlyexit kernel...")
    compiled = cute.compile(topk_radix_cutedsl, *make_fakes())
    print("Done.\n")

    print("=== Correctness test (v4_fuse_earlyexit) ===")
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


if __name__ == "__main__":
    test_correctness()

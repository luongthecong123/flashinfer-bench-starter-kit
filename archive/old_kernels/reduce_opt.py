"""
reduce_opt.py — Progressive optimisation of the FlashInfer split-K reduce kernel.

Run standalone (locally) or via modal for B200 benchmarking.

Architecture: Blackwell SM100a only.
Kernel configuration: BLOCK_SIZE_REDUCE=512, head_dim_ckv=512, num_splits=8.

-----------------------------------------------------------------------
Current baseline (v0) issues
-----------------------------------------------------------------------
  ISSUE 1  Thread 0 does two serial passes over num_splits=8 to compute
           global_max and global_denom.  511 threads sit idle.
           Fix: assign one thread per split (8 threads), warp-reduce the
           results.  Eliminates 2 × 7 = 14 serial stall iterations.

  ISSUE 2  The output loop runs in 512 threads, each computing
               scale[s] = exp(local_max[s] - g_max) / g_denom
           for s in range(8).  That is 512 × 8 = 4096 exp() calls, but
           there are only 8 UNIQUE scale values.
           Fix: precompute smem_scales[8] in 8 threads (warp 0), broadcast
           to all 512 output threads via smem.

  ISSUE 3  The sentinel is broadcast through smem (1 float + sync_threads
           just to let every thread read a single value written by thread 0).
           Fix: all threads read partial_lse[bidx, bidy, 0, 0] directly.
           The value is small (8 B) and hits L2 for all threads
           simultaneously.  Eliminates smem allocation + sync.

  ISSUE 4  The output loop re-reads partial_lse[bidx, bidy, s, 0 / 1] for
           every one of 512 threads × 8 splits.  After ISSUE 2 fix, these
           reads are only done once in the precompute step.

-----------------------------------------------------------------------
Kernel versions
-----------------------------------------------------------------------
  v0  baseline  — exact copy of kvsplit_reduce_kernel_clc
  v1  fix ISSUE 3  — direct sentinel read, no smem for sentinel
  v2  fix ISSUE 1  — 8-thread max+denom, warp-reduce
  v3  fix ISSUE 2  — precompute smem_scales[8], broadcast to 512 threads
     (v3 is the target; v4 explores multi-pair block assignment)

Run:
  python -m src.kernels.reduce_opt
"""
import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils as utils

import torch


# ── Constants ──────────────────────────────────────────────────────────────────
NUM_HEADS:    cutlass.Constexpr = 16
DV:           cutlass.Constexpr = 512
NUM_SPLITS:   cutlass.Constexpr = 8
T_MAX:        cutlass.Constexpr = 8
ROW_MAX_SUM_PAIR: cutlass.Constexpr = 2

BLOCK_SIZE_REDUCE: cutlass.Constexpr = 512
LN2 = 0.6931471805599453


# ─────────────────────────────────────────────────────────────────────────────
# v0: BASELINE — exact replica of kvsplit_reduce_kernel_clc
# -  Issue 1: thread 0 serial max+denom (511 threads idle)
# -  Issue 2: 512 × 8 exp() in output loop
# -  Issue 3: smem sentinel (1 float smem + sync_threads)
# ─────────────────────────────────────────────────────────────────────────────

@cute.kernel
def reduce_v0(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output:      cute.Tensor,
    lse:         cute.Tensor,
):
    """v0: baseline — identical logic to kvsplit_reduce_kernel_clc."""
    head_dim_ckv   = partial_out.shape[3]
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _    = cute.arch.thread_idx()

    allocator = cutlass.utils.SmemAllocator()
    # Issue 3: sentinel broadcast through smem
    smem_sentinel = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

    if tidx == 0:
        smem_sentinel[0] = partial_lse[bidx, bidy, 0, 0]
    cute.arch.sync_threads()
    sentinel_val = smem_sentinel[0]

    if sentinel_val < cutlass.Float32(1e30):
        allocator2 = cutlass.utils.SmemAllocator()
        smem_global_max   = allocator2.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_global_denom = allocator2.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

        # Issue 1: thread 0 serial over num_splits
        if tidx == 0:
            g_max = -cutlass.Float32(math.inf)
            for s in range(num_splits):
                local_max = partial_lse[bidx, bidy, s, 0]
                if local_max > g_max:
                    g_max = local_max
            smem_global_max[0] = g_max

            g_denom = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                g_denom += local_denom * cute.math.exp(local_max - g_max)
            smem_global_denom[0] = g_denom

        cute.arch.sync_threads()
        g_max   = smem_global_max[0]
        g_denom = smem_global_denom[0]

        if tidx == 0:
            lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

        # Issue 2: 512 threads × 8 exp() = 4096 total exp() calls
        if tidx < head_dim_ckv:
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                scale = cute.math.exp(local_max - g_max) / g_denom
                acc += partial_out[bidx, bidy, s, tidx] * scale
            output[bidx, bidy, tidx] = cutlass.BFloat16(acc)


# ─────────────────────────────────────────────────────────────────────────────
# v1: fix ISSUE 3 — direct sentinel read, no smem alloc for sentinel
# -  All threads read partial_lse[bidx, bidy, 0, 0] directly.
# -  Saves: 1 float smem + sync_threads.
# ─────────────────────────────────────────────────────────────────────────────

@cute.kernel
def reduce_v1(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output:      cute.Tensor,
    lse:         cute.Tensor,
):
    """v1: fix Issue 3 — direct sentinel read, no smem sentinel."""
    head_dim_ckv   = partial_out.shape[3]
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _    = cute.arch.thread_idx()

    # All threads read the sentinel directly — tiny value, hits L2 broadcast
    sentinel_val = partial_lse[bidx, bidy, 0, 0]

    if sentinel_val < cutlass.Float32(1e30):
        allocator = cutlass.utils.SmemAllocator()
        smem_global_max   = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_global_denom = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

        # Issue 1: still thread 0 serial
        if tidx == 0:
            g_max = -cutlass.Float32(math.inf)
            for s in range(num_splits):
                local_max = partial_lse[bidx, bidy, s, 0]
                if local_max > g_max:
                    g_max = local_max
            smem_global_max[0] = g_max

            g_denom = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                g_denom += local_denom * cute.math.exp(local_max - g_max)
            smem_global_denom[0] = g_denom

        cute.arch.sync_threads()
        g_max   = smem_global_max[0]
        g_denom = smem_global_denom[0]

        if tidx == 0:
            lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

        # Issue 2: still 512 × 8 exp()
        if tidx < head_dim_ckv:
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                local_max   = partial_lse[bidx, bidy, s, 0]
                local_denom = partial_lse[bidx, bidy, s, 1]
                scale = cute.math.exp(local_max - g_max) / g_denom
                acc += partial_out[bidx, bidy, s, tidx] * scale
            output[bidx, bidy, tidx] = cutlass.BFloat16(acc)


# ─────────────────────────────────────────────────────────────────────────────
# v2: fix ISSUE 3 + ISSUE 1 — 8-thread max+denom + warp-reduce
#
# Thread s (for s < num_splits) reads partial_lse[..., s, *] directly.
# A warp-reduce computes global_max across 8 threads, then a second pass
# computes global_denom.  One sync_threads distributes to all 512 threads.
# ─────────────────────────────────────────────────────────────────────────────

@cute.kernel
def reduce_v2(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output:      cute.Tensor,
    lse:         cute.Tensor,
):
    """v2: fix Issue 1+3 — 8-thread max/denom with warp-reduce."""
    head_dim_ckv   = partial_out.shape[3]
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _    = cute.arch.thread_idx()

    sentinel_val = partial_lse[bidx, bidy, 0, 0]

    if sentinel_val < cutlass.Float32(1e30):
        allocator = cutlass.utils.SmemAllocator()
        smem_global_max   = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_global_denom = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)

        # Fix Issue 1: each of the 8 threads reads one split
        # Thread s reads split s.  Threads 8-511 read -inf / 0 (neutral elements).
        if tidx < num_splits:
            local_max   = partial_lse[bidx, bidy, tidx, 0]
            local_denom = partial_lse[bidx, bidy, tidx, 1]
        else:
            local_max   = -cutlass.Float32(math.inf)
            local_denom = cutlass.Float32(0)

        # Warp-reduce max across lanes 0..31 (only lanes 0..7 have real data)
        g_max = local_max
        for offset in (1, 2, 4, 8, 16):
            g_max = (lambda a, b: a if a > b else b)(
                g_max, cute.arch.shuffle_sync_bfly(g_max, offset=offset))

        # Now compute scaled denom (only in lanes 0..7; others contribute 0)
        scaled = local_denom * cute.math.exp(local_max - g_max)
        g_denom = scaled
        for offset in (1, 2, 4, 8, 16):
            g_denom = g_denom + cute.arch.shuffle_sync_bfly(g_denom, offset=offset)

        if tidx == 0:
            smem_global_max[0]   = g_max
            smem_global_denom[0] = g_denom
            lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

        cute.arch.sync_threads()
        g_max   = smem_global_max[0]
        g_denom = smem_global_denom[0]

        # Issue 2: still 512 × 8 exp() (not yet fixed)
        if tidx < head_dim_ckv:
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                local_max_s   = partial_lse[bidx, bidy, s, 0]
                local_denom_s = partial_lse[bidx, bidy, s, 1]
                scale = cute.math.exp(local_max_s - g_max) / g_denom
                acc += partial_out[bidx, bidy, s, tidx] * scale
            output[bidx, bidy, tidx] = cutlass.BFloat16(acc)


# ─────────────────────────────────────────────────────────────────────────────
# v3: fix ISSUE 1 + 2 + 3 — precompute smem_scales[8], all fixes applied
#
# Warp 0, lanes 0..7: compute global_max and smem_scales[s] once.
# The 512 output threads read smem_scales[s] by index instead of calling exp().
# 4096 exp() → 8 exp() total.  Also fixes Issues 1 and 3.
# ─────────────────────────────────────────────────────────────────────────────

@cute.kernel
def reduce_v3(
    partial_out: cute.Tensor,
    partial_lse: cute.Tensor,
    output:      cute.Tensor,
    lse:         cute.Tensor,
):
    """v3: all fixes — precompute smem_scales[8]; 8 exp() instead of 4096."""
    head_dim_ckv   = partial_out.shape[3]
    num_splits: cutlass.Constexpr = NUM_SPLITS

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _    = cute.arch.thread_idx()

    sentinel_val = partial_lse[bidx, bidy, 0, 0]

    if sentinel_val < cutlass.Float32(1e30):
        allocator = cutlass.utils.SmemAllocator()
        smem_global_max   = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        smem_global_denom = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((1,), stride=(1,)), 16, None)
        # smem_scales[num_splits] — broadcast from warp 0 to all 512 output threads
        smem_scales = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((num_splits,), stride=(1,)), 16, None)

        # Fix Issue 1+2: threads 0..7 (lanes in warp 0) each own one split
        if tidx < num_splits:
            local_max   = partial_lse[bidx, bidy, tidx, 0]
            local_denom = partial_lse[bidx, bidy, tidx, 1]
        else:
            local_max   = -cutlass.Float32(math.inf)
            local_denom = cutlass.Float32(0)

        # Warp-reduce max (lanes 0..15 suffice for 8 splits, but full warp is fine)
        g_max = local_max
        for offset in (1, 2, 4, 8, 16):
            g_max = (lambda a, b: a if a > b else b)(
                g_max, cute.arch.shuffle_sync_bfly(g_max, offset=offset))

        scaled = local_denom * cute.math.exp(local_max - g_max)
        g_denom = scaled
        for offset in (1, 2, 4, 8, 16):
            g_denom = g_denom + cute.arch.shuffle_sync_bfly(g_denom, offset=offset)

        # Thread s writes its pre-computed scale (only threads 0..7 have real data)
        if tidx < num_splits:
            smem_scales[tidx] = cute.math.exp(local_max - g_max) / g_denom

        if tidx == 0:
            smem_global_max[0]   = g_max
            smem_global_denom[0] = g_denom
            lse[bidx, bidy] = (g_max + cute.math.log(g_denom)) / cutlass.Float32(LN2)

        cute.arch.sync_threads()
        g_max   = smem_global_max[0]
        g_denom = smem_global_denom[0]

        # Fix Issue 2: read smem_scales[s] instead of calling exp()
        if tidx < head_dim_ckv:
            acc = cutlass.Float32(0)
            for s in range(num_splits):
                scale = smem_scales[s]             # smem read, NOT exp()
                acc += partial_out[bidx, bidy, s, tidx] * scale
            output[bidx, bidy, tidx] = cutlass.BFloat16(acc)


# ═══════════════════════════════════════════════════════════════════════════════
# Compilation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fake(dtype, shape, stride_order, align):
    from cutlass.cute.runtime import make_fake_compact_tensor
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align)


def _fake_inputs():
    T         = 8
    num_heads = 16
    num_s     = 8
    dv        = 512
    partial_out = _fake(cute.Float32,  (T, num_heads, num_s, dv), (3, 2, 1, 0), 16)
    partial_lse = _fake(cute.Float32,  (T, num_heads, num_s, 2),  (3, 2, 1, 0), 16)
    output      = _fake(cute.BFloat16, (T, num_heads, dv),        (2, 1, 0),     16)
    lse         = _fake(cute.Float32,  (T, num_heads),             (1, 0),         4)
    return partial_out, partial_lse, output, lse


def compile_all():
    print("Compiling reduce kernels v0 … v3 …", end=" ", flush=True)
    inp = _fake_inputs()
    opts = "--enable-tvm-ffi"
    from cutlass.cute.runtime import make_fake_stream

    def _comp(fn):
        return cute.compile(fn, *inp, options=opts)

    compiled = {
        "v0": _comp(reduce_v0),
        "v1": _comp(reduce_v1),
        "v2": _comp(reduce_v2),
        "v3": _comp(reduce_v3),
    }
    print("done.")
    return compiled


# ═══════════════════════════════════════════════════════════════════════════════
# Correctness + benchmark harness
# ═══════════════════════════════════════════════════════════════════════════════

def _reference_reduce(partial_out, partial_lse):
    """Float32 reference: compute combined output and lse from splits."""
    T, H, S, D = partial_out.shape
    out_ref = torch.zeros(T, H, D, dtype=torch.float32, device=partial_out.device)
    lse_ref = torch.zeros(T, H,    dtype=torch.float32, device=partial_out.device)
    SENTINEL = 1e30
    LN2_ref  = 0.6931471805599453

    for t in range(T):
        for h in range(H):
            sentinel = partial_lse[t, h, 0, 0].item()
            if sentinel >= SENTINEL:
                # token was handled by single-split fast path
                # output was already written; lse is set directly in compute kernel
                lse_ref[t, h] = 0.0  # placeholder
                continue
            lse_vals   = partial_lse[t, h, :, 0]   # (S,)
            denom_vals = partial_lse[t, h, :, 1]   # (S,)
            g_max = lse_vals.max().item()
            g_denom = (denom_vals * (lse_vals - g_max).exp()).sum().item()
            lse_ref[t, h] = (g_max + math.log(g_denom)) / LN2_ref
            for s in range(S):
                scale = math.exp(lse_vals[s].item() - g_max) / g_denom
                out_ref[t, h] += partial_out[t, h, s] * scale
    return out_ref.to(torch.bfloat16), lse_ref


def _make_test_tensors(T=8, H=16, S=8, D=512):
    """Create realistic partial lse / partial out tensors for benchmarking."""
    torch.manual_seed(42)
    partial_max  = torch.randn(T, H, S,    device="cuda", dtype=torch.float32) * 3.0
    partial_sum  = torch.rand( T, H, S,    device="cuda", dtype=torch.float32) * 10 + 1.0
    partial_lse  = torch.stack([partial_max, partial_sum], dim=-1)  # (T, H, S, 2)
    partial_out  = torch.randn(T, H, S, D, device="cuda", dtype=torch.float32)
    out_buf      = torch.empty(T, H, D,    device="cuda", dtype=torch.bfloat16)
    lse_buf      = torch.empty(T, H,       device="cuda", dtype=torch.float32)
    return partial_out, partial_lse, out_buf, lse_buf


def check_correctness(compiled, T=8, H=16, S=8, D=512):
    partial_out, partial_lse, out_v, lse_v = _make_test_tensors(T, H, S, D)
    ref_out, ref_lse = _reference_reduce(partial_out, partial_lse)

    all_pass = True
    for name, fn in compiled.items():
        out_v.zero_()
        lse_v.zero_()
        fn(partial_out, partial_lse, out_v, lse_v)
        torch.cuda.synchronize()

        max_out_err = (out_v.float() - ref_out.float()).abs().max().item()
        # lse correctness: only rows that had sentinel >= 1e30 were skipped
        mask = partial_lse[:, :, 0, 0] < 1e30
        max_lse_err = (lse_v[mask] - ref_lse[mask]).abs().max().item() if mask.any() else 0.0

        status = "PASS" if max_out_err < 1e-2 and max_lse_err < 1e-4 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name}  out_err={max_out_err:.2e}  lse_err={max_lse_err:.2e}  → {status}")

    return all_pass


def benchmark(compiled, T=8, H=16, S=8, D=512, warmup=50, reps=200):
    """Time each version using CUDA events: (T, H) blocks × BLOCK_SIZE_REDUCE threads."""
    partial_out, partial_lse, out_v, lse_v = _make_test_tensors(T, H, S, D)

    results = {}
    for name, fn in compiled.items():
        # warmup
        for _ in range(warmup):
            fn(partial_out, partial_lse, out_v, lse_v)
        torch.cuda.synchronize()

        start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        end_ev   = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            start_ev[i].record()
            fn(partial_out, partial_lse, out_v, lse_v)
            end_ev[i].record()
        torch.cuda.synchronize()

        times_us = [s.elapsed_time(e) * 1e3 for s, e in zip(start_ev, end_ev)]
        med = sorted(times_us)[reps // 2]
        results[name] = med
        print(f"  {name}  median={med:.2f} µs")

    return results


def main():
    compiled = compile_all()

    print("\n── Correctness ─────────────────────────────────────────────────")
    passed = check_correctness(compiled)

    print("\n── Benchmark (reduce only, T=8 H=16 S=8 D=512) ────────────────")
    results = benchmark(compiled)

    print("\n── Summary ─────────────────────────────────────────────────────")
    base = results.get("v0", 1.0)
    for name, t in results.items():
        speedup = base / t
        print(f"  {name}  {t:.2f} µs  ({speedup:.2f}×  vs v0)")

    return 0 if passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

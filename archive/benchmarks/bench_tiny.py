#!/usr/bin/env python3
"""Benchmark CuTeDSL kernel on small workloads (max valid count per token < 64)."""
import math
import torch
from pathlib import Path
from cook import get_inputs, get_outputs, _call

MAX_VALID = 32
WARMUP = 10
ITERS = 50
H, D, Dp = 16, 512, 64

ROOT = Path(__file__).parent.parent
CONTEST = ROOT.parent / "flashinfer26dsa" / "mlsys26-contest"
JSONL = str(CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl")


def bench_small():
    from ref import run as ref_fn
    from impl_cutedsl import run as impl_fn

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Filter: all tokens in workload have valid_count < {MAX_VALID}\n")

    # First pass: collect qualifying workloads
    workloads = []
    for i, inp in enumerate(get_inputs(JSONL)):
        if max(inp["valid_per_token"]) <= MAX_VALID:
            workloads.append((i, inp))

    print(f"Found {len(workloads)} / {sum(1 for _ in open(JSONL))} workloads with max valid <= {MAX_VALID}\n")
    if not workloads:
        print("No qualifying workloads found.")
        return

    hdr = f"{'#':>3} {'Orig#':>5} {'T':>2} {'Valid':>40} {'Ref ms':>8} {'Impl ms':>8} {'Speedup':>8} {'GFLOPS':>8}"
    print(hdr)
    print("-" * len(hdr))

    durations, gflops_list, speedups = [], [], []
    for j, (orig_idx, inp) in enumerate(workloads):
        T = inp["num_tokens"]
        vpt = inp["valid_per_token"]
        fl = sum(2*H*D*v + 2*H*Dp*v + 2*H*v*D + 5*H*v for v in vpt if v > 0)

        def _run_impl():
            o, l = get_outputs(T)
            _call(impl_fn, inp, o, l)

        def _run_ref():
            o, l = get_outputs(T)
            _call(ref_fn, inp, o, l)

        # Warmup
        for _ in range(WARMUP):
            _run_impl()
            _run_ref()
        torch.cuda.synchronize()

        # Bench impl
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
        for s, e in evs:
            s.record(); _run_impl(); e.record()
        torch.cuda.synchronize()
        i_ms = sum(s.elapsed_time(e) for s, e in evs) / ITERS

        # Bench ref
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
        for s, e in evs:
            s.record(); _run_ref(); e.record()
        torch.cuda.synchronize()
        r_ms = sum(s.elapsed_time(e) for s, e in evs) / ITERS

        sp = r_ms / i_ms if i_ms > 0 else 0
        gf = fl / (i_ms * 1e-3) / 1e9 if i_ms > 0 else 0
        valids = ",".join(str(v) for v in vpt)
        print(f"{j+1:>3} {orig_idx+1:>5} {T:>2} {valids:>40} {r_ms:>8.3f} {i_ms:>8.3f} {sp:>7.2f}x {gf:>8.2f}")
        durations.append(i_ms)
        gflops_list.append(gf)
        speedups.append(sp)

    if durations:
        gmean = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
        print(f"\n  Geomean:  duration={gmean(durations):.3f} ms  GFLOPS={gmean(gflops_list):.2f}  speedup={gmean(speedups):.2f}x")


if __name__ == "__main__":
    bench_small()

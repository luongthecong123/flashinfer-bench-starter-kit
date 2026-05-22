#!/usr/bin/env python3
"""Run hybrid dispatch + per-token split benchmarks on Modal B200.
Uses the same image as modal_cook.py — no image rebuild needed."""
import modal
from pathlib import Path

DEV_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(DEV_DIR, remote_path="/root/dev")
)

app = modal.App("dsa-hybrid-split", image=image)
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)


@app.function(
    gpu="B200:1",
    timeout=900,
    volumes={"/data": trace_volume},
)
def run_bench():
    import sys, os
    os.chdir("/root/dev")
    sys.path.insert(0, "/root/dev")

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    from cook import check_correctness, benchmark
    from ref import run as ref_fn

    JSONL = "/data/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    # ─── 1. Hybrid dispatch ───────────────────────────────────
    print("=" * 70)
    print("STRATEGY 1: HYBRID DISPATCH (CuTeDSL small / torch.compile large)")
    print("=" * 70)
    from impl_hybrid import run as hybrid_fn

    print("\n--- Correctness ---")
    ok1 = check_correctness(hybrid_fn, ref_fn, jsonl_path=JSONL)

    print("\n--- Benchmark ---")
    benchmark(hybrid_fn, ref_fn, jsonl_path=JSONL)

    # ─── 2. Per-token split ───────────────────────────────────
    print("\n")
    print("=" * 70)
    print("STRATEGY 2: PER-TOKEN SPLIT (CuTeDSL small tokens + torch.compile large tokens)")
    print("=" * 70)
    from impl_split import run as split_fn

    print("\n--- Correctness ---")
    ok2 = check_correctness(split_fn, ref_fn, jsonl_path=JSONL)

    print("\n--- Benchmark ---")
    benchmark(split_fn, ref_fn, jsonl_path=JSONL)

    # ─── 3. Baselines for comparison ──────────────────────────
    print("\n")
    print("=" * 70)
    print("BASELINE: TORCH.COMPILE BATCHED (current solution)")
    print("=" * 70)
    from impl import run as tc_fn

    print("\n--- Benchmark ---")
    benchmark(tc_fn, ref_fn, jsonl_path=JSONL)

    print("\n")
    print("=" * 70)
    print("BASELINE: CuTeDSL MULTI-STREAM (per-token)")
    print("=" * 70)
    from impl_cutedsl_forT import run as cutedsl_ms_fn

    print("\n--- Benchmark ---")
    benchmark(cutedsl_ms_fn, ref_fn, jsonl_path=JSONL)


@app.local_entrypoint()
def main():
    run_bench.remote()

#!/usr/bin/env python3
"""Benchmark hybrid dispatch variants on Modal B200.

Compares 4 approaches:
1. Standalone tc (baseline — no branching, pure torch.compile)
2. Hybrid original (column-check .item() branching)
3. Hybrid overlap (speculative tc prep on second stream)
4. Hybrid cache (decision cached after first call, zero sync)
"""
import modal
from pathlib import Path

DEV_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(DEV_DIR, remote_path="/root/dev")
)

app = modal.App("dsa-hybrid-variants", image=image)
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)


@app.function(
    gpu="B200:1",
    timeout=1800,
    volumes={"/data": trace_volume},
)
def run_bench():
    import sys, os
    os.chdir("/root/dev")
    sys.path.insert(0, "/root/dev")

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    from cook import check_correctness, benchmark
    from ref import run as ref_fn

    JSONL = "/data/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    # --- 1. Standalone tc (baseline) ---
    print("=" * 70)
    print("1. STANDALONE TC (baseline)")
    print("=" * 70)
    from impl import run as tc_fn
    print("Correctness:")
    check_correctness(tc_fn, ref_fn, jsonl_path=JSONL)
    print("\nBenchmark:")
    benchmark(tc_fn, ref_fn, jsonl_path=JSONL)

    # --- 2. Hybrid original ---
    print("\n" + "=" * 70)
    print("2. HYBRID ORIGINAL (.item() branch)")
    print("=" * 70)
    from impl_hybrid import run as hybrid_fn, THRESHOLD as HT
    print(f"THRESHOLD = {HT}")
    print("Correctness:")
    check_correctness(hybrid_fn, ref_fn, jsonl_path=JSONL)
    print("\nBenchmark:")
    benchmark(hybrid_fn, ref_fn, jsonl_path=JSONL)

    # --- 3. Hybrid overlap ---
    print("\n" + "=" * 70)
    print("3. HYBRID OVERLAP (speculative tc prep on stream 1)")
    print("=" * 70)
    from impl_hybrid_overlap import run as overlap_fn, THRESHOLD as OT
    print(f"THRESHOLD = {OT}")
    print("Correctness:")
    check_correctness(overlap_fn, ref_fn, jsonl_path=JSONL)
    print("\nBenchmark:")
    benchmark(overlap_fn, ref_fn, jsonl_path=JSONL)

    # --- 4. Hybrid cache ---
    print("\n" + "=" * 70)
    print("4. HYBRID CACHE (decision cached after warmup)")
    print("=" * 70)
    from impl_hybrid_cache import run as cache_fn, THRESHOLD as CT
    print(f"THRESHOLD = {CT}")
    print("Correctness:")
    check_correctness(cache_fn, ref_fn, jsonl_path=JSONL)
    print("\nBenchmark:")
    benchmark(cache_fn, ref_fn, jsonl_path=JSONL)


@app.local_entrypoint()
def main():
    run_bench.remote()

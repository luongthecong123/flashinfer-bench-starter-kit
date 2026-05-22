#!/usr/bin/env python3
"""Benchmark CUDA graph variants on Modal B200.

Compares 4 approaches:
1. Standalone tc (mode=default) — baseline
2. Standalone tc (mode=reduce-overhead) — CUDA graph
3. Hybrid (mode=default) — current hybrid
4. Hybrid (mode=reduce-overhead) — hybrid + CUDA graph
"""
import modal
from pathlib import Path

DEV_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(DEV_DIR, remote_path="/root/dev")
)

app = modal.App("dsa-graph-bench", image=image)
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

    # --- 1. Standalone tc (default mode) ---
    print("=" * 70)
    print("1. STANDALONE TC (mode=default)")
    print("=" * 70)
    from impl import run as tc_fn
    print("Correctness:")
    check_correctness(tc_fn, ref_fn, jsonl_path=JSONL)
    print("\nBenchmark:")
    benchmark(tc_fn, ref_fn, jsonl_path=JSONL)

    # --- 2. Standalone tc + CUDA graph (reduce-overhead) ---
    print("\n" + "=" * 70)
    print("2. STANDALONE TC + CUDA GRAPH (mode=reduce-overhead)")
    print("=" * 70)
    from impl_graph import run as tc_graph_fn
    print("Correctness:")
    check_correctness(tc_graph_fn, ref_fn, jsonl_path=JSONL)
    print("\nBenchmark:")
    benchmark(tc_graph_fn, ref_fn, jsonl_path=JSONL, warmup=20)

    # --- 3. Hybrid (default mode) ---
    print("\n" + "=" * 70)
    print("3. HYBRID (mode=default)")
    print("=" * 70)
    from impl_hybrid import run as hybrid_fn, THRESHOLD as HT
    print(f"THRESHOLD = {HT}")
    print("Correctness:")
    check_correctness(hybrid_fn, ref_fn, jsonl_path=JSONL)
    print("\nBenchmark:")
    benchmark(hybrid_fn, ref_fn, jsonl_path=JSONL)

    # --- 4. Hybrid + CUDA graph (reduce-overhead) ---
    print("\n" + "=" * 70)
    print("4. HYBRID + CUDA GRAPH (mode=reduce-overhead)")
    print("=" * 70)
    from impl_hybrid_graph import run as hybrid_graph_fn, THRESHOLD as HGT
    print(f"THRESHOLD = {HGT}")
    print("Correctness:")
    check_correctness(hybrid_graph_fn, ref_fn, jsonl_path=JSONL)
    print("\nBenchmark:")
    benchmark(hybrid_graph_fn, ref_fn, jsonl_path=JSONL, warmup=20)


@app.local_entrypoint()
def main():
    run_bench.remote()

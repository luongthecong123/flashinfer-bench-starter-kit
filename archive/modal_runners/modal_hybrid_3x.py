#!/usr/bin/env python3
"""Run hybrid dispatch benchmark 3x on Modal B200."""
import modal
from pathlib import Path

DEV_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(DEV_DIR, remote_path="/root/dev")
)

app = modal.App("dsa-hybrid-3x", image=image)
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
    from impl_hybrid import run as hybrid_fn, THRESHOLD

    JSONL = "/data/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    print(f"THRESHOLD = {THRESHOLD}")
    print()

    print("--- Correctness ---")
    check_correctness(hybrid_fn, ref_fn, jsonl_path=JSONL)

    for trial in range(1, 4):
        print(f"\n{'='*70}")
        print(f"HYBRID DISPATCH — Run {trial}/3")
        print(f"{'='*70}")
        benchmark(hybrid_fn, ref_fn, jsonl_path=JSONL)


@app.local_entrypoint()
def main():
    run_bench.remote()

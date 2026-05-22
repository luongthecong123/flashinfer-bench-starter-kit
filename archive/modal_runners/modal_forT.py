#!/usr/bin/env python3
"""Run impl_forT.py (per-token torch.compile with CUDA streams) on Modal B200."""
import modal
from pathlib import Path

DEV_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(DEV_DIR, remote_path="/root/dev")
)

app = modal.App("dsa-cook", image=image)
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)


@app.function(
    gpu="B200:1",
    timeout=600,
    volumes={"/data": trace_volume},
)
def run_forT():
    import sys, os
    os.chdir("/root/dev")
    sys.path.insert(0, "/root/dev")

    from cook import check_correctness, benchmark
    from ref import run as ref_fn
    from impl_forT import run as impl_fn

    JSONL = "/data/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print("=== Correctness ===")
    check_correctness(impl_fn, ref_fn, jsonl_path=JSONL)

    print("\n=== Benchmark ===")
    benchmark(impl_fn, ref_fn, jsonl_path=JSONL)


@app.local_entrypoint()
def main():
    run_forT.remote()

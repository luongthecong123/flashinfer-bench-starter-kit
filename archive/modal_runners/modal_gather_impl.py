#!/usr/bin/env python3
"""Run gather_impl.py on Modal B200."""
import modal
from pathlib import Path

ZEN_DIR = Path(__file__).parent
ROOT_DIR = ZEN_DIR.parent
DEV_DIR = ROOT_DIR / "dev"
CONTEST_DIR = ROOT_DIR.parent / "flashinfer26dsa" / "mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(ZEN_DIR, remote_path="/app/zen")
    .add_local_dir(DEV_DIR, remote_path="/app/dev")
)

app = modal.App("dsa-cook", image=image)
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

@app.function(
    gpu="B200:1",
    timeout=600,
    volumes={"/data": trace_volume},
)
def run_cook():
    import sys, os
    os.chdir("/app/dev")
    sys.path.insert(0, "/app/dev")
    sys.path.insert(0, "/app/zen")
    sys.path.insert(0, "/app")

    # Patch cook to use our impl and Modal paths
    import cook
    from pathlib import Path
    cook.CONTEST = Path("/data")
    cook.JSONL = cook.CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    # Override impl_fn with our gather_impl
    from zen.gather_impl import run as impl_fn
    cook.impl_fn = impl_fn

    cook.main()


@app.local_entrypoint()
def main():
    run_cook.remote()

#!/usr/bin/env python3
"""Run cook.py on Modal B200."""
import modal
from pathlib import Path

DEV_DIR = Path(__file__).parent
CONTEST_DIR = DEV_DIR.parent.parent / "flashinfer26dsa" / "mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja")
    .add_local_dir(DEV_DIR, remote_path="/root/dev")
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
    os.chdir("/root/dev")
    sys.path.insert(0, "/root/dev")

    # Patch CONTEST path to Modal volume
    import cook
    from pathlib import Path
    cook.CONTEST = Path("/data")
    cook.JSONL = cook.CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    cook.main()

@app.local_entrypoint()
def main():
    run_cook.remote()

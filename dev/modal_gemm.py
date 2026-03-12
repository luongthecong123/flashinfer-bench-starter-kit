#!/usr/bin/env python3
"""Run gemm.py on Modal B200."""
import modal
from pathlib import Path

DEV_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja")
    .add_local_dir(DEV_DIR, remote_path="/root/dev")
)

app = modal.App("dsa-gemm", image=image)

@app.function(
    gpu="B200:1",
    timeout=600,
)
def run_gemm():
    import sys, os
    os.chdir("/root/dev")
    sys.path.insert(0, "/root/dev")
    import gemm
    gemm.main()

@app.local_entrypoint()
def main():
    run_gemm.remote()

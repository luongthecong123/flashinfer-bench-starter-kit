#!/usr/bin/env python3
"""
Run intra-kernel profiling of Fused_DSA on Modal B200.
Measures time of each phase: score_nope, score_pe, softmax, output_load, output_gemm, epilogue.

Usage: modal run zen/modal_profile_dsa_intrakernel.py
"""
import modal
from pathlib import Path

ZEN_DIR = Path(__file__).parent
ROOT_DIR = ZEN_DIR.parent
DEV_DIR = ROOT_DIR / "dev"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(ZEN_DIR, remote_path="/app/zen")
    .add_local_dir(DEV_DIR, remote_path="/app/dev")
)

app = modal.App("dsa-intrakernel-profile", image=image)


@app.function(gpu="B200:1", timeout=600)
def run():
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/zen")
    sys.path.insert(0, "/app/dev")

    from zen.profile_dsa_intrakernel import run_profiling
    probe_cpu, trace_json = run_profiling()
    return trace_json


@app.local_entrypoint()
def main():
    trace_json = run.remote()

    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / "dsa_intrakernel_trace.json"
    out.write_text(trace_json)
    print(f"\nSaved: {out}")
    print("Open with chrome://tracing or https://ui.perfetto.dev")

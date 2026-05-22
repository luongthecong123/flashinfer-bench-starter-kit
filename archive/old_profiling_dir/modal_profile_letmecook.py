#!/usr/bin/env python3
"""
Run intra-kernel profiling of letmecook on Modal B200.
Measures time of each phase: load_indices, score, softmax, output, epilogue.

Usage: modal run profiling/modal_profile_letmecook.py
"""
import modal
from pathlib import Path

PROF_DIR = Path(__file__).parent
ROOT_DIR = PROF_DIR.parent
DEV_DIR  = ROOT_DIR / "dev"
ZEN_DIR  = ROOT_DIR / "zen"
CONTEST  = ROOT_DIR.parent / "flashinfer26dsa" / "mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(PROF_DIR, remote_path="/app/profiling")
    .add_local_dir(DEV_DIR, remote_path="/app/dev")
    .add_local_dir(ZEN_DIR, remote_path="/app/zen")
    .add_local_dir(str(CONTEST), remote_path="/flashinfer26dsa/mlsys26-contest")
)

app = modal.App("letmecook-intrakernel-profile", image=image)


@app.function(gpu="B200:1", timeout=600)
def run():
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/dev")
    sys.path.insert(0, "/app/profiling")

    from profile_letmecook_intrakernel import run_profiling
    trace_json = run_profiling()
    return trace_json


@app.local_entrypoint()
def main():
    import json
    traces_json = run.remote()
    traces = json.loads(traces_json)

    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    for wid, trace in traces.items():
        out = reports_dir / f"letmecook_wl{wid}_trace.json"
        out.write_text(trace)
        print(f"Saved: {out}")
    print("\nOpen with chrome://tracing or https://ui.perfetto.dev")

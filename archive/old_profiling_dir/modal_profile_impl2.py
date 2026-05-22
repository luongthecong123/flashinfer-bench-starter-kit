#!/usr/bin/env python3
"""
Run intra-kernel profiling of impl2 (32-warp parallel-keys) on Modal B200.

Usage: modal run profiling/modal_profile_impl2.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run():
    import sys
    sys.path.insert(0, "/app")
    from src.profile_impl2_intrakernel import run_profiling
    return run_profiling()


@app.local_entrypoint()
def main():
    import json
    traces_json = run.remote()
    traces = json.loads(traces_json)

    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    for wid, trace in traces.items():
        out = reports_dir / f"impl2_wl{wid}_trace.json"
        out.write_text(trace)
        print(f"Saved: {out}")
    print("\nOpen with chrome://tracing or https://ui.perfetto.dev")

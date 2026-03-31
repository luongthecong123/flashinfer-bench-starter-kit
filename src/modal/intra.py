"""Intra-kernel profiling on Modal B200.
Change IMPL_MODULE / WORKLOAD_IDX to select the implementation and workload.
Usage: modal run src/modal/intra.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

# ── Pick workload: WL23 = index 22 (large: 7 tokens, MaxValid~462) ──
WORKLOAD_IDX = 22

COMPARE = [
    "src.modal.fused_tiny5v2_intra",
    "src.modal.fused_tiny5v4_intra",
]


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_intra(impl_module: str, workload_idx: int):
    import sys
    sys.path.insert(0, "/app")
    from importlib import import_module
    mod = import_module(impl_module)
    return mod.run_single(workload_idx)


@app.local_entrypoint()
def main():
    for impl_module in COMPARE:
        print(f"\n{'='*60}\nProfiling {impl_module}  WL{WORKLOAD_IDX + 1}\n{'='*60}")
        trace_json = run_intra.remote(impl_module, WORKLOAD_IDX)
        impl_short = impl_module.split(".")[-1]
        out_path = Path(f"reports/intra_{impl_short}_w{WORKLOAD_IDX}.json")
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(trace_json)
        print(f"Saved trace to {out_path}")

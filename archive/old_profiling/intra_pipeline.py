"""Modal runner: intra-phase profiling for fused_pipeline (PipelineAsync v1).
Workload 13 (index 12): T=8, MaxValid includes 2048.

Usage:
    modal run src/modal/intra_pipeline.py
Output:
    reports/intra_fused_pipeline_intra_w12.json
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 12
IMPL_MODULE  = "src.kernels.fused_pipeline_intra"


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_intra(impl_module: str, workload_idx: int):
    import sys
    sys.path.insert(0, "/app")
    from importlib import import_module
    mod = import_module(impl_module)
    return mod.run_single(workload_idx)


@app.local_entrypoint()
def main():
    print(f"\n{'='*60}\nProfiling {IMPL_MODULE}  WL{WORKLOAD_IDX + 1}\n{'='*60}")
    trace_json = run_intra.remote(IMPL_MODULE, WORKLOAD_IDX)
    out_path = Path(f"reports/intra_fused_pipeline_intra_w{WORKLOAD_IDX}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(trace_json)
    print(f"Saved trace to {out_path}")

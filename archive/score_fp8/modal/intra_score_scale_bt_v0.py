"""Modal runner: intra-kernel profiling for score_scale_bt_v0 (baseline).

Runs WL 70 (longest case, pg=89, M=5760) by default.
Outputs: reports/intra_score_scale_bt_v0_w<idx>.json (Chrome trace)
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, image

WORKLOAD_IDX = 4   # WL 70: pg=89, M=5760 (longest)
IMPL_MODULE  = "src.kernels.score_scale_bt_v0_intra"


@app.function(image=image, gpu="B200:1", timeout=600)
def run_intra(impl_module: str, workload_idx: int):
    sys.path.insert(0, "/app")
    from importlib import import_module
    mod = import_module(impl_module)
    return mod.run_single(workload_idx)


@app.local_entrypoint()
def main():
    print(f"\n{'='*60}\nProfiling {IMPL_MODULE}  WL{WORKLOAD_IDX}\n{'='*60}")
    trace_json = run_intra.remote(IMPL_MODULE, WORKLOAD_IDX)
    out_path = Path(f"reports/intra_score_scale_bt_v0_w{WORKLOAD_IDX}.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(trace_json)
    print(f"\nSaved trace to {out_path}")

"""Modal runner: kv_split intra-profiling on WL20 with DIM_SPLIT=512.

Workload 20: uuid=7a389715  T=8  MaxValid=[8,11,11,16,1641,73,1,1]

DIM_SPLIT=512 → NUM_SPLITS=4 (vs DIM_SPLIT=256 → NUM_SPLITS=8 baseline).

Output:
  reports/intra_kv_split_v3_thr_warpv3_intra_s512_w19.json

Usage:
    modal run src/modal/intra_w19_s512.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 19  # WL20: T=8, MaxValid=[8,11,11,16,1641,73,1,1]


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
def run_intra(impl_module: str, workload_idx: int):
    import sys
    sys.path.insert(0, "/app")
    from importlib import import_module
    mod = import_module(impl_module)
    return mod.run_single(workload_idx)


@app.local_entrypoint()
def main():
    impl_module = "src.kernels.kv_split_v3_thr_warpv3_intra_s512"
    out_path = Path("reports/intra_kv_split_v3_thr_warpv3_intra_s512_w19.json")
    print(f"\n{'='*60}\nProfiling {impl_module}  WL{WORKLOAD_IDX + 1} (DIM_SPLIT=512)\n{'='*60}")
    trace_json = run_intra.remote(impl_module, WORKLOAD_IDX)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(trace_json)
    print(f"Saved trace to {out_path}")

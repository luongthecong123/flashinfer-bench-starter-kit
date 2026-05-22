"""Modal runner: intra-phase profiling for workload 20 (index 19).

Workload 20: uuid=7a389715  T=8  MaxValid=[8,11,11,16,1641,73,1,1]
Very unbalanced — one token has 1641 valid entries while six have ≤16.

Runs two kernels for comparison:
  1. fused_tiny_thr_warpv3_intra   → single per-token kernel, no splits
  2. kv_split_v3_thr_warpv3_intra → 8-way KV-split, shows load imbalance
                                     + kernel launch overhead + reduction tax

Output files:
  reports/intra_fused_tiny_thr_warpv3_intra_w19.json
  reports/intra_kv_split_v3_thr_warpv3_intra_w19.json

Usage:
    modal run src/modal/intra_w19.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 19  # 0-based → workload 20 (uuid=7a389715, T=8, MaxValid=[8,11,11,16,1641,73,1,1])

RUNS = [
    ("src.kernels.fused_tiny_thr_warpv3_intra", "intra_fused_tiny_thr_warpv3_intra_w19.json"),
    ("src.kernels.kv_split_v3_thr_warpv3_intra", "intra_kv_split_v3_thr_warpv3_intra_w19.json"),
]


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
def run_intra(impl_module: str, workload_idx: int):
    import sys
    sys.path.insert(0, "/app")
    from importlib import import_module
    mod = import_module(impl_module)
    return mod.run_single(workload_idx)


@app.local_entrypoint()
def main():
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    for impl_module, out_filename in RUNS:
        print(f"\n{'='*60}\nProfiling {impl_module}  WL{WORKLOAD_IDX + 1}\n{'='*60}")
        trace_json = run_intra.remote(impl_module, WORKLOAD_IDX)
        out_path = out_dir / out_filename
        out_path.write_text(trace_json)
        print(f"Saved trace to {out_path}")

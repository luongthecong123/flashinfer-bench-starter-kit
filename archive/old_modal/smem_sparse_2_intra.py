"""Modal runner: compare baseline vs Strategy 2 (vec4 LDG.128) for smem_sparse.

Usage:
    modal run src/modal/smem_sparse_2_intra.py
Output:
    reports/intra_smem_sparse_w17.json    (baseline)
    reports/intra_smem_sparse_2_w17.json  (Strategy 2: vec4)
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 16

COMPARE = [
    ("baseline",   "src.kernels.smem_sparse_intra",   "reports/intra_smem_sparse_w17.json"),
    ("strategy_2", "src.kernels.smem_sparse_2_intra",  "reports/intra_smem_sparse_2_w17.json"),
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
    for label, impl_module, out_path_str in COMPARE:
        print(f"\n{'='*60}\n[{label}]  {impl_module}  WL{WORKLOAD_IDX + 1}\n{'='*60}")
        trace_json = run_intra.remote(impl_module, WORKLOAD_IDX)
        out_path = Path(out_path_str)
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(trace_json)
        print(f"Saved trace to {out_path}")

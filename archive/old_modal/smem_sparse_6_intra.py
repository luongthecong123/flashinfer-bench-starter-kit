"""Modal runner: compare S5 (cp.async) vs S6 (cp.async + early-exit).

Usage:
    modal run src/modal/smem_sparse_6_intra.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 16

COMPARE = [
    ("strategy_5", "src.kernels.smem_sparse_5_intra", "reports/intra_smem_sparse_5_w17.json"),
    ("strategy_6", "src.kernels.smem_sparse_6_intra", "reports/intra_smem_sparse_6_w17.json"),
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

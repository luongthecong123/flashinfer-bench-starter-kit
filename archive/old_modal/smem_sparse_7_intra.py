"""Modal runner: S6 vs S7 across WL16/17/18 (the three workloads with T=6,8,7).

WL16 (idx=15): T=6, MaxValid=[19,20,32,12,25,3]       — S6 leaves WG6,7 idle
WL17 (idx=16): T=8, MaxValid=[288,4,1884,21,136,2048,42,335] — all WGs busy baseline
WL18 (idx=17): T=7, MaxValid=[19,12,2048,21,26,46,136] — S6 leaves WG7 idle

Usage:
    modal run src/modal/smem_sparse_7_intra.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

# Workloads to benchmark (0-based indices)
WORKLOAD_INDICES = [15, 16, 17]   # WL16, WL17, WL18

IMPLS = [
    ("S6", "src.kernels.smem_sparse_6_intra"),
    ("S7", "src.kernels.smem_sparse_7_intra"),
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
    for wl_idx in WORKLOAD_INDICES:
        print(f"\n{'#'*70}")
        print(f"# Workload {wl_idx + 1} (0-based index {wl_idx})")
        print(f"{'#'*70}")
        for label, impl_module in IMPLS:
            print(f"\n{'='*60}\n[{label}]  {impl_module}  WL{wl_idx + 1}\n{'='*60}")
            trace_json = run_intra.remote(impl_module, wl_idx)
            out_path = Path(f"reports/intra_smem_sparse_{label.lower()}_wl{wl_idx + 1}.json")
            out_path.parent.mkdir(exist_ok=True)
            out_path.write_text(trace_json)
            print(f"Saved trace to {out_path}")

"""Modal runner: intra-phase comparison across kv_split variants vs kv_split_v3_thr_warpv3.

Workload 17 (index 16, 0-based): uuid=564007ac  T=8  MaxValid=[288,4,1884,21,136,2048,42,335]

Variants:
  xor    – baseline xor kernel (2D smem_partial + smem_out intermediate)
  xor_v2 – removes smem_out; reduces smem_partial → gmem directly in write phase
  xor_3D – v2 + 3D smem_partial layout with padding to avoid bank conflicts
  v3_thr_warpv3 – reference kernel

Usage:
    modal run src/modal/kv_split_xor_intra.py
Output:
    reports/intra_kv_split_xor_w16.json
    reports/intra_kv_split_xor_v2_w16.json
    reports/intra_kv_split_xor_3D_w16.json
    reports/intra_kv_split_xor_sentinel_w16.json
    reports/intra_kv_split_v3_thr_warpv3_w16.json
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 16   # 0-based → workload 17 (uuid=564007ac, T=8)

COMPARE = [
    ("src.kernels.kv_split_xor_intra",           "reports/intra_kv_split_xor_w16.json"),
    ("src.kernels.kv_split_xor_v2_intra",        "reports/intra_kv_split_xor_v2_w16.json"),
    ("src.kernels.kv_split_xor_3D_intra",        "reports/intra_kv_split_xor_3D_w16.json"),
    ("src.kernels.kv_split_xor_sentinel_intra",  "reports/intra_kv_split_xor_sentinel_w16.json"),
    ("src.kernels.kv_split_v3_thr_warpv3_intra", "reports/intra_kv_split_v3_thr_warpv3_w16.json"),
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
    for impl_module, out_path_str in COMPARE:
        print(f"\n{'='*60}\nProfiling {impl_module}  WL{WORKLOAD_IDX + 1}\n{'='*60}")
        trace_json = run_intra.remote(impl_module, WORKLOAD_IDX)
        out_path = Path(out_path_str)
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(trace_json)
        print(f"Saved trace to {out_path}")

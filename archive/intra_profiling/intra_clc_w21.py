"""Modal runner: intra-phase profiling for kv_split_v3_thr_warpv3_clc_v2 on WL#21.

Workload #21: uuid=5096e459, T=8, max_valid=[17,13,1887,16,180,1986,413,1]
Very unbalanced: two tokens have ~1900 valid entries while six have ≤180.

This runner exercises the CLC (work-stealing) compute kernel with three separate
probe buffers so that Perfetto shows concurrency between phases:

  pid = sm_id        → compute (load/score/softmax/output/clc_wait)
  pid = sm_id + 100  → epilogue / write (smem_partial → GMEM)
  pid = sm_id + 200  → reduce

The epilogue and the compute being on DIFFERENT Perfetto rows lets us visually
inspect whether the epilogue GMEM stores of tile N overlap in time with the
load/score GMEM reads of tile N+1 on the same SM.

Output:
  reports/intra_kv_split_v3_thr_warpv3_clc_intra_w21.json

Usage:
    modal run src/modal/intra_clc_w21.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 20  # 0-based → workload #21 (uuid=5096e459)
IMPL_MODULE  = "src.kernels.kv_split_v3_thr_warpv3_clc_intra"
OUT_FILENAME = "intra_kv_split_v3_thr_warpv3_clc_intra_w21.json"


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
    print(f"\n{'='*60}")
    print(f"Profiling {IMPL_MODULE}  WL{WORKLOAD_IDX + 1}")
    print(f"Perfetto layout:")
    print(f"  pid = sm_id        → compute phases")
    print(f"  pid = sm_id + 100  → epilogue/write phases  (SEPARATE ROW)")
    print(f"  pid = sm_id + 200  → reduce phases")
    print(f"{'='*60}")
    trace_json = run_intra.remote(IMPL_MODULE, WORKLOAD_IDX)
    out_path = out_dir / OUT_FILENAME
    out_path.write_text(trace_json)
    print(f"\nSaved trace to {out_path}")
    print(f"Open at https://ui.perfetto.dev  (load the JSON file)")

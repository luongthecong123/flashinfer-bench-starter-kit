"""Modal runner: intra-phase profiling for kv_split_v3_thr_warpv3_clc_pdl on WL#21.

Workload #21: uuid=5096e459, T=8, max_valid=[17,13,1887,16,180,1986,413,1]

PDL-specific Perfetto layout:
  pid = sm_id        → compute (load/valid_count/score/softmax/output/clc_wait)
  pid = sm_id + 100  → epilogue / write
  pid = sm_id + 200  → reduce  (includes pdl_wait phase + reduce phase)

The pdl_wait phase shows how long griddepcontrol_wait stalls the reduce kernel
while waiting for the compute kernel's grid-ending membar.  If PDL is effective,
we expect the reduce kernel's start time to OVERLAP with the last compute tiles
because griddepcontrol_launch_dependents was fired early in the compute loop.

Usage:
    modal run src/modal/intra_clc_pdl_w21.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 20  # 0-based → workload #21 (uuid=5096e459)
IMPL_MODULE  = "src.kernels.kv_split_v3_thr_warpv3_clc_pdl_intra"
OUT_FILENAME = "intra_kv_split_v3_thr_warpv3_clc_pdl_intra_w21.json"


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
    print(f"\nPDL Perfetto layout:")
    print(f"  pid = sm_id        → compute phases (load/score/softmax/output/clc_wait)")
    print(f"  pid = sm_id + 100  → epilogue phases")
    print(f"  pid = sm_id + 200  → reduce (pdl_wait + reduce)")
    print(f"\nKey metrics to look for:")
    print(f"  pdl_wait duration: how long reduce blocks stall at griddepcontrol_wait")
    print(f"  overlap: reduce start times vs last compute tile end times")
    print(f"{'='*60}")

    trace_json = run_intra.remote(IMPL_MODULE, WORKLOAD_IDX)
    out_path = out_dir / OUT_FILENAME
    out_path.write_text(trace_json)
    print(f"\nSaved trace to {out_path}")
    print(f"Open at https://ui.perfetto.dev  (load the JSON file)")

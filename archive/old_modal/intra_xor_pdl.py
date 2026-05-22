"""Modal runner: intra-phase profiling for kv_split_xor_pdl on WL#17.

Workload #17: uuid=564007ac, T=8, max_valid=[288,4,1884,21,136,2048,42,335]

PDL-specific Perfetto layout:
  pid = sm_id        → compute (upfront/score/softmax_max/softmax_exp_sum/output/write)
  pid = sm_id + 200  → reduce  (pdl_wait = prologue + griddepcontrol_wait stall, reduce = per-T merge)

If PDL overlap is effective, reduce's pdl_wait phase should START while compute's
last write phases are still running.

Usage:
    modal run src/modal/intra_xor_pdl.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 16  # 0-based → workload #17 (uuid=564007ac, T=8)
IMPL_MODULE  = "src.kernels.kv_split_xor_pdl_intra"
OUT_FILENAME = "intra_kv_split_xor_pdl_w17.json"


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
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
    out_path = Path(f"reports/{OUT_FILENAME}")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(trace_json)
    print(f"Saved trace to {out_path}")

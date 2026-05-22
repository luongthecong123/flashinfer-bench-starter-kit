"""Modal runner: intra profiling for topk_aten_cutedsl_v4_fuse.

Run all 6 workloads (or just one) and save trace JSON to reports/.
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, image

# Either an int (one WL) or "all" (all WLs)
WORKLOAD_IDX = "all"
IMPL_MODULE  = "src.kernels.topk_aten_cutedsl_v4_fuse_intra"
NAME         = "topk_v4_fuse"


@app.function(image=image, gpu="B200:1", timeout=900)
def run_intra(impl_module: str, workload_idx: int):
    sys.path.insert(0, "/app")
    from importlib import import_module
    mod = import_module(impl_module)
    return mod.run_single(workload_idx)


N_WORKLOADS = 6


@app.local_entrypoint()
def main():
    if WORKLOAD_IDX == "all":
        idxs = list(range(N_WORKLOADS))
    else:
        idxs = [int(WORKLOAD_IDX)]

    Path("reports").mkdir(exist_ok=True)
    for i in idxs:
        print(f"\n{'='*60}\nProfiling {IMPL_MODULE}  WL{i}\n{'='*60}")
        trace_json = run_intra.remote(IMPL_MODULE, i)
        out_path = Path(f"reports/intra_{NAME}_w{i}.json")
        out_path.write_text(trace_json)
        print(f"Saved trace to {out_path}")

"""Modal runner: intra profiling for kv_split_umma_v3_stages_pe_prologue_v2.

Usage:
    modal run src/modal/intra_v3_stages_pe_prologue_v2.py
    WORKLOAD_IDX=20 modal run src/modal/intra_v3_stages_pe_prologue_v2.py
    WORKLOAD_IDX=-1 modal run src/modal/intra_v3_stages_pe_prologue_v2.py   # all 23
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = int(os.environ.get("WORKLOAD_IDX", 20))
IMPL_MODULE  = "src.kernels.kv_split_umma_v3_stages_pe_prologue_intra_v2"
NAME         = "v3_stages_pe_prologue_v2"


@app.function(image=image, gpu="B200:1", timeout=1200, volumes={"/data": trace_volume})
def run_intra(impl_module: str, workload_idx: int):
    sys.path.insert(0, "/app")
    from importlib import import_module
    mod = import_module(impl_module)
    return mod.run_single(workload_idx)


@app.local_entrypoint()
def main():
    idxs = [WORKLOAD_IDX] if WORKLOAD_IDX != -1 else list(range(23))
    Path("reports").mkdir(exist_ok=True)
    for i in idxs:
        print(f"\n{'='*64}\nProfiling {IMPL_MODULE}  workload={i+1} (0-based {i})\n{'='*64}")
        trace_json = run_intra.remote(IMPL_MODULE, i)
        out_path = Path(f"reports/intra_{NAME}_w{i+1}.json")
        out_path.write_text(trace_json)
        print(f"Saved trace to {out_path}")

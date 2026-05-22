"""Modal runner: TMA S2G reduce intra profiling."""
import sys, os
from pathlib import Path
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

IMPL_MODULE  = "src.kernels.tma_s2g_reduce_intra"
OUT_FILENAME = "intra_tma_s2g_reduce.json"
WORKLOAD_IDX = 0


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
def run_intra(impl_module: str, workload_idx: int):
    import sys; sys.path.insert(0, "/app")
    from importlib import import_module
    mod = import_module(impl_module)
    return mod.run_single(workload_idx)


@app.local_entrypoint()
def main():
    trace_json = run_intra.remote(IMPL_MODULE, WORKLOAD_IDX)
    Path(f"reports/{OUT_FILENAME}").write_text(trace_json)
    print(f"Saved → reports/{OUT_FILENAME}")

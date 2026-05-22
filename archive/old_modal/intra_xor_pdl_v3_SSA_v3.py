"""Modal runner: intra profiling for SSA_v3 (smem_partial vectorized store)."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 16  # 0-based → workload #17 (uuid=564007ac, T=8)
IMPL_MODULE  = "src.kernels.kv_split_xor_pdl_intra_v3_SSA_v3"
OUT_FILENAME = "intra_kv_split_xor_pdl_v3_SSA_v3_w17.json"


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

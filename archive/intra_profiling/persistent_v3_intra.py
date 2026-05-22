"""Modal runner: intra-phase profiling for fused_persistent_v3 on WL20.

Workload 20 (index 19): uuid=7a389715  T=8  MaxValid=[8,11,11,16,1641,73,1,1]
Very unbalanced — tok4 has 1641 valid entries, six others have ≤16.

Phases logged per CTA (128 CTAs total):
  startup  — smem_sparse + q_nope/q_pe load + valid-count reduce (once)
  score    — K·q dot products (per task × 8 rounds)
  softmax  — max + exp + sum reductions (per task)
  output   — weighted V accumulation → smem_partial (per task)
  reduce   — cross-warp smem_partial → partial_out / direct output (per task)

Output: reports/intra_persistent_v3_w19.json  (Perfetto-compatible trace)

Usage:
    modal run src/modal/persistent_v3_intra.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 19   # 0-based → WL20: uuid=7a389715  T=8  MaxValid=[8,11,11,16,1641,73,1,1]
IMPL_MODULE  = "src.kernels.fused_persistent_v3_intra"


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
def run_intra(impl_module: str, workload_idx: int):
    import sys
    sys.path.insert(0, "/app")
    from importlib import import_module
    mod = import_module(impl_module)
    return mod.run_single(workload_idx)


@app.local_entrypoint()
def main():
    out_path = Path("reports/intra_persistent_v3_w19.json")
    out_path.parent.mkdir(exist_ok=True)
    print(f"\n{'='*60}\nProfiling {IMPL_MODULE}  WL{WORKLOAD_IDX + 1}\n{'='*60}")
    trace_json = run_intra.remote(IMPL_MODULE, WORKLOAD_IDX)
    out_path.write_text(trace_json)
    print(f"Saved trace to {out_path}")

"""Modal runner: intra-phase profiling of the smem_sparse (upfront) phase.

Runs the standalone smem_sparse_intra kernel against WL17 and prints:
  q_load      — time to load q_nope + q_pe into smem     (thread-0 / T=0 group)
  sparse_load — time to load sparse_indices + count valid (thread-0 / T=0 group)
  sync_wait   — time blocked at final sync_threads        (straggler overhead)

Usage:
    modal run src/modal/smem_sparse_intra.py
Output:
    reports/intra_smem_sparse_w17.json
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 16   # 0-based → workload 17 (uuid=564007ac, T=8)


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_intra(workload_idx: int):
    import sys
    sys.path.insert(0, "/app")
    from src.kernels.smem_sparse_intra import run_single
    return run_single(workload_idx)


@app.local_entrypoint()
def main():
    print(f"\n{'='*60}\nProfiling smem_sparse upfront phase  WL{WORKLOAD_IDX + 1}\n{'='*60}")
    trace_json = run_intra.remote(WORKLOAD_IDX)
    out_path = Path("reports/intra_smem_sparse_w17.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(trace_json)
    print(f"Saved trace to {out_path}")

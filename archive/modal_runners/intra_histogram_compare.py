"""Modal runner: compare histogram (single-CTA) vs histogram_dsmem (4-CTA cluster).

Both use intra-kernel %globaltimer probing — the definitive GPU-time measurement.
Sweep matches the topk use case: seq_len ∈ [2049, 6000].
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_single(n: int):
    sys.path.insert(0, "/app")
    from src.kernels.histogram import run_case
    run_case(n=n)
    return "ok"


@app.function(image=image, gpu="B200:1", timeout=600)
def run_cluster(n: int):
    sys.path.insert(0, "/app")
    from src.kernels.histogram_dsmem import run_case
    run_case(n=n)
    return "ok"


@app.local_entrypoint()
def main():
    # Topk use case: per-batch slow-path with seq_len 2049-6000.
    cases = [2049, 3072, 4096, 5120, 6000]
    print("\n" + "="*70 + "\nSINGLE-CTA (histogram.py, grid=1)\n" + "="*70)
    for n in cases:
        print(f"\n--- N={n} ---")
        run_single.remote(n)
    print("\n" + "="*70 + "\nCLUSTER (histogram_dsmem.py, grid=cluster=4)\n" + "="*70)
    for n in cases:
        print(f"\n--- N={n} ---")
        run_cluster.remote(n)

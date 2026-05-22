"""Modal runner: single-cluster DSMEM histogram with intra-kernel probe timing.

Intra-kernel timing is the definitive GPU-time measurement. torch.cuda.Event
includes launch overhead which dominates for small N.
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_histogram(n: int):
    sys.path.insert(0, "/app")
    from src.kernels.histogram_dsmem import run_case
    run_case(n=n)
    return "ok"


@app.local_entrypoint()
def main():
    cases = [2049, 8192, 16384, 65536, 1 << 20]
    for n in cases:
        print(f"\n{'='*60}\nhistogram_dsmem  N={n}\n{'='*60}")
        run_histogram.remote(n)

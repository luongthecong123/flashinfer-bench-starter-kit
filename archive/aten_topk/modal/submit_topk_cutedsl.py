"""Modal runner: correctness + benchmark for topk_aten_cutedsl on B200.

Usage:
    modal run src/modal/submit_topk_cutedsl.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_topk_cutedsl():
    import sys
    sys.path.insert(0, "/app")
    from src.kernels.topk_aten_cutedsl import test_correctness, benchmark_vs_torch
    test_correctness()
    benchmark_vs_torch()


@app.local_entrypoint()
def main():
    run_topk_cutedsl.remote()

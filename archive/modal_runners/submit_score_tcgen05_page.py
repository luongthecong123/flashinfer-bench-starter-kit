"""submit_score_tcgen05_page.py — Modal runner: paged tcgen05 fp8 GEMM + scale.

Usage:
    modal run src/modal/submit_score_tcgen05_page.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_test():
    sys.path.insert(0, "/app")
    from src.kernels.score_scale_tcgen05_page import main
    main()


@app.local_entrypoint()
def go():
    run_test.remote()

"""Modal runner: single-block no-TMA tcgen05 score test.

Usage:
    modal run src/modal/score_tcgen05_direct_1block.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_score_test():
    import sys
    sys.path.insert(0, "/app")
    from src.kernels.score_tcgen05_direct_1block import main
    main()


@app.local_entrypoint()
def main():
    run_score_test.remote()

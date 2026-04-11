"""Modal runner: single-block UMMA TS-mode (A from TMEM) score kernel.

Usage:
    modal run src/modal/score_tcgen05_direct_tmem_1block.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_score_test():
    import sys, traceback
    sys.path.insert(0, "/app")
    try:
        from src.kernels.score_tcgen05_direct_tmem_1block import main
        main()
    except Exception as e:
        print("ERROR:", type(e).__name__, str(e))
        traceback.print_exc()
        raise RuntimeError(str(e)) from None


@app.local_entrypoint()
def main():
    run_score_test.remote()

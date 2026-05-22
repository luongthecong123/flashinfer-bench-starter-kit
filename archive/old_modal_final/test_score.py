"""Modal runner: test_score — (2,512)@(512,128) + (2,64)@(64,128) with cp.async prologue.

Usage:
    modal run src/modal/test_score.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_remote():
    sys.path.insert(0, "/app")
    from src.kernels import test_score as t
    ok = t.run()
    return ok


@app.local_entrypoint()
def main():
    ok = run_remote.remote()
    print(f"\nPASS={ok}")

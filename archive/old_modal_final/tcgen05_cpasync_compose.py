"""Modal runner: tcgen05_cpasync_compose kernel smoke test."""
import os
import sys

if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_remote():
    sys.path.insert(0, "/app")
    from src.kernels import tcgen05_cpasync_compose as t
    ok = t.run()
    return ok


@app.local_entrypoint()
def main():
    ok = run_remote.remote()
    print(f"\nPASS={ok}")

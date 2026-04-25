"""Modal runner: test zipped_divide(VEC=4) + fma_packed_f32x2 + warp_reduce_f32x2_add.

Usage:
    modal run src/modal/test_warp_reduce_packed.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_remote():
    sys.path.insert(0, "/app")
    from src.kernels import test_warp_reduce_packed as t
    ok = t.run()
    return ok


@app.local_entrypoint()
def main():
    ok = run_remote.remote()
    print(f"\nPASS={ok}")

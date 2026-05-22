"""Modal runner for load_a_v2_swizzled_cmp_faithful2."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_cmp(num_pg: int, seed: int) -> float:
    sys.path.insert(0, "/app")
    from src.kernels.load_a_v2_swizzled_cmp_faithful2 import run_test
    return run_test(num_pg=num_pg, seed=seed)


@app.local_entrypoint()
def main():
    for num_pg, seed in [(4, 0), (8, 1), (16, 2)]:
        print(f"\n=== num_pg={num_pg} seed={seed} ===")
        max_abs = run_cmp.remote(num_pg, seed)
        print(f"   → max_abs={max_abs}")

"""Modal runner: simple histogram POC test on B200."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_histogram(mode: str, n: int, grid: int):
    sys.path.insert(0, "/app")
    from src.kernels.histogram import run
    run(n=n, grid=grid, mode=mode)
    return "ok"


@app.local_entrypoint()
def main():
    # Sweep a few sizes/modes
    cases = [
        ("top8", 8192, 1),
        ("top8", 65536, 1),
        ("top8", 65536, 4),
        ("byte", 65536, 1),
        ("byte", 1 << 20, 16),
    ]
    for mode, n, grid in cases:
        print(f"\n{'='*60}\nhistogram mode={mode}  N={n}  grid={grid}\n{'='*60}")
        run_histogram.remote(mode, n, grid)

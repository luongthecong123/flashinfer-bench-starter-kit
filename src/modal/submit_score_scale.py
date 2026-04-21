"""Run score_scale.py on Modal B200."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_bench():
    import sys
    sys.path.insert(0, "/app")
    from src.kernels.score_scale_simt import main
    main()

@app.local_entrypoint()
def go():
    run_bench.remote()

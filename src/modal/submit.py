"""Submit a benchmark run to Modal B200.
Change IMPL_MODULE to select which implementation to benchmark.
Usage: modal run src/modal/submit.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

# ── Pick implementation (override via env: IMPL_MODULE=src.toco_impl modal run submit.py) ──
import os
IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.gather_dsa_impl")


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_bench(impl_module: str):
    import sys
    from pathlib import Path as P
    from importlib import import_module

    sys.path.insert(0, "/app")

    impl = import_module(impl_module)

    from src import utils
    from src.ref import run as ref_run
    utils.ref_fn = ref_run
    utils.CONTEST = P("/data")
    utils.JSONL = utils.CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    utils.CHECK = True
    utils.MEASURE = True
    utils.impl_fn = impl.run
    utils.main()


@app.local_entrypoint()
def main():
    run_bench.remote(IMPL_MODULE)

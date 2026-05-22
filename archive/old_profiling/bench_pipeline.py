"""Submit correctness + benchmark for fused_pipeline kernel.
Usage: modal run src/modal/bench_pipeline.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

IMPL_MODULE = "src.kernels.fused_pipeline_v2"


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

"""Modal runner: kv_split_umma_v3_stages_pe_prologue correctness check.

Invokes the same harness as src/modal/submit.py but pinned to this module.
Use START/END env vars to restrict workload range.
"""
import os, sys
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

IMPL_MODULE = "src.kernels.kv_split_umma_v3_stages_pe_prologue"
START = int(os.environ.get("START", 0))
END   = int(os.environ.get("END",   0))


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_bench(impl_module: str, start: int = 0, end: int = 0):
    from pathlib import Path as P
    from importlib import import_module
    sys.path.insert(0, "/app")

    impl = import_module(impl_module)

    from src import utils
    from src.ref import run as ref_run
    utils.ref_fn = ref_run
    utils.CONTEST = P("/data")
    utils.JSONL = (utils.CONTEST / "workloads" / "dsa_paged"
                   / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl")
    utils.CHECK = True
    utils.MEASURE = True
    utils.START = start
    utils.END = end
    utils.impl_fn = impl.run
    return utils.main()


@app.local_entrypoint()
def main():
    data = run_bench.remote(IMPL_MODULE, start=START, end=END)
    if data is None:
        return
    status = "ALL PASS" if data["all_pass"] else "SOME FAILED"
    print(f"\n>> {status}")

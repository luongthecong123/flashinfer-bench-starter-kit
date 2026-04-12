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
# IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_v3_thr_warpv3") # Correct code
# IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_v3_thr_warpv3_clc") # Correct code
# IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_v3_thr_warpv3_clc_v2") # Correct code
# IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_v3_thr_warpv3_clc_pdl") # a few cases have nan output
# IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_v3_thr_warpv3_clc_upfront") # Correct code
# IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_v3_thr_warpv3_pdl")
IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_xor")
# IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_xor_skew")
# IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_xor_pdl")
# IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_xor_pdl_v3_pro_v2")
# IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_xor_sentinel") # sentinel-skip variant of kv_split_xor

print("IMPL_MODULE: ", IMPL_MODULE)

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
    return utils.main()


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path as P

    data = run_bench.remote(IMPL_MODULE)
    if data is None:
        return

    # Derive short variant name from module path
    variant = IMPL_MODULE.replace("src.kernels.", "")

    csv_path = P("bench_results.csv")
    write_header = not csv_path.exists()

    # If file exists, remove old rows for this variant
    existing_rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r["variant"] != variant]
        write_header = True  # rewrite whole file

    rows = existing_rows
    for r in data["results"]:
        rows.append({
            "variant":  variant,
            "workload": r["workload"],
            "uuid":     r["uuid"],
            "T":        r["T"],
            "ref_ms":   f"{r.get('ref_ms', 0):.3f}",
            "impl_ms":  f"{r.get('impl_ms', 0):.3f}",
            "speedup":  f"{r.get('speedup', 0):.2f}",
        })

    fields = ["variant", "workload", "uuid", "T", "ref_ms", "impl_ms", "speedup"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    status = "ALL PASS" if data["all_pass"] else "SOME FAILED"
    print(f"\n>> {status} — wrote {variant} to {csv_path}")

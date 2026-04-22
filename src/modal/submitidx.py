"""Submit an indexer benchmark run to Modal B200.
Change IMPL_MODULE to select which indexer implementation to benchmark.
Usage: modal run src/modal/submitidx.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

# ── Pick implementation ──
IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.draftv4_hist_pdl")
START = int(os.environ.get("START", 0))   # 0-indexed
END   = int(os.environ.get("END", 0))     # 0 = all

print("IMPL_MODULE: ", IMPL_MODULE)
print(f"RANGE: {START}..{END or 'end'}")


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_bench(impl_module: str, start: int = 0, end: int = 0):
    import sys
    from pathlib import Path as P
    from importlib import import_module

    sys.path.insert(0, "/app")

    impl = import_module(impl_module)

    from src import idx_utils
    from src.kernels.idxer_ref import run as ref_run
    idx_utils.ref_fn = ref_run
    idx_utils.CONTEST = P("/data")
    idx_utils.JSONL = (idx_utils.CONTEST / "workloads" / "dsa_paged"
                       / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl")
    idx_utils.CHECK = True
    idx_utils.MEASURE = True
    idx_utils.START = start
    idx_utils.END = end
    idx_utils.impl_fn = impl.run
    return idx_utils.main()


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path as P

    data = run_bench.remote(IMPL_MODULE, start=START, end=END)
    if data is None:
        return

    variant = IMPL_MODULE.replace("src.kernels.", "")

    csv_path = P("bench_idx_results.csv")
    write_header = not csv_path.exists()

    existing_rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r["variant"] != variant]
        write_header = True

    rows = existing_rows
    for r in data["results"]:
        rows.append({
            "variant":    variant,
            "workload":   r["workload"],
            "uuid":       r["uuid"],
            "batch_size": r["batch_size"],
            "max_pages":  r["max_num_pages"],
            "num_pages":  r["num_pages"],
            "ref_ms":     f"{r.get('ref_ms', 0):.3f}",
            "impl_ms":    f"{r.get('impl_ms', 0):.3f}",
            "speedup":    f"{r.get('speedup', 0):.2f}",
        })

    fields = ["variant", "workload", "uuid", "batch_size", "max_pages",
              "num_pages", "ref_ms", "impl_ms", "speedup"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    status = "ALL PASS" if data["all_pass"] else "SOME FAILED"
    print(f"\n>> {status} — wrote {variant} to {csv_path}")

"""NCU comparison: kv_split_v3_thr_warpv3 vs kv_split_v3_thr_warpv3_clc.

Profiles workload #20 (7a389715, T=8, max_valid=[8,11,11,16,1641,73,1,1])
— the most imbalanced workload in the contest dataset.

Usage:
    modal run src/modal/ncu_clc_vs_thr_warpv3_w20.py
    modal run src/modal/ncu_clc_vs_thr_warpv3_w20.py --reps 5
"""
import sys, os, json, math, random
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, trace_volume, image

# Workload #20: T=8, most imbalanced (5 near-empty + 1 large + 1 medium + 2 tiny)
WORKLOAD_IDX = 19   # 0-based
WORKLOAD_UUID = "7a389715"
WORKLOAD_T    = 8
WORKLOAD_MAX_VALID = 1641

KERNELS = [
    ("thr_warpv3",     "src.kernels.kv_split_v3_thr_warpv3"),
    ("thr_warpv3_clc", "src.kernels.kv_split_v3_thr_warpv3_clc"),
]

_TARGET_TEMPLATE = """\
import sys, json, os
from pathlib import Path
sys.path.insert(0, "/app")
import torch
from importlib import import_module
from safetensors.torch import load_file

IMPL_MODULE  = "__IMPL_MODULE__"
WORKLOAD_IDX = __WORKLOAD_IDX__
CONTEST      = Path(os.environ.get("CONTEST_DIR", "/data"))

H, D, Dp, PS = 16, 512, 64, 64
SCALE = 0.1352337788608801
JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

workloads = [json.loads(l) for l in open(JSONL)]
w   = workloads[WORKLOAD_IDX]
ax  = w["workload"]["axes"]
inp = w["workload"]["inputs"]
T, P = ax["num_tokens"], ax["num_pages"]
print(f"WL{WORKLOAD_IDX+1}: uuid={w['workload']['uuid'][:8]} T={T} impl={IMPL_MODULE}")

q_nope = torch.randn(T, H, D,  dtype=torch.bfloat16, device="cuda")
q_pe   = torch.randn(T, H, Dp, dtype=torch.bfloat16, device="cuda")
ckv    = torch.randn(P, PS, D,  dtype=torch.bfloat16, device="cuda")
kpe    = torch.randn(P, PS, Dp, dtype=torch.bfloat16, device="cuda")
sf     = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
si     = sf[inp["sparse_indices"]["tensor_key"]].cuda()
output = torch.zeros(T, H, D,  dtype=torch.bfloat16, device="cuda")
lse    = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

run = import_module(IMPL_MODULE).run

run(q_nope, q_pe, ckv, kpe, si, SCALE, output, lse)
torch.cuda.synchronize()
"""


def _parse_ncu_csv(stdout):
    import csv, io
    lines = stdout.splitlines()
    header_idx = None
    for i, l in enumerate(lines):
        if l.startswith('"ID"'):
            header_idx = i
            break
    if header_idx is None:
        return [], []
    csv_lines = [l for l in lines[header_idx:] if not l.startswith("==")]
    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    all_rows = [row for row in reader if (row.get("Metric Name") or "").strip()]
    cute_rows = [row for row in all_rows
                 if "cutlass" in (row.get("Kernel Name") or "").lower()
                 or "nvjet"   in (row.get("Kernel Name") or "").lower()]
    return cute_rows, all_rows


def _row_to_us(row):
    val  = float(row["Metric Value"].strip().replace(",", ""))
    unit = (row.get("Metric Unit") or "").strip()
    if "nsecond" in unit or unit == "ns":
        val /= 1000.0
    return val


@app.function(image=image, gpu="B200:1", timeout=7200,
              volumes={"/data": trace_volume})
def benchmark():
    import subprocess, glob

    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    results = {}

    for label, impl_module in KERNELS:
        script = (
            _TARGET_TEMPLATE
            .replace("__IMPL_MODULE__",  impl_module)
            .replace("__WORKLOAD_IDX__", str(WORKLOAD_IDX))
        )
        target = f"/tmp/_bench_{label}_w{WORKLOAD_IDX}.py"
        with open(target, "w") as f:
            f.write(script)

        print(f"\n{'='*60}")
        print(f"NCU: {label} | WL{WORKLOAD_IDX+1} ({WORKLOAD_UUID}) T={WORKLOAD_T} max_valid={WORKLOAD_MAX_VALID}")
        print(f"{'='*60}")

        env = {**os.environ, "CONTEST_DIR": "/data"}
        cmd = [ncu,
               "--kernel-name", "regex:.*",
               "--metrics", "gpu__time_duration.sum",
               "--csv",
               "--target-processes", "all",
               "python", target]

        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=1800, env=env, cwd="/app")

        print(f"NCU exit={r.returncode}, stdout={len(r.stdout)}B")
        if r.stderr:
            print(f"STDERR (tail 500):\n{r.stderr[-500:]}")

        cute_rows, all_rows = _parse_ncu_csv(r.stdout)
        print(f"Captured {len(cute_rows)} CuTe launches out of {len(all_rows)} total")

        if not cute_rows:
            results[label] = None
            print("  NO CuTe kernels captured!")
            continue

        print("  Kernel breakdown:")
        kernel_times = {}
        total_us = 0.0
        for row in cute_rows:
            kname = (row.get("Kernel Name") or "?")[:50]
            t_us  = _row_to_us(row)
            print(f"    {kname}: {t_us:.2f} µs")
            kernel_times[kname] = t_us
            total_us += t_us

        print(f"  TOTAL: {total_us:.2f} µs")

        results[label] = {
            "total_us":     round(total_us, 2),
            "kernel_times": kernel_times,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — WL{WORKLOAD_IDX+1} ({WORKLOAD_UUID}) T={WORKLOAD_T} max_valid={WORKLOAD_MAX_VALID}")
    print(f"{'='*60}")
    for label, res in results.items():
        if res is None:
            print(f"  {label:<20}: FAILED")
        else:
            print(f"  {label:<20}: {res['total_us']:.2f} µs")

    if all(results.get(k) for k in [k for k, _ in KERNELS]):
        base = results[KERNELS[0][0]]["total_us"]
        new  = results[KERNELS[1][0]]["total_us"]
        sp   = base / new if new > 0 else 0
        print(f"\n  Speedup (clc / thr_warpv3): {sp:.3f}x")

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    raw = benchmark.remote()

    out_path = f"reports/ncu_clc_vs_thr_warpv3_w{WORKLOAD_IDX+1}.json"
    with open(out_path, "w") as f:
        f.write(raw)
    print(f"\nSaved to {out_path}")

"""bench_kvsplit.py: NCU benchmark — compare kv_split (2-kernel) vs kv_split_dsmem (1-kernel).

For kv_split: NCU captures *both* compute_partial and reduce_splits per rep.
We report individual kernel timings + total.

For kv_split_dsmem: NCU captures 1 kernel per rep.

Usage:
    modal run src/modal/bench_kvsplit.py              # single-shot
    modal run src/modal/bench_kvsplit.py --reps 10    # 10 reps, mean±std
"""
import sys, os, json, random, math
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, trace_volume, image

# Kernel variants
KERNELS = [
    ("kv_split",       "src.kernels.kv_split"),
    ("kv_split_dsmem", "src.kernels.kv_split_dsmem"),
]

# Same 6 workloads as bench_fused
WORKLOADS = [
    (1,  "9d4a5f21", 2,   18),
    (2,  "b7668cfd", 2,   52),
    (6,  "e6b849f2", 2,   92),
    (4,  "05f6de65", 2,  337),
    (9,  "385742b2", 8, 1044),
    (12, "02d6ae9c", 8, 2048),
]

# Inline target script
_TARGET_TEMPLATE = """\
import sys, json, os
from pathlib import Path
sys.path.insert(0, "/app")
import torch
from importlib import import_module
from safetensors.torch import load_file
from src.utils import WORKLOAD_INFO

IMPL_MODULE  = "{impl_module}"
WORKLOAD_IDX = {workload_idx}
REPS         = {reps}
CONTEST      = Path(os.environ.get("CONTEST_DIR", "/data"))

H, D, Dp, PS = 16, 512, 64, 64
SCALE = 0.1352337788608801
JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

workloads = [json.loads(l) for l in open(JSONL)]
w   = workloads[WORKLOAD_IDX]
ax  = w["workload"]["axes"]
inp = w["workload"]["inputs"]
T, P = ax["num_tokens"], ax["num_pages"]
_uuid, _T, _max_valid = WORKLOAD_INFO[WORKLOAD_IDX]
print(f"WL{{WORKLOAD_IDX+1}}: uuid={{_uuid[:8]}} T={{_T}} MaxValid={{_max_valid}} impl={{IMPL_MODULE}} reps={{REPS}}")

q_nope = torch.randn(T, H, D,  dtype=torch.bfloat16, device="cuda")
q_pe   = torch.randn(T, H, Dp, dtype=torch.bfloat16, device="cuda")
ckv    = torch.randn(P, PS, D,  dtype=torch.bfloat16, device="cuda")
kpe    = torch.randn(P, PS, Dp, dtype=torch.bfloat16, device="cuda")
sf     = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
si     = sf[inp["sparse_indices"]["tensor_key"]].cuda()
output = torch.zeros(T, H, D,  dtype=torch.bfloat16, device="cuda")
lse    = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

run = import_module(IMPL_MODULE).run

for _ in range(REPS):
    run(q_nope, q_pe, ckv, kpe, si, SCALE, output, lse)
    torch.cuda.synchronize()
"""


def _parse_ncu_csv(stdout):
    """Parse NCU CSV, strip ==PROF== lines and non-CSV preamble."""
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
    csv_text = "\n".join(csv_lines)
    reader = csv.DictReader(io.StringIO(csv_text))
    all_rows = [row for row in reader if (row.get("Metric Name") or "").strip()]
    cute_rows = [row for row in all_rows
                 if "cutlass" in (row.get("Kernel Name") or "").lower()
                 or "nvjet" in (row.get("Kernel Name") or "").lower()]
    return cute_rows, all_rows


def _row_to_us(row):
    """Extract duration in µs from an NCU CSV row."""
    val = float(row["Metric Value"].strip().replace(",", ""))
    unit = (row.get("Metric Unit") or "").strip()
    if "nsecond" in unit or unit == "ns":
        val /= 1000.0
    return val


def _get_kernel_short_name(name):
    """Classify a kernel row by its name."""
    lower = name.lower()
    if "reduce" in lower:
        return "reduce"
    return "compute"


@app.function(image=image, gpu="B200:1", timeout=7200,
              volumes={"/data": trace_volume})
def benchmark(reps: int = 1):
    import subprocess, glob

    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    results = {}

    kernel_order = list(range(len(KERNELS)))
    if reps > 1:
        random.seed(42)
        random.shuffle(kernel_order)

    for ki in kernel_order:
        label, impl_module = KERNELS[ki]
        results[label] = {}

        wl_order = list(range(len(WORKLOADS)))
        if reps > 1:
            random.shuffle(wl_order)

        for wi in wl_order:
            wl_idx, uuid, T, max_valid = WORKLOADS[wi]
            tag = f"WL{wl_idx+1}(max={max_valid})"

            script = _TARGET_TEMPLATE.format(
                impl_module=impl_module,
                workload_idx=wl_idx,
                reps=reps,
            )
            target = f"/tmp/_bench_kvsplit_{label}_w{wl_idx}.py"
            with open(target, "w") as f:
                f.write(script)

            print(f"\n{'='*60}")
            print(f"NCU: {label} | {tag} | T={T} | reps={reps}")
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

            print(f"NCU exit={r.returncode}, stdout={len(r.stdout)}B, stderr={len(r.stderr)}B")
            if r.stderr:
                print(f"STDERR (tail 500): {r.stderr[-500:]}")

            cute_rows, all_rows = _parse_ncu_csv(r.stdout)
            print(f"Captured {len(cute_rows)} CuTe launches out of {len(all_rows)} total")

            if not cute_rows:
                results[label][tag] = None
                print("  NO CuTe kernel captured!")
                continue

            # For kv_split (2-kernel): group compute vs reduce
            # For kv_split_dsmem (1-kernel): all are "fused"
            is_two_kernel = (label == "kv_split")

            if is_two_kernel:
                # Group by kernel type (compute vs reduce)
                compute_vals = []
                reduce_vals  = []
                for row in cute_rows:
                    kname = row.get("Kernel Name", "")
                    us = _row_to_us(row)
                    if "reduce" in kname.lower():
                        reduce_vals.append(us)
                    else:
                        compute_vals.append(us)

                def _stats(vals, name):
                    if not vals:
                        return None
                    if reps == 1:
                        return round(vals[-1], 2)
                    mean = sum(vals) / len(vals)
                    if len(vals) > 1:
                        var = sum((v - mean)**2 for v in vals) / (len(vals) - 1)
                        std = math.sqrt(var)
                    else:
                        std = 0.0
                    print(f"  {name}: {mean:.2f} ± {std:.2f} µs  (n={len(vals)})")
                    return {"mean": round(mean, 2), "std": round(std, 2), "count": len(vals)}

                c_stat = _stats(compute_vals, "compute_partial")
                r_stat = _stats(reduce_vals,  "reduce_splits")

                # Total
                total_vals = []
                for i in range(min(len(compute_vals), len(reduce_vals))):
                    total_vals.append(compute_vals[i] + reduce_vals[i])
                t_stat = _stats(total_vals, "total")

                results[label][tag] = {
                    "compute": c_stat,
                    "reduce": r_stat,
                    "total": t_stat,
                }
            else:
                # Single fused kernel
                vals = [_row_to_us(row) for row in cute_rows]
                if reps == 1:
                    results[label][tag] = round(vals[-1], 2)
                    print(f"  Duration: {vals[-1]:.2f} µs")
                else:
                    mean = sum(vals) / len(vals)
                    if len(vals) > 1:
                        var = sum((v - mean)**2 for v in vals) / (len(vals) - 1)
                        std = math.sqrt(var)
                    else:
                        std = 0.0
                    results[label][tag] = {"mean": round(mean, 2), "std": round(std, 2), "count": len(vals)}
                    print(f"  Duration: {mean:.2f} ± {std:.2f} µs  (n={len(vals)})")

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(reps: int = 1):
    raw = benchmark.remote(reps=reps)

    if reps == 1:
        out_path = "reports/bench_kvsplit.json"
    else:
        out_path = f"reports/bench_kvsplit_{reps}reps.json"

    with open(out_path, "w") as f:
        f.write(raw)
    print(f"\nSaved results to {out_path}")

"""bench_fused_v3.py: NCU benchmark — tiny5v2 vs ldgv1b vs thr_warpv2 vs thr_warpv3.

Focused comparison: scalar-score baseline vs LDG.128 variants with bf16/fp32 precision.

Usage:
    modal run src/modal/bench_fused_v3.py              # single-shot
    modal run src/modal/bench_fused_v3.py --reps 10    # 10 reps, mean±std
"""
import sys, os, json, random, math
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, trace_volume, image

# 4 kernel variants to compare
KERNELS = [
    ("tiny5v2",             "src.kernels.fused_tiny5v2"),
    ("tiny_ldgv1b",         "src.kernels.fused_tiny_ldgv1b"),
    ("tiny_thr_warpv2",     "src.kernels.fused_tiny_thr_warpv2"),
    ("tiny_thr_warpv3",     "src.kernels.fused_tiny_thr_warpv3"),
]

# 6 workloads: (index_in_JSONL, uuid, T, max_valid)
WORKLOADS = [
    (1,  "9d4a5f21", 2,   18),    # ~16
    (2,  "b7668cfd", 2,   52),    # ~64
    (6,  "e6b849f2", 2,   92),    # ~128
    (4,  "05f6de65", 2,  337),    # ~337
    (9,  "385742b2", 8, 1044),    # ~1024
    (12, "02d6ae9c", 8, 2048),    # ~2048
]

# Inline target script template — runs one workload with one kernel, N times
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
    # Find the actual CSV header line (starts with '"ID"')
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


@app.function(image=image, gpu="B200:1", timeout=7200,
              volumes={"/data": trace_volume})
def benchmark(reps: int = 1):
    import subprocess, glob

    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    results = {}

    # Shuffle kernel order to avoid bias
    kernel_order = list(range(len(KERNELS)))
    if reps > 1:
        random.seed(42)
        random.shuffle(kernel_order)

    for ki in kernel_order:
        label, impl_module = KERNELS[ki]
        results[label] = {}

        # Shuffle workload order too
        wl_order = list(range(len(WORKLOADS)))
        if reps > 1:
            random.shuffle(wl_order)

        for wi in wl_order:
            wl_idx, uuid, T, max_valid = WORKLOADS[wi]
            tag = f"WL{wl_idx+1}(max={max_valid})"

            # Write target script
            script = _TARGET_TEMPLATE.format(
                impl_module=impl_module,
                workload_idx=wl_idx,
                reps=reps,
            )
            target = f"/tmp/_bench_fused_{label}_w{wl_idx}.py"
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
                print(f"STDERR (tail 300): {r.stderr[-300:]}")

            cute_rows, all_rows = _parse_ncu_csv(r.stdout)
            print(f"Captured {len(cute_rows)} CuTe launches out of {len(all_rows)} total (expected {reps})")

            if cute_rows:
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
            else:
                results[label][tag] = None
                print(f"  NO CuTe kernel captured!")

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(reps: int = 1):
    raw = benchmark.remote(reps=reps)

    if reps == 1:
        out_path = "reports/bench_fused_v3.json"
    else:
        out_path = f"reports/bench_fused_v3_{reps}reps.json"

    with open(out_path, "w") as f:
        f.write(raw)
    print(f"\nSaved results to {out_path}")
    print(raw)

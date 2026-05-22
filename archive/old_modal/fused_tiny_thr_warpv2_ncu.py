"""NCU full-report profiling for fused_tiny_thr_warpv2 on WL13 (max_valid=2048).
Emits .ncu-rep for SASS inspection.
Usage: modal run src/modal/fused_tiny_thr_warpv2_ncu.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

IMPL_MODULE  = "src.kernels.fused_tiny_thr_warpv2"
WORKLOAD_IDX = 12   # WL13: 02d6ae9c  T=8  MaxValid=2048

_RUN_ONCE = """\
import sys, json, os
from pathlib import Path
sys.path.insert(0, "/app")
import torch
from importlib import import_module
from safetensors.torch import load_file
from src.utils import WORKLOAD_INFO

IMPL_MODULE  = os.environ["IMPL_MODULE"]
WORKLOAD_IDX = int(os.environ["WORKLOAD_IDX"])
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
print(f"Workload {WORKLOAD_IDX + 1}: MaxValid={_max_valid}  impl={IMPL_MODULE}")

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

"""


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_ncu(impl_module: str, workload_idx: int):
    import subprocess, glob, os, csv, io
    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]

    target = "/tmp/_ncu_target.py"
    with open(target, "w") as f:
        f.write(_RUN_ONCE)

    env = {**os.environ,
           "WORKLOAD_IDX": str(workload_idx),
           "CONTEST_DIR": "/data",
           "IMPL_MODULE": impl_module}
    cmd = [ncu, "--set", "full", "--target-processes", "all",
           "--print-summary", "per-kernel",
           "--kernel-name", "regex:.*(nvjet|cublas|cudnn|cutlass|gemm|Gemm|GEMM|sgemm|dgemm|hgemm|bmm|triton).*",
           "--import-source", "yes",
           "--source-folders", "/app/src",
           "-f", "--export", "/tmp/ncu_out",
           "python", target]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=540, env=env, cwd="/app/src")
    print(r.stdout[-5000:] if len(r.stdout) > 5000 else r.stdout)
    if r.stderr:
        print(f"STDERR (last 2000): {r.stderr[-2000:]}")
    print(f"exit: {r.returncode}")

    dur = subprocess.run([ncu, "--import", "/tmp/ncu_out.ncu-rep", "--csv",
                          "--metrics", "gpu__time_duration.sum"],
                         capture_output=True, text=True, timeout=60)
    for row in csv.DictReader(io.StringIO(dur.stdout)):
        if row.get("Metric Name", "").strip():
            print(f"\nDuration: {row['Metric Value']} {row['Metric Unit']}")

    rules = subprocess.run([ncu, "--import", "/tmp/ncu_out.ncu-rep", "--csv"],
                           capture_output=True, text=True, timeout=60)
    bottlenecks = [
        row for row in csv.DictReader(io.StringIO(rules.stdout))
        if (row.get("Estimated Speedup") or "").strip()
        and (row.get("Rule Description") or "").strip()
    ]
    bottlenecks.sort(key=lambda r: float(r["Estimated Speedup"] or 0), reverse=True)
    print("\nTop 3 bottlenecks:")
    for row in bottlenecks[:3]:
        print(f"  [{row['Estimated Speedup']}%] {row['Rule Description'][:120]}")

    rep_files = glob.glob("/tmp/ncu_out*.ncu-rep")
    if rep_files:
        with open(rep_files[0], "rb") as f:
            data = f.read()
        print(f"Report size: {len(data)} bytes")
        return data
    else:
        print("No .ncu-rep file found!")
        return None


@app.local_entrypoint()
def main():
    data = run_ncu.remote(IMPL_MODULE, WORKLOAD_IDX)
    if data:
        impl_short = IMPL_MODULE.split(".")[-1]
        out_path = f"reports/ncu_{impl_short}_w{WORKLOAD_IDX}.ncu-rep"
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved NCU report to {out_path} ({len(data)} bytes)")

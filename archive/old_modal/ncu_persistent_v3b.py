"""Quick NCU runner for v3b — reuses ncu_persistent_v4fa4.py pattern."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image, get_ncu_compute_cmd

WORKLOAD_IDX = 19

_RUN_SCRIPT = """\
import sys, json, os
sys.path.insert(0, "/app")
import torch
from importlib import import_module
from safetensors.torch import load_file
from pathlib import Path

IMPL_MODULE  = os.environ["IMPL_MODULE"]
WORKLOAD_IDX = int(os.environ["WORKLOAD_IDX"])
CONTEST      = Path("/data")

H, D, Dp, PS = 16, 512, 64, 64
SCALE = 0.1352337788608801
JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

workloads = [json.loads(l) for l in open(JSONL)]
w   = workloads[WORKLOAD_IDX]
ax  = w["workload"]["axes"]
inp = w["workload"]["inputs"]
T, P = ax["num_tokens"], ax["num_pages"]

print(f"Workload {WORKLOAD_IDX + 1}: T={T}  impl={IMPL_MODULE}")

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

run(q_nope, q_pe, ckv, kpe, si, SCALE, output, lse)
torch.cuda.synchronize()
print("done")
"""


def _run_ncu(impl_module, workload_idx, out_stem):
    import subprocess, glob, os, csv, io
    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    target = f"/tmp/_ncu_{out_stem}.py"
    with open(target, "w") as f:
        f.write(_RUN_SCRIPT)
    env = {**os.environ, "WORKLOAD_IDX": str(workload_idx), "CONTEST_DIR": "/data", "IMPL_MODULE": impl_module}
    out_rep = f"/tmp/ncu_{out_stem}"
    cmd = get_ncu_compute_cmd(ncu, target, out_rep)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=540, env=env, cwd="/app/src")
    tail = r.stdout[-8000:] if len(r.stdout) > 8000 else r.stdout
    print(f"\n{'='*60}\n  NCU: {impl_module}\n{'='*60}")
    print(tail)
    if r.stderr:
        print(f"STDERR: {r.stderr[-3000:]}")
    print(f"exit: {r.returncode}")
    dur = subprocess.run([ncu, "--import", f"{out_rep}.ncu-rep", "--csv", "--metrics", "gpu__time_duration.sum"],
        capture_output=True, text=True, timeout=60)
    print("\n-- Duration by kernel --")
    for row in csv.DictReader(io.StringIO(dur.stdout)):
        if row.get("Metric Name", "").strip():
            print(f"  {row.get('Kernel Name','')[:60]:<60}  {row['Metric Value']:>10} {row['Metric Unit']}")
    rules = subprocess.run([ncu, "--import", f"{out_rep}.ncu-rep", "--csv"], capture_output=True, text=True, timeout=60)
    bottlenecks = [row for row in csv.DictReader(io.StringIO(rules.stdout))
        if (row.get("Estimated Speedup") or "").strip() and (row.get("Rule Description") or "").strip()]
    bottlenecks.sort(key=lambda r: float(r["Estimated Speedup"] or 0), reverse=True)
    print("\n-- Top 5 bottlenecks --")
    for row in bottlenecks[:5]:
        print(f"  [{row['Estimated Speedup']:>6}%] {row['Rule Description'][:100]}")
    rep_files = glob.glob(f"{out_rep}*.ncu-rep")
    if rep_files:
        with open(rep_files[0], "rb") as f:
            return f.read()
    return None


@app.function(image=image, gpu="B200:1", timeout=1200, volumes={"/data": trace_volume})
def run_ncu_v3b(workload_idx):
    import sys
    sys.path.insert(0, "/app")
    return _run_ncu("src.kernels.fused_persistent_v3b", workload_idx, f"persistent_v3b_w{workload_idx}")


@app.local_entrypoint()
def main():
    data = run_ncu_v3b.remote(WORKLOAD_IDX)
    out_dir = Path("reports"); out_dir.mkdir(exist_ok=True)
    if data:
        stem = f"ncu_persistent_v3b_w{WORKLOAD_IDX}.ncu-rep"
        (out_dir / stem).write_bytes(data)
        print(f"Saved {out_dir / stem}  ({len(data)} bytes)")
    else:
        print("No report generated")

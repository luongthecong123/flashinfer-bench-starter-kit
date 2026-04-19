"""NCU profiling for score_scale_full_bt_ws_cpasync on Modal B200.
Single run — ncu's replay does internal warm-up, no manual loop needed.
Usage: modal run src/modal/ncu_score_scale_full_bt_ws_cpasync.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image, get_ncu_compute_cmd

# Pick one representative workload for the captured kernel launch.
WORKLOAD_LABEL = "WL 64 backwards-jump pg=82"
BLOCK_TABLE_LIST = (
    list(range(44, 65)) + [25, 18] + list(range(65, 95)) + [42, 33] + list(range(95, 122))
)
NUM_PAGES_POOL = 11923

_RUN_ONCE = f"""\
import sys, os
sys.path.insert(0, "/app")
import torch
from src.kernels.score_scale_full_bt_ws_cpasync import (
    get_compiled, PAGE_SIZE, N, HEAD_DIM, ROW_STRIDE,
)

device = "cuda"
torch.manual_seed(0)

bt_list = {BLOCK_TABLE_LIST!r}
num_pg_real = len(bt_list)
num_pg = num_pg_real if num_pg_real % 2 == 0 else num_pg_real + 1
bt_padded = bt_list + ([0] if num_pg != num_pg_real else [])

K_fp8_used    = torch.randn(num_pg_real, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
K_scales_used = torch.rand(num_pg_real, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5
kv_pool = torch.zeros({NUM_PAGES_POOL}, PAGE_SIZE, ROW_STRIDE, device=device, dtype=torch.uint8)
for i, pid in enumerate(bt_list):
    kv_pool[pid, :, :HEAD_DIM] = K_fp8_used[i].view(torch.uint8)
    kv_pool[pid, :, HEAD_DIM:HEAD_DIM + 4] = (
        K_scales_used[i].view(torch.uint8).reshape(PAGE_SIZE, 4)
    )
block_table = torch.tensor(bt_padded, dtype=torch.int32, device=device)
q_fp8 = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
w     = torch.randn(N, device=device, dtype=torch.float32)

print("Compiling...")
kernel, compiled = get_compiled()
workspace = kernel.workspace
print("Compile done. Launching kernel for NCU capture (workload: {WORKLOAD_LABEL})")

compiled(kv_pool, block_table, q_fp8, w, workspace)
torch.cuda.synchronize()
print("kernel done")
"""


@app.function(image=image, gpu="B200:1", timeout=600)
def run_ncu():
    import subprocess, glob, os, csv, io
    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]

    target = "/tmp/_ncu_score_scale_cpasync_target.py"
    with open(target, "w") as f:
        f.write(_RUN_ONCE)

    cmd = get_ncu_compute_cmd(ncu, target, "/tmp/ncu_score_scale_cpasync")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=540,
                       env=os.environ, cwd="/app/src")
    print(r.stdout[-8000:] if len(r.stdout) > 8000 else r.stdout)
    if r.stderr:
        print(f"STDERR (last 2000): {r.stderr[-2000:]}")
    print(f"exit: {r.returncode}")

    # Duration
    dur = subprocess.run([ncu, "--import", "/tmp/ncu_score_scale_cpasync.ncu-rep", "--csv",
                          "--metrics", "gpu__time_duration.sum"],
                         capture_output=True, text=True, timeout=60)
    for row in csv.DictReader(io.StringIO(dur.stdout)):
        if row.get("Metric Name", "").strip():
            print(f"\nDuration: {row['Metric Value']} {row['Metric Unit']}")

    # Top 3 bottlenecks
    rules = subprocess.run([ncu, "--import", "/tmp/ncu_score_scale_cpasync.ncu-rep", "--csv"],
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

    rep_files = glob.glob("/tmp/ncu_score_scale_cpasync*.ncu-rep")
    if rep_files:
        with open(rep_files[0], "rb") as f:
            data = f.read()
        print(f"Report size: {len(data)} bytes")
        return data
    print("No .ncu-rep file found!")
    return None


@app.local_entrypoint()
def main():
    data = run_ncu.remote()
    if data:
        out_path = "reports/ncu_score_scale_full_bt_ws_cpasync.ncu-rep"
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved NCU report to {out_path} ({len(data)} bytes)")

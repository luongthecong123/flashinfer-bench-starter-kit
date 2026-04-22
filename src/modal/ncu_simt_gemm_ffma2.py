"""NCU profiling for simt_gemm_ffma2 on Modal B200.
Usage: modal run src/modal/ncu_simt_gemm_ffma2.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image, get_ncu_compute_cmd

_RUN_ONCE = """\
import sys
sys.path.insert(0, "/app")
import torch
from cutlass.cute.runtime import from_dlpack
import cutlass.cute as cute
from src.kernels.simt_gemm_ffma2 import naive_ffma2

M, N, K = 1024, 1024, 1024

A = torch.randn((M, K), device="cuda", dtype=torch.float32)
B = torch.randn((N, K), device="cuda", dtype=torch.float32)
C = torch.empty((M, N), device="cuda", dtype=torch.float32)

A_ = from_dlpack(A, assumed_align=16)
B_ = from_dlpack(B, assumed_align=16)
C_ = from_dlpack(C, assumed_align=16)

print("Compiling...")
compiled = cute.compile(naive_ffma2, A_, B_, C_)
print("Compile done. Launching kernel for NCU capture...")

compiled(A_, B_, C_)
torch.cuda.synchronize()
print("kernel done")
"""


@app.function(image=image, gpu="B200:1", timeout=600)
def run_ncu():
    import subprocess, glob, os, csv, io
    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]

    target = "/tmp/_ncu_simt_ffma2_target.py"
    with open(target, "w") as f:
        f.write(_RUN_ONCE)

    cmd = get_ncu_compute_cmd(ncu, target, "/tmp/ncu_simt_ffma2")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=540,
                       env=os.environ, cwd="/app/src")
    print(r.stdout[-8000:] if len(r.stdout) > 8000 else r.stdout)
    if r.stderr:
        print(f"STDERR (last 2000): {r.stderr[-2000:]}")
    print(f"exit: {r.returncode}")

    # Duration
    dur = subprocess.run([ncu, "--import", "/tmp/ncu_simt_ffma2.ncu-rep", "--csv",
                          "--metrics", "gpu__time_duration.sum"],
                         capture_output=True, text=True, timeout=60)
    for row in csv.DictReader(io.StringIO(dur.stdout)):
        if row.get("Metric Name", "").strip():
            print(f"\nDuration: {row['Metric Value']} {row['Metric Unit']}")

    # Top 3 bottlenecks by estimated speedup
    rules = subprocess.run([ncu, "--import", "/tmp/ncu_simt_ffma2.ncu-rep", "--csv"],
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

    rep_files = glob.glob("/tmp/ncu_simt_ffma2*.ncu-rep")
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
    data = run_ncu.remote()
    if data:
        out_path = "reports/ncu_simt_gemm_ffma2.ncu-rep"
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved NCU report to {out_path} ({len(data)} bytes)")

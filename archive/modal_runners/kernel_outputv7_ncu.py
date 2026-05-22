"""kernel_outputv7_ncu: correctness test + NCU profile for kernel_outputv7.

Usage:
    modal run src/modal/kernel_outputv7_ncu.py              # NCU profile
    modal run src/modal/kernel_outputv7_ncu.py::check_fn    # correctness only
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, image

_RUN_ONCE = """\
import sys
sys.path.insert(0, "/app")
import torch
from src.kernels.kernel_outputv7 import run, N, D

torch.manual_seed(42)
scores = torch.randn(N,    dtype=torch.float32, device="cuda")
V      = torch.randn(N, D, dtype=torch.float32, device="cuda")
output = torch.zeros(D,    dtype=torch.float32, device="cuda")

run(scores, V, output)
"""


@app.function(image=image, gpu="B200:1", timeout=600)
def run_ncu():
    import subprocess, glob, csv, io

    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]

    target = "/tmp/_ncu_target.py"
    with open(target, "w") as f:
        f.write(_RUN_ONCE)

    cmd = [ncu, "--set", "full", "--target-processes", "all",
           "--print-summary", "per-kernel",
           "--kernel-name", "regex:.*(nvjet|cublas|cudnn|cutlass|gemm|Gemm|GEMM|sgemm|dgemm|hgemm|bmm|triton).*",
           "--import-source", "yes",
           "--source-folders", "/app/src",
           "-f", "--export", "/tmp/ncu_out",
           "python", target]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=540, cwd="/app/src")
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


@app.function(image=image, gpu="B200", timeout=120)
def check_fn():
    import torch
    from src.kernels.kernel_outputv7 import run, N, D

    torch.manual_seed(42)
    scores = torch.softmax(torch.randn(N, device="cuda"), dim=0).float()
    V      = torch.randn(N, D, dtype=torch.float32, device="cuda")
    output = torch.zeros(D,    dtype=torch.float32, device="cuda")
    ref    = (scores.unsqueeze(0) @ V).squeeze(0)

    run(scores, V, output)
    torch.cuda.synchronize()

    max_diff  = (output - ref).abs().max().item()
    mean_diff = (output - ref).abs().mean().item()
    print(f"max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
    assert max_diff < 1e-4, f"FAIL: max_diff={max_diff}"
    print("PASS ✓")


@app.local_entrypoint()
def main():
    data = run_ncu.remote()
    if data:
        out_path = "reports/ncu_kernel_outputv7.ncu-rep"
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved NCU report to {out_path} ({len(data)} bytes)")

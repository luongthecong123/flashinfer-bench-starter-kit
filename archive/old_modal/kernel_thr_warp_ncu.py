"""kernel_thr_warp_ncu: correctness test + NCU profile for kernel_thr_warp.

Mirrors ncu.py exactly — only the _RUN_ONCE script and tensor setup differ.

Usage:
    modal run src/modal/kernel_thr_warp_ncu.py              # NCU profile
    modal run src/modal/kernel_thr_warp_ncu.py::check_fn    # correctness only
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, image

_RUN_ONCE = """\
import sys
sys.path.insert(0, "/app")
import torch
from src.kernels.kernel_thr_warp import run, D, N

torch.manual_seed(42)
q      = torch.randn(D,    dtype=torch.bfloat16, device="cuda")
K      = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")
scores = torch.zeros(N,    dtype=torch.float32,  device="cuda")

run(q, K, scores)
"""


@app.function(image=image, gpu="B200:1", timeout=600)
def run_ncu():
    import subprocess, glob, os, csv, io

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

    # Duration
    dur = subprocess.run([ncu, "--import", "/tmp/ncu_out.ncu-rep", "--csv",
                          "--metrics", "gpu__time_duration.sum"],
                         capture_output=True, text=True, timeout=60)
    for row in csv.DictReader(io.StringIO(dur.stdout)):
        if row.get("Metric Name", "").strip():
            print(f"\nDuration: {row['Metric Value']} {row['Metric Unit']}")

    # Top 3 bottlenecks by estimated speedup
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
    from src.kernels.kernel_thr_warp import run, D, N

    torch.manual_seed(42)
    q      = torch.randn(D,    dtype=torch.bfloat16, device="cuda")
    K      = torch.randn(N, D, dtype=torch.bfloat16, device="cuda")
    scores = torch.zeros(N,    dtype=torch.float32,  device="cuda")
    ref    = q.float() @ K.float().T   # [D] @ [D, N] = [N]

    run(q, K, scores)
    torch.cuda.synchronize()

    max_diff  = (scores - ref).abs().max().item()
    mean_diff = (scores - ref).abs().mean().item()
    print(f"max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
    assert max_diff < 0.01, f"FAIL: max_diff={max_diff}"
    print("PASS ✓")


@app.local_entrypoint()
def main():
    data = run_ncu.remote()
    if data:
        out_path = "reports/ncu_kernel_thr_warp.ncu-rep"
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved NCU report to {out_path} ({len(data)} bytes)")

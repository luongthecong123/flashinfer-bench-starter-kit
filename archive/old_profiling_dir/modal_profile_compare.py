#!/usr/bin/env python3
"""
Profile gather_impl vs gather_dsa_impl on Modal B200.
Runs: torch.cuda.Event timing, NCU, and NSYS for both implementations.
Reports saved to reports/ with clear names.

Usage: modal run zen/modal_profile_compare.py
"""
import modal
from pathlib import Path

ZEN_DIR = Path(__file__).parent
ROOT_DIR = ZEN_DIR.parent
DEV_DIR = ROOT_DIR / "dev"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .apt_install("wget", "gnupg")
    .run_commands(
        # NCU
        "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /' > /etc/apt/sources.list.d/cuda.list",
        "apt-get update && apt-get install -y nsight-compute-2026.1.0",
        # NSYS
        "apt-key adv --fetch-keys https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub",
        "echo 'deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' > /etc/apt/sources.list.d/nsight.list",
        "apt-get update && apt-get install -y nsight-systems-2026.2.1 || apt-get install -y $(apt-cache search '^nsight-systems-202' | sort -r | head -1 | awk '{print $1}')",
    )
    .add_local_dir(ZEN_DIR, remote_path="/app/zen")
    .add_local_dir(DEV_DIR, remote_path="/app/dev")
)

app = modal.App("profile-gather-compare", image=image)


# ── Torch timing benchmark ──────────────────────────────────────────────────────

@app.function(gpu="B200:1", timeout=600)
def run_torch_bench():
    """Torch CUDA event timing for both implementations."""
    import sys, math
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/zen")
    sys.path.insert(0, "/app/dev")
    import torch

    T = 8
    P = 512
    H = 16
    D_ckv = 512
    D_kpe = 64
    PAGE_SIZE = 64
    TOPK = 2048
    sm_scale = 1.0 / math.sqrt(D_ckv + D_kpe)

    q_nope = torch.randn(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn(T, H, D_kpe, dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn(P, PAGE_SIZE, D_ckv, dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn(P, PAGE_SIZE, D_kpe, dtype=torch.bfloat16, device="cuda")
    total_kv = P * PAGE_SIZE
    sparse_indices = torch.randint(0, total_kv, (T, TOPK), dtype=torch.int32, device="cuda")

    from zen.gather_impl import run as gather_impl_run
    from zen.gather_dsa_impl import run as gather_dsa_impl_run

    def bench(fn, label, warmup=10, iters=50):
        output = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
        lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
        args = (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)

        # Warmup
        for _ in range(warmup):
            output.zero_(); lse.fill_(-float("inf"))
            fn(*args)
        torch.cuda.synchronize()

        # Timed runs
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
        for s, e in evs:
            output.zero_(); lse.fill_(-float("inf"))
            s.record()
            fn(*args)
            e.record()
        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in evs]
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        med = sorted(times)[len(times)//2]
        print(f"[{label}]  avg={avg:.3f}ms  min={mn:.3f}ms  max={mx:.3f}ms  median={med:.3f}ms  (iters={iters})")
        return {"label": label, "avg": avg, "min": mn, "max": mx, "median": med, "times": times}

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"T={T}, P={P}, TOPK={TOPK}\n")

    r1 = bench(gather_impl_run, "gather_impl (CuTe gather + torch.compile attn)")
    r2 = bench(gather_dsa_impl_run, "gather_dsa_impl (CuTe gather + CuTe fused DSA)")

    speedup = r1["avg"] / r2["avg"] if r2["avg"] > 0 else 0
    print(f"\nSpeedup (dsa/compile): {speedup:.3f}x")
    return r1, r2


# ── NCU profiling ────────────────────────────────────────────────────────────────

def run_ncu_profile(ncu, script, report_name):
    """Run NCU on a profile script, print metrics, return .ncu-rep bytes."""
    import subprocess, glob, os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app:/app/zen:/app/dev"

    report_path = f"/tmp/{report_name}"
    cmd = [
        ncu, "--set", "full", "--target-processes", "all",
        "--print-summary", "per-kernel",
        "--replay-mode", "kernel",
        "--import-source", "yes",
        "--source-folders", "/app/zen",
        "--export", report_path,
        "--nvtx", "--nvtx-include", "gather_impl_run/,gather_dsa_impl_run/",
        "python", script,
    ]

    print(f"\n{'='*60}")
    print(f"NCU PROFILING: {script} -> {report_name}")
    print(f"{'='*60}")

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    print(r.stdout[-5000:] if len(r.stdout) > 5000 else r.stdout)
    if r.stderr:
        print(f"STDERR (last 2000): {r.stderr[-2000:]}")
    print(f"exit: {r.returncode}")

    # Duration
    d = subprocess.run(
        [ncu, "--import", f"{report_path}.ncu-rep", "--csv",
         "--metrics", "gpu__time_duration.sum"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    print(f"\n===== DURATION ({report_name}) =====")
    print(d.stdout)

    # Memory metrics
    mem_metrics = ",".join([
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ])
    q = subprocess.run(
        [ncu, "--import", f"{report_path}.ncu-rep", "--csv",
         "--metrics", mem_metrics],
        capture_output=True, text=True, timeout=60, env=env,
    )
    print(f"\n===== MEMORY & THROUGHPUT ({report_name}) =====")
    print(q.stdout)

    rep_files = glob.glob(f"{report_path}*.ncu-rep")
    if rep_files:
        with open(rep_files[0], "rb") as f:
            data = f.read()
        print(f"Report size: {len(data)} bytes")
        return data
    print("No .ncu-rep file found!")
    return None


@app.function(gpu="B200:1", timeout=1800)
def run_ncu():
    import subprocess, glob
    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    print(f"Using NCU: {ncu}")

    d1 = run_ncu_profile(ncu, "/app/zen/profile_gather_impl.py", "ncu_gather_impl")
    d2 = run_ncu_profile(ncu, "/app/zen/profile_gather_dsa_impl.py", "ncu_gather_dsa_impl")
    return d1, d2


# ── NSYS profiling ───────────────────────────────────────────────────────────────

def run_nsys_profile(nsys, script, output_name):
    """Run nsys profile on a script, print reports, return .nsys-rep bytes."""
    import subprocess, glob, os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app:/app/zen:/app/dev"

    cmd = [
        nsys, "profile",
        "--inherit-environment=true",
        "-w", "true",
        "-t", "cuda,nvtx",
        "-s", "process-tree",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--backtrace=fp",
        "-x", "true",
        "--gpu-metrics-devices=0",
        "-o", f"/tmp/{output_name}",
        "python", script,
    ]

    print(f"\n{'='*60}")
    print(f"NSYS PROFILING: {script} -> {output_name}")
    print(f"{'='*60}")

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
    if r.stderr:
        print(f"STDERR (last 1000): {r.stderr[-1000:]}")
    print(f"exit: {r.returncode}")

    rep_files = glob.glob(f"/tmp/{output_name}*.nsys-rep")
    if rep_files:
        rep = rep_files[0]

        stats = subprocess.run(
            [nsys, "stats", "--report", "cuda_gpu_kern_sum", rep],
            capture_output=True, text=True, timeout=120,
        )
        print(f"\n===== CUDA KERNEL SUMMARY ({output_name}) =====")
        print(stats.stdout[-3000:] if len(stats.stdout) > 3000 else stats.stdout)

        nvtx = subprocess.run(
            [nsys, "stats", "--report", "nvtx_sum", rep],
            capture_output=True, text=True, timeout=120,
        )
        print(f"\n===== NVTX SUMMARY ({output_name}) =====")
        print(nvtx.stdout[-3000:] if len(nvtx.stdout) > 3000 else nvtx.stdout)

        with open(rep, "rb") as f:
            data = f.read()
        print(f"Report size: {len(data)} bytes")
        return data
    print("No .nsys-rep file found!")
    return None


@app.function(gpu="B200:1", timeout=900)
def run_nsys():
    nsys = "nsys"

    d1 = run_nsys_profile(nsys, "/app/zen/profile_gather_impl.py", "nsys_gather_impl")
    d2 = run_nsys_profile(nsys, "/app/zen/profile_gather_dsa_impl.py", "nsys_gather_dsa_impl")
    return d1, d2


# ── Entrypoint ───────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    from pathlib import Path

    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 1) Torch timing (fast)
    print("=" * 70)
    print("TORCH CUDA EVENT TIMING")
    print("=" * 70)
    r1, r2 = run_torch_bench.remote()

    # 2) NCU + NSYS in parallel (separate GPU containers)
    print("\n" + "=" * 70)
    print("NCU + NSYS PROFILING (parallel)")
    print("=" * 70)
    ncu_handle = run_ncu.spawn()
    nsys_handle = run_nsys.spawn()

    ncu_gather_impl, ncu_gather_dsa = ncu_handle.get()
    if ncu_gather_impl:
        p = reports_dir / "ncu_gather_impl.ncu-rep"
        p.write_bytes(ncu_gather_impl)
        print(f"Saved: {p} ({len(ncu_gather_impl)} bytes)")
    if ncu_gather_dsa:
        p = reports_dir / "ncu_gather_dsa_impl.ncu-rep"
        p.write_bytes(ncu_gather_dsa)
        print(f"Saved: {p} ({len(ncu_gather_dsa)} bytes)")

    nsys_gather_impl, nsys_gather_dsa = nsys_handle.get()
    if nsys_gather_impl:
        p = reports_dir / "nsys_gather_impl.nsys-rep"
        p.write_bytes(nsys_gather_impl)
        print(f"Saved: {p} ({len(nsys_gather_impl)} bytes)")
    if nsys_gather_dsa:
        p = reports_dir / "nsys_gather_dsa_impl.nsys-rep"
        p.write_bytes(nsys_gather_dsa)
        print(f"Saved: {p} ({len(nsys_gather_dsa)} bytes)")

    print("\n=== DONE ===")
    print("Reports saved:")
    print("  reports/ncu_gather_impl.ncu-rep")
    print("  reports/ncu_gather_dsa_impl.ncu-rep")
    print("  reports/nsys_gather_impl.nsys-rep")
    print("  reports/nsys_gather_dsa_impl.nsys-rep")

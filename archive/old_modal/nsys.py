"""nsys profiling on Modal B200.
Change IMPL_MODULE / WORKLOAD_IDX to select the implementation and workload.
Usage: modal run src/modal/nsys.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

# ── Pick implementation ──
IMPL_MODULE  = "src.gather_dsa_impl"
WORKLOAD_IDX = 0


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_nsys(impl_module: str, workload_idx: int):
    import subprocess, glob, os
    env = {**os.environ,
           "WORKLOAD_IDX": str(workload_idx),
           "CONTEST_DIR": "/data",
           "IMPL_MODULE": impl_module,
           "WARMUP": "3",
           "USE_NVTX": "1"}
    cmd = ["nsys", "profile",
           "--inherit-environment=true",
           "-w", "true",
           "-t", "cuda,nvtx",
           "-s", "process-tree",
           "--capture-range=cudaProfilerApi",
           "--capture-range-end=stop",
           "--backtrace=fp",
           "-x", "true",
           "--gpu-metrics-devices=0",
           "-o", "/tmp/nsys_out",
           "python", "/app/src/profiler.py"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=480, env=env, cwd="/app/src")
    print(r.stdout[-5000:] if len(r.stdout) > 5000 else r.stdout)
    if r.stderr:
        print(f"STDERR (last 2000): {r.stderr[-2000:]}")
    print(f"exit: {r.returncode}")

    # Kernel summary
    rep_files = glob.glob("/tmp/nsys_out*.nsys-rep")
    if rep_files:
        rep = rep_files[0]
        stats = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", rep],
                               capture_output=True, text=True, timeout=120)
        print("\n===== CUDA KERNEL SUMMARY =====")
        print(stats.stdout[-3000:] if len(stats.stdout) > 3000 else stats.stdout)

        nvtx = subprocess.run(["nsys", "stats", "--report", "nvtx_sum", rep],
                              capture_output=True, text=True, timeout=120)
        print("\n===== NVTX SUMMARY =====")
        print(nvtx.stdout[-3000:] if len(nvtx.stdout) > 3000 else nvtx.stdout)

        with open(rep, "rb") as f:
            data = f.read()
        print(f"Report size: {len(data)} bytes")
        return data
    else:
        print("No .nsys-rep file found!")
        return None


@app.local_entrypoint()
def main():
    data = run_nsys.remote(IMPL_MODULE, WORKLOAD_IDX)
    if data:
        impl_short = IMPL_MODULE.split(".")[-1]
        out_path = f"reports/nsys_{impl_short}_w{WORKLOAD_IDX}.nsys-rep"
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved nsys report to {out_path} ({len(data)} bytes)")

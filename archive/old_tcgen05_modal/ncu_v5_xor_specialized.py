"""NCU profiling for kv_split_tcgen05_exp_persistent_v5_xor_specialized on Modal B200.
Usage: modal run src/modal/ncu_v5_xor_specialized.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

IMPL_MODULE  = "src.kernels.kv_split_tcgen05_exp_persistent_v5_xor_specialized"
WORKLOAD_IDX = 22  # WL23 T=7

_NCU_KERNEL_REGEX = r".*(cutlass|kv_split|KvSplit).*"

# Inline target — compiles the v5 kernel and runs ONE launch.
_RUN_ONCE = """\
import sys, os
sys.path.insert(0, "/app")
from importlib import import_module

IMPL_MODULE  = os.environ["IMPL_MODULE"]
WORKLOAD_IDX = int(os.environ["WORKLOAD_IDX"])

mod = import_module(IMPL_MODULE)
mod.run_workload(WORKLOAD_IDX)
"""


@app.function(image=image, gpu="B200:1", timeout=900,
              volumes={"/data": trace_volume})
def run_ncu(impl_module: str, workload_idx: int):
    import subprocess, glob, csv, io
    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]

    target = "/tmp/_ncu_target.py"
    with open(target, "w") as f:
        f.write(_RUN_ONCE)

    out_rep = "/tmp/ncu_out"
    env = {**os.environ,
           "WORKLOAD_IDX": str(workload_idx),
           "CONTEST_DIR": "/data",
           "IMPL_MODULE": impl_module}
    cmd = [
        ncu,
        "--set", "full",
        "--target-processes", "all",
        "--print-summary", "per-kernel",
        "--kernel-name", f"regex:{_NCU_KERNEL_REGEX}",
        "--launch-count", "1",
        "--launch-skip-before-match", "3",  # skip warm-up launches
        "--import-source", "yes",
        "--source-folders", "/app/src",
        "-f", "--export", out_rep,
        "python", target,
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=840,
                       env=env, cwd="/app/src")
    print(r.stdout[-8000:] if len(r.stdout) > 8000 else r.stdout)
    if r.stderr:
        print(f"STDERR (last 2000): {r.stderr[-2000:]}")
    print(f"exit: {r.returncode}")

    rep = f"{out_rep}.ncu-rep"

    # Register / occupancy / SMEM metrics
    metrics = ",".join([
        "launch__registers_per_thread",
        "launch__shared_mem_per_block",
        "launch__shared_mem_per_block_static",
        "launch__shared_mem_per_block_dynamic",
        "launch__block_size",
        "launch__grid_size",
        "launch__thread_count",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_active.avg.pct_of_peak_sustained_active",
        "gpu__time_duration.sum",
    ])
    m = subprocess.run([ncu, "--import", rep, "--csv",
                        "--metrics", metrics],
                       capture_output=True, text=True, timeout=60)
    print("\n--- KEY METRICS ---")
    print(m.stdout)

    # Top bottlenecks by estimated speedup
    rules = subprocess.run([ncu, "--import", rep, "--csv"],
                           capture_output=True, text=True, timeout=60)
    bottlenecks = [
        row for row in csv.DictReader(io.StringIO(rules.stdout))
        if (row.get("Estimated Speedup") or "").strip()
        and (row.get("Rule Description") or "").strip()
    ]
    bottlenecks.sort(key=lambda r: float(r["Estimated Speedup"] or 0), reverse=True)
    print("\nTop 5 bottlenecks:")
    for row in bottlenecks[:5]:
        print(f"  [{row['Estimated Speedup']}%] {row['Rule Description'][:140]}")

    rep_files = glob.glob(f"{out_rep}*.ncu-rep")
    if rep_files:
        with open(rep_files[0], "rb") as f:
            data = f.read()
        print(f"Report size: {len(data)} bytes")
        return data
    print("No .ncu-rep file found!")
    return None


@app.local_entrypoint()
def main():
    data = run_ncu.remote(IMPL_MODULE, WORKLOAD_IDX)
    if data:
        impl_short = IMPL_MODULE.split(".")[-1]
        out_path = f"reports/ncu_{impl_short}_w{WORKLOAD_IDX + 1}.ncu-rep"
        os.makedirs("reports", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved NCU report to {out_path} ({len(data)} bytes)")

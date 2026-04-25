"""NCU register pressure comparison: v1 (VEC=8, 2 chunks) vs v2 (VEC=16, 1 chunk).

Usage:
    modal run src/modal/ncu_warp_reduce_packed.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image, get_ncu_compute_cmd

_RUN_V1 = """\
import sys
sys.path.insert(0, "/app")
from src.kernels.test_warp_reduce_packed_v1 import run
run()
"""

_RUN_V2 = """\
import sys
sys.path.insert(0, "/app")
from src.kernels.test_warp_reduce_packed_v2 import run
run()
"""

_RUN_V3 = """\
import sys
sys.path.insert(0, "/app")
from src.kernels.test_warp_reduce_packed_v3 import run
run()
"""

_RUN_V4 = """\
import sys
sys.path.insert(0, "/app")
from src.kernels.test_warp_reduce_packed_v4 import run
run()
"""



def _query_regs(ncu: str, rep: str, label: str):
    import subprocess, csv, io
    # Try both common metric names for register count
    for metric in ("launch__registers_per_thread",
                   "sm__registers_per_thread_allocated"):
        r = subprocess.run(
            [ncu, "--import", rep, "--csv", "--metrics", metric],
            capture_output=True, text=True, timeout=60,
        )
        rows = [row for row in csv.DictReader(io.StringIO(r.stdout))
                if row.get("Metric Value", "").strip()]
        if rows:
            print(f"\n[{label}] metric={metric}")
            for row in rows:
                kname = row.get("Kernel Name", "?")[:60]
                val   = row.get("Metric Value", "?")
                unit  = row.get("Metric Unit", "")
                print(f"  {kname:60s}  {val} {unit}")
            return
    # Fallback: dump raw stdout of first metric attempt
    r = subprocess.run(
        [ncu, "--import", rep, "--csv",
         "--metrics", "launch__registers_per_thread"],
        capture_output=True, text=True, timeout=60,
    )
    print(f"[{label}] raw CSV (first 800 chars):\n{r.stdout[:800]}")


@app.function(image=image, gpu="B200:1", timeout=600)
def run_ncu():
    import subprocess, glob

    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    print(f"Using ncu: {ncu}")

    for label, script_src, out_rep in [
        ("v4 VEC=8  SSA-reduce", _RUN_V4, "/tmp/ncu_v4"),
    ]:
        target = f"/tmp/_ncu_{label[:2].strip()}.py"
        with open(target, "w") as f:
            f.write(script_src)

        print(f"\n{'='*60}")
        print(f"Profiling {label}")
        print(f"{'='*60}")

        cmd = get_ncu_compute_cmd(ncu, target, out_rep)
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd="/app/src"
        )
        out = r.stdout
        print(out[-4000:] if len(out) > 4000 else out)
        if r.stderr:
            print(f"STDERR: {r.stderr[-1000:]}")
        print(f"exit: {r.returncode}")

        rep_file = f"{out_rep}.ncu-rep"
        if os.path.exists(rep_file):
            _query_regs(ncu, rep_file, label)
        else:
            print(f"No report at {rep_file}")

    # Return both reports for local download
    reports = {}
    for name, path in [("v4", "/tmp/ncu_v4.ncu-rep")]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                reports[name] = f.read()
    return reports


@app.local_entrypoint()
def main():
    reports = run_ncu.remote()
    for name, data in reports.items():
        out = f"reports/ncu_warp_reduce_packed_{name}.ncu-rep"
        with open(out, "wb") as f:
            f.write(data)
        print(f"Saved {out} ({len(data)} bytes)")
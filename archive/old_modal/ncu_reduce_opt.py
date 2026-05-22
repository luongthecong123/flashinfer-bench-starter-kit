"""NCU comparison: reduce_v0 vs reduce_v1/v2/v3 (Task C progressive reduce).

Each version is run under NCU to measure actual GPU kernel duration.
No contest workloads needed — uses synthetic partial_out/lse tensors.

Optimisations compared:
  v0  baseline      — thread 0 serial max+denom, smem sentinel, 512×8 exp()
  v1  fix Issue 3   — direct sentinel read (all threads); no smem+sync for sentinel
  v2  fix Issue 1+3 — 8-thread max+denom with warp-reduce
  v3  fix all       — precompute smem_scales[8]; 4096→8 exp() calls

Usage:
    modal run src/modal/ncu_reduce_opt.py
"""
import sys, os, json
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, image

VERSIONS = ["v0", "v1", "v2", "v3"]

# Standalone script compiled and run under NCU for each version.
# Uses compile_all() (compiles all 4 variants at once) and calls the one
# requested.  _fake_inputs() returns cutlass cute tensors matching the signatures.
_TARGET_TEMPLATE = """\
import sys
sys.path.insert(0, "/app")
import torch
import cutlass.cute as cute

from src.kernels.reduce_opt import _fake_inputs, compile_all

# Compile all variants (same JIT cost whether we compile one or four)
compiled = compile_all()
fn = compiled["{version}"]

# Run the chosen variant once to warm up, then under NCU
inp = _fake_inputs()
fn(*inp)
torch.cuda.synchronize()
fn(*inp)
torch.cuda.synchronize()
print("reduce_{version} done")
"""


def _parse_ncu_csv(stdout):
    import csv, io
    lines = stdout.splitlines()
    header_idx = None
    for i, l in enumerate(lines):
        if l.startswith('"ID"'):
            header_idx = i
            break
    if header_idx is None:
        return [], []
    csv_lines = [l for l in lines[header_idx:] if not l.startswith("==")]
    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    all_rows = [row for row in reader if (row.get("Metric Name") or "").strip()]
    cute_rows = [row for row in all_rows
                 if "cutlass" in (row.get("Kernel Name") or "").lower()
                 or "nvjet"   in (row.get("Kernel Name") or "").lower()]
    return cute_rows, all_rows


def _row_to_us(row):
    val  = float(row["Metric Value"].strip().replace(",", ""))
    unit = (row.get("Metric Unit") or "").strip()
    if "nsecond" in unit or unit == "ns":
        val /= 1000.0
    return val


def _keepalive_run(fn, *args, **kwargs):
    """Run fn(*args,**kwargs) in a background thread; print a '.' every 10 s
    so the gRPC stream never goes silent during compilations / subprocesses."""
    import threading, time
    result, error, done = [None], [None], [False]
    def _worker():
        try:    result[0] = fn(*args, **kwargs)
        except Exception as e: error[0] = e
        finally: done[0] = True
    threading.Thread(target=_worker, daemon=True).start()
    while not done[0]:
        print(".", end="", flush=True)
        time.sleep(10)
    print()
    if error[0]: raise error[0]
    return result[0]


@app.function(image=image, gpu="B200:1", timeout=7200)
def run_all():
    import subprocess, glob
    sys.path.insert(0, "/app")

    # ── Correctness check (in-process, no NCU overhead) ───────────────────────
    import torch

    print("Importing reduce_opt (compiling all variants) ...", end=" ", flush=True)
    import importlib
    reduce_mod = _keepalive_run(importlib.import_module, "src.kernels.reduce_opt")
    compile_all     = reduce_mod.compile_all
    check_correctness = reduce_mod.check_correctness
    print("Done.")

    print(f"\n{'='*60}")
    print("CORRECTNESS CHECK — reduce v0..v3")
    print(f"{'='*60}")
    compiled = compile_all()
    all_pass = check_correctness(compiled)
    print(f"Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")

    # ── NCU per version ───────────────────────────────────────────────────────
    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    results = {}

    for ver in VERSIONS:
        script = _TARGET_TEMPLATE.replace("{version}", ver)
        target = f"/tmp/_reduce_{ver}.py"
        with open(target, "w") as f:
            f.write(script)

        print(f"\n{'='*60}")
        print(f"NCU: reduce_{ver}  (T=8 H=16 S=8 D=512)")
        print(f"{'='*60}")

        cmd = [ncu,
               "--kernel-name", "regex:.*",
               "--metrics", "gpu__time_duration.sum",
               "--csv",
               "--target-processes", "all",
               "python", target]

        print(f"Running NCU (dots = keepalive every 10 s)...", end=" ", flush=True)
        r = _keepalive_run(subprocess.run, cmd,
                           capture_output=True, text=True,
                           timeout=1800, env=os.environ, cwd="/app")

        print(f"NCU exit={r.returncode}, stdout={len(r.stdout)}B")
        if r.stderr:
            print(f"STDERR (tail 300):\n{r.stderr[-300:]}")

        cute_rows, all_rows = _parse_ncu_csv(r.stdout)
        print(f"Captured {len(cute_rows)} CuTe launches out of {len(all_rows)} total")

        if not cute_rows:
            results[ver] = None
            print("  NO CuTe kernels captured!")
            continue

        print("  Kernel breakdown:")
        kernel_times = {}
        total_us = 0.0
        for row in cute_rows:
            kname = (row.get("Kernel Name") or "?")[:60]
            t_us  = _row_to_us(row)
            print(f"    {kname}: {t_us:.2f} µs")
            kernel_times[kname] = t_us
            total_us += t_us

        print(f"  TOTAL: {total_us:.2f} µs")
        results[ver] = {"total_us": round(total_us, 2), "kernel_times": kernel_times}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — reduce progressive optimisation")
    print(f"{'='*60}")
    base = (results.get("v0") or {}).get("total_us")
    for ver in VERSIONS:
        res = results.get(ver)
        if res is None:
            print(f"  reduce_{ver}: FAILED to capture")
        else:
            speedup = f"({base/res['total_us']:.3f}x vs v0)" if base else ""
            print(f"  reduce_{ver}: {res['total_us']:.2f} µs  {speedup}")

    return json.dumps({"correctness_pass": all_pass, "ncu": results}, indent=2)


@app.local_entrypoint()
def main():
    out = run_all.remote()
    print("\nFinal JSON result:")
    print(out)
    import pathlib
    out_path = pathlib.Path("reports/ncu_reduce_opt.json")
    out_path.write_text(out)
    print(f"Saved to {out_path}")

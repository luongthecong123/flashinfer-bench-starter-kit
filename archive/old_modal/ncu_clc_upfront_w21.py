"""NCU comparison: kv_split_v3_thr_warpv3_clc vs clc_upfront on WL#21.

Task B: upfront-preload variant eliminates per-tile sparse_indices GMEM reads
        (8 KB × num_tiles) by preloading all T tokens' indices into smem once.

Measures:
  thr_warpv3_clc      — baseline (per-tile sparse_indices load)
  thr_warpv3_clc_upfront — upfront preload (sparse_indices in smem)

Also runs correctness check vs thr_warpv3 on all 23 workloads first.

Usage:
    modal run src/modal/ncu_clc_upfront_w21.py
"""
import sys, os, json
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX  = 20   # 0-based → WL#21
WORKLOAD_UUID = "5096e459"
WORKLOAD_T    = 8
WORKLOAD_MAX_VALID = "[17,13,1887,16,180,1986,413,1]"

KERNELS = [
    ("clc",      "src.kernels.kv_split_v3_thr_warpv3_clc"),
    ("upfront",  "src.kernels.kv_split_v3_thr_warpv3_clc_upfront"),
]

_TARGET_TEMPLATE = """\
import sys, json, os
from pathlib import Path
sys.path.insert(0, "/app")
import torch
from importlib import import_module
from safetensors.torch import load_file

IMPL_MODULE  = "__IMPL_MODULE__"
WORKLOAD_IDX = __WORKLOAD_IDX__
CONTEST      = Path(os.environ.get("CONTEST_DIR", "/data"))

H, D, Dp, PS = 16, 512, 64, 64
SCALE = 0.1352337788608801
JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

workloads = [json.loads(l) for l in open(JSONL)]
w   = workloads[WORKLOAD_IDX]
ax  = w["workload"]["axes"]
inp = w["workload"]["inputs"]
T, P = ax["num_tokens"], ax["num_pages"]
print(f"WL{WORKLOAD_IDX+1}: uuid={w['workload']['uuid'][:8]} T={T} impl={IMPL_MODULE}")

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
    so the gRPC stream never goes silent during long compilations / subprocesses."""
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


@app.function(image=image, gpu="B200:1", timeout=7200,
              volumes={"/data": trace_volume})
def run_all():
    import subprocess, glob, importlib
    sys.path.insert(0, "/app")

    # ── Import kernels (may compile for 4+ minutes on cold container) ─────────
    import torch
    from pathlib import Path
    from safetensors.torch import load_file

    print("Importing kv_split_v3_thr_warpv3 (ref) ...", end=" ", flush=True)
    run_ref     = _keepalive_run(importlib.import_module, "src.kernels.kv_split_v3_thr_warpv3").run
    print("Importing kv_split_v3_thr_warpv3_clc_upfront ...", end=" ", flush=True)
    run_upfront = _keepalive_run(importlib.import_module, "src.kernels.kv_split_v3_thr_warpv3_clc_upfront").run
    print("Both kernels ready.")

    # ── Correctness check ─────────────────────────────────────────────────────

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    H, D, Dp, PS = 16, 512, 64, 64
    SCALE = 0.1352337788608801
    ATOL  = 0.01

    workloads = [json.loads(l) for l in open(JSONL)]
    print(f"\n{'='*60}")
    print("CORRECTNESS CHECK — clc_upfront vs baseline (all 23 workloads)")
    print(f"{'='*60}")
    print(f"{'#':>3} {'UUID':>10} {'T':>2}  {'out_err':>10} {'lse_err':>10}  {'Status':>6}")
    print("-" * 52)

    all_pass = True
    for i_w, w in enumerate(workloads):
        ax  = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        T, P = ax["num_tokens"], ax["num_pages"]
        uuid = w["workload"]["uuid"][:8]

        q_nope = torch.randn(T, H, D,  dtype=torch.bfloat16, device="cuda")
        q_pe   = torch.randn(T, H, Dp, dtype=torch.bfloat16, device="cuda")
        ckv    = torch.randn(P, PS, D,  dtype=torch.bfloat16, device="cuda")
        kpe    = torch.randn(P, PS, Dp, dtype=torch.bfloat16, device="cuda")
        sf     = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
        si     = sf[inp["sparse_indices"]["tensor_key"]].cuda()

        r_out = torch.zeros(T, H, D, dtype=torch.bfloat16, device="cuda")
        r_lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
        u_out = torch.zeros(T, H, D, dtype=torch.bfloat16, device="cuda")
        u_lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

        run_ref(q_nope, q_pe, ckv, kpe, si, SCALE, r_out, r_lse)
        run_upfront(q_nope, q_pe, ckv, kpe, si, SCALE, u_out, u_lse)
        torch.cuda.synchronize()

        o_err = (r_out.float() - u_out.float()).abs().max().item()
        l_err = (r_lse - u_lse).abs().max().item()
        ok    = o_err < ATOL and l_err < ATOL
        if not ok:
            all_pass = False
        print(f"{i_w+1:>3} {uuid:>10} {T:>2}  {o_err:>10.2e} {l_err:>10.2e}  {'PASS' if ok else 'FAIL':>6}")

    print(f"\nOverall upfront correctness: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")

    # ── NCU comparison ────────────────────────────────────────────────────────
    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    results = {}

    for label, impl_module in KERNELS:
        script = (
            _TARGET_TEMPLATE
            .replace("__IMPL_MODULE__",  impl_module)
            .replace("__WORKLOAD_IDX__", str(WORKLOAD_IDX))
        )
        target = f"/tmp/_bench_{label}_w{WORKLOAD_IDX}.py"
        with open(target, "w") as f:
            f.write(script)

        print(f"\n{'='*60}")
        print(f"NCU: {label} | WL{WORKLOAD_IDX+1} ({WORKLOAD_UUID}) T={WORKLOAD_T}")
        print(f"     max_valid={WORKLOAD_MAX_VALID}")
        print(f"{'='*60}")

        env = {**os.environ, "CONTEST_DIR": "/data"}
        cmd = [ncu,
               "--kernel-name", "regex:.*",
               "--metrics", "gpu__time_duration.sum",
               "--csv",
               "--target-processes", "all",
               "python", target]

        print(f"Running NCU (dots = keepalive every 10 s)...", end=" ", flush=True)
        r = _keepalive_run(subprocess.run, cmd,
                           capture_output=True, text=True,
                           timeout=1800, env=env, cwd="/app")

        print(f"NCU exit={r.returncode}, stdout={len(r.stdout)}B")
        if r.stderr:
            print(f"STDERR (tail 300):\n{r.stderr[-300:]}")

        cute_rows, all_rows = _parse_ncu_csv(r.stdout)
        print(f"Captured {len(cute_rows)} CuTe launches out of {len(all_rows)} total")

        if not cute_rows:
            results[label] = None
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
        results[label] = {"total_us": round(total_us, 2), "kernel_times": kernel_times}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY — WL{WORKLOAD_IDX+1} ({WORKLOAD_UUID}) T={WORKLOAD_T}")
    print(f"  max_valid={WORKLOAD_MAX_VALID}")
    print(f"{'='*60}")
    for label, res in results.items():
        if res is None:
            print(f"  {label:<22}: FAILED to capture")
        else:
            print(f"  {label:<22}: {res['total_us']:.2f} µs")

    base = (results.get("clc") or {}).get("total_us")
    upf  = (results.get("upfront") or {}).get("total_us")
    if base and upf:
        print(f"\n  upfront vs clc: {base/upf:.3f}x speedup")

    return json.dumps({"correctness_pass": all_pass, "ncu": results}, indent=2)


@app.local_entrypoint()
def main():
    out = run_all.remote()
    print("\nFinal JSON result:")
    print(out)
    import pathlib
    out_path = pathlib.Path("reports/ncu_clc_upfront_w21.json")
    out_path.write_text(out)
    print(f"Saved to {out_path}")

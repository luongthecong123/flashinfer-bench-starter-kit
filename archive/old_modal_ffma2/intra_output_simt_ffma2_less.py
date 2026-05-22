"""Modal runner: baseline ffma2 vs less (direct smem accumulate, racy).

Usage:
    modal run src/modal/intra_output_simt_ffma2_less.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_both():
    import sys, json
    sys.path.insert(0, "/app")

    from src.kernels import output_simt_ffma2_intra       # baseline (probed)
    from src.kernels import output_simt_ffma2_less_intra  # less (probed)

    print("\n" + "="*60)
    print("Baseline  — register accumulate + smem_partial + serial reduce")
    print("="*60)
    r_base = json.loads(output_simt_ffma2_intra.run_single())

    print("\n" + "="*60)
    print("Less      — direct smem_output accumulate (racy, no reduce phase)")
    print("="*60)
    r_less = json.loads(output_simt_ffma2_less_intra.run_single())

    return json.dumps({"baseline": r_base, "less": r_less}, indent=2)


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result_json = run_both.remote()
    data = json.loads(result_json)

    def get_phase(d, name):
        for p in d.get("probes", []):
            if p["phase"] == name:
                return p["us"]
        return None

    def fmt(label, d):
        print(f"\n{label}")
        for p in d["probes"]:
            print(f"  {p['phase']:8s}: {p['us']:7.3f} µs")
        return get_phase(d, "total")

    t_base = fmt("Baseline (register accum + reduce):", data["baseline"])
    t_less = fmt("Less     (direct smem accum, racy):", data["less"])

    r_base = get_phase(data["baseline"], "reduce")
    g_base = get_phase(data["baseline"], "gemv")
    g_less = get_phase(data["less"],     "gemv")

    print("\n--- Comparison ---")
    print(f"  baseline total: {t_base:.3f} µs   gemv: {g_base:.3f} µs   reduce: {r_base:.3f} µs")
    print(f"  less     total: {t_less:.3f} µs   gemv: {g_less:.3f} µs   (no reduce)")

    if t_base and t_less:
        winner = "less" if t_less < t_base else "baseline"
        ratio  = max(t_base, t_less) / min(t_base, t_less)
        print(f"  Winner: {winner}  ({ratio:.2f}x faster)")

    out_path = Path("reports/intra_output_simt_ffma2_less.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

"""Modal runner: baseline ffma2 vs atomic (direct gmem atomic add, no reduce).

Usage:
    modal run src/modal/intra_output_simt_ffma2_atomic.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_both():
    import sys, json
    sys.path.insert(0, "/app")

    from src.kernels import output_simt_ffma2_intra        # baseline (probed)
    from src.kernels import output_simt_ffma2_atomic_intra # atomic (probed)

    print("\n" + "="*60)
    print("Baseline  — register accum + smem_partial + serial reduce")
    print("="*60)
    r_base = json.loads(output_simt_ffma2_intra.run_single())

    print("\n" + "="*60)
    print("Atomic    — register accum + atomic add to gmem (no reduce)")
    print("="*60)
    r_atom = json.loads(output_simt_ffma2_atomic_intra.run_single())

    return json.dumps({"baseline": r_base, "atomic": r_atom}, indent=2)


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
            print(f"  {p['phase']:10s}: {p['us']:7.3f} µs")
        return get_phase(d, "total")

    t_base = fmt("Baseline (smem_partial + serial reduce):", data["baseline"])
    t_atom = fmt("Atomic   (gmem atomic add, no reduce):",   data["atomic"])

    r_base   = get_phase(data["baseline"], "reduce")
    r_atom   = get_phase(data["atomic"],   "atomic_out")
    g_base   = get_phase(data["baseline"], "gemv")
    g_atom   = get_phase(data["atomic"],   "gemv")

    print("\n--- Comparison ---")
    print(f"  baseline total: {t_base:.3f} µs   gemv: {g_base:.3f} µs   reduce: {r_base:.3f} µs")
    print(f"  atomic   total: {t_atom:.3f} µs   gemv: {g_atom:.3f} µs   atomic_out: {r_atom:.3f} µs")

    if t_base and t_atom:
        winner = "atomic" if t_atom < t_base else "baseline"
        ratio  = max(t_base, t_atom) / min(t_base, t_atom)
        print(f"  Winner: {winner}  ({ratio:.2f}x faster)")

    out_path = Path("reports/intra_output_simt_ffma2_atomic.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

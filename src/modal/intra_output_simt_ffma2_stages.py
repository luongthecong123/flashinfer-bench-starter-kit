"""Modal runner: baseline vs stages-2 vs stages-4.

Usage:
    modal run src/modal/intra_output_simt_ffma2_stages.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all():
    import sys, json
    sys.path.insert(0, "/app")

    from src.kernels import output_simt_ffma2_intra
    from src.kernels import output_simt_ffma2_stages

    print("\n" + "="*60)
    print("Baseline  — 1 pass, 16 regs/row, 1 sync")
    print("="*60)
    r_base = json.loads(output_simt_ffma2_intra.run_single())

    print("\n" + "="*60)
    print("Stages=2  — 2 passes, 8 regs/row, 2 syncs, gmem overlap")
    print("="*60)
    r_s2 = json.loads(output_simt_ffma2_stages.run_stages2())

    print("\n" + "="*60)
    print("Stages=4  — 4 passes, 4 regs/row, 4 syncs, gmem overlap")
    print("="*60)
    r_s4 = json.loads(output_simt_ffma2_stages.run_stages4())

    return json.dumps({"baseline": r_base, "stages2": r_s2, "stages4": r_s4}, indent=2)


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result_json = run_all.remote()
    data = json.loads(result_json)

    def get_phase(d, name):
        for p in d.get("probes", []):
            if p["phase"] == name:
                return p["us"]
        return None

    def fmt(label, d):
        print(f"\n{label}")
        for p in d["probes"]:
            print(f"  {p['phase']:12s}: {p['us']:7.3f} µs")
        return get_phase(d, "total")

    t_base = fmt("Baseline  (1 pass, 16 regs/row):", data["baseline"])
    t_s2   = fmt("Stages=2  (2 passes, 8 regs/row, gmem overlap):", data["stages2"])
    t_s4   = fmt("Stages=4  (4 passes, 4 regs/row, gmem overlap):", data["stages4"])

    g_base = get_phase(data["baseline"], "gemv")
    g_s2   = get_phase(data["stages2"],  "stages_loop")
    g_s4   = get_phase(data["stages4"],  "stages_loop")

    print("\n--- Comparison ---")
    print(f"  {'variant':10s}  {'total':>8s}  {'gemv/loop':>10s}")
    print(f"  {'baseline':10s}  {t_base:8.3f}  {g_base:10.3f}")
    print(f"  {'stages=2':10s}  {t_s2:8.3f}  {g_s2:10.3f}")
    print(f"  {'stages=4':10s}  {t_s4:8.3f}  {g_s4:10.3f}")

    for label, t in [("stages=2", t_s2), ("stages=4", t_s4)]:
        if t and t_base:
            diff = t - t_base
            if diff < 0:
                print(f"  {label} vs baseline: {diff:.3f} µs ({t_base/t:.2f}x faster)")
            else:
                print(f"  {label} vs baseline: +{diff:.3f} µs ({t/t_base:.2f}x slower)")

    out_path = Path("reports/intra_output_simt_ffma2_stages.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

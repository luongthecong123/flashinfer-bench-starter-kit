"""Modal runner: cpasync (no swizzle) vs cpasync_swz (Sw128 swizzle) vs baseline.

Usage:
    modal run src/modal/intra_output_simt_ffma2_cpasync_swz.py
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
    from src.kernels import output_simt_ffma2_stages_smem_cpasync
    from src.kernels import output_simt_ffma2_stages_smem_cpasync_swz

    print("\n" + "="*60)
    print("Baseline — 1 pass, gmem CKV, scalar loads")
    print("="*60)
    r_base = json.loads(output_simt_ffma2_intra.run_single())

    print("\n" + "="*60)
    print("cpasync=4 — smem CKV 128KB, cp.async 128b, no swizzle")
    print("="*60)
    r_cp = json.loads(output_simt_ffma2_stages_smem_cpasync.run_smem_cpasync4())

    print("\n" + "="*60)
    print("cpasync_swz=4 — smem CKV 128KB, cp.async 128b, Sw128(3,4,3)")
    print("="*60)
    r_swz = json.loads(output_simt_ffma2_stages_smem_cpasync_swz.run_smem_cpasync_swz4())

    return json.dumps({"baseline": r_base, "cpasync4": r_cp,
                       "cpasync_swz4": r_swz}, indent=2)


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

    t_base = fmt("Baseline  (1 pass, gmem CKV):", data["baseline"])
    t_cp   = fmt("cpasync=4  (smem CKV, no swizzle):", data["cpasync4"])
    t_swz  = fmt("cpasync_swz=4  (smem CKV, Sw128):", data["cpasync_swz4"])

    loop_cp  = get_phase(data["cpasync4"],   "stages_loop")
    loop_swz = get_phase(data["cpasync_swz4"], "stages_loop")
    ckv_cp   = get_phase(data["cpasync4"],   "load_ckv")
    ckv_swz  = get_phase(data["cpasync_swz4"], "load_ckv")

    print("\n--- Comparison ---")
    print(f"  {'variant':24s}  {'total':>8s}  {'load_ckv':>9s}  {'loop':>8s}")
    g_base = get_phase(data["baseline"], "gemv")
    print(f"  {'baseline':24s}  {t_base:8.3f}  {'N/A':>9s}  {g_base:8.3f}  µs")
    print(f"  {'cpasync=4':24s}  {t_cp:8.3f}  {ckv_cp:9.3f}  {loop_cp:8.3f}  µs")
    print(f"  {'cpasync_swz=4':24s}  {t_swz:8.3f}  {ckv_swz:9.3f}  {loop_swz:8.3f}  µs")

    for label, t in [("cpasync=4", t_cp), ("cpasync_swz=4", t_swz)]:
        if t and t_base:
            diff = t - t_base
            sign = "" if diff < 0 else "+"
            tag = f"({t_base/t:.2f}x faster)" if diff < 0 else f"({t/t_base:.2f}x slower)"
            print(f"  {label} vs baseline: {sign}{diff:.3f} µs {tag}")

    if t_cp and t_swz:
        diff = t_swz - t_cp
        sign = "" if diff < 0 else "+"
        tag = f"({t_cp/t_swz:.2f}x faster)" if diff < 0 else f"({t_swz/t_cp:.2f}x slower)"
        print(f"  swz vs no-swz: {sign}{diff:.3f} µs {tag}")
        if loop_cp and loop_swz:
            diff_loop = loop_swz - loop_cp
            sign2 = "" if diff_loop < 0 else "+"
            print(f"  stages_loop swz vs no-swz: {sign2}{diff_loop:.3f} µs")

    out_path = Path("reports/intra_output_simt_ffma2_cpasync_swz.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

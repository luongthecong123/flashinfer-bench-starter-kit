"""Modal runner: baseline vs stages_smem (scalar) vs stages_smem_cpasync (cp.async 128b).

Usage:
    modal run src/modal/intra_output_simt_ffma2_cpasync.py
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
    from src.kernels import output_simt_ffma2_stages_smem
    from src.kernels import output_simt_ffma2_stages_smem_cpasync

    print("\n" + "="*60)
    print("Baseline — 1 pass, gmem CKV, scalar loads")
    print("="*60)
    r_base = json.loads(output_simt_ffma2_intra.run_single())

    print("\n" + "="*60)
    print("Stages_smem=4 — smem CKV 128KB, scalar loop, 8 syncs")
    print("="*60)
    r_smem = json.loads(output_simt_ffma2_stages_smem.run_smem4())

    print("\n" + "="*60)
    print("Stages_smem_cpasync=4 — smem CKV 128KB, cp.async 128b, 8 syncs")
    print("="*60)
    r_cp = json.loads(output_simt_ffma2_stages_smem_cpasync.run_smem_cpasync4())

    return json.dumps({"baseline": r_base, "stages_smem4": r_smem,
                       "stages_smem_cpasync4": r_cp}, indent=2)


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
    t_smem = fmt("Stages_smem=4  (smem CKV, scalar load):", data["stages_smem4"])
    t_cp   = fmt("Stages_smem_cpasync=4  (smem CKV, cp.async 128b):", data["stages_smem_cpasync4"])

    ckv_smem = get_phase(data["stages_smem4"],         "load_ckv")
    ckv_cp   = get_phase(data["stages_smem_cpasync4"], "load_ckv")
    loop_smem = get_phase(data["stages_smem4"],         "stages_loop")
    loop_cp   = get_phase(data["stages_smem_cpasync4"], "stages_loop")

    print("\n--- Comparison ---")
    print(f"  {'variant':24s}  {'total':>8s}  {'load_ckv':>9s}  {'loop':>8s}")
    print(f"  {'baseline':24s}  {t_base:8.3f}  {'N/A':>9s}  "
          f"  {get_phase(data['baseline'], 'gemv'):8.3f}  µs")
    if ckv_smem:
        print(f"  {'stages_smem=4':24s}  {t_smem:8.3f}  {ckv_smem:9.3f}  {loop_smem:8.3f}  µs")
    if ckv_cp:
        print(f"  {'stages_smem_cpasync=4':24s}  {t_cp:8.3f}  {ckv_cp:9.3f}  {loop_cp:8.3f}  µs")

    for label, t in [("stages_smem=4", t_smem), ("stages_smem_cpasync=4", t_cp)]:
        if t and t_base:
            diff = t - t_base
            tag = f"({t_base/t:.2f}x faster)" if diff < 0 else f"({t/t_base:.2f}x slower)"
            sign = "" if diff < 0 else "+"
            print(f"  {label} vs baseline: {sign}{diff:.3f} µs {tag}")

    if t_smem and t_cp:
        diff = t_cp - t_smem
        if diff < 0:
            print(f"  cpasync vs scalar: {diff:.3f} µs ({t_smem/t_cp:.2f}x faster)")
        else:
            print(f"  cpasync vs scalar: +{diff:.3f} µs ({t_cp/t_smem:.2f}x slower)")
        if ckv_smem and ckv_cp:
            diff_ckv = ckv_cp - ckv_smem
            sign = "" if diff_ckv < 0 else "+"
            print(f"  load_ckv cpasync vs scalar: {sign}{diff_ckv:.3f} µs")

    out_path = Path("reports/intra_output_simt_ffma2_cpasync.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

"""Modal runner: stages_smem 512T (baseline) vs stages_smem 1024T.

Usage:
    modal run src/modal/intra_output_simt_ffma2_stages_smem_1024.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all():
    import sys, json
    sys.path.insert(0, "/app")

    from src.kernels import output_simt_ffma2_stages_smem
    from src.kernels import output_simt_ffma2_stages_smem_1024

    print("\n" + "="*60)
    print("Baseline  — 512T, 16 warps, num_rounds=8, smem_partial=16KB")
    print("="*60)
    r_512 = json.loads(output_simt_ffma2_stages_smem.run_smem4())

    print("\n" + "="*60)
    print("1024T     — 1024T, 32 warps, num_rounds=4, smem_partial=32KB")
    print("="*60)
    r_1024 = json.loads(output_simt_ffma2_stages_smem_1024.run_smem4_1024())

    return json.dumps({"smem_512T": r_512, "smem_1024T": r_1024}, indent=2)


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

    t_512  = fmt("512T  (16 warps, num_rounds=8, smem_partial=16KB):", data["smem_512T"])
    t_1024 = fmt("1024T (32 warps, num_rounds=4, smem_partial=32KB):", data["smem_1024T"])

    g_512  = get_phase(data["smem_512T"],  "stages_loop")
    g_1024 = get_phase(data["smem_1024T"], "stages_loop")
    ld_512  = get_phase(data["smem_512T"],  "load_ckv")
    ld_1024 = get_phase(data["smem_1024T"], "load_ckv")

    print("\n--- Comparison ---")
    print(f"  {'variant':10s}  {'total':>8s}  {'stages_loop':>12s}  {'load_ckv':>10s}  {'smem_partial':>12s}")
    print(f"  {'512T':10s}  {t_512:8.3f}  {g_512:12.3f}  {ld_512:10.3f}  16 KB")
    print(f"  {'1024T':10s}  {t_1024:8.3f}  {g_1024:12.3f}  {ld_1024:10.3f}  32 KB")

    if t_512 and t_1024:
        diff = t_1024 - t_512
        if diff < 0:
            print(f"\n  1024T vs 512T: {diff:.3f} µs ({t_512/t_1024:.2f}x faster)")
        else:
            print(f"\n  1024T vs 512T: +{diff:.3f} µs ({t_1024/t_512:.2f}x slower)")

    if g_512 and g_1024:
        diff_g = g_1024 - g_512
        if diff_g < 0:
            print(f"  stages_loop: {diff_g:.3f} µs ({g_512/g_1024:.2f}x faster)")
        else:
            print(f"  stages_loop: +{diff_g:.3f} µs ({g_1024/g_512:.2f}x slower)")

    out_path = Path("reports/intra_output_simt_ffma2_stages_smem_1024.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

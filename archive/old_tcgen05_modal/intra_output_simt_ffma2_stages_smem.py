"""Modal runner: baseline vs stages=4 (dedicated slots) vs stages_smem=4 (CKV in smem).

Usage:
    modal run src/modal/intra_output_simt_ffma2_stages_smem.py
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
    from src.kernels import output_simt_ffma2_stages_smem

    print("\n" + "="*60)
    print("Baseline  — 1 pass, 16 regs/row, 1 sync, gmem CKV")
    print("="*60)
    r_base = json.loads(output_simt_ffma2_intra.run_single())

    print("\n" + "="*60)
    print("Stages=4  — 4 passes, 4 regs/row, 4 syncs, gmem CKV, dedicated smem slots")
    print("="*60)
    r_s4 = json.loads(output_simt_ffma2_stages.run_stages4())

    print("\n" + "="*60)
    print("Stages_smem=4 — 4 passes, 4 regs/row, 8 syncs, smem CKV (128KB), reused smem_partial (16KB)")
    print("="*60)
    r_smem4 = json.loads(output_simt_ffma2_stages_smem.run_smem4())

    return json.dumps({"baseline": r_base, "stages4": r_s4, "stages_smem4": r_smem4}, indent=2)


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

    t_base   = fmt("Baseline  (1 pass, 16 regs/row, gmem CKV):", data["baseline"])
    t_s4     = fmt("Stages=4  (4 passes, 4 regs/row, gmem CKV, 64KB partial):", data["stages4"])
    t_smem4  = fmt("Stages_smem=4 (4 passes, 4 regs/row, smem CKV 128KB, 16KB partial):", data["stages_smem4"])

    g_base  = get_phase(data["baseline"],    "gemv")
    g_s4    = get_phase(data["stages4"],     "stages_loop")
    g_smem4 = get_phase(data["stages_smem4"], "stages_loop")
    ckv_ld  = get_phase(data["stages_smem4"], "load_ckv")

    print("\n--- Comparison ---")
    print(f"  {'variant':16s}  {'total':>8s}  {'gemv/loop':>10s}")
    print(f"  {'baseline':16s}  {t_base:8.3f}  {g_base:10.3f}  µs")
    if g_s4 is not None:
        print(f"  {'stages=4':16s}  {t_s4:8.3f}  {g_s4:10.3f}  µs")
    if g_smem4 is not None:
        print(f"  {'stages_smem=4':16s}  {t_smem4:8.3f}  {g_smem4:10.3f}  µs  (load_ckv={ckv_ld:.3f} µs)")

    for label, t in [("stages=4", t_s4), ("stages_smem=4", t_smem4)]:
        if t and t_base:
            diff = t - t_base
            if diff < 0:
                print(f"  {label} vs baseline: {diff:.3f} µs ({t_base/t:.2f}x faster)")
            else:
                print(f"  {label} vs baseline: +{diff:.3f} µs ({t/t_base:.2f}x slower)")

    if t_s4 and t_smem4:
        diff = t_smem4 - t_s4
        if diff < 0:
            print(f"  stages_smem=4 vs stages=4: {diff:.3f} µs ({t_s4/t_smem4:.2f}x faster)")
        else:
            print(f"  stages_smem=4 vs stages=4: +{diff:.3f} µs ({t_smem4/t_s4:.2f}x slower)")

    out_path = Path("reports/intra_output_simt_ffma2_stages_smem.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

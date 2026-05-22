"""Modal runner: baseline vs tile_N=2 vs tile_N=4.

Usage:
    modal run src/modal/intra_output_simt_ffma2_tileN.py
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
    from src.kernels import output_simt_ffma2_tileN_intra

    print("\n" + "="*60)
    print("Baseline  — smem_partial (16,2,512) = 64 KB")
    print("="*60)
    r_base  = json.loads(output_simt_ffma2_intra.run_single())

    print("\n" + "="*60)
    print("tile_N=2  — smem_partial (16,2,256) = 32 KB  VEC_SIZE=8")
    print("="*60)
    r_tile2 = json.loads(output_simt_ffma2_tileN_intra.run_tile2())

    print("\n" + "="*60)
    print("tile_N=4  — smem_partial (16,2,128) = 16 KB  VEC_SIZE=4")
    print("="*60)
    r_tile4 = json.loads(output_simt_ffma2_tileN_intra.run_tile4())

    return json.dumps({"baseline": r_base, "tile2": r_tile2, "tile4": r_tile4}, indent=2)


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

    t_base  = fmt("Baseline  (smem 64KB, no tile):", data["baseline"])
    t_tile2 = fmt("tile_N=2  (smem 32KB, +1 sync):", data["tile2"])
    t_tile4 = fmt("tile_N=4  (smem 16KB, +3 sync):", data["tile4"])

    g_base  = get_phase(data["baseline"], "gemv")
    g_tile2 = get_phase(data["tile2"],    "gemv")
    g_tile4 = get_phase(data["tile4"],    "gemv")
    wr_base  = get_phase(data["baseline"], "reduce")
    wr_tile2 = get_phase(data["tile2"],    "write_reduce")
    wr_tile4 = get_phase(data["tile4"],    "write_reduce")

    print("\n--- Comparison ---")
    print(f"  {'variant':10s}  {'total':>8s}  {'gemv':>8s}  {'wr_phase':>10s}")
    print(f"  {'baseline':10s}  {t_base:8.3f}  {g_base:8.3f}  {wr_base:10.3f}")
    print(f"  {'tile_N=2':10s}  {t_tile2:8.3f}  {g_tile2:8.3f}  {wr_tile2:10.3f}")
    print(f"  {'tile_N=4':10s}  {t_tile4:8.3f}  {g_tile4:8.3f}  {wr_tile4:10.3f}")

    for label, t in [("tile_N=2", t_tile2), ("tile_N=4", t_tile4)]:
        if t and t_base:
            diff = t - t_base
            sign = "+" if diff > 0 else ""
            print(f"  {label} vs baseline: {sign}{diff:.3f} µs ({t_base/t:.2f}x)" if t < t_base
                  else f"  {label} vs baseline: +{diff:.3f} µs ({t/t_base:.2f}x slower)")

    out_path = Path("reports/intra_output_simt_ffma2_tileN.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

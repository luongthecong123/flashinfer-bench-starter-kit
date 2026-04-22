"""Modal runner: intra profiling for output_simt_ffma2 GEMV.

Runs both output_simt_intra and output_simt_ffma2_intra back-to-back on the
same B200 so their latencies are directly comparable.

Usage:
    modal run src/modal/intra_output_simt_ffma2.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_both():
    import sys, json
    sys.path.insert(0, "/app")

    from src.kernels import output_simt_intra, output_simt_ffma2_intra

    print("\n" + "="*60)
    print("output_simt  (1×256) @ (256×512) — 1024 threads, plain FMA")
    print("="*60)
    r1 = json.loads(output_simt_intra.run_single())

    print("\n" + "="*60)
    print("output_simt_ffma2  (2×128) @ (128×512) — 512 threads, FFMA2")
    print("="*60)
    r2 = json.loads(output_simt_ffma2_intra.run_single())

    return json.dumps({"simt": r1, "ffma2": r2}, indent=2)


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result_json = run_both.remote()
    data = json.loads(result_json)

    def fmt(label, d):
        total = next((p["us"] for p in d["probes"] if p["phase"] == "total"), None)
        print(f"\n{label}")
        for p in d["probes"]:
            print(f"  {p['phase']:8s}: {p['us']:7.3f} µs")
        return total

    t1 = fmt("output_simt   (1×256 @ 256×512, 1024 thr, FMA) :", data["simt"])
    t2 = fmt("output_simt_ffma2 (2×128 @ 128×512,  512 thr, FFMA2):", data["ffma2"])

    print(f"\n--- Comparison ---")
    print(f"  simt    total: {t1:.3f} µs")
    print(f"  ffma2   total: {t2:.3f} µs")
    if t1 and t2:
        winner = "ffma2" if t2 < t1 else "simt"
        ratio  = max(t1, t2) / min(t1, t2)
        print(f"  Winner: {winner}  ({ratio:.2f}x faster)")

    out_path = Path("reports/intra_output_simt_ffma2.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

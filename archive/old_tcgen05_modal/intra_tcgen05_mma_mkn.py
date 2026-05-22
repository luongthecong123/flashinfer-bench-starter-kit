"""Modal runner: tcgen05 FP8 MMA (M=128, K=512) x (K=512, N=8) = (M=128, N=8).

Usage:
    modal run src/modal/intra_tcgen05_mma_mkn.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all():
    import sys, json
    sys.path.insert(0, "/app")

    from src.kernels import tcgen05_mma_mkn

    print("\n" + "="*60)
    print("tcgen05 FP8 MMA  M=128  K=512  N=8")
    print("="*60)
    return tcgen05_mma_mkn.run_intra()


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result_json = run_all.remote()
    data = json.loads(result_json)

    print("\n--- Phase breakdown ---")
    for p in data["probes"]:
        print(f"  {p['phase']:10s}: {p['us']:7.3f} µs")

    out_path = Path("reports/intra_tcgen05_mma_mkn.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

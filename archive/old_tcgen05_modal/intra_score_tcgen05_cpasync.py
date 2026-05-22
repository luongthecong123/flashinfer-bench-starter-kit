"""Modal runner: score_tcgen05_cpasync — tcgen05 BF16 MMA with cp.async load.

Same problem as score_tcgen05.py:
  (M=128, K=512) x (N_mma=8, K=512) → (M=128, N_real=2)
but loads A and B using cp.async (instead of TMA) with the
int32-recast + make_swizzle(3,3,3) composed-layout pattern.

Usage:
    modal run src/modal/intra_score_tcgen05_cpasync.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all():
    import sys, json
    sys.path.insert(0, "/app")

    from src.kernels import score_tcgen05_cpasync as sc

    print("\n" + "="*60)
    print("score_tcgen05_cpasync — (128,512)×(8,512)→(128,2) [N_real=2, cp.async]")
    print("="*60)
    return sc.run_intra()


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result_json = run_all.remote()
    data = json.loads(result_json)

    print("\n--- Summary ---")
    print(f"  kernel   : {data['kernel']}")
    print(f"  M={data['M']}  K={data['K']}  N_real={data['N_real']}  N_mma={data['N_mma']}")
    print(f"  correct  : {'PASS' if data['correct'] else 'FAIL'}  max_diff={data['max_diff']:.6f}")
    print()
    for p in data.get("probes", []):
        print(f"  {p['phase']:10s}: {p['us']:7.3f} µs")

    out_path = Path("reports/intra_score_tcgen05_cpasync.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

"""Modal runner: score_tcgen05_debug — TMA + tcgen05 with raw physical SMEM dump.

Verifies:
  - tcgen05 BF16 MMA correctness on score dims (M=128, K=512, N_mma=8 → N_real=2)
  - Dumps the RAW physical SMEM for sA[128,512] and sB[8,512] using
    a non-swizzled pointer constructed via cute.make_ptr at the same byte
    address as the swizzled allocation. Reveals the actual swizzle pattern.

Usage:
    modal run src/modal/intra_score_tcgen05_debug.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all():
    import sys
    sys.path.insert(0, "/app")
    from src.kernels import score_tcgen05_debug as sd

    print("\n" + "=" * 60)
    print("score_tcgen05_debug — TMA + tcgen05 + RAW SMEM dump")
    print("=" * 60)
    return sd.run()


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result_json = run_all.remote()
    data = json.loads(result_json)

    print("\n--- Summary ---")
    print(f"  kernel       : {data['kernel']}")
    print(f"  M={data['M']}  K={data['K']}  N_real={data['N_real']}  N_mma={data['N_mma']}")
    print(f"  mma_correct  : {'PASS' if data['mma_correct'] else 'FAIL'}  max_diff={data['mma_max_diff']:.6f}")

    out_path = Path("reports/intra_score_tcgen05_debug.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

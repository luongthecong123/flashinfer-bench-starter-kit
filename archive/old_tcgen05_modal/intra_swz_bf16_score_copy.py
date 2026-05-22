"""Modal runner: swz_bf16_score_copy — TMA + autovec swizzle copy verification.

Verifies that cp-copy methods produce the correct physical SMEM layout for
score-kernel tiles:
  sA: [128, 512] bfloat16
  sB: [  8, 512] bfloat16

Usage:
    # Step 1 (TMA only):
    modal run src/modal/intra_swz_bf16_score_copy.py

    # Steps 1+2 (TMA + autovec):
    modal run src/modal/intra_swz_bf16_score_copy.py --steps 2
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all(steps: int = 1):
    import sys
    sys.path.insert(0, "/app")

    from src.kernels import swz_bf16_score_copy as sc

    print("\n" + "=" * 60)
    print(f"swz_bf16_score_copy  sA=[128,512]  sB=[8,512]  steps={steps}")
    print("=" * 60)
    return sc.run(steps=steps)


@app.local_entrypoint()
def main(steps: int = 1):
    import json
    from pathlib import Path

    result_json = run_all.remote(steps=steps)
    data = json.loads(result_json)

    print("\n--- Result ---")
    for k, v in data.items():
        print(f"  {k}: {v}")

    out_path = Path("reports/swz_bf16_score_copy.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

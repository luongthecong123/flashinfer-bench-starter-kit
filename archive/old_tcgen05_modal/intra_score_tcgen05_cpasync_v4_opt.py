"""Modal runner: score_tcgen05_cpasync_v4_opt — single tiled_copy + split probes."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all():
    import sys
    sys.path.insert(0, "/app")
    from src.kernels import score_tcgen05_cpasync_v4_opt as sc
    print("\n" + "="*60)
    print("score_tcgen05_cpasync_v4_opt — single tiled_copy + split load_a/b")
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
    print(f"  correct  : {'PASS' if data['correct'] else 'FAIL'}  max_diff={data['max_diff']:.6f}")
    for p in data.get("probes", []):
        print(f"  {p['phase']:10s}: {p['us']:7.3f} µs")
    out_path = Path("reports/intra_score_tcgen05_cpasync_v4_opt.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

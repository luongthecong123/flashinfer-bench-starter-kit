"""Modal runner: score_warp_red — cp.async load + FastGEMV-style score."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all():
    import sys
    sys.path.insert(0, "/app")

    from src.kernels import score_warp_red as sc

    print("\n" + "=" * 60)
    print("score_warp_red — cp.async load + FastGEMV score")
    print("=" * 60)
    return sc.run_intra()


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result_json = run_all.remote()
    data = json.loads(result_json)

    print("\n--- Summary ---")
    print(f"  {'case':6s}  {'seq_len':>8s}  {'correct':>8s}  {'load_ab':>10s}  {'score':>10s}  {'total':>10s}")
    for case, d in data.items():
        ok = "PASS" if d["correct"] else f"FAIL({d['max_diff']:.4f})"
        print(
            f"  {case:6s}  {d['seq_len']:8d}  {ok:>8s}"
            f"  {d.get('load_ab_us', 0.0):10.3f}"
            f"  {d.get('score_us', 0.0):10.3f}"
            f"  {d.get('total_us', 0.0):10.3f}  us"
        )

    out_path = Path("reports/intra_score_warp_red.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

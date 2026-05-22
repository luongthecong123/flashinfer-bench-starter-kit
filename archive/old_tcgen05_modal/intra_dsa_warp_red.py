"""Modal runner: dsa_warp_red - score + softmax + output."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all():
    import sys
    sys.path.insert(0, "/app")

    from src.kernels import dsa_warp_red as dsa

    print("\n" + "=" * 60)
    print("dsa_warp_red - cp.async load + score/softmax/output")
    print("=" * 60)
    return dsa.run_intra()


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result_json = run_all.remote()
    data = json.loads(result_json)

    print("\n--- Summary ---")
    print(
        f"  {'case':6s}  {'seq_len':>8s}  {'correct':>8s}"
        f"  {'load_ab':>10s}  {'score':>10s}  {'softmax':>10s}"
        f"  {'output':>10s}  {'total':>10s}"
    )
    for case, d in data.items():
        ok = "PASS" if d["correct"] else f"FAIL({d['max_diff']:.4f})"
        print(
            f"  {case:6s}  {d['seq_len']:8d}  {ok:>8s}"
            f"  {d.get('load_ab_us', 0.0):10.3f}"
            f"  {d.get('score_us', 0.0):10.3f}"
            f"  {d.get('softmax_us', 0.0):10.3f}"
            f"  {d.get('output_us', 0.0):10.3f}"
            f"  {d.get('total_us', 0.0):10.3f}  us"
        )

    out_path = Path("reports/intra_dsa_warp_red.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

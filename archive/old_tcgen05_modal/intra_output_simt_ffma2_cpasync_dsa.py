"""Modal runner: DSA paged-KV cpasync kernel — full (2048 valid) vs short (128 valid) cases.

Tests two sparse_indices patterns:
  full:  sparse_indices = [0, 1, ..., 2047]  — all K_topk entries valid
  short: sparse_indices = [0, 1, ..., 127, -1, -1, ..., -1]  — only 128 valid

Usage:
    modal run src/modal/intra_output_simt_ffma2_cpasync_dsa.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all():
    import sys, json
    sys.path.insert(0, "/app")

    from src.kernels import output_simt_ffma2_stages_smem_cpasync_dsa as dsa

    print("\n" + "="*60)
    print("DSA cpasync kernel — full vs short sparse_indices")
    print("="*60)
    results = dsa.run_dsa_cases(save_dir="/tmp")
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    result_json = run_all.remote()
    data = json.loads(result_json)

    print("\n--- Summary ---")
    print(f"  {'case':6s}  {'seq_len':>8s}  {'correct':>8s}  {'total_us':>10s}")
    for case_name, r in data.items():
        ok_str = "PASS" if r["correct"] else f"FAIL(maxdiff={r['max_diff']:.4f})"
        print(f"  {case_name:6s}  {r['seq_len']:8d}  {ok_str:>8s}  {r['total_us']:10.3f} µs")
        for p in r.get("probes", []):
            print(f"    {p['phase']:12s}: {p['us']:7.3f} µs")

    out_path = Path("reports/intra_output_simt_ffma2_cpasync_dsa.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

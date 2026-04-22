"""Modal runner: intra profiling for output_simt GEMV.

Usage:
    modal run src/modal/intra_output_simt.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, image

IMPL_MODULE  = "src.kernels.output_simt_intra"
OUT_FILENAME = "intra_output_simt.json"


@app.function(image=image, gpu="B200:1", timeout=600)
def run_intra():
    import sys
    sys.path.insert(0, "/app")
    from importlib import import_module
    mod = import_module(IMPL_MODULE)
    return mod.run_single()


@app.local_entrypoint()
def main():
    print(f"\n{'='*60}\nProfiling {IMPL_MODULE}\n{'='*60}")
    result_json = run_intra.remote()

    out_path = Path(f"reports/{OUT_FILENAME}")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)

    import json
    data = json.loads(result_json)
    print("\nPhase breakdown:")
    for p in data["probes"]:
        print(f"  {p['phase']:8s}: {p['us']:7.3f} µs")
    print(f"\nSaved to {out_path}")

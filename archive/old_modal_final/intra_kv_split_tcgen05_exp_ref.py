"""Modal runner: kv_split_tcgen05_exp_ref (correctness check only)."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 22
OUT_NAME     = f"intra_kv_split_tcgen05_exp_ref_w{WORKLOAD_IDX + 1}"


@app.function(image=image, gpu="B200:1", timeout=900,
              volumes={"/data": trace_volume})
def run_remote(workload_idx: int):
    sys.path.insert(0, "/app")
    from src.kernels import kv_split_tcgen05_exp_ref as exp
    print("\n" + "=" * 60)
    print(f"kv_split_tcgen05_exp_ref — WL{workload_idx + 1}")
    print("=" * 60)
    summary = exp.run_intra(workload_idx)
    return summary


@app.local_entrypoint()
def main():
    print(f"\nRunning kv_split_tcgen05_exp_ref on WL{WORKLOAD_IDX + 1}")
    summary = run_remote.remote(WORKLOAD_IDX)
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{OUT_NAME}.json").write_text(summary)
    print(f"\nSaved {out_dir / (OUT_NAME + '.json')}")
    print("\n--- summary ---\n" + summary)

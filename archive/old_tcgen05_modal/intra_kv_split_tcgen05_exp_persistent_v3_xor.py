"""Modal runner: kv_split_tcgen05_exp_persistent_v3_xor (dynamic per-(t,split) routing)."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 22
OUT_NAME     = f"intra_kv_split_tcgen05_exp_persistent_v3_xor_w{WORKLOAD_IDX + 1}"


@app.function(image=image, gpu="B200:1", timeout=900,
              volumes={"/data": trace_volume})
def run_remote(workload_idx: int):
    os.environ["CUTE_DSL_KEEP_PTX"]   = "1"
    os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
    sys.path.insert(0, "/app")
    from src.kernels import kv_split_tcgen05_exp_persistent_v3_xor as exp
    print("\n" + "=" * 60)
    print(f"kv_split_tcgen05_exp_persistent_v3_xor — WL{workload_idx + 1}")
    print("=" * 60)
    summary, trace = exp.run_intra(workload_idx)
    return summary, trace


@app.local_entrypoint()
def main():
    print(f"\nProfiling kv_split_tcgen05_exp_persistent_v3_xor on WL{WORKLOAD_IDX + 1}")
    summary, trace = run_remote.remote(WORKLOAD_IDX)
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{OUT_NAME}.json").write_text(summary)
    (out_dir / f"{OUT_NAME}_trace.json").write_text(trace)
    print(f"\nSaved {out_dir / (OUT_NAME + '.json')}")
    print(f"Saved {out_dir / (OUT_NAME + '_trace.json')}")
    print("\n--- summary ---\n" + summary)

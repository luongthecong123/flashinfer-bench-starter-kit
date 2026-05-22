"""Modal runner: kv_split_tcgen05_exp_persistent_v5_xor_specialized_v3b."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 22
OUT_NAME     = f"intra_kv_split_tcgen05_exp_persistent_v5_xor_specialized_v3b_w{WORKLOAD_IDX + 1}"


@app.function(image=image, gpu="B200:1", timeout=900,
              volumes={"/data": trace_volume})
def run_remote(workload_idx: int):
    os.environ["CUTE_DSL_KEEP_PTX"]   = "1"
    os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
    sys.path.insert(0, "/app")
    from src.kernels import kv_split_tcgen05_exp_persistent_v5_xor_specialized_v3b as exp
    print("\n" + "=" * 60)
    print(f"kv_split_tcgen05_exp_persistent_v5_xor_specialized_v3b — WL{workload_idx + 1}")
    print("=" * 60)
    summary, trace = exp.run_intra(workload_idx)

    # ── Collect generated PTX / cubin artifacts ────────────────────────────
    import glob, base64
    artifacts = {}
    for ext in ("ptx", "cubin"):
        for path in (sorted(glob.glob(f"/tmp/**/*.{ext}", recursive=True))
                     + sorted(glob.glob(f"./*.{ext}"))
                     + sorted(glob.glob(f"/root/**/*.{ext}", recursive=True))):
            try:
                if ext == "ptx":
                    artifacts[path] = open(path).read()
                else:
                    artifacts[path] = base64.b64encode(
                        open(path, "rb").read()).decode()
                print(f"  artifact: {path}  ({os.path.getsize(path)} bytes)")
            except Exception as e:
                print(f"  artifact-skip: {path}  err={e}")
    return summary, trace, artifacts


@app.local_entrypoint()
def main():
    import base64
    print(f"\nProfiling kv_split_tcgen05_exp_persistent_v5_xor_specialized_v3b "
          f"on WL{WORKLOAD_IDX + 1}")
    summary, trace, artifacts = run_remote.remote(WORKLOAD_IDX)
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{OUT_NAME}.json").write_text(summary)
    (out_dir / f"{OUT_NAME}_trace.json").write_text(trace)
    print(f"\nSaved {out_dir / (OUT_NAME + '.json')}")
    print(f"Saved {out_dir / (OUT_NAME + '_trace.json')}")

    asm_dir = Path("reports/asm/kv_split_tcgen05_exp_persistent_v5_xor_specialized_v3b")
    asm_dir.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts.items():
        name = Path(path).name
        local = asm_dir / name
        if name.endswith(".ptx"):
            local.write_text(content)
        else:
            local.write_bytes(base64.b64decode(content))
        print(f"Saved {local}")

    print("\n--- summary ---\n" + summary)

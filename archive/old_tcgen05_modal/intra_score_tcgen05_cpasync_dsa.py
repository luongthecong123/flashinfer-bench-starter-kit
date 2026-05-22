"""Modal runner: score_tcgen05_cpasync_dsa — sparse-indexed score kernel."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def run_all():
    os.environ["CUTE_DSL_KEEP_PTX"]   = "1"
    os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"

    import sys, glob, json
    sys.path.insert(0, "/app")
    from src.kernels import score_tcgen05_cpasync_dsa as sc
    print("\n" + "="*60)
    print("score_tcgen05_cpasync_dsa — sparse-indexed A loads")
    print("="*60)
    result = sc.run_intra()

    artifacts = {}
    for ext in ("ptx", "cubin"):
        for path in sorted(glob.glob(f"/tmp/**/*.{ext}", recursive=True)) + \
                    sorted(glob.glob(f"./*.{ext}")) + \
                    sorted(glob.glob(f"/root/*.{ext}")):
            try:
                if ext == "ptx":
                    artifacts[path] = open(path).read()
                else:
                    import base64
                    artifacts[path] = base64.b64encode(open(path, "rb").read()).decode()
                print(f"  artifact: {path}  ({os.path.getsize(path)} bytes)")
            except Exception as e:
                print(f"  artifact-skip: {path}  err={e}")
    return result, artifacts


@app.local_entrypoint()
def main():
    import json, base64
    from pathlib import Path
    result_json, artifacts = run_all.remote()
    data = json.loads(result_json)
    print("\n--- Summary ---")
    for case, d in data.items():
        print(f"  [{case}]  seq_len={d['seq_len']}  correct={d['correct']}  "
              f"max_diff={d['max_diff']:.6f}  total={d['total_us']:.3f} µs")

    out_dir = Path("reports/asm/score_tcgen05_cpasync_dsa")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts.items():
        name = Path(path).name
        local = out_dir / name
        if name.endswith(".ptx"):
            local.write_text(content)
        else:
            local.write_bytes(base64.b64decode(content))
        print(f"  saved   : {local}")

    out_path = Path("reports/intra_score_tcgen05_cpasync_dsa.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(result_json)
    print(f"\nSaved to {out_path}")

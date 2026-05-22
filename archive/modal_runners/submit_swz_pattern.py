"""Run swizzle pattern visualization on Modal B200 and download PNGs.
Usage: modal run src/modal/submit_swz_pattern.py
"""
import sys, os, modal
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent  # src/

app = modal.App("swz-pattern")

image_mpl = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "numpy", "nvidia-cutlass-dsl",
                 "ninja", "apache-tvm-ffi", "matplotlib")
    .add_local_dir(SRC_DIR, remote_path="/app/src")
)


@app.function(image=image_mpl, gpu="B200:1", timeout=300)
def run_swz():
    import sys
    sys.path.insert(0, "/app")
    from src.kernels.score_tcgen05_fp8_swz_pattern import main
    main()

    # Read generated PNGs and return as dict
    results = {}
    for name in [
        "swizzle_sA_fp8_128x128.png",
        "swizzle_sA_fp8_banks.png",
        "swizzle_sA_fp8_comparison.png",
    ]:
        if os.path.exists(name):
            with open(name, "rb") as f:
                results[name] = f.read()
            print(f"  read {name}: {len(results[name])} bytes")
    return results


@app.local_entrypoint()
def go():
    pngs = run_swz.remote()
    if not pngs:
        print("No PNGs returned!")
        return
    for name, data in pngs.items():
        with open(name, "wb") as f:
            f.write(data)
        print(f"Saved {name} ({len(data)} bytes)")

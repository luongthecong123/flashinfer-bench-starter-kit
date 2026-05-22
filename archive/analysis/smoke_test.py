"""Smoke test: compile kernel.cu on B200 via Modal, run with random PyTorch tensors.

Usage:
    modal run scripts/smoke_test.py
"""

import modal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
KERNEL_CU = PROJECT_ROOT / "solution" / "cuda" / "kernel4.cu"

app = modal.App("smoke-test-cuda")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("torch", "apache-tvm-ffi", "numpy")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_file(str(KERNEL_CU), remote_path="/app/kernel.cu")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def smoke_test():
    import subprocess
    import torch
    import tvm_ffi

    # ── Step 1: Compile kernel.cu → kernel.so ────────────────────────────────
    print("=== Step 1: Compiling kernel.cu ===")

    # Get tvm-ffi compiler flags
    cxxflags = subprocess.check_output(
        ["tvm-ffi-config", "--cxxflags"], text=True
    ).strip()
    ldflags = subprocess.check_output(
        ["tvm-ffi-config", "--ldflags"], text=True
    ).strip()
    libs = subprocess.check_output(
        ["tvm-ffi-config", "--libs"], text=True
    ).strip()

    print(f"  cxxflags: {cxxflags}")
    print(f"  ldflags:  {ldflags}")
    print(f"  libs:     {libs}")

    # Build with nvcc (following TVM-FFI quickstart pattern)
    cmd = (
        f"nvcc -shared -O3 /app/kernel.cu "
        f"-Xcompiler -fPIC,-fvisibility=hidden "
        f"{cxxflags} {ldflags} {libs} "
        f"-arch=sm_100 "
        f"-o /tmp/kernel.so"
    )
    print(f"  cmd: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"COMPILATION FAILED (exit {result.returncode}):\n{result.stderr}")
        return
    print("Compilation successful!\n")

    # ── Step 2: Load the compiled module ─────────────────────────────────────
    print("=== Step 2: Loading module ===")
    mod = tvm_ffi.load_module("/tmp/kernel.so")
    kernel_fn = mod.kernel
    print(f"  Loaded: {kernel_fn}\n")

    # ── Step 3: Create test tensors ──────────────────────────────────────────
    # ── Step 3: Test BOTH dispatch paths ────────────────────────────────────
    for T, path_name in [(1, "FUSED (T<3)"), (4, "KV-SPLIT (T>=3)")]:
        print(f"\n=== Step 3: Testing {path_name} path (T={T}) ===")
        device = "cuda"

        q_nope = torch.randn(T, 16, 512, dtype=torch.bfloat16, device=device)
        q_pe = torch.randn(T, 16, 64, dtype=torch.bfloat16, device=device)
        ckv_cache = torch.randn(8462, 64, 512, dtype=torch.bfloat16, device=device)
        kpe_cache = torch.randn(8462, 64, 64, dtype=torch.bfloat16, device=device)
        sparse_indices = torch.randint(
            0, 8462 * 64, (T, 2048), dtype=torch.int32, device=device
        )
        sm_scale = 0.1352337788608801
        output = torch.zeros(T, 16, 512, dtype=torch.bfloat16, device=device)
        lse = torch.zeros(T, 16, dtype=torch.float32, device=device)

        kernel_fn(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)
        torch.cuda.synchronize()

        print(f"  Output range: [{output.float().min():.4f}, {output.float().max():.4f}]")
        print(f"  LSE range:    [{lse.min():.4f}, {lse.max():.4f}]")
        print(f"  NaN output:   {torch.isnan(output.float()).any().item()}")
        print(f"  NaN LSE:      {torch.isnan(lse).any().item()}")

    print("\n=== SMOKE TEST PASSED ===")


@app.local_entrypoint()
def main():
    smoke_test.remote()

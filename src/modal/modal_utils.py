"""Shared Modal resources: single image with all profiling tools.

Usage in submit.py / ncu.py / nsys.py:
    from src.modal.modal_utils import app, trace_volume, image
"""
import modal
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent          # src/

app = modal.App("dsa-cook")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

# Single image shared across all workflows — prevents cache invalidation.
# Installs both nsight-compute and nsight-systems in one layer.
_NCU_KERNEL_REGEX = r".*(nvjet|cublas|cudnn|cutlass|gemm|Gemm|GEMM|sgemm|dgemm|hgemm|bmm|triton).*"


def get_ncu_compute_cmd(
    ncu_path: str,
    target: str,
    out_rep: str,
) -> list[str]:
    """Build the ncu command list for subprocess.run.

    Filters to compute kernels only (nvjet, cublas, triton, cutlass, gemm…)
    so that CUDA graph / driver noise doesn't pollute the report.

    Args:
        ncu_path: Absolute path to the ncu binary.
        target: Python script to profile.
        out_rep: Output report path (without .ncu-rep extension).
    """
    return [
        ncu_path,
        "--set", "full",
        "--target-processes", "all",
        "--print-summary", "per-kernel",
        "--kernel-name", f"regex:{_NCU_KERNEL_REGEX}",
        "--import-source", "yes",
        "--source-folders", "/app/src",
        "-f", "--export", out_rep,
        "python", target,
    ]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging",
                 "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .apt_install("wget", "gnupg")
    .run_commands(
        # nsight-compute (debian12 CUDA repo)
        "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /' > /etc/apt/sources.list.d/cuda.list",
        # nsight-systems (ubuntu2204 devtools repo)
        "wget -qO- https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | gpg --dearmor -o /usr/share/keyrings/nsight-systems-keyring.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/nsight-systems-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' > /etc/apt/sources.list.d/nsight.list",
        "apt-get update && apt-get install -y nsight-compute-2026.1.0 nsight-systems-2026.2.1",
    )
    .add_local_dir(SRC_DIR, remote_path="/app/src")
)

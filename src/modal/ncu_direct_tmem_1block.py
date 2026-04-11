"""NCU profiling for score_tcgen05_direct_tmem_1block on Modal B200.
Usage: modal run src/modal/ncu_direct_tmem_1block.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image, get_ncu_compute_cmd

_RUN_ONCE = """\
import sys
sys.path.insert(0, "/app")
import torch
from cutlass.cute.runtime import from_dlpack
from src.kernels.score_tcgen05_direct_tmem_1block import (
    ScoreGEMM_Direct_TMEM_1Block, M, N, K
)
import cutlass.cute as cute

kv    = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
q     = torch.randn((1, K), device="cuda", dtype=torch.bfloat16)
q_pad = torch.empty((N, K), device="cuda", dtype=torch.bfloat16)
q_pad[0] = q[0]
c_out = torch.zeros((M, 1), device="cuda", dtype=torch.float32)

kv_    = from_dlpack(kv,    assumed_align=16)
q_pad_ = from_dlpack(q_pad, assumed_align=16)
c_out_ = from_dlpack(c_out, assumed_align=16)

gemm     = ScoreGEMM_Direct_TMEM_1Block()
compiled = cute.compile(gemm, kv_, q_pad_, c_out_)

# Run once — NCU captures this single invocation
compiled(kv_, q_pad_, c_out_)
print("kernel done")
"""


@app.function(image=image, gpu="B200:1", timeout=600)
def run_ncu():
    import subprocess, glob, os

    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]

    target = "/tmp/_ncu_tmem_target.py"
    with open(target, "w") as f:
        f.write(_RUN_ONCE)

    cmd = get_ncu_compute_cmd(ncu, target, "/tmp/ncu_tmem_out")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=540,
                       env=os.environ, cwd="/app/src")
    print(r.stdout[-8000:] if len(r.stdout) > 8000 else r.stdout)
    if r.stderr:
        print(f"STDERR (last 2000): {r.stderr[-2000:]}")
    print(f"exit: {r.returncode}")

    rep_files = glob.glob("/tmp/ncu_tmem_out*.ncu-rep")
    if rep_files:
        with open(rep_files[0], "rb") as f:
            data = f.read()
        print(f"Report size: {len(data)} bytes")
        return data
    else:
        print("No .ncu-rep file found!")
        return None


@app.local_entrypoint()
def main():
    data = run_ncu.remote()
    if data:
        out_path = "reports/ncu_direct_tmem_1block.ncu-rep"
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved NCU report to {out_path} ({len(data)} bytes)")

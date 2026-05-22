#!/usr/bin/env python3
"""Run impl3 (32-warp parallel-keys + gathered dense buffers) on Modal B200."""
import modal
from pathlib import Path

DEV_DIR = Path(__file__).parent
ROOT_DIR = DEV_DIR.parent
ZEN_DIR = ROOT_DIR / "zen"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(DEV_DIR, remote_path="/app/dev")
    .add_local_dir(ZEN_DIR, remote_path="/app/zen")
)

app = modal.App("dsa-impl3", image=image)
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

D_ckv, D_kpe, TOPK = 512, 64, 2048

@app.function(gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_cook():
    import sys, os
    import torch
    os.chdir("/app/dev")
    sys.path.insert(0, "/app/dev")
    sys.path.append("/app/zen")

    from impl3 import fused_dsa_v3_compiled
    from gather import gather_compiled

    def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
        T = q_nope.shape[0]
        ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
        kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])

        Kc = torch.empty(T, TOPK, D_ckv, dtype=torch.bfloat16, device="cuda")
        Kp = torch.empty(T, TOPK, D_kpe, dtype=torch.bfloat16, device="cuda")
        max_valid = torch.zeros(T, dtype=torch.int32, device="cuda")
        gather_compiled(ckv_flat, kpe_flat, sparse_indices, Kc, Kp, max_valid)
        torch.cuda.synchronize()

        fused_dsa_v3_compiled(q_nope, q_pe, Kc, Kp, max_valid, output, lse)

    import cook
    cook.CONTEST = Path("/data")
    cook.JSONL = cook.CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    cook.impl_fn = run
    cook.main()

@app.local_entrypoint()
def main():
    run_cook.remote()

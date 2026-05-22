#!/usr/bin/env python3
"""Run impl2 (32-warp parallel-keys letmecook) on Modal B200."""
import modal
from pathlib import Path

DEV_DIR = Path(__file__).parent
ROOT_DIR = DEV_DIR.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(DEV_DIR, remote_path="/app/dev")
)

app = modal.App("dsa-impl2", image=image)
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

@app.function(gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_cook():
    import sys, os
    os.chdir("/app/dev")
    sys.path.insert(0, "/app/dev")
    sys.path.insert(0, "/app")

    from impl2 import fused_dsa_v2_compiled

    def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
        ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
        kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
        fused_dsa_v2_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)

    import cook
    cook.CONTEST = Path("/data")
    cook.JSONL = cook.CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    cook.impl_fn = run
    cook.main()

@app.local_entrypoint()
def main():
    run_cook.remote()

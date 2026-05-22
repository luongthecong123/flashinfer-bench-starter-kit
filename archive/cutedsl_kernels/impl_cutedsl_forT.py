"""Per-token CuTeDSL kernel — loops over T on host, one kernel launch per token.
Each token launched on its own CUDA stream via explicit CUstream passing."""
import math
import torch
from cuda.bindings import driver as cuda
from letmecook_forT import fused_dsa_single_compiled

# Pre-allocate streams (max T=8) and their CUstream handles
_streams = [torch.cuda.Stream() for _ in range(8)]
_cu_streams = [cuda.CUstream(s.cuda_stream) for s in _streams]

@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, is_profiling=False):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape

    # Flatten paged cache to 2D — just a view, no copy
    ckv_flat = ckv_cache.reshape(-1, head_dim_ckv)   # [N, 512]
    kpe_flat = kpe_cache.reshape(-1, head_dim_kpe)   # [N, 64]

    # Launch each token on its own stream — tokens are fully independent
    for t in range(num_tokens):
        fused_dsa_single_compiled(
            q_nope[t], q_pe[t], ckv_flat, kpe_flat,
            sparse_indices[t], output[t], lse[t],
            _cu_streams[t]
        )

    # Wait for all streams to finish
    main_stream = torch.cuda.current_stream()
    for t in range(num_tokens):
        main_stream.wait_stream(_streams[t])

if __name__ == "__main__":
    from pathlib import Path
    from cook import check_correctness, benchmark
    from ref import run as ref_fn
    ROOT = Path(__file__).parent.parent
    JSONL = str(ROOT.parent / "flashinfer26dsa" / "mlsys26-contest" / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl")

    print("=== Correctness ===")
    check_correctness(run, jsonl_path=JSONL)
    print("\n=== Benchmark ===")
    benchmark(run, ref_fn=ref_fn, jsonl_path=JSONL)

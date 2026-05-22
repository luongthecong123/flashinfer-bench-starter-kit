"""Per-token split: within a single workload, route each token to the best kernel.

Small-valid tokens → CuTeDSL (on their own CUDA streams)
Large-valid tokens → torch.compile batched BMM (grouped + batched together)
Both groups run concurrently on different streams.

This attacks mixed workloads like T=8, valid=[18, 11, 2048, 20, 25, 45, 135, 326]
where most tokens are small but 1-2 tokens have high valid count.
"""
import math
import torch
from cuda.bindings import driver as cuda
from letmecook_forT import fused_dsa_single_compiled

torch.backends.cuda.matmul.allow_tf32 = False

# --- CuTeDSL multi-stream machinery ---
_streams = [torch.cuda.Stream() for _ in range(8)]
_cu_streams = [cuda.CUstream(s.cuda_stream) for s in _streams]

# Extra stream for torch.compile batched path
_tc_stream = torch.cuda.Stream()

# --- torch.compile batched machinery ---
@torch.compile
def compute_attention_batched(qn, qp, Kc, Kp, mask, sm_scale, output, lse):
    logits = torch.bmm(qn, Kc.transpose(1, 2), out_dtype=torch.float32) + torch.bmm(qp, Kp.transpose(1, 2), out_dtype=torch.float32)
    logits.masked_fill_(mask.unsqueeze(1), float('-inf'))
    logits_scaled = logits * sm_scale
    lse.copy_(torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0))
    attn = torch.softmax(logits_scaled, dim=-1)
    output.copy_(torch.bmm(attn, Kc.float()).to(torch.bfloat16))

@torch.compile
def prepare_indices(sparse_indices):
    mask = sparse_indices == -1
    safe_indices = sparse_indices.clamp(min=0).long()
    return mask, safe_indices

# Per-token threshold: if a token's valid count > this, route to torch.compile
TOKEN_THRESHOLD = 100


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, is_profiling=False):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]

    # Compute per-token valid counts (cheap)
    valid_counts = (sparse_indices != -1).sum(dim=1)  # [T]

    # Split tokens into small (CuTeDSL) and large (torch.compile)
    small_mask = valid_counts <= TOKEN_THRESHOLD
    small_indices = torch.where(small_mask)[0]
    large_indices = torch.where(~small_mask)[0]

    n_small = small_indices.numel()
    n_large = large_indices.numel()

    # Flatten paged cache (shared by both paths)
    ckv_flat = ckv_cache.reshape(-1, head_dim_ckv)
    kpe_flat = kpe_cache.reshape(-1, head_dim_kpe)

    # --- Launch CuTeDSL for small tokens on their own streams ---
    if n_small > 0:
        for i in range(n_small):
            t = small_indices[i].item()
            fused_dsa_single_compiled(
                q_nope[t], q_pe[t], ckv_flat, kpe_flat,
                sparse_indices[t], output[t], lse[t],
                _cu_streams[i]
            )

    # --- Launch torch.compile for large tokens (batched) on _tc_stream ---
    if n_large > 0:
        with torch.cuda.stream(_tc_stream):
            # Gather the large-valid-count tokens into contiguous batch
            lg_idx = large_indices  # [n_large] on GPU
            qn_lg = q_nope[lg_idx]           # [n_large, 16, 512]
            qp_lg = q_pe[lg_idx]             # [n_large, 16, 64]
            si_lg = sparse_indices[lg_idx]   # [n_large, 2048]

            mask, safe_indices = prepare_indices(si_lg)
            flat_idx = safe_indices.reshape(-1)
            Kc = ckv_flat[flat_idx].reshape(n_large, topk, head_dim_ckv)
            Kp = kpe_flat[flat_idx].reshape(n_large, topk, head_dim_kpe)

            # Allocate temp output on this stream
            out_lg = torch.empty(n_large, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device=q_nope.device)
            lse_lg = torch.empty(n_large, num_qo_heads, dtype=torch.float32, device=q_nope.device)

            compute_attention_batched(qn_lg, qp_lg, Kc, Kp, mask, sm_scale, out_lg, lse_lg)

            # Scatter results back to original positions
            for i in range(n_large):
                t = large_indices[i].item()
                output[t].copy_(out_lg[i])
                lse[t].copy_(lse_lg[i])

    # --- Synchronize all streams ---
    main_stream = torch.cuda.current_stream()
    if n_small > 0:
        for i in range(n_small):
            main_stream.wait_stream(_streams[i])
    if n_large > 0:
        main_stream.wait_stream(_tc_stream)


if __name__ == "__main__":
    from pathlib import Path
    from cook import check_correctness, benchmark
    from ref import run as ref_fn

    JSONL = "/data/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    import torch as _t
    print(f"GPU: {_t.cuda.get_device_name(0)}")
    print(f"Per-token threshold: {TOKEN_THRESHOLD}\n")

    print("=== Correctness ===")
    check_correctness(run, ref_fn, jsonl_path=JSONL)
    print("\n=== Benchmark ===")
    benchmark(run, ref_fn, jsonl_path=JSONL)

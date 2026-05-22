"""Hybrid dispatch: CuTeDSL for small workloads, torch.compile for large ones.

Decision rule: compute max_valid from sparse_indices on CPU.
  - If max_valid <= THRESHOLD → CuTeDSL multi-stream (per-token, each on own stream)
  - Else → torch.compile batched padded BMM

The crossover on B200 (from benchmarks):
  - CuTeDSL wins when valid counts are small (workloads 1-4, 6-7, 9, 16)
  - torch.compile wins when any token has high valid count (>~200)
"""
import math
import torch
from letmecook import fused_dsa_compiled

torch.backends.cuda.matmul.allow_tf32 = False

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

# Threshold: if max valid count across all tokens > this, use torch.compile
# From B200 data: CuTeDSL wins clearly when max_valid < ~100
# torch.compile is ~0.236ms constant; CuTeDSL scales linearly with max_valid
THRESHOLD = 64


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, is_profiling=False):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]

    # Decision: check if ALL tokens have < THRESHOLD valid entries.
    # sparse_indices is sorted (valid first, then -1 padding), so checking
    # column THRESHOLD-1 tells us: if it's -1 for all rows, all have < THRESHOLD.
    # This reads only T values (max 8) instead of summing T*2048.
    use_cutedsl = sparse_indices[:, THRESHOLD - 1].max().item() < 0

    if use_cutedsl:
        # --- CuTeDSL batched path (single kernel launch) ---
        ckv_flat = ckv_cache.reshape(-1, head_dim_ckv)
        kpe_flat = kpe_cache.reshape(-1, head_dim_kpe)

        fused_dsa_compiled(
            q_nope, q_pe, ckv_flat, kpe_flat,
            sparse_indices, output, lse
        )
    else:
        # --- torch.compile batched path ---
        T = num_tokens
        topk = sparse_indices.shape[-1]
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv)
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe)

        mask, safe_indices = prepare_indices(sparse_indices)
        flat_idx = safe_indices.reshape(-1)
        Kc = Kc_all[flat_idx].reshape(T, topk, head_dim_ckv)
        Kp = Kp_all[flat_idx].reshape(T, topk, head_dim_kpe)
        compute_attention_batched(q_nope, q_pe, Kc, Kp, mask, sm_scale, output, lse)


if __name__ == "__main__":
    from pathlib import Path
    from cook import check_correctness, benchmark
    from ref import run as ref_fn

    JSONL = "/data/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    import torch as _t
    print(f"GPU: {_t.cuda.get_device_name(0)}")
    print(f"Threshold: {THRESHOLD}\n")

    print("=== Correctness ===")
    check_correctness(run, ref_fn, jsonl_path=JSONL)
    print("\n=== Benchmark ===")
    benchmark(run, ref_fn, jsonl_path=JSONL)

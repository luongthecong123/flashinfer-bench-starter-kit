"""Hybrid dispatch with speculative tc-path preparation on a separate stream.

The .item() GPU→CPU sync for the branch decision is overlapped with speculative
tc path preparation (prepare_indices + gather) on a second CUDA stream.

Timeline (tc path — large workloads):
  S0: max() ──────── .item() ────── wait_stream(S1) ── attention ──
  S1:        prepare_indices ── gather ──────────────
  The .item() blocks CPU ~5µs (max() already done), while S1 does useful work.

Timeline (CuTeDSL path — small workloads):
  S0: max() ── .item() ── CuTeDSL kernel ──
  S1:    prepare_indices ── gather ── (wasted, but concurrent with CuTeDSL)
  On B200 (148 SMs), CuTeDSL (16-128 blocks) and S1 prep coexist easily.
"""
import math
import torch
from letmecook import fused_dsa_compiled

torch.backends.cuda.matmul.allow_tf32 = False

THRESHOLD = 64

# Pre-allocate stream to avoid per-call allocation overhead
_spec_stream = None


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


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, is_profiling=False):
    global _spec_stream
    if _spec_stream is None:
        _spec_stream = torch.cuda.Stream()

    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    T = num_tokens
    topk = sparse_indices.shape[-1]

    # 1. Launch decision kernel on default stream (fast, ~3µs GPU)
    decision_val = sparse_indices[:, THRESHOLD - 1].max()

    # 2. Speculatively prepare tc path on a separate stream
    #    These GPU ops execute concurrently with the .item() sync below
    s_tc = _spec_stream
    s_tc.wait_stream(torch.cuda.current_stream())  # ensure sparse_indices is ready
    with torch.cuda.stream(s_tc):
        Kc_all = ckv_cache.reshape(-1, head_dim_ckv)
        Kp_all = kpe_cache.reshape(-1, head_dim_kpe)
        mask, safe_indices = prepare_indices(sparse_indices)
        flat_idx = safe_indices.reshape(-1)
        Kc = Kc_all[flat_idx].reshape(T, topk, head_dim_ckv)
        Kp = Kp_all[flat_idx].reshape(T, topk, head_dim_kpe)

    # 3. CPU blocks here waiting for max() result (~5µs since max() already done)
    #    Meanwhile, s_tc continues executing prep on GPU
    use_cutedsl = decision_val.item() < 0

    if use_cutedsl:
        # CuTeDSL path — s_tc prep runs concurrently but result is discarded
        ckv_flat = ckv_cache.reshape(-1, head_dim_ckv)
        kpe_flat = kpe_cache.reshape(-1, head_dim_kpe)
        fused_dsa_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)
    else:
        # TC path — wait for speculative prep to finish, then run attention
        torch.cuda.current_stream().wait_stream(s_tc)
        compute_attention_batched(q_nope, q_pe, Kc, Kp, mask, sm_scale, output, lse)


if __name__ == "__main__":
    from cook import check_correctness, benchmark
    from ref import run as ref_fn

    JSONL = "/data/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    import torch as _t
    print(f"GPU: {_t.cuda.get_device_name(0)}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Mode: speculative overlap\n")

    print("=== Correctness ===")
    check_correctness(run, ref_fn, jsonl_path=JSONL)
    print("\n=== Benchmark ===")
    benchmark(run, ref_fn, jsonl_path=JSONL)

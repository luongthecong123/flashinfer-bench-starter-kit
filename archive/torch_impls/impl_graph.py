"""Standalone torch.compile with mode='reduce-overhead' (CUDA graph).

Same logic as impl.py but with CUDA graph capture via mode='reduce-overhead'.
This establishes the best-possible tc latency with graph replay,
so we can isolate whether the hybrid overhead is from .item() or from dispatch.
"""
import math
import torch

torch.backends.cuda.matmul.allow_tf32 = False


@torch.compile(mode="reduce-overhead")
def tc_fused(q_nope, q_pe, Kc_all, Kp_all, sparse_indices, sm_scale):
    mask = sparse_indices == -1
    safe_indices = sparse_indices.clamp(min=0).long()
    flat_idx = safe_indices.reshape(-1)
    T = q_nope.shape[0]
    topk = 2048
    Kc = Kc_all[flat_idx].reshape(T, topk, 512)
    Kp = Kp_all[flat_idx].reshape(T, topk, 64)
    logits = torch.bmm(q_nope, Kc.transpose(1, 2), out_dtype=torch.float32) + \
             torch.bmm(q_pe, Kp.transpose(1, 2), out_dtype=torch.float32)
    logits.masked_fill_(mask.unsqueeze(1), float('-inf'))
    logits_scaled = logits * sm_scale
    log_sum_exp = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
    attn = torch.softmax(logits_scaled, dim=-1)
    out = torch.bmm(attn, Kc.float()).to(torch.bfloat16)
    return out, log_sum_exp


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, is_profiling=False):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]

    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)
    out, log_sum_exp = tc_fused(q_nope, q_pe, Kc_all, Kp_all, sparse_indices, sm_scale)
    output.copy_(out)
    lse.copy_(log_sum_exp)


if __name__ == "__main__":
    from cook import check_correctness, benchmark
    from ref import run as ref_fn

    JSONL = "/data/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    import torch as _t
    print(f"GPU: {_t.cuda.get_device_name(0)}")
    print(f"TC mode: reduce-overhead (CUDA graph)\n")

    print("=== Correctness ===")
    check_correctness(run, ref_fn, jsonl_path=JSONL)
    print("\n=== Benchmark ===")
    benchmark(run, ref_fn, jsonl_path=JSONL, warmup=20)

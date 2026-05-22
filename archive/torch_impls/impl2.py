import math
import torch

# Disable TF32 — use true f32 or bf16→f32 accumulation for precision
torch.backends.cuda.matmul.allow_tf32 = False


@torch.compile
def compute_attention_batched(q_combined, Kc, K_combined, mask, sm_scale, output, lse):
    """Batched padded attention with fused q concat and bf16 output BMM.

    Inputs:
        q_combined: [T, 16, 576]   bf16  — cat(qn, qp) along head dim
        Kc:         [T, 2048, 512] bf16  — gathered compressed KV (padded)
        K_combined: [T, 2048, 576] bf16  — cat(Kc, Kp) along head dim
        mask:       [T, 2048]      bool  — True for INVALID (padding) positions
        sm_scale:   float                — 1/sqrt(192)
    """
    # Single fused score BMM: [T,16,576] @ [T,576,2048] → [T,16,2048] f32
    logits = torch.bmm(q_combined, K_combined.transpose(1, 2), out_dtype=torch.float32)

    # Mask invalid positions to -inf
    logits.masked_fill_(mask.unsqueeze(1), float('-inf'))
    logits_scaled = logits * sm_scale

    # LSE and softmax
    lse.copy_(torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0))
    attn = torch.softmax(logits_scaled, dim=-1)

    # Output BMM in bf16: cast attn to bf16, bmm with bf16 Kc directly
    output.copy_(torch.bmm(attn.to(torch.bfloat16), Kc))


@torch.compile
def prepare_indices(sparse_indices):
    """Fused mask + safe indices computation."""
    mask = sparse_indices == -1
    safe_indices = sparse_indices.clamp(min=0).long()
    return mask, safe_indices


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, is_profiling=False):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64
    assert topk == 2048
    assert sparse_indices.shape[0] == num_tokens
    assert ckv_cache.shape[1] == page_size

    T = num_tokens

    # Flatten paged KV cache: [num_pages, 64, D] → [num_pages*64, D]
    if is_profiling: torch.cuda.nvtx.range_push('flatten_kv_cache')
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)   # [total, 512] bf16
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)    # [total, 64]  bf16
    if is_profiling: torch.cuda.nvtx.range_pop()

    # Fused mask + safe indices
    if is_profiling: torch.cuda.nvtx.range_push('prepare_indices')
    mask, safe_indices = prepare_indices(sparse_indices)
    if is_profiling: torch.cuda.nvtx.range_pop()

    # Batched gather: [T, 2048, D]
    if is_profiling: torch.cuda.nvtx.range_push('gather')
    flat_idx = safe_indices.reshape(-1)
    Kc = Kc_all[flat_idx].reshape(T, topk, head_dim_ckv)
    Kp = Kp_all[flat_idx].reshape(T, topk, head_dim_kpe)
    if is_profiling: torch.cuda.nvtx.range_pop()

    # Concat queries: [T, 16, 512] + [T, 16, 64] → [T, 16, 576]
    if is_profiling: torch.cuda.nvtx.range_push('concat')
    q_combined = torch.cat([q_nope, q_pe], dim=-1)
    # Concat keys: [T, 2048, 512] + [T, 2048, 64] → [T, 2048, 576]
    K_combined = torch.cat([Kc, Kp], dim=-1)
    if is_profiling: torch.cuda.nvtx.range_pop()

    # Batched attention
    if is_profiling: torch.cuda.nvtx.range_push('compute_attention_batched')
    compute_attention_batched(q_combined, Kc, K_combined, mask, sm_scale, output, lse)
    if is_profiling: torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    from pathlib import Path
    from cook import check_correctness
    ROOT = Path(__file__).parent.parent
    JSONL = str(ROOT.parent / "flashinfer26dsa" / "mlsys26-contest" / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl")
    MAX_SAMPLES = None
    check_correctness(run, jsonl_path=JSONL, max_samples=MAX_SAMPLES)

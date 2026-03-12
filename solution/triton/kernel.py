import math
import torch

# Disable TF32 — use true f32 or bf16→f32 accumulation for precision
torch.backends.cuda.matmul.allow_tf32 = False


@torch.compile
def compute_attention_batched(qn, qp, Kc, Kp, mask, sm_scale):
    """Batched padded attention — single fused kernel for all T tokens.

    Inputs:
        qn:       [T, 16, 512]    bf16  — query nope
        qp:       [T, 16, 64]     bf16  — query PE
        Kc:       [T, 2048, 512]  bf16  — gathered compressed KV (padded)
        Kp:       [T, 2048, 64]   bf16  — gathered key PE (padded)
        mask:     [T, 2048]       bool  — True for INVALID (padding) positions
        sm_scale: float                  — 1/sqrt(192)

        logits  = einsum('thd,tvd->thv', qn, Kc)  → [T, 16, 2048] f32  (nope score)
                + einsum('thd,tvd->thv', qp, Kp)  → [T, 16, 2048] f32  (PE score)
        logits[mask] = -inf                                              (mask padding)
        logits_scaled = logits * sm_scale          → [T, 16, 2048] f32
        lse = logsumexp(logits_scaled, dim=-1)/ln2 → [T, 16]       f32
        attn = softmax(logits_scaled, dim=-1)      → [T, 16, 2048] f32
        out = einsum('thv,tvd->thd', attn, Kc)    → [T, 16, 512]  bf16

    Outputs:
        out: [T, 16, 512]  bf16
        lse: [T, 16]       f32
    """
    qn_f = qn.float()
    qp_f = qp.float()
    Kc_f = Kc.float()
    Kp_f = Kp.float()

    # Score matmuls
    logits = torch.bmm(qn_f, Kc_f.transpose(1, 2)) + torch.bmm(qp_f, Kp_f.transpose(1, 2))

    # Mask invalid positions to -inf
    logits.masked_fill_(mask.unsqueeze(1), float('-inf'))
    logits_scaled = logits * sm_scale

    # LSE and softmax
    token_lse = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)  # [T, 16]
    attn = torch.softmax(logits_scaled, dim=-1)  # [T, 16, 2048]

    # Output matmul in f32 for precision, then cast
    out = torch.bmm(attn, Kc_f).to(torch.bfloat16)

    return out, token_lse


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
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
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)   # [total, 512] bf16
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)    # [total, 64]  bf16

    # Build mask and safe indices (replace -1 with 0)
    mask = sparse_indices == -1                     # [T, 2048] True = invalid
    safe_indices = sparse_indices.clamp(min=0).long()  # [T, 2048]

    # Batched gather: [T, 2048, D]
    Kc = Kc_all[safe_indices.reshape(-1)].reshape(T, topk, head_dim_ckv)
    Kp = Kp_all[safe_indices.reshape(-1)].reshape(T, topk, head_dim_kpe)

    # Single batched attention call
    out, token_lse = compute_attention_batched(q_nope, q_pe, Kc, Kp, mask, sm_scale)

    output.copy_(out)
    lse.copy_(token_lse)

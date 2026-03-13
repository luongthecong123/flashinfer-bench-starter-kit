import math
import torch

# Disable TF32 — use true f32 or bf16→f32 accumulation for precision
torch.backends.cuda.matmul.allow_tf32 = False


@torch.compile
def compute_attention_batched(qn, qp, Kc, Kp, mask, sm_scale):
    """Batched padded attention — transposed layout [T, *, 16] throughout.

    Transpose queries once at entry, work in [T, KV, heads] layout,
    transpose output once at exit. Softmax runs over dim=1 (the KV dim).

    Original layout:    [T, 16, 2048]   softmax dim=-1   M=16 (bad)
    Transposed layout:  [T, 2048, 16]   softmax dim=1    M=2048 (good)

    Inputs:
        qn:       [T, 16, 512]    bf16
        qp:       [T, 16, 64]     bf16
        Kc:       [T, 2048, 512]  bf16
        Kp:       [T, 2048, 64]   bf16
        mask:     [T, 2048]       bool  — True for INVALID positions
        sm_scale: float

    Outputs:
        out: [T, 16, 512]  bf16
        lse: [T, 16]       f32
    """
    # Transpose queries once → [T, D, 16]
    qn_t = qn.float().transpose(1, 2)  # [T, 512, 16]
    qp_t = qp.float().transpose(1, 2)  # [T, 64, 16]
    Kc_f = Kc.float()                  # [T, 2048, 512]
    Kp_f = Kp.float()                  # [T, 2048, 64]

    # Score: [T, 2048, 512] @ [T, 512, 16] → [T, 2048, 16]   M=2048 ✓
    logits = torch.bmm(Kc_f, qn_t) + torch.bmm(Kp_f, qp_t)  # [T, 2048, 16]

    # Mask + scale (mask broadcasts over heads on dim=-1)
    logits.masked_fill_(mask.unsqueeze(-1), float('-inf'))
    logits.mul_(sm_scale)

    # Softmax over KV dim=1 (the 2048 dimension)
    token_lse = torch.logsumexp(logits, dim=1) / math.log(2.0)  # [T, 16]
    attn = torch.softmax(logits, dim=1)                          # [T, 2048, 16]

    # Output: [T, 512, 2048] @ [T, 2048, 16] → [T, 512, 16]   M=512 ✓
    out = torch.bmm(Kc_f.transpose(1, 2), attn)  # [T, 512, 16]

    # Transpose once at exit → [T, 16, 512]
    out = out.transpose(1, 2).to(torch.bfloat16)

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

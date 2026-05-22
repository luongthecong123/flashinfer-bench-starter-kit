"""Sliced torch.compile DSA attention.

Instead of padding ALL tokens to 2048, find the max valid count across
all tokens and slice sparse_indices to that. Single batched BMM, no loop.
"""
import math
import torch

torch.backends.cuda.matmul.allow_tf32 = False

PAD_ALIGN = 128


@torch.compile
def compute_attention_batched(qn, qp, Kc, Kp, mask, sm_scale, output, lse):
    logits = torch.bmm(qn, Kc.transpose(1, 2), out_dtype=torch.float32) + \
             torch.bmm(qp, Kp.transpose(1, 2), out_dtype=torch.float32)
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
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    T = num_tokens

    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)

    # Find max valid count across all tokens, pad to nearest PAD_ALIGN
    max_valid = (sparse_indices != -1).sum(dim=1).max().item()
    pad_n = min(((max_valid + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN, 2048)

    # Slice to [T, pad_n]
    sparse_sliced = sparse_indices[:, :pad_n]

    mask, safe_indices = prepare_indices(sparse_sliced)

    flat_idx = safe_indices.reshape(-1)
    Kc = Kc_all[flat_idx].reshape(T, pad_n, head_dim_ckv)
    Kp = Kp_all[flat_idx].reshape(T, pad_n, head_dim_kpe)

    compute_attention_batched(q_nope, q_pe, Kc, Kp, mask, sm_scale, output, lse)

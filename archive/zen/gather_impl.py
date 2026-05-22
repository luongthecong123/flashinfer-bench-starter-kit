"""DSA sparse attention using CuTe DSL gather kernel + torch.compile attention."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from zen.gather import gather_compiled
from zen.dsa import fused_dsa_compiled


@torch.compile
def compute_attention_batched(qn, qp, Kc, Kp, mask, sm_scale, output, lse):
    """Batched padded attention — single fused kernel for all T tokens.

    Inputs:
        qn:       [T, 16, 512]    bf16
        qp:       [T, 16, 64]     bf16
        Kc:       [T, 2048, 512]  bf16
        Kp:       [T, 2048, 64]   bf16
        mask:     [T, 2048]       bool  — True for INVALID positions
        sm_scale: float
    Outputs (written in-place):
        output: [T, 16, 512]  bf16
        lse:    [T, 16]       f32
    """
    logits = torch.bmm(qn, Kc.transpose(1, 2), out_dtype=torch.float32) + \
             torch.bmm(qp, Kp.transpose(1, 2), out_dtype=torch.float32)
    logits.masked_fill_(mask.unsqueeze(1), float('-inf'))
    logits_scaled = logits * sm_scale
    lse.copy_(torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0))
    attn = torch.softmax(logits_scaled, dim=-1)
    output.copy_(torch.bmm(attn.float(), Kc.float())).bfloat16()


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]
    # Async D2H copy of sparse_indices — runs concurrently with gather kernel

    
    T = num_tokens
    # Build mask from sparse_indices for attention masking
    mask = sparse_indices == -1  # [T, 2048]
    # Flatten paged KV cache: [num_pages, 64, D] → [num_pages*64, D]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)   # [total, 512] bf16
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)    # [total, 64]  bf16

    # Allocate gathered outputs
    Kc = torch.empty(T, topk, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
    Kp = torch.empty(T, topk, head_dim_kpe, dtype=torch.bfloat16, device="cuda")
    max_valid = torch.empty(T, dtype=torch.int32, device="cuda")

    # CuTe DSL gather kernel — runs concurrently with D2H copy
    gather_compiled(Kc_all, Kp_all, sparse_indices, Kc, Kp, max_valid)
    # sparse_indices_cpu = sparse_indices.to("cpu")
    # CPU computation of max_valid (D2H should be done by now)
    # max_valid_cpu = (sparse_indices_cpu != -1).sum(dim=1).int()  


    # Batched attention — writes directly into output/lse
    compute_attention_batched(q_nope, q_pe, Kc, Kp, mask, sm_scale, output, lse)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))
    import cook
    cook.impl_fn = run
    cook.CHECK = True
    cook.MEASURE = False
    cook.TOY_CHECK = False
    cook.main()

import math
import torch

torch.backends.cuda.matmul.allow_tf32 = False


@torch.compile
def prepare_indices(sparse_indices):
    """Fused mask + safe indices computation."""
    mask = sparse_indices == -1
    safe_indices = sparse_indices.clamp(min=0).long()
    return mask, safe_indices


@torch.compile
def compute_attention_batched(qn, qp, Kc, Kp, mask, sm_scale, output, lse):
    """Batched padded attention — score BMMs in f32, output BMM in bf16."""
    logits = torch.bmm(qn, Kc.transpose(1, 2), out_dtype=torch.float32) + \
             torch.bmm(qp, Kp.transpose(1, 2), out_dtype=torch.float32)
    logits.masked_fill_(mask.unsqueeze(1), float('-inf'))
    logits_scaled = logits * sm_scale
    lse.copy_(torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0))
    attn = torch.softmax(logits_scaled, dim=-1)
    output.copy_(torch.bmm(attn.float(), Kc.float()).bfloat16())


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]
    T = num_tokens

    # Flatten paged KV cache: [num_pages, 64, D] → [num_pages*64, D]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)   # [total, 512] bf16
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)    # [total, 64]  bf16

    # Fused mask + safe indices
    mask, safe_indices = prepare_indices(sparse_indices)

    # Batched gather: [T, 2048, D]
    flat_idx = safe_indices.reshape(-1)
    Kc = Kc_all[flat_idx].reshape(T, topk, head_dim_ckv)
    Kp = Kp_all[flat_idx].reshape(T, topk, head_dim_kpe)

    compute_attention_batched(q_nope, q_pe, Kc, Kp, mask, sm_scale, output, lse)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import utils
    from src.ref import run as ref_run
    utils.ref_fn = ref_run
    utils.impl_fn = run
    utils.CHECK = True
    utils.MEASURE = False
    utils.TOY_CHECK = False
    utils.main()

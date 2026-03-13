import math
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from b2_wmma_smem_batched import Gemm_TC_Batched

# Disable TF32 — use true f32 or bf16→f32 accumulation for precision
torch.backends.cuda.matmul.allow_tf32 = False

# One-time WMMA kernel compilation (bf16 in, f32 out, C = A @ B^T)
_kernel = None

def _compile_kernel():
    global _kernel
    if _kernel is None:
        A = torch.empty(1, 16, 512, device='cuda', dtype=torch.bfloat16)
        B = torch.empty(1, 2048, 512, device='cuda', dtype=torch.bfloat16)
        C = torch.empty(1, 16, 2048, device='cuda', dtype=torch.float32)
        _kernel = cute.compile(
            Gemm_TC_Batched(),
            from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=A.dim_order()).mark_compact_shape_dynamic(mode=2, divisibility=64),
            from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=B.dim_order()).mark_compact_shape_dynamic(mode=2, divisibility=64),
            from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=C.dim_order()),
        )
    return _kernel

def _wrap(t, divisible_k=False):
    w = from_dlpack(t, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=t.dim_order())
    if divisible_k:
        w = w.mark_compact_shape_dynamic(mode=2, divisibility=64)
    return w

def cute_bmm(A, B, C):
    """C = A @ B^T  (bf16 → f32) via WMMA kernel."""
    _compile_kernel()(_wrap(A, True), _wrap(B, True), _wrap(C))

def compute_attention_batched(qn, qp, Kc, Kp, mask, sm_scale):
    """Batched padded attention — same interface as impl.py version.

    Same as original except first 2 score BMMs use cute WMMA kernel
    (bf16 → f32 natively) instead of torch.bmm on float-casted tensors.
    """
    T = qn.shape[0]

    # Score matmuls via cute WMMA (replaces torch.bmm(qn_f, Kc_f.T) etc.)
    logits_nope = torch.empty(T, 16, 2048, device=qn.device, dtype=torch.float32)
    logits_pe   = torch.empty(T, 16, 2048, device=qn.device, dtype=torch.float32)
    cute_bmm(qn, Kc, logits_nope)
    cute_bmm(qp, Kp, logits_pe)
    logits = logits_nope + logits_pe

    # Mask invalid positions to -inf
    logits.masked_fill_(mask.unsqueeze(1), float('-inf'))
    logits_scaled = logits * sm_scale

    # LSE and softmax
    token_lse = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
    attn = torch.softmax(logits_scaled, dim=-1)

    # Output matmul in f32 for precision, then cast
    Kc_f = Kc.float()
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

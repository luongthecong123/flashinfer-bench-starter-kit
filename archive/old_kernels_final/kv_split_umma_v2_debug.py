"""Debug wrapper: GPU kernel writes lse directly (Phase 2).
Output is computed purely in PyTorch so only lse correctness is tested.
"""
import torch

from src.kernels.kv_split_umma_v2 import (
    _hybrid, _compiled,
    NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE,
    NUM_PAGES, PAGE_SIZE, FLAT_CACHE,
)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    # ── Step 1: GPU kernel — writes lse directly (Phase 2) ──────────────────
    _compiled(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
              _hybrid.partial_out, _hybrid.partial_lse, output, lse)
    torch.cuda.synchronize()
    # lse is now written by the GPU — do NOT overwrite it.

    # ── Step 2: PyTorch output (independent of GPU, so output never masks lse error) ──
    T = q_nope.shape[0]
    dev = q_nope.device

    ckv_flat = ckv_cache.view(FLAT_CACHE, HEAD_DIM_CKV).float()
    kpe_flat = kpe_cache.view(FLAT_CACHE, HEAD_DIM_KPE).float()
    q_n = q_nope.float()
    q_p = q_pe.float()

    out_f = torch.zeros(T, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.float32, device=dev)
    for t in range(T):
        all_idx = sparse_indices[t].long()
        valid_mask = (all_idx >= 0) & (all_idx < FLAT_CACHE)
        valid_idx = all_idx[valid_mask]
        if valid_idx.numel() == 0:
            continue
        kv_c = ckv_flat[valid_idx]   # [N, Dc]
        kv_p = kpe_flat[valid_idx]   # [N, Dp]
        sc = (q_n[t] @ kv_c.T + q_p[t] @ kv_p.T) * sm_scale  # [H, N]
        softmax_w = torch.softmax(sc, dim=-1)                   # [H, N]
        out_f[t] = softmax_w @ kv_c                             # [H, Dc]

    output.copy_(out_f.to(output.dtype))

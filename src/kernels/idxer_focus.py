"""idxer_focus.py — focused slow-path indexer, batch_size=1, seq_len > 2048.

This kernel handles ONLY the case where the single request's seq_len exceeds
the TOPK budget (2048).  All fast-path / batch-dimension complexity is stripped.

Interface matches every other idxer kernel:
    run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)

Caller guarantees:
  - q_index_fp8.shape  == [1, 64, 128]
  - weights.shape      == [1, 64]
  - seq_lens.shape     == [1],  seq_lens[0] > 2048
  - block_table.shape  == [1, max_num_pages]
  - topk_indices.shape == [1, 2048]  (output, pre-allocated, filled with -1)
"""
import torch

PAGE_SIZE = 64
HEAD_DIM  = 128
NUM_HEADS = 64
TOPK      = 2048


def dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV-cache from deep_gemm packing.

    Input:  [pool_pages, page_size, 1, 132] int8 (uint8 reinterpret)
    Output: [pool_pages, page_size, 128] float32
    """
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4  # 132 - 4 = 128

    kv_flat    = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)
    fp8_bytes  = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float  = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale       = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float * scale


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    """Slow-path top-K indexer for a single request with seq_len > TOPK.

    Steps:
      1. Dequantize the FP8 KV pool.
      2. Gather only the pages belonging to this request.
      3. Compute per-head dot products, ReLU, weighted-sum across heads.
      4. top-K select and map indices back to global token IDs.
    """
    seq_len = int(seq_lens[0].item())
    device  = q_index_fp8.device

    # ── 1. Dequantize entire KV pool ──────────────────────────────────────────
    K_all  = dequant_fp8_kv_cache(k_index_cache_fp8)   # [pool_pages, 64, 128]
    K_flat = K_all.reshape(-1, HEAD_DIM)                # [pool_pages * 64, 128]

    # ── 2. Gather tokens for this request ────────────────────────────────────
    num_pages_for_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    pages   = block_table[0, :num_pages_for_seq].long()  # [P]

    offsets      = torch.arange(PAGE_SIZE, device=device)
    flat_tok_ids = (pages.unsqueeze(1) * PAGE_SIZE
                    + offsets.unsqueeze(0)).reshape(-1)  # [P * 64]

    K_seq = K_flat[flat_tok_ids][:seq_len]  # [seq_len, 128]  — trim padding

    # ── 3. Score and reduce ───────────────────────────────────────────────────
    q = q_index_fp8[0].to(torch.float32)        # [64, 128]
    # [64, 128] @ [128, seq_len] -> [64, seq_len]
    scores = torch.mm(q, K_seq.T)
    scores = torch.relu(scores)

    w     = weights[0]                           # [64]
    final = (scores * w.unsqueeze(1)).sum(dim=0) # [seq_len]

    # ── 4. Top-K and map to global token IDs ─────────────────────────────────
    actual_topk = min(TOPK, seq_len)
    _, topk_idx = torch.topk(final, actual_topk, dim=0)  # [actual_topk]

    # topk_idx is in [0, seq_len); map back through block table
    page_of_tok  = topk_idx // PAGE_SIZE   # which page slot  (into `pages`)
    off_of_tok   = topk_idx %  PAGE_SIZE   # offset within page

    global_page  = pages[page_of_tok]                                         # actual page number
    global_toks  = (global_page * PAGE_SIZE + off_of_tok).to(torch.int32)

    topk_indices.fill_(-1)
    topk_indices[0, :actual_topk] = global_toks

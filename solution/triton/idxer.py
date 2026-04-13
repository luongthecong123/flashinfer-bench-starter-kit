import torch

PAGE_SIZE = 64
HEAD_DIM = 128
NUM_HEADS = 64
TOPK = 2048


@torch.compile
def dequant_fp8_kv_cache(k_index_cache_fp8):
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4

    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)
    fp8_bytes = kv_flat[:, :page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim:].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float * scale


@torch.compile
def _score_and_reduce(q, K_gathered, weights, mask):
    """Batched score + relu + weighted reduce.
    q:          [B, 64, 128]   f32
    K_gathered: [B, max_sl, 128] f32
    weights:    [B, 64]        f32
    mask:       [B, max_sl]    bool  (True = padding)
    returns:    [B, max_sl]    f32   (padded positions = -inf)
    """
    # [B, 64, 128] @ [B, 128, max_sl] -> [B, 64, max_sl]
    scores = torch.bmm(q, K_gathered.transpose(1, 2))
    scores = torch.relu(scores)
    # weighted sum over heads: [B, max_sl]
    final = torch.einsum("bhs,bh->bs", scores, weights)
    # mask padding to -inf so topk ignores them
    final.masked_fill_(mask, float("-inf"))
    return final


@torch.compile
def _topk_remap_and_write(final, block_table, mask, topk_indices, topk, page_size):
    actual_k = min(topk, final.shape[1])
    _, topk_idx = torch.topk(final, actual_k, dim=1)
    topk_page = topk_idx // page_size
    topk_off = topk_idx % page_size
    global_pages = torch.gather(block_table, 1, topk_page)
    global_tokens = (global_pages * page_size + topk_off).to(torch.int32)
    invalid = torch.gather(mask, 1, topk_idx)
    global_tokens[invalid] = -1
    topk_indices.fill_(-1)
    topk_indices[:, :actual_k] = global_tokens


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    B, num_index_heads, index_head_dim = q_index_fp8.shape
    device = q_index_fp8.device

    q = q_index_fp8.to(torch.float32)  # [B, 64, 128]
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages, 64, 128]
    K_flat = K_all.reshape(-1, HEAD_DIM)  # [num_pages*64, 128]

    max_num_pages = block_table.shape[1]
    max_sl = max_num_pages * PAGE_SIZE  # upper bound on seq_len

    # Build [B, max_sl] gather index from block_table
    # block_table: [B, max_num_pages] int32 -> page indices
    # Each page has PAGE_SIZE=64 tokens, so token index = page_idx * 64 + offset
    offsets = torch.arange(PAGE_SIZE, device=device)  # [64]
    # [B, max_num_pages, 64] -> token indices into K_flat
    token_indices = (block_table.long().unsqueeze(2) * PAGE_SIZE +
                     offsets.view(1, 1, PAGE_SIZE))  # [B, max_num_pages, 64]
    token_indices = token_indices.reshape(B, max_sl)  # [B, max_sl]

    # Build mask: True for positions >= seq_len (padding)
    positions = torch.arange(max_sl, device=device).unsqueeze(0)  # [1, max_sl]
    mask = positions >= seq_lens.unsqueeze(1)  # [B, max_sl]

    # Clamp indices for safe gather (masked positions will be ignored)
    token_indices = token_indices.clamp(0, K_flat.shape[0] - 1)

    # Gather: [B, max_sl, 128]
    K_gathered = K_flat[token_indices.reshape(-1)].reshape(B, max_sl, HEAD_DIM)

    # Score + reduce
    final = _score_and_reduce(q, K_gathered, weights, mask)

    # Top-k + remap + write output
    _topk_remap_and_write(final, block_table.long(), mask, topk_indices, TOPK, PAGE_SIZE)

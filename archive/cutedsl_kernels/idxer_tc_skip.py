import torch


PAGE_SIZE = 64
HEAD_DIM  = 128
NUM_HEADS = 64
TOPK      = 2048


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
    # [B,64,128] @ [B,128,max_sl] -> [B,64,max_sl]
    scores = torch.bmm(q, K_gathered.transpose(1, 2))
    scores = torch.relu(scores)
    # [B,64,max_sl] * [B,64] -> [B,max_sl]
    final = torch.einsum("bhs,bh->bs", scores, weights)
    final.masked_fill_(mask, float("-inf"))
    return final


@torch.compile
def _fast_path(block_table, seq_lens, topk_indices, page_size: int):
    """max_sl <= TOPK: skip GEMM, scatter all valid global token ids directly."""
    B, max_num_pages = block_table.shape
    device = block_table.device
    max_sl = max_num_pages * page_size

    offsets = torch.arange(page_size, device=device)
    global_tokens = (block_table.long().unsqueeze(2) * page_size
                     + offsets.view(1, 1, page_size))              # [B, max_num_pages, 64]
    global_tokens = global_tokens.reshape(B, max_sl).to(torch.int32)

    positions = torch.arange(max_sl, device=device).unsqueeze(0)
    global_tokens[positions >= seq_lens.unsqueeze(1)] = -1

    topk_indices.fill_(-1)
    topk_indices[:, :max_sl] = global_tokens


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    B             = block_table.shape[0]
    max_num_pages = block_table.shape[1]
    max_sl        = max_num_pages * PAGE_SIZE
    device        = q_index_fp8.device

    # --- fast path: all tokens already fit in the TOPK budget ---
    if max_sl <= TOPK:
        _fast_path(block_table, seq_lens, topk_indices, PAGE_SIZE)
        return

    # --- slow path: need GEMM + top-k selection ---
    q      = q_index_fp8.to(torch.float32)                        # [B, 64, 128]
    K_all  = dequant_fp8_kv_cache(k_index_cache_fp8)             # [num_pages, 64, 128]
    K_flat = K_all.reshape(-1, HEAD_DIM)

    offsets       = torch.arange(PAGE_SIZE, device=device)
    token_indices = (block_table.long().unsqueeze(2) * PAGE_SIZE
                     + offsets.view(1, 1, PAGE_SIZE))             # [B, max_num_pages, 64]
    token_indices = token_indices.reshape(B, max_sl)
    token_indices = token_indices.clamp(0, K_flat.shape[0] - 1)

    positions = torch.arange(max_sl, device=device).unsqueeze(0)
    mask      = positions >= seq_lens.unsqueeze(1)                # [B, max_sl]

    K_gathered = K_flat[token_indices.reshape(-1)].reshape(B, max_sl, HEAD_DIM)
    final      = _score_and_reduce(q, K_gathered, weights, mask)

    _, topk_idx   = torch.topk(final, TOPK, dim=1)               # [B, TOPK]
    topk_page     = topk_idx // PAGE_SIZE
    topk_off      = topk_idx %  PAGE_SIZE
    global_pages  = torch.gather(block_table.long(), 1, topk_page)
    global_tokens = (global_pages * PAGE_SIZE + topk_off).to(torch.int32)
    global_tokens[torch.gather(mask, 1, topk_idx)] = -1

    topk_indices.fill_(-1)
    topk_indices[:, :TOPK] = global_tokens

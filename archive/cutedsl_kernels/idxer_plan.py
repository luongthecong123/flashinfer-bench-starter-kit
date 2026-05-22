import torch

PAGE_SIZE = 64
HEAD_DIM  = 128
NUM_HEADS = 64
TOPK      = 2048


def kernel_gather_dequant_gemm_reduce(
    k_cache_fp8:  torch.Tensor,   # [num_pages, 64, 1, 132]  int8 (fp8+scale bytes)
    block_table:  torch.Tensor,   # [B, max_num_pages]        int32
    seq_lens:     torch.Tensor,   # [B]                       int32
    q_fp8:        torch.Tensor,   # [B, 64, 128]              float8_e4m3fn
    weights:      torch.Tensor,   # [B, 64]                   float32
    final:        torch.Tensor,   # [B, max_sl]               float32  (output)
    mask:         torch.Tensor,   # [B, max_sl]               bool     (output)
):
    q = q_fp8.to(torch.float32)
    num_pages = k_cache_fp8.shape[0]
    k_uint8 = k_cache_fp8.view(torch.uint8)

    bt = block_table.long().clamp(0, num_pages - 1)
    pages = k_uint8[bt]                                            # [B, max_num_pages, 64, 1, 132]

    B, max_num_pages, ps, _, head_dim_sf = pages.shape
    head_dim = head_dim_sf - 4
    max_sl   = max_num_pages * ps

    pages_flat = pages.reshape(B, max_num_pages, ps * head_dim_sf)

    fp8_bytes = pages_flat[:, :, : ps * head_dim].contiguous()
    K_fp32 = (fp8_bytes
              .reshape(B * max_num_pages, ps, head_dim)
              .view(torch.float8_e4m3fn)
              .to(torch.float32)
              .reshape(B, max_num_pages, ps, head_dim))

    scale_bytes = pages_flat[:, :, ps * head_dim :].contiguous()
    scale = (scale_bytes
             .reshape(B * max_num_pages, ps, 4)
             .view(torch.float32)
             .reshape(B, max_num_pages, ps, 1))

    K_gathered = (K_fp32 * scale).reshape(B, max_sl, head_dim)    # [B, max_sl, 128]

    # [B,64,128] @ [B,128,max_sl] -> [B,64,max_sl]
    scores = torch.bmm(q, K_gathered.transpose(1, 2))
    scores = torch.relu(scores)
    # [B,64,max_sl] * [B,64] -> [B,max_sl]  (weighted sum over heads)
    out = torch.einsum("bhs,bh->bs", scores, weights)

    device    = k_cache_fp8.device
    positions = torch.arange(max_sl, device=device).unsqueeze(0)
    mask.copy_(positions >= seq_lens.unsqueeze(1))
    out.masked_fill_(mask, float("-inf"))
    final.copy_(out)


@torch.compile
def kernel_topk_remap(
    scores:       torch.Tensor,   # [B, max_sl]        float32
    block_table:  torch.Tensor,   # [B, max_num_pages] int64
    mask:         torch.Tensor,   # [B, max_sl]        bool
    topk_indices: torch.Tensor,   # [B, 2048]          int32  (output, filled in-place)
    topk:         int,
    page_size:    int,
):
    actual_k   = min(topk, scores.shape[1])
    _, topk_idx = torch.topk(scores, actual_k, dim=1)

    page_slot  = topk_idx // page_size
    offset     = topk_idx %  page_size

    global_page  = torch.gather(block_table, 1, page_slot)
    global_token = (global_page * page_size + offset).to(torch.int32)

    invalid = torch.gather(mask, 1, topk_idx)
    global_token[invalid] = -1

    topk_indices.fill_(-1)
    topk_indices[:, :actual_k] = global_token


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    B, max_num_pages = block_table.shape
    max_sl = max_num_pages * PAGE_SIZE
    device = k_index_cache_fp8.device
    final = torch.empty(B, max_sl, dtype=torch.float32, device=device)
    mask  = torch.empty(B, max_sl, dtype=torch.bool,    device=device)
    kernel_gather_dequant_gemm_reduce(
        k_index_cache_fp8, block_table, seq_lens, q_index_fp8, weights, final, mask
    )
    kernel_topk_remap(final, block_table.long(), mask, topk_indices, TOPK, PAGE_SIZE)

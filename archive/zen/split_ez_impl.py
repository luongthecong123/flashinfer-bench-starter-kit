"""Flash-decoding style DSA: gather + split-K attention + reduction.

Simplified version — no row-max stability trick in split_attention.
  1. split_attention(): one call per (token, split, head_block)
  2. reduce(): combines partial results across splits
"""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from zen.gather import gather_compiled
from zen.reduce import gather_compiled as reduce_compiled

# ── Tuning knobs ──
num_splits = 4          # KV sequence splits
BM = 8                  # heads per block (16 heads / 2 blocks)

# ── Model constants ──
H = 16
D = 512
Dp = 64
TOPK = 2048
N_split = TOPK // num_splits  # 512 keys per split


def split_attention(
    qn,         # [T, H, D]        bf16   — q nope
    qp,         # [T, H, Dp]       bf16   — q pe
    Kc,         # [T, TOPK, D]     bf16   — gathered ckv
    Kp,         # [T, TOPK, Dp]    bf16   — gathered kpe
    mask,       # [T, TOPK]        bool   — True = invalid
    sm_scale,   # float
    partial_O,  # [T, num_splits, H, D]   fp32 — output per split
    partial_lse,# [T, num_splits, H]      fp32 — log-sum-exp per split
    split_idx,  # int — which split we are processing
    head_start, # int — starting head index for this head block
):
    """Process one (split, head_block) tile — mimics one CTA of kernel 1."""
    T = qn.shape[0]
    head_end = min(head_start + BM, H)
    k_start = split_idx * N_split
    k_end = k_start + N_split

    # Slice inputs for this tile
    qn_tile = qn[:, head_start:head_end, :]            # [T, BM, D]
    qp_tile = qp[:, head_start:head_end, :]            # [T, BM, Dp]
    Kc_tile = Kc[:, k_start:k_end, :]                  # [T, N_split, D]
    Kp_tile = Kp[:, k_start:k_end, :]                  # [T, N_split, Dp]
    mask_tile = mask[:, k_start:k_end]                  # [T, N_split]

    # GEMM1: logits = Q @ K^T  →  [T, BM, N_split]
    logits = (torch.bmm(qn_tile.float(), Kc_tile.float().transpose(1, 2)) +
              torch.bmm(qp_tile.float(), Kp_tile.float().transpose(1, 2)))

    # Mask invalid positions
    logits.masked_fill_(mask_tile.unsqueeze(1), float('-inf'))
    logits_scaled = logits * sm_scale

    # Softmax over this split's keys (no row-max subtraction)
    exp_logits = torch.exp(logits_scaled)                       # [T, BM, N_split]
    row_sum = exp_logits.sum(dim=-1)                            # [T, BM]

    # Handle all-masked splits: row_sum=0 → set attn to 0, lse to -inf
    valid = row_sum > 0                                         # [T, BM]
    row_sum = row_sum.clamp(min=1e-20)                          # avoid div-by-zero

    # Attention weights (within this split)
    attn = exp_logits / row_sum.unsqueeze(-1)                   # [T, BM, N_split]

    # GEMM2: output = attn @ V  →  [T, BM, D]
    out_tile = torch.bmm(attn, Kc_tile.float())                # [T, BM, D]

    # Zero out invalid rows to avoid NaN propagation in reduce
    out_tile[~valid] = 0.0

    # Store partial results
    partial_O[:, split_idx, head_start:head_end, :] = out_tile
    # LSE = log(row_sum)  (no row_max offset)
    split_lse = torch.log(row_sum)
    split_lse = torch.where(valid, split_lse, torch.tensor(float('-inf'), device=split_lse.device))
    partial_lse[:, split_idx, head_start:head_end] = split_lse


def reduce_block(
    partial_O,      # [T, num_splits, H, D]   fp32
    partial_lse,    # [T, num_splits, H]      fp32
    output,         # [T, H, D]               bf16 — final output
    lse,            # [T, H]                  fp32 — final lse
    t,              # int — token index (grid X)
    head_start,     # int — starting head index (grid Y)
):
    """Reduce one (token, head_block) tile — mimics one CTA of kernel 2.

    Sequentially folds splits using online log-sum-exp.
    """
    S = partial_O.shape[1]
    head_end = min(head_start + BM, H)

    # Gather all splits for this (token, head_block)
    pO = partial_O[t, :, head_start:head_end, :]        # [S, BM, D]
    pL = partial_lse[t, :, head_start:head_end]          # [S, BM]

    # Global LSE across splits
    global_lse = torch.logsumexp(pL, dim=0)              # [BM]

    # Scale factors per split
    scale = torch.exp(pL - global_lse.unsqueeze(0))      # [S, BM]

    # Weighted sum: [S, BM, D] * [S, BM, 1] → sum → [BM, D]
    combined = (pO * scale.unsqueeze(-1)).sum(dim=0)

    output[t, head_start:head_end, :] = combined.bfloat16()
    lse[t, head_start:head_end] = global_lse / math.log(2.0)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    T = q_nope.shape[0]
    num_pages, page_size, _ = ckv_cache.shape

    mask = sparse_indices == -1  # [T, TOPK]

    # ── Gather (reuse fast CuTe kernel) ──
    Kc_all = ckv_cache.reshape(-1, D)
    Kp_all = kpe_cache.reshape(-1, Dp)
    Kc = torch.empty(T, TOPK, D, dtype=torch.bfloat16, device="cuda")
    Kp = torch.empty(T, TOPK, Dp, dtype=torch.bfloat16, device="cuda")
    max_valid = torch.empty(T, dtype=torch.int32, device="cuda")
    gather_compiled(Kc_all, Kp_all, sparse_indices, Kc, Kp, max_valid)

    # ── Allocate partial buffers (gmem) ──
    partial_O = torch.empty(T, num_splits, H, D, dtype=torch.float32, device="cuda")
    partial_lse = torch.empty(T, num_splits, H, dtype=torch.float32, device="cuda")

    # ── Kernel 1: split attention (sequential over blocks) ──
    num_head_blocks = H // BM
    for s in range(num_splits):
        for hb in range(num_head_blocks):
            split_attention(
                q_nope, q_pe, Kc, Kp, mask, sm_scale,
                partial_O, partial_lse,
                split_idx=s, head_start=hb * BM,
            )

    # ── Kernel 2: reduce across splits (CuTeDSL kernel) ──
    reduce_compiled(partial_O, partial_lse, output, lse)

    # # ── Kernel 2 (reference): reduce across splits (sequential over grid) ──
    # for t in range(T):
    #     for hb in range(num_head_blocks):
    #         reduce_block(partial_O, partial_lse, output, lse,
    #                      t=t, head_start=hb * BM)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))
    import cook
    cook.impl_fn = run
    cook.main()

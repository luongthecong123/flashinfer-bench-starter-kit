#!/usr/bin/env python3
"""Generate Excalidraw diagram for idxer_tc.py — Batched Index Attention (torch.compile).

Vectorized (no Python loop): all B requests processed together.
  1. Dequant FP8 → K_all, reshape → K_flat [P*64, 128]
  2. Build token_indices [B, max_sl] from block_table (broadcast)
  3. Build mask [B, max_sl] (True = padding)
  4. Gather K_gathered [B, max_sl, 128]
  5. @torch.compile _score_and_reduce:
       bmm(q, K_gathered.T) → [B,64,max_sl], relu, einsum bhs,bh->bs,
       masked_fill → final [B, max_sl]
  6. topk → [B, actual_k]  →  remap to global page tokens
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from excalib import (Diagram, px,
                     C_Q, C_KV, C_SCR, C_ATTN, C_OUT, C_IDX,
                     C_DIM, C_TXT, C_OP, BG_Q, BG_KV, BG_SCR, BG_ATTN,
                     BG_OUT, BG_IDX, BG_MASK,
                     FS_DIM, FS_NAME, FS_TITLE, FS_NOTE, FS_BIG,
                     LGAP, SECTION_GAP, FILL_IN, FILL_OUT)

d = Diagram()

# ── Pixel sizes ──
D128    = px(128)   # 120 — index_head_dim
D64     = px(64)    # 80  — num_index_heads / page_size
DV      = px(289)   # 200 — mean valid seq_len (representative)
DP      = px(8462)  # 180 — num_pages
DTOPK   = 260       # topk=2048, compact
DMAXSL  = 240       # max_sl = max_num_pages * 64, compact
DB      = px(5)     # 60  — batch size ~5
DEPTH_B = 10        # 3D batch depth

LEFT = 100
y = 0

# ━━ TITLE ━━
d.text(LEFT, y, "idxer_tc.py — Batched Index Attention  (@torch.compile, no Python loop)", FS_BIG, C_TXT)
d.text(LEFT, y + 38,
       "Real workload:  B∈[1,8]  |  P=8462 pages  |  page_size=64"
       "  |  max_sl = max_num_pages × 64  |  topk=2048",
       FS_NOTE + 1, C_DIM)
d.text(LEFT, y + 58,
       "All B requests processed together — no per-request Python loop."
       "  3D boxes = batched tensors.  L-layout: A (left) × B (top) = C.",
       FS_NOTE + 1, C_DIM)
y += 90

# ━━ DIMENSION LEGEND ━━
d.text(LEFT, y, "Dimensions", FS_TITLE, C_TXT); y += 50
for line in [
    "B       = batch_size: number of concurrent decode requests (1–8)",
    "64      = num_index_heads: heads used for scoring",
    "128     = index_head_dim: dimension of each index head",
    "P       = num_pages = 8462: pages in shared paged KV cache",
    "64      = page_size: tokens per page",
    "max_sl  = max_num_pages × 64: padded sequence length (all requests aligned)",
    "V       ≈ 289: mean valid tokens — positions < V have real data, rest are masked",
    "2048    = topk: max sparse tokens selected per request",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 20
y += SECTION_GAP // 2

# ━━━━ INPUTS ━━━━
d.text(LEFT, y, "INPUTS", FS_TITLE, C_TXT); y += 54

# q_index_fp8 [B, 64, 128]
d.labeled_rect_3d(LEFT, y, D128, D64, DEPTH_B, "q_index_fp8", C_Q, BG_Q,
                  dim_top="128", dim_side="64", dim_depth="B",
                  shape="[B, 64, 128] fp8_e4m3fn", fill=FILL_IN)

# k_index_cache_fp8 [P, 64, 1, 132]
kc_x = LEFT + D128 + 180
d.labeled_rect_3d(kc_x, y, 90, DP, DEPTH_B, "k_index_cache_fp8", C_KV, BG_KV,
                  dim_top="132", dim_side="P=8462", dim_depth="64",
                  shape="[8462, 64, 1, 132] int8", fill=FILL_IN)

# weights [B, 64]
wt_x = kc_x + 90 + 200
mid_y = y + (DP - DB) // 2
d.labeled_rect(wt_x, mid_y, D64, DB, "weights", C_ATTN, BG_ATTN,
               dim_top="64", dim_side="B",
               shape="[B, 64] f32", fill=FILL_IN)

# block_table [B, max_num_pages]
bt_x = wt_x + D64 + 140
d.labeled_rect(bt_x, mid_y - 50, DTOPK // 2, DB, "block_table", C_IDX, BG_IDX,
               dim_top="max_pages", dim_side="B",
               shape="[B, max_pages] int32", fill=FILL_IN)

# seq_lens [B]
d.labeled_rect(bt_x, mid_y + DB + 30, 30, DB, "seq_lens", C_IDX, BG_IDX,
               dim_side="B", shape="[B] int32", fill=FILL_IN)

y += DP + SECTION_GAP

# ━━━━ ① DEQUANT + FLATTEN ━━━━
d.text(LEFT, y, "① Dequant FP8  +  Flatten  →  K_flat [P×64, 128] f32", FS_TITLE, C_TXT)
d.text(LEFT, y + 32,
       "Same fp8×scale dequant as idxer_ref, then reshape [P,64,128] → [P×64, 128]",
       FS_NOTE + 1, C_DIM)
y += 60

d.labeled_rect_3d(LEFT, y, 90, DP, DEPTH_B, "k_index_cache_fp8", C_KV, BG_KV,
                  dim_top="132", dim_side="P=8462", dim_depth="64", fill=FILL_IN)
ax1 = LEFT + 90 + DEPTH_B + 20; ax2 = ax1 + 90
d.transform_arrow(ax1, y + DP // 2, ax2, "fp8×scale", "+reshape")

kflat_x = ax2 + 12
kflat_h = px(541568)  # 160
d.labeled_rect(kflat_x, y, D128, kflat_h, "K_flat", C_KV, BG_KV,
               dim_top="128", dim_side="P×64=541568",
               shape="[541568, 128] f32", fill=FILL_OUT)

y += max(DP, kflat_h) + SECTION_GAP

# ━━━━ ② BUILD token_indices + mask ━━━━
d.text(LEFT, y, "② Build token_indices [B, max_sl]  and  mask [B, max_sl]", FS_TITLE, C_TXT)
d.text(LEFT, y + 32,
       "token_indices = block_table[:, :, None]*64 + arange(64)[None,None,:]  →  reshape [B, max_sl]",
       FS_NOTE + 1, C_DIM)
d.text(LEFT, y + 52,
       "mask = arange(max_sl)[None] >= seq_lens[:, None]   (True = padding)",
       FS_NOTE + 1, C_DIM)
y += 76

# block_table
d.labeled_rect(LEFT, y, DMAXSL // 2, DB, "block_table", C_IDX, BG_IDX,
               dim_top="max_pages", dim_side="B", fill=FILL_IN)
ax1 = LEFT + DMAXSL // 2 + 15; ax2 = ax1 + 90
d.transform_arrow(ax1, y + DB // 2, ax2, "*64+offset", "broadcast")
d.labeled_rect_3d(ax2 + 12, y, DMAXSL, DB, DEPTH_B, "token_indices", C_IDX, BG_IDX,
                  dim_top="max_sl", dim_side="B",
                  shape="[B, max_sl] int64", fill=FILL_OUT)
y += DB + 50

# mask
d.labeled_rect(LEFT, y, 30, DB, "seq_lens", C_IDX, BG_IDX,
               dim_side="B", fill=FILL_IN)
ax1 = LEFT + 30 + 15; ax2 = ax1 + 90
d.transform_arrow(ax1, y + DB // 2, ax2, ">=arange", "broadcast")
d.labeled_rect_3d(ax2 + 12, y, DMAXSL, DB, DEPTH_B, "mask", C_IDX, BG_MASK,
                  dim_top="max_sl", dim_side="B",
                  shape="[B, max_sl] bool", fill=FILL_OUT)

y += DB + SECTION_GAP

# ━━━━ ③ GATHER K_gathered ━━━━
d.text(LEFT, y, "③ Gather:  K_flat[token_indices] → K_gathered [B, max_sl, 128]", FS_TITLE, C_TXT)
d.text(LEFT, y + 32,
       "K_flat[token_indices.reshape(-1)].reshape(B, max_sl, 128)  —  one big indexed lookup",
       FS_NOTE + 1, C_DIM)
y += 56

d.labeled_rect(LEFT, y, D128, kflat_h, "K_flat", C_KV, BG_KV,
               dim_top="128", dim_side="P×64", fill=FILL_IN)
ax1 = LEFT + D128 + 20; ax2 = ax1 + 90
d.transform_arrow(ax1, y + kflat_h // 2, ax2, "[tok_idx]", "reshape")

kg_x = ax2 + 12
d.labeled_rect_3d(kg_x, y, D128, DMAXSL, DEPTH_B, "K_gathered", C_KV, BG_KV,
                  dim_top="128", dim_side="max_sl", dim_depth="B",
                  shape="[B, max_sl, 128] f32", fill=FILL_OUT)

y += max(kflat_h, DMAXSL) + SECTION_GAP

# ━━━━ ④ BATCHED BMM (inside @torch.compile) ━━━━
d.text(LEFT, y,
       "④  @torch.compile  _score_and_reduce(q, K_gathered, weights, mask)",
       FS_TITLE, C_TXT)
y += 35
d.text(LEFT, y, "Step A — bmm:  q [B,64,128] @ K_gathered.T [B,128,max_sl] → scores [B,64,max_sl]",
       FS_NOTE + 1, C_DIM)
y += 30

y_c = y + D128 + LGAP
bot, _, _, _, _ = d.bmm_L_3d(LEFT, y_c, DEPTH_B,
    "q", C_Q, BG_Q, D64, D128,
    "K_gathered.T", C_KV, BG_KV, D128, DMAXSL,
    "scores", C_SCR, BG_SCR,
    "64", "128", "max_sl", batch_dim="B",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT)
y = bot + SECTION_GAP

# ━━━━ ⑤ RELU + EINSUM ━━━━
d.text(LEFT, y,
       "⑤ relu  +  einsum 'bhs,bh->bs':  scores [B,64,max_sl] × w [B,64] → final [B,max_sl]",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "Weighted sum over heads: final[b,s] = Σ_h  relu(scores[b,h,s]) * weights[b,h]",
       FS_NOTE + 1, C_DIM)
y += 60

d.labeled_rect_3d(LEFT, y, DMAXSL, D64, DEPTH_B, "scores", C_SCR, BG_SCR,
                  dim_top="max_sl", dim_side="64", dim_depth="B", fill=FILL_IN)
ax1 = LEFT + DMAXSL + DEPTH_B + 20; ax2 = ax1 + 80
d.transform_arrow(ax1, y + D64 // 2, ax2, "relu(·)", "einsum×w")

d.labeled_rect_3d(ax2 + 12, y + D64 // 2 - DB // 2, DMAXSL, DB, DEPTH_B,
                  "final", C_SCR, BG_SCR,
                  dim_top="max_sl", dim_side="B",
                  shape="[B, max_sl] f32", fill=FILL_OUT)

y += D64 + SECTION_GAP

# ━━━━ ⑥ MASKED_FILL ━━━━
d.text(LEFT, y, "⑥ masked_fill:  final.masked_fill_(mask, −∞)", FS_TITLE, C_TXT)
y += 56

x = LEFT
d.labeled_rect_3d(x, y, DMAXSL, DB, DEPTH_B, "final", C_SCR, BG_SCR,
                  dim_top="max_sl", dim_side="B", fill=FILL_IN)
x += DMAXSL + DEPTH_B + 20
d.op_text(x, y + DB // 2 - 14, "masked_fill_(mask, −∞)  →")
x += 280
d.labeled_rect_3d(x, y, DMAXSL, DB, DEPTH_B, "final_masked", C_SCR, BG_SCR,
                  dim_top="max_sl", dim_side="B",
                  shape="[B, max_sl] f32  (padding = −∞)", fill=FILL_OUT)

y += DB + SECTION_GAP

# ━━━━ ⑦ TOP-K ━━━━
d.text(LEFT, y, "⑦ Top-K (batched):  torch.topk(final, k=2048, dim=1) → topk_idx [B, actual_k]",
       FS_TITLE, C_TXT)
y += 56

d.labeled_rect_3d(LEFT, y, DMAXSL, DB, DEPTH_B, "final_masked", C_SCR, BG_SCR,
                  dim_top="max_sl", dim_side="B", fill=FILL_IN)
ax1 = LEFT + DMAXSL + DEPTH_B + 20; ax2 = ax1 + 90
d.transform_arrow(ax1, y + DB // 2, ax2, "topk(2048,", "dim=1)")
d.labeled_rect_3d(ax2 + 12, y, DTOPK, DB, DEPTH_B, "topk_idx", C_IDX, BG_IDX,
                  dim_top="actual_k≤2048", dim_side="B",
                  shape="[B, actual_k] int64", fill=FILL_OUT)

y += DB + SECTION_GAP

# ━━━━ ⑧ REMAP ━━━━
d.text(LEFT, y, "⑧ Remap → global token index  (page * page_size + offset)",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "topk_page = topk_idx // 64  |  topk_off = topk_idx % 64"
       "  |  global_pages = gather(block_table, topk_page)  |  × 64 + offset",
       FS_NOTE + 1, C_DIM)
y += 60

d.labeled_rect_3d(LEFT, y, DTOPK, DB, DEPTH_B, "topk_idx", C_IDX, BG_IDX,
                  dim_top="actual_k", dim_side="B", fill=FILL_IN)
ax1 = LEFT + DTOPK + DEPTH_B + 20; ax2 = ax1 + 100
d.transform_arrow(ax1, y + DB // 2, ax2, "//64, %64,", "gather, ×64+off")
d.labeled_rect_3d(ax2 + 12, y, DTOPK, DB, DEPTH_B, "global_tokens", C_IDX, BG_IDX,
                  dim_top="actual_k", dim_side="B",
                  shape="[B, actual_k] int32", fill=FILL_OUT)

y += DB + SECTION_GAP

# mask out invalid (topk picked padding slot)
d.text(LEFT, y, "Mask invalid:  where topk picked a −∞ score slot → set −1", FS_NOTE + 1, C_DIM)
y += 30
d.labeled_rect_3d(LEFT, y, DTOPK, DB, DEPTH_B, "global_tokens (clipped to −1 if invalid)",
                  C_IDX, BG_IDX,
                  dim_top="actual_k", dim_side="B",
                  shape="[B, actual_k] int32  (−1 = padding)", fill=FILL_OUT)

y += DB + SECTION_GAP

# ━━━━ OUTPUT ━━━━
d.text(LEFT, y, "OUTPUT", FS_TITLE, C_TXT); y += 56
d.labeled_rect_3d(LEFT, y, DTOPK, DB, DEPTH_B, "topk_indices", C_OUT, BG_OUT,
                  dim_top="2048", dim_side="B",
                  shape="[B, 2048] int32  (−1 = padding)", fill=FILL_OUT)

out = os.path.join(os.path.dirname(__file__), "idxer_tc.excalidraw")
d.write(out)

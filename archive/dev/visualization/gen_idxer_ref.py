#!/usr/bin/env python3
"""Generate Excalidraw diagram for idxer_ref.py — Sparse Index Attention (per-batch loop).

Each decode request:
  1. Dequant FP8 KV cache → K_all [P, 64, 128]
  2. Per-batch: gather K [V, 128], score q_b @ K.T → [64, V],
     relu + weighted sum → [V], topk → remap to global page tokens.
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
D128   = px(128)   # 120 — index_head_dim
D64    = px(64)    # 80  — num_index_heads / page_size
DV     = px(289)   # 200 — mean seq_len (valid tokens per request)
DP     = px(8462)  # 180 — num_pages
DTOPK  = 260       # topk=2048 (compact)
DB     = px(5)     # 60  — batch size ~5
DEPTH_B = 10       # 3D batch depth

LEFT = 100
y = 0

# ━━ TITLE ━━
d.text(LEFT, y, "idxer_ref.py — Sparse Index Attention (per-batch loop)", FS_BIG, C_TXT)
d.text(LEFT, y + 38,
       "Real workload:  B∈[1,8]  |  P=8462 pages  |  page_size=64"
       "  |  V∈[1,2048] seq_len per request  |  topk=2048",
       FS_NOTE + 1, C_DIM)
d.text(LEFT, y + 58,
       "Per-batch loop: for b in range(B).  L-layout: A (left) × B (top) = C.",
       FS_NOTE + 1, C_DIM)
y += 90

# ━━ DIMENSION LEGEND ━━
d.text(LEFT, y, "Dimensions", FS_TITLE, C_TXT); y += 50
for line in [
    "B    = batch_size: number of concurrent decode requests (1–8)",
    "64   = num_index_heads: heads used for scoring",
    "128  = index_head_dim: dimension of each index head",
    "P    = num_pages = 8462: pages in the shared paged KV cache",
    "64   = page_size: tokens per page (= num_index_heads)",
    "V    = seq_len: valid tokens for this request (1–2048, mean~289)",
    "2048 = topk: max sparse tokens selected per request",
    "132  = 128 fp8 data bytes + 4 scale bytes (per-token scale, packed)",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 20
y += SECTION_GAP // 2

# ━━━━ INPUTS ━━━━
d.text(LEFT, y, "INPUTS", FS_TITLE, C_TXT); y += 54

# q_index_fp8 [B, 64, 128]
d.labeled_rect_3d(LEFT, y, D128, D64, DEPTH_B, "q_index_fp8", C_Q, BG_Q,
                  dim_top="128", dim_side="64", dim_depth="B",
                  shape="[B, 64, 128] fp8_e4m3fn", fill=FILL_IN)

# k_index_cache_fp8 [num_pages, 64, 1, 132]
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

# ━━━━ ① DEQUANT FP8 KV CACHE ━━━━
d.text(LEFT, y, "① Dequant FP8 KV Cache  →  K_all [P, 64, 128] f32", FS_TITLE, C_TXT)
d.text(LEFT, y + 32,
       "view(uint8) → split fp8 bytes [P,64,128] × scale [P,64,1] → f32",
       FS_NOTE + 1, C_DIM)
y += 60

d.labeled_rect_3d(LEFT, y, 90, DP, DEPTH_B, "k_index_cache_fp8", C_KV, BG_KV,
                  dim_top="132=128+4", dim_side="P=8462", dim_depth="64", fill=FILL_IN)
ax1 = LEFT + 90 + DEPTH_B + 20; ax2 = ax1 + 90
d.transform_arrow(ax1, y + DP // 2, ax2, "fp8×scale", "+ cast f32")

kall_x = ax2 + 12
d.labeled_rect_3d(kall_x, y, D128, DP, DEPTH_B, "K_all", C_KV, BG_KV,
                  dim_top="128", dim_side="P=8462", dim_depth="64",
                  shape="[8462, 64, 128] f32", fill=FILL_OUT)

y += DP + SECTION_GAP

# ━━━━ ② PER-BATCH LOOP ━━━━
d.text(LEFT, y, "② Per-batch loop  —  for b in range(B):", FS_TITLE, C_TXT)
y += 56

# page_indices
d.text(LEFT, y, "Gather page indices for request b:", FS_NOTE + 1, C_TXT); y += 28
d.labeled_rect(LEFT, y, DTOPK // 2, DB, "block_table[b]", C_IDX, BG_IDX,
               dim_top="max_pages", fill=FILL_IN)
ax1 = LEFT + DTOPK // 2 + 15; ax2 = ax1 + 80
d.transform_arrow(ax1, y + DB // 2, ax2, "[:P_b]", "int64")
d.labeled_rect(ax2 + 12, y, 120, DB, "page_indices", C_IDX, BG_IDX,
               dim_top="P_b", shape="[P_b] int64", fill=FILL_OUT)
y += DB + 60

# K gather
d.text(LEFT, y, "Gather K pages → reshape → slice to seq_len V:", FS_NOTE + 1, C_TXT); y += 28
d.labeled_rect_3d(LEFT, y, D128, DP, DEPTH_B, "K_all", C_KV, BG_KV,
                  dim_top="128", dim_side="P=8462", dim_depth="64", fill=FILL_IN)
ax1 = LEFT + D128 + DEPTH_B + 20; ax2 = ax1 + 90
d.transform_arrow(ax1, y + DP // 2, ax2, "[page_idx]", "reshape[:V]")

kv_top = y + (DP - DV) // 2
d.labeled_rect(ax2 + 12, kv_top, D128, DV, "K", C_KV, BG_KV,
               dim_side="V=seq_len", dim_top="128",
               shape="[V, 128] f32", fill=FILL_OUT)

# q_b
d.labeled_rect(LEFT, kv_top + DV + 40, D128, D64, "q_b = q[b]", C_Q, BG_Q,
               dim_top="128", dim_side="64",
               shape="[64, 128] f32", fill=FILL_IN)

y += DP + SECTION_GAP

# ━━━━ ③ SCORE ━━━━
d.text(LEFT, y, "③ Score:  q_b [64,128] @ K.T [128,V] → scores [64,V]", FS_TITLE, C_TXT)
y += 56

y_c = y + D128 + LGAP
bot, _, _, _, _ = d.matmul_L(LEFT, y_c,
    "q_b", C_Q, BG_Q, D64, D128,
    "K.T", C_KV, BG_KV, D128, DV,
    "scores", C_SCR, BG_SCR,
    "64", "128", "V",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT)
y = bot + SECTION_GAP

# ━━━━ ④ RELU + WEIGHTED SUM ━━━━
d.text(LEFT, y, "④ ReLU + weighted sum  →  final_scores [V]", FS_TITLE, C_TXT)
d.text(LEFT, y + 32,
       "relu(scores) × weights[b, :, None]  →  .sum(dim=0)",
       FS_NOTE + 1, C_DIM)
y += 60

d.labeled_rect(LEFT, y, DV, D64, "scores", C_SCR, BG_SCR,
               dim_top="V", dim_side="64", fill=FILL_IN)
ax1 = LEFT + DV + 15; ax2 = ax1 + 70
d.transform_arrow(ax1, y + D64 // 2, ax2, "relu(·)", "×w[:,None]")

relu_x = ax2 + 12
d.labeled_rect(relu_x, y, DV, D64, "relu×w", C_SCR, BG_SCR,
               dim_top="V", dim_side="64")

ax3 = relu_x + DV + 15; ax4 = ax3 + 70
d.transform_arrow(ax3, y + D64 // 2, ax4, ".sum(dim=0)", "")

fs_y = y + D64 // 2 - 12
d.labeled_rect(ax4 + 12, fs_y, DV, 24, "final_scores", C_SCR, BG_SCR,
               dim_top="V", shape="[V] f32", fill=FILL_OUT)

y += D64 + SECTION_GAP

# ━━━━ ⑤ TOP-K ━━━━
d.text(LEFT, y, "⑤ Top-K:  torch.topk(final_scores, k=2048) → topk_idx", FS_TITLE, C_TXT)
y += 56

d.labeled_rect(LEFT, y, DV, 24, "final_scores", C_SCR, BG_SCR,
               dim_top="V", fill=FILL_IN)
ax1 = LEFT + DV + 15; ax2 = ax1 + 90
d.transform_arrow(ax1, y + 12, ax2, "topk(2048)", "")
d.labeled_rect(ax2 + 12, y, DTOPK, 24, "topk_idx", C_IDX, BG_IDX,
               dim_top="actual_k ≤ 2048", shape="[actual_k] int64", fill=FILL_OUT)

y += 24 + SECTION_GAP

# ━━━━ ⑥ REMAP TO GLOBAL PAGE·OFFSET TOKEN INDEX ━━━━
d.text(LEFT, y, "⑥ Remap → global token index  (page * 64 + offset)", FS_TITLE, C_TXT)
y += 56

d.labeled_rect(LEFT, y, DTOPK, 24, "topk_idx", C_IDX, BG_IDX,
               dim_top="actual_k", fill=FILL_IN)
d.text(LEFT + DTOPK + 20, y + 4, "// 64  →", FS_NOTE + 1, C_IDX)

y += 36
d.labeled_rect(LEFT, y, DTOPK, 24, "page_idx_per_token", C_IDX, BG_IDX)
d.text(LEFT + DTOPK + 20, y + 4,
       "→  global_page = page_indices[page_idx_per_token]", FS_NOTE + 1, C_IDX)

y += 36
d.labeled_rect(LEFT, y, DTOPK, 24, "offset = topk_idx % 64", C_IDX, BG_IDX)

y += 36
d.labeled_rect(LEFT, y, DTOPK, 24, "global_tokens = global_page * 64 + offset", C_IDX, BG_IDX,
               dim_top="actual_k", shape="[actual_k] int32  →  written to topk_indices[b, :]",
               fill=FILL_OUT)

y += 24 + SECTION_GAP

# ━━━━ OUTPUT ━━━━
d.text(LEFT, y, "OUTPUT", FS_TITLE, C_TXT); y += 56
d.labeled_rect_3d(LEFT, y, DTOPK, DB, DEPTH_B, "topk_indices", C_OUT, BG_OUT,
                  dim_top="2048", dim_side="B",
                  shape="[B, 2048] int32  (−1 = padding)", fill=FILL_OUT)

out = os.path.join(os.path.dirname(__file__), "idxer_ref.excalidraw")
d.write(out)

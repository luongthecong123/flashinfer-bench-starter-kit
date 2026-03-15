#!/usr/bin/env python3
"""Generate Excalidraw diagram for impl.py — batched DSA Sparse Attention.

Key differences from ref.py:
  - No per-token loop: all T tokens processed as a batch
  - Padded to topk=2048 with masking (instead of filtering to V valid)
  - torch.compile fuses score + mask + scale + softmax + output
  - Uses bf16 matmuls (bmm with out_dtype=f32)

Uses 3D tensor boxes (parallelogram faces) for batched tensors (T depth).
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

D16   = px(16)     # 40
D64   = px(64)     # 80
D512  = px(512)    # 400
D2048 = px(2048)   # 550
DP    = px(8462)   # 180
DTOT  = px(541568) # 160
DT    = px(5)      # 60

DEPTH_T = 15   # T batch depth
DEPTH_P = 10   # page_size depth

# Extra y-offset before bmm_L_3d (accounts for B's 3D extrusion above C)
DY_T = DEPTH_T * 2  # = 30

C_MASK = "#f08c00"

LEFT = 100
y = 0

# ━━ TITLE ━━
d.text(LEFT, y, "impl.py — Batched DSA Sparse Attention", FS_BIG, C_TXT)
d.text(LEFT, y + 38, "Batched over T tokens | padded to 2048 + mask | torch.compile fused", FS_NOTE + 1, C_DIM)
d.text(LEFT, y + 58, "bf16 matmuls (bmm out_dtype=f32), no per-token loop, ~1.3 ms total", FS_NOTE + 1, C_DIM)
y += 90

# ━━ DIMENSION LEGEND ━━
d.text(LEFT, y, "Dimensions", FS_TITLE, C_TXT); y += 50
for line in [
    "16 = num_qo_heads: attention heads per GPU (MLA: shared KV, 16 query heads)",
    "512 = head_dim_ckv: compressed KV dimension (MLA: K and V fused into one vector)",
    "64 = head_dim_kpe: positional key encoding dim (= page_size)",
    "2048 = topk: max sparse-selected KV tokens per query (Stage 1 indexer output)",
    "T ≈ 5 = num_tokens: decode batch size (1–8, mean ≈ 5 concurrent requests)",
    "P = 8462: num_pages in shared paged KV cache pool (541,568 token slots total)",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 20
y += SECTION_GAP // 2

# ━━━━ INPUTS ━━━━
d.text(LEFT, y, "INPUTS", FS_TITLE, C_TXT); y += 54

# q_nope [T, 16, 512] bf16 — 3D
d.labeled_rect_3d(LEFT, y, D512, D16, DEPTH_T, "q_nope", C_Q, BG_Q,
                  dim_top="512", dim_side="16", dim_depth="T=5",
                  shape="[5, 16, 512] bf16", fill=FILL_IN)

qp_x = LEFT + D512 + 160
d.labeled_rect_3d(qp_x, y, D64, D16, DEPTH_T, "q_pe", C_Q, BG_Q,
                  dim_top="64", dim_side="16", dim_depth="T",
                  shape="[5, 16, 64] bf16", fill=FILL_IN)
y += D16 + 80

# caches — 3D with page_size depth
d.labeled_rect_3d(LEFT, y, D512, DP, DEPTH_P, "ckv_cache", C_KV, BG_KV,
                  dim_top="512", dim_side="P=8462", dim_depth="64",
                  shape="[8462, 64, 512] bf16", fill=FILL_IN)

kpe_x = LEFT + D512 + 140
d.labeled_rect_3d(kpe_x, y, D64, DP, DEPTH_P, "kpe_cache", C_KV, BG_KV,
                  dim_top="64", dim_side="P=8462", dim_depth="64",
                  shape="[8462, 64, 64] bf16", fill=FILL_IN)

si_x = kpe_x + D64 + 220
d.labeled_rect(si_x, y, D2048, DT, "sparse_indices", C_IDX, BG_IDX,
               dim_top="2048", dim_side="T=5",
               shape="[5, 2048] int32", fill=FILL_IN)
y += DP + SECTION_GAP

# ━━━━ ① FLATTEN (stays bf16) ━━━━
d.text(LEFT, y, "① Flatten KV Cache  ~13 µs  (reshape only, stays bf16)", FS_TITLE, C_TXT)
y += 56

d.rect_3d(LEFT, y, D512, DTOT, DEPTH_P, C_KV, BG_KV, fill=FILL_IN)
d.name_inside(LEFT, y, D512, DTOT, "ckv_cache", C_KV)
d.dim_above(LEFT, y, D512, "512"); d.dim_left(LEFT, y, DTOT, "P")
d.text(LEFT - len("64")*FS_DIM*0.6 - 4, y - DEPTH_P*2 - FS_DIM*0.6, "64", FS_DIM, C_DIM)

ax1 = LEFT + D512 + DEPTH_P + 20; ax2 = ax1 + 60
d.transform_arrow(ax1, y + DTOT/2, ax2, "reshape")

kca_x = ax2 + 15
d.labeled_rect(kca_x, y + 10, D512, DTOT - 20, "Kc_all", C_KV, BG_KV,
               dim_top="512", dim_side="541568", shape="[541568, 512] bf16")
y += DTOT + 40

d.rect_3d(LEFT, y, D64, DTOT, DEPTH_P, C_KV, BG_KV, fill=FILL_IN)
d.name_inside(LEFT, y, D64, DTOT, "kpe_cache", C_KV)
d.dim_above(LEFT, y, D64, "64"); d.dim_left(LEFT, y, DTOT, "P")
d.text(LEFT - len("64")*FS_DIM*0.6 - 4, y - DEPTH_P*2 - FS_DIM*0.6, "64", FS_DIM, C_DIM)

ax1 = LEFT + D64 + DEPTH_P + 20; ax2 = ax1 + 60
d.transform_arrow(ax1, y + DTOT/2, ax2, "reshape")

kpa_x = ax2 + 15
d.labeled_rect(kpa_x, y + 10, D64, DTOT - 20, "Kp_all", C_KV, BG_KV,
               dim_top="64", dim_side="541568", shape="[541568, 64] bf16")
y += DTOT + SECTION_GAP

# ━━━━ ② PREPARE INDICES ━━━━
d.text(LEFT, y, "② Prepare Indices  ~492 µs  (torch.compile — fused mask + clamp)", FS_TITLE, C_TXT)
y += 56

d.labeled_rect(LEFT, y, D2048, DT, "sparse_indices", C_IDX, BG_IDX,
               dim_top="2048", dim_side="T=5", fill=FILL_IN)

ax1 = LEFT + D2048 + 15; ax2 = ax1 + 100
d.arrow_h(ax1, y + DT/2 - 12, ax2, C_OP)
d.text(ax1 + 15, y + DT/2 - 28, "== -1", FS_NOTE, C_OP)
d.arrow_h(ax1, y + DT/2 + 12, ax2, C_OP)
d.text(ax1 + 10, y + DT/2 + 2, "clamp(0)", FS_NOTE, C_OP)

res_x = ax2 + 12
half_h = DT // 2 + 4
d.rect(res_x, y - 8, D2048, half_h, C_MASK, BG_MASK)
d.auto_name(res_x, y - 8, D2048, half_h, "mask", C_MASK)
d.shape_right(res_x, y - 8, D2048, half_h, "[5, 2048] bool — True=invalid")

d.rect(res_x, y + half_h - 4, D2048, half_h, C_IDX, BG_IDX)
d.auto_name(res_x, y + half_h - 4, D2048, half_h, "safe_indices", C_IDX)
d.shape_right(res_x, y + half_h - 4, D2048, half_h, "[5, 2048] int64 — clamped ≥0")
y += DT + SECTION_GAP

# ━━━━ ③ BATCHED GATHER ━━━━
d.text(LEFT, y, "③ Batched Gather  ~112 µs  (index flattened cache, all T at once)", FS_TITLE, C_TXT)
y += 56

# Kc_all → Kc [T, 2048, 512] — 3D result
d.labeled_rect(LEFT, y, D512, DTOT, "Kc_all", C_KV, BG_KV,
               dim_side="541568", dim_top="512")
ax1 = LEFT + D512 + 15; ax2 = ax1 + 80
d.transform_arrow(ax1, y + DTOT/2, ax2, "[safe_indices]", "gather+reshape")

kc_x = ax2 + 12
d.labeled_rect_3d(kc_x, y + 10, D512, D2048, DEPTH_T, "Kc", C_KV, BG_KV,
                  dim_top="512", dim_side="2048", dim_depth="T=5",
                  shape="[5, 2048, 512] bf16")
y += max(DTOT, D2048 + 10) + 50

# Kp_all → Kp [T, 2048, 64]
d.labeled_rect(LEFT, y, D64, DTOT, "Kp_all", C_KV, BG_KV,
               dim_side="541568", dim_top="64")
ax1 = LEFT + D64 + 15; ax2 = ax1 + 80
d.transform_arrow(ax1, y + DTOT/2, ax2, "[safe_indices]", "gather+reshape")

kp_x = ax2 + 12
d.labeled_rect_3d(kp_x, y + 10, D64, DTOT, DEPTH_T, "Kp", C_KV, BG_KV,
                  dim_top="64", dim_side="2048", dim_depth="T=5",
                  shape="[5, 2048, 64] bf16")
y += DTOT + 10 + SECTION_GAP

# ━━━━ ④ BATCHED ATTENTION (torch.compile fused) ━━━━
d.text(LEFT, y, "④ Batched Attention  ~563 µs  (torch.compile — single fused kernel)", FS_TITLE, C_TXT)
y += 10
d.text(LEFT, y + 26, "bmm(bf16 × bf16, out_dtype=f32)", FS_NOTE + 1, C_DIM)
y += 76

# ④a: Score bmm — 3D matmuls
d.text(LEFT, y, "④a Score:  bmm(qn, Kc.T) + bmm(qp, Kp.T)", FS_TITLE - 2, C_TXT); y += 52

y_c = y + D512 + 2 * LGAP + 2 * DY_T
bot, _, _, _, _ = d.bmm_L_3d(LEFT, y_c, DEPTH_T,
    "qn", C_Q, BG_Q, D16, D512,
    "Kc.T", C_KV, BG_KV, D512, D2048,
    "score_nope", C_SCR, BG_SCR,
    "16", "512", "2048", batch_dim="T=5",
    a_fill=FILL_IN, b_fill=FILL_IN)

# + score_pe
plus_x = LEFT + D512 + LGAP + DEPTH_T + D2048 + 30
d.op_text(plus_x, y_c + D16//2 - 14, "+")

pe_x = plus_x + 40
bot2, _, _, _, _ = d.bmm_L_3d(pe_x, y_c, DEPTH_T,
    "qp", C_Q, BG_Q, D16, D64,
    "Kp.T", C_KV, BG_KV, D64, D2048,
    "score_pe", C_SCR, BG_SCR,
    "16", "64", "2048",
    a_fill=FILL_IN, b_fill=FILL_IN)
y = max(bot, bot2) + 60

# ④b: Mask + scale
d.text(LEFT, y, "④b Mask + Scale:", FS_TITLE - 2, C_TXT); y += 52

x = LEFT
d.labeled_rect_3d(x, y, D2048, D16, DEPTH_T, "logits", C_SCR, BG_SCR,
                  dim_top="2048", dim_side="16", dim_depth="T=5")
x += D2048 + DEPTH_T + 20
d.text(x, y + D16//2 - 10, "mask_fill(-inf) × sm_scale →", FS_NOTE, C_OP)
x += 240
d.labeled_rect_3d(x, y, D2048, D16, DEPTH_T, "logits_scaled", C_SCR, BG_SCR,
                  dim_top="2048", dim_side="16", dim_depth="T=5",
                  shape="[5, 16, 2048] f32")
y += D16 + DEPTH_T * 2 + 60

# ④c: LSE + softmax
d.text(LEFT, y, "④c LSE + Softmax:", FS_TITLE - 2, C_TXT); y += 52

d.labeled_rect_3d(LEFT, y, D2048, D16, DEPTH_T, "logits_scaled", C_SCR, BG_SCR,
                  dim_top="2048", dim_side="16", dim_depth="T=5")
d.text(LEFT + D2048 + DEPTH_T + 15, y + D16//2 - 10, "→ logsumexp/ln2 →", 14, C_OP)
lse_x = LEFT + D2048 + DEPTH_T + 200
d.labeled_rect_3d(lse_x, y, 24, D16, DEPTH_T, "lse", C_OUT, BG_OUT,
                  dim_depth="T=5", shape="[5, 16] f32")
y += D16 + DEPTH_T * 2 + 30

d.labeled_rect_3d(LEFT, y, D2048, D16, DEPTH_T, "logits_scaled", C_SCR, BG_SCR,
                  dim_top="2048", dim_side="16", dim_depth="T=5")
d.text(LEFT + D2048 + DEPTH_T + 15, y + D16//2 - 10, "→ softmax →", 14, C_OP)
attn_x = LEFT + D2048 + DEPTH_T + 170
d.labeled_rect_3d(attn_x, y, D2048, D16, DEPTH_T, "attn", C_ATTN, BG_ATTN,
                  dim_top="2048", dim_side="16", dim_depth="T=5",
                  shape="[5, 16, 2048] f32")
y += D16 + DEPTH_T * 2 + SECTION_GAP

# ④d: Output bmm — 3D
d.text(LEFT, y, "④d Output:  bmm(attn, Kc.float()) → output", FS_TITLE - 2, C_TXT); y += 56

y_c = y + D2048 + 2 * LGAP + 2 * DY_T
bot, cx, cy, cw, ch = d.bmm_L_3d(LEFT, y_c, DEPTH_T,
    "attn", C_ATTN, BG_ATTN, D16, D2048,
    "Kc (f32)", C_KV, BG_KV, D2048, D512,
    "output", C_OUT, BG_OUT,
    "16", "2048", "512", batch_dim="T=5",
    a_fill=FILL_IN, b_fill=FILL_IN)
d.shape_right(cx, cy, cw, ch, "[5, 16, 512] → bf16")
y = bot + SECTION_GAP

# ━━━━ OUTPUTS ━━━━
d.text(LEFT, y, "OUTPUTS", FS_TITLE, C_TXT); y += 56
d.labeled_rect_3d(LEFT, y, D512, D16, DEPTH_T, "output", C_OUT, BG_OUT,
                  dim_top="512", dim_side="16", dim_depth="T=5",
                  shape="[5, 16, 512] bf16", fill=FILL_OUT)
lse_x = LEFT + D512 + 200
d.labeled_rect_3d(lse_x, y, D16, D16, DEPTH_T, "lse", C_OUT, BG_OUT,
                  dim_top="16", dim_depth="T=5",
                  shape="[5, 16] f32", fill=FILL_OUT)

out = os.path.join(os.path.dirname(__file__), "impl.excalidraw")
d.write(out)

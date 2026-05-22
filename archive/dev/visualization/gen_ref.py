#!/usr/bin/env python3
"""Generate Excalidraw diagram for ref.py — DSA Sparse Attention (per-token loop).

Uses 3D tensor boxes (parallelogram faces) for batched / paged tensors.
Real workload stats (23 workloads):
  T∈[1,8] mean=5  |  P=8462 always  |  V∈[1,2048] mean=289
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from excalib import (Diagram, px,
                     C_Q, C_KV, C_SCR, C_ATTN, C_OUT, C_IDX,
                     C_DIM, C_TXT, C_OP, BG_Q, BG_KV, BG_SCR, BG_ATTN,
                     BG_OUT, BG_IDX, FS_DIM, FS_NAME, FS_TITLE, FS_NOTE,
                     FS_BIG, LGAP, SECTION_GAP, FILL_IN, FILL_OUT)

d = Diagram()

# ── Pixel sizes (consistent across all diagrams via excalib.DIM) ──
D16   = px(16)     # 40
D64   = px(64)     # 80
DV    = px(289)    # 200
D512  = px(512)    # 400
D2048 = px(2048)   # 550
DP    = px(8462)   # 180
DTOT  = px(541568) # 160
DT    = px(5)      # 60

# 3D depth pixels
DEPTH_T = 15   # T (batch)
DEPTH_P = 10   # page_size=64

LEFT = 100
y = 0

# ━━ TITLE ━━
d.text(LEFT, y, "ref.py — DSA Sparse Attention (per-token loop)", FS_BIG, C_TXT)
d.text(LEFT, y + 38, "Real workload:  T∈[1,8] mean=5  |  P=8462  |  V∈[1,2048] mean=289", FS_NOTE + 1, C_DIM)
d.text(LEFT, y + 58, "3D boxes = batched/paged tensors.  L-layout: A (left) × B (top) = C.", FS_NOTE + 1, C_DIM)
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
    "V ≈ 289: valid (non-padding) tokens per query (1–2048, mean 289)",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 20
y += SECTION_GAP // 2

# ━━━━ INPUTS ━━━━
d.text(LEFT, y, "INPUTS", FS_TITLE, C_TXT); y += 54

# q_nope [T, 16, 512] bf16 — 3D with T depth
d.labeled_rect_3d(LEFT, y, D512, D16, DEPTH_T, "q_nope", C_Q, BG_Q,
                  dim_top="512", dim_side="16", dim_depth="T=5",
                  shape="[5, 16, 512] bf16", fill=FILL_IN)

# q_pe [T, 16, 64] bf16
qp_x = LEFT + D512 + 180
d.labeled_rect_3d(qp_x, y, D64, D16, DEPTH_T, "q_pe", C_Q, BG_Q,
                  dim_top="64", dim_side="16", dim_depth="T",
                  shape="[5, 16, 64] bf16", fill=FILL_IN)
y += D16 + 80

# ckv_cache [8462, 64, 512] bf16 — 3D with page_size depth
d.labeled_rect_3d(LEFT, y, D512, DP, DEPTH_P, "ckv_cache", C_KV, BG_KV,
                  dim_top="512", dim_side="P=8462", dim_depth="64",
                  shape="[8462, 64, 512] bf16", fill=FILL_IN)

# kpe_cache [8462, 64, 64] bf16
kpe_x = LEFT + D512 + 140
d.labeled_rect_3d(kpe_x, y, D64, DP, DEPTH_P, "kpe_cache", C_KV, BG_KV,
                  dim_top="64", dim_side="P=8462", dim_depth="64",
                  shape="[8462, 64, 64] bf16", fill=FILL_IN)

# sparse_indices [T, 2048] int32
si_x = kpe_x + D64 + 220
d.labeled_rect(si_x, y, D2048, DT, "sparse_indices", C_IDX, BG_IDX,
               dim_top="2048 (topk)", dim_side="T=5",
               shape="[5, 2048] int32 (−1 = pad)", fill=FILL_IN)
y += DP + SECTION_GAP

# ━━━━ ① FLATTEN KV CACHE ━━━━
d.text(LEFT, y, "① Flatten KV Cache  ~67 µs  (reshape [P,64,D] → [P×64,D] + cast f32)", FS_TITLE, C_TXT)
y += 56

# ckv_cache 3D → Kc_all 2D
d.rect_3d(LEFT, y, D512, DTOT, DEPTH_P, C_KV, BG_KV, fill=FILL_IN)
d.name_inside(LEFT, y, D512, DTOT, "ckv_cache", C_KV)
d.dim_above(LEFT, y, D512, "512"); d.dim_left(LEFT, y, DTOT, "8462")
d.text(LEFT - len("64")*FS_DIM*0.6 - 4, y - DEPTH_P*2 - FS_DIM*0.6, "64", FS_DIM, C_DIM)

ax1 = LEFT + D512 + DEPTH_P + 20; ax2 = ax1 + 70
d.transform_arrow(ax1, y + DTOT/2, ax2, "reshape", "+ f32")

kca_x = ax2 + 15
d.labeled_rect(kca_x, y + 10, D512, DTOT - 20, "Kc_all", C_KV, BG_KV,
               dim_top="512", dim_side="541568", shape="[541568, 512] f32")
y += DTOT + 40

# kpe_cache → Kp_all
d.rect_3d(LEFT, y, D64, DTOT, DEPTH_P, C_KV, BG_KV, fill=FILL_IN)
d.name_inside(LEFT, y, D64, DTOT, "kpe_cache", C_KV)
d.dim_above(LEFT, y, D64, "64"); d.dim_left(LEFT, y, DTOT, "8462")
d.text(LEFT - len("64")*FS_DIM*0.6 - 4, y - DEPTH_P*2 - FS_DIM*0.6, "64", FS_DIM, C_DIM)

ax1 = LEFT + D64 + DEPTH_P + 20; ax2 = ax1 + 70
d.transform_arrow(ax1, y + DTOT/2, ax2, "reshape", "+ f32")

kpa_x = ax2 + 15
d.labeled_rect(kpa_x, y + 10, D64, DTOT - 20, "Kp_all", C_KV, BG_KV,
               dim_top="64", dim_side="541568", shape="[541568, 64] f32")
y += DTOT + SECTION_GAP

# ━━━━ ② FILTER & GATHER ━━━━
d.text(LEFT, y, "② Filter & Gather  (per token t, inside loop over T)", FS_TITLE, C_TXT)
y += 54

d.labeled_rect(LEFT, y, D2048, D16, "sparse_indices[t]", C_IDX, BG_IDX,
               dim_top="2048", fill=FILL_IN)
ax1 = LEFT + D2048 + 15; ax2 = ax1 + 80
d.transform_arrow(ax1, y + D16/2, ax2, "!= -1", "filter")
vi_x = ax2 + 12
d.labeled_rect(vi_x, y, DV, D16, "valid_indices", C_IDX, BG_IDX, dim_top="V=289 (mean)")
y += D16 + 50

d.text(LEFT, y, "Gather from flattened cache:", FS_NOTE + 1, C_TXT); y += 24
d.labeled_rect(LEFT, y, D512, DTOT, "Kc_all", C_KV, BG_KV, dim_side="541568", dim_top="512")
ax1 = LEFT + D512 + 15; ax2 = ax1 + 70
d.transform_arrow(ax1, y + DTOT/2, ax2, "[tok_idx]", "gather")
kc_y = y + (DTOT - DV)//2
d.labeled_rect(ax2 + 12, kc_y, D512, DV, "Kc", C_KV, BG_KV,
               dim_side="V=289", dim_top="512", shape="[289, 512] f32", fill=FILL_OUT)
y += DTOT + 30

d.labeled_rect(LEFT, y, D64, DTOT, "Kp_all", C_KV, BG_KV, dim_side="541568", dim_top="64")
ax1 = LEFT + D64 + 15; ax2 = ax1 + 70
d.transform_arrow(ax1, y + DTOT/2, ax2, "[tok_idx]", "gather")
kp_y = y + (DTOT - DV)//2
d.labeled_rect(ax2 + 12, kp_y, D64, DV, "Kp", C_KV, BG_KV,
               dim_side="V=289", dim_top="64", shape="[289, 64] f32", fill=FILL_OUT)
y += DTOT + 30

d.text(LEFT, y, "Slice query for token t (cast bf16 → f32):", FS_NOTE + 1, C_TXT); y += 24
d.labeled_rect(LEFT, y, D512, D16, "qn = q_nope[t]", C_Q, BG_Q,
               dim_top="512", dim_side="16", shape="[16, 512] f32", fill=FILL_IN)
qp2_x = LEFT + D512 + 180
d.labeled_rect(qp2_x, y, D64, D16, "qp = q_pe[t]", C_Q, BG_Q,
               dim_top="64", dim_side="16", shape="[16, 64] f32", fill=FILL_IN)
y += D16 + 50 + SECTION_GAP

# ━━━━ ③ SCORE (NOPE) ━━━━
d.text(LEFT, y, "③ Score (nope)  ~282 µs (③+④):  qn[16,512] × Kc.T[512,V] → score_nope[16,V]", FS_TITLE, C_TXT)
y += 56
y_c = y + D512 + LGAP
bot, _, _, _, _ = d.matmul_L(LEFT, y_c,
    "qn", C_Q, BG_Q, D16, D512,
    "Kc.T", C_KV, BG_KV, D512, DV,
    "score_nope", C_SCR, BG_SCR,
    "16", "512", "V=289",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT)
y = bot + SECTION_GAP

# ━━━━ ④ SCORE (PE) ━━━━
d.text(LEFT, y, "④ Score (PE):  qp[16,64] × Kp.T[64,V] → score_pe[16,V]", FS_TITLE, C_TXT)
y += 56
y_c = y + D64 + LGAP
bot, _, _, _, _ = d.matmul_L(LEFT, y_c,
    "qp", C_Q, BG_Q, D16, D64,
    "Kp.T", C_KV, BG_KV, D64, DV,
    "score_pe", C_SCR, BG_SCR,
    "16", "64", "V=289",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT)
y = bot + SECTION_GAP

# ━━━━ ⑤ ADD + SCALE ━━━━
d.text(LEFT, y, "⑤ Add + scale:  (score_nope + score_pe) × sm_scale", FS_TITLE, C_TXT); y += 60
x = LEFT
d.rect(x, y, DV, D16, C_SCR, BG_SCR); d.name_below(x, y, DV, D16, "score_nope", C_SCR)
x += DV; d.op_text(x + 10, y + D16//2 - 14, "+")
x += 40
d.rect(x, y, DV, D16, C_SCR, BG_SCR); d.name_below(x, y, DV, D16, "score_pe", C_SCR)
x += DV; d.op_text(x + 10, y + D16//2 - 14, "× s =", 20)
x += 80
d.rect(x, y, DV, D16, C_SCR, BG_SCR, FILL_OUT); d.name_below(x, y, DV, D16, "logits_scaled", C_SCR)
d.dim_above(x, y, DV, "V=289"); d.dim_left(x, y, D16, "16")
d.shape_right(x, y, DV, D16, "[16, 289] f32")
y += D16 + SECTION_GAP

# ━━━━ ⑥ LSE ━━━━
d.text(LEFT, y, "⑥ LSE  ~336 µs:  logsumexp(logits_scaled, dim=-1) / ln2", FS_TITLE, C_TXT)
y += 60
d.rect(LEFT, y, DV, D16, C_SCR, BG_SCR)
d.name_below(LEFT, y, DV, D16, "logits_scaled", C_SCR)
d.dim_above(LEFT, y, DV, "V=289"); d.dim_left(LEFT, y, D16, "16")
d.text(LEFT + DV + 15, y + D16//2 - 10, "→ logsumexp / ln2 →", 15, C_OP)
lse_x = LEFT + DV + 240
d.rect(lse_x, y, 24, D16, C_OUT, BG_OUT, FILL_OUT)
d.name_right(lse_x, y, 24, D16, "lse[t]  [16] f32", C_OUT)
y += D16 + SECTION_GAP

# ━━━━ ⑦ SOFTMAX ━━━━
d.text(LEFT, y, "⑦ Softmax  ~150 µs (⑦+⑧) → attn", FS_TITLE, C_TXT)
y += 60
d.rect(LEFT, y, DV, D16, C_SCR, BG_SCR)
d.name_below(LEFT, y, DV, D16, "logits_scaled", C_SCR)
d.text(LEFT + DV + 15, y + D16//2 - 10, "→ softmax →", 15, C_OP)
attn_x = LEFT + DV + 170
d.labeled_rect(attn_x, y, DV, D16, "attn", C_ATTN, BG_ATTN,
               dim_top="V=289", dim_side="16", shape="[16, 289] f32", fill=FILL_OUT)
y += D16 + SECTION_GAP + 30

# ━━━━ ⑧ OUTPUT ━━━━
d.text(LEFT, y, "⑧ Output:  attn[16,V] × Kc[V,512] → output[t]", FS_TITLE, C_TXT); y += 56
y_c = y + DV + LGAP
bot, cx, cy, cw, ch = d.matmul_L(LEFT, y_c,
    "attn", C_ATTN, BG_ATTN, D16, DV,
    "Kc", C_KV, BG_KV, DV, D512,
    "output[t]", C_OUT, BG_OUT,
    "16", "V=289", "512",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT)
d.shape_right(cx, cy, cw, ch, "[16, 512] → bf16")
y = bot + SECTION_GAP

# ━━━━ OUTPUTS ━━━━
d.text(LEFT, y, "OUTPUTS (stacked over T tokens)", FS_TITLE, C_TXT); y += 56
d.labeled_rect_3d(LEFT, y, D512, D16, DEPTH_T, "output", C_OUT, BG_OUT,
                  dim_top="512", dim_side="16", dim_depth="T=5",
                  shape="[5, 16, 512] bf16", fill=FILL_OUT)
lse_x = LEFT + D512 + 200
d.labeled_rect_3d(lse_x, y, D16, D16, DEPTH_T, "lse", C_OUT, BG_OUT,
                  dim_top="16", dim_depth="T=5",
                  shape="[5, 16] f32", fill=FILL_OUT)

out = os.path.join(os.path.dirname(__file__), "ref.excalidraw")
d.write(out)

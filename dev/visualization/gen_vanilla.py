#!/usr/bin/env python3
"""Generate vanilla_attention.excalidraw — Multi-Head Self-Attention.

Two cases visualized with 3D tensor boxes (H=heads as depth dimension):

  Case 1 — Naive: Full S×S attention, no KV cache
  Case 2 — KV Cache Inference:
    a) Prefill: process full prompt, build KV cache
    b) Decode: single new token attends to full KV cache

Dimensions: B=1 (omitted), H=16, D=64, D_model=1024, S=1024
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

# ── Pixel sizes (shared DIM map) ──
DS = px(1024)    # S = 1024 → 480
DD = px(64)      # D = 64  → 80
DM = px(1024)    # D_model → 480
D1 = px(1)       # 1 token → 8

DEPTH_H = 12     # H = 16 heads — 3D depth pixels
DY_H = DEPTH_H * 2  # = 24, vertical offset for 3D

LEFT = 100
y = 0

# ══════════════════════════════════════════════════════════════════
#  CASE 1: NAIVE SELF-ATTENTION (no KV cache)
# ══════════════════════════════════════════════════════════════════

d.text(LEFT, y, "Case 1 — Naive Self-Attention (no KV cache)", FS_BIG, C_TXT)
d.text(LEFT, y + 38,
       "B=1 (omitted)  ·  H=16  ·  D=64  ·  D_model=1024  ·  S=1024",
       FS_NOTE + 1, C_DIM)
d.text(LEFT, y + 58,
       "Full S×S causal attention  ·  O(S² · H · D) compute",
       FS_NOTE + 1, C_DIM)
y += 90

# ── Dimension Legend ──
d.text(LEFT, y, "Dimensions", FS_TITLE, C_TXT); y += 50
for line in [
    "H = 16: number of attention heads",
    "D = 64: head dimension (per head)",
    "D_model = H × D = 1024: model embedding dimension",
    "S = 1024: sequence length / KV cache length",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 20
y += SECTION_GAP // 2

# ── Input ──
d.text(LEFT, y, "Input", FS_TITLE, C_TXT); y += 54
d.labeled_rect(LEFT, y, DM, DS, "X", C_TXT, "#f1f3f5",
               dim_top="D_model=1024", dim_side="S=1024",
               shape="[B, S, D_model]", fill=FILL_IN)
y += DS + SECTION_GAP

# ── ① Linear Projections + Reshape ──
d.text(LEFT, y, "① Linear Projections + Reshape to heads", FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "X @ W → [S, H, D] → transpose → [H, S, D]   (Q, K, V separately)",
       FS_NOTE + 1, C_DIM)
y += 76

proj_gap = 180

# Q [H, S, D]
d.labeled_rect_3d(LEFT, y, DD, DS, DEPTH_H, "Q", C_Q, BG_Q,
                  dim_top="D=64", dim_side="S=1024", dim_depth="H=16",
                  shape="[H, S, D]", fill=FILL_IN)

# K [H, S, D]
kx = LEFT + DD + proj_gap
d.labeled_rect_3d(kx, y, DD, DS, DEPTH_H, "K", C_KV, BG_KV,
                  dim_top="D=64", dim_side="S", dim_depth="H",
                  shape="[H, S, D]", fill=FILL_IN)

# V [H, S, D]
vx = kx + DD + proj_gap
d.labeled_rect_3d(vx, y, DD, DS, DEPTH_H, "V", C_KV, BG_KV,
                  dim_top="D=64", dim_side="S", dim_depth="H",
                  shape="[H, S, D]", fill=FILL_IN)
y += DS + SECTION_GAP

# ── ② Score: Q × K.T → scores [H, S, S] ──
d.text(LEFT, y, "② Score:  Q × K.T / √D  →  scores [H, S, S]", FS_TITLE, C_TXT)
y += 56

y_c = y + DD + 2 * LGAP + 2 * DY_H
bot, cx, cy, cw, ch = d.bmm_L_3d(LEFT, y_c, DEPTH_H,
    "Q", C_Q, BG_Q, DS, DD,         # A: rows=S, cols=D
    "K.T", C_KV, BG_KV, DD, DS,     # B: rows=D, cols=S
    "scores", C_SCR, BG_SCR,
    "S=1024", "D=64", "S=1024", batch_dim="H=16",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT)
d.shape_right(cx, cy, cw, ch, "[H, S, S] f32")
y = bot + 30

# Mask + softmax — inline note
d.text(LEFT, y,
       "③ Causal mask (upper triangle → -∞)  +  softmax  →  attn [H, S, S]",
       FS_TITLE - 2, C_TXT)
y += SECTION_GAP

# ── ④ Output: attn × V → out ──
d.text(LEFT, y, "④ Output:  attn × V  →  out [H, S, D]", FS_TITLE, C_TXT)
y += 56

y_c = y + DS + 2 * LGAP + 2 * DY_H
bot, cx, cy, cw, ch = d.bmm_L_3d(LEFT, y_c, DEPTH_H,
    "attn", C_ATTN, BG_ATTN, DS, DS,   # A: rows=S, cols=S
    "V", C_KV, BG_KV, DS, DD,          # B: rows=S, cols=D
    "out", C_OUT, BG_OUT,
    "S=1024", "S=1024", "D=64", batch_dim="H=16",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT)
d.shape_right(cx, cy, cw, ch, "[H, S, D]")
y = bot + SECTION_GAP

# ── ⑤ Output Projection ──
d.text(LEFT, y, "⑤ Reshape + Wo  →  output [S, D_model]", FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "concat heads: [H,S,D] → [S, H·D] = [S, D_model]  →  @ Wo",
       FS_NOTE + 1, C_DIM)
y += 76
d.labeled_rect(LEFT, y, DM, DS, "output", C_OUT, BG_OUT,
               dim_top="D_model=1024", dim_side="S=1024",
               shape="[B, S, D_model]", fill=FILL_OUT)
y += DS + SECTION_GAP * 3

# ══════════════════════════════════════════════════════════════════
#  CASE 2: KV CACHE INFERENCE
# ══════════════════════════════════════════════════════════════════

d.text(LEFT, y, "Case 2 — KV Cache Inference", FS_BIG, C_TXT)
d.text(LEFT, y + 38,
       "Cache K, V from prefill;  decode one token at a time with O(S) per step",
       FS_NOTE + 1, C_DIM)
y += 80

# ━━ 2a: Prefill ━━
d.text(LEFT, y, "2a — Prefill Stage", FS_TITLE + 4, C_TXT)
d.text(LEFT, y + 36,
       "Same attention as Case 1 (naive).  Additionally, store K and V into KV cache:",
       FS_NOTE + 1, C_DIM)
y += 86

# K → K_cache and V → V_cache  (side by side)
d.labeled_rect_3d(LEFT, y, DD, DS, DEPTH_H, "K", C_KV, BG_KV,
                  dim_top="D", dim_side="S", dim_depth="H", fill=FILL_IN)
ax1 = LEFT + DD + DEPTH_H + 15
ax2 = ax1 + 70
d.transform_arrow(ax1, y + DS // 2, ax2, "store")
kc_x = ax2 + 12
d.labeled_rect_3d(kc_x, y, DD, DS, DEPTH_H, "K_cache", C_KV, BG_KV,
                  dim_top="D", dim_side="S", dim_depth="H",
                  shape="[H, S, D]", fill=FILL_OUT)

v_off = kc_x + DD + DEPTH_H + 120
d.labeled_rect_3d(v_off, y, DD, DS, DEPTH_H, "V", C_KV, BG_KV,
                  dim_top="D", dim_side="S", dim_depth="H", fill=FILL_IN)
ax1v = v_off + DD + DEPTH_H + 15
ax2v = ax1v + 70
d.transform_arrow(ax1v, y + DS // 2, ax2v, "store")
vc_x = ax2v + 12
d.labeled_rect_3d(vc_x, y, DD, DS, DEPTH_H, "V_cache", C_KV, BG_KV,
                  dim_top="D", dim_side="S", dim_depth="H",
                  shape="[H, S, D]", fill=FILL_OUT)

y += DS + SECTION_GAP * 2

# ━━ 2b: Decode ━━
d.text(LEFT, y, "2b — Decode Stage (1 new token)", FS_TITLE + 4, C_TXT)
d.text(LEFT, y + 36,
       "Process ONE token  ·  O(S·H·D) per step — linear, not quadratic!",
       FS_NOTE + 1, C_DIM)
y += 90

# ── Decode Input ──
d.text(LEFT, y, "Input", FS_TITLE, C_TXT); y += 50
d.labeled_rect(LEFT, y, DM, D1, "x (new token)", C_TXT, "#f1f3f5",
               dim_top="D_model=1024",
               shape="[B, 1, D_model]", fill=FILL_IN)
d.dim_left(LEFT, y, D1, "1")
y += D1 + 60

# ── ⑥ Projections ──
d.text(LEFT, y, "⑥ Projections (single token)", FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "x @ Wq → q [H, 1, D]   (same for k, v)",
       FS_NOTE + 1, C_DIM)
y += 76

q_gap = 140
d.labeled_rect_3d(LEFT, y, DD, D1, DEPTH_H, "q", C_Q, BG_Q,
                  dim_top="D=64", dim_depth="H",
                  shape="[H, 1, D]", fill=FILL_IN)
d.dim_left(LEFT, y, D1, "1")

kx2 = LEFT + DD + q_gap
d.labeled_rect_3d(kx2, y, DD, D1, DEPTH_H, "k", C_KV, BG_KV,
                  dim_top="D", dim_depth="H",
                  shape="[H, 1, D]", fill=FILL_IN)
d.dim_left(kx2, y, D1, "1")

vx2 = kx2 + DD + q_gap
d.labeled_rect_3d(vx2, y, DD, D1, DEPTH_H, "v", C_KV, BG_KV,
                  dim_top="D", dim_depth="H",
                  shape="[H, 1, D]", fill=FILL_IN)
d.dim_left(vx2, y, D1, "1")
y += D1 + DEPTH_H * 2 + 60

# ── ⑦ Append to KV Cache ──
d.text(LEFT, y, "⑦ Append to KV Cache", FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "K_cache [H,S,D]  +  k [H,1,D]  →  K_cache [H, S+1, D]   (same for V)",
       FS_NOTE + 1, C_DIM)
y += 76

d.labeled_rect_3d(LEFT, y, DD, DS, DEPTH_H, "K_cache", C_KV, BG_KV,
                  dim_top="D", dim_side="S+1", dim_depth="H",
                  shape="[H, S+1, D]  (≈ 1025)", fill=FILL_IN)

vc_off = LEFT + DD + DEPTH_H + 160
d.labeled_rect_3d(vc_off, y, DD, DS, DEPTH_H, "V_cache", C_KV, BG_KV,
                  dim_top="D", dim_side="S+1", dim_depth="H",
                  shape="[H, S+1, D]", fill=FILL_IN)
y += DS + SECTION_GAP

# ── ⑧ Score: q × K_cache.T ──
d.text(LEFT, y, "⑧ Score:  q × K_cache.T / √D  →  scores [H, 1, S+1]", FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "Compare: 1×(S+1) vs S×S in naive — 1024× fewer elements!",
       FS_NOTE + 1, C_OP)
y += 76

y_c = y + DD + 2 * LGAP + 2 * DY_H
bot, cx, cy, cw, ch = d.bmm_L_3d(LEFT, y_c, DEPTH_H,
    "q", C_Q, BG_Q, D1, DD,               # A: rows=1, cols=D
    "K_cache.T", C_KV, BG_KV, DD, DS,     # B: rows=D, cols=S+1
    "scores", C_SCR, BG_SCR,
    "1", "D=64", "S+1", batch_dim="H=16",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT)
d.shape_right(cx, cy, cw, ch, "[H, 1, S+1]  ← just 1 row!")
y = bot + 30

# Softmax note
d.text(LEFT, y,
       "⑨ Softmax → attn [H, 1, S+1]  (no causal mask — single query token)",
       FS_TITLE - 2, C_TXT)
y += SECTION_GAP

# ── ⑩ Output: attn × V_cache → out ──
d.text(LEFT, y, "⑩ Output:  attn × V_cache  →  out [H, 1, D]", FS_TITLE, C_TXT)
y += 56

y_c = y + DS + 2 * LGAP + 2 * DY_H
bot, cx, cy, cw, ch = d.bmm_L_3d(LEFT, y_c, DEPTH_H,
    "attn", C_ATTN, BG_ATTN, D1, DS,       # A: rows=1, cols=S+1
    "V_cache", C_KV, BG_KV, DS, DD,         # B: rows=S+1, cols=D
    "out", C_OUT, BG_OUT,
    "1", "S+1", "D=64", batch_dim="H=16",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT)
d.shape_right(cx, cy, cw, ch, "[H, 1, D]  ← tiny output vector!")
y = bot + SECTION_GAP

# ── ⑪ Output ──
d.text(LEFT, y, "⑪ Reshape + @ Wo  →  output [1, D_model]", FS_TITLE, C_TXT)
y += 56
d.labeled_rect(LEFT, y, DM, D1, "output", C_OUT, BG_OUT,
               dim_top="D_model=1024",
               shape="[B, 1, D_model]", fill=FILL_OUT)
d.dim_left(LEFT, y, D1, "1")

# ── Write ──
out = os.path.join(os.path.dirname(__file__), "vanilla_attention.excalidraw")
d.write(out)

#!/usr/bin/env python3
"""Generate Excalidraw diagram for Dense SwiGLU FFN.

Architecture:
  gate = SiLU(x @ w_gate.T)   [S, INNER]
  up   = x @ w_up.T            [S, INNER]
  h    = gate * up              [S, INNER]  (element-wise)
  out  = h @ w_down.T           [S, M]

Dimensions:
  S     = 4096  (sequence length)
  M     = 2048  (hidden / model dim)
  INNER = 4096  (FFN intermediate dim)

L-layout: A (left) x B (top) = C (intersection)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from excalib import (Diagram,
                     C_Q, C_KV, C_SCR, C_ATTN, C_OUT, C_IDX,
                     C_DIM, C_TXT, C_OP, BG_Q, BG_KV, BG_SCR, BG_ATTN,
                     BG_OUT, BG_IDX,
                     FS_DIM, FS_NAME, FS_TITLE, FS_NOTE, FS_BIG,
                     LGAP, SECTION_GAP, FILL_IN, FILL_OUT)

d = Diagram()

# ── Custom pixel sizes for these dims ──
DS  = 180   # S = 4096  (rows of x / output rows)
DM  = 120   # M = 2048  (model dim / weight cols)
DI  = 180   # INNER = 4096  (FFN inner dim)

# Color scheme
C_X    = C_Q        # input x — blue
BG_X   = BG_Q
C_W    = C_KV       # weight matrices — green
BG_W   = BG_KV
C_ACT  = C_SCR      # activations / intermediate — purple
BG_ACT = BG_SCR
C_MUL  = C_ATTN     # hadamard product — orange
BG_MUL = BG_ATTN
C_O    = C_OUT      # output — teal
BG_O   = BG_OUT

LEFT = 120
y    = 0

# ━━ TITLE ━━
d.text(LEFT, y,
       "Dense SwiGLU FFN  ·  S=4096  M=2048  INNER=4096",
       FS_BIG, C_TXT)
d.text(LEFT, y + 38,
       "gate = SiLU(x @ w_gate.T)  ·  up = x @ w_up.T  "
       "·  h = gate ⊙ up  ·  out = h @ w_down.T",
       FS_NOTE + 1, C_DIM)
d.text(LEFT, y + 58,
       "L-layout: A (left) × B (top) = C (intersection).  "
       "Param budget: 3 × M × INNER  =  3 × 2048 × 4096  ≈ 25 B params",
       FS_NOTE + 1, C_DIM)
y += 100

# ━━ DIMENSION LEGEND ━━
d.text(LEFT, y, "Dimensions", FS_TITLE, C_TXT); y += 46
for line in [
    "S = 4096 : sequence length (batch of tokens)",
    "M = 2048 : model / hidden dimension",
    "INNER = 4096 : FFN intermediate dimension  (= 2 × M  for SwiGLU, same param count as 4 × M ReLU FFN)",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 20
y += SECTION_GAP // 2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INPUT x
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "Input", FS_TITLE, C_TXT); y += 46
d.labeled_rect(LEFT, y, DM, DS, "x", C_X, BG_X,
               dim_top="M=2048", dim_side="S=4096",
               shape="[4096, 2048] bf16", fill=FILL_IN)
y += DS + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ① GATE PATH:  x @ w_gate.T → SiLU → gate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "① Gate path:  x @ w_gate.T  →  SiLU", FS_TITLE, C_TXT); y += 46

bot_y, cx, cy, cw, ch = d.matmul_L(
    LEFT, y,
    "x",      C_X,  BG_X,  DS, DM,    # A: [S, M]
    "w_gate", C_W,  BG_W,  DM, DI,    # B: [M, INNER]
    "pre_gate", C_ACT, BG_ACT,         # C: [S, INNER]
    row_dim="S=4096", shared_dim="M=2048", col_dim="INNER=4096",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT,
)
# shape annotation on C
d.shape_right(cx, cy, cw, ch, "[4096, 4096] bf16")

# SiLU arrow + gate box
silu_gap = 40
ax1 = cx + cw + LGAP
ax2 = ax1 + 80
d.transform_arrow(ax1, cy + ch / 2, ax2, "SiLU")

gx = ax2 + LGAP
d.labeled_rect(gx, cy, DI, DS, "gate", C_ACT, BG_ACT,
               dim_top="INNER=4096", dim_side="S=4096",
               shape="[4096, 4096] bf16", fill=FILL_OUT)
gate_cx, gate_cy = gx, cy

y = bot_y + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ② UP PATH:  x @ w_up.T → up
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "② Up path:  x @ w_up.T", FS_TITLE, C_TXT); y += 46

bot_y2, ux, uy, uw, uh = d.matmul_L(
    LEFT, y,
    "x",    C_X, BG_X, DS, DM,    # A: [S, M]
    "w_up", C_W, BG_W, DM, DI,    # B: [M, INNER]
    "up",   C_ACT, BG_ACT,        # C: [S, INNER]
    row_dim="S=4096", shared_dim="M=2048", col_dim="INNER=4096",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(ux, uy, uw, uh, "[4096, 4096] bf16")

y = bot_y2 + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ③ HADAMARD:  gate ⊙ up = h
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "③ Hadamard product:  gate ⊙ up  =  h", FS_TITLE, C_TXT); y += 46

# gate (left)
hm_gx = LEFT
d.labeled_rect(hm_gx, y, DI, DS, "gate", C_ACT, BG_ACT,
               dim_top="INNER=4096", dim_side="S=4096", fill=FILL_OUT)

# ⊙ operator
op_x = hm_gx + DI + 20
d.op_text(op_x, y + DS / 2 - 18, "⊙", 36)

# up (right of ⊙)
hm_ux = op_x + 50
d.labeled_rect(hm_ux, y, DI, DS, "up", C_ACT, BG_ACT,
               dim_top="INNER=4096", dim_side="S=4096", fill=FILL_OUT)

# = operator
eq_x = hm_ux + DI + 20
d.op_text(eq_x, y + DS / 2 - 18, "=", 36)

# h (result)
hm_hx = eq_x + 50
d.labeled_rect(hm_hx, y, DI, DS, "h", C_MUL, BG_MUL,
               dim_top="INNER=4096", dim_side="S=4096",
               shape="[4096, 4096] bf16", fill=FILL_OUT)
h_cx, h_cy = hm_hx, y

y += DS + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ④ DOWN PROJECTION:  h @ w_down.T → out
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "④ Down projection:  h @ w_down.T  →  out", FS_TITLE, C_TXT); y += 46

bot_y3, ox2, oy2, ow, oh = d.matmul_L(
    LEFT, y,
    "h",      C_MUL, BG_MUL, DS, DI,   # A: [S, INNER]
    "w_down", C_W,   BG_W,   DI, DM,   # B: [INNER, M]
    "out",    C_O,   BG_O,              # C: [S, M]
    row_dim="S=4096", shared_dim="INNER=4096", col_dim="M=2048",
    a_fill=FILL_OUT, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(ox2, oy2, ow, oh, "[4096, 2048] bf16")

y = bot_y3 + SECTION_GAP // 2

# ━━ FOOTER NOTE ━━
d.text(LEFT, y,
       "Param count:  w_gate [4096,2048] + w_up [4096,2048] + w_down [2048,4096]  "
       "= 3 × 2048 × 4096 = 25,165,824 ≈ 25 M params per layer",
       FS_NOTE + 1, C_DIM)

out = os.path.join(os.path.dirname(__file__), "dense_swiglu.excalidraw")
d.write(out)

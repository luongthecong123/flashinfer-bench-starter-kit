#!/usr/bin/env python3
"""Generate Excalidraw diagram for Balanced MoE SwiGLU FFN.

E=8 experts, K=2 active per token.
With balanced routing: each expert gets CAP = S*K/E = 4096*2/8 = 1024 tokens.

FLOPs ratio vs dense:
  dense:         3 × S  × M × INNER        = 3 × 4096 × 2048 × 4096
  balanced MoE:  3 × E  × CAP × M × EI     = 3 × 8 × 1024 × 2048 × 512
  ratio = (8 × 1024 × 512) / (4096 × 4096) = 4194304 / 16777216 = 1/4

Stages:
  ① Routing:      x [S,M] × wg.T [M,E] → logits [S,E] → topK → idx[S,K], scores[S,K]
  ② ScatterPack:  x + idx → expert_buf [E, CAP, M]   (CUDA kernel ①)
  ③ Gate BMM:     expert_buf × ew_gate → SiLU → gate [E, CAP, EI]
  ④ Up BMM:       expert_buf × ew_up   → up   [E, CAP, EI]
  ⑤ Hadamard:     gate ⊙ up = h         [E, CAP, EI]
  ⑥ Down BMM:     h × ew_down → yo      [E, CAP, M]
  ⑦ ScatterComb:  yo × weights → out     [S, M]      (CUDA kernel ②)
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

# ── Pixel sizes (custom, small so 3D fits nicely) ──
DS   = 160   # S = 4096  (full sequence, rows)
DM   = 100   # M = 2048  (hidden dim, cols)
DCAP = 50    # CAP = 1024 (tokens per expert = S*K/E)
DEI  = 50    # EXPERT_INNER = 512
DE   = 8     # E = 8 experts — depth pixels for 3D boxes
DK   = 20    # K = 2

# Colors
C_X   = C_Q;    BG_X   = BG_Q       # input x — blue
C_W   = C_KV;   BG_W   = BG_KV      # weights — green
C_BUF = C_IDX;  BG_BUF = BG_IDX     # expert buffer — red
C_ACT = C_SCR;  BG_ACT = BG_SCR     # activations — purple
C_MUL = C_ATTN; BG_MUL = BG_ATTN    # hadamard result — orange
C_O   = C_OUT;  BG_O   = BG_OUT     # output — teal
C_RTR = "#e03131"; BG_RTR = "#ffe3e3"  # router / indices — red

LEFT = 120
y    = 0

# ━━ TITLE ━━
d.text(LEFT, y, "Balanced MoE SwiGLU  ·  K=2/E=8  ·  ¼ dense FLOPs", FS_BIG, C_TXT)
d.text(LEFT, y + 38,
       "S=4096  M=2048  E=8  K=2  EXPERT_INNER=512  CAP=S×K/E=1024 tokens/expert",
       FS_NOTE + 1, C_DIM)
d.text(LEFT, y + 58,
       "FLOPs: (8×1024×512) / (4096×4096) = 1/4 dense  — same param count as dense",
       FS_NOTE + 1, C_DIM)
y += 100

# ━━ DIMENSION LEGEND ━━
d.text(LEFT, y, "Dimensions", FS_TITLE, C_TXT); y += 46
for line in [
    "S = 4096 : total decode tokens",
    "M = 2048 : model / hidden dim",
    "E = 8    : total experts  (3D depth axis on all batched tensors)",
    "K = 2    : active experts per token  →  S×K = 8192 total assignments",
    "EXPERT_INNER = 512 : FFN inner dim per expert  (= INNER/E = 4096/8)",
    "CAP = 1024 : tokens per expert after balanced sort  (= S×K/E, balanced)",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 20
y += SECTION_GAP // 2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INPUT  x [S, M]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "Input", FS_TITLE, C_TXT); y += 46
d.labeled_rect(LEFT, y, DM, DS, "x", C_X, BG_X,
               dim_top="M=2048", dim_side="S=4096",
               shape="[4096, 2048] bf16", fill=FILL_IN)
y += DS + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ① ROUTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "① Routing:  x @ wg.T  →  topK  →  softmax over K", FS_TITLE, C_TXT)
y += 46

bot_y, cx, cy, cw, ch = d.matmul_L(
    LEFT, y,
    "x",      C_X,   BG_X,   DS,  DM,   # A: [S, M]
    "wg",     C_RTR, BG_RTR, DM,  DK*4, # B: [M, E=8]  (thin)
    "logits", C_RTR, BG_RTR,             # C: [S, E]
    row_dim="S=4096", shared_dim="M=2048", col_dim="E=8",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(cx, cy, cw, ch, "[4096, 8] f32")

# topK + softmax arrow
ax1 = cx + cw + LGAP; ax2 = ax1 + 100
d.transform_arrow(ax1, cy + ch / 2, ax2, "topK(2)+softmax")

# idx  [S, K]  and  scores  [S, K]  side by side
idx_x = ax2 + LGAP
d.labeled_rect(idx_x, cy, DK * 2, DS, "idx", C_RTR, BG_RTR,
               dim_top="K=2", dim_side="S=4096",
               shape="[4096,2] int32", fill=FILL_OUT)
sc_x = idx_x + DK * 2 + 30
d.labeled_rect(sc_x, cy, DK * 2, DS, "scores", C_RTR, BG_RTR,
               dim_top="K=2", dim_side="S=4096",
               shape="[4096,2] f32", fill=FILL_OUT)

y = bot_y + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ② SCATTER PACK  (CUDA kernel ①)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "② Scatter Pack  [CUDA kernel ①]  — sort by expert_id, gather x rows",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "  token_ids + expert_ids → argsort → coalesced gather  →  expert_buf[e, slot, :]",
       FS_NOTE + 1, C_DIM)
y += 60

# x again (small)
d.labeled_rect(LEFT, y, DM, DS, "x", C_X, BG_X,
               dim_top="M=2048", dim_side="S=4096", fill=FILL_IN)
d.text(LEFT + DM // 2 - 30, y + DS + 4, "idx, scores", FS_NOTE, C_RTR)

ax1 = LEFT + DM + 12; ax2 = ax1 + 90
d.transform_arrow(ax1, y + DS / 2, ax2, "sort + gather", "CUDA kernel ①")

# expert_buf  [E, CAP, M]  — 3D box
eb_x = ax2 + LGAP
d.labeled_rect_3d(eb_x, y + (DS - DCAP) // 2, DM, DCAP, DE,
                  "expert_buf", C_BUF, BG_BUF,
                  dim_top="M=2048", dim_side="CAP=1024", dim_depth="E=8",
                  shape="[8, 1024, 2048] bf16", fill=FILL_OUT)

# weight_buf  [E, CAP]  — slim 3D box
wb_x = eb_x + DM + DE * 2 + 50
d.labeled_rect_3d(wb_x, y + (DS - DCAP) // 2, 28, DCAP, DE,
                  "weight_buf", C_RTR, BG_RTR,
                  dim_side="CAP=1024", dim_depth="E=8",
                  shape="[8,1024] f32", fill=FILL_OUT)

y += DS + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ③ GATE BMM  (3D batched)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "③ Gate BMM:  expert_buf @ ew_gate.T  →  SiLU  →  gate     (E=8 experts in parallel)",
       FS_TITLE, C_TXT)
y += 46

bot_y, cx, cy, cw, ch = d.bmm_L_3d(
    LEFT, y, DE,
    "expert_buf", C_BUF, BG_BUF, DCAP, DM,    # A: [E, CAP, M]
    "ew_gate",    C_W,   BG_W,   DM,   DEI,   # B: [E, M, EI]
    "pre_gate",   C_ACT, BG_ACT,               # C: [E, CAP, EI]
    row_dim="CAP=1024", shared_dim="M=2048", col_dim="EI=512", batch_dim="E=8",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(cx, cy, cw, ch, "[8, 1024, 512] bf16")

ax1 = cx + cw + DE + LGAP; ax2 = ax1 + 80
d.transform_arrow(ax1, cy + ch / 2, ax2, "SiLU")

gx = ax2 + LGAP
d.labeled_rect_3d(gx, cy, DEI, DCAP, DE, "gate", C_ACT, BG_ACT,
                  dim_top="EI=512", dim_side="CAP=1024", dim_depth="E=8",
                  shape="[8,1024,512]", fill=FILL_OUT)

y = bot_y + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ④ UP BMM  (3D batched)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "④ Up BMM:  expert_buf @ ew_up.T  →  up     (same A tiles, different B)",
       FS_TITLE, C_TXT)
y += 46

bot_y2, ux, uy, uw, uh = d.bmm_L_3d(
    LEFT, y, DE,
    "expert_buf", C_BUF, BG_BUF, DCAP, DM,
    "ew_up",      C_W,   BG_W,   DM,   DEI,
    "up",         C_ACT, BG_ACT,
    row_dim="CAP=1024", shared_dim="M=2048", col_dim="EI=512", batch_dim="E=8",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(ux, uy, uw, uh, "[8, 1024, 512] bf16")

y = bot_y2 + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ⑤ HADAMARD  gate ⊙ up = h  (all 3D)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "⑤ Hadamard:  gate ⊙ up  =  h", FS_TITLE, C_TXT); y += 46

hm_gx = LEFT
d.labeled_rect_3d(hm_gx, y, DEI, DCAP, DE, "gate", C_ACT, BG_ACT,
                  dim_top="EI=512", dim_side="CAP=1024", dim_depth="E=8", fill=FILL_OUT)

op_x = hm_gx + DEI + DE * 2 + 20
d.op_text(op_x, y + DCAP / 2 - 18, "⊙", 36)

hm_ux = op_x + 55
d.labeled_rect_3d(hm_ux, y, DEI, DCAP, DE, "up", C_ACT, BG_ACT,
                  dim_top="EI=512", dim_side="CAP=1024", fill=FILL_OUT)

eq_x = hm_ux + DEI + DE * 2 + 20
d.op_text(eq_x, y + DCAP / 2 - 18, "=", 36)

hm_hx = eq_x + 55
d.labeled_rect_3d(hm_hx, y, DEI, DCAP, DE, "h", C_MUL, BG_MUL,
                  dim_top="EI=512", dim_side="CAP=1024", dim_depth="E=8",
                  shape="[8,1024,512] bf16", fill=FILL_OUT)

y += DCAP + DE * 2 + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ⑥ DOWN BMM  (3D batched)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "⑥ Down BMM:  h @ ew_down.T  →  yo",
       FS_TITLE, C_TXT)
y += 46

bot_y3, ox2, oy2, ow2, oh2 = d.bmm_L_3d(
    LEFT, y, DE,
    "h",       C_MUL, BG_MUL, DCAP, DEI,   # A: [E, CAP, EI]
    "ew_down", C_W,   BG_W,   DEI,  DM,    # B: [E, EI, M]
    "yo",      C_O,   BG_O,                 # C: [E, CAP, M]
    row_dim="CAP=1024", shared_dim="EI=512", col_dim="M=2048", batch_dim="E=8",
    a_fill=FILL_OUT, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(ox2, oy2, ow2, oh2, "[8, 1024, 2048] bf16")

y = bot_y3 + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ⑦ SCATTER COMBINE  (CUDA kernel ②)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "⑦ Scatter Combine  [CUDA kernel ②]  — weighted scatter-add yo back to out[S,M]",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "  out[token_id, :] += weight × yo[e, slot, :]   (atomic-add or two-pass reduce)",
       FS_NOTE + 1, C_DIM)
y += 60

# yo 3D box
d.labeled_rect_3d(LEFT, y, DM, DCAP, DE, "yo", C_O, BG_O,
                  dim_top="M=2048", dim_side="CAP=1024", dim_depth="E=8", fill=FILL_OUT)
d.text(LEFT + DM // 2 - 30, y + DCAP + DE * 2 + 4, "weight_buf", FS_NOTE, C_RTR)

ax1 = LEFT + DM + DE * 2 + 12; ax2 = ax1 + 100
d.transform_arrow(ax1, y + DCAP / 2, ax2, "scatter_add ×w", "CUDA kernel ②")

out_x = ax2 + LGAP
d.labeled_rect(out_x, y - (DS - DCAP) // 2, DM, DS, "out", C_O, BG_O,
               dim_top="M=2048", dim_side="S=4096",
               shape="[4096, 2048] bf16", fill=FILL_OUT)

y += max(DCAP + DE * 2, DS) + SECTION_GAP

# ━━ FLOPS SUMMARY ━━
d.text(LEFT, y, "FLOPs summary  (3 GEMMs, forward only):", FS_TITLE, C_TXT); y += 40
for line in [
    "Dense:        S × M × INNER    × 2 × 3  =  4096 × 2048 × 4096 × 6  ≈  206 GFLOPs",
    "Balanced MoE: E × CAP × M × EI × 2 × 3  =  8 × 1024 × 2048 × 512 × 6  ≈  51.5 GFLOPs  (¼)",
    "Routing:      S × M × E        × 2      =  4096 × 2048 × 8    × 2  ≈   0.13 GFLOPs  (tiny)",
    "FLOPs ratio = (E×CAP×EI) / (S×INNER) = (8×1024×512) / (4096×4096) = 4194304/16777216 = 1/4",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 22

out = os.path.join(os.path.dirname(__file__), "balanced_moe.excalidraw")
d.write(out)

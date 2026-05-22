#!/usr/bin/env python3
"""Generate Excalidraw diagram for Mixtral Sparse MoE SwiGLU FFN.

Faithful HF Mixtral port from sparse_moe() in moe.py:
  E=8 experts, K=2 active per token, EXPERT_INNER = INNER/E = 512.

FLOPs ratio vs dense:
  dense:       3 × S × M × INNER         = 3 × 4096 × 2048 × 4096
  sparse MoE:  3 × n × M × EI × E   (expected n = S×K/E = 1024)
             = 3 × 1024 × 2048 × 512 × 8 ≈ 51.5 GFLOPs  →  1/4 dense
  ratio = K/E = 2/8 = 1/4  (compute)
  wall-time overhead >> balanced_moe due to torch.where + serial per-expert GEMMs.

Stages:
  ① Routing:    x [S,M] × wg.T [M,E] → logits [S,E]
                → softmax(all E, fp32) → topK(K) → top_probs [S,K], idx [S,K]
                → renormalize → scores [S,K]
  ② Mask:       one_hot(idx, E) [S,K,E] → permute(2,1,0) → expert_mask [E,K,S]
  ③ Expert loop  (for e in range(E)):
      A. Gather:   torch.where(expert_mask[e]) → token_idx [n]
                   xi = x[token_idx]                          [n, M]
      B. Gate+Up:  xi @ ew_gate_up[e].T → gate_up [n, 2×EI]
                   → chunk(2) → gate [n,EI],  up [n,EI]
      C. SwiGLU:   SiLU(gate) ⊙ up  →  h  [n, EI]
      D. Down:     h @ ew_down[e].T  →  yo [n, M]
      E. Combine:  out.index_add_(0, token_idx, yo × weights)
  ④ Output:     out [S, M]
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

# ── Pixel sizes ──────────────────────────────────────────────────────
DS   = 160   # S = 4096  (full sequence rows)
DM   = 100   # M = 2048  (model / hidden dim)
DEI  = 50    # EXPERT_INNER = 512
D2EI = 100   # 2 × EXPERT_INNER = 1024  (fused gate_up output width)
DN   = 50    # n ≈ 1024  (tokens routed to one expert, variable)
DK   = 20    # K = 2
DES  = 30    # E = 8  (width for [S,E] logits matrix)
DE   = 32    # depth pixels for 3D expert_mask box  (E = 8 experts)

# Color scheme (matches other gen_*.py diagrams)
C_X   = C_Q;    BG_X   = BG_Q       # input x — blue
C_W   = C_KV;   BG_W   = BG_KV      # weight matrices — green
C_ACT = C_SCR;  BG_ACT = BG_SCR     # intermediate activations — purple
C_MUL = C_ATTN; BG_MUL = BG_ATTN    # SwiGLU result h — orange
C_O   = C_OUT;  BG_O   = BG_OUT     # output tensors — teal
C_RTR = "#e03131"; BG_RTR = "#ffe3e3"  # router / indices — red
C_MSK = C_IDX;  BG_MSK = BG_IDX        # expert mask — red (same as indices)

LEFT = 120
y    = 0

# ━━ TITLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "Mixtral Sparse MoE  ·  K=2/E=8  ·  one_hot mask + torch.where dispatch",
       FS_BIG, C_TXT)
d.text(LEFT, y + 38,
       "S=4096  M=2048  E=8  K=2  EXPERT_INNER=512  (n ≈ S×K/E = 1024 tokens/expert expected)",
       FS_NOTE + 1, C_DIM)
d.text(LEFT, y + 58,
       "Routing: softmax(logits, all E, fp32) → topK(2) → renormalize.  "
       "Dispatch: one_hot + torch.where loop.  FLOPs ratio = K/E = ¼ dense (same as balanced).",
       FS_NOTE + 1, C_DIM)
y += 100

# ━━ DIMENSION LEGEND ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "Dimensions", FS_TITLE, C_TXT); y += 46
for line in [
    "S = 4096           : total decode tokens",
    "M = 2048           : model / hidden dim",
    "E = 8              : total experts",
    "K = 2              : active experts per token  →  S×K = 8192 total assignments",
    "EXPERT_INNER = 512 : FFN inner dim per expert  (= INNER/E = 4096/8)",
    "n ≈ 1024           : tokens routed to one expert (expected S×K/E; varies per step)",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 20
y += SECTION_GAP // 2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INPUT  x [S, M]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "Input", FS_TITLE, C_TXT); y += 46
d.labeled_rect(LEFT, y, DM, DS, "x", C_X, BG_X,
               dim_top="M=2048", dim_side="S=4096",
               shape="[4096, 2048] bf16", fill=FILL_IN)
y += DS + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ① ROUTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "① Routing:  x @ wg.T  →  softmax(all E, fp32)  →  topK  →  renormalize",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "  logits = F.linear(x, wg)  →  probs = softmax(fp32)  →  topK(K=2)  →  top_probs / sum(top_probs)",
       FS_NOTE + 1, C_DIM)
y += 60

bot_y, cx, cy, cw, ch = d.matmul_L(
    LEFT, y,
    "x",      C_X,   BG_X,   DS,  DM,   # A: [S, M]
    "wg",     C_RTR, BG_RTR, DM,  DES,  # B: [M, E=8]  (thin — E=8)
    "logits", C_RTR, BG_RTR,             # C: [S, E]
    row_dim="S=4096", shared_dim="M=2048", col_dim="E=8",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(cx, cy, cw, ch, "[4096, 8] f32")

# softmax(all E) → topK → renorm arrow
ax1 = cx + cw + LGAP; ax2 = ax1 + 160
d.transform_arrow(ax1, cy + ch / 2, ax2, "softmax(all E)→topK(2)→renorm")

# idx [S, K]  and  scores [S, K]
idx_x = ax2 + LGAP
d.labeled_rect(idx_x, cy, DK * 2, DS, "idx", C_RTR, BG_RTR,
               dim_top="K=2", dim_side="S=4096",
               shape="[4096,2] int32", fill=FILL_OUT)
sc_x = idx_x + DK * 2 + 35
d.labeled_rect(sc_x, cy, DK * 2, DS, "scores", C_RTR, BG_RTR,
               dim_top="K=2", dim_side="S=4096",
               shape="[4096,2] f32", fill=FILL_OUT)

y = bot_y + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ② EXPERT MASK  — one_hot dispatch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "② Expert Mask:  one_hot(idx, num_classes=E)  →  permute(2,1,0)  →  expert_mask [E, K, S]",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "  one_hot: [S,K] → [S,K,E]  →  permute(2,1,0)  →  [E,K,S].  "
       "Slice expert_mask[e] → [K,S] used in torch.where each iteration.",
       FS_NOTE + 1, C_DIM)
y += 60

# idx (small, FILL_IN as it was output from routing)
d.labeled_rect(LEFT, y, DK * 2, DS, "idx", C_RTR, BG_RTR,
               dim_top="K=2", dim_side="S=4096", fill=FILL_IN)

ax1 = LEFT + DK * 2 + LGAP; ax2 = ax1 + 130
d.transform_arrow(ax1, y + DS / 2, ax2, "one_hot(E=8) → permute")

# expert_mask [E, K, S] — 3D box (depth = E axis)
em_x = ax2 + LGAP
D_EM_K = DK * 2    # K=2 axis (width, 40 px)
D_EM_S = DS        # S=4096 axis (height)
D_EM_E = DE        # E=8 depth (32 px)
d.labeled_rect_3d(em_x, y, D_EM_K, D_EM_S, D_EM_E,
                  "expert_mask", C_MSK, BG_MSK,
                  dim_top="K=2", dim_side="S=4096", dim_depth="E=8",
                  shape="[8, 2, 4096] bool", fill=FILL_OUT)

y += DS + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ③ EXPERT LOOP  (one expert shown, ×8)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "③ Expert loop  (for e in range(8)):  one expert shown — same pattern × 8",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "  Each iteration is independent; E=8 serial GEMMs (no batched bmm).",
       FS_NOTE + 1, C_DIM)
y += 60

# ─────────────────────────────────────────────────────────────────────
#  A. Gather  xi = x[token_idx]
# ─────────────────────────────────────────────────────────────────────
d.text(LEFT, y, "  A.  Gather:  xi = x[token_idx]", FS_TITLE - 2, C_TXT)
d.text(LEFT, y + 26,
       "      expert_mask[e] [K,S]  →  torch.where  →  (top_k_pos, token_idx)  "
       "→  xi = x[token_idx]  [n, M]",
       FS_NOTE + 1, C_DIM)
y += 56

# expert_mask[e]  [K, S]  — one slice
d.labeled_rect(LEFT, y, D_EM_K, D_EM_S, "mask[e]", C_MSK, BG_MSK,
               dim_top="K=2", dim_side="S=4096", fill=FILL_IN)

d.text(LEFT + D_EM_K + LGAP, y + D_EM_S // 2 - 10, "torch.where →", FS_NOTE + 1, C_OP)

# token_idx [n]
ti_x = LEFT + D_EM_K + LGAP + 115
d.labeled_rect(ti_x, y + (DS - DN) // 2, 22, DN,
               "tok_idx", C_RTR, BG_RTR,
               dim_side="n≈1024", shape="[n] int32", fill=FILL_OUT)

# x (full) on the right
xg_x = ti_x + 22 + 50
d.labeled_rect(xg_x, y, DM, DS, "x", C_X, BG_X,
               dim_top="M=2048", dim_side="S=4096", fill=FILL_IN)

# gather arrow
ax1 = xg_x + DM + LGAP; ax2 = ax1 + 90
d.transform_arrow(ax1, y + DS / 2, ax2, "gather rows")

# xi [n, M]
xi_x = ax2 + LGAP
d.labeled_rect(xi_x, y + (DS - DN) // 2, DM, DN, "xi", C_X, BG_X,
               dim_top="M=2048", dim_side="n≈1024",
               shape="[n, 2048] bf16", fill=FILL_OUT)

y += DS + SECTION_GAP

# ─────────────────────────────────────────────────────────────────────
#  B. Fused Gate+Up projection  xi @ ew_gate_up[e].T  →  chunk
# ─────────────────────────────────────────────────────────────────────
d.text(LEFT, y,
       "  B.  Gate+Up (fused):  xi @ ew_gate_up[e].T  →  chunk(2)  →  gate [n,EI], up [n,EI]",
       FS_TITLE - 2, C_TXT)
d.text(LEFT, y + 26,
       "      ew_gate_up[e]: [2×EI, M] = [1024, 2048]  (gate and up projections stored fused)",
       FS_NOTE + 1, C_DIM)
y += 56

bot_y2, cx2, cy2, cw2, ch2 = d.matmul_L(
    LEFT, y,
    "xi",          C_X,   BG_X,   DN,  DM,    # A: [n, M]
    "ew_gate_up",  C_W,   BG_W,   DM,  D2EI,  # B: [M, 2×EI]
    "gate_up",     C_ACT, BG_ACT,              # C: [n, 2×EI]
    row_dim="n≈1024", shared_dim="M=2048", col_dim="2×EI=1024",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(cx2, cy2, cw2, ch2, "[n, 1024] bf16")

# chunk arrow → gate, up (side by side)
ax1 = cx2 + cw2 + LGAP; ax2 = ax1 + 80
d.transform_arrow(ax1, cy2 + ch2 / 2, ax2, "chunk(2, dim=-1)")

g_x = ax2 + LGAP
d.labeled_rect(g_x, cy2, DEI, DN, "gate", C_ACT, BG_ACT,
               dim_top="EI=512", dim_side="n≈1024", fill=FILL_OUT)
up_x = g_x + DEI + 30
d.labeled_rect(up_x, cy2, DEI, DN, "up", C_ACT, BG_ACT,
               dim_top="EI=512", dim_side="n≈1024", fill=FILL_OUT)

y = bot_y2 + SECTION_GAP

# ─────────────────────────────────────────────────────────────────────
#  C. SwiGLU  SiLU(gate) ⊙ up = h
# ─────────────────────────────────────────────────────────────────────
d.text(LEFT, y, "  C.  SwiGLU:  SiLU(gate) ⊙ up  =  h", FS_TITLE - 2, C_TXT); y += 46

# gate (input to SiLU)
hm_gx = LEFT
d.labeled_rect(hm_gx, y, DEI, DN, "gate", C_ACT, BG_ACT,
               dim_top="EI=512", dim_side="n≈1024", fill=FILL_OUT)

# SiLU arrow
silu_ax1 = hm_gx + DEI + LGAP
silu_ax2 = silu_ax1 + 70
d.transform_arrow(silu_ax1, y + DN / 2, silu_ax2, "SiLU")

# gate* after SiLU
sg_x = silu_ax2 + LGAP
d.labeled_rect(sg_x, y, DEI, DN, "gate*", C_ACT, BG_ACT,
               dim_top="EI=512", fill=FILL_OUT)

# ⊙  operator
d.op_text(sg_x + DEI + 15, y + DN / 2 - 18, "⊙", 36)

# up
up_x2 = sg_x + DEI + 55
d.labeled_rect(up_x2, y, DEI, DN, "up", C_ACT, BG_ACT,
               dim_top="EI=512", fill=FILL_OUT)

# =
d.op_text(up_x2 + DEI + 15, y + DN / 2 - 18, "=", 36)

# h
h_x = up_x2 + DEI + 55
d.labeled_rect(h_x, y, DEI, DN, "h", C_MUL, BG_MUL,
               dim_top="EI=512", dim_side="n≈1024",
               shape="[n, 512] bf16", fill=FILL_OUT)

y += DN + SECTION_GAP

# ─────────────────────────────────────────────────────────────────────
#  D. Down projection  h @ ew_down[e].T → yo
# ─────────────────────────────────────────────────────────────────────
d.text(LEFT, y, "  D.  Down:  h @ ew_down[e].T  →  yo", FS_TITLE - 2, C_TXT)
d.text(LEFT, y + 26,
       "      ew_down[e]: [M, EI] = [2048, 512]",
       FS_NOTE + 1, C_DIM)
y += 56

bot_y3, ox3, oy3, ow3, oh3 = d.matmul_L(
    LEFT, y,
    "h",       C_MUL, BG_MUL, DN,  DEI,   # A: [n, EI]
    "ew_down", C_W,   BG_W,   DEI, DM,    # B: [EI, M]
    "yo",      C_O,   BG_O,               # C: [n, M]
    row_dim="n≈1024", shared_dim="EI=512", col_dim="M=2048",
    a_fill=FILL_OUT, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(ox3, oy3, ow3, oh3, "[n, 2048] bf16")

y = bot_y3 + SECTION_GAP

# ─────────────────────────────────────────────────────────────────────
#  E. Weighted combine  out.index_add_(0, token_idx, yo × weights)
# ─────────────────────────────────────────────────────────────────────
d.text(LEFT, y,
       "  E.  Combine:  out.index_add_(0, token_idx,  yo × weights)",
       FS_TITLE - 2, C_TXT)
d.text(LEFT, y + 28,
       "      yo [n,M] × weights [n,1]  →  scatter-add into out[S,M].  "
       "No race: each token appears in ≤ K=2 experts → ≤ 2 additions per output row.",
       FS_NOTE + 1, C_DIM)
y += 60

# yo [n, M]
d.labeled_rect(LEFT, y, DM, DN, "yo", C_O, BG_O,
               dim_top="M=2048", dim_side="n≈1024", fill=FILL_OUT)
# weights [n, 1]  (slim column)
d.labeled_rect(LEFT + DM + 10, y + (DN - DK * 2) // 2, 22, DK * 2,
               "w", C_RTR, BG_RTR,
               dim_side="n", fill=FILL_OUT)

# index_add arrow
ax1 = LEFT + DM + 45 + LGAP; ax2 = ax1 + 110
d.transform_arrow(ax1, y + DN / 2, ax2, "index_add_ ×w", "token_idx [n]")

# out [S, M]  (full sequence, offset upward to centre-align with yo)
out_x = ax2 + LGAP
out_top = y - (DS - DN) // 2
d.labeled_rect(out_x, out_top, DM, DS, "out", C_O, BG_O,
               dim_top="M=2048", dim_side="S=4096",
               shape="[4096, 2048] bf16", fill=FILL_OUT)

y += max(DN, DS - (DS - DN) // 2) + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KEY DIFFERENCES  vs Balanced MoE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "Key differences vs Balanced MoE:", FS_TITLE, C_TXT); y += 40
for line in [
    "• Routing scope:  softmax over ALL E experts first, then topK → renormalize  "
    "(balanced: same gate)",
    "• Dispatch:  one_hot [S,K,E] + permute + torch.where per expert  "
    "(vs sort-by-expert + coalesced scatter_pack kernel)",
    "• GEMM pattern:  8 separate small GEMMs [n,M]×[M,2EI] — serial, no batching  "
    "(vs 2 batched bmm [E,CAP,M]×[E,M,EI])",
    "• Token count:  variable n per expert — no capacity buffer, no padding  "
    "(balanced: fixed CAP = S×K/E with 1.25× overflow)",
    "• index_add_ per expert  (balanced: scatter_combine kernel with pre-sorted indices)",
    "• Wall-time:  higher overhead per FLOP — torch.where + control flow + no bmm batching",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 22

y += SECTION_GAP // 2

# ━━ FLOPS SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "FLOPs summary  (3 GEMMs × E=8 experts):", FS_TITLE, C_TXT); y += 40
for line in [
    "Dense:       S × M × INNER         × 2 × 3  =  4096 × 2048 × 4096 × 6  ≈ 206 GFLOPs",
    "Sparse MoE:  n × M × EI            × 2 × 3 × E  (n = S×K/E ≈ 1024)",
    "           = 1024 × 2048 × 512 × 6 × 8  ≈  51.5 GFLOPs  (¼ dense)  [same FLOPs as balanced]",
    "Routing:     S × M × E             × 2      =  4096 × 2048 × 8 × 2  ≈  0.13 GFLOPs  (tiny)",
    "FLOPs ratio = K/E = 2/8 = ¼  — but wall-time ~2–4× slower than balanced due to serial dispatch",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 22

out = os.path.join(os.path.dirname(__file__), "mixtral_moe.excalidraw")
d.write(out)

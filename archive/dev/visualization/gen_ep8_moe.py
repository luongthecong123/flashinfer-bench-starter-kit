#!/usr/bin/env python3
"""Generate Excalidraw diagram for DeepSpeed MoE with Expert Parallelism = 8.

Target: DiT-3D (dit3d_moe.py / dit3d_config.yaml) trained on 8×H100.
  S_local = 4096 tokens per GPU,  M = 4096 hidden,
  E = 8 experts,  K = 2 top-k,  EXPERT_INNER = 7168 (Mixtral ratio 1.75),
  ep_size = 8  →  num_local_experts = E / ep_size = 1 expert per GPU.

DeepSpeed MoE forward (sharded_moe.MOELayer.forward, ep_size=G=8):

  per-GPU input  x_g [S_local, M]
        │
  ① ROUTE  (local)    logits = x @ wg.T  →  softmax(fp32)  →  topK(2)  →  renorm
                       l_aux load-balance loss accumulated for training
        ↓
  ② DISPATCH (local)  combine_weights, dispatch_mask  [S_local, E, C]
                       dispatched = einsum("sec,sm->ecm", mask, x)  →  [E, C, M]
        │
        │  ── AllToAll #1 ──────────────────────────────────────────────
        │  dist.all_to_all_single(out, dispatched, group=ep_group)
        │  GPU i sends slice for expert j to GPU j.  After A2A each GPU
        │  holds tokens from ALL 8 GPUs but only for its 1 local expert:
        │  shape becomes  [G=8, local_E=1, C, M]
        ↓
  ③ EXPERT  (local, 1 expert per GPU)
            SwiGLU FFN:  x @ Wg.T, x @ Wu.T,  SiLU(g)*u,  h @ Wd.T
            Each GPU runs ITS OWN expert weights only.
        │
        │  ── AllToAll #2 ──────────────────────────────────────────────
        │  Send results back.  After A2A each GPU again holds [E, C, M]
        │  (its own slice of every expert's output).
        ↓
  ④ COMBINE (local)   einsum("sec,ecm->sm", combine_weights, expert_out)
                       →  out [S_local, M]

ZeRO interaction:
  Expert weights tagged param.allreduce=False, group_name="ep_size_8".
  → Gradient AllReduce uses expert_dp_process_group (skipped at full EP=world_size).
  → Optimizer states sharded only within expert DP group.
  Non-expert weights (attention, gates, norms) follow standard ZeRO-2 DP.
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
DS    = 160   # S_local = 4096
DM    = 100   # M = 4096
DEI   = 80    # EXPERT_INNER = 7168
D2EI  = 160   # 2×EI = 14336 (fused gate_up output)
DC    = 60    # capacity C = 1024 (S_local * K / E with cf=1)
DE    = 50    # E = 8 (depth for 3D dispatched tensor)
DG    = 50    # G = ep_size = 8 (depth axis post-AllToAll)
DK    = 20    # K = 2

# Color scheme
C_X    = C_Q;    BG_X    = BG_Q       # input x
C_W    = C_KV;   BG_W    = BG_KV      # expert weights
C_ACT  = C_SCR;  BG_ACT  = BG_SCR     # gate/up activations
C_MUL  = C_ATTN; BG_MUL  = BG_ATTN    # SwiGLU result h
C_O    = C_OUT;  BG_O    = BG_OUT     # output
C_RTR  = "#e03131"; BG_RTR = "#ffe3e3"  # router / gate
C_MSK  = C_IDX;  BG_MSK  = BG_IDX        # dispatch mask
C_A2A  = "#0c8599"; BG_A2A = "#c5f6fa"  # AllToAll bands

LEFT = 120
y    = 0

# ━━ TITLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "DeepSpeed MoE  ·  Expert Parallelism EP = 8  ·  1 expert / GPU  (DiT-3D, 8×H100)",
       FS_BIG, C_TXT)
d.text(LEFT, y + 38,
       "Per GPU:  S_local=4096  M=4096  E=8  K=2  EXPERT_INNER=7168  C=1024  "
       "(num_local_experts = E / ep_size = 1)",
       FS_NOTE + 1, C_DIM)
d.text(LEFT, y + 58,
       "Two AllToAll collectives wrap the expert compute.  Routing + dispatch + combine are LOCAL on every GPU.  "
       "Expert weights are SHARDED across the 8 GPUs.",
       FS_NOTE + 1, C_DIM)
y += 100

# ━━ DIMENSION LEGEND ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "Dimensions per GPU", FS_TITLE, C_TXT); y += 46
for line in [
    "G  = ep_size = 8        : number of EP ranks (one per H100)",
    "S_local = 4096          : tokens visible to this GPU (after DP shard)",
    "M  = 4096               : hidden dim",
    "E  = 8                  : total experts in the MoE layer",
    "K  = 2                  : top-k experts per token",
    "C  = 1024               : capacity slot count = ceil(S_local × cf × K / E),  cf=1",
    "EXPERT_INNER = 7168     : SwiGLU inner dim (4096 × 1.75, Mixtral ratio)",
    "num_local_experts = E/G = 1   ← each GPU owns ONE expert",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 20
y += SECTION_GAP // 2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "Input on each GPU  (this view = GPU 0; GPUs 1–7 mirror it)",
       FS_TITLE, C_TXT); y += 46
d.labeled_rect(LEFT, y, DM, DS, "x", C_X, BG_X,
               dim_top="M=4096", dim_side="S_local=4096",
               shape="[4096, 4096] bf16", fill=FILL_IN)
y += DS + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ① LOCAL ROUTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "①  Local routing  (TopKGate.forward, runs independently on every GPU)",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "   logits = x @ wg.T  →  softmax(fp32, all E)  →  topK(2)  →  renormalize  "
       "→  combine_weights, dispatch_mask  [S_local, E, C]",
       FS_NOTE + 1, C_DIM)
y += 60

DES = 30
bot_y, cx, cy, cw, ch = d.matmul_L(
    LEFT, y,
    "x",        C_X,   BG_X,   DS,  DM,
    "wg",       C_RTR, BG_RTR, DM,  DES,
    "logits",   C_RTR, BG_RTR,
    row_dim="S_local=4096", shared_dim="M=4096", col_dim="E=8",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(cx, cy, cw, ch, "[4096, 8] f32")

ax1 = cx + cw + LGAP; ax2 = ax1 + 200
d.transform_arrow(ax1, cy + ch / 2, ax2,
                  "softmax(fp32) → topK(2) → renorm",
                  "→ combine_weights, dispatch_mask")

# combine_weights [S, E, C] as 3D box (depth = C)
cw_x = ax2 + LGAP
d.labeled_rect_3d(cw_x, cy, DES, DS, DC,
                  "combine_w", C_RTR, BG_RTR,
                  dim_top="E=8", dim_side="S_local=4096", dim_depth="C=1024",
                  shape="[4096, 8, 1024] bf16", fill=FILL_OUT)

y = bot_y + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ② DISPATCH einsum  →  [E, C, M]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "②  Local dispatch  (einsum 'sec,sm → ecm')  — pack tokens into per-expert capacity buffer",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "   dispatched = einsum(dispatch_mask.bool→type, x)  →  [E=8, C=1024, M=4096]  on each GPU",
       FS_NOTE + 1, C_DIM)
y += 60

# Show 3D dispatch_mask  + x  → 3D dispatched
d.labeled_rect_3d(LEFT, y, DES, DS, DC,
                  "disp_mask", C_MSK, BG_MSK,
                  dim_top="E=8", dim_side="S_local", dim_depth="C", fill=FILL_IN)

ix = LEFT + DES + DC + 60
d.labeled_rect(ix, y, DM, DS, "x", C_X, BG_X,
               dim_top="M=4096", dim_side="S_local", fill=FILL_IN)

ax1 = ix + DM + LGAP; ax2 = ax1 + 130
d.transform_arrow(ax1, y + DS / 2, ax2, "einsum sec,sm→ecm")

dx = ax2 + LGAP
# dispatched: 3D with depth=E (use rect_3d depth axis = E)
DISP_W = DM       # M
DISP_H = DC       # C
DISP_D = DE       # E
d.labeled_rect_3d(dx, y + (DS - DISP_H)//2, DISP_W, DISP_H, DISP_D,
                  "dispatched", C_O, BG_O,
                  dim_top="M=4096", dim_side="C=1024", dim_depth="E=8",
                  shape="[8, 1024, 4096] bf16", fill=FILL_OUT)

y += DS + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AllToAll #1  banner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BANNER_H = 80
d.rect(LEFT - 20, y, 1500, BANNER_H, C_A2A, BG_A2A, "hachure")
d.text(LEFT, y + 14,
       "── AllToAll #1   dist.all_to_all_single(out, dispatched, group=ep_group)  "
       "── 8-way exchange across H100s ──",
       FS_TITLE - 2, C_A2A)
d.text(LEFT, y + 44,
       "Before:  every GPU has [E=8, C, M]  (full E, partial S).      "
       "After:  every GPU has [G=8, local_E=1, C, M]  (full G stitched, only its 1 expert).",
       FS_NOTE + 1, C_TXT)
y += BANNER_H + SECTION_GAP // 2

# ── 8-rank exchange visual: small 8×8 grid of arrows (compact) ──────
d.text(LEFT, y,
       "GPU i sends its expert-j slice to GPU j  →  GPU j receives its expert from all 8 GPUs",
       FS_NOTE + 1, C_DIM)
y += 26

# Draw 8 source mini-tensors on left, 8 dest on right
SRC_X = LEFT
DST_X = LEFT + 700
ROW_H = 22
ROW_GAP = 6
for i in range(8):
    yy = y + i * (ROW_H + ROW_GAP)
    # source [E, C, M] tile per GPU i
    d.rect(SRC_X, yy, 80, ROW_H, C_O, BG_O, FILL_IN)
    d.text(SRC_X + 6, yy + 2, f"GPU {i}: [E=8, C, M]", FS_NOTE, C_TXT)
for j in range(8):
    yy = y + j * (ROW_H + ROW_GAP)
    d.rect(DST_X, yy, 230, ROW_H, C_O, BG_O, FILL_OUT)
    d.text(DST_X + 6, yy + 2,
           f"GPU {j}: tokens for expert {j} from all 8 GPUs  →  [G=8, 1, C, M]",
           FS_NOTE, C_TXT)

# Crossing arrows: every i→every j (full 8×8 mesh)
for i in range(8):
    for j in range(8):
        y_src = y + i * (ROW_H + ROW_GAP) + ROW_H / 2
        y_dst = y + j * (ROW_H + ROW_GAP) + ROW_H / 2
        # color the diagonal (i==j) hot, off-diagonal lighter
        col = C_A2A if i != j else "#e03131"
        d.line(SRC_X + 80, y_src,
               [[0, 0], [DST_X - SRC_X - 80, y_dst - y_src]],
               col, "solid")

y += 8 * (ROW_H + ROW_GAP) + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ③ LOCAL EXPERT COMPUTE   (only this GPU's 1 expert weights live here)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "③  Local expert compute  on GPU j  —  only expert j's weights are present  (1 expert / GPU)",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "   Input slab:  [G=8, 1, C, M] reshaped → [G·C = 8192, M].  "
       "SwiGLU FFN with this expert's gate_proj, up_proj, down_proj.",
       FS_NOTE + 1, C_DIM)
y += 60

# Local expert input: G·C rows, M cols
DGC = 200    # G * C = 8192 → tall
d.labeled_rect(LEFT, y, DM, DGC, "expert_in_j", C_X, BG_X,
               dim_top="M=4096", dim_side="G·C = 8192",
               shape="[8192, 4096] bf16  (this GPU only)", fill=FILL_IN)

# Gate+Up matmul for expert j
mx = LEFT + DM + LGAP + 40
d.text(mx, y, "Stage A — Gate+Up (fused)", FS_TITLE - 2, C_TXT)
bot_y, cxA, cyA, cwA, chA = d.matmul_L(
    mx, y + 40,
    "expert_in_j",  C_X,  BG_X,  DGC, DM,
    "Wgu_j",        C_W,  BG_W,  DM,  D2EI,
    "gate_up_j",    C_ACT, BG_ACT,
    row_dim="G·C", shared_dim="M=4096", col_dim="2·EI=14336",
    a_fill=FILL_IN, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(cxA, cyA, cwA, chA, "[8192, 14336] bf16")

# Note about weight residency
d.text(mx, bot_y + 16,
       "Wgu_j only exists on GPU j  (param.allreduce=False, group_name='ep_size_8')",
       FS_NOTE + 1, C_DIM)

y = bot_y + 50 + SECTION_GAP

# Stage B: chunk + SwiGLU
d.text(LEFT, y, "Stage B — chunk → SiLU(gate) ⊙ up = h", FS_TITLE - 2, C_TXT); y += 40
d.labeled_rect(LEFT, y, DEI, DGC, "gate_j", C_ACT, BG_ACT,
               dim_top="EI=7168", dim_side="G·C", fill=FILL_OUT)
sx = LEFT + DEI + LGAP
d.transform_arrow(sx, y + DGC/2, sx + 70, "SiLU")
sg_x = sx + 70 + LGAP
d.labeled_rect(sg_x, y, DEI, DGC, "gate*", C_ACT, BG_ACT,
               dim_top="EI", fill=FILL_OUT)
d.op_text(sg_x + DEI + 12, y + DGC/2 - 18, "⊙", 36)
up_x = sg_x + DEI + 50
d.labeled_rect(up_x, y, DEI, DGC, "up_j", C_ACT, BG_ACT,
               dim_top="EI", fill=FILL_OUT)
d.op_text(up_x + DEI + 12, y + DGC/2 - 18, "=", 36)
h_x = up_x + DEI + 50
d.labeled_rect(h_x, y, DEI, DGC, "h_j", C_MUL, BG_MUL,
               dim_top="EI=7168", dim_side="G·C",
               shape="[8192, 7168] bf16", fill=FILL_OUT)
y += DGC + SECTION_GAP

# Stage C: down projection
d.text(LEFT, y, "Stage C — Down projection  h_j @ Wd_j.T", FS_TITLE - 2, C_TXT); y += 40
bot_y, cxC, cyC, cwC, chC = d.matmul_L(
    LEFT, y,
    "h_j",   C_MUL, BG_MUL, DGC, DEI,
    "Wd_j",  C_W,   BG_W,   DEI, DM,
    "exp_out_j", C_O, BG_O,
    row_dim="G·C", shared_dim="EI=7168", col_dim="M=4096",
    a_fill=FILL_OUT, b_fill=FILL_IN, c_fill=FILL_OUT,
)
d.shape_right(cxC, cyC, cwC, chC, "[8192, 4096] bf16")
y = bot_y + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AllToAll #2  banner  (reverse direction)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.rect(LEFT - 20, y, 1500, BANNER_H, C_A2A, BG_A2A, "hachure")
d.text(LEFT, y + 14,
       "── AllToAll #2   dist.all_to_all_single(out, expert_out, group=ep_group)  "
       "── reverse 8-way exchange ──",
       FS_TITLE - 2, C_A2A)
d.text(LEFT, y + 44,
       "Before:  GPU j has [G=8, 1, C, M] expert-j outputs for every GPU.        "
       "After:  every GPU again has [E=8, C, M]  (its tokens, all 8 experts).",
       FS_NOTE + 1, C_TXT)
y += BANNER_H + SECTION_GAP // 2

# Mirror crossing arrows (right→left this time)
for i in range(8):
    yy = y + i * (ROW_H + ROW_GAP)
    d.rect(SRC_X, yy, 230, ROW_H, C_O, BG_O, FILL_IN)
    d.text(SRC_X + 6, yy + 2,
           f"GPU {i}: expert-{i} out for all 8 GPUs  [G=8, 1, C, M]",
           FS_NOTE, C_TXT)
for j in range(8):
    yy = y + j * (ROW_H + ROW_GAP)
    d.rect(DST_X, yy, 80, ROW_H, C_O, BG_O, FILL_OUT)
    d.text(DST_X + 6, yy + 2, f"GPU {j}: [E=8, C, M]", FS_NOTE, C_TXT)
for i in range(8):
    for j in range(8):
        y_src = y + i * (ROW_H + ROW_GAP) + ROW_H / 2
        y_dst = y + j * (ROW_H + ROW_GAP) + ROW_H / 2
        col = C_A2A if i != j else "#e03131"
        d.line(SRC_X + 230, y_src,
               [[0, 0], [DST_X - SRC_X - 230, y_dst - y_src]],
               col, "solid")
y += 8 * (ROW_H + ROW_GAP) + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ④ LOCAL COMBINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y,
       "④  Local combine  (einsum 'sec,ecm → sm')  — weighted sum back to token positions",
       FS_TITLE, C_TXT)
d.text(LEFT, y + 30,
       "   out = einsum(combine_weights, expert_out)  →  [S_local=4096, M=4096]  "
       "(rescaled by topK gate scores for the 2 experts each token visited)",
       FS_NOTE + 1, C_DIM)
y += 60

# combine_w  [S, E, C] (3D)
d.labeled_rect_3d(LEFT, y, DES, DS, DC,
                  "combine_w", C_RTR, BG_RTR,
                  dim_top="E=8", dim_side="S_local", dim_depth="C", fill=FILL_IN)

ex = LEFT + DES + DC + 60
d.labeled_rect_3d(ex, y + (DS - DC)//2, DM, DC, DE,
                  "expert_out", C_O, BG_O,
                  dim_top="M=4096", dim_side="C", dim_depth="E=8", fill=FILL_IN)

ax1 = ex + DM + LGAP + 30; ax2 = ax1 + 130
d.transform_arrow(ax1, y + DS/2, ax2, "einsum sec,ecm→sm")

ox = ax2 + LGAP
d.labeled_rect(ox, y, DM, DS, "out", C_O, BG_O,
               dim_top="M=4096", dim_side="S_local=4096",
               shape="[4096, 4096] bf16", fill=FILL_OUT)
y += DS + SECTION_GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WHAT MAKES IT ZeRO-COMPATIBLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "Why this composes with ZeRO-2 (DeepSpeed magic):",
       FS_TITLE, C_TXT); y += 40
for line in [
    "• Every expert weight gets stamped:  param.allreduce = False  +  param.group_name = 'ep_size_8'",
    "  (deepspeed/moe/experts.py — the ONLY two attributes that do the trick)",
    "• ZeRO-2 engine checks is_moe_param(p) → routes its grad AllReduce to expert_dp_process_group,",
    "  not the full DP group.  At ep_size = world_size = 8 that group has size 1 → grad AllReduce skipped.",
    "• Optimizer states for expert params are sharded only within that smaller group → 1/G memory.",
    "• Non-expert params (attention, qkv, norms, router wg) follow the standard ZeRO-2 DP path unchanged.",
    "• split_params_into_different_moe_groups_for_optimizer() must be called before optimizer init",
    "  to carve expert params into their own param_group with {'moe': True}.",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 22
y += SECTION_GAP // 2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COMM / COMPUTE SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(LEFT, y, "Per-layer comm + compute (forward, one micro-batch):",
       FS_TITLE, C_TXT); y += 40
for line in [
    "AllToAll #1 volume per GPU:  E × C × M × 2B  =  8 × 1024 × 4096 × 2  ≈  64 MiB out (and 64 MiB in)",
    "AllToAll #2 volume per GPU:  same  ≈  64 MiB each direction",
    "Local expert FLOPs (1 expert × G·C tokens):",
    "    gate_up:  8192 × 4096 × 14336 × 2  ≈  962 GFLOPs",
    "    down:     8192 × 7168 × 4096  × 2  ≈  481 GFLOPs",
    "    total:    ≈ 1.44 TFLOPs / GPU / layer  (matches dense FFN cost since G·C ≈ S_local × K)",
    "Routing + dispatch + combine einsums:  ≪ 1 GFLOP, cheap.",
    "DiT-3D depth = 32 layers  →  ~64 AllToAll calls per forward step on the EP group.",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 22

out = os.path.join(os.path.dirname(__file__), "ep8_moe.excalidraw")
d.write(out)

#!/usr/bin/env python3
"""System-level view v2 — central-blob dataflow.

v1 (gen_ep8_system.py) drew the dispatched tensor and the post-AllToAll tensor
as 8 identical column-aligned boxes (one per GPU).  That visually duplicated
information that is conceptually a SINGLE per-GPU tensor.

v2 collapses those duplicates into ONE central blob and then draws explicit
"routed-by-topK" fan-out arrows from that blob to the 8 expert columns.

Layout:
    [single x_g blob]                     ─ one blob, lives on every GPU
            │ (local attention)
    [single attn-out blob]                ─ one blob
            │ (local router: softmax → topK(2) → dispatched [E,C,M])
    [single dispatched blob]              ─ INPUT of AllToAll #1, single blob
            │  ╲   ╲     │     ╱   ╱      ─ topK fan-out arrows colored per
            ▼   ▼   ▼   ▼   ▼   ▼   ▼     destination expert
    [exp e_0][e_1][e_2]…[e_7]             ─ per-column expert compute
            │   │   │   │   │   │
            ╲   ╲   ╲   │   ╱   ╱         ─ symmetric fan-in arrows
    [single combined blob]                ─ OUTPUT of AllToAll #2, single blob
            │ (local weighted combine using topK weights)
    [single out_g blob]
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from excalib import (Diagram,
                     C_DIM, C_TXT,
                     FS_DIM, FS_NAME, FS_TITLE, FS_NOTE, FS_BIG,
                     SECTION_GAP)
import random

d = Diagram()

# ── GPU grid geometry ───────────────────────────────────────────────
N_GPUS    = 8
COL_W     = 170
COL_GAP   = 30
GRID_LEFT = 200
GRID_X    = lambda g: GRID_LEFT + g * (COL_W + COL_GAP)
GRID_W    = N_GPUS * (COL_W + COL_GAP) - COL_GAP
RIGHT_EDGE = GRID_LEFT + GRID_W
CENTER_X  = GRID_LEFT + GRID_W / 2

# Central-blob geometry
BLOB_W = 720
BLOB_X = CENTER_X - BLOB_W / 2

# ── Color scheme (matches v1) ──────────────────────────────────────
C_REP   = "#1971c2"; BG_REP   = "#dbe4ff"
C_ZERO  = "#5f3dc4"; BG_ZERO  = "#e5dbff"
EXPERT_PALETTE = [
    ("#c92a2a", "#ffe3e3"),
    ("#e8590c", "#ffe8cc"),
    ("#e67700", "#fff3bf"),
    ("#5c940d", "#d8f5a2"),
    ("#087f5b", "#c3fae8"),
    ("#1864ab", "#a5d8ff"),
    ("#5f3dc4", "#d0bfff"),
    ("#a61e4d", "#fcc2d7"),
]
C_ACT   = "#0c8599"; BG_ACT   = "#c5f6fa"
C_AR    = "#2f9e44"; BG_AR    = "#d3f9d8"
C_A2A   = "#0c8599"; BG_A2A   = "#99e9f2"
C_NOOP  = "#868e96"; BG_NOOP  = "#f1f3f5"


# ── helpers ────────────────────────────────────────────────────────
def gpu_box(g, y, h, color, bg, label, sub=None, fs=FS_NOTE, fill="solid"):
    x = GRID_X(g)
    d.rect(x, y, COL_W, h, color, bg, fill)
    d.text(x + 8, y + 6, label, fs + 1, color)
    if sub:
        d.text(x + 8, y + 6 + fs + 6, sub, fs - 1, C_DIM)


def row_label(y, h, text, color=C_TXT):
    d.text(20, y + h/2 - FS_NOTE * 0.6, text, FS_NOTE + 1, color)


def central_blob(y, h, color, bg, label, sub=None, badge=None, fill="solid"):
    """A single tensor blob centered above the GPU grid."""
    d.rect(BLOB_X, y, BLOB_W, h, color, bg, fill)
    d.text(BLOB_X + 14, y + 8, label, FS_TITLE - 4, color)
    if sub:
        d.text(BLOB_X + 14, y + 8 + FS_TITLE - 2, sub, FS_NOTE + 1, C_DIM)
    if badge:
        # right-aligned hint ("lives on every GPU" etc.)
        d.text(BLOB_X + BLOB_W - 280, y + 10, badge, FS_NOTE, C_DIM)


def diag_arrow(x1, y1, x2, y2, color):
    """Append a diagonal arrow with arrowhead (excalib only ships axis-aligned)."""
    dx, dy = x2 - x1, y2 - y1
    d.elements.append({
        "id": d._uid(), "type": "arrow",
        "x": x1, "y": y1, "width": abs(dx), "height": abs(dy),
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "index": d._next_index(), "roundness": {"type": 2},
        "seed": random.randint(1, 2**31), "version": 1,
        "versionNonce": random.randint(1, 2**31),
        "isDeleted": False, "boundElements": None,
        "updated": 1773529200000, "link": None, "locked": False,
        "points": [[0, 0], [dx, dy]],
        "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "arrow",
        "elbowed": False,
    })


def band(y, h, color, bg, label, sub=None):
    d.rect(GRID_LEFT - 10, y, GRID_W + 20, h, color, bg, "hachure")
    d.text(GRID_LEFT, y + 6, label, FS_TITLE - 4, color)
    if sub:
        d.text(GRID_LEFT, y + 30, sub, FS_NOTE + 1, C_TXT)


# ━━ TITLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
y = 0
d.text(GRID_LEFT, y,
       "DiT-3D Transformer Block  ·  8 × H100  ·  EP=8 + ZeRO-2  ·  v2 central-blob view",
       FS_BIG, C_TXT)
d.text(GRID_LEFT, y + 38,
       "Per-GPU activations that share the same shape on every rank are drawn ONCE as a single "
       "central blob.  TopK routing → fan-out arrows pick which expert column receives each slice.",
       FS_NOTE + 1, C_DIM)
y += 80

# ━━ LEGEND ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LH = 26
LBX = GRID_LEFT
d.text(LBX, y, "Legend", FS_TITLE - 2, C_TXT); y += 32
items = [
    ("[R]  Replicated weight",   C_REP,  BG_REP,  "Full tensor on every GPU."),
    ("[Z]  ZeRO-2 grad shard",   C_ZERO, BG_ZERO, "Params replicated; grads + Adam states split 1/8."),
    ("[E]  EP-owned expert",     EXPERT_PALETTE[0][0], EXPERT_PALETTE[0][1],
     "1 of 8 experts lives on this GPU only.  Distinct color per GPU column."),
    ("[A]  Activation blob",     C_ACT,  BG_ACT,  "Same shape on every GPU → drawn as ONE central blob."),
    ("topK route",               C_A2A,  BG_A2A,  "Fan-out arrow colored by destination expert."),
]
for label, col, bg, sub in items:
    d.rect(LBX, y, 28, 20, col, bg, "solid")
    d.text(LBX + 38, y + 2, label, FS_NOTE + 2, col)
    d.text(LBX + 280, y + 2, sub, FS_NOTE + 1, C_DIM)
    y += LH
y += SECTION_GAP // 2

# ━━ GPU COLUMN HEADERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HDR_H = 36
for g in range(N_GPUS):
    x = GRID_X(g)
    fg, bg = EXPERT_PALETTE[g]
    d.rect(x, y, COL_W, HDR_H, fg, bg, "solid")
    d.text(x + 8, y + 6, f"GPU {g}  (rank {g})", FS_TITLE - 4, fg)
    d.text(x + 8, y + 22, f"owns expert e_{g}", FS_NOTE, C_TXT)
y += HDR_H + 6

# ━━ WEIGHT RESIDENCY (kept per-column — these genuinely differ) ━━━━━
ROW_H = 60
row_label(y, ROW_H, "Attention\nweights")
for g in range(N_GPUS):
    gpu_box(g, y, ROW_H, C_REP, BG_REP, "[R] Wq Wk Wv Wo",
            "GQA 32Q/8KV, replicated")
y += ROW_H + 4

ROW_H2 = 50
row_label(y, ROW_H2, "Norms +\nAdaLN +\nRouter wg")
for g in range(N_GPUS):
    gpu_box(g, y, ROW_H2, C_REP, BG_REP, "[R] norms + wg",
            "RMSNorm, AdaLN, wg [E,M]")
y += ROW_H2 + 4

ROW_H4 = 56
row_label(y, ROW_H4, "ZeRO-2 shard\n(non-expert)")
for g in range(N_GPUS):
    gpu_box(g, y, ROW_H4, C_ZERO, BG_ZERO, f"[Z] shard {g}/8",
            "grad + Adam m,v + fp32 master")
y += ROW_H4 + 4

ROW_H5 = 88
row_label(y, ROW_H5, "Expert FFN\n(EP-sharded)")
for g in range(N_GPUS):
    fg, bg = EXPERT_PALETTE[g]
    gpu_box(g, y, ROW_H5, fg, bg, f"[E] expert e_{g}",
            "Wgu_j [2·EI, M]\nWd_j [M, EI]\n~3.4 B params  ·  ~6.7 GB bf16",
            fill="cross-hatch")
y += ROW_H5 + SECTION_GAP // 2

# ━━ FORWARD-PASS DATAFLOW (central-blob design) ━━━━━━━━━━━━━━━━━━━━
d.text(GRID_LEFT, y,
       "Forward pass  ·  central blob = 'this tensor exists on every GPU with the same shape'",
       FS_TITLE, C_TXT)
y += 40

# ── 1. Single x_g input blob ────────────────────────────────────────
BLOB_H = 50
central_blob(y, BLOB_H, C_ACT, BG_ACT,
             "[A] x_g   input tokens",
             "[S_local = 4096, M = 4096]   bf16",
             badge="(one per GPU, shape identical)",
             fill="hachure")
y += BLOB_H + 6
d.arrow_v(CENTER_X, y, y + 24, C_TXT)
d.text(CENTER_X + 14, y + 4, "local attention (no comm)", FS_NOTE, C_DIM)
y += 30

# ── 2. Single attention-output blob ─────────────────────────────────
central_blob(y, BLOB_H, C_ACT, BG_ACT,
             "[A] h_g   post-attention",
             "[S_local = 4096, M = 4096]   bf16",
             badge="(one per GPU, shape identical)",
             fill="hachure")
y += BLOB_H + 6
d.arrow_v(CENTER_X, y, y + 24, C_TXT)
d.text(CENTER_X + 14, y + 4, "local router  softmax → topK(2)  →  dispatched [E,C,M]", FS_NOTE, C_DIM)
y += 30

# ── 3. SINGLE dispatched blob (input of AllToAll #1) ────────────────
DISP_H = 70
central_blob(y, DISP_H, C_ACT, BG_ACT,
             "[A] dispatched   (input of AllToAll #1)",
             "[E = 8, C = 1024, M = 4096]   bf16   ·   slice [j, :, :] is destined for expert e_j",
             badge="ONE blob per GPU  ·  not 8 copies",
             fill="solid")
disp_bottom_y = y + DISP_H
# 8 colored anchor ticks inside the blob — one per expert slot
slot_w = (BLOB_W - 24) / N_GPUS
for j in range(N_GPUS):
    fg, bg = EXPERT_PALETTE[j]
    sx = BLOB_X + 12 + j * slot_w
    d.rect(sx, disp_bottom_y - 22, slot_w - 4, 18, fg, bg, "cross-hatch")
    d.text(sx + 4, disp_bottom_y - 20, f"slot j={j}", FS_DIM, fg)
y = disp_bottom_y + 8

# ── 4. AllToAll #1 banner + topK fan-out arrows ─────────────────────
A2A_LABEL_H = 26
d.text(GRID_LEFT, y,
       "AllToAll #1   dist.all_to_all_single(out, dispatched, group=ep_group)   "
       "→  slice j of every GPU is delivered to GPU j (the owner of expert e_j)",
       FS_NOTE + 2, C_A2A)
y += A2A_LABEL_H

fan_src_y = disp_bottom_y - 4         # arrows leave from inside the blob's slot ticks
fan_dst_y = y + 60                     # land just above the expert-compute row
for j in range(N_GPUS):
    sx = BLOB_X + 12 + j * slot_w + (slot_w - 4) / 2
    dx_ = GRID_X(j) + COL_W / 2
    fg, _ = EXPERT_PALETTE[j]
    diag_arrow(sx, fan_src_y, dx_, fan_dst_y, fg)
    # mid-arrow label: which expert receives this slice
    midx = (sx + dx_) / 2
    midy = (fan_src_y + fan_dst_y) / 2
    d.text(midx - 12, midy - 16, f"→ e_{j}", FS_DIM, fg)
y = fan_dst_y + 4

# ── 5. Per-column expert compute (each GPU runs its OWN expert) ─────
EXP_H = 84
row_label(y, EXP_H, "expert compute\n(local, EP-sharded)")
for g in range(N_GPUS):
    fg, bg = EXPERT_PALETTE[g]
    gpu_box(g, y, EXP_H, fg, bg,
            f"run e_{g}  on G·C tokens",
            "SwiGLU using Wgu_j, Wd_j\n[E·C = 8192, M] tokens\n→ [E·C, M] output",
            fill="cross-hatch")
y += EXP_H + 8

# ── 6. AllToAll #2 banner + symmetric fan-IN arrows to one blob ─────
d.text(GRID_LEFT, y,
       "AllToAll #2   dist.all_to_all_single(out, expert_out, group=ep_group)   "
       "→  reverse exchange: each expert's outputs return to the GPUs that contributed tokens",
       FS_NOTE + 2, C_A2A)
y += A2A_LABEL_H

fan_src_y2 = y + 4
fan_dst_y2 = y + 70
for j in range(N_GPUS):
    sx = GRID_X(j) + COL_W / 2
    dx_ = BLOB_X + 12 + j * slot_w + (slot_w - 4) / 2
    fg, _ = EXPERT_PALETTE[j]
    diag_arrow(sx, fan_src_y2, dx_, fan_dst_y2, fg)
y = fan_dst_y2

# ── 7. SINGLE combined blob (output of AllToAll #2) ─────────────────
COMB_H = 70
central_blob(y, COMB_H, C_ACT, BG_ACT,
             "[A] combined   (output of AllToAll #2)",
             "[E = 8, C = 1024, M = 4096]   bf16   ·   slice [j, :, :] holds tokens that visited e_j",
             badge="ONE blob per GPU  ·  mirror of dispatched",
             fill="solid")
# colored slot ticks at the TOP of this blob to mirror the fan-in
for j in range(N_GPUS):
    fg, bg = EXPERT_PALETTE[j]
    sx = BLOB_X + 12 + j * slot_w
    d.rect(sx, y + 4, slot_w - 4, 18, fg, bg, "cross-hatch")
    d.text(sx + 4, y + 6, f"from e_{j}", FS_DIM, fg)
y += COMB_H + 6

d.arrow_v(CENTER_X, y, y + 24, C_TXT)
d.text(CENTER_X + 14, y + 4,
       "local combine: weighted sum using top-K weights  →  out_g",
       FS_NOTE, C_DIM)
y += 30

# ── 8. Single output blob ───────────────────────────────────────────
central_blob(y, BLOB_H, C_ACT, BG_ACT,
             "[A] out_g   block output",
             "[S_local = 4096, M = 4096]   bf16",
             badge="(one per GPU, shape identical)",
             fill="hachure")
y += BLOB_H + SECTION_GAP // 2

# ━━ NOTES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(GRID_LEFT, y,
       "Why the central-blob view is more honest than v1's 8-column dataflow:",
       FS_TITLE - 2, C_TXT); y += 32
for line in [
    "•  The dispatched [E,C,M] tensor IS ONE per-GPU tensor — it just has shape [8, C, M]; v1's 8 boxes were 8 copies of the SAME object.",
    "•  TopK routing is the algorithmic reason a slice ends up in slot j; the colored fan-out arrows make 'token → expert e_j' visible.",
    "•  AllToAll #1 then physically MOVES slot j of GPU i to GPU j (no compute, just a rearrangement of bytes).",
    "•  After expert compute, AllToAll #2 mirrors the move — slot j of the combined blob on GPU i holds whatever expert e_j on GPU j produced for GPU i's tokens.",
    "•  Local combine (weighted sum with top-K scores) collapses the 8 slots back into one [S_local, M] activation = out_g.",
]:
    d.text(GRID_LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 22

out = os.path.join(os.path.dirname(__file__), "ep8_system_v2.excalidraw")
d.write(out)

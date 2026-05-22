#!/usr/bin/env python3
"""System-level view: per-GPU residency for one DiT-3D transformer block on 8×H100.

Goal: make WHO HOLDS WHAT immediately legible.  Three sharding regimes coexist
in a single block:

    [R]  Replicated      — every GPU stores the full tensor      (attn weights, router wg, norms)
    [Z]  ZeRO-2 sharded  — params replicated, grads + opt state sharded across DP=8
    [E]  EP-sharded      — one expert per GPU, never replicated   (Wgu_j, Wd_j)

For each GPU column we draw the residency stack (top: weights, bottom: activations
flowing through the block) and overlay the cross-GPU collectives as horizontal
bands of arrows.  This complements gen_ep8_moe.py (which zooms into ONE GPU's
tensor shapes) by showing the FULL CLUSTER STATE.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from excalib import (Diagram,
                     C_Q, C_KV, C_SCR, C_ATTN, C_OUT, C_IDX,
                     C_DIM, C_TXT, C_OP, BG_Q, BG_KV, BG_SCR, BG_ATTN,
                     BG_OUT, BG_IDX, BG_MASK,
                     FS_DIM, FS_NAME, FS_TITLE, FS_NOTE, FS_BIG,
                     LGAP, SECTION_GAP, FILL_IN, FILL_OUT)

d = Diagram()

# ── GPU grid geometry ───────────────────────────────────────────────
N_GPUS    = 8
COL_W     = 170          # per-GPU column width
COL_GAP   = 30
GRID_LEFT = 200          # left margin (room for row labels)
GRID_X    = lambda g: GRID_LEFT + g * (COL_W + COL_GAP)
GRID_W    = N_GPUS * (COL_W + COL_GAP) - COL_GAP
RIGHT_EDGE = GRID_LEFT + GRID_W

# ── Color scheme (3 sharding regimes + activations) ────────────────
# Replicated weights: blue family
C_REP   = "#1971c2"; BG_REP   = "#dbe4ff"
# ZeRO-sharded slice (always different shade per GPU to convey "different bytes")
C_ZERO  = "#5f3dc4"; BG_ZERO  = "#e5dbff"
# EP-owned expert: each GPU gets its own hue (8 distinct colors)
EXPERT_PALETTE = [
    ("#c92a2a", "#ffe3e3"),   # red
    ("#e8590c", "#ffe8cc"),   # orange
    ("#e67700", "#fff3bf"),   # amber
    ("#5c940d", "#d8f5a2"),   # lime
    ("#087f5b", "#c3fae8"),   # teal
    ("#1864ab", "#a5d8ff"),   # blue
    ("#5f3dc4", "#d0bfff"),   # violet
    ("#a61e4d", "#fcc2d7"),   # pink
]
# Activations
C_ACT   = "#0c8599"; BG_ACT   = "#c5f6fa"
# Comm bands
C_AR    = "#2f9e44"; BG_AR    = "#d3f9d8"   # AllReduce / ReduceScatter
C_A2A   = "#0c8599"; BG_A2A   = "#99e9f2"   # AllToAll
C_NOOP  = "#868e96"; BG_NOOP  = "#f1f3f5"   # no-op band


def gpu_box(g, y, h, color, bg, label, sub=None, fs=FS_NOTE, fill="solid"):
    """Draw a box spanning one GPU column with a label and optional sub-text."""
    x = GRID_X(g)
    d.rect(x, y, COL_W, h, color, bg, fill)
    d.text(x + 8, y + 6, label, fs + 1, color)
    if sub:
        d.text(x + 8, y + 6 + fs + 6, sub, fs - 1, C_DIM)


def row_label(y, h, text, color=C_TXT):
    """Left-margin label for a horizontal row."""
    d.text(20, y + h/2 - FS_NOTE * 0.6, text, FS_NOTE + 1, color)


def band(y, h, color, bg, label, sub=None):
    """Horizontal band spanning the entire 8-GPU grid (used for comm ops)."""
    d.rect(GRID_LEFT - 10, y, GRID_W + 20, h, color, bg, "hachure")
    d.text(GRID_LEFT, y + 6, label, FS_TITLE - 4, color)
    if sub:
        d.text(GRID_LEFT, y + 30, sub, FS_NOTE + 1, C_TXT)


# ━━ TITLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
y = 0
d.text(GRID_LEFT, y,
       "DiT-3D Transformer Block  ·  8 × H100  ·  EP=8 + ZeRO-2 residency map",
       FS_BIG, C_TXT)
d.text(GRID_LEFT, y + 38,
       "One column per GPU.  Vertical stack = what lives in HBM.  "
       "Horizontal bands = collectives that touch all 8 GPUs at the same logical step.",
       FS_NOTE + 1, C_DIM)
y += 80

# ━━ LEGEND ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LH = 26
LBX = GRID_LEFT
d.text(LBX, y, "Sharding regimes", FS_TITLE - 2, C_TXT); y += 32
items = [
    ("[R]  Replicated",        C_REP,  BG_REP,  "Full tensor on every GPU. Forward uses local copy."),
    ("[Z]  ZeRO-2 grad shard", C_ZERO, BG_ZERO, "Params replicated; gradients + Adam states split 1/8 across DP."),
    ("[E]  EP-owned expert",   C_PURP := "#c92a2a", "#ffe3e3",
     "1 of 8 expert FFNs lives on this GPU only. Distinct color per GPU."),
    ("[A]  Activations",       C_ACT,  BG_ACT,  "Per-GPU tensors flowing through the block."),
    ("[C]  Collective band",   C_A2A,  BG_A2A,  "AllToAll / AllReduce / ReduceScatter spanning all 8 GPUs."),
]
for label, col, bg, sub in items:
    d.rect(LBX, y, 28, 20, col, bg, "solid")
    d.text(LBX + 38, y + 2, label, FS_NOTE + 2, col)
    d.text(LBX + 250, y + 2, sub, FS_NOTE + 1, C_DIM)
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

# ━━ ROW 1: Replicated attention weights ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROW_H = 78
row_label(y, ROW_H, "Attention\nweights")
for g in range(N_GPUS):
    gpu_box(g, y, ROW_H, C_REP, BG_REP,
            "[R] Wq Wk Wv Wo",
            "GQA 32Q/8KV  ·  ~50M params\nidentical bytes on every GPU")
y += ROW_H + 4

# ━━ ROW 2: AdaLN + RMSNorm + LayerScale ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROW_H2 = 56
row_label(y, ROW_H2, "Norms +\nAdaLN")
for g in range(N_GPUS):
    gpu_box(g, y, ROW_H2, C_REP, BG_REP,
            "[R] norms",
            "RMSNorm × 3, AdaLN linear\nreplicated, ~6M params")
y += ROW_H2 + 4

# ━━ ROW 3: Router wg ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROW_H3 = 50
row_label(y, ROW_H3, "Router wg")
for g in range(N_GPUS):
    gpu_box(g, y, ROW_H3, C_REP, BG_REP,
            "[R] wg [E,M]",
            "[8, 4096]  routing decisions LOCAL")
y += ROW_H3 + 4

# ━━ ROW 4: ZeRO-2 grad/opt-state shard for non-expert params ━━━━━━━
ROW_H4 = 70
row_label(y, ROW_H4, "ZeRO-2 shard\n(non-expert)")
for g in range(N_GPUS):
    gpu_box(g, y, ROW_H4, C_ZERO, BG_ZERO,
            f"[Z] shard {g}/8",
            "grad + Adam m,v + fp32 master\nfor 1/8 of attn+norm+wg params")
y += ROW_H4 + 4

# ━━ ROW 5: EP-owned expert weights ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROW_H5 = 110
row_label(y, ROW_H5, "Expert FFN\n(EP-sharded)")
for g in range(N_GPUS):
    fg, bg = EXPERT_PALETTE[g]
    gpu_box(g, y, ROW_H5, fg, bg,
            f"[E] expert e_{g}",
            "Wgu_j [2·EI, M]  +  Wd_j [M, EI]\n~3.4 B params  ·  ~6.7 GB bf16\n+ ~40 GB Adam states (NOT sharded\nat ep_size=world_size)",
            fill="cross-hatch")
y += ROW_H5 + SECTION_GAP // 2

# ━━ FORWARD-PASS DATAFLOW SECTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(GRID_LEFT, y,
       "Forward pass through one block  (top → bottom on every GPU column simultaneously)",
       FS_TITLE, C_TXT)
y += 40

# ── Activation: x_g (input) ────────────────────────────────────────
ACT_H = 44
row_label(y, ACT_H, "x_g  (input)")
for g in range(N_GPUS):
    gpu_box(g, y, ACT_H, C_ACT, BG_ACT,
            f"[A] x_{g}",
            "[S_local=4096, M=4096]  bf16",
            fill="hachure")
y += ACT_H + 6

# vertical down-arrows from each GPU column
for g in range(N_GPUS):
    cx = GRID_X(g) + COL_W / 2
    d.arrow_v(cx, y, y + 24, C_TXT)
y += 30

# ── Local attention (no comm) ──────────────────────────────────────
row_label(y, ACT_H, "attention")
for g in range(N_GPUS):
    gpu_box(g, y, ACT_H, C_ACT, BG_ACT,
            "GQA self+cross",
            "Wq/Wk/Wv/Wo (replicated)\nLOCAL, no comm")
y += ACT_H + 6

# down arrows
for g in range(N_GPUS):
    cx = GRID_X(g) + COL_W / 2
    d.arrow_v(cx, y, y + 24, C_TXT)
y += 30

# ── Local routing ──────────────────────────────────────────────────
row_label(y, ACT_H, "route + dispatch")
for g in range(N_GPUS):
    gpu_box(g, y, ACT_H, C_ACT, BG_ACT,
            "softmax → topK(2)",
            "→ dispatch [E=8, C, M]  LOCAL")
y += ACT_H + 6

# ── AllToAll #1 band ───────────────────────────────────────────────
A2A_H = 100
band(y, A2A_H, C_A2A, BG_A2A,
     "AllToAll #1   dist.all_to_all_single(out, dispatched, group=ep_group)",
     "GPU i sends its slice for expert j to GPU j   ·   8×8 = 64 transfers   ·   ~64 MiB / direction / GPU")

# crossing arrows: each GPU column → each GPU column at the bottom edge
src_y = y + 56
dst_y = y + A2A_H - 4
for i in range(N_GPUS):
    sx = GRID_X(i) + COL_W / 2
    for j in range(N_GPUS):
        dx_ = GRID_X(j) + COL_W / 2
        col = "#e03131" if i == j else C_A2A
        d.line(sx, src_y, [[0, 0], [dx_ - sx, dst_y - src_y]], col, "solid")
y += A2A_H + 6

# ── Per-GPU expert compute (each GPU runs ITS OWN expert) ──────────
EXP_H = 84
row_label(y, EXP_H, "expert compute")
for g in range(N_GPUS):
    fg, bg = EXPERT_PALETTE[g]
    gpu_box(g, y, EXP_H, fg, bg,
            f"run e_{g}  on G·C tokens",
            "SwiGLU using Wgu_j, Wd_j\n[8192, M] → [8192, 2·EI]\n→ SiLU·⊙ → [8192, M]",
            fill="cross-hatch")
y += EXP_H + 6

# ── AllToAll #2 band (mirror) ──────────────────────────────────────
band(y, A2A_H, C_A2A, BG_A2A,
     "AllToAll #2   dist.all_to_all_single(out, expert_out, group=ep_group)",
     "Reverse exchange: GPU j returns expert-j outputs to originating GPUs   ·   shape → [E=8, C, M] on every GPU")

src_y = y + 56
dst_y = y + A2A_H - 4
for i in range(N_GPUS):
    sx = GRID_X(i) + COL_W / 2
    for j in range(N_GPUS):
        dx_ = GRID_X(j) + COL_W / 2
        col = "#e03131" if i == j else C_A2A
        d.line(sx, src_y, [[0, 0], [dx_ - sx, dst_y - src_y]], col, "solid")
y += A2A_H + 6

# ── Combine + output ───────────────────────────────────────────────
for g in range(N_GPUS):
    cx = GRID_X(g) + COL_W / 2
    d.arrow_v(cx, y, y + 24, C_TXT)
y += 30

row_label(y, ACT_H, "combine → out_g")
for g in range(N_GPUS):
    gpu_box(g, y, ACT_H, C_ACT, BG_ACT,
            f"[A] out_{g}",
            "[S_local=4096, M=4096]  bf16",
            fill="cross-hatch")
y += ACT_H + SECTION_GAP // 2

# ━━ BACKWARD-PASS COMM SECTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(GRID_LEFT, y,
       "Backward pass — gradient communication (3 distinct paths)",
       FS_TITLE, C_TXT)
y += 40

# Path 1: ZeRO-2 ReduceScatter for non-expert grads
band(y, 70, C_AR, BG_AR,
     "Non-expert grads  →  ReduceScatter on full DP group (size 8)",
     "Wq/Wk/Wv/Wo, norms, router wg → grad split 1/8 → each GPU updates its ZeRO-2 shard.\n"
     "Volume per GPU ≈ (full non-expert grad bytes) / 8.")
y += 80

# Path 2: Expert grads — no comm at full EP
band(y, 70, C_NOOP, BG_NOOP,
     "Expert grads  →  AllReduce on expert_dp_process_group  (size 1 at ep_size = world_size)  →  NO-OP",
     "Each GPU's expert grad stays local. Optimizer state for that expert also stays local (NOT sharded).\n"
     "→ memory hotspot: ~40 GB Adam state per GPU for one 3.4 B-param expert.")
y += 80

# Path 3: Activation grads through AllToAll (in reverse)
band(y, 70, C_A2A, BG_A2A,
     "Activation grads through MoE  →  symmetric AllToAlls in reverse (autograd of _AllToAll.apply)",
     "Same 64 MiB × 2 directions × 2 calls = ~256 MiB / GPU / layer of comm during backward,\n"
     "× 32 layers ≈ 8 GiB / GPU / step just for MoE activation grads.")
y += 80 + SECTION_GAP // 2

# ━━ SUMMARY TABLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
d.text(GRID_LEFT, y, "Per-GPU memory budget (1 transformer block)",
       FS_TITLE, C_TXT)
y += 40
rows = [
    ("Tensor",                "Regime", "Per-GPU bytes (bf16)", "Notes"),
    ("Attention Wq/Wk/Wv/Wo", "[R]",    "~100 MB",              "GQA 32Q/8KV, replicated"),
    ("RMSNorm + AdaLN",       "[R]",    "~12 MB",               "tiny, replicated"),
    ("Router wg",             "[R]",    "<1 MB",                "[E=8, M=4096]"),
    ("Expert FFN (1 of 8)",   "[E]",    "~6.7 GB",              "Wgu_j [2·EI, M] + Wd_j [M, EI]"),
    ("ZeRO-2 grads + Adam",   "[Z]",    "~3 GB",                "for non-expert params, 1/8 shard"),
    ("Expert Adam states",    "(local)","~40 GB",              "NOT sharded at ep_size=world_size"),
    ("Activations (recomp)",  "[A]",    "~5–10 GB",            "depends on checkpointing"),
]
COL_X = [GRID_LEFT, GRID_LEFT + 280, GRID_LEFT + 380, GRID_LEFT + 600]
for r, row in enumerate(rows):
    is_hdr = (r == 0)
    fs = FS_NOTE + (2 if is_hdr else 1)
    col = C_TXT if is_hdr else C_DIM
    for c, cell in enumerate(row):
        d.text(COL_X[c], y, cell, fs, col)
    y += 24

y += SECTION_GAP // 2
d.text(GRID_LEFT, y,
       "Mitigations if expert Adam state (~40 GB) won't fit:",
       FS_TITLE - 2, C_TXT); y += 30
for line in [
    "•  Lower ep_size to 4 → 2 experts/GPU but expert_dp_group has size 2 → ZeRO can shard expert opt-state 2-way.",
    "•  cpu_offloading: true in YAML → DeepSpeedCPUAdam moves Adam m, v to host RAM (slower step, fits).",
    "•  Switch to bf16 Adam moments (saves 2/3 of optimizer bytes).",
    "•  ZeRO-3 across the full world (shards expert params too, but expensive AllGather every layer).",
]:
    d.text(GRID_LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 22

out = os.path.join(os.path.dirname(__file__), "ep8_system.excalidraw")
d.write(out)

"""
Visualize KV-split workload distribution across 8 requests and 8 splits of 256 tokens.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

seq_lens = [288, 4, 1884, 21, 136, 2048, 42, 335]

N_REQUESTS  = len(seq_lens)
N_SPLITS    = 8
SPLIT_SIZE  = 256          # tokens per split
TOTAL_TOKENS = N_SPLITS * SPLIT_SIZE  # 2048

# ── colours ──────────────────────────────────────────────────────────────────
BG_COLOR     = "#ffffff"
GRID_COLOR   = "#333333"
FILL_COLOR   = "#2c7bb6"   # filled (active) portion
EMPTY_COLOR  = "#f5f5f5"   # empty cell fill

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

CELL_W = 1.0
CELL_H = 1.0

for row, seq_len in enumerate(seq_lens):
    # request 0 at the top → invert row index for y
    y = (N_REQUESTS - 1 - row) * CELL_H

    for col in range(N_SPLITS):
        x = col * CELL_W

        token_start = col * SPLIT_SIZE
        token_end   = (col + 1) * SPLIT_SIZE

        # fraction of this split that is occupied
        tokens_here  = max(0, min(seq_len, token_end) - token_start)
        fill_frac    = tokens_here / SPLIT_SIZE

        # empty cell background
        ax.add_patch(mpatches.Rectangle(
            (x, y), CELL_W, CELL_H,
            linewidth=0,
            facecolor=EMPTY_COLOR,
            zorder=1,
        ))

        # filled rectangle (left-aligned inside the cell)
        if fill_frac > 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y), CELL_W * fill_frac, CELL_H,
                boxstyle="square,pad=0",
                linewidth=0,
                facecolor=FILL_COLOR,
                zorder=2,
            ))

        # cell border (drawn on top)
        ax.add_patch(mpatches.Rectangle(
            (x, y), CELL_W, CELL_H,
            linewidth=1.2,
            edgecolor=GRID_COLOR,
            facecolor="none",
            zorder=3,
        ))

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0, N_SPLITS  * CELL_W)
ax.set_ylim(0, N_REQUESTS * CELL_H)

# x-axis: block labels at the top
ax.set_xticks([i + 0.5 for i in range(N_SPLITS)])
ax.set_xticklabels([f"block {i}" for i in range(N_SPLITS)],
                   color="black", fontsize=11)
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.tick_params(axis="x", top=True, bottom=False,
               labeltop=True, labelbottom=False, length=0)

# y-axis: request labels
ax.set_yticks([(N_REQUESTS - 1 - i) + 0.5 for i in range(N_REQUESTS)])
ax.set_yticklabels(
    [f"request {i}  (seq={seq_lens[i]})" for i in range(N_REQUESTS)],
    color="black", fontsize=10,
)
ax.tick_params(axis="y", left=True, right=False,
               labelleft=True, labelright=False, length=0)

# outer border
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COLOR)
    spine.set_linewidth(1.5)

# ── optional: annotate token count inside each filled cell ───────────────────
for row, seq_len in enumerate(seq_lens):
    y = (N_REQUESTS - 1 - row) * CELL_H
    for col in range(N_SPLITS):
        token_start = col * SPLIT_SIZE
        token_end   = (col + 1) * SPLIT_SIZE
        tokens_here = max(0, min(seq_len, token_end) - token_start)
        if tokens_here == 0:
            continue
        fill_frac = tokens_here / SPLIT_SIZE
        # place text centered in the filled area
        ax.text(
            col * CELL_W + fill_frac * CELL_W / 2,
            y + CELL_H / 2,
            str(tokens_here),
            ha="center", va="center",
            color="white", fontsize=8, fontweight="bold",
            zorder=4,
        )
        # label empty remainder if partial fill
        remainder = SPLIT_SIZE - tokens_here
        if 0 < fill_frac < 1 and remainder > 0:
            ax.text(
                col * CELL_W + fill_frac * CELL_W + (1 - fill_frac) * CELL_W / 2,
                y + CELL_H / 2,
                "",
                ha="center", va="center",
                color="#999999", fontsize=7,
                zorder=4,
            )

plt.title("KV-split workload distribution  (split size = 256 tokens)",
          color="black", fontsize=13, pad=14)
plt.tight_layout()

out_path = "kv_split_workload.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved → {out_path}")

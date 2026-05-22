"""
Visualize XOR Swizzle Load Balancing for KV-split.

split_idx_new = XOR(split_idx_old, request_idx)

Each cell (row=request, col=sequence_range) shows which block handles it
after the XOR remapping.  Cell colour encodes the split type:
  - OOB      (sequence doesn't reach that range)  → red,    label "OOB"
  - Full     (entire 256-token range is covered)  → green,  label = block idx
  - Partial  (partially covered)                  → yellow, label = block idx
The entire tile is filled with the colour (no partial-fill bar).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

seq_lens = [288, 4, 1884, 21, 136, 2048, 42, 335]

N_REQUESTS = len(seq_lens)
N_SPLITS   = 8
SPLIT_SIZE = 256          # tokens per split

# ── colours ───────────────────────────────────────────────────────────────────
BG_COLOR      = "#ffffff"
GRID_COLOR    = "#333333"
OOB_COLOR     = "#d73027"   # red
FULL_COLOR    = "#1a9850"   # green
PARTIAL_COLOR = "#fec44f"   # amber-yellow (good contrast with dark text)

fig, ax = plt.subplots(figsize=(16, 5.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

CELL_W = 1.0
CELL_H = 1.0

for row, seq_len in enumerate(seq_lens):
    # request 0 at top → invert row for y position
    y = (N_REQUESTS - 1 - row) * CELL_H

    for col in range(N_SPLITS):
        x = col * CELL_W

        token_start = col * SPLIT_SIZE
        token_end   = (col + 1) * SPLIT_SIZE
        tokens_here = max(0, min(seq_len, token_end) - token_start)

        # ── classify cell ─────────────────────────────────────────────────────
        if tokens_here == 0:
            cell_type = "oob"
        elif tokens_here == SPLIT_SIZE:
            cell_type = "full"
        else:
            cell_type = "partial"

        # block assigned to this cell after XOR swizzle
        block_idx = col ^ row   # split_idx_new = split_idx_old XOR request_idx

        # ── choose colour / label ─────────────────────────────────────────────
        if cell_type == "oob":
            facecolor  = OOB_COLOR
            label      = "OOB"
            text_color = "white"
            fontsize   = 9
        elif cell_type == "full":
            facecolor  = FULL_COLOR
            label      = f"block {block_idx}"
            text_color = "white"
            fontsize   = 10
        else:  # partial
            facecolor  = PARTIAL_COLOR
            label      = f"block {block_idx}"
            text_color = "#222222"
            fontsize   = 10

        # ── draw fully-filled tile ────────────────────────────────────────────
        ax.add_patch(mpatches.Rectangle(
            (x, y), CELL_W, CELL_H,
            linewidth=1.4,
            edgecolor=GRID_COLOR,
            facecolor=facecolor,
            zorder=2,
        ))

        # ── centred label ─────────────────────────────────────────────────────
        ax.text(
            x + CELL_W / 2,
            y + CELL_H / 2,
            label,
            ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold",
            zorder=3,
        )

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0, N_SPLITS  * CELL_W)
ax.set_ylim(0, N_REQUESTS * CELL_H)

# x-axis: sequence-length ranges (shown at top)
ax.set_xticks([i + 0.5 for i in range(N_SPLITS)])
ax.set_xticklabels(
    [f"{i * SPLIT_SIZE}–{(i + 1) * SPLIT_SIZE - 1}" for i in range(N_SPLITS)],
    color="black", fontsize=10,
)
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.tick_params(axis="x", top=True, bottom=False,
               labeltop=True, labelbottom=False, length=0)
ax.set_xlabel("Sequence Length  (token index range per split)",
              color="black", fontsize=11, labelpad=10)

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

# ── legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=FULL_COLOR,    edgecolor=GRID_COLOR,
                   label="Full split"),
    mpatches.Patch(facecolor=PARTIAL_COLOR, edgecolor=GRID_COLOR,
                   label="Partial split"),
    mpatches.Patch(facecolor=OOB_COLOR,     edgecolor=GRID_COLOR,
                   label="Out of Bounds (OOB)"),
]
ax.legend(handles=legend_patches, loc="upper left",
          bbox_to_anchor=(1.01, 1), borderaxespad=0,
          fontsize=9, framealpha=0.95, edgecolor=GRID_COLOR)

# ── title ─────────────────────────────────────────────────────────────────────
plt.title(
    "XOR Swizzle Load Balancing\n"
    r"$\mathtt{split\_idx\_new = XOR(split\_idx\_old,\ request\_idx)}$",
    color="black", fontsize=13, pad=22,
)

plt.tight_layout()

out_path = "xor_swizzle_load_balancing.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved → {out_path}")

"""
gen_swizzle_table_sA.py — Generate swizzle tables for sA fp8 [128×128] S<3,4,3>.

Outputs (in images/):
  1. swizzle_table_sA_fp8.png            — cell value = swizzled byte offset
  2. swizzle_table_sA_fp8_mat_indices.png — cell value = swizzled column index

Usage:
    python src/kernels/gen_swizzle_table_sA.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images")

# ── Swizzle S<3,4,3> ─────────────────────────────────────────────────────────
BITS, BASE, SHIFT = 3, 4, 3
MASK = ((1 << BITS) - 1) << BASE  # 0x70


def swizzle(addr):
    return addr ^ ((addr >> SHIFT) & MASK)


# 8 pastel colors — one per 16-byte chunk (bits [6:4] of swizzled address)
CHUNK_COLORS = [
    '#ff9999',  # 0: light red
    '#99ff99',  # 1: light green
    '#9999ff',  # 2: light blue
    '#ffff99',  # 3: light yellow
    '#ff99ff',  # 4: light magenta
    '#99ffff',  # 5: light cyan
    '#ffcc99',  # 6: light orange
    '#cc99ff',  # 7: light purple
]

EXPLANATION = (
    "Swizzle S<3,4,3>:  addr_swz = addr ⊕ ((addr >> 3) & 0x70)\n"
    "\n"
    "XOR source bits [9:7] (= row % 8) onto target bits [6:4] (= 16B chunk within 128B cache line).\n"
    "Each of the 8 XOR groups (row 0–7) gets a unique permutation of the 8 chunks,\n"
    "so the same column in different rows maps to different SMEM banks → zero bank conflicts.\n"
    "\n"
    "Cache line = 128 bytes = 8 chunks of 16 bytes = 32 banks of 4 bytes.\n"
    "For fp8 (1 byte/element): stride = 128 bytes/row = 1 cache line/row.\n"
    "\n"
    "Pattern repeats every 8 rows (row % 8 determines the XOR group).\n"
    "Within each row, each 16B chunk is colored consistently.\n"
    "Chunks are reordered per-row so that column-wise reads (same col, different rows)\n"
    "access different banks — this is the key benefit of the swizzle."
)


def _draw_table(row_ranges, nc, stride, cell_texts_groups, row_labels_groups,
                title, out_name, legend_label):
    """
    Draw a colored swizzle table with rectangular cells (height = 2 × width).

    row_ranges       — list of (start, end) tuples, e.g. [(0,8), (64,72)]
    cell_texts_groups — list of 2D lists, one per row range
    row_labels_groups — list of lists of row label strings, one per row range
    Colors are always by 16-byte chunk of the swizzled address.
    """
    cell_w = 1.0
    cell_h = 2.0  # height = 2 × width
    gap = 2.0     # vertical gap between row groups

    # Determine max text length to pick appropriate font size
    max_len = max(len(t) for texts in cell_texts_groups
                  for row in texts for t in row)
    cell_fontsize = 24 if max_len <= 3 else 22

    # Precompute chunk colors for each group
    chunk_groups = []
    for (rstart, rend) in row_ranges:
        actual_rows = np.arange(rstart, rend)
        cols = np.arange(nc)
        linear = actual_rows[:, None] * stride + cols[None, :]
        swizzled = np.vectorize(swizzle)(linear)
        chunk = (swizzled >> BASE) & ((1 << BITS) - 1)
        chunk_groups.append(chunk)

    # Layout metrics
    n_groups = len(row_ranges)
    total_rows = sum(rend - rstart for rstart, rend in row_ranges)
    table_w = nc * cell_w
    table_h = total_rows * cell_h + (n_groups - 1) * gap
    legend_w = 7.0
    explain_h = 8.0

    fig_w = table_w + legend_w + 3.0
    fig_h = table_h + explain_h + 4.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-2.0, table_w + legend_w + 2.0)
    ax.set_ylim(table_h + explain_h + 2.0, -3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(table_w / 2, -2.5, title,
            ha='center', va='center', fontsize=16, fontweight='bold')

    # Column headers
    for c in range(nc):
        ax.text(c * cell_w + cell_w / 2, -0.5, str(c),
                ha='center', va='center', fontsize=20, color='#555555',
                fontweight='bold')

    # Draw each row group
    y_offset = 0.0
    for gi, ((rstart, rend), texts, labels, chunk) in enumerate(
            zip(row_ranges, cell_texts_groups, row_labels_groups, chunk_groups)):
        nr = rend - rstart

        # Group separator label (for 2nd group onward)
        if gi > 0:
            sep_y = y_offset - gap / 2
            ax.text(table_w / 2, sep_y, "⋮", ha='center', va='center',
                    fontsize=18, color='#999999')

        for r in range(nr):
            # Row label (actual row index)
            ax.text(-0.6, y_offset + r * cell_h + cell_h / 2, labels[r],
                    ha='right', va='center', fontsize=22, fontweight='bold',
                    color='#555555')
            for c in range(nc):
                ch = chunk[r, c]
                x = c * cell_w
                y = y_offset + r * cell_h
                rect = Rectangle((x, y), cell_w, cell_h,
                                  facecolor=CHUNK_COLORS[ch],
                                  edgecolor='#888888', linewidth=0.5)
                ax.add_patch(rect)
                ax.text(x + cell_w / 2, y + cell_h / 2,
                        texts[r][c],
                        ha='center', va='center',
                        fontsize=cell_fontsize, fontweight='bold')

        y_offset += nr * cell_h + gap

    # Legend (right side)
    leg_x = table_w + 2.0
    ax.text(leg_x + 1.0, 0.5, legend_label, fontsize=13, fontweight='bold',
            ha='left', va='center')
    for i, col in enumerate(CHUNK_COLORS):
        y_leg = 2.5 + i * cell_h
        rect = Rectangle((leg_x, y_leg), cell_w, cell_h * 0.8,
                          facecolor=col, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(leg_x + cell_w + 0.6, y_leg + cell_h * 0.4,
                f"chunk {i}  (bytes {i*16}–{i*16+15})",
                ha='left', va='center', fontsize=11)

    # Explanation below the table
    exp_y = table_h + 1.5
    ax.text(0, exp_y, EXPLANATION,
            fontsize=11, family='monospace', va='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f0f0',
                      edgecolor='#cccccc'))

    out = os.path.join(OUT_DIR, out_name)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved → {out}")
    plt.close()


def make_byte_offset_table(M, K, stride, label, out_name):
    """Table where cell value = swizzled byte offset (rows 0:7 only)."""
    row_ranges = [(0, min(8, M))]
    all_texts = []
    all_labels = []
    for (rstart, rend) in row_ranges:
        actual_rows = np.arange(rstart, rend)
        cols = np.arange(K)
        linear = actual_rows[:, None] * stride + cols[None, :]
        swizzled = np.vectorize(swizzle)(linear)
        nr = rend - rstart
        texts = [[str(swizzled[r, c]) for c in range(K)] for r in range(nr)]
        labels = [str(rstart + r) for r in range(nr)]
        all_texts.append(texts)
        all_labels.append(labels)
    _draw_table(row_ranges, K, stride, all_texts, all_labels,
                f"{label}  —  Swizzled byte offsets",
                out_name,
                "Color = 16B chunk\n(bits [6:4])")


def make_matrix_index_table(M, K, stride, label, out_name):
    """Table where cell value = swizzled column index.

    Shows rows 0:7 and 64:71 to demonstrate the repeating pattern.
    Given linear (r, c) → byte_addr = r*stride + c  (fp8: 1 byte/elem)
    Swizzled addr → swizzled_col = swizzled_addr % stride
    """
    row_ranges = [(0, min(8, M)), (64, min(72, M))]
    all_texts = []
    all_labels = []
    for (rstart, rend) in row_ranges:
        actual_rows = np.arange(rstart, rend)
        cols = np.arange(K)
        linear = actual_rows[:, None] * stride + cols[None, :]
        swizzled = np.vectorize(swizzle)(linear)
        swz_col = swizzled % stride
        nr = rend - rstart
        texts = [[str(swz_col[r, c]) for c in range(K)] for r in range(nr)]
        labels = [str(rstart + r) for r in range(nr)]
        all_texts.append(texts)
        all_labels.append(labels)
    _draw_table(row_ranges, K, stride, all_texts, all_labels,
                f"{label}  —  Swizzled column indices",
                out_name,
                "Color = 16B chunk\n(bits [6:4])")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    make_byte_offset_table(
        128, 128, stride=128,
        label="sA fp8 [128×128]  S<3,4,3>",
        out_name="swizzle_table_sA_fp8.png",
    )
    make_matrix_index_table(
        128, 128, stride=128,
        label="sA fp8 [128×128]  S<3,4,3>",
        out_name="swizzle_table_sA_fp8_mat_indices.png",
    )

    print("\nDone!")

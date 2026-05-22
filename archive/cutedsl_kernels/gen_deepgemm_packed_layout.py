"""
gen_deepgemm_packed_layout.py — Visualize deepGEMM packed q_f8_scale_f32 layout.

Flat 1D buffer, 8448 bytes total:
  Bytes 0–8191:    fp8 Q tile [64 rows × 128 cols], row-major, flattened
  Bytes 8192–8447: f32 per-row scales [64 scales × 4 bytes each]

Shown as a 1D horizontal strip with byte-offset labels and ellipsis in the middle.

Output: images/deepgemm_packed_q_f8_scale_f32.png

Usage:
    python src/kernels/gen_deepgemm_packed_layout.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images")

# Colors
ELLIPSIS_BG   = '#f0f0f0'   # light gray
ELLIPSIS_FG   = '#999999'
BOUNDARY_CLR  = '#e74c3c'   # red for region boundaries
SCALE_COLOR   = '#f9e79f'   # light gold — f32 scale bytes

# High-contrast cycling palette (same as swizzle table)
ROW_PALETTE = [
    '#ff9999',  # light red
    '#99ff99',  # light green
    '#9999ff',  # light blue
    '#ffff99',  # light yellow
    '#ff99ff',  # light magenta
    '#99ffff',  # light cyan
    '#ffcc99',  # light orange
    '#cc99ff',  # light purple
]


def _row_color(row_idx):
    """High-contrast color for a row by cycling through 8 distinct hues."""
    return ROW_PALETTE[row_idx % len(ROW_PALETTE)]


def main(out_name="deepgemm_packed_q_f8_scale_f32.png"):
    """Draw a 1D horizontal byte-strip of the packed buffer."""

    cell_w = 1.0
    cell_h = 2.0

    # ── Build display cells: (byte_offset_or_None, color, text) ──────
    cells = []

    # --- fp8 region: show row 0 start, row 0 end, row 1 start,
    #     ellipsis, row 63 end  ---

    # Row 0: first 5 bytes
    for c in range(5):
        b = 0 * 128 + c
        cells.append((b, _row_color(0), f"r0c{c}"))

    cells.append((None, ELLIPSIS_BG, "⋯"))  # ... middle of row 0

    # Row 0: last 3 bytes
    for c in [125, 126, 127]:
        b = 0 * 128 + c
        cells.append((b, _row_color(0), f"r0c{c}"))

    # Row 1: first 4 bytes
    for c in range(4):
        b = 1 * 128 + c
        cells.append((b, _row_color(1), f"r1c{c}"))

    cells.append((None, ELLIPSIS_BG, "⋯"))  # ... middle of row 1

    # Row 1: last 3 bytes
    for c in [125, 126, 127]:
        b = 1 * 128 + c
        cells.append((b, _row_color(1), f"r1c{c}"))

    # Row 2: first 3 bytes
    for c in range(3):
        b = 2 * 128 + c
        cells.append((b, _row_color(2), f"r2c{c}"))

    cells.append((None, ELLIPSIS_BG, "⋯"))  # ... rest of row 2

    # Big ellipsis for rows 3..62
    cells.append((None, ELLIPSIS_BG, "⋯\nrows\n3–62\n⋯"))

    # Row 63: first 3 bytes
    for c in range(3):
        b = 63 * 128 + c
        cells.append((b, _row_color(63), f"r63c{c}"))

    cells.append((None, ELLIPSIS_BG, "⋯"))

    # Row 63: last 3 bytes
    for c in [125, 126, 127]:
        b = 63 * 128 + c
        cells.append((b, _row_color(63), f"r63c{c}"))

    # --- Boundary index ---
    boundary_after = len(cells)

    # --- Scale region: just s[0] and s[63] with ellipsis ---
    # s[0] (4 bytes) — same color as row 0
    for i in range(4):
        cells.append((8192 + i, _row_color(0), f"r0[{i}]"))

    cells.append((None, ELLIPSIS_BG, "⋯"))

    # s[63] (4 bytes) — same color as row 63
    for i in range(4):
        cells.append((8444 + i, _row_color(63), f"r63[{i}]"))

    n_cells = len(cells)

    # ── Layout ────────────────────────────────────────────────────────
    strip_w = n_cells * cell_w
    explain_h = 8.0

    fig_w = strip_w + 4.0
    fig_h = cell_h + explain_h + 14.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-1.5, strip_w + 2.0)
    ax.set_ylim(cell_h + explain_h + 10.0, -7.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(strip_w / 2, -6.0,
            "deepGEMM packed q_f8_scale_f32  —  1D Byte Layout",
            ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(strip_w / 2, -4.2,
            "8448 bytes = 8192 B fp8 Q [64×128]  +  256 B f32 scales [64]",
            ha='center', va='center', fontsize=14, color='#444444')

    # ── "1 byte" width arrow above first cell ─────────────────────────
    arrow_y = -1.2
    ax.annotate('', xy=(0, arrow_y), xytext=(cell_w, arrow_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(cell_w / 2, arrow_y - 0.4, "1 byte",
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ── Draw cells ────────────────────────────────────────────────────
    for i, (byte_off, color, text) in enumerate(cells):
        x = i * cell_w
        y = 0

        rect = Rectangle((x, y), cell_w, cell_h,
                          facecolor=color, edgecolor='#888888',
                          linewidth=0.5)
        ax.add_patch(rect)

        is_ell = byte_off is None
        # Multi-line ellipsis (the big gap) gets smaller font
        if '\n' in text:
            fs, fc, fw = 9, ELLIPSIS_FG, 'normal'
        elif is_ell:
            fs, fc, fw = 16, ELLIPSIS_FG, 'normal'
        else:
            fs, fc, fw = 10, 'black', 'bold'

        ax.text(x + cell_w / 2, y + cell_h / 2, text,
                ha='center', va='center',
                fontsize=fs, fontweight=fw, color=fc)

        # Byte offset label below
        if byte_off is not None:
            ax.text(x + cell_w / 2, cell_h + 0.6, str(byte_off),
                    ha='center', va='top', fontsize=8,
                    fontweight='bold', color='#555555', rotation=55)

    # ── Red boundary line at fp8 → scale transition ───────────────────
    bx = boundary_after * cell_w
    ax.plot([bx, bx], [-0.5, cell_h + 0.3], color=BOUNDARY_CLR,
            linewidth=3, linestyle='-', zorder=5)
    ax.text(bx, cell_h + 0.8, "byte 8192\n(fp8 → scales)",
            ha='center', va='top', fontsize=10, fontweight='bold',
            color=BOUNDARY_CLR)

    # ── Region brackets / labels above ────────────────────────────────
    bracket_y = -2.5
    # fp8 region
    fp8_end_x = boundary_after * cell_w
    mid_fp8 = fp8_end_x / 2
    ax.annotate('', xy=(0, bracket_y), xytext=(fp8_end_x, bracket_y),
                arrowprops=dict(arrowstyle='<->', color='#2471a3', lw=2.5))
    ax.text(mid_fp8, bracket_y - 0.3, "fp8 Q data  (8192 bytes, 64 rows × 128 cols)",
            ha='center', va='top', fontsize=12, fontweight='bold',
            color='#2471a3')

    # scale region
    sc_start_x = boundary_after * cell_w
    sc_end_x = n_cells * cell_w
    mid_sc = (sc_start_x + sc_end_x) / 2
    ax.annotate('', xy=(sc_start_x, bracket_y), xytext=(sc_end_x, bracket_y),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=2.5))
    ax.text(mid_sc, bracket_y - 0.3, "f32 scales (256 bytes, 64 rows × 4 cols)",
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='#555555')

    # ── Legend ─────────────────────────────────────────────────────────
    leg_y = cell_h + 4.5
    # Row color samples
    ax.text(0, leg_y - 0.5, "Legend — color = row index (cycles every 8 rows, same for fp8 data and its scale)",
            fontsize=13, fontweight='bold')

    sample_rows = [0, 1, 2, 3, 4, 5, 6, 7]
    for j, r in enumerate(sample_rows):
        rx = j * 2.5
        rect = Rectangle((rx, leg_y + 1.0), cell_w, cell_h * 0.6,
                          facecolor=_row_color(r), edgecolor='black',
                          linewidth=0.8)
        ax.add_patch(rect)
        ax.text(rx + cell_w / 2, leg_y + 1.0 + cell_h * 0.3,
                f"r{r}", ha='center', va='center', fontsize=10,
                fontweight='bold')
    ax.text(len(sample_rows) * 2.5 + 0.3, leg_y + 1.0 + cell_h * 0.3,
            "← fp8 bytes and f32 scale share the same row color",
            ha='left', va='center', fontsize=11)

    # ── Explanation below ─────────────────────────────────────────────
    exp_y = cell_h + 8.5
    explanation = (
        "deepGEMM packs Q fp8 data and per-row f32 scales into one contiguous 1D buffer.\n"
        "\n"
        "Bytes 0–8191:     fp8 Q tile [64×128], row-major, flattened.\n"
        "                  Byte N → row = N // 128,  col = N % 128.\n"
        "                  Same-row bytes share a color (rainbow by row).\n"
        "Bytes 8192–8447:  64 f32 scales (4 bytes each), one per row of Q.\n"
        "                  Scale for row r at byte offset 8192 + r × 4.\n"
        "\n"
        "Cell text:  rRcC  = fp8 element at (row R, col C)\n"
        "            rN[b] = byte b of f32 scale for row N"
    )
    ax.text(0, exp_y, explanation,
            fontsize=11, family='monospace', va='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f0f0',
                      edgecolor='#cccccc'))

    out = os.path.join(OUT_DIR, out_name)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    main()

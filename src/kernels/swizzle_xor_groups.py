"""
swizzle_xor_groups.py — CuTe-viz style swizzle tables.

Generates two tables for sA fp8 [128×128] with S<3,4,3>:
  1. swizzle_table_sA_fp8.png          — cell value = swizzled byte offset
  2. swizzle_table_sA_fp8_mat_indices.png — cell value = (row, col) matrix index

Usage:
    python src/kernels/swizzle_xor_groups.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images")

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
    "XOR source bits [9:7] (= row % 8) onto target bits [6:4] (= 16B chunk within 128B cache line).\n"
    "Each of the 8 XOR groups (row 0–7) gets a unique permutation of chunks,\n"
    "so the same column in different rows maps to different banks → zero bank conflicts.\n"
    "\n"
    "Cache line = 128 bytes = 8 chunks of 16 bytes = 32 banks of 4 bytes.\n"
    "For fp8 (1 byte/element): stride = 128 bytes/row = 1 cache line/row."
)


def _draw_table(nr, nc, stride, cell_texts, title, out_name, legend_label):
    """
    Draw a colored swizzle table.

    cell_texts[r][c] — string to display in cell (r, c)
    Colors are always by 16-byte chunk of the swizzled address.
    """
    rows = np.arange(nr)
    cols = np.arange(nc)
    linear = rows[:, None] * stride + cols[None, :]
    swizzled = np.vectorize(swizzle)(linear)
    chunk = (swizzled >> BASE) & ((1 << BITS) - 1)

    cell_w = 1.0
    cell_h = 2.0  # height = 2 × width

    # Layout metrics
    table_w = nc * cell_w
    table_h = nr * cell_h
    legend_w = 6.0
    explain_h = 5.0

    fig_w = table_w + legend_w + 2.0
    fig_h = table_h + explain_h + 3.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-1.5, table_w + legend_w + 1.0)
    ax.set_ylim(table_h + explain_h + 1.0, -2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(table_w / 2, -2.0, title,
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Column headers
    for c in range(nc):
        ax.text(c * cell_w + cell_w / 2, -0.3, str(c),
                ha='center', va='center', fontsize=9, color='#555555',
                fontstyle='italic')

    # Row headers + cells
    for r in range(nr):
        # Row label
        ax.text(-0.4, r * cell_h + cell_h / 2, str(r),
                ha='right', va='center', fontsize=10, fontstyle='italic',
                color='#555555')
        for c in range(nc):
            ch = chunk[r, c]
            x = c * cell_w
            y = r * cell_h
            rect = Rectangle((x, y), cell_w, cell_h,
                              facecolor=CHUNK_COLORS[ch],
                              edgecolor='#888888', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(x + cell_w / 2, y + cell_h / 2,
                    cell_texts[r][c],
                    ha='center', va='center',
                    fontsize=9, fontweight='bold')

    # Legend (right side)
    leg_x = table_w + 1.5
    ax.text(leg_x + 1.0, 0.5, legend_label, fontsize=11, fontweight='bold',
            ha='left', va='center')
    for i, col in enumerate(CHUNK_COLORS):
        y_leg = 2.0 + i * cell_h
        rect = Rectangle((leg_x, y_leg), cell_w, cell_h * 0.8,
                          facecolor=col, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(leg_x + cell_w + 0.5, y_leg + cell_h * 0.4,
                f"chunk {i}  (bytes {i*16}–{i*16+15})",
                ha='left', va='center', fontsize=10)

    # Explanation below
    exp_y = table_h + 1.5
    ax.text(0, exp_y, EXPLANATION,
            fontsize=10, family='monospace', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                      edgecolor='#cccccc'))

    out = os.path.join(OUT_DIR, out_name)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved → {out}")
    plt.close()


def make_byte_offset_table(M, K, stride, label, out_name):
    """Table where cell value = swizzled byte offset."""
    nr = min(8, M)
    rows = np.arange(nr)
    cols = np.arange(K)
    linear = rows[:, None] * stride + cols[None, :]
    swizzled = np.vectorize(swizzle)(linear)
    texts = [[str(swizzled[r, c]) for c in range(K)] for r in range(nr)]
    _draw_table(nr, K, stride, texts,
                f"{label}  —  Swizzled byte offsets",
                out_name,
                "Color = 16B chunk\n(bits [6:4])")


def make_matrix_index_table(M, K, stride, label, out_name):
    """Table where cell value = swizzled matrix (row, col) index.

    Given linear (r, c) → byte_addr = r*stride + c (fp8: 1 byte/elem)
    Swizzled addr → swizzled_row = addr // stride, swizzled_col = addr % stride
    We show "swizzled_col" since the row doesn't change (swizzle only
    affects intra-cache-line bits for stride=128).
    """
    nr = min(8, M)
    rows = np.arange(nr)
    cols = np.arange(K)
    linear = rows[:, None] * stride + cols[None, :]
    swizzled = np.vectorize(swizzle)(linear)
    # For stride=128 and fp8, swizzled byte offset → col index directly
    swz_col = swizzled % stride
    texts = [[str(swz_col[r, c]) for c in range(K)] for r in range(nr)]
    _draw_table(nr, K, stride, texts,
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
"""
swizzle_xor_groups.py — CuTe-viz style swizzle table with matplotlib.

Shows where each (row, col) element ends up after S<3,4,3> XOR swizzle,
colored by which 16-byte chunk (bits [6:4]) the swizzled address falls in.

Usage:
    python src/kernels/swizzle_xor_groups.py

Output:
    images/swizzle_table_sA_fp8.png   — first 8 rows × 32 cols of sA [128×128]
    images/swizzle_table_sA_fp8_full.png — all 128 rows × 128 cols (overview)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images")

# ── Swizzle S<3,4,3> ─────────────────────────────────────────────────────────
BITS, BASE, SHIFT = 3, 4, 3
MASK = ((1 << BITS) - 1) << BASE  # 0x70


def swizzle(addr):
    return addr ^ ((addr >> SHIFT) & MASK)


# 8 pastel colors matching the cute-viz style (one per 16-byte chunk = bits [6:4])
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


def make_table(M, K, stride, label, out_name, show_full=False):
    """
    Draw the swizzle table.

    M, K: tile dimensions (rows, cols)
    stride: row stride in bytes (= K for fp8, = 2*K for fp16)
    """
    # Build offset arrays
    rows = np.arange(M)
    cols = np.arange(K)
    linear = rows[:, None] * stride + cols[None, :]  # byte offsets
    swizzled = np.vectorize(swizzle)(linear)

    # Chunk index = bits [6:4] of swizzled address (which 16B section in cache line)
    chunk = (swizzled >> BASE) & ((1 << BITS) - 1)  # 0..7

    if not show_full:
        # ── Detailed table: show first 8 rows, all K cols ────────────
        nr = min(8, M)
        nc = K
        cell_w, cell_h = 1.0, 1.0
        fig_w = nc * cell_w + 2.5
        fig_h = nr * cell_h + 2.5

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(-0.5, nc + 0.5)
        ax.set_ylim(nr + 0.5, -1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        fig.suptitle(
            f"{label}  —  Swizzle S<{BITS},{BASE},{SHIFT}>   "
            f"addr_swz = addr ⊕ ((addr >> {SHIFT}) & 0x{MASK:02X})\n"
            f"Cell value = swizzled byte offset   |   "
            f"Color = 16-byte chunk (bits [6:4] of swizzled addr)",
            fontsize=11, fontweight='bold', y=0.98,
        )

        # Column headers
        for c in range(nc):
            ax.text(c + 0.5, -0.7, str(c), ha='center', va='center',
                    fontsize=7, color='#666666', fontstyle='italic')

        # Row headers + cells
        for r in range(nr):
            ax.text(-0.3, r + 0.5, str(r), ha='right', va='center',
                    fontsize=8, fontstyle='italic', color='#666666')
            for c in range(nc):
                val = swizzled[r, c]
                ch = chunk[r, c]
                color = CHUNK_COLORS[ch]
                rect = Rectangle((c, r), cell_w, cell_h,
                                 facecolor=color, edgecolor='#888888',
                                 linewidth=0.5)
                ax.add_patch(rect)
                # Text: show the swizzled byte offset
                ax.text(c + 0.5, r + 0.5, str(val),
                        ha='center', va='center', fontsize=6,
                        fontweight='bold')

        # Legend
        for i, col in enumerate(CHUNK_COLORS):
            x_leg = nc + 0.8
            y_leg = i * 1.0 + 0.5
            rect = Rectangle((x_leg - 0.4, y_leg - 0.3), 0.6, 0.6,
                              facecolor=col, edgecolor='black', linewidth=0.8)
            ax.add_patch(rect)
            ax.text(x_leg + 0.5, y_leg, f"chunk {i}  (bytes {i*16}–{i*16+15})",
                    ha='left', va='center', fontsize=7)

        plt.tight_layout(rect=[0, 0, 0.95, 0.94])
        out = os.path.join(OUT_DIR, out_name)
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"Saved → {out}")
        plt.close()

    else:
        # ── Overview: full M×K tile, no cell text ────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(28, 12))
        fig.suptitle(
            f"{label}  —  S<{BITS},{BASE},{SHIFT}> full tile\n"
            f"Left: swizzled byte offset   |   Right: 16-byte chunk color",
            fontsize=14, fontweight='bold', y=0.99,
        )

        # Left: swizzled offset as heatmap
        ax = axes[0]
        im = ax.imshow(swizzled, aspect='auto', cmap='viridis',
                        interpolation='nearest')
        ax.set_title("Swizzled byte offset", fontsize=13)
        ax.set_xlabel("Column (K)"); ax.set_ylabel("Row (M)")
        # Tick every 8
        ax.set_xticks(range(0, K, 8))
        ax.set_yticks(range(0, M, 8))
        plt.colorbar(im, ax=ax, shrink=0.7, label="byte offset")

        # Right: chunk color
        ax = axes[1]
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(CHUNK_COLORS)
        norm = BoundaryNorm(np.arange(9) - 0.5, 8)
        im = ax.imshow(chunk, aspect='auto', cmap=cmap, norm=norm,
                        interpolation='nearest')
        ax.set_title("16-byte chunk (bits [6:4])", fontsize=13)
        ax.set_xlabel("Column (K)"); ax.set_ylabel("Row (M)")
        ax.set_xticks(range(0, K, 8))
        ax.set_yticks(range(0, M, 8))
        cb = plt.colorbar(im, ax=ax, shrink=0.7, ticks=range(8))
        cb.set_label("chunk #")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(OUT_DIR, out_name)
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved → {out}")
        plt.close()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # fp8: stride = K bytes (1 byte per element)
    make_table(128, 128, stride=128,
               label="sA fp8 [128×128]",
               out_name="swizzle_table_sA_fp8.png",
               show_full=False)

    make_table(128, 128, stride=128,
               label="sA fp8 [128×128]",
               out_name="swizzle_table_sA_fp8_full.png",
               show_full=True)

    make_table(64, 128, stride=128,
               label="sB fp8 [64×128]",
               out_name="swizzle_table_sB_fp8.png",
               show_full=False)

    print("\nDone!")
"""
swizzle_xor_groups.py — Matplotlib visualization of S<3,4,3> XOR groups.

S<3,4,3> swizzle: addr_swz = addr ^ ((addr >> 3) & 0x70)
  - Source bits: [9:7] of byte address → row % 8 for stride-128 layout
  - Target bits: [6:4] of byte address → which 16-byte chunk within cache line
  - 8 XOR groups (2^3), each gets a different bank remapping

Each row belongs to one of 8 XOR groups (row % 8). The swizzle ensures
that rows in different groups map the same column to different banks,
eliminating bank conflicts when threads read the same column across rows.

Usage:
    python src/kernels/swizzle_xor_groups.py

Output:
    images/swizzle_xor_groups_sA_fp8.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images")

# ── Swizzle S<3,4,3> ─────────────────────────────────────────────────────────
BITS, BASE, SHIFT = 3, 4, 3
MASK = ((1 << BITS) - 1) << BASE          # 0x70 = bits [6:4]
CACHE_LINE = 128                           # bytes
BANK_COUNT = 32
BANK_WIDTH = 4                             # bytes per bank


def swizzle(addr):
    return addr ^ ((addr >> SHIFT) & MASK)

vswizzle = np.vectorize(swizzle)


def make_figure(M, K, label):
    """Visualize S<3,4,3> XOR groups for an [M×K] fp8 tile."""
    rows = np.arange(M)[:, None]
    cols = np.arange(K)[None, :]
    linear = rows * K + cols
    swizzled = vswizzle(linear)

    # XOR group = source bits = (addr >> SHIFT) & ((1 << BITS) - 1)
    # For stride=128 (=2^7), addr bits [9:7] = row bits [2:0] = row % 8
    n_groups = 1 << BITS  # 8
    xor_group = rows % n_groups  # (M, 1) broadcast to (M, K)
    xor_group = np.broadcast_to(xor_group, (M, K))

    # Bank index: (byte_within_cacheline) // 4
    bank_lin = (linear % CACHE_LINE) // BANK_WIDTH
    bank_swz = (swizzled % CACHE_LINE) // BANK_WIDTH

    # 8 vivid colors for 8 XOR groups
    group_colors = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
    ]
    grp_cmap = ListedColormap(group_colors)
    grp_norm = BoundaryNorm(np.arange(n_groups + 1) - 0.5, n_groups)

    # ══════════════════════════════════════════════════════════════════
    # FIGURE: 3 × 2 grid
    # ══════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 2, figsize=(26, 24),
                             gridspec_kw={'height_ratios': [1, 1, 0.8]})
    fig.suptitle(
        f"{label}  —  Swizzle S<{BITS},{BASE},{SHIFT}>\n"
        f"addr_swz = addr ⊕ ((addr >> {SHIFT}) & 0x{MASK:02X})   |   "
        f"8 XOR groups (row % 8)   |   32 banks × 4B = 128B cache line",
        fontsize=16, fontweight='bold', y=0.99,
    )

    # ── (0,0): XOR group membership — full tile ──────────────────────
    ax = axes[0, 0]
    im = ax.imshow(xor_group, aspect='auto', cmap=grp_cmap, norm=grp_norm,
                   interpolation='nearest')
    ax.set_title(f"XOR group (row % 8) — full [{M}×{K}]", fontsize=13)
    ax.set_xlabel("K (column)"); ax.set_ylabel("M (row)")
    cb = plt.colorbar(im, ax=ax, shrink=0.8, ticks=range(n_groups))
    cb.set_label("XOR group")

    # ── (0,1): Swizzled bank index — full tile ───────────────────────
    ax = axes[0, 1]
    im = ax.imshow(bank_swz, aspect='auto', cmap='tab20',
                   interpolation='nearest', vmin=0, vmax=31)
    ax.set_title(f"Swizzled bank index — full [{M}×{K}]", fontsize=13)
    ax.set_xlabel("K (column)"); ax.set_ylabel("M (row)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="bank #")

    # ── (1,0): Zoomed — linear bank per (row, col) ───────────────────
    zr = min(16, M)
    zc = 32  # show first 32 columns (covers 1 XOR group's full bank range)
    ax = axes[1, 0]
    z_lin = bank_lin[:zr, :zc]
    im = ax.imshow(z_lin, aspect='equal', cmap='tab20',
                   interpolation='nearest', vmin=0, vmax=31)
    ax.set_title(f"Linear bank# (no swizzle) — first {zr}×{zc}", fontsize=13)
    ax.set_xlabel("K (column)"); ax.set_ylabel("M (row)")
    for r in range(zr):
        for c in range(zc):
            ax.text(c, r, str(z_lin[r, c]), ha='center', va='center',
                    fontsize=6, color='white')
    # Draw XOR group boundaries
    for g in range(1, zr // (n_groups if zr >= n_groups else 1)):
        ax.axhline(y=g * n_groups - 0.5, color='red', lw=2, ls='--')
    plt.colorbar(im, ax=ax, shrink=0.8, label="bank #")

    # ── (1,1): Zoomed — swizzled bank per (row, col) ─────────────────
    ax = axes[1, 1]
    z_swz = bank_swz[:zr, :zc]
    im = ax.imshow(z_swz, aspect='equal', cmap='tab20',
                   interpolation='nearest', vmin=0, vmax=31)
    ax.set_title(f"Swizzled bank# — first {zr}×{zc}", fontsize=13)
    ax.set_xlabel("K (column)"); ax.set_ylabel("M (row)")
    for r in range(zr):
        for c in range(zc):
            ax.text(c, r, str(z_swz[r, c]), ha='center', va='center',
                    fontsize=6, color='white')
    for g in range(1, zr // (n_groups if zr >= n_groups else 1)):
        ax.axhline(y=g * n_groups - 0.5, color='red', lw=2, ls='--')
    plt.colorbar(im, ax=ax, shrink=0.8, label="bank #")

    # ── (2,0): Column-0 bank histogram — linear vs swizzled ─────────
    ax = axes[2, 0]
    x = np.arange(BANK_COUNT)
    w = 0.35
    hist_lin = np.bincount(bank_lin[:, 0].astype(int), minlength=BANK_COUNT)
    hist_swz = np.bincount(bank_swz[:, 0].astype(int), minlength=BANK_COUNT)
    ax.bar(x - w/2, hist_lin, w, label='Linear', color='lightcoral', edgecolor='black')
    ax.bar(x + w/2, hist_swz, w, label='Swizzled', color='steelblue', edgecolor='black')
    ax.set_title(f"Bank usage — col 0, all {M} rows", fontsize=13)
    ax.set_xlabel("Bank #"); ax.set_ylabel("# rows → bank")
    ax.set_xticks(x)
    ax.axhline(y=M / BANK_COUNT, color='red', ls='--', lw=1.5,
               label=f'ideal = {M / BANK_COUNT:.0f}')
    ax.legend(fontsize=10)

    # ── (2,1): 8 groups × bank index for col 0 ──────────────────────
    ax = axes[2, 1]
    # Show which bank each group's rows hit at column 0
    group_bank_at_col0 = []
    for g in range(n_groups):
        # All rows in group g hit the same bank at col 0 (after swizzle)
        row = g  # representative row
        addr = row * K + 0
        swz_addr = swizzle(addr)
        bank = (swz_addr % CACHE_LINE) // BANK_WIDTH
        group_bank_at_col0.append(bank)

    bars = ax.bar(range(n_groups), group_bank_at_col0,
                  color=[group_colors[i] for i in range(n_groups)],
                  edgecolor='black', linewidth=1.5)
    ax.set_title("Swizzled bank# at col 0 — by XOR group", fontsize=13)
    ax.set_xlabel("XOR group (row % 8)"); ax.set_ylabel("Bank #")
    ax.set_xticks(range(n_groups))
    ax.set_yticks(range(0, BANK_COUNT, 4))
    for i, v in enumerate(group_bank_at_col0):
        ax.text(i, v + 0.3, str(v), ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, f"swizzle_xor_groups_{label.split()[0].lower()}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_figure(128, 128, "sA fp8 [128×128]")
    make_figure(64,  128, "sB fp8 [64×128]")
    print("\nDone!")
"""
swizzle_xor_groups.py — Matplotlib visualization of S<3,4,3> XOR groups.

Each cell is colored by its swizzled cache line index (byte_offset // 128).
This shows how the XOR swizzle redistributes rows across cache lines
to eliminate bank conflicts.

Usage:
    python src/kernels/swizzle_xor_groups.py

Output:
    images/swizzle_xor_groups_sA_fp8.png
    images/swizzle_xor_groups_sB_fp8.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images")

# ── Swizzle S<3,4,3> ─────────────────────────────────────────────────────────
BITS, BASE, SHIFT = 3, 4, 3
MASK = ((1 << BITS) - 1) << BASE          # 0x70 = bits [6:4]
CACHE_LINE = 128                           # bytes


def swizzle(addr):
    return addr ^ ((addr >> SHIFT) & MASK)

vswizzle = np.vectorize(swizzle)


def make_figure(M, K, title_prefix, out_name):
    """Build swizzle visualization for an [M × K] fp8 tile (1 byte/element)."""
    # Linear byte offsets: row-major (M, K):(K, 1)
    rows = np.arange(M)[:, None]
    cols = np.arange(K)[None, :]
    linear = rows * K + cols                   # shape (M, K)

    swizzled = vswizzle(linear)
    cache_line = swizzled // CACHE_LINE        # which 128-byte cache line

    n_lines = (M * K) // CACHE_LINE           # total cache lines
    # Build a colormap with enough distinct colors
    # Use a cyclic HSV-based palette so adjacent lines are visually distinct
    base_colors = plt.cm.hsv(np.linspace(0, 1, n_lines, endpoint=False))
    cmap = ListedColormap(base_colors)
    norm = BoundaryNorm(np.arange(n_lines + 1) - 0.5, n_lines)

    # ── Figure 1: Full tile ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    fig.suptitle(
        f"{title_prefix} — Swizzle S<{BITS},{BASE},{SHIFT}> XOR groups\n"
        f"[{M}×{K}] fp8  |  {n_lines} cache lines of {CACHE_LINE} bytes  |  "
        f"mask=0x{MASK:02X} (bits [{BASE+BITS-1}:{BASE}] ⊕ bits [{BASE+BITS+SHIFT-1}:{SHIFT}])",
        fontsize=15, fontweight='bold', y=0.98,
    )

    # (0,0) — Swizzled cache line index (full tile)
    ax = axes[0, 0]
    im = ax.imshow(cache_line, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')
    ax.set_title(f"Swizzled cache line index [{M}×{K}]", fontsize=13)
    ax.set_xlabel("K (column = byte within row)")
    ax.set_ylabel("M (row)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="cache line #")

    # (0,1) — Linear (no swizzle) cache line index
    ax = axes[0, 1]
    linear_cl = linear // CACHE_LINE
    im = ax.imshow(linear_cl, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')
    ax.set_title(f"Linear cache line index (no swizzle)", fontsize=13)
    ax.set_xlabel("K (column)")
    ax.set_ylabel("M (row)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="cache line #")

    # (1,0) — Zoomed swizzled (first 8 rows × 128 cols) with annotations
    zoom_r = min(8, M)
    ax = axes[1, 0]
    z_cl = cache_line[:zoom_r, :]
    im = ax.imshow(z_cl, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')
    ax.set_title(f"Zoom: first {zoom_r} rows — swizzled cache line #", fontsize=13)
    ax.set_xlabel("K (column)")
    ax.set_ylabel("M (row)")
    # Annotate every 8th column with cache line #
    for r in range(zoom_r):
        for c in range(0, K, 8):
            v = z_cl[r, c]
            ax.text(c + 3.5, r, str(v), ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='white' if base_colors[v % n_lines][:3].mean() < 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8, label="cache line #")

    # (1,1) — XOR delta (swizzled_cl - linear_cl) for first 8 rows
    ax = axes[1, 1]
    delta = cache_line[:zoom_r, :].astype(int) - linear_cl[:zoom_r, :].astype(int)
    vmax = max(abs(delta.min()), abs(delta.max()), 1)
    im = ax.imshow(delta, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(f"XOR delta (swizzled − linear) cache line", fontsize=13)
    ax.set_xlabel("K (column)")
    ax.set_ylabel("M (row)")
    for r in range(zoom_r):
        for c in range(0, K, 8):
            v = delta[r, c]
            ax.text(c + 3.5, r, f"{v:+d}" if v != 0 else "0",
                    ha='center', va='center', fontsize=7, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label="Δ cache line")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close()

    # ── Figure 2: Bank conflict analysis ──────────────────────────────
    BANK_COUNT = 32
    BANK_WIDTH = 4  # bytes per bank
    bank_swz = (swizzled % CACHE_LINE) // BANK_WIDTH
    bank_lin = (linear % CACHE_LINE) // BANK_WIDTH

    fig2, axes2 = plt.subplots(1, 3, figsize=(30, 8))
    fig2.suptitle(
        f"{title_prefix} — Bank conflict analysis  |  "
        f"32 banks × 4 bytes = 128B cache line",
        fontsize=14, fontweight='bold',
    )

    # Bank index heatmap (swizzled, zoom 8 rows)
    ax = axes2[0]
    zb = bank_swz[:zoom_r, :]
    bank_cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20)))
    im = ax.imshow(zb, aspect='auto', cmap=bank_cmap, interpolation='nearest')
    ax.set_title(f"Swizzled bank index (first {zoom_r} rows)")
    ax.set_xlabel("K"); ax.set_ylabel("M")
    plt.colorbar(im, ax=ax, shrink=0.8, label="bank #")

    # Column-0 bank histogram (swizzled vs linear)
    ax = axes2[1]
    hist_swz = np.bincount(bank_swz[:, 0].astype(int), minlength=BANK_COUNT)
    hist_lin = np.bincount(bank_lin[:, 0].astype(int), minlength=BANK_COUNT)
    x = np.arange(BANK_COUNT)
    w = 0.35
    ax.bar(x - w/2, hist_lin, w, label='Linear', color='lightcoral', edgecolor='black')
    ax.bar(x + w/2, hist_swz, w, label='Swizzled', color='steelblue', edgecolor='black')
    ax.set_title(f"Bank usage for column 0 (all {M} rows)")
    ax.set_xlabel("Bank #"); ax.set_ylabel("# rows hitting bank")
    ax.set_xticks(x)
    ax.axhline(y=M / BANK_COUNT, color='red', ls='--', label=f'ideal={M // BANK_COUNT}')
    ax.legend()

    # Column-63 bank histogram
    col = min(63, K - 1)
    ax = axes2[2]
    hist_swz2 = np.bincount(bank_swz[:, col].astype(int), minlength=BANK_COUNT)
    hist_lin2 = np.bincount(bank_lin[:, col].astype(int), minlength=BANK_COUNT)
    ax.bar(x - w/2, hist_lin2, w, label='Linear', color='lightcoral', edgecolor='black')
    ax.bar(x + w/2, hist_swz2, w, label='Swizzled', color='steelblue', edgecolor='black')
    ax.set_title(f"Bank usage for column {col} (all {M} rows)")
    ax.set_xlabel("Bank #"); ax.set_ylabel("# rows hitting bank")
    ax.set_xticks(x)
    ax.axhline(y=M / BANK_COUNT, color='red', ls='--', label=f'ideal={M // BANK_COUNT}')
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path2 = os.path.join(OUT_DIR, out_name.replace(".png", "_banks.png"))
    plt.savefig(out_path2, dpi=150)
    print(f"Saved → {out_path2}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_figure(128, 128, "sA fp8 [128×128]", "swizzle_xor_groups_sA_fp8.png")
    make_figure(64,  128, "sB fp8 [64×128]",  "swizzle_xor_groups_sB_fp8.png")
    print("\nDone!")

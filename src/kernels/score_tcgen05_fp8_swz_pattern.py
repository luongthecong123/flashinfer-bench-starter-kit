"""
score_tcgen05_fp8_swz_pattern.py — Visualise tcgen05 MMA layouts for fp8 sA.

Uses @cute.jit (compile-time only, no GPU needed) + cute_viz SVG rendering.

Usage:
    CUTE_DSL_ARCH=sm_100a python src/kernels/score_tcgen05_fp8_swz_pattern.py

Output SVGs in images/:
    swizzle_sA_fp8.svg          — swizzled sA [128×128] composed layout
    layout_sA_outer.svg         — outer layout (stage 0)
    mma_tiled_ABC.svg           — tiled MMA thread mapping for A, B, C
    swizzle_sB_fp8.svg          — swizzled sB [64×128]
    layout_sB_outer.svg         — outer layout sB (stage 0)
"""

import os
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cute_viz import (
    render_swizzle_layout_svg,
    render_layout_svg,
    render_tiled_mma_svg,
)

# ── Tile config (matches score_scale.py / score_tcgen05_fp8.py) ──────────────
M_TILE, N_TILE, K_TILE = 128, 64, 128
CTA_TILE_MNK = (M_TILE, N_TILE, K_TILE)
MMA_INST_MNK = (128, 64, 32)
FP8_TYPE = cutlass.Float8E4M3FN
ACC_TYPE = cutlass.Float32
NUM_STAGES = 1

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images")


@cute.jit
def visualize():
    op = tcgen05.MmaFP8Op(
        FP8_TYPE,
        ACC_TYPE,
        MMA_INST_MNK,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    a_smem = sm100_utils.make_smem_layout_a(
        tiled_mma, CTA_TILE_MNK, FP8_TYPE, NUM_STAGES,
    )
    b_smem = sm100_utils.make_smem_layout_b(
        tiled_mma, CTA_TILE_MNK, FP8_TYPE, NUM_STAGES,
    )

    print("tiled_mma:", tiled_mma)
    print("a_smem_layout:", a_smem)
    print("  outer:", a_smem.outer)
    print("  inner (swizzle):", a_smem.inner)
    print("b_smem_layout:", b_smem)
    print("  outer:", b_smem.outer)
    print("  inner (swizzle):", b_smem.inner)

    # ── Extract stage-0 outer layouts ─────────────────────────────────
    # outer has rank 4: (M_hier, N, K, Stage). Select modes [0,1,2] → rank 3.
    stage0_a = cute.select(a_smem.outer, mode=[0, 1, 2])
    stage0_b = cute.select(b_smem.outer, mode=[0, 1, 2])
    print("outer_stage0_a:", stage0_a)
    print("outer_stage0_b:", stage0_b)

    # ── Build flat rank-2 layouts for readable visualization ─────────
    # The hierarchical outer ((128,32), 4) : ((128,1), 32) flattens to
    # 4096 rows × 4 cols which is unreadable.
    # Instead, build simple (M, K) : (K, 1) row-major layouts matching
    # the actual SMEM tile dimensions.
    flat_a = cute.make_layout((M_TILE, K_TILE), stride=(K_TILE, 1))  # (128, 128):(128, 1)
    flat_b = cute.make_layout((N_TILE, K_TILE), stride=(K_TILE, 1))  # (64,  128):(128, 1)
    print("flat_a (MxK):", flat_a)
    print("flat_b (NxK):", flat_b)

    swizzled_a = cute.make_composed_layout(a_smem.inner, 0, flat_a)
    swizzled_b = cute.make_composed_layout(b_smem.inner, 0, flat_b)
    print("swizzled_a:", swizzled_a)
    print("swizzled_b:", swizzled_b)

    # ── 1. Swizzled sA [128×128] ─────────────────────────────────────
    out1 = os.path.join(OUT_DIR, "swizzle_sA_fp8.svg")
    render_swizzle_layout_svg(swizzled_a, out1)
    print(f"\nSaved → {out1}")

    # ── 2. Outer layout sA (flat 128×128) ────────────────────────────
    out2 = os.path.join(OUT_DIR, "layout_sA_outer.svg")
    render_layout_svg(flat_a, out2)
    print(f"Saved → {out2}")

    # ── 3. Tiled MMA thread mapping ──────────────────────────────────
    out3 = os.path.join(OUT_DIR, "mma_tiled_ABC.svg")
    render_tiled_mma_svg(tiled_mma, CTA_TILE_MNK, out3)
    print(f"Saved → {out3}")

    # ── 4. Swizzled sB [64×128] ──────────────────────────────────────
    out4 = os.path.join(OUT_DIR, "swizzle_sB_fp8.svg")
    render_swizzle_layout_svg(swizzled_b, out4)
    print(f"Saved → {out4}")

    # ── 5. Outer layout sB (flat 64×128) ─────────────────────────────
    out5 = os.path.join(OUT_DIR, "layout_sB_outer.svg")
    render_layout_svg(flat_b, out5)
    print(f"Saved → {out5}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    visualize()
    print("\nDone!")

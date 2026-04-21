"""Compare swizzle parameters for fp8 vs fp16 tcgen05 MMA."""
import cutlass, cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils

CTA = (128, 64, 128)
INST = (128, 64, 32)

@cute.jit
def compare_fp8():
    op = tcgen05.MmaFP8Op(
        cutlass.Float8E4M3FN, cutlass.Float32, INST,
        tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
    )
    tmma = cute.make_tiled_mma(op)
    sA = sm100_utils.make_smem_layout_a(tmma, CTA, cutlass.Float8E4M3FN, 1)
    sB = sm100_utils.make_smem_layout_b(tmma, CTA, cutlass.Float8E4M3FN, 1)
    s0A = cute.select(sA.outer, mode=[0, 1, 2])
    s0B = cute.select(sB.outer, mode=[0, 1, 2])
    print("=== fp8 (1 byte per element) ===")
    print("  sA swizzle:", sA.inner, "  outer_stage0:", s0A)
    print("  sB swizzle:", sB.inner, "  outer_stage0:", s0B)

@cute.jit
def compare_fp16():
    INST16 = (128, 64, 16)  # fp16: K_inst=16 (not 32)
    op = tcgen05.MmaF16BF16Op(
        cutlass.Float16, cutlass.Float32, INST16,
        tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
    )
    tmma = cute.make_tiled_mma(op)
    sA = sm100_utils.make_smem_layout_a(tmma, CTA, cutlass.Float16, 1)
    sB = sm100_utils.make_smem_layout_b(tmma, CTA, cutlass.Float16, 1)
    s0A = cute.select(sA.outer, mode=[0, 1, 2])
    s0B = cute.select(sB.outer, mode=[0, 1, 2])
    print("=== fp16 (2 bytes per element) ===")
    print("  sA swizzle:", sA.inner, "  outer_stage0:", s0A)
    print("  sB swizzle:", sB.inner, "  outer_stage0:", s0B)

compare_fp8()
print()
compare_fp16()

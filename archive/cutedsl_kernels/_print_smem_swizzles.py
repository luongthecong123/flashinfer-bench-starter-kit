"""_print_smem_swizzles.py — Print sA.inner swizzle for fp8 and bf16 MMA ops.

Both tiles have the same 128-byte row width, so the swizzle should be equal.
Run locally on SM100+ (B200):

  python src/kernels/_print_smem_swizzles.py
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils


@cute.jit
def print_swizzles():
    # fp8: sA = [128, 128] fp8 — 128 bytes/row
    op_fp8 = tcgen05.MmaFP8Op(
        cutlass.Float8E4M3FN, cutlass.Float32, (128, 64, 32),
        tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
    )
    tmma_fp8 = cute.make_tiled_mma(op_fp8)
    sA_fp8   = sm100_utils.make_smem_layout_a(
        tmma_fp8, (128, 64, 128), cutlass.Float8E4M3FN, 1,
    )
    print("fp8  [128,128] sA.inner:", sA_fp8.inner)

    # bf16: sA = [128, 64] bf16 — 128 bytes/row (same byte width)
    op_bf16 = tcgen05.MmaF16BF16Op(
        cutlass.BFloat16, cutlass.Float32, (128, 64, 16),
        tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
    )
    tmma_bf16 = cute.make_tiled_mma(op_bf16)
    sA_bf16   = sm100_utils.make_smem_layout_a(
        tmma_bf16, (128, 64, 64), cutlass.BFloat16, 1,
    )
    print("bf16 [128, 64] sA.inner:", sA_bf16.inner)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA device found.")
    else:
        major, minor = torch.cuda.get_device_capability()
        if major >= 10:
            print_swizzles()
        else:
            print(f"SM{major}{minor} < SM100 — Blackwell required.")

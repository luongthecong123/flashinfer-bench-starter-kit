"""
v1 — VEC_SIZE=8, 2 explicit chunks, N=512 bf16.

Register pressure goal: only one set of 3×4=12 bf16 regs live at a time.

Each thread loads 8 bf16 (one ldg.128), converts to f32, runs 8 fma_packed_f32x2,
then the chunk registers are dead before the next ldg.128 is issued.

Layout:
  zipped_divide(a0, (8,)) on (512,) → ((8,), (64,))
  Thread lane_idx owns columns:
    chunk 0 → col  lane_idx       (elements 8*lane_idx .. 8*lane_idx+7)
    chunk 1 → col  lane_idx + 32  (elements 8*(lane_idx+32) .. +7)
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

VEC_SIZE: cutlass.Constexpr = 8   # bf16 elements per ldg.128 (128b / 16b = 8)
N: cutlass.Constexpr = 512        # vector length = 8 * 64 = 8 * 32 * 2


@cute.jit
def warp_reduce_f32x2_add(
    val0: cutlass.Float32,
    val1: cutlass.Float32,
    width: cutlass.Constexpr = 32,
):
    for i in range(int(math.log2(width))):
        s0 = cute.arch.shuffle_sync_bfly(val0, offset=1 << i)
        s1 = cute.arch.shuffle_sync_bfly(val1, offset=1 << i)
        val0, val1 = cute.arch.add_packed_f32x2((val0, val1), (s0, s1))
    return val0, val1


class TestDotBF16_V1:
    @cute.kernel
    def _kernel(
        self,
        a0: cute.Tensor,
        a1: cute.Tensor,
        b:  cute.Tensor,
        output: cute.Tensor,
    ):
        lane_idx = cute.arch.lane_idx()

        # ((8,), (64,)) — 64 column-slices of 8 bf16 each
        a0_z = cute.zipped_divide(a0, (VEC_SIZE,))
        a1_z = cute.zipped_divide(a1, (VEC_SIZE,))
        b_z  = cute.zipped_divide(b,  (VEC_SIZE,))

        acc0 = cutlass.Float32(0.0)
        acc1 = cutlass.Float32(0.0)

        # ── chunk 0: columns [lane_idx] ─────────────────────────────────────
        # 3 × ldg.128 (3 × 4 regs); dead after this block
        a0_c0 = a0_z[None, lane_idx].load()
        a1_c0 = a1_z[None, lane_idx].load()
        b_c0  = b_z [None, lane_idx].load()
        for v in cutlass.range_constexpr(VEC_SIZE):
            a0_v = cutlass.Float32(a0_c0[v])
            a1_v = cutlass.Float32(a1_c0[v])
            b_v  = cutlass.Float32(b_c0[v])
            acc0, acc1 = cute.arch.fma_packed_f32x2(
                (a0_v, a1_v), (b_v, b_v), (acc0, acc1)
            )

        # ── chunk 1: columns [lane_idx + 32] ────────────────────────────────
        a0_c1 = a0_z[None, lane_idx + 32].load()
        a1_c1 = a1_z[None, lane_idx + 32].load()
        b_c1  = b_z [None, lane_idx + 32].load()
        for v in cutlass.range_constexpr(VEC_SIZE):
            a0_v = cutlass.Float32(a0_c1[v])
            a1_v = cutlass.Float32(a1_c1[v])
            b_v  = cutlass.Float32(b_c1[v])
            acc0, acc1 = cute.arch.fma_packed_f32x2(
                (a0_v, a1_v), (b_v, b_v), (acc0, acc1)
            )

        acc0, acc1 = warp_reduce_f32x2_add(acc0, acc1, width=32)

        if lane_idx == 0:
            output[0] = acc0
            output[1] = acc1

    @cute.jit
    def __call__(self, a0, a1, b, output, stream):
        self._kernel(a0, a1, b, output).launch(
            grid=[1, 1, 1], block=[32, 1, 1], stream=stream
        )


def _fake(dtype, shape):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=(0,), assumed_align=16
    )


def compile_test():
    a0     = _fake(cute.BFloat16, (N,))
    a1     = _fake(cute.BFloat16, (N,))
    b      = _fake(cute.BFloat16, (N,))
    output = _fake(cute.Float32,  (2,))
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    test = TestDotBF16_V1()
    compiled = cute.compile(test, a0, a1, b, output, stream, options="--enable-tvm-ffi")
    return test, compiled


_test, _compiled = compile_test()


def run():
    a0 = torch.arange(0,   N,   dtype=torch.bfloat16, device="cuda")
    a1 = torch.arange(N,   2*N, dtype=torch.bfloat16, device="cuda")
    b  = torch.ones(N,          dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(2,     dtype=torch.float32,  device="cuda")

    _compiled(a0, a1, b, output)
    torch.cuda.synchronize()

    c0 = output[0].item()
    c1 = output[1].item()

    # Reference in f32 (matches kernel: bf16 inputs, f32 accumulation)
    c0_ref = (a0.float() * b.float()).sum().item()
    c1_ref = (a1.float() * b.float()).sum().item()

    print("\n=== v1: VEC_SIZE=8, 2 chunks, N=512 bf16 ===")
    print(f"c0 = {c0:.1f}   ref={c0_ref:.1f}")
    print(f"c1 = {c1:.1f}   ref={c1_ref:.1f}")

    ok = (abs(c0 - c0_ref) < 1.0 and abs(c1 - c1_ref) < 1.0)
    print(f"PASS={ok}")
    return ok

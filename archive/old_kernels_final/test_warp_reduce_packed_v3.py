"""
v3 — VEC_SIZE=4, 4 explicit chunks, N=512 bf16.

VEC_SIZE=4 → ldg.64 per load (4×bf16=64b). 4 chunks of 128 elements each.
Per chunk: 3 × 2 regs bf16 live → freed after the 4-iter FMA loop.

Layout:
  zipped_divide(a0, (4,)) on (512,) → ((4,), (128,))
  Thread lane_idx owns columns: lane_idx + 32*k  for k in 0..3
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

VEC_SIZE: cutlass.Constexpr = 4   # bf16 elements per ldg.64 (64b / 16b = 4)
N: cutlass.Constexpr = 512        # vector length = 4 * 128 = 4 * 32 * 4


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


class TestDotBF16_V3:
    @cute.kernel
    def _kernel(
        self,
        a0: cute.Tensor,
        a1: cute.Tensor,
        b:  cute.Tensor,
        output: cute.Tensor,
    ):
        lane_idx = cute.arch.lane_idx()

        # ((4,), (128,)) — 128 column-slices of 4 bf16 each
        a0_z = cute.zipped_divide(a0, (VEC_SIZE,))
        a1_z = cute.zipped_divide(a1, (VEC_SIZE,))
        b_z  = cute.zipped_divide(b,  (VEC_SIZE,))

        acc0 = cutlass.Float32(0.0)
        acc1 = cutlass.Float32(0.0)

        # 4 chunks; each loads 3 × ldg.64 = 3 × 2 regs bf16, then freed
        for chunk in cutlass.range_constexpr(4):
            col = lane_idx + 32 * chunk
            a0_c = a0_z[None, col].load()
            a1_c = a1_z[None, col].load()
            b_c  = b_z [None, col].load()
            for v in cutlass.range_constexpr(VEC_SIZE):
                a0_v = cutlass.Float32(a0_c[v])
                a1_v = cutlass.Float32(a1_c[v])
                b_v  = cutlass.Float32(b_c[v])
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
    test = TestDotBF16_V3()
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

    c0_ref = (a0.float() * b.float()).sum().item()
    c1_ref = (a1.float() * b.float()).sum().item()

    print("\n=== v3: VEC_SIZE=4, 4 chunks, N=512 bf16 ===")
    print(f"c0 = {c0:.1f}   ref={c0_ref:.1f}")
    print(f"c1 = {c1:.1f}   ref={c1_ref:.1f}")

    ok = (abs(c0 - c0_ref) < 1.0 and abs(c1 - c1_ref) < 1.0)
    print(f"PASS={ok}")
    return ok

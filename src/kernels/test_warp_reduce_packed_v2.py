"""
v2 — VEC_SIZE=16, 1 chunk, N=512 bf16.

Register pressure comparison: all 3×8=24 bf16 regs live simultaneously,
and potentially 3×16=48 f32 regs if PTXAS bulk-converts before the FMA chain.

Layout:
  zipped_divide(a0, (16,)) on (512,) → ((16,), (32,))
  Thread lane_idx owns column lane_idx (elements 16*lane_idx .. 16*lane_idx+15)
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

VEC_SIZE: cutlass.Constexpr = 16  # bf16 elements per thread (2 × ldg.128 = 256b)
N: cutlass.Constexpr = 512        # vector length = 16 * 32


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


class TestDotBF16_V2:
    @cute.kernel
    def _kernel(
        self,
        a0: cute.Tensor,
        a1: cute.Tensor,
        b:  cute.Tensor,
        output: cute.Tensor,
    ):
        lane_idx = cute.arch.lane_idx()

        # ((16,), (32,)) — each thread owns a 16-element slice
        a0_z = cute.zipped_divide(a0, (VEC_SIZE,))
        a1_z = cute.zipped_divide(a1, (VEC_SIZE,))
        b_z  = cute.zipped_divide(b,  (VEC_SIZE,))

        # 3 × 8 regs bf16; potentially 3 × 16 regs f32 if bulk-converted
        a0_local = a0_z[None, lane_idx].load()
        a1_local = a1_z[None, lane_idx].load()
        b_local  = b_z [None, lane_idx].load()

        acc0 = cutlass.Float32(0.0)
        acc1 = cutlass.Float32(0.0)
        for v in cutlass.range_constexpr(VEC_SIZE):
            a0_v = cutlass.Float32(a0_local[v])
            a1_v = cutlass.Float32(a1_local[v])
            b_v  = cutlass.Float32(b_local[v])
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
    test = TestDotBF16_V2()
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

    print("\n=== v2: VEC_SIZE=16, 1 chunk, N=512 bf16 ===")
    print(f"c0 = {c0:.1f}   ref={c0_ref:.1f}")
    print(f"c1 = {c1:.1f}   ref={c1_ref:.1f}")

    ok = (abs(c0 - c0_ref) < 1.0 and abs(c1 - c1_ref) < 1.0)
    print(f"PASS={ok}")
    return ok

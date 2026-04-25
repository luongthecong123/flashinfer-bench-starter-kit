"""
v4 — VEC_SIZE=8, 1 chunk, N=512 bf16, SSA-style reduce.

Element-wise multiply is done as a tensor operation (tensorSSA),
then .reduce(ReductionOp.ADD, init_val=0, reduction_profile=0) collapses
the 8-element fragment to a single f32 — no manual accumulator init needed.
Warp butterfly reduce follows. Compiler handles packed instructions.

Layout:
  zipped_divide(a0, (8,)) on (512,) → ((8,), (64,))
  2 outer iterations: group = it*32 + lane_idx  for it in 0..1
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

VEC_SIZE: cutlass.Constexpr = 8   # bf16 elements per ldg.128
N: cutlass.Constexpr = 512        # vector length = 8 * 64
ITERS: cutlass.Constexpr = 2      # N / (VEC_SIZE * 32) = 512 / 256 = 2


@cute.jit
def warp_reduce_add_f32(val: cutlass.Float32, width: cutlass.Constexpr = 32) -> cutlass.Float32:
    for i in range(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


class TestDotBF16_V4:
    @cute.kernel
    def _kernel(
        self,
        a0: cute.Tensor,
        a1: cute.Tensor,
        b:  cute.Tensor,
        output: cute.Tensor,
    ):
        lane_idx = cute.arch.lane_idx()

        # ((8,), (64,)) — 64 groups of 8 bf16
        a0_z = cute.zipped_divide(a0, (VEC_SIZE,))
        a1_z = cute.zipped_divide(a1, (VEC_SIZE,))
        b_z  = cute.zipped_divide(b,  (VEC_SIZE,))

        sp0 = cutlass.Float32(0.0)
        sp1 = cutlass.Float32(0.0)

        for it in cutlass.range_constexpr(ITERS):
            group = it * 32 + lane_idx   # column index into the ((8,),(64,)) layout

            a0_frag = a0_z[None, group].load()
            a1_frag = a1_z[None, group].load()
            b_frag  = b_z [None, group].load()

            # cast to f32 before multiply so reduction is in f32
            sp0 = sp0 + cutlass.Float32(
                (a0_frag.to(cutlass.Float32) * b_frag.to(cutlass.Float32)).reduce(
                    cute.ReductionOp.ADD, init_val=float(0), reduction_profile=0
                )
            )
            sp1 = sp1 + cutlass.Float32(
                (a1_frag.to(cutlass.Float32) * b_frag.to(cutlass.Float32)).reduce(
                    cute.ReductionOp.ADD, init_val=float(0), reduction_profile=0
                )
            )

        c0 = warp_reduce_add_f32(sp0, width=32)
        c1 = warp_reduce_add_f32(sp1, width=32)

        if lane_idx == 0:
            output[0] = c0
            output[1] = c1

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
    test = TestDotBF16_V4()
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

    print("\n=== v4: VEC_SIZE=8, SSA reduce, N=512 bf16 ===")
    print(f"c0 = {c0:.1f}   ref={c0_ref:.1f}")
    print(f"c1 = {c1:.1f}   ref={c1_ref:.1f}")

    ok = (abs(c0 - c0_ref) < 1.0 and abs(c1 - c1_ref) < 1.0)
    print(f"PASS={ok}")
    return ok

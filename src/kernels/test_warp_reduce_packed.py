"""
Generic dot-product kernel: c0 = dot(a0, b), c1 = dot(a1, b) for N=512 bf16.

Tune by changing VEC_SIZE:
  VEC_SIZE=8  → ITERS=2  (like v1: 2 explicit chunks, ldg.128 bf16)
  VEC_SIZE=16 → ITERS=1  (like v2: 1 chunk,  wider load)
  VEC_SIZE=4  → ITERS=4  (like v3: 4 chunks, narrower load)

Layout:
  zipped_divide(tensor, (VEC_SIZE,)) on (512,) → ((VEC_SIZE,), (512//VEC_SIZE,))
  Each thread handles ITERS columns: col = it*32 + lane_idx
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math
import torch

# ── tuneable knob ─────────────────────────────────────────────────────────────
VEC_SIZE: cutlass.Constexpr = 8   # 4 → v3-like, 8 → v1-like, 16 → v2-like
N: cutlass.Constexpr = 512
ITERS: cutlass.Constexpr = N // (VEC_SIZE * 32)  # chunks per thread


# ── warp_reduce helpers ───────────────────────────────────────────────────────

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


# ── kernel + wrapper ──────────────────────────────────────────────────────────

class TestZippedDot:
    @cute.kernel
    def _kernel(
        self,
        a0: cute.Tensor,
        a1: cute.Tensor,
        b:  cute.Tensor,
        output: cute.Tensor,
    ):
        lane_idx = cute.arch.lane_idx()

        a0_z = cute.zipped_divide(a0, (VEC_SIZE,))
        a1_z = cute.zipped_divide(a1, (VEC_SIZE,))
        b_z  = cute.zipped_divide(b,  (VEC_SIZE,))

        acc0 = cutlass.Float32(0.0)
        acc1 = cutlass.Float32(0.0)

        for it in cutlass.range_constexpr(ITERS):
            col = it * 32 + lane_idx
            a0_frag = a0_z[None, col].load()
            a1_frag = a1_z[None, col].load()
            b_frag  = b_z [None, col].load()
            for v in cutlass.range_constexpr(VEC_SIZE):
                a0_v = cutlass.Float32(a0_frag[v])
                a1_v = cutlass.Float32(a1_frag[v])
                b_v  = cutlass.Float32(b_frag[v])
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


# ── compile ───────────────────────────────────────────────────────────────────

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
    test = TestZippedDot()
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

    print(f"\n=== VEC_SIZE={int(VEC_SIZE)}, ITERS={int(ITERS)}, N={int(N)} bf16 ===")
    print(f"c0 = {c0:.1f}   ref={c0_ref:.1f}")
    print(f"c1 = {c1:.1f}   ref={c1_ref:.1f}")

    ok = (abs(c0 - c0_ref) < 1.0 and abs(c1 - c1_ref) < 1.0)
    print(f"PASS={ok}")
    return ok

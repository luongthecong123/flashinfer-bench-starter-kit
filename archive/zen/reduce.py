import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cute.testing import benchmark, JitArguments

from typing import Tuple
import math
import torch


def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class Reduce():
    """Reduce kernel: combine split-K partial attention results via logsumexp rescaling.

    Grid: [T, head_dim, 1]   — one CTA per (token, head).
    Each CTA reduces num_splits partial results for a single head.
    256 threads cooperatively handle the dv=512 dimension.
    """
    def __init__(self):
        self.num_threads = 256
        self.warp_size = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        partial_O: cute.Tensor,    # [T, num_splits, 16, 512] float32
        partial_lse: cute.Tensor,       # [T, num_splits, 16] float32
        output: cute.Tensor,            # [T, 16, 512] bf16
        lse: cute.Tensor,               # [T, 16] float32
        stream,                         # CUDA stream
    ):
        T, num_splits, head_dim, dv = partial_O.shape
        self.kernel(partial_O, partial_lse, output, lse).launch(
            grid=[T, head_dim, 1], block=[self.num_threads, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        partial_O: cute.Tensor,    # [T, num_splits, 16, 512] float32
        partial_lse: cute.Tensor,       # [T, num_splits, 16] float32
        output: cute.Tensor,            # [T, 16, 512] bf16
        lse: cute.Tensor,               # [T, 16] float32
    ):
        T, num_splits, head_dim, dv = partial_O.shape
        batch_idx, h, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # ── Global LSE across splits (direct sum, no max trick) ──
        sum_exp = cutlass.Float32(0.0)
        for s in range(num_splits):
            sum_exp += cute.math.exp(partial_lse[batch_idx, s, h])
        global_lse = cute.math.log(sum_exp)

        # ── Weighted combination of partial outputs ──
        for d in range(tidx, dv, self.num_threads):
            acc = cutlass.Float32(0.0)
            for s in range(num_splits):
                scale = cute.math.exp(partial_lse[batch_idx, s, h] - global_lse)
                acc += partial_O[batch_idx, s, h, d] * scale
            output[batch_idx, h, d] = cutlass.BFloat16(acc)

        # ── Store LSE (convert to log base 2) ──
        if tidx == 0:
            lse[batch_idx, h] = global_lse / cutlass.Float32(0.6931471805599453)


# ── Compilation ────────────────────────────────────────────────────────────────

def fake_wrapper(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=assumed_align)


def compile_gather():
    T = cute.sym_int()
    num_splits = cute.sym_int(divisibility=2)
    head_dim, dv = 16, 512

    partial_O = fake_wrapper(cute.Float32, (T, num_splits, head_dim, dv), (3, 2, 1, 0), 16)
    partial_lse = fake_wrapper(cute.Float32, (T, num_splits, head_dim), (2, 1, 0), 16)
    output = fake_wrapper(cute.BFloat16, (T, head_dim, dv), (2, 1, 0), 16)
    lse = fake_wrapper(cute.Float32, (T, head_dim), (1, 0), 16)
    
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    reduce = Reduce()
    return cute.compile(
        reduce,
        partial_O, partial_lse, output, lse, stream,
        options="--enable-tvm-ffi"
    )


gather_compiled = compile_gather()
"""bench_warp: class-based kernel_warp with parameterised N.

Operation: scores = q @ K.T
  q      : [D]    bf16
  K      : [N, D] bf16
  scores : [N]    fp32

Usage:
    kern = BenchWarp(N=2048)
    kern.run(q, K, scores)
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math


@cute.jit
def _warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class BenchWarp:
    def __init__(self, N: int = 64, D: int = 512):
        self.N = N
        self.D = D
        self.BLOCK_SIZE = 1024
        self.NUM_WARPS = self.BLOCK_SIZE // 32
        self.K_PER_LANE = self.D // 32
        self._compiled = None

    @cute.jit
    def __call__(self, q: cute.Tensor, K: cute.Tensor, scores: cute.Tensor, stream):
        self.kernel(q, K, scores).launch(
            grid=[1, 1, 1],
            block=[self.BLOCK_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, q: cute.Tensor, K: cute.Tensor, scores: cute.Tensor):
        d          : cutlass.Constexpr = self.D
        n          : cutlass.Constexpr = self.N
        num_warps  : cutlass.Constexpr = self.NUM_WARPS
        k_per_lane : cutlass.Constexpr = self.K_PER_LANE
        num_threads: cutlass.Constexpr = self.BLOCK_SIZE

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.warp_idx()
        warp_idx   = cute.arch.make_warp_uniform(warp_idx)
        lane_idx   = cute.arch.lane_idx()
        wsize      = cute.arch.WARP_SIZE

        allocator   = cutlass.utils.SmemAllocator()
        smem_q      = allocator.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((d,), stride=(1,)), 16, None)
        smem_scores = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None)

        for i in range(tidx, d, num_threads):
            smem_q[i] = q[i]
        cute.arch.sync_threads()

        num_rounds: cutlass.Constexpr = (n + num_warps - 1) // num_warps

        for round_idx in range(num_rounds):
            token_idx = round_idx * num_warps + warp_idx
            sum_partial = cutlass.Float32(0)
            for k in range(k_per_lane):
                dim         = k * wsize + lane_idx
                q_n         = cutlass.Float32(smem_q[dim])
                kv          = cutlass.Float32(K[token_idx, dim])
                sum_partial = sum_partial + q_n * kv

            s = _warp_reduce(sum_partial, lambda a, b: a + b, width=32)
            if lane_idx == 0:
                smem_scores[token_idx] = s

        cute.arch.sync_threads()

        for i in range(tidx, n, num_threads):
            scores[i] = smem_scores[i]

    def compile(self):
        q      = make_fake_compact_tensor(dtype=cute.BFloat16, shape=(self.D,),          stride_order=(0,),   assumed_align=16)
        K      = make_fake_compact_tensor(dtype=cute.BFloat16, shape=(self.N, self.D),   stride_order=(1, 0), assumed_align=16)
        scores = make_fake_compact_tensor(dtype=cute.Float32,  shape=(self.N,),          stride_order=(0,),   assumed_align=16)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        self._compiled = cute.compile(self, q, K, scores, stream, options="--enable-tvm-ffi")

    def run(self, q, K, scores):
        if self._compiled is None:
            self.compile()
        self._compiled(q, K, scores)

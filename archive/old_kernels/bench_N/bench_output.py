"""bench_output: class-based kernel_output with parameterised N.

Operation: output = scores @ V
  scores : [N]    fp32
  V      : [N, D] fp32
  output : [D]    fp32
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math


class BenchOutput:
    def __init__(self, N: int = 64, D: int = 512):
        self.N = N
        self.D = D
        self.BLOCK_SIZE = 1024
        self.NUM_WARPS = self.BLOCK_SIZE // 32
        self.DIMS_PER_LANE = self.D // 32
        self.NUM_ROUNDS = (self.N + self.NUM_WARPS - 1) // self.NUM_WARPS
        self._compiled = None

    @cute.jit
    def __call__(self, scores: cute.Tensor, V: cute.Tensor, output: cute.Tensor, stream):
        self.kernel(scores, V, output).launch(
            grid=[1, 1, 1],
            block=[self.BLOCK_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, scores: cute.Tensor, V: cute.Tensor, output: cute.Tensor):
        n             : cutlass.Constexpr = self.N
        d             : cutlass.Constexpr = self.D
        num_warps     : cutlass.Constexpr = self.NUM_WARPS
        dims_per_lane : cutlass.Constexpr = self.DIMS_PER_LANE
        num_rounds    : cutlass.Constexpr = self.NUM_ROUNDS
        num_threads   : cutlass.Constexpr = self.BLOCK_SIZE
        wsize         = cute.arch.WARP_SIZE

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.warp_idx()
        warp_idx   = cute.arch.make_warp_uniform(warp_idx)
        lane_idx   = cute.arch.lane_idx()

        allocator    = cutlass.utils.SmemAllocator()
        smem_scores  = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None)
        smem_partial = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((num_warps, d), stride=(d, 1)), 16, None)
        smem_output  = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((d,), stride=(1,)), 16, None)

        for i in range(tidx, n, num_threads):
            smem_scores[i] = scores[i]
        cute.arch.sync_threads()

        out_regs = cute.make_rmem_tensor(
            cute.make_layout((dims_per_lane,), stride=(1,)),
            cutlass.Float32,
        )
        for k in range(dims_per_lane):
            out_regs[k] = cutlass.Float32(0)

        for round_idx in range(num_rounds):
            j      = round_idx * num_warps + warp_idx
            weight = smem_scores[j]
            for k in range(dims_per_lane):
                out_regs[k] += weight * V[j, k * wsize + lane_idx]

        for k in range(dims_per_lane):
            smem_partial[warp_idx, k * wsize + lane_idx] = out_regs[k]

        cute.arch.sync_threads()

        for i in range(tidx, d, num_threads):
            acc = cutlass.Float32(0)
            for w in range(num_warps):
                acc += smem_partial[w, i]
            smem_output[i] = acc

        cute.arch.sync_threads()

        for i in range(tidx, d, num_threads):
            output[i] = smem_output[i]

    def compile(self):
        scores = make_fake_compact_tensor(dtype=cute.Float32, shape=(self.N,),          stride_order=(0,),   assumed_align=16)
        V      = make_fake_compact_tensor(dtype=cute.Float32, shape=(self.N, self.D),   stride_order=(1, 0), assumed_align=16)
        output = make_fake_compact_tensor(dtype=cute.Float32, shape=(self.D,),          stride_order=(0,),   assumed_align=16)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        self._compiled = cute.compile(self, scores, V, output, stream, options="--enable-tvm-ffi")

    def run(self, scores, V, output):
        if self._compiled is None:
            self.compile()
        self._compiled(scores, V, output)

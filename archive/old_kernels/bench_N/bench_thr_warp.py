"""bench_thr_warp: class-based kernel_thr_warp with parameterised N.

Operation: scores = q @ K.T  (vectorised LDG.64, thread→warp reduce)
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


class BenchThrWarp:
    def __init__(self, N: int = 64, D: int = 512):
        self.N = N
        self.D = D
        self.BLOCK_SIZE = 1024
        self.NUM_WARPS = self.BLOCK_SIZE // 32
        self.K_PER_LANE = self.D // 32
        self.NUM_VEC = 8
        self.ITERS_PER_LANE = self.K_PER_LANE // self.NUM_VEC
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
        d              : cutlass.Constexpr = self.D
        n              : cutlass.Constexpr = self.N
        num_warps      : cutlass.Constexpr = self.NUM_WARPS
        num_vec        : cutlass.Constexpr = self.NUM_VEC
        iters_per_lane : cutlass.Constexpr = self.ITERS_PER_LANE
        num_threads    : cutlass.Constexpr = self.BLOCK_SIZE

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
        q_z = cute.zipped_divide(smem_q, (num_vec,))

        for round_idx in range(num_rounds):
            token_idx = round_idx * num_warps + warp_idx
            K_row = K[token_idx, None]
            K_z   = cute.zipped_divide(K_row, (num_vec,))

            sum_partial = cutlass.Float32(0)
            for it in range(iters_per_lane):
                group  = it * wsize + lane_idx
                q_frag = q_z[(None, (group,))].load()
                K_frag = K_z[(None, (group,))].load()
                sumSSA = q_frag * K_frag
                partial = cutlass.Float32(
                    sumSSA.reduce(cute.ReductionOp.ADD, init_val=float(0), reduction_profile=0)
                )
                sum_partial = sum_partial + partial

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

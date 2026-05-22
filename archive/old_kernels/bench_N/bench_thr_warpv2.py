"""bench_thr_warpv2: class-based kernel_thr_warpv2 with parameterised N.

Operation: scores = q @ K.T  (FastGEMV-style 4-row interleaved batching)

Requires N >= ROWS_PER_ROUND = NUM_WARPS * ROWS_PER_WARP = 128.
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


class BenchThrWarpV2:
    def __init__(self, N: int = 128, D: int = 512):
        self.N = N
        self.D = D
        self.BLOCK_SIZE = 1024
        self.NUM_WARPS = self.BLOCK_SIZE // 32
        self.ROWS_PER_WARP = 4
        self.NUM_VEC = 8
        self.K_PER_LANE = self.D // 32
        self.ITERS_PER_LANE = self.K_PER_LANE // self.NUM_VEC
        self.ROWS_PER_ROUND = self.NUM_WARPS * self.ROWS_PER_WARP
        assert N >= self.ROWS_PER_ROUND, f"N={N} must be >= {self.ROWS_PER_ROUND}"
        assert N % self.ROWS_PER_ROUND == 0, f"N={N} must be divisible by {self.ROWS_PER_ROUND}"
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
        d               : cutlass.Constexpr = self.D
        n               : cutlass.Constexpr = self.N
        num_warps       : cutlass.Constexpr = self.NUM_WARPS
        num_vec         : cutlass.Constexpr = self.NUM_VEC
        rows_per_warp   : cutlass.Constexpr = self.ROWS_PER_WARP
        iters_per_lane  : cutlass.Constexpr = self.ITERS_PER_LANE
        rows_per_round  : cutlass.Constexpr = self.ROWS_PER_ROUND
        num_threads     : cutlass.Constexpr = self.BLOCK_SIZE

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

        num_rounds: cutlass.Constexpr = (n + rows_per_round - 1) // rows_per_round
        q_z = cute.zipped_divide(smem_q, (num_vec,))

        for round_idx in range(num_rounds):
            base_token = round_idx * rows_per_round + warp_idx * rows_per_warp

            K_z0 = cute.zipped_divide(K[base_token + 0, None], (num_vec,))
            K_z1 = cute.zipped_divide(K[base_token + 1, None], (num_vec,))
            K_z2 = cute.zipped_divide(K[base_token + 2, None], (num_vec,))
            K_z3 = cute.zipped_divide(K[base_token + 3, None], (num_vec,))

            sums = cute.make_rmem_tensor(
                cute.make_layout((rows_per_warp,), stride=(1,)),
                cutlass.Float32,
            )
            for r in range(rows_per_warp):
                sums[r] = cutlass.Float32(0)

            for it in range(iters_per_lane):
                group  = it * wsize + lane_idx
                q_frag = q_z[(None, (group,))].load()

                K_f0 = K_z0[(None, (group,))].load()
                K_f1 = K_z1[(None, (group,))].load()
                K_f2 = K_z2[(None, (group,))].load()
                K_f3 = K_z3[(None, (group,))].load()

                for v in range(num_vec):
                    qv = cutlass.Float32(q_frag[v])
                    sums[0] = sums[0] + qv * cutlass.Float32(K_f0[v])
                    sums[1] = sums[1] + qv * cutlass.Float32(K_f1[v])
                    sums[2] = sums[2] + qv * cutlass.Float32(K_f2[v])
                    sums[3] = sums[3] + qv * cutlass.Float32(K_f3[v])

            for r in range(rows_per_warp):
                sums[r] = _warp_reduce(sums[r], lambda a, b: a + b, width=32)
            if lane_idx == 0:
                for r in range(rows_per_warp):
                    smem_scores[base_token + r] = sums[r]

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

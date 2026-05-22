"""bench_output_ldg: class-based kernel_output_ldg with parameterised N.

Operation: output = scores @ V  (LDG.128 + 3D smem, zero bank conflicts)
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream


@cute.jit
def _warp_reduce_add(val: cute.Numeric) -> cute.Numeric:
    for i in range(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


class BenchOutputLdg:
    def __init__(self, N: int = 64, D: int = 512):
        self.N = N
        self.D = D
        self.BLOCK_SIZE = 1024
        self.NUM_WARPS = self.BLOCK_SIZE // 32
        self.VEC = 4
        self.NUM_VEC_GROUPS = self.D // self.VEC
        self.GROUPS_PER_PASS = 32
        self.NUM_DIM_PASSES = self.NUM_VEC_GROUPS // self.GROUPS_PER_PASS
        self.NUM_ROUNDS = (self.N + self.NUM_WARPS - 1) // self.NUM_WARPS
        self.SMEM_S_VEC = 1
        self.SMEM_S_WARP = self.VEC + 1           # 5
        self.SMEM_S_VG = (self.NUM_WARPS + 1) * self.SMEM_S_WARP  # 165
        self.VG_PER_WARP = self.NUM_VEC_GROUPS // self.NUM_WARPS   # 4
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
        n              : cutlass.Constexpr = self.N
        d              : cutlass.Constexpr = self.D
        num_warps      : cutlass.Constexpr = self.NUM_WARPS
        num_rounds     : cutlass.Constexpr = self.NUM_ROUNDS
        num_threads    : cutlass.Constexpr = self.BLOCK_SIZE
        vec            : cutlass.Constexpr = self.VEC
        num_dim_passes : cutlass.Constexpr = self.NUM_DIM_PASSES
        groups_per_pass: cutlass.Constexpr = self.GROUPS_PER_PASS
        vg_per_warp    : cutlass.Constexpr = self.VG_PER_WARP
        s_vg           : cutlass.Constexpr = self.SMEM_S_VG
        s_warp         : cutlass.Constexpr = self.SMEM_S_WARP
        s_vec          : cutlass.Constexpr = self.SMEM_S_VEC
        num_vec_groups : cutlass.Constexpr = self.NUM_VEC_GROUPS

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.warp_idx()
        warp_idx   = cute.arch.make_warp_uniform(warp_idx)
        lane_idx   = cute.arch.lane_idx()

        allocator   = cutlass.utils.SmemAllocator()
        smem_scores = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None)
        smem_partial = allocator.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (num_vec_groups, num_warps, vec),
                stride=(s_vg, s_warp, s_vec),
            ), 16, None)

        for i in range(tidx, n, num_threads):
            smem_scores[i] = scores[i]
        cute.arch.sync_threads()

        out_regs = cute.make_rmem_tensor(
            cute.make_layout((num_dim_passes, vec), stride=(vec, 1)),
            cutlass.Float32,
        )
        for dp in range(num_dim_passes):
            for e in range(vec):
                out_regs[dp, e] = cutlass.Float32(0)

        for round_idx in range(num_rounds):
            j      = round_idx * num_warps + warp_idx
            weight = smem_scores[j]
            V_row_z = cute.zipped_divide(V[j, None], (vec,))

            for dp in range(num_dim_passes):
                group = dp * groups_per_pass + lane_idx
                frag  = V_row_z[(None, (group,))].load()

                for v in range(vec):
                    out_regs[dp, v] += weight * frag[v]

        for dp in range(num_dim_passes):
            vg = dp * groups_per_pass + lane_idx
            for e in range(vec):
                smem_partial[vg, warp_idx, e] = out_regs[dp, e]

        cute.arch.sync_threads()

        for vg_off in range(vg_per_warp):
            vg = warp_idx * vg_per_warp + vg_off
            for e in range(vec):
                val = smem_partial[vg, lane_idx, e]
                val = _warp_reduce_add(val)
                if lane_idx == 0:
                    dim = vg * vec + e
                    output[dim] = val

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

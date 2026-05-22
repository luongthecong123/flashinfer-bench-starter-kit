"""kernel_output_ldg: output-phase GEMV with LDG.128 vectorized loads + 3D smem layout.

Operation: output = scores @ V
  scores : [N]       fp32
  V      : [N, D]    fp32
  output : [D]       fp32

Key changes from baseline (kernel_output.py):
  1. LDG.128 vectorised V loads via zipped_divide + .load().
     Baseline: lane l owns dims {k*32+l}  (stride-32, scalar LDG per dim)
     LDG:      lane l owns dims {pass*128 + l*4 .. l*4+3} (4 contiguous, LDG.128)
     4 passes × 32 lanes × 4 elems = 512 dims.

  2. 3D smem_partial layout to eliminate bank conflicts on both write and reduce.
     Shape:  [NUM_VEC_GROUPS, NUM_WARPS, VEC] = [128, 32, 4]
     Stride: ((NUM_WARPS+1)*(VEC+1), VEC+1, 1) = (165, 5, 1)

     VEC is the FASTEST dimension (stride 1) — enables STS coalescing for 4 fp32.

     Write (all lanes of one warp, simultaneous):
       Lane l → smem_partial[dp*32+l, warp_idx, 0..3]
       Stride across lanes = 165.  165 % 32 = 5.  gcd(5,32) = 1 → zero conflicts.

     Read / reduce (each warp reduces over 32 warps):
       Lane l → smem_partial[vg, l, e]
       Stride across lanes = 5.  gcd(5,32) = 1 → zero conflicts.

     Total: max_offset = 127*165 + 31*5 + 3 = 21113 elems ≈ 82.5 KB.
     Plus smem_scores (8 KB) → ~90.5 KB total.  Fine for B200 (228 KB smem).

Grid: [1, 1, 1]   Block: [1024, 1, 1]
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

N              : cutlass.Constexpr = 64
D              : cutlass.Constexpr = 512
BLOCK_SIZE     : cutlass.Constexpr = 1024
NUM_WARPS      : cutlass.Constexpr = BLOCK_SIZE // 32        # 32
VEC            : cutlass.Constexpr = 4                       # fp32 × 4 = 128 bits
NUM_VEC_GROUPS : cutlass.Constexpr = D // VEC                # 128
GROUPS_PER_PASS: cutlass.Constexpr = 32                      # one vec-group per lane
NUM_DIM_PASSES : cutlass.Constexpr = NUM_VEC_GROUPS // GROUPS_PER_PASS  # 4
NUM_ROUNDS     : cutlass.Constexpr = N // NUM_WARPS          # 64

# 3D smem strides: [VG, WARP, VEC] with VEC fastest
SMEM_S_VEC  : cutlass.Constexpr = 1
SMEM_S_WARP : cutlass.Constexpr = VEC + 1                   # 5
SMEM_S_VG   : cutlass.Constexpr = (NUM_WARPS + 1) * SMEM_S_WARP  # 165

# Reduction constants
VG_PER_WARP : cutlass.Constexpr = NUM_VEC_GROUPS // NUM_WARPS  # 4


@cute.jit
def warp_reduce_add(val: cute.Numeric) -> cute.Numeric:
    for i in range(5):  # log2(32) = 5
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def kernel_output_ldg_fn(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
    stream,
):
    kernel_output_ldg_kernel(scores, V, output).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_output_ldg_kernel(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
):
    n              : cutlass.Constexpr = N
    d              : cutlass.Constexpr = D
    num_warps      : cutlass.Constexpr = NUM_WARPS
    num_rounds     : cutlass.Constexpr = NUM_ROUNDS
    num_threads    : cutlass.Constexpr = BLOCK_SIZE
    vec            : cutlass.Constexpr = VEC
    num_dim_passes : cutlass.Constexpr = NUM_DIM_PASSES
    groups_per_pass: cutlass.Constexpr = GROUPS_PER_PASS
    vg_per_warp    : cutlass.Constexpr = VG_PER_WARP
    s_vg           : cutlass.Constexpr = SMEM_S_VG
    s_warp         : cutlass.Constexpr = SMEM_S_WARP
    s_vec          : cutlass.Constexpr = SMEM_S_VEC

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx   = cute.arch.warp_idx()
    warp_idx   = cute.arch.make_warp_uniform(warp_idx)
    lane_idx   = cute.arch.lane_idx()

    # ── Smem allocation ───────────────────────────────────────────────────
    allocator   = cutlass.utils.SmemAllocator()
    smem_scores = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None)
    # 3D layout: [NUM_VEC_GROUPS, NUM_WARPS, VEC] with stride (165, 5, 1)
    num_vec_groups : cutlass.Constexpr = NUM_VEC_GROUPS
    smem_partial = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(
            (num_vec_groups, num_warps, vec),
            stride=(s_vg, s_warp, s_vec),
        ), 16, None)

    # ── Load scores → smem ────────────────────────────────────────────────
    for i in range(tidx, n, num_threads):
        smem_scores[i] = scores[i]
    cute.arch.sync_threads()

    # ── Per-warp register accumulation with LDG.128 ───────────────────────
    # out_regs[dim_pass, elem]: 4 passes × 4 elements = 16 regs per lane
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

        # zipped_divide V row into groups of 4: ((VEC,), (D//VEC,))
        V_row_z = cute.zipped_divide(V[j, None], (vec,))

        for dp in range(num_dim_passes):
            # Lane l loads group = dp*32 + l → V[j, group*4 : group*4+3]
            group = dp * groups_per_pass + lane_idx
            frag  = V_row_z[(None, (group,))].load()   # LDG.128

            for v in range(vec):
                out_regs[dp, v] += weight * frag[v]

    # ── Write partial sums → smem_partial ─────────────────────────────────
    # Lane l writes smem_partial[dp*32+l, warp_idx, 0..3]
    # Vec dimension is fastest (stride 1) → 4 contiguous fp32 per lane.
    # Stride across lanes = s_vg = 165, 165 % 32 = 5 → zero bank conflicts.
    for dp in range(num_dim_passes):
        vg = dp * groups_per_pass + lane_idx
        for e in range(vec):
            smem_partial[vg, warp_idx, e] = out_regs[dp, e]

    cute.arch.sync_threads()

    # ── Cross-warp reduce via warp butterfly ──────────────────────────────
    # Each warp handles vg_per_warp=4 vec-groups, each with 4 elements → 16 dims.
    # Lane l reads smem_partial[vg, l, e] — stride across lanes = 5.
    # gcd(5, 32) = 1 → zero bank conflicts.
    for vg_off in range(vg_per_warp):
        vg = warp_idx * vg_per_warp + vg_off
        for e in range(vec):
            val = smem_partial[vg, lane_idx, e]
            val = warp_reduce_add(val)
            if lane_idx == 0:
                dim = vg * vec + e
                output[dim] = val


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_output_ldg():
    scores = _fake(cute.Float32,  (N,),    (0,),    16)
    V      = _fake(cute.Float32,  (N, D),  (1, 0),  16)
    output = _fake(cute.Float32,  (D,),    (0,),    16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_output_ldg_fn,
        scores, V, output, stream,
        options="--enable-tvm-ffi",
    )


kernel_output_ldg_compiled = compile_kernel_output_ldg()


def run(scores, V, output):
    """scores: [N] fp32, V: [N,D] fp32, output: [D] fp32 (pre-allocated)."""
    kernel_output_ldg_compiled(scores, V, output)

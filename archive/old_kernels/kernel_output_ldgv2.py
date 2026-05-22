"""kernel_output_ldgv2: output-phase GEMV with LDG.128 + STS.128 + LDS.128.

Operation: output = scores @ V
  scores : [N]       fp32
  V      : [N, D]    fp32
  output : [D]       fp32

Key changes from kernel_output_ldg (v1):
  v1: LDG.128 for V loads, but STS.32 / LDS.32 for smem writes/reads.
      Root cause: S_WARP=5 (padded) → addresses not 16B-aligned → compiler
      can't merge 4 consecutive stores/loads into 128-bit ops.
  v2: S_WARP=4 (=VEC, no pad) → offsets always % 4 == 0 → 16B-aligned.
      Compiler auto-vectorizes unrolled constexpr inner loops to STS.128/LDS.128.
      No inline PTX — compiler retains full loop unrolling + scheduling freedom.

  3D smem_partial layout: [NUM_VEC_GROUPS, NUM_WARPS, VEC] = [128, 32, 4]
  Stride: (S_VG=132, S_WARP=4, 1) — VEC fastest, 16B-aligned.

  Bank conflict analysis (128-bit accesses):
    Write: stride across lanes = S_VG=132. 132%32=4 → 4-way per phase.
    Read:  stride across lanes = S_WARP=4. 4%32=4 → 4-way per phase.
    4-way × 128-bit = same effective throughput as 1-way × 32-bit (16 phases each),
    but 4× fewer instruction dispatches.

  Reduce phase: load all 4 VEC elements first (→ LDS.128),
  THEN butterfly-reduce each separately. Separation ensures loads aren't
  interleaved with shuffles so compiler can merge them.

  Smem: 128*132 = 16896 elems ≈ 66 KB + 8 KB scores = 74 KB.

Grid: [1, 1, 1]   Block: [1024, 1, 1]
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

N              : cutlass.Constexpr = 2048
D              : cutlass.Constexpr = 512
BLOCK_SIZE     : cutlass.Constexpr = 1024
NUM_WARPS      : cutlass.Constexpr = BLOCK_SIZE // 32        # 32
VEC            : cutlass.Constexpr = 4                       # fp32 × 4 = 128 bits
NUM_VEC_GROUPS : cutlass.Constexpr = D // VEC                # 128
GROUPS_PER_PASS: cutlass.Constexpr = 32                      # one vec-group per lane
NUM_DIM_PASSES : cutlass.Constexpr = NUM_VEC_GROUPS // GROUPS_PER_PASS  # 4
NUM_ROUNDS     : cutlass.Constexpr = N // NUM_WARPS          # 64

# 3D smem strides: [VG, WARP, VEC] — VEC fastest, S_WARP=4 for 16B align
SMEM_S_VEC  : cutlass.Constexpr = 1
SMEM_S_WARP : cutlass.Constexpr = VEC                        # 4 (= VEC, 16B aligned)
SMEM_S_VG   : cutlass.Constexpr = (NUM_WARPS + 1) * SMEM_S_WARP  # 132

# Reduction constants
VG_PER_WARP : cutlass.Constexpr = NUM_VEC_GROUPS // NUM_WARPS  # 4


@cute.jit
def warp_reduce_add(val: cute.Numeric) -> cute.Numeric:
    for i in range(5):  # log2(32) = 5
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def kernel_output_ldgv2_fn(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
    stream,
):
    kernel_output_ldgv2_kernel(scores, V, output).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_output_ldgv2_kernel(
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

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx   = cute.arch.warp_idx()
    warp_idx   = cute.arch.make_warp_uniform(warp_idx)
    lane_idx   = cute.arch.lane_idx()

    # ── Smem allocation ───────────────────────────────────────────────────
    allocator   = cutlass.utils.SmemAllocator()
    smem_scores = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None)
    # 3D layout: [NUM_VEC_GROUPS, NUM_WARPS, VEC] stride (132, 4, 1)
    # VEC is fastest dim (stride 1), S_WARP=4 → 16B aligned for any vg, warp_idx.
    num_vec_groups : cutlass.Constexpr = NUM_VEC_GROUPS
    smem_partial = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(
            (num_vec_groups, num_warps, vec),
            stride=(s_vg, s_warp, 1),
        ), 16, None)

    # ── Load scores → smem ────────────────────────────────────────────────
    for i in range(tidx, n, num_threads):
        smem_scores[i] = scores[i]
    cute.arch.sync_threads()

    # ── Per-warp register accumulation with LDG.128 ───────────────────────
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

        # zipped_divide V row: ((VEC,), (D//VEC,))
        V_row_z = cute.zipped_divide(V[j, None], (vec,))

        for dp in range(num_dim_passes):
            group = dp * groups_per_pass + lane_idx
            frag  = V_row_z[(None, (group,))].load()   # LDG.128

            for v in range(vec):
                out_regs[dp, v] += weight * frag[v]

    # ── Write partial sums → smem_partial ─────────────────────────────────
    # Lane l writes smem_partial[vg, warp_idx, 0..3].
    # VEC stride=1, S_WARP=4 → 4 contiguous 16B-aligned fp32 per lane.
    # Constexpr inner loop unrolled → compiler merges to STS.128.
    for dp in range(num_dim_passes):
        vg = dp * groups_per_pass + lane_idx
        for e in range(vec):
            smem_partial[vg, warp_idx, e] = out_regs[dp, e]

    cute.arch.sync_threads()

    # ── Cross-warp reduce ─────────────────────────────────────────────────
    # Each warp handles vg_per_warp=4 vec-groups × 4 VEC elements = 16 dims.
    # For each vg: load all 4 VEC elements FIRST (→ LDS.128 from contiguous smem),
    # THEN butterfly-reduce each. Separating loads from shuffles lets compiler merge.
    for vg_off in range(vg_per_warp):
        vg = warp_idx * vg_per_warp + vg_off

        # Load phase: smem_partial[vg, lane_idx, 0..3] — 4 contiguous fp32 → LDS.128
        # Use zipped_divide on the VEC slice for explicit vectorized load
        smem_slice = smem_partial[vg, lane_idx, None]          # shape (VEC,) stride (1,)
        smem_z     = cute.zipped_divide(smem_slice, (vec,))    # ((VEC,), (1,))
        ld_frag    = smem_z[(None, (0,))].load()               # LDS.128

        # Reduce phase: butterfly reduce each element, lane 0 writes output
        for e in range(vec):
            val = warp_reduce_add(ld_frag[e])
            if lane_idx == 0:
                output[vg * vec + e] = val


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_output_ldgv2():
    scores = _fake(cute.Float32,  (N,),    (0,),    16)
    V      = _fake(cute.Float32,  (N, D),  (1, 0),  16)
    output = _fake(cute.Float32,  (D,),    (0,),    16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_output_ldgv2_fn,
        scores, V, output, stream,
        options="--enable-tvm-ffi",
    )


kernel_output_ldgv2_compiled = compile_kernel_output_ldgv2()


def run(scores, V, output):
    """scores: [N] fp32, V: [N,D] fp32, output: [D] fp32 (pre-allocated)."""
    kernel_output_ldgv2_compiled(scores, V, output)

"""kernel_outputv6: output-phase GEMV with FFMA2 + split load/compute loops.

Root cause of v5 slowdown (from SASS analysis):
  The compiler reuses only 2 physical registers (R22/R23) for ALL 8 groups' V loads.
  Each group's LDG overwrites R22/R23 immediately before its FFMA2, leaving only
  1-3 instructions of gap — far too short to hide global memory latency.

  Pattern in v5 SASS:
    LDG R22  (group 0 v_lo)
    [2 instructions]
    FFMA2 R6, R22  ← 3-inst gap → 12 stall samples
    LDG R22  (group 1 v_lo)  ← R22 reused!
    FFMA2 R18, R22 ← 1-inst gap → 14 stall samples (worst case)
  5 out of 8 FFMA2s stall because load/compute are back-to-back.

Fix: separate the g-loop into two phases:
  Phase 1 (load):   all 16 LDGs issued before any FFMA2.
  Phase 2 (compute): all 8 FFMA2s after all loads are in flight.

  With all 16 v_all values live simultaneously (forced by the split),
  LLVM must allocate 16 distinct physical registers.  The first FFMA2 now
  has a 16-instruction gap from the first LDG → latency is hidden by warp
  scheduling across 32 warps.

Grid: [1, 1, 1]   Block: [1024, 1, 1]
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir

N              : cutlass.Constexpr = 2048
D              : cutlass.Constexpr = 512
BLOCK_SIZE     : cutlass.Constexpr = 1024
NUM_WARPS      : cutlass.Constexpr = BLOCK_SIZE // 32   # 32
DIMS_PER_LANE  : cutlass.Constexpr = D // 32            # 16
NUM_GROUPS     : cutlass.Constexpr = DIMS_PER_LANE // 2 # 8
NUM_ROUNDS     : cutlass.Constexpr = N // NUM_WARPS     # 64


@dsl_user_op
def fma_f32x2(
    a0: cutlass.Float32, a1: cutlass.Float32,
    b0: cutlass.Float32, b1: cutlass.Float32,
    c0: cutlass.Float32, c1: cutlass.Float32,
    *, loc=None, ip=None,
):
    """fma.rn.f32x2: d[i] = a[i]*b[i] + c[i], i=0,1.
    Pack two f32 → <2xf32> → i64 (same as CUTLASS simd_sm100.hpp).
    """
    f32x2_ty = ir.VectorType.get([2], T.f32())
    i32_ty   = ir.IntegerType.get_signless(32)

    def to_i64(x, y):
        undef = llvm.mlir_undef(f32x2_ty, loc=loc, ip=ip)
        i0 = llvm.mlir_constant(ir.IntegerAttr.get(i32_ty, 0), loc=loc, ip=ip)
        i1 = llvm.mlir_constant(ir.IntegerAttr.get(i32_ty, 1), loc=loc, ip=ip)
        v  = llvm.insertelement(undef, cutlass.Float32(x).ir_value(loc=loc, ip=ip), i0, loc=loc, ip=ip)
        v  = llvm.insertelement(v,     cutlass.Float32(y).ir_value(loc=loc, ip=ip), i1, loc=loc, ip=ip)
        return llvm.bitcast(T.i64(), v, loc=loc, ip=ip)

    d_i64 = llvm.inline_asm(
        T.i64(),
        [to_i64(a0, a1), to_i64(b0, b1), to_i64(c0, c1)],
        "fma.rn.f32x2 $0, $1, $2, $3;",
        "=l,l,l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )

    d_vec = llvm.bitcast(f32x2_ty, d_i64, loc=loc, ip=ip)
    i0    = llvm.mlir_constant(ir.IntegerAttr.get(i32_ty, 0), loc=loc, ip=ip)
    i1    = llvm.mlir_constant(ir.IntegerAttr.get(i32_ty, 1), loc=loc, ip=ip)
    return (cutlass.Float32(llvm.extractelement(d_vec, i0, loc=loc, ip=ip)),
            cutlass.Float32(llvm.extractelement(d_vec, i1, loc=loc, ip=ip)))


@cute.jit
def kernel_outputv6_fn(
    scores: cute.Tensor,
    V:      cute.Tensor,
    output: cute.Tensor,
    stream,
):
    kernel_outputv6_kernel(scores, V, output).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_outputv6_kernel(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
):
    n              : cutlass.Constexpr = N
    d              : cutlass.Constexpr = D
    num_warps      : cutlass.Constexpr = NUM_WARPS
    dims_per_lane  : cutlass.Constexpr = DIMS_PER_LANE
    num_groups     : cutlass.Constexpr = NUM_GROUPS
    num_rounds     : cutlass.Constexpr = NUM_ROUNDS
    num_threads    : cutlass.Constexpr = BLOCK_SIZE
    wsize          = cute.arch.WARP_SIZE

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx   = cute.arch.warp_idx()
    warp_idx   = cute.arch.make_warp_uniform(warp_idx)
    lane_idx   = cute.arch.lane_idx()

    # ── Smem allocation ───────────────────────────────────────────────────
    allocator    = cutlass.utils.SmemAllocator()
    smem_scores  = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None)
    smem_partial = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((num_warps, d), stride=(d, 1)), 16, None)
    smem_output  = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((d,), stride=(1,)), 16, None)

    # ── Load scores → smem ────────────────────────────────────────────────
    for i in range(tidx, n, num_threads):
        smem_scores[i] = scores[i]
    cute.arch.sync_threads()

    # ── Per-warp register accumulation: FFMA2 with split load/compute ────
    #
    # Key fix over v5: issue ALL 16 V loads BEFORE any FFMA2 fires.
    #   - v_all has make_layout((num_groups, 2), stride=(2,1)) → 16 adjacent slots.
    #   - Load loop: 16 LDGs, all live simultaneously → 16 distinct registers.
    #   - Compute loop: FFMA2[g] uses v_all[g,*] loaded >= 8 instructions ago.
    #   - First FFMA2 sees v_all[0,*] from 16 instructions prior → full latency hiding.
    #
    out_regs = cute.make_rmem_tensor(
        cute.make_layout((num_groups, 2), stride=(2, 1)),
        cutlass.Float32,
    )
    for g in range(num_groups):
        out_regs[g, 0] = cutlass.Float32(0)
        out_regs[g, 1] = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        j      = round_idx * num_warps + warp_idx
        weight = smem_scores[j]

        # ── Phase 1: issue all 16 V loads → 16 distinct registers ────────
        # With all v_all[g,*] live simultaneously until Phase 2,
        # the register allocator CANNOT reuse any of the 16 slots.
        v_all = cute.make_rmem_tensor(
            cute.make_layout((num_groups, 2), stride=(2, 1)),
            cutlass.Float32,
        )
        for g in range(num_groups):
            v_all[g, 0] = cutlass.Float32(V[j, (2 * g)     * wsize + lane_idx])
            v_all[g, 1] = cutlass.Float32(V[j, (2 * g + 1) * wsize + lane_idx])

        # ── Phase 2: all 8 FFMA2s after all loads are in flight ──────────
        # FFMA2 for group g fires with v_all[g,*] loaded 8+(7-g)*2 instructions ago.
        for g in range(num_groups):
            new0, new1 = fma_f32x2(v_all[g, 0], v_all[g, 1], weight, weight,
                                   out_regs[g, 0], out_regs[g, 1])
            out_regs[g, 0] = new0
            out_regs[g, 1] = new1

    # ── Write partial sums → smem_partial ────────────────────────────────
    for g in range(num_groups):
        smem_partial[warp_idx, (2 * g)     * wsize + lane_idx] = out_regs[g, 0]
        smem_partial[warp_idx, (2 * g + 1) * wsize + lane_idx] = out_regs[g, 1]

    cute.arch.sync_threads()

    # ── Cross-warp reduce ─────────────────────────────────────────────────
    for i in range(tidx, d, num_threads):
        acc = cutlass.Float32(0)
        for w in range(num_warps):
            acc += smem_partial[w, i]
        smem_output[i] = acc

    cute.arch.sync_threads()

    # ── Epilogue ──────────────────────────────────────────────────────────
    for i in range(tidx, d, num_threads):
        output[i] = smem_output[i]


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_outputv6():
    scores = _fake(cute.Float32, (N,),    (0,),   16)
    V      = _fake(cute.Float32, (N, D),  (1, 0), 16)
    output = _fake(cute.Float32, (D,),    (0,),   16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_outputv6_fn,
        scores, V, output, stream,
        options="--enable-tvm-ffi",
    )


kernel_outputv6_compiled = compile_kernel_outputv6()


def run(scores, V, output):
    """scores: [N] fp32, V: [N,D] fp32, output: [D] fp32 (pre-allocated)."""
    kernel_outputv6_compiled(scores, V, output)

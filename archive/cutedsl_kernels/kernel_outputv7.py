"""kernel_outputv7: output-phase GEMV with FFMA2 + register barriers.

Root cause of v5/v6 slowdown (from SASS analysis):
  LLVM sinks LDG loads to right before each FFMA2 asm block because
  inline_asm is opaque. The compiler reuses only 2 temporary registers
  (R22/R23) for ALL 16 V loads, creating LDG→FFMA2 back-to-back chains
  with 1-3 instruction gaps. 5/8 FFMA2s stall → 146 µs (3× worse).

  Meanwhile the scalar FFMA baseline compiles to a double-buffered pipeline:
  32 LDGs → 16 FFMAs (interleaved with next-round 32 LDGs) → 0 stalls.
  The compiler uses 60+ registers (2 V banks of 16 + 16 accumulators).

Fix: add 16 barrier register constraints to each FFMA2 asm call.
  Each fma_f32x2_barrier() takes all 16 V values as `f` (float register)
  asm inputs in addition to the 3 real packed i64 inputs.
  LLVM must ensure all 16 values are in registers at the asm point.
  → All 16 LDGs are forced before the first FFMA2 → pipeline fills.

  The barrier values aren't referenced in the asm string, but their
  register constraints are binding — LLVM can't sink any load past them.

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
def fma_f32x2_barrier(
    v_lo: cutlass.Float32, v_hi: cutlass.Float32,
    w: cutlass.Float32,
    c_lo: cutlass.Float32, c_hi: cutlass.Float32,
    b0:  cutlass.Float32, b1:  cutlass.Float32,
    b2:  cutlass.Float32, b3:  cutlass.Float32,
    b4:  cutlass.Float32, b5:  cutlass.Float32,
    b6:  cutlass.Float32, b7:  cutlass.Float32,
    b8:  cutlass.Float32, b9:  cutlass.Float32,
    b10: cutlass.Float32, b11: cutlass.Float32,
    b12: cutlass.Float32, b13: cutlass.Float32,
    b14: cutlass.Float32, b15: cutlass.Float32,
    *, loc=None, ip=None,
):
    """fma.rn.f32x2 with 16 V-value barrier constraints.

    Real work: d[i] = v[i]*w + c[i], i=0,1  (via fma.rn.f32x2)
    Barrier:   b0..b15 are listed as `f` asm inputs (float registers).
               LLVM must keep all 16 in registers at this asm point,
               forcing all 16 LDGs to be issued before this FFMA2 fires.
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

    # 3 real packed i64 inputs
    real_inputs = [
        to_i64(v_lo, v_hi),   # $1: packed V pair for this group
        to_i64(w, w),          # $2: packed weight (broadcast)
        to_i64(c_lo, c_hi),   # $3: packed accumulator
    ]

    # 16 barrier f32 inputs — forces all V loads alive at asm point
    barrier_inputs = [
        cutlass.Float32(b).ir_value(loc=loc, ip=ip)
        for b in [b0, b1, b2, b3, b4, b5, b6, b7,
                  b8, b9, b10, b11, b12, b13, b14, b15]
    ]

    #   $0       : output i64         (=l)
    #   $1..$3   : real packed inputs  (l,l,l)
    #   $4..$19  : barrier f32 inputs  (f × 16, unreferenced in asm)
    d_i64 = llvm.inline_asm(
        T.i64(),
        real_inputs + barrier_inputs,
        "fma.rn.f32x2 $0, $1, $2, $3;",
        "=l,l,l,l" + ",f" * 16,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )

    # Unpack i64 → two f32
    d_vec = llvm.bitcast(f32x2_ty, d_i64, loc=loc, ip=ip)
    i0    = llvm.mlir_constant(ir.IntegerAttr.get(i32_ty, 0), loc=loc, ip=ip)
    i1    = llvm.mlir_constant(ir.IntegerAttr.get(i32_ty, 1), loc=loc, ip=ip)
    return (cutlass.Float32(llvm.extractelement(d_vec, i0, loc=loc, ip=ip)),
            cutlass.Float32(llvm.extractelement(d_vec, i1, loc=loc, ip=ip)))


@cute.jit
def kernel_outputv7_fn(
    scores: cute.Tensor,
    V:      cute.Tensor,
    output: cute.Tensor,
    stream,
):
    kernel_outputv7_kernel(scores, V, output).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_outputv7_kernel(
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

    # ── Per-warp register accumulation: FFMA2 with barriers ──────────────
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

        # ── Phase 1: load all 16 V values into registers ────────────────
        # These become independent LDGs. The barriers in Phase 2 ensure
        # LLVM cannot sink any of them past the first FFMA2.
        v_buf = cute.make_rmem_tensor(
            cute.make_layout((dims_per_lane,), stride=(1,)),
            cutlass.Float32,
        )
        for k in range(dims_per_lane):
            v_buf[k] = cutlass.Float32(V[j, k * wsize + lane_idx])

        # ── Phase 2: 8 FFMA2s, each with all 16 V values as barriers ───
        for g in range(num_groups):
            new0, new1 = fma_f32x2_barrier(
                v_buf[2 * g], v_buf[2 * g + 1],       # V pair for this group
                weight,                                 # weight (broadcast)
                out_regs[g, 0], out_regs[g, 1],        # accumulator pair
                v_buf[0],  v_buf[1],  v_buf[2],  v_buf[3],   # barriers
                v_buf[4],  v_buf[5],  v_buf[6],  v_buf[7],
                v_buf[8],  v_buf[9],  v_buf[10], v_buf[11],
                v_buf[12], v_buf[13], v_buf[14], v_buf[15],
            )
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


def compile_kernel_outputv7():
    scores = _fake(cute.Float32, (N,),    (0,),   16)
    V      = _fake(cute.Float32, (N, D),  (1, 0), 16)
    output = _fake(cute.Float32, (D,),    (0,),   16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_outputv7_fn,
        scores, V, output, stream,
        options="--enable-tvm-ffi",
    )


kernel_outputv7_compiled = compile_kernel_outputv7()


def run(scores, V, output):
    """scores: [N] fp32, V: [N,D] fp32, output: [D] fp32 (pre-allocated)."""
    kernel_outputv7_compiled(scores, V, output)

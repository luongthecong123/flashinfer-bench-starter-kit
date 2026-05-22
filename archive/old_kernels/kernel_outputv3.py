"""kernel_outputv3: output-phase GEMV with column-major smem_partial.

Operation: output = scores @ V
  scores : [N]       fp32
  V      : [N, D]    fp32
  output : [D]       fp32

Key change from baseline (kernel_output.py):
  v1 smem_partial[32, 512] stride (512, 1) — row-major
      write: conflict-free (stride 1 across lanes)
      reduce: 1024 threads loop 32 warps → 32 LDS per thread, conflict-free per inst

  v3 smem_partial[32, 512] stride (1, 33) — column-major + 1 pad (warp axis is fast)
      smem_partial[w, d] → linear offset = w + d*33
      write stride across lanes = 33; 33%32=1 → all 32 distinct banks → zero write conflicts
      read  stride across lanes = 1                              → zero read  conflicts
      reduce: each warp handles 16 dims
              lane_idx loads smem_partial[lane_idx, d] (contiguous), butterfly reduce,
              lane 0 writes output[d] directly → no smem_output needed

Grid: [1, 1, 1]   Block: [1024, 1, 1]
"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import math

N              : cutlass.Constexpr = 2048
D              : cutlass.Constexpr = 512
BLOCK_SIZE     : cutlass.Constexpr = 1024
NUM_WARPS      : cutlass.Constexpr = BLOCK_SIZE // 32   # 32
DIMS_PER_LANE  : cutlass.Constexpr = D // 32            # 16
NUM_ROUNDS     : cutlass.Constexpr = N // NUM_WARPS     # 64


@cute.jit
def warp_reduce_add(val: cute.Numeric) -> cute.Numeric:
    for i in range(5):  # log2(32) = 5
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def kernel_outputv3_fn(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
    stream,
):
    kernel_outputv3_kernel(scores, V, output).launch(
        grid=[1, 1, 1],
        block=[BLOCK_SIZE, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel_outputv3_kernel(
    scores: cute.Tensor,   # [N]    fp32
    V:      cute.Tensor,   # [N, D] fp32
    output: cute.Tensor,   # [D]    fp32
):
    n              : cutlass.Constexpr = N
    d              : cutlass.Constexpr = D
    num_warps      : cutlass.Constexpr = NUM_WARPS
    dims_per_lane  : cutlass.Constexpr = DIMS_PER_LANE
    num_rounds     : cutlass.Constexpr = NUM_ROUNDS
    num_threads    : cutlass.Constexpr = BLOCK_SIZE
    wsize          = cute.arch.WARP_SIZE   # 32

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx   = cute.arch.warp_idx()
    warp_idx   = cute.arch.make_warp_uniform(warp_idx)
    lane_idx   = cute.arch.lane_idx()

    # ── Smem allocation ───────────────────────────────────────────────────
    allocator   = cutlass.utils.SmemAllocator()
    smem_scores = allocator.allocate_tensor(
        cutlass.Float32, cute.make_layout((n,), stride=(1,)), 16, None)
    # Column-major layout with +1 padding: smem_partial[w, dim] at offset w + dim*(num_warps+1)
    # Column pitch = 33 (odd) → stride between lanes during write = 33 % 32 = 1
    # → all 32 lanes hit distinct banks for both reads and writes.
    # Size: max offset = 31 + 511*33 = 16894 elements ≈ 66 KB
    smem_partial = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((num_warps, d), stride=(1, num_warps + 2)), 16, None)

    # ── Load scores → smem ────────────────────────────────────────────────
    for i in range(tidx, n, num_threads):
        smem_scores[i] = scores[i]
    cute.arch.sync_threads()

    # ── Per-warp register accumulation (same as v1) ───────────────────────
    # Lane lane_idx owns dims: k*32 + lane_idx for k in 0..dims_per_lane-1
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

    # ── Write partial sums → smem_partial (32-way bank conflict) ─────────
    # offset = warp_idx + (k*32 + lane_idx) * 32 → stride 32 across lanes
    for k in range(dims_per_lane):
        smem_partial[warp_idx, k * wsize + lane_idx] = out_regs[k]

    cute.arch.sync_threads()

    # ── Cross-warp reduce via warp butterfly ──────────────────────────────
    # Warp warp_idx handles dims [warp_idx*16 .. warp_idx*16+15].
    # For dim d, smem_partial[0..31, d] are contiguous (offset = lane_idx + d*32).
    # Each lane loads one warp's contribution → warp butterfly reduce → lane 0 → gmem.
    for k in range(dims_per_lane):
        dim = warp_idx * dims_per_lane + k
        # smem_partial[lane_idx, dim] = lane_idx + dim * num_warps
        # For 32 lanes: offsets dim*32, dim*32+1, ..., dim*32+31 → stride-1, zero conflict
        val = smem_partial[lane_idx, dim]
        val = warp_reduce_add(val)
        if lane_idx == 0:
            output[dim] = val


# ── Compilation ───────────────────────────────────────────────────────────────

def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(
        dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=align
    )


def compile_kernel_outputv3():
    scores = _fake(cute.Float32,  (N,),    (0,),    16)
    V      = _fake(cute.Float32,  (N, D),  (1, 0),  16)
    output = _fake(cute.Float32,  (D,),    (0,),    16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel_outputv3_fn,
        scores, V, output, stream,
        options="--enable-tvm-ffi",
    )


kernel_outputv3_compiled = compile_kernel_outputv3()


def run(scores, V, output):
    """scores: [N] fp32, V: [N,D] fp32, output: [D] fp32 (pre-allocated)."""
    kernel_outputv3_compiled(scores, V, output)

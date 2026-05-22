"""
output_simt_ffma2_tileN_intra.py — Intra profiling for N-tiled FFMA2 variants.

Reduces smem_partial by processing N columns in num_tile_N sequential passes.
GEMV is unchanged (all registers accumulated in one pass over K).
After GEMV: num_tile_N × (write tile → sync → reduce tile → output).

Variants:
  tile_N=2  VEC_SIZE=8  smem_partial (16,2,256) = 32 KB  (+1 sync vs baseline)
  tile_N=4  VEC_SIZE=4  smem_partial (16,2,128) = 16 KB  (+3 syncs vs baseline)
"""
import json
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass.cute.runtime import from_dlpack


# ── Probe infra ───────────────────────────────────────────────────────────────

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(MLIR_T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int64(t)


@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(MLIR_T.i32(), [], "mov.u32 $0, %smid;", "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Int32(t)


PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES  = 5
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY

TAGS      = {"total": 2, "load_w": 4, "gemv": 6, "write_reduce": 8}
TAG_NAMES = {v: k for k, v in TAGS.items()}


def range_start(probe, row, cnt, sm_val, tag_val):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 0] = cutlass.Int64(sm_val)
    probe[row, off + 1] = cutlass.Int64(tag_val)
    probe[row, off + 2] = globaltimer_u64()


def range_stop(probe, row, cnt):
    off = PROBE_HEADER + cnt * PROBE_ENTRY
    probe[row, off + 3] = globaltimer_u64() - probe[row, off + 2]
    return cnt + cutlass.Int32(1)


def range_finalize(probe, row, cnt):
    probe[row, 0] = cutlass.Int64(cnt)


M = 2
K = 128
N = 512
NUM_THREADS = 512
NUM_WARPS   = NUM_THREADS // 32   # 16
NUM_ROUNDS  = K // NUM_WARPS      # 8


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


# ══════════════════════════════════════════════════════════════════════════════
#  tile_N = 2   (VEC_SIZE=8, smem_partial 32 KB)
# ══════════════════════════════════════════════════════════════════════════════

VEC_SIZE_2       = 8
ITERS_PER_LANE_2 = N // (32 * VEC_SIZE_2)   # 2
NUM_TILE_N_2     = 4
N_TILE_2         = N // NUM_TILE_N_2          # 256
ITERS_PER_TILE_2 = ITERS_PER_LANE_2 // NUM_TILE_N_2  # 1


@cute.jit
def output_gemv_ffma2_tile2_intra_jit(
    weights: cute.Tensor,
    ckv:     cute.Tensor,
    output:  cute.Tensor,
    probe:   cute.Tensor,
):
    output_gemv_ffma2_tile2_intra_kernel(weights, ckv, output, probe).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1])


@cute.kernel
def output_gemv_ffma2_tile2_intra_kernel(
    weights: cute.Tensor,
    ckv:     cute.Tensor,
    output:  cute.Tensor,
    probe:   cute.Tensor,
):
    M_:             cutlass.Constexpr = M
    K_:             cutlass.Constexpr = K
    N_:             cutlass.Constexpr = N
    num_threads:    cutlass.Constexpr = NUM_THREADS
    num_warps:      cutlass.Constexpr = NUM_WARPS
    vec_size:       cutlass.Constexpr = VEC_SIZE_2
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE_2
    num_rounds:     cutlass.Constexpr = NUM_ROUNDS
    num_tile_n:     cutlass.Constexpr = NUM_TILE_N_2
    n_tile:         cutlass.Constexpr = N_TILE_2
    iters_per_tile: cutlass.Constexpr = ITERS_PER_TILE_2

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    alloc = cutlass.utils.SmemAllocator()
    smem_weight  = _smem(alloc, cutlass.Float32, (M_ * K_,), (1,), 16)
    # (16, 2, 256)  stride (512, 256, 1) = 32 KB
    smem_partial = _smem(alloc, cutlass.Float32,
                         (num_warps, M_, n_tile), (M_ * n_tile, n_tile, 1), 16)

    sm_val = smid_u32()
    if tidx == cutlass.Int32(0):
        range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])
        range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["load_w"])

    for col in range(tidx, K_, num_threads):
        smem_weight[col * 2 + 0] = weights[0, col]
        smem_weight[col * 2 + 1] = weights[1, col]
    cute.arch.sync_threads()

    smem_w_vec2 = cute.zipped_divide(smem_weight, (2,))
    ckv_        = cute.zipped_divide(ckv, (1, vec_size))

    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(1))
        range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["gemv"])

    out_regs_r0 = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane * vec_size,), stride=(1,)), cutlass.Float32)
    out_regs_r1 = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane * vec_size,), stride=(1,)), cutlass.Float32)
    for i in range(iters_per_lane * vec_size):
        out_regs_r0[i] = cutlass.Float32(0)
        out_regs_r1[i] = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < K_:
            w_frag = smem_w_vec2[(None,), (sparse_idx,)].load()
            w0 = w_frag[0]
            w1 = w_frag[1]
            ckv_row = ckv_[(0, None), (sparse_idx, None)]
            for it in range(iters_per_lane):
                rest_idx = it * wsize + lane_idx
                ckv_vec = ckv_row[None, rest_idx].load()
                for v in range(vec_size):
                    ckv_f32 = cutlass.Float32(ckv_vec[v])
                    reg_idx = it * vec_size + v
                    out_regs_r0[reg_idx], out_regs_r1[reg_idx] = \
                        cute.arch.fma_packed_f32x2(
                            (w0, w1), (ckv_f32, ckv_f32),
                            (out_regs_r0[reg_idx], out_regs_r1[reg_idx]))

    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))
        range_start(probe, cutlass.Int32(0), cutlass.Int32(3), sm_val, TAGS["write_reduce"])

    # Tiled write → sync → reduce → (sync before next tile)
    for tile_idx in range(num_tile_n):
        reg_base = tile_idx * iters_per_tile * vec_size  # 0 or 8
        for v in range(vec_size):
            n_col_local = lane_idx * vec_size + v          # 0..255
            smem_partial[warp_idx, 0, n_col_local] = out_regs_r0[reg_base + v]
            smem_partial[warp_idx, 1, n_col_local] = out_regs_r1[reg_base + v]
        cute.arch.sync_threads()

        n_global_base = tile_idx * n_tile
        for m in range(M_):
            for i in range(tidx, n_tile, num_threads):
                acc = cutlass.Float32(0)
                for w in range(num_warps):
                    acc += smem_partial[w, m, i]
                output[m, n_global_base + i] = acc

        if tile_idx < num_tile_n - 1:
            cute.arch.sync_threads()  # before next tile overwrites smem

    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(3))
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(0))
        range_finalize(probe, cutlass.Int32(0), cutlass.Int32(4))


# ══════════════════════════════════════════════════════════════════════════════
#  tile_N = 4   (VEC_SIZE=4, smem_partial 16 KB)
# ══════════════════════════════════════════════════════════════════════════════

VEC_SIZE_4       = 4
ITERS_PER_LANE_4 = N // (32 * VEC_SIZE_4)   # 4
NUM_TILE_N_4     = 4
N_TILE_4         = N // NUM_TILE_N_4          # 128
ITERS_PER_TILE_4 = ITERS_PER_LANE_4 // NUM_TILE_N_4  # 1


@cute.jit
def output_gemv_ffma2_tile4_intra_jit(
    weights: cute.Tensor,
    ckv:     cute.Tensor,
    output:  cute.Tensor,
    probe:   cute.Tensor,
):
    output_gemv_ffma2_tile4_intra_kernel(weights, ckv, output, probe).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1])


@cute.kernel
def output_gemv_ffma2_tile4_intra_kernel(
    weights: cute.Tensor,
    ckv:     cute.Tensor,
    output:  cute.Tensor,
    probe:   cute.Tensor,
):
    M_:             cutlass.Constexpr = M
    K_:             cutlass.Constexpr = K
    N_:             cutlass.Constexpr = N
    num_threads:    cutlass.Constexpr = NUM_THREADS
    num_warps:      cutlass.Constexpr = NUM_WARPS
    vec_size:       cutlass.Constexpr = VEC_SIZE_4       # 4 (64-bit CKV loads)
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE_4 # 4
    num_rounds:     cutlass.Constexpr = NUM_ROUNDS
    num_tile_n:     cutlass.Constexpr = NUM_TILE_N_4
    n_tile:         cutlass.Constexpr = N_TILE_4
    iters_per_tile: cutlass.Constexpr = ITERS_PER_TILE_4

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    alloc = cutlass.utils.SmemAllocator()
    smem_weight  = _smem(alloc, cutlass.Float32, (M_ * K_,), (1,), 16)
    # (16, 2, 128)  stride (256, 128, 1) = 16 KB
    smem_partial = _smem(alloc, cutlass.Float32,
                         (num_warps, M_, n_tile), (M_ * n_tile, n_tile, 1), 16)

    sm_val = smid_u32()
    if tidx == cutlass.Int32(0):
        range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])
        range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["load_w"])

    for col in range(tidx, K_, num_threads):
        smem_weight[col * 2 + 0] = weights[0, col]
        smem_weight[col * 2 + 1] = weights[1, col]
    cute.arch.sync_threads()

    smem_w_vec2 = cute.zipped_divide(smem_weight, (2,))
    ckv_        = cute.zipped_divide(ckv, (1, vec_size))   # vec_size=4

    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(1))
        range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["gemv"])

    # 4 iters × 4 vec = 16 regs per row (same count as baseline 2×8=16)
    out_regs_r0 = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane * vec_size,), stride=(1,)), cutlass.Float32)
    out_regs_r1 = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane * vec_size,), stride=(1,)), cutlass.Float32)
    for i in range(iters_per_lane * vec_size):
        out_regs_r0[i] = cutlass.Float32(0)
        out_regs_r1[i] = cutlass.Float32(0)

    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < K_:
            w_frag = smem_w_vec2[(None,), (sparse_idx,)].load()
            w0 = w_frag[0]
            w1 = w_frag[1]
            ckv_row = ckv_[(0, None), (sparse_idx, None)]
            for it in range(iters_per_lane):
                rest_idx = it * wsize + lane_idx
                ckv_vec = ckv_row[None, rest_idx].load()   # 4 × BF16 (64-bit)
                for v in range(vec_size):
                    ckv_f32 = cutlass.Float32(ckv_vec[v])
                    reg_idx = it * vec_size + v
                    out_regs_r0[reg_idx], out_regs_r1[reg_idx] = \
                        cute.arch.fma_packed_f32x2(
                            (w0, w1), (ckv_f32, ckv_f32),
                            (out_regs_r0[reg_idx], out_regs_r1[reg_idx]))

    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))
        range_start(probe, cutlass.Int32(0), cutlass.Int32(3), sm_val, TAGS["write_reduce"])

    for tile_idx in range(num_tile_n):
        reg_base = tile_idx * iters_per_tile * vec_size  # 0, 4, 8, 12
        for v in range(vec_size):
            n_col_local = lane_idx * vec_size + v          # 0..127
            smem_partial[warp_idx, 0, n_col_local] = out_regs_r0[reg_base + v]
            smem_partial[warp_idx, 1, n_col_local] = out_regs_r1[reg_base + v]
        cute.arch.sync_threads()

        n_global_base = tile_idx * n_tile
        for m in range(M_):
            for i in range(tidx, n_tile, num_threads):
                acc = cutlass.Float32(0)
                for w in range(num_warps):
                    acc += smem_partial[w, m, i]
                output[m, n_global_base + i] = acc

        if tile_idx < num_tile_n - 1:
            cute.arch.sync_threads()

    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(3))
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(0))
        range_finalize(probe, cutlass.Int32(0), cutlass.Int32(4))


# ══════════════════════════════════════════════════════════════════════════════
#  run helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run(label, jit_fn, M, K, N, num_threads) -> str:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {label}  M={M}  K={K}  N={N}  threads={num_threads}")

    torch.manual_seed(42)
    weights = torch.rand((M, K), device="cuda", dtype=torch.float32)
    ckv     = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    output  = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    probe   = torch.zeros((1, PROBE_COLS), dtype=torch.int64, device="cuda")

    weights_ = from_dlpack(weights, assumed_align=16)
    ckv_     = from_dlpack(ckv,     assumed_align=16)
    output_  = from_dlpack(output,  assumed_align=16)
    probe_   = from_dlpack(probe,   assumed_align=8)

    compiled = cute.compile(jit_fn, weights_, ckv_, output_, probe_)

    for _ in range(3):
        probe.zero_(); output.zero_()
        compiled(weights_, ckv_, output_, probe_)
    torch.cuda.synchronize()

    ref      = weights.float() @ ckv.float()
    ok       = torch.allclose(output, ref, atol=1e-2, rtol=1e-2)
    max_diff = (output - ref).abs().max().item()
    print(f"Correctness: {'PASS' if ok else 'FAIL'}  max_diff={max_diff:.6f}")

    probe.zero_(); output.zero_()
    compiled(weights_, ckv_, output_, probe_)
    torch.cuda.synchronize()

    p   = probe[0].cpu().tolist()
    cnt = int(p[0])
    probes = []
    for i in range(cnt):
        off    = PROBE_HEADER + i * PROBE_ENTRY
        tag_v  = int(p[off + 1])
        dur_ns = int(p[off + 3])
        name   = TAG_NAMES.get(tag_v, f"tag{tag_v}")
        us     = dur_ns / 1000.0
        probes.append({"phase": name, "us": us})
        print(f"  {name:12s}: {us:7.3f} µs")

    return json.dumps({
        "kernel": label, "M": M, "K": K, "N": N,
        "correct": ok, "max_diff": float(max_diff), "probes": probes,
    }, indent=2)


def run_tile2() -> str:
    return _run("output_simt_ffma2_tile2",
                output_gemv_ffma2_tile2_intra_jit, M, K, N, NUM_THREADS)

def run_tile4() -> str:
    return _run("output_simt_ffma2_tile4",
                output_gemv_ffma2_tile4_intra_jit, M, K, N, NUM_THREADS)

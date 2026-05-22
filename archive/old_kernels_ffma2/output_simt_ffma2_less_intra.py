"""
output_simt_ffma2_less_intra.py — Intra-kernel profiling for output_simt_ffma2_less.

Probes: total / load_w / gemv / copy_out
"""
import json
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass.cute.runtime import from_dlpack


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

TAGS      = {"total": 2, "load_w": 4, "gemv": 6, "copy_out": 8}
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

NUM_THREADS     = 512
NUM_WARPS       = NUM_THREADS // 32
VEC_SIZE        = 8
ITERS_PER_LANE  = N // (32 * VEC_SIZE)
NUM_ROUNDS      = K // NUM_WARPS


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


@cute.jit
def output_gemv_ffma2_less_intra_jit(
    weights: cute.Tensor,
    ckv:     cute.Tensor,
    output:  cute.Tensor,
    probe:   cute.Tensor,
):
    output_gemv_ffma2_less_intra_kernel(weights, ckv, output, probe).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1])


@cute.kernel
def output_gemv_ffma2_less_intra_kernel(
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
    vec_size:       cutlass.Constexpr = VEC_SIZE
    iters_per_lane: cutlass.Constexpr = ITERS_PER_LANE
    num_rounds:     cutlass.Constexpr = NUM_ROUNDS

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = cute.arch.lane_idx()
    wsize    = cute.arch.WARP_SIZE

    alloc = cutlass.utils.SmemAllocator()
    smem_weight = _smem(alloc, cutlass.Float32, (M_ * K_,), (1,), 16)
    smem_output = _smem(alloc, cutlass.Float32, (M_ * N_,), (1,), 16)

    # ── Probe: total + load_w start ───────────────────────────────────────────
    sm_val = smid_u32()
    if tidx == cutlass.Int32(0):
        range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])
        range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["load_w"])

    # ── Init: load weights + zero smem_output ─────────────────────────────────
    for col in range(tidx, K_, num_threads):
        smem_weight[col * 2 + 0] = weights[0, col]
        smem_weight[col * 2 + 1] = weights[1, col]
    for i in range(tidx, N_, num_threads):
        smem_output[i * 2 + 0] = cutlass.Float32(0)
        smem_output[i * 2 + 1] = cutlass.Float32(0)
    cute.arch.sync_threads()

    smem_w_vec2   = cute.zipped_divide(smem_weight, (2,))
    smem_out_vec2 = cute.zipped_divide(smem_output, (2,))
    ckv_          = cute.zipped_divide(ckv, (1, vec_size))

    # ── Probe: stop load_w, start gemv ────────────────────────────────────────
    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(1))
        range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["gemv"])

    # ── GEMV: LDS.64 → FFMA2 → STS.64 on smem_output (racy) ─────────────────
    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < K_:
            w_frag = smem_w_vec2[(None,), (sparse_idx,)].load()
            w0 = w_frag[0]
            w1 = w_frag[1]
            ckv_row = ckv_[(0, None), (sparse_idx, None)]

            for it in range(iters_per_lane):
                rest_idx = it * wsize + lane_idx
                ckv_vec  = ckv_row[None, rest_idx].load()
                for v in range(vec_size):
                    n_col   = (it * wsize + lane_idx) * vec_size + v
                    ckv_f32 = cutlass.Float32(ckv_vec[v])
                    out_pair = smem_out_vec2[(None,), (n_col,)].load()
                    r0 = out_pair[0]
                    r1 = out_pair[1]
                    r0, r1 = cute.arch.fma_packed_f32x2(
                        (w0, w1), (ckv_f32, ckv_f32), (r0, r1))
                    smem_output[n_col * 2 + 0] = r0
                    smem_output[n_col * 2 + 1] = r1

    cute.arch.sync_threads()

    # ── Probe: stop gemv, start copy_out ──────────────────────────────────────
    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))
        range_start(probe, cutlass.Int32(0), cutlass.Int32(3), sm_val, TAGS["copy_out"])

    # ── Copy smem_output → global ─────────────────────────────────────────────
    for i in range(tidx, N_, num_threads):
        output[0, i] = smem_output[i * 2 + 0]
        output[1, i] = smem_output[i * 2 + 1]

    # ── Probe: stop copy_out + total, finalize ────────────────────────────────
    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(3))
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(0))
        range_finalize(probe, cutlass.Int32(0), cutlass.Int32(4))


def run_single() -> str:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: output_simt_ffma2_less  M={M}  K={K}  N={N}  threads={NUM_THREADS}")

    torch.manual_seed(42)
    weights = torch.rand((M, K), device="cuda", dtype=torch.float32)
    ckv     = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    output  = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    probe   = torch.zeros((1, PROBE_COLS), dtype=torch.int64, device="cuda")

    weights_ = from_dlpack(weights, assumed_align=16)
    ckv_     = from_dlpack(ckv,     assumed_align=16)
    output_  = from_dlpack(output,  assumed_align=16)
    probe_   = from_dlpack(probe,   assumed_align=8)

    compiled = cute.compile(output_gemv_ffma2_less_intra_jit, weights_, ckv_, output_, probe_)

    for _ in range(3):
        probe.zero_()
        compiled(weights_, ckv_, output_, probe_)
    torch.cuda.synchronize()

    ref      = weights.float() @ ckv.float()
    ok       = torch.allclose(output, ref, atol=1e-2, rtol=1e-2)
    max_diff = (output - ref).abs().max().item()
    print(f"Correctness: {'PASS' if ok else 'FAIL'}  max_diff={max_diff:.6f}")

    probe.zero_()
    compiled(weights_, ckv_, output_, probe_)
    torch.cuda.synchronize()

    p   = probe[0].cpu().tolist()
    cnt = int(p[0])
    probes = []
    for i in range(cnt):
        off    = PROBE_HEADER + i * PROBE_ENTRY
        sm_v   = int(p[off + 0])
        tag_v  = int(p[off + 1])
        dur_ns = int(p[off + 3])
        name   = TAG_NAMES.get(tag_v, f"tag{tag_v}")
        us     = dur_ns / 1000.0
        probes.append({"phase": name, "sm": sm_v, "us": us})
        print(f"  {name:8s}: {us:7.3f} µs")

    result = {
        "kernel": "output_simt_ffma2_less",
        "M": M, "K": K, "N": N,
        "num_threads": NUM_THREADS,
        "correct": ok,
        "max_diff": float(max_diff),
        "probes": probes,
    }
    return json.dumps(result, indent=2)

"""
output_simt_intra.py — Intra-kernel profiling for output_simt GEMV.

Adds globaltimer probes around:
  - total:  entire kernel
  - gemv:   FMA accumulation + smem partial store
  - reduce: cross-warp reduction to output

run_single() compiles, warms up, checks correctness, runs with probes,
and returns a JSON string with phase durations in µs.
"""
import json
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass.cute.runtime import from_dlpack


# ── Probe infra (matches draftv4_hist_pdl_intra.py) ──────────────────────────

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


PROBE_HEADER = 1                                      # probe[row, 0] = entry count
PROBE_ENTRY  = 4                                      # sm, tag, t_start_ns, dur_ns
MAX_ENTRIES  = 5                                      # total, load_w, gemv, reduce + spare
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY  # 21

TAGS      = {"total": 2, "load_w": 4, "gemv": 6, "reduce": 8}
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


# ── Constants ─────────────────────────────────────────────────────────────────

K = 256
N = 512

NUM_THREADS     = 1024
NUM_WARPS       = NUM_THREADS // 32   # 32
VEC_SIZE        = 8                   # 8 × BF16 = 128-bit load
ITERS_PER_LANE  = N // (32 * VEC_SIZE)  # 2
NUM_ROUNDS      = (K + NUM_WARPS - 1) // NUM_WARPS  # 8


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


# ── Instrumented kernel ───────────────────────────────────────────────────────

@cute.jit
def output_gemv_intra_jit(
    weights: cute.Tensor,
    ckv:     cute.Tensor,
    output:  cute.Tensor,
    probe:   cute.Tensor,
):
    output_gemv_intra_kernel(weights, ckv, output, probe).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1])


@cute.kernel
def output_gemv_intra_kernel(
    weights: cute.Tensor,   # (K,)       float32
    ckv:     cute.Tensor,   # (K, N)     BF16
    output:  cute.Tensor,   # (N,)       float32
    probe:   cute.Tensor,   # (1, PROBE_COLS)  int64
):
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
    smem_weight  = _smem(alloc, cutlass.Float32, (K_,),           (1,),    16)
    smem_partial = _smem(alloc, cutlass.Float32, (num_warps, N_), (N_, 1), 16)

    ckv_ = cute.zipped_divide(ckv, (1, vec_size))

    out_regs = cute.make_rmem_tensor(
        cute.make_layout((iters_per_lane * vec_size,), stride=(1,)),
        cutlass.Float32,
    )
    for i in range(iters_per_lane * vec_size):
        out_regs[i] = cutlass.Float32(0)

    # ── Probe: start total + load_w (thread 0) ───────────────────────────────
    sm_val = smid_u32()
    if tidx == cutlass.Int32(0):
        range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])
        range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["load_w"])

    # ── Load weights → smem_weight ────────────────────────────────────────────
    for i in range(tidx, K_, num_threads):
        smem_weight[i] = weights[i]
    cute.arch.sync_threads()

    # ── Probe: stop load_w, start gemv ───────────────────────────────────────
    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(1))
        range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["gemv"])

    # ── GEMV loop: each warp processes one row of ckv per round ──────────────
    for round_idx in range(num_rounds):
        sparse_idx = round_idx * num_warps + warp_idx
        if sparse_idx < K_:
            e = smem_weight[sparse_idx]
            ckv_row = ckv_[(0, None), (sparse_idx, None)]
            for it in range(iters_per_lane):
                rest_idx = it * wsize + lane_idx
                ckv_vec = ckv_row[None, rest_idx].load()
                for v in range(vec_size):
                    out_regs[it * vec_size + v] += e * cutlass.Float32(ckv_vec[v])

    # ── Write warp partial sums to smem ──────────────────────────────────────
    for it in range(iters_per_lane):
        for v in range(vec_size):
            smem_partial[warp_idx, (it * wsize + lane_idx) * vec_size + v] = \
                out_regs[it * vec_size + v]

    cute.arch.sync_threads()

    # ── Probe: stop gemv, start reduce (thread 0) ────────────────────────────
    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))
        range_start(probe, cutlass.Int32(0), cutlass.Int32(3), sm_val, TAGS["reduce"])

    # ── Reduce across all warps → final output ────────────────────────────────
    for i in range(tidx, N_, num_threads):
        acc = cutlass.Float32(0)
        for w in range(num_warps):
            acc += smem_partial[w, i]
        output[i] = acc

    cute.arch.sync_threads()

    # ── Probe: stop reduce, stop total, finalize (thread 0) ──────────────────
    if tidx == cutlass.Int32(0):
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(3))
        range_stop(probe, cutlass.Int32(0), cutlass.Int32(0))
        range_finalize(probe, cutlass.Int32(0), cutlass.Int32(4))


# ── run_single ────────────────────────────────────────────────────────────────

def run_single() -> str:
    import torch
    from cutlass.cute.runtime import from_dlpack
    import cutlass.cute as cute

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: output_simt  K={K}  N={N}  threads={NUM_THREADS}")

    torch.manual_seed(42)
    weights = torch.rand((K,), device="cuda", dtype=torch.float32)
    ckv     = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    output  = torch.zeros((N,), device="cuda", dtype=torch.float32)
    probe   = torch.zeros((1, PROBE_COLS), dtype=torch.int64, device="cuda")

    weights_ = from_dlpack(weights, assumed_align=16)
    ckv_     = from_dlpack(ckv,     assumed_align=16)
    output_  = from_dlpack(output,  assumed_align=16)
    probe_   = from_dlpack(probe,   assumed_align=8)

    compiled = cute.compile(output_gemv_intra_jit, weights_, ckv_, output_, probe_)

    # Warmup
    for _ in range(3):
        probe.zero_()
        compiled(weights_, ckv_, output_, probe_)
    torch.cuda.synchronize()

    # Correctness check
    ref = (weights.float().unsqueeze(0) @ ckv.float()).squeeze(0)
    ok      = torch.allclose(output, ref, atol=1e-2, rtol=1e-2)
    max_diff = (output - ref).abs().max().item()
    print(f"Correctness: {'PASS' if ok else 'FAIL'}  max_diff={max_diff:.6f}")

    # Final instrumented run
    probe.zero_()
    compiled(weights_, ckv_, output_, probe_)
    torch.cuda.synchronize()

    # Parse probe
    p    = probe[0].cpu().tolist()
    cnt  = int(p[0])
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
        "kernel": "output_simt",
        "K": K, "N": N,
        "num_threads": NUM_THREADS,
        "correct": ok,
        "max_diff": float(max_diff),
        "probes": probes,
    }
    return json.dumps(result, indent=2)

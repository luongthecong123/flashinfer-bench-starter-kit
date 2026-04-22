"""
output_simt_ffma2_stages.py — Staged FFMA2 GEMV with dedicated smem slots per stage.

Design
------
  Full smem_partial (num_warps, M, N) = 64 KB allocated once.
  Split into num_stages dedicated slots along N:  slot s → [:, :, s*stage_dim:(s+1)*stage_dim].

  For each stage s:
    1. GEMV over all K for stage s's N-column range  → vec_size regs per row (iters_per_stage=1)
    2. Write regs to smem_partial[:, :, s*stage_dim:(s+1)*stage_dim]  (unique slot, no conflict)
    3. sync_threads()  — ensures all warps wrote before reduce
    4. Reduce across warps → output[:, s*stage_dim:(s+1)*stage_dim]  (gmem stores posted)
    [no sync] — gmem stores overlap with next stage's GEMV; next stage writes a different slot.

  Total syncs: num_stages (vs 1 for baseline).
  Register pressure per row: vec_size (8 or 4) vs 16 for baseline.

vec_size = stage_dim // 32  (guarantees iters_per_stage = 1 always)
  num_stages=2 → stage_dim=256  vec_size=8  (128-bit CKV loads, same as baseline)
  num_stages=4 → stage_dim=128  vec_size=4  ( 64-bit CKV loads)

Probe phases
  total       : entire kernel
  load_w      : weight load to smem
  stages_loop : entire stage for-loop (all stages: GEMV + write + sync + reduce)

Usage
-----
    kernel = OutputSIMTFfma2Stages(num_stages=2)
    compiled = cute.compile(kernel, weights_, ckv_, output_, probe_)
    compiled(weights_, ckv_, output_, probe_)
"""

import json
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass.cute.runtime import from_dlpack


# ── Probe infra ────────────────────────────────────────────────────────────────

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

TAGS      = {"total": 2, "load_w": 4, "stages_loop": 6}
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


def _smem(allocator, dtype, shape, stride, align):
    return allocator.allocate_tensor(dtype, cute.make_layout(shape, stride=stride), align, None)


# ── Default problem size ───────────────────────────────────────────────────────

_M           = 2
_K           = 128
_N           = 512
_NUM_THREADS = 512


# ══════════════════════════════════════════════════════════════════════════════
#  Class: OutputSIMTFfma2Stages
# ══════════════════════════════════════════════════════════════════════════════

class OutputSIMTFfma2Stages:
    """
    SIMT FFMA2 GEMV with N split into num_stages sequential passes.

    Full smem_partial (num_warps, M, N) = 64 KB allocated once.
    Each stage owns a dedicated N-slice — no smem conflict between stages.

    Per-stage loop:
      GEMV → write smem slot → sync → reduce → gmem  [no sync after]
      Next stage GEMV starts immediately; gmem stores can overlap.

    Parameters
    ----------
    num_stages : int
        2 or 4 typical. vec_size = stage_dim // 32 (always iters_per_stage=1).
    M, K, N : int
        Problem dimensions. Default: (2, 128, 512).
    num_threads : int
        CTA thread count. Default: 512 (16 warps).
    """

    def __init__(
        self,
        num_stages:  int,
        M:           int = _M,
        K:           int = _K,
        N:           int = _N,
        num_threads: int = _NUM_THREADS,
    ):
        assert N % num_stages == 0, \
            f"N={N} must be divisible by num_stages={num_stages}"
        stage_dim = N // num_stages
        assert stage_dim % 32 == 0, \
            f"stage_dim={stage_dim} must be divisible by 32 (warp size)"

        self.M           = M
        self.K           = K
        self.N           = N
        self.num_threads = num_threads
        self.num_warps   = num_threads // 32
        self.num_rounds  = K // (num_threads // 32)
        self.num_stages  = num_stages
        self.stage_dim   = stage_dim
        # num_stages=2 → vec_size=8 (128-bit LDG, same as baseline)
        # num_stages=4 → vec_size=4 ( 64-bit LDG)
        self.vec_size    = stage_dim // 32   # iters_per_stage=1 always

    # ── JIT host wrapper ──────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        weights: cute.Tensor,   # (M, K) fp32
        ckv:     cute.Tensor,   # (K, N) bf16
        output:  cute.Tensor,   # (M, N) fp32
        probe:   cute.Tensor,   # (1, PROBE_COLS) int64
    ):
        self.kernel(weights, ckv, output, probe).launch(
            grid=[1, 1, 1], block=[self.num_threads, 1, 1])

    # ── Kernel ────────────────────────────────────────────────────────────────

    @cute.kernel
    def kernel(
        self,
        weights: cute.Tensor,
        ckv:     cute.Tensor,
        output:  cute.Tensor,
        probe:   cute.Tensor,
    ):
        M_:          cutlass.Constexpr = self.M
        K_:          cutlass.Constexpr = self.K
        N_:          cutlass.Constexpr = self.N
        num_threads: cutlass.Constexpr = self.num_threads
        num_warps:   cutlass.Constexpr = self.num_warps
        num_rounds:  cutlass.Constexpr = self.num_rounds
        num_stages:  cutlass.Constexpr = self.num_stages
        stage_dim:   cutlass.Constexpr = self.stage_dim
        vec_size:    cutlass.Constexpr = self.vec_size

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        wsize    = cute.arch.WARP_SIZE   # 32

        alloc = cutlass.utils.SmemAllocator()

        # smem_weight: (M*K,)  stride=(1,)  = 1 KB  — interleaved M rows
        smem_weight = _smem(alloc, cutlass.Float32, (M_ * K_,), (1,), 16)

        # smem_partial: full (num_warps, M, N) = 64 KB.
        # Stage s owns columns [s*stage_dim, (s+1)*stage_dim) — no overlap, no conflicts.
        smem_partial = _smem(alloc, cutlass.Float32,
                             (num_warps, M_, N_), (M_ * N_, N_, 1), 16)

        sm_val = smid_u32()
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])
            range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["load_w"])

        # ── Load weights to smem ──────────────────────────────────────────────
        for col in range(tidx, K_, num_threads):
            smem_weight[col * 2 + 0] = weights[0, col]
            smem_weight[col * 2 + 1] = weights[1, col]
        cute.arch.sync_threads()

        smem_w_vec2 = cute.zipped_divide(smem_weight, (2,))
        ckv_        = cute.zipped_divide(ckv, (1, vec_size))

        if tidx == cutlass.Int32(0):
            range_stop(probe,  cutlass.Int32(0), cutlass.Int32(1))
            range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["stages_loop"])

        # Output regs: only vec_size per row live at once (iters_per_stage=1).
        out_regs_r0 = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        out_regs_r1 = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

        # ── Stage loop ────────────────────────────────────────────────────────
        for stage in range(num_stages):
            stage_offset = stage * stage_dim

            # Zero accumulators for this stage.
            for v in range(vec_size):
                out_regs_r0[v] = cutlass.Float32(0)
                out_regs_r1[v] = cutlass.Float32(0)

            # GEMV: all K rounds, stage s's N-column range.
            # rest_idx = stage*32 + lane_idx → maps to N cols [stage_offset..stage_offset+stage_dim).
            for round_idx in range(num_rounds):
                sparse_idx = round_idx * num_warps + warp_idx
                if sparse_idx < K_:
                    w_frag = smem_w_vec2[(None,), (sparse_idx,)].load()
                    w0 = w_frag[0]
                    w1 = w_frag[1]
                    ckv_row = ckv_[(0, None), (sparse_idx, None)]
                    rest_idx = stage * wsize + lane_idx
                    ckv_vec = ckv_row[None, rest_idx].load()
                    for v in range(vec_size):
                        ckv_f32 = cutlass.Float32(ckv_vec[v])
                        out_regs_r0[v], out_regs_r1[v] = \
                            cute.arch.fma_packed_f32x2(
                                (w0, w1), (ckv_f32, ckv_f32),
                                (out_regs_r0[v], out_regs_r1[v]))

            # Write regs to dedicated smem slot (unique per stage → no conflict with
            # any other stage's read/write even without a preceding sync).
            for v in range(vec_size):
                n_col = stage_offset + lane_idx * vec_size + v
                smem_partial[warp_idx, 0, n_col] = out_regs_r0[v]
                smem_partial[warp_idx, 1, n_col] = out_regs_r1[v]

            # Sync: all warps wrote their smem slot for this stage → reduce can start.
            cute.arch.sync_threads()

            # Reduce across warps → gmem.
            # Gmem stores posted here; no sync after → next stage GEMV begins
            # while stores are still in flight (hardware overlap).
            for m in range(M_):
                for i in range(tidx, stage_dim, num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_warps):
                        acc += smem_partial[w, m, stage_offset + i]
                    output[m, stage_offset + i] = acc

            # No sync: next stage GEMV doesn't access smem_partial,
            # and next stage writes to a non-overlapping smem slot.

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(0))
            range_finalize(probe, cutlass.Int32(0), cutlass.Int32(3))

# ══════════════════════════════════════════════════════════════════════════════
#  Run helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_intra(num_stages: int) -> str:
    """Compile, warm-up, correctness-check, then profile one launch. Returns JSON."""
    label = f"output_simt_ffma2_stages{num_stages}"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {label}  num_stages={num_stages}  "
          f"M={_M}  K={_K}  N={_N}  threads={_NUM_THREADS}")

    kernel = OutputSIMTFfma2Stages(num_stages=num_stages)
    smem_kb = kernel.num_warps * kernel.M * kernel.N * 4 // 1024
    print(f"  vec_size={kernel.vec_size}  stage_dim={kernel.stage_dim}  "
          f"smem_partial={smem_kb} KB")

    torch.manual_seed(42)
    weights = torch.rand((_M, _K), device="cuda", dtype=torch.float32)
    ckv     = torch.randn((_K, _N), device="cuda", dtype=torch.bfloat16)
    output  = torch.zeros((_M, _N), device="cuda", dtype=torch.float32)
    probe   = torch.zeros((1, PROBE_COLS), dtype=torch.int64, device="cuda")

    weights_ = from_dlpack(weights, assumed_align=16)
    ckv_     = from_dlpack(ckv,     assumed_align=16)
    output_  = from_dlpack(output,  assumed_align=16)
    probe_   = from_dlpack(probe,   assumed_align=8)

    compiled = cute.compile(kernel, weights_, ckv_, output_, probe_)

    # Warm-up
    for _ in range(3):
        probe.zero_(); output.zero_()
        compiled(weights_, ckv_, output_, probe_)
    torch.cuda.synchronize()

    # Correctness
    ref      = weights.float() @ ckv.float()
    ok       = torch.allclose(output, ref, atol=1e-2, rtol=1e-2)
    max_diff = (output - ref).abs().max().item()
    print(f"Correctness: {'PASS' if ok else 'FAIL'}  max_diff={max_diff:.6f}")

    # Profile
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
        "kernel": label, "num_stages": num_stages,
        "M": _M, "K": _K, "N": _N,
        "vec_size": kernel.vec_size, "stage_dim": kernel.stage_dim,
        "correct": ok, "max_diff": float(max_diff),
        "probes": probes,
    }, indent=2)


def run_stages2() -> str:
    return run_intra(num_stages=2)


def run_stages4() -> str:
    return run_intra(num_stages=4)

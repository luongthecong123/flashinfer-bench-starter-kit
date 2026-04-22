"""
output_simt_ffma2_stages_smem.py — Staged FFMA2 GEMV with CKV (128×512) preloaded to smem.

Design
------
  Allocate smem_ckv (K=128, N=512) BF16 = 128 KB before the stage loop.
  Allocate smem_partial (num_warps, M, stage_dim) = (16, 2, 128) float32 = 16 KB — reused
  across all 4 stages (no dedicated slots; overwrite each iteration).

  Smem budget (B200: 228 KB max):
    smem_weight:  1 KB   (M*K float32, interleaved)
    smem_ckv:   128 KB   (K*N BF16, row-major)
    smem_partial: 16 KB  (num_warps * M * stage_dim float32, reused)
    Total:       145 KB  ✓

  CKV load:
    With num_threads = N = 512, thread `tidx` loads smem_ckv[k, tidx] = ckv[k, tidx]
    for all K=128 rows — perfectly coalesced (32 consecutive BF16 per warp per row).

  Stage loop (4 stages, 2 syncs each = 8 total vs 4 in stages=4 baseline):
    1. Zero regs
    2. GEMV over all K, reading from smem_ckv (L1 smem, ~4 cycle latency)
    3. sync_threads()  — ensure previous stage's reduce is done before overwriting smem_partial
    4. Write vec_size regs to smem_partial[:, :, 0:stage_dim]  (overwrite)
    5. sync_threads()  — ensure all warps wrote before reduce
    6. Reduce across warps → output[:, stage_offset:stage_offset+stage_dim]

  Trade-offs vs stages=4 baseline (gmem CKV, dedicated smem slots):
    + GEMV reads from L1 smem (~4 cyc) instead of L2/HBM
    + smem_partial 4× smaller (16 KB vs 64 KB) — better occupancy for smem_partial
    − 128 KB smem_ckv overhead (reduces SM wave occupancy)
    − 4 extra syncs (8 vs 4) — ~400 ns cost on B200

Probe phases
  total       : entire kernel
  load_w      : weight→smem load
  load_ckv    : CKV→smem load
  stages_loop : all 4 stage iterations

Usage
-----
    kernel   = OutputSIMTFfma2StagesSmem(num_stages=4)
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

# 4 phases: total, load_w, load_ckv, stages_loop
TAGS      = {"total": 2, "load_w": 4, "load_ckv": 6, "stages_loop": 8}
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
#  Class: OutputSIMTFfma2StagesSmem
# ══════════════════════════════════════════════════════════════════════════════

class OutputSIMTFfma2StagesSmem:
    """
    SIMT FFMA2 GEMV — full CKV (K×N) preloaded to smem, smem_partial reused per stage.

    Smem layout:
      smem_weight  (M*K,)            float32  = 1 KB   (interleaved M rows)
      smem_ckv     (K, N)            bf16     = 128 KB  (row-major)
      smem_partial (num_warps, M, stage_dim) float32  = 16 KB  (reused across stages)

    Parameters
    ----------
    num_stages : int
        Number of N-column stages. Default 4 → stage_dim=128, vec_size=4.
    """

    def __init__(
        self,
        num_stages:  int = 4,
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
        assert N == num_threads, \
            f"N={N} must equal num_threads={num_threads} for the CKV load pattern"

        self.M           = M
        self.K           = K
        self.N           = N
        self.num_threads = num_threads
        self.num_warps   = num_threads // 32
        self.num_rounds  = K // (num_threads // 32)   # K / num_warps = 8
        self.num_stages  = num_stages
        self.stage_dim   = stage_dim
        self.vec_size    = stage_dim // 32            # =4 for num_stages=4

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

        # smem_weight: (M*K,) float32 = 1 KB — interleaved M rows
        smem_weight = _smem(alloc, cutlass.Float32, (M_ * K_,), (1,), 16)

        # smem_ckv: (K, N) BF16 = 128 KB — row-major
        smem_ckv = _smem(alloc, cutlass.BFloat16, (K_, N_), (N_, 1), 16)

        # smem_partial: (num_warps, M, stage_dim) float32 = 16 KB — reused each stage
        smem_partial = _smem(alloc, cutlass.Float32,
                             (num_warps, M_, stage_dim), (M_ * stage_dim, stage_dim, 1), 16)

        sm_val = smid_u32()
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])
            range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["load_w"])

        # ── Load weights to smem (interleaved M rows) ─────────────────────────
        for col in range(tidx, K_, num_threads):
            smem_weight[col * 2 + 0] = weights[0, col]
            smem_weight[col * 2 + 1] = weights[1, col]

        if tidx == cutlass.Int32(0):
            range_stop(probe,  cutlass.Int32(0), cutlass.Int32(1))
            range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["load_ckv"])

        # ── Load CKV to smem ──────────────────────────────────────────────────
        # Thread tidx ∈ [0, N) loads column tidx for all K rows.
        # Each warp covers 32 consecutive N-columns per K-row — perfectly coalesced.
        for k in range(K_):
            smem_ckv[k, tidx] = ckv[k, tidx]

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))

        # Barrier: all of smem_weight and smem_ckv are ready.
        cute.arch.sync_threads()

        # Vectorized views for GEMV
        smem_w_vec2 = cute.zipped_divide(smem_weight, (2,))
        smem_ckv_   = cute.zipped_divide(smem_ckv, (1, vec_size))

        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(3), sm_val, TAGS["stages_loop"])

        # Accumulators: vec_size regs per row (iters_per_stage = 1).
        out_regs_r0 = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        out_regs_r1 = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

        # ── Stage loop ────────────────────────────────────────────────────────
        for stage in range(num_stages):
            stage_offset = stage * stage_dim

            # 1. Zero accumulators for this stage.
            for v in range(vec_size):
                out_regs_r0[v] = cutlass.Float32(0)
                out_regs_r1[v] = cutlass.Float32(0)

            # 2. GEMV over all K, reading from smem_ckv (L1 smem, ~4 cycle latency).
            #    rest_idx = stage*32 + lane_idx  →  N-group within [stage_offset, stage_offset+stage_dim).
            for round_idx in range(num_rounds):
                sparse_idx = round_idx * num_warps + warp_idx
                w_frag = smem_w_vec2[(None,), (sparse_idx,)].load()
                w0 = w_frag[0]
                w1 = w_frag[1]
                ckv_row  = smem_ckv_[(0, None), (sparse_idx, None)]
                rest_idx = stage * wsize + lane_idx
                ckv_vec  = ckv_row[None, rest_idx].load()
                for v in range(vec_size):
                    ckv_f32 = cutlass.Float32(ckv_vec[v])
                    out_regs_r0[v], out_regs_r1[v] = \
                        cute.arch.fma_packed_f32x2(
                            (w0, w1), (ckv_f32, ckv_f32),
                            (out_regs_r0[v], out_regs_r1[v]))

            # 3. Sync: ensure previous stage's reduce (reads smem_partial) is done
            #    before we overwrite smem_partial.  Harmless no-op for stage 0.
            cute.arch.sync_threads()

            # 4. Write regs to smem_partial (overwrite the single reused slot).
            for v in range(vec_size):
                n_local = lane_idx * vec_size + v
                smem_partial[warp_idx, 0, n_local] = out_regs_r0[v]
                smem_partial[warp_idx, 1, n_local] = out_regs_r1[v]

            # 5. Sync: ensure all warps wrote before reduce reads.
            cute.arch.sync_threads()

            # 6. Reduce across warps → output for this stage's N-slice.
            for m in range(M_):
                for i in range(tidx, stage_dim, num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_warps):
                        acc += smem_partial[w, m, i]
                    output[m, stage_offset + i] = acc

            # No sync after reduce: next stage's GEMV reads smem_ckv (not smem_partial),
            # and the sync at the TOP of the next iteration guards the write to smem_partial.

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(3))
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(0))
            range_finalize(probe, cutlass.Int32(0), cutlass.Int32(4))


# ══════════════════════════════════════════════════════════════════════════════
#  Run helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_intra(num_stages: int = 4) -> str:
    """Compile, warm-up, correctness-check, profile one launch. Returns JSON."""
    label = f"output_simt_ffma2_stages_smem{num_stages}"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {label}  num_stages={num_stages}  "
          f"M={_M}  K={_K}  N={_N}  threads={_NUM_THREADS}")

    kernel = OutputSIMTFfma2StagesSmem(num_stages=num_stages)
    smem_ckv_kb     = kernel.K * kernel.N * 2 // 1024
    smem_partial_kb = kernel.num_warps * kernel.M * kernel.stage_dim * 4 // 1024
    print(f"  vec_size={kernel.vec_size}  stage_dim={kernel.stage_dim}")
    print(f"  smem_ckv={smem_ckv_kb} KB  smem_partial={smem_partial_kb} KB  "
          f"total_smem={1 + smem_ckv_kb + smem_partial_kb} KB")

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
        "smem_ckv_kb": smem_ckv_kb, "smem_partial_kb": smem_partial_kb,
        "correct": ok, "max_diff": float(max_diff),
        "probes": probes,
    }, indent=2)


def run_smem4() -> str:
    return run_intra(num_stages=4)

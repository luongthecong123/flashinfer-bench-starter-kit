"""
output_simt_ffma2_stages_smem_cpasync.py — Same as stages_smem but loads CKV
via cp.async 128-bit (8×BF16 = 128 bits) instead of scalar LDS stores.

Differences vs output_simt_ffma2_stages_smem.py
------------------------------------------------
  CKV load: replaced  `smem_ckv[k, tidx] = ckv[k, tidx]`  (1×BF16 per thread)
       with cp.async  128-bit transactions  (8×BF16 per transaction).

  Load pattern (warp/lane 2D striding — avoids runtime loop bounds):
    outer loop: k_rnd ∈ [0, K//num_warps)  → k = k_rnd*num_warps + warp_idx
    inner loop: g_rnd ∈ [0, N//vec_ckv//32) → grp = g_rnd*32 + lane_idx
    Each thread: (K//num_warps) × (N//vec_ckv//32) = 8×2 = 16 cp.async calls
    Per call: 8×BF16 = 128 bits  ✓

  After the loop:
    cp_async_commit_group() — seal all pending transactions into one group
    cp_async_wait_group(0)  — wait until 0 groups remain outstanding (per thread)
    sync_threads()          — broadcast to all threads

  Everything else (GEMV + reduce) is identical to stages_smem.

Smem budget (B200: 228 KB max):
    smem_weight:  1 KB   (M*K float32, interleaved)
    smem_ckv:   128 KB   (K*N BF16, plain row-major — no swizzle)
    smem_partial: 16 KB  (num_warps × M × stage_dim float32, reused)
    Total:       145 KB  ✓

Probe phases
  total       : entire kernel
  load_w      : weight→smem (scalar, same as before)
  load_ckv    : CKV→smem via cp.async
  stages_loop : all 4 stage iterations (GEMV from smem + reduce)
"""

import json
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync


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
#  Class: OutputSIMTFfma2StagesSmemCpAsync
# ══════════════════════════════════════════════════════════════════════════════

class OutputSIMTFfma2StagesSmemCpAsync:
    """
    SIMT FFMA2 GEMV — CKV preloaded to smem via cp.async 128-bit (8×BF16).

    Load pattern:
      vec_ckv = 8  (128-bit per cp.async call)
      Each thread issues K//num_warps × (N//vec_ckv)//32 = 8×2 = 16 calls.
    """

    def __init__(
        self,
        num_stages:  int = 4,
        M:           int = _M,
        K:           int = _K,
        N:           int = _N,
        num_threads: int = _NUM_THREADS,
    ):
        assert N % num_stages == 0
        stage_dim = N // num_stages
        assert stage_dim % 32 == 0
        assert N == num_threads, \
            f"N={N} must equal num_threads={num_threads}"

        self.M           = M
        self.K           = K
        self.N           = N
        self.num_threads = num_threads
        self.num_warps   = num_threads // 32
        self.num_rounds  = K // (num_threads // 32)   # = 8
        self.num_stages  = num_stages
        self.stage_dim   = stage_dim
        self.vec_size    = stage_dim // 32            # = 4 for num_stages=4
        self.vec_ckv     = 8                          # 8×BF16 = 128-bit cp.async

    # ── JIT host wrapper ──────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        weights: cute.Tensor,
        ckv:     cute.Tensor,
        output:  cute.Tensor,
        probe:   cute.Tensor,
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
        vec_ckv:     cutlass.Constexpr = self.vec_ckv

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        wsize    = cute.arch.WARP_SIZE   # 32

        alloc = cutlass.utils.SmemAllocator()

        # smem_weight: (M*K,) float32 = 1 KB
        smem_weight = _smem(alloc, cutlass.Float32, (M_ * K_,), (1,), 16)

        # smem_ckv: (K, N) BF16 = 128 KB — plain row-major, no swizzle.
        # 16-byte alignment ensures each 8×BF16 cp.async destination is 16B aligned.
        smem_ckv = _smem(alloc, cutlass.BFloat16, (K_, N_), (N_, 1), 16)

        # smem_partial: (num_warps, M, stage_dim) float32 = 16 KB — reused per stage
        smem_partial = _smem(alloc, cutlass.Float32,
                             (num_warps, M_, stage_dim), (M_ * stage_dim, stage_dim, 1), 16)

        sm_val = smid_u32()
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])
            range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["load_w"])

        # ── Load weights to smem (interleaved M rows, scalar) ─────────────────
        for col in range(tidx, K_, num_threads):
            smem_weight[col * 2 + 0] = weights[0, col]
            smem_weight[col * 2 + 1] = weights[1, col]

        if tidx == cutlass.Int32(0):
            range_stop(probe,  cutlass.Int32(0), cutlass.Int32(1))
            range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["load_ckv"])

        # ── Load CKV to smem via cp.async 128-bit (8×BF16 per transaction) ───
        #
        # Each thread fires 16 cp.async calls (no commit inside the loop —
        # the hardware can pipeline all of them):
        #
        #   k_rnd ∈ [0, K//num_warps) = [0, 8)
        #     k = k_rnd * num_warps + warp_idx
        #   g_rnd ∈ [0, N//vec_ckv//wsize) = [0, 2)
        #     grp = g_rnd * wsize + lane_idx
        #
        # Covers: 16 warps × 8 K-rows per warp = 128 K rows  ✓
        #         32 lanes × 2 grps per lane   = 64 N-groups  ✓  (64×8=512 cols)
        #
        # Destination alignment: smem_ckv base is 16B aligned;
        #   byte offset = (k*N + grp*vec_ckv)*2 — each grp step = 16 bytes ✓

        copy_atom_ckv = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )

        ckv_vec      = cute.zipped_divide(ckv,      (1, vec_ckv))
        smem_ckv_vec = cute.zipped_divide(smem_ckv, (1, vec_ckv))

        for k_rnd in range(K_ // num_warps):              # = 8 iters, constexpr
            k = cutlass.Int32(k_rnd) * num_warps + warp_idx
            for g_rnd in range(N_ // vec_ckv // wsize):   # = 2 iters, constexpr
                grp = cutlass.Int32(g_rnd) * wsize + lane_idx
                cute.copy(
                    copy_atom_ckv,
                    ckv_vec[(0, None), (k, grp)],
                    smem_ckv_vec[(0, None), (k, grp)],
                )

        # Commit all pending cp.async into one group, then stall until complete.
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))

        # Barrier: smem_weight and smem_ckv are all ready.
        cute.arch.sync_threads()

        # Vectorized views for GEMV
        smem_w_vec2 = cute.zipped_divide(smem_weight, (2,))
        smem_ckv_   = cute.zipped_divide(smem_ckv, (1, vec_size))

        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(3), sm_val, TAGS["stages_loop"])

        out_regs_r0 = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        out_regs_r1 = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

        # ── Stage loop (unchanged from stages_smem) ───────────────────────────
        for stage in range(num_stages):
            stage_offset = stage * stage_dim

            for v in range(vec_size):
                out_regs_r0[v] = cutlass.Float32(0)
                out_regs_r1[v] = cutlass.Float32(0)

            for round_idx in range(num_rounds):
                sparse_idx = round_idx * num_warps + warp_idx
                w_frag = smem_w_vec2[(None,), (sparse_idx,)].load()
                w0 = w_frag[0]
                w1 = w_frag[1]
                ckv_row  = smem_ckv_[(0, None), (sparse_idx, None)]
                rest_idx = stage * wsize + lane_idx
                ckv_vec_gemv = ckv_row[None, rest_idx].load()
                for v in range(vec_size):
                    ckv_f32 = cutlass.Float32(ckv_vec_gemv[v])
                    out_regs_r0[v], out_regs_r1[v] = \
                        cute.arch.fma_packed_f32x2(
                            (w0, w1), (ckv_f32, ckv_f32),
                            (out_regs_r0[v], out_regs_r1[v]))

            # Sync: guard overwrite of smem_partial (previous stage may still reduce)
            cute.arch.sync_threads()

            for v in range(vec_size):
                n_local = lane_idx * vec_size + v
                smem_partial[warp_idx, 0, n_local] = out_regs_r0[v]
                smem_partial[warp_idx, 1, n_local] = out_regs_r1[v]

            # Sync: ensure all warps wrote before reduce
            cute.arch.sync_threads()

            for m in range(M_):
                for i in range(tidx, stage_dim, num_threads):
                    acc = cutlass.Float32(0)
                    for w in range(num_warps):
                        acc += smem_partial[w, m, i]
                    output[m, stage_offset + i] = acc

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(3))
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(0))
            range_finalize(probe, cutlass.Int32(0), cutlass.Int32(4))


# ══════════════════════════════════════════════════════════════════════════════
#  Run helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_intra(num_stages: int = 4) -> str:
    label = f"output_simt_ffma2_stages_smem_cpasync{num_stages}"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {label}  num_stages={num_stages}  "
          f"M={_M}  K={_K}  N={_N}  threads={_NUM_THREADS}")

    kernel = OutputSIMTFfma2StagesSmemCpAsync(num_stages=num_stages)
    smem_ckv_kb     = kernel.K * kernel.N * 2 // 1024
    smem_partial_kb = kernel.num_warps * kernel.M * kernel.stage_dim * 4 // 1024
    print(f"  vec_size={kernel.vec_size}  vec_ckv={kernel.vec_ckv}  stage_dim={kernel.stage_dim}")
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
        "kernel": label, "num_stages": num_stages,
        "M": _M, "K": _K, "N": _N,
        "vec_size": kernel.vec_size, "vec_ckv": kernel.vec_ckv,
        "stage_dim": kernel.stage_dim,
        "smem_ckv_kb": smem_ckv_kb, "smem_partial_kb": smem_partial_kb,
        "correct": ok, "max_diff": float(max_diff),
        "probes": probes,
    }, indent=2)


def run_smem_cpasync4() -> str:
    return run_intra(num_stages=4)

"""
output_simt_ffma2_stages_smem_cpasync_dsa.py — DSA-realistic variant of _cpasync.

Identical to output_simt_ffma2_stages_smem_cpasync.py EXCEPT:

  * Adds smem_sparse[K] (int32) and smem_num_valid (int32 scalar in smem).
  * Prologue (before cp.async):
      1. Load sparse_indices[k] -> smem_sparse[k] for k in [tidx, K, num_threads)
      2. Count valid (>=0) entries via warp+block reduce -> smem_num_valid
      3. sync_threads
      4. Clamp: for invalid smem_sparse[k] entries, write 0 so cp.async/GEMV
         can issue unconditionally (safe OOB-free FastGEMV sentinel pattern)
  * num_rounds = ceil(num_valid / num_warps) — early exit for both load and GEMV.
  * cp.async loop and GEMV loop both iterate `num_rounds` (not K//num_warps),
    with guard `if k_local < num_valid` to skip remainder slots.
  * Everything else unchanged.

Two test cases in run_dsa_cases():
  full:  sparse_indices = [0..127]       -- all K=128 valid
  short: sparse_indices = [0..63, -1..]  -- first 64 valid, rest -1
"""

import json
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
import math


# -- Probe infra ---------------------------------------------------------------

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


@cute.jit
def warp_reduce_add(val: cutlass.Int32, width: cutlass.Constexpr = 32) -> cutlass.Int32:
    for i in range(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


# -- Default problem size ------------------------------------------------------

_M           = 2
_K           = 128
_N           = 512
_NUM_THREADS = 512


# =============================================================================
#  Class
# =============================================================================

class OutputSIMTFfma2StagesSmemCpAsyncDSA:
    """
    Same as OutputSIMTFfma2StagesSmemCpAsync but:
      - CKV loaded indirectly via smem_sparse (cached sparse_indices)
      - num_valid reduces early-exit for cp.async + GEMV rounds
      - zero sentinels written for remainder slots (safe unconditional FMA)
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
        assert N == num_threads

        self.M           = M
        self.K           = K
        self.N           = N
        self.num_threads = num_threads
        self.num_warps   = num_threads // 32
        self.num_stages  = num_stages
        self.stage_dim   = stage_dim
        self.vec_size    = stage_dim // 32   # = 4
        self.vec_ckv     = 8                 # 8xBF16 = 128-bit

    @cute.jit
    def __call__(
        self,
        weights:        cute.Tensor,   # [M, K]    float32
        ckv_flat:       cute.Tensor,   # [pool, N] BF16
        sparse_indices: cute.Tensor,   # [K]       int32
        output:         cute.Tensor,   # [M, N]   float32
        probe:          cute.Tensor,
    ):
        self.kernel(weights, ckv_flat, sparse_indices, output, probe).launch(
            grid=[1, 1, 1], block=[self.num_threads, 1, 1])

    @cute.kernel
    def kernel(
        self,
        weights:        cute.Tensor,
        ckv_flat:       cute.Tensor,
        sparse_indices: cute.Tensor,
        output:         cute.Tensor,
        probe:          cute.Tensor,
    ):
        M_:          cutlass.Constexpr = self.M
        K_:          cutlass.Constexpr = self.K
        N_:          cutlass.Constexpr = self.N
        num_threads: cutlass.Constexpr = self.num_threads
        num_warps:   cutlass.Constexpr = self.num_warps
        num_stages:  cutlass.Constexpr = self.num_stages
        stage_dim:   cutlass.Constexpr = self.stage_dim
        vec_size:    cutlass.Constexpr = self.vec_size
        vec_ckv:     cutlass.Constexpr = self.vec_ckv

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        wsize    = cute.arch.WARP_SIZE

        alloc = cutlass.utils.SmemAllocator()

        # smem_weight: (M*K,) float32 = 1 KB
        smem_weight = _smem(alloc, cutlass.Float32, (M_ * K_,), (1,), 16)

        # smem_ckv: (K, N) BF16 = 128 KB
        smem_ckv = _smem(alloc, cutlass.BFloat16, (K_, N_), (N_, 1), 16)

        # smem_partial: (num_warps, M, stage_dim) float32 = 16 KB
        smem_partial = _smem(alloc, cutlass.Float32,
                             (num_warps, M_, stage_dim), (M_ * stage_dim, stage_dim, 1), 16)

        # smem_sparse: (K,) int32 — cached sparse_indices
        smem_sparse = _smem(alloc, cutlass.Int32, (K_,), (1,), 4)

        # smem_red: (num_warps,) int32 — warp partial sums for valid count
        smem_red = _smem(alloc, cutlass.Int32, (num_warps,), (1,), 4)

        sm_val = smid_u32()
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])
            range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["load_w"])

        # -- Prologue: load sparse_indices -> smem_sparse, count valid ---------
        #
        # Each thread: for k in [tidx, K, num_threads) load idx, accumulate cnt
        partial_valid = cutlass.Int32(0)
        for k in range(tidx, K_, num_threads):
            idx = sparse_indices[k]
            smem_sparse[k] = idx
            if idx >= cutlass.Int32(0):
                partial_valid = partial_valid + cutlass.Int32(1)

        # Block-wide reduce to get num_valid
        warp_sum = warp_reduce_add(partial_valid, width=32)
        if lane_idx == cutlass.Int32(0):
            smem_red[warp_idx] = warp_sum
        cute.arch.sync_threads()

        if warp_idx == cutlass.Int32(0):
            val = smem_red[lane_idx]
            block_sum = warp_reduce_add(val, width=num_warps)
            if lane_idx == cutlass.Int32(0):
                smem_red[0] = block_sum
        cute.arch.sync_threads()

        num_valid  = smem_red[0]
        num_rounds = (num_valid + num_warps - cutlass.Int32(1)) // num_warps

        # Clamp invalid smem_sparse entries to 0 (safe sentinel for FastGEMV):
        # only slots [num_valid .. num_rounds*num_warps) need clamping.
        for k in range(tidx, K_, num_threads):
            if smem_sparse[k] < cutlass.Int32(0):
                smem_sparse[k] = cutlass.Int32(0)
        cute.arch.sync_threads()

        # -- Load weights to smem ----------------------------------------------
        for col in range(tidx, K_, num_threads):
            smem_weight[col * 2 + 0] = weights[0, col]
            smem_weight[col * 2 + 1] = weights[1, col]

        if tidx == cutlass.Int32(0):
            range_stop(probe,  cutlass.Int32(0), cutlass.Int32(1))
            range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["load_ckv"])

        # -- cp.async: load num_rounds*num_warps rows from paged ckv_flat -----
        #
        # Warp w handles k_local = k_rnd*num_warps + w.
        # For k_local < num_valid: load ckv_flat[smem_sparse[k_local], :]
        # For k_local in [num_valid, num_rounds*num_warps): sentinel idx=0,
        #   smem_ckv[k_local, :] will be loaded from ckv_flat[0] but weight=0.

        copy_atom_ckv = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )

        ckv_flat_vec = cute.zipped_divide(ckv_flat, (1, vec_ckv))
        smem_ckv_vec = cute.zipped_divide(smem_ckv, (1, vec_ckv))

        for k_rnd in range(K_ // num_warps):   # = 8, constexpr upper bound
            k_local = cutlass.Int32(k_rnd) * num_warps + warp_idx
            if k_local < num_rounds * num_warps:
                idx = smem_sparse[k_local]     # already clamped to >= 0
                for g_rnd in range(N_ // vec_ckv // wsize):   # = 2, constexpr
                    grp = cutlass.Int32(g_rnd) * wsize + lane_idx
                    cute.copy(
                        copy_atom_ckv,
                        ckv_flat_vec[(0, None), (idx, grp)],
                        smem_ckv_vec[(0, None), (k_local, grp)],
                    )

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))

        cute.arch.sync_threads()

        # -- GEMV: num_rounds rounds, weight=0 sentinel ensures no contribution
        #    for k_local >= num_valid (smem_weight[k_local*2] comes from the
        #    original weights tensor which may be non-zero — so we rely on the
        #    fact that for k_local >= num_valid the contribution should be zero).
        #
        # To zero the weight for sentinel rows: we wrote smem_sparse[k]=0 but
        # smem_weight[k*2+{0,1}] still has real weight values.  Guard: only
        # accumulate if k_local < num_valid.

        smem_w_vec2 = cute.zipped_divide(smem_weight, (2,))
        smem_ckv_   = cute.zipped_divide(smem_ckv, (1, vec_size))

        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(3), sm_val, TAGS["stages_loop"])

        out_regs_r0 = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
        out_regs_r1 = cute.make_rmem_tensor(
            cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)

        for stage in range(num_stages):
            stage_offset = stage * stage_dim

            for v in range(vec_size):
                out_regs_r0[v] = cutlass.Float32(0)
                out_regs_r1[v] = cutlass.Float32(0)

            for round_idx in range(K_ // num_warps):   # constexpr upper bound
                k_local = cutlass.Int32(round_idx) * num_warps + warp_idx
                if k_local < num_rounds * num_warps:
                    if k_local < num_valid:
                        w_frag = smem_w_vec2[(None,), (k_local,)].load()
                        w0 = w_frag[0]
                        w1 = w_frag[1]
                        ckv_row      = smem_ckv_[(0, None), (k_local, None)]
                        rest_idx     = stage * wsize + lane_idx
                        ckv_vec_gemv = ckv_row[None, rest_idx].load()
                        for v in range(vec_size):
                            ckv_f32 = cutlass.Float32(ckv_vec_gemv[v])
                            out_regs_r0[v], out_regs_r1[v] = \
                                cute.arch.fma_packed_f32x2(
                                    (w0, w1), (ckv_f32, ckv_f32),
                                    (out_regs_r0[v], out_regs_r1[v]))

            cute.arch.sync_threads()

            for v in range(vec_size):
                n_local = lane_idx * vec_size + v
                smem_partial[warp_idx, 0, n_local] = out_regs_r0[v]
                smem_partial[warp_idx, 1, n_local] = out_regs_r1[v]

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


# =============================================================================
#  Two-case test
# =============================================================================

def run_dsa_cases(num_stages: int = 4, save_dir: str = "/tmp") -> dict:
    """
    Case A -- "full":  sparse_indices = [0..127]       (all 128 valid)
    Case B -- "short": sparse_indices = [0..63, -1..]  (64 valid, 64 padding)
    """
    import os
    label = f"output_simt_ffma2_stages_smem_cpasync_dsa{num_stages}"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {label}  M={_M}  K={_K}  N={_N}  threads={_NUM_THREADS}")

    kernel = OutputSIMTFfma2StagesSmemCpAsyncDSA(num_stages=num_stages)

    torch.manual_seed(42)
    weights  = torch.rand((_M, _K), device="cuda", dtype=torch.float32)
    ckv_flat = torch.randn((_K, _N), device="cuda", dtype=torch.bfloat16)

    si_full  = torch.arange(_K, device="cuda", dtype=torch.int32)

    SEQ_SHORT = 64
    si_short  = torch.full((_K,), -1, device="cuda", dtype=torch.int32)
    si_short[:SEQ_SHORT] = torch.arange(SEQ_SHORT, dtype=torch.int32)

    torch.save({"weights": weights.cpu(), "ckv_flat": ckv_flat.cpu(),
                "sparse_indices": si_full.cpu(), "seq_len": _K},
               os.path.join(save_dir, "dsa_case_full.pt"))
    torch.save({"weights": weights.cpu(), "ckv_flat": ckv_flat.cpu(),
                "sparse_indices": si_short.cpu(), "seq_len": SEQ_SHORT},
               os.path.join(save_dir, "dsa_case_short.pt"))
    print(f"Saved to {save_dir}/dsa_case_{{full,short}}.pt")

    output = torch.zeros((_M, _N), device="cuda", dtype=torch.float32)
    probe  = torch.zeros((1, PROBE_COLS), dtype=torch.int64, device="cuda")

    weights_  = from_dlpack(weights,  assumed_align=16)
    ckv_flat_ = from_dlpack(ckv_flat, assumed_align=16)
    output_   = from_dlpack(output,   assumed_align=16)
    probe_    = from_dlpack(probe,    assumed_align=8)

    si_full_  = from_dlpack(si_full, assumed_align=16)
    compiled  = cute.compile(kernel, weights_, ckv_flat_, si_full_, output_, probe_)

    results = {}
    for case_name, si_tensor, seq_len in [
        ("full",  si_full,  _K),
        ("short", si_short, SEQ_SHORT),
    ]:
        si_ = from_dlpack(si_tensor, assumed_align=16)

        for _ in range(3):
            probe.zero_(); output.zero_()
            compiled(weights_, ckv_flat_, si_, output_, probe_)
        torch.cuda.synchronize()

        valid_mask = (si_tensor >= 0)
        valid_idx  = si_tensor[valid_mask].long()
        ref      = weights[:, valid_mask].float() @ ckv_flat[valid_idx].float()
        ok       = torch.allclose(output, ref, atol=1e-1, rtol=1e-2)
        max_diff = (output - ref).abs().max().item()

        probe.zero_(); output.zero_()
        compiled(weights_, ckv_flat_, si_, output_, probe_)
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

        total_us = next((x["us"] for x in probes if x["phase"] == "total"), 0)
        print(f"\n[{case_name:5s}]  seq_len={seq_len:3d}  "
              f"{'PASS' if ok else f'FAIL(max_diff={max_diff:.4f})'}  "
              f"total={total_us:.3f} us")
        for p_ in probes:
            print(f"  {p_['phase']:12s}: {p_['us']:7.3f} us")

        results[case_name] = {
            "seq_len": seq_len, "correct": ok, "max_diff": float(max_diff),
            "total_us": total_us, "probes": probes,
        }

    return results


def run_smem_cpasync_dsa4() -> str:
    return json.dumps(run_dsa_cases(num_stages=4), indent=2)


if __name__ == "__main__":
    print(json.dumps(run_dsa_cases(), indent=2))

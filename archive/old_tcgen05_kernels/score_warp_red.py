"""score_warp_red.py — cp.async AB loads + FastGEMV-style score reduction.

Goal:
  * Reuse cp.async shared-memory load path from score_tcgen05_cpasync_dsa_full.py
    for ckv+kpe (A) and q_rope+q_nope (B).
  * Replace tcgen05 MMA score compute with a FastGEMV-style SIMT score path,
    interleaving 4 rows per warp and reducing with warp shuffles.
  * Report probe latencies focused on `load_ab` and `score`.
"""

import json
import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T


# ── Problem dims ──────────────────────────────────────────────────────────────
M               = 128
N_REAL          = 2
N_MMA           = 8
K_CKV           = 512
K_KPE           = 64
K_FULL          = K_CKV + K_KPE
POOL            = 256

THREADS_PER_CTA = 512
NUM_WARPS       = THREADS_PER_CTA // 32
NUM_ROUNDS_MAX  = M // NUM_WARPS

# FastGEMV-style score constants
ROWS_PER_WARP         = 4
ROWS_PER_ROUND_SCORE  = NUM_WARPS * ROWS_PER_WARP   # 64 rows per score round
NUM_SCORE_ROUNDS      = M // ROWS_PER_ROUND_SCORE   # 2 rounds for M=128


# ── Probe helpers ─────────────────────────────────────────────────────────────

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(
        MLIR_T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return cutlass.Int64(t)


@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(
        MLIR_T.i32(), [], "mov.u32 $0, %smid;", "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return cutlass.Int32(t)


PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES  = 12
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY

TAGS = {
    "total":     2,
    "prologue":  4,
    "load_ab":   6,
    "score":     8,
}
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


@cute.jit
def warp_reduce_add_i32(val: cutlass.Int32, width: cutlass.Constexpr = 32) -> cutlass.Int32:
    for i in range(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def warp_reduce_add_f32(val: cutlass.Float32, width: cutlass.Constexpr = 32) -> cutlass.Float32:
    for i in range(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


# ══════════════════════════════════════════════════════════════════════════════
class ScoreWarpRed:
    @cute.jit
    def __call__(
        self,
        ckv_flat:       cute.Tensor,   # (POOL, K_CKV) bf16
        kpe_flat:       cute.Tensor,   # (POOL, K_KPE) bf16
        q_rope:         cute.Tensor,   # (N_MMA, K_CKV) bf16
        q_nope:         cute.Tensor,   # (N_MMA, K_KPE) bf16
        sparse_indices: cute.Tensor,   # (M,) int32
        C:              cute.Tensor,   # (M, N_REAL) f32
        probe:          cute.Tensor,   # (1, PROBE_COLS) int64
    ):
        self.kernel(
            ckv_flat, kpe_flat, q_rope, q_nope, sparse_indices, C, probe,
        ).launch(grid=[1, 1, 1], block=[THREADS_PER_CTA, 1, 1])

    @cute.kernel
    def kernel(
        self,
        ckv_flat:       cute.Tensor,
        kpe_flat:       cute.Tensor,
        q_rope:         cute.Tensor,
        q_nope:         cute.Tensor,
        sparse_indices: cute.Tensor,
        C:              cute.Tensor,
        probe:          cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx   = cute.arch.lane_idx()
        ab_dtype = cutlass.BFloat16

        K_TILE:          cutlass.Constexpr = 64
        K_OUTER_CKV:     cutlass.Constexpr = K_CKV // K_TILE      # 8
        VEC_BF16:        cutlass.Constexpr = 8                    # 128b atom
        K_OUTER_HALF:    cutlass.Constexpr = K_OUTER_CKV // 2     # 4
        VEC_BF16_KPE:    cutlass.Constexpr = 2                    # 32b atom

        # ── SMEM allocation ────────────────────────────────────────────────
        alloc = cutlass.utils.SmemAllocator()
        sA = alloc.allocate_tensor(
            ab_dtype,
            cute.make_layout((M, K_FULL), stride=(K_FULL, 1)),
            128,
            None,
        )
        sB = alloc.allocate_tensor(
            ab_dtype,
            cute.make_layout((N_MMA, K_FULL), stride=(K_FULL, 1)),
            128,
            None,
        )
        smem_sparse = alloc.allocate_tensor(
            cutlass.Int32, cute.make_layout((M,), stride=(1,)), 4, None,
        )
        smem_red = alloc.allocate_tensor(
            cutlass.Int32, cute.make_layout((NUM_WARPS,), stride=(1,)), 4, None,
        )

        sm_val = smid_u32()
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])

        # ── Prologue: cache sparse indices + count valid + clamp ───────────
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["prologue"])

        partial_valid = cutlass.Int32(0)
        for m in cutlass.range_constexpr(M // THREADS_PER_CTA + 1):
            idx_lin = cutlass.Int32(m) * cutlass.Int32(THREADS_PER_CTA) + tidx
            if idx_lin < cutlass.Int32(M):
                idx = sparse_indices[idx_lin]
                smem_sparse[idx_lin] = idx
                if idx >= cutlass.Int32(0):
                    partial_valid = partial_valid + cutlass.Int32(1)

        warp_sum = warp_reduce_add_i32(partial_valid, width=32)
        if lane_idx == cutlass.Int32(0):
            smem_red[warp_idx] = warp_sum
        cute.arch.sync_threads()

        if warp_idx == cutlass.Int32(0):
            val = cutlass.Int32(0)
            if lane_idx < cutlass.Int32(NUM_WARPS):
                val = smem_red[lane_idx]
            block_sum = warp_reduce_add_i32(val, width=NUM_WARPS)
            if lane_idx == cutlass.Int32(0):
                smem_red[0] = block_sum
        cute.arch.sync_threads()

        num_valid = smem_red[0]
        num_rounds = (num_valid + cutlass.Int32(NUM_WARPS - 1)) // cutlass.Int32(NUM_WARPS)
        round_limit = num_rounds * cutlass.Int32(NUM_WARPS)

        for m in cutlass.range_constexpr(M // THREADS_PER_CTA + 1):
            idx_lin = cutlass.Int32(m) * cutlass.Int32(THREADS_PER_CTA) + tidx
            if idx_lin < cutlass.Int32(M):
                if smem_sparse[idx_lin] < cutlass.Int32(0):
                    smem_sparse[idx_lin] = cutlass.Int32(0)
        cute.arch.sync_threads()

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(1))

        # ── cp.async load_ab: same path as dsa_full/dsa_kpe ────────────────
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["load_ab"])

        atom_cpa = cute.make_copy_atom(
            cpasync.CopyG2SOp(), ab_dtype, num_bits_per_copy=128,
        )
        thr_layout_warp = cute.make_layout(
            (1, (8, K_OUTER_HALF)), stride=(32, (1, 8)),
        )
        val_layout_warp = cute.make_layout(
            (1, (VEC_BF16, 1)), stride=(0, (1, 0)),
        )
        tiled_copy_warp = cute.make_tiled_copy_tv(
            atom_cpa, thr_layout_warp, val_layout_warp,
        )
        lane_copy = tiled_copy_warp.get_slice(lane_idx)

        atom_cpa_kpe = cute.make_copy_atom(
            cpasync.CopyG2SOp(), ab_dtype, num_bits_per_copy=32,
        )
        thr_layout_kpe = cute.make_layout((1, 32), stride=(32, 1))
        val_layout_kpe = cute.make_layout((1, VEC_BF16_KPE), stride=(0, 1))
        tiled_copy_kpe = cute.make_tiled_copy_tv(
            atom_cpa_kpe, thr_layout_kpe, val_layout_kpe,
        )
        lane_copy_kpe = tiled_copy_kpe.get_slice(lane_idx)

        ckv_full = cute.make_tensor(
            ckv_flat.iterator,
            cute.make_layout(
                (1, POOL, (K_TILE, K_OUTER_CKV)),
                stride=(0, K_CKV, (1, K_TILE)),
            ),
        )
        kpe_full = cute.make_tensor(
            kpe_flat.iterator,
            cute.make_layout((1, POOL, K_TILE), stride=(0, K_KPE, 1)),
        )
        q_rope_full = cute.make_tensor(
            q_rope.iterator,
            cute.make_layout(
                (1, N_MMA, (K_TILE, K_OUTER_CKV)),
                stride=(0, K_CKV, (1, K_TILE)),
            ),
        )
        q_nope_full = cute.make_tensor(
            q_nope.iterator,
            cute.make_layout((1, N_MMA, K_TILE), stride=(0, K_KPE, 1)),
        )

        sA_ckv = cute.make_tensor(
            sA.iterator,
            cute.make_layout(
                (1, M, (K_TILE, K_OUTER_CKV)),
                stride=(0, K_FULL, (1, K_TILE)),
            ),
        )
        sA_kpe = cute.make_tensor(
            sA.iterator + K_CKV,
            cute.make_layout((1, M, K_TILE), stride=(0, K_FULL, 1)),
        )
        sB_qr = cute.make_tensor(
            sB.iterator,
            cute.make_layout(
                (1, N_MMA, (K_TILE, K_OUTER_CKV)),
                stride=(0, K_FULL, (1, K_TILE)),
            ),
        )
        sB_qn = cute.make_tensor(
            sB.iterator + K_CKV,
            cute.make_layout((1, N_MMA, K_TILE), stride=(0, K_FULL, 1)),
        )

        for rnd in cutlass.range_constexpr(NUM_ROUNDS_MAX):
            m_local = cutlass.Int32(rnd) * cutlass.Int32(NUM_WARPS) + warp_idx
            if m_local < round_limit:
                pool_idx = smem_sparse[m_local]

                gA_row = ckv_full[None, pool_idx, None]
                sA_ckv_row = sA_ckv[None, m_local, None]
                cute.copy(
                    atom_cpa,
                    lane_copy.partition_S(gA_row),
                    lane_copy.partition_D(sA_ckv_row),
                )

                gA_kpe_row = kpe_full[None, pool_idx, None]
                sA_kpe_row = sA_kpe[None, m_local, None]
                cute.copy(
                    atom_cpa_kpe,
                    lane_copy_kpe.partition_S(gA_kpe_row),
                    lane_copy_kpe.partition_D(sA_kpe_row),
                )

        if warp_idx < cutlass.Int32(N_REAL):
            gB_row = q_rope_full[None, warp_idx, None]
            sB_qr_row = sB_qr[None, warp_idx, None]
            cute.copy(
                atom_cpa,
                lane_copy.partition_S(gB_row),
                lane_copy.partition_D(sB_qr_row),
            )

            gB_qn_row = q_nope_full[None, warp_idx, None]
            sB_qn_row = sB_qn[None, warp_idx, None]
            cute.copy(
                atom_cpa_kpe,
                lane_copy_kpe.partition_S(gB_qn_row),
                lane_copy_kpe.partition_D(sB_qn_row),
            )

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))

        # ── Score: FastGEMV-style 4-row interleaved warp reduction ─────────
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(3), sm_val, TAGS["score"])

        atom_s2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), ab_dtype, num_bits_per_copy=32,
        )
        thr_layout_vec = cute.make_layout((32,), stride=(1,))
        val_layout_vec = cute.make_layout((2,), stride=(1,))
        tiled_copy_vec = cute.make_tiled_copy_tv(
            atom_s2r, thr_layout_vec, val_layout_vec,
        )
        lane_copy_vec = tiled_copy_vec.get_slice(lane_idx)

        for round_idx in cutlass.range_constexpr(NUM_SCORE_ROUNDS):
            base_row = (
                cutlass.Int32(round_idx) * cutlass.Int32(ROWS_PER_ROUND_SCORE)
                + warp_idx * cutlass.Int32(ROWS_PER_WARP)
            )

            for h in cutlass.range_constexpr(N_REAL):
                sums = cute.make_rmem_tensor(
                    cute.make_layout((ROWS_PER_WARP,), stride=(1,)),
                    cutlass.Float32,
                )
                for r in cutlass.range_constexpr(ROWS_PER_WARP):
                    sums[r] = cutlass.Float32(0)

                # CKV contribution over 8 x K_TILE(64)
                for ko in cutlass.range_constexpr(K_OUTER_CKV):
                    q_chunk = sB_qr[0, h, (None, ko)]
                    q_part = lane_copy_vec.partition_S(q_chunk)
                    q_rmem = cute.make_rmem_tensor(q_part.shape, ab_dtype)
                    cute.copy(atom_s2r, q_part, q_rmem)

                    for r in cutlass.range_constexpr(ROWS_PER_WARP):
                        row = base_row + cutlass.Int32(r)
                        if row < num_valid:
                            a_chunk = sA_ckv[0, row, (None, ko)]
                            a_part = lane_copy_vec.partition_S(a_chunk)
                            a_rmem = cute.make_rmem_tensor(a_part.shape, ab_dtype)
                            cute.copy(atom_s2r, a_part, a_rmem)
                            for v in cutlass.range_constexpr(2):
                                sums[r] = (
                                    sums[r]
                                    + cutlass.Float32(a_rmem[v]) * cutlass.Float32(q_rmem[v])
                                )

                # KPE contribution over 1 x K_TILE(64)
                qn_chunk = sB_qn[0, h, None]
                qn_part = lane_copy_vec.partition_S(qn_chunk)
                qn_rmem = cute.make_rmem_tensor(qn_part.shape, ab_dtype)
                cute.copy(atom_s2r, qn_part, qn_rmem)

                for r in cutlass.range_constexpr(ROWS_PER_WARP):
                    row = base_row + cutlass.Int32(r)
                    if row < num_valid:
                        ak_chunk = sA_kpe[0, row, None]
                        ak_part = lane_copy_vec.partition_S(ak_chunk)
                        ak_rmem = cute.make_rmem_tensor(ak_part.shape, ab_dtype)
                        cute.copy(atom_s2r, ak_part, ak_rmem)
                        for v in cutlass.range_constexpr(2):
                            sums[r] = (
                                sums[r]
                                + cutlass.Float32(ak_rmem[v]) * cutlass.Float32(qn_rmem[v])
                            )

                for r in cutlass.range_constexpr(ROWS_PER_WARP):
                    sums[r] = warp_reduce_add_f32(sums[r], width=32)
                    row = base_row + cutlass.Int32(r)
                    if lane_idx == cutlass.Int32(0) and row < num_valid:
                        C[row, h] = sums[r]

        # Keep outputs deterministic for invalid rows.
        for i in cutlass.range_constexpr(M // THREADS_PER_CTA + 1):
            row = cutlass.Int32(i) * cutlass.Int32(THREADS_PER_CTA) + tidx
            if row < cutlass.Int32(M) and row >= num_valid:
                C[row, 0] = cutlass.Float32(0)
                C[row, 1] = cutlass.Float32(0)

        cute.arch.sync_threads()

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(3))
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(0))
            range_finalize(probe, cutlass.Int32(0), cutlass.Int32(4))


def run_cases() -> dict:
    label = "score_warp_red"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Kernel: {label}  M={M} K_CKV={K_CKV} K_KPE={K_KPE} "
        f"N_real={N_REAL} threads={THREADS_PER_CTA}"
    )

    kernel = ScoreWarpRed()

    torch.manual_seed(42)
    ckv_flat = torch.randn((POOL, K_CKV), device="cuda", dtype=torch.bfloat16) * 0.1
    kpe_flat = torch.randn((POOL, K_KPE), device="cuda", dtype=torch.bfloat16) * 0.1
    q_rope   = torch.randn((N_MMA, K_CKV), device="cuda", dtype=torch.bfloat16) * 0.1
    q_nope   = torch.randn((N_MMA, K_KPE), device="cuda", dtype=torch.bfloat16) * 0.1

    si_full = torch.arange(M, device="cuda", dtype=torch.int32)

    seq_short = 64
    si_short = torch.full((M,), -1, device="cuda", dtype=torch.int32)
    si_short[:seq_short] = torch.arange(seq_short, dtype=torch.int32, device="cuda")

    C = torch.zeros((M, N_REAL), device="cuda", dtype=torch.float32)
    probe = torch.zeros((1, PROBE_COLS), device="cuda", dtype=torch.int64)

    ckv_ = from_dlpack(ckv_flat, assumed_align=128)
    kpe_ = from_dlpack(kpe_flat, assumed_align=128)
    qr_  = from_dlpack(q_rope, assumed_align=128)
    qn_  = from_dlpack(q_nope, assumed_align=128)
    C_   = from_dlpack(C, assumed_align=16)
    p_   = from_dlpack(probe, assumed_align=8)

    si_full_ = from_dlpack(si_full, assumed_align=16)
    compiled = cute.compile(kernel, ckv_, kpe_, qr_, qn_, si_full_, C_, p_)

    results = {}
    for case_name, si_tensor, seq_len in [
        ("full", si_full, M),
        ("short", si_short, seq_short),
    ]:
        si_ = from_dlpack(si_tensor, assumed_align=16)

        for _ in range(3):
            C.zero_()
            probe.zero_()
            compiled(ckv_, kpe_, qr_, qn_, si_, C_, p_)
        torch.cuda.synchronize()

        valid_mask = si_tensor >= 0
        valid_idx = si_tensor[valid_mask].long()
        ref_ckv = ckv_flat[valid_idx].float() @ q_rope[:N_REAL].float().T
        ref_kpe = kpe_flat[valid_idx].float() @ q_nope[:N_REAL].float().T
        ref = ref_ckv + ref_kpe

        nv = int(valid_mask.sum().item())
        ok = torch.allclose(C[:nv], ref, atol=1e-2, rtol=1e-2)
        max_diff = (C[:nv] - ref).abs().max().item()

        C.zero_()
        probe.zero_()
        compiled(ckv_, kpe_, qr_, qn_, si_, C_, p_)
        torch.cuda.synchronize()

        p_log = probe[0].cpu().tolist()
        cnt = int(p_log[0])
        probes = []
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag_v = int(p_log[off + 1])
            dur_ns = int(p_log[off + 3])
            probes.append({
                "phase": TAG_NAMES.get(tag_v, f"tag{tag_v}"),
                "us": dur_ns / 1000.0,
            })

        total_us = next((x["us"] for x in probes if x["phase"] == "total"), 0.0)
        load_us = next((x["us"] for x in probes if x["phase"] == "load_ab"), 0.0)
        score_us = next((x["us"] for x in probes if x["phase"] == "score"), 0.0)

        print(
            f"\n[{case_name:5s}] seq_len={seq_len:3d} "
            f"{'PASS' if ok else f'FAIL(max_diff={max_diff:.4f})'} "
            f"total={total_us:.3f} us  load_ab={load_us:.3f} us  score={score_us:.3f} us"
        )
        for pr in probes:
            print(f"  {pr['phase']:10s}: {pr['us']:7.3f} us")

        results[case_name] = {
            "seq_len": seq_len,
            "correct": bool(ok),
            "max_diff": float(max_diff),
            "total_us": float(total_us),
            "load_ab_us": float(load_us),
            "score_us": float(score_us),
            "probes": probes,
        }

    return results


def run_intra() -> str:
    return json.dumps(run_cases(), indent=2)


if __name__ == "__main__":
    print(run_intra())

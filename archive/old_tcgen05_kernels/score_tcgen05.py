"""score_tcgen05.py — tcgen05 BF16 MMA score kernel: (128,K) × (K,N_real) = (128,N_real).

Problem
-------
  A      : (M=128, K=512) bf16, K-major   — Q_nope / token queries
  B      : (N_MMA=8, K=512) bf16, K-major — only first N_real=2 rows are real;
                                             rows [N_real:N_MMA] are garbage
  C      : (M=128, N_real=2) fp32         — output scores

  C[:, :N_real] = A @ B[:N_real].T

The MMA minimum N is 8 for tcgen05 BF16 so we always run the (128,8,K) MMA.
During the epilogue we load all 8 tmem columns into registers but only store
the first N_real=2 to C.  Columns [N_real:8] are computed but discarded.

Pipeline (identical to tcgen05_mma_mkn.py)
------------------------------------------
  1. TMA G2S A and B (full 8-row B) → swizzled smem.
  2. mbarrier wait.
  3. tcgen05 BF16 MMA (warp 0, K_inst=16, 32 k-blocks over K=512).
  4. tmem → regs epilogue, write only first N_real columns to C.
"""

import json
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import tcgen05, cpasync
from cutlass.cutlass_dsl import dsl_user_op, T as MLIR_T


# ── Problem dims ──────────────────────────────────────────────────────────────
M               = 128
N_REAL          = 2       # actual output columns to write
N_MMA           = 8       # tcgen05 minimum N; MMA always runs (128,8,K)
K               = 512

THREADS_PER_CTA = 512
MMA_INST_MNK    = (128, N_MMA, 16)    # bf16/bf16: K_inst = 16
CTA_TILE_MNK    = (M, N_MMA, K)


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


@dsl_user_op
def tcgen05_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None, [], "tcgen05.fence::after_thread_sync;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


PROBE_HEADER = 1
PROBE_ENTRY  = 4
MAX_ENTRIES  = 8
PROBE_COLS   = PROBE_HEADER + MAX_ENTRIES * PROBE_ENTRY

TAGS      = {"total": 2, "load_ab": 4, "mma": 6, "epilogue": 8}
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


# ══════════════════════════════════════════════════════════════════════════════
#  Kernel
# ══════════════════════════════════════════════════════════════════════════════

class ScoreTcgen05:
    """tcgen05 BF16 MMA: (M=128,K=512) x (N_MMA=8,K=512) → only first N_real=2 cols written."""

    def __init__(self):
        self.num_stages  = 1
        self.tmem_ld_rep = N_MMA   # load all 8 tmem columns into regs

    @cute.jit
    def __call__(
        self,
        A:      cute.Tensor,   # (M, K) bf16
        B:      cute.Tensor,   # (N_MMA, K) bf16  — only first N_real rows real
        C:      cute.Tensor,   # (M, N_real) fp32
        probe:  cute.Tensor,   # (1, PROBE_COLS) int64
    ):
        ab_dtype  = cutlass.BFloat16
        acc_dtype = cutlass.Float32

        op = tcgen05.MmaF16BF16Op(
            ab_dtype, acc_dtype, MMA_INST_MNK,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE_MNK, ab_dtype, self.num_stages,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, CTA_TILE_MNK, ab_dtype, self.num_stages,
        )

        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op_g2s, A, cute.select(a_smem_layout, mode=[0, 1, 2]),
            CTA_TILE_MNK, tiled_mma,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, B, cute.select(b_smem_layout, mode=[0, 1, 2]),
            CTA_TILE_MNK, tiled_mma,
        )

        @cute.struct
        class SharedStorage:
            mma_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, 1]
            tma_mbar_a_ptr:   cute.struct.MemRange[cutlass.Int64, 1]
            tma_mbar_b_ptr:   cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma, a_smem_layout, b_smem_layout, ab_dtype, acc_dtype,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            C, probe,
        ).launch(grid=[1, 1, 1], block=[THREADS_PER_CTA, 1, 1])

    @cute.kernel
    def kernel(
        self,
        tiled_mma, a_smem_layout, b_smem_layout, ab_dtype, acc_dtype,
        tma_atom_a:    cute.CopyAtom,
        mA_tma:        cute.Tensor,
        tma_atom_b:    cute.CopyAtom,
        mB_tma:        cute.Tensor,
        C:      cute.Tensor,
        probe:  cute.Tensor,
    ):
        N_real:      cutlass.Constexpr = N_REAL
        tmem_ld_rep: cutlass.Constexpr = self.tmem_ld_rep

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        alloc = cutlass.utils.SmemAllocator()
        sA = alloc.allocate_tensor(
            element_type=ab_dtype, layout=a_smem_layout.outer,
            byte_alignment=128, swizzle=a_smem_layout.inner,
        )
        sB = alloc.allocate_tensor(
            element_type=ab_dtype, layout=b_smem_layout.outer,
            byte_alignment=128, swizzle=b_smem_layout.inner,
        )
        storage = alloc.allocate(self.shared_storage)
        mma_mbar   = storage.mma_mbar_ptr.data_ptr()
        tma_mbar_a = storage.tma_mbar_a_ptr.data_ptr()
        tma_mbar_b = storage.tma_mbar_b_ptr.data_ptr()
        tma_bytes_a = cute.size_in_bytes(
            ab_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]),
        )
        tma_bytes_b = cute.size_in_bytes(
            ab_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]),
        )

        sm_val = smid_u32()
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(0), sm_val, TAGS["total"])

        # ── TMA G2S partitions ────────────────────────────────────────────────
        gA_tma  = cute.local_tile(mA_tma, CTA_TILE_MNK, (0, 0, None), proj=(1, None, 1))
        gB_tma  = cute.local_tile(mB_tma, CTA_TILE_MNK, (0, 0, None), proj=(None, 1, 1))
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA    = thr_mma.partition_A(gA_tma)
        tCgB    = thr_mma.partition_B(gB_tma)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_init(tma_mbar_a, cnt=1)
                cute.arch.mbarrier_init(tma_mbar_b, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.barrier(barrier_id=1, number_of_threads=THREADS_PER_CTA)

        # ── load_ab phase ─────────────────────────────────────────────────────
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(1), sm_val, TAGS["load_ab"])

        if warp_idx == 0:
            cute.copy(tma_atom_a, tAgA[None, 0], tAsA[None, 0], tma_bar_ptr=tma_mbar_a)
            cute.copy(tma_atom_b, tBgB[None, 0], tBsB[None, 0], tma_bar_ptr=tma_mbar_b)
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar_a, tma_bytes_a)
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar_b, tma_bytes_b)
        cute.arch.mbarrier_wait(tma_mbar_a, 0)
        cute.arch.mbarrier_wait(tma_mbar_b, 0)
        cute.arch.sync_threads()

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(1))

        # ── MMA setup: tmem alloc + mbar init ─────────────────────────────────
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape       = tiled_mma.partition_shape_C(CTA_TILE_MNK[:2])
        tCtAcc_tmpl     = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols   = utils.get_num_tmem_alloc_cols(tCtAcc_tmpl)
        tmem_alloc_cols = cutlass.Int32(num_tmem_cols)

        if warp_idx == 0:
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)

        tmem_barrier_id = 1
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            acc_dtype, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)

        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)

        num_k_blocks = cute.size(tCrA, mode=[2])

        # ── mma phase ─────────────────────────────────────────────────────────
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(2), sm_val, TAGS["mma"])

        tcgen05_fence()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        if warp_idx == 0:
            for k_block_idx in range(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, 0)
                cute.gemm(
                    tiled_mma, tCtAcc,
                    tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if tidx == 0:
                tcgen05.commit(mma_mbar)

        cute.arch.mbarrier_wait(mma_mbar, cutlass.Int32(0))

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(2))

        # ── epilogue: tmem → regs → gmem ──────────────────────────────────────
        # Load all N_MMA=8 tmem columns per thread, write only first N_real=2.
        if tidx == cutlass.Int32(0):
            range_start(probe, cutlass.Int32(0), cutlass.Int32(3), sm_val, TAGS["epilogue"])

        M_acc          = cute.size(tCtAcc, mode=[0, 0])
        ld_op          = tcgen05.Ld32x32bOp(tcgen05.Repetition(tmem_ld_rep))
        epi_tiler      = ((M_acc, tmem_ld_rep),)
        tCtAcc_epi     = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r  = cute.make_copy_atom(ld_op, acc_dtype)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy  = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc       = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc       = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, acc_dtype)

        if tidx < cutlass.Int32(M):
            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)
            for n_idx in cutlass.range_constexpr(N_REAL):
                C[tidx, n_idx] = tTR_rAcc[n_idx]

        cute.arch.sync_threads()

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(3))

        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier(barrier_id=tmem_barrier_id)
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        if tidx == cutlass.Int32(0):
            range_stop(probe, cutlass.Int32(0), cutlass.Int32(0))
            range_finalize(probe, cutlass.Int32(0), cutlass.Int32(4))


# ══════════════════════════════════════════════════════════════════════════════
#  Run helper
# ══════════════════════════════════════════════════════════════════════════════

def run_intra() -> str:
    label = "score_tcgen05_bf16"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kernel: {label}  M={M}  K={K}  N_real={N_REAL}  N_mma={N_MMA}  threads={THREADS_PER_CTA}")

    kernel = ScoreTcgen05()

    torch.manual_seed(42)
    # A: (M, K) — real query data
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.1
    # B: (N_MMA, K) — first N_REAL rows real, rest garbage (random)
    B = torch.randn((N_MMA, K), device="cuda", dtype=torch.bfloat16) * 0.1
    # C: (M, N_REAL) — output scores
    C = torch.zeros((M, N_REAL), device="cuda", dtype=torch.float32)
    probe = torch.zeros((1, PROBE_COLS), dtype=torch.int64, device="cuda")

    A_     = from_dlpack(A,     assumed_align=128)
    B_     = from_dlpack(B,     assumed_align=128)
    C_     = from_dlpack(C,     assumed_align=16)
    probe_ = from_dlpack(probe, assumed_align=8)

    compiled = cute.compile(kernel, A_, B_, C_, probe_)

    for _ in range(3):
        probe.zero_(); C.zero_()
        compiled(A_, B_, C_, probe_)
    torch.cuda.synchronize()

    # Reference: A @ B[:N_REAL].T  (only the first N_REAL rows of B matter)
    ref = A.float() @ B[:N_REAL].float().T
    ok  = torch.allclose(C, ref, atol=1e-2, rtol=1e-2)
    max_diff = (C - ref).abs().max().item()
    print(f"Correctness: {'PASS' if ok else 'FAIL'}  max_diff={max_diff:.6f}")

    probe.zero_(); C.zero_()
    compiled(A_, B_, C_, probe_)
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
        print(f"  {name:10s}: {us:7.3f} µs")

    return json.dumps({
        "kernel": label,
        "M": M, "K": K, "N_real": N_REAL, "N_mma": N_MMA, "threads": THREADS_PER_CTA,
        "correct": ok, "max_diff": float(max_diff),
        "probes": probes,
    }, indent=2)


if __name__ == "__main__":
    print(run_intra())

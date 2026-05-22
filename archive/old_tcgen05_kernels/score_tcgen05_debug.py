"""score_tcgen05_debug.py — debug variant of score_tcgen05.

Goals
-----
1. Use the SAME TMA + tcgen05 BF16 MMA pipeline as score_tcgen05.py
   (proven correct). Dims:
     A : (M=128, K=512) bf16
     B : (N_MMA=8, K=512) bf16
     C : (M=128, N_real=2) fp32
2. Use a probe-friendly input pattern  src[row, col] = col  so any swizzle
   on physical SMEM is visible by inspecting raw bytes.
3. Dump the raw physical SMEM bytes after the TMA load by constructing a
   FRESH non-swizzled pointer at the same byte address as sA / sB:
       sA_raw_addr = cute.recast_ptr(sA.iterator, dtype=BF16).toint()
       sA_raw_ptr  = cute.make_ptr(BF16, sA_raw_addr,
                                   mem_space=cute.AddressSpace.smem,
                                   assumed_align=1024)
       sA_raw      = cute.make_tensor(sA_raw_ptr, cute.make_layout(N))
   This bypasses the ComposedLayout swizzle wrapper that lives on
   `sA.iterator`, so reading `sA_raw[i]` returns the actual physical byte
   at offset i (no XOR round-trip).

Why
---
The previous swz_bf16_score_copy.py readback indexed through `sA.iterator`,
which still carries the swizzle wrapper — so `sA_raw[i]` round-tripped
through the swizzle XOR and reproduced the *logical* layout, not the
physical SMEM layout.  This file demonstrates the correct readback.
"""

import json
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import tcgen05, cpasync


# ── Problem dims (match score_tcgen05.py) ────────────────────────────────────
M               = 128
N_REAL          = 2
N_MMA           = 8
K               = 512

THREADS_PER_CTA = 512
MMA_INST_MNK    = (128, N_MMA, 16)
CTA_TILE_MNK    = (M, N_MMA, K)

BF16_DTYPE = cutlass.BFloat16
ACC_DTYPE  = cutlass.Float32

NUM_ELEMS_A = M * K       # 65 536
NUM_ELEMS_B = N_MMA * K   #  4 096


# ── Display helper ────────────────────────────────────────────────────────────
#
# The TMA SMEM layout for sA / sB packs each `m_outer` index as 64
# contiguous bf16 elements (128 bytes = one swizzle row). To observe
# Sw<3,4,3>, we print each PHYSICAL row independently:
#   row m_outer = raw_flat[m_outer*64 : m_outer*64 + 64]
# The Sw<3,4,3> XOR uses byte-address bits [7..9] = (m_outer & 7).

def _row_str(values, elems_per_row: int) -> str:
    return " ".join(f"{int(v):3d}" for v in values[:elems_per_row])


def expected_xor_row(m_outer: int, elems_per_row: int = 64,
                     atom: int = 8) -> list[int]:
    """Predicted physical SMEM row for src[m,k]=k under Sw<3,4,3>.

    Each atom of `atom` bf16 elems lives at a byte-aligned offset whose
    bits [7..9] encode the atom index within the row (for 16-byte atoms,
    bits [4..6]). Sw<3,4,3> XORs atom bits [4..6] with m_outer's bits [0..2].
    Result: the atom at column-atom `a` holds source-column-atom `a XOR (m_outer&7)`.
    """
    xor_atom = m_outer & 7
    out = []
    for a in range(elems_per_row // atom):       # atom-index within the row
        src_a = a ^ xor_atom
        for k in range(atom):                    # bf16 within the atom
            out.append(src_a * atom + k)
    return out


def print_phys_rows(raw_flat: torch.Tensor, label: str,
                    m_outer_list, k_outer_outer: int = 0,
                    elems_per_row: int = 64):
    """Print physical SMEM rows side-by-side: MEASURED vs EXPECTED.

    MEASURED = raw bytes captured from SMEM via the bypass pointer
               (no swizzle wrapper on indexing — see file docstring).
    EXPECTED = analytically computed `(col XOR atom_swap_by_(m_outer&7))`
               printed in the SAME format for direct visual comparison.

    raw_flat: 1D bf16 tensor of dumped SMEM bytes (in element order).
    m_outer_list: which m_outer rows to print (each row = 64 elems).
    k_outer_outer: which K-outer-outer block to look at (offset = k_oo*8192).
    """
    f = raw_flat.cpu().to(torch.float32)
    base = k_outer_outer * 8192
    hdr  = " ".join(f"{c:3d}" for c in range(elems_per_row))

    # ── MEASURED ──────────────────────────────────────────────────────────
    print(f"  [{label} — MEASURED]  (k_outer_outer={k_outer_outer}, elems/row={elems_per_row})")
    print(f"    {'':10s}  {hdr}")
    measured_rows = []
    for m_outer in m_outer_list:
        off  = base + m_outer * elems_per_row
        vals = [f[off + c].item() for c in range(elems_per_row)]
        measured_rows.append(vals)
        xor_atom = m_outer & 7
        print(f"    m_o={m_outer:3d} (XOR={xor_atom}): {_row_str(vals, elems_per_row)}")

    # ── EXPECTED (analytical, NOT measured) ───────────────────────────────
    print()
    print(f"  [{label} — EXPECTED (analytical from Sw<3,4,3> XOR rule)]")
    print(f"    {'':10s}  {hdr}")
    mismatches = 0
    for idx, m_outer in enumerate(m_outer_list):
        exp = expected_xor_row(m_outer, elems_per_row=elems_per_row)
        xor_atom = m_outer & 7
        print(f"    m_o={m_outer:3d} (XOR={xor_atom}): {_row_str(exp, elems_per_row)}")
        # diff vs measured
        for c in range(elems_per_row):
            if int(measured_rows[idx][c]) != exp[c]:
                mismatches += 1

    total = len(m_outer_list) * elems_per_row
    verdict = "MATCH ✓" if mismatches == 0 else f"MISMATCH ✗ ({mismatches}/{total})"
    print(f"\n    diff(measured, expected): {verdict}")


# ── Probe-friendly inputs: src[row, col] = col ───────────────────────────────

def make_input_a() -> torch.Tensor:
    cols = torch.arange(K, dtype=torch.float32).unsqueeze(0)
    return cols.expand(M, -1).contiguous().to(torch.bfloat16).cuda()

def make_input_b() -> torch.Tensor:
    cols = torch.arange(K, dtype=torch.float32).unsqueeze(0)
    return cols.expand(N_MMA, -1).contiguous().to(torch.bfloat16).cuda()


# ══════════════════════════════════════════════════════════════════════════════
#  Debug kernel: TMA + tcgen05 MMA + RAW physical SMEM dump
# ══════════════════════════════════════════════════════════════════════════════

class ScoreTcgen05Debug:

    def __init__(self):
        self.num_stages  = 1
        self.tmem_ld_rep = N_MMA

    @cute.jit
    def __call__(
        self,
        A:        cute.Tensor,   # (M, K) bf16
        B:        cute.Tensor,   # (N_MMA, K) bf16
        C:        cute.Tensor,   # (M, N_real) fp32
        rawA_out: cute.Tensor,   # (NUM_ELEMS_A,) bf16 — raw physical sA bytes
        rawB_out: cute.Tensor,   # (NUM_ELEMS_B,) bf16 — raw physical sB bytes
    ):
        op = tcgen05.MmaF16BF16Op(
            BF16_DTYPE, ACC_DTYPE, MMA_INST_MNK,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)

        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE_MNK, BF16_DTYPE, self.num_stages,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, CTA_TILE_MNK, BF16_DTYPE, self.num_stages,
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

        print("[Debug] a_smem_layout:", a_smem_layout)
        print("[Debug] b_smem_layout:", b_smem_layout)

        self.kernel(
            tiled_mma, a_smem_layout, b_smem_layout,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            C, rawA_out, rawB_out,
        ).launch(grid=[1, 1, 1], block=[THREADS_PER_CTA, 1, 1])

    @cute.kernel
    def kernel(
        self,
        tiled_mma, a_smem_layout, b_smem_layout,
        tma_atom_a:    cute.CopyAtom,
        mA_tma:        cute.Tensor,
        tma_atom_b:    cute.CopyAtom,
        mB_tma:        cute.Tensor,
        C:        cute.Tensor,
        rawA_out: cute.Tensor,
        rawB_out: cute.Tensor,
    ):
        tmem_ld_rep: cutlass.Constexpr = self.tmem_ld_rep

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        alloc = cutlass.utils.SmemAllocator()
        sA = alloc.allocate_tensor(
            element_type=BF16_DTYPE, layout=a_smem_layout.outer,
            byte_alignment=1024, swizzle=a_smem_layout.inner,
        )
        sB = alloc.allocate_tensor(
            element_type=BF16_DTYPE, layout=b_smem_layout.outer,
            byte_alignment=1024, swizzle=b_smem_layout.inner,
        )
        storage     = alloc.allocate(self.shared_storage)
        mma_mbar    = storage.mma_mbar_ptr.data_ptr()
        tma_mbar_a  = storage.tma_mbar_a_ptr.data_ptr()
        tma_mbar_b  = storage.tma_mbar_b_ptr.data_ptr()
        tma_bytes_a = cute.size_in_bytes(BF16_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2]))
        tma_bytes_b = cute.size_in_bytes(BF16_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2]))

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

        # ── load A,B via TMA ─────────────────────────────────────────────────
        if warp_idx == 0:
            cute.copy(tma_atom_a, tAgA[None, 0], tAsA[None, 0], tma_bar_ptr=tma_mbar_a)
            cute.copy(tma_atom_b, tBgB[None, 0], tBsB[None, 0], tma_bar_ptr=tma_mbar_b)
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar_a, tma_bytes_a)
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar_b, tma_bytes_b)
        cute.arch.mbarrier_wait(tma_mbar_a, 0)
        cute.arch.mbarrier_wait(tma_mbar_b, 0)
        cute.arch.sync_threads()

        # ════════════════════════════════════════════════════════════════════
        #  RAW PHYSICAL SMEM DUMP  — bypass the swizzle wrapper.
        #
        #  sA.iterator carries the ComposedLayout(swizzle, ...) from
        #  allocate_tensor, so indexing sA[i] applies the swizzle XOR.
        #  We extract the byte address with recast_ptr().toint() and rebuild
        #  a FRESH plain bf16 pointer with cute.make_ptr; the resulting
        #  tensor's [i] == the physical byte at offset i.
        # ════════════════════════════════════════════════════════════════════
        sA_addr = cute.recast_ptr(sA.iterator, dtype=BF16_DTYPE).toint()
        sB_addr = cute.recast_ptr(sB.iterator, dtype=BF16_DTYPE).toint()

        sA_raw_ptr = cute.make_ptr(
            BF16_DTYPE, sA_addr,
            mem_space=cute.AddressSpace.smem, assumed_align=1024,
        )
        sB_raw_ptr = cute.make_ptr(
            BF16_DTYPE, sB_addr,
            mem_space=cute.AddressSpace.smem, assumed_align=1024,
        )
        sA_raw = cute.make_tensor(sA_raw_ptr, cute.make_layout(NUM_ELEMS_A))
        sB_raw = cute.make_tensor(sB_raw_ptr, cute.make_layout(NUM_ELEMS_B))

        gOutA = cute.make_tensor(
            cute.recast_ptr(rawA_out.iterator, dtype=BF16_DTYPE),
            cute.make_layout(NUM_ELEMS_A),
        )
        gOutB = cute.make_tensor(
            cute.recast_ptr(rawB_out.iterator, dtype=BF16_DTYPE),
            cute.make_layout(NUM_ELEMS_B),
        )

        # 512 threads × 128 elems = 65536 = NUM_ELEMS_A
        for k in cutlass.range_constexpr(NUM_ELEMS_A // THREADS_PER_CTA):
            gOutA[tidx + k * THREADS_PER_CTA] = sA_raw[tidx + k * THREADS_PER_CTA]
        # 512 threads × 8 elems = 4096 = NUM_ELEMS_B
        for k in cutlass.range_constexpr(NUM_ELEMS_B // THREADS_PER_CTA):
            gOutB[tidx + k * THREADS_PER_CTA] = sB_raw[tidx + k * THREADS_PER_CTA]
        cute.arch.sync_threads()

        # ── tcgen05 MMA: sanity check that the swizzled sA/sB layouts ARE
        #    consumed correctly by hardware (writes scores into C). ──────────
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
            ACC_DTYPE, alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)

        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_init(mma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()
        cute.arch.barrier(barrier_id=tmem_barrier_id, number_of_threads=THREADS_PER_CTA)

        num_k_blocks = cute.size(tCrA, mode=[2])
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        if warp_idx == 0:
            for k_block_idx in range(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, 0)
                cute.gemm(tiled_mma, tCtAcc,
                          tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if tidx == 0:
                tcgen05.commit(mma_mbar)
        cute.arch.mbarrier_wait(mma_mbar, cutlass.Int32(0))

        # ── Epilogue: tmem → regs → C ────────────────────────────────────────
        M_acc           = cute.size(tCtAcc, mode=[0, 0])
        ld_op           = tcgen05.Ld32x32bOp(tcgen05.Repetition(tmem_ld_rep))
        epi_tiler       = ((M_acc, tmem_ld_rep),)
        tCtAcc_epi      = cute.zipped_divide(tCtAcc, epi_tiler)
        copy_atom_t2r   = cute.make_copy_atom(ld_op, ACC_DTYPE)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[None, 0])
        tmem_thr_copy   = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc        = tmem_thr_copy.partition_S(tCtAcc_epi)
        tTR_rAcc        = cute.make_rmem_tensor(tTR_tAcc[None, None, 0].shape, ACC_DTYPE)

        if tidx < cutlass.Int32(M):
            cute.copy(tmem_tiled_copy, tTR_tAcc[None, None, 0], tTR_rAcc)
            for n_idx in cutlass.range_constexpr(N_REAL):
                C[tidx, n_idx] = tTR_rAcc[n_idx]
        cute.arch.sync_threads()

        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier(barrier_id=tmem_barrier_id)
        if warp_idx == 0:
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run() -> str:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"score_tcgen05_debug  M={M}  K={K}  N_real={N_REAL}  N_mma={N_MMA}")

    # Probe inputs: src[row, col] = col so swizzle pattern is visible.
    A = make_input_a()                                # (M, K) bf16
    B = make_input_b()                                # (N_MMA, K) bf16
    C = torch.zeros((M, N_REAL), device="cuda", dtype=torch.float32)
    rawA_out = torch.zeros(NUM_ELEMS_A, device="cuda", dtype=torch.bfloat16)
    rawB_out = torch.zeros(NUM_ELEMS_B, device="cuda", dtype=torch.bfloat16)

    A_   = from_dlpack(A,   assumed_align=128)
    B_   = from_dlpack(B,   assumed_align=128)
    C_   = from_dlpack(C,   assumed_align=16)
    rA_  = from_dlpack(rawA_out, assumed_align=128)
    rB_  = from_dlpack(rawB_out, assumed_align=128)

    kernel   = ScoreTcgen05Debug()
    compiled = cute.compile(kernel, A_, B_, C_, rA_, rB_)
    compiled(A_, B_, C_, rA_, rB_)
    torch.cuda.synchronize()

    # ── MMA sanity check ─────────────────────────────────────────────────────
    ref = A.float() @ B[:N_REAL].float().T
    ok  = torch.allclose(C, ref, atol=1e-2, rtol=1e-2)
    max_diff = (C - ref).abs().max().item()
    print(f"\nMMA correctness: {'PASS' if ok else 'FAIL'}  max_diff={max_diff:.6f}")

    # ── Raw physical SMEM dump ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RAW physical SMEM after TMA load (Sw<3,4,3>, src[m,k]=k)")
    print("  Each printed row = ONE physical SMEM row = 64 bf16 elems = 128 B.")
    print("  Sw<3,4,3> XOR uses byte-address bits [7..9] = m_outer & 7.")
    print("  Atom = 8 bf16 elems (16 B). Both MEASURED and EXPECTED tables")
    print("  are printed in the same row format for direct comparison:")
    print("    - MEASURED: bytes captured from SMEM via the bypass pointer.")
    print("    - EXPECTED: analytically computed `(c XOR atom_swap_by_(m_outer&7))`.")
    print()
    print("  EXPECTED summary (analytical — NOT measured):")
    print("    m_outer=0 (XOR=0):  0..7 | 8..15 | 16..23 | 24..31 | ... (identity)")
    print("    m_outer=1 (XOR=1):  8..15| 0..7  | 24..31 | 16..23 | ... (atom swap by 1)")
    print("    m_outer=2 (XOR=2):  16..23|24..31| 0..7   | 8..15  | ...")
    print("    m_outer=4 (XOR=4):  32..39|...   | ...    | ...    | 0..7 | 8..15 | ...")
    print("=" * 70)
    print()
    print_phys_rows(rawA_out, "sA — physical SMEM rows (k_outer_outer=0)",
                    m_outer_list=list(range(8)))
    print()
    print_phys_rows(rawB_out, "sB — physical SMEM rows (k_outer_outer=0)",
                    m_outer_list=list(range(N_MMA)))

    return json.dumps({
        "kernel": "score_tcgen05_debug",
        "M": M, "K": K, "N_real": N_REAL, "N_mma": N_MMA,
        "mma_correct": ok, "mma_max_diff": float(max_diff),
    }, indent=2)


if __name__ == "__main__":
    print(run())

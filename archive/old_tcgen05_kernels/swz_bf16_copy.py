"""swz_bf16_copy.py — Swizzled bf16 sA+sB tiles: TMA vs autovec copy comparison.

sA: [128, 64] bfloat16 — 128 bytes/row × 128 rows = 16 384 bytes.
sB: [ 64, 64] bfloat16 — 128 bytes/row ×  64 rows =  8 192 bytes.
Swizzle: Sw<3,4,3> for both (same 128-byte row width).

Input: src[row, col] = col  (all rows identical; value = column index 0..63).
Each physical SMEM slot shows exactly which source column landed there.

Allocation order: sA first → sB → barriers.
  sA at byte 0:     k = 0  → swizzle XOR row r matches relative offset exactly.
  sB at byte 16384: k = (16384>>7)&7 = 0  → same zero-bias for sB.
  Both TMA (absolute address) and autovec (relative) agree.

Step 1: TMA G2S for sA and sB — ground-truth physical SMEM layout.
Step 2: cute.autovec_copy G→S for sA and sB — compare vs TMA reference.
Step 3: cute.copy with cp.async G→S for sA and sB — compare vs TMA reference.

cp.async note: 128-bit atom = 16B = 8 bf16 per instruction.
128 threads × 8 atoms = 1024 bf16 = sA in 8 passes; sB in 4 passes per thread.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, cpasync
import cutlass.utils.blackwell_helpers as sm100_utils

# ── Constants ─────────────────────────────────────────────────────────────────
BM       = 128
BN       = 64
HEAD_DIM = 64

NUM_ELEMS_A = BM * HEAD_DIM   # 8 192
NUM_ELEMS_B = BN * HEAD_DIM   # 4 096

BF16_DTYPE = cutlass.BFloat16
ACC_DTYPE  = cutlass.Float32
MMA_INST   = (128, 64, 16)          # bf16/bf16 K_inst=16
CTA_TILE   = (BM, BN, HEAD_DIM)     # (128, 64, 64)

THREADS      = 512
N_PER_THR_A  = NUM_ELEMS_A // THREADS   # 16
N_PER_THR_B  = NUM_ELEMS_B // THREADS   # 8

# cp.async (Step 3) — 128-bit atom: 16B = 8×bf16 per instruction
N_ATOMS_THR_A = N_PER_THR_A // 8   # 2  × 128b cp.asyncs per thread (sA)
N_ATOMS_THR_B = N_PER_THR_B // 8   # 1  × 128b cp.asyncs per thread (sB)


# ── Inputs ─────────────────────────────────────────────────────────────────────

def make_bf16_input_a() -> torch.Tensor:
    """[BM, HEAD_DIM] bf16, src[row, col] = col."""
    cols = torch.arange(HEAD_DIM, dtype=torch.float32).unsqueeze(0)
    return cols.expand(BM, -1).clone().to(torch.bfloat16).cuda()

def make_bf16_input_b() -> torch.Tensor:
    """[BN, HEAD_DIM] bf16, src[row, col] = col."""
    cols = torch.arange(HEAD_DIM, dtype=torch.float32).unsqueeze(0)
    return cols.expand(BN, -1).clone().to(torch.bfloat16).cuda()


# ── Display helper ─────────────────────────────────────────────────────────────

def print_chunk(t: torch.Tensor, label: str, n_rows: int, n_cols: int = 32):
    """Print first n_rows × n_cols bf16 values as integers."""
    _, ncols_total = t.cpu().to(torch.float32).shape
    flat = t.cpu().to(torch.float32).reshape(-1, ncols_total)
    print(f"  [{label}]")
    hdr = " ".join(f"{c:3d}" for c in range(n_cols))
    print(f"    {'':8s}  {hdr}")
    for r in range(n_rows):
        vals = " ".join(f"{flat[r, c].item():3.0f}" for c in range(n_cols))
        print(f"    row {r:3d}:  {vals}")


# ── Step 1: TMA G2S (reference) ───────────────────────────────────────────────

class _Bf16TmaCopy:
    """TMA G2S into sA and sB; raw bf16 readback exposes physical SMEM layout."""

    NUM_STAGES = 1

    @cute.jit
    def __call__(
        self,
        mA:    cute.Tensor,   # [BM, HEAD_DIM] bf16
        mB:    cute.Tensor,   # [BN, HEAD_DIM] bf16
        mOutA: cute.Tensor,   # [BM, HEAD_DIM] bf16 output
        mOutB: cute.Tensor,   # [BN, HEAD_DIM] bf16 output
    ):
        op = tcgen05.MmaF16BF16Op(
            BF16_DTYPE, ACC_DTYPE, MMA_INST,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma     = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE, BF16_DTYPE, self.NUM_STAGES,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, CTA_TILE, BF16_DTYPE, self.NUM_STAGES,
        )
        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op_g2s, mA, cute.select(a_smem_layout, mode=[0, 1, 2]), CTA_TILE, tiled_mma,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, mB, cute.select(b_smem_layout, mode=[0, 1, 2]), CTA_TILE, tiled_mma,
        )
        self.kernel(
            tiled_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            mOutA, mOutB,
            a_smem_layout, b_smem_layout,
        ).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))

    @cute.kernel
    def kernel(
        self,
        tiled_mma:     cute.TiledMma,
        tma_atom_a:    cute.CopyAtom,
        mA_tma:        cute.Tensor,
        tma_atom_b:    cute.CopyAtom,
        mB_tma:        cute.Tensor,
        mOutA:         cute.Tensor,
        mOutB:         cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx   = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()

        # CRITICAL: sA and sB allocated FIRST, before barriers.
        # TMA uses absolute SMEM addresses for swizzle XOR. sA lands at byte 0
        # (k=0), sB lands at byte 16384 = 128*128 → (16384>>7)&7 = 0 (k=0).
        # Both autovec (relative) and TMA (absolute) agree at k=0.
        sA = smem.allocate_tensor(
            element_type=BF16_DTYPE,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=BF16_DTYPE,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        @cute.struct
        class Barriers:
            tma_mbar_a: cute.struct.MemRange[cutlass.Int64, 1]
            tma_mbar_b: cute.struct.MemRange[cutlass.Int64, 1]

        storage   = smem.allocate(Barriers)
        tma_mbar_a = storage.tma_mbar_a.data_ptr()
        tma_mbar_b = storage.tma_mbar_b.data_ptr()
        tma_bytes_a = cute.size_in_bytes(BF16_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2]))
        tma_bytes_b = cute.size_in_bytes(BF16_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2]))

        # TMA partition for A
        gA_tma  = cute.local_tile(mA_tma, CTA_TILE, (0, 0, None), proj=(1, None, 1))
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA    = thr_mma.partition_A(gA_tma)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )

        # TMA partition for B
        gB_tma  = cute.local_tile(mB_tma, CTA_TILE, (0, 0, None), proj=(None, 1, 1))
        tCgB    = thr_mma.partition_B(gB_tma)
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
        cute.arch.barrier(barrier_id=1, number_of_threads=THREADS)

        if warp_idx == 0:
            cute.copy(tma_atom_a, tAgA[None, 0], tAsA[None, 0], tma_bar_ptr=tma_mbar_a)
            cute.copy(tma_atom_b, tBgB[None, 0], tBsB[None, 0], tma_bar_ptr=tma_mbar_b)
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar_a, tma_bytes_a)
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar_b, tma_bytes_b)
        cute.arch.mbarrier_wait(tma_mbar_a, 0)
        cute.arch.mbarrier_wait(tma_mbar_b, 0)
        cute.arch.sync_threads()

        # Raw physical readback for sA
        sA_raw   = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=BF16_DTYPE),
                                    cute.make_layout(NUM_ELEMS_A))
        gOutA_raw = cute.make_tensor(cute.recast_ptr(mOutA.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        for k in range(N_PER_THR_A):
            gOutA_raw[tidx + k * THREADS] = sA_raw[tidx + k * THREADS]

        # Raw physical readback for sB
        sB_raw   = cute.make_tensor(cute.recast_ptr(sB.iterator, dtype=BF16_DTYPE),
                                    cute.make_layout(NUM_ELEMS_B))
        gOutB_raw = cute.make_tensor(cute.recast_ptr(mOutB.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        for k in range(N_PER_THR_B):
            gOutB_raw[tidx + k * THREADS] = sB_raw[tidx + k * THREADS]


def get_tma_reference(src_a: torch.Tensor, src_b: torch.Tensor):
    """Step 1: TMA G2S for sA and sB; return raw physical SMEM layouts."""
    from cutlass.cute.runtime import from_dlpack
    print("=" * 60)
    print("Step 1: TMA G2S — physical SMEM reference (sA and sB)")
    print("=" * 60)

    out_a = torch.zeros(BM, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    out_b = torch.zeros(BN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _Bf16TmaCopy()(from_dlpack(src_a), from_dlpack(src_b),
                   from_dlpack(out_a), from_dlpack(out_b))
    torch.cuda.synchronize()

    ref_a, ref_b = out_a.cpu(), out_b.cpu()
    print_chunk(ref_a, "TMA sA — physical SMEM [128×64]", n_rows=8)
    print()
    print_chunk(ref_b, "TMA sB — physical SMEM [64×64]", n_rows=8)
    print("\n  Step 1 done.")
    return ref_a, ref_b


# ── Step 2: cute.autovec_copy G→S ─────────────────────────────────────────────

class _Bf16AutovecCopy:
    """autovec_copy G→S into sA and sB; raw bf16 readback for physical layout."""

    NUM_STAGES = 1

    @cute.jit
    def __call__(
        self,
        mA:    cute.Tensor,
        mB:    cute.Tensor,
        mOutA: cute.Tensor,
        mOutB: cute.Tensor,
    ):
        op = tcgen05.MmaF16BF16Op(
            BF16_DTYPE, ACC_DTYPE, MMA_INST,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma     = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE, BF16_DTYPE, self.NUM_STAGES,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, CTA_TILE, BF16_DTYPE, self.NUM_STAGES,
        )
        self.kernel(tiled_mma, mA, mB, mOutA, mOutB, a_smem_layout, b_smem_layout).launch(
            grid=(1, 1, 1), block=(THREADS, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma:     cute.TiledMma,
        mA:            cute.Tensor,
        mB:            cute.Tensor,
        mOutA:         cute.Tensor,
        mOutB:         cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        # Same allocation order as TMA kernel: sA first, sB second.
        sA = smem.allocate_tensor(
            element_type=BF16_DTYPE,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=BF16_DTYPE,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        thr_mma    = tiled_mma.get_slice(thr_idx=0)
        thr_layout = cute.make_layout(THREADS)
        print("[Step2] thr_layout:          ", thr_layout)

        # ── autovec copy for A ────────────────────────────────────────
        gA_flat = cute.make_tensor(
            mA.iterator,
            cute.make_layout((BM, HEAD_DIM), stride=(HEAD_DIM, 1)),
        )
        print("[Step2] gA_flat:             ", gA_flat)
        print("[Step2] sA (full):           ", sA)
        print("[Step2] sA[None,None,None,0]:", sA[None, None, None, 0])
        tCgA   = thr_mma.partition_A(gA_flat)
        print("[Step2] tCgA (partition_A):  ", tCgA)
        sA_thr = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        print("[Step2] sA_thr (local_part): ", sA_thr)
        gA_thr = cute.local_partition(tCgA, thr_layout, tidx)
        print("[Step2] gA_thr (local_part): ", gA_thr)
        cute.autovec_copy(gA_thr, sA_thr)

        # ── autovec copy for B ────────────────────────────────────────
        gB_flat = cute.make_tensor(
            mB.iterator,
            cute.make_layout((BN, HEAD_DIM), stride=(HEAD_DIM, 1)),
        )
        print("[Step2] gB_flat:             ", gB_flat)
        print("[Step2] sB (full):           ", sB)
        print("[Step2] sB[None,None,None,0]:", sB[None, None, None, 0])
        tCgB   = thr_mma.partition_B(gB_flat)
        print("[Step2] tCgB (partition_B):  ", tCgB)
        sB_thr = cute.local_partition(sB[None, None, None, 0], thr_layout, tidx)
        print("[Step2] sB_thr (local_part): ", sB_thr)
        gB_thr = cute.local_partition(tCgB, thr_layout, tidx)
        print("[Step2] gB_thr (local_part): ", gB_thr)
        cute.autovec_copy(gB_thr, sB_thr)

        cute.arch.sync_threads()

        # Raw physical readback for sA
        sA_raw   = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=BF16_DTYPE),
                                    cute.make_layout(NUM_ELEMS_A))
        gOutA_raw = cute.make_tensor(cute.recast_ptr(mOutA.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        for k in range(N_PER_THR_A):
            gOutA_raw[tidx + k * THREADS] = sA_raw[tidx + k * THREADS]

        # Raw physical readback for sB
        sB_raw   = cute.make_tensor(cute.recast_ptr(sB.iterator, dtype=BF16_DTYPE),
                                    cute.make_layout(NUM_ELEMS_B))
        gOutB_raw = cute.make_tensor(cute.recast_ptr(mOutB.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        for k in range(N_PER_THR_B):
            gOutB_raw[tidx + k * THREADS] = sB_raw[tidx + k * THREADS]


def test_autovec_copy(
    src_a: torch.Tensor, src_b: torch.Tensor,
    ref_a: torch.Tensor, ref_b: torch.Tensor,
) -> None:
    """Step 2: autovec_copy G→S for sA and sB; compare vs TMA reference."""
    from cutlass.cute.runtime import from_dlpack
    print("=" * 60)
    print("Step 2: cute.autovec_copy G→S — compare vs TMA reference")
    print("=" * 60)

    out_a = torch.zeros(BM, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    out_b = torch.zeros(BN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _Bf16AutovecCopy()(from_dlpack(src_a), from_dlpack(src_b),
                       from_dlpack(out_a), from_dlpack(out_b))
    torch.cuda.synchronize()

    got_a, got_b = out_a.cpu(), out_b.cpu()

    diff_a = (got_a != ref_a).sum().item()
    diff_b = (got_b != ref_b).sum().item()
    match_a, match_b = diff_a == 0, diff_b == 0

    print(f"\n  sA autovec == TMA ref : {match_a}  ({diff_a}/{NUM_ELEMS_A} mismatches)")
    print(f"  sB autovec == TMA ref : {match_b}  ({diff_b}/{NUM_ELEMS_B} mismatches)")
    print()
    print_chunk(got_a, "autovec sA — physical SMEM [128×64]", n_rows=8)
    print()
    print_chunk(got_b, "autovec sB — physical SMEM [64×64]", n_rows=8)
    if not match_a:
        print()
        print_chunk(ref_a, "TMA ref sA", n_rows=8)
    if not match_b:
        print()
        print_chunk(ref_b, "TMA ref sB", n_rows=8)
    ok = match_a and match_b
    print(f"\n  Step 2 {'PASSED ✓' if ok else 'FAILED ✗'}")


# ── Step 3: cute.copy with cp.async G→S ──────────────────────────────────────

class _Bf16CpAsyncCopy:
    """cp.async G→S into sA and sB via TV tiled copy; raw bf16 readback."""

    NUM_STAGES = 1

    @cute.jit
    def __call__(
        self,
        mA:    cute.Tensor,
        mB:    cute.Tensor,
        mOutA: cute.Tensor,
        mOutB: cute.Tensor,
    ):
        op = tcgen05.MmaF16BF16Op(
            BF16_DTYPE, ACC_DTYPE, MMA_INST,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma     = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE, BF16_DTYPE, self.NUM_STAGES,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, CTA_TILE, BF16_DTYPE, self.NUM_STAGES,
        )
        self.kernel(tiled_mma, mA, mB, mOutA, mOutB, a_smem_layout, b_smem_layout).launch(
            grid=(1, 1, 1), block=(THREADS, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma:     cute.TiledMma,
        mA:            cute.Tensor,
        mB:            cute.Tensor,
        mOutA:         cute.Tensor,
        mOutB:         cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        # Same allocation order: sA first (byte 0), sB second (byte 16384).
        sA = smem.allocate_tensor(
            element_type=BF16_DTYPE,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=BF16_DTYPE,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        # ── bf16 GMEM views ───────────────────────────────────────────────
        # assumed_align=128 is baked into mA/mB via from_dlpack at call site.
        gA = cute.make_tensor(mA.iterator, cute.make_layout((BM, HEAD_DIM), stride=(HEAD_DIM, 1)))
        gB = cute.make_tensor(mB.iterator, cute.make_layout((BN, HEAD_DIM), stride=(HEAD_DIM, 1)))
        print("[Step3] gA:                  ", gA)
        print("[Step3] gB:                  ", gB)
        print("[Step3] sA (full):           ", sA)
        print("[Step3] sB (full):           ", sB)
        print("[Step3] sA[None,None,None,0]:", sA[None, None, None, 0])
        print("[Step3] sB[None,None,None,0]:", sB[None, None, None, 0])

        # ── cp.async copy atom — 128b per instruction, bf16 ───────────────
        atom_cpa = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            BF16_DTYPE, num_bits_per_copy=128,
        )
        print("[Step3] atom_cpa:            ", atom_cpa)

        # thr_layout: (64, 8) stride (64, 8) — 64 row-groups × 8 col-groups = 512 threads.
        # Thread (t_m, t_n) starts at t_m*64 + t_n*8; val (v,) adds v.
        # No overlap: bijection over [0, 4096). Tile = (64, 64) rows×cols.
        thr_layout = cute.make_layout((64, 8), stride=(64, 8))
        print("[Step3] thr_layout:          ", thr_layout)

        # ── copy A ────────────────────────────────────────────────────
        # val_layout = (8,): 8 bf16 per thread = 128b per atom.
        val_layout_a = cute.make_layout((8,), stride=(1,))
        print("[Step3] val_layout_a:        ", val_layout_a)
        tiled_copy_a = cute.make_tiled_copy_tv(atom_cpa, thr_layout, val_layout_a)
        print("[Step3] tiled_copy_a:        ", tiled_copy_a)
        thr_copy_a   = tiled_copy_a.get_slice(tidx)
        print("[Step3] thr_copy_a:          ", thr_copy_a)
        tAgA = thr_copy_a.partition_S(gA)
        print("[Step3] tAgA (partition_S):  ", tAgA)
        tAsA = thr_copy_a.partition_D(sA[None, None, None, 0])  # strip STAGE before partition
        print("[Step3] tAsA (partition_D):  ", tAsA)
        tAgA = cute.group_modes(tAgA, 1, cute.rank(tAgA))  # (CPY, Rest_all)
        print("[Step3] tAgA (grouped):      ", tAgA)
        tAsA = cute.group_modes(tAsA, 1, cute.rank(tAsA))  # (CPY, Rest_all)
        print("[Step3] tAsA (grouped):      ", tAsA)
        cute.copy(atom_cpa, tAgA, tAsA)

        # ── copy B ────────────────────────────────────────────────────
        # val_layout = (8,): 8 bf16 per thread = 128b per atom.
        val_layout_b = cute.make_layout((8,), stride=(1,))
        print("[Step3] val_layout_b:        ", val_layout_b)
        tiled_copy_b = cute.make_tiled_copy_tv(atom_cpa, thr_layout, val_layout_b)
        print("[Step3] tiled_copy_b:        ", tiled_copy_b)
        thr_copy_b   = tiled_copy_b.get_slice(tidx)
        print("[Step3] thr_copy_b:          ", thr_copy_b)
        tBgB = thr_copy_b.partition_S(gB)
        print("[Step3] tBgB (partition_S):  ", tBgB)
        tBsB = thr_copy_b.partition_D(sB[None, None, None, 0])  # strip STAGE before partition
        print("[Step3] tBsB (partition_D):  ", tBsB)
        tBgB = cute.group_modes(tBgB, 1, cute.rank(tBgB))  # (CPY, Rest_all)
        print("[Step3] tBgB (grouped):      ", tBgB)
        tBsB = cute.group_modes(tBsB, 1, cute.rank(tBsB))  # (CPY, Rest_all)
        print("[Step3] tBsB (grouped):      ", tBsB)
        cute.copy(atom_cpa, tBgB, tBsB)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # ── raw physical readback ─────────────────────────────────────
        sA_raw    = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        gOutA_raw = cute.make_tensor(cute.recast_ptr(mOutA.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        for k in range(N_PER_THR_A):
            gOutA_raw[tidx + k * THREADS] = sA_raw[tidx + k * THREADS]

        sB_raw    = cute.make_tensor(cute.recast_ptr(sB.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        gOutB_raw = cute.make_tensor(cute.recast_ptr(mOutB.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        for k in range(N_PER_THR_B):
            gOutB_raw[tidx + k * THREADS] = sB_raw[tidx + k * THREADS]


def test_cpasync_copy(
    src_a: torch.Tensor, src_b: torch.Tensor,
    ref_a: torch.Tensor, ref_b: torch.Tensor,
) -> None:
    """Step 3: cp.async G→S for sA and sB; compare vs TMA reference."""
    from cutlass.cute.runtime import from_dlpack
    print("=" * 60)
    print("Step 3: cute.copy cp.async G→S — compare vs TMA reference")
    print("=" * 60)

    out_a = torch.zeros(BM, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    out_b = torch.zeros(BN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _Bf16CpAsyncCopy()(from_dlpack(src_a, assumed_align=128), from_dlpack(src_b, assumed_align=128),
                       from_dlpack(out_a), from_dlpack(out_b))
    torch.cuda.synchronize()

    got_a, got_b = out_a.cpu(), out_b.cpu()

    diff_a = (got_a != ref_a).sum().item()
    diff_b = (got_b != ref_b).sum().item()
    match_a, match_b = diff_a == 0, diff_b == 0

    print(f"\n  sA cp.async == TMA ref : {match_a}  ({diff_a}/{NUM_ELEMS_A} mismatches)")
    print(f"  sB cp.async == TMA ref : {match_b}  ({diff_b}/{NUM_ELEMS_B} mismatches)")
    print()
    print_chunk(got_a, "cp.async sA — physical SMEM [128×64]", n_rows=8)
    print()
    print_chunk(got_b, "cp.async sB — physical SMEM [64×64]", n_rows=8)
    if not match_a:
        print()
        print_chunk(ref_a, "TMA ref sA", n_rows=8)
    if not match_b:
        print()
        print_chunk(ref_b, "TMA ref sB", n_rows=8)
    ok = match_a and match_b
    print(f"\n  Step 3 {'PASSED ✓' if ok else 'FAILED ✗'}")

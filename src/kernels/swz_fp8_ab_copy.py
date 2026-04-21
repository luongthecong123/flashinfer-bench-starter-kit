"""swz_fp8_ab_copy.py — Swizzled fp8 sA+sB tiles: TMA vs autovec copy comparison.

sA: [128, 128] Float8E4M3FN — 128 bytes/row × 128 rows = 16 384 bytes.
sB: [ 64, 128] Float8E4M3FN — 128 bytes/row ×  64 rows =  8 192 bytes.
Swizzle: Sw<3,4,3> for both (same 128-byte row width).

Input: src[row, col] = fp8(col)  (all rows identical; value ≈ column index 0..127).
Physical SMEM readback stored as fp8; displayed as float32.

Allocation order: sA first → sB → barriers.
  sA at byte 0:      k = (0     >> 7) & 7 = 0  → zero XOR bias.
  sB at byte 16384:  k = (16384 >> 7) & 7 = 0  → same zero bias.
  Both TMA (absolute) and autovec (relative) agree at k=0.

Step 1: TMA G2S for sA and sB — ground-truth physical SMEM layout.
Step 2: cute.autovec_copy G→S for sA and sB — compare vs TMA reference.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, cpasync
import cutlass.utils.blackwell_helpers as sm100_utils

# ── Constants ─────────────────────────────────────────────────────────────────
BM       = 128
BN       = 64
HEAD_DIM = 128

NUM_ELEMS_A = BM * HEAD_DIM   # 16 384
NUM_ELEMS_B = BN * HEAD_DIM   #  8 192

FP8_DTYPE  = cutlass.Float8E4M3FN
ACC_DTYPE  = cutlass.Float32
MMA_INST   = (128, 64, 32)       # fp8/fp8 K_inst=32
CTA_TILE   = (BM, BN, HEAD_DIM)  # (128, 64, 128)

THREADS     = 512
N_PER_THR_A = NUM_ELEMS_A // THREADS   # 32
N_PER_THR_B = NUM_ELEMS_B // THREADS   # 16


# ── Inputs ────────────────────────────────────────────────────────────────────

def make_fp8_input_a() -> torch.Tensor:
    """[BM, HEAD_DIM] fp8 E4M3FN, src[row, col] = fp8(col)."""
    cols = torch.arange(HEAD_DIM, dtype=torch.float32).unsqueeze(0)
    return cols.expand(BM, -1).clone().to(torch.float8_e4m3fn).cuda()

def make_fp8_input_b() -> torch.Tensor:
    """[BN, HEAD_DIM] fp8 E4M3FN, src[row, col] = fp8(col)."""
    cols = torch.arange(HEAD_DIM, dtype=torch.float32).unsqueeze(0)
    return cols.expand(BN, -1).clone().to(torch.float8_e4m3fn).cuda()


# ── Display helper ────────────────────────────────────────────────────────────

def print_chunk(t: torch.Tensor, label: str, n_rows: int, n_cols: int = 32):
    """Print first n_rows × n_cols fp8 values as float32 integers."""
    flat = t.cpu().to(torch.float32).reshape(t.shape[0], -1)
    print(f"  [{label}]")
    hdr = " ".join(f"{c:5d}" for c in range(n_cols))
    print(f"    {'':8s}  {hdr}")
    for r in range(n_rows):
        vals = " ".join(f"{flat[r, c].item():5.0f}" for c in range(n_cols))
        print(f"    row {r:3d}:  {vals}")


# ── Step 1: TMA G2S (reference) ───────────────────────────────────────────────

class _Fp8TmaCopy:
    """TMA G2S into sA and sB; raw fp8 readback exposes physical SMEM layout."""

    NUM_STAGES = 1

    @cute.jit
    def __call__(
        self,
        mA:    cute.Tensor,   # [BM, HEAD_DIM] fp8
        mB:    cute.Tensor,   # [BN, HEAD_DIM] fp8
        mOutA: cute.Tensor,   # [BM, HEAD_DIM] fp8 output
        mOutB: cute.Tensor,   # [BN, HEAD_DIM] fp8 output
    ):
        op = tcgen05.MmaFP8Op(
            FP8_DTYPE, ACC_DTYPE, MMA_INST,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma     = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE, FP8_DTYPE, self.NUM_STAGES,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, CTA_TILE, FP8_DTYPE, self.NUM_STAGES,
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

        # sA first (byte 0), sB second (byte 16384), barriers last.
        sA = smem.allocate_tensor(
            element_type=FP8_DTYPE,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=FP8_DTYPE,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        @cute.struct
        class Barriers:
            tma_mbar_a: cute.struct.MemRange[cutlass.Int64, 1]
            tma_mbar_b: cute.struct.MemRange[cutlass.Int64, 1]

        storage    = smem.allocate(Barriers)
        tma_mbar_a = storage.tma_mbar_a.data_ptr()
        tma_mbar_b = storage.tma_mbar_b.data_ptr()
        tma_bytes_a = cute.size_in_bytes(FP8_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2]))
        tma_bytes_b = cute.size_in_bytes(FP8_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2]))

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
        sA_raw    = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=FP8_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        gOutA_raw = cute.make_tensor(cute.recast_ptr(mOutA.iterator, dtype=FP8_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        for k in range(N_PER_THR_A):
            gOutA_raw[tidx + k * THREADS] = sA_raw[tidx + k * THREADS]

        # Raw physical readback for sB
        sB_raw    = cute.make_tensor(cute.recast_ptr(sB.iterator, dtype=FP8_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        gOutB_raw = cute.make_tensor(cute.recast_ptr(mOutB.iterator, dtype=FP8_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        for k in range(N_PER_THR_B):
            gOutB_raw[tidx + k * THREADS] = sB_raw[tidx + k * THREADS]


def get_tma_reference(src_a: torch.Tensor, src_b: torch.Tensor):
    """Step 1: TMA G2S for sA and sB; return raw physical SMEM layouts."""
    from cutlass.cute.runtime import from_dlpack
    print("=" * 60)
    print("Step 1: TMA G2S — physical SMEM reference (sA and sB)")
    print("=" * 60)

    out_a = torch.zeros(BM, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    out_b = torch.zeros(BN, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    _Fp8TmaCopy()(from_dlpack(src_a), from_dlpack(src_b),
                  from_dlpack(out_a), from_dlpack(out_b))
    torch.cuda.synchronize()

    ref_a, ref_b = out_a.cpu(), out_b.cpu()
    print_chunk(ref_a, "TMA sA — physical SMEM [128×128]", n_rows=8)
    print()
    print_chunk(ref_b, "TMA sB — physical SMEM [64×128]", n_rows=8)
    print("\n  Step 1 done.")
    return ref_a, ref_b


# ── Step 2: cute.autovec_copy G→S ─────────────────────────────────────────────

class _Fp8AutovecCopy:
    """autovec_copy G→S into sA and sB; raw fp8 readback for physical layout."""

    NUM_STAGES = 1

    @cute.jit
    def __call__(
        self,
        mA:    cute.Tensor,
        mB:    cute.Tensor,
        mOutA: cute.Tensor,
        mOutB: cute.Tensor,
    ):
        op = tcgen05.MmaFP8Op(
            FP8_DTYPE, ACC_DTYPE, MMA_INST,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma     = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE, FP8_DTYPE, self.NUM_STAGES,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, CTA_TILE, FP8_DTYPE, self.NUM_STAGES,
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
            element_type=FP8_DTYPE,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            element_type=FP8_DTYPE,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        thr_mma    = tiled_mma.get_slice(thr_idx=0)
        thr_layout = cute.make_layout(THREADS)

        # ── autovec copy for A ────────────────────────────────────────
        gA_flat = cute.make_tensor(
            mA.iterator,
            cute.make_layout((BM, HEAD_DIM), stride=(HEAD_DIM, 1)),
        )
        tCgA   = thr_mma.partition_A(gA_flat)
        sA_thr = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr = cute.local_partition(tCgA, thr_layout, tidx)
        cute.autovec_copy(gA_thr, sA_thr)

        # ── autovec copy for B ────────────────────────────────────────
        gB_flat = cute.make_tensor(
            mB.iterator,
            cute.make_layout((BN, HEAD_DIM), stride=(HEAD_DIM, 1)),
        )
        tCgB   = thr_mma.partition_B(gB_flat)
        sB_thr = cute.local_partition(sB[None, None, None, 0], thr_layout, tidx)
        gB_thr = cute.local_partition(tCgB, thr_layout, tidx)
        cute.autovec_copy(gB_thr, sB_thr)

        cute.arch.sync_threads()

        # Raw physical readback for sA
        sA_raw    = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=FP8_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        gOutA_raw = cute.make_tensor(cute.recast_ptr(mOutA.iterator, dtype=FP8_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        for k in range(N_PER_THR_A):
            gOutA_raw[tidx + k * THREADS] = sA_raw[tidx + k * THREADS]

        # Raw physical readback for sB
        sB_raw    = cute.make_tensor(cute.recast_ptr(sB.iterator, dtype=FP8_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        gOutB_raw = cute.make_tensor(cute.recast_ptr(mOutB.iterator, dtype=FP8_DTYPE),
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

    out_a = torch.zeros(BM, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    out_b = torch.zeros(BN, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    _Fp8AutovecCopy()(from_dlpack(src_a), from_dlpack(src_b),
                      from_dlpack(out_a), from_dlpack(out_b))
    torch.cuda.synchronize()

    got_a, got_b = out_a.cpu(), out_b.cpu()

    diff_a = (got_a.view(torch.uint8) != ref_a.view(torch.uint8)).sum().item()
    diff_b = (got_b.view(torch.uint8) != ref_b.view(torch.uint8)).sum().item()
    match_a, match_b = diff_a == 0, diff_b == 0

    print(f"\n  sA autovec == TMA ref : {match_a}  ({diff_a}/{NUM_ELEMS_A} mismatches)")
    print(f"  sB autovec == TMA ref : {match_b}  ({diff_b}/{NUM_ELEMS_B} mismatches)")
    print()
    print_chunk(got_a, "autovec sA — physical SMEM [128×128]", n_rows=8)
    print()
    print_chunk(got_b, "autovec sB — physical SMEM [64×128]", n_rows=8)
    if not match_a:
        print()
        print_chunk(ref_a, "TMA ref sA", n_rows=8)
    if not match_b:
        print()
        print_chunk(ref_b, "TMA ref sB", n_rows=8)
    ok = match_a and match_b
    print(f"\n  Step 2 {'PASSED ✓' if ok else 'FAILED ✗'}")

"""swz_bf16_score_copy.py — Swizzled bf16 sA+sB tiles for score kernel dims.

sA: [128, 512] bfloat16 — 1024 bytes/row × 128 rows = 131 072 bytes.
sB: [  8, 512] bfloat16 — 1024 bytes/row ×   8 rows =   8 192 bytes.
Swizzle: computed by sm100_utils for 1024-byte rows (both tiles same width).

Input: src[row, col] = col  (all rows identical; value = column index 0..511).
Each physical SMEM slot shows exactly which source column landed there.

Step 1: TMA G2S for sA and sB — ground-truth physical SMEM layout.  [done]
Step 2: cute.autovec_copy G→S for sA and sB — compare vs TMA reference.  [TODO]
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, cpasync
import cutlass.utils.blackwell_helpers as sm100_utils

# ── Constants ─────────────────────────────────────────────────────────────────
BM       = 128
BN       = 8       # N_MMA for score: tcgen05 minimum N
HEAD_DIM = 512     # K

NUM_ELEMS_A = BM * HEAD_DIM   # 65 536
NUM_ELEMS_B = BN * HEAD_DIM   #  4 096

BF16_DTYPE = cutlass.BFloat16
ACC_DTYPE  = cutlass.Float32
MMA_INST   = (128, BN, 16)          # bf16/bf16 K_inst=16
CTA_TILE   = (BM, BN, HEAD_DIM)     # (128, 8, 512)

THREADS     = 512
N_PER_THR_A = NUM_ELEMS_A // THREADS   # 128
N_PER_THR_B = NUM_ELEMS_B // THREADS   #   8


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

        # sA first → sB → barriers (same order as score_tcgen05.py).
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

        storage    = smem.allocate(Barriers)
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
        sA_raw    = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        gOutA_raw = cute.make_tensor(cute.recast_ptr(mOutA.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        for k in range(N_PER_THR_A):
            gOutA_raw[tidx + k * THREADS] = sA_raw[tidx + k * THREADS]

        # Raw physical readback for sB
        sB_raw    = cute.make_tensor(cute.recast_ptr(sB.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        gOutB_raw = cute.make_tensor(cute.recast_ptr(mOutB.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        for k in range(N_PER_THR_B):
            gOutB_raw[tidx + k * THREADS] = sB_raw[tidx + k * THREADS]


def get_tma_reference(src_a: torch.Tensor, src_b: torch.Tensor):
    """Step 1: TMA G2S for sA [128,512] and sB [8,512]; return raw physical layouts."""
    from cutlass.cute.runtime import from_dlpack
    print("=" * 60)
    print("Step 1: TMA G2S — physical SMEM reference (sA [128,512] and sB [8,512])")
    print("=" * 60)

    out_a = torch.zeros(BM, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    out_b = torch.zeros(BN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _Bf16TmaCopy()(from_dlpack(src_a), from_dlpack(src_b),
                   from_dlpack(out_a), from_dlpack(out_b))
    torch.cuda.synchronize()

    ref_a, ref_b = out_a.cpu(), out_b.cpu()
    print_chunk(ref_a, "TMA sA — physical SMEM [128×512]", n_rows=8)
    print()
    print_chunk(ref_b, "TMA sB — physical SMEM [8×512]", n_rows=8)
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

        thr_layout = cute.make_layout(THREADS)
        print("[Step2] thr_layout:          ", thr_layout)

        # Build a GMEM view that has the SAME logical layout shape as
        # sA[None,None,None,0] / sB[None,None,None,0] so autovec_copy can
        # distribute across the CTA. The SMEM logical layout is
        #   ((M, K_inner=16), 1, (K_oi=4, K_oo=K/64))
        # with K-inner stride 1, K_oi stride 16, K_oo stride 64 in K-elems.
        K_OI = 4
        K_OO_A = HEAD_DIM // (16 * K_OI)   # 8 for HEAD_DIM=512
        K_OO_B = HEAD_DIM // (16 * K_OI)

        gA_view = cute.make_tensor(
            mA.iterator,
            cute.make_layout(((BM, 16), 1, (K_OI, K_OO_A)),
                             stride=((HEAD_DIM, 1), 0, (16, 64))),
        )
        gB_view = cute.make_tensor(
            mB.iterator,
            cute.make_layout(((BN, 16), 1, (K_OI, K_OO_B)),
                             stride=((HEAD_DIM, 1), 0, (16, 64))),
        )
        print("[Step2] gA_view:             ", gA_view)
        print("[Step2] sA[...,0]:           ", sA[None, None, None, 0])
        print("[Step2] gB_view:             ", gB_view)
        print("[Step2] sB[...,0]:           ", sB[None, None, None, 0])

        # CTA-collective autovec_copy. Each thread gets its own slice
        # automatically; the swizzle wrapper on the SMEM side routes writes
        # to the correct physical bytes — TMA-equivalent placement.
        cute.autovec_copy(gA_view, sA[None, None, None, 0])
        cute.autovec_copy(gB_view, sB[None, None, None, 0])

        cute.arch.sync_threads()

        # Raw physical readback for sA
        sA_raw    = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        gOutA_raw = cute.make_tensor(cute.recast_ptr(mOutA.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_A))
        for k in range(N_PER_THR_A):
            gOutA_raw[tidx + k * THREADS] = sA_raw[tidx + k * THREADS]

        # Raw physical readback for sB
        sB_raw    = cute.make_tensor(cute.recast_ptr(sB.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        gOutB_raw = cute.make_tensor(cute.recast_ptr(mOutB.iterator, dtype=BF16_DTYPE),
                                     cute.make_layout(NUM_ELEMS_B))
        for k in range(N_PER_THR_B):
            gOutB_raw[tidx + k * THREADS] = sB_raw[tidx + k * THREADS]


def test_autovec_copy(
    src_a: torch.Tensor, src_b: torch.Tensor,
    ref_a: torch.Tensor, ref_b: torch.Tensor,
) -> bool:
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
    print_chunk(got_a, "autovec sA — physical SMEM [128×512]", n_rows=8)
    print()
    print_chunk(got_b, "autovec sB — physical SMEM [8×512]", n_rows=8)
    if not match_a:
        print()
        print_chunk(ref_a, "TMA ref sA", n_rows=8)
    if not match_b:
        print()
        print_chunk(ref_b, "TMA ref sB", n_rows=8)
    ok = match_a and match_b
    print(f"\n  Step 2 {'PASSED ✓' if ok else 'FAILED ✗'}")
    return ok


# ── Step 3: cute.copy with cp.async G→S ──────────────────────────────────────
#
# 128b atom = 16B = 8 bf16 per cp.async instruction.
# sA [128, 512]: 65 536 bf16 = 8 192 atoms → 16 atoms/thread (512 threads).
#   thr_layout (64, 8) stride (64, 8): 64 row-thr × 8 col-thr × 8 val
#   = (64, 64) per pass → 2 row-passes × 8 col-passes = 16 atoms/thread.
# sB [8, 512]:    4 096 bf16 =   512 atoms →  1 atom/thread.
#   thr_layout (8, 64) stride (8, 64): 8 row-thr × 64 col-thr × 8 val
#   = (8, 512) per pass → exactly 1 atom/thread.

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

        # ── bf16 GMEM views (assumed_align=128 baked in via from_dlpack) ──
        gA = cute.make_tensor(mA.iterator,
                              cute.make_layout((BM, HEAD_DIM), stride=(HEAD_DIM, 1)))
        gB = cute.make_tensor(mB.iterator,
                              cute.make_layout((BN, HEAD_DIM), stride=(HEAD_DIM, 1)))

        # Flat composed views of sA / sB with the SAME Sw<3,4,3> wrapper, so
        # partition_D sees a clean (BM, K) / (BN, K) shape and tile-divides
        # cleanly with the (64,64) / (8,512) thr×val tiles. The swizzle still
        # routes writes to the correct physical bytes.
        sA_flat_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0,
            cute.make_layout((BM, HEAD_DIM), stride=(HEAD_DIM, 1)),
        )
        sB_flat_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0,
            cute.make_layout((BN, HEAD_DIM), stride=(HEAD_DIM, 1)),
        )
        # Build fresh non-swizzled pointers from sA/sB raw byte addresses;
        # the composed layout attaches the Sw<3,4,3> wrapper itself. Two
        # swizzles on the same pointer is invalid.
        sA_ptr = cute.make_ptr(BF16_DTYPE,
                               cute.recast_ptr(sA.iterator, dtype=BF16_DTYPE).toint(),
                               mem_space=cute.AddressSpace.smem,
                               assumed_align=1024)
        sB_ptr = cute.make_ptr(BF16_DTYPE,
                               cute.recast_ptr(sB.iterator, dtype=BF16_DTYPE).toint(),
                               mem_space=cute.AddressSpace.smem,
                               assumed_align=1024)
        sA_flat = cute.make_tensor(sA_ptr, sA_flat_layout)
        sB_flat = cute.make_tensor(sB_ptr, sB_flat_layout)
        print("[Step3] gA:                  ", gA)
        print("[Step3] gB:                  ", gB)
        print("[Step3] sA_flat:             ", sA_flat)
        print("[Step3] sB_flat:             ", sB_flat)

        # ── 128b cp.async atom ───────────────────────────────────────
        atom_cpa = cute.make_copy_atom(
            cpasync.CopyG2SOp(),
            BF16_DTYPE, num_bits_per_copy=128,
        )
        print("[Step3] atom_cpa:            ", atom_cpa)

        # Use the SAME multi-mode SMEM tile as Step 2 (matches physical
        # bytes via the original swizzle) and a matching GMEM view.
        K_OI = 4
        K_OO_A = HEAD_DIM // (16 * K_OI)
        K_OO_B = HEAD_DIM // (16 * K_OI)
        gA_view = cute.make_tensor(
            mA.iterator,
            cute.make_layout(((BM, 16), 1, (K_OI, K_OO_A)),
                             stride=((HEAD_DIM, 1), 0, (16, 64))),
        )
        gB_view = cute.make_tensor(
            mB.iterator,
            cute.make_layout(((BN, 16), 1, (K_OI, K_OO_B)),
                             stride=((HEAD_DIM, 1), 0, (16, 64))),
        )
        sA_tile = sA[None, None, None, 0]
        sB_tile = sB[None, None, None, 0]

        # zipped_divide by VEC=8 along K_inner dim.
        VEC = 8
        gA_vec = cute.zipped_divide(gA_view, ((1, VEC), 1, (1, 1)))
        sA_vec = cute.zipped_divide(sA_tile, ((1, VEC), 1, (1, 1)))
        gB_vec = cute.zipped_divide(gB_view, ((1, VEC), 1, (1, 1)))
        sB_vec = cute.zipped_divide(sB_tile, ((1, VEC), 1, (1, 1)))
        print("[Step3] gA_vec:              ", gA_vec)
        print("[Step3] sA_vec:              ", sA_vec)
        print("[Step3] gB_vec:              ", gB_vec)
        print("[Step3] sB_vec:              ", sB_vec)

        nA_chunks = cute.size(gA_vec, mode=[1])
        nB_chunks = cute.size(gB_vec, mode=[1])
        N_CHUNK_A_PER_THR = nA_chunks // THREADS
        N_CHUNK_B_PER_THR = nB_chunks // THREADS
        print("[Step3] nA/THR/per:", nA_chunks, THREADS, N_CHUNK_A_PER_THR)
        print("[Step3] nB/THR/per:", nB_chunks, THREADS, N_CHUNK_B_PER_THR)

        for k in cutlass.range_constexpr(N_CHUNK_A_PER_THR):
            ck = tidx + k * THREADS
            cute.copy(atom_cpa, gA_vec[None, ck], sA_vec[None, ck])
        for k in cutlass.range_constexpr(N_CHUNK_B_PER_THR):
            ck = tidx + k * THREADS
            cute.copy(atom_cpa, gB_vec[None, ck], sB_vec[None, ck])

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # ── raw physical readback ────────────────────────────────────
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
) -> bool:
    """Step 3: cp.async G→S for sA and sB; compare vs TMA reference."""
    from cutlass.cute.runtime import from_dlpack
    print("=" * 60)
    print("Step 3: cute.copy cp.async G→S — compare vs TMA reference")
    print("=" * 60)

    out_a = torch.zeros(BM, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    out_b = torch.zeros(BN, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _Bf16CpAsyncCopy()(from_dlpack(src_a, assumed_align=128),
                       from_dlpack(src_b, assumed_align=128),
                       from_dlpack(out_a), from_dlpack(out_b))
    torch.cuda.synchronize()

    got_a, got_b = out_a.cpu(), out_b.cpu()
    diff_a = (got_a != ref_a).sum().item()
    diff_b = (got_b != ref_b).sum().item()
    match_a, match_b = diff_a == 0, diff_b == 0

    print(f"\n  sA cp.async == TMA ref : {match_a}  ({diff_a}/{NUM_ELEMS_A} mismatches)")
    print(f"  sB cp.async == TMA ref : {match_b}  ({diff_b}/{NUM_ELEMS_B} mismatches)")
    print()
    print_chunk(got_a, "cp.async sA — physical SMEM [128×512]", n_rows=8)
    print()
    print_chunk(got_b, "cp.async sB — physical SMEM [8×512]", n_rows=8)
    if not match_a:
        print()
        print_chunk(ref_a, "TMA ref sA", n_rows=8)
    if not match_b:
        print()
        print_chunk(ref_b, "TMA ref sB", n_rows=8)
    ok = match_a and match_b
    print(f"\n  Step 3 {'PASSED ✓' if ok else 'FAILED ✗'}")
    return ok


# ── Entry point ───────────────────────────────────────────────────────────────

def run(steps: int = 1) -> dict:
    """Run copy comparison steps and return a result dict.

    Args:
        steps: 1 = TMA only, 2 = TMA + autovec, 3 = TMA + autovec + cp.async.
    """
    import json

    src_a = make_bf16_input_a()
    src_b = make_bf16_input_b()

    ref_a, ref_b = get_tma_reference(src_a, src_b)
    results = {"step1_tma": "done"}

    if steps >= 2:
        ok2 = test_autovec_copy(src_a, src_b, ref_a, ref_b)
        results["step2_autovec"] = "PASS" if ok2 else "FAIL"

    if steps >= 3:
        ok3 = test_cpasync_copy(src_a, src_b, ref_a, ref_b)
        results["step3_cpasync"] = "PASS" if ok3 else "FAIL"

    return json.dumps(results)


if __name__ == "__main__":
    run(steps=1)

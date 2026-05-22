"""swz_fp8_copy.py — Test copy methods into Sw<3,4,3> sA fp8 tile.

Tests 4 methods of populating a swizzled sA SMEM tile:
  Step 1: pure-Python/PyTorch swizzle reference (this file)
  Step 2: TMA copy  (TODO)
  Step 3: cute.autovec_copy  (TODO)
  Step 4: cp.async TV-layout  (TODO)

Layout: sA = (128, 128) fp8, tcgen05 MMA tile, Sw<B=3,M=4,S=3>.
Input:  fp8 tensor [BM, HEAD_DIM] created via clamped fp32 (no NaN/Inf).

Sw<3,4,3> formula (byte offsets):
    swizzled = x ^ (((x >> 7) & 7) << 4)

  Decomposition: XOR bits [6:4] of x with bits [9:7] (= row & 7).
  The mapping is its own inverse (self-XOR), and is bijective on any
  range whose top bits (>=bit 7) are preserved.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.nvgpu import tcgen05, cpasync
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils

# ── Constants ────────────────────────────────────────────────────────────────
PAGE_ROWS    = 64      # rows per KV page
HEAD_DIM     = 128     # fp8 cols per row
BM           = 128     # M-tile = 2 pages
MMA_INST_MNK = (128, 64, 32)
CTA_TILE_MNK = (BM, 64, HEAD_DIM)
FP8_DTYPE    = cutlass.Float8E4M3FN
CLAMP_MAX    = 240.0   # well below fp8 E4M3FN max (448.0); avoids boundary values
NUM_TOTAL    = BM * HEAD_DIM   # 16 384


# ── Swizzle formula ──────────────────────────────────────────────────────────

def _swizzle_343(byte_off: int) -> int:
    """Sw<B=3, M=4, S=3>: x ^ (((x >> 7) & 7) << 4).
    
    XORs bits [6:4] of byte_off with bits [9:7] (= row_in_tile & 7).
    Self-inverse bijection on [0, NUM_TOTAL).
    """
    return byte_off ^ (((byte_off >> 7) & 7) << 4)


# ── Input generation ─────────────────────────────────────────────────────────

def make_fp8_input(seed: int = 42) -> torch.Tensor:
    """Return [BM, HEAD_DIM] fp8 E4M3FN tensor (on CUDA).
    
    Created by clamping fp32 randn to ±CLAMP_MAX before cast to fp8,
    guaranteeing no NaN (fp8 E4M3FN NaN = 0x7F / 0xFF, only at ±448).
    """
    torch.manual_seed(seed)
    fp32 = torch.randn(BM, HEAD_DIM, device="cuda") * 20.0
    fp32 = fp32.clamp(-CLAMP_MAX, CLAMP_MAX)
    return fp32.to(torch.float8_e4m3fn)


# ── Step 1: pure-Python/PyTorch reference ────────────────────────────────────

def compute_swz343_reference(fp8_src: torch.Tensor) -> torch.Tensor:
    """Compute the expected physical SMEM byte layout after Sw<3,4,3>.

    For each logical element at (row, col), the fp8 byte at linear offset
    (col + HEAD_DIM * row) lands at physical SMEM byte swizzle_343(linear).

    Args:
        fp8_src: [BM, HEAD_DIM] fp8 tensor (any device).
    Returns:
        ref: [NUM_TOTAL] uint8 tensor (CPU) — expected SMEM physical bytes.
             ref[swizzle_343(i)] == src_bytes[i]  for all i.
    """
    rows, cols = fp8_src.shape
    assert rows == BM and cols == HEAD_DIM, \
        f"Expected [{BM},{HEAD_DIM}], got {list(fp8_src.shape)}"

    src_bytes = fp8_src.cpu().view(torch.uint8).flatten()   # [16384]
    ref = torch.zeros(NUM_TOTAL, dtype=torch.uint8)

    for row in range(rows):
        for col in range(cols):
            linear = col + cols * row
            swz    = _swizzle_343(linear)
            ref[swz] = src_bytes[linear]

    return ref


def test_reference() -> tuple:
    """Step 1: build fp8 input + PyTorch swizzle reference; verify correctness.

    Checks:
      1. No NaN bytes in fp8 input.
      2. ref[swizzle_343(i)] == src_bytes[i]  for all i  (exact mapping).
      3. ref is a permutation of src_bytes  (bijection sanity).

    Returns: (fp8_src, ref) for downstream comparison with kernel outputs.
    """
    print("=" * 60)
    print("Step 1: PyTorch swizzle_343 reference")
    print("=" * 60)

    fp8_src = make_fp8_input()
    print(f"  fp8_src shape : {fp8_src.shape}, dtype: {fp8_src.dtype}")
    print(f"  device        : {fp8_src.device}")

    # ── Check 1: no NaN in fp8 input ─────────────────────────────────
    raw = fp8_src.cpu().view(torch.uint8).flatten()
    # fp8 E4M3FN NaN: exp=1111, mant=111  →  lower 7 bits all 1  (0x7F)
    nan_mask = (raw & 0x7F) == 0x7F
    n_nan = nan_mask.sum().item()
    print(f"\n  [Check 1] NaN count in fp8 input : {n_nan}  (expected 0)")
    assert n_nan == 0, f"fp8 input contains {n_nan} NaN values!"
    print("  [Check 1] PASSED")

    # ── Compute reference ─────────────────────────────────────────────
    ref = compute_swz343_reference(fp8_src)
    print(f"\n  ref shape : {ref.shape}, dtype: {ref.dtype}")

    # ── Check 2: exact mapping  ref[swz(i)] == src_bytes[i] ─────────
    src_bytes = fp8_src.cpu().view(torch.uint8).flatten()
    mismatches = 0
    for i in range(NUM_TOTAL):
        swz = _swizzle_343(i)
        if ref[swz].item() != src_bytes[i].item():
            mismatches += 1
            if mismatches <= 3:
                print(f"    MISMATCH i={i}: ref[{swz}]={ref[swz].item()} "
                      f"!= src[{i}]={src_bytes[i].item()}")
    print(f"\n  [Check 2] Exact mapping mismatches : {mismatches}  (expected 0)")
    assert mismatches == 0, f"Reference has {mismatches} mapping errors!"
    print("  [Check 2] PASSED")

    # ── Print sample mapping for visual inspection ────────────────────
    print("\n  Sample: src (logical) → SMEM (physical) for first 8 elements:")
    print(f"  {'i':>5}  {'(row,col)':>10}  {'byte_val':>10}  {'swz_off':>8}  {'smem(r,c)':>12}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*12}")
    for i in range(8):
        row, col = divmod(i, HEAD_DIM)
        swz = _swizzle_343(i)
        sr, sc = divmod(swz, HEAD_DIM)
        bval = src_bytes[i].item()
        print(f"  {i:>5}  ({row:>3},{col:>3})      0x{bval:02x}       {swz:>6}   ({sr:>3},{sc:>3})")

    # Also print the XOR group pattern for rows 0..7 col 0
    print("\n  XOR group check — col=0, rows 0..7:")
    print(f"  {'row':>4}  {'linear':>8}  {'swz':>8}  {'swz_col':>8}")
    for row in range(8):
        lin = 0 + HEAD_DIM * row
        swz = _swizzle_343(lin)
        print(f"  {row:>4}  {lin:>8}  {swz:>8}  {swz%HEAD_DIM:>8}")

    print("\n  Step 1 PASSED ✓")
    return fp8_src, ref


# ── Step 2: TMA copy into sA ─────────────────────────────────────────────────
# Load [BM, HEAD_DIM] fp8 into sA via TMA (tcgen05 bulk-copy).
# Readback via symmetric S→G using partition_A + local_partition + autovec_copy.
# Final mOut [BM, HEAD_DIM] fp8 should equal fp8_src exactly.
#
# NOTE: raw physical byte comparison against the Python swizzle_343 ref does NOT
# work because a_smem_layout.outer is a hierarchical layout (not plain row-major).
# The logical S→G readback is the correct check.

THREADS_STEP2 = 128          # 4 warps


class _SwzTmaCopy:
    """Step 2 kernel: TMA G2S into sA, then symmetric S→G readback."""

    NUM_STAGES = 1

    @cute.jit
    def __call__(self, mA: cute.Tensor, mOut: cute.Tensor):
        """mA: [BM, HEAD_DIM] fp8  |  mOut: [BM, HEAD_DIM] fp8 output."""
        op = tcgen05.MmaFP8Op(
            FP8_DTYPE, cutlass.Float32, MMA_INST_MNK,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE_MNK, FP8_DTYPE, self.NUM_STAGES,
        )
        a_smem_stage0 = cute.select(a_smem_layout, mode=[0, 1, 2])
        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op_g2s, mA, a_smem_stage0, CTA_TILE_MNK, tiled_mma,
        )
        self.kernel(
            tiled_mma, tma_atom_a, tma_tensor_a, mOut, a_smem_layout,
        ).launch(grid=(1, 1, 1), block=(THREADS_STEP2, 1, 1))

    @cute.kernel
    def kernel(
        self,
        tiled_mma:     cute.TiledMma,
        tma_atom_a:    cute.CopyAtom,
        mA_tma:        cute.Tensor,        # TMA descriptor view
        mOut:          cute.Tensor,        # [BM, HEAD_DIM] fp8 — readback dest
        a_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _  = cute.arch.thread_idx()
        warp_idx    = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Barriers:
            tma_mbar: cute.struct.MemRange[cutlass.Int64, 1]

        storage = smem.allocate(Barriers)
        sA = smem.allocate_tensor(
            element_type=FP8_DTYPE,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )

        # ── TMA partition ─────────────────────────────────────────────
        gA_tma  = cute.local_tile(mA_tma, CTA_TILE_MNK, (0, 0, None), proj=(1, None, 1))
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA    = thr_mma.partition_A(gA_tma)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )

        tma_mbar  = storage.tma_mbar.data_ptr()
        tma_bytes = cute.size_in_bytes(FP8_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2]))

        if warp_idx == 0:
            if tidx == 0:
                cute.arch.mbarrier_init(tma_mbar, cnt=1)
                cute.arch.mbarrier_init_fence()

        cute.arch.barrier(barrier_id=1, number_of_threads=THREADS_STEP2)

        # ── Fire TMA (warp 0) ────────────────────────────────────────
        if warp_idx == 0:
            cute.copy(tma_atom_a, tAgA[None, 0], tAsA[None, 0], tma_bar_ptr=tma_mbar)
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_bytes)

        cute.arch.mbarrier_wait(tma_mbar, 0)
        cute.arch.sync_threads()

        # -- Raw byte readback: flat i32 ptr, NO CuTe layout --
        # Symmetric autovec S->G would cancel the swizzle and always
        # return fp8_src regardless of whether the swizzle was correct.
        N_I32     = BM * HEAD_DIM // 4
        N_PER_THR = N_I32 // THREADS_STEP2
        sA_raw   = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=cutlass.Int32),
                                    cute.make_layout(N_I32))
        gOut_raw = cute.make_tensor(cute.recast_ptr(mOut.iterator, dtype=cutlass.Int32),
                                    cute.make_layout(N_I32))
        for k in range(N_PER_THR):
            gOut_raw[tidx + k * THREADS_STEP2] = sA_raw[tidx + k * THREADS_STEP2]


def test_step2_tma(fp8_src: torch.Tensor, ref: torch.Tensor) -> None:
    """Step 2: TMA into sA; logical S→G readback; compare output vs fp8_src."""
    from cutlass.cute.runtime import from_dlpack
    print("=" * 60)
    print("Step 2: TMA copy into sA  [readback via partition_A S→G]")
    print("=" * 60)

    mOut = torch.zeros(BM, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    fn = _SwzTmaCopy()
    fn(from_dlpack(fp8_src), from_dlpack(mOut))
    torch.cuda.synchronize()

    got_u8  = mOut.cpu().view(torch.uint8).flatten()
    exp_u8  = ref   # uint8 [NUM_TOTAL] -- expected physical SMEM layout
    ok = torch.equal(got_u8, exp_u8)
    print(f"  [Check] raw SMEM bytes == swizzle_343 ref : {ok}  (expected True)")
    if not ok:
        diff = (got_u8 != exp_u8)
        n_err = diff.sum().item()
        idx = diff.flatten().nonzero(as_tuple=True)[0][0].item()
        g, e = got_u8[idx].item(), exp_u8[idx].item()
        print(f"  {n_err} mismatches; first at byte {idx}: got=0x{g:02x} expected=0x{e:02x}")
        assert False, f"Step 2 TMA: {n_err} mismatches!"
    print("  Step 2 PASSED ✓")


# ── Step 3: autovec_copy G→S into sA ─────────────────────────────────────────
# Directly copy [BM, HEAD_DIM] fp8 from GMEM into swizzled sA using
# cute.autovec_copy (thread-parallel load/store).
# Readback via the same symmetric S→G pattern as Step 2.

THREADS_STEP3 = 128


class _SwzAutovecCopy:
    """Step 3 kernel: autovec_copy G→S into sA, then symmetric S→G readback."""

    NUM_STAGES = 1

    @cute.jit
    def __call__(self, mA: cute.Tensor, mOut: cute.Tensor):
        """mA: [BM, HEAD_DIM] fp8  |  mOut: [BM, HEAD_DIM] fp8 output."""
        op = tcgen05.MmaFP8Op(
            FP8_DTYPE, cutlass.Float32, MMA_INST_MNK,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE_MNK, FP8_DTYPE, self.NUM_STAGES,
        )
        self.kernel(tiled_mma, mA, mOut, a_smem_layout).launch(
            grid=(1, 1, 1), block=(THREADS_STEP3, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma:     cute.TiledMma,
        mA:            cute.Tensor,        # [BM, HEAD_DIM] fp8 GMEM input
        mOut:          cute.Tensor,        # [BM, HEAD_DIM] fp8 GMEM output
        a_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(
            element_type=FP8_DTYPE,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )

        # ── Build logical GMEM view and partition across threads ──────
        gA_flat = cute.make_tensor(
            mA.iterator,
            cute.make_layout((BM, HEAD_DIM), stride=(HEAD_DIM, 1)),
        )
        thr_mma  = tiled_mma.get_slice(thr_idx=0)
        tCgA     = thr_mma.partition_A(gA_flat)
        thr_layout = cute.make_layout(THREADS_STEP3)
        sA_thr   = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr   = cute.local_partition(tCgA, thr_layout, tidx)

        # ── Copy G→S ─────────────────────────────────────────────────
        cute.autovec_copy(gA_thr, sA_thr)
        cute.arch.sync_threads()

        # ── Raw byte readback: flat i32 pointer, NO CuTe layout ──────
        # Reads physical SMEM bytes as-is so we can verify the swizzle
        # pattern against the Python swizzle_343 reference.
        # Symmetric autovec S→G would cancel the swizzle and always
        # give back fp8_src regardless of whether swizzle was correct.
        N_I32     = BM * HEAD_DIM // 4          # 4096 i32s = 16384 bytes
        N_PER_THR = N_I32 // THREADS_STEP3      # 32 per thread
        sA_raw   = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=cutlass.Int32),
                                    cute.make_layout(N_I32))
        gOut_raw = cute.make_tensor(cute.recast_ptr(mOut.iterator, dtype=cutlass.Int32),
                                    cute.make_layout(N_I32))
        for k in range(N_PER_THR):
            gOut_raw[tidx + k * THREADS_STEP3] = sA_raw[tidx + k * THREADS_STEP3]


def test_step3_autovec(fp8_src: torch.Tensor, ref: torch.Tensor) -> None:
    """Step 3: autovec_copy G→S into sA; symmetric S→G readback; compare vs fp8_src."""
    from cutlass.cute.runtime import from_dlpack
    print("=" * 60)
    print("Step 3: autovec_copy G→S into sA  [readback via S→G]")
    print("=" * 60)

    mOut = torch.zeros(BM, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    fn = _SwzAutovecCopy()
    fn(from_dlpack(fp8_src), from_dlpack(mOut))
    torch.cuda.synchronize()

    # mOut now contains the RAW physical SMEM bytes of sA.
    # ref[swizzle_343(i)] == src_bytes[i]  encodes the expected physical layout.
    got_u8 = mOut.cpu().view(torch.uint8).flatten()
    exp_u8 = ref   # uint8 [NUM_TOTAL] — expected physical SMEM layout
    ok = torch.equal(got_u8, exp_u8)
    print(f"  [Check] raw SMEM bytes == swizzle_343 ref : {ok}  (expected True)")
    if not ok:
        diff = (got_u8 != exp_u8)
        n_err = diff.sum().item()
        idx = diff.flatten().nonzero(as_tuple=True)[0][0].item()
        g, e = got_u8[idx].item(), exp_u8[idx].item()
        print(f"  {n_err} mismatches; first at byte {idx}: got=0x{g:02x} expected=0x{e:02x}")
        assert False, f"Step 3 autovec: {n_err} mismatches!"
    print("  Step 3 PASSED ✓")


# ── Step 4: cp.async TV-layout G→S into sA ───────────────────────────────────
# Load [BM, HEAD_DIM] fp8 into swizzled sA via 128-thread cp.async using a
# TV-layout tiled copy on an i32 view of both GMEM and SMEM.
#
# SMEM view: Sw<3,2,3>∘row_major on i32 — byte Sw<3,4,3> / 4 = i32 Sw<3,2,3>.
# CRITICAL: sA must be allocated FIRST (SMEM offset 0). Sw<3,2,3> on recast_ptr
# does NOT subtract the SMEM base, so a non-zero offset XORs against absolute
# address bits -> wrong bytes. See repo memory: cpasync-mma-A-load-SOLVED.md.
#
# Readback: raw flat i32 pointer (same as Step 3) -> compare vs swizzle_343 ref.

THREADS_STEP4 = 128
HEAD_DIM_I32  = HEAD_DIM // 4   # 32  (4 fp8 bytes per i32)


class _SwzCpAsyncCopy:
    """Step 4 kernel: cp.async TV-layout G->S into sA, then raw i32 readback."""

    NUM_STAGES = 1

    @cute.jit
    def __call__(self, mA: cute.Tensor, mOut: cute.Tensor):
        """mA: [BM, HEAD_DIM] fp8  |  mOut: [BM, HEAD_DIM] fp8 output."""
        op = tcgen05.MmaFP8Op(
            FP8_DTYPE, cutlass.Float32, MMA_INST_MNK,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, CTA_TILE_MNK, FP8_DTYPE, self.NUM_STAGES,
        )
        self.kernel(mA, mOut, a_smem_layout).launch(
            grid=(1, 1, 1), block=(THREADS_STEP4, 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        mA:            cute.Tensor,        # [BM, HEAD_DIM] fp8 GMEM input
        mOut:          cute.Tensor,        # [BM, HEAD_DIM] fp8 GMEM output
        a_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        # CRITICAL: sA at SMEM offset 0 — allocate before anything else.
        sA = smem.allocate_tensor(
            element_type=FP8_DTYPE,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )

        # -- Sw<3,2,3>*row_major i32 view of sA --
        # byte Sw<3,4,3>(x) on offset x  ==  i32 Sw<3,2,3>(x//4) on x//4
        sA_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BM, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1)),
        )
        sA_i32_ptr = cute.recast_ptr(sA.iterator, dtype=cutlass.Int32)
        sA_load    = cute.make_tensor(sA_i32_ptr, sA_load_layout)

        # -- GMEM source: i32 view, explicit assumed_align=4 --
        gA_i32_base = cute.make_ptr(
            cutlass.Int32,
            cute.recast_ptr(mA.iterator, dtype=cutlass.Int32).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=4,
        )
        gA_i32 = cute.make_tensor(
            gA_i32_base,
            cute.make_layout((BM, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1)),
        )

        # -- cp.async TV tiled copy --
        # thr_layout (4, 32): 128 threads across the (BM, HEAD_DIM_I32) tile
        # val_layout (32, 1): 32 i32s per thread  (128 * 32 = 4096 i32s total)
        N_PER_THREAD_I32 = (BM * HEAD_DIM_I32) // THREADS_STEP4   # 32
        atom_cpa = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32, num_bits_per_copy=cutlass.Int32.width,
        )
        thr_layout_load = cute.make_layout(
            (THREADS_STEP4 // HEAD_DIM_I32, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1),
        )  # (4, 32)
        val_layout_load = cute.make_layout((N_PER_THREAD_I32, 1), stride=(1, 1))  # (32, 1)
        tiled_copy_a = cute.make_tiled_copy_tv(atom_cpa, thr_layout_load, val_layout_load)
        thr_copy_a   = tiled_copy_a.get_slice(tidx)
        tAgA = thr_copy_a.partition_S(gA_i32)
        tAsA = thr_copy_a.partition_D(sA_load)
        cute.copy(atom_cpa, tAgA, tAsA)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # -- Raw byte readback: flat i32 ptr, no CuTe layout --
        N_I32     = BM * HEAD_DIM // 4        # 4096
        N_PER_THR = N_I32 // THREADS_STEP4    # 32
        sA_raw   = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=cutlass.Int32),
                                    cute.make_layout(N_I32))
        gOut_raw = cute.make_tensor(cute.recast_ptr(mOut.iterator, dtype=cutlass.Int32),
                                    cute.make_layout(N_I32))
        for k in range(N_PER_THR):
            gOut_raw[tidx + k * THREADS_STEP4] = sA_raw[tidx + k * THREADS_STEP4]


def test_step4_cpasync(fp8_src: torch.Tensor, ref: torch.Tensor) -> None:
    """Step 4: cp.async TV-layout G->S into sA; raw i32 readback; compare vs ref."""
    from cutlass.cute.runtime import from_dlpack
    print("=" * 60)
    print("Step 4: cp.async TV-layout G->S into sA  [raw i32 readback]")
    print("=" * 60)

    mOut = torch.zeros(BM, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    fn = _SwzCpAsyncCopy()
    fn(from_dlpack(fp8_src), from_dlpack(mOut))
    torch.cuda.synchronize()

    got_u8 = mOut.cpu().view(torch.uint8).flatten()
    exp_u8 = ref   # uint8 [NUM_TOTAL] -- expected physical SMEM layout
    ok = torch.equal(got_u8, exp_u8)
    print(f"  [Check] raw SMEM bytes == swizzle_343 ref : {ok}  (expected True)")
    if not ok:
        diff = (got_u8 != exp_u8)
        n_err = diff.sum().item()
        idx = diff.flatten().nonzero(as_tuple=True)[0][0].item()
        g, e = got_u8[idx].item(), exp_u8[idx].item()
        print(f"  {n_err} mismatches; first at byte {idx}: got=0x{g:02x} expected=0x{e:02x}")
        assert False, f"Step 4 cp.async: {n_err} mismatches!"
    print("  Step 4 PASSED ✓")


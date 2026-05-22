"""load_a_v2_swizzled_cmp.py — byte-level equivalence test of two A-load paths.

Two SMEM A-buffers are allocated side-by-side, each laid out with the canonical
UMMA-swizzled layout (`a_smem_layout` from sm100_utils):

  smem_ref   ← filled by the autovec_copy pattern from
               score_scale_full_bt.py (uses thr_mma.partition_A(gA) + flat
               threadlayout + cute.autovec_copy).

  smem_impl  ← filled by the cp.async TV-layout pattern from
               load_a_v2_swizzled.py (Int32 view + Sw<3,2,3>∘row_major,
               make_tiled_copy_tv + cute.copy on a CopyG2SOp atom).

After both copies finish (sync_threads + cp_async_wait_group), every thread
reads BM*HEAD_DIM/THREADS bytes from each buffer as Uint8 and XORs them
(scalar fp8→fp32 cast is illegal in MLIR; XOR on the byte view is exact
since identical bytes ⇔ identical fp8 values for pure copies).  Result is
written to a global `diff` tensor; host asserts `diff.abs().max() == 0`.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from src.kernels.load_a_common import (
    globaltimer_u64, smid_u32,
    range_start, range_stop, range_finalize,
    PROBE_HEADER, PROBE_ENTRY, PROBE_COLS,
)

TAGS = {"total": 0, "ref_autovec": 1, "impl_cpasync": 2}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["total", "ref_autovec", "impl_cpasync"]

# ── Workload constants (mirror load_a_common) ────────────────────────────────
PAGE_SIZE       = 64
HEAD_DIM        = 128
ROW_STRIDE      = 132            # bytes per kv row: 128 fp8 + 4 scale
PAGES_PER_TILE  = 2
BM, BK          = 128, HEAD_DIM
N               = 64             # for tiled_mma B side (unused for copy)
NUM_PAGES_POOL  = 11923
HEAD_DIM_I32    = HEAD_DIM // 4  # 32
ROW_STRIDE_I32  = ROW_STRIDE // 4  # 33

THREADS_PER_CTA = 128            # match the autovec path in score_scale_full_bt
NUM_THREADS_LOAD = 512           # cp.async path (matches load_a_v2_swizzled)
# Use the larger thread count so both paths can use the same CTA.
THREADS_CTA = NUM_THREADS_LOAD


class LoadAV2SwizzledCmp:
    def __init__(self):
        self.threads_per_cta = THREADS_CTA

    @cute.jit
    def __call__(self, kv_pool, block_table, q, diff, probe, stream):
        self.fp8_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32

        op = tcgen05.MmaFP8Op(
            self.fp8_dtype, self.acc_dtype, (128, 64, 32),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma     = cute.make_tiled_mma(op)
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, (BM, N, BK), self.fp8_dtype, 1)

        num_pg = cute.size(block_table, mode=[0])
        grid_m = num_pg // PAGES_PER_TILE
        self.kernel(self.tiled_mma, kv_pool, block_table, diff, probe,
                    self.a_smem_layout).launch(
            grid=(grid_m, 1, 1), block=(self.threads_per_cta, 1, 1),
            stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma, kv_pool, block_table, diff, probe, a_smem_layout):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        # Two raw byte blocks, 128-B aligned, BM*HEAD_DIM bytes each.
        sA_ref_raw  = smem.allocate(BM * HEAD_DIM, byte_alignment=128)
        sA_impl_raw = smem.allocate(BM * HEAD_DIM, byte_alignment=128)

        # Both buffers viewed as the canonical UMMA fp8 PDSL tensor.
        # `a_smem_layout` is a ComposedLayout (Sw<3,4,3> ∘ atom_tiled_outer).
        # We attach the swizzle to the pointer (PDSL) and bind the *outer*
        # layout — this is what `make_fragment_A` (and the full kernel) expect.
        sA_swiz = cute.make_swizzle(3, 4, 3)
        sA_ref_fp8_ptr  = cute.recast_ptr(sA_ref_raw,  sA_swiz, dtype=self.fp8_dtype)
        sA_impl_fp8_ptr = cute.recast_ptr(sA_impl_raw, sA_swiz, dtype=self.fp8_dtype)
        sA_ref  = cute.make_tensor(sA_ref_fp8_ptr,  a_smem_layout.outer)
        sA_impl = cute.make_tensor(sA_impl_fp8_ptr, a_smem_layout.outer)

        # ── gA (fp8, 2D BM×HEAD_DIM, dynamic stride between pages) ──────────
        page0_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 0])
        page1_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 1])
        page_stride_b = PAGE_SIZE * ROW_STRIDE
        page0_off_b   = page0_id * page_stride_b
        jump_b        = (page1_id - page0_id) * page_stride_b
        fp8_base = cute.recast_ptr(kv_pool.iterator, dtype=self.fp8_dtype) + page0_off_b
        gA = cute.make_tensor(fp8_base, cute.make_layout(
            ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM),
            stride=((ROW_STRIDE, jump_b), 1),
        ))

        # ── REF fill: autovec_copy pattern (score_scale_full_bt.py) ─────────
        # Use only the first THREADS_PER_CTA threads, mirroring the original.
        sm = cutlass.Int64(smid_u32())
        if tidx == 0:
            range_start(probe, bidx, cutlass.Int32(0), sm, TAGS["total"])
            range_start(probe, bidx, cutlass.Int32(1), sm, TAGS["ref_autovec"])

        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA    = thr_mma.partition_A(gA)
        if tidx < THREADS_PER_CTA:
            thr_layout = cute.make_layout(THREADS_PER_CTA)
            sA_ref_thr = cute.local_partition(sA_ref[None, None, None, 0], thr_layout, tidx)
            gA_thr     = cute.local_partition(tCgA, thr_layout, tidx)
            cute.autovec_copy(gA_thr, sA_ref_thr)
        cute.arch.sync_threads()

        if tidx == 0:
            range_stop(probe, bidx, cutlass.Int32(1))
            range_start(probe, bidx, cutlass.Int32(2), sm, TAGS["impl_cpasync"])

        # ── IMPL fill: cp.async TV-layout pattern (load_a_v2_swizzled) ──────
        # Build an Int32 view of the SAME smem block + Sw<3,2,3>∘row_major.
        sA_impl_i32_ptr = cute.recast_ptr(sA_impl_raw, dtype=cutlass.Int32)
        sA_impl_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BM, HEAD_DIM_I32),
                             stride=(HEAD_DIM_I32, 1)),
        )
        sA_impl_load = cute.make_tensor(sA_impl_i32_ptr, sA_impl_load_layout)

        # Int32 view of gA.
        page_stride_i32 = page_stride_b // 4
        page0_off_i32   = page0_id * page_stride_i32
        jump_i32        = (page1_id - page0_id) * page_stride_i32
        i32_base = cute.make_ptr(
            cutlass.Int32,
            (cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Int32) + page0_off_i32).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=4,
        )
        gA_i32 = cute.make_tensor(i32_base, cute.make_layout(
            ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM_I32),
            stride=((ROW_STRIDE_I32, jump_i32), 1),
        ))

        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32, num_bits_per_copy=cutlass.Int32.width,
        )
        # Quack-blog TV layout: (16, 32) thr × (8, 1) val = (128, 32) tile, 512 thr.
        N_PER_THREAD = (BM * HEAD_DIM_I32) // NUM_THREADS_LOAD  # 8
        thr_layout = cute.make_layout((16, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1))
        val_layout = cute.make_layout((N_PER_THREAD, 1),  stride=(1, 1))
        tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        tAgA = thr_copy.partition_S(gA_i32)
        tAsA = thr_copy.partition_D(sA_impl_load)
        cute.copy(atom, tAgA, tAsA)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if tidx == 0:
            range_stop(probe, bidx, cutlass.Int32(2))
            range_stop(probe, bidx, cutlass.Int32(0))   # close 'total'
            range_finalize(probe, bidx, cutlass.Int32(3))

        # ── Compare: each thread reads its share of bytes from both views ───
        # Read both buffers as Uint8 with a row-major (BM, HEAD_DIM) layout, so
        # logical (m, k) hits the same physical byte in each (the swizzle is on
        # the pointer in both views).  XOR the bytes — 0 means identical.
        diff_tile = diff[bidx, None, None]   # (BM, HEAD_DIM) uint8

        sA_ref_u8_ptr  = cute.recast_ptr(sA_ref_raw,  sA_swiz, dtype=cutlass.Uint8)
        sA_impl_u8_ptr = cute.recast_ptr(sA_impl_raw, sA_swiz, dtype=cutlass.Uint8)
        sA_ref_flat  = cute.make_tensor(sA_ref_u8_ptr,  cute.make_layout(
            (BM, HEAD_DIM), stride=(HEAD_DIM, 1)))
        sA_impl_flat = cute.make_tensor(sA_impl_u8_ptr, cute.make_layout(
            (BM, HEAD_DIM), stride=(HEAD_DIM, 1)))

        ELEMS_PER_THREAD = (BM * HEAD_DIM) // self.threads_per_cta
        for i in cutlass.range_constexpr(ELEMS_PER_THREAD):
            flat = tidx * ELEMS_PER_THREAD + i
            m = flat // HEAD_DIM
            k = flat - m * HEAD_DIM
            r = sA_ref_flat[m, k]
            s = sA_impl_flat[m, k]
            diff_tile[m, k] = r ^ s


# ── Host-side launcher / test ────────────────────────────────────────────────
def run_test(num_pg: int = 4, seed: int = 0):
    device = "cuda"
    torch.manual_seed(seed)
    assert num_pg % 2 == 0, "num_pg must be even"

    K_fp8 = (torch.randn(num_pg, PAGE_SIZE, HEAD_DIM, device=device)
             .clamp(-100, 100).to(torch.float8_e4m3fn))
    K_scl = torch.rand(num_pg, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5

    kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE,
                          device=device, dtype=torch.uint8)
    bt = torch.randperm(NUM_PAGES_POOL - 1, device=device)[:num_pg].to(torch.int32) + 1
    for i in range(num_pg):
        pid = int(bt[i].item())
        kv_pool[pid, :, :HEAD_DIM] = K_fp8[i].view(torch.uint8)
        kv_pool[pid, :, HEAD_DIM:HEAD_DIM + 4] = (
            K_scl[i].view(torch.uint8).reshape(PAGE_SIZE, 4))

    grid_m = num_pg // PAGES_PER_TILE
    diff = torch.zeros(grid_m, BM, HEAD_DIM, device=device, dtype=torch.uint8)
    probe = torch.zeros((grid_m, PROBE_COLS), dtype=torch.int64, device=device)

    kv_pool_ = from_dlpack(kv_pool, assumed_align=16, enable_tvm_ffi=True)
    bt_      = from_dlpack(bt,      assumed_align=4,  enable_tvm_ffi=True)
    diff_    = from_dlpack(diff,    assumed_align=16, enable_tvm_ffi=True)
    probe_   = from_dlpack(probe,   assumed_align=16, enable_tvm_ffi=True)
    # `q` is unused on-device but the JIT signature wants a tensor.
    q = torch.zeros(N, HEAD_DIM, device=device, dtype=torch.float8_e4m3fn)
    q_ = from_dlpack(q, assumed_align=16, enable_tvm_ffi=True)

    # Compile with fake-stream + tvm-ffi (env-stream resolved at call time).
    NUM_PG = cute.sym_int(divisibility=2)
    NUM_BL = cute.sym_int()
    fake_kv = make_fake_compact_tensor(dtype=cute.Uint8,
        shape=(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE), stride_order=(2, 1, 0), assumed_align=16)
    fake_bt = make_fake_compact_tensor(dtype=cute.Int32, shape=(NUM_PG,), stride_order=(0,), assumed_align=4)
    fake_q  = make_fake_compact_tensor(dtype=cute.Float8E4M3FN, shape=(N, HEAD_DIM), stride_order=(1, 0), assumed_align=16)
    fake_diff = make_fake_compact_tensor(dtype=cute.Uint8, shape=(NUM_BL, BM, HEAD_DIM), stride_order=(2, 1, 0), assumed_align=16)
    fake_probe = make_fake_compact_tensor(dtype=cute.Int64, shape=(NUM_BL, PROBE_COLS), stride_order=(1, 0), assumed_align=16)
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = LoadAV2SwizzledCmp()
    compiled = cute.compile(kernel, fake_kv, fake_bt, fake_q, fake_diff, fake_probe, fake_stream,
                            options="--enable-tvm-ffi")
    # Warm-up
    for _ in range(3):
        probe.zero_()
        compiled(kv_pool_, bt_, q_, diff_, probe_)
    torch.cuda.synchronize()
    # Timed run
    probe.zero_()
    compiled(kv_pool_, bt_, q_, diff_, probe_)
    torch.cuda.synchronize()

    max_abs = diff.abs().max().item()
    nz      = (diff != 0).sum().item()
    print(f"grid_m={grid_m}  diff.max()={max_abs}  nonzero={nz}")
    if max_abs == 0:
        print("PASS: byte-level equivalence — both A-load paths agree.")
    else:
        print("FAIL: paths diverge.")
        idx = (diff != 0).nonzero()[:8]
        for row in idx.tolist():
            b, m, k = row
            print(f"  diff[b={b}, m={m}, k={k}] = {diff[b, m, k].item()}")

    # ── Latency dump per phase ───────────────────────────────────────────────
    p = probe.cpu().tolist()
    totals = {n: 0 for n in PHASE_ORDER}
    counts = {n: 0 for n in PHASE_ORDER}
    for bid in range(grid_m):
        cnt = int(p[bid][0])
        for i in range(cnt):
            off = PROBE_HEADER + i * PROBE_ENTRY
            tag = int(p[bid][off + 1])
            dur = int(p[bid][off + 3])
            name = TAG_NAMES.get(tag, f"tag_{tag}")
            if name in totals:
                totals[name] += dur
                counts[name] += 1
    print(f"\n{'Phase':>16s} {'Avg (ns)':>12s} {'Avg (µs)':>12s} {'Count':>8s}")
    print(f"{'-'*52}")
    for name in PHASE_ORDER:
        c = counts[name] or 1
        avg = totals[name] / c
        print(f"{name:>16s} {avg:>12.1f} {avg/1000:>12.3f} {counts[name]:>8d}")
    return float(max_abs)


if __name__ == "__main__":
    run_test()

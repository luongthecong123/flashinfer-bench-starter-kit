"""load_a_v2_swizzled_cmp_faithful.py — atom-tiled byte-equivalence test.

Successor to load_a_v2_swizzled_cmp.py. The original cmp test compared the
two A-load paths via a row_major Uint8 readback view, which only proved that
the two writes agreed *under that view*. It did NOT validate against the
layout the MMA actually reads from (`a_smem_layout.outer`, which is
**atom-tiled**, NOT row_major). As a result the original cmp passed even
when integrating into the full kernel produced max_err≈347 (cp.async
wrote bytes via row_major addresses; MMA dereferenced atom-tiled
addresses → wrong bytes).

This test reads back via `a_smem_layout.outer` ∘ Sw<3,4,3> on a Uint8
PDSL view of the SAME raw SMEM block — i.e., the EXACT physical
addresses the MMA's `make_fragment_A` would dereference. If the cp.async
path lands bytes anywhere other than where MMA reads, the XOR diff is
non-zero.

Per-thread comparison uses the same `local_partition(...,
thr_layout=COMPUTE_THREADS)` pattern as the autovec_copy path in
score_scale_full_bt_ws_cpasync_intra.py, so the cmp tracks the actual
working kernel structure.
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

# Match the full kernel's structure: 512-thread CTA, 128 compute threads
# (autovec writes only on threads [0, COMPUTE_THREADS)).
COMPUTE_THREADS  = 128
NUM_THREADS_LOAD = 512
THREADS_CTA      = NUM_THREADS_LOAD


class LoadAV2SwizzledCmpFaithful:
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

        # Canonical UMMA fp8 PDSL view (Sw<3,4,3> ∘ atom_tiled_outer) — what
        # `make_fragment_A` and the MMA actually use.
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

        # ── REF fill: autovec_copy via partition_A (the WORKING path) ───────
        # Mirrors score_scale_full_bt_ws_cpasync_intra.py exactly: only the
        # first COMPUTE_THREADS=128 threads write, using local_partition over
        # the MMA-atom-tiled SMEM view.
        sm = cutlass.Int64(smid_u32())
        if tidx == 0:
            range_start(probe, bidx, cutlass.Int32(0), sm, TAGS["total"])
            range_start(probe, bidx, cutlass.Int32(1), sm, TAGS["ref_autovec"])

        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA    = thr_mma.partition_A(gA)
        if tidx < COMPUTE_THREADS:
            thr_layout_compute = cute.make_layout(COMPUTE_THREADS)
            sA_ref_thr = cute.local_partition(sA_ref[None, None, None, 0],
                                              thr_layout_compute, tidx)
            gA_thr     = cute.local_partition(tCgA, thr_layout_compute, tidx)
            cute.autovec_copy(gA_thr, sA_ref_thr)
        cute.arch.sync_threads()

        if tidx == 0:
            range_stop(probe, bidx, cutlass.Int32(1))
            range_start(probe, bidx, cutlass.Int32(2), sm, TAGS["impl_cpasync"])

        # ── IMPL fill: cp.async TV-layout pattern (load_a_v2_swizzled) ──────
        # This is the path under test. Writes via Sw<3,2,3>∘row_major Int32
        # view — the SAME pattern integrated into the full kernel.
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
            range_stop(probe, bidx, cutlass.Int32(0))
            range_finalize(probe, bidx, cutlass.Int32(3))

        # ════════════════════════════════════════════════════════════════════
        # FAITHFUL READBACK: read via the atom-tiled layout MMA actually uses.
        # ════════════════════════════════════════════════════════════════════
        # Build Uint8 PDSL views over the SAME atom-tiled outer layout. The
        # swizzle is bound to the byte-pointer (Sw<3,4,3> XORs byte-address
        # bits), and Uint8 is also 1 byte, so the swizzle behaves identically.
        # Indexing these tensors by logical (m, k) coords reaches the EXACT
        # same physical addresses MMA's `make_fragment_A` dereferences.
        sA_ref_u8_ptr  = cute.recast_ptr(sA_ref_raw,  sA_swiz, dtype=cutlass.Uint8)
        sA_impl_u8_ptr = cute.recast_ptr(sA_impl_raw, sA_swiz, dtype=cutlass.Uint8)
        sA_ref_u8  = cute.make_tensor(sA_ref_u8_ptr,  a_smem_layout.outer)
        sA_impl_u8 = cute.make_tensor(sA_impl_u8_ptr, a_smem_layout.outer)

        # Per-thread split via the SAME local_partition the autovec path used,
        # ensuring per-thread slice topology matches the MMA-aligned write.
        # 512 threads × ELEMS_PER_THREAD = BM × HEAD_DIM = 16384 bytes.
        thr_layout_cmp = cute.make_layout(self.threads_per_cta)
        sA_ref_thr_u8  = cute.local_partition(sA_ref_u8[None, None, None, 0],
                                              thr_layout_cmp, tidx)
        sA_impl_thr_u8 = cute.local_partition(sA_impl_u8[None, None, None, 0],
                                              thr_layout_cmp, tidx)

        # Each thread XORs its share into diff[bidx, tidx, :]. Non-zero diff
        # ⇒ cp.async wrote to addresses MMA does not read from.
        ELEMS_PER_THREAD = cute.size(sA_ref_thr_u8)
        for i in cutlass.range_constexpr(ELEMS_PER_THREAD):
            r = sA_ref_thr_u8[i]
            s = sA_impl_thr_u8[i]
            diff[bidx, tidx, i] = r ^ s


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

    grid_m           = num_pg // PAGES_PER_TILE
    elems_per_thread = (BM * HEAD_DIM) // THREADS_CTA   # 32
    diff = torch.zeros(grid_m, THREADS_CTA, elems_per_thread,
                       device=device, dtype=torch.uint8)
    probe = torch.zeros((grid_m, PROBE_COLS), dtype=torch.int64, device=device)

    kv_pool_ = from_dlpack(kv_pool, assumed_align=16, enable_tvm_ffi=True)
    bt_      = from_dlpack(bt,      assumed_align=4,  enable_tvm_ffi=True)
    diff_    = from_dlpack(diff,    assumed_align=16, enable_tvm_ffi=True)
    probe_   = from_dlpack(probe,   assumed_align=16, enable_tvm_ffi=True)
    q = torch.zeros(N, HEAD_DIM, device=device, dtype=torch.float8_e4m3fn)
    q_ = from_dlpack(q, assumed_align=16, enable_tvm_ffi=True)

    NUM_PG = cute.sym_int(divisibility=2)
    NUM_BL = cute.sym_int()
    fake_kv = make_fake_compact_tensor(dtype=cute.Uint8,
        shape=(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE), stride_order=(2, 1, 0), assumed_align=16)
    fake_bt = make_fake_compact_tensor(dtype=cute.Int32, shape=(NUM_PG,), stride_order=(0,), assumed_align=4)
    fake_q  = make_fake_compact_tensor(dtype=cute.Float8E4M3FN, shape=(N, HEAD_DIM), stride_order=(1, 0), assumed_align=16)
    fake_diff = make_fake_compact_tensor(dtype=cute.Uint8,
        shape=(NUM_BL, THREADS_CTA, elems_per_thread), stride_order=(2, 1, 0), assumed_align=16)
    fake_probe = make_fake_compact_tensor(dtype=cute.Int64,
        shape=(NUM_BL, PROBE_COLS), stride_order=(1, 0), assumed_align=16)
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = LoadAV2SwizzledCmpFaithful()
    compiled = cute.compile(kernel, fake_kv, fake_bt, fake_q, fake_diff, fake_probe, fake_stream,
                            options="--enable-tvm-ffi")

    for _ in range(3):
        probe.zero_()
        compiled(kv_pool_, bt_, q_, diff_, probe_)
    torch.cuda.synchronize()

    probe.zero_()
    compiled(kv_pool_, bt_, q_, diff_, probe_)
    torch.cuda.synchronize()

    max_abs = diff.abs().max().item()
    nz      = (diff != 0).sum().item()
    print(f"grid_m={grid_m}  diff.max()={max_abs}  nonzero={nz}/{diff.numel()}")
    if max_abs == 0:
        print("PASS: atom-tiled byte-equivalence — cp.async writes "
              "land at the addresses MMA reads from.")
    else:
        print("FAIL: cp.async writes diverge from MMA's read addresses.")
        # Show first few mismatched (bidx, tid, i) tuples.
        idx = (diff != 0).nonzero()[:8]
        for row in idx.tolist():
            b, t, i = row
            print(f"  diff[bidx={b}, tid={t}, i={i}] = {diff[b, t, i].item()}")
        # Per-thread fail count summary (top 5)
        per_thread_fail = (diff != 0).sum(dim=(0, 2))
        topk = torch.topk(per_thread_fail, k=5)
        print("  top-5 failing tids (across all blocks):")
        for cnt, tid in zip(topk.values.tolist(), topk.indices.tolist()):
            print(f"    tid={tid:3d}  fails={cnt}")

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

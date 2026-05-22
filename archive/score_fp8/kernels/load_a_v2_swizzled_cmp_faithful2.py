"""load_a_v2_swizzled_cmp_faithful2.py — TRULY faithful cmp.

The previous faithful cmp built ref/impl from RAW byte allocs and bound the
swizzle via recast_ptr. The full kernel uses `smem.allocate_tensor(swizzle=...)`
which bakes the swizzle into the PDSL iterator itself. This test mirrors that
exactly:

  sA_ref  = smem.allocate_tensor(fp8, a_smem_layout.outer, swizzle=a_smem_layout.inner)
  sA_impl = smem.allocate_tensor(fp8, a_smem_layout.outer, swizzle=a_smem_layout.inner)

  # cp.async target: i32 view derived from sA_impl.iterator
  sA_impl_load = make_tensor(recast_ptr(sA_impl.iterator, i32), Sw<3,2,3>∘row_major)

Comparison is done via the FP8 PDSL views (the EXACT way MMA reads bytes).
Each thread XORs its share through the fp8 PDSL — if cp.async + iterator
swizzle interaction lands bytes anywhere differently, this XOR will be non-zero.

Per the user's request: also dump raw bytes for bit-level inspection on FAIL.
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

PAGE_SIZE       = 64
HEAD_DIM        = 128
ROW_STRIDE      = 132
PAGES_PER_TILE  = 2
BM, BK          = 128, HEAD_DIM
N               = 64
NUM_PAGES_POOL  = 11923
HEAD_DIM_I32    = HEAD_DIM // 4
ROW_STRIDE_I32  = ROW_STRIDE // 4

COMPUTE_THREADS  = 128
NUM_THREADS_LOAD = 512
THREADS_CTA      = NUM_THREADS_LOAD


class LoadAV2SwizzledCmpFaithful2:
    def __init__(self):
        self.threads_per_cta = THREADS_CTA

    @cute.jit
    def __call__(self, kv_pool, block_table, q, diff, ref_raw, impl_raw, probe, stream):
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
        self.kernel(self.tiled_mma, kv_pool, block_table, diff, ref_raw,
                    impl_raw, probe, self.a_smem_layout).launch(
            grid=(grid_m, 1, 1), block=(self.threads_per_cta, 1, 1),
            stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma, kv_pool, block_table, diff, ref_raw, impl_raw,
               probe, a_smem_layout):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        # ── EXACT integrated-kernel pattern: allocate_tensor with swizzle ──
        sA_ref = smem.allocate_tensor(
            element_type=self.fp8_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sA_impl = smem.allocate_tensor(
            element_type=self.fp8_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )

        # ── gA (fp8) ────────────────────────────────────────────────────────
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

        # ── IMPL fill: cp.async via i32 view DERIVED FROM sA_impl.iterator ──
        # This is what the integrated kernel does: recast_ptr(sA.iterator, i32)
        # and bind a NEW Sw<3,2,3>∘row_major composed layout.
        sA_impl_i32_ptr = cute.recast_ptr(sA_impl.iterator, dtype=cutlass.Int32)
        sA_impl_load_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 2, 3), 0,
            cute.make_layout((BM, HEAD_DIM_I32),
                             stride=(HEAD_DIM_I32, 1)),
        )
        sA_impl_load = cute.make_tensor(sA_impl_i32_ptr, sA_impl_load_layout)

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
        N_PER_THREAD = (BM * HEAD_DIM_I32) // NUM_THREADS_LOAD
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
        cute.arch.fence_view_async_shared()

        if tidx == 0:
            range_stop(probe, bidx, cutlass.Int32(2))
            range_stop(probe, bidx, cutlass.Int32(0))
            range_finalize(probe, bidx, cutlass.Int32(3))

        # ════════════════════════════════════════════════════════════════════
        # COMPARE via the FP8 PDSL views — the EXACT way MMA reads.
        # ════════════════════════════════════════════════════════════════════
        # Per-thread split via local_partition over the MMA-atom-tiled fp8 view.
        thr_layout_cmp = cute.make_layout(self.threads_per_cta)
        sA_ref_thr_fp8  = cute.local_partition(sA_ref[None, None, None, 0],
                                               thr_layout_cmp, tidx)
        sA_impl_thr_fp8 = cute.local_partition(sA_impl[None, None, None, 0],
                                               thr_layout_cmp, tidx)

        ELEMS_PER_THREAD = cute.size(sA_ref_thr_fp8)
        # Cast fp8 → i32 (bit reinterpret via recast_tensor) before XOR, since
        # fp8→fp32 cast is illegal in MLIR. Each thread's slice is contiguous
        # in fp8 so we can recast_tensor to a uint8 view safely.
        sA_ref_thr_u8  = cute.recast_tensor(sA_ref_thr_fp8,  dtype=cutlass.Uint8)
        sA_impl_thr_u8 = cute.recast_tensor(sA_impl_thr_fp8, dtype=cutlass.Uint8)
        for i in cutlass.range_constexpr(ELEMS_PER_THREAD):
            r = sA_ref_thr_u8[i]
            s = sA_impl_thr_u8[i]
            diff[bidx, tidx, i] = r ^ s

        # ── Also dump RAW bytes (logical-by-logical via fp8 PDSL) for the
        # first CTA so host can do bit-level inspection. We dump using the
        # SAME local_partition slicing into per-thread rows.
        if bidx == 0:
            for i in cutlass.range_constexpr(ELEMS_PER_THREAD):
                ref_raw[tidx, i]  = sA_ref_thr_u8[i]
                impl_raw[tidx, i] = sA_impl_thr_u8[i]


def run_test(num_pg: int = 4, seed: int = 0):
    device = "cuda"
    torch.manual_seed(seed)
    assert num_pg % 2 == 0

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
    ELEMS_PER_THREAD = (BM * HEAD_DIM) // THREADS_CTA  # 32
    diff     = torch.zeros(grid_m, THREADS_CTA, ELEMS_PER_THREAD,
                           device=device, dtype=torch.uint8)
    ref_raw  = torch.zeros(THREADS_CTA, ELEMS_PER_THREAD,
                           device=device, dtype=torch.uint8)
    impl_raw = torch.zeros(THREADS_CTA, ELEMS_PER_THREAD,
                           device=device, dtype=torch.uint8)
    probe    = torch.zeros((grid_m, PROBE_COLS), dtype=torch.int64, device=device)

    kv_pool_  = from_dlpack(kv_pool,  assumed_align=16, enable_tvm_ffi=True)
    bt_       = from_dlpack(bt,       assumed_align=4,  enable_tvm_ffi=True)
    diff_     = from_dlpack(diff,     assumed_align=16, enable_tvm_ffi=True)
    ref_raw_  = from_dlpack(ref_raw,  assumed_align=16, enable_tvm_ffi=True)
    impl_raw_ = from_dlpack(impl_raw, assumed_align=16, enable_tvm_ffi=True)
    probe_    = from_dlpack(probe,    assumed_align=16, enable_tvm_ffi=True)
    q = torch.zeros(N, HEAD_DIM, device=device, dtype=torch.float8_e4m3fn)
    q_ = from_dlpack(q, assumed_align=16, enable_tvm_ffi=True)

    NUM_PG = cute.sym_int(divisibility=2)
    NUM_BL = cute.sym_int()
    fake_kv = make_fake_compact_tensor(dtype=cute.Uint8,
        shape=(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE), stride_order=(2, 1, 0), assumed_align=16)
    fake_bt = make_fake_compact_tensor(dtype=cute.Int32, shape=(NUM_PG,), stride_order=(0,), assumed_align=4)
    fake_q  = make_fake_compact_tensor(dtype=cute.Float8E4M3FN, shape=(N, HEAD_DIM), stride_order=(1, 0), assumed_align=16)
    fake_diff = make_fake_compact_tensor(dtype=cute.Uint8, shape=(NUM_BL, THREADS_CTA, ELEMS_PER_THREAD), stride_order=(2, 1, 0), assumed_align=16)
    fake_raw  = make_fake_compact_tensor(dtype=cute.Uint8, shape=(THREADS_CTA, ELEMS_PER_THREAD), stride_order=(1, 0), assumed_align=16)
    fake_probe = make_fake_compact_tensor(dtype=cute.Int64, shape=(NUM_BL, PROBE_COLS), stride_order=(1, 0), assumed_align=16)
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = LoadAV2SwizzledCmpFaithful2()
    compiled = cute.compile(kernel, fake_kv, fake_bt, fake_q, fake_diff,
                            fake_raw, fake_raw, fake_probe, fake_stream,
                            options="--enable-tvm-ffi")
    for _ in range(3):
        probe.zero_()
        compiled(kv_pool_, bt_, q_, diff_, ref_raw_, impl_raw_, probe_)
    torch.cuda.synchronize()
    probe.zero_()
    compiled(kv_pool_, bt_, q_, diff_, ref_raw_, impl_raw_, probe_)
    torch.cuda.synchronize()

    max_abs = diff.abs().max().item()
    nz      = (diff != 0).sum().item()
    print(f"\ngrid_m={grid_m}  diff.max()={max_abs}  nonzero={nz}/{diff.numel()}")
    if max_abs == 0:
        print("PASS: cp.async writes (via sA.iterator i32 view) are byte-identical "
              "to autovec writes when read through the FP8 PDSL — so MMA sees "
              "the same bytes. The full-kernel bug is NOT in the cp.async path.")
    else:
        print("FAIL: cp.async + sA.iterator i32 view diverges from autovec under "
              "the FP8 PDSL readback. The full-kernel bug IS this divergence.")
        # Bit-level inspection: find first 32 differing positions and dump bits.
        ref_b  = ref_raw.cpu()
        impl_b = impl_raw.cpu()
        delta  = ref_b ^ impl_b
        idx = delta.nonzero()[:32]
        print(f"\n  First {len(idx)} differing (thread, elem) positions in CTA0:")
        print(f"  {'thr':>4s} {'elem':>4s} {'ref':>10s} {'impl':>10s} {'xor':>10s}")
        for row in idx.tolist():
            t, i = row
            r = int(ref_b[t, i].item());  rb = format(r, '08b')
            s = int(impl_b[t, i].item()); sb = format(s, '08b')
            x = int(delta[t, i].item());  xb = format(x, '08b')
            print(f"  {t:>4d} {i:>4d}   {rb}   {sb}   {xb}")
    return float(max_abs)


if __name__ == "__main__":
    run_test()

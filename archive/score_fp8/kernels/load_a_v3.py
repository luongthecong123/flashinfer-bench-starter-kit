"""load-A v3: 2 SMEM buffers — cp.async into linear, then smem→smem swizzle.

Pipeline:
  Phase 1 (gmem_to_lin): cp.async fire-and-forget GMEM → linear SMEM (v2 path).
  Phase 2 (lin_to_swiz): autovec_copy linear SMEM → UMMA-swizzled SMEM via
                         partition_A. Pure on-chip data movement.

This is what UMMA needs end-to-end. Both phases are timed individually plus
total load_A.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from src.kernels.load_a_common import (
    globaltimer_u64, smid_u32,
    range_start, range_stop, range_finalize,
    PROBE_HEADER, PROBE_ENTRY, PROBE_COLS,
    PAGE_SIZE, HEAD_DIM, ROW_STRIDE, PAGES_PER_TILE, BM, BN, BK, N,
    NUM_PAGES_POOL, WORKLOAD_CASES,
)

NUM_THREADS = 512
TAGS = {"total": 0, "load_A": 2, "gmem_to_lin": 4, "lin_to_swiz": 6}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["total", "load_A", "gmem_to_lin", "lin_to_swiz"]

ROW_STRIDE_I32 = ROW_STRIDE // 4
HEAD_DIM_I32   = HEAD_DIM   // 4


class LoadAV3:
    def __init__(self):
        self.threads_per_cta = NUM_THREADS
        self.num_stages      = 1
        self.cta_tile_mnk    = (BM, BN, BK)
        self.mma_inst_mnk    = (128, 64, 32)

    @cute.jit
    def __call__(self, kv_pool, block_table, sink, probe, stream):
        self.fp8_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32

        op = tcgen05.MmaFP8Op(
            self.fp8_dtype, self.acc_dtype, self.mma_inst_mnk,
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        )
        self.tiled_mma = cute.make_tiled_mma(op)
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.cta_tile_mnk, self.fp8_dtype, self.num_stages)

        num_pg = cute.size(block_table, mode=[0])
        grid_m = num_pg // PAGES_PER_TILE

        self.kernel(
            self.tiled_mma, kv_pool, block_table, sink, probe,
            self.a_smem_layout,
        ).launch(grid=(grid_m, 1, 1), block=(self.threads_per_cta, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma, kv_pool, block_table, sink, probe, a_smem_layout):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        # Buffer 1: linear (non-swizzled) Int32 SMEM (gather target).
        sLin = smem.allocate_array(cutlass.Int32,
                                   num_elems=BM * HEAD_DIM_I32)
        # Buffer 2: UMMA-swizzled SMEM (final).
        sA   = smem.allocate_tensor(
            self.fp8_dtype, a_smem_layout.outer, 128, a_smem_layout.inner)

        page0_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 0])
        page1_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 1])
        page_stride_b   = PAGE_SIZE * ROW_STRIDE
        page_stride_i32 = page_stride_b // 4
        page0_off_i32   = page0_id * page_stride_i32
        jump_i32        = (page1_id - page0_id) * page_stride_i32

        # GMEM Int32 view (4-byte aligned).
        i32_base = cute.make_ptr(
            cutlass.Int32,
            (cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Int32) + page0_off_i32).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=4,
        )
        gA = cute.make_tensor(i32_base, cute.make_layout(
            (HEAD_DIM_I32, PAGE_SIZE, PAGES_PER_TILE),
            stride=(1, ROW_STRIDE_I32, jump_i32),
        ))
        # Linear SMEM Int32 view, same logical shape.
        sLinT = cute.make_tensor(sLin, cute.make_layout(
            (HEAD_DIM_I32, PAGE_SIZE, PAGES_PER_TILE),
            stride=(1, HEAD_DIM_I32, HEAD_DIM_I32 * PAGE_SIZE),
        ))

        sm = cutlass.Int64(smid_u32())
        probe_cnt = cutlass.Int32(0)

        if tidx == 0:
            range_start(probe, bidx, probe_cnt, sm, TAGS["total"])
            probe_cnt = cutlass.Int32(1)
            range_start(probe, bidx, probe_cnt, sm, TAGS["load_A"])
            probe_cnt = cutlass.Int32(2)
            range_start(probe, bidx, probe_cnt, sm, TAGS["gmem_to_lin"])

        # ── Phase 1: cp.async GMEM → linear SMEM ──────────────────────
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32, num_bits_per_copy=cutlass.Int32.width,
        )
        thr_layout_g = cute.make_layout((32, 16, 1), stride=(1, 32, 0))
        gA_thr  = cute.local_partition(gA,    thr_layout_g, tidx)
        sLin_thr = cute.local_partition(sLinT, thr_layout_g, tidx)
        cute.copy(atom, gA_thr, sLin_thr)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if tidx == 0:
            probe_cnt = range_stop(probe, bidx, probe_cnt)
            probe_cnt = cutlass.Int32(3)
            range_start(probe, bidx, probe_cnt, sm, TAGS["lin_to_swiz"])

        # ── Phase 2: autovec linear SMEM → swizzled SMEM via partition_A ──
        # Build a dummy GMEM-style "tensor" over sLin recast to fp8 with the
        # same logical (BM, HEAD_DIM) layout as gA expected by partition_A.
        sLin_fp8 = cute.recast_tensor(
            cute.make_tensor(sLin, cute.make_layout(
                (BM * HEAD_DIM_I32,), stride=(1,),
            )),
            dtype=self.fp8_dtype,
        )
        # Re-view as ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM) row-major.
        sLin_view = cute.make_tensor(
            sLin_fp8.iterator,
            cute.make_layout(
                ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM),
                stride=((HEAD_DIM, HEAD_DIM * PAGE_SIZE), 1),
            ),
        )
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCsLin  = thr_mma.partition_A(sLin_view)

        thr_layout_s = cute.make_layout(NUM_THREADS)
        sLin_part = cute.local_partition(tCsLin, thr_layout_s, tidx)
        sA_part   = cute.local_partition(sA[None, None, None, 0], thr_layout_s, tidx)
        cute.autovec_copy(sLin_part, sA_part)

        cute.arch.sync_threads()

        if tidx == 0:
            probe_cnt = range_stop(probe, bidx, probe_cnt)
            probe_cnt = range_stop(probe, bidx, cutlass.Int32(1))   # close load_A
            sA_u8 = cute.recast_tensor(sA_part, dtype=cutlass.Uint8)
            sink[bidx] = cutlass.Int32(sA_u8[0])
            off = PROBE_HEADER + 0 * PROBE_ENTRY
            probe[bidx, off + 3] = globaltimer_u64() - probe[bidx, off + 2]
            range_finalize(probe, bidx, cutlass.Int32(4))


def _fake(dtype, shape, stride_order, align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape,
                                    stride_order=stride_order, assumed_align=align)

def compile_kernel():
    NUM_PG = cute.sym_int(divisibility=2)
    NUM_PP = cute.sym_int()
    NUM_BL = cute.sym_int()

    kv_pool     = _fake(cute.Uint8,  (NUM_PP, PAGE_SIZE, ROW_STRIDE), (2, 1, 0), 16)
    block_table = _fake(cute.Int32,  (NUM_PG,),                       (0,),      4)
    sink        = _fake(cute.Int32,  (NUM_BL,),                       (0,),      4)
    probe       = _fake(cute.Int64,  (NUM_BL, PROBE_COLS),            (1, 0),    8)
    stream      = make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = LoadAV3()
    compiled = cute.compile(
        kernel, kv_pool, block_table, sink, probe, stream,
        options="--enable-tvm-ffi",
    )
    return kernel, compiled


def run_single(workload_idx: int) -> str:
    from src.kernels.load_a_common import run_single as _rs
    import sys
    return _rs(sys.modules[__name__], workload_idx)

"""load-A v1: autovec_copy into UMMA-swizzled SMEM, 512 threads.

Plain "vectorized" baseline. Uses the MMA's tCgA partition so destination
sA is in the exact swizzled layout UMMA needs. Loads via cute.autovec_copy.
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
TAGS = {"total": 0, "load_A": 2}
TAG_NAMES   = {v: k for k, v in TAGS.items()}
PHASE_ORDER = ["total", "load_A"]


class LoadAV1:
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
        sA = smem.allocate_tensor(
            self.fp8_dtype, a_smem_layout.outer, 128, a_smem_layout.inner)

        page0_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 0])
        page1_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 1])
        page_stride_b = PAGE_SIZE * ROW_STRIDE
        page0_off_b = page0_id * page_stride_b
        jump_b      = (page1_id - page0_id) * page_stride_b

        fp8_base = cute.recast_ptr(kv_pool.iterator, dtype=self.fp8_dtype) + page0_off_b
        gA_layout = cute.make_layout(((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM),
                                     stride=((ROW_STRIDE, jump_b), 1))
        gA = cute.make_tensor(fp8_base, gA_layout)

        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA    = thr_mma.partition_A(gA)

        sm = cutlass.Int64(smid_u32())
        probe_cnt = cutlass.Int32(0)

        if tidx == 0:
            range_start(probe, bidx, probe_cnt, sm, TAGS["total"])
            probe_cnt = cutlass.Int32(1)
            range_start(probe, bidx, probe_cnt, sm, TAGS["load_A"])

        # ── autovec_copy across 512 threads into UMMA-swizzled sA ─────
        thr_layout = cute.make_layout(self.threads_per_cta)
        sA_thr = cute.local_partition(sA[None, None, None, 0], thr_layout, tidx)
        gA_thr = cute.local_partition(tCgA, thr_layout, tidx)
        cute.autovec_copy(gA_thr, sA_thr)

        cute.arch.sync_threads()

        if tidx == 0:
            probe_cnt = range_stop(probe, bidx, probe_cnt)
            # Recast sA to Uint8 to read back without fp8 conversion.
            sA_u8 = cute.recast_tensor(sA_thr, dtype=cutlass.Uint8)
            sink[bidx] = cutlass.Int32(sA_u8[0])
            off = PROBE_HEADER + 0 * PROBE_ENTRY
            probe[bidx, off + 3] = globaltimer_u64() - probe[bidx, off + 2]
            range_finalize(probe, bidx, probe_cnt)


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

    kernel = LoadAV1()
    compiled = cute.compile(
        kernel, kv_pool, block_table, sink, probe, stream,
        options="--enable-tvm-ffi",
    )
    return kernel, compiled


def run_single(workload_idx: int) -> str:
    from src.kernels.load_a_common import run_single as _rs
    import sys
    return _rs(sys.modules[__name__], workload_idx)

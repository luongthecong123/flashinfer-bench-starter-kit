"""load-A v2: cp.async fire-and-forget, 512 threads, LINEAR SMEM dest.

Quack-blog pattern: build a cp.async copy atom (non-bulk) + a TV layout, then
`cute.copy(atom, partition_S(gA), partition_D(sA))`.

Source/dest both Int32-typed for 4-B alignment (kv_pool row stride = 132 B = 33
Int32). 32-bit cp.async (rows aren't 16-aligned in gmem). Destination is a flat
(BM, BK) Int32 buffer — NOT the UMMA swizzled layout (handled by v2_swizzled).

Each thread issues 8 cp.async.ALWAYS instructions (4096 Int32 / 512 threads).
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.nvgpu import cpasync

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

ROW_STRIDE_I32 = ROW_STRIDE // 4   # 33
HEAD_DIM_I32   = HEAD_DIM   // 4   # 32
N_PER_THREAD   = (BM * HEAD_DIM_I32) // NUM_THREADS  # 4096/512 = 8


class LoadAV2:
    def __init__(self):
        self.threads_per_cta = NUM_THREADS

    @cute.jit
    def __call__(self, kv_pool, block_table, sink, probe, stream):
        num_pg = cute.size(block_table, mode=[0])
        grid_m = num_pg // PAGES_PER_TILE
        self.kernel(kv_pool, block_table, sink, probe).launch(
            grid=(grid_m, 1, 1), block=(self.threads_per_cta, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, kv_pool, block_table, sink, probe):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sA_i32 = smem.allocate_array(cutlass.Int32, num_elems=BM * HEAD_DIM_I32)

        page0_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 0])
        page1_id = cutlass.Int32(block_table[bidx * PAGES_PER_TILE + 1])
        page_stride_b   = PAGE_SIZE * ROW_STRIDE
        page_stride_i32 = page_stride_b // 4
        page0_off_i32   = page0_id * page_stride_i32
        jump_i32        = (page1_id - page0_id) * page_stride_i32

        i32_base = cute.make_ptr(
            cutlass.Int32,
            (cute.recast_ptr(kv_pool.iterator, dtype=cutlass.Int32) + page0_off_i32).toint(),
            mem_space=cute.AddressSpace.gmem, assumed_align=4,
        )
        # gA / sA as clean 2D (M, K) tensors, K-major (K stride = 1).
        # M = (PAGE_SIZE, PAGES_PER_TILE) = 128 rows; K = HEAD_DIM_I32 = 32 Int32 per row.
        gA = cute.make_tensor(i32_base, cute.make_layout(
            ((PAGE_SIZE, PAGES_PER_TILE), HEAD_DIM_I32),
            stride=((ROW_STRIDE_I32, jump_i32), 1),
        ))
        sA = cute.make_tensor(sA_i32, cute.make_layout(
            (BM, HEAD_DIM_I32),
            stride=(HEAD_DIM_I32, 1),
        ))

        sm = cutlass.Int64(smid_u32())
        probe_cnt = cutlass.Int32(0)

        if tidx == 0:
            range_start(probe, bidx, probe_cnt, sm, TAGS["total"])
            probe_cnt = cutlass.Int32(1)
            range_start(probe, bidx, probe_cnt, sm, TAGS["load_A"])

        # ── cp.async copy atom: 32-bit per instruction (4-B aligned rows) ─────
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32,
            num_bits_per_copy=cutlass.Int32.width,
        )

        # ── TV layout (quack-blog pattern) ────────────────────────────────────
        # Tile = (BM=128, BK=32) Int32. 512 threads × 8 vals = 4096 elements.
        # thr_layout (M=16, K=32) stride (K, 1): each warp (32 threads) covers
        #   one full K-row of 32 Int32 = 128 B → coalesced GMEM read.
        # val_layout (M=8, K=1) stride (1, 1): each thread owns 8 rows in M
        #   (strided by 16 since 16 thread-rows tile M).
        thr_layout = cute.make_layout((16, HEAD_DIM_I32), stride=(HEAD_DIM_I32, 1))
        val_layout = cute.make_layout((N_PER_THREAD, 1),  stride=(1, 1))
        tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)

        thr_copy = tiled_copy.get_slice(tidx)
        tAgA = thr_copy.partition_S(gA)
        tAsA = thr_copy.partition_D(sA)
        cute.copy(atom, tAgA, tAsA)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if tidx == 0:
            probe_cnt = range_stop(probe, bidx, probe_cnt)
            sink[bidx] = sA[0, 0]
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

    kernel = LoadAV2()
    compiled = cute.compile(
        kernel, kv_pool, block_table, sink, probe, stream,
        options="--enable-tvm-ffi",
    )
    return kernel, compiled


def run_single(workload_idx: int) -> str:
    from src.kernels.load_a_common import run_single as _rs
    import sys
    return _rs(sys.modules[__name__], workload_idx)

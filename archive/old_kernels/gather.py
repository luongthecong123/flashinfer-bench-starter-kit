import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cute.testing import benchmark, JitArguments

from typing import Tuple
import math
import torch


def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


class Gather():
    """Gather kernel: scatter-read from flat paged KV cache into dense [T, topk, D] buffers.

    Grid: [T, topk // BM, 1]   — one CTA per (token, tile-of-topk)
    Each CTA handles BM rows: threads cooperatively copy D elements per row.
    """
    def __init__(self):
        self.BM = 2
        self.num_threads = 256
        self.warp_size = cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        ckv_cache: cute.Tensor,         # [N, 512]   flat page pool
        kpe_cache: cute.Tensor,         # [N, 64]    flat page pool
        sparse_indices: cute.Tensor,    # [T, 2048]
        kc: cute.Tensor,                # [T, 2048, 512]
        Kp: cute.Tensor,                # [T, 2048, 64]
        max_valid: cute.Tensor,         # [T] int32  — output: count of valid (!=-1) per token
        stream,                         # CUDA stream
    ):
        T, topk = sparse_indices.shape
        self.kernel(ckv_cache, kpe_cache, sparse_indices, kc, Kp, max_valid).launch(
            grid=[T, topk // self.BM, 1], block=[self.num_threads, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        ckv_cache: cute.Tensor,         # [N, 512]
        kpe_cache: cute.Tensor,         # [N, 64]
        sparse_indices: cute.Tensor,    # [T, 2048]
        kc: cute.Tensor,                # [T, 2048, 512]
        Kp: cute.Tensor,                # [T, 2048, 64]
        max_valid: cute.Tensor,         # [T] int32
    ):
        N, dkc = ckv_cache.shape
        N2, dkp = kpe_cache.shape
        T, topk = sparse_indices.shape

        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        # Each CTA handles BM consecutive rows in the topk dimension
        # bidx = token index,  bidy = tile index (topk // BM tiles)
        for m in range(self.BM):
            row = bidy * self.BM + m
            idx = sparse_indices[bidx, row]

            if idx >= cutlass.Int32(0):
                # Valid index: gather from cache
                for d in range(tidx, dkc, self.num_threads):
                    kc[bidx, row, d] = ckv_cache[idx, d]
                for d in range(tidx, dkp, self.num_threads):
                    Kp[bidx, row, d] = kpe_cache[idx, d]
            else:
                # Invalid (-1) sentinel: write zeros
                for d in range(tidx, dkc, self.num_threads):
                    kc[bidx, row, d] = cutlass.BFloat16(0)
                for d in range(tidx, dkp, self.num_threads):
                    Kp[bidx, row, d] = cutlass.BFloat16(0)

        # ── Option 1: all-warp reduction via smem ─────────────────────────────
        # 256 threads = 8 warps. Each thread counts (2048/256)=8 elements.
        # Each warp reduces via shuffle → partial sums → smem[32].
        # Warp 0 reads smem, reduces with masked warp_reduce(width=num_warps).
        if bidy == 0:
            num_warps = self.num_threads // self.warp_size

            allocator = cutlass.utils.SmemAllocator()
            smem_counts = allocator.allocate_tensor(
                cutlass.Int32, cute.make_layout((self.warp_size,), stride=(1,)), 4, None
            )

            # Each thread counts its chunk: 2048 / 256 = 8 elements
            local_count = cutlass.Int32(0)
            for i in range(topk // self.num_threads):
                if sparse_indices[bidx, tidx * (topk // self.num_threads) + i] >= cutlass.Int32(0):
                    local_count += cutlass.Int32(1)

            # Intra-warp reduction
            warp_sum = warp_reduce(local_count, lambda a, b: a + b)

            # Lane 0 of each warp writes to smem
            if lane_idx == 0:
                smem_counts[warp_idx] = warp_sum

            cute.arch.sync_threads()

            # Warp 0 reads smem and does final masked reduction
            if warp_idx == 0:
                partial = smem_counts[lane_idx]
                total = warp_reduce(partial, lambda a, b: a + b, width=self.num_threads // self.warp_size)
                if lane_idx == 0:
                    max_valid[bidx] = total

        # # ── Option 2: single-warp reduction ──────────────────────────────────
        # # bidy==0 CTA: first warp counts valid entries via warp shuffle reduction
        # if bidy == 0:
        #     warp_idx = tidx // 32
        #     lane_idx = cute.arch.lane_idx()
        #     if warp_idx == 0:
        #         # Each lane counts its chunk: 2048 / 32 = 64 elements per lane
        #         local_count = cutlass.Int32(0)
        #         for i in range(topk // 32):
        #             if sparse_indices[bidx, lane_idx * (topk // 32) + i] >= cutlass.Int32(0):
        #                 local_count += cutlass.Int32(1)
        #         total = warp_reduce(local_count, lambda a, b: a + b)
        #         if lane_idx == 0:
        #             max_valid[bidx] = total


# ── Compilation ────────────────────────────────────────────────────────────────

def fake_wrapper(dtype, shape, stride_order, assumed_align):
    return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=assumed_align)


def compile_gather():
    T = cute.sym_int()
    # N = cute.sym_int()
    N = 8462 * 64
    head_dim_ckv, head_dim_kpe, top_k_len = 512, 64, 2048

    ckv_cache = fake_wrapper(cute.BFloat16, (N, head_dim_ckv), (1, 0), 16)
    kpe_cache = fake_wrapper(cute.BFloat16, (N, head_dim_kpe), (1, 0), 16)
    sparse_indices = fake_wrapper(cute.Int32, (T, top_k_len), (1, 0), 4)
    kc = fake_wrapper(cute.BFloat16, (T, top_k_len, head_dim_ckv), (2, 1, 0), 16)
    Kp = fake_wrapper(cute.BFloat16, (T, top_k_len, head_dim_kpe), (2, 1, 0), 16)
    max_valid = fake_wrapper(cute.Int32, (T,), (0,), 4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    gather = Gather()
    return cute.compile(
        gather,
        ckv_cache, kpe_cache, sparse_indices, kc, Kp, max_valid, stream,
        options="--enable-tvm-ffi"
    )


gather_compiled = compile_gather()

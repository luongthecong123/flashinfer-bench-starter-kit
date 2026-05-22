"""
Exact top-K — CUDA C++ port of PyTorch's sbtopk (single-block-per-slice) path.

Loaded via torch.utils.cpp_extension.load_inline (no build step needed).

Algorithm: mirrors SortingRadixSelect.cuh + gatherTopK from TensorTopK.cu
  RADIX_BITS = 2  ->  16 passes for float32
  Phase 1 (find k-th value):
    Each pass: threads scan their stripe -> per-thread bin counts (4 bins)
    Warp-reduce with __shfl_xor_sync -> lane-0 atomically accumulates into smem[4]
    __syncthreads -> all threads read globals -> scan bins descending to narrow desired
  Phase 2 (gather):
    Scan again: strict > threshold -> exclusive warp prefix scan -> write
    Second loop: exact-tie elements fill the remainder up to exactly TOPK
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.utils.cpp_extension import load_inline
from src.idx_utils import check_topk_indices

TOPK        = 2048
NUM_THREADS = 1024

CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

__device__ __forceinline__ uint32_t float_to_radix(float v) {
    uint32_t x;
    memcpy(&x, &v, sizeof(x));
    if (v != v) return 0xFFFFFFFFu;
    uint32_t mask = (x & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return x ^ mask;
}

__device__ __forceinline__ void warp_reduce_bins(
    uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3)
{
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        c0 += __shfl_xor_sync(0xFFFFFFFFu, c0, offset);
        c1 += __shfl_xor_sync(0xFFFFFFFFu, c1, offset);
        c2 += __shfl_xor_sync(0xFFFFFFFFu, c2, offset);
        c3 += __shfl_xor_sync(0xFFFFFFFFu, c3, offset);
    }
}

__global__ void __launch_bounds__(1024)
topk_radix_sbtopk(
    const float* __restrict__ scores,
    int32_t*     __restrict__ out_idx,
    const int*   __restrict__ seq_lens,
    int max_sl,
    int topk)
{
    __shared__ uint32_t smem[4 + 32];

    const int b         = blockIdx.x;
    const int tid       = threadIdx.x;
    const int lane      = tid & 31;
    const int warp      = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    const int sl    = seq_lens[b];
    const float* row = scores + (long long)b * max_sl;
    int32_t*     out = out_idx + (long long)b * topk;

    // ---- Phase 1: 16-pass radix select to find k-th value ----
    uint32_t desired      = 0u;
    uint32_t desired_mask = 0u;
    uint32_t k_to_find    = (uint32_t)topk;

    #pragma unroll 1
    for (int pass = 0; pass < 16; ++pass) {
        const int digit_pos = 30 - pass * 2;

        if (tid < 4) smem[tid] = 0u;
        __syncthreads();

        uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (int i = tid; i < sl; i += blockDim.x) {
            uint32_t bits = float_to_radix(row[i]);
            if ((bits & desired_mask) != (desired & desired_mask)) continue;
            uint32_t digit = (bits >> digit_pos) & 3u;
            c0 += (digit == 0u);
            c1 += (digit == 1u);
            c2 += (digit == 2u);
            c3 += (digit == 3u);
        }

        warp_reduce_bins(c0, c1, c2, c3);

        if (lane == 0) {
            atomicAdd(&smem[0], c0);
            atomicAdd(&smem[1], c1);
            atomicAdd(&smem[2], c2);
            atomicAdd(&smem[3], c3);
        }
        __syncthreads();

        uint32_t g0 = smem[0], g1 = smem[1], g2 = smem[2], g3 = smem[3];
        __syncthreads();

        // scan from largest digit down
        if (g3 >= k_to_find) {
            desired = (desired & ~(3u << digit_pos)) | (3u << digit_pos);
            desired_mask |= (3u << digit_pos);
        } else {
            k_to_find -= g3;
            if (g2 >= k_to_find) {
                desired = (desired & ~(3u << digit_pos)) | (2u << digit_pos);
                desired_mask |= (3u << digit_pos);
            } else {
                k_to_find -= g2;
                if (g1 >= k_to_find) {
                    desired = (desired & ~(3u << digit_pos)) | (1u << digit_pos);
                    desired_mask |= (3u << digit_pos);
                } else {
                    k_to_find -= g1;
                    desired = (desired & ~(3u << digit_pos));
                    desired_mask |= (3u << digit_pos);
                }
            }
        }
    }

    const uint32_t kth_bits = desired;

    // ---- Phase 2a: gather strictly-better elements ----
    __shared__ uint32_t warp_totals[32];
    __shared__ uint32_t write_cursor;
    if (tid == 0) write_cursor = 0u;
    __syncthreads();

    for (int base = 0; base < sl; base += blockDim.x) {
        int i = base + tid;
        uint32_t bits  = (i < sl) ? float_to_radix(row[i]) : 0u;
        uint32_t is_b  = (i < sl && bits > kth_bits) ? 1u : 0u;

        // Kogge-Stone inclusive prefix scan on is_b within warp
        uint32_t val = is_b;
        #pragma unroll
        for (int s = 1; s < 32; s <<= 1) {
            uint32_t peer = __shfl_up_sync(0xFFFFFFFFu, val, s);  // fetch accumulated val
            if (lane >= s) val += peer;
        }
        // val is now inclusive prefix sum; convert to exclusive
        uint32_t my_excl  = val - is_b;
        uint32_t warp_tot = __shfl_sync(0xFFFFFFFFu, val, 31);  // total of warp = inclusive at lane31

        if (lane == 31) warp_totals[warp] = warp_tot;
        __syncthreads();

        // warp 0: Kogge-Stone exclusive prefix of warp totals
        if (warp == 0) {
            uint32_t orig = (lane < num_warps) ? warp_totals[lane] : 0u;
            uint32_t v = orig;
            #pragma unroll
            for (int s = 1; s < 32; s <<= 1) {
                uint32_t peer = __shfl_up_sync(0xFFFFFFFFu, v, s);
                if (lane >= s) v += peer;
            }
            if (lane < num_warps) warp_totals[lane] = v - orig;  // exclusive prefix
        }
        __syncthreads();

        uint32_t woff = warp_totals[warp];
        uint32_t goff = write_cursor + woff + my_excl;
        if (is_b && goff < (uint32_t)topk) out[goff] = i;

        // count this round and advance cursor
        __syncthreads();
        if (tid == 0) smem[0] = 0u;
        __syncthreads();
        if (is_b) atomicAdd(&smem[0], 1u);
        __syncthreads();
        if (tid == 0) write_cursor += smem[0];
        __syncthreads();
    }

    // ---- Phase 2b: fill tie elements up to topk ----
    __shared__ uint32_t tie_cursor;
    if (tid == 0) tie_cursor = 0u;
    __syncthreads();

    for (int base = 0; base < sl; base += blockDim.x) {
        if (write_cursor + tie_cursor >= (uint32_t)topk) break;

        int i = base + tid;
        uint32_t bits  = (i < sl) ? float_to_radix(row[i]) : 0u;
        uint32_t is_t  = (i < sl && bits == kth_bits) ? 1u : 0u;

        // Kogge-Stone inclusive prefix scan on is_t within warp
        uint32_t val_t = is_t;
        #pragma unroll
        for (int s = 1; s < 32; s <<= 1) {
            uint32_t peer = __shfl_up_sync(0xFFFFFFFFu, val_t, s);
            if (lane >= s) val_t += peer;
        }
        uint32_t my_excl  = val_t - is_t;
        uint32_t warp_tot = __shfl_sync(0xFFFFFFFFu, val_t, 31);

        if (lane == 31) warp_totals[warp] = warp_tot;
        __syncthreads();

        if (warp == 0) {
            uint32_t orig = (lane < num_warps) ? warp_totals[lane] : 0u;
            uint32_t v = orig;
            #pragma unroll
            for (int s = 1; s < 32; s <<= 1) {
                uint32_t peer = __shfl_up_sync(0xFFFFFFFFu, v, s);
                if (lane >= s) v += peer;
            }
            if (lane < num_warps) warp_totals[lane] = v - orig;  // exclusive prefix
        }
        __syncthreads();

        uint32_t woff    = warp_totals[warp];
        uint32_t toff    = tie_cursor + woff + my_excl;
        uint32_t wrt_pos = write_cursor + toff;
        uint32_t need    = (uint32_t)topk - write_cursor;

        if (is_t && toff < need && wrt_pos < (uint32_t)topk) out[wrt_pos] = i;

        __syncthreads();
        if (tid == 0) smem[0] = 0u;
        __syncthreads();
        if (is_t) atomicAdd(&smem[0], 1u);
        __syncthreads();
        if (tid == 0) tie_cursor += smem[0];
        __syncthreads();
    }
}

void topk_radix_launch(
    torch::Tensor scores,
    torch::Tensor out_idx,
    torch::Tensor seq_lens)
{
    const int B      = scores.size(0);
    const int max_sl = scores.size(1);
    const int topk   = out_idx.size(1);

    topk_radix_sbtopk<<<B, 1024>>>(
        scores.data_ptr<float>(),
        out_idx.data_ptr<int32_t>(),
        seq_lens.data_ptr<int>(),
        max_sl,
        topk
    );
}
"""

CPP_SOURCE = r"""
void topk_radix_launch(
    torch::Tensor scores,
    torch::Tensor out_idx,
    torch::Tensor seq_lens);
"""

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="topk_radix_sbtopk",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["topk_radix_launch"],
            verbose=True,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _ext


def topk_radix(scores: torch.Tensor, seq_lens: torch.Tensor, topk: int = TOPK) -> torch.Tensor:
    """
    Exact top-K indices via sbtopk radix select.

    Args:
        scores   : [B, max_sl] float32, positions beyond seq_lens[b] padded with -inf
        seq_lens : [B]         int32
        topk     : number of top elements to return (default TOPK=2048)

    Returns:
        out_idx  : [B, topk]   int32
    """
    B = scores.size(0)
    out_idx = torch.full((B, topk), -1, dtype=torch.int32, device=scores.device)
    _get_ext().topk_radix_launch(scores, out_idx, seq_lens)
    return out_idx


def test_correctness():
    torch.manual_seed(0)
    device = torch.device("cuda")

    print("=== Correctness test ===")
    cases = [
        (1, 4096),
        (4, 6000),
        (8, 3000),
        (1, 2049),
        (2, 5806),   # max seq_len from contest
    ]
    for B, max_sl in cases:
        scores = torch.full((B, max_sl), float("-inf"), dtype=torch.float32, device=device)
        sl_list = [max_sl - i * (max_sl // B // 2) for i in range(B)]
        sl_list = [max(TOPK + 1, s) for s in sl_list]
        seq_lens = torch.tensor(sl_list, dtype=torch.int32, device=device)

        for b, sl in enumerate(sl_list):
            scores[b, :sl] = torch.randn(sl, device=device)

        ref_idx = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
        for b, sl in enumerate(sl_list):
            k = min(TOPK, sl)
            _, idx = torch.topk(scores[b, :sl], k)
            ref_idx[b, :k] = idx.int()

        out = topk_radix(scores, seq_lens)
        torch.cuda.synchronize()

        ok, miss = check_topk_indices(ref_idx, out, seq_lens)
        status = "PASS" if ok else "FAIL"
        print(f"  B={B:2d}  max_sl={max_sl:5d}  seq_lens={sl_list}  worst_miss={miss:.6f}  [{status}]")


def benchmark_vs_torch():
    import time
    torch.manual_seed(1)
    device = torch.device("cuda")

    B, max_sl = 8, 6000
    scores = torch.randn(B, max_sl, device=device, dtype=torch.float32)
    seq_lens = torch.full((B,), max_sl, dtype=torch.int32, device=device)

    for _ in range(20):
        topk_radix(scores, seq_lens)
        torch.topk(scores, TOPK, dim=1)
    torch.cuda.synchronize()

    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        topk_radix(scores, seq_lens)
    torch.cuda.synchronize()
    our_ms = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        torch.topk(scores, TOPK, dim=1)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) / N * 1000

    print(f"\n=== Benchmark (B={B}, max_sl={max_sl}) ===")
    print(f"  topk_radix  : {our_ms*1000:.1f} us")
    print(f"  torch.topk  : {ref_ms*1000:.1f} us")
    print(f"  speedup     : {ref_ms/our_ms:.2f}x")


if __name__ == "__main__":
    test_correctness()
    benchmark_vs_torch()

# Indexer TC Kernel — Profiling Results (B200)

---

## Profile 1 — NVTX (idxer_tc v1, no torch.compile on dequant/remap)

### Setup

- **Kernel**: `src/kernels/idxer_tc_nvtx.py` (NVTX-instrumented `idxer_tc.py`)
- **Profiler**: `src/modal/nsys_idx.py` → `src/profiler_idx.py`
- **GPU**: NVIDIA B200
- **Workload 0**: uuid=30cecff1, B=1, num_pages=11923, max_pages=1, max_sl=64
- **Warmup**: 3 iterations (is_profiling=False), then 1 profiled run (is_profiling=True)

### NVTX Phase Breakdown

| Phase              | Time (ns) |  Time (µs) | % of Total |
|--------------------|----------:|----------:|-----------:|
| **Total**          | 1,910,304 |  1,910.3  |    100.0%  |
| score_and_reduce   |   506,612 |    506.6  |     26.5%  |
| remap_indices      |   332,891 |    332.9  |     17.4%  |
| dequant_kv_cache   |   261,121 |    261.1  |     13.7%  |
| dequant_q          |   248,935 |    248.9  |     13.0%  |
| build_indices      |   246,706 |    246.7  |     12.9%  |
| topk               |    78,440 |     78.4  |      4.1%  |
| gather             |    50,497 |     50.5  |      2.6%  |

**Unaccounted overhead** (host-side dispatch, NVTX push/pop, etc.): ~185 µs (9.7%)

**Root cause of uniform latency across workloads**: `dequant_kv_cache` always processes all 11923 pages (~97 MB) regardless of actual usage. The variable work (scoring B×seq_len tokens) is negligible by comparison.

### Key Observations

1. **score_and_reduce dominates** at 26.5% — fused `torch.compile`d BMM + ReLU + einsum + masked_fill.
2. **remap_indices** at 17.4% — ~9 separate eager kernel launches doing trivial integer arithmetic (pure dispatch overhead on tiny tensors).
3. **dequant_q + dequant_kv_cache** at 26.7% — FP8 → FP32 conversion overhead; dequant processes full 11923-page pool unconditionally.
4. **build_indices** at 12.9%; **topk** at 4.1%; **gather** at 2.6%.

### CUDA Kernel Summary

| Kernel | Time (ns) |
|--------|-------:|
| triton_red_fused_bmm_1 (score GEMM) | 150,496 |
| radix_topk::... (topk) | 52,736 |
| void at::native::vectorized_gather_kernel (gather) | 41,920 |
| triton_poi_fused_bmm_relu_0 | 1,345 |
| triton_poi_fused_bmm_masked_fill_permute_view_1 | 960 |

---

## Profile 2 — CUDA Events (idxer_tc v2, all phases torch.compiled)

### Setup

- **Kernel**: `src/kernels/idxer_tc.py` — `dequant_fp8_kv_cache`, `_score_and_reduce`, `_topk_remap_and_write` all `@torch.compile`d
- **Profiler**: `src/modal/profile_idxer_tc.py` (CUDA event timing)
- **GPU**: NVIDIA B200
- **Workload 127 (last)**: uuid=dba1e960, B=25, num_pages=11923, max_pages=30, max_sl=1920
- **Warmup**: 10 iterations, **Reps**: 50 (averaged)

### CUDA Event Phase Breakdown

| Phase              |   µs   |   ms   | % of Total |
|--------------------|-------:|-------:|-----------:|
| **TOTAL**          | 300.8  | 0.301  |   100.0%   |
| dequant_kv_cache   |  82.4  | 0.082  |    27.4%   |
| topk_remap_write   |  63.9  | 0.064  |    21.2%   |
| score_and_reduce   |  61.3  | 0.061  |    20.4%   |
| gather             |  28.8  | 0.029  |     9.6%   |
| build_indices      |  26.3  | 0.026  |     8.7%   |
| build_mask         |  16.5  | 0.017  |     5.5%   |
| dequant_q          |   7.8  | 0.008  |     2.6%   |

### Comparison vs Profile 1

| Phase              | v1 (µs) | v2 (µs) | Speedup |
|--------------------|--------:|--------:|--------:|
| remap/topk_remap   |  332.9  |   63.9  | **5.2x** |
| dequant_kv_cache   |  261.1  |   82.4  | **3.2x** |
| dequant_q          |  248.9  |    7.8  | **31.9x** |
| score_and_reduce   |  506.6  |   61.3  | **8.3x** |
| gather             |   50.5  |   28.8  | **1.8x** |
| **TOTAL**          | ~1910   |  300.8  | **~6.4x** |

> Note: Profile 1 was WL0 (B=1, max_sl=64); Profile 2 is WL127 (B=25, max_sl=1920).
> The v2 total reflects both the torch.compile improvements and larger workload (more actual work).

### Key Observations

1. **dequant_kv_cache still dominates** at 27.4% — processes all 11923 pages every call. Primary remaining bottleneck.
2. **topk_remap_write** fused down to 63.9 µs (was 333 µs for remap alone) — `@torch.compile` eliminated the 9-launch overhead.
3. **score_and_reduce** at 20.4% — efficient at this size; will grow with batch.
4. **Remaining opportunity**: dequant only the pages actually used by the batch (`block_table.unique()` → slice → dequant).

## Files

- [src/kernels/idxer_tc_nvtx.py](src/kernels/idxer_tc_nvtx.py) — NVTX-instrumented v1
- [src/modal/profile_idxer_tc.py](src/modal/profile_idxer_tc.py) — CUDA event profiler (v2)
- [src/profiler_idx.py](src/profiler_idx.py) — profiler harness
- [src/modal/nsys_idx.py](src/modal/nsys_idx.py) — Modal nsys launcher


```python
# k_cache_fp8: [num_pages, PAGE_SIZE, 1, 132] int8 (bytes)
# Per token: [fp8_0..fp8_127 | scale_f32 (4 bytes)]

N_tokens = NUM_PAGES * PAGE_SIZE  # total tokens = num_pages * 64

# Base uint8 iterator over the flat byte stream
uint8_ptr = cute.recast_ptr(k_cache_fp8.iterator, dtype=cutlass.UInt8)

# ── FP8 data view: [N_tokens, 128] float8_e4m3fn ─────────────────────────────
# uint8 → float8_e4m3fn is 1-to-1 (both 1 byte), so stride=(132,1) is unchanged
fp8_view = cute.make_tensor(
    cute.recast_ptr(uint8_ptr, dtype=cutlass.Float8E4M3FN),
    cute.make_layout((N_tokens, HEAD_DIM), stride=(132, 1))
)

# ── Scale view: [N_tokens] float32 ───────────────────────────────────────────
# Offset base ptr by HEAD_DIM bytes to land on scale bytes within each row,
# then recast uint8 → float32 (4 bytes each) → stride becomes 132/4 = 33
scale_view = cute.make_tensor(
    cute.recast_ptr(uint8_ptr + HEAD_DIM, dtype=cutlass.Float32),
    cute.make_layout((N_tokens,), stride=(33,))
)
```
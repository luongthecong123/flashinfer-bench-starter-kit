# Batched Padded Attention — Removing the Per-Token Loop

## The Problem

The reference iterates `for t in range(T)`:
- Each iteration gathers variable-length KV (1–2048 valid tokens), runs 3 matmuls
- T separate `torch.compile` kernel launches → huge launch overhead for T=8
- Each matmul has a different reduction length (some 6 tokens, some 2048)

## The Idea

Pad every token's selected KV to the full 2048, mask invalid positions, and run
ONE batched matmul call for all T tokens simultaneously.

---

## Shapes Step-by-Step

### Inputs (given by the benchmark)

```
q_nope        [T, 16, 512]      bf16    contiguous, row-major
q_pe          [T, 16,  64]      bf16    contiguous, row-major
ckv_cache     [P, 64, 512]      bf16    contiguous, row-major   P=8462 pages
kpe_cache     [P, 64,  64]      bf16    contiguous, row-major
sparse_indices[T, 2048]         int32   contiguous, row-major   -1 = padding
sm_scale      scalar            f32
output        [T, 16, 512]      bf16    pre-allocated (DPS)
lse           [T, 16]           f32     pre-allocated (DPS)
```

T = 1–8 (batch of decode queries), P = 8462 (shared page pool, 541,568 token slots).

### Step 1: Flatten Paged Cache

```python
Kc_all = ckv_cache.reshape(-1, 512)     # [541568, 512]  bf16  contiguous
Kp_all = kpe_cache.reshape(-1, 64)      # [541568,  64]  bf16  contiguous
```

Just a view — no copy. Pages are already laid out as `[page0_tok0, page0_tok1, ..., page0_tok63, page1_tok0, ...]` in memory. The reshape merges the first two dims into one flat token index.

### Step 2: Build Mask & Safe Indices

```python
mask         = (sparse_indices == -1)               # [T, 2048]  bool   contiguous
safe_indices = sparse_indices.clamp(min=0).long()    # [T, 2048]  int64  contiguous
```

`mask[t, v] = True` means position `v` is padding (original index was -1).
`safe_indices` replaces -1 with 0 so the gather doesn't segfault — the gathered values at masked positions are garbage but don't matter (masked to -inf before softmax).

### Step 3: Batched Gather (the key reshape trick)

```python
# Flatten indices for a single index_select call:
flat_idx = safe_indices.reshape(-1)                 # [T*2048]  int64

Kc = Kc_all[flat_idx].reshape(T, 2048, 512)        # [T, 2048, 512]  bf16  contiguous
Kp = Kp_all[flat_idx].reshape(T, 2048,  64)        # [T, 2048,  64]  bf16  contiguous
```

**Memory access pattern:**

```
Kc_all:  [═══════════════════════════════════════════]  541,568 × 512  bf16
              ↑  ↑↑↑ ↑        ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑          ↑↑↑
              scattered reads from sparse_indices
                    ↓
Kc:      [████████████████████]  T × 2048 × 512  bf16   ← NEW contiguous buffer
              dense output
```

Each gather reads 2048 × 512 × 2B = **2 MB per token** from scattered locations in the 529 MiB pool. The output `Kc` is a freshly-allocated contiguous tensor.

**Why not truly random:** ~96.7% of selected tokens share a page. Touched pages have ~60/64 slots selected. So the access is more like reading ~32 nearly-full 4 KB pages than 2048 random cacheline fetches.

### Step 4: Cast to f32 (inside @torch.compile)

```python
qn_f = q_nope.float()    # [T, 16, 512]     f32  contiguous
qp_f = q_pe.float()      # [T, 16,  64]     f32  contiguous
Kc_f = Kc.float()        # [T, 2048, 512]   f32  contiguous
Kp_f = Kp.float()        # [T, 2048,  64]   f32  contiguous
```

Required because TF32 doesn't have enough precision (10-bit mantissa → errors > 0.01 on the 512-dim reductions). With `allow_tf32 = False`, these f32 matmuls use true IEEE f32 on tensor cores.

### Step 5: Score Matmuls (BMM)

```
torch.bmm(qn_f, Kc_f.T₂)  →  [T, 16, 512] × [T, 512, 2048]  →  [T, 16, 2048]  f32
torch.bmm(qp_f, Kp_f.T₂)  →  [T, 16,  64] × [T,  64, 2048]  →  [T, 16, 2048]  f32
logits = sum of above       →  [T, 16, 2048]  f32
```

`.transpose(1, 2)` on Kc_f makes it `[T, 512, 2048]` — this is a **non-contiguous view** (stride changes from `[2048*512, 512, 1]` to `[2048*512, 1, 512]`). BMM handles non-contiguous inputs natively via cuBLAS strided batched GEMM.

**FLOP count per batch:**
- MM1: T × 2 × 16 × 2048 × 512 = T × 33.6M
- MM2: T × 2 × 16 × 2048 × 64  = T × 4.2M

### Step 6: Mask + Scale + Softmax

```python
logits.masked_fill_(mask.unsqueeze(1), -inf)   # mask: [T,1,2048] broadcast → [T,16,2048]
logits_scaled = logits * sm_scale               # [T, 16, 2048]  f32

token_lse = logsumexp(logits_scaled, dim=-1) / ln2   # [T, 16]  f32
attn      = softmax(logits_scaled, dim=-1)            # [T, 16, 2048]  f32
```

The mask broadcasts across heads (dim=1). Masked positions get -inf → exp(-inf)=0 after softmax, so they contribute nothing regardless of the garbage values gathered in step 3.

All shapes are contiguous in-place/elementwise — no layout issues.

### Step 7: Output Matmul + Cast

```
torch.bmm(attn, Kc_f)  →  [T, 16, 2048] × [T, 2048, 512]  →  [T, 16, 512]  f32
.to(bf16)               →  [T, 16, 512]  bf16
```

**FLOP:** T × 2 × 16 × 512 × 2048 = T × 33.6M (same as MM1)

### Step 8: Write Outputs

```python
output.copy_(out)          # [T, 16, 512]  bf16 → bf16
lse.copy_(token_lse)       # [T, 16]       f32  → f32
```

---

## Memory Layout Summary

| Tensor | Shape | Dtype | Contiguous? | Notes |
|--------|-------|-------|-------------|-------|
| `q_nope` | [T, 16, 512] | bf16 | ✅ yes | Given input |
| `q_pe` | [T, 16, 64] | bf16 | ✅ yes | Given input |
| `ckv_cache` | [8462, 64, 512] | bf16 | ✅ yes | Page pool |
| `Kc_all` | [541568, 512] | bf16 | ✅ yes (view) | Reshape, no copy |
| `safe_indices` | [T, 2048] | int64 | ✅ yes | Clamped copy |
| `Kc` | [T, 2048, 512] | bf16 | ✅ yes | Gathered copy |
| `Kp` | [T, 2048, 64] | bf16 | ✅ yes | Gathered copy |
| `Kc_f` | [T, 2048, 512] | f32 | ✅ yes | Cast copy |
| `Kc_f.T` | [T, 512, 2048] | f32 | ❌ no (view) | Stride-transposed for BMM |
| `logits` | [T, 16, 2048] | f32 | ✅ yes | BMM output |
| `attn` | [T, 16, 2048] | f32 | ✅ yes | Softmax output |
| `out` | [T, 16, 512] | bf16 | ✅ yes | BMM + cast output |

All tensors are **row-major** (C-contiguous), meaning the last dimension is the fastest-changing in memory. The only non-contiguous tensor is the transposed view `Kc_f.transpose(1,2)` passed to BMM.

---

## Why This Is Faster: Reference vs Batched

```
Reference (per-token loop):             Batched (this solution):
─────────────────────────               ────────────────────────
for t in range(T):                      1× gather [T*2048] from Kc_all
  gather V valid tokens (V=1..2048)     1× compute_attention_batched()
  cast to f32                             → 1× torch.compile invocation
  MM1: [16,512] × [512,V] → [16,V]       → 2× BMM (batched T matrices)
  MM2: [16, 64] × [ 64,V] → [16,V]       → 1× mask + softmax
  softmax + LSE                           → 1× BMM
  MM3: [16,V] × [V,512] → [16,512]     1× copy to output
  copy to output

T kernel launches                       1 kernel launch (fused by torch.compile)
T different V dimensions                Fixed 2048 dim (padded)
No batching benefit                     cuBLAS batched GEMM (T matrices at once)
```

**Waste from padding:** 56% of rows have V < 50 (median = 33), so we compute ~60× more than needed for those. But the single fused kernel launch + batched GEMM efficiency far outweighs this waste.

**Result:** ~7–10× speedup over reference on B200 (steady state).

---

## Latency Profiling (NVIDIA B200)

Profiled with CUDA events per stage, averaged over 23 workloads (steady-state, excluding torch.compile warmup):

### Breakdown (T=2 tokens, representative)

| Stage | Avg (ms) | % of Total | Description |
|-------|----------|------------|-------------|
| **attention** | 0.17 | **63–66%** | `@torch.compile`-d batched attention (4× bf16→f32 casts, 2× BMM scores, mask+scale, logsumexp, softmax, 1× BMM output, cast back) |
| **mask_idx** | 0.036 | ~13% | `sparse_indices == -1` comparison + `clamp(min=0).long()` |
| **gather** | 0.033 | ~12% | Advanced indexing gather of KV cache entries from flat pool |
| **copy_out** | 0.013 | ~5% | `output.copy_(out)` + `lse.copy_(token_lse)` |
| **flatten** | 0.011 | ~4% | `ckv_cache.reshape(-1, D)` (just a view, near-free) |
| **TOTAL** | ~0.28 | 100% | |

### Scaling with T (batch size)

| T | Total (ms) | Attention (ms) | Notes |
|---|-----------|----------------|-------|
| 1 | 0.25 | 0.17 | Minimal batch |
| 2 | 0.28 | 0.17 | Most workloads |
| 6 | 0.28 | 0.19 | Slight attention increase |
| 7 | 0.31 | 0.21 | |
| 8 | 0.31 | 0.21 | Largest batch |

### Bottleneck Analysis

1. **Primary bottleneck: `attention` (~65%)** — The fused `@torch.compile` block doing score computation (2× BMM), softmax, and output matmul (1× BMM). The bf16→f32 casts, materializing the full [T, 16, 2048] attention matrix, and 3 separate BMM calls are the main costs.

2. **Secondary bottleneck: `mask_idx` + `gather` (~25%)** — Index preparation and scattered memory reads from the 529 MiB KV cache pool. The gather reads ~2 MB per token from scattered locations.

3. **Negligible: `flatten` + `copy_out` (~9%)** — Reshapes (views) and final copies are near-free.

### Optimization Targets

- **Attention**: A fused FlashAttention-style kernel (CUDA or Triton) that avoids materializing the full [T, 16, 2048] score matrix and eliminates the f32 cast overhead would cut this stage significantly. Online softmax with tiling would reduce memory bandwidth.
- **Gather + Mask**: Could be fused into a single custom kernel that reads sparse_indices, clamps, gathers, and masks in one pass instead of multiple PyTorch ops.

### Optimization Difficulty Ranking (easiest → hardest)

| Rank | Stage | Time (ms) | % | Difficulty | Rationale |
|------|-------|-----------|---|------------|-----------|
| 1 | **flatten** | 0.011 | 4% | Already optimal | `.reshape()` view — zero-copy, zero-compute. Nothing to optimize. |
| 2 | **copy_out** | 0.013 | 5% | Easy | Eliminate by writing directly into pre-allocated `output`/`lse` buffers instead of creating temporaries. |
| 3 | **mask_idx** | 0.036 | 13% | Easy–Moderate | Two elementwise ops (`== -1`, `clamp + cast`). Fuse into the gather kernel to handle -1 indices on-the-fly, eliminating this stage entirely. |
| 4 | **gather** | 0.033 | 12% | Moderate | Memory-bandwidth bound (~2 MB/token from scattered 529 MiB pool). Custom CUDA kernel can fuse mask+gather+bf16→f32 cast, but can't beat the memory bandwidth wall. Gains from fusion (eliminating intermediate buffers). |
| 5 | **attention** | 0.17–0.21 | 65% | Hard | Requires fused FlashAttention-style tiled kernel: online softmax (avoid materializing [T,16,2048] matrix), split nope+PE score accumulation, lse in log2 space, bf16 I/O with f32 accumulation, custom padding mask. Highest effort but biggest payoff. |

### Recommended Attack Order (bang-for-buck)

1. **copy_out** — trivial fix, saves ~0.013 ms
2. **mask_idx + gather fusion** — moderate effort, saves ~0.07 ms combined by eliminating intermediate tensors and kernel launches
3. **attention** — high effort but 65% of wall time. A fused kernel could cut attention by 2–5×, saving 0.08–0.15 ms

**Realistic optimization ceiling:** from ~0.28 ms down to ~0.10–0.15 ms with fused attention kernel and gather fusion.

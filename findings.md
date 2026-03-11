# DSA Sparse Attention — Findings

Focus: `dsa_ref.py` (track: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`)


---

## Table of Contents

**Part I: DSA in Theory**

- [1. Background](#1-background)
  - [1.1 Vanilla Multi-Head Attention (MHA)](#11-vanilla-multi-head-attention-mha)
  - [1.2 MLA — Multi-head Latent Attention](#12-mla-multi-head-latent-attention)
  - [1.3 DSA — DeepSeek Sparse Attention](#13-dsa-deepseek-sparse-attention)
  - [1.4 Comparison Table](#14-comparison-table)
- [2. Algorithm — Operation Flow](#2-algorithm-operation-flow)
  - [2.1 Dataflow Diagram](#21-dataflow-diagram)
  - [2.2 Operations in Order](#22-operations-in-order)

**Part II: DSA in Production**

- [3. The Inference Pipeline — Where Do These Tensors Come From?](#3-the-inference-pipeline-where-do-these-tensors-come-from)
  - [3.1 Prefill Phase](#31-prefill-phase)
  - [3.2 Decode Phase](#32-decode-phase)
  - [3.3 Batched Decode (What the Benchmark Measures)](#33-batched-decode-what-the-benchmark-measures)
  - [3.4 Tensor Parallelism](#34-tensor-parallelism)
  - [3.5 Tensor Origins Summary](#35-tensor-origins-summary)
- [4. Paging — How the KV Cache is Organized](#4-paging-how-the-kv-cache-is-organized)
  - [4.1 The Problem: KV Cache Size](#41-the-problem-kv-cache-size)
  - [4.2 The Solution: Paged KV Cache](#42-the-solution-paged-kv-cache)
  - [4.3 How the Reference Code Uses Paging](#43-how-the-reference-code-uses-paging)
  - [4.4 Why It Matters for Performance](#44-why-it-matters-for-performance)
- [5. Inputs and Outputs](#5-inputs-and-outputs)
  - [5.1 Inputs](#51-inputs)
  - [5.2 Outputs (DPS)](#52-outputs-dps)
  - [5.3 sm_scale Note](#53-sm_scale-note)
  - [5.4 Decode vs Prefill](#54-decode-vs-prefill)
  - [5.5 sparse_indices Explained](#55-sparse_indices-explained)
- [6. How the Benchmark Works](#6-how-the-benchmark-works)
  - [6.1 Reference Code Source](#61-reference-code-source)
  - [6.2 Evaluation Pipeline](#62-evaluation-pipeline)
  - [6.3 Key Details](#63-key-details)
  - [6.4 What dsa_ref.py Actually Is](#64-what-dsa_refpy-actually-is)
  - [6.5 Precision Strategy](#65-precision-strategy)
- [7. Performance Characteristics](#7-performance-characteristics)
  - [7.1 Bottleneck Analysis](#71-bottleneck-analysis)
  - [7.2 Arithmetic Intensity](#72-arithmetic-intensity)
- [8. Benchmark Data](#8-benchmark-data)
  - [8.1 Track Definition](#81-track-definition)
  - [8.2 Workload Distribution](#82-workload-distribution)
  - [8.3 Input Data Generation](#83-input-data-generation)
  - [8.4 Memory Footprint](#84-memory-footprint)
  - [8.5 Current Performance](#85-current-performance)
- [9. Optimization Opportunities](#9-optimization-opportunities)

---

# Part I: DSA in Theory

<a id="1-background"></a>

## 1. Background

<a id="11-vanilla-multi-head-attention-mha"></a>

### 1.1 Vanilla Multi-Head Attention (MHA)

Standard transformer attention (GPT-style). For a single head:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- $Q$ = query, shape `[1, d_k]` (one new token during decode)
- $K$ = keys, shape `[seq_len, d_k]` (all previous tokens)
- $V$ = values, shape `[seq_len, d_v]` (all previous tokens)
- $d_k$ = head dimension (typically 128)

With `H` heads, each head has its **own separate** K and V. The KV cache stores:
- K cache: `[seq_len, H, d_k]` — one key per head per token
- V cache: `[seq_len, H, d_v]` — one value per head per token

For a model with 128 heads and $d_k = d_v = 128$, the KV cache per token = $128 \times 128 \times 2 \times 2$ bytes (K+V, bf16) = **65,536 bytes/token**. At 100K context, that's ~6.5 GB per sequence per layer — extremely expensive.

<a id="12-mla-multi-head-latent-attention"></a>

### 1.2 MLA — Multi-head Latent Attention

DeepSeek-V2/V3 dramatically reduces KV cache size by **compressing** K and V into a shared low-rank latent vector. Instead of storing separate K and V per head:

**Standard MHA** stores per token: `K[H, d_k]` + `V[H, d_v]` = `128 * 128 * 2` = 32,768 values

**MLA** stores per token: `c_kv[d_c]` = one vector of dim 512 (the "compressed KV"), shared across all heads

The key insight: during training, the model learns **absorption matrices** $W_K^{absorb}$ and $W_V^{absorb}$ that let each query head "decode" K and V from the shared compressed vector on-the-fly. But at inference time, these absorption matrices get **folded into the query projection** (pre-absorption), so the query directly dots with the compressed KV — no need to ever decompress K and V explicitly.

This is why:
- **`q_nope` has dim 512** (not the usual 128) — the query has been "pre-absorbed" to match the compressed KV dim
- **`ckv_cache` is shared across all 16 heads** — only one `[512]` vector per KV token, not `[16, 128]`
- **The output of `attn @ Kc` produces the final output directly** — no separate V multiplication needed

**Positional encoding caveat**: RoPE positional encoding can't be folded into the absorption matrices (it's position-dependent), so it's handled separately:
- `kpe_cache` `[64]` per token — stores the positional key component
- `q_pe` `[64]` per head — the positional query component
- These are added to the attention logits: $\text{logits} = q_{nope} \cdot c_{kv}^T + q_{pe} \cdot k_{pe}^T$

**MLA KV cache per token** = 512 (ckv) + 64 (kpe) = 576 values × 2 bytes = **1,152 bytes/token**
**vs MHA** = 65,536 bytes/token → **~57x reduction**

<a id="13-dsa-deepseek-sparse-attention"></a>

### 1.3 DSA — DeepSeek Sparse Attention

Even with MLA's compressed cache, attending to ALL tokens is $O(\text{seq\_len})$ per query. For 500K+ context, that's still slow. DSA adds a sparsity layer:

1. **Stage 1 (TopK Indexer)**: Use cheap FP8 approximation to score all KV tokens, pick the top-2048 most relevant
2. **Stage 2 (Sparse Attention — this track)**: Run full-precision MLA attention on ONLY those 2048 tokens

The result: instead of attending to 500K tokens, you attend to 2048 — a **~250x reduction** in attention compute.

> **DSA is defined in the DeepSeek-V3.2 technical report** ([arxiv:2512.02556](https://arxiv.org/abs/2512.02556)),
> Section 2.1. DSA stands for **DeepSeek Sparse Attention** — sometimes also referred to as
> "Native Sparse Attention" in benchmark descriptions and SGLang code (`nsa_backend.py`),
> but the canonical name from the paper is **DSA**. It should not be confused with the
> earlier NSA research paper ([arxiv:2502.11089](https://arxiv.org/abs/2502.11089)), which
> describes a different 3-branch architecture (compressed + selected + sliding window) with
> block-level selection — DSA uses token-level selection in a 2-stage pipeline instead.
>
> The paper defines the **lightning indexer** score as:
> $$I_{t,s} = \sum_{j=1}^{H_I} w_{t,j}^I \cdot \text{ReLU}(q_{t,j}^I \cdot k_s^I)$$
> where $H_I$ = number of indexer heads, $q_{t,j}^I \in \mathbb{R}^{d_I}$ and $w_{t,j}^I \in \mathbb{R}$
> are derived from the query token, and $k_s^I \in \mathbb{R}^{d_I}$ from the KV token.
> The paper explicitly states the indexer "has a small number of heads and can be implemented in FP8".
>
> Key DSA properties (from the paper):
> - **2-stage pipeline**: lightning indexer (FP8, cheap) → sparse MLA attention (full precision)
> - **Token-level selection**: "fine-grained token selection" — individual top-2048 tokens, not blocks
> - **Dedicated indexer heads**: $H_I$ separate heads with learned weights, distinct from the MLA attention heads
> - **ReLU activation**: chosen "for throughput consideration"
> - **Instantiated under MLA's MQA mode**: each latent vector (KV entry) is shared across all query heads
>
> The FlashInfer benchmark tracks implement DSA, matching the
> [SGLang](https://github.com/sgl-project/sglang) / [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
> implementation (MIT-licensed). DeepGEMM provides the Stage 1 indexer kernel
> (`fp8_paged_mqa_logits`), while the Stage 2 sparse attention kernel (this benchmark track)
> is what competitors must optimize.
>

<a id="14-comparison-table"></a>

### 1.4 Comparison Table

```
Standard MHA              MLA                        DSA (MLA + Sparse)
─────────────            ─────                      ──────────────────
K: [seq, H, d]           ckv: [seq, d_c]            ckv: [seq, d_c]
V: [seq, H, d]           kpe: [seq, d_kpe]          kpe: [seq, d_kpe]
                                                     sparse_indices: [topk]
                                                       ↓
Q @ K.T                  q_nope @ ckv.T              GATHER 2048 entries
  [H, seq]               + q_pe @ kpe.T              then same as MLA
                           [H, seq]                    [H, 2048]

softmax(logits/√d)       softmax(logits * sm_scale)  softmax(logits * sm_scale)

attn @ V                 attn @ ckv                  attn @ ckv_gathered
  [H, d_v]               [H, d_c]  ← output!         [H, d_c]  ← output!
                          (no separate V needed)

KV cache/token:          KV cache/token:             KV cache/token:
  65,536 B                 1,152 B (57x less)          1,152 B
                                                     Attention tokens:
Attend to: all seq_len   Attend to: all seq_len        2048 (not all)
```


---

<a id="2-algorithm-operation-flow"></a>

## 2. Algorithm — Operation Flow

<a id="21-dataflow-diagram"></a>

### 2.1 Dataflow Diagram

For each token `t` in `[0, T)`:

```
                     ckv_cache [P, 64, 512]
                     kpe_cache [P, 64, 64]
                            │
                     ┌──────┴──────┐
                     │   Flatten   │   reshape(-1, dim) → [P*64, dim]
                     │  to tokens  │
                     └──────┬──────┘
                            │
                    Kc_all [541568, 512]  f32
                    Kp_all [541568, 64]   f32
                            │
      sparse_indices[t]     │
      [2048] int32 ────────►│
                            │
                     ┌──────┴──────┐
                     │   Gather    │   index_select by sparse_indices
                     │  (scatter   │   scattered memory access
                     │   read)     │   ~2.36 MB/token (bf16), ~4.72 MB (f32 ref)
                     └──────┬──────┘
                            │
                    Kc [2048, 512]  f32
                    Kp [2048, 64]   f32
                            │
      q_nope[t] ───────────►│
      [16, 512] f32         │
                     ┌──────┴──────┐
                     │   MatMul 1  │   q_nope @ Kc.T
                     │  (GEMM)     │   [16,512] x [512,2048] = [16,2048]
                     │  33.6M FLOP │
                     └──────┬──────┘
                            │
      q_pe[t] ─────────────►│
      [16, 64] f32          │
                     ┌──────┴──────┐
                     │   MatMul 2  │   q_pe @ Kp.T
                     │  (GEMM)     │   [16,64] x [64,2048] = [16,2048]
                     │   4.2M FLOP │
                     └──────┬──────┘
                            │
                     ┌──────┴──────┐
                     │     Add     │   logits = MM1 + MM2
                     │             │   [16, 2048]
                     └──────┬──────┘
                            │
                     ┌──────┴──────┐
                     │    Scale    │   logits_scaled = logits * sm_scale
                     │             │   [16, 2048] (elementwise)
                     └──────┬──────┘
                            │
                    ┌───────┴───────┐
            ┌───────┴──────┐ ┌──────┴───────┐
            │  LogSumExp   │ │   Softmax    │
            │  (base 2)    │ │              │
            │  → lse[t]    │ │  → attn      │
            │  [16]  f32   │ │  [16, 2048]  │
            └──────────────┘ └──────┬───────┘
                                    │
                             ┌──────┴──────┐
                             │   MatMul 3  │   attn @ Kc
                             │  (GEMM)     │   [16,2048] x [2048,512] = [16,512]
                             │  33.6M FLOP │
                             └──────┬──────┘
                                    │
                             ┌──────┴──────┐
                             │   Cast      │   f32 → bf16
                             │             │
                             └──────┬──────┘
                                    │
                              output[t] [16, 512] bf16
```

<a id="22-operations-in-order"></a>

### 2.2 Operations in Order

1. **Flatten** KV caches from paged layout `[P, 64, dim]` → flat `[P*64, dim]`, cast bf16→f32
2. **Gather** — index into flattened KV using `sparse_indices[t]` (scattered access, ~2.36 MB/token bf16 or ~4.72 MB f32)
3. **MatMul 1** — `q_nope[t] @ Kc.T` → `[16, 2048]` (33.6M FLOP)
4. **MatMul 2** — `q_pe[t] @ Kp.T` → `[16, 2048]` (4.2M FLOP)
5. **Add** — `logits = MM1 + MM2` (elementwise)
6. **Scale** — `logits *= sm_scale` (elementwise)
7. **LogSumExp** — `lse[t] = logsumexp(logits_scaled) / ln(2)` (reduction per head)
8. **Softmax** — `attn = softmax(logits_scaled)` (per head, over 2048 entries)
9. **MatMul 3** — `attn @ Kc` → `[16, 512]` (33.6M FLOP) — note: output dims = ckv dim (MLA: K=V compressed together)
10. **Cast** — f32 → bf16 and write to `output[t]`

**Total per token: ~71.3M FLOP** across 3 matrix multiplications, plus gather + softmax + elementwise ops.


---

# Part II: DSA in Production

<a id="3-the-inference-pipeline-where-do-these-tensors-come-from"></a>

## 3. The Inference Pipeline — Where Do These Tensors Come From?

When a user sends a prompt to a DeepSeek-V3 server, here's what happens:

<a id="31-prefill-phase"></a>

### 3.1 Prefill Phase

User types: *"Explain quantum computing"* (say 5 tokens after tokenization)

```
Tokens: [<bos>, "Explain", "quantum", "computing", <eos>]

For each transformer layer:
  1. Project all 5 tokens through Q, K, V projections
  2. MLA compression: K,V → ckv (dim 512) + kpe (dim 64) per token
  3. Store in KV cache page pool: allocate 1 page (holds 64 tokens, only 5 used)
  4. Run full attention across all 5 tokens (no sparsity needed — context is tiny)
  5. Produce output for each token
```

After prefill, the KV cache has 5 entries for this sequence.

<a id="32-decode-phase"></a>

### 3.2 Decode Phase

The model generates the response token by token. For EACH new token:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  New token generated: "Quantum"                                          │
│                                                                          │
│  1. PROJECT QUERY                                                        │
│     Input: embedding of "Quantum"                                        │
│     → q_nope [1, 16, 512]  ← query (pre-absorbed, 16 heads × 512 dim)   │
│     → q_pe   [1, 16, 64]   ← positional query component                 │
│                                                                          │
│  2. UPDATE KV CACHE                                                      │
│     Compress "Quantum" → ckv [512], kpe [64]                             │
│     Append to this sequence's pages in the pool                          │
│     ckv_cache and kpe_cache now have one more token                      │
│                                                                          │
│  3. STAGE 1: TOPK INDEXING (the other track)                             │
│     Use FP8 approximate scores to find top-2048 most relevant KV tokens  │
│     → sparse_indices [1, 2048]  ← flat indices into the page pool        │
│     (if context < 2048 tokens, remaining slots filled with -1)           │
│                                                                          │
│  4. STAGE 2: SPARSE ATTENTION (this track — what dsa_ref.py computes)    │
│     Inputs:                                                              │
│       q_nope     [1, 16, 512]   ← from step 1                           │
│       q_pe       [1, 16, 64]    ← from step 1                           │
│       ckv_cache  [P, 64, 512]   ← shared page pool (all sequences)      │
│       kpe_cache  [P, 64, 64]    ← shared page pool                      │
│       sparse_indices [1, 2048]  ← from step 3                           │
│       sm_scale                  ← model constant                         │
│     Outputs:                                                             │
│       output [1, 16, 512]       ← attention result                       │
│       lse    [1, 16]            ← log-sum-exp (for numerical stability)  │
│                                                                          │
│  5. REST OF LAYER                                                        │
│     output → output projection → residual add → FFN → next layer         │
│                                                                          │
│  6. NEXT TOKEN                                                           │
│     Final layer output → language model head → sample next token          │
│     Repeat from step 1 with the next token                               │
└──────────────────────────────────────────────────────────────────────────┘
```

<a id="33-batched-decode-what-the-benchmark-measures"></a>

### 3.3 Batched Decode (What the Benchmark Measures)

In production, the server handles **multiple users simultaneously**. With batch size T=8:

```
User A: "Explain quantum..."  (context: 191 tokens)  → 63 selected by topk
User B: "Write code for..."   (context: 200 tokens)  → 9 selected
User C: "Summarize this..."   (context: 2948 tokens) → 2048 selected (full topk)
User D: "Hello"               (context: 3220 tokens) → 212 selected
User E: (short context)                               → 11 selected
User F: (short context)                               → 25 selected
User G: (short context)                               → 6 selected
User H: (medium context)                              → 50 selected
```

All 8 sequences share the same `ckv_cache` / `kpe_cache` page pool (8462 pages), but each indexes into **its own non-overlapping region**. The queries are batched:

- `q_nope`: `[8, 16, 512]` — 8 queries (one per sequence)
- `sparse_indices`: `[8, 2048]` — each row points to different parts of the page pool

The kernel processes all 8 in a single call — this is the setting our optimization needs to be fast for.

<a id="34-tensor-parallelism"></a>

### 3.4 Tensor Parallelism

DeepSeek-V3.2 has **128 attention heads** total. In production, these are split across 8 GPUs using Tensor Parallelism (TP=8), so each GPU runs **16 heads** independently.

```
Full model:    128 heads × 61 layers
Per GPU (TP=8): 16 heads × 61 layers  ← what the benchmark simulates
```

Key points:
- **MLA's KV cache is NOT split by TP** — the compressed `ckv` (dim 512) and `kpe` (dim 64) are shared across all heads, so every GPU holds its own full copy of the KV cache
- **What IS split**: the query projections and output projections — each GPU only computes its 16 heads
- **No cross-GPU communication** during the attention kernel itself — all-reduce happens after the output projection, outside the attention step
- **The benchmark operates on one TP shard**: the track name `h16` means 16 heads, which is exactly one GPU's worth. All workloads represent the single-GPU view — no multi-GPU handling needed

For this benchmark, you're optimizing the single-GPU attention kernel for 16 heads. The TP=8 context only matters for understanding the production VRAM budget (Section 4.2).

<a id="35-tensor-origins-summary"></a>

### 3.5 Tensor Origins Summary

| Tensor | Produced by | When |
|--------|-------------|------|
| `q_nope` | Query projection + absorption (W_Q × embedding) | Each decode step |
| `q_pe` | Query RoPE projection | Each decode step |
| `ckv_cache` | KV compression of all past tokens, stored in page pool | Accumulated over all steps |
| `kpe_cache` | Positional key encoding of all past tokens | Accumulated over all steps |
| `sparse_indices` | Stage 1 TopK indexer (FP8 scoring) | Each decode step |
| `sm_scale` | Model architecture constant | Fixed |
| `output` | **This kernel produces it** — attention result | Each decode step |
| `lse` | **This kernel produces it** — for FlashAttention-style log-sum-exp merging | Each decode step |


---

<a id="4-paging-how-the-kv-cache-is-organized"></a>

## 4. Paging — How the KV Cache is Organized

<a id="41-the-problem-kv-cache-size"></a>

### 4.1 The Problem: KV Cache Size

DeepSeek-V3.2 uses MLA with compressed KV. Per token, per layer, per TP shard:

| Component | Dimensions | Size (bf16) |
|---|---|---|
| `ckv` (compressed KV) | 512 | 1,024 B |
| `kpe` (positional key) | 64 | 128 B |
| **Total per token per layer** | | **1,152 B** |

DeepSeek-V3.2 has **61 transformer layers**. Across all layers on one TP shard:
- **1,152 B × 61 = 68.6 KB per token** (all layers, one GPU)
- With TP=8, each GPU holds its own copy (MLA's `ckv` is shared across all heads, not split by TP)

For context, here's what that means for a single user's KV cache across all 61 layers:

| Context length | Pages (64 tok/page) | KV per layer | KV all 61 layers |
|---|---|---|---|
| 500 tokens (short chat) | 8 | 0.55 MB | 0.03 GB |
| 4K tokens (conversation) | 64 | 4.5 MB | 0.27 GB |
| 32K tokens (document) | 512 | 36 MB | 2.14 GB |
| 128K tokens (long reasoning) | 2,048 | 144 MB | 8.58 GB |

In a serving system, multiple user sequences run simultaneously. Allocating a contiguous `[max_seq_len, dim]` buffer per user wastes memory — a 500-token chat would reserve the same 8.58 GB as a 128K reasoning trace.

<a id="42-the-solution-paged-kv-cache"></a>

### 4.2 The Solution: Paged KV Cache

Borrowed from OS virtual memory — instead of contiguous per-sequence buffers, the KV cache is a **shared pool of fixed-size pages** (like a memory allocator). Each page holds 64 tokens.

#### The Benchmark Pool (one layer, one TP shard)

```
ckv_cache: [8462 pages, 64 tokens/page, 512 dim] bf16 = 529 MiB (555 MB)
kpe_cache: [8462 pages, 64 tokens/page,  64 dim] bf16 =  66 MiB ( 69 MB)
─────────────────────────────────────────────────────────────────────────
Total pool: 595 MiB (624 MB)                     541,568 token slots
```

(All "MB" figures in this document use **MiB** = 2²⁰ bytes unless noted otherwise.)

In production, this pool would be replicated across all 61 layers, totaling **~35.4 GiB per TP shard**.

#### How Users Share the Pool

Pages are assigned dynamically. The 8,462-page pool can hold:

| Scenario | Pages/user | Max concurrent users |
|---|---|---|
| Short chats (500 tokens) | 8 | **1,057 users** |
| Conversations (4K tokens) | 64 | **132 users** |
| Documents (32K tokens) | 512 | **16 users** |
| Long reasoning (128K tokens) | 2,048 | **4 users** |

In practice, a batch has a **mix** — some short, some long. The benchmark's actual batch (T=8, one workload) uses only ~106 pages out of 8,462 (1.3%), serving 8 users with a mix of tiny (6-token) to medium (3K-token) contexts.

#### B200 Production VRAM Budget

On an NVIDIA B200 GPU (192 GB HBM3e) running DeepSeek-V3.2 with TP=8:

```
Model weights (671B / 8 GPUs):    ~84 GB
Activations & workspace:          ~10 GB
Available for KV cache:           ~98 GB

At 68.6 KB/token (all 61 layers), that's ~1.5M token slots total.
→ 11 users at 128K context
→ 365 users at 4K context
→ Benchmark pool (541K slots) = 36% of max capacity
```

<a id="43-how-the-reference-code-uses-paging"></a>

### 4.3 How the Reference Code Uses Paging

In `dsa_ref.py`, the paged cache is simply **flattened** into a token-level array:

```python
# Flatten: [num_pages, 64, dim] → [num_pages * 64, dim]
Kc_all = ckv_cache.reshape(-1, 512)   # [541568, 512]
Kp_all = kpe_cache.reshape(-1, 64)    # [541568, 64]

# Then index with sparse_indices[t] (flat offsets)
tok_idx = sparse_indices[t][sparse_indices[t] != -1]
Kc = Kc_all[tok_idx]   # random gather from 541K entries
```

The flat index encodes page structure: `flat_index = page_idx × 64 + offset_within_page`.
For example, index `65` = page 1, offset 1. Index `8201` = page 128, offset 9.

A **block table** (mapping logical→physical pages per user) exists in the TopK indexer track but is **not used in this track** — `sparse_indices` already contains resolved flat offsets.

<a id="44-why-it-matters-for-performance"></a>

### 4.4 Why It Matters for Performance

**Naive expectation**: The gather reads from 541K scattered positions — terrible cache locality.

**Reality from the data** (measured across all 23 workloads, 120 token rows):
- **96.7% of selected tokens share a page** with another selected token (mean intra-page clustering)
- When a page is touched, on average **60-63 out of 64 slots** are selected (nearly the full page)
- For fully-filled rows (2048 tokens), **75-95% of consecutive gaps are exactly 1** — nearly sequential
- **94% of rows are sorted** in ascending order (only fully-filled rows may be unsorted from topk)

This means the access pattern is **not truly random** — DSA's lightning indexer tends to select nearly-whole pages. A page-aware kernel that loads full 64-token pages and masks unused slots could achieve **coalesced memory access**, reading ~4 KB chunks (64 × 64B per ckv entry) instead of scattered individual tokens.


---

<a id="5-inputs-and-outputs"></a>

## 5. Inputs and Outputs

<a id="51-inputs"></a>

### 5.1 Inputs

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `q_nope` | `[T, 16, 512]` | bf16 | Query, non-positional component |
| `q_pe` | `[T, 16, 64]` | bf16 | Query, positional encoding component |
| `ckv_cache` | `[P, 64, 512]` | bf16 | Paged compressed KV cache |
| `kpe_cache` | `[P, 64, 64]` | bf16 | Paged key positional encoding cache |
| `sparse_indices` | `[T, 2048]` | int32 | Pre-selected top-K token indices (-1 = padding) |
| `sm_scale` | scalar | f32 | Softmax scale (benchmark value: **0.1352**) * |

Where:
- **T** = `num_tokens` (variable: 1–8 in benchmark)
- **P** = `num_pages` (fixed at 8462 in benchmark, total 541,568 KV tokens)
- Page size is always 64 tokens per page

<a id="52-outputs-dps"></a>

### 5.2 Outputs (DPS)

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `output` | `[T, 16, 512]` | bf16 | Attention output per token per head |
| `lse` | `[T, 16]` | f32 | Log-sum-exp (base 2) of attention logits |

<a id="53-sm_scale-note"></a>

### 5.3 sm_scale Note

\* The benchmark definition JSON describes the scale formula as $\frac{1}{\sqrt{head\_dim\_qk + head\_dim\_kpe}} = \frac{1}{\sqrt{128 + 64}} = \frac{1}{\sqrt{192}} \approx 0.0722$. However, the actual benchmark value is **0.1352** ($\approx \frac{1}{\sqrt{54.7}}$), suggesting a different scaling convention may be in use within DeepSeek-V3.2. The kernel treats `sm_scale` as an opaque constant — it simply multiplies the raw logits by whatever value the benchmark provides.

<a id="54-decode-vs-prefill"></a>

### 5.4 Decode vs Prefill

This is **batched decode**. Each "token" in `num_tokens` is **one query from a different sequence** running simultaneously. Evidence:

- The definition says: *"num_tokens: Number of tokens (batch_size for decode, total_num_tokens for prefill)"*
- `num_tokens` ranges 1–8 in the workloads — these are batch sizes, not sequence lengths
- Each token's `sparse_indices` row indexes into a **non-overlapping region** of the shared KV cache page pool (verified from real data — zero overlap between any two tokens' index ranges)
- The valid index counts vary wildly per token within a batch (e.g., `[288, 4, 1884, 21, 136, 2048, 42, 335]`), corresponding to sequences of very different lengths sharing the same page pool

So `T=8` means 8 different user requests are being served simultaneously, each with its own KV cache pages, all packed into a shared page pool of 8462 pages.

<a id="55-sparse_indices-explained"></a>

### 5.5 sparse_indices Explained

`sparse_indices` is a `[T, 2048]` int32 tensor. Each row selects **which KV tokens a given query should attend to**, out of 541,568 total KV token slots in the page pool.

**Index encoding**: Each index is a flat offset into the flattened KV cache:
```
flat_index = page_idx * 64 + offset_within_page
```
For example, index `65` means page 1, offset 1 (`65 = 1*64 + 1`).

**Padding**: Values of `-1` mean "no token here" — the slot is unused. This happens when a sequence has fewer than 2048 KV tokens (so topk can't fill all 2048 slots).

**Real data patterns** (from the benchmark safetensors):

```
Workload example (T=8, num_pages=8462):

token 0:    63 valid, range=[  128..  190],  1 page   ← short sequence (~190 tokens)
token 1:     9 valid, range=[  192..  200],  1 page   ← very short (~200 tokens)
token 2:  2048 valid, range=[  257..2948], 43 pages   ← long, topk fully filled
token 3:   212 valid, range=[ 3008..3219],  4 pages   ← medium
token 4:    11 valid, range=[ 3264..3274],  1 page    ← very short
token 5:    25 valid, range=[ 3328..3352],  1 page    ← short
token 6:     6 valid, range=[ 3392..3397],  1 page    ← tiny
token 7:    50 valid, range=[ 3456..3505],  2 pages   ← short
```

#### Distribution Across All 23 Workloads (120 rows total)

**Valid count per row** (how many of the 2048 slots are used):

| Statistic | Value |
|---|---|
| Min | 1 |
| Max | 2048 |
| Mean | 289 |
| **Median** | **33** |
| P25 | 16 |
| P75 | 225 |
| P95 | 1,987 |

**Histogram** — most rows are short-context sequences with very few valid tokens:

```
[    0,    10):  18 rows (15.0%)  ████████
[   10,    50):  49 rows (40.8%)  ██████████████████████   ← dominant bucket
[   50,   100):  12 rows (10.0%)  █████
[  100,   200):  10 rows ( 8.3%)  ████
[  200,   500):  15 rows (12.5%)  ██████
[  500, 1000):    1 rows ( 0.8%)  
[ 1000, 1500):    4 rows ( 3.3%)  █
[ 1500, 2048):    7 rows ( 5.8%)  ███
      = 2048 :    4 rows ( 3.3%)  █       ← fully filled
```

**Structural properties**:

| Property | Value | Implication |
|---|---|---|
| Rows fully sorted (ascending) | 94% (109/116) | Unsorted rows are the 4 fully-filled (2048) ones |
| Intra-page clustering (mean) | 96.7% | When a page is touched, almost all its tokens are selected |
| Tokens per touched page (500+ rows) | avg 60-63 / 64 | Pages are selected nearly-whole |
| Index range | 64 to 8,202 | Page 0 never used (reserved?), max used = page 128 |
| Approximate context span (median) | 33 tokens | Most sequences are very short |
| Approximate context span (max) | 8,075 tokens | Longest context in the benchmark |

**Gap analysis** (fully-filled 2048 rows only — the hardest case for memory access):

| Workload | Index span | Consecutive gaps = 1 | Max gap |
|---|---|---|---|
| 0, token 2 | 2,692 | 75.7% | 5 |
| 10, token 5 | 2,405 | 85.4% | 6 |
| 12, token 2 | 2,161 | 95.2% | 4 |
| 16, token 2 | 2,162 | 95.1% | 5 |

Even in the worst case (2048 tokens from a ~2700-token span), 76-95% of indices are consecutive — the access is nearly sequential, not random.

Key observations:
- **Non-overlapping**: each token occupies its own contiguous slice of the page pool (no two tokens share pages)
- **Huge variance**: valid counts range from 1 to 2048 — most sequences are short (median 33), a few are long
- **Heavily skewed**: 56% of rows have < 50 valid tokens, only 3.3% are fully filled
- **Nearly-whole page selection**: intra-page clustering of 96.7% means the indexer selects almost all tokens in a page when it selects any — this is exploitable for coalesced loads
- **Sorted by default**: 94% of rows are fully sorted ascending — only the fully-filled rows (where topk doesn't preserve order) may be unsorted
- **page_idx=0 is never used** — indices start at 64 (page 1), suggesting page 0 may be reserved


---

<a id="6-how-the-benchmark-works"></a>

## 6. How the Benchmark Works

<a id="61-reference-code-source"></a>

### 6.1 Reference Code Source

The reference code comes from the **definition JSON file** (`dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json`). The JSON has a `"reference"` field containing a complete Python `run()` function — the same naive loop code as `ref1.py`. This is **not** an optimized FlashInfer kernel.

At evaluation time, the framework:
1. Calls `BuilderRegistry.build_reference(definition)` — creates a pseudo `Solution` from the `definition.reference` string with `destination_passing_style=False` and `entry_point="main.py::run"`
2. Runs this reference on each workload to produce **ground-truth outputs** (`output`, `lse`)
3. Times the reference execution to get **reference latency**

<a id="62-evaluation-pipeline"></a>

### 6.2 Evaluation Pipeline

Per workload:

```
┌─────────────────────────────────────────────────────────────┐
│  1. BUILD BASELINE                                          │
│                                                             │
│  ref_runnable = build_reference(definition)                 │
│    └── source: definition JSON "reference" field            │
│    └── naive Python loop, value-returning, NO torch.compile │
│                                                             │
│  For each trial (3x):                                       │
│    inputs[i] = gen_inputs(definition, workload)             │
│    outputs[i] = ref_runnable(*inputs[i])   ← ground truth  │
│                                                             │
│  Time reference: 10 warmup + 50 iters per trial             │
│    └── L2 cache cleared before each iter                    │
│    └── tensor args cloned per iter (setup not timed)        │
│  ref_latency = mean across trials                           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  2. CHECK CORRECTNESS                                       │
│                                                             │
│  For each trial:                                            │
│    Allocate output tensors (DPS)                            │
│    sol_runnable(*inputs[i], *out_tensors)                   │
│    Compare out_tensors vs ref outputs[i]                    │
│    Check: atol < 0.01, rtol < 0.01                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  3. MEASURE PERFORMANCE                                     │
│                                                             │
│  For each trial:                                            │
│    args = [*inputs[i], *output_tensors]                     │
│    sol_latency = do_bench(sol, warmup=10, rep=50)           │
│      └── same L2 clear + clone methodology                  │
│                                                             │
│  speedup = ref_latency / sol_latency                        │
└─────────────────────────────────────────────────────────────┘
```

<a id="63-key-details"></a>

### 6.3 Key Details

- **Correctness** is checked against the reference code's output (computed live), not pre-stored golden outputs
- **Speedup** = `ref_mean_latency / sol_mean_latency` — timed against the **same naive Python reference**
- **Timing** uses CUDA events with L2 cache flush (256 MB zero-fill) before each iteration
- **3 trials** with different random inputs (except `sparse_indices` from safetensors), 10 warmup + 50 timed iterations each
- The reference runs with `destination_passing_style=False` (returns tensors); your solution runs with `destination_passing_style=True` (writes in-place to pre-allocated tensors)

<a id="64-what-dsa_refpy-actually-is"></a>

### 6.4 What dsa_ref.py Actually Is

`dsa_ref.py` is **essentially the reference code** with two changes:
1. DPS signature (writes to `output`/`lse` params instead of returning them)
2. `@torch.compile` on inner `compute_attention()` function

This gives ~**1.09x speedup** — torch.compile fuses the matmuls/softmax/logsumexp into one kernel, slightly beating the unfused PyTorch reference. It's not a real optimization.

<a id="65-precision-strategy"></a>

### 6.5 Precision Strategy

- KV cache stored in **bf16** (paged)
- All compute done in **f32** (`.to(torch.float32)`)
- Output cast back to **bf16** at the end
- LSE stays in **f32**
- Tolerance: atol < 0.01, rtol < 0.01 (generous for bf16)


---

<a id="7-performance-characteristics"></a>

## 7. Performance Characteristics

<a id="71-bottleneck-analysis"></a>

### 7.1 Bottleneck Analysis

| Component | Per token (bf16 gather) | Per token (ref code: f32 gather) | Notes |
|-----------|------------------------|----------------------------------|-------|
| Gather Kc | 2.10 MB | 4.19 MB | 2048 × 512 × {2,4} B |
| Gather Kp | 0.26 MB | 0.52 MB | 2048 × 64 × {2,4} B |
| **Total gather** | **2.36 MB** | **4.72 MB** | |
| MatMul 1 (qn@Kc.T) | 33.6M FLOP | 33.6M FLOP | Largest compute |
| MatMul 2 (qp@Kp.T) | 4.2M FLOP | 4.2M FLOP | Small |
| MatMul 3 (attn@Kc) | 33.6M FLOP | 33.6M FLOP | Largest compute |
| Softmax/LSE | ~65K elements | ~65K elements | Per head: 2048 values |

At topk=2048 with dim=512, the per-token compute is only **71.3M FLOP** — this is small. The gather is the dominant cost, making this operation **memory-bandwidth bound**.

**Important distinction**: The reference code casts the *entire* KV cache to f32 before gathering (4 bytes/element), doubling the gather bandwidth. An optimized kernel should gather in bf16 and upcast in registers (2 bytes/element).

<a id="72-arithmetic-intensity"></a>

### 7.2 Arithmetic Intensity

| Scenario | Gather bandwidth | AI |
|---|---|---|
| Optimized (bf16 gather) | 2.36 MB | ~30 FLOP/Byte |
| Reference code (f32 gather) | 4.72 MB | ~15 FLOP/Byte |

Despite the seemingly reasonable arithmetic intensity in the bf16 case, the **scattered access pattern** of the gather (2048 reads from 541K positions) destroys cache locality and limits effective bandwidth well below peak.


---

<a id="8-benchmark-data"></a>

## 8. Benchmark Data

<a id="81-track-definition"></a>

### 8.1 Track Definition

- **Track**: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`
- **Model**: DeepSeek-V3.2
- **Tensor Parallel**: TP=8 in production — see [Section 3.4](#34-tensor-parallelism) for details. The benchmark operates on **one shard** (`h16` = 16 heads).

<a id="82-workload-distribution"></a>

### 8.2 Workload Distribution

23 total workloads:

| num_tokens | num_pages | Count | Total KV tokens |
|:----------:|:---------:|:-----:|:---------------:|
| 1 | 8462 | 1 | 541,568 |
| 2 | 8462 | 8 | 541,568 |
| 6 | 8462 | 3 | 541,568 |
| 7 | 8462 | 3 | 541,568 |
| 8 | 8462 | 8 | 541,568 |

All workloads share `num_pages=8462`, only `num_tokens` varies (1–8).

<a id="83-input-data-generation"></a>

### 8.3 Input Data Generation

| Tensor | Source | Notes |
|--------|--------|-------|
| `q_nope` | Random | Generated by harness |
| `q_pe` | Random | Generated by harness |
| `ckv_cache` | Random | Generated by harness |
| `kpe_cache` | Random | Generated by harness |
| `sparse_indices` | **Safetensors file** | Real indices captured from DeepSeek-V3.2 inference |
| `sm_scale` | Scalar constant | 0.1352337788608801 |

The `sparse_indices` are the only non-random input — they are **captured from real model inference**, preserving realistic sparsity patterns and index distributions. This means the gather access pattern in benchmarks reflects production workloads.

<a id="84-memory-footprint"></a>

### 8.4 Memory Footprint

Largest workload (T=8):

| Tensor | Size | Shape |
|--------|------|-------|
| `ckv_cache` | **529 MiB** | [8462, 64, 512] bf16 |
| `kpe_cache` | 66 MiB | [8462, 64, 64] bf16 |
| `q_nope` | 0.13 MiB | [8, 16, 512] bf16 |
| `q_pe` | 0.02 MiB | [8, 16, 64] bf16 |
| `sparse_indices` | 0.06 MiB | [8, 2048] int32 |
| **Total inputs** | **~595 MiB** | Dominated by KV cache |

<a id="85-current-performance"></a>

### 8.5 Current Performance

Modal B200:

- Reference baseline (naive Python loop from definition JSON): ~1.20 ms (1 token, 8462 pages)
- `dsa_ref.py` with `@torch.compile`: **1.105 ms, 1.09x speedup**
- This is **not competitive** — the reference is a naive Python loop, and we're barely beating it


---

<a id="9-optimization-opportunities"></a>

## 9. Optimization Opportunities

1. **Fuse gather + attention** — avoid materializing gathered Kc/Kp tensors; load directly in the attention kernel (Flash Attention style)
2. **Batch across tokens** — current implementation loops over tokens sequentially; a custom kernel can parallelize across T, heads, and topk simultaneously
3. **Keep KV in bf16 during gather, upcast in registers** — reduces gather bandwidth by avoiding the f32 pre-cast of the entire cache
4. **Tiled attention** — split the 2048 KV tokens into tiles, compute partial attention per tile, merge with online softmax (FlashAttention algorithm)
5. **Triton kernel** — write a fused Triton kernel that handles gather + MLA attention + softmax + output in a single GPU kernel launch
6. **Exploit page structure** — sparse_indices encode `page_idx * 64 + offset`; tokens within the same page are contiguous in memory, enabling coalesced loads when multiple selected tokens share a page

"""Phase-level CUDA timing for idxer_tc.py (no NVTX, just cudaEvents).

Usage:
  WORKLOAD_IDX=-1  →  last workload (default)
  CONTEST_DIR      →  path to contest data (default: env FIB_DATASET_PATH)
  WARMUP           →  warmup iterations (default: 5)
  REPS             →  timed iterations to average (default: 20)

Example:
  python src/profile_idxer_tc.py
  WORKLOAD_IDX=0 python src/profile_idxer_tc.py
"""
import os, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from safetensors.torch import load_file

# ── Config ──
WORKLOAD_IDX = int(os.environ.get("WORKLOAD_IDX", "-1"))
CONTEST      = Path(os.environ.get("CONTEST_DIR",
                    os.environ.get("FIB_DATASET_PATH", "/data")))
WARMUP       = int(os.environ.get("WARMUP", "5"))
REPS         = int(os.environ.get("REPS", "20"))

NUM_HEADS  = 64
HEAD_DIM   = 128
TOPK       = 2048
PAGE_SIZE  = 64

JSONL = (CONTEST / "workloads" / "dsa_paged" /
         "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl")

workloads = [json.loads(l) for l in open(JSONL)]
idx = WORKLOAD_IDX if WORKLOAD_IDX >= 0 else len(workloads) - 1
w   = workloads[idx]
ax  = w["workload"]["axes"]
inp = w["workload"]["inputs"]

batch_size    = ax["batch_size"]
num_pages     = ax["num_pages"]
max_num_pages = ax["max_num_pages"]
uuid_short    = w["workload"]["uuid"][:8]
print(f"Workload {idx} / {len(workloads)-1}: uuid={uuid_short}  "
      f"B={batch_size}  num_pages={num_pages}  max_pages={max_num_pages}")

# ── Tensors ──
device = "cuda"
q_index_fp8 = torch.randn(batch_size, NUM_HEADS, HEAD_DIM,
                           dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
k_index_cache_fp8 = torch.randint(0, 256,
                                   (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
                                   dtype=torch.uint8, device=device).view(torch.int8)
weights     = torch.randn(batch_size, NUM_HEADS, dtype=torch.float32, device=device)
topk_indices = torch.full((batch_size, TOPK), -1, dtype=torch.int32, device=device)

sf          = load_file(str(CONTEST / inp["seq_lens"]["path"]))
seq_lens    = sf[inp["seq_lens"]["tensor_key"]].to(device)
block_table = sf[inp["block_table"]["tensor_key"]].to(device)

# ── Import kernel pieces directly so we can time them individually ──
from src.kernels.idxer_tc import (
    dequant_fp8_kv_cache, _score_and_reduce, _topk_remap_and_write
)


def run_phases(q_fp8, k_cache_fp8, wts, slens, btable, topk_out):
    """Run all phases, returning intermediate tensors for individual timing."""
    B = q_fp8.shape[0]
    q   = q_fp8.to(torch.float32)
    K_all  = dequant_fp8_kv_cache(k_cache_fp8)
    K_flat = K_all.reshape(-1, HEAD_DIM)

    max_sl = btable.shape[1] * PAGE_SIZE
    offsets      = torch.arange(PAGE_SIZE, device=device)
    token_indices = (btable.long().unsqueeze(2) * PAGE_SIZE +
                     offsets.view(1, 1, PAGE_SIZE)).reshape(B, max_sl)
    positions    = torch.arange(max_sl, device=device).unsqueeze(0)
    mask         = positions >= slens.unsqueeze(1)
    token_indices = token_indices.clamp(0, K_flat.shape[0] - 1)
    K_gathered   = K_flat[token_indices.reshape(-1)].reshape(B, max_sl, HEAD_DIM)
    final        = _score_and_reduce(q, K_gathered, wts, mask)
    _topk_remap_and_write(final, btable.long(), mask, topk_out, TOPK, PAGE_SIZE)


def timed(fn, *args, reps=REPS):
    """Return mean elapsed time in µs over `reps` runs using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    # warmup this specific fn
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / reps  # ms→µs


# ── Warmup full pipeline to trigger torch.compile ──
print(f"Warming up ({WARMUP} iters)...")
for _ in range(WARMUP):
    run_phases(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
torch.cuda.synchronize()
print("Done warming up.\n")

# ── Phase definitions ──
B = batch_size
max_sl = max_num_pages * PAGE_SIZE

# Pre-build intermediates once for isolated phase timing
q        = q_index_fp8.to(torch.float32)
K_all    = dequant_fp8_kv_cache(k_index_cache_fp8)
K_flat   = K_all.reshape(-1, HEAD_DIM)
offsets  = torch.arange(PAGE_SIZE, device=device)
token_indices = (block_table.long().unsqueeze(2) * PAGE_SIZE +
                 offsets.view(1, 1, PAGE_SIZE)).reshape(B, max_sl)
positions = torch.arange(max_sl, device=device).unsqueeze(0)
mask      = positions >= seq_lens.unsqueeze(1)
token_indices_c = token_indices.clamp(0, K_flat.shape[0] - 1)
K_gathered = K_flat[token_indices_c.reshape(-1)].reshape(B, max_sl, HEAD_DIM)
final     = _score_and_reduce(q, K_gathered, weights, mask)

phases = [
    ("dequant_q         ",
     lambda: q_index_fp8.to(torch.float32)),
    ("dequant_kv_cache  ",
     lambda: dequant_fp8_kv_cache(k_index_cache_fp8)),
    ("build_indices     ",
     lambda: (block_table.long().unsqueeze(2) * PAGE_SIZE +
              offsets.view(1, 1, PAGE_SIZE)).reshape(B, max_sl)),
    ("build_mask        ",
     lambda: torch.arange(max_sl, device=device).unsqueeze(0) >= seq_lens.unsqueeze(1)),
    ("gather            ",
     lambda: K_flat[token_indices_c.reshape(-1)].reshape(B, max_sl, HEAD_DIM)),
    ("score_and_reduce  ",
     lambda: _score_and_reduce(q, K_gathered, weights, mask)),
    ("topk_remap_write  ",
     lambda: _topk_remap_and_write(final, block_table.long(), mask, topk_indices, TOPK, PAGE_SIZE)),
    ("TOTAL (end-to-end)",
     lambda: run_phases(q_index_fp8, k_index_cache_fp8, weights, seq_lens,
                        block_table, topk_indices)),
]

# ── Print results ──
print(f"{'Phase':<26}  {'µs':>8}  {'ms':>8}")
print("-" * 50)
results = {}
for name, fn in phases:
    t_us = timed(fn)
    results[name] = t_us
    print(f"{name}  {t_us:8.1f}  {t_us/1000:8.3f}")

total = results["TOTAL (end-to-end)"]
print("-" * 50)
print(f"\n{'Phase':<26}  {'%':>6}")
for name, fn in phases[:-1]:
    print(f"{name}  {results[name]/total*100:6.1f}%")

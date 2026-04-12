"""Generic profiling target for nsys — indexer workload.

Environment variables:
  IMPL_MODULE   Python module path of the implementation (default: src.kernels.idxer_tc_nvtx)
  WORKLOAD_IDX  0-based workload index (default: 0)
  CONTEST_DIR   Path to the contest data directory (default: /data)
  WARMUP        Number of warmup runs before the profiled pass (default: 3)
  USE_NVTX      Set to 1 to wrap the profiled run with cudaProfilerApi (default: 0)
"""
import os, sys, json
from pathlib import Path

for p in ["/app", "/app/dev"]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from importlib import import_module
from safetensors.torch import load_file

# ── Config from env ──
WORKLOAD_IDX = int(os.environ.get("WORKLOAD_IDX", "0"))
IMPL_MODULE  = os.environ.get("IMPL_MODULE", "src.kernels.idxer_tc_nvtx")
CONTEST      = Path(os.environ.get("CONTEST_DIR", "/data"))
WARMUP       = int(os.environ.get("WARMUP", "3"))
USE_NVTX     = int(os.environ.get("USE_NVTX", "0"))

# ── Model constants ──
NUM_HEADS  = 64
HEAD_DIM   = 128
TOPK       = 2048
PAGE_SIZE  = 64

JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"

# ── Load workload ──
workloads = [json.loads(l) for l in open(JSONL)]
w   = workloads[WORKLOAD_IDX]
ax  = w["workload"]["axes"]
inp = w["workload"]["inputs"]
batch_size     = ax["batch_size"]
num_pages      = ax["num_pages"]
max_num_pages  = ax["max_num_pages"]
uuid_short     = w["workload"]["uuid"][:8]
print(f"Workload {WORKLOAD_IDX}: uuid={uuid_short}  B={batch_size}  num_pages={num_pages}  max_pages={max_num_pages}  impl={IMPL_MODULE}")

# ── Tensors ──
q_index_fp8 = torch.randn(batch_size, NUM_HEADS, HEAD_DIM,
                           dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
k_index_cache_fp8 = torch.randint(0, 256,
                                   (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
                                   dtype=torch.uint8, device="cuda").view(torch.int8)
weights = torch.randn(batch_size, NUM_HEADS, dtype=torch.float32, device="cuda")

# Load seq_lens and block_table from safetensors
sf = load_file(str(CONTEST / inp["seq_lens"]["path"]))
seq_lens    = sf[inp["seq_lens"]["tensor_key"]].cuda()
block_table = sf[inp["block_table"]["tensor_key"]].cuda()

topk_indices = torch.full((batch_size, TOPK), -1, dtype=torch.int32, device="cuda")

# ── Load impl ──
impl = import_module(IMPL_MODULE)
run  = impl.run

# Trigger JIT compilation
run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices, is_profiling=False)
torch.cuda.synchronize()

# ── Warmup (no profiling) ──
for i in range(WARMUP):
    print(f"  warmup {i+1}/{WARMUP}")
    run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices, is_profiling=False)
    torch.cuda.synchronize()

# ── Profiled run ──
if USE_NVTX:
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(IMPL_MODULE)

print("  profiled run (is_profiling=True)")
run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices, is_profiling=True)
torch.cuda.synchronize()

if USE_NVTX:
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

print("Done.")

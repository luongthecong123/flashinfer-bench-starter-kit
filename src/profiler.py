"""Generic profiling target for ncu / nsys.

Environment variables:
  IMPL_MODULE   Python module path of the implementation (default: src.gather_impl)
  WORKLOAD_IDX  0-based workload index (default: 0)
  CONTEST_DIR   Path to the contest data directory (default: /data)
  WARMUP        Number of warmup runs before the profiled pass (default: 0, ncu handles replay)
  USE_NVTX      Set to 1 to wrap the profiled run in a CUDA profiler / NVTX range (default: 0)
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
IMPL_MODULE  = os.environ.get("IMPL_MODULE", "src.gather_impl")
CONTEST      = Path(os.environ.get("CONTEST_DIR", "/data"))
WARMUP       = int(os.environ.get("WARMUP", "0"))
USE_NVTX     = int(os.environ.get("USE_NVTX", "0"))

# ── Model constants ──
H, D, Dp, TOPK, PS = 16, 512, 64, 2048, 64
SCALE = 0.1352337788608801

JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

# ── Load workload ──
workloads = [json.loads(l) for l in open(JSONL)]
w   = workloads[WORKLOAD_IDX]
ax  = w["workload"]["axes"]
inp = w["workload"]["inputs"]
T, P = ax["num_tokens"], ax["num_pages"]
print(f"Workload {WORKLOAD_IDX}: uuid={w['workload']['uuid'][:8]}  T={T}  P={P}  impl={IMPL_MODULE}")

# ── Tensors ──
q_nope = torch.randn(T, H, D,  dtype=torch.bfloat16, device="cuda")
q_pe   = torch.randn(T, H, Dp, dtype=torch.bfloat16, device="cuda")
ckv    = torch.randn(P, PS, D,  dtype=torch.bfloat16, device="cuda")
kpe    = torch.randn(P, PS, Dp, dtype=torch.bfloat16, device="cuda")

sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

output = torch.zeros(T, H, D,  dtype=torch.bfloat16, device="cuda")
lse    = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

# ── Load impl ──
impl = import_module(IMPL_MODULE)
run  = impl.run

# Trigger JIT compilation (ncu replays from here)
run(q_nope, q_pe, ckv, kpe, si, SCALE, output, lse)
torch.cuda.synchronize()

# ── Warmup ──
for _ in range(WARMUP):
    run(q_nope, q_pe, ckv, kpe, si, SCALE, output, lse)
    torch.cuda.synchronize()

# ── Profiled run ──
if USE_NVTX:
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(IMPL_MODULE)

run(q_nope, q_pe, ckv, kpe, si, SCALE, output, lse)
torch.cuda.synchronize()

if USE_NVTX:
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

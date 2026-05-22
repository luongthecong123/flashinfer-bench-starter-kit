"""Standalone profiling script for gather_dsa_impl (CuTe gather + CuTe fused DSA attention)."""
import sys, os
# Ensure /app is on PYTHONPATH so 'zen.*' imports work when run as subprocess
for p in ["/app", "/app/zen", "/app/dev"]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import math

def main():
    T = 8
    P = 512
    H = 16
    D_ckv = 512
    D_kpe = 64
    PAGE_SIZE = 64
    TOPK = 2048
    sm_scale = 1.0 / math.sqrt(D_ckv + D_kpe)

    q_nope = torch.randn(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn(T, H, D_kpe, dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn(P, PAGE_SIZE, D_ckv, dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn(P, PAGE_SIZE, D_kpe, dtype=torch.bfloat16, device="cuda")

    total_kv = P * PAGE_SIZE
    sparse_indices = torch.randint(0, total_kv, (T, TOPK), dtype=torch.int32, device="cuda")

    output = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

    from zen.gather_dsa_impl import run

    # Warmup
    for _ in range(5):
        output.zero_()
        lse.fill_(-float("inf"))
        run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)
        torch.cuda.synchronize()

    # Profiled run
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("gather_dsa_impl_run")
    output.zero_()
    lse.fill_(-float("inf"))
    run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    main()

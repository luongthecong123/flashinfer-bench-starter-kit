"""Standalone profiling script for dev2/impl.py with synthetic data."""
import torch
import math
from impl import run

def main():
    # Typical workload dimensions
    T = 8          # num_tokens
    P = 512        # num_pages
    H = 16         # num_qo_heads
    D_ckv = 512    # head_dim_ckv
    D_kpe = 64     # head_dim_kpe
    PAGE_SIZE = 64
    TOPK = 2048
    sm_scale = 1.0 / math.sqrt(D_ckv + D_kpe)

    # Generate synthetic inputs
    q_nope = torch.randn(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn(T, H, D_kpe, dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn(P, PAGE_SIZE, D_ckv, dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn(P, PAGE_SIZE, D_kpe, dtype=torch.bfloat16, device="cuda")

    total_kv = P * PAGE_SIZE
    sparse_indices = torch.randint(0, total_kv, (T, TOPK), dtype=torch.int32, device="cuda")

    output = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

    # Single run — NCU handles replay internally
    run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)

if __name__ == "__main__":
    main()

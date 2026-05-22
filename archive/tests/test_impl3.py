#!/usr/bin/env python3
"""Quick local correctness check for impl3 (32-warp + precomputed max_valid)."""
import sys, os
# dev/ must come first so we import dev/cook.py, not zen/cook.py
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'zen'))

import torch
from impl3 import fused_dsa_v3_compiled
from gather import gather_compiled

D_ckv, D_kpe, TOPK = 512, 64, 2048

def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    T = q_nope.shape[0]
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])

    # Run gather to get kc, Kp, max_valid
    Kc = torch.empty(T, TOPK, D_ckv, dtype=torch.bfloat16, device="cuda")
    Kp = torch.empty(T, TOPK, D_kpe, dtype=torch.bfloat16, device="cuda")
    max_valid = torch.zeros(T, dtype=torch.int32, device="cuda")
    gather_compiled(ckv_flat, kpe_flat, sparse_indices, Kc, Kp, max_valid)
    torch.cuda.synchronize()

    fused_dsa_v3_compiled(q_nope, q_pe, Kc, Kp, max_valid, output, lse)

import cook
cook.impl_fn = run
cook.CHECK = True
cook.MEASURE = False
cook.TOY_CHECK = False
cook.main()

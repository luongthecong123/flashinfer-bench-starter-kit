#!/usr/bin/env python3
"""Quick local correctness check for impl2 (32-warp parallel-keys letmecook)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from impl2 import fused_dsa_v2_compiled
import cook

def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[-1])
    kpe_flat = kpe_cache.reshape(-1, kpe_cache.shape[-1])
    fused_dsa_v2_compiled(q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices, output, lse)

cook.impl_fn = run
cook.CHECK = True
cook.MEASURE = False
cook.TOY_CHECK = False
cook.main()

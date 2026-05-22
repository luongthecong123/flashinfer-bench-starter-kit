"""DSA sparse attention using CuTe DSL gather kernel + CuTe DSL fused attention."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from zen.gather import gather_compiled
from zen.dsa import fused_dsa_compiled


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    topk = sparse_indices.shape[-1]
    T = num_tokens

    # Flatten paged KV cache: [num_pages, 64, D] → [N, D]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)

    # CuTe DSL gather kernel
    Kc = torch.empty(T, topk, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
    Kp = torch.empty(T, topk, head_dim_kpe, dtype=torch.bfloat16, device="cuda")
    max_valid = torch.empty(T, dtype=torch.int32, device="cuda")
    gather_compiled(Kc_all, Kp_all, sparse_indices, Kc, Kp, max_valid)

    # CuTe DSL fused attention kernel
    sm_scale_tensor = torch.tensor([sm_scale], dtype=torch.float32, device="cuda")
    fused_dsa_compiled(q_nope, q_pe, Kc, Kp, sparse_indices, max_valid, sm_scale_tensor, output, lse)

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dev"))
    import cook
    cook.impl_fn = run
    cook.CHECK = True
    cook.MEASURE = False
    cook.TOY_CHECK = False
    cook.main()

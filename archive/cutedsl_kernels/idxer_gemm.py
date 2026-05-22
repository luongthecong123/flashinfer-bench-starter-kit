"""idxer_gemm.py — stripped-down core score kernel.

Focuses purely on:
  fp8 dequant  →  GEMM  →  ReLU  →  weighted reduction over 64 heads

Inputs:
  q_fp8    : [64, 128]   float8_e4m3fn  (query)
  K_fp8    : [2048, 128] float8_e4m3fn  (keys, already gathered)
  K_scales : [2048]      float32        (per-token scale)
  weights  : [64]        float32

Output:
  scores   : [2048]  float32

seq_len is fixed at 2048 for easy early kernel development.
"""
import torch

SEQ_LEN   = 2048
NUM_HEADS = 64
HEAD_DIM  = 128


def compute_scores(q_fp8: torch.Tensor,
                   K_fp8: torch.Tensor,
                   K_scales: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
    """
    q_fp8    : [64, 128]   float8_e4m3fn
    K_fp8    : [2048, 128] float8_e4m3fn
    K_scales : [2048]      float32
    weights  : [64]        float32
    returns  : [2048]      float32
    """
    # fp8 → fp32
    q = q_fp8.to(torch.float32)                         # [64, 128]
    K = K_fp8.to(torch.float32) * K_scales[:, None]     # [2048, 128]
    # GEMM: [64, 128] @ [128, 2048] -> [64, 2048]
    scores = torch.mm(q, K.T)
    # ReLU
    scores = torch.relu(scores)
    # weighted reduction over 64 heads -> [2048]
    return (scores * weights[:, None]).sum(dim=0)


# ── self-contained correctness check ──────────────────────────────────────────
if __name__ == "__main__":
    import sys
    device = "cuda"

    # Use randn → to(fp8) so values are valid (no NaN bit patterns)
    q_fp8    = torch.randn(NUM_HEADS, HEAD_DIM,  device=device).to(torch.float8_e4m3fn)
    K_fp8    = torch.randn(SEQ_LEN,   HEAD_DIM,  device=device).to(torch.float8_e4m3fn)
    K_scales = torch.rand (SEQ_LEN,              device=device) + 0.5   # positive scales
    weights  = torch.randn(NUM_HEADS,            device=device)

    # ref: inline pytorch (ground truth, same math)
    q_f32 = q_fp8.to(torch.float32)
    K_f32 = K_fp8.to(torch.float32) * K_scales[:, None]
    ref   = torch.mm(q_f32, K_f32.T)
    ref   = torch.relu(ref)
    ref   = (ref * weights[:, None]).sum(dim=0)

    impl = compute_scores(q_fp8, K_fp8, K_scales, weights)

    torch.cuda.synchronize()
    max_diff = (ref - impl).abs().max().item()
    ok = max_diff == 0.0
    print(f"seq_len={SEQ_LEN}  max_diff={max_diff:.2e}  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

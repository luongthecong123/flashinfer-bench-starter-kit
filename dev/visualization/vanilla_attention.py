"""Vanilla Multi-Head Self-Attention — two modes for comparison.

Dimensions (for diagram):
  B = 1 (batch, omitted from diagram for clarity)
  H = 16 (attention heads)
  D = 64 (head dimension)
  D_model = H * D = 1024 (model dimension / embedding dim)
  S = 1024 (sequence length / KV cache length)

Case 1 — Naive (no KV cache):
  Full self-attention over the entire sequence.  Used during training or
  non-optimized inference.  Every token attends to all previous tokens.
  Prefill cost: O(S² · H · D) for the attention matmuls.

Case 2 — With KV cache (efficient inference):
  a) Prefill stage: process full prompt, build KV cache.
     Same compute as naive, but we SAVE K,V into cache.
  b) Decode stage: process ONE new token at a time.
     Only compute Q for the new token, attend to full KV cache.
     Cost per decode step: O(S · H · D) — linear in S, not S².
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def naive_self_attention(X: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                         Wv: torch.Tensor, Wo: torch.Tensor,
                         H: int = 16, causal: bool = True) -> torch.Tensor:
    """Naive multi-head self-attention — no KV cache.

    Args:
        X:  [B, S, D_model]  input embeddings
        Wq: [D_model, D_model]  query projection
        Wk: [D_model, D_model]  key projection
        Wv: [D_model, D_model]  value projection
        Wo: [D_model, D_model]  output projection
        H:  number of attention heads
        causal: whether to apply causal mask

    Returns:
        output: [B, S, D_model]
    """
    B, S, D_model = X.shape
    D = D_model // H  # head dim

    # Linear projections: [B, S, D_model] @ [D_model, D_model] → [B, S, D_model]
    Q = (X @ Wq).reshape(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
    K = (X @ Wk).reshape(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
    V = (X @ Wv).reshape(B, S, H, D).transpose(1, 2)  # [B, H, S, D]

    # Attention scores: [B, H, S, D] @ [B, H, D, S] → [B, H, S, S]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

    # Causal mask: prevent attending to future tokens
    if causal:
        mask = torch.triu(torch.ones(S, S, device=X.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn = F.softmax(scores, dim=-1)  # [B, H, S, S]

    # Output: [B, H, S, S] @ [B, H, S, D] → [B, H, S, D]
    out = torch.matmul(attn, V)
    out = out.transpose(1, 2).reshape(B, S, D_model)  # [B, S, D_model]

    # Output projection
    return out @ Wo


class KVCache:
    """Simple KV cache for efficient autoregressive inference."""

    def __init__(self, B: int, H: int, D: int, max_len: int, device: torch.device):
        self.k_cache = torch.zeros(B, H, max_len, D, device=device)
        self.v_cache = torch.zeros(B, H, max_len, D, device=device)
        self.seq_len = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple:
        """Append new K,V to cache and return full cache."""
        new_len = k.shape[2]
        end = self.seq_len + new_len
        self.k_cache[:, :, self.seq_len:end] = k
        self.v_cache[:, :, self.seq_len:end] = v
        self.seq_len = end
        return self.k_cache[:, :, :end], self.v_cache[:, :, :end]


@torch.no_grad()
def prefill_with_kv_cache(X: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                          Wv: torch.Tensor, Wo: torch.Tensor,
                          kv_cache: KVCache, H: int = 16) -> torch.Tensor:
    """Prefill stage: process full prompt, populate KV cache.

    Args:
        X:  [B, S, D_model]  full prompt embeddings
        kv_cache: KVCache to populate
        (other args same as naive)

    Returns:
        output: [B, S, D_model]
    """
    B, S, D_model = X.shape
    D = D_model // H

    Q = (X @ Wq).reshape(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
    K = (X @ Wk).reshape(B, S, H, D).transpose(1, 2)  # [B, H, S, D]
    V = (X @ Wv).reshape(B, S, H, D).transpose(1, 2)  # [B, H, S, D]

    # Store K, V into cache
    K_full, V_full = kv_cache.update(K, V)  # [B, H, S, D]

    # Attention (causal)
    scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(D)
    mask = torch.triu(torch.ones(S, S, device=X.device, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn = F.softmax(scores, dim=-1)  # [B, H, S, S]

    out = torch.matmul(attn, V_full)
    out = out.transpose(1, 2).reshape(B, S, D_model)
    return out @ Wo


@torch.no_grad()
def decode_with_kv_cache(x: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor,
                         Wv: torch.Tensor, Wo: torch.Tensor,
                         kv_cache: KVCache, H: int = 16) -> torch.Tensor:
    """Decode stage: process ONE new token, attend to full KV cache.

    Args:
        x:  [B, 1, D_model]  single new token embedding
        kv_cache: KVCache with previous K,V already stored
        (other args same as naive)

    Returns:
        output: [B, 1, D_model]
    """
    B, T, D_model = x.shape  # T = 1
    D = D_model // H

    q = (x @ Wq).reshape(B, T, H, D).transpose(1, 2)   # [B, H, 1, D]
    k = (x @ Wk).reshape(B, T, H, D).transpose(1, 2)    # [B, H, 1, D]
    v = (x @ Wv).reshape(B, T, H, D).transpose(1, 2)    # [B, H, 1, D]

    # Append to cache → K_full [B, H, S+1, D], V_full [B, H, S+1, D]
    K_full, V_full = kv_cache.update(k, v)
    S_total = K_full.shape[2]

    # Attention: q [B,H,1,D] @ K_full.T [B,H,D,S+1] → [B,H,1,S+1]
    scores = torch.matmul(q, K_full.transpose(-2, -1)) / math.sqrt(D)
    # No causal mask needed: single query can attend to all previous tokens
    attn = F.softmax(scores, dim=-1)  # [B, H, 1, S+1]

    # Output: [B,H,1,S+1] @ V_full [B,H,S+1,D] → [B,H,1,D]
    out = torch.matmul(attn, V_full)
    out = out.transpose(1, 2).reshape(B, T, D_model)
    return out @ Wo


# ── Quick test ──
if __name__ == "__main__":
    B, S, H, D = 1, 1024, 16, 64
    D_model = H * D  # 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.randn(B, S, D_model, device=device)
    Wq = torch.randn(D_model, D_model, device=device) * 0.01
    Wk = torch.randn(D_model, D_model, device=device) * 0.01
    Wv = torch.randn(D_model, D_model, device=device) * 0.01
    Wo = torch.randn(D_model, D_model, device=device) * 0.01

    # Case 1: Naive
    out_naive = naive_self_attention(X, Wq, Wk, Wv, Wo, H)
    print(f"Naive output:   {out_naive.shape}")  # [1, 1024, 1024]

    # Case 2: With KV cache
    # Prefill
    cache = KVCache(B, H, D, max_len=S + 10, device=device)
    out_prefill = prefill_with_kv_cache(X, Wq, Wk, Wv, Wo, cache, H)
    print(f"Prefill output: {out_prefill.shape}")  # [1, 1024, 1024]

    # Decode (one new token)
    x_new = torch.randn(B, 1, D_model, device=device)
    out_decode = decode_with_kv_cache(x_new, Wq, Wk, Wv, Wo, cache, H)
    print(f"Decode output:  {out_decode.shape}")   # [1, 1, 1024]

    # Verify naive == prefill for same input
    diff = (out_naive - out_prefill).abs().max().item()
    print(f"Naive vs Prefill max diff: {diff:.6e}")
    print("PASS" if diff < 1e-3 else "FAIL")

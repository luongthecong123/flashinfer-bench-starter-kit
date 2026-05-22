"""Modal test: verify gemv_kvsplit_v1 against PyTorch on B200.

Usage:
    modal run src/modal/gemv_kvsplit_test.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200", timeout=300)
def test_gemv_kvsplit():
    import torch
    from src.kernels.gemv_kvsplit_v1 import run, K, D

    torch.manual_seed(42)
    logit  = torch.randn(K,    dtype=torch.float32,  device="cuda")
    V      = torch.randn(K, D, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(D,    dtype=torch.float32,  device="cuda")

    # Reference: output[d] = sum_k logit[k] * V[k, d]
    # = (V.float().T @ logit) = [D, K] @ [K] = [D]
    ref = torch.mv(V.float().T.contiguous(), logit)

    run(logit, V, output)
    torch.cuda.synchronize()

    abs_diff = (output - ref).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    rel_diff  = (abs_diff / (ref.abs() + 1e-6)).max().item()

    print(f"K={K}, D={D}, CLUSTER_N=4, K_PER_SPLIT=512")
    print(f"max |output - ref| = {max_diff:.6f}")
    print(f"mean|output - ref| = {mean_diff:.6f}")
    print(f"max rel diff       = {rel_diff:.6f}")
    print(f"output[:8] = {output[:8].tolist()}")
    print(f"ref   [:8] = {ref[:8].tolist()}")

    assert max_diff < 0.05, f"FAIL: max_diff={max_diff:.4f} too large"
    print("PASS ✓")
    return {"max_diff": max_diff, "mean_diff": mean_diff}


@app.local_entrypoint()
def main():
    result = test_gemv_kvsplit.remote()
    print(f"\nResult: {result}")

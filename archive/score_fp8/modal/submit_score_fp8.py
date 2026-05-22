"""submit_score_fp8.py — correctness test for score_tcgen05_fp8 GEMM on B200.

Runs ScoreGEMMFP8 on random fp8 inputs and checks output against
K_fp8.float() @ q_fp8.float().T (reference GEMM in float32).

Usage:
    modal run src/modal/submit_score_fp8.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_correctness():
    import sys
    import torch
    sys.path.insert(0, "/app")

    from cutlass.cute.runtime import from_dlpack
    import cutlass.cute as cute
    from cutlass.cute.testing import benchmark, JitArguments
    from src.kernels.score_tcgen05_fp8 import ScoreGEMMFP8, M, N, K

    device = "cuda"
    NUM_TRIALS = 5

    print(f"Testing ScoreGEMMFP8: M={M}, N={N}, K={K}")
    print("=" * 60)

    all_pass = True
    for trial in range(NUM_TRIALS):
        K_fp8 = torch.randn(M, K, device=device).to(torch.float8_e4m3fn)
        q_fp8 = torch.randn(N, K, device=device).to(torch.float8_e4m3fn)
        c_out = torch.zeros((M, N), device=device, dtype=torch.float32)

        kv_ = from_dlpack(K_fp8, assumed_align=16)
        q_  = from_dlpack(q_fp8, assumed_align=16)
        c_  = from_dlpack(c_out, assumed_align=16)

        gemm     = ScoreGEMMFP8()
        compiled = cute.compile(gemm, kv_, q_, c_)
        compiled(kv_, q_, c_)

        # Reference: float32 GEMM
        ref_c = K_fp8.float() @ q_fp8.float().T   # [2048, 64]

        atol, rtol = 1.0, 0.5
        match   = torch.allclose(c_out, ref_c, atol=atol, rtol=rtol)
        max_err = (c_out - ref_c).abs().max().item()
        status  = "PASS" if match else "FAIL"
        print(f"Trial {trial}: {status}  max_err={max_err:.4f}")
        if not match:
            all_pass = False
            print(f"  c_out[0,:8]: {c_out[0, :8].tolist()}")
            print(f"  ref_c[0,:8]: {ref_c[0, :8].tolist()}")

    # Benchmark
    t = benchmark(compiled, kernel_arguments=JitArguments(kv_, q_, c_))
    print(f"\nDURATION: {t:.4f} µs")
    print(f"\nOVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass


@app.local_entrypoint()
def main():
    ok = run_correctness.remote()
    if not ok:
        raise SystemExit(1)

"""submit_score_fp8_direct.py — correctness test for score_tcgen05_fp8_direct on B200."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_correctness():
    import sys, torch
    sys.path.insert(0, "/app")
    from cutlass.cute.runtime import from_dlpack
    import cutlass.cute as cute
    from cutlass.cute.testing import benchmark, JitArguments
    from src.kernels.score_tcgen05_fp8_direct import ScoreGEMMFP8Direct, M, N, K

    device = "cuda"
    all_pass = True
    for trial in range(5):
        K_fp8 = torch.randn(M, K, device=device).to(torch.float8_e4m3fn)
        q_fp8 = torch.randn(N, K, device=device).to(torch.float8_e4m3fn)
        c_out = torch.zeros((M, N), device=device, dtype=torch.float32)
        kv_ = from_dlpack(K_fp8, assumed_align=16)
        q_  = from_dlpack(q_fp8, assumed_align=16)
        c_  = from_dlpack(c_out, assumed_align=16)
        gemm = ScoreGEMMFP8Direct()
        compiled = cute.compile(gemm, kv_, q_, c_)
        compiled(kv_, q_, c_)
        ref_c = K_fp8.float() @ q_fp8.float().T
        match = torch.allclose(c_out, ref_c, atol=1.0, rtol=0.5)
        max_err = (c_out - ref_c).abs().max().item()
        print(f"Trial {trial}: {'PASS' if match else 'FAIL'}  max_err={max_err:.4f}")
        if not match:
            all_pass = False
            print(f"  c_out[0,:8]: {c_out[0, :8].tolist()}")
            print(f"  ref_c[0,:8]: {ref_c[0, :8].tolist()}")
    t = benchmark(compiled, kernel_arguments=JitArguments(kv_, q_, c_))
    print(f"DURATION: {t:.4f} us")
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


@app.local_entrypoint()
def main():
    ok = run_correctness.remote()
    if not ok:
        raise SystemExit(1)

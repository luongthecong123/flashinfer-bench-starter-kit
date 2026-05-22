"""submit_score_scale_full_page.py — correctness test for score_scale_full_page.py on B200."""
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
    from src.kernels.score_scale_full_page import (
        ScoreScaleFullPage, M, N, HEAD_DIM, ROW_STRIDE, K_PAGES, PAGE_SIZE,
    )

    device = "cuda"
    all_pass = True
    compiled = None

    for trial in range(5):
        K_fp8    = torch.randn(M, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        K_scales = torch.rand(M, device=device, dtype=torch.float32) + 0.5
        q_fp8    = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        w        = torch.randn(N, device=device, dtype=torch.float32)

        # Paged kv tensor [K_pages, PAGE_SIZE, ROW_STRIDE]
        kv_raw = torch.zeros(K_PAGES, PAGE_SIZE, ROW_STRIDE, device=device, dtype=torch.uint8)
        kv_raw[:, :, :HEAD_DIM] = K_fp8.view(torch.uint8).reshape(K_PAGES, PAGE_SIZE, HEAD_DIM)
        kv_raw[:, :, HEAD_DIM:HEAD_DIM + 4] = (
            K_scales.view(torch.uint8).reshape(K_PAGES, PAGE_SIZE, 4)
        )
        c_out = torch.zeros(M, device=device, dtype=torch.float32)

        kv_raw_ = from_dlpack(kv_raw, assumed_align=16)
        q_      = from_dlpack(q_fp8,  assumed_align=16)
        w_      = from_dlpack(w,      assumed_align=16)
        c_      = from_dlpack(c_out,  assumed_align=16)

        k = ScoreScaleFullPage()
        compiled = cute.compile(k, kv_raw_, q_, w_, c_)
        compiled(kv_raw_, q_, w_, c_)

        scores  = (K_fp8.float() @ q_fp8.float().T) * K_scales[:, None]
        ref_out = (torch.relu(scores) @ w)

        match   = torch.allclose(c_out, ref_out, atol=1.0, rtol=0.5)
        max_err = (c_out - ref_out).abs().max().item()
        print(f"Trial {trial}: {'PASS' if match else 'FAIL'}  max_err={max_err:.4f}")
        if not match:
            all_pass = False
            print(f"  c_out[:4]:   {c_out[:4].tolist()}")
            print(f"  ref_out[:4]: {ref_out[:4].tolist()}")

    t = benchmark(compiled, kernel_arguments=JitArguments(kv_raw_, q_, w_, c_))
    print(f"DURATION: {t:.4f} us")
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


@app.local_entrypoint()
def main():
    ok = run_correctness.remote()
    if not ok:
        raise SystemExit(1)

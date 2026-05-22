"""submit_score_scale.py — correctness test for score_scale.py on B200."""
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
    from src.kernels.score_scale import ScoreScale, M, N, HEAD_DIM, ROW_STRIDE

    device = "cuda"
    all_pass = True
    compiled = None

    for trial in range(5):
        # Clamp to fp8 e4m3fn safe range (±100 << ±448) to avoid 0x7F NaN pattern
        K_fp8    = torch.randn(M, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        K_scales = torch.rand(M, device=device, dtype=torch.float32) + 0.5
        q_fp8    = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)

        # Pack into kv_raw [M, 132] uint8: real format (128 fp8 + 4 scale bytes)
        kv_raw = torch.zeros(M, ROW_STRIDE, device=device, dtype=torch.uint8)
        kv_raw[:, :HEAD_DIM] = K_fp8.view(torch.uint8)
        kv_raw[:, HEAD_DIM:HEAD_DIM + 4] = K_scales.view(torch.uint8).reshape(M, 4)
        c_out = torch.zeros((M, N), device=device, dtype=torch.float32)

        kv_raw_ = from_dlpack(kv_raw, assumed_align=16)
        q_      = from_dlpack(q_fp8,  assumed_align=16)
        c_      = from_dlpack(c_out,  assumed_align=16)

        k = ScoreScale()
        compiled = cute.compile(k, kv_raw_, q_, c_)
        compiled(kv_raw_, q_, c_)

        ref_c = (K_fp8.float() @ q_fp8.float().T) * K_scales[:, None]

        # ── Packing verification: kv_raw viewed as float32 ───────────
        kv_as_f32   = kv_raw.view(torch.float32)          # [M, 33] float32 (132//4)
        scales_check = kv_as_f32[:, 32]                   # [M] scale column (col 32 = byte 128)
        pack_err     = (scales_check - K_scales).abs().max().item()
        print(f"  Packing max_err: {pack_err}")

        match   = torch.allclose(c_out, ref_c, atol=1.0, rtol=0.5)
        max_err = (c_out - ref_c).abs().max().item()
        print(f"Trial {trial}: {'PASS' if match else 'FAIL'}  max_err={max_err}")
        if not match:
            all_pass = False
            cout_nan_rows = c_out.isnan().any(dim=1).nonzero().squeeze()  # NaN row indices
            print(f"  c_out NaN count: {c_out.isnan().sum().item()}, NaN rows: {cout_nan_rows.shape}")
            for r in cout_nan_rows[:4].tolist():
                # What bytes are at the expected scale position?
                b = kv_raw[r, HEAD_DIM:HEAD_DIM + 4].tolist()
                fv = kv_as_f32[r, HEAD_DIM // 4].item()
                print(f"  Row {r}: K_scales={K_scales[r].item():.4f}, kv_raw bytes={b}, float32={fv:.4f}")
            print(f"  c_out[0,:4]: {c_out[0, :4].tolist()}")
            print(f"  ref_c[0,:4]: {ref_c[0, :4].tolist()}")

    t = benchmark(compiled, kernel_arguments=JitArguments(kv_raw_, q_, c_))
    print(f"DURATION: {t:.4f} us")
    print(f"OVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


@app.local_entrypoint()
def main():
    ok = run_correctness.remote()
    if not ok:
        raise SystemExit(1)

"""Modal runner: fp8 SIMT GEMM test on B200.

Usage:
    modal run src/modal/submit_fp8_simt.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_test():
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    # ── Constants ──
    BM, BN, K = 128, 64, 128
    THREADS = 128
    NUM_VEC = 4
    K_ITERS = K // NUM_VEC  # 32

    @cute.jit
    def simt_fp8_gemm_jit(mA_i8: cute.Tensor, mB_i8: cute.Tensor, mC: cute.Tensor):
        M = mA_i8.shape[0]
        N = mB_i8.shape[0]
        Kd = mA_i8.shape[1]
        simt_fp8_gemm_kernel(mA_i8, mB_i8, mC, M, N, Kd).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1])

    @cute.kernel
    def simt_fp8_gemm_kernel(
        gA_i8: cute.Tensor, gB_i8: cute.Tensor, gC: cute.Tensor,
        M: int, N: int, Kd: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        num_vec: cutlass.Constexpr = NUM_VEC
        k_iters: cutlass.Constexpr = K_ITERS

        fp8_A_ptr = cute.recast_ptr(gA_i8.iterator, dtype=cutlass.Float8E4M3FN)
        gA = cute.make_tensor(fp8_A_ptr, cute.make_layout((M, Kd), stride=(Kd, 1)))
        fp8_B_ptr = cute.recast_ptr(gB_i8.iterator, dtype=cutlass.Float8E4M3FN)
        gB = cute.make_tensor(fp8_B_ptr, cute.make_layout((N, Kd), stride=(Kd, 1)))

        m = tidx
        if m < M:
            A_row = gA[m, None]
            A_z = cute.zipped_divide(A_row, (num_vec,))
            for n in range(N):
                B_row = gB[n, None]
                B_z = cute.zipped_divide(B_row, (num_vec,))
                acc = cutlass.Float32(0)
                for k4 in range(k_iters):
                    a_frag = A_z[(None, (k4,))].load()
                    b_frag = B_z[(None, (k4,))].load()
                    a_f32 = a_frag.to(cutlass.Float32)
                    b_f32 = b_frag.to(cutlass.Float32)
                    for v in cutlass.range_constexpr(num_vec):
                        acc += a_f32[v] * b_f32[v]
                gC[m, n] = acc

    # ── Test data ──
    torch.manual_seed(42)
    A_f32 = torch.randn(BM, K, device="cuda", dtype=torch.float32).clamp(-240, 240)
    B_f32 = torch.randn(BN, K, device="cuda", dtype=torch.float32).clamp(-240, 240)
    A_fp8 = A_f32.to(torch.float8_e4m3fn)
    B_fp8 = B_f32.to(torch.float8_e4m3fn)
    C_ref = A_fp8.float() @ B_fp8.float().T
    C_out = torch.zeros(BM, BN, device="cuda", dtype=torch.float32)
    A_i8 = A_fp8.view(torch.int8)
    B_i8 = B_fp8.view(torch.int8)

    A_ = from_dlpack(A_i8, assumed_align=16)
    B_ = from_dlpack(B_i8, assumed_align=16)
    C_ = from_dlpack(C_out, assumed_align=16)

    print(f"A: {A_i8.shape} i8, B: {B_i8.shape} i8, C: {C_out.shape} f32")

    compiled = cute.compile(simt_fp8_gemm_jit, A_, B_, C_)
    compiled(A_, B_, C_)

    print(C_out[0, :10])

    diff = (C_out - C_ref).abs()
    max_diff = diff.max().item()
    print(f"max_abs_diff = {max_diff:.6f}")
    if max_diff < 2.0:
        print("CORRECTNESS PASS")
    else:
        print("CORRECTNESS FAIL")
        bad = (diff > 2.0).nonzero(as_tuple=False)[:5]
        for idx in bad:
            m, n = idx[0].item(), idx[1].item()
            print(f"  [{m},{n}] kern={C_out[m,n]:.4f} ref={C_ref[m,n]:.4f}")


@app.local_entrypoint()
def go():
    run_test.remote()

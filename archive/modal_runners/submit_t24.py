"""Synthetic T=24 correctness test: stack a T=8 workload 3 times.
Usage: modal run src/modal/submit_t24.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_xor_pdl_v3_pro_v2_1024T")
print("IMPL_MODULE: ", IMPL_MODULE)


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_t24_test(impl_module: str):
    import sys
    sys.path.insert(0, "/app")
    import torch
    from importlib import import_module

    impl = import_module(impl_module)
    from src.ref import run as ref_run
    from src.utils import make_tensors, alloc_out, SCALE, H, D

    ATOL = 0.01

    # ── Synthetic T=24: 3 × T=8 with varied valid counts ────────────────
    T = 24
    P = 8462  # must match NUM_PAGES compiled into the kernel
    # 3 groups of 8 tokens with varied valid counts
    valid_per_token = (
        [92, 48, 1044, 14, 411, 30, 16, 8]   +   # group 0: from workload 385742b2
        [18, 19, 1002, 31, 11, 316, 24, 2]   +   # group 1: from workload 4c46a94b
        [63, 9, 2048, 212, 11, 25, 6, 50]        # group 2: from workload 02d6ae9c
    )

    q_nope, q_pe, ckv, kpe, si = make_tensors(T, P, valid_per_token)

    r_out, r_lse = alloc_out(T)
    i_out, i_lse = alloc_out(T)

    ref_run(q_nope, q_pe, ckv, kpe, si, SCALE, r_out, r_lse)
    impl.run(q_nope, q_pe, ckv, kpe, si, SCALE, i_out, i_lse)
    torch.cuda.synchronize()

    o_abs = (r_out.float() - i_out.float()).abs().max().item()
    l_abs = (r_lse - i_lse).abs().max().item()

    ok = o_abs < ATOL and l_abs < ATOL
    status = "PASS" if ok else "FAIL"

    print(f"\n=== SYNTHETIC T={T} TEST ===")
    print(f"  valid_per_token: {valid_per_token}")
    print(f"  output abs err:  {o_abs:.6e}")
    print(f"  lse abs err:     {l_abs:.6e}")
    print(f"  Status:          {status}")

    return {"ok": ok, "o_abs": o_abs, "l_abs": l_abs, "T": T}


@app.local_entrypoint()
def main():
    result = run_t24_test.remote(IMPL_MODULE)
    status = "PASS" if result["ok"] else "FAIL"
    print(f"\n>> T={result['T']} synthetic test: {status}  (output_err={result['o_abs']:.6e}, lse_err={result['l_abs']:.6e})")

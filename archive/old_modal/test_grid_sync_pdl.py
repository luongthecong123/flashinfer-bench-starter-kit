"""Modal runner: correctness test for PDL-based two-kernel grid-sync softmax on B200.

Tests test_grid_sync_pdl.py which uses:
  cute.arch.griddepcontrol_launch_dependents()  (end of kernel 1)
  cute.arch.griddepcontrol_wait()               (start of kernel 2)
  use_pdl=True in .launch()

Usage:
    modal run src/modal/test_grid_sync_pdl.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_pdl_test():
    import sys
    sys.path.insert(0, "/app")

    import torch
    from cutlass.cute.runtime import from_dlpack
    import src.kernels.test_grid_sync_pdl as pdl_mod

    # tensors matching the compile-time fakes
    N          = pdl_mod.N
    NUM_SPLITS = pdl_mod.NUM_SPLITS

    x_t           = torch.randn(N, device="cuda", dtype=torch.float32)
    partial_lse_t = torch.empty(NUM_SPLITS, 2, device="cuda", dtype=torch.float32)
    lse_out_t     = torch.empty(1, device="cuda", dtype=torch.float32)

    x_c           = from_dlpack(x_t,           assumed_align=4, enable_tvm_ffi=True)
    partial_lse_c = from_dlpack(partial_lse_t,  assumed_align=4, enable_tvm_ffi=True)
    lse_out_c     = from_dlpack(lse_out_t,      assumed_align=4, enable_tvm_ffi=True)

    pdl_mod._compiled(x_c, partial_lse_c, lse_out_c)
    torch.cuda.synchronize()

    ref = torch.logsumexp(x_t, dim=0)
    our = lse_out_t.item()
    err = abs(our - ref.item())

    result = (
        f"our lse : {our:.6f}\n"
        f"ref lse : {ref.item():.6f}\n"
        f"abs err : {err:.2e}\n"
        f"{'CORRECTNESS PASS' if err < 1e-4 else 'MISMATCH!'}"
    )
    print(result)
    return result


@app.local_entrypoint()
def main():
    print(f"\n{'='*60}\nPDL grid-sync softmax correctness test on B200\n{'='*60}")
    result = run_pdl_test.remote()
    print(result)

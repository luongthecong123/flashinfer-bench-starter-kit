"""submit_swz_fp8.py — Modal runner: test swizzled sA fp8 copy methods.

Each step is a separate function; run them one at a time.

Current status:
  run_step1_reference  ← implemented, ready to test
  run_step2_tma        ← TODO
  run_step3_autovec    ← TODO
  run_step4_cpasync    ← TODO

Usage:
  modal run src/modal/submit_swz_fp8.py          # default: step 1
  modal run src/modal/submit_swz_fp8.py::go      # same
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


# ── Step 1: pure-Python/PyTorch reference ────────────────────────────────────

@app.function(image=image, gpu="B200:1", timeout=120)
def run_step1_reference():
    sys.path.insert(0, "/app")
    from src.kernels.swz_fp8_copy import test_reference
    fp8_src, ref = test_reference()
    import torch
    print(f"\n  fp8_src sum (uint8 view): {fp8_src.cpu().view(torch.uint8).sum().item()}")
    print(f"  ref   sum              : {ref.sum().item()}  (must equal fp8_src sum)")
    print("\n  [Step 1 done]")


@app.local_entrypoint()
def go():
    run_step1_reference.remote()


# ── fp8 sA+sB: TMA reference + autovec comparison ────────────────────────────

@app.function(image=image, gpu="B200:1", timeout=300)
def run_fp8_ab():
    sys.path.insert(0, "/app")
    from src.kernels.swz_fp8_ab_copy import (
        make_fp8_input_a,
        make_fp8_input_b,
        get_tma_reference,
        test_autovec_copy,
    )
    src_a        = make_fp8_input_a()
    src_b        = make_fp8_input_b()
    ref_a, ref_b = get_tma_reference(src_a, src_b)
    test_autovec_copy(src_a, src_b, ref_a, ref_b)


@app.local_entrypoint()
def run_all():
    run_fp8_ab.remote()


# ── Step 2: TMA copy ─────────────────────────────────────────────────────────

@app.function(image=image, gpu="B200:1", timeout=300)
def run_step2_tma():
    sys.path.insert(0, "/app")
    from src.kernels.swz_fp8_copy import test_reference, test_step2_tma
    fp8_src, ref = test_reference()
    test_step2_tma(fp8_src, ref)


# ── Step 3: autovec_copy G→S ─────────────────────────────────────────────────

@app.function(image=image, gpu="B200:1", timeout=300)
def run_step3_autovec():
    sys.path.insert(0, "/app")
    from src.kernels.swz_fp8_copy import test_reference, test_step3_autovec
    fp8_src, ref = test_reference()
    test_step3_autovec(fp8_src, ref)


# ── Step 4: cp.async TV-layout G→S ──────────────────────────────────────────

@app.function(image=image, gpu="B200:1", timeout=300)
def run_step4_cpasync():
    sys.path.insert(0, "/app")
    from src.kernels.swz_fp8_copy import test_reference, test_step4_cpasync
    fp8_src, ref = test_reference()
    test_step4_cpasync(fp8_src, ref)

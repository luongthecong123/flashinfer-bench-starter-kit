"""submit_swz_bf16.py — Modal runner: bf16 swizzle copy comparison.

Usage:
  modal run src/modal/submit_swz_bf16.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=300)
def run_all():
    sys.path.insert(0, "/app")
    from src.kernels.swz_bf16_copy import (
        make_bf16_input_a,
        make_bf16_input_b,
        get_tma_reference,
        test_autovec_copy,
        test_cpasync_copy,
    )
    src_a        = make_bf16_input_a()
    src_b        = make_bf16_input_b()
    ref_a, ref_b = get_tma_reference(src_a, src_b)
    test_autovec_copy(src_a, src_b, ref_a, ref_b)
    test_cpasync_copy(src_a, src_b, ref_a, ref_b)


@app.local_entrypoint()
def go():
    run_all.remote()

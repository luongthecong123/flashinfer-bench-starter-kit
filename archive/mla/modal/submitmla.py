"""Benchmark dense MLA (mla_wmma.py) on Modal B200 for T = 1..8.

No workload JSONL — all inputs are random with fixed shapes.
Usage:
    modal run src/modal/submitmla.py
"""
import os, sys
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def bench_mla():
    import sys, math
    sys.path.insert(0, "/app")

    import torch
    from src.kernels.mla_reduction import (
        run, ref_run,
        NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, SEQ_LEN,
    )
    from src.utils import bench

    SM_SCALE = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)
    device   = "cuda"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SEQ_LEN={SEQ_LEN}  H={NUM_HEADS}  D={HEAD_DIM_CKV}  Dp={HEAD_DIM_KPE}")
    print()
    print(f"{'T':>3}  {'abs_err':>10}  {'Status':>6}  {'ref_ms':>8}  {'impl_ms':>8}  {'Speedup':>8}  {'GFLOPS/s':>10}")
    print("-" * 68)

    # FLOPs per call: T × SEQ_LEN × H × (4D + 2Dp + 5)  (same formula as solution.md)
    FLOPS_PER_TOKEN = NUM_HEADS * (4 * HEAD_DIM_CKV + 2 * HEAD_DIM_KPE + 5)

    results = []
    torch.manual_seed(42)

    for T in range(1, 9):
        q_nope = torch.randn(T, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        q_pe   = torch.randn(T, NUM_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
        kc     = torch.randn(T, SEQ_LEN, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        kp     = torch.randn(T, SEQ_LEN, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
        output = torch.empty(T, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        lse    = torch.empty(T, NUM_HEADS, dtype=torch.float32, device=device)

        # ── Correctness ───────────────────────────────────────────────────────
        run(q_nope, q_pe, kc, kp, SM_SCALE, output, lse)
        torch.cuda.synchronize()
        ref_out, _ = ref_run(q_nope, q_pe, kc, kp, SM_SCALE)
        abs_err = (output.float() - ref_out.float()).abs().max().item()
        status  = "PASS" if abs_err < 1e-1 else "FAIL"

        # ── Timing via utils.bench (L2-flush, per-iter clone, pre-sync) ───────
        ref_ms  = bench(ref_run,  [q_nope, q_pe, kc, kp, SM_SCALE])
        impl_ms = bench(run,      [q_nope, q_pe, kc, kp, SM_SCALE, output, lse])

        speedup  = ref_ms / impl_ms
        flops    = T * SEQ_LEN * FLOPS_PER_TOKEN
        gflops_s = (flops / impl_ms) * 1e-6   # GFLOPs/s

        print(f"{T:>3}  {abs_err:>10.2e}  {status:>6}  {ref_ms:>8.3f}  {impl_ms:>8.3f}  {speedup:>8.2f}x  {gflops_s:>10.1f}")
        results.append(dict(T=T, abs_err=abs_err, status=status,
                            ref_ms=ref_ms, impl_ms=impl_ms,
                            speedup=speedup, gflops_s=gflops_s))

    return results


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path

    data = bench_mla.remote()
    if data is None:
        return

    csv_path = Path("bench_mla_results.csv")
    fields   = ["T", "abs_err", "status", "ref_ms", "impl_ms", "speedup", "gflops_s"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)
    print(f"\nResults written to {csv_path}")

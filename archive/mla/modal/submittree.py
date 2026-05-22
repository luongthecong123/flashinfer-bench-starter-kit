"""Benchmark tree-masked MLA (mla_tree.py) on Modal B200 for T = 1..8.

Usage:
    modal run src/modal/submittree.py
"""
import os, sys
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=600)
def bench_tree():
    import sys, math
    sys.path.insert(0, "/app")

    import torch
    from src.kernels.mla_wmma_tree import (
        run, ref_run,
        NUM_HEADS, HEAD_DIM_CKV, HEAD_DIM_KPE, SEQ_LEN, NUM_DRAFT,
    )
    from src.utils import bench

    SM_SCALE = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)
    device   = "cuda"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SEQ_LEN={SEQ_LEN}  H={NUM_HEADS}  D={HEAD_DIM_CKV}  Dp={HEAD_DIM_KPE}  NUM_DRAFT={NUM_DRAFT}")
    print()
    print(f"{'T':>3}  {'abs_err':>10}  {'Status':>6}  {'ref_ms':>8}  {'impl_ms':>8}  {'Speedup':>8}")
    print("-" * 58)

    results = []
    torch.manual_seed(42)

    # Random 0/1 tree mask — same for all T
    tree_table = torch.randint(0, 2, (SEQ_LEN, SEQ_LEN), dtype=torch.int32, device=device)

    for T in range(1, 9):
        q_nope = torch.randn(T, NUM_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        q_pe   = torch.randn(T, NUM_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
        kc     = torch.randn(T, SEQ_LEN, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        kp     = torch.randn(T, SEQ_LEN, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
        output = torch.empty(T, NUM_HEADS, NUM_DRAFT, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
        lse    = torch.empty(T, NUM_HEADS, NUM_DRAFT, dtype=torch.float32, device=device)

        # ── Correctness ───────────────────────────────────────────────────────
        run(q_nope, q_pe, kc, kp, SM_SCALE, tree_table, output, lse)
        torch.cuda.synchronize()
        ref_out, _ = ref_run(q_nope, q_pe, kc, kp, SM_SCALE, tree_table)
        abs_err = (output.float() - ref_out.float()).abs().max().item()
        status  = "PASS" if abs_err < 1e-1 else "FAIL"

        # ── Timing ────────────────────────────────────────────────────────────
        ref_ms  = bench(ref_run,  [q_nope, q_pe, kc, kp, SM_SCALE, tree_table])
        impl_ms = bench(run,      [q_nope, q_pe, kc, kp, SM_SCALE, tree_table, output, lse])

        speedup = ref_ms / impl_ms

        print(f"{T:>3}  {abs_err:>10.2e}  {status:>6}  {ref_ms:>8.3f}  {impl_ms:>8.3f}  {speedup:>8.2f}x")
        results.append(dict(T=T, abs_err=abs_err, status=status,
                            ref_ms=ref_ms, impl_ms=impl_ms, speedup=speedup))

    return results


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path

    data = bench_tree.remote()
    if data is None:
        return

    csv_path = Path("bench_tree_results.csv")
    fields   = ["T", "abs_err", "status", "ref_ms", "impl_ms", "speedup"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)
    print(f"\nResults written to {csv_path}")

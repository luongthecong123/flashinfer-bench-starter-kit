"""submit_score_scale_full_bt.py — correctness test for score_scale_full_bt.py on B200.

Tests several real workload cases from solution_idxer.md:
  - Contiguous: WL 14 (pg=34, contiguous from page 3)
  - Small scatter (1 gap): WL 21 (pg=35, [3..36, 38])
  - Multi-gap small: WL 25 (pg=36, [3..36, 38, 42])
  - Backwards-jump (large case): WL 64 (pg=82, includes 64→25→18 backwards)
  - Long-tail scatter: WL 70 (pg=89, starts with 7 then jumps to 65..152)

Pads odd num_pg to even (the extra garbage rows are filtered downstream by seq_len mask).
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


# ── Real block_tables extracted from contest safetensors (see solution_idxer.md) ──
WORKLOAD_CASES = [
    # (label, seq_len, block_table)
    ("WL 14 contig pg=34",
     2161,
     list(range(3, 37))),

    ("WL 21 1-gap pg=35",
     2177,
     list(range(3, 37)) + [38]),

    ("WL 25 2-gap pg=36",
     2241,
     list(range(3, 37)) + [38, 42]),

    ("WL 64 backwards-jump pg=82",
     5194,
     list(range(44, 65)) + [25, 18] + list(range(65, 95)) + [42, 33] + list(range(95, 122))),

    ("WL 70 long-tail pg=89",
     5679,
     [7] + list(range(65, 153))),
]

NUM_PAGES_POOL = 11923   # match contest's k_index_cache_fp8 first dim


@app.function(image=image, gpu="B200:1", timeout=600)
def run_correctness():
    import sys, math, torch
    sys.path.insert(0, "/app")
    from cutlass.cute.runtime import from_dlpack
    import cutlass.cute as cute
    from cutlass.cute.testing import benchmark, JitArguments
    from src.kernels.score_scale_full_bt import (
        ScoreScaleFullBT, PAGE_SIZE, N, HEAD_DIM, ROW_STRIDE,
    )

    device = "cuda"
    all_pass = True
    cache = {}   # num_pg → compiled kernel

    for label, seq_len, bt_list in WORKLOAD_CASES:
        torch.manual_seed(len(bt_list))

        num_pg_real = len(bt_list)
        # Pad to even number of pages (extra garbage page id = 0)
        num_pg = num_pg_real if num_pg_real % 2 == 0 else num_pg_real + 1
        if num_pg != num_pg_real:
            bt_padded = bt_list + [0]
        else:
            bt_padded = bt_list
        M = num_pg * PAGE_SIZE

        print(f"\n── {label}  (seq_len={seq_len}, num_pg_real={num_pg_real}, padded={num_pg}, M={M}) ──")

        # Build per-used-page random kv data
        K_fp8_used    = torch.randn(num_pg_real, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        K_scales_used = (torch.rand(num_pg_real, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5)

        # Pack into a fresh global pool
        kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE, device=device, dtype=torch.uint8)
        for i, pid in enumerate(bt_list):
            kv_pool[pid, :, :HEAD_DIM] = K_fp8_used[i].view(torch.uint8)
            kv_pool[pid, :, HEAD_DIM:HEAD_DIM + 4] = (
                K_scales_used[i].view(torch.uint8).reshape(PAGE_SIZE, 4)
            )

        block_table = torch.tensor(bt_padded, dtype=torch.int32, device=device)
        q_fp8 = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        w     = torch.randn(N, device=device, dtype=torch.float32)
        c_out = torch.zeros(M, device=device, dtype=torch.float32)

        kv_pool_ = from_dlpack(kv_pool, assumed_align=16)
        bt_      = from_dlpack(block_table, assumed_align=4)
        q_       = from_dlpack(q_fp8, assumed_align=16)
        w_       = from_dlpack(w, assumed_align=16)
        c_       = from_dlpack(c_out, assumed_align=16)

        # Compile per unique num_pg (shape-specialized)
        if num_pg not in cache:
            k = ScoreScaleFullBT(num_pg=num_pg)
            cache[num_pg] = cute.compile(k, kv_pool_, bt_, q_, w_, c_)
        compiled = cache[num_pg]
        compiled(kv_pool_, bt_, q_, w_, c_)

        # Reference: K is per-page data flattened in block_table order, then garbage pad
        K_ref_used = K_fp8_used.reshape(num_pg_real * PAGE_SIZE, HEAD_DIM).float()
        K_sc_used  = K_scales_used.reshape(num_pg_real * PAGE_SIZE)
        scores_used = (K_ref_used @ q_fp8.float().T) * K_sc_used[:, None]   # [M_real, N]
        ref_used    = torch.relu(scores_used) @ w                            # [M_real]

        # Compare only the first M_real rows (the rest are garbage / page 0 reads)
        M_real = num_pg_real * PAGE_SIZE
        c_real = c_out[:M_real]

        match   = torch.allclose(c_real, ref_used, atol=1.0, rtol=0.5)
        max_err = (c_real - ref_used).abs().max().item()
        print(f"  → {'PASS' if match else 'FAIL'}  max_err={max_err:.4f}  (compared first {M_real} rows)")
        if not match:
            all_pass = False
            mism = (c_real - ref_used).abs() > (1.0 + 0.5 * ref_used.abs())
            n_bad = mism.sum().item()
            bad_idx = mism.nonzero(as_tuple=True)[0][:5].tolist()
            print(f"    {n_bad} bad rows; first idx: {bad_idx}")
            for i in bad_idx[:3]:
                print(f"    row {i}: got={c_real[i].item():.3f} ref={ref_used[i].item():.3f}")

        # Benchmark
        t = benchmark(compiled, kernel_arguments=JitArguments(kv_pool_, bt_, q_, w_, c_))
        print(f"  duration: {t:.4f} us")

    print(f"\n{'='*60}\nOVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


@app.local_entrypoint()
def main():
    ok = run_correctness.remote()
    if not ok:
        raise SystemExit(1)

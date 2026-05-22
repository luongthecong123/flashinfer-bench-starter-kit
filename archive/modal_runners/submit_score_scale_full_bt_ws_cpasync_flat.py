"""submit_score_scale_full_bt_ws_cpasync_flat — correctness + bench for the
FLAT-layout cpasync kernel.

Per-page byte layout (contest layout):
  bytes [0    .. 8192) : packed fp8  (64 tokens × 128 dims)
  bytes [8192 .. 8448) : packed fp32 scales (64 tokens × 4 B)

Same workload cases as submit_score_scale_full_bt_ws_cpasync.py.
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


WORKLOAD_CASES = [
    ("WL 14 contig pg=34",          2161, list(range(3, 37))),
    ("WL 21 1-gap pg=35",           2177, list(range(3, 37)) + [38]),
    ("WL 25 2-gap pg=36",           2241, list(range(3, 37)) + [38, 42]),
    ("WL 64 backwards-jump pg=82",  5194, list(range(44, 65)) + [25, 18] + list(range(65, 95)) + [42, 33] + list(range(95, 122))),
    ("WL 70 long-tail pg=89",       5679, [7] + list(range(65, 153))),
    # Explicit ODD num_pg cases (no even-padding) — exercises the new partial-tile path:
    ("ODD pg=33 contig",            2113, list(range(3, 36))),
    ("ODD pg=89 long-tail",         5679, [7] + list(range(65, 153))),
]

NUM_PAGES_POOL = 11923


@app.function(image=image, gpu="B200:1", timeout=600)
def run_correctness_and_bench():
    import torch
    sys.path.insert(0, "/app")
    from src.kernels.score_scale_full_bt_ws_cpasync_flat import (
        get_compiled, PAGE_SIZE, N, HEAD_DIM, ROW_STRIDE,
        FP8_REGION, PAGE_BYTES,
    )

    device = "cuda"
    all_pass = True

    print("Compiling tvm-ffi kernel (one-time)...")
    kernel, compiled = get_compiled()
    print("Compile done.\n")
    workspace = kernel.workspace

    def bench(fn, args, warmup=5, iters=50):
        cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")
        def _clone(a):
            return [x.clone() if isinstance(x, torch.Tensor) else x for x in a]
        for _ in range(warmup):
            cache.zero_(); fn(*_clone(args))
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            cache.zero_(); cl = _clone(args); torch.cuda.synchronize()
            starts[i].record(); fn(*cl); ends[i].record()
        torch.cuda.synchronize()
        ts = [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]
        return sum(ts) / len(ts)

    for label, seq_len, bt_list in WORKLOAD_CASES:
        torch.manual_seed(len(bt_list))

        # Kernel handles odd num_pg natively. We only need block_table to be
        # padded to even length (one extra readable slot — value can be any
        # valid page id; downstream topk only consumes the first seq_len out).
        num_pg = len(bt_list)
        if num_pg % 2 == 1:
            bt_padded = bt_list + [bt_list[0]]   # pad with page0 (any valid id is fine)
        else:
            bt_padded = bt_list
        M = num_pg * PAGE_SIZE
        M_real = M
        max_seq = M_real

        print(f"── {label}  (seq_len={seq_len}, num_pg={num_pg}, M={M}) ──")

        # Build per-page reference data
        K_fp8_used    = torch.randn(num_pg, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        K_scales_used = torch.rand(num_pg, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5

        # FLAT pool: shape (NUM_PAGES, PAGE_SIZE, 1, ROW_STRIDE) but bytes laid
        # out as [fp8_region | scale_region] within each page. We treat the
        # 4D tensor as a contiguous byte buffer per page.
        kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, 1, ROW_STRIDE,
                              device=device, dtype=torch.uint8)
        # View as (NUM_PAGES, PAGE_BYTES) bytes per page for flat writes
        kv_flat = kv_pool.view(NUM_PAGES_POOL, PAGE_BYTES)
        for i, pid in enumerate(bt_list):
            # fp8 region: PAGE_SIZE*HEAD_DIM = 8192 bytes
            kv_flat[pid, :FP8_REGION] = K_fp8_used[i].reshape(-1).view(torch.uint8)
            # scale region: PAGE_SIZE*4 = 256 bytes
            kv_flat[pid, FP8_REGION:PAGE_BYTES] = K_scales_used[i].view(torch.uint8).reshape(-1)

        block_table = torch.tensor(bt_padded, dtype=torch.int32, device=device)
        q_fp8 = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        w     = torch.randn(N, device=device, dtype=torch.float32)

        compiled(kv_pool, block_table, q_fp8, w, workspace)
        torch.cuda.synchronize()

        # Reference
        K_ref = K_fp8_used.reshape(M_real, HEAD_DIM).float()
        K_sc  = K_scales_used.reshape(M_real)
        scores = (K_ref @ q_fp8.float().T) * K_sc[:, None]
        ref    = torch.relu(scores) @ w

        c_view = workspace[0, :max_seq]
        match  = torch.allclose(c_view, ref, atol=1.0, rtol=0.5)
        max_err = (c_view - ref).abs().max().item()
        print(f"  → {'PASS' if match else 'FAIL'}  max_err={max_err:.4f}  (sliced [0, :{max_seq}])")
        if not match:
            all_pass = False
            mism = (c_view - ref).abs() > (1.0 + 0.5 * ref.abs())
            n_bad = mism.sum().item()
            bad = mism.nonzero(as_tuple=True)[0][:5].tolist()
            print(f"    {n_bad} bad rows; first idx: {bad}")
            for i in bad[:3]:
                print(f"    row {i}: got={c_view[i].item():.3f} ref={ref[i].item():.3f}")

        t_us = bench(compiled, [kv_pool, block_table, q_fp8, w, workspace])
        print(f"  duration: {t_us:.3f} us\n")

    print(f"{'='*60}\nOVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


@app.local_entrypoint()
def main():
    ok = run_correctness_and_bench.remote()
    if not ok:
        raise SystemExit(1)

"""submit_score_scale_full_bt_ws.py — correctness + bench for score_scale_full_bt_ws.py.

Same workload cases as submit_score_scale_full_bt.py, but uses the workspace
+ tvm-ffi pattern. Single compile, no per-shape cache (NUM_PG is sym_int).

Each request writes to workspace row 0 only (experimental). Correctness
slice = workspace[0, :max_seq_len_in_batch] vs reference.

Benchmark via torch.cuda.Event with L2 flush + arg clone (same methodology
as scripts/run_modal.py / src/utils.py).
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
]

NUM_PAGES_POOL = 11923


@app.function(image=image, gpu="B200:1", timeout=600)
def run_correctness_and_bench():
    import torch
    sys.path.insert(0, "/app")
    from src.kernels.score_scale_full_bt_ws import (
        get_compiled, PAGE_SIZE, N, HEAD_DIM, ROW_STRIDE, WS_COLS,
    )

    device = "cuda"
    all_pass = True

    # Compile once (sym_int over NUM_PG, NUM_PAGES_POOL)
    print("Compiling tvm-ffi kernel (one-time)...")
    kernel, compiled = get_compiled()
    print("Compile done.\n")
    workspace = kernel.workspace

    # ── Bench helper: L2 flush + arg clone + torch.cuda.Event ────────
    def bench(fn, args, warmup=5, iters=50):
        cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")
        def _clone(a):
            return [x.clone() if isinstance(x, torch.Tensor) else x for x in a]

        for _ in range(warmup):
            cache.zero_()
            fn(*_clone(args))
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            cache.zero_()
            cl = _clone(args)
            torch.cuda.synchronize()
            starts[i].record()
            fn(*cl)
            ends[i].record()
        torch.cuda.synchronize()
        ts = [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]   # → us
        return sum(ts) / len(ts)

    for label, seq_len, bt_list in WORKLOAD_CASES:
        torch.manual_seed(len(bt_list))

        num_pg_real = len(bt_list)
        num_pg = num_pg_real if num_pg_real % 2 == 0 else num_pg_real + 1
        bt_padded = bt_list + ([0] if num_pg != num_pg_real else [])
        M = num_pg * PAGE_SIZE
        M_real = num_pg_real * PAGE_SIZE
        # "Max seq_len in batch" — for single request = this request's seq_len rounded up
        max_seq = M_real

        print(f"── {label}  (seq_len={seq_len}, pg_real={num_pg_real}→pad={num_pg}, M={M}) ──")

        K_fp8_used    = torch.randn(num_pg_real, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        K_scales_used = torch.rand(num_pg_real, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5

        kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE, device=device, dtype=torch.uint8)
        for i, pid in enumerate(bt_list):
            kv_pool[pid, :, :HEAD_DIM] = K_fp8_used[i].view(torch.uint8)
            kv_pool[pid, :, HEAD_DIM:HEAD_DIM + 4] = (
                K_scales_used[i].view(torch.uint8).reshape(PAGE_SIZE, 4)
            )

        block_table = torch.tensor(bt_padded, dtype=torch.int32, device=device)
        q_fp8 = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        w     = torch.randn(N, device=device, dtype=torch.float32)

        # tvm-ffi: pass torch tensors directly
        compiled(kv_pool, block_table, q_fp8, w, workspace)
        torch.cuda.synchronize()

        # Reference
        K_ref = K_fp8_used.reshape(M_real, HEAD_DIM).float()
        K_sc  = K_scales_used.reshape(M_real)
        scores = (K_ref @ q_fp8.float().T) * K_sc[:, None]
        ref    = torch.relu(scores) @ w

        # Slice workspace row 0 by max_seq_len_in_batch
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

        # Bench
        t_us = bench(compiled, [kv_pool, block_table, q_fp8, w, workspace])
        print(f"  duration: {t_us:.3f} us\n")

    print(f"{'='*60}\nOVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


@app.local_entrypoint()
def main():
    ok = run_correctness_and_bench.remote()
    if not ok:
        raise SystemExit(1)

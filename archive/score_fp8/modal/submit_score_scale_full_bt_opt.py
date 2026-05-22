"""submit_score_scale_full_bt_opt.py — correctness + bench for opt kernel.

Bench modes:
  • WARM    : back-to-back calls, no L2 flush, no clone — matches the cute
              `benchmark()` methodology (apples-to-apples vs score_scale_full).
  • COLD    : 256 MB L2 flush + arg clone every iter — flashinfer-bench-style.

Reports both for each workload. Compares against score_scale_full_bt_ws on
the same 5 real workloads.
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
    from src.kernels.score_scale_full_bt_opt import (
        get_compiled, PAGE_SIZE, N, HEAD_DIM, ROW_STRIDE,
    )

    device = "cuda"
    all_pass = True

    print("Compiling tvm-ffi opt kernel (one-time)...")
    kernel, compiled = get_compiled()
    print("Compile done.\n")
    workspace = kernel.workspace

    # ── Bench helpers ────────────────────────────────────────────────
    def bench_warm(fn, args, warmup=10, iters=200):
        """Warm cache, no clone. Matches cute.benchmark methodology."""
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn(*args)
            ends[i].record()
        torch.cuda.synchronize()
        ts = [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]
        return sum(ts) / len(ts), min(ts)

    def bench_cold(fn, args, warmup=5, iters=50):
        """L2 flush + arg clone. Flashinfer-bench style."""
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
        ts = [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]
        return sum(ts) / len(ts), min(ts)

    for label, seq_len, bt_list in WORKLOAD_CASES:
        torch.manual_seed(len(bt_list))

        num_pg_real = len(bt_list)
        num_pg = num_pg_real if num_pg_real % 2 == 0 else num_pg_real + 1
        bt_padded = bt_list + ([0] if num_pg != num_pg_real else [])
        M = num_pg * PAGE_SIZE
        M_real = num_pg_real * PAGE_SIZE
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

        compiled(kv_pool, block_table, q_fp8, w, workspace)
        torch.cuda.synchronize()

        # Reference
        K_ref  = K_fp8_used.reshape(M_real, HEAD_DIM).float()
        K_sc   = K_scales_used.reshape(M_real)
        scores = (K_ref @ q_fp8.float().T) * K_sc[:, None]
        ref    = torch.relu(scores) @ w

        c_view = workspace[0, :max_seq]
        match  = torch.allclose(c_view, ref, atol=1.0, rtol=0.5)
        max_err = (c_view - ref).abs().max().item()
        print(f"  → {'PASS' if match else 'FAIL'}  max_err={max_err:.4f}  (sliced [0, :{max_seq}])")
        if not match:
            all_pass = False

        warm_avg, warm_min = bench_warm(compiled, [kv_pool, block_table, q_fp8, w, workspace])
        cold_avg, cold_min = bench_cold(compiled, [kv_pool, block_table, q_fp8, w, workspace])
        print(f"  warm: avg={warm_avg:.3f} us  min={warm_min:.3f} us")
        print(f"  cold: avg={cold_avg:.3f} us  min={cold_min:.3f} us\n")

    print(f"{'='*60}\nOVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


@app.local_entrypoint()
def main():
    ok = run_correctness_and_bench.remote()
    if not ok:
        raise SystemExit(1)

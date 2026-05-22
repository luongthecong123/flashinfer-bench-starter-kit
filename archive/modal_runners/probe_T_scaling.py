"""probe_T_scaling — measure latency vs T (all tiny, 1 tile each) to fit
   latency(T) = F + T * P
   so we can split the fixed prologue/epilogue from per-iter cost.
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


T_VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
SEQ      = 128            # one BM-tile per request (smallest possible)

NUM_PAGES_POOL = 11923
PAGE_SIZE      = 64
HEAD_DIM       = 128
ROW_STRIDE     = HEAD_DIM + 4
NUM_HEADS      = 64


@app.function(image=image, gpu="B200:1", timeout=900)
def run_probe():
    import torch
    sys.path.insert(0, "/app")
    from src.kernels.score_scale_full_bt_ws_cpasync_flat_T import (
        get_compiled, FP8_REGION, PAGE_BYTES, BM,
    )

    device = "cuda"
    print("Compiling FLAT-T persistent kernel (one-time)...")
    kernel, compiled = get_compiled()
    workspace = kernel.workspace
    print("Compile done.\n")

    kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, 1, ROW_STRIDE,
                          device=device, dtype=torch.uint8)

    def bench(fn, args, warmup=10, iters=100):
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
        return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters * 1e3

    npg_per_req = (SEQ + PAGE_SIZE - 1) // PAGE_SIZE        # = 2 pages
    max_num_pages = npg_per_req if npg_per_req % 2 == 0 else npg_per_req + 1

    results = []
    print(f"{'T':>3} {'µs':>10}")
    print("-" * 16)
    for T in T_VALUES:
        torch.manual_seed(T * 7 + SEQ)
        block_table = torch.zeros(T, max_num_pages, dtype=torch.int32, device=device)
        next_pid = 1
        for t in range(T):
            pids = list(range(next_pid, next_pid + npg_per_req))
            next_pid += npg_per_req
            block_table[t, :npg_per_req] = torch.tensor(pids, dtype=torch.int32, device=device)
            K_fp8 = torch.randn(npg_per_req, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
            K_sc  = torch.rand(npg_per_req, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5
            kv_flat = kv_pool.view(NUM_PAGES_POOL, PAGE_BYTES)
            for i, pid in enumerate(pids):
                kv_flat[pid, :FP8_REGION]           = K_fp8[i].reshape(-1).view(torch.uint8)
                kv_flat[pid, FP8_REGION:PAGE_BYTES] = K_sc[i].view(torch.uint8).reshape(-1)

        seq_lens_t = torch.full((T,), SEQ, dtype=torch.int32, device=device)
        q_real = torch.randn(T, NUM_HEADS, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        w = torch.randn(T, NUM_HEADS, device=device, dtype=torch.float32)

        t_us = bench(compiled, [kv_pool, block_table, seq_lens_t, q_real, w, workspace])
        results.append((T, t_us))
        print(f"{T:>3} {t_us:>10.3f}")

    # Linear fit  y = F + P*T  via least squares.
    n = len(results)
    sx  = sum(r[0] for r in results)
    sy  = sum(r[1] for r in results)
    sxy = sum(r[0]*r[1] for r in results)
    sxx = sum(r[0]*r[0] for r in results)
    P = (n*sxy - sx*sy) / (n*sxx - sx*sx)
    F = (sy - P*sx) / n
    print(f"\nLinear fit: latency(T) ≈ {F:.2f} µs (fixed) + {P:.3f} µs/iter * T")

    # Residuals
    print("\nResiduals (measured − fit):")
    for T, y in results:
        print(f"  T={T:>3}  measured={y:7.3f}  fit={F+P*T:7.3f}  resid={y-(F+P*T):+.3f}")

    return True


@app.local_entrypoint()
def main():
    run_probe.remote()

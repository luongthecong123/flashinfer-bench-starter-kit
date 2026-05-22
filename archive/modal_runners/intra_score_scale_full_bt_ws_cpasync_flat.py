"""intra_score_scale_full_bt_ws_cpasync_flat — per-phase profile of FLAT cpasync.

For each WORKLOAD_CASE (same as submit_score_scale_full_bt_ws_cpasync_flat.py):
  • verify correctness
  • run kernel ONCE with probes enabled, dump per-phase ranges
  • bench 50 iters total wall-clock for context
"""
import json, sys, os
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
def run_intra():
    import torch
    sys.path.insert(0, "/app")
    from src.kernels.score_scale_full_bt_ws_cpasync_flat_intra import (
        compile_intra, dump_probe, PROBE_COLS,
        PAGE_SIZE, N, HEAD_DIM, ROW_STRIDE, FP8_REGION, PAGE_BYTES,
    )

    device = "cuda"
    print("Compiling intra kernel (one-time)...")
    ker, compiled = compile_intra()
    print("Compile done.\n")
    workspace = ker.workspace

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
        return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters * 1e3

    all_traces = {}

    for label, seq_len, bt_list in WORKLOAD_CASES:
        torch.manual_seed(len(bt_list))
        num_pg_real = len(bt_list)
        num_pg = num_pg_real if num_pg_real % 2 == 0 else num_pg_real + 1
        bt_padded = bt_list + ([0] if num_pg != num_pg_real else [])
        M = num_pg * PAGE_SIZE
        M_real = num_pg_real * PAGE_SIZE
        max_seq = M_real
        grid_m = num_pg // 2

        print(f"── {label}  pg_real={num_pg_real}→pad={num_pg}  grid_m={grid_m} ──")

        K_fp8_used    = torch.randn(num_pg_real, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        K_scales_used = torch.rand(num_pg_real, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5

        kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, 1, ROW_STRIDE,
                              device=device, dtype=torch.uint8)
        kv_flat = kv_pool.view(NUM_PAGES_POOL, PAGE_BYTES)
        for i, pid in enumerate(bt_list):
            kv_flat[pid, :FP8_REGION]            = K_fp8_used[i].reshape(-1).view(torch.uint8)
            kv_flat[pid, FP8_REGION:PAGE_BYTES]  = K_scales_used[i].view(torch.uint8).reshape(-1)

        block_table = torch.tensor(bt_padded, dtype=torch.int32, device=device)
        q_fp8 = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        w     = torch.randn(N, device=device, dtype=torch.float32)
        probe = torch.zeros(grid_m, PROBE_COLS, dtype=torch.int64, device=device)

        compiled(kv_pool, block_table, q_fp8, w, workspace, probe)
        torch.cuda.synchronize()

        # correctness
        K_ref  = K_fp8_used.reshape(M_real, HEAD_DIM).float()
        K_sc   = K_scales_used.reshape(M_real)
        scores = (K_ref @ q_fp8.float().T) * K_sc[:, None]
        ref    = torch.relu(scores) @ w
        c_view = workspace[0, :max_seq]
        ok     = torch.allclose(c_view, ref, atol=1.0, rtol=0.5)
        print(f"  correctness: {'PASS' if ok else 'FAIL'}  max_err={(c_view - ref).abs().max().item():.4f}")

        # dump probe
        trace = dump_probe(probe, grid_m, label=label)
        all_traces[label] = trace

        # bench wall-clock for context (without probe to avoid skew? probe is tidx0 only — small)
        t_us = bench(compiled, [kv_pool, block_table, q_fp8, w, workspace, probe])
        print(f"  wall-clock (50 iters avg): {t_us:.3f} µs\n")

    # Save the trace from the largest workload
    out_path = "/tmp/intra_score_scale_full_bt_ws_cpasync_flat.json"
    biggest = max(all_traces.items(), key=lambda kv: len(kv[1]["traceEvents"]))
    with open(out_path, "w") as f:
        json.dump(biggest[1], f)
    print(f"\nSaved chrome trace ({biggest[0]}, {len(biggest[1]['traceEvents'])} events) → {out_path}")
    return biggest[1]


@app.local_entrypoint()
def main():
    trace = run_intra.remote()
    out = "reports/intra_score_scale_full_bt_ws_cpasync_flat.json"
    os.makedirs("reports", exist_ok=True)
    with open(out, "w") as f:
        json.dump(trace, f)
    print(f"\nSaved chrome trace locally → {out}  ({len(trace['traceEvents'])} events)")

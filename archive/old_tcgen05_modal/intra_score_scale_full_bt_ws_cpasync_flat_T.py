"""intra_score_scale_full_bt_ws_cpasync_flat_T — per-CTA per-iter profile.

For each WORKLOAD_CASE: verify correctness vs torch, run with probes, dump.
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


WORKLOAD_CASES = [
    ("1req_tiny",   [128]),
    ("4req_tiny",   [128, 128, 128, 128]),
    ("8req_tiny",   [128] * 8),
    ("16req_tiny",  [128] * 16),
    ("32req_tiny",  [128] * 32),
    ("4req_mixed",  [256, 1024, 4096, 768]),
    ("8req_mixed",  [256, 512, 1024, 5194, 768, 5679, 1500, 320]),
]

NUM_PAGES_POOL = 11923


@app.function(image=image, gpu="B200:1", timeout=900)
def run_intra():
    import torch
    sys.path.insert(0, "/app")
    from src.kernels.score_scale_full_bt_ws_cpasync_flat_T_intra import (
        compile_intra, dump_probe, PROBE_COLS,
        PAGE_SIZE, NUM_HEADS, HEAD_DIM, ROW_STRIDE, FP8_REGION, PAGE_BYTES, BM,
    )

    device = "cuda"
    print("Compiling _flat_T intra kernel (one-time)...")
    ker, compiled = compile_intra()
    workspace = ker.workspace
    print("Compile done.\n")

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

    def seq_to_pages(s): return (s + PAGE_SIZE - 1) // PAGE_SIZE

    for label, seqs in WORKLOAD_CASES:
        T = len(seqs)
        pages_per_req      = [seq_to_pages(s) for s in seqs]
        max_num_pages_real = max(pages_per_req)
        max_num_pages      = max_num_pages_real + (max_num_pages_real & 1)
        num_splits         = (max_num_pages + 1) // 2

        kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, 1, ROW_STRIDE,
                              device=device, dtype=torch.uint8)
        torch.manual_seed(T * 7 + max(seqs))
        block_table = torch.zeros(T, max_num_pages, dtype=torch.int32, device=device)
        next_pid = 1
        all_K_fp8, all_K_sc = {}, {}
        for t, npg in enumerate(pages_per_req):
            pids = list(range(next_pid, next_pid + npg)); next_pid += npg
            block_table[t, :npg] = torch.tensor(pids, dtype=torch.int32, device=device)
            if npg < max_num_pages:
                block_table[t, npg:] = pids[0]
            K_fp8 = torch.randn(npg, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
            K_sc  = torch.rand(npg, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5
            all_K_fp8[t], all_K_sc[t] = K_fp8, K_sc
            kv_flat = kv_pool.view(NUM_PAGES_POOL, PAGE_BYTES)
            for i, pid in enumerate(pids):
                kv_flat[pid, :FP8_REGION]           = K_fp8[i].reshape(-1).view(torch.uint8)
                kv_flat[pid, FP8_REGION:PAGE_BYTES] = K_sc[i].view(torch.uint8).reshape(-1)

        seq_lens_t = torch.tensor(seqs, dtype=torch.int32, device=device)
        q_real = torch.randn(T, NUM_HEADS, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        w = torch.randn(T, NUM_HEADS, device=device, dtype=torch.float32)
        probe = torch.zeros(num_splits, PROBE_COLS, dtype=torch.int64, device=device)

        workspace.zero_()
        compiled(kv_pool, block_table, seq_lens_t, q_real, w, workspace, probe)
        torch.cuda.synchronize()

        # Correctness
        all_ok, max_err_g = True, 0.0
        for t, npg in enumerate(pages_per_req):
            M_real = pages_per_req[t] * PAGE_SIZE
            tile_M = ((seqs[t] + BM - 1) // BM) * BM
            cmp_M = min(tile_M, M_real)
            K_ref = all_K_fp8[t].reshape(M_real, HEAD_DIM).float()
            K_sc  = all_K_sc[t].reshape(M_real)
            scores = (K_ref @ q_real[t].float().T) * K_sc[:, None]
            ref = torch.relu(scores) @ w[t]
            got = workspace[t, :cmp_M]
            ok = torch.allclose(got, ref[:cmp_M], atol=1.0, rtol=0.5)
            err = (got - ref[:cmp_M]).abs().max().item()
            max_err_g = max(max_err_g, err)
            if not ok: all_ok = False
        print(f"── {label}  T={T}  num_splits={num_splits}  "
              f"correctness: {'PASS' if all_ok else 'FAIL'}  max_err={max_err_g:.4f}")

        dump_probe(probe, num_splits, label=label)

        t_us = bench(compiled, [kv_pool, block_table, seq_lens_t, q_real, w, workspace, probe])
        print(f"  wall-clock (100 iters): {t_us:.3f} µs\n")


@app.local_entrypoint()
def main():
    run_intra.remote()

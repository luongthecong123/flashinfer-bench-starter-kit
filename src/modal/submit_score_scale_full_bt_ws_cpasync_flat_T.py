"""submit_score_scale_full_bt_ws_cpasync_flat_T — correctness + bench for the
T-dim persistent FLAT cpasync-with-static-TMA kernel.

q is host-padded to MAX_T=32 rows so the TMA descriptor is fully static.
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


WORKLOAD_CASES = [
    ("1req_tiny",    [128]),                              # smoke test: T=1, single tile
    ("2req_tiny",    [128, 128]),
    ("4req_1long",   [512, 1024, 4096, 768]),
    ("8req_2long",   [256, 512, 1024, 5194, 768, 5679, 1500, 320]),
    ("12req_3long",  [256, 512, 768, 1024, 4096, 1500, 5194, 320, 768, 5679, 256, 1024]),
    ("16req_short",  [512] * 16),
    ("6req_1huge",   [256, 512, 8192, 768, 1024, 320]),
]

NUM_PAGES_POOL = 11923
PAGE_SIZE      = 64
HEAD_DIM       = 128
ROW_STRIDE     = HEAD_DIM + 4
NUM_HEADS      = 64
MAX_T          = 32


def seq_to_pages(s):
    return (s + PAGE_SIZE - 1) // PAGE_SIZE


@app.function(image=image, gpu="B200:1", timeout=900)
def run_correctness_and_bench():
    import torch
    sys.path.insert(0, "/app")
    from src.kernels.score_scale_full_bt_ws_cpasync_flat_T import (
        get_compiled, FP8_REGION, PAGE_BYTES, BM,
    )

    device = "cuda"
    all_pass = True

    print("Compiling FLAT-TMA persistent kernel (one-time)...")
    kernel, compiled = get_compiled()
    print("Compile done.\n")
    workspace = kernel.workspace

    kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, 1, ROW_STRIDE,
                          device=device, dtype=torch.uint8)

    def bench(fn, args, warmup=5, iters=30):
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

    for label, seqs in WORKLOAD_CASES:
        T = len(seqs)
        assert T <= MAX_T, f"T={T} exceeds MAX_T={MAX_T}"
        pages_per_req      = [seq_to_pages(s) for s in seqs]
        max_num_pages_real = max(pages_per_req)
        max_num_pages      = max_num_pages_real + (max_num_pages_real & 1)

        total_tiles = sum((s + BM - 1) // BM for s in seqs)
        num_splits  = (max_num_pages + 1) // 2
        print(f"── {label}  T={T}  max_seq={max(seqs)}  max_pg={max_num_pages_real}  "
              f"num_splits={num_splits}  total_M_tiles={total_tiles} ──")

        torch.manual_seed(T * 7 + max(seqs))
        block_table = torch.zeros(T, max_num_pages, dtype=torch.int32, device=device)
        next_pid = 1
        page_ids_per_req = []
        all_K_fp8 = {}
        all_K_sc  = {}
        for t, npg in enumerate(pages_per_req):
            pids = list(range(next_pid, next_pid + npg))
            next_pid += npg
            page_ids_per_req.append(pids)
            block_table[t, :npg] = torch.tensor(pids, dtype=torch.int32, device=device)
            if npg < max_num_pages:
                block_table[t, npg:] = pids[0]

            K_fp8 = torch.randn(npg, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
            K_sc  = torch.rand(npg, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5
            all_K_fp8[t] = K_fp8
            all_K_sc [t] = K_sc
            kv_flat = kv_pool.view(NUM_PAGES_POOL, PAGE_BYTES)
            for i, pid in enumerate(pids):
                kv_flat[pid, :FP8_REGION]           = K_fp8[i].reshape(-1).view(torch.uint8)
                kv_flat[pid, FP8_REGION:PAGE_BYTES] = K_sc[i].view(torch.uint8).reshape(-1)

        seq_lens_t = torch.tensor(seqs, dtype=torch.int32, device=device)
        q_real = torch.randn(T, NUM_HEADS, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
        w = torch.randn(T, NUM_HEADS, device=device, dtype=torch.float32)

        workspace.zero_()
        compiled(kv_pool, block_table, seq_lens_t, q_real, w, workspace)
        torch.cuda.synchronize()

        case_ok = True
        max_err_global = 0.0
        for t, npg in enumerate(pages_per_req):
            M_real = pages_per_req[t] * PAGE_SIZE
            tile_M = ((seqs[t] + BM - 1) // BM) * BM
            # Compare only over [0:cmp_M] where cmp_M = min(tile_M, M_real). The
            # kernel writes whole BM tiles (potentially past the last real page);
            # downstream topk discards values past seq_len.
            cmp_M = min(tile_M, M_real)
            K_ref  = all_K_fp8[t].reshape(M_real, HEAD_DIM).float()
            K_sc   = all_K_sc [t].reshape(M_real)
            scores = (K_ref @ q_real[t].float().T) * K_sc[:, None]
            ref    = torch.relu(scores) @ w[t]
            ref_t  = ref[:cmp_M]
            got    = workspace[t, :cmp_M]
            ok = torch.allclose(got, ref_t, atol=1.0, rtol=0.5)
            err = (got - ref_t).abs().max().item()
            max_err_global = max(max_err_global, err)
            if not ok:
                case_ok = False
                bad = ((got - ref_t).abs() > (1.0 + 0.5 * ref_t.abs())).nonzero(as_tuple=True)[0][:5].tolist()
                print(f"    t={t} seq={seqs[t]} FAIL  max_err={err:.4f}  first_bad={bad}")
                for i in bad[:3]:
                    print(f"      row {i}: got={got[i].item():.3f} ref={ref_t[i].item():.3f}")
        print(f"  → {'PASS' if case_ok else 'FAIL'}  max_err={max_err_global:.4f}")
        if not case_ok:
            all_pass = False

        t_us = bench(compiled, [kv_pool, block_table, seq_lens_t, q_real, w, workspace])
        print(f"  duration: {t_us:.3f} µs\n")

    print(f"{'='*60}\nOVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


@app.local_entrypoint()
def main():
    ok = run_correctness_and_bench.remote()
    if not ok:
        raise SystemExit(1)

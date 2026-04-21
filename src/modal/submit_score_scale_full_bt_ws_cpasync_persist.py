"""submit_score_scale_full_bt_ws_cpasync_persist.py — correctness + bench
for the request-persistent variant.

Tests with NUM_REQ requests sharing the same num_pg shape (worst-case
across requests). For each request we compute the standalone reference
and compare to workspace[req_idx, :M_real].

Exercises:
  - persistent loop over requests inside one CTA per tile
  - per-request runtime TMA-B slicing on q_3d[req_idx, :, :]
  - per-request weights row, block_table row, workspace row
  - mbarrier phase toggle (req_idx & 1)
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


# Each case: (label, [(seq_len, bt_list), ...]) — multiple requests per case
WORKLOAD_CASES = [
    ("WL multi-req pg=34 (3 reqs)",
        [
            (2161, list(range(3, 37))),
            (2155, list(range(40, 74))),
            (2100, list(range(80, 114))),
        ],
    ),
    ("WL multi-req pg=82 (2 reqs)",
        [
            (5194, list(range(44, 65)) + [25, 18] + list(range(65, 95)) + [42, 33] + list(range(95, 122))),
            (5180, list(range(150, 232))),
        ],
    ),
    ("WL single-req pg=34 (1 req)",
        [
            (2161, list(range(3, 37))),
        ],
    ),
]

NUM_PAGES_POOL = 11923


@app.function(image=image, gpu="B200:1", timeout=600)
def run_correctness_and_bench():
    import torch
    sys.path.insert(0, "/app")
    from src.kernels.score_scale_full_bt_ws_cpasync_persist import (
        get_compiled, PAGE_SIZE, N, HEAD_DIM, ROW_STRIDE, WS_COLS,
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
        return sum(ts) / len(ts)

    for label, reqs in WORKLOAD_CASES:
        torch.manual_seed(len(reqs))

        num_pg_real_max = max(len(bt) for _, bt in reqs)
        num_pg = num_pg_real_max if num_pg_real_max % 2 == 0 else num_pg_real_max + 1
        num_requests = len(reqs)
        M = num_pg * PAGE_SIZE

        print(f"── {label}  (num_req={num_requests}, num_pg={num_pg}, M={M}) ──")

        kv_pool = torch.zeros(NUM_PAGES_POOL, PAGE_SIZE, ROW_STRIDE,
                              device=device, dtype=torch.uint8)
        block_table_2d = torch.zeros(num_requests, num_pg, dtype=torch.int32, device=device)
        q_3d           = torch.zeros(num_requests, N, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device)
        w_2d           = torch.zeros(num_requests, N, dtype=torch.float32, device=device)

        ref_per_req = []

        for r, (seq_len, bt_list) in enumerate(reqs):
            num_pg_real = len(bt_list)
            bt_padded = bt_list + [0] * (num_pg - num_pg_real)
            M_real = num_pg_real * PAGE_SIZE

            K_fp8_used    = torch.randn(num_pg_real, PAGE_SIZE, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
            K_scales_used = torch.rand(num_pg_real, PAGE_SIZE, device=device, dtype=torch.float32) + 0.5

            for i, pid in enumerate(bt_list):
                kv_pool[pid, :, :HEAD_DIM] = K_fp8_used[i].view(torch.uint8)
                kv_pool[pid, :, HEAD_DIM:HEAD_DIM + 4] = (
                    K_scales_used[i].view(torch.uint8).reshape(PAGE_SIZE, 4)
                )

            block_table_2d[r] = torch.tensor(bt_padded, dtype=torch.int32, device=device)
            q_fp8 = torch.randn(N, HEAD_DIM, device=device).clamp(-100, 100).to(torch.float8_e4m3fn)
            w     = torch.randn(N, device=device, dtype=torch.float32)
            q_3d[r] = q_fp8
            w_2d[r] = w

            K_ref = K_fp8_used.reshape(M_real, HEAD_DIM).float()
            K_sc  = K_scales_used.reshape(M_real)
            scores = (K_ref @ q_fp8.float().T) * K_sc[:, None]
            ref    = torch.relu(scores) @ w
            ref_per_req.append((r, M_real, ref))

        compiled(kv_pool, block_table_2d, q_3d, w_2d, workspace)
        torch.cuda.synchronize()

        case_pass = True
        for r, M_real, ref in ref_per_req:
            c_view = workspace[r, :M_real]
            match  = torch.allclose(c_view, ref, atol=1.0, rtol=0.5)
            max_err = (c_view - ref).abs().max().item()
            tag = 'PASS' if match else 'FAIL'
            print(f"  req {r}: {tag}  max_err={max_err:.4f}  (sliced [{r}, :{M_real}])")
            if not match:
                case_pass = False
                mism = (c_view - ref).abs() > (1.0 + 0.5 * ref.abs())
                n_bad = mism.sum().item()
                bad = mism.nonzero(as_tuple=True)[0][:5].tolist()
                print(f"    {n_bad} bad rows; first idx: {bad}")
                for i in bad[:3]:
                    print(f"    row {i}: got={c_view[i].item():.3f} ref={ref[i].item():.3f}")

        if not case_pass:
            all_pass = False

        t_us = bench(compiled, [kv_pool, block_table_2d, q_3d, w_2d, workspace])
        print(f"  duration: {t_us:.3f} us  ({t_us / num_requests:.3f} us/req)\n")

    print(f"{'='*60}\nOVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")
    return all_pass


@app.local_entrypoint()
def main():
    ok = run_correctness_and_bench.remote()
    if not ok:
        raise SystemExit(1)

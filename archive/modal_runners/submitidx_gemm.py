"""submitidx_gemm.py — benchmark idxer_gemm.compute_scores on all seq_len > 2048 cases.

For each long-seq request the script:
  1. Gathers K_fp8 and K_scales for this request (prep, not benchmarked).
  2. Benchmarks compute_scores(q_fp8, K_fp8, K_scales, weights) vs an inline ref.
     Both ref and impl receive fp8 inputs; dequant happens inside the kernel.
  3. seq_len is padded/trimmed to exactly 2048 tokens to match the static kernel.
  4. Checks correctness (max absolute diff must be 0).

Usage:
    modal run src/modal/submitidx_gemm.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.idxer_gemm")
print("IMPL_MODULE:", IMPL_MODULE)


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_bench(impl_module: str):
    import json
    import sys
    import torch
    from pathlib import Path as P
    from importlib import import_module
    from safetensors.torch import load_file

    sys.path.insert(0, "/app")

    impl = import_module(impl_module)

    CONTEST   = P("/data")
    JSONL     = (CONTEST / "workloads" / "dsa_paged"
                 / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl")
    PAGE_SIZE = 64
    HEAD_DIM  = 128
    NUM_HEADS = 64
    TOPK      = 2048  # also the static seq_len we feed to the kernel

    # ── helpers ───────────────────────────────────────────────────────────────
    def gather_K_fp8_and_scales(k_cache_raw, block_table_row, seq_len, device):
        """Return K_fp8 [seq_len, HEAD_DIM] and K_scales [seq_len] for one request.

        k_cache_raw: [num_pages, page_size, 1, 132] int8 (uint8 reinterpret)
        Layout per page: first page_size*128 bytes = fp8, last page_size*4 bytes = scales.
        """
        k_u8 = k_cache_raw.view(torch.uint8)
        pool_pages = k_u8.shape[0]
        flat  = k_u8.view(pool_pages, PAGE_SIZE * (HEAD_DIM + 4))

        # fp8 part: first PAGE_SIZE*128 bytes per page
        fp8_flat   = flat[:, :PAGE_SIZE * HEAD_DIM]               # [pool_pages, P*128]
        fp8_tensor = fp8_flat.view(pool_pages, PAGE_SIZE, HEAD_DIM).view(torch.float8_e4m3fn)

        # scale part: last PAGE_SIZE*4 bytes per page → 1 float32 per token
        sc_flat    = flat[:, PAGE_SIZE * HEAD_DIM:].contiguous()  # [pool_pages, P*4]
        scales     = sc_flat.view(pool_pages, PAGE_SIZE, 4).view(torch.float32)  # [pool_pages, P]

        # gather pages for this request
        num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        pages     = block_table_row[:num_pages].long()
        offsets   = torch.arange(PAGE_SIZE, device=device)
        tok_ids   = (pages.unsqueeze(1) * PAGE_SIZE + offsets.unsqueeze(0)).reshape(-1)

        K_fp8_all  = fp8_tensor.reshape(-1, HEAD_DIM)
        scales_all = scales.reshape(-1)

        K_fp8_seq   = K_fp8_all[tok_ids][:seq_len]    # [seq_len, 128]
        K_scales_seq = scales_all[tok_ids][:seq_len]  # [seq_len]
        return K_fp8_seq, K_scales_seq

    def ref_compute(q_fp8, K_fp8, K_scales, weights):
        q = q_fp8.to(torch.float32)
        K = K_fp8.to(torch.float32) * K_scales[:, None]
        s = torch.mm(q, K.T)
        s = torch.relu(s)
        return (s * weights[:, None]).sum(dim=0)

    cache_buf = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")

    def bench_fn(fn, args, warmup=3, iters=20):
        for _ in range(warmup):
            cache_buf.zero_()
            fn(*[a.clone() if isinstance(a, torch.Tensor) else a for a in args])
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            cache_buf.zero_()
            cloned = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
            torch.cuda.synchronize()
            starts[i].record()
            fn(*cloned)
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return sum(times) / len(times)

    # ── main loop ─────────────────────────────────────────────────────────────
    all_workloads = [json.loads(l) for l in open(JSONL)]

    hdr = (f"{'WL':>4} {'B':>3} {'UUID':>10} {'seq_len':>8}"
           f" {'MaxDiff':>10} {'Status':>6} {'Ref ms':>8} {'Impl ms':>8} {'Speedup':>8}")
    print(hdr)
    print("-" * len(hdr))

    all_pass = True
    results  = []

    for i_w, w in enumerate(all_workloads):
        ax  = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        uuid          = w["workload"]["uuid"][:8]
        batch_size    = ax["batch_size"]
        num_pages     = ax["num_pages"]
        max_num_pages = ax["max_num_pages"]

        sf_path     = str(CONTEST / inp["seq_lens"]["path"])
        sf          = load_file(sf_path)
        seq_lens    = sf[inp["seq_lens"]["tensor_key"]].cuda()
        block_table = sf[inp["block_table"]["tensor_key"]].cuda()

        long_idxs = (seq_lens > TOPK).nonzero(as_tuple=True)[0]
        if long_idxs.numel() == 0:
            continue

        # random q, weights (use randn→fp8 for valid bit patterns, no NaNs)
        q_fp8_batch   = torch.randn(batch_size, NUM_HEADS, HEAD_DIM,
                                    device="cuda").to(torch.float8_e4m3fn)
        weights_batch = torch.randn(batch_size, NUM_HEADS, device="cuda")

        # K cache: use randn→fp8 for fp8 part, rand for scales (positive)
        k_fp8_pool   = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM,
                                   device="cuda").to(torch.float8_e4m3fn)
        k_scale_pool = torch.rand(num_pages, PAGE_SIZE, device="cuda") + 0.5

        # Pack into the expected [num_pages, page_size, 1, 132] int8 format
        fp8_bytes  = k_fp8_pool.view(torch.uint8).reshape(num_pages, PAGE_SIZE * HEAD_DIM)
        scale_bytes = k_scale_pool.reshape(num_pages * PAGE_SIZE).view(torch.uint8
                      ).reshape(num_pages, PAGE_SIZE * 4)
        packed = torch.cat([fp8_bytes, scale_bytes], dim=1).view(torch.int8
                 ).reshape(num_pages, PAGE_SIZE, 1, HEAD_DIM + 4)

        for b in long_idxs.tolist():
            sl = int(seq_lens[b].item())

            q_fp8   = q_fp8_batch[b]              # [64, 128]  float8_e4m3fn
            weights = weights_batch[b]             # [64]       float32

            # Gather exactly TOPK tokens (static seq_len for the kernel)
            K_fp8, K_scales = gather_K_fp8_and_scales(
                packed, block_table[b], TOPK, "cuda")   # [2048, 128], [2048]

            # correctness
            ref_s  = ref_compute(q_fp8, K_fp8, K_scales, weights)
            impl_s = impl.compute_scores(q_fp8, K_fp8, K_scales, weights)
            torch.cuda.synchronize()
            max_diff = (ref_s - impl_s).abs().max().item()
            ok = max_diff == 0.0
            if not ok:
                all_pass = False

            # benchmark
            args = (q_fp8, K_fp8, K_scales, weights)
            r_ms = bench_fn(ref_compute,          args)
            i_ms = bench_fn(impl.compute_scores,  args)
            sp   = r_ms / i_ms if i_ms > 0 else 0.0

            line = (f"{i_w+1:>4} {b:>3} {uuid:>10} {sl:>8}"
                    f" {max_diff:>10.2e} {'PASS' if ok else 'FAIL':>6}"
                    f" {r_ms:>8.3f} {i_ms:>8.3f} {sp:>7.2f}x")
            print(line)

            results.append({
                "workload": i_w + 1, "batch_elem": b, "uuid": uuid,
                "seq_len": sl, "batch_size": batch_size,
                "max_num_pages": max_num_pages, "num_pages": num_pages,
                "ok": ok, "max_diff": max_diff,
                "ref_ms": r_ms, "impl_ms": i_ms, "speedup": sp,
            })

    status = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"\n>> {status}  ({len(results)} long-seq requests tested)")
    return {"all_pass": all_pass, "results": results}


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path as P

    data = run_bench.remote(IMPL_MODULE)
    if data is None:
        return

    variant  = IMPL_MODULE.replace("src.kernels.", "")
    csv_path = P("bench_idx_gemm_results.csv")

    existing_rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r["variant"] != variant]

    rows = existing_rows
    for r in data["results"]:
        rows.append({
            "variant":    variant,
            "workload":   r["workload"],
            "batch_elem": r["batch_elem"],
            "uuid":       r["uuid"],
            "seq_len":    r["seq_len"],
            "batch_size": r["batch_size"],
            "max_pages":  r["max_num_pages"],
            "num_pages":  r["num_pages"],
            "ok":         r["ok"],
            "max_diff":   f"{r['max_diff']:.2e}",
            "ref_ms":     f"{r['ref_ms']:.3f}",
            "impl_ms":    f"{r['impl_ms']:.3f}",
            "speedup":    f"{r['speedup']:.2f}",
        })

    fields = ["variant", "workload", "batch_elem", "uuid", "seq_len",
              "batch_size", "max_pages", "num_pages", "ok", "max_diff",
              "ref_ms", "impl_ms", "speedup"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    status = "ALL PASS" if data["all_pass"] else "SOME FAILED"
    print(f"\n>> {status} — wrote {len(data['results'])} rows to {csv_path}")

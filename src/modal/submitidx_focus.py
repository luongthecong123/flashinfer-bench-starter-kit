"""submitidx_focus.py — run idxer_focus.py on every request with seq_len > 2048.

For each of the 128 workloads the script:
  1. Loads seq_lens from safetensors.
  2. Finds batch elements where seq_lens[b] > TOPK (= 2048).
  3. Slices the inputs to batch_size=1 for each such element.
  4. Correctness-checks and benchmarks idxer_focus against the reference.

Usage:
    modal run src/modal/submitidx_focus.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.idxer_focus")

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

    from src.kernels.idxer_ref import run as ref_run

    CONTEST  = P("/data")
    JSONL    = (CONTEST / "workloads" / "dsa_paged"
                / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl")
    PAGE_SIZE = 64
    TOPK      = 2048
    NUM_HEADS = 64
    HEAD_DIM  = 128

    def make_q_and_weights(batch_size):
        q = torch.randn(batch_size, NUM_HEADS, HEAD_DIM,
                        dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
        w = torch.randn(batch_size, NUM_HEADS, dtype=torch.float32, device="cuda")
        return q, w

    def make_k_cache(num_pages):
        return torch.randint(
            0, 256,
            (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
            dtype=torch.uint8, device="cuda",
        ).view(torch.int8)

    def alloc_topk(device="cuda"):
        return torch.full((1, TOPK), -1, dtype=torch.int32, device=device)

    def check(ref_idx, impl_idx, sl):
        actual_k = min(TOPK, sl)
        ref_set  = set(ref_idx[0, :actual_k].tolist()) - {-1}
        impl_set = set(impl_idx[0, :actual_k].tolist()) - {-1}
        if not ref_set:
            return True, 0.0
        missing   = len(ref_set - impl_set)
        miss_frac = missing / len(ref_set)
        return miss_frac < 0.01, miss_frac

    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")

    def bench_fn(fn, args, warmup=3, iters=20):
        for _ in range(warmup):
            cache.zero_()
            out = alloc_topk()
            fn(*args, out)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            cache.zero_()
            out = alloc_topk()
            torch.cuda.synchronize()
            starts[i].record()
            fn(*args, out)
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return sum(times) / len(times)

    all_workloads = [json.loads(l) for l in open(JSONL)]

    hdr = f"{'WL':>4} {'B':>3} {'UUID':>10} {'seq_len':>8} {'MissFrac':>10} {'Status':>6} {'Ref ms':>8} {'Impl ms':>8} {'Speedup':>8}"
    print(hdr)
    print("-" * len(hdr))

    all_pass = True
    results  = []

    for i_w, w in enumerate(all_workloads):
        ax  = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        uuid      = w["workload"]["uuid"][:8]
        batch_size    = ax["batch_size"]
        max_num_pages = ax["max_num_pages"]
        num_pages     = ax["num_pages"]

        # Load seq_lens and block_table from safetensors
        sf_path = str(CONTEST / inp["seq_lens"]["path"])
        sf      = load_file(sf_path)
        seq_lens    = sf[inp["seq_lens"]["tensor_key"]].cuda()     # [B]
        block_table = sf[inp["block_table"]["tensor_key"]].cuda()  # [B, max_num_pages]

        # Find batch elements with seq_len > TOPK
        long_mask = seq_lens > TOPK
        long_idxs = long_mask.nonzero(as_tuple=True)[0]
        if long_idxs.numel() == 0:
            continue  # skip fast-path workloads

        # Shared KV pool and random q/weights (same pool for all elements)
        q_fp8, weights_full = make_q_and_weights(batch_size)
        k_cache = make_k_cache(num_pages)

        for b in long_idxs.tolist():
            sl = int(seq_lens[b].item())

            # Slice to batch_size=1
            q1          = q_fp8[b:b+1]          # [1, 64, 128]
            w1          = weights_full[b:b+1]    # [1, 64]
            sl1         = seq_lens[b:b+1]        # [1]
            bt1         = block_table[b:b+1]     # [1, max_num_pages]

            # Correctness check
            ref_out  = alloc_topk()
            impl_out = alloc_topk()
            ref_run(q1, k_cache, w1, sl1, bt1, ref_out)
            impl.run(q1, k_cache, w1, sl1, bt1, impl_out)
            torch.cuda.synchronize()
            ok, miss_frac = check(ref_out, impl_out, sl)
            if not ok:
                all_pass = False

            # Benchmark
            ref_args  = (q1, k_cache, w1, sl1, bt1)
            r_ms = bench_fn(ref_run,  ref_args)
            i_ms = bench_fn(impl.run, ref_args)
            sp   = r_ms / i_ms if i_ms > 0 else 0.0

            line = (f"{i_w+1:>4} {b:>3} {uuid:>10} {sl:>8}"
                    f" {miss_frac:>10.4f} {'PASS' if ok else 'FAIL':>6}"
                    f" {r_ms:>8.3f} {i_ms:>8.3f} {sp:>7.2f}x")
            print(line)

            results.append({
                "workload":   i_w + 1,
                "batch_elem": b,
                "uuid":       uuid,
                "seq_len":    sl,
                "batch_size": batch_size,
                "max_num_pages": max_num_pages,
                "num_pages":  num_pages,
                "ok":         ok,
                "miss_frac":  miss_frac,
                "ref_ms":     r_ms,
                "impl_ms":    i_ms,
                "speedup":    sp,
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

    variant   = IMPL_MODULE.replace("src.kernels.", "")
    csv_path  = P("bench_idx_focus_results.csv")

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
            "ref_ms":     f"{r['ref_ms']:.3f}",
            "impl_ms":    f"{r['impl_ms']:.3f}",
            "speedup":    f"{r['speedup']:.2f}",
        })

    fields = ["variant", "workload", "batch_elem", "uuid", "seq_len",
              "batch_size", "max_pages", "num_pages", "ok", "ref_ms", "impl_ms", "speedup"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    status = "ALL PASS" if data["all_pass"] else "SOME FAILED"
    print(f"\n>> {status} — wrote {len(data['results'])} rows to {csv_path}")

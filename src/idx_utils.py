"""Correctness check + benchmark harness for DSA top-k indexer.

Mirrors src/utils.py but for the indexer track:
  dsa_topk_indexer_fp8_h64_d128_topk2048_ps64

Usage:
    from src import idx_utils
    idx_utils.impl_fn = my_run
    idx_utils.main()
"""
import json, math, torch
from pathlib import Path
from safetensors.torch import load_file

# ── Flags ──
CHECK   = True
MEASURE = True
START = 0    # 0-indexed start workload
END   = 0    # 0 = all

# ── Model params ──
NUM_HEADS  = 64
HEAD_DIM   = 128
TOPK       = 2048
PAGE_SIZE  = 64

# ── Paths (override on Modal) ──
import os as _os
CONTEST = Path(_os.environ.get(
    "CONTEST_DIR",
    _os.environ.get("FIB_DATASET_PATH",
                    "/home/luongt/codeCuda/flashinfer26dsa/mlsys26-contest")
))
JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"

# ── Impl hooks ──
def _missing(*a, **kw):
    raise RuntimeError("Set idx_utils.ref_fn / idx_utils.impl_fn before calling main()")

ref_fn  = _missing
impl_fn = _missing


def make_tensors(batch_size, num_pages, max_num_pages, device="cuda"):
    """Allocate random indexer inputs (matching flashinfer_bench random generation)."""
    q_index_fp8 = torch.randn(batch_size, NUM_HEADS, HEAD_DIM,
                               dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    # k_index_cache_fp8: [num_pages, page_size, 1, 132] int8
    # The last 4 bytes per (page, token) are the fp32 scale
    k_index_cache_fp8 = torch.randint(0, 256,
                                       (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
                                       dtype=torch.uint8, device=device).view(torch.int8)
    weights = torch.randn(batch_size, NUM_HEADS, dtype=torch.float32, device=device)
    # seq_lens and block_table are loaded from safetensors, not random
    return q_index_fp8, k_index_cache_fp8, weights


def alloc_topk_indices(batch_size, device="cuda"):
    return torch.full((batch_size, TOPK), -1, dtype=torch.int32, device=device)


def _clone_args(args):
    return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]


def bench(fn, args, warmup=3, iters=20):
    """Benchmark with L2 flush + arg clone (matches flashinfer_bench methodology)."""
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")

    for _ in range(warmup):
        cache.zero_()
        fn(*_clone_args(args))
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        cache.zero_()
        cloned = _clone_args(args)
        torch.cuda.synchronize()
        start_events[i].record()
        fn(*cloned)
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


def check_topk_indices(ref_idx, impl_idx, seq_lens):
    """Check that impl topk_indices match ref (order-independent per batch).
    Returns (ok, max_missing_fraction)."""
    batch_size = ref_idx.shape[0]
    max_miss = 0.0
    for b in range(batch_size):
        sl = int(seq_lens[b].item())
        actual_k = min(TOPK, sl)
        if actual_k == 0:
            continue
        ref_set  = set(ref_idx[b, :actual_k].tolist())
        impl_set = set(impl_idx[b, :actual_k].tolist())
        # Remove sentinel -1 entries
        ref_set.discard(-1)
        impl_set.discard(-1)
        if len(ref_set) == 0:
            continue
        missing = len(ref_set - impl_set)
        miss_frac = missing / len(ref_set)
        max_miss = max(max_miss, miss_frac)
    return max_miss < 0.01, max_miss


# ────────────────────────────────────────────
def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    all_workloads = [json.loads(l) for l in open(JSONL)]
    end = END if END > 0 else len(all_workloads)
    workloads = all_workloads[START:end]
    print(f"=== {len(workloads)} INDEXER WORKLOADS (#{START+1}..#{end} of {len(all_workloads)}) ===")

    hdr = f"{'#':>3} {'UUID':>10} {'B':>3} {'MaxPg':>5} {'NumPg':>6}"
    if CHECK:   hdr += f" {'MissFrac':>10} {'Status':>6}"
    if MEASURE: hdr += f" {'Ref ms':>8} {'Impl ms':>8} {'Speedup':>8}"
    print(hdr)
    print("-" * len(hdr))

    all_pass = True
    durations, ref_ms_list, speedups = [], [], []

    for i_w, w in enumerate(workloads, start=START):
        ax  = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        batch_size = ax["batch_size"]
        max_num_pages = ax["max_num_pages"]
        num_pages = ax["num_pages"]
        uuid = w["workload"]["uuid"][:8]

        # Build inputs
        q_fp8, k_cache_fp8, weights = make_tensors(batch_size, num_pages, max_num_pages)

        # Load seq_lens and block_table from safetensors
        sf_path = str(CONTEST / inp["seq_lens"]["path"])
        sf = load_file(sf_path)
        seq_lens = sf[inp["seq_lens"]["tensor_key"]].cuda()
        block_table = sf[inp["block_table"]["tensor_key"]].cuda()

        line = f"{i_w+1:>3} {uuid:>10} {batch_size:>3} {max_num_pages:>5} {num_pages:>6}"

        args = (q_fp8, k_cache_fp8, weights, seq_lens, block_table)

        if CHECK:
            ref_topk = alloc_topk_indices(batch_size)
            impl_topk = alloc_topk_indices(batch_size)
            ref_fn(*args, ref_topk)
            impl_fn(*args, impl_topk)
            torch.cuda.synchronize()
            ok, miss_frac = check_topk_indices(ref_topk, impl_topk, seq_lens)
            if not ok:
                all_pass = False
            line += f" {miss_frac:>10.4f} {'PASS' if ok else 'FAIL':>6}"
            if not ok:
                print(line)
                continue

        if MEASURE:
            ref_topk = alloc_topk_indices(batch_size)
            impl_topk = alloc_topk_indices(batch_size)
            ref_args  = (*args, ref_topk)
            impl_args = (*args, impl_topk)
            r_ms = bench(ref_fn, ref_args)
            i_ms = bench(impl_fn, impl_args)
            sp = r_ms / i_ms if i_ms > 0 else 0
            line += f" {r_ms:>8.3f} {i_ms:>8.3f} {sp:>7.2f}x"
            durations.append(i_ms)
            ref_ms_list.append(r_ms)
            speedups.append(sp)

        print(line)

    if CHECK:
        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")

    # ── Return structured results ──
    results = []
    for idx in range(len(workloads)):
        ax = workloads[idx]["workload"]["axes"]
        uuid = workloads[idx]["workload"]["uuid"][:8]
        r = {"workload": idx + 1, "uuid": uuid,
             "batch_size": ax["batch_size"],
             "max_num_pages": ax["max_num_pages"],
             "num_pages": ax["num_pages"]}
        if idx < len(speedups):
            r["ref_ms"] = ref_ms_list[idx]
            r["impl_ms"] = durations[idx]
            r["speedup"] = speedups[idx]
        results.append(r)

    if MEASURE and durations:
        n = len(durations)
        arith_ms = sum(durations) / n
        arith_sp = sum(speedups) / n
        geo_ms = math.exp(sum(math.log(x) for x in durations) / n)
        geo_sp = math.exp(sum(math.log(x) for x in speedups) / n) if all(x > 0 for x in speedups) else 0.0
        print()
        for label, ims, sp in [("Arith", arith_ms, arith_sp), ("Geo", geo_ms, geo_sp)]:
            print(f"{'':>3} {label:>10} {'':>3} {'':>5} {'':>6}  {'':>10} {'':>6} {ims:>8.3f} {sp:>7.2f}x")

    return {"all_pass": all_pass, "results": results}

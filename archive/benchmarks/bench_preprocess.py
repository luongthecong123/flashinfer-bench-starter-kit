#!/usr/bin/env python3
"""Benchmark the 3 preprocessing lines in impl_cutedsl.py across all 23 workloads."""
import torch
from pathlib import Path
from cook import get_inputs

HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
WARMUP = 50
ITERS = 200


def bench_reshape(ckv_cache, kpe_cache):
    """ckv_cache.reshape(-1, 512) + kpe_cache.reshape(-1, 64) — should be ~free (views)."""
    ckv_flat = ckv_cache.reshape(-1, HEAD_DIM_CKV)
    kpe_flat = kpe_cache.reshape(-1, HEAD_DIM_KPE)
    return ckv_flat, kpe_flat


def bench_valid_count(sparse_indices):
    """(sparse_indices != -1).sum(dim=-1).unsqueeze(0).to(torch.int32)"""
    return (sparse_indices != -1).sum(dim=-1).unsqueeze(0).to(torch.int32)


def bench_all(ckv_cache, kpe_cache, sparse_indices):
    """All 3 lines together."""
    ckv_flat = ckv_cache.reshape(-1, HEAD_DIM_CKV)
    kpe_flat = kpe_cache.reshape(-1, HEAD_DIM_KPE)
    valid_indices = (sparse_indices != -1).sum(dim=-1).unsqueeze(0).to(torch.int32)
    return ckv_flat, kpe_flat, valid_indices


def time_fn(fn, *args, warmup=WARMUP, iters=ITERS):
    """Time a GPU function using CUDA events. Returns mean time in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return (elapsed_ms / iters) * 1000  # convert ms → µs


def main():
    ROOT = Path(__file__).parent.parent
    JSONL = str(ROOT.parent / "flashinfer26dsa" / "mlsys26-contest" / "workloads" / "dsa_paged"
                / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl")

    print(f"{'WL':>3} {'T':>2} {'P':>6} | {'reshape(µs)':>12} {'valid_cnt(µs)':>14} {'all_3(µs)':>10}")
    print("-" * 70)

    reshape_times = []
    valid_times = []
    all_times = []

    for i, inp in enumerate(get_inputs(JSONL)):
        T, P = inp["num_tokens"], inp["num_pages"]
        ckv = inp["ckv_cache"]
        kpe = inp["kpe_cache"]
        si = inp["sparse_indices"]

        t_reshape = time_fn(bench_reshape, ckv, kpe)
        t_valid = time_fn(bench_valid_count, si)
        t_all = time_fn(bench_all, ckv, kpe, si)

        reshape_times.append(t_reshape)
        valid_times.append(t_valid)
        all_times.append(t_all)

        print(f"{i+1:>3} {T:>2} {P:>6} | {t_reshape:>12.2f} {t_valid:>14.2f} {t_all:>10.2f}")

    # Summary
    import statistics
    print("-" * 70)
    print(f"{'':>3} {'':>2} {'MEAN':>6} | {statistics.mean(reshape_times):>12.2f} "
          f"{statistics.mean(valid_times):>14.2f} {statistics.mean(all_times):>10.2f}")
    print(f"{'':>3} {'':>2} {'MED':>6} | {statistics.median(reshape_times):>12.2f} "
          f"{statistics.median(valid_times):>14.2f} {statistics.median(all_times):>10.2f}")
    print(f"{'':>3} {'':>2} {'MAX':>6} | {max(reshape_times):>12.2f} "
          f"{max(valid_times):>14.2f} {max(all_times):>10.2f}")
    print(f"{'':>3} {'':>2} {'MIN':>6} | {min(reshape_times):>12.2f} "
          f"{min(valid_times):>14.2f} {min(all_times):>10.2f}")

    geo = lambda xs: statistics.geometric_mean(xs)
    print(f"{'':>3} {'':>2} {'GEO':>6} | {geo(reshape_times):>12.2f} "
          f"{geo(valid_times):>14.2f} {geo(all_times):>10.2f}")


if __name__ == "__main__":
    main()

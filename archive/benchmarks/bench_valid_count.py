"""Benchmark different methods to find valid_count per row in sparse_indices.
Tests on all 23 real workloads, both CPU and GPU sparse_indices."""
import json, math, statistics, torch
from functools import partial
from pathlib import Path
from safetensors.torch import load_file

torch.backends.cuda.matmul.allow_tf32 = False

ROOT    = Path(__file__).parent.parent
CONTEST = ROOT.parent / "flashinfer26dsa" / "mlsys26-contest"
JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"


# ════════════════════════════════════════════
# GPU methods
# ════════════════════════════════════════════

global FLAG_CPU
FLAG_CPU = True
global FLAG_GPU
FLAG_GPU = True

def gpu_sum(si):
    """Count non-(-1) entries per row."""
    global FLAG_GPU
    result =  (si != -1).sum(dim=1)
    if FLAG_GPU:
        print("gpu_sum shape:", result.shape, "dtype:", result.dtype)
        FLAG_GPU = True
    return result

def gpu_argmax(si):
    """argmax of (si == -1) mask — first -1 position."""
    mask = (si == -1)
    has_padding = mask.any(dim=1)
    first_neg = mask.int().argmax(dim=1)
    return torch.where(has_padding, first_neg, si.shape[1])

def gpu_flip_argmax(si):
    """Flip + argmax of valid entries from the tail."""
    flipped = si.flip(dims=[1])
    first_valid = (flipped != -1).int().argmax(dim=1)
    return si.shape[1] - first_valid


# ════════════════════════════════════════════
# CPU methods
# ════════════════════════════════════════════

def cpu_sum(si):
    """Simple sum on CPU."""
    global FLAG_CPU
    result = (si != -1).sum(dim=1)
    
    if FLAG_CPU:
        print("cpu_sum shape:", result.shape, "dtype:", result.dtype)
        FLAG_CPU = True
    return result

def cpu_bisect(si):
    """Standard binary search per row."""
    T, topk = si.shape
    counts = torch.empty(T, dtype=torch.int64)
    for t in range(T):
        lo, hi = 0, topk
        row = si[t]
        while lo < hi:
            mid = (lo + hi) // 2
            if row[mid].item() == -1:
                hi = mid
            else:
                lo = mid + 1
        counts[t] = lo
    return counts

def cpu_bisect_with_hint(si, hint=1024):
    """Binary search with pre-computed starting hint.
    
    First probe at `hint` instead of middle. Halves the remaining
    search space on the correct side.
    """
    T, topk = si.shape
    hint = max(0, min(hint, topk))
    counts = torch.empty(T, dtype=torch.int64)
    for t in range(T):
        row = si[t]
        if hint < topk and row[hint].item() == -1:
            lo, hi = 0, hint
        else:
            lo, hi = hint + 1 if hint < topk else hint, topk
        while lo < hi:
            mid = (lo + hi) // 2
            if row[mid].item() == -1:
                hi = mid
            else:
                lo = mid + 1
        counts[t] = lo
    return counts

def cpu_march_with_hint(si, hint=1024):
    """Linear march from pre-computed hint.
    
    Probe hint position, then march sequentially left/right.
    Good cache behaviour for sequential access.
    """
    T, topk = si.shape
    hint = max(0, min(hint, topk - 1))
    counts = torch.empty(T, dtype=torch.int64)
    for t in range(T):
        row = si[t]
        if row[hint].item() == -1:
            pos = hint
            while pos > 0 and row[pos - 1].item() == -1:
                pos -= 1
            counts[t] = pos
        else:
            pos = hint + 1
            while pos < topk and row[pos].item() != -1:
                pos += 1
            counts[t] = pos
    return counts

def cpu_forward_scan(si):
    """March from position 0 rightward."""
    T, topk = si.shape
    counts = torch.empty(T, dtype=torch.int64)
    for t in range(T):
        row = si[t]
        pos = 0
        while pos < topk and row[pos].item() != -1:
            pos += 1
        counts[t] = pos
    return counts


# ════════════════════════════════════════════
# Benchmark harness
# ════════════════════════════════════════════

def bench_gpu(fn, si, warmup=50, iters=200):
    for _ in range(warmup):
        fn(si)
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in evs:
        s.record(); result = fn(si); e.record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in evs]
    return sum(times) / len(times) * 1000, result  # return in microseconds

def bench_cpu(fn, si, warmup=5, iters=20):
    for _ in range(warmup):
        fn(si)
    import time
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn(si)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds
    return sum(times) / len(times), result


def gmean(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Load all real workloads
    workloads = [json.loads(l) for l in open(JSONL)]
    sparse_list = []
    all_valids = []  # flat list of every per-token valid count
    for w in workloads:
        ax, inp = w["workload"]["axes"], w["workload"]["inputs"]
        T = ax["num_tokens"]
        sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
        si = sf[inp["sparse_indices"]["tensor_key"]]
        valids = [(si[t] != -1).sum().item() for t in range(T)]
        sparse_list.append((T, valids, si))
        all_valids.extend(valids)

    # ── Pre-compute offline statistics ──
    global_mean   = int(statistics.mean(all_valids))
    global_median = int(statistics.median(all_valids))
    per_wl_means   = [int(statistics.mean(v)) for _, v, _ in sparse_list]
    per_wl_medians = [int(statistics.median(v)) for _, v, _ in sparse_list]

    print(f"Offline statistics (computed from {len(all_valids)} tokens across {len(sparse_list)} workloads):")
    print(f"  Global mean  = {global_mean}")
    print(f"  Global median= {global_median}")
    print(f"  Per-workload means  : {per_wl_means}")
    print(f"  Per-workload medians: {per_wl_medians}")
    print()

    gpu_methods = [
        ("sum(!=−1)",       gpu_sum),
        ("argmax(==−1)",    gpu_argmax),
        ("flip+argmax",     gpu_flip_argmax),
    ]

    # CPU methods: static ones + hint-based ones built per-workload
    cpu_static = [
        ("cpu sum",          cpu_sum),
        ("cpu bisect",       cpu_bisect),
        ("cpu fwd scan",     cpu_forward_scan),
    ]

    hint_labels = [
        "bisect@gmean",   "bisect@gmedian",   "bisect@wlmean",   "bisect@wlmedian",
        "march@gmean",    "march@gmedian",    "march@wlmean",    "march@wlmedian",
    ]

    # ── GPU benchmark ──
    print("=" * 100)
    print("GPU sparse_indices")
    print("=" * 100)
    header = f"{'#':>3} {'T':>2} {'Valid':>40}"
    for name, _ in gpu_methods:
        header += f" {name:>16}"
    print(header)
    print("-" * len(header))

    gpu_times = {name: [] for name, _ in gpu_methods}
    for i, (T, valids, si_cpu) in enumerate(sparse_list):
        si_gpu = si_cpu.cuda()
        valids_str = ",".join(str(v) for v in valids)
        line = f"{i+1:>3} {T:>2} {valids_str:>40}"
        ref_result = None
        for name, fn in gpu_methods:
            us, result = bench_gpu(fn, si_gpu)
            result_cpu = result.cpu()
            if ref_result is None:
                ref_result = result_cpu
            else:
                assert torch.equal(ref_result, result_cpu), f"MISMATCH {name}: {result_cpu} vs {ref_result}"
            gpu_times[name].append(us)
            line += f" {us:>13.1f} us"
        print(line)

    line = f"{'':>3} {'':>2} {'GEOMEAN':>40}"
    for name, _ in gpu_methods:
        g = gmean(gpu_times[name])
        line += f" {g:>13.1f} us"
    print(line)

    # ── CPU benchmark ──
    all_cpu_names = [n for n, _ in cpu_static] + hint_labels
    col_w = 18
    print()
    print("=" * (50 + col_w * len(all_cpu_names)))
    print("CPU sparse_indices")
    print("=" * (50 + col_w * len(all_cpu_names)))
    header = f"{'#':>3} {'T':>2} {'Valid':>40}"
    for name in all_cpu_names:
        header += f" {name:>{col_w}}"
    print(header)
    print("-" * len(header))

    cpu_times = {name: [] for name, _ in cpu_static}
    for h in hint_labels:
        cpu_times[h] = []

    for i, (T, valids, si_cpu) in enumerate(sparse_list):
        valids_str = ",".join(str(v) for v in valids)
        line = f"{i+1:>3} {T:>2} {valids_str:>40}"

        # Build hint-based methods for this workload
        wlm, wlmed = per_wl_means[i], per_wl_medians[i]
        hint_methods = [
            ("bisect@gmean",    partial(cpu_bisect_with_hint, hint=global_mean)),
            ("bisect@gmedian",  partial(cpu_bisect_with_hint, hint=global_median)),
            ("bisect@wlmean",   partial(cpu_bisect_with_hint, hint=wlm)),
            ("bisect@wlmedian", partial(cpu_bisect_with_hint, hint=wlmed)),
            ("march@gmean",     partial(cpu_march_with_hint, hint=global_mean)),
            ("march@gmedian",   partial(cpu_march_with_hint, hint=global_median)),
            ("march@wlmean",    partial(cpu_march_with_hint, hint=wlm)),
            ("march@wlmedian",  partial(cpu_march_with_hint, hint=wlmed)),
        ]

        all_methods = list(cpu_static) + hint_methods
        ref_result = None
        for name, fn in all_methods:
            us, result = bench_cpu(fn, si_cpu)
            if ref_result is None:
                ref_result = result
            else:
                assert torch.equal(ref_result, result), f"MISMATCH {name}: {result} vs {ref_result}"
            cpu_times[name].append(us)
            line += f" {us:>{col_w - 3}.1f} us"
        print(line)

    line = f"{'':>3} {'':>2} {'GEOMEAN':>40}"
    for name in all_cpu_names:
        g = gmean(cpu_times[name])
        line += f" {g:>{col_w - 3}.1f} us"
    print(line)

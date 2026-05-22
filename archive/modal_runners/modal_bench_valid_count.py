"""Run bench_valid_count on Modal B200 GPU.
Reuses the same image spec as flashinfer-bench (cached, no rebuild)."""
import modal

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
)

@app.function(image=image, gpu="B200:1", timeout=600, volumes={TRACE_SET_PATH: trace_volume})
def run_bench():
    import json, math, statistics, time, torch
    from functools import partial
    from pathlib import Path
    from safetensors.torch import load_file

    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"GPU: {torch.cuda.get_device_name(0)}\n", flush=True)

    # Find the JSONL file inside the trace volume
    root = Path(TRACE_SET_PATH)
    jsonl_candidates = list(root.rglob("dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"))
    if not jsonl_candidates:
        raise FileNotFoundError(f"No JSONL found under {root}")
    JSONL = jsonl_candidates[0]
    contest_dir = JSONL.parent.parent.parent
    print(f"JSONL: {JSONL}", flush=True)

    # Load all workloads
    workloads = [json.loads(l) for l in open(JSONL)]
    sparse_list = []
    all_valids = []
    for w in workloads:
        ax, inp = w["workload"]["axes"], w["workload"]["inputs"]
        T = ax["num_tokens"]
        sf = load_file(str(contest_dir / inp["sparse_indices"]["path"]))
        si = sf[inp["sparse_indices"]["tensor_key"]]
        valids = [(si[t] != -1).sum().item() for t in range(T)]
        sparse_list.append((T, valids, si))
        all_valids.extend(valids)

    global_mean   = int(statistics.mean(all_valids))
    global_median = int(statistics.median(all_valids))
    per_wl_means   = [int(statistics.mean(v)) for _, v, _ in sparse_list]
    per_wl_medians = [int(statistics.median(v)) for _, v, _ in sparse_list]

    # ── GPU methods ──
    def gpu_sum(si):
        return (si != -1).sum(dim=1)
    def gpu_argmax(si):
        mask = (si == -1)
        has_padding = mask.any(dim=1)
        first_neg = mask.int().argmax(dim=1)
        return torch.where(has_padding, first_neg, si.shape[1])
    def gpu_flip_argmax(si):
        flipped = si.flip(dims=[1])
        first_valid = (flipped != -1).int().argmax(dim=1)
        return si.shape[1] - first_valid

    # ── CPU methods ──
    def cpu_sum(si):
        return (si != -1).sum(dim=1)
    def cpu_bisect(si):
        T, topk = si.shape
        counts = torch.empty(T, dtype=torch.int64)
        for t in range(T):
            lo, hi = 0, topk
            row = si[t]
            while lo < hi:
                mid = (lo + hi) // 2
                if row[mid].item() == -1: hi = mid
                else: lo = mid + 1
            counts[t] = lo
        return counts
    def cpu_bisect_with_hint(si, hint=1024):
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
                if row[mid].item() == -1: hi = mid
                else: lo = mid + 1
            counts[t] = lo
        return counts
    def cpu_march_with_hint(si, hint=1024):
        T, topk = si.shape
        hint = max(0, min(hint, topk - 1))
        counts = torch.empty(T, dtype=torch.int64)
        for t in range(T):
            row = si[t]
            if row[hint].item() == -1:
                pos = hint
                while pos > 0 and row[pos - 1].item() == -1: pos -= 1
                counts[t] = pos
            else:
                pos = hint + 1
                while pos < topk and row[pos].item() != -1: pos += 1
                counts[t] = pos
        return counts
    def cpu_forward_scan(si):
        T, topk = si.shape
        counts = torch.empty(T, dtype=torch.int64)
        for t in range(T):
            row = si[t]
            pos = 0
            while pos < topk and row[pos].item() != -1: pos += 1
            counts[t] = pos
        return counts

    def bench_gpu(fn, si, warmup=50, iters=200):
        for _ in range(warmup): fn(si)
        torch.cuda.synchronize()
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
        for s, e in evs:
            s.record(); result = fn(si); e.record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in evs]
        return sum(times) / len(times) * 1000, result

    def bench_cpu(fn, si, warmup=5, iters=20):
        for _ in range(warmup): fn(si)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            result = fn(si)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
        return sum(times) / len(times), result

    def gmean(vals):
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    output_lines = []
    def prt(s=""):
        print(s, flush=True)
        output_lines.append(s)

    prt(f"GPU: {torch.cuda.get_device_name(0)}")
    prt()
    prt(f"Offline statistics ({len(all_valids)} tokens, {len(sparse_list)} workloads):")
    prt(f"  Global mean={global_mean}, median={global_median}")
    prt(f"  Per-wl means  : {per_wl_means}")
    prt(f"  Per-wl medians: {per_wl_medians}")
    prt()

    gpu_methods = [
        ("sum(!=−1)",    gpu_sum),
        ("argmax(==−1)", gpu_argmax),
        ("flip+argmax",  gpu_flip_argmax),
    ]
    cpu_static = [
        ("cpu sum",      cpu_sum),
        ("cpu bisect",   cpu_bisect),
        ("cpu fwd scan", cpu_forward_scan),
    ]
    hint_labels = [
        "bisect@gmean", "bisect@gmedian", "bisect@wlmean", "bisect@wlmedian",
        "march@gmean",  "march@gmedian",  "march@wlmean",  "march@wlmedian",
    ]

    # GPU benchmark
    prt("=" * 100)
    prt("GPU sparse_indices")
    prt("=" * 100)
    header = f"{'#':>3} {'T':>2} {'Valid':>40}"
    for name, _ in gpu_methods: header += f" {name:>16}"
    prt(header)
    prt("-" * len(header))

    gpu_times = {n: [] for n, _ in gpu_methods}
    for i, (T, valids, si_cpu) in enumerate(sparse_list):
        si_gpu = si_cpu.cuda()
        vs = ",".join(str(v) for v in valids)
        line = f"{i+1:>3} {T:>2} {vs:>40}"
        ref = None
        for name, fn in gpu_methods:
            us, result = bench_gpu(fn, si_gpu)
            rc = result.cpu()
            if ref is None: ref = rc
            else: assert torch.equal(ref, rc), f"MISMATCH {name}"
            gpu_times[name].append(us)
            line += f" {us:>13.1f} us"
        prt(line)
    line = f"{'':>3} {'':>2} {'GEOMEAN':>40}"
    for name, _ in gpu_methods:
        line += f" {gmean(gpu_times[name]):>13.1f} us"
    prt(line)

    # CPU benchmark
    all_cpu = [n for n, _ in cpu_static] + hint_labels
    cw = 18
    prt()
    prt("=" * (50 + cw * len(all_cpu)))
    prt("CPU sparse_indices")
    prt("=" * (50 + cw * len(all_cpu)))
    header = f"{'#':>3} {'T':>2} {'Valid':>40}"
    for n in all_cpu: header += f" {n:>{cw}}"
    prt(header)
    prt("-" * len(header))

    cpu_times = {n: [] for n in all_cpu}
    for n, _ in cpu_static: cpu_times.setdefault(n, [])

    for i, (T, valids, si_cpu) in enumerate(sparse_list):
        vs = ",".join(str(v) for v in valids)
        line = f"{i+1:>3} {T:>2} {vs:>40}"
        wlm, wlmed = per_wl_means[i], per_wl_medians[i]
        hint_methods = [
            ("bisect@gmean",    partial(cpu_bisect_with_hint, hint=global_mean)),
            ("bisect@gmedian",  partial(cpu_bisect_with_hint, hint=global_median)),
            ("bisect@wlmean",   partial(cpu_bisect_with_hint, hint=wlm)),
            ("bisect@wlmedian", partial(cpu_bisect_with_hint, hint=wlmed)),
            ("march@gmean",     partial(cpu_march_with_hint,  hint=global_mean)),
            ("march@gmedian",   partial(cpu_march_with_hint,  hint=global_median)),
            ("march@wlmean",    partial(cpu_march_with_hint,  hint=wlm)),
            ("march@wlmedian",  partial(cpu_march_with_hint,  hint=wlmed)),
        ]
        all_methods = list(cpu_static) + hint_methods
        ref = None
        for name, fn in all_methods:
            us, result = bench_cpu(fn, si_cpu)
            if ref is None: ref = result
            else: assert torch.equal(ref, result), f"MISMATCH {name}"
            cpu_times[name].append(us)
            line += f" {us:>{cw-3}.1f} us"
        prt(line)

    line = f"{'':>3} {'':>2} {'GEOMEAN':>40}"
    for n in all_cpu:
        line += f" {gmean(cpu_times[n]):>{cw-3}.1f} us"
    prt(line)

    return "\n".join(output_lines)


@app.local_entrypoint()
def bench_valid_count():
    result = run_bench.remote()
    print("\n\n=== FULL OUTPUT ===\n")
    print(result)

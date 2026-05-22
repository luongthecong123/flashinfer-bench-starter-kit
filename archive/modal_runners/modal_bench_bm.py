#!/usr/bin/env python3
"""Benchmark gather kernel with different BM values on Modal B200.

Wave analysis (B200 = 148 SMs, 256 threads/block, max 8 blocks/SM = 1184 blocks/wave):
  BM=4  → 512 blocks/token → single wave for T≤2
  BM=8  → 256 blocks/token → single wave for T≤4
  BM=16 → 128 blocks/token → single wave for T≤9
  BM=32 →  64 blocks/token → single wave for T≤18
"""
import modal
from pathlib import Path

ZEN_DIR = Path(__file__).parent
DEV_DIR = ZEN_DIR.parent / "dev"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "safetensors", "packaging", "numpy", "nvidia-cutlass-dsl", "ninja", "apache-tvm-ffi")
    .add_local_dir(ZEN_DIR, remote_path="/root/zen")
    .add_local_dir(DEV_DIR, remote_path="/root/dev")
)

app = modal.App("gather-bench-bm", image=image)
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)


@app.function(
    gpu="B200:1",
    timeout=900,
    volumes={"/data": trace_volume},
)
def run_bench():
    import sys, math
    sys.path.insert(0, "/root/dev")
    sys.path.insert(0, "/root/zen")

    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor
    from cook import get_inputs
    from functools import reduce
    import operator

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    JSONL = "/data/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    HEAD_DIM_CKV, HEAD_DIM_KPE, TOPK = 512, 64, 2048
    WARMUP, ITERS = 20, 100
    BM_VALUES = [4, 8, 16, 32]
    NUM_ROUNDS = 3

    # ── Helpers ──
    @cute.jit
    def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
        for i in range(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val

    def fake_wrapper(dtype, shape, stride_order, assumed_align):
        return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=assumed_align)

    # ── Parameterized Gather kernel ──
    class GatherBM():
        def __init__(self, BM):
            self.BM = BM
            self.num_threads = 256
            self.warp_size = cute.arch.WARP_SIZE

        @cute.jit
        def __call__(self, ckv_cache: cute.Tensor, kpe_cache: cute.Tensor,
                     sparse_indices: cute.Tensor, kc: cute.Tensor, Kp: cute.Tensor,
                     max_valid: cute.Tensor):
            T, topk = sparse_indices.shape
            self.kernel(ckv_cache, kpe_cache, sparse_indices, kc, Kp, max_valid).launch(
                grid=[T, topk // self.BM, 1], block=[self.num_threads, 1, 1])

        @cute.kernel
        def kernel(self, ckv_cache: cute.Tensor, kpe_cache: cute.Tensor,
                   sparse_indices: cute.Tensor, kc: cute.Tensor, Kp: cute.Tensor,
                   max_valid: cute.Tensor):
            N, dkc = ckv_cache.shape
            N2, dkp = kpe_cache.shape
            T, topk = sparse_indices.shape
            bidx, bidy, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx = cute.arch.warp_idx()
            warp_idx = cute.arch.make_warp_uniform(warp_idx)
            lane_idx = cute.arch.lane_idx()

            for m in range(self.BM):
                row = bidy * self.BM + m
                idx = sparse_indices[bidx, row]
                if idx >= cutlass.Int32(0):
                    for d in range(tidx, dkc, self.num_threads):
                        kc[bidx, row, d] = ckv_cache[idx, d]
                    for d in range(tidx, dkp, self.num_threads):
                        Kp[bidx, row, d] = kpe_cache[idx, d]
                else:
                    for d in range(tidx, dkc, self.num_threads):
                        kc[bidx, row, d] = cutlass.BFloat16(0)
                    for d in range(tidx, dkp, self.num_threads):
                        Kp[bidx, row, d] = cutlass.BFloat16(0)

            if bidy == 0:
                num_warps = self.num_threads // self.warp_size
                allocator = cutlass.utils.SmemAllocator()
                smem_counts = allocator.allocate_tensor(
                    cutlass.Int32, cute.make_layout((self.warp_size,), stride=(1,)), 4, None)
                local_count = cutlass.Int32(0)
                for i in range(topk // self.num_threads):
                    if sparse_indices[bidx, tidx * (topk // self.num_threads) + i] >= cutlass.Int32(0):
                        local_count += cutlass.Int32(1)
                warp_sum = warp_reduce(local_count, lambda a, b: a + b)
                if lane_idx == 0:
                    smem_counts[warp_idx] = warp_sum
                cute.arch.sync_threads()
                if warp_idx == 0:
                    partial = smem_counts[lane_idx]
                    total = warp_reduce(partial, lambda a, b: a + b, width=self.num_threads // self.warp_size)
                    if lane_idx == 0:
                        max_valid[bidx] = total

    # ── Compile all BM variants ──
    def compile_gather(bm):
        T = cute.sym_int()
        N = cute.sym_int()
        ckv = fake_wrapper(cute.BFloat16, (N, HEAD_DIM_CKV), (1, 0), 16)
        kpe = fake_wrapper(cute.BFloat16, (N, HEAD_DIM_KPE), (1, 0), 16)
        si  = fake_wrapper(cute.Int32, (T, TOPK), (1, 0), 4)
        kc  = fake_wrapper(cute.BFloat16, (T, TOPK, HEAD_DIM_CKV), (2, 1, 0), 16)
        Kp  = fake_wrapper(cute.BFloat16, (T, TOPK, HEAD_DIM_KPE), (2, 1, 0), 16)
        mv  = fake_wrapper(cute.Int32, (T,), (0,), 4)
        return cute.compile(GatherBM(bm), ckv, kpe, si, kc, Kp, mv, options="--enable-tvm-ffi")

    compiled = {}
    for bm in BM_VALUES:
        print(f"Compiling BM={bm} (grid per token: {TOPK//bm} blocks)...")
        compiled[bm] = compile_gather(bm)
    print("All compiled.\n")

    # ── Load workloads ──
    workloads = []
    for inp in get_inputs(JSONL):
        T = inp["num_tokens"]
        ckv_flat = inp["ckv_cache"].reshape(-1, HEAD_DIM_CKV).contiguous()
        kpe_flat = inp["kpe_cache"].reshape(-1, HEAD_DIM_KPE).contiguous()
        si = inp["sparse_indices"]
        workloads.append((T, ckv_flat, kpe_flat, si))

    # Categorize: tiny (T<=2), medium (3<=T<=5), large (T>=6)
    tiny_idx   = [i for i, (T, *_) in enumerate(workloads) if T <= 2]
    large_idx  = [i for i, (T, *_) in enumerate(workloads) if T >= 6]
    medium_idx = [i for i, (T, *_) in enumerate(workloads) if 3 <= T <= 5]

    print(f"Workload categories:")
    print(f"  Tiny  (T<=2): {len(tiny_idx)} workloads, T values: {[workloads[i][0] for i in tiny_idx]}")
    print(f"  Medium(3-5):  {len(medium_idx)} workloads, T values: {[workloads[i][0] for i in medium_idx]}")
    print(f"  Large (T>=6): {len(large_idx)} workloads, T values: {[workloads[i][0] for i in large_idx]}")
    print()

    # ── Benchmark function ──
    def bench_one(fn, ckv_flat, kpe_flat, si, T):
        kc = torch.zeros(T, TOPK, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")
        Kp = torch.zeros(T, TOPK, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cuda")
        mv = torch.zeros(T, dtype=torch.int32, device="cuda")
        for _ in range(WARMUP):
            fn(ckv_flat, kpe_flat, si, kc, Kp, mv)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            fn(ckv_flat, kpe_flat, si, kc, Kp, mv)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / ITERS * 1000  # us

    def geomean(vals):
        return reduce(operator.mul, vals) ** (1/len(vals)) if vals else 0

    # ── Run rounds ──
    # all_times[bm][workload_idx] = list of times across rounds
    all_times = {bm: {i: [] for i in range(len(workloads))} for bm in BM_VALUES}

    for rnd in range(NUM_ROUNDS):
        print(f"{'='*70}")
        print(f" ROUND {rnd+1}/{NUM_ROUNDS}")
        print(f"{'='*70}")

        bm_header = "".join(f"  BM={bm:>2} (us)" for bm in BM_VALUES)
        print(f"{'#':>3} {'T':>2}{bm_header}  {'Best':>6}")
        print("-" * 70)

        for i, (T, ckv_flat, kpe_flat, si) in enumerate(workloads):
            times = {}
            for bm in BM_VALUES:
                t = bench_one(compiled[bm], ckv_flat, kpe_flat, si, T)
                times[bm] = t
                all_times[bm][i].append(t)
            best_bm = min(times, key=times.get)
            row = f"{i:>3} {T:>2}"
            for bm in BM_VALUES:
                marker = " *" if bm == best_bm else "  "
                row += f"  {times[bm]:>9.1f}{marker}"
            row += f"  BM={best_bm:>2}"
            print(row)

        # Per-round geomean
        print("-" * 70)
        row = f"{'GM':>3} {'':>2}"
        gm_times = {}
        for bm in BM_VALUES:
            vals = [all_times[bm][i][-1] for i in range(len(workloads))]
            gm = geomean(vals)
            gm_times[bm] = gm
            best_mark = " *" if bm == min(gm_times, key=gm_times.get) else "  "
            row += f"  {gm:>9.1f}{best_mark}"
        row += f"  BM={min(gm_times, key=gm_times.get):>2}"
        print(row)
        print()

    # ── Final summary across rounds ──
    print(f"\n{'='*70}")
    print(f" FINAL SUMMARY ({NUM_ROUNDS} rounds averaged)")
    print(f"{'='*70}")

    def avg_geomean(bm, idx_list):
        """Avg across rounds of per-round geomean for workloads in idx_list."""
        round_gms = []
        for rnd in range(NUM_ROUNDS):
            vals = [all_times[bm][i][rnd] for i in idx_list]
            round_gms.append(geomean(vals))
        return sum(round_gms) / len(round_gms) if round_gms else 0

    categories = [
        ("ALL",    list(range(len(workloads)))),
        ("TINY",   tiny_idx),
        ("LARGE",  large_idx),
    ]

    for cat_name, idx_list in categories:
        if not idx_list:
            continue
        print(f"\n  {cat_name} workloads ({len(idx_list)} workloads):")
        bm_header = "".join(f"  BM={bm:>2}" for bm in BM_VALUES)
        print(f"    {'':>8}{bm_header}")

        gm_vals = {}
        for bm in BM_VALUES:
            gm_vals[bm] = avg_geomean(bm, idx_list)

        row = f"    {'GM (us)':>8}"
        for bm in BM_VALUES:
            marker = " *" if bm == min(gm_vals, key=gm_vals.get) else "  "
            row += f"  {gm_vals[bm]:>4.1f}{marker}"
        print(row)

        # Speedup relative to BM=4
        row = f"    {'vs BM=4':>8}"
        base = gm_vals[4]
        for bm in BM_VALUES:
            ratio = gm_vals[bm] / base if base > 0 else 0
            row += f"  {ratio:>5.3f} "
        print(row)

    # ── Wave info ──
    print(f"\n  Wave analysis (148 SMs × 8 blocks/SM = 1184 max blocks/wave):")
    for bm in BM_VALUES:
        bpt = TOPK // bm
        for cat_name, idx_list in categories:
            t_vals = sorted(set(workloads[i][0] for i in idx_list))
            waves = {t: math.ceil(t * bpt / 1184) for t in t_vals}
            wave_str = ", ".join(f"T={t}→{w}w" for t, w in waves.items())
            print(f"    BM={bm:>2} {cat_name:>5}: {wave_str}")


@app.local_entrypoint()
def main():
    run_bench.remote()

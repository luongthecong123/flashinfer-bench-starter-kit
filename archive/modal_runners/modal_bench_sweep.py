#!/usr/bin/env python3
"""Benchmark gather kernel sweeping BM × num_threads on Modal B200."""
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

app = modal.App("gather-bench-sweep", image=image)
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)


@app.function(
    gpu="B200:1",
    timeout=1800,
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

    BM_VALUES = [1, 2, 4, 8]
    NT_VALUES = [128, 256, 512]
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
    class GatherParam():
        def __init__(self, BM, num_threads):
            self.BM = BM
            self.num_threads = num_threads
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

    # ── Compile all variants ──
    configs = [(bm, nt) for bm in BM_VALUES for nt in NT_VALUES]

    def compile_gather(bm, nt):
        T = cute.sym_int()
        N = cute.sym_int()
        ckv = fake_wrapper(cute.BFloat16, (N, HEAD_DIM_CKV), (1, 0), 16)
        kpe = fake_wrapper(cute.BFloat16, (N, HEAD_DIM_KPE), (1, 0), 16)
        si  = fake_wrapper(cute.Int32, (T, TOPK), (1, 0), 4)
        kc  = fake_wrapper(cute.BFloat16, (T, TOPK, HEAD_DIM_CKV), (2, 1, 0), 16)
        Kp  = fake_wrapper(cute.BFloat16, (T, TOPK, HEAD_DIM_KPE), (2, 1, 0), 16)
        mv  = fake_wrapper(cute.Int32, (T,), (0,), 4)
        return cute.compile(GatherParam(bm, nt), ckv, kpe, si, kc, Kp, mv, options="--enable-tvm-ffi")

    compiled = {}
    for bm, nt in configs:
        blocks_per_tok = TOPK // bm
        elems_per_thread = HEAD_DIM_CKV // nt
        print(f"Compiling BM={bm:>2} NT={nt:>3}  ({blocks_per_tok} blocks/tok, {elems_per_thread} ckv elems/thread)...")
        compiled[(bm, nt)] = compile_gather(bm, nt)
    print(f"All {len(configs)} variants compiled.\n")

    # ── Load workloads ──
    workloads = []
    for inp in get_inputs(JSONL):
        T = inp["num_tokens"]
        ckv_flat = inp["ckv_cache"].reshape(-1, HEAD_DIM_CKV).contiguous()
        kpe_flat = inp["kpe_cache"].reshape(-1, HEAD_DIM_KPE).contiguous()
        si = inp["sparse_indices"]
        workloads.append((T, ckv_flat, kpe_flat, si))

    tiny_idx  = [i for i, (T, *_) in enumerate(workloads) if T <= 2]
    large_idx = [i for i, (T, *_) in enumerate(workloads) if T >= 6]

    print(f"Workloads: {len(workloads)} total, {len(tiny_idx)} tiny (T<=2), {len(large_idx)} large (T>=6)")
    print(f"T values: {sorted(set(w[0] for w in workloads))}\n")

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
    # all_times[(bm,nt)][workload_idx] = list of times across rounds
    all_times = {cfg: {i: [] for i in range(len(workloads))} for cfg in configs}

    for rnd in range(NUM_ROUNDS):
        print(f"\n{'='*80}")
        print(f" ROUND {rnd+1}/{NUM_ROUNDS}")
        print(f"{'='*80}")

        cfg_header = "".join(f" {bm}/{nt:>3}" for bm, nt in configs)
        print(f"{'#':>3} {'T':>2}{cfg_header}  Best")
        print("-" * 80)

        for i, (T, ckv_flat, kpe_flat, si) in enumerate(workloads):
            times = {}
            for cfg in configs:
                t = bench_one(compiled[cfg], ckv_flat, kpe_flat, si, T)
                times[cfg] = t
                all_times[cfg][i].append(t)
            best = min(times, key=times.get)
            row = f"{i:>3} {T:>2}"
            for cfg in configs:
                v = times[cfg]
                marker = "*" if cfg == best else " "
                row += f" {v:>5.1f}{marker}"
            row += f"  {best[0]}/{best[1]}"
            print(row)

        # Geomean row
        print("-" * 80)
        row = f"{'GM':>3} {'':>2}"
        gm = {}
        for cfg in configs:
            vals = [all_times[cfg][i][-1] for i in range(len(workloads))]
            gm[cfg] = geomean(vals)
        best_gm = min(gm, key=gm.get)
        for cfg in configs:
            marker = "*" if cfg == best_gm else " "
            row += f" {gm[cfg]:>5.1f}{marker}"
        row += f"  {best_gm[0]}/{best_gm[1]}"
        print(row)

    # ── Final summary ──
    print(f"\n{'='*80}")
    print(f" FINAL SUMMARY ({NUM_ROUNDS} rounds averaged)")
    print(f"{'='*80}")

    def avg_geomean(cfg, idx_list):
        round_gms = []
        for rnd in range(NUM_ROUNDS):
            vals = [all_times[cfg][i][rnd] for i in idx_list]
            round_gms.append(geomean(vals))
        return sum(round_gms) / len(round_gms)

    categories = [
        ("ALL",   list(range(len(workloads)))),
        ("TINY",  tiny_idx),
        ("LARGE", large_idx),
    ]

    for cat_name, idx_list in categories:
        if not idx_list:
            continue
        print(f"\n  {cat_name} ({len(idx_list)} workloads):")

        # Table header
        print(f"  {'BM':>4} \\ NT", end="")
        for nt in NT_VALUES:
            print(f"  {nt:>7}", end="")
        print()
        print(f"  {'-'*4}------", end="")
        for _ in NT_VALUES:
            print(f"  {'-'*7}", end="")
        print()

        gm_all = {}
        for cfg in configs:
            gm_all[cfg] = avg_geomean(cfg, idx_list)
        best_cfg = min(gm_all, key=gm_all.get)

        for bm in BM_VALUES:
            print(f"  {bm:>4}     ", end="")
            for nt in NT_VALUES:
                cfg = (bm, nt)
                marker = " *" if cfg == best_cfg else "  "
                print(f"  {gm_all[cfg]:>5.1f}{marker}", end="")
            print()

        print(f"\n  Best: BM={best_cfg[0]}, NT={best_cfg[1]} → {gm_all[best_cfg]:.1f} us")

    # ── Config properties ──
    print(f"\n  Config properties (B200: 148 SMs, max 2048 threads/SM):")
    print(f"  {'BM':>4} {'NT':>4} {'blk/tok':>8} {'elem/thr':>9} {'blk/SM':>7} {'waves(T=1)':>11} {'waves(T=8)':>11}")
    for bm, nt in configs:
        bpt = TOPK // bm
        ept = HEAD_DIM_CKV // nt
        max_blk_sm = 2048 // nt
        max_blk_wave = 148 * max_blk_sm
        w1 = math.ceil(1 * bpt / max_blk_wave)
        w8 = math.ceil(8 * bpt / max_blk_wave)
        print(f"  {bm:>4} {nt:>4} {bpt:>8} {ept:>9} {max_blk_sm:>7} {w1:>11} {w8:>11}")


@app.local_entrypoint()
def main():
    run_bench.remote()

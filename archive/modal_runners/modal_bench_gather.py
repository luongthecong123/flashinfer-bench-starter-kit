#!/usr/bin/env python3
"""Benchmark gather kernel Option 1 vs Option 2 on Modal B200."""
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

app = modal.App("gather-bench", image=image)
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)


@app.function(
    gpu="B200:1",
    timeout=600,
    volumes={"/data": trace_volume},
)
def run_bench():
    import sys, os, time, math
    sys.path.insert(0, "/root/dev")
    sys.path.insert(0, "/root/zen")

    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor
    from cook import get_inputs

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    JSONL = "/data/workloads/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    HEAD_DIM_CKV, HEAD_DIM_KPE, TOPK = 512, 64, 2048
    WARMUP, ITERS = 20, 100

    # ── Helper ──
    def warp_reduce(val: cute.Numeric, op: callable, width: cutlass.Constexpr = 32) -> cute.Numeric:
        for i in range(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val

    def fake_wrapper(dtype, shape, stride_order, assumed_align):
        return make_fake_compact_tensor(dtype=dtype, shape=shape, stride_order=stride_order, assumed_align=assumed_align)

    # ── Option 1: all-warp reduction via smem ─────────────────────────────
    class GatherOpt1():
        def __init__(self):
            self.BM = 4
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

    # ── Option 2: single-warp reduction ───────────────────────────────────
    class GatherOpt2():
        def __init__(self):
            self.BM = 4
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
                if warp_idx == 0:
                    local_count = cutlass.Int32(0)
                    for i in range(topk // self.warp_size):
                        if sparse_indices[bidx, lane_idx * (topk // self.warp_size) + i] >= cutlass.Int32(0):
                            local_count += cutlass.Int32(1)
                    total = warp_reduce(local_count, lambda a, b: a + b)
                    if lane_idx == 0:
                        max_valid[bidx] = total

    # ── Compile both ──────────────────────────────────────────────────────
    def compile_gather(cls):
        T = cute.sym_int()
        N = cute.sym_int()
        ckv = fake_wrapper(cute.BFloat16, (N, HEAD_DIM_CKV), (1, 0), 16)
        kpe = fake_wrapper(cute.BFloat16, (N, HEAD_DIM_KPE), (1, 0), 16)
        si  = fake_wrapper(cute.Int32, (T, TOPK), (1, 0), 4)
        kc  = fake_wrapper(cute.BFloat16, (T, TOPK, HEAD_DIM_CKV), (2, 1, 0), 16)
        Kp  = fake_wrapper(cute.BFloat16, (T, TOPK, HEAD_DIM_KPE), (2, 1, 0), 16)
        mv  = fake_wrapper(cute.Int32, (T,), (0,), 4)
        return cute.compile(cls(), ckv, kpe, si, kc, Kp, mv, options="--enable-tvm-ffi")

    print("Compiling Option 1 (all-warp smem reduction)...")
    opt1 = compile_gather(GatherOpt1)
    print("Compiling Option 2 (single-warp reduction)...")
    opt2 = compile_gather(GatherOpt2)
    print("Done.\n")

    # ── Benchmark ─────────────────────────────────────────────────────────
    NUM_ROUNDS = 5

    def bench_one(compiled_fn, ckv_flat, kpe_flat, sparse_indices, T):
        kc = torch.zeros(T, TOPK, HEAD_DIM_CKV, dtype=torch.bfloat16, device="cuda")
        Kp = torch.zeros(T, TOPK, HEAD_DIM_KPE, dtype=torch.bfloat16, device="cuda")
        mv = torch.zeros(T, dtype=torch.int32, device="cuda")
        # warmup
        for _ in range(WARMUP):
            compiled_fn(ckv_flat, kpe_flat, sparse_indices, kc, Kp, mv)
        torch.cuda.synchronize()
        # timed
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            compiled_fn(ckv_flat, kpe_flat, sparse_indices, kc, Kp, mv)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / ITERS, mv

    # Load workloads once
    workloads = []
    for inp in get_inputs(JSONL):
        T = inp["num_tokens"]
        ckv_flat = inp["ckv_cache"].reshape(-1, HEAD_DIM_CKV).contiguous()
        kpe_flat = inp["kpe_cache"].reshape(-1, HEAD_DIM_KPE).contiguous()
        si = inp["sparse_indices"]
        workloads.append((T, ckv_flat, kpe_flat, si))

    from functools import reduce
    import operator

    round_gm1s = []
    round_gm2s = []

    for rnd in range(NUM_ROUNDS):
        print(f"\n{'='*60}")
        print(f" ROUND {rnd+1}/{NUM_ROUNDS}")
        print(f"{'='*60}")
        print(f"{'#':>3} {'T':>2} {'Opt1 (us)':>10} {'Opt2 (us)':>10} {'Ratio':>7} {'Winner':>8}")
        print("-" * 50)

        geomeans = {"opt1": [], "opt2": []}

        for i, (T, ckv_flat, kpe_flat, si) in enumerate(workloads):
            t1_ms, mv1 = bench_one(opt1, ckv_flat, kpe_flat, si, T)
            t2_ms, mv2 = bench_one(opt2, ckv_flat, kpe_flat, si, T)

            assert (mv1 == mv2).all(), f"Workload {i}: max_valid mismatch!"

            t1_us = t1_ms * 1000
            t2_us = t2_ms * 1000
            ratio = t1_us / t2_us if t2_us > 0 else float("inf")
            winner = "Opt1" if t1_us < t2_us else "Opt2"
            geomeans["opt1"].append(t1_us)
            geomeans["opt2"].append(t2_us)

            print(f"{i:>3} {T:>2} {t1_us:>10.1f} {t2_us:>10.1f} {ratio:>7.3f} {winner:>8}")

        n = len(geomeans["opt1"])
        gm1 = reduce(operator.mul, geomeans["opt1"]) ** (1/n)
        gm2 = reduce(operator.mul, geomeans["opt2"]) ** (1/n)
        round_gm1s.append(gm1)
        round_gm2s.append(gm2)
        print("-" * 50)
        print(f"{'GM':>3} {'':>2} {gm1:>10.1f} {gm2:>10.1f} {gm1/gm2:>7.3f} {'Opt1' if gm1 < gm2 else 'Opt2':>8}")

    # ── Summary across rounds ──
    print(f"\n{'='*60}")
    print(f" SUMMARY ({NUM_ROUNDS} rounds)")
    print(f"{'='*60}")
    print(f"{'Round':>6} {'Opt1 GM':>10} {'Opt2 GM':>10} {'Ratio':>7} {'Winner':>8}")
    print("-" * 50)
    for rnd, (g1, g2) in enumerate(zip(round_gm1s, round_gm2s)):
        print(f"{rnd+1:>6} {g1:>10.1f} {g2:>10.1f} {g1/g2:>7.3f} {'Opt1' if g1 < g2 else 'Opt2':>8}")
    avg1 = sum(round_gm1s) / NUM_ROUNDS
    avg2 = sum(round_gm2s) / NUM_ROUNDS
    min1 = min(round_gm1s)
    min2 = min(round_gm2s)
    print("-" * 50)
    print(f"{'AVG':>6} {avg1:>10.1f} {avg2:>10.1f} {avg1/avg2:>7.3f} {'Opt1' if avg1 < avg2 else 'Opt2':>8}")
    print(f"{'BEST':>6} {min1:>10.1f} {min2:>10.1f} {min1/min2:>7.3f} {'Opt1' if min1 < min2 else 'Opt2':>8}")


@app.local_entrypoint()
def main():
    run_bench.remote()

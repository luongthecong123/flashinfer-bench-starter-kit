"""Phase-level CUDA event timing for idxer_tc.py — runs on Modal B200.

Usage:
  modal run src/modal/profile_idxer_tc.py
  WORKLOAD_IDX=-1 modal run src/modal/profile_idxer_tc.py   # last workload (default)
  WORKLOAD_IDX=0  modal run src/modal/profile_idxer_tc.py   # first workload
"""
import os, sys
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = int(os.environ.get("WORKLOAD_IDX", "-1"))
WARMUP       = int(os.environ.get("WARMUP", "10"))
REPS         = int(os.environ.get("REPS", "50"))


@app.function(image=image, gpu="B200:1", timeout=300, volumes={"/data": trace_volume})
def run_profile(workload_idx: int, warmup: int, reps: int):
    import sys, json
    from pathlib import Path
    sys.path.insert(0, "/app")
    import torch
    from safetensors.torch import load_file

    NUM_HEADS = 64
    HEAD_DIM  = 128
    TOPK      = 2048
    PAGE_SIZE = 64

    JSONL = (Path("/data") / "workloads" / "dsa_paged" /
             "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl")
    workloads = [json.loads(l) for l in open(JSONL)]
    idx = workload_idx if workload_idx >= 0 else len(workloads) - 1
    w   = workloads[idx]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]

    batch_size    = ax["batch_size"]
    num_pages     = ax["num_pages"]
    max_num_pages = ax["max_num_pages"]
    uuid_short    = w["workload"]["uuid"][:8]

    device = "cuda"
    q_index_fp8 = torch.randn(batch_size, NUM_HEADS, HEAD_DIM,
                               dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    k_index_cache_fp8 = torch.randint(0, 256,
                                       (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
                                       dtype=torch.uint8, device=device).view(torch.int8)
    weights      = torch.randn(batch_size, NUM_HEADS, dtype=torch.float32, device=device)
    topk_indices = torch.full((batch_size, TOPK), -1, dtype=torch.int32, device=device)

    sf          = load_file(str(Path("/data") / inp["seq_lens"]["path"]))
    seq_lens    = sf[inp["seq_lens"]["tensor_key"]].to(device)
    block_table = sf[inp["block_table"]["tensor_key"]].to(device)

    from src.kernels.idxer_tc import (
        dequant_fp8_kv_cache, _score_and_reduce, _topk_remap_and_write
    )

    # ── Build intermediates (used for isolated phase timing) ──
    B      = batch_size
    max_sl = max_num_pages * PAGE_SIZE
    offsets = torch.arange(PAGE_SIZE, device=device)

    def build_intermediates():
        q = q_index_fp8.to(torch.float32)
        K_all  = dequant_fp8_kv_cache(k_index_cache_fp8)
        K_flat = K_all.reshape(-1, HEAD_DIM)
        token_indices = (block_table.long().unsqueeze(2) * PAGE_SIZE +
                         offsets.view(1, 1, PAGE_SIZE)).reshape(B, max_sl)
        positions = torch.arange(max_sl, device=device).unsqueeze(0)
        mask = positions >= seq_lens.unsqueeze(1)
        token_indices = token_indices.clamp(0, K_flat.shape[0] - 1)
        K_gathered = K_flat[token_indices.reshape(-1)].reshape(B, max_sl, HEAD_DIM)
        final = _score_and_reduce(q, K_gathered, weights, mask)
        return q, K_all, K_flat, token_indices, mask, K_gathered, final

    def run_full():
        q = q_index_fp8.to(torch.float32)
        K_all  = dequant_fp8_kv_cache(k_index_cache_fp8)
        K_flat = K_all.reshape(-1, HEAD_DIM)
        token_indices = (block_table.long().unsqueeze(2) * PAGE_SIZE +
                         offsets.view(1, 1, PAGE_SIZE)).reshape(B, max_sl)
        positions = torch.arange(max_sl, device=device).unsqueeze(0)
        mask = positions >= seq_lens.unsqueeze(1)
        token_indices = token_indices.clamp(0, K_flat.shape[0] - 1)
        K_gathered = K_flat[token_indices.reshape(-1)].reshape(B, max_sl, HEAD_DIM)
        final = _score_and_reduce(q, K_gathered, weights, mask)
        _topk_remap_and_write(final, block_table.long(), mask, topk_indices, TOPK, PAGE_SIZE)

    # ── Warmup to trigger torch.compile ──
    for _ in range(warmup):
        run_full()
    torch.cuda.synchronize()

    # ── Pre-build fixed intermediates for isolated phase timing ──
    q, K_all, K_flat, token_indices, mask, K_gathered, final = build_intermediates()
    torch.cuda.synchronize()

    def timed(fn, r=reps):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(r):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / r  # ms → µs

    phases = {
        "dequant_q":          lambda: q_index_fp8.to(torch.float32),
        "dequant_kv_cache":   lambda: dequant_fp8_kv_cache(k_index_cache_fp8),
        "build_indices":      lambda: (block_table.long().unsqueeze(2) * PAGE_SIZE +
                                       offsets.view(1, 1, PAGE_SIZE)).reshape(B, max_sl),
        "build_mask":         lambda: (torch.arange(max_sl, device=device).unsqueeze(0)
                                       >= seq_lens.unsqueeze(1)),
        "gather":             lambda: K_flat[token_indices.reshape(-1)].reshape(B, max_sl, HEAD_DIM),
        "score_and_reduce":   lambda: _score_and_reduce(q, K_gathered, weights, mask),
        "topk_remap_write":   lambda: _topk_remap_and_write(
                                  final, block_table.long(), mask, topk_indices, TOPK, PAGE_SIZE),
        "TOTAL":              run_full,
    }

    results = {name: timed(fn) for name, fn in phases.items()}
    total = results["TOTAL"]

    return {
        "workload_idx": idx,
        "uuid": uuid_short,
        "batch_size": batch_size,
        "num_pages": num_pages,
        "max_num_pages": max_num_pages,
        "max_sl": max_sl,
        "warmup": warmup,
        "reps": reps,
        "phases_us": results,
        "total_us": total,
    }


@app.local_entrypoint()
def main():
    r = run_profile.remote(WORKLOAD_IDX, WARMUP, REPS)

    print(f"\n── idxer_tc.py Phase Profile (B200) ──")
    print(f"Workload {r['workload_idx']}: uuid={r['uuid']}  "
          f"B={r['batch_size']}  num_pages={r['num_pages']}  "
          f"max_pages={r['max_num_pages']}  max_sl={r['max_sl']}")
    print(f"Warmup={r['warmup']}  Reps={r['reps']}\n")

    total = r["total_us"]
    print(f"{'Phase':<22}  {'µs':>8}  {'ms':>7}  {'% total':>8}")
    print("-" * 56)
    for name, us in r["phases_us"].items():
        if name == "TOTAL":
            continue
        print(f"{name:<22}  {us:8.1f}  {us/1000:7.3f}  {us/total*100:7.1f}%")
    print("-" * 56)
    print(f"{'TOTAL':<22}  {total:8.1f}  {total/1000:7.3f}  {'100.0%':>8}")

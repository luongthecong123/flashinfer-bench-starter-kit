"""Count NaN in inputs across all 128 workloads.
Usage: modal run src/modal/count_nan.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image


@app.function(image=image, gpu="B200:1", timeout=300, volumes={"/data": trace_volume})
def run_bench():
    import json, torch
    from pathlib import Path

    CONTEST = Path("/data")
    JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"

    PAGE_SIZE = 64
    HEAD_DIM = 128

    all_workloads = [json.loads(l) for l in open(JSONL)]
    print(f"Total workloads: {len(all_workloads)}\n")

    print(f"{'#':>3} {'B':>3} {'MaxPg':>5} {'NumPg':>6} {'TotalTokens':>12} "
          f"{'fp8_NaN':>8} {'scale_NaN':>10} {'scale_Inf':>10} {'scale_bad%':>10}")
    print("-" * 90)

    total_fp8_nan = 0
    total_scale_nan = 0
    total_scale_inf = 0
    total_tokens = 0

    for i, w in enumerate(all_workloads):
        ax = w["workload"]["axes"]
        batch_size = ax["batch_size"]
        max_num_pages = ax["max_num_pages"]
        num_pages = ax["num_pages"]

        # Generate inputs exactly like make_tensors
        torch.manual_seed(42 + i)  # reproducible but different per workload
        k_cache = torch.randint(0, 256,
                                (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
                                dtype=torch.uint8, device="cuda")

        # Extract fp8 bytes (first 128) and scale bytes (last 4)
        fp8_bytes = k_cache[:, :, 0, :HEAD_DIM]       # [num_pages, 64, 128] uint8
        scale_bytes = k_cache[:, :, 0, HEAD_DIM:]      # [num_pages, 64, 4] uint8

        # Check fp8: view as float8_e4m3fn
        fp8_vals = fp8_bytes.contiguous().view(torch.float8_e4m3fn).to(torch.float32)
        fp8_nan = fp8_vals.isnan().sum().item()

        # Check scales: view 4 bytes as float32
        scale_vals = scale_bytes.contiguous().view(torch.float32)  # [num_pages, 64]
        s_nan = scale_vals.isnan().sum().item()
        s_inf = scale_vals.isinf().sum().item()

        n_tokens = num_pages * PAGE_SIZE
        bad_pct = 100.0 * (s_nan + s_inf) / n_tokens if n_tokens > 0 else 0

        total_fp8_nan += fp8_nan
        total_scale_nan += s_nan
        total_scale_inf += s_inf
        total_tokens += n_tokens

        print(f"{i+1:>3} {batch_size:>3} {max_num_pages:>5} {num_pages:>6} {n_tokens:>12} "
              f"{fp8_nan:>8} {s_nan:>10} {s_inf:>10} {bad_pct:>9.2f}%")

    print("-" * 90)
    total_bad_pct = 100.0 * (total_scale_nan + total_scale_inf) / total_tokens if total_tokens > 0 else 0
    print(f"TOTAL: {total_tokens:,} tokens  |  "
          f"fp8_NaN={total_fp8_nan}  scale_NaN={total_scale_nan:,}  scale_Inf={total_scale_inf:,}  "
          f"bad={total_bad_pct:.3f}%")

    # Also check: does make_tensors use a seed? (it doesn't — check if random is truly random)
    print(f"\nNOTE: make_tensors() does NOT set a seed, so each run produces different random data.")
    print(f"The counts above use manual seeds for reproducibility.")

    # Now do it WITHOUT seed to show typical range
    print(f"\n--- Without seed (single sample, workload #1) ---")
    ax = all_workloads[0]["workload"]["axes"]
    k2 = torch.randint(0, 256, (ax["num_pages"], PAGE_SIZE, 1, HEAD_DIM + 4),
                        dtype=torch.uint8, device="cuda")
    s2 = k2[:, :, 0, HEAD_DIM:].contiguous().view(torch.float32)
    fp8_2 = k2[:, :, 0, :HEAD_DIM].contiguous().view(torch.float8_e4m3fn).to(torch.float32)
    print(f"  fp8 NaN: {fp8_2.isnan().sum().item()}")
    print(f"  scale NaN: {s2.isnan().sum().item()}, Inf: {s2.isinf().sum().item()}")
    print(f"  tokens: {ax['num_pages'] * PAGE_SIZE}")


@app.local_entrypoint()
def main():
    run_bench.remote()

"""Count NaN from ref dequant_fp8_kv_cache across all 128 workloads.
Usage: modal run src/modal/count_nan_ref.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image


@app.function(image=image, gpu="B200:1", timeout=300, volumes={"/data": trace_volume})
def run_bench():
    import json, torch
    from pathlib import Path

    sys.path.insert(0, "/app")
    from src.kernels.idxer_ref import dequant_fp8_kv_cache

    CONTEST = Path("/data")
    JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
    PAGE_SIZE = 64
    HEAD_DIM = 128

    all_workloads = [json.loads(l) for l in open(JSONL)]
    print(f"Total workloads: {len(all_workloads)}\n")

    print(f"{'#':>3} {'B':>3} {'NumPg':>6} {'tokens':>10} | "
          f"{'out_NaN':>10} {'out_Inf':>10} {'out_NaN%':>8} {'tok_w_NaN':>10} {'tok_w_NaN%':>10}")
    print("-" * 100)

    tot_out_nan = tot_out_inf = 0
    tot_tok_nan = 0
    tot_tokens = 0

    for i, w in enumerate(all_workloads):
        ax = w["workload"]["axes"]
        num_pages = ax["num_pages"]

        torch.manual_seed(42 + i)
        k_cache = torch.randint(0, 256,
                                (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
                                dtype=torch.uint8, device="cuda").view(torch.int8)

        # Call the ACTUAL ref dequant directly
        output = dequant_fp8_kv_cache(k_cache)  # [num_pages, 64, 128]

        out_nan = output.isnan().sum().item()
        out_inf = output.isinf().sum().item()

        # Per-token: how many tokens have at least one NaN
        out_flat = output.view(-1, HEAD_DIM)
        tok_nan = out_flat.isnan().any(dim=1).sum().item()

        n_tokens = num_pages * PAGE_SIZE
        total_elems = n_tokens * HEAD_DIM
        out_nan_pct = 100.0 * out_nan / total_elems if total_elems > 0 else 0
        tok_nan_pct = 100.0 * tok_nan / n_tokens if n_tokens > 0 else 0

        tot_out_nan += out_nan
        tot_out_inf += out_inf
        tot_tok_nan += tok_nan
        tot_tokens += n_tokens

        print(f"{i+1:>3} {ax['batch_size']:>3} {num_pages:>6} {n_tokens:>10} | "
              f"{out_nan:>10} {out_inf:>10} {out_nan_pct:>7.3f}% {tok_nan:>10} {tok_nan_pct:>9.2f}%")

    print("-" * 100)
    total_elems = tot_tokens * HEAD_DIM
    print(f"\nTOTAL across {len(all_workloads)} workloads:")
    print(f"  tokens:    {tot_tokens:,}")
    print(f"  elements:  {total_elems:,}")
    print(f"  output NaN:  {tot_out_nan:,}  ({100*tot_out_nan/total_elems:.3f}%)")
    print(f"  output Inf:  {tot_out_inf:,}  ({100*tot_out_inf/total_elems:.3f}%)")
    print(f"  tokens with at least 1 NaN: {tot_tok_nan:,}  ({100*tot_tok_nan/tot_tokens:.2f}%)")


@app.local_entrypoint()
def main():
    run_bench.remote()

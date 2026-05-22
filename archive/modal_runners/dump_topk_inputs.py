"""Gather torch.topk inputs from idxer_tc.run for workloads with max_sl > 2048.

Replays each indexer workload, computes the `final` tensor that gets fed into
torch.topk, and writes one .pt per workload to /data/topk_inputs/wl{idx}.pt
on the Modal volume, then downloads them locally to
archive/analysis/idxer_topk_inputs/.

Also prints zero-count statistics for the user-visible (sl_b)-sized prefix.
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
def collect():
    import sys, json, torch
    from pathlib import Path
    from safetensors.torch import load_file
    sys.path.insert(0, "/app")

    from src.kernels.idxer_tc import (
        dequant_fp8_kv_cache, _score_and_reduce,
        PAGE_SIZE, NUM_HEADS, HEAD_DIM,
    )

    JSONL = (Path("/data") / "workloads" / "dsa_paged" /
             "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl")
    out_dir = Path("/data") / "topk_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    workloads = [json.loads(l) for l in open(JSONL)]
    print(f"Loaded {len(workloads)} workloads")

    stats = []
    device = "cuda"

    for i_w, w in enumerate(workloads):
        ax  = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        B   = ax["batch_size"]
        max_num_pages = ax["max_num_pages"]
        num_pages     = ax["num_pages"]
        uuid_short    = w["workload"]["uuid"][:8]

        sf = load_file(str(Path("/data") / inp["seq_lens"]["path"]))
        seq_lens    = sf[inp["seq_lens"]["tensor_key"]].cuda()
        block_table = sf[inp["block_table"]["tensor_key"]].cuda()
        max_sl      = max_num_pages * PAGE_SIZE
        max_sl_actual = int(seq_lens.max().item())

        if max_sl_actual <= 2048:
            print(f"  [{i_w:>2}] {uuid_short}  B={B} max_sl={max_sl_actual}  SKIP (<=2048)")
            continue

        # Replay scoring path from idxer_tc.run
        torch.manual_seed(0)
        q_index_fp8 = torch.randn(B, NUM_HEADS, HEAD_DIM,
                                   dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
        k_index_cache_fp8 = torch.randint(0, 256,
                                           (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
                                           dtype=torch.uint8, device=device).view(torch.int8)
        weights = torch.randn(B, NUM_HEADS, dtype=torch.float32, device=device)

        q = q_index_fp8.to(torch.float32)
        K_all  = dequant_fp8_kv_cache(k_index_cache_fp8)
        K_flat = K_all.reshape(-1, HEAD_DIM)
        offsets = torch.arange(PAGE_SIZE, device=device)
        token_indices = (block_table.long().unsqueeze(2) * PAGE_SIZE +
                         offsets.view(1, 1, PAGE_SIZE)).reshape(B, max_sl)
        positions = torch.arange(max_sl, device=device).unsqueeze(0)
        mask = positions >= seq_lens.unsqueeze(1)
        token_indices = token_indices.clamp(0, K_flat.shape[0] - 1)
        K_gathered = K_flat[token_indices.reshape(-1)].reshape(B, max_sl, HEAD_DIM)
        final = _score_and_reduce(q, K_gathered, weights, mask)  # [B, max_sl] f32, padding = -inf

        # Statistics on the *valid* prefix only (padded positions are -inf).
        z_total = 0
        v_total = 0
        nonzero_per_b = []
        for b in range(B):
            sl_b = int(seq_lens[b].item())
            if sl_b <= 2048:
                continue
            row = final[b, :sl_b]
            zeros = int((row == 0).sum().item())
            z_total += zeros
            v_total += sl_b
            nonzero_per_b.append(sl_b - zeros)

        stats.append(dict(
            wl=i_w, uuid=uuid_short, B=B,
            max_sl=max_sl_actual,
            valid_total=v_total, zero_total=z_total,
            zero_frac=(z_total / v_total) if v_total else 0.0,
            nonzero_min=min(nonzero_per_b) if nonzero_per_b else 0,
            nonzero_max=max(nonzero_per_b) if nonzero_per_b else 0,
            nonzero_mean=(sum(nonzero_per_b) / len(nonzero_per_b)) if nonzero_per_b else 0,
        ))

        # Save final tensor + seq_lens
        out_pt = out_dir / f"wl{i_w}.pt"
        torch.save(dict(final=final.cpu(), seq_lens=seq_lens.cpu(),
                        uuid=uuid_short, B=B, max_sl=max_sl_actual),
                   out_pt)
        print(f"  [{i_w:>2}] {uuid_short}  B={B} max_sl={max_sl_actual} "
              f"valid={v_total} zeros={z_total} ({100*z_total/v_total:.2f}%)  -> {out_pt.name}")

    summary_path = out_dir / "stats.json"
    summary_path.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote stats to {summary_path}")
    return json.dumps(stats)


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    stats_json = collect.remote()
    out_local = Path("archive/analysis/idxer_topk_inputs")
    out_local.mkdir(parents=True, exist_ok=True)
    (out_local / "stats.json").write_text(stats_json)
    print(f"\nLocal stats saved: {out_local / 'stats.json'}")
    print("Note: .pt files remain on Modal volume `/data/topk_inputs/`. "
          "Use `modal volume get flashinfer-trace topk_inputs ...` to fetch.")

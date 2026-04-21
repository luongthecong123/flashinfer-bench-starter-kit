"""Race-condition test for draftv3 indexer.

Fix seed, run ONE failing workload N times, compare impl outputs across runs
and against reference. If outputs differ run-to-run → race condition.

Usage: modal run src/modal/race_test.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.draftv3")
WORKLOAD_IDX = int(os.environ.get("WORKLOAD_IDX", "123"))  # 0-indexed, 123 == #124 27c3374f FAIL
N_RUNS = int(os.environ.get("N_RUNS", "8"))
SEED = int(os.environ.get("SEED", "42"))


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_race(impl_module: str, workload_idx: int, n_runs: int, seed: int):
    import json, torch
    from pathlib import Path
    from importlib import import_module
    from safetensors.torch import load_file

    sys.path.insert(0, "/app")

    impl = import_module(impl_module)
    from src.kernels.idxer_ref import run as ref_run

    NUM_HEADS, HEAD_DIM, PAGE_SIZE, TOPK = 64, 128, 64, 2048
    CONTEST = Path("/data")
    JSONL = CONTEST / "workloads" / "dsa_paged" / "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"

    all_workloads = [json.loads(l) for l in open(JSONL)]
    w = all_workloads[workload_idx]
    ax, inp = w["workload"]["axes"], w["workload"]["inputs"]
    bs, num_pages = ax["batch_size"], ax["num_pages"]
    uuid = w["workload"]["uuid"][:8]

    sf = load_file(str(CONTEST / inp["seq_lens"]["path"]))
    seq_lens    = sf[inp["seq_lens"]["tensor_key"]].cuda()
    block_table = sf[inp["block_table"]["tensor_key"]].cuda()

    print(f"Workload #{workload_idx+1}  uuid={uuid}  batch_size={bs}  num_pages={num_pages}  max_pg={ax['max_num_pages']}")
    print(f"seq_lens: min={seq_lens.min().item()}  max={seq_lens.max().item()}  >2048={(seq_lens>2048).sum().item()}")

    # ── Fixed-seed inputs ──
    g = torch.Generator(device="cuda").manual_seed(seed)
    q_f32   = torch.randn(bs, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device="cuda", generator=g)
    q_fp8   = q_f32.to(torch.float8_e4m3fn)
    k_cache = torch.randint(0, 256, (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4),
                            dtype=torch.uint8, device="cuda", generator=g).view(torch.int8)
    weights = torch.randn(bs, NUM_HEADS, dtype=torch.float32, device="cuda", generator=g)

    args = (q_fp8, k_cache, weights, seq_lens, block_table)

    # ── Reference ──
    ref_topk = torch.full((bs, TOPK), -1, dtype=torch.int32, device="cuda")
    ref_run(*args, ref_topk)
    torch.cuda.synchronize()

    # ── N impl runs on same inputs ──
    impl_outs = []
    for r in range(n_runs):
        impl_topk = torch.full((bs, TOPK), -1, dtype=torch.int32, device="cuda")
        # Clone args (match bench methodology)
        cloned = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
        impl.run(*cloned, impl_topk)
        torch.cuda.synchronize()
        impl_outs.append(impl_topk)

    # ── Analyze ──
    def miss(ref, imp, sl_tensor):
        maxm = 0.0
        per_batch = []
        for b in range(ref.shape[0]):
            sl = int(sl_tensor[b].item())
            k  = min(TOPK, sl)
            if k == 0:
                per_batch.append(0.0); continue
            rs = set(ref[b, :k].tolist()) - {-1}
            ps = set(imp[b, :k].tolist()) - {-1}
            if not rs:
                per_batch.append(0.0); continue
            mf = len(rs - ps) / len(rs)
            per_batch.append(mf)
            maxm = max(maxm, mf)
        return maxm, per_batch

    print(f"\n{'run':>4} {'vs ref':>10} {'vs run0':>10}  worst-batch-miss-frac")
    print("-" * 60)
    mf0_vs_ref, pb0_ref = miss(ref_topk, impl_outs[0], seq_lens)
    print(f"{0:>4} {mf0_vs_ref:>10.4f} {0.0:>10.4f}  (batch-level vs ref)")
    # print top-5 worst batches for run 0 vs ref
    worst = sorted(range(bs), key=lambda b: -pb0_ref[b])[:5]
    for b in worst:
        print(f"       batch {b:3d}  seq_len={int(seq_lens[b].item()):5d}  miss_frac_vs_ref={pb0_ref[b]:.4f}")

    same_across_runs = True
    for r in range(1, n_runs):
        mf_ref, _      = miss(ref_topk,     impl_outs[r], seq_lens)
        mf_run0, _     = miss(impl_outs[0], impl_outs[r], seq_lens)
        if mf_run0 > 0.0:
            same_across_runs = False
        # Exact-equality flag (order may differ → sets)
        eq_to_run0 = all(
            set(impl_outs[0][b].tolist()) == set(impl_outs[r][b].tolist())
            for b in range(bs)
        )
        print(f"{r:>4} {mf_ref:>10.4f} {mf_run0:>10.4f}  set_eq_to_run0={eq_to_run0}")

    print()
    if same_across_runs:
        print("==> DETERMINISTIC across runs (same impl output every run)")
        print("    The FAIL is SYSTEMATIC, not a race condition.")
    else:
        print("==> NON-DETERMINISTIC across runs — RACE CONDITION suspected.")
    return None


@app.local_entrypoint()
def main():
    run_race.remote(IMPL_MODULE, WORKLOAD_IDX, N_RUNS, SEED)

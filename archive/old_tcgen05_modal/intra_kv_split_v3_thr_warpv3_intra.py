"""Modal runner: intra profiling for kv_split_v3_thr_warpv3_intra."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 22  # match tcgen05_v3b_full_tma run
IMPL_MODULE  = "src.kernels.kv_split_v3_thr_warpv3_intra"
OUT_FILENAME = "intra_kv_split_v3_thr_warpv3_w23.json"


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
def run_intra(impl_module: str, workload_idx: int):
    import sys, os
    sys.path.insert(0, "/app")
    from importlib import import_module
    from pathlib import Path
    from safetensors.torch import load_file
    import torch, json
    mod = import_module(impl_module)

    # Run the kernel's own profiling first (per-CTA probe).
    trace = mod.run_single(workload_idx)

    # Now do CUDA-event e2e timing for apples-to-apples comparison.
    from src.utils import WORKLOAD_INFO, make_tensors
    H, D_ckv = mod.NUM_HEADS, mod.DV
    NUM_SPLITS = mod.NUM_SPLITS
    ROW_MAX_SUM_PAIR = mod.ROW_MAX_SUM_PAIR
    PROBE_COLS = mod.PROBE_COLS

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [json.loads(l) for l in open(JSONL)]
    w   = workloads[workload_idx]
    inp = w["workload"]["inputs"]
    ax  = w["workload"]["axes"]
    T, P = ax["num_tokens"], ax["num_pages"]

    compiled = mod.compile_kernel()
    q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)
    sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

    output      = torch.zeros(T, H, D_ckv, dtype=torch.bfloat16, device="cuda")
    lse         = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
    partial_out = torch.empty(8, H, NUM_SPLITS, D_ckv, dtype=torch.float32, device="cuda")
    partial_lse = torch.empty(8, H, NUM_SPLITS, ROW_MAX_SUM_PAIR, dtype=torch.float32, device="cuda")
    num_compute_blocks = T * H * NUM_SPLITS
    num_reduce_blocks  = T * H
    probe_compute = torch.zeros((num_compute_blocks, PROBE_COLS), dtype=torch.int64, device="cuda")
    probe_reduce  = torch.zeros((num_reduce_blocks,  PROBE_COLS), dtype=torch.int64, device="cuda")

    for _ in range(5):
        compiled(q_nope, q_pe, ckv, kpe, si, partial_out, partial_lse,
                 output, lse, probe_compute, probe_reduce)
    torch.cuda.synchronize()

    BENCH_ITERS = 50
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(BENCH_ITERS)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(BENCH_ITERS)]
    for i in range(BENCH_ITERS):
        cache.zero_()
        torch.cuda.synchronize()
        starts[i].record()
        compiled(q_nope, q_pe, ckv, kpe, si, partial_out, partial_lse,
                 output, lse, probe_compute, probe_reduce)
        ends[i].record()
    torch.cuda.synchronize()
    e2e = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    print(f"\n[e2e compute+reduce] mean={sum(e2e)/len(e2e):.3f} µs  "
          f"min={min(e2e):.3f}  max={max(e2e):.3f}  ({BENCH_ITERS} iters)")
    return trace


@app.local_entrypoint()
def main():
    print(f"\n{'='*60}\nProfiling {IMPL_MODULE}  WL{WORKLOAD_IDX + 1}\n{'='*60}")
    trace_json = run_intra.remote(IMPL_MODULE, WORKLOAD_IDX)
    out_path = Path(f"reports/{OUT_FILENAME}")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(trace_json)
    print(f"Saved trace to {out_path}")

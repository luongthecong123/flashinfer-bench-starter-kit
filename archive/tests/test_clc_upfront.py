"""Modal runner: correctness + benchmark for Task B — kv_split_v3_thr_warpv3_clc_upfront.

Upfront preload variant. Before the CLC tile loop, ALL T tokens' sparse_indices are
loaded into smem once (smem_sparse[T_MAX=8, TOP_K=2048] = 64 KB) and num_valid is
cached in smem. The tile loop then skips all sparse_indices GMEM reads.

Compared against kv_split_v3_thr_warpv3 (verified baseline) on all 23 workloads.

Uses a generator function to stream output line-by-line; this keeps the gRPC
connection alive during the long compilation phase (~3-4 min on B200).

Usage:
    modal run src/modal/test_clc_upfront.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

WARMUP = 50
REPS   = 200


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/data": trace_volume})
def run_all():
    """Generator: streams correctness + benchmark results line by line."""
    import sys, json
    from pathlib import Path
    sys.path.insert(0, "/app")

    import torch
    from safetensors.torch import load_file

    yield "Compiling kernels (run_ref + run_upfront) …\n"
    from src.kernels.kv_split_v3_thr_warpv3             import run as run_ref      # compiled at import
    from src.kernels.kv_split_v3_thr_warpv3_clc_upfront import run as run_upfront  # compiled at import
    yield "Compilation done.\n"

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / \
              "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    H, D, Dp, PS = 16, 512, 64, 64
    SCALE = 0.1352337788608801
    ATOL  = 0.01

    workloads = [json.loads(l) for l in open(JSONL)]

    # ── Correctness ────────────────────────────────────────────────────────────
    yield f"\n{'='*60}\n"
    yield "TASK B: kv_split_v3_thr_warpv3_clc_upfront — CORRECTNESS\n"
    yield f"{'='*60}\n"
    yield f"{'#':>3} {'uuid':>10} {'T':>2}  {'out_err':>10} {'lse_err':>10}  {'Status':>6}\n"
    yield "-" * 52 + "\n"

    all_pass = True
    bench_data = []
    for i_w, w in enumerate(workloads):
        ax  = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        T, P = ax["num_tokens"], ax["num_pages"]
        uuid = w["workload"]["uuid"][:8]

        q_nope = torch.randn(T, H, D,  dtype=torch.bfloat16, device="cuda")
        q_pe   = torch.randn(T, H, Dp, dtype=torch.bfloat16, device="cuda")
        ckv    = torch.randn(P, PS, D,  dtype=torch.bfloat16, device="cuda")
        kpe    = torch.randn(P, PS, Dp, dtype=torch.bfloat16, device="cuda")
        sf     = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
        si     = sf[inp["sparse_indices"]["tensor_key"]].cuda()

        r_out = torch.zeros(T, H, D, dtype=torch.bfloat16, device="cuda")
        r_lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
        u_out = torch.zeros(T, H, D, dtype=torch.bfloat16, device="cuda")
        u_lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

        run_ref(q_nope, q_pe, ckv, kpe, si, SCALE, r_out, r_lse)
        run_upfront(q_nope, q_pe, ckv, kpe, si, SCALE, u_out, u_lse)
        torch.cuda.synchronize()

        o_err = (r_out.float() - u_out.float()).abs().max().item()
        l_err = (r_lse - u_lse).abs().max().item()
        ok    = o_err < ATOL and l_err < ATOL
        status = "PASS" if ok else "FAIL"
        yield f"{i_w+1:>3} {uuid:>10} {T:>2}  {o_err:>10.2e} {l_err:>10.2e}  {status:>6}\n"
        if not ok:
            all_pass = False

        bench_data.append((T, P, q_nope, q_pe, ckv, kpe, si))

    yield f"\nOverall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}\n"

    # ── Benchmark ──────────────────────────────────────────────────────────────
    yield f"\n{'='*60}\n"
    yield f"TASK B: clc_upfront — BENCHMARK  (warmup={WARMUP} reps={REPS})\n"
    yield f"{'='*60}\n"
    yield f"{'#':>3} {'uuid':>10} {'T':>2}  {'ref µs':>9} {'upfrt µs':>9}  {'speedup':>8}\n"
    yield "-" * 58 + "\n"

    for i_w, (w, (T, P, q_nope, q_pe, ckv, kpe, si)) in enumerate(zip(workloads, bench_data)):
        uuid = w["workload"]["uuid"][:8]
        out  = torch.zeros(T, H, D, dtype=torch.bfloat16, device="cuda")
        lse  = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

        def _bench(fn):
            for _ in range(WARMUP):
                fn(q_nope, q_pe, ckv, kpe, si, SCALE, out, lse)
            torch.cuda.synchronize()
            evs = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)) for _ in range(REPS)]
            for s, e in evs:
                s.record(); fn(q_nope, q_pe, ckv, kpe, si, SCALE, out, lse); e.record()
            torch.cuda.synchronize()
            return sorted(s.elapsed_time(e) * 1e3 for s, e in evs)[REPS // 2]

        ref_t     = _bench(run_ref)
        upfront_t = _bench(run_upfront)
        speedup = ref_t / upfront_t
        yield f"{i_w+1:>3} {uuid:>10} {T:>2}  {ref_t:>9.2f} {upfront_t:>9.2f}  {speedup:>8.3f}×\n"


@app.local_entrypoint()
def main():
    for line in run_all.remote_gen():
        print(line, end="", flush=True)

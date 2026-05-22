"""Modal runner: correctness + benchmark for CLC variant kernels on B200.

Tests three CLC variants against the verified baseline (kv_split_v3_thr_warpv3):
  clc          -- original CLC (work-stealing)
  clc_upfront  -- CLC with upfront sparse_indices preload
  clc_pdl      -- CLC with PDL (Programmatic Dependent Launch)

Correctness: all 23 contest workloads.
Benchmark: configurable warm-up + repetitions on each workload, reports
           median kernel time per variant.

Usage:
    modal run src/modal/test_kvsplit_v3_clc_variants.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
def check_and_bench(warmup: int = 50, reps: int = 200):
    import sys, json, math, time
    from pathlib import Path
    sys.path.insert(0, "/app")

    import torch
    from safetensors.torch import load_file

    from src.kernels.kv_split_v3_thr_warpv3         import run as run_ref
    from src.kernels.kv_split_v3_thr_warpv3_clc     import run as run_clc
    from src.kernels.kv_split_v3_thr_warpv3_clc_upfront import run as run_upfront
    from src.kernels.kv_split_v3_thr_warpv3_clc_pdl import run as run_pdl

    VARIANTS = {
        "clc":      run_clc,
        "upfront":  run_upfront,
        "pdl":      run_pdl,
    }

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / \
              "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    H, D, Dp, PS = 16, 512, 64, 64
    SCALE = 0.1352337788608801
    ATOL  = 0.01

    workloads = [json.loads(l) for l in open(JSONL)]
    n_wl = len(workloads)

    # ── Correctness pass ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("CORRECTNESS CHECK")
    print(f"{'='*72}")
    hdr = f"{'#':>3} {'uuid':>10} {'T':>2}"
    for v in VARIANTS:
        hdr += f"  {v:>12}"
    print(hdr)
    print("-" * len(hdr))

    all_pass = {v: True for v in VARIANTS}
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
        run_ref(q_nope, q_pe, ckv, kpe, si, SCALE, r_out, r_lse)
        torch.cuda.synchronize()

        row = f"{i_w+1:>3} {uuid:>10} {T:>2}"
        for vname, vfn in VARIANTS.items():
            v_out = torch.zeros(T, H, D, dtype=torch.bfloat16, device="cuda")
            v_lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")
            vfn(q_nope, q_pe, ckv, kpe, si, SCALE, v_out, v_lse)
            torch.cuda.synchronize()
            o_err = (r_out.float() - v_out.float()).abs().max().item()
            l_err = (r_lse - v_lse).abs().max().item()
            ok = o_err < ATOL and l_err < ATOL
            if not ok:
                all_pass[vname] = False
            status = "PASS" if ok else f"FAIL({o_err:.1e})"
            row += f"  {status:>12}"
        print(row)

    print()
    for vname, passed in all_pass.items():
        print(f"  {vname}: {'ALL PASS' if passed else 'SOME FAILED'}")

    # ── Benchmark pass ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"BENCHMARK  (warmup={warmup}  reps={reps})")
    print(f"{'='*72}")
    hdr = f"{'#':>3} {'uuid':>10} {'T':>2}"
    for v in VARIANTS:
        hdr += f"  {v:>11}"
    print(hdr)
    print("-" * len(hdr))

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
        out    = torch.zeros(T, H, D, dtype=torch.bfloat16, device="cuda")
        lse    = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

        row = f"{i_w+1:>3} {uuid:>10} {T:>2}"
        for vname, vfn in VARIANTS.items():
            for _ in range(warmup):
                vfn(q_nope, q_pe, ckv, kpe, si, SCALE, out, lse)
            torch.cuda.synchronize()

            evs = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
                   for _ in range(reps)]
            for s, e in evs:
                s.record()
                vfn(q_nope, q_pe, ckv, kpe, si, SCALE, out, lse)
                e.record()
            torch.cuda.synchronize()

            times_us = sorted(s.elapsed_time(e) * 1e3 for s, e in evs)
            med = times_us[reps // 2]
            row += f"  {med:>8.1f} µs"
        print(row)

    return all_pass


@app.local_entrypoint()
def main():
    results = check_and_bench.remote()
    overall = all(results.values())
    if overall:
        print("\nAll correctness checks PASSED.")
    else:
        failed = [v for v, p in results.items() if not p]
        print(f"\nFAILED variants: {failed}")

"""Debug benchmark runner — GPU UMMA kernel + PyTorch GEMV fallback.

Usage: modal run src/modal/submit_debug.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

IMPL_MODULE = os.environ.get("IMPL_MODULE", "src.kernels.kv_split_umma_v2_debug")

print("IMPL_MODULE: ", IMPL_MODULE)

TARGET_WORKLOAD_INDEX = 21
TARGET_WORKLOAD_UUID  = "5096e459"


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def run_bench(impl_module: str):
    import json
    import sys
    from pathlib import Path as P
    from importlib import import_module

    sys.path.insert(0, "/app")

    impl = import_module(impl_module)

    from src import utils
    from src.ref import run as ref_run
    utils.ref_fn = ref_run
    utils.CONTEST = P("/data")
    full_jsonl = utils.CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    if TARGET_WORKLOAD_INDEX is not None:
        workloads = [json.loads(l) for l in open(full_jsonl)]
        target_row = workloads[TARGET_WORKLOAD_INDEX - 1]
        target_uuid = target_row["workload"]["uuid"][:8]
        assert target_uuid == TARGET_WORKLOAD_UUID, (
            f"Target mismatch: expected {TARGET_WORKLOAD_UUID}, got {target_uuid} "
            f"at row {TARGET_WORKLOAD_INDEX}"
        )
        one_jsonl = P("/tmp") / f"single_workload_{TARGET_WORKLOAD_UUID}.jsonl"
        with open(one_jsonl, "w") as f:
            f.write(json.dumps(target_row) + "\n")
        utils.JSONL = one_jsonl
        print(f"Running single workload: row={TARGET_WORKLOAD_INDEX}, uuid={TARGET_WORKLOAD_UUID}")
    else:
        utils.JSONL = full_jsonl
        print("Running all workloads")

    utils.CHECK   = False  # skip combined check; we do our own split check below
    utils.MEASURE = True
    utils.impl_fn = impl.run

    # ── Custom split correctness check ──────────────────────────────────────
    import json as _json
    from src.ref import run as _ref_run

    workloads_all = [_json.loads(l) for l in open(utils.JSONL)]
    w = workloads_all[0]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]
    T, Pg = ax["num_tokens"], ax["num_pages"]

    q_nope, q_pe, ckv, kpe, _ = utils.make_tensors(T, Pg)
    from safetensors.torch import load_file as _lf
    sf = _lf(str(utils.CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

    import torch as _torch
    r_out, r_lse = utils.alloc_out(T)
    i_out, i_lse = utils.alloc_out(T)
    _ref_run(q_nope, q_pe, ckv, kpe, si, utils.SCALE, r_out, r_lse)
    impl.run(q_nope, q_pe, ckv, kpe, si, utils.SCALE, i_out, i_lse)
    _torch.cuda.synchronize()

    o_err = (r_out.float() - i_out.float()).abs().max().item()
    l_err = (r_lse - i_lse).abs().max().item()
    o_ok  = o_err < utils.ATOL
    l_ok  = l_err < utils.ATOL

    print(f"\n=== Split correctness check (workload {TARGET_WORKLOAD_UUID}) ===")
    print(f"  output : {'PASS' if o_ok else 'FAIL'}  max_err={o_err:.3e}  (atol={utils.ATOL})")
    print(f"  lse    : {'PASS' if l_ok else 'FAIL'}  max_err={l_err:.3e}  (atol={utils.ATOL})")
    if not l_ok:
        print(f"\n  lse per (token, head) — ref vs impl vs diff:")
        for t in range(T):
            for h in range(utils.H):
                rv = r_lse[t, h].item()
                iv = i_lse[t, h].item()
                print(f"    t={t} h={h:2d}  ref={rv:+.4f}  impl={iv:+.4f}  diff={iv-rv:+.4f}")
    print()
    # ────────────────────────────────────────────────────────────────────────

    return utils.main()


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path as P

    data = run_bench.remote(IMPL_MODULE)
    if data is None:
        return

    variant = IMPL_MODULE.replace("src.kernels.", "")

    csv_path = P("bench_results.csv")
    write_header = not csv_path.exists()

    existing_rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r["variant"] != variant]
        write_header = True

    rows = existing_rows
    for r in data["results"]:
        rows.append({
            "variant":  variant,
            "workload": r["workload"],
            "uuid":     r["uuid"],
            "T":        r["T"],
            "ref_ms":   f"{r.get('ref_ms', 0):.3f}",
            "impl_ms":  f"{r.get('impl_ms', 0):.3f}",
            "speedup":  f"{r.get('speedup', 0):.2f}",
        })

    fields = ["variant", "workload", "uuid", "T", "ref_ms", "impl_ms", "speedup"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    status = "ALL PASS" if data["all_pass"] else "SOME FAILED"
    print(f"\n>> {status} — wrote {variant} to {csv_path}")

"""Modal runner: correctness check for kv_split_v3_thr_warpv3_clc on B200.

Compares CLC kernel output against kv_split_v3_thr_warpv3 (verified baseline)
across all 23 contest workloads.

Usage:
    modal run src/modal/test_kvsplit_v3_clc.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/data": trace_volume})
def check_fn():
    import sys, json, math
    from pathlib import Path
    sys.path.insert(0, "/app")

    import torch
    from cutlass.cute.runtime import from_dlpack
    from safetensors.torch import load_file

    from src.kernels.kv_split_v3_thr_warpv3     import run as run_ref
    from src.kernels.kv_split_v3_thr_warpv3_clc import run as run_clc

    CONTEST  = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL    = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    H, D, Dp, PS = 16, 512, 64, 64
    SCALE = 0.1352337788608801
    ATOL  = 0.01

    workloads = [json.loads(l) for l in open(JSONL)]
    print(f"Running correctness check on {len(workloads)} workloads...\n")
    print(f"{'#':>3} {'UUID':>10} {'T':>2}  {'out_err':>10} {'lse_err':>10}  {'Status':>6}")
    print("-" * 52)

    all_pass = True
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
        c_out = torch.zeros(T, H, D, dtype=torch.bfloat16, device="cuda")
        c_lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device="cuda")

        run_ref(q_nope, q_pe, ckv, kpe, si, SCALE, r_out, r_lse)
        run_clc(q_nope, q_pe, ckv, kpe, si, SCALE, c_out, c_lse)
        torch.cuda.synchronize()

        o_err = (r_out.float() - c_out.float()).abs().max().item()
        l_err = (r_lse - c_lse).abs().max().item()
        ok    = o_err < ATOL and l_err < ATOL

        status = "PASS" if ok else "FAIL"
        print(f"{i_w+1:>3} {uuid:>10} {T:>2}  {o_err:>10.2e} {l_err:>10.2e}  {status:>6}")

        if not ok:
            all_pass = False

    print()
    print("=" * 52)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


@app.local_entrypoint()
def main():
    result = check_fn.remote()
    if result:
        print("\nCORRECTNESS PASS — CLC kernel matches baseline on all workloads.")
    else:
        print("\nCORRECTNESS FAIL — see per-workload errors above.")

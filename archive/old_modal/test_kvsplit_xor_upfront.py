"""Modal runner: correctness check for kv_split_xor upfront phase on B200.

Tests the 'Load sparse_indices and calculate OOB tiles up front' logic by:
  1. Running the stripped upfront kernel on real contest sparse_indices.
  2. Comparing GPU smem_num_valid output against CPU reference:
       (sparse_indices >= 0).sum(dim=1)

If XID-13 'Illegal Instruction Parameter' fires here the crash is isolated to
the barrier + count-reduce logic, which means cute.arch.barrier with a dynamic
barrier_id is the root cause.

Usage:
    modal run src/modal/test_kvsplit_xor_upfront.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image


@app.function(image=image, gpu="B200:1", timeout=300, volumes={"/data": trace_volume})
def check_fn():
    import sys, json
    from pathlib import Path
    sys.path.insert(0, "/app")

    import torch
    from safetensors.torch import load_file
    from src.kernels.test_kvsplit_xor_upfront import run_upfront_test, TOP_K_LEN

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / \
              "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    workloads = [json.loads(l) for l in open(JSONL)]
    print(f"Upfront-phase correctness check on {len(workloads)} workloads\n")
    print(f"{'#':>3} {'UUID':>10} {'T':>2}  {'max_err':>8}  {'Status':>6}")
    print("-" * 38)

    all_pass = True
    for i_w, w in enumerate(workloads):
        ax   = w["workload"]["axes"]
        inp  = w["workload"]["inputs"]
        T    = ax["num_tokens"]
        uuid = w["workload"]["uuid"][:8]

        sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
        si = sf[inp["sparse_indices"]["tensor_key"]].cuda()   # (T, TOP_K_LEN) i32

        gpu_out = run_upfront_test(si)                        # (T,) on CPU
        cpu_ref = (si.cpu() >= 0).sum(dim=1).int()           # (T,) on CPU

        max_err = (gpu_out - cpu_ref).abs().max().item()
        ok      = max_err == 0
        status  = "PASS" if ok else "FAIL"
        print(f"{i_w+1:>3} {uuid:>10} {T:>2}  {max_err:>8}  {status:>6}")

        if not ok:
            all_pass = False
            print(f"     GPU : {gpu_out.tolist()}")
            print(f"     CPU : {cpu_ref.tolist()}")

    print()
    print("=" * 38)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


@app.local_entrypoint()
def main():
    result = check_fn.remote()
    if result:
        print("\nUPFRONT PHASE PASS — smem_num_valid matches CPU reference on all workloads.")
    else:
        print("\nUPFRONT PHASE FAIL — see per-workload errors above.")

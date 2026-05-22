"""Modal runner: e2e timing for kv_split_xor_pdl_v3_pro_v2 on a single workload."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from pathlib import Path
from src.modal.modal_utils import app, trace_volume, image

WORKLOAD_IDX = 22  # 0-based → workload #23 (uuid=2207f0fd, T=7) — same as tcgen05
IMPL_MODULE  = "src.kernels.kv_split_xor_pdl_v3_pro_v2"
OUT_FILENAME = "intra_kv_split_xor_pdl_v3_pro_v2_w23.txt"


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
def run_intra(workload_idx: int):
    import sys, os, json, torch
    sys.path.insert(0, "/app")
    from pathlib import Path
    from safetensors.torch import load_file
    from importlib import import_module

    mod = import_module(IMPL_MODULE)
    from src.utils import WORKLOAD_INFO, make_tensors

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    CONTEST = Path(os.environ.get("CONTEST_DIR", "/data"))
    JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"
    workloads = [json.loads(l) for l in open(JSONL)]
    w   = workloads[workload_idx]
    ax  = w["workload"]["axes"]
    inp = w["workload"]["inputs"]
    T, P = ax["num_tokens"], ax["num_pages"]
    _uuid, _T, max_valid = WORKLOAD_INFO[workload_idx]
    sm_scale = 0.1352337788608801
    print(f"Workload {workload_idx + 1}: uuid={_uuid}  T={T}  P={P}  MaxValid={max_valid}")

    q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)
    sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
    si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

    output = torch.zeros(T, mod.NUM_HEADS, mod.HEAD_DIM_CKV,
                         dtype=torch.bfloat16, device="cuda")
    lse    = torch.full((T, mod.NUM_HEADS), -float("inf"),
                        dtype=torch.float32, device="cuda")

    # Warmup
    for _ in range(5):
        output.zero_(); lse.fill_(-float("inf"))
        mod.run(q_nope, q_pe, ckv, kpe, si, sm_scale, output, lse)
    torch.cuda.synchronize()

    # ─── Correctness on first up-to-4 (T,head) ──────────────────────────────
    si_cpu = si.cpu()
    P_n = P * mod.PAGE_SIZE
    ckv_f = ckv.view(P_n, mod.HEAD_DIM_CKV).float()
    kpe_f = kpe.view(P_n, mod.HEAD_DIM_KPE).float()
    qn_f, qp_f = q_nope.float(), q_pe.float()
    out_pass = out_fail = 0; out_max = 0.0; lse_max = 0.0
    for t in range(min(T, 4)):
        idx = si_cpu[t]
        valid_idx = idx[idx >= 0].long()
        if valid_idx.numel() == 0:
            continue
        ckv_v = ckv_f[valid_idx]; kpe_v = kpe_f[valid_idx]
        for h in range(mod.NUM_HEADS):
            score = (ckv_v @ qn_f[t, h] + kpe_v @ qp_f[t, h]) * sm_scale
            row_max = score.max(); e = torch.exp(score - row_max); ssum = e.sum()
            ref = (e / ssum) @ ckv_v
            ref_lse = (row_max + torch.log(ssum)) / 0.6931471805599453
            got = output[t, h].float().cpu()
            diff = (got - ref.cpu()).abs().max().item()
            out_max = max(out_max, diff)
            lse_max = max(lse_max, abs(lse[t, h].item() - ref_lse.item()))
            (out_pass if diff < 5e-2 else out_fail).__class__  # noop
            if diff < 5e-2: out_pass += 1
            else: out_fail += 1
    print(f"Final output correctness: {out_pass} PASS / {out_fail} FAIL  "
          f"out_max={out_max:.5f}  lse_max={lse_max:.5f}")

    # ─── e2e timing via CUDA events (apples-to-apples with tcgen05 runner) ─
    BENCH_ITERS = 50
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(BENCH_ITERS)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(BENCH_ITERS)]
    for i in range(BENCH_ITERS):
        cache.zero_()
        torch.cuda.synchronize()
        starts[i].record()
        mod.run(q_nope, q_pe, ckv, kpe, si, sm_scale, output, lse)
        ends[i].record()
    torch.cuda.synchronize()
    e2e = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    line = (f"\n[e2e compute+reduce] mean={sum(e2e)/len(e2e):.3f} µs  "
            f"min={min(e2e):.3f}  max={max(e2e):.3f}  ({BENCH_ITERS} iters)")
    print(line)
    return line


@app.local_entrypoint()
def main():
    print(f"\n{'='*60}\nProfiling {IMPL_MODULE}  WL{WORKLOAD_IDX + 1}\n{'='*60}")
    s = run_intra.remote(WORKLOAD_IDX)
    out_path = Path(f"reports/{OUT_FILENAME}")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(s)
    print(f"Saved to {out_path}")

"""Smoke test: kv_split_xor_pdl_v3_pro_v2_tcgen05 on synthetic workload."""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=900)
def run_smoke():
    import sys, json
    sys.path.insert(0, "/app")
    import torch
    from src.kernels import kv_split_xor_pdl_v3_pro_v2_tcgen05 as kk
    from src import utils as u

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Use workload 21 metadata: T=8, valid per token from WORKLOAD_INFO[20]
    uuid_, T, max_valid = u.WORKLOAD_INFO[20]
    P = u.PS  # not actually used, num_pages comes from kernel constants
    # Use NUM_PAGES from kernel
    P = kk.NUM_PAGES

    torch.manual_seed(0)
    q_nope, q_pe, ckv, kpe, si = u.make_tensors(T, P, valid_per_token=max_valid)
    output = torch.zeros(T, kk.NUM_HEADS, kk.HEAD_DIM_CKV,
                         dtype=torch.bfloat16, device="cuda")
    lse = torch.full((T, kk.NUM_HEADS), -float("inf"),
                     dtype=torch.float32, device="cuda")

    # Reference (per token)
    sm_scale = u.SCALE
    ckv_flat = ckv.reshape(-1, kk.HEAD_DIM_CKV).float()
    kpe_flat = kpe.reshape(-1, kk.HEAD_DIM_KPE).float()

    print(f"\nWorkload 21 ({uuid_}): T={T}, max_valid={max_valid}")
    kk.run(q_nope, q_pe, ckv, kpe, si, sm_scale, output, lse)
    torch.cuda.synchronize()

    # Per-token reference
    max_diff = 0.0
    for t in range(T):
        v = max_valid[t]
        if v == 0:
            continue
        idx = si[t, :v].long()
        ckv_v = ckv_flat[idx]   # (v, 512)
        kpe_v = kpe_flat[idx]   # (v, 64)
        qn = q_nope[t].float()  # (16, 512)
        qp = q_pe[t].float()    # (16, 64)

        score = (ckv_v @ qn.T + kpe_v @ qp.T) * sm_scale  # (v, 16)
        smax = score.max(dim=0, keepdim=True).values
        e = torch.exp(score - smax)
        p = e / e.sum(dim=0, keepdim=True)
        ref = p.T @ ckv_v   # (16, 512)

        diff = (output[t].float() - ref).abs().max().item()
        max_diff = max(max_diff, diff)
        print(f"  t={t} v={v:>4d}  max_diff={diff:.4f}  "
              f"{'PASS' if diff < 0.05 else 'FAIL'}")

    print(f"\nGlobal max_diff = {max_diff:.4f}")

    # Quick latency probe
    for _ in range(3):
        kk.run(q_nope, q_pe, ckv, kpe, si, sm_scale, output, lse)
    torch.cuda.synchronize()

    n_iter = 50
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        kk.run(q_nope, q_pe, ckv, kpe, si, sm_scale, output, lse)
    end.record()
    torch.cuda.synchronize()
    ms_avg = start.elapsed_time(end) / n_iter
    print(f"\nLatency: {ms_avg*1000:.2f} µs/iter (avg of {n_iter})")

    return json.dumps({"max_diff": max_diff, "us": ms_avg*1000})


@app.local_entrypoint()
def main():
    print(run_smoke.remote())

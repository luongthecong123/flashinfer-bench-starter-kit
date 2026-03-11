#!/usr/bin/env python3
"""One-click correctness + benchmark for DSA sparse attention."""
import json, math, torch
from pathlib import Path
from safetensors.torch import load_file

# ── Flags ──
CHECK   = True
MEASURE = True
TOY_CHECK = True   # quick small-scale sanity check first
TOY_CHECK = False

# ── Model params ──
H      = 16    # num_qo_heads
D      = 512   # head_dim_ckv
Dp     = 64    # head_dim_kpe
TOPK   = 2048
PS     = 64    # page_size
SCALE  = 0.1352337788608801
ATOL   = 0.01
RTOL   = 0.01

# ── Paths ──
ROOT    = Path(__file__).parent.parent
CONTEST = ROOT.parent / "flashinfer26dsa" / "mlsys26-contest"
JSONL   = CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

# ── Imports ──
from ref import run as ref_fn
from impl import run as impl_fn

# ────────────────────────────────────────────
def make_tensors(T, P, valid_per_token=None, device="cuda"):
    """Generate random inputs. valid_per_token: list of ints or None (=TOPK)."""
    q_nope   = torch.randn(T, H, D,  dtype=torch.bfloat16, device=device)
    q_pe     = torch.randn(T, H, Dp, dtype=torch.bfloat16, device=device)
    ckv      = torch.randn(P, PS, D,  dtype=torch.bfloat16, device=device)
    kpe      = torch.randn(P, PS, Dp, dtype=torch.bfloat16, device=device)
    total    = P * PS
    si       = torch.full((T, TOPK), -1, dtype=torch.int32, device=device)
    for t in range(T):
        v = (valid_per_token[t] if valid_per_token else TOPK)
        v = min(v, total, TOPK)
        si[t, :v] = torch.randperm(total, device=device)[:v].int()
    return q_nope, q_pe, ckv, kpe, si

def alloc_out(T, device="cuda"):
    out = torch.zeros(T, H, D, dtype=torch.bfloat16, device=device)
    lse = torch.full((T, H), -float("inf"), dtype=torch.float32, device=device)
    return out, lse

def flops(sparse_indices):
    total = 0
    for t in range(sparse_indices.shape[0]):
        V = (sparse_indices[t] != -1).sum().item()
        if V == 0: continue
        total += 2*H*D*V + 2*H*Dp*V + 2*H*V*D + 5*H*V
    return total

def pretty_diff(name, ref, impl):
    """Print first/last 10 values side-by-side."""
    r, i = ref.flatten().float(), impl.flatten().float()
    n = r.numel()
    k = min(10, n)
    print(f"  {name} (shape={list(ref.shape)}, numel={n}):")
    print(f"  {'':>5} {'ref':>12} {'impl':>12} {'diff':>12}")
    idxs = list(range(k)) + (["..."] if n > 2*k else []) + list(range(n-k, n))
    for j in idxs:
        if j == "...":
            print(f"  {'...':>5}")
            continue
        d = abs(r[j].item() - i[j].item())
        print(f"  {j:>5} {r[j].item():>12.6f} {i[j].item():>12.6f} {d:>12.2e}")

def check(tag, args, atol=ATOL, rtol=RTOL):
    r_out, r_lse = alloc_out(args[0].shape[0])
    i_out, i_lse = alloc_out(args[0].shape[0])
    ref_fn(*args, SCALE, r_out, r_lse)
    impl_fn(*args, SCALE, i_out, i_lse)
    torch.cuda.synchronize()
    o_ok = torch.allclose(r_out.float(), i_out.float(), atol=atol, rtol=rtol)
    l_ok = torch.allclose(r_lse, i_lse, atol=atol, rtol=rtol)
    o_abs = (r_out.float() - i_out.float()).abs().max().item()
    l_abs = (r_lse - i_lse).abs().max().item()
    ok = o_ok and l_ok
    status = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {status}  out_err={o_abs:.2e}  lse_err={l_abs:.2e}")
    if not ok:
        pretty_diff("output", r_out, i_out)
        pretty_diff("lse", r_lse, i_lse)
    return ok

def bench(fn, args, warmup=10, iters=50):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in evs:
        s.record(); fn(*args); e.record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in evs) / iters

# ────────────────────────────────────────────
def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # ── Toy check ──
    if TOY_CHECK:
        print("=== TOY CHECK (T=2, P=4) ===")
        args = make_tensors(2, 4, valid_per_token=[30, 100])
        check("toy", args)
        return

    # ── Load real workloads ──
    workloads = [json.loads(l) for l in open(JSONL)]
    print(f"=== {len(workloads)} REAL WORKLOADS ===")

    if CHECK or MEASURE:
        hdr = f"{'#':>3} {'UUID':>10} {'T':>2} {'Valid':>6}"
        if CHECK: hdr += f" {'abs_err':>10} {'Status':>6}"
        if MEASURE: hdr += f" {'Ref ms':>8} {'Impl ms':>8} {'Speedup':>8} {'GFLOPS':>8}"
        print(hdr)
        print("-" * len(hdr))

    all_pass = True
    durations, gflops_list, speedups = [], [], []
    for i, w in enumerate(workloads):
        ax = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        T, P = ax["num_tokens"], ax["num_pages"]
        uuid = w["workload"]["uuid"][:8]

        # generate inputs
        q_nope, q_pe, ckv, kpe, _ = make_tensors(T, P)
        sf = load_file(str(CONTEST / inp["sparse_indices"]["path"]))
        si = sf[inp["sparse_indices"]["tensor_key"]].cuda()
        valid = (si != -1).sum().item()
        fl = flops(si)
        args = (q_nope, q_pe, ckv, kpe, si)

        line = f"{i+1:>3} {uuid:>10} {T:>2} {valid:>6}"

        if CHECK:
            r_out, r_lse = alloc_out(T)
            i_out, i_lse = alloc_out(T)
            ref_fn(*args, SCALE, r_out, r_lse)
            impl_fn(*args, SCALE, i_out, i_lse)
            torch.cuda.synchronize()
            o_abs = (r_out.float() - i_out.float()).abs().max().item()
            l_abs = (r_lse - i_lse).abs().max().item()
            ok = o_abs < ATOL and l_abs < ATOL
            if not ok: all_pass = False
            line += f" {max(o_abs,l_abs):>10.2e} {'PASS' if ok else 'FAIL':>6}"
            if not ok:
                print(line)
                pretty_diff("output", r_out, i_out)
                pretty_diff("lse", r_lse, i_lse)
                continue

        if MEASURE:
            def run_ref(qn, qp, c, k, s):
                o, l = alloc_out(qn.shape[0])
                ref_fn(qn, qp, c, k, s, SCALE, o, l)
            def run_impl(qn, qp, c, k, s):
                o, l = alloc_out(qn.shape[0])
                impl_fn(qn, qp, c, k, s, SCALE, o, l)
            r_ms = bench(run_ref, args)
            i_ms = bench(run_impl, args)
            sp = r_ms / i_ms if i_ms > 0 else 0
            gf = fl / (i_ms * 1e-3) / 1e9 if i_ms > 0 else 0
            line += f" {r_ms:>8.3f} {i_ms:>8.3f} {sp:>7.2f}x {gf:>8.2f}"
            durations.append(i_ms)
            gflops_list.append(gf)
            speedups.append(sp)

        print(line)

    if CHECK:
        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")

    if MEASURE and durations:
        def gmean(vals):
            return math.exp(sum(math.log(v) for v in vals) / len(vals))
        print(f"\n  Geomean:  duration={gmean(durations):.3f} ms  GFLOPS={gmean(gflops_list):.2f}  speedup={gmean(speedups):.3f}x")

if __name__ == "__main__":
    main()

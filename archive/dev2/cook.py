#!/usr/bin/env python3
"""Composable correctness + benchmark helpers for DSA sparse attention."""
import json, math, torch
from pathlib import Path
from safetensors.torch import load_file


def get_inputs(jsonl_path, max_samples=None,
               num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64, page_size=64, topk=2048,
               sm_scale=0.1352337788608801):
    """Load workloads and yield inputs matching ref.py's run() signature.

    Args:
        jsonl_path:    Path to the workload JSONL file.
        max_samples:   Max workloads to load (None = all).
        num_qo_heads:  Number of query/output heads (H).
        head_dim_ckv:  Compressed KV head dimension (D).
        head_dim_kpe:  Key positional-encoding head dimension.
        page_size:     Entries per KV-cache page.
        topk:          Max sparse indices per token (padded with -1).
        sm_scale:      Softmax scale factor, 1/sqrt(head_dim_ckv + head_dim_kpe).

    Yields per workload:
        dict with keys:
            q_nope:         [num_tokens, num_qo_heads, head_dim_ckv]  bf16
            q_pe:           [num_tokens, num_qo_heads, head_dim_kpe]  bf16
            ckv_cache:      [num_pages, page_size, head_dim_ckv]      bf16
            kpe_cache:      [num_pages, page_size, head_dim_kpe]      bf16
            sparse_indices: [num_tokens, topk]                        int32 (-1 = padding)
            sm_scale:       float
            num_tokens:     int (T)
            num_pages:      int (P)
            valid_per_token: list[int] — number of real (non-padding) indices per token
    """
    contest_dir = Path(jsonl_path).parent.parent.parent
    wls = [json.loads(l) for l in open(jsonl_path)]
    if max_samples is not None:
        wls = wls[:max_samples]
    for w in wls:
        ax, inp = w["workload"]["axes"], w["workload"]["inputs"]
        T, P = ax["num_tokens"], ax["num_pages"]
        q_nope    = torch.randn(T, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
        q_pe      = torch.randn(T, num_qo_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda")
        ckv_cache = torch.randn(P, page_size, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
        kpe_cache = torch.randn(P, page_size, head_dim_kpe, dtype=torch.bfloat16, device="cuda")
        sf = load_file(str(contest_dir / inp["sparse_indices"]["path"]))
        sparse_indices = sf[inp["sparse_indices"]["tensor_key"]].cuda()
        valid_per_token = [(sparse_indices[t] != -1).sum().item() for t in range(T)]
        yield dict(
            q_nope=q_nope, q_pe=q_pe, ckv_cache=ckv_cache, kpe_cache=kpe_cache,
            sparse_indices=sparse_indices, sm_scale=sm_scale,
            num_tokens=T, num_pages=P, valid_per_token=valid_per_token,
        )


def get_outputs(num_tokens, num_qo_heads=16, head_dim_ckv=512):
    """Allocate zeroed output + lse buffers matching ref.py's run() outputs.

    Args:
        num_tokens:   Number of tokens (T).
        num_qo_heads: Number of query/output heads.
        head_dim_ckv: Compressed KV head dimension.

    Returns:
        output: [num_tokens, num_qo_heads, head_dim_ckv]  bf16  — attention output
        lse:    [num_tokens, num_qo_heads]                 f32   — log-sum-exp / ln2
    """
    output = torch.zeros(num_tokens, num_qo_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device="cuda")
    return output, lse


def _call(fn, inp, output, lse):
    """Call fn with ref.py-style positional args."""
    fn(inp["q_nope"], inp["q_pe"], inp["ckv_cache"], inp["kpe_cache"],
       inp["sparse_indices"], inp["sm_scale"], output, lse)


def _count_workloads(jsonl_path, max_samples=None):
    """Count workloads without loading tensors."""
    n = sum(1 for _ in open(jsonl_path))
    return min(n, max_samples) if max_samples is not None else n


def check_correctness(impl_fn, ref_fn=None, jsonl_path=None, max_samples=None, atol=0.01):
    """Check impl_fn against ref_fn on real workloads. Returns True if all pass."""
    if ref_fn is None:
        from ref import run as ref_fn
    total = _count_workloads(jsonl_path, max_samples)
    print(f"Checking {total} workloads...")
    all_pass = True
    for i, inp in enumerate(get_inputs(jsonl_path, max_samples=max_samples)):
        T = inp["num_tokens"]
        r_out, r_lse = get_outputs(T)
        i_out, i_lse = get_outputs(T)
        _call(ref_fn, inp, r_out, r_lse)
        _call(impl_fn, inp, i_out, i_lse)
        torch.cuda.synchronize()
        o_err = (r_out.float() - i_out.float()).abs().max().item()
        l_err = (r_lse - i_lse).abs().max().item()
        ok = o_err < atol and l_err < atol
        if not ok:
            all_pass = False
        valids = ",".join(str(v) for v in inp["valid_per_token"])
        print(f"  [{i+1}/{total}] T={T}  valid=[{valids}]  out_err={o_err:.2e} lse_err={l_err:.2e} {'PASS' if ok else 'FAIL'}")
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def benchmark(impl_fn, ref_fn=None, jsonl_path=None, max_samples=None,
              warmup=10, iters=50, num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64):
    """Benchmark impl_fn (and optionally ref_fn) on real workloads."""
    total = _count_workloads(jsonl_path, max_samples)
    show_ref = ref_fn is not None
    hdr = f"{'#':>3} {'T':>2} {'Valid':>40}"
    if show_ref:
        hdr += f" {'Ref ms':>8}"
    hdr += f" {'Impl ms':>8}"
    if show_ref:
        hdr += f" {'Speedup':>8}"
    hdr += f" {'GFLOPS':>8}"
    print(hdr)
    print("-" * len(hdr))

    H, D, Dp = num_qo_heads, head_dim_ckv, head_dim_kpe
    durations, gflops_list, speedups = [], [], []
    for i, inp in enumerate(get_inputs(jsonl_path, max_samples=max_samples)):
        T = inp["num_tokens"]
        vpt = inp["valid_per_token"]
        fl = sum(2*H*D*v + 2*H*Dp*v + 2*H*v*D + 5*H*v for v in vpt if v > 0)

        def _run_impl():
            o, l = get_outputs(T, num_qo_heads=H, head_dim_ckv=D)
            _call(impl_fn, inp, o, l)

        for _ in range(warmup):
            _run_impl()
        torch.cuda.synchronize()
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
        for s, e in evs:
            s.record(); _run_impl(); e.record()
        torch.cuda.synchronize()
        i_ms = sum(s.elapsed_time(e) for s, e in evs) / iters

        valids = ",".join(str(v) for v in vpt)
        line = f"{i+1:>3} {T:>2} {valids:>40}"

        if show_ref:
            def _run_ref():
                o, l = get_outputs(T, num_qo_heads=H, head_dim_ckv=D)
                _call(ref_fn, inp, o, l)
            for _ in range(warmup):
                _run_ref()
            torch.cuda.synchronize()
            evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
            for s, e in evs:
                s.record(); _run_ref(); e.record()
            torch.cuda.synchronize()
            r_ms = sum(s.elapsed_time(e) for s, e in evs) / iters
            sp = r_ms / i_ms if i_ms > 0 else 0
            line += f" {r_ms:>8.3f}"
            speedups.append(sp)

        gf = fl / (i_ms * 1e-3) / 1e9 if i_ms > 0 else 0
        line += f" {i_ms:>8.3f}"
        if show_ref:
            line += f" {sp:>7.2f}x"
        line += f" {gf:>8.2f}"
        durations.append(i_ms)
        gflops_list.append(gf)
        print(line)

    if durations:
        gmean = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
        summary = f"\n  Geomean:  duration={gmean(durations):.3f} ms  GFLOPS={gmean(gflops_list):.2f}"
        if speedups:
            summary += f"  speedup={gmean(speedups):.3f}x"
        print(summary)


if __name__ == "__main__":
    from ref import run as ref_fn
    # from impl import run as impl_fn
    from impl_cutedsl import run as impl_fn

    ROOT    = Path(__file__).parent.parent
    CONTEST = ROOT.parent / "flashinfer26dsa" / "mlsys26-contest"
    JSONL   = str(CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl")

    CHECK   = True
    MEASURE = True
    MAX_SAMPLES = None

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    if CHECK:
        check_correctness(impl_fn, ref_fn, jsonl_path=JSONL, max_samples=MAX_SAMPLES)
    if MEASURE:
        benchmark(impl_fn, ref_fn, jsonl_path=JSONL, max_samples=MAX_SAMPLES)

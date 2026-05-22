"""PoC: evaluate threshold-estimation for 1-pass topk.

For each request in a dumped .pt file, we:
  1) Compute the true top-K threshold tau_true.
  2) Partition scores into coarse radix buckets (top-N bits) and find the
     smallest tau_hat at a bucket boundary such that
        count(score >= tau_hat) >= K           (no under-shoot)
     then measure miss-rate if we emit *all* elements with score >= tau_hat
     and fill the remainder (if any) arbitrarily.
  3) Report worst-case miss fraction and "overshoot" (how many extra
     elements we need to cull by a tie-breaker).

This tells us whether a single histogram pass over ceil(log2 bucket_count)
radix bits yields a threshold within the 1% grader tolerance.
"""
from pathlib import Path
import json
import struct
import torch

TOPK = 2048
TOLERANCE = 0.01

def float_to_radix_u32(t: torch.Tensor) -> torch.Tensor:
    """Monotone-increasing uint32 key for float32 (same as kernel).

    NaNs are mapped to 0 (lowest) so they never appear in topk.
    """
    t = torch.where(torch.isnan(t), torch.tensor(-float("inf"), dtype=t.dtype), t)
    u = t.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = (u >> 31) & 1
    pos = u ^ 0x80000000
    neg = (~u) & 0xFFFFFFFF
    out = torch.where(sign.bool(), neg, pos)
    return out.to(torch.int64) & 0xFFFFFFFF


def analyze(scores: torch.Tensor, bits: int):
    """
    scores: [sl]  (one request)
    bits:   how many top radix bits to histogram with (bucket_count = 2**bits)

    Returns dict with:
      bucket_count, tau_bucket (threshold bucket id),
      count_above_eq, count_above_strict, overshoot, undershoot, miss_fraction
    """
    sl = scores.numel()
    if sl <= TOPK:
        return dict(skipped=True, sl=sl)

    radix = float_to_radix_u32(scores)  # [sl]
    shift = 32 - bits
    bucket = (radix >> shift).to(torch.int64)  # [sl] in [0, 2**bits)
    bucket_count = 1 << bits

    # Histogram of buckets.
    hist = torch.bincount(bucket, minlength=bucket_count)

    # Cumulative count from the top bucket downward.
    # We want smallest bucket_id b such that sum_{b' >= b} hist[b'] >= TOPK.
    rev_cum = torch.flip(torch.cumsum(torch.flip(hist, [0]), 0), [0])
    # rev_cum[b] = count of elements with bucket >= b

    # Find threshold bucket: largest b with rev_cum[b] >= TOPK.
    ge_topk = (rev_cum >= TOPK).nonzero().flatten()
    if ge_topk.numel() == 0:
        # Shouldn't happen if sl > TOPK.
        return dict(bad=True)
    tau_bucket = int(ge_topk.max().item())
    count_above_strict = int((rev_cum[tau_bucket + 1] if tau_bucket + 1 < bucket_count else torch.tensor(0)).item())
    count_above_eq     = int(rev_cum[tau_bucket].item())
    tie_bucket_size    = count_above_eq - count_above_strict

    # Miss analysis:
    # If we emit ALL elements with bucket >= tau_bucket, we get count_above_eq
    # elements. We need exactly TOPK. overshoot = count_above_eq - TOPK.
    # We then either: (a) randomly drop `overshoot` ties → possibly miss up to
    # `overshoot` true topk entries in the tie bucket. Worst case:
    #   miss = min(overshoot, tie_bucket_size)
    # but since all ties are in the tie bucket, all ties *could* be in topk or
    # below-the-line. The true number of tie-bucket elements actually in topk is
    # (TOPK - count_above_strict). So dropping any `overshoot` ties
    # misclassifies at most `overshoot` elements.
    overshoot = count_above_eq - TOPK
    miss_fraction = overshoot / TOPK if overshoot > 0 else 0.0
    return dict(
        skipped=False,
        sl=sl,
        bits=bits,
        tau_bucket=tau_bucket,
        count_above_strict=count_above_strict,
        count_above_eq=count_above_eq,
        tie_bucket_size=tie_bucket_size,
        overshoot=overshoot,
        miss_fraction=miss_fraction,
    )


def main():
    here = Path(__file__).parent
    pt_path = here / "wl126.pt"
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    # Expect dict with 'scores' and 'seq_lens'.
    if isinstance(data, dict):
        print("keys:", list(data.keys()))
        scores = data.get("final") if "final" in data else data.get("scores")
        seq_lens = data.get("seq_lens")
    else:
        # Fallback if it's a tuple.
        scores, seq_lens = data[0], data[1]
    print(f"scores shape: {tuple(scores.shape)}  dtype: {scores.dtype}")
    print(f"seq_lens: {seq_lens.tolist() if hasattr(seq_lens,'tolist') else seq_lens}")

    B = scores.shape[0]
    out = {}
    for bits in [5, 6, 7, 8, 10, 12]:
        bucket_count = 1 << bits
        misses = []
        overshoots = []
        for b in range(B):
            sl = int(seq_lens[b])
            sub = scores[b, :sl].float()
            r = analyze(sub, bits)
            if r.get("skipped"):
                continue
            misses.append(r["miss_fraction"])
            overshoots.append(r["overshoot"])
        if not misses:
            continue
        m = torch.tensor(misses)
        o = torch.tensor(overshoots, dtype=torch.float32)
        out[bits] = dict(
            bucket_count=bucket_count,
            requests=len(misses),
            miss_max=float(m.max()),
            miss_mean=float(m.mean()),
            miss_p99=float(m.quantile(0.99)),
            overshoot_max=float(o.max()),
            overshoot_mean=float(o.mean()),
            pass_1pct=int((m <= TOLERANCE).sum()),
        )

    print("\n── threshold-estimation PoC (wl126, 1-pass) ──")
    print(f"{'bits':>4} {'buckets':>8} {'reqs':>5} {'pass@1%':>8} {'miss_max':>9} {'miss_mean':>10} {'over_max':>9} {'over_mean':>10}")
    for bits, r in out.items():
        print(f"{bits:>4} {r['bucket_count']:>8} {r['requests']:>5} "
              f"{r['pass_1pct']:>8} {r['miss_max']:>9.4f} {r['miss_mean']:>10.4f} "
              f"{int(r['overshoot_max']):>9} {r['overshoot_mean']:>10.1f}")

    (here / "histogram_poc_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

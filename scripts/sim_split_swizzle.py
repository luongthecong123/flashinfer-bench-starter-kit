"""
Simulator: per-CTA latency under various (t, split) → split permutations.

Model
-----
- Grid = NUM_SPLITS CTAs (per head-group; head-group axis is symmetric so
  we collapse it — each head-group sees the same nv pattern).
- Each CTA owns one split_idx s and loops over T tokens.
- For token t, CTA s actually services split  perm(t, s).
- Cost(t, split) = 5 µs if nv(t, split) > 0 else 0 µs.
- Total kernel latency = max over CTAs of the sum of per-token costs.

nv(t, split) is derived from the workload's max_valid[t] array:
the first max_valid[t] entries of sparse_indices are non-negative,
the rest are -1.  → nv(t, s) = clip(max_valid[t] - s*128, 0, 128).
"""

NUM_SPLITS = 16
SPLIT_SIZE = 128
COST_US    = 5.0

WORKLOADS = {
    17: ("564007ac", 8, [288, 4, 1884, 21, 136, 2048, 42, 335]),
    23: ("2207f0fd", 7, [415, 131, 2011, 148, 263, 169, 462]),
}

def nv_table(max_valid):
    """nv[t][s] = number of valid entries CTA s handles for token t."""
    T = len(max_valid)
    nv = [[0] * NUM_SPLITS for _ in range(T)]
    for t in range(T):
        mv = max_valid[t]
        for s in range(NUM_SPLITS):
            nv[t][s] = max(0, min(SPLIT_SIZE, mv - s * SPLIT_SIZE))
    return nv

# ── Permutations: perm(t, s) returns the split-id this CTA-s handles for token t.
def perm_baseline(t, s, NS): return s
def perm_xor     (t, s, NS): return t ^ s
def perm_rot     (t, s, NS): return (s + t) % NS
def perm_stride  (stride):
    return lambda t, s, NS, k=stride: (s + t * k) % NS

SCHEMES = {
    "baseline":      perm_baseline,
    "xor":           perm_xor,
    "rot+1":         perm_rot,
    "rot stride=3":  perm_stride(3),
    "rot stride=5":  perm_stride(5),
    "rot stride=7":  perm_stride(7),
    "rot stride=9":  perm_stride(9),
}

def simulate(nv, perm):
    T  = len(nv)
    NS = NUM_SPLITS
    per_cta_active = [0] * NS         # # of active tokens this CTA does
    per_cta_visits = [[0]*NS for _ in range(NS)]  # CTA s → which splits visited
    for s in range(NS):
        for t in range(T):
            split = perm(t, s, NS)
            per_cta_visits[s][split] = 1
            if nv[t][split] > 0:
                per_cta_active[s] += 1
    per_cta_us  = [a * COST_US for a in per_cta_active]
    total_us    = max(per_cta_us)
    # Bijection check: every (t, split) must be visited exactly once.
    bij_ok = True
    for t in range(T):
        seen = [0]*NS
        for s in range(NS):
            seen[perm(t, s, NS)] += 1
        if seen != [1]*NS:
            bij_ok = False
            break
    return total_us, per_cta_active, bij_ok

def fmt_active(arr):
    return "[" + " ".join(f"{a:>2d}" for a in arr) + "]"

def run_workload(wl_id):
    uuid, T, mv = WORKLOADS[wl_id]
    nv = nv_table(mv)

    # Per-split totals (active token count per split, summed over T)
    total_active_per_split = [sum(1 for t in range(T) if nv[t][s] > 0)
                              for s in range(NUM_SPLITS)]
    grand_total_active = sum(total_active_per_split)
    ideal_us = grand_total_active * COST_US / NUM_SPLITS

    print(f"\n══════ Workload {wl_id}  (uuid={uuid}, T={T}) ══════")
    print(f"max_valid                : {mv}")
    print(f"active CTAs per split    : {fmt_active(total_active_per_split)}")
    print(f"sum active (t,split)     : {grand_total_active}")
    print(f"ideal balanced latency   : {ideal_us:.2f} µs")
    print(f"")
    print(f"{'scheme':<14} {'total µs':>10}  {'min/max active per CTA':>25}  bij  per-CTA active count")
    print(f"{'-'*14} {'-'*10}  {'-'*25}  ---  --------------------")
    for name, perm in SCHEMES.items():
        total, per_cta, bij = simulate(nv, perm)
        bij_s = "OK" if bij else "FAIL"
        mn, mx = min(per_cta), max(per_cta)
        print(f"{name:<14} {total:>10.2f}  {f'{mn} / {mx}':>25}  {bij_s:>3}  {fmt_active(per_cta)}")

if __name__ == "__main__":
    for wl in (17, 23):
        run_workload(wl)

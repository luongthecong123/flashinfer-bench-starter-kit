"""Simulate per-request split-rotation strategies for the tcgen05 v2 kernel.

Geometry (kv_split_umma_v2):
  - Grid: [NUM_HEADS // HEADS_PER_SPLIT, NUM_SPLITS, 1] = [8, 16, 1]
  - Each CTA loops over T_idx in [0..T-1]
  - Inside the loop, a CTA with grid coord (h, split_old) processes one MMA over
    DIM_SPLIT=128 KV positions selected by `split_new = π_t(split_old)`.
  - Cost per (CTA, T_idx) ≈ task_cost(num_valid_in_split_new(T_idx))

For load balance, only `split_old` matters (h-CTAs share work along split_old).
We want a family of permutations π_t : split_old → split_new (one per request t)
minimising  max over split_old of  sum_t cost_t[π_t(split_old)].

Three families considered:
  1. Pure additive rotation per request:  π_t(o) = (o + R[t]) mod 16
  2. Pure XOR per request:                π_t(o) = o ^ X[t]
  3. Composition (XOR then ROT):          π_t(o) = ((o ^ X[t]) + R[t]) mod 16

For each family we report:
  - per-workload best (oracle)
  - one global schedule that minimises arithmetic mean (over all 23 workloads)
    of max-CTA cost — to compare against the kernel's current fixed ROT=7.
"""
import math
from itertools import product

NUM_SPLITS = 16
DIM_SPLIT = 128
LIMIT_REQUEST = 8

# Same workloads as utils.WORKLOAD_INFO
WORKLOADS = [
    ("0c23b10c", 1, [2]),
    ("9d4a5f21", 2, [18, 11]),
    ("b7668cfd", 2, [33, 52]),
    ("0a63b87b", 2, [63, 9]),
    ("05f6de65", 2, [6, 337]),
    ("fc85411e", 2, [17, 13]),
    ("e6b849f2", 2, [92, 48]),
    ("9f3f891b", 2, [288, 4]),
    ("f77df5ce", 2, [18, 19]),
    ("385742b2", 8, [92, 48, 1044, 14, 411, 30, 16, 8]),
    ("4c46a94b", 8, [18, 19, 1002, 31, 11, 316, 24, 2]),
    ("38389961", 7, [33, 52, 72, 17, 18, 401, 1089]),
    ("02d6ae9c", 8, [63, 9, 2048, 212, 11, 25, 6, 50]),
    ("ddfa9e34", 6, [6, 9, 9, 14, 1639, 71]),
    ("78b2e11c", 8, [18, 11, 2048, 20, 25, 45, 135, 326]),
    ("68d6817d", 6, [19, 20, 32, 12, 25, 3]),
    ("564007ac", 8, [288, 4, 1884, 21, 136, 2048, 42, 335]),
    ("ae4219a9", 7, [19, 12, 2048, 21, 26, 46, 136]),
    ("232ed014", 8, [35, 54, 74, 19, 20, 403, 1091, 1]),
    ("7a389715", 8, [8, 11, 11, 16, 1641, 73, 1, 1]),
    ("5096e459", 8, [17, 13, 1887, 16, 180, 1986, 413, 1]),
    ("d57eb9e1", 6, [143, 139, 2013, 142, 306, 539]),
    ("2207f0fd", 7, [415, 131, 2011, 148, 263, 169, 462]),
]

# ── Cost model ────────────────────────────────────────────────────────────────
# The MMA writes a fixed (128 x N) tile regardless of num_valid, but cp.async
# work + softmax/output inner loops are bounded by ceil(num_valid). Use
# linear-with-fixed-overhead.
FIXED_OH   = 8.0     # MMA + softmax fixed overhead per active task
PER_VALID  = 1.0     # per-KV-row cost (cp.async + FMA)

def task_cost(local_valid: int) -> float:
    if local_valid <= 0:
        return 0.0     # CTA short-circuits via `if num_valid > 0`
    return FIXED_OH + PER_VALID * local_valid


def per_split_work(vc: int):
    """work[split=0..15] for a single token with valid_count=vc."""
    return [max(0, min(DIM_SPLIT, vc - s * DIM_SPLIT)) for s in range(NUM_SPLITS)]


def workload_costs(vc_list):
    """costs[t][s] for t in [0..T-1], s in [0..15]."""
    return [[task_cost(w) for w in per_split_work(vc)] for vc in vc_list]


# ── Permutation evaluation ────────────────────────────────────────────────────

def evaluate(costs, perms):
    """Given costs[t][s] and perms[t]: split_old → split_new,
    return (max_cta_cost, arith_mean_cta_cost)."""
    T = len(costs)
    cta = [0.0] * NUM_SPLITS
    for t in range(T):
        pt = perms[t]
        ct = costs[t]
        for o in range(NUM_SPLITS):
            cta[o] += ct[pt[o]]
    return max(cta), sum(cta) / NUM_SPLITS


def perm_rot(r):
    return [(o + r) % NUM_SPLITS for o in range(NUM_SPLITS)]

def perm_xor(x):
    return [o ^ x for o in range(NUM_SPLITS)]

def perm_xor_rot(x, r):
    return [((o ^ x) + r) % NUM_SPLITS for o in range(NUM_SPLITS)]


# ── Per-workload oracle via coord-descent (was: brute-force 16^T = 4.3e9 for T=8) ──

def _oracle_coord(costs, perm_fns, n_choices, init=None, max_passes=8, restarts=8, seed=0):
    """Greedy coord-descent over per-T choices to minimise max-CTA cost.
    perm_fns[v] returns a permutation list[16] for choice v.
    Returns (best_max, best_choices).
    """
    import random
    rng = random.Random(seed)
    T = len(costs)
    perms_table = [perm_fns(v) for v in range(n_choices)]

    def eval_cur(choices):
        cta = [0.0] * NUM_SPLITS
        for t in range(T):
            p = perms_table[choices[t]]
            ct = costs[t]
            for o in range(NUM_SPLITS):
                cta[o] += ct[p[o]]
        return max(cta), cta

    best_overall = math.inf
    best_choice  = None
    for r in range(restarts):
        if r == 0:
            cur = [0] * T if init is None else list(init)
        else:
            cur = [rng.randrange(n_choices) for _ in range(T)]
        cur_max, cta = eval_cur(cur)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            for t in range(T):
                # Subtract this slot's current contribution
                old_p = perms_table[cur[t]]
                ct = costs[t]
                base = [cta[o] - ct[old_p[o]] for o in range(NUM_SPLITS)]
                best_v = cur[t]
                best_m = cur_max
                best_cta = cta
                for v in range(n_choices):
                    p = perms_table[v]
                    new_cta = [base[o] + ct[p[o]] for o in range(NUM_SPLITS)]
                    m = max(new_cta)
                    if m < best_m - 1e-9:
                        best_m = m
                        best_v = v
                        best_cta = new_cta
                if best_v != cur[t]:
                    cur[t] = best_v
                    cta = best_cta
                    cur_max = best_m
                    improved = True
            passes += 1
        if cur_max < best_overall:
            best_overall = cur_max
            best_choice = list(cur)
    return best_overall, best_choice


def oracle_rot(costs):
    return _oracle_coord(costs, lambda r: perm_rot(r), NUM_SPLITS)

def oracle_xor(costs):
    return _oracle_coord(costs, lambda x: perm_xor(x), NUM_SPLITS)


# ── Global schedules: a single (R[0..7] or X[0..7]) used for ALL workloads ──
# Search by greedy coordinate descent (pure brute force is 16^8 = 4.3e9).

def score_schedule_R(R, all_costs):
    """Aggregate (sum of max-CTA cost over workloads) for given global rotation R.
    Each workload uses R[:T_w]."""
    total = 0.0
    for costs in all_costs:
        T = len(costs)
        perms = [perm_rot(R[t]) for t in range(T)]
        m, _ = evaluate(costs, perms)
        total += m
    return total

def score_schedule_X(X, all_costs):
    total = 0.0
    for costs in all_costs:
        T = len(costs)
        perms = [perm_xor(X[t]) for t in range(T)]
        m, _ = evaluate(costs, perms)
        total += m
    return total

def score_schedule_XR(XR, all_costs):
    total = 0.0
    for costs in all_costs:
        T = len(costs)
        perms = [perm_xor_rot(*XR[t]) for t in range(T)]
        m, _ = evaluate(costs, perms)
        total += m
    return total

def coord_descent(score_fn, init, n_per_slot, all_costs, max_passes=8):
    """Greedy: cycle through slots, pick best value for each."""
    cur = list(init)
    cur_score = score_fn(cur, all_costs)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        for i in range(LIMIT_REQUEST):
            best_v = cur[i]
            best_s = cur_score
            for v in range(n_per_slot):
                if v == cur[i]:
                    continue
                cur[i] = v
                s = score_fn(cur, all_costs)
                if s < best_s - 1e-9:
                    best_s = s
                    best_v = v
            cur[i] = best_v
            if best_s < cur_score - 1e-9:
                cur_score = best_s
                improved = True
        passes += 1
    return cur, cur_score


def random_restart(score_fn, n_per_slot, all_costs, restarts=30, seed=0):
    """Multi-restart coord descent."""
    import random
    rng = random.Random(seed)
    best = None
    best_s = math.inf
    for r in range(restarts):
        if r == 0:
            init = [0] * LIMIT_REQUEST           # identity baseline
        elif r == 1:
            init = [(7 * t) % n_per_slot for t in range(LIMIT_REQUEST)]  # current ROT=7
        else:
            init = [rng.randrange(n_per_slot) for _ in range(LIMIT_REQUEST)]
        sched, s = coord_descent(score_fn, init, n_per_slot, all_costs)
        if s < best_s:
            best_s = s
            best = sched
    return best, best_s


# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    all_costs = [workload_costs(vc) for _, _, vc in WORKLOADS]

    # Baseline: identity (no rotation), as if swz_rot_shift=0
    print("=" * 110)
    print("Per-workload max-CTA cost — different schedules")
    print("=" * 110)
    print(f"{'WL':>3}  {'uuid':>9}  {'T':>2}  "
          f"{'identity':>9}  {'rot=7':>9}  "
          f"{'best fixed ROT (per-WL)':>26}  "
          f"{'best per-T ROT (per-WL)':>26}  "
          f"{'best per-T XOR (per-WL)':>26}")
    print("-" * 110)

    for wl_idx, (uuid, T, vc) in enumerate(WORKLOADS):
        costs = all_costs[wl_idx]

        ident, _ = evaluate(costs, [perm_rot(0)] * T)
        rot7,  _ = evaluate(costs, [perm_rot(7)] * T)

        # Best single ROT applied to all requests of this workload
        best_fixed = math.inf
        best_fixed_r = 0
        for r in range(NUM_SPLITS):
            m, _ = evaluate(costs, [perm_rot(r)] * T)
            if m < best_fixed:
                best_fixed = m
                best_fixed_r = r

        m_or_R, R = oracle_rot(costs)
        m_or_X, X = oracle_xor(costs)

        print(f"{wl_idx+1:>3}  {uuid:>9}  {T:>2}  "
              f"{ident:>9.1f}  {rot7:>9.1f}  "
              f"{best_fixed:>10.1f} (R={best_fixed_r:>2}) {'':>4}  "
              f"{m_or_R:>10.1f} {str(list(R)):>15}  "
              f"{m_or_X:>10.1f} {str(list(X)):>15}")

    # ── Global schedules ─────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("GLOBAL schedules (one R[0..7] or X[0..7] used for ALL workloads)")
    print("Optimised via coord-descent w/ random restarts on Σ max-CTA over 23 WLs.")
    print("=" * 110)

    # Global rotation
    R_glb, R_score = random_restart(score_schedule_R, NUM_SPLITS, all_costs, restarts=40)
    X_glb, X_score = random_restart(score_schedule_X, NUM_SPLITS, all_costs, restarts=40)

    # Global XOR-then-ROT (search XR pairs jointly)
    def score_schedule_XR_pair(pairs, all_costs):
        total = 0.0
        for costs in all_costs:
            T = len(costs)
            perms = [perm_xor_rot(*pairs[t]) for t in range(T)]
            m, _ = evaluate(costs, perms)
            total += m
        return total

    def coord_descent_pairs(all_costs, init, max_passes=8):
        cur = list(init)
        cur_s = score_schedule_XR_pair(cur, all_costs)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            for i in range(LIMIT_REQUEST):
                best_p = cur[i]
                best_s = cur_s
                for x in range(NUM_SPLITS):
                    for r in range(NUM_SPLITS):
                        cur[i] = (x, r)
                        s = score_schedule_XR_pair(cur, all_costs)
                        if s < best_s - 1e-9:
                            best_s = s
                            best_p = (x, r)
                cur[i] = best_p
                if best_s < cur_s - 1e-9:
                    cur_s = best_s
                    improved = True
            passes += 1
        return cur, cur_s

    import random
    rng = random.Random(0)
    best_XR = None
    best_XR_s = math.inf
    for r in range(20):
        if r == 0:
            init = [(0, 0)] * LIMIT_REQUEST
        elif r == 1:
            init = [(0, 7)] * LIMIT_REQUEST
        else:
            init = [(rng.randrange(NUM_SPLITS), rng.randrange(NUM_SPLITS)) for _ in range(LIMIT_REQUEST)]
        pairs, s = coord_descent_pairs(all_costs, init)
        if s < best_XR_s:
            best_XR_s = s
            best_XR = pairs

    # Fixed-rot baselines for comparison
    sum_ident = sum(evaluate(c, [perm_rot(0)] * len(c))[0] for c in all_costs)
    sum_rot7  = sum(evaluate(c, [perm_rot(7)] * len(c))[0] for c in all_costs)

    # Best single global rotation R applied to all requests
    best_gR = 0
    best_gR_s = math.inf
    for r in range(NUM_SPLITS):
        s = sum(evaluate(c, [perm_rot(r)] * len(c))[0] for c in all_costs)
        if s < best_gR_s:
            best_gR_s = s
            best_gR = r

    print(f"\n  Sum of max-CTA cost over 23 workloads (lower = better):")
    print(f"    identity (R=0)              : {sum_ident:>10.1f}")
    print(f"    fixed rot=7 (current)       : {sum_rot7:>10.1f}   ({sum_ident/sum_rot7:.3f}x vs identity)")
    print(f"    best single global ROT (R={best_gR:>2}): {best_gR_s:>10.1f}   ({sum_ident/best_gR_s:.3f}x vs identity)")
    print(f"    per-T ROT  (global R[0..7]) : {R_score:>10.1f}   ({sum_ident/R_score:.3f}x vs identity)")
    print(f"      R = {R_glb}")
    print(f"    per-T XOR  (global X[0..7]) : {X_score:>10.1f}   ({sum_ident/X_score:.3f}x vs identity)")
    print(f"      X = {X_glb}")
    print(f"    per-T XOR+ROT (global)      : {best_XR_s:>10.1f}   ({sum_ident/best_XR_s:.3f}x vs identity)")
    print(f"      XR = {best_XR}")

    # ── Per-workload comparison with the chosen global schedules ─────────────
    print("\n" + "=" * 110)
    print("Per-workload max-CTA cost with each global schedule")
    print("=" * 110)
    hdr = (f"{'WL':>3}  {'uuid':>9}  {'T':>2}  "
           f"{'identity':>9}  {'rot=7':>9}  {'gROT='+str(best_gR):>9}  "
           f"{'per-T ROT':>10}  {'per-T XOR':>10}  {'per-T XR':>10}")
    print(hdr)
    print("-" * len(hdr))

    geo_pairs = []
    for wl_idx, (uuid, T, vc) in enumerate(WORKLOADS):
        costs = all_costs[wl_idx]
        ident, _ = evaluate(costs, [perm_rot(0)] * T)
        rot7,  _ = evaluate(costs, [perm_rot(7)] * T)
        gR,    _ = evaluate(costs, [perm_rot(best_gR)] * T)
        pR,    _ = evaluate(costs, [perm_rot(R_glb[t]) for t in range(T)])
        pX,    _ = evaluate(costs, [perm_xor(X_glb[t]) for t in range(T)])
        pXR,   _ = evaluate(costs, [perm_xor_rot(*best_XR[t]) for t in range(T)])
        geo_pairs.append((rot7, pXR))
        print(f"{wl_idx+1:>3}  {uuid:>9}  {T:>2}  "
              f"{ident:>9.1f}  {rot7:>9.1f}  {gR:>9.1f}  "
              f"{pR:>10.1f}  {pX:>10.1f}  {pXR:>10.1f}")

    # Geometric mean speedup over rot=7
    log_sum = 0.0
    for rot7, pXR in geo_pairs:
        log_sum += math.log(rot7 / pXR) if pXR > 0 else 0.0
    print(f"\n  Geo-mean speedup of (per-T XOR+ROT) over (fixed rot=7): "
          f"{math.exp(log_sum / len(geo_pairs)):.3f}x")

    # ── Formulaic search ─────────────────────────────────────────────────────
    # split_new = ((split_old ^ (T_idx * X_MUL % 16)) + T_idx * R_MUL % 16) % 16
    # X_MUL=0 → pure additive rotation; R_MUL=0 → pure XOR.
    print("\n" + "=" * 110)
    print("FORMULAIC search: split_new = ((split_old ^ (T_idx * X_MUL % 16)) + T_idx * R_MUL % 16) % 16")
    print("Sweep all 16×16=256 (X_MUL, R_MUL) pairs, score = Σ max-CTA over 23 WLs")
    print("=" * 110)

    best_formula_score = math.inf
    formula_scores = {}
    for x_mul in range(NUM_SPLITS):
        for r_mul in range(NUM_SPLITS):
            total = 0.0
            for costs in all_costs:
                T = len(costs)
                perms = [perm_xor_rot((t * x_mul) % NUM_SPLITS,
                                      (t * r_mul) % NUM_SPLITS)
                         for t in range(T)]
                m, _ = evaluate(costs, perms)
                total += m
            formula_scores[(x_mul, r_mul)] = total
            if total < best_formula_score:
                best_formula_score = total
                best_x_mul, best_r_mul = x_mul, r_mul

    # Show top-15
    ranked = sorted(formula_scores.items(), key=lambda kv: kv[1])
    print(f"\n  Top-15 formulas (lower score = more balanced):")
    print(f"  {'X_MUL':>7}  {'R_MUL':>7}  {'Σ max-CTA':>12}  {'vs identity':>12}  {'vs lookup XR':>14}  {'geo-mean spdup vs rot=7':>24}")
    print(f"  {'-'*90}")
    for (xm, rm), s in ranked[:15]:
        geo_ls = 0.0
        for wl_idx, (uuid, T, vc) in enumerate(WORKLOADS):
            costs = all_costs[wl_idx]
            perms = [perm_xor_rot((t * xm) % NUM_SPLITS, (t * rm) % NUM_SPLITS) for t in range(T)]
            m, _ = evaluate(costs, perms)
            rot7_m, _ = evaluate(costs, [perm_rot(7)] * T)
            geo_ls += math.log(rot7_m / m) if m > 0 else 0.0
        geo_sp = math.exp(geo_ls / len(WORKLOADS))
        print(f"  {xm:>7}  {rm:>7}  {s:>12.1f}  {sum_ident/s:>11.3f}x  {best_XR_s/s:>13.3f}x  {geo_sp:>23.3f}x")

    print(f"\n  Best formula: X_MUL={best_x_mul}, R_MUL={best_r_mul}")
    print(f"    score={best_formula_score:.1f}  ({sum_ident/best_formula_score:.3f}x vs identity, "
          f"{best_XR_s/best_formula_score:.3f}x vs lookup XR)")
    print(f"\n  In kernel (constexpr, no table):")
    print(f"    split_idx_new = ((split_idx_old ^ (T_idx * {best_x_mul} % NUM_SPLITS))")
    print(f"                   + T_idx * {best_r_mul} % NUM_SPLITS) % NUM_SPLITS")

    # Per-workload detail for best formula vs rot=7 vs lookup XR
    print(f"\n  Per-workload cost: best formula vs rot=7 vs lookup XR")
    print(f"  {'WL':>3}  {'T':>2}  {'rot=7':>9}  {'lookup XR':>10}  {'formula':>9}  {'formula/rot7':>13}")
    print(f"  {'-'*55}")
    geo_log = 0.0
    for wl_idx, (uuid, T, vc) in enumerate(WORKLOADS):
        costs = all_costs[wl_idx]
        rot7_m, _ = evaluate(costs, [perm_rot(7)] * T)
        lxr_m,  _ = evaluate(costs, [perm_xor_rot(*best_XR[t]) for t in range(T)])
        fml_m,  _ = evaluate(costs, [perm_xor_rot((t * best_x_mul) % NUM_SPLITS,
                                                   (t * best_r_mul) % NUM_SPLITS)
                                     for t in range(T)])
        ratio = fml_m / rot7_m if rot7_m > 0 else 1.0
        geo_log += math.log(rot7_m / fml_m) if fml_m > 0 else 0.0
        print(f"  {wl_idx+1:>3}  {T:>2}  {rot7_m:>9.1f}  {lxr_m:>10.1f}  {fml_m:>9.1f}  {ratio:>12.3f}x")
    print(f"\n  Geo-mean speedup of best formula over fixed rot=7: "
          f"{math.exp(geo_log / len(WORKLOADS)):.3f}x")



if __name__ == "__main__":
    main()

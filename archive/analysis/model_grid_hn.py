"""Model compute time for H×nsplit grid launch strategy.

Grid = [H * nsplit, 1, 1].  Each CTA owns one (head, split) pair and
loops over all T tokens — processing the token if local_valid > 0, else
skipping (zero cost).

Wall-clock  = max over all CTAs of sum_tok f(local_valid[tok, split])
            = max over splits of sum_tok f(local_valid[tok, split])
              (all heads are symmetric → same cost, so we only need
               to max over splits)

Cost model  f(N) — piecewise-linear interpolation through fused thr_warpv2 data
  (solution_exp.md, "Speedup vs thr_warp" table, thr_warpv2 column, 10-rep mean).

  These are single-block fused timings (score + softmax + output) for a CTA
  processing N valid KV entries.  Early-exit at small N is captured by the data.

    N      fused thr_warpv2 (µs)
      0      0.00   (anchor: empty tile)
     18      6.52
     52      7.68
     92      8.94
    337     17.84
   1044     41.94
   2048     75.18

  Interpolation: piecewise linear (numpy.interp).
  Extrapolation beyond N=2048: linear extension of the last segment.

  Roofline formula:
    tile_cost(lv, dim_split) = 0           if lv ≤ 0
                              = interp(N)  otherwise, N = min(lv, dim_split)

    roof = max(max_tile_cost,  Σ_{tok,split,head} tile_cost / n_sms)
         = max(critical_path,  total_work / n_sms)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.utils import WORKLOAD_INFO

# ── Per-tile cost model — fused thr_warpv2 interpolation (solution_exp.md) ──
# Single-block fused (score+softmax+output) measured timings.
# Source: "Speedup vs thr_warp" table, thr_warpv2 column, 10-rep mean.
# N=0 anchor ensures interp returns 0 for empty tiles.
N_FUSED    = np.array([   0.,   18.,   52.,   92.,  337.,  1044.,  2048.])
COST_FUSED = np.array([3,  6.52,  7.68,  8.94, 17.84,  41.94,  75.18])

# Linear extrapolation for N > 2048 (extend last segment)
_extrap_slope = (COST_FUSED[-1] - COST_FUSED[-2]) / (N_FUSED[-1] - N_FUSED[-2])
_extrap_base  = COST_FUSED[-1] - _extrap_slope * N_FUSED[-1]

print("Tile cost model: piecewise-linear interp of fused thr_warpv2 (score+softmax+output)")
print(f"  N    = {N_FUSED.tolist()}")
print(f"  cost = {COST_FUSED.tolist()} µs")
print(f"  Extrap slope (N>2048): {_extrap_slope:.5f} µs/N")
print(f"  f(92)={np.interp(92,N_FUSED,COST_FUSED):.2f}  "
      f"f(256)={np.interp(256,N_FUSED,COST_FUSED):.2f}  "
      f"f(512)={np.interp(512,N_FUSED,COST_FUSED):.2f}  "
      f"f(1044)={np.interp(1044,N_FUSED,COST_FUSED):.2f} µs\n")

def tile_cost_us(lv: int, dim_split: int) -> float:
    """Cost of one (tok, head, split) tile with lv valid entries.

    Piecewise-linear interpolation through fused thr_warpv2 measurements.
    N = min(lv, dim_split). Extrapolates linearly beyond N=2048.
    The flat 6–9 µs floor at small N naturally captures early-exit behaviour.
    """
    if lv <= 0:
        return 0.0
    N = float(min(lv, dim_split))
    if N >= N_FUSED[-1]:
        return float(_extrap_slope * N + _extrap_base)
    return float(np.interp(N, N_FUSED, COST_FUSED))

TOP_K = 2048

def analyze_workload(label, valid_list, dim_split, nsplit):
    """
    valid_list : list of per-token valid counts  (length = T)
    Grid = H * nsplit CTAs.  Each CTA = (head, split) loops over T tokens.

    Returns wall_clock_us = max over splits of (sum over tok of tile_cost).
    (All H heads are identical so we only need to model over splits.)
    """
    T = len(valid_list)
    H = 16

    # For each split, compute sum_tok cost
    max_cta_cost = 0.0
    cta_costs = []
    for s in range(nsplit):
        split_start = s * dim_split
        cta_cost = 0.0
        for tok in range(T):
            lv = max(0, min(dim_split, valid_list[tok] - split_start))
            cta_cost += tile_cost_us(lv, dim_split)
        cta_costs.append(cta_cost)
        if cta_cost > max_cta_cost:
            max_cta_cost = cta_cost

    # Wall clock = max CTA cost (all H heads run in parallel on different CTAs)
    return max_cta_cost, cta_costs

def analyze_xor_swizzle(valid_list, dim_split=256, nsplit=None):
    """H×S static grid with diagonal swizzle over the split dimension.

    CTA(head=h, split=s) processes split (s ^ (tok % nsplit)) for power-of-2
    nsplit (single XOR, free), or (s + tok) % nsplit for general nsplit.
    Both are bijections: each CTA visits all nsplit splits exactly once per
    nsplit-aligned token block.  Wall-clock = max over CTAs.
    """
    T = len(valid_list)
    if nsplit is None:
        nsplit = TOP_K // dim_split  # 8 for dim_split=256

    is_pow2 = nsplit > 0 and (nsplit & (nsplit - 1)) == 0

    cta_costs = [0.0] * nsplit
    for s in range(nsplit):
        for tok in range(T):
            eff_split = (s ^ (tok % nsplit)) if is_pow2 else ((s + tok) % nsplit)
            lv = max(0, min(dim_split, valid_list[tok] - eff_split * dim_split))
            cta_costs[s] += tile_cost_us(lv, dim_split)

    return max(cta_costs)


def analyze_v3b_swizzled(valid_list, dim_split=256, n_ctas=128, phase_shift=20):
    """Model 128-CTA persistent with per-round phase swizzle replicating the
    148-CTA diagonal.  Each round the within-block offset is shifted by
    phase_shift, so CTA i processes a different (split, head) each round.

    swizzled_flat = (round * 128) + (cta_id + round * phase_shift) % 128
    """
    T  = len(valid_list)
    H  = 16
    nsplit = TOP_K // dim_split
    n_tiles_per_cta = (T * nsplit * H + n_ctas - 1) // n_ctas  # rounds per CTA

    def decode_TSH(flat):
        n_sh = nsplit * H
        tok   =  flat // n_sh
        split = (flat %  n_sh) // H
        head  =  flat % H
        return tok, split, head

    cta_totals = [0.0] * n_ctas
    block_size = n_ctas  # = 128
    total_tiles = T * nsplit * H

    for rnd in range(n_tiles_per_cta):
        for cta in range(n_ctas):
            block_start  = rnd * block_size
            phased_off   = (cta + rnd * phase_shift) % block_size
            swizzled     = block_start + phased_off
            if swizzled >= total_tiles:
                continue
            tok, split, head = decode_TSH(swizzled)
            lv = max(0, min(dim_split, valid_list[tok] - split * dim_split))
            cta_totals[cta] += tile_cost_us(lv, dim_split)

    return max(cta_totals)


def roofline_us(valid_list, dim_split=256, n_sms=148):
    """Theoretical minimum wall-clock for n_sms perfectly balanced SMs.

    roof = max(critical_path, throughput_bound)
      critical_path     = cost of the single most expensive tile
                          (no schedule can hide a long tile)
      throughput_bound  = total_work / n_sms
                          (work summed over ALL T×nsplit×H tiles;
                           H=16 identical heads each count separately)

    Any schedule that achieves this is optimal.
    """
    T      = len(valid_list)
    H      = 16
    nsplit = TOP_K // dim_split

    max_cost   = 0.0
    total_work = 0.0
    for tok in range(T):
        for split in range(nsplit):
            lv   = max(0, min(dim_split, valid_list[tok] - split * dim_split))
            c    = tile_cost_us(lv, dim_split)
            if c > max_cost:
                max_cost = c
            total_work += c * H   # H identical head-tiles per (tok,split)

    return max(max_cost, total_work / n_sms)


def analyze_v3b_sorted_rr(valid_list, dim_split=256, n_ctas=148):
    """CUTLASS-style: sort all tiles by descending cost, then assign
    round-robin to n_ctas.  Every CTA naturally gets one heavy + several
    light tiles — equivalent to the grouped-scheduler sort_problems() idea.

    Tile pool = T×nsplit×H tiles (all 1024 for T=8, nsplit=8, H=16).
    H head-tiles per (tok,split) pair are identical in cost but are separate
    physical tiles that must each run on some SM — they must be included.
    """
    T      = len(valid_list)
    H      = 16
    nsplit = TOP_K // dim_split

    # H copies of each (tok,split) cost — these are the 1024 real tiles
    tile_costs = sorted(
        (tile_cost_us(max(0, min(dim_split, valid_list[tok] - split * dim_split)), dim_split)
         for tok in range(T) for split in range(nsplit) for _ in range(H)),
        reverse=True
    )

    cta_totals = [0.0] * n_ctas
    for i, cost in enumerate(tile_costs):
        cta_totals[i % n_ctas] += cost

    return max(cta_totals)


def analyze_xor128_steal20(valid_list, dim_split=256, n_steal=20):
    """128 CTAs with XOR swizzle (static, generalizable).  20 extra CTAs
    opportunistically steal tiles at runtime.

    Steal policy: each extra CTA grabs the single most expensive remaining
    tile from the currently max-loaded base CTA.  This reduces that CTA's
    total and the extra CTA runs the stolen tile independently in parallel.

    Models are at the (tok, split) granularity — H heads are symmetric so
    every split CTA represents all H identical head CTAs.
    """
    T      = len(valid_list)
    nsplit = TOP_K // dim_split

    # XOR assignment — split CTA s owns T tiles
    col_tiles = []
    for s in range(nsplit):
        tiles = []
        for tok in range(T):
            eff_split = s ^ (tok % nsplit)
            lv = max(0, min(dim_split, valid_list[tok] - eff_split * dim_split))
            tiles.append(tile_cost_us(lv, dim_split))
        col_tiles.append(tiles)

    col_totals = [sum(t) for t in col_tiles]

    # Greedy steal: each extra CTA picks the most expensive tile from the
    # max-loaded base CTA, reducing that CTA's load and running it in parallel
    extra_costs = []
    for _ in range(n_steal):
        max_col = max(range(nsplit), key=lambda c: col_totals[c])
        if not col_tiles[max_col]:
            break
        max_tile_idx = max(range(len(col_tiles[max_col])),
                           key=lambda i: col_tiles[max_col][i])
        stolen = col_tiles[max_col].pop(max_tile_idx)
        col_totals[max_col] -= stolen
        extra_costs.append(stolen)

    return max(max(col_totals), max(extra_costs) if extra_costs else 0.0)


def analyze_v3b_xor_swizzled(valid_list, dim_split=256, n_ctas=148):
    """148-CTA persistent round-robin + XOR permutation within each token block.

    Same schedule as v3b (stride=n_ctas), but tile-decode applies:
        split_eff = split_orig ^ (tok % nsplit)
    This is a bijection within each 128-tile block (nsplit must be power-of-2),
    so every (tok, split, head) tile is still covered exactly once.
    No tuned parameter — fully determined by nsplit at compile time.
    """
    T      = len(valid_list)
    H      = 16
    nsplit = TOP_K // dim_split  # must be power-of-2 for XOR bijection
    total_tiles = T * nsplit * H
    n_sh  = nsplit * H  # tiles per token block (128 for nsplit=8)

    def decode_xor(flat_idx):
        tok        =  flat_idx // n_sh
        off        =  flat_idx %  n_sh
        split_orig =  off // H
        head       =  off %  H
        split_eff  =  split_orig ^ (tok % nsplit)
        return tok, split_eff, head

    cta_totals = [0.0] * n_ctas
    tile = 0
    while tile < total_tiles:
        for cta in range(n_ctas):
            if tile >= total_tiles:
                break
            tok, split, _ = decode_xor(tile)
            lv = max(0, min(dim_split, valid_list[tok] - split * dim_split))
            cta_totals[cta] += tile_cost_us(lv, dim_split)
            tile += 1

    return max(cta_totals)


def analyze_v3b_roundrobin(valid_list, dim_split=256, n_ctas=148):
    """Model v3b persistent round-robin: 148 CTAs pick tiles in order.
    Each CTA i gets tiles: i, i+n_ctas, i+2*n_ctas, ...
    Tile flat_idx = tok*(nsplit*H) + split*H + head  (T>S>H ordering)
    Returns wall-clock = max CTA total cost.
    """
    T  = len(valid_list)
    H  = 16
    nsplit = TOP_K // dim_split
    total_tiles = T * nsplit * H

    # Precompute tile cost for every tile
    def tile_cost_flat(flat_idx):
        n_sh = nsplit * H
        tok   =  flat_idx // n_sh
        split = (flat_idx %  n_sh) // H
        split_start = split * dim_split
        lv = max(0, min(dim_split, valid_list[tok] - split_start))
        return tile_cost_us(lv, dim_split)

    cta_totals = [0.0] * n_ctas
    tile = 0
    while tile < total_tiles:
        for cta in range(n_ctas):
            if tile >= total_tiles:
                break
            cta_totals[cta] += tile_cost_flat(tile)
            tile += 1

    return max(cta_totals)

# ── Grid configs ──────────────────────────────────────────────────────────────
configs = [
    ("DS=256  nsplit=8  CTAs=128",  256, 8),
    ("DS=228  nsplit=9  CTAs=144",  228, 9),
    ("DS=512  nsplit=4  CTAs=64",   512, 4),
]

# ── Print summary table (pandas for clean alignment) ────────────────────────
#   roof-fml = formula lower bound: max(max_tile_cost, total_work/148)
#   XOR-128-stl = upper bound only: stealing can only improve XOR-128, shown as <XX.XX
#   Ratio columns (X/fml): 1.0x = optimal, higher = farther from ideal

import pandas as pd

def _gm(lst): return np.exp(np.mean(np.log(lst)))

gm_all = {k: [] for k in ['v3b', 'srr', 'x128', 'x144', 'sw80']}
gm_t2p = {k: [] for k in ['v3b', 'srr', 'x128', 'x144', 'sw80']}  # T > 2 only

rows = []
for i, (uuid, T, valid_list) in enumerate(WORKLOAD_INFO):
    label    = f"WL{i+1}"
    maxv     = max(valid_list)
    v3b148   = analyze_v3b_roundrobin(valid_list, dim_split=256, n_ctas=148)
    roof     = roofline_us(valid_list, dim_split=256, n_sms=148)
    srr      = analyze_v3b_sorted_rr(valid_list, dim_split=256, n_ctas=148)
    x128     = analyze_xor_swizzle(valid_list, dim_split=256, nsplit=8)
    x144     = analyze_xor_swizzle(valid_list, dim_split=228, nsplit=9)
    sw80     = analyze_v3b_swizzled(valid_list, dim_split=256, n_ctas=128, phase_shift=80)
    ratios = {'v3b': v3b148/roof, 'srr': srr/roof,
              'x128': x128/roof, 'x144': x144/roof, 'sw80': sw80/roof}
    for k, r in ratios.items():
        gm_all[k].append(r)
        if T > 2:
            gm_t2p[k].append(r)
    bound = "T" if roof == max(tile_cost_us(max(0, min(256, valid_list[tok] - s*256)), 256)
                               for tok in range(T) for s in range(TOP_K//256)) else "W"
    # XOR-128-stl: stealing can only reduce XOR-128; show strict upper bound
    xs_str = f"<{x128:.2f}" if x128 > roof + 0.01 else f" {x128:.2f}"
    rows.append({
        'WL': label, 'T': T, 'MaxN': maxv,
        'v3b(µs)':    f"{v3b148:.2f}",
        'roof(µs)':   f"{roof:.2f}{bound}",
        'srr(µs)':    f"{srr:.2f}",
        'XOR-128':    f"{x128:.2f}",
        'XOR-144':    f"{x144:.2f}",
        'XOR-128-stl': xs_str,
        'SW80':       f"{sw80:.2f}",
        'v3b/fml':    f"{v3b148/roof:.3f}x",
        'srr/fml':    f"{srr/roof:.3f}x",
        'x128/fml':   f"{x128/roof:.3f}x",
        'x144/fml':   f"{x144/roof:.3f}x",
        'sw80/fml':   f"{sw80/roof:.3f}x",
    })

# Geo-mean rows
_E = ''
rows.append({
    'WL': 'Geo(all)', 'T': _E, 'MaxN': _E,
    'v3b(µs)': _E, 'roof(µs)': _E, 'srr(µs)': _E,
    'XOR-128': _E, 'XOR-144': _E, 'XOR-128-stl': _E, 'SW80': _E,
    'v3b/fml':  f"{_gm(gm_all['v3b']):.3f}x",
    'srr/fml':  f"{_gm(gm_all['srr']):.3f}x",
    'x128/fml': f"{_gm(gm_all['x128']):.3f}x",
    'x144/fml': f"{_gm(gm_all['x144']):.3f}x",
    'sw80/fml': f"{_gm(gm_all['sw80']):.3f}x",
})
rows.append({
    'WL': 'Geo(T>2)', 'T': _E, 'MaxN': _E,
    'v3b(µs)': _E, 'roof(µs)': _E, 'srr(µs)': _E,
    'XOR-128': _E, 'XOR-144': _E, 'XOR-128-stl': _E, 'SW80': _E,
    'v3b/fml':  f"{_gm(gm_t2p['v3b']):.3f}x",
    'srr/fml':  f"{_gm(gm_t2p['srr']):.3f}x",
    'x128/fml': f"{_gm(gm_t2p['x128']):.3f}x",
    'x144/fml': f"{_gm(gm_t2p['x144']):.3f}x",
    'sw80/fml': f"{_gm(gm_t2p['sw80']):.3f}x",
})

df = pd.DataFrame(rows)
tbl = df.to_string(index=False, col_space=2)
lines = tbl.splitlines()
sep = '-' * len(lines[0])
# Insert separators: after header (line 0), after WL9 (line 9+1=10), before geo rows (last 2)
out = [lines[0], sep]
for j, ln in enumerate(lines[1:], start=1):
    out.append(ln)
    if j == 9:           # after WL9 — last T≤2 row
        out.append(sep)
    elif j == len(lines) - 3:  # after WL23 — before geo rows
        out.append(sep)
print('\n' + '\n'.join(out) + '\n')
print("roof-fml     (T=tile-bound, W=work-bound) = max(max_tile_cost, Σ tile_cost / 148)  [true lower bound]")
print("sorted-RR    = 148 CTAs, all 1024 tiles sorted desc, round-robin  [no atomic, static]")
print("XOR-128      = 128 static CTAs (8 splits×16 heads), eff_split = s ^ (tok%8)   [ds=256]")
print("XOR-144      = 144 static CTAs (9 splits×16 heads), eff_split = (s+tok)%9      [ds=228]")
print("XOR-128-stl  = XOR-128 + work stealing; upper bound shown as <XOR-128 (exact not modeled)")
print("SW80         = 128-CTA persistent + phase_shift=80  [tuned, 128 SMs only]")
print()
print(f"\n{'Workload':<8} {'T':>2}  {'MaxValid':>9}  {'v3b-RR':>8}  {'HxNS-256':>10}  {'HxNS-228':>10}  {'ratio-256':>10}  {'ratio-228':>10}")
print("-" * 90)

geomean_256, geomean_228 = [], []
for i, (uuid, T, valid_list) in enumerate(WORKLOAD_INFO):
    label  = f"WL{i+1}"
    maxv   = max(valid_list)
    v3b    = analyze_v3b_roundrobin(valid_list, dim_split=256)
    hn256, _ = analyze_workload(label, valid_list, 256, 8)
    hn228, _ = analyze_workload(label, valid_list, 228, 9)
    r256 = hn256 / v3b
    r228 = hn228 / v3b
    geomean_256.append(r256)
    geomean_228.append(r228)
    print(f"{label:<8} {T:>2}  {maxv:>9}  {v3b:>8.2f}  {hn256:>10.2f}  {hn228:>10.2f}  {r256:>10.2f}x  {r228:>10.2f}x")

gm256 = np.exp(np.mean(np.log(geomean_256)))
gm228 = np.exp(np.mean(np.log(geomean_228)))
# Row prefix width = 8+1+2+2+9+2+8+2+10+2+10+2 = 58 chars before ratios
print(f"\n{'Geo-mean (HxNS/v3b)':>56}  {gm256:>10.3f}x  {gm228:>10.3f}x")
print()
print("v3b-RR  = modeled v3b persistent 148 CTAs round-robin (matches NCU ~25µs at WL20)")
print("HxNS-*  = H×nsplit grid: each CTA owns (head,split), serializes over T tokens")
print("ratio   = HxNS / v3b-RR  (>1 means H×nsplit is SLOWER)")

# ── Per-split breakdown ────────────────────────────────────────────────────────
PROBE = [9, 12, 16, 19, 22]   # 0-indexed: WL10, WL13, WL17, WL20, WL23

print()
for wl_idx in PROBE:
    uuid, T, valid_list = WORKLOAD_INFO[wl_idx]
    label = f"WL{wl_idx+1}"
    v3b = analyze_v3b_roundrobin(valid_list)
    print(f"\n{label}  T={T}  valid={valid_list}  v3b-RR-model={v3b:.2f}µs")
    for name, ds, ns in configs[:2]:
        wall, cta_costs = analyze_workload(label, valid_list, ds, ns)
        costs_str = "  ".join(f"s{s}:{c:.1f}" for s, c in enumerate(cta_costs))
        print(f"  [{name}]  wall={wall:.2f} µs   CTA split-costs: {costs_str}")



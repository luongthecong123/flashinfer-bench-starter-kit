#!/usr/bin/env python3
"""
kv_split_v3 block scheduling simulator — WL20 load-balance analysis.

Explores how block-ordering strategies affect heavy-work distribution across SMs.

WL20: T=8  H=16  S=8 (DIM_SPLIT=256)  valid=[8,11,11,16,1641,73,1,1]
B200: 148 SMs × 2 blk/SM = 296 concurrent slots (thread-limited)

Key question:
  All H=16 heads for the same (tok, split) do IDENTICAL work.
  Can we exploit this symmetry to achieve perfect load balance?
"""

import heapq
from collections import Counter

# ── WL20 setup ────────────────────────────────────────────────────────────
T, H, S    = 8, 16, 8
DIM_SPLIT  = 256
VALID      = [8, 11, 11, 16, 1641, 73, 1, 1]

HEAVY      = 5.0   # µs  — split overlaps with valid token range
OOB        = 0.3   # µs  — split is entirely beyond valid range

# B200: 148 SMs, thread-limited at 2 blk/SM (16 warps × 128 threads = 2048 max)
N_SM       = 148
C          = N_SM * 2   # 296 concurrent slots

def work(tok: int, split: int) -> float:
    return HEAVY if VALID[tok] > split * DIM_SPLIT else OOB

# ── Block classification ──────────────────────────────────────────────────
print("=" * 72)
print("  kv_split_v3  WL20  block distribution simulator")
print("=" * 72)

all_blocks = [(t, h, s) for t in range(T) for h in range(H) for s in range(S)]
n_total = len(all_blocks)   # 1024
n_heavy = sum(1 for t, _, s in all_blocks if work(t, s) == HEAVY)
n_oob   = n_total - n_heavy

print(f"\nConfig: T={T}  H={H}  S={S}  DIM_SPLIT={DIM_SPLIT}")
print(f"        {N_SM} SMs × 2 blk/SM = {C} concurrent slots")
print(f"        {n_total} total blocks   {n_heavy} heavy ({n_heavy*100//n_total}%)   {n_oob} OOB ({n_oob*100//n_total}%)")

print(f"\n  tok  valid  heavy_splits  heavy_blks  OOB_blks")
print(f"  {'─'*52}")
for t in range(T):
    hs = sum(1 for s in range(S) if work(t, s) == HEAVY)
    tag = "  ◀ DOMINANT" if t == 4 else ""
    print(f"  {t}    {VALID[t]:>4}    {hs}/8         {hs*H:>4} ({hs*H*HEAVY:.0f}µs)   {(S-hs)*H:>4} ({(S-hs)*H*OOB:.0f}µs){tag}")

# Worst case: 1 SM gets ALL heavy blocks of the worst token
worst_tok = max(range(T), key=lambda t: sum(work(t,s)==HEAVY for s in range(S)))
worst_heavy_per_tok = sum(1 for s in range(S) if work(worst_tok, s) == HEAVY) * H
print(f"\n  Worst-case (1 SM gets all heavy blocks of tok{worst_tok}): "
      f"{worst_heavy_per_tok} blks × {HEAVY}µs = {worst_heavy_per_tok * HEAVY:.0f}µs")
print(f"  Ideal (perfect balance):  {n_heavy} heavy / {C} slots × {HEAVY}µs = {n_heavy * HEAVY / C:.1f}µs")

# ── Simulation models ──────────────────────────────────────────────────────

def wave_sim(blocks):
    """
    CUDA default: block i → SM (i % C). SM executes its queue sequentially.
    This is a simplified model where all blocks in a wave start at t=0.
    Returns (makespan, per-SM load list).
    """
    sm = [0.0] * C
    for i, (t, h, s) in enumerate(blocks):
        sm[i % C] += work(t, s)
    return max(sm), sm

def greedy_sim(blocks):
    """
    Persistent / ideal scheduler: each slot picks next block when free.
    Lower bound on real makespan given perfect FIFO scheduling.
    """
    hp = [0.0] * C
    heapq.heapify(hp)
    for t, h, s in blocks:
        heapq.heappush(hp, heapq.heappop(hp) + work(t, s))
    return max(hp)

def max_heavy_per_sm(blocks):
    hv = [0] * C
    for i, (t, _, s) in enumerate(blocks):
        if work(t, s) == HEAVY:
            hv[i % C] += 1
    return max(hv), hv

# ── Ordering strategies ───────────────────────────────────────────────────
#
# All 6 permutations of the 3 loop dimensions: T (token), H (head), S (split)
# The head dimension is the free dimension — heads for the same (tok,split)
# do exactly the same work, so we can place them anywhere in the launch order.

strategies = {
    "T > H > S  (default)": [(t,h,s) for t in range(T) for h in range(H) for s in range(S)],
    "T > S > H":             [(t,h,s) for t in range(T) for s in range(S) for h in range(H)],
    "H > T > S":             [(t,h,s) for h in range(H) for t in range(T) for s in range(S)],
    "H > S > T":             [(t,h,s) for h in range(H) for s in range(S) for t in range(T)],
    "S > T > H":             [(t,h,s) for s in range(S) for t in range(T) for h in range(H)],
    "S > H > T":             [(t,h,s) for s in range(S) for h in range(H) for t in range(T)],
    "heavy-first ↓ (oracle)":sorted(all_blocks, key=lambda b: -work(b[0], b[2])),
    "light-first ↑ (oracle)":sorted(all_blocks, key=lambda b:  work(b[0], b[2])),
}

# ── Print comparison ──────────────────────────────────────────────────────
print("\n\n" + "─" * 78)
print(f"{'Strategy':<28}  {'wave_ms':>8}  {'greedy_ms':>10}  {'max_H/SM':>8}  {'heavy blks/SM distribution':>28}")
print("─" * 78)

results = {}
for name, blks in strategies.items():
    mk_w, sm_loads   = wave_sim(blks)
    mk_g             = greedy_sim(blks)
    mh, hv_list      = max_heavy_per_sm(blks)
    hist             = Counter(hv_list)
    hist_str         = "  ".join(f"{hist[k]}×[{k}H]" for k in sorted(hist) if k in hist)
    results[name]    = (mk_w, mk_g, mh, hist, sm_loads, hv_list)
    print(f"{name:<28}  {mk_w:>7.1f}µs  {mk_g:>9.1f}µs  {mh:>7}H  {hist_str}")

# ── Per-wave breakdown for default vs S>T>H ───────────────────────────────

def wave_breakdown(blocks, name):
    print(f"\n── Wave breakdown: {name} ─────────────────────────────────────────────────")
    waves = max(1, (len(blocks) + C - 1) // C)
    tok4_heavy_per_wave = []
    for w_idx in range(waves):
        start, end = w_idx * C, min((w_idx + 1) * C, len(blocks))
        wave_blks  = [blocks[i] for i in range(start, end)]
        n_h        = sum(1 for t, _, s in wave_blks if work(t, s) == HEAVY)
        n_o        = len(wave_blks) - n_h
        tok4_h     = sum(1 for t, _, s in wave_blks if t == 4 and work(t, s) == HEAVY)
        max_sm_h   = max((sum(1 for i, (t,_,sp) in enumerate(blocks)
                              if i % C == (start + j) % C and work(t, sp) == HEAVY)
                         for j in range(len(wave_blks))), default=0)
        print(f"  wave {w_idx+1} [{start:4}..{end-1:4}]  {len(wave_blks):3} blks  heavy={n_h:3}  OOB={n_o:3}  tok4_heavy={tok4_h:3}")

wave_breakdown(strategies["T > H > S  (default)"], "T > H > S  (default)")
wave_breakdown(strategies["S > T > H"],             "S > T > H")

# ── Top-N worst SMs for select strategies ────────────────────────────────
def show_top_sms(name, n=8):
    mk_w, sm_loads, mh, hist, _, hv_list = (
        results[name][0], results[name][4], results[name][2],
        results[name][3], results[name][5], results[name][5])
    print(f"\n── Top {n} loaded SMs: {name} ──")
    ranked = sorted(range(C), key=lambda i: -sm_loads[i])[:n]
    for sm in ranked:
        # show which (tok,split) blocks this SM processes
        blks = strategies[name]
        assigned = [(t, h, s) for i, (t, h, s) in enumerate(blks) if i % C == sm]
        detail = "  ".join(
            f"tok{t}s{s}({'H' if work(t,s)==HEAVY else 'O'})" for t,h,s in assigned)
        print(f"  SM {sm:3d}: {sm_loads[sm]:5.1f}µs  [{hv_list[sm]}H]  {detail}")

show_top_sms("T > H > S  (default)")
show_top_sms("S > T > H")

# ── Swizzle formula ────────────────────────────────────────────────────────
print(f"""
{"=" * 72}
  Swizzle formula for CUDA / CuTe kernel
{"=" * 72}

  Default (T > H > S) — blockIdx.x encodes as:
    tok   = blockIdx.x // (H * S)     # {H*S} consecutive blocks share same token
    head  = (blockIdx.x // S) % H
    split = blockIdx.x % S

    → all H={H} heads of heavy tok4 land in block IDs [{4*H*S}..{5*H*S - 1}]
    → clustered in waves 2–3, max {H} heavy blocks per SM possible

  Rotated (S > T > H) — reorder outer loop to split:
    split = blockIdx.x // (T * H)     # {T*H} consecutive blocks share same split
    tok   = (blockIdx.x // H) % T
    head  = blockIdx.x % H

    → tok4's 7 heavy splits land at block groups spaced {T*H} = {T*H} apart
    → each group has only H={H} consecutive tok4 heavy blocks
    → {T*H}-spacing across C={C} slots → spread across ~{(T*H*100)//C}% of each wave
    → max heavy blocks per SM drops from {results["T > H > S  (default)"][2]} → {results["S > T > H"][2]}

  In CuTe DSL (Python):
    # Grid: (S, T, H) instead of (T, H, S)
    # Inside kernel: use blockIdx.z=head, blockIdx.y=tok, blockIdx.x=split

  Key insight: since all H heads for the same (tok, split) do IDENTICAL work,
  we can place them anywhere in the launch order without changing correctness.
  Rotating the outermost dimension from T → S (or H) breaks the cluster of
  heavy tok4 blocks across {T*H} positions instead of just {H*S}.
""")

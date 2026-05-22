"""Simulate persistent-kernel work distribution strategies.

Goal: find a task ordering that minimises max-CTA wall-clock
(= the critical path) across all 23 workloads.

Current problem with T>S>H round-robin:
  - 75-88% of tasks are OOB (local_valid=0)
  - Some CTAs get only OOB work, while others are overloaded
  - Wall-clock = max(CTA work), so idle CTAs are waste

Strategies tested:
  1. baseline    — current T>S>H encoding, round-robin
  2. skip_oob    — only schedule tasks with local_valid > 0, round-robin
  3. sorted_desc — sort valid tasks by work (desc), round-robin
  4. greedy_lpt  — longest-processing-time-first greedy bin-packing
"""

NUM_HEADS = 16
NUM_SPLITS = 8
DIM_SPLIT = 256
N_SMs = 148

WORKLOADS = [
    (1,  1, [2]),
    (2,  2, [18, 11]),
    (3,  2, [33, 52]),
    (4,  2, [63, 9]),
    (5,  2, [6, 337]),
    (6,  2, [17, 13]),
    (7,  2, [92, 48]),
    (8,  2, [288, 4]),
    (9,  2, [18, 19]),
    (10, 8, [92, 48, 1044, 14, 411, 30, 16, 8]),
    (11, 8, [18, 19, 1002, 31, 11, 316, 24, 2]),
    (12, 7, [33, 52, 72, 17, 18, 401, 1089]),
    (13, 8, [63, 9, 2048, 212, 11, 25, 6, 50]),
    (14, 6, [6, 9, 9, 14, 1639, 71]),
    (15, 8, [18, 11, 2048, 20, 25, 45, 135, 326]),
    (16, 6, [19, 20, 32, 12, 25, 3]),
    (17, 8, [288, 4, 1884, 21, 136, 2048, 42, 335]),
    (18, 7, [19, 12, 2048, 21, 26, 46, 136]),
    (19, 8, [35, 54, 74, 19, 20, 403, 1091, 1]),
    (20, 8, [8, 11, 11, 16, 1641, 73, 1, 1]),
    (21, 8, [17, 13, 1887, 16, 180, 1986, 413, 1]),
    (22, 6, [143, 139, 2013, 142, 306, 539]),
    (23, 7, [415, 131, 2011, 148, 263, 169, 462]),
]

# ── Cost model ────────────────────────────────────────────────────────────────
# From NCU data: score scales ~linearly with local_valid, softmax/output too.
# Approximate: cost(task) ∝ local_valid for valid tasks, ~0.1 for OOB sentinel write.
OOB_COST = 0.1  # relative cost of an OOB task (sentinel write)

def task_cost(local_valid: int) -> float:
    """Approximate relative cost of a task."""
    if local_valid == 0:
        return OOB_COST
    return float(local_valid)


def build_all_tasks(T: int, valid_list: list[int]):
    """Build the full task list: [(tok, split, head, local_valid), ...]"""
    tasks = []
    for tok in range(T):
        vc = valid_list[tok]
        for split in range(NUM_SPLITS):
            lv = max(0, min(DIM_SPLIT, vc - split * DIM_SPLIT))
            for head in range(NUM_HEADS):
                tasks.append((tok, split, head, lv))
    return tasks


def assign_round_robin(tasks, n_ctas):
    """Assign tasks to CTAs in order: task[i] → CTA i % n_ctas."""
    cta_cost = [0.0] * n_ctas
    for i, (tok, split, head, lv) in enumerate(tasks):
        cta_cost[i % n_ctas] += task_cost(lv)
    return cta_cost


def assign_greedy_lpt(tasks, n_ctas):
    """Longest Processing Time first: assign each task to the least-loaded CTA."""
    import heapq
    # Sort tasks by cost descending
    sorted_tasks = sorted(tasks, key=lambda t: task_cost(t[3]), reverse=True)
    # Min-heap of (total_cost, cta_id)
    heap = [(0.0, i) for i in range(n_ctas)]
    heapq.heapify(heap)
    for tok, split, head, lv in sorted_tasks:
        cost, cta = heapq.heappop(heap)
        heapq.heappush(heap, (cost + task_cost(lv), cta))
    return [c for c, _ in sorted(heap, key=lambda x: x[1])]


# ── Strategies ────────────────────────────────────────────────────────────────

def strategy_baseline(T, valid_list):
    """Current T>S>H encoding, all 1024 tasks, round-robin over 148 CTAs."""
    tasks = build_all_tasks(T, valid_list)
    n_ctas = min(N_SMs, len(tasks))
    return assign_round_robin(tasks, n_ctas), n_ctas, len(tasks)


def strategy_skip_oob(T, valid_list):
    """Only schedule tasks with local_valid > 0, round-robin."""
    all_tasks = build_all_tasks(T, valid_list)
    valid_tasks = [(tok, split, head, lv) for tok, split, head, lv in all_tasks if lv > 0]
    n_ctas = min(N_SMs, len(valid_tasks))
    return assign_round_robin(valid_tasks, n_ctas), n_ctas, len(valid_tasks)


def strategy_sorted_desc(T, valid_list):
    """Sort valid tasks by work descending, then round-robin."""
    all_tasks = build_all_tasks(T, valid_list)
    valid_tasks = [(tok, split, head, lv) for tok, split, head, lv in all_tasks if lv > 0]
    valid_tasks.sort(key=lambda t: t[3], reverse=True)
    n_ctas = min(N_SMs, len(valid_tasks))
    return assign_round_robin(valid_tasks, n_ctas), n_ctas, len(valid_tasks)


def strategy_greedy_lpt(T, valid_list):
    """Skip OOB, then greedy LPT bin-packing into CTAs."""
    all_tasks = build_all_tasks(T, valid_list)
    valid_tasks = [(tok, split, head, lv) for tok, split, head, lv in all_tasks if lv > 0]
    n_ctas = min(N_SMs, len(valid_tasks))
    cta_costs = assign_greedy_lpt(valid_tasks, n_ctas)
    return cta_costs, n_ctas, len(valid_tasks)


def strategy_skip_oob_sorted_snake(T, valid_list):
    """Skip OOB, sort desc, snake (boustrophedon) assignment.
    Tasks 0..N-1 sorted by work desc.
    Row 0: CTA 0,1,...,N_CTA-1 gets tasks 0,1,...
    Row 1: CTA N_CTA-1,...,1,0 gets next batch (reversed)
    This ensures heaviest tasks are spread across all CTAs."""
    all_tasks = build_all_tasks(T, valid_list)
    valid_tasks = [(tok, split, head, lv) for tok, split, head, lv in all_tasks if lv > 0]
    valid_tasks.sort(key=lambda t: t[3], reverse=True)
    n_ctas = min(N_SMs, len(valid_tasks))
    cta_cost = [0.0] * n_ctas
    for i, (tok, split, head, lv) in enumerate(valid_tasks):
        row = i // n_ctas
        col = i % n_ctas
        if row % 2 == 1:
            col = n_ctas - 1 - col  # reverse direction
        cta_cost[col] += task_cost(lv)
    return cta_cost, n_ctas, len(valid_tasks)


# ── Run all strategies ────────────────────────────────────────────────────────

STRATEGIES = {
    "baseline":    strategy_baseline,
    "skip_oob":    strategy_skip_oob,
    "sorted_desc": strategy_sorted_desc,
    "snake":       strategy_skip_oob_sorted_snake,
    "greedy_lpt":  strategy_greedy_lpt,
}

print(f"{'':>4s} {'':>3s} {'':>38s}  ", end="")
for name in STRATEGIES:
    print(f"  {name:>12s}", end="")
print()

print(f"{'WL':>4s} {'T':>3s} {'valid_list':>38s}  ", end="")
for name in STRATEGIES:
    print(f"  {'max_cta':>12s}", end="")
print()
print("=" * (50 + 14 * len(STRATEGIES)))

for wl_id, T, valid_list in WORKLOADS:
    vl_str = str(valid_list)
    if len(vl_str) > 38:
        vl_str = vl_str[:35] + "..."
    print(f"{wl_id:>4d} {T:>3d} {vl_str:>38s}  ", end="")

    baseline_max = None
    for name, fn in STRATEGIES.items():
        cta_costs, n_ctas, n_tasks = fn(T, valid_list)
        max_cost = max(cta_costs)
        total_cost = sum(cta_costs)
        ideal = total_cost / n_ctas

        if name == "baseline":
            baseline_max = max_cost
            print(f"  {max_cost:>10.1f}  ", end="")
        else:
            speedup = baseline_max / max_cost if max_cost > 0 else float("inf")
            print(f"  {max_cost:>7.1f} {speedup:>4.2f}x", end="")
    print()

# ── Detailed analysis for WL20 ───────────────────────────────────────────────
print("\n" + "=" * 80)
print("DETAILED: WL20 — skip_oob vs baseline")
print("=" * 80)

wl_id, T, valid_list = WORKLOADS[19]
all_tasks = build_all_tasks(T, valid_list)
valid_tasks = [(tok, split, head, lv) for tok, split, head, lv in all_tasks if lv > 0]

print(f"\nAll tasks: {len(all_tasks)}, Valid tasks: {len(valid_tasks)}")
print(f"OOB ratio: {1 - len(valid_tasks)/len(all_tasks):.0%}")

# Show valid task breakdown per token
print(f"\nValid tasks per token:")
for tok in range(T):
    vc = valid_list[tok]
    active_splits = (vc + DIM_SPLIT - 1) // DIM_SPLIT
    n_valid = active_splits * NUM_HEADS
    print(f"  tok{tok}: valid_count={vc:5d}, active_splits={active_splits}, "
          f"valid_tasks={n_valid:3d}, work/task={[max(0, min(DIM_SPLIT, vc - s*DIM_SPLIT)) for s in range(active_splits)]}")

# Skip OOB strategy detail
n_ctas = min(N_SMs, len(valid_tasks))
print(f"\nWith skip_oob: {len(valid_tasks)} tasks across {n_ctas} CTAs")
print(f"  Tasks/CTA: {len(valid_tasks)/n_ctas:.1f}")
print(f"  Ideal work/CTA: {sum(task_cost(lv) for _,_,_,lv in valid_tasks)/n_ctas:.1f}")

# Greedy LPT detail
cta_costs_lpt, _, _ = strategy_greedy_lpt(T, valid_list)
cta_costs_base, _, _ = strategy_baseline(T, valid_list)
print(f"\n  Baseline max CTA cost: {max(cta_costs_base):.1f}")
print(f"  Greedy LPT max CTA cost: {max(cta_costs_lpt):.1f}")
print(f"  Speedup: {max(cta_costs_base)/max(cta_costs_lpt):.2f}x")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("GEOMETRIC MEAN SPEEDUP OVER BASELINE (across all 23 workloads)")
print("=" * 80)

import math
for name, fn in STRATEGIES.items():
    if name == "baseline":
        continue
    log_sum = 0
    for wl_id, T, valid_list in WORKLOADS:
        base_costs, _, _ = strategy_baseline(T, valid_list)
        test_costs, _, _ = fn(T, valid_list)
        speedup = max(base_costs) / max(test_costs) if max(test_costs) > 0 else 1.0
        log_sum += math.log(speedup)
    geo_mean = math.exp(log_sum / len(WORKLOADS))
    print(f"  {name:>12s}: {geo_mean:.3f}x geo mean speedup")

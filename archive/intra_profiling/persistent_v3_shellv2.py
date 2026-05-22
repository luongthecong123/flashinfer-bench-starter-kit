"""Modal runner: persistent_v3_shellv2 — scheduling strategy comparison.

Runs the real CUTLASS shell kernel to get per-token valid counts, then
simulates 5 scheduling strategies and compares max-CTA-work (critical path).

Usage:
    modal run src/modal/persistent_v3_shellv2.py
"""
import sys, os, math, heapq
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, trace_volume, image

# ── Which workloads get the detailed 148-row table? ──────────────────────────
# None  → summary only
# int   → single workload, e.g. 17
# list  → multiple, e.g. [17, 21, 22]
# "all" → every workload
PROBE_WL = None

NUM_HEADS  = 16
NUM_SPLITS = 8
DIM_SPLIT  = 256
N_CTA      = 148
OOB_COST   = 0.1  # relative cost of an OOB task


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduling strategies (pure Python, using real valid counts from kernel)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_tasks(T, valid_list):
    """Build full T*S*H task list with local_valid per task."""
    tasks = []
    for tok in range(T):
        vc = valid_list[tok]
        for split in range(NUM_SPLITS):
            lv = max(0, min(DIM_SPLIT, vc - split * DIM_SPLIT))
            for head in range(NUM_HEADS):
                tasks.append((tok, split, head, lv))
    return tasks


def _cost(lv):
    return float(lv) if lv > 0 else OOB_COST


def _max_cta_cost(cta_costs):
    return max(cta_costs) if cta_costs else 0


def strat_baseline(T, valid_list):
    """Current: all T*128 tasks, round-robin over 148 CTAs."""
    tasks = _build_tasks(T, valid_list)
    n = min(N_CTA, len(tasks))
    costs = [0.0] * n
    for i, (_, _, _, lv) in enumerate(tasks):
        costs[i % n] += _cost(lv)
    return costs, tasks, "baseline"


def strat_skip_oob(T, valid_list):
    """Only schedule tasks with local_valid > 0, round-robin."""
    tasks = [t for t in _build_tasks(T, valid_list) if t[3] > 0]
    n = min(N_CTA, len(tasks))
    costs = [0.0] * n
    for i, (_, _, _, lv) in enumerate(tasks):
        costs[i % n] += _cost(lv)
    return costs, tasks, "skip_oob"


def strat_sorted_desc(T, valid_list):
    """Skip OOB, sort by work desc, round-robin."""
    tasks = sorted([t for t in _build_tasks(T, valid_list) if t[3] > 0],
                   key=lambda t: t[3], reverse=True)
    n = min(N_CTA, len(tasks))
    costs = [0.0] * n
    for i, (_, _, _, lv) in enumerate(tasks):
        costs[i % n] += _cost(lv)
    return costs, tasks, "sorted_desc"


def strat_snake(T, valid_list):
    """Skip OOB, sort desc, boustrophedon (snake) assignment."""
    tasks = sorted([t for t in _build_tasks(T, valid_list) if t[3] > 0],
                   key=lambda t: t[3], reverse=True)
    n = min(N_CTA, len(tasks))
    costs = [0.0] * n
    for i, (_, _, _, lv) in enumerate(tasks):
        row = i // n
        col = i % n
        if row % 2 == 1:
            col = n - 1 - col
        costs[col] += _cost(lv)
    return costs, tasks, "snake"


def strat_greedy_lpt(T, valid_list):
    """Skip OOB, greedy longest-processing-time-first bin-packing."""
    tasks = sorted([t for t in _build_tasks(T, valid_list) if t[3] > 0],
                   key=lambda t: t[3], reverse=True)
    n = min(N_CTA, len(tasks))
    heap = [(0.0, i) for i in range(n)]
    heapq.heapify(heap)
    cta_tasks = [[] for _ in range(n)]
    for tok, split, head, lv in tasks:
        c, cta = heapq.heappop(heap)
        heapq.heappush(heap, (c + _cost(lv), cta))
        cta_tasks[cta].append((tok, split, head, lv))
    costs = [sum(_cost(t[3]) for t in cta_tasks[i]) for i in range(n)]
    return costs, tasks, "greedy_lpt"


ALL_STRATEGIES = [strat_baseline, strat_skip_oob, strat_sorted_desc, strat_snake, strat_greedy_lpt]
STRAT_NAMES = ["baseline", "skip_oob", "sorted_desc", "snake", "greedy_lpt"]


# ═══════════════════════════════════════════════════════════════════════════════
# Detailed per-CTA table (only for PROBE_WL workloads, greedy_lpt strategy)
# ═══════════════════════════════════════════════════════════════════════════════

def _print_detail(wl_num, T, valid_list):
    """Print 148-row table for greedy_lpt showing per-CTA task assignments."""
    tasks = sorted([t for t in _build_tasks(T, valid_list) if t[3] > 0],
                   key=lambda t: t[3], reverse=True)
    n = min(N_CTA, len(tasks))
    heap = [(0.0, i) for i in range(n)]
    heapq.heapify(heap)
    cta_tasks = [[] for _ in range(N_CTA)]
    for tok, split, head, lv in tasks:
        c, cta = heapq.heappop(heap)
        heapq.heappush(heap, (c + _cost(lv), cta))
        cta_tasks[cta].append((tok, split, head, lv))

    max_len = max((len(ct) for ct in cta_tasks), default=0)
    CW = 16

    print(f"\n{'─'*60}")
    print(f"WL{wl_num} GREEDY_LPT detail  T={T}  tasks={len(tasks)}  CTAs_used={n}")
    print(f"{'─'*60}")
    hdr = f"{'CTA':>4} {'work':>6} {'#t':>3}"
    for wi in range(max_len):
        hdr += f"  {'w'+str(wi):>{CW}}"
    print(hdr)
    print(f"{'─'*4} {'─'*6} {'─'*3}" + f"  {'─'*CW}" * max_len)

    for cta in range(N_CTA):
        ct = cta_tasks[cta]
        vsum = sum(t[3] for t in ct)
        row = f"{cta:>4} {vsum:>6} {len(ct):>3}"
        for wi in range(max_len):
            if wi < len(ct):
                tok, split, head, lv = ct[wi]
                cell = f"t{tok}s{split}h{head}={lv}"
            else:
                cell = ""
            row += f"  {cell:>{CW}}"
        print(row)
    print()


def _fmt_valid(per_tok):
    if len(per_tok) <= 4:
        return "[" + ",".join(str(v) for v in per_tok) + "]"
    shown = ",".join(str(v) for v in per_tok[:3])
    return f"[{shown},...+{len(per_tok)-3}]"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/data": trace_volume})
def run_shell_remote():
    import sys, json
    sys.path.insert(0, "/app")

    import torch
    from pathlib import Path
    from safetensors.torch import load_file

    from src.kernels.fused_persistent_v3_shell import (
        run_shell, MAX_ACTIVE_CLUSTERS,
    )

    from src import utils
    utils.CONTEST = Path("/data")
    utils.JSONL = utils.CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl"

    workloads = [json.loads(l) for l in open(utils.JSONL)]

    print(f"GPU: {torch.cuda.get_device_name(0)}  |  SMs: {N_CTA}")
    print(f"H={NUM_HEADS}  S={NUM_SPLITS}  DIM_SPLIT={DIM_SPLIT}  OOB_COST={OOB_COST}")
    print(f"Strategies: {', '.join(STRAT_NAMES)}")
    print()

    # ── Collect data: one row per workload ────────────────────────────────
    rows = []

    for i_w, w in enumerate(workloads):
        ax  = w["workload"]["axes"]
        inp = w["workload"]["inputs"]
        T_val = ax["num_tokens"]

        sf = load_file(str(utils.CONTEST / inp["sparse_indices"]["path"]))
        si = sf[inp["sparse_indices"]["tensor_key"]].cuda()

        # Get real valid counts from kernel
        _, _, gvc = run_shell(si, T_val)
        valid_list = gvc.cpu()[:T_val].tolist()

        wl_num = i_w + 1

        # Run all strategies
        strat_results = {}
        for fn in ALL_STRATEGIES:
            costs, tasks, name = fn(T_val, valid_list)
            mx = _max_cta_cost(costs)
            n_tasks = len(tasks)
            n_ctas = min(N_CTA, n_tasks)
            active = sum(1 for c in costs if c > OOB_COST)
            strat_results[name] = dict(max=mx, tasks=n_tasks, ctas=n_ctas, active=active)

        base_max = strat_results["baseline"]["max"]
        rows.append(dict(wl=wl_num, T=T_val, valid=valid_list, strats=strat_results, base_max=base_max))

        # ── Detailed table if requested ───────────────────────────────────
        show_detail = (
            PROBE_WL == "all"
            or (isinstance(PROBE_WL, int) and PROBE_WL == wl_num)
            or (isinstance(PROBE_WL, (list, tuple)) and wl_num in PROBE_WL)
        )
        if show_detail:
            _print_detail(wl_num, T_val, valid_list)

    # ══════════════════════════════════════════════════════════════════════
    # Summary table: baseline max_cost + speedup for each strategy
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*130}")
    print(f"STRATEGY COMPARISON — max CTA cost (critical path), lower = better")
    print(f"{'═'*130}")

    # header
    hdr = f"{'WL':>3}  {'T':>2}  {'valid':<22}  {'#real':>5}"
    hdr += f"  {'baseline':>10}"
    for name in STRAT_NAMES[1:]:
        hdr += f"  {name:>14}"
    print(hdr)

    sep = f"{'─'*3}  {'─'*2}  {'─'*22}  {'─'*5}"
    sep += f"  {'─'*10}"
    for _ in STRAT_NAMES[1:]:
        sep += f"  {'─'*14}"
    print(sep)

    log_sums = {name: 0.0 for name in STRAT_NAMES[1:]}

    for r in rows:
        vstr = _fmt_valid(r["valid"])
        s = r["strats"]
        real = s["skip_oob"]["tasks"]  # = number of non-OOB tasks
        line = f"{r['wl']:>3}  {r['T']:>2}  {vstr:<22}  {real:>5}"
        line += f"  {s['baseline']['max']:>10.1f}"
        for name in STRAT_NAMES[1:]:
            mx = s[name]["max"]
            sp = r["base_max"] / mx if mx > 0 else float("inf")
            line += f"  {mx:>8.1f} {sp:>4.2f}x"
            log_sums[name] += math.log(sp)
        print(line)

    # ── Geo mean row ──────────────────────────────────────────────────────
    print(sep)
    geo_line = f"{'':>3}  {'':>2}  {'GEO MEAN SPEEDUP':<22}  {'':>5}  {'1.00x':>10}"
    for name in STRAT_NAMES[1:]:
        gm = math.exp(log_sums[name] / len(rows))
        geo_line += f"  {'':>8} {gm:>4.2f}x"
    print(geo_line)
    print(f"{'═'*130}\n")

    return "done"


@app.local_entrypoint()
def main():
    run_shell_remote.remote()

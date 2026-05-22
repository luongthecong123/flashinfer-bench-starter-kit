"""
Comprehensive analysis graphs for GEMV kernel optimization.

Generates:
  1. Linearization graph — Stage 1 (Score GEMV): runtime vs N with linear reference lines
  2. Linearization graph — Stage 3 (Output GEMV): runtime vs N with linear reference lines
  3. Speedup ratio chart — all kernels vs baseline at each N
  4. Fused kernel breakdown — stacked contributions of thr_warp vs ldgv1b optimizations
  5. Wave analysis — KV-split planning with actual workloads
  6. KV-split cost model — estimated wall-clock per dim_split

Outputs PNGs to images/
"""
import json
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
with open(os.path.join(DATA_DIR, "bench_all_10reps.json")) as f:
    standalone = json.load(f)
with open(os.path.join(DATA_DIR, "bench_fused_10reps.json")) as f:
    fused = json.load(f)

# ── Standalone kernel data (Stage 1 + Stage 3) ───────────────────────────────
N_VALUES = [32, 64, 128, 256, 512, 1024, 2048]

def get_means(kernel_name, n_values=N_VALUES):
    """Extract mean timings, returning NaN for missing entries."""
    d = standalone[kernel_name]
    return [d[str(n)]["mean"] if d.get(str(n)) else float("nan") for n in n_values]

# Stage 1 kernels
warp_means = get_means("warp")
thr_warp_means = get_means("thr_warp")
thr_warpv2_means = get_means("thr_warpv2")

# Stage 3 kernels
output_means = get_means("output")
output_ldg_means = get_means("output_ldg")
output_ldgv1b_means = get_means("output_ldgv1b")

# ── Fused kernel data ────────────────────────────────────────────────────────
# Workloads sorted by max valid count
FUSED_WL_ORDER = ["WL2(max=18)", "WL3(max=52)", "WL7(max=92)",
                   "WL5(max=337)", "WL10(max=1044)", "WL13(max=2048)"]
FUSED_N = [18, 52, 92, 337, 1044, 2048]
FUSED_LABELS = ["18", "52", "92", "337", "1044", "2048"]

def get_fused_means(kernel_name):
    d = fused[kernel_name]
    return [d[wl]["mean"] for wl in FUSED_WL_ORDER]

def get_fused_stds(kernel_name):
    d = fused[kernel_name]
    return [d[wl]["std"] for wl in FUSED_WL_ORDER]

fused_baseline = get_fused_means("tiny5v2")
fused_thr = get_fused_means("tiny_thr_warp")
fused_ldg = get_fused_means("tiny_ldgv1b")
fused_both = get_fused_means("tiny_thr_warp_ldgv1b")

fused_baseline_std = get_fused_stds("tiny5v2")
fused_thr_std = get_fused_stds("tiny_thr_warp")
fused_ldg_std = get_fused_stds("tiny_ldgv1b")
fused_both_std = get_fused_stds("tiny_thr_warp_ldgv1b")

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.size": 11,
})

COLORS = {
    "warp": "#8b949e",       # gray (baseline)
    "thr_warp": "#58a6ff",   # blue
    "thr_warpv2": "#3fb950", # green
    "output": "#8b949e",     # gray (baseline)
    "output_ldg": "#d2a8ff", # purple
    "output_ldgv1b": "#f0883e", # orange
    "ref_min": "#484f58",    # dim gray
    "ref_max": "#484f58",    # dim gray
    "tiny5v2": "#8b949e",
    "tiny_thr_warp": "#58a6ff",
    "tiny_ldgv1b": "#f0883e",
    "tiny_thr_warp_ldgv1b": "#3fb950",
}

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 1: Stage 1 Linearization
# ═══════════════════════════════════════════════════════════════════════════════
def plot_linearization_stage1():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot kernel data
    ax.plot(N_VALUES, warp_means, "o-", color=COLORS["warp"], label="warp (baseline)", linewidth=2, markersize=6)
    ax.plot(N_VALUES, thr_warp_means, "s-", color=COLORS["thr_warp"], label="thr_warp (LDG.128)", linewidth=2, markersize=6)
    # thr_warpv2 starts at N=128
    valid_n = [n for n, v in zip(N_VALUES, thr_warpv2_means) if not math.isnan(v)]
    valid_v = [v for v in thr_warpv2_means if not math.isnan(v)]
    ax.plot(valid_n, valid_v, "D-", color=COLORS["thr_warpv2"], label="thr_warpv2 (4-row interleaved)", linewidth=2, markersize=6)

    # Linear reference lines from min (N=32) and max (N=2048) for warp baseline
    n_min, n_max = 32, 2048
    v_min, v_max = warp_means[0], warp_means[-1]  # warp at N=32, N=2048
    slope_min = v_min / n_min  # µs per N, extrapolated from N=32
    slope_max = v_max / n_max  # µs per N, extrapolated from N=2048
    n_line = np.linspace(16, 2200, 100)
    ax.plot(n_line, slope_min * n_line, "--", color=COLORS["ref_min"], alpha=0.6,
            label=f"Linear ref (from N=32): {slope_min:.4f} µs/N")
    ax.plot(n_line, slope_max * n_line, color=COLORS["ref_max"], alpha=0.6,
            label=f"Linear ref (from N=2048): {slope_max:.4f} µs/N", linestyle=":")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (valid KV count, log₂ scale)")
    ax.set_ylabel("Kernel duration (µs)")
    ax.set_title("Stage 1: Score GEMV — Linearization Analysis (10-rep mean)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(N_VALUES)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(24, 2800)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "linearization_stage1_score.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 2: Stage 3 Linearization
# ═══════════════════════════════════════════════════════════════════════════════
def plot_linearization_stage3():
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(N_VALUES, output_means, "o-", color=COLORS["output"], label="output (baseline)", linewidth=2, markersize=6)
    ax.plot(N_VALUES, output_ldg_means, "s-", color=COLORS["output_ldg"], label="output_ldg (LDG.128)", linewidth=2, markersize=6)
    ax.plot(N_VALUES, output_ldgv1b_means, "D-", color=COLORS["output_ldgv1b"], label="output_ldgv1b (LDG.128 + coalesced)", linewidth=2, markersize=6)

    # Linear reference lines from min (N=32) and max (N=2048) for output_ldgv1b
    n_min, n_max = 32, 2048
    v_min, v_max = output_ldgv1b_means[0], output_ldgv1b_means[-1]
    slope_min = v_min / n_min
    slope_max = v_max / n_max
    n_line = np.linspace(16, 2200, 100)
    ax.plot(n_line, slope_min * n_line, "--", color=COLORS["ref_min"], alpha=0.6,
            label=f"Linear ref (from N=32): {slope_min:.4f} µs/N")
    ax.plot(n_line, slope_max * n_line, color=COLORS["ref_max"], alpha=0.6,
            label=f"Linear ref (from N=2048): {slope_max:.4f} µs/N", linestyle=":")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (valid KV count, log₂ scale)")
    ax.set_ylabel("Kernel duration (µs)")
    ax.set_title("Stage 3: Output GEMV — Linearization Analysis (10-rep mean)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(N_VALUES)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(24, 2800)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "linearization_stage3_output.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 3: Speedup ratio chart — all standalone kernels vs their baseline
# ═══════════════════════════════════════════════════════════════════════════════
def plot_speedup_ratios():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Stage 1 speedups
    s1_thr = [w / t if not math.isnan(t) else float("nan") for w, t in zip(warp_means, thr_warp_means)]
    s1_v2 = [w / v if not math.isnan(v) else float("nan") for w, v in zip(warp_means, thr_warpv2_means)]

    ax1.bar([n - 30 for n in N_VALUES], s1_thr, width=[n * 0.18 for n in N_VALUES],
            color=COLORS["thr_warp"], alpha=0.85, label="thr_warp / warp")
    valid_n_v2 = [(n, s) for n, s in zip(N_VALUES, s1_v2) if not math.isnan(s)]
    if valid_n_v2:
        ns, ss = zip(*valid_n_v2)
        ax1.bar([n + 30 for n in ns], ss, width=[n * 0.18 for n in ns],
                color=COLORS["thr_warpv2"], alpha=0.85, label="thr_warpv2 / warp")

    ax1.axhline(y=1.0, color="#f85149", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("N (log₂ scale)")
    ax1.set_ylabel("Speedup (×)")
    ax1.set_title("Stage 1: Score Speedup vs Baseline", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2, axis="y")
    ax1.set_xticks(N_VALUES)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())

    # Stage 3 speedups
    s3_ldg = [o / l for o, l in zip(output_means, output_ldg_means)]
    s3_v1b = [o / l for o, l in zip(output_means, output_ldgv1b_means)]

    x = np.arange(len(N_VALUES))
    w = 0.35
    ax2.bar(x - w/2, s3_ldg, w, color=COLORS["output_ldg"], alpha=0.85, label="output_ldg / output")
    ax2.bar(x + w/2, s3_v1b, w, color=COLORS["output_ldgv1b"], alpha=0.85, label="output_ldgv1b / output")
    ax2.axhline(y=1.0, color="#f85149", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_xlabel("N")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("Stage 3: Output Speedup vs Baseline", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in N_VALUES])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "speedup_ratios.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 4: Fused kernel — optimization breakdown (stacked area showing
#           contribution of thr_warp vs ldgv1b savings)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fused_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Absolute timings with error bars
    x = np.arange(len(FUSED_N))
    w = 0.2
    for i, (name, means, stds, color) in enumerate([
        ("tiny5v2 (baseline)", fused_baseline, fused_baseline_std, COLORS["tiny5v2"]),
        ("+thr_warp (score)", fused_thr, fused_thr_std, COLORS["tiny_thr_warp"]),
        ("+ldgv1b (output)", fused_ldg, fused_ldg_std, COLORS["tiny_ldgv1b"]),
        ("+both", fused_both, fused_both_std, COLORS["tiny_thr_warp_ldgv1b"]),
    ]):
        ax1.bar(x + (i - 1.5) * w, means, w, yerr=stds, capsize=2,
                color=color, alpha=0.85, label=name)

    ax1.set_xlabel("Max Valid Count")
    ax1.set_ylabel("Duration (µs)")
    ax1.set_title("Fused Kernel: Absolute Timings (10-rep)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(FUSED_LABELS)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.2, axis="y")

    # Right: Savings breakdown — how much each optimization contributes
    # thr_warp savings = baseline - thr_warp
    # ldgv1b savings = baseline - ldgv1b
    # combined savings = baseline - both
    # interaction = combined - (thr_savings + ldg_savings)  # synergy / overlap
    thr_savings = [b - t for b, t in zip(fused_baseline, fused_thr)]
    ldg_savings = [b - l for b, l in zip(fused_baseline, fused_ldg)]
    both_savings = [b - c for b, c in zip(fused_baseline, fused_both)]
    interaction = [bo - (t + l) for bo, t, l in zip(both_savings, thr_savings, ldg_savings)]

    ax2.bar(x, thr_savings, 0.5, color=COLORS["tiny_thr_warp"], alpha=0.85,
            label="Score vectorization savings")
    ax2.bar(x, [max(0, l) for l in ldg_savings], 0.5, bottom=thr_savings,
            color=COLORS["tiny_ldgv1b"], alpha=0.85,
            label="Output vectorization savings")
    # Show interaction (synergy) as additional bar
    bottom_inter = [t + max(0, l) for t, l in zip(thr_savings, ldg_savings)]
    ax2.bar(x, [max(0, i) for i in interaction], 0.5, bottom=bottom_inter,
            color="#d29922", alpha=0.85, label="Synergy (interaction)")
    # Show negative savings (regressions) as hatched bars below zero
    neg_ldg = [min(0, l) for l in ldg_savings]
    if any(v < 0 for v in neg_ldg):
        ax2.bar(x, neg_ldg, 0.5, color=COLORS["tiny_ldgv1b"], alpha=0.4,
                hatch="//", label="Output regression (small N)")

    ax2.axhline(y=0, color="#f85149", linestyle="-", alpha=0.3, linewidth=1)
    ax2.set_xlabel("Max Valid Count")
    ax2.set_ylabel("Time Saved (µs)")
    ax2.set_title("Optimization Breakdown: Where Savings Come From", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(FUSED_LABELS)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fused_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 5: Wave Analysis — KV-split planning for all 23 workloads
# ═══════════════════════════════════════════════════════════════════════════════
WORKLOAD_INFO = [
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

NUM_SM = 148
H = 16
BLOCK_SIZE = 1024
MAX_SMEM_PER_SM = 228
SMEM_PER_BLOCK = 200
MAX_BLOCK_PER_SM = MAX_SMEM_PER_SM // SMEM_PER_BLOCK  # = 1

DIM_SPLIT_SWEEP = [2048, 1024, 512, 256, 128, 64]

def compute_wave_analysis():
    """For each dim_split, compute total blocks, waves, and imbalance per workload."""
    results = {}
    for dim_split in DIM_SPLIT_SWEEP:
        wl_data = []
        for idx, (wl_id, num_tokens, work_list) in enumerate(WORKLOAD_INFO):
            blocks_per_token = [(v + dim_split - 1) // dim_split for v in work_list]
            total_blocks = sum(blocks_per_token) * H
            blocks_per_sm = min(2048 // BLOCK_SIZE, MAX_BLOCK_PER_SM)
            num_waves = total_blocks / (blocks_per_sm * NUM_SM)
            max_splits = max(blocks_per_token)
            avg_splits = sum(blocks_per_token) / len(blocks_per_token)
            imbalance = max_splits / avg_splits if avg_splits > 0 else 0
            wl_data.append({
                "wl_idx": idx + 1,
                "wl_id": wl_id,
                "T": num_tokens,
                "work_list": work_list,
                "blocks_per_token": blocks_per_token,
                "total_blocks": total_blocks,
                "num_waves": num_waves,
                "imbalance": imbalance,
                "max_valid": max(work_list),
            })
        results[dim_split] = wl_data
    return results

def plot_wave_analysis():
    wave_data = compute_wave_analysis()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Total waves per workload for each dim_split
    x = np.arange(len(WORKLOAD_INFO))
    width = 0.13
    ds_colors = ["#8b949e", "#58a6ff", "#3fb950", "#f0883e", "#d2a8ff", "#f85149"]
    for i, (ds, color) in enumerate(zip(DIM_SPLIT_SWEEP, ds_colors)):
        waves = [d["num_waves"] for d in wave_data[ds]]
        ax1.bar(x + (i - 2.5) * width, waves, width, color=color, alpha=0.8,
                label=f"DS={ds}")

    ax1.set_xlabel("Workload Index")
    ax1.set_ylabel("Number of Waves")
    ax1.set_title("Wave Count per Workload by DIM_SPLIT", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i + 1) for i in range(len(WORKLOAD_INFO))], fontsize=7)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.2, axis="y")

    # Right: Imbalance (max/avg blocks per token) vs dim_split for worst workloads
    # Show the 5 most imbalanced workloads
    worst_wls = sorted(range(len(WORKLOAD_INFO)),
                       key=lambda i: max(WORKLOAD_INFO[i][2]) / (sum(WORKLOAD_INFO[i][2]) / len(WORKLOAD_INFO[i][2])),
                       reverse=True)[:5]
    for wl_idx in worst_wls:
        imbalances = [wave_data[ds][wl_idx]["imbalance"] for ds in DIM_SPLIT_SWEEP]
        label = f"WL{wl_idx+1} ({WORKLOAD_INFO[wl_idx][0][:8]})"
        ax2.plot(DIM_SPLIT_SWEEP, imbalances, "o-", label=label, linewidth=1.5, markersize=5)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("DIM_SPLIT (log₂ scale)")
    ax2.set_ylabel("Imbalance (max/avg blocks per token)")
    ax2.set_title("Token-level Imbalance vs DIM_SPLIT", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(DIM_SPLIT_SWEEP)
    ax2.get_xaxis().set_major_formatter(ScalarFormatter())
    ax2.invert_xaxis()

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "wave_analysis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 6: KV-split Cost Model — estimated wall-clock per dim_split
# Uses standalone kernel timings to estimate per-block cost, then multiplies
# by wave count for each workload.
# ═══════════════════════════════════════════════════════════════════════════════
def interpolate_kernel_time(n, kernel_means, n_values=N_VALUES):
    """Linear interpolation of kernel time for arbitrary N."""
    if n <= 0:
        return 0.0
    # Clamp to range
    if n <= n_values[0]:
        return kernel_means[0]
    if n >= n_values[-1]:
        return kernel_means[-1]
    # Find bracketing indices
    for i in range(len(n_values) - 1):
        if n_values[i] <= n <= n_values[i + 1]:
            if math.isnan(kernel_means[i]) or math.isnan(kernel_means[i + 1]):
                return kernel_means[i + 1] if math.isnan(kernel_means[i]) else kernel_means[i]
            frac = (n - n_values[i]) / (n_values[i + 1] - n_values[i])
            return kernel_means[i] + frac * (kernel_means[i + 1] - kernel_means[i])
    return kernel_means[-1]

def estimate_wall_clock(dim_split, work_list, score_kernel, output_kernel):
    """Estimate wall-clock for a single workload with KV-split.

    Each token's valid count is split into ceil(valid/dim_split) blocks.
    Each block processes min(dim_split, remaining) KV entries.
    Blocks run in waves of NUM_SM.

    Returns: estimated total µs (waves × per-wave-cost).
    Per-wave cost = max time across all blocks in that wave (critical path).
    Simplified: all blocks in a wave take the same time ≈ kernel_time(dim_split).
    """
    blocks_per_token = [(v + dim_split - 1) // dim_split for v in work_list]
    total_blocks = sum(blocks_per_token) * H
    blocks_per_sm = min(2048 // BLOCK_SIZE, MAX_BLOCK_PER_SM)
    num_waves = math.ceil(total_blocks / (blocks_per_sm * NUM_SM))

    # Per-block N is at most dim_split (last block may be smaller)
    # Use dim_split as the representative N for cost estimation
    n_eff = min(dim_split, 2048)
    score_time = interpolate_kernel_time(n_eff, score_kernel)
    output_time = interpolate_kernel_time(n_eff, output_kernel)
    per_block_time = score_time + output_time  # simplified: no softmax overhead

    # Wall clock ≈ num_waves × per_block_time
    return num_waves * per_block_time

def plot_kvsplit_cost():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Compare 4 kernel combinations across dim_splits
    kernel_configs = [
        ("warp + output", warp_means, output_means, COLORS["warp"]),
        ("thr_warp + output_ldgv1b", thr_warp_means, output_ldgv1b_means, COLORS["thr_warp"]),
        ("thr_warpv2 + output_ldgv1b", thr_warpv2_means, output_ldgv1b_means, COLORS["thr_warpv2"]),
    ]

    # Left: Geometric mean wall-clock across all 23 workloads for each dim_split
    for config_name, score_k, output_k, color in kernel_configs:
        gm_times = []
        for ds in DIM_SPLIT_SWEEP:
            times = []
            for wl_id, num_tokens, work_list in WORKLOAD_INFO:
                t = estimate_wall_clock(ds, work_list, score_k, output_k)
                if t > 0:
                    times.append(t)
            gm = np.exp(np.mean(np.log(times))) if times else 0
            gm_times.append(gm)
        ax1.plot(DIM_SPLIT_SWEEP, gm_times, "o-", color=color, label=config_name,
                 linewidth=2, markersize=6)

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("DIM_SPLIT (log₂ scale)")
    ax1.set_ylabel("Geomean Wall-Clock (µs)")
    ax1.set_title("KV-Split Cost Model: GM Wall-Clock\n(score + output stages, ignoring reduction)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(DIM_SPLIT_SWEEP)
    ax1.get_xaxis().set_major_formatter(ScalarFormatter())
    ax1.invert_xaxis()

    # Right: Per-workload comparison at the optimal dim_split
    # Find best dim_split for each kernel config
    best_ds_per_config = {}
    for config_name, score_k, output_k, color in kernel_configs:
        best_ds = None
        best_gm = float("inf")
        for ds in DIM_SPLIT_SWEEP:
            times = []
            for wl_id, num_tokens, work_list in WORKLOAD_INFO:
                t = estimate_wall_clock(ds, work_list, score_k, output_k)
                if t > 0:
                    times.append(t)
            gm = np.exp(np.mean(np.log(times)))
            if gm < best_gm:
                best_gm = gm
                best_ds = ds
        best_ds_per_config[config_name] = (best_ds, best_gm)

    # Show per-workload times at best dim_split for each config
    x = np.arange(len(WORKLOAD_INFO))
    width = 0.25
    for i, (config_name, score_k, output_k, color) in enumerate(kernel_configs):
        best_ds = best_ds_per_config[config_name][0]
        times = [estimate_wall_clock(best_ds, wl[2], score_k, output_k)
                 for wl in WORKLOAD_INFO]
        ax2.bar(x + (i - 1) * width, times, width, color=color, alpha=0.8,
                label=f"{config_name}\n(DS={best_ds})")

    ax2.set_xlabel("Workload Index")
    ax2.set_ylabel("Estimated Wall-Clock (µs)")
    ax2.set_title("Per-Workload Cost at Optimal DIM_SPLIT", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i + 1) for i in range(len(WORKLOAD_INFO))], fontsize=7)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kvsplit_cost_model.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Print comprehensive analysis table  
# ═══════════════════════════════════════════════════════════════════════════════
def print_kvsplit_analysis():
    print("\n" + "=" * 90)
    print("KV-SPLIT WAVE ANALYSIS — All 23 Workloads × 6 DIM_SPLITs")
    print("B200: 148 SMs, 200 KB smem/block → 1 block/SM max")
    print("=" * 90)

    # Summarize: for each dim_split, compute GM waves and GM estimated time
    kernel_configs = [
        ("warp+output", warp_means, output_means),
        ("thr_warp+ldgv1b", thr_warp_means, output_ldgv1b_means),
        ("thr_warpv2+ldgv1b", thr_warpv2_means, output_ldgv1b_means),
    ]

    print(f"\n{'DIM_SPLIT':>10}  {'GM Waves':>10}  {'Max Waves':>10}  ", end="")
    for name, _, _ in kernel_configs:
        print(f"{'GM µs (' + name + ')':>28}  ", end="")
    print()
    print("-" * 110)

    best_overall = {}
    for ds in DIM_SPLIT_SWEEP:
        waves_list = []
        for wl_id, num_tokens, work_list in WORKLOAD_INFO:
            bpt = [(v + ds - 1) // ds for v in work_list]
            total = sum(bpt) * H
            nw = math.ceil(total / NUM_SM)
            waves_list.append(nw)

        gm_waves = np.exp(np.mean(np.log(waves_list)))
        max_waves = max(waves_list)

        print(f"{ds:>10}  {gm_waves:>10.2f}  {max_waves:>10}  ", end="")
        for name, score_k, output_k in kernel_configs:
            times = [estimate_wall_clock(ds, wl[2], score_k, output_k) for wl in WORKLOAD_INFO]
            gm = np.exp(np.mean(np.log([t for t in times if t > 0])))
            print(f"{gm:>28.1f}  ", end="")
            if name not in best_overall or gm < best_overall[name][1]:
                best_overall[name] = (ds, gm)
        print()

    print("\n── Best DIM_SPLIT per kernel config ──")
    for name, (ds, gm) in best_overall.items():
        print(f"  {name}: DIM_SPLIT={ds}, GM wall-clock={gm:.1f} µs")

    # Mixed strategy: per-workload, pick the best (kernel, dim_split) combo
    print("\n── Mixed Strategy: best per-workload ──")
    print(f"{'WL':>4}  {'T':>3}  {'MaxValid':>8}  {'BestConfig':>25}  {'DS':>6}  {'µs':>8}")
    print("-" * 70)
    total_times_mixed = []
    total_times_single = []
    for idx, (wl_id, num_tokens, work_list) in enumerate(WORKLOAD_INFO):
        best_t = float("inf")
        best_cfg = ""
        best_ds_wl = 0
        for ds in DIM_SPLIT_SWEEP:
            for name, score_k, output_k in kernel_configs:
                t = estimate_wall_clock(ds, work_list, score_k, output_k)
                if t < best_t:
                    best_t = t
                    best_cfg = name
                    best_ds_wl = ds
        total_times_mixed.append(best_t)
        # Best single config
        best_single_name, (best_single_ds, _) = min(best_overall.items(), key=lambda x: x[1][1])
        best_single_k = [(n, s, o) for n, s, o in kernel_configs if n == best_single_name][0]
        t_single = estimate_wall_clock(best_single_ds, work_list, best_single_k[1], best_single_k[2])
        total_times_single.append(t_single)

        print(f"{idx+1:>4}  {num_tokens:>3}  {max(work_list):>8}  {best_cfg:>25}  {best_ds_wl:>6}  {best_t:>8.1f}")

    gm_mixed = np.exp(np.mean(np.log(total_times_mixed)))
    gm_single = np.exp(np.mean(np.log(total_times_single)))
    print(f"\nGM (mixed):  {gm_mixed:.1f} µs")
    print(f"GM (single): {gm_single:.1f} µs")
    print(f"Mixed advantage: {gm_single / gm_mixed:.2f}×")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating graphs...")
    plot_linearization_stage1()
    plot_linearization_stage3()
    plot_speedup_ratios()
    plot_fused_breakdown()
    plot_wave_analysis()
    plot_kvsplit_cost()
    print_kvsplit_analysis()
    print("\nDone!")

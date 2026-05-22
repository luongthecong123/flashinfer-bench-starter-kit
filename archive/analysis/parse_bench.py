#!/usr/bin/env python3
"""Generate solution.md tables from bench_results.csv.

Usage:
    python scripts/parse_bench.py                  # read bench_results.csv → solution.md
    python scripts/parse_bench.py --from-logs      # legacy: parse logs/full_bench/*.log → csv + md
"""
import re, math, sys, argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "full_bench"
CSV_PATH = ROOT / "bench_results.csv"
MD_PATH = ROOT / "solution.md"

# ── Variant display names mapped from IMPL_MODULE kernel names ───────────────
VARIANT_MAP = {
    "kv_split_v3_thr_warpv3":                "thr_warpv3",
    "kv_split_xor":                           "xor",
    "kv_split_xor_pdl_v3":                    "xor_pdl_v3",
    "kv_split_xor_pdl_v3_SSA_v2":             "SSA_v2",
    "kv_split_xor_pdl_v3_SSA_v2_TMA_static":  "SSA_TMA_static",
    "kv_split_xor_pdl_v3_SSA_v2_TMA":         "SSA_TMA",
    "kv_split_xor_skew":                      "xor_skew",
    "kv_split_xor_pdl_v3_pro":                "xor_pdl_v3_pro",
    "kv_split_rot_pdl_v3_pro":                "rot_pdl_v3_pro",
    "kv_split_148_v3_pro":                    "148_v3_pro",
}

# Modules that are *reference* runs (not a target variant in their own right
# when they appear as the second IMPL_MODULE in a pair).  We still accept
# xor_pdl_v3 when it appears under its own === RUNNING: header.
REFERENCE_MODULES = {"kv_split_xor_pdl_v3"}

# Variants we want in the final tables, in display order
VARIANTS = [
    "thr_warpv3", "xor_pdl_v3_pro", "148_v3_pro", "rot_pdl_v3_pro", "xor", "xor_pdl_v3",
    "SSA_v2", "SSA_TMA_static", "SSA_TMA", "xor_skew",
]

ROW_PAT = re.compile(
    r"^\s+(\d+)\s+(\S+)\s+(\d+)\s+\S+\s+(PASS|FAIL)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)x",
    re.MULTILINE,
)


def _parse_data_block(text: str):
    """Extract workload rows and all_pass from a text block (ignoring module name)."""
    rows = []
    for rm in ROW_PAT.finditer(text):
        rows.append({
            "workload": int(rm.group(1)),
            "uuid":     rm.group(2),
            "T":        int(rm.group(3)),
            "status":   rm.group(4),
            "ref_ms":   float(rm.group(5)),
            "impl_ms":  float(rm.group(6)),
            "speedup":  float(rm.group(7)),
        })
    return rows, "ALL PASS" in text


def _extract_module(text: str) -> str | None:
    m = re.match(r"IMPL_MODULE:\s+src\.kernels\.(\S+)", text)
    return m.group(1) if m else None


def _parse_impl_pairs(content: str) -> list[tuple[str, list[dict], bool]]:
    """Split content into IMPL_MODULE blocks, pair them: (variant_module, rows, all_pass).

    Log structure: pairs of IMPL_MODULE blocks where:
    - 1st block = variant name (from submit.py print, no workload data)
    - 2nd block = benchmark data (workload rows + ALL PASS), IMPL_MODULE may show reference name
    The data in the 2nd block belongs to the 1st block's variant.
    """
    parts = re.split(r"(?=IMPL_MODULE:\s+)", content)
    blocks = []
    for p in parts:
        mod = _extract_module(p)
        if mod is None:
            continue
        rows, all_pass = _parse_data_block(p)
        blocks.append((mod, rows, all_pass))

    results = []
    i = 0
    while i < len(blocks):
        mod, rows, ap = blocks[i]
        if rows:
            # This block has data — standalone
            results.append((mod, rows, ap))
            i += 1
        elif i + 1 < len(blocks):
            # No data — next block has the data for this variant
            _, data_rows, data_ap = blocks[i + 1]
            if data_rows:
                results.append((mod, data_rows, data_ap))
                i += 2
            else:
                i += 1
        else:
            i += 1
    return results


def collect_variants() -> dict[str, list[dict]]:
    """Parse all log files, return {display_name: [23 row dicts]} keeping last ALL PASS."""
    result: dict[str, list[dict]] = {}
    log_files = sorted(LOG_DIR.glob("*.log"))

    for lf in log_files:
        text = lf.read_text()

        # ── Try === RUNNING: header-based splitting first ──
        running_parts = re.split(r"=== (?:RUNNING|RERUNNING)[:\s]+(\S+)\s+===", text)
        if len(running_parts) > 1:
            for i in range(1, len(running_parts), 2):
                running_name = running_parts[i]
                content = running_parts[i + 1] if i + 1 < len(running_parts) else ""
                for mod, rows, ap in _parse_impl_pairs(content):
                    vname = VARIANT_MAP.get(running_name) or VARIANT_MAP.get(mod)
                    if vname and ap and len(rows) == 23:
                        result[vname] = rows
        else:
            # ── No RUNNING headers: pair IMPL_MODULE blocks directly ──
            for mod, rows, ap in _parse_impl_pairs(text):
                vname = VARIANT_MAP.get(mod)
                if vname and ap and len(rows) == 23:
                    result[vname] = rows

    return result


def build_csv(data: dict[str, list[dict]], out_path: Path):
    """Write flat CSV: variant, workload, uuid, T, ref_ms, impl_ms, speedup."""
    rows = []
    for vname in VARIANTS:
        if vname not in data:
            continue
        for r in data[vname]:
            rows.append({
                "variant":  vname,
                "workload": r["workload"],
                "uuid":     r["uuid"],
                "T":        r["T"],
                "ref_ms":   r["ref_ms"],
                "impl_ms":  r["impl_ms"],
                "speedup":  r["speedup"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(df)} rows, {df['variant'].nunique()} variants)")
    return df


def geo_mean(s: pd.Series) -> float:
    return np.exp(np.log(s).mean())


def pivot_table(df: pd.DataFrame, value_col: str, fmt: str, wl_filter=None):
    """Pivot to workload rows × variant columns, add Geo + Arith rows."""
    sub = df if wl_filter is None else df[wl_filter(df)]
    piv = sub.pivot(index="workload", columns="variant", values=value_col)
    present = [v for v in VARIANTS if v in piv.columns]
    piv = piv[present]  # enforce column order, skip missing
    # merge T column
    t_map = sub.drop_duplicates("workload").set_index("workload")["T"]
    piv.insert(0, "T", t_map)

    # format values
    formatted = piv.copy()
    for v in present:
        formatted[v] = piv[v].map(lambda x: fmt.format(x))

    # Geo + Arith rows
    geo_row = {"T": ""}
    arith_row = {"T": ""}
    for v in present:
        vals = piv[v].dropna()
        geo_row[v] = f"**{fmt.format(geo_mean(vals))}**"
        arith_row[v] = f"**{fmt.format(vals.mean())}**"
    geo_df = pd.DataFrame([geo_row], index=["**Geo**"])
    arith_df = pd.DataFrame([arith_row], index=["**Arith**"])
    formatted = pd.concat([formatted, geo_df, arith_df])
    formatted.index.name = "#"
    return formatted


def df_to_md(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table string."""
    lines = []
    cols = df.columns.tolist()
    lines.append("| # | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---"] * len(cols)) + "|")
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_solution_md(df: pd.DataFrame, out_path: Path):
    sections = []
    sections.append("# Benchmark Results — Kernel Variants (ALL PASS)\n")
    sections.append("B200, Modal, L2 flush, arg clone, 100 reps, 3 warmup, DPS-style (output pre-allocated)\n")

    # Summary
    sections.append("## Summary\n")
    summary_rows = []
    for i, v in enumerate(VARIANTS, 1):
        vdf = df[df["variant"] == v]
        summary_rows.append({
            "#": i, "Variant": v,
            "Geo Speedup": f"{geo_mean(vdf['speedup']):.2f}x",
            "Geo Impl ms": f"{geo_mean(vdf['impl_ms']):.3f}",
            "Arith Speedup": f"{vdf['speedup'].mean():.2f}x",
        })
    sdf = pd.DataFrame(summary_rows).set_index("#")
    sections.append(df_to_md(sdf) + "\n")

    # All workloads — speedup
    sections.append("## Per-Workload Speedup Comparison\n")
    t = pivot_table(df, "speedup", "{:.2f}x")
    sections.append(df_to_md(t) + "\n")

    # All workloads — impl time
    sections.append("## Per-Workload Impl Time (ms)\n")
    t = pivot_table(df, "impl_ms", "{:.3f}")
    sections.append(df_to_md(t) + "\n")

    # T > 2 — speedup
    sections.append("## Speedup — T > 2 only (multi-split workloads)\n")
    t = pivot_table(df, "speedup", "{:.2f}x", wl_filter=lambda d: d["T"] > 2)
    sections.append(df_to_md(t) + "\n")

    # T > 2 — impl time
    sections.append("## Impl Time (ms) — T > 2 only\n")
    t = pivot_table(df, "impl_ms", "{:.3f}", wl_filter=lambda d: d["T"] > 2)
    sections.append(df_to_md(t) + "\n")

    out_path.write_text("\n".join(sections))
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate solution.md from benchmark data.")
    parser.add_argument("--from-logs", action="store_true",
                        help="Parse logs/full_bench/*.log → bench_results.csv first, then generate MD.")
    args = parser.parse_args()

    if args.from_logs:
        print("Parsing logs → CSV …")
        data = collect_variants()
        missing = [v for v in VARIANTS if v not in data]
        if missing:
            print(f"WARNING: missing variants: {missing}", file=sys.stderr)
        for v in VARIANTS:
            if v in data:
                g = geo_mean(pd.Series([r["speedup"] for r in data[v]]))
                print(f"  {v}: {g:.2f}x")
        df = build_csv(data, CSV_PATH)
    else:
        if not CSV_PATH.exists():
            print(f"ERROR: {CSV_PATH} not found. Run with --from-logs or benchmark via submit.py first.", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(CSV_PATH)
        present = [v for v in VARIANTS if v in df["variant"].unique()]
        missing = [v for v in VARIANTS if v not in df["variant"].unique()]
        if missing:
            print(f"WARNING: missing variants in CSV: {missing}", file=sys.stderr)
        print(f"Read {CSV_PATH}: {len(df)} rows, variants: {present}")

    build_solution_md(df, MD_PATH)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

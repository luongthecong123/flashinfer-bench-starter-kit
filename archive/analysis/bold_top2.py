"""
Read solution.md, bold the top-2 per workload row in comparison tables,
write result to solution_compare.md.

- Speedup tables: bold top-2 HIGHEST values per row
- Latency tables: bold top-2 LOWEST values per row
- Paired tables (Lat+Spd columns per kernel): find top-2 by speedup,
  bold BOTH Lat and Spd columns for those kernels.

Data columns start at index 2 (after # and T, or # and Ref ms).
Skip Geo/Arith summary rows (start with '|' but contain '**').
"""

import numpy as np
from pathlib import Path

SRC = Path(__file__).parent / "solution.md"
DST = Path(__file__).parent / "solution_compare.md"

# Simple tables: (section heading, higher_is_better)
TABLES = [
    ("## Per-Workload Speedup Comparison",      True),
    ("## Per-Workload Impl Time (ms)",           False),
    ("## Speedup — T > 2 only",                 True),
    ("## Impl Time (ms) — T > 2 only",          False),
]

# Paired tables: alternating Lat/Spd columns per kernel, top-2 by speedup
PAIRED_TABLES = [
    "### Per-Workload: Latency (ms) and Speedup",
]

def parse_val(s: str):
    """Strip bold markers and parse float. Speedup values end with 'x'."""
    s = s.strip().strip("*")
    s = s.rstrip("x")
    try:
        return float(s)
    except ValueError:
        return None

def bold(s: str) -> str:
    s = s.strip()
    if s.startswith("**") and s.endswith("**"):
        return s  # already bold
    return f"**{s}**"

def process_table_row(cells: list[str], higher_is_better: bool) -> list[str]:
    """
    cells[0] = row number, cells[1] = T, cells[2:] = data values.
    Bold top-2 in data columns by highest (speedup) or lowest (latency).
    """
    data_cells = cells[2:]
    vals = []
    for c in data_cells:
        v = parse_val(c)
        vals.append(v)

    indices = [i for i, v in enumerate(vals) if v is not None]
    if len(indices) < 2:
        return cells

    arr = np.array([vals[i] for i in indices], dtype=float)
    if higher_is_better:
        top2_local = np.argsort(arr)[-2:][::-1]
    else:
        top2_local = np.argsort(arr)[:2]

    top2_global = {indices[i] for i in top2_local}

    new_data = []
    for i, c in enumerate(data_cells):
        if i in top2_global and parse_val(c) is not None:
            new_data.append(bold(c.strip()))
        else:
            new_data.append(c)

    return cells[:2] + new_data


def process_paired_row(cells: list[str]) -> list[str]:
    """
    Paired Lat/Spd columns per kernel:
      cells[0] = #, cells[1] = Ref(ms), cells[2:] = k1_lat, k1_spd, k2_lat, k2_spd, ...
    Find top-2 kernels by speedup (higher=better), bold both Lat and Spd for those kernels.
    """
    data_cells = cells[2:]
    n_kernels = len(data_cells) // 2
    spd_vals = [parse_val(data_cells[2 * k + 1]) for k in range(n_kernels)]

    valid = [(k, v) for k, v in enumerate(spd_vals) if v is not None]
    if len(valid) < 2:
        return cells

    idxs, vals = zip(*valid)
    arr = np.array(vals, dtype=float)
    top2_local = np.argsort(arr)[-2:]
    top2_kernels = {idxs[i] for i in top2_local}

    new_data = []
    for k in range(n_kernels):
        lat_cell = data_cells[2 * k]
        spd_cell = data_cells[2 * k + 1]
        if k in top2_kernels:
            new_data.append(bold(lat_cell.strip()))
            new_data.append(bold(spd_cell.strip()))
        else:
            new_data.append(lat_cell)
            new_data.append(spd_cell)

    return cells[:2] + new_data

def is_data_row(line: str) -> bool:
    """True if this is a table row with actual workload data (not header/separator/summary)."""
    if not line.startswith("|"):
        return False
    if "---" in line:
        return False
    # summary rows: contain **Geo** or **Arith**
    cells = [c.strip() for c in line.split("|")[1:-1]]
    if not cells:
        return False
    first = cells[0].strip("*").strip()
    if first in ("Geo", "Arith", "", "#"):
        return False
    # header row has non-numeric first cell like "#" or "Geo T>2"
    try:
        int(first)
    except ValueError:
        return False
    return True

lines = SRC.read_text().splitlines(keepends=True)

# Build a map: heading line index -> table type
# type is ('simple', higher_is_better) or ('paired', None)
heading_map: dict[int, tuple] = {}
for heading, hib in TABLES:
    for i, line in enumerate(lines):
        if line.strip() == heading:
            heading_map[i] = ('simple', hib)
            break
for heading in PAIRED_TABLES:
    for i, line in enumerate(lines):
        if line.strip() == heading:
            heading_map[i] = ('paired', None)
            break

# Find for each line which table it belongs to (if any)
def find_table_for_line(idx: int) -> tuple | None:
    """Return (type, hib) for the active table, or None if not in a table."""
    last_heading = None
    last_info = None
    for h_idx, info in sorted(heading_map.items()):
        if h_idx <= idx:
            last_heading = h_idx
            last_info = info
        else:
            break
    if last_heading is None:
        return None
    # Check we haven't left the table (blank line or new ## section after heading)
    for j in range(last_heading + 1, idx):
        l = lines[j].strip()
        if l.startswith("##") and j != last_heading:
            return None
        # Blank line after table content ends table
        if l == "" and j > last_heading + 3 and not lines[j-1].strip().startswith("|"):
            return None
    return last_info

out_lines = []
for i, line in enumerate(lines):
    table_info = find_table_for_line(i)
    if table_info is not None and is_data_row(line):
        ttype, hib = table_info
        cells = line.split("|")
        inner = cells[1:-1]
        if ttype == 'paired':
            new_inner = process_paired_row(inner)
        else:
            new_inner = process_table_row(inner, hib)
        out_lines.append("|" + "|".join(
            f" {c.strip()} " for c in new_inner
        ) + "|\n")
    else:
        out_lines.append(line)

DST.write_text("".join(out_lines))
print(f"Written to {DST}")

# Quick sanity: count bolded cells
bolded = sum(line.count("**") for line in out_lines if is_data_row(line))
print(f"Total bold markers in data rows: {bolded} (expect ~4 per row across 4 tables)")

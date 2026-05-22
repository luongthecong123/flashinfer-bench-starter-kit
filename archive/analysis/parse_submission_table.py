#!/usr/bin/env python3
"""Parse the 'Per-Workload: Latency (ms) and Speedup' table from solution.md into sub_results.csv."""

import re, csv, sys
from pathlib import Path

HERE = Path(__file__).parent
MD = HERE / "solution.md"
OUT = HERE / "sub_results.csv"

text = MD.read_text()

# Find the per-workload table (starts after "### Per-Workload: Latency (ms) and Speedup")
m = re.search(r"### Per-Workload: Latency \(ms\) and Speedup\n+(\|.+?)(?:\n\n|\n>|\n---)", text, re.DOTALL)
if not m:
    sys.exit("ERROR: Could not find 'Per-Workload: Latency (ms) and Speedup' table")

table_text = m.group(1).strip()
lines = [l.strip() for l in table_text.split("\n") if l.strip()]

# Parse header
header_raw = [c.strip().replace("**", "") for c in lines[0].split("|")[1:-1]]  # strip outer pipes
# Skip separator line (line 1)

# Parse data rows
rows = []
for line in lines[2:]:
    cells = [c.strip().replace("**", "").replace("x", "") for c in line.split("|")[1:-1]]
    rows.append(cells)

# Build CSV header from the markdown header
# Header: #, Ref (ms), k1 Lat, k1 Spd, k2 Lat, k2 Spd, ...
csv_header = [h.strip() for h in header_raw]

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(csv_header)
    for row in rows:
        w.writerow([c.strip() for c in row])

print(f"Wrote {len(rows)} rows to {OUT}")
print(f"Columns: {csv_header}")

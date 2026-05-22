#!/usr/bin/env python3
"""
Replace k5c column with k5cl (4-run average) in sub_results.csv,
then rewrite the corresponding tables in solution.md.

Sanity checks:
- Verifies old k5c values in CSV match what's in solution.md (≥80% match)
- Recomputes Geo/Arith summary rows from scratch
"""

import csv, math, re, sys
from pathlib import Path

HERE = Path(__file__).parent
CSV_PATH = HERE / "sub_results.csv"
MD_PATH  = HERE / "solution.md"

# ── k5cl 4-run average data (workloads 1-23) ──
K5CL = [
    # (lat_ms, speedup)
    (0.013, 90.84),   # 1
    (0.013, 111.06),  # 2
    (0.013, 103.62),  # 3
    (0.014, 100.69),  # 4
    (0.024, 57.75),   # 5
    (0.013, 108.36),  # 6
    (0.015, 92.84),   # 7
    (0.022, 63.25),   # 8
    (0.013, 110.58),  # 9
    (0.029, 102.38),  # 10
    (0.028, 106.05),  # 11
    (0.035, 77.76),   # 12
    (0.033, 90.75),   # 13
    (0.029, 85.78),   # 14
    (0.037, 80.77),   # 15
    (0.018, 132.97),  # 16
    (0.042, 70.90),   # 17
    (0.032, 83.71),   # 18
    (0.036, 82.51),   # 19
    (0.029, 99.90),   # 20
    (0.044, 67.19),   # 21
    (0.041, 60.63),   # 22
    (0.038, 70.90),   # 23
]

# ── Step 1: Read CSV ──
with open(CSV_PATH) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Find k5c columns
k5c_lat_idx = header.index("k5c Lat")
k5c_spd_idx = header.index("k5c Spd")

# ── Step 2: Sanity check — old k5c values vs expected ──
OLD_K5C_SPD = [94.91, 111.49, 104.66, 101.22, 57.42, 112.38, 91.96, 63.37,
               110.09, 96.12, 99.53, 74.56, 87.30, 81.79, 76.93, 134.55,
               68.19, 81.25, 80.00, 95.86, 63.64, 58.85, 68.50]

data_rows = [r for r in rows if r[0] not in ("Geo", "Arith", "Geo T>2", "Arith T>2")]
match_count = 0
for i, row in enumerate(data_rows):
    csv_spd = float(row[k5c_spd_idx])
    if abs(csv_spd - OLD_K5C_SPD[i]) < 0.1:
        match_count += 1
    else:
        print(f"  MISMATCH row {i+1}: CSV={csv_spd}, expected={OLD_K5C_SPD[i]}")

pct = match_count / len(data_rows) * 100
print(f"Sanity check: {match_count}/{len(data_rows)} k5c speedup values match ({pct:.0f}%)")
if pct < 80:
    sys.exit(f"ABORT: Only {pct:.0f}% match — something is wrong with the CSV")

# ── Step 3: Rename header k5c → k5cl ──
header[k5c_lat_idx] = "k5cl Lat"
header[k5c_spd_idx] = "k5cl Spd"

# ── Step 4: Replace k5c data with k5cl in data rows ──
for i, row in enumerate(data_rows):
    lat, spd = K5CL[i]
    row[k5c_lat_idx] = f"{lat:.4f}"
    row[k5c_spd_idx] = f"{spd:.2f}"

# ── Step 5: Recompute summary rows ──
def geo_mean(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))

# Collect all kernel column pairs (lat_idx, spd_idx)
kernel_pairs = []
for ci, h in enumerate(header):
    if h.endswith(" Lat") and ci + 1 < len(header) and header[ci + 1].endswith(" Spd"):
        kernel_pairs.append((ci, ci + 1, h.replace(" Lat", "")))

# Remove old summary rows
rows = [r for r in rows if r[0] not in ("Geo", "Arith", "Geo T>2", "Arith T>2")]

# Build new summary rows
def make_summary_row(label, row_indices):
    row = [""] * len(header)
    row[0] = label
    for lat_i, spd_i, kname in kernel_pairs:
        lats = []
        spds = []
        for ri in row_indices:
            lv = rows[ri][lat_i].strip()
            sv = rows[ri][spd_i].strip()
            if lv and sv:
                lats.append(float(lv))
                spds.append(float(sv))
        if lats:
            row[lat_i] = f"{geo_mean(lats):.4f}"
            row[spd_i] = f"{geo_mean(spds):.2f}"
    return row

def make_arith_row(label, row_indices):
    row = [""] * len(header)
    row[0] = label
    for lat_i, spd_i, kname in kernel_pairs:
        lats = []
        spds = []
        for ri in row_indices:
            lv = rows[ri][lat_i].strip()
            sv = rows[ri][spd_i].strip()
            if lv and sv:
                lats.append(float(lv))
                spds.append(float(sv))
        if lats:
            row[lat_i] = f"{sum(lats)/len(lats):.4f}"
            row[spd_i] = f"{sum(spds)/len(spds):.2f}"
    return row

all_idx = list(range(23))
t_gt2_idx = list(range(9, 23))  # workloads 10-23

geo_row = make_summary_row("Geo", all_idx)
arith_row = make_arith_row("Arith", all_idx)
geo_t2_row = make_summary_row("Geo T>2", t_gt2_idx)
arith_t2_row = make_arith_row("Arith T>2", t_gt2_idx)

# For Geo T>2 and Arith T>2, only k5+ kernels have values — blank out k1-k4
for row in [geo_t2_row, arith_t2_row]:
    for lat_i, spd_i, kname in kernel_pairs:
        if kname in ("k1", "k2", "k3", "k4"):
            row[lat_i] = ""
            row[spd_i] = ""

rows.extend([geo_row, arith_row, geo_t2_row, arith_t2_row])

# ── Step 6: Write CSV ──
with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for row in rows:
        w.writerow(row)

print(f"\nWrote updated CSV to {CSV_PATH}")

# ── Step 7: Rebuild markdown tables in solution.md ──
md_text = MD_PATH.read_text()

# --- 7a: Update summary table row for k5c → k5cl ---
# Find and replace the k5c summary row
old_summary_line = "| k5c | k5b + FastGEMV 4-row score | **85.44x** | 0.0238 | 87.59x | 0.0264 |"
# Get k5cl summary values from our computed rows
k5cl_geo_spd = float(geo_row[k5c_spd_idx])
k5cl_geo_lat = float(geo_row[k5c_lat_idx])
k5cl_arith_spd = float(arith_row[k5c_spd_idx])
k5cl_arith_lat = float(arith_row[k5c_lat_idx])
new_summary_line = f"| k5cl | k5c class + no sentinel (4-run avg) | **{k5cl_geo_spd:.2f}x** | {k5cl_geo_lat:.4f} | {k5cl_arith_spd:.2f}x | {k5cl_arith_lat:.4f} |"

if old_summary_line not in md_text:
    print("WARNING: Could not find k5c summary row in solution.md")
else:
    md_text = md_text.replace(old_summary_line, new_summary_line)
    print(f"Updated summary row: k5c → k5cl")

# --- 7b: Update the note ---
old_note = "> **Note:** k5 values are 4-run means; k5c values are 3-run means; k5cb and kcn values are 4-run means. k1–k4 values are single-run."
new_note = "> **Note:** k5 and k5cl values are 4-run means; k5cb and kcn values are 4-run means. k1–k4 values are single-run."
if old_note in md_text:
    md_text = md_text.replace(old_note, new_note)
    print("Updated note")

# --- 7c: Rebuild per-workload table ---
def fmt_lat(v):
    """Format latency: 4 decimal places"""
    return f"{float(v):.4f}" if v.strip() else ""

def fmt_spd(v):
    """Format speedup with 'x' suffix"""
    return f"{float(v):.2f}x" if v.strip() else ""

def fmt_bold(v):
    return f"**{v}**" if v else ""

# Build the new table header
new_header = "| # | Ref (ms) | k1 Lat | k1 Spd | k2 Lat | k2 Spd | k3 Lat | k3 Spd | k4 Lat | k4 Spd | k5 Lat | k5 Spd | k5cl Lat | k5cl Spd | k5cb Lat | k5cb Spd | kcn Lat | kcn Spd |"
new_sep    = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

# Data rows (1-23)
md_data_lines = []
for row in rows[:23]:
    num = row[0]
    ref = row[1]
    cells = [f"| {num} | {ref}"]
    for lat_i, spd_i, kname in kernel_pairs:
        cells.append(f" {fmt_lat(row[lat_i])} | {fmt_spd(row[spd_i])}")
    md_data_lines.append(" |".join(cells) + " |")

# Summary rows (bold)
for summary_row in rows[23:]:
    label = summary_row[0]
    cells = [f"| **{label}** |"]
    for lat_i, spd_i, kname in kernel_pairs:
        lv = summary_row[lat_i].strip()
        sv = summary_row[spd_i].strip()
        if lv and sv:
            cells.append(f" **{fmt_lat(lv)}** | **{fmt_spd(sv)}**")
        else:
            cells.append(f" | ")
    md_data_lines.append(" |".join(cells) + " |")

new_table = "\n".join([new_header, new_sep] + md_data_lines)

# Find and replace the old table
old_table_pattern = re.compile(
    r"(\| # \| Ref \(ms\) \| k1 Lat.*?\n\|---:.*?\n(?:\|.*\n)*?\| \*\*Arith T>2\*\*.*?\|)",
    re.MULTILINE
)
m = old_table_pattern.search(md_text)
if m:
    md_text = md_text[:m.start()] + new_table + md_text[m.end():]
    print("Rebuilt per-workload table")
else:
    print("WARNING: Could not find per-workload table to replace")

# --- 7d: Update the Adjusted note ---
old_adj = 'between k5, k5c, k5cb, and kcn'
new_adj = 'between k5, k5cl, k5cb, and kcn'
md_text = md_text.replace(old_adj, new_adj)

# Also update the gain note if needed — recompute
k5_geo_t2 = float(geo_t2_row[header.index("k5 Spd")])
kcn_geo_t2 = float(geo_t2_row[header.index("kcn Spd")])
gain_pct = (kcn_geo_t2 / k5_geo_t2 - 1) * 100
old_gain_pat = re.compile(r'kcn gains \*\*\+[\d.]+%\*\* geo speedup over k5')
md_text = old_gain_pat.sub(f'kcn gains **+{gain_pct:.1f}%** geo speedup over k5', md_text)

MD_PATH.write_text(md_text)
print(f"\nUpdated {MD_PATH}")

# ── Step 8: Final sanity — print k5cl summary ──
print(f"\n── k5cl summary ──")
print(f"  Geo  Speedup: {k5cl_geo_spd:.2f}x   Lat: {k5cl_geo_lat:.4f} ms")
print(f"  Arith Speedup: {k5cl_arith_spd:.2f}x  Lat: {k5cl_arith_lat:.4f} ms")
k5cl_geo_t2  = float(geo_t2_row[k5c_spd_idx])
k5cl_arith_t2 = float(arith_t2_row[k5c_spd_idx])
print(f"  Geo T>2:  {k5cl_geo_t2:.2f}x")
print(f"  Arith T>2: {k5cl_arith_t2:.2f}x")

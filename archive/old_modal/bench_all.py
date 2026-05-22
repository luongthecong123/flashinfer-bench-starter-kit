"""bench_all.py: NCU-timed benchmark of Stage 1 + Stage 3 kernels across N shapes.

Uses class-based kernels from bench_N/ so N is passed via __init__ (no regex
patching).  Each kernel is a separate NCU invocation for process isolation.

Usage:
    modal run src/modal/bench_all.py              # single-shot (default)
    modal run src/modal/bench_all.py --reps 10    # 10 reps, shuffled, mean±std
"""
import sys, os, json, random
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, image

D = 512
SHAPES = [16, 32, 64, 128, 256, 512, 1024, 2048]

# (label, module_path, class_name, stage, min_n)
KERNELS = [
    ("warp",           "src.kernels.bench_N.bench_warp",           "BenchWarp",         1,    32),
    ("thr_warp",       "src.kernels.bench_N.bench_thr_warp",       "BenchThrWarp",      1,    32),
    ("thr_warpv2",     "src.kernels.bench_N.bench_thr_warpv2",     "BenchThrWarpV2",    1,   128),
    ("output",         "src.kernels.bench_N.bench_output",          "BenchOutput",       3,    32),
    ("output_ldg",     "src.kernels.bench_N.bench_output_ldg",      "BenchOutputLdg",    3,    32),
    ("output_ldgv1b",  "src.kernels.bench_N.bench_output_ldgv1b",   "BenchOutputLdgV1b", 3,    32),
]


def _make_target_script(module_path, class_name, stage, shapes, reps=1, seed=42):
    """Generate a Python script that instantiates the class per N and runs.

    When reps > 1, creates a shuffled schedule of (N, rep) pairs and writes the
    schedule to /tmp/_schedule_{class_name}.json so the caller can map NCU rows
    back to N values.
    """
    max_n = max(shapes)
    lines = [
        "import sys, json, random",
        "sys.path.insert(0, '/app')",
        "import torch",
        f"from {module_path} import {class_name}",
        "",
    ]
    if stage == 1:
        lines += [
            f"q = torch.randn({D}, dtype=torch.bfloat16, device='cuda')",
            f"K = torch.randn({max_n}, {D}, dtype=torch.bfloat16, device='cuda')",
            f"scores = torch.zeros({max_n}, dtype=torch.float32, device='cuda')",
            "torch.cuda.synchronize()",
            "",
        ]
    else:
        lines += [
            f"sc = torch.randn({max_n}, dtype=torch.float32, device='cuda')",
            f"V = torch.randn({max_n}, {D}, dtype=torch.float32, device='cuda')",
            f"out = torch.zeros({D}, dtype=torch.float32, device='cuda')",
            "torch.cuda.synchronize()",
            "",
        ]

    # Compile all N variants first
    lines.append("# Pre-compile all variants")
    lines.append("kernels = {}")
    for n in shapes:
        lines.append(f"kernels[{n}] = {class_name}(N={n}, D={D})")
        lines.append(f"kernels[{n}].compile()")
    lines.append("")

    if reps == 1:
        # Simple single-shot: run each N once in order
        lines.append("schedule = " + repr(shapes))
        for n in shapes:
            if stage == 1:
                lines.append(f"kernels[{n}].run(q[:], K[:{n}], scores[:{n}])")
            else:
                lines.append(f"kernels[{n}].run(sc[:{n}], V[:{n}], out)")
            lines.append("torch.cuda.synchronize()")
    else:
        # Build shuffled schedule: reps repetitions of each N, shuffled
        schedule = []
        for n in shapes:
            schedule.extend([n] * reps)
        rng = random.Random(seed)
        rng.shuffle(schedule)
        lines.append(f"schedule = {schedule}")
        lines.append(f"with open('/tmp/_schedule_{class_name}.json', 'w') as f:")
        lines.append(f"    json.dump(schedule, f)")
        lines.append("")
        lines.append("for n in schedule:")
        if stage == 1:
            lines.append("    kernels[n].run(q[:], K[:n], scores[:n])")
        else:
            lines.append("    kernels[n].run(sc[:n], V[:n], out)")
        lines.append("    torch.cuda.synchronize()")

    lines.append("")
    return "\n".join(lines), schedule if reps > 1 else shapes


def _parse_ncu_csv(stdout):
    """Parse NCU CSV output, stripping ==PROF== lines, returning CuTe rows."""
    import csv, io
    csv_lines = [l for l in stdout.splitlines() if not l.startswith("==")]
    csv_text = "\n".join(csv_lines)
    reader = csv.DictReader(io.StringIO(csv_text))
    all_rows = [row for row in reader if (row.get("Metric Name") or "").strip()]
    rows = [row for row in all_rows
            if "cutlass" in (row.get("Kernel Name") or "").lower()
            or "nvjet" in (row.get("Kernel Name") or "").lower()]
    return rows, all_rows


def _row_to_us(row):
    """Extract duration in µs from an NCU CSV row."""
    val = float(row["Metric Value"].strip().replace(",", ""))
    unit = (row.get("Metric Unit") or "").strip()
    if "nsecond" in unit or unit == "ns":
        val /= 1000.0
    return val


@app.function(image=image, gpu="B200:1", timeout=3600)
def benchmark(reps: int = 1):
    import subprocess, glob, math

    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    results = {}

    # Shuffle kernel order to avoid first-kernel-always-first bias
    kernel_order = list(range(len(KERNELS)))
    if reps > 1:
        random.seed(42)
        random.shuffle(kernel_order)
        print(f"Kernel execution order (shuffled): {[KERNELS[i][0] for i in kernel_order]}")
    else:
        print(f"Kernel execution order: {[KERNELS[i][0] for i in kernel_order]}")

    for ki in kernel_order:
        label, mod_path, cls_name, stage, min_n = KERNELS[ki]
        valid = [n for n in SHAPES if n >= min_n]

        script, schedule = _make_target_script(mod_path, cls_name, stage, valid, reps=reps)
        target = f"/tmp/_bench_target_{label}.py"
        with open(target, "w") as f:
            f.write(script)

        expected = len(schedule)
        print(f"\n{'='*60}")
        print(f"NCU: {label} (stage {stage}), N = {valid}, reps={reps}, launches={expected}")
        print(f"{'='*60}")

        cmd = [ncu,
               "--kernel-name", "regex:.*",
               "--metrics", "gpu__time_duration.sum",
               "--csv",
               "--target-processes", "all",
               "python", target]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if r.returncode != 0:
            print(f"NCU exit code {r.returncode}")
        if r.stderr:
            print(f"STDERR (tail): {r.stderr[-300:]}")

        rows, all_rows = _parse_ncu_csv(r.stdout)
        print(f"Captured {len(rows)} CuTe launches out of {len(all_rows)} total (expected {expected})")

        if len(rows) != expected:
            if not rows and all_rows:
                unique = set((row.get("Kernel Name") or "")[:80] for row in all_rows[:20])
                print("Kernel names seen:")
                for name in sorted(unique):
                    print(f"  {name}")

        # Map rows back to N values via schedule
        timings_per_n = {str(n): [] for n in valid}
        for i, n_val in enumerate(schedule):
            if i < len(rows):
                try:
                    timings_per_n[str(n_val)].append(_row_to_us(rows[i]))
                except (ValueError, KeyError):
                    pass

        if reps == 1:
            timings = {}
            for n in valid:
                vals = timings_per_n[str(n)]
                timings[str(n)] = round(vals[0], 2) if vals else None
            for n in SHAPES:
                if str(n) not in timings:
                    timings[str(n)] = None
            results[label] = timings
        else:
            stats = {}
            for n in valid:
                vals = timings_per_n[str(n)]
                if vals:
                    mean = sum(vals) / len(vals)
                    if len(vals) > 1:
                        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                        std = math.sqrt(var)
                    else:
                        std = 0.0
                    stats[str(n)] = {"mean": round(mean, 2), "std": round(std, 2),
                                     "count": len(vals)}
                else:
                    stats[str(n)] = None
            for n in SHAPES:
                if str(n) not in stats:
                    stats[str(n)] = None
            results[label] = stats

        # Print summary
        for n in SHAPES:
            entry = (results[label] or {}).get(str(n))
            if entry is None:
                print(f"  N={n:5d}:        N/A")
            elif isinstance(entry, dict):
                print(f"  N={n:5d}: {entry['mean']:>8.2f} ± {entry['std']:>5.2f} µs  (n={entry['count']})")
            else:
                print(f"  N={n:5d}: {entry:>10.2f} µs")

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main(reps: int = 1):
    raw = benchmark.remote(reps=reps)

    if reps == 1:
        out_path = "reports/bench_all.json"
    else:
        out_path = f"reports/bench_all_{reps}reps.json"

    with open(out_path, "w") as f:
        f.write(raw)
    print(f"\nSaved results to {out_path}")
    print(raw)

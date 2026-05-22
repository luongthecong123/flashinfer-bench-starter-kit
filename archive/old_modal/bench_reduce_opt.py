"""Modal runner: benchmark the progressive reduce kernel optimisations on B200.

Runs reduce_v0 through reduce_v3 from src/kernels/reduce_opt.py on the B200
and reports median latency and correctness for each version.

Usage:
    modal run src/modal/bench_reduce_opt.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
from src.modal.modal_utils import app, image


@app.function(image=image, gpu="B200:1", timeout=1800)
def run_reduce_opt():
    """Generator: streams reduce_opt correctness + benchmark results line by line."""
    import sys
    sys.path.insert(0, "/app")

    yield "Compiling reduce v0…v3 kernels…\n"
    from src.kernels.reduce_opt import compile_all, check_correctness, benchmark
    compiled = compile_all()
    yield "Compilation done.\n"

    yield "\n── Correctness ──────────────────────────────────────────────────\n"
    passed = check_correctness(compiled)
    yield f"Correctness: {'ALL PASS ✓' if passed else 'SOME FAILED ✗'}\n"

    yield "\n── Benchmark (reduce only, T=8 H=16 S=8 D=512) ─────────────────\n"
    results = benchmark(compiled, warmup=100, reps=500)

    yield "\n── Summary ──────────────────────────────────────────────────────\n"
    base = results.get("v0", 1.0)
    for name, t in results.items():
        speedup = base / t
        yield f"  {name}  {t:.2f} µs  ({speedup:.2f}×  vs v0)\n"


@app.local_entrypoint()
def main():
    for line in run_reduce_opt.remote_gen():
        print(line, end="", flush=True)

"""kv_split_v3_thr_warpv3 intra-profiling with DIM_SPLIT=512 (NUM_SPLITS=4).

Patches the module-level constants before JIT compilation so the same kernel
code runs with 4 splits of 512 tokens each instead of 8×256.
"""
import src.kernels.kv_split_v3_thr_warpv3_intra as _mod

# Override constants before compile_kernel() is called (CuTe JIT reads these
# from the module's __globals__ during tracing, not at decoration time).
_mod.DIM_SPLIT  = 512
_mod.NUM_SPLITS = 4   # TOP_K // DIM_SPLIT = 2048 // 512


def run_single(workload_idx: int) -> str:
    return _mod.run_single(workload_idx)

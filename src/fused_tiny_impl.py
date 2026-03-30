"""DSA sparse attention using CuTe DSL fused_tiny2 kernel (32-warp parallel-keys)."""
import torch
from src.kernels.fused_tiny2 import run


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import utils
    from src.ref import run as ref_run
    utils.ref_fn = ref_run
    utils.impl_fn = run
    utils.CHECK = True
    utils.MEASURE = False
    utils.TOY_CHECK = False
    utils.main()

"""DSA sparse attention using fused_tiny5v4 kernel (smem-staged CKV output GEMV)."""
import torch
from archive.fused_tiny5v7 import run

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

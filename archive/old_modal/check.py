"""Local correctness check against the reference implementation.
Change IMPL_MODULE to select which implementation to test.
Usage: python src/modal/check.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Pick implementation ──
IMPL_MODULE = "src.gather_dsa_impl"

from importlib import import_module
impl = import_module(IMPL_MODULE)

from src import utils
from src.ref import run as ref_run
utils.ref_fn  = ref_run
utils.impl_fn = impl.run
utils.CHECK   = True
utils.MEASURE = False
utils.TOY_CHECK = False
utils.main()

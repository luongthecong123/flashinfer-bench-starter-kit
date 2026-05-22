#!/usr/bin/env python3
"""Standalone test: draw a single 3D tensor box to debug parallelogram rendering."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from excalib import Diagram, FILL_IN, C_Q, BG_Q

d = Diagram()

# Draw one 3D tensor at a comfortable position
d.labeled_rect_3d(200, 200, 400, 200, 20, "test_tensor", C_Q, BG_Q,
                  dim_top="512", dim_side="16", dim_depth="T=5",
                  shape="[5, 16, 512] bf16", fill=FILL_IN)

out = os.path.join(os.path.dirname(__file__), "test_3d.excalidraw")
d.write(out)

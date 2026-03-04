#!/bin/bash
# Run dsa_dequant_ref.py (topk indexer track) on Modal

# Temporarily update config.toml
cat > config.toml << 'EOF'
[solution]
name = "cong-dequant-v1"          # Solution name
definition = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"  # Track definition
author = "Cong"                   # Team/author name

[build]
language = "triton"               # triton | cuda
entry_point = "dsa_dequant_ref.py::run"   # Kernel function name
EOF

python scripts/pack_solution.py
MAX_WORKLOADS="${N:-0}" modal run scripts/run_modal.py

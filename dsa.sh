#!/bin/bash
# Run dsa_ref.py (sparse attention track) on Modal

# Temporarily update config.toml
cat > config.toml << 'EOF'
[solution]
name = "cong-dsa-ref-v1"          # Solution name
definition = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"  # Track definition
author = "Cong"                   # Team/author name

[build]
language = "triton"               # triton | cuda
entry_point = "dsa_ref.py::run"   # Kernel function name
EOF

python scripts/pack_solution.py
MAX_WORKLOADS="${N:-0}" modal run scripts/run_modal.py

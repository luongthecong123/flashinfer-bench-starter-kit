#!/usr/bin/env bash
set -e
# IMPL_MODULE=src.toco_impl modal run submit.py
# IMPL_MODULE=src.gather_impl modal run submit.py
# IMPL_MODULE=src.gather_dsa_impl modal run submit.py
# IMPL_MODULE=src.fused_tiny_impl modal run submit.py
IMPL_MODULE=src.mixed_impl modal run src/modal/submit.py

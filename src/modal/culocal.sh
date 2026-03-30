#!/usr/bin/env bash
set -e
python src/toco_impl.py
python src/gather_impl.py
python src/gather_dsa_impl.py
python src/fused_tiny_impl.py

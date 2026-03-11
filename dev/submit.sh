#!/bin/bash
# Run cook.py on Modal B200
set -e
cd "$(dirname "$0")"
modal run modal_cook.py

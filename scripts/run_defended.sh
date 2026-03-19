#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python3 -m agent_security.runner \
  --mode defended \
  --subset small \
  --attack on \
  --max_tasks 20 \
  --seed 42 \
  --out_dir results \
  --model_backend heuristic

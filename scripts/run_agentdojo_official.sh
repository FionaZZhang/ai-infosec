#!/usr/bin/env bash
set -euo pipefail

# Run official AgentDojo benchmark (repo/paper-aligned pipeline)
# Requires:
#   1) Official repo cloned locally (default: ../tmp_agentdojo_official)
#   2) Python 3.10 environment via uv
#   3) Provider credentials configured for chosen model
#
# Example:
#   MODEL=gemini-2.0-flash-001 ATTACK=important_instructions DEFENSE=none \
#   bash scripts/run_agentdojo_official.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AGENTDOJO_DIR="${AGENTDOJO_DIR:-$ROOT_DIR/../tmp_agentdojo_official}"

MODEL="${MODEL:-gemini-2.0-flash-001}"
BENCHMARK_VERSION="${BENCHMARK_VERSION:-v1.2.2}"
ATTACK="${ATTACK:-important_instructions}"   # set to 'none' for clean-only run
DEFENSE="${DEFENSE:-none}"                    # use 'none' for baseline
LOGDIR="${LOGDIR:-$AGENTDOJO_DIR/runs}"
FORCE_RERUN="${FORCE_RERUN:-0}"

if [[ ! -d "$AGENTDOJO_DIR" ]]; then
  echo "[ERROR] AGENTDOJO_DIR not found: $AGENTDOJO_DIR"
  exit 1
fi

cd "$AGENTDOJO_DIR"

echo "[INFO] AgentDojo dir: $AGENTDOJO_DIR"
echo "[INFO] Model: $MODEL"
echo "[INFO] Benchmark version: $BENCHMARK_VERSION"
echo "[INFO] Attack: $ATTACK"
echo "[INFO] Defense: $DEFENSE"
echo "[INFO] Logdir: $LOGDIR"

CMD=(uv run --python 3.10 -m agentdojo.scripts.benchmark
  --benchmark-version "$BENCHMARK_VERSION"
  --model "$MODEL"
  --logdir "$LOGDIR")

if [[ "$ATTACK" != "none" ]]; then
  CMD+=(--attack "$ATTACK")
fi

if [[ "$DEFENSE" != "none" ]]; then
  CMD+=(--defense "$DEFENSE")
fi

if [[ "$FORCE_RERUN" == "1" ]]; then
  CMD+=(--force-rerun)
fi

printf '[INFO] Running: %q ' "${CMD[@]}"
echo
"${CMD[@]}"

echo "[INFO] Done. Logs under: $LOGDIR"

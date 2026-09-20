#!/usr/bin/env bash
# Re-render every figure and every RESEARCH.md table from the committed results JSONs (no simulation is re-run).
set -euo pipefail
cd "$(dirname "$0")/../brain"
for exp in sugar_mn9 benchmark looming_gf naive_play; do
  if [ -f "experiments/results/${exp}.json" ] && [ -f "experiments/${exp}.py" ]; then
    uv run --no-sync python -m "experiments.${exp}" --plot-only
  fi
done
uv run --no-sync python -m experiments.report

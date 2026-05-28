#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-runs/scheme_d_paper_suite/thermal_lstm_spawnhist_cover_v9_seed0/checkpoints/scheme_d_paper_base_latest_full.pt}"
OUTPUT_DIR="${2:-runs/matrix/thermal_lstm_spawnhist_cover_v9_quick}"

.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies horizon2,horizon3,dynamic_weighted,planner_ensemble,ppo \
  --stages multi_realistic,multi_hard \
  --seeds 100,101,102 \
  --episodes 5 \
  --steps 800 \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR"

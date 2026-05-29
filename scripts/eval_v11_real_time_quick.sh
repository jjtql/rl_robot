#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-runs/scheme_d_paper_suite/thermal_lstm_spawnhist_latency_v11_seed1/checkpoints/scheme_d_paper_base_latest_full.pt}"
OUTPUT_DIR="${2:-runs/matrix/thermal_lstm_spawnhist_latency_v11_real_time_quick}"
DEVICE="${3:-cpu}"
DECISION_DT_SECONDS="${DECISION_DT_SECONDS:-0.05}"

.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies horizon2,dynamic_weighted,planner_ensemble,ppo \
  --stages multi_realistic,multi_hard,multi_extreme \
  --seeds 100,101,102 \
  --episodes 3 \
  --steps 3200 \
  --demo-mode \
  --decision-dt-seconds "$DECISION_DT_SECONDS" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR"

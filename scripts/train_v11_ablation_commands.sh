#!/usr/bin/env bash
set -euo pipefail

METHODS="${METHODS:-thermal_lstm_spawnhist_latency_v11,thermal_lstm_spawnhist_latency_v11_no_pred,thermal_lstm_spawnhist_latency_v11_no_carry,thermal_lstm_spawnhist_latency_v11_no_latency_reward,thermal_lstm_spawnhist_latency_v11_no_attention,thermal_lstm_spawnhist_latency_v11_no_residual}"
SEEDS="${SEEDS:-0,1,2}"
RUN_DIR="${RUN_DIR:-runs/v11_ablation_suite}"
OUTPUT="${OUTPUT:-runs/v11_ablation_suite/commands.txt}"

.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_training_suite \
  --methods "$METHODS" \
  --seeds "$SEEDS" \
  --run-dir "$RUN_DIR" \
  --output "$OUTPUT" \
  "$@"

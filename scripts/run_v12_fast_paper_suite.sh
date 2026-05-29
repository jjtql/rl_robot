#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHODS="${METHODS:-thermal_lstm_spawnhist_latency_v12_fast,thermal_lstm_spawnhist_latency_v12_fast_no_pred,thermal_lstm_spawnhist_latency_v12_fast_no_carry,thermal_lstm_spawnhist_latency_v12_fast_no_latency_reward,thermal_lstm_spawnhist_latency_v12_fast_no_attention,thermal_lstm_spawnhist_latency_v12_fast_no_residual}"
export SUITE_LABEL="${SUITE_LABEL:-V12 fast-response paper suite}"
export SUITE_NAME="${SUITE_NAME:-v12_fast_paper_suite_$(date +%Y%m%d_%H%M%S)}"
export DECISION_DT_SECONDS="${DECISION_DT_SECONDS:-0.05}"
export CHECKPOINT_SWEEP_METHODS="${CHECKPOINT_SWEEP_METHODS:-thermal_lstm_spawnhist_latency_v12_fast}"

exec "$SCRIPT_DIR/run_v11_full_paper_suite.sh" "$@"

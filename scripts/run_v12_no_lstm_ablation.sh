#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/v12_small_paper_suite}"
METHOD="thermal_lstm_spawnhist_latency_v12_fast_no_lstm_residual"
SEEDS="${SEEDS:-0,1,2}"
EVAL_SEEDS="${EVAL_SEEDS:-100,101,102}"
EVAL_STAGES="${EVAL_STAGES:-multi_low,multi_realistic,multi_hard,multi_extreme}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
EVAL_STEPS="${EVAL_STEPS:-3200}"
DEVICE="${DEVICE:-cuda}"
JOBS="${JOBS:-1}"

"$PYTHON" -m coverage_schemes.scheme_d_paper_base.run_training_suite \
  --methods "$METHOD" \
  --seeds "$SEEDS" \
  --run-dir "$RUN_ROOT/train/main_ablation" \
  --output "$RUN_ROOT/train/main_ablation/${METHOD}_commands.txt" \
  --execute \
  --jobs "$JOBS" \
  --device "$DEVICE"

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
for seed in "${SEED_ARRAY[@]}"; do
  checkpoint="$RUN_ROOT/train/main_ablation/${METHOD}_seed${seed}/checkpoints/scheme_d_paper_base_latest_full.pt"
  "$PYTHON" -m coverage_schemes.scheme_d_paper_base.run_matrix \
    --policies ppo \
    --stages "$EVAL_STAGES" \
    --seeds "$EVAL_SEEDS" \
    --episodes "$EVAL_EPISODES" \
    --steps "$EVAL_STEPS" \
    --model "$checkpoint" \
    --device "$DEVICE" \
    --decision-dt-seconds 0.05 \
    --output-dir "$RUN_ROOT/eval/main_ablation/${METHOD}_seed${seed}" \
    --demo-mode
done

"$PYTHON" -m coverage_schemes.scheme_d_paper_base.run_v12_supplemental_suite \
  --run-root "$RUN_ROOT" \
  --skip-train \
  --skip-eval \
  --skip-sweep \
  --skip-robustness \
  --skip-visuals \
  --skip-runtime \
  --device "$DEVICE"

"$PYTHON" -m coverage_schemes.scheme_d_paper_base.plot_v12_paper_figures \
  --run-root "$RUN_ROOT"

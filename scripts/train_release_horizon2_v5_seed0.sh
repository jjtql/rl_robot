#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m coverage_schemes.scheme_d_paper_base.train \
  --run-dir runs/scheme_d_paper_suite \
  --run-name thermal_lstm_spawnhist_release_v5_seed0 \
  --seed 0 \
  --headless \
  --target-selector risk_aware \
  --bc-policy horizon2 \
  --steam-attention \
  --spawn-history-observation \
  --residual-policy \
  --residual-base-policy horizon2 \
  --residual-beta 0.35 \
  --residual-beta-start 0.08 \
  --residual-beta-end 0.35 \
  --residual-beta-warmup-steps 450000 \
  --stage-episodes multi_low:150,multi_realistic:250,multi_hard:700,multi_extreme:150 \
  --update-episodes 4 \
  --no-action-smoothing-penalty \
  --thermal-hotspot-strength 2.2 \
  --thermal-background-weight 0.25 \
  --ppo-lr 1e-5 \
  --ppo-epochs 1 \
  --ppo-clip 0.05 \
  --ppo-value-clip 0.05 \
  --ppo-entropy-start 0.0007 \
  --ppo-entropy-end 0.00015 \
  --ppo-entropy-decay-steps 900000 \
  --bc-supervised-coef 0.14 \
  --bc-supervised-min-coef 0.0 \
  --bc-supervised-decay-steps 360000 \
  --bc-episodes 180 \
  --bc-epochs 12 \
  --bc-stage-episodes multi_low:30,multi_realistic:50,multi_hard:80,multi_extreme:20

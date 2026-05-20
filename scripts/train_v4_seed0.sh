#!/usr/bin/env bash
set -euo pipefail

python -m coverage_schemes.scheme_d_paper_base.train \
  --run-dir runs/scheme_d_paper_suite \
  --run-name planner_residual_attention_v4_seed0 \
  --seed 0 \
  --headless \
  --target-selector risk_aware \
  --bc-policy horizon2 \
  --steam-attention \
  --residual-policy \
  --residual-base-policy horizon2 \
  --residual-beta 0.20 \
  --stage-episodes multi_low:350,multi_realistic:350,multi_hard:400 \
  --no-action-smoothing-penalty \
  --ppo-lr 8e-6 \
  --ppo-epochs 1 \
  --ppo-clip 0.04 \
  --ppo-value-clip 0.04 \
  --ppo-entropy-start 0.0004 \
  --ppo-entropy-end 0.00008 \
  --ppo-entropy-decay-steps 1200000 \
  --bc-supervised-coef 0.18 \
  --bc-supervised-min-coef 0.12 \
  --bc-supervised-decay-steps 1200000 \
  --bc-episodes 180 \
  --bc-epochs 12 \
  --bc-stage-episodes multi_low:50,multi_realistic:60,multi_hard:70

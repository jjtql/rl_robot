#!/usr/bin/env bash
set -euo pipefail

python -m coverage_schemes.scheme_d_paper_base.train \
  --run-dir runs/smoke_train \
  --run-name github_smoke \
  --seed 0 \
  --headless \
  --episode-steps 20 \
  --update-episodes 1 \
  --save-interval 1 \
  --target-selector risk_aware \
  --bc-policy horizon2 \
  --steam-attention \
  --residual-policy \
  --residual-base-policy horizon2 \
  --residual-beta 0.20 \
  --stage-episodes multi_hard:1 \
  --bc-stage-episodes multi_hard:1 \
  --bc-episodes 1 \
  --bc-epochs 1 \
  --no-action-smoothing-penalty

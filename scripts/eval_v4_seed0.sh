#!/usr/bin/env bash
set -euo pipefail

python -m coverage_schemes.scheme_d_paper_base.run_matrix \
  --policies nearest,oldest,distance_age,risk_aware,dynamic_weighted,horizon2,horizon3,aco_tsp,ppo \
  --model runs/scheme_d_paper_suite/planner_residual_attention_v4_seed0/checkpoints/scheme_d_paper_base_latest_full.pt \
  --stages multi_low,multi_realistic,multi_hard \
  --seeds 100,101,102 \
  --episodes 10 \
  --steps 800 \
  --output-dir runs/matrix/planner_residual_attention_v4_seed0_eval

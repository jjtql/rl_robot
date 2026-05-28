#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_training_suite \
  --methods thermal_lstm_spawnhist_latency_v11 \
  --seeds 0,1 \
  --output runs/scheme_d_paper_suite/memory_v11_latency_commands.txt \
  --execute \
  --jobs 2

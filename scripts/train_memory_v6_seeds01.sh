#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m coverage_schemes.scheme_d_paper_base.run_training_suite \
  --methods thermal_lstm_spawnhist_memory_v6 \
  --seeds 0,1 \
  --output runs/scheme_d_paper_suite/memory_v6_commands.txt \
  --execute \
  --jobs 2

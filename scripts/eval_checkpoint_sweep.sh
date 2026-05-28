#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-runs/scheme_d_paper_suite/thermal_lstm_spawnhist_cover_v9_seed0}"
shift || true

.venv/bin/python -m coverage_schemes.scheme_d_paper_base.checkpoint_sweep \
  --run-dir "$RUN_DIR" \
  "$@"

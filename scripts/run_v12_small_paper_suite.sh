#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
exec "$PYTHON" -m coverage_schemes.scheme_d_paper_base.run_v12_supplemental_suite "$@"

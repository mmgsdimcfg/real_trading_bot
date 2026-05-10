#!/usr/bin/env bash
# R76 simulation launcher.
#
# Purpose:
# - Convenience wrapper for running r76 date simulation with environment checks.
#
# Usage examples:
# - ./r004_run_simulation.sh 20260508
# - ./r004_run_simulation.sh 20260508 --codes 003490 018880
#
# Update log format (append only):
# - [YYYY-MM-DD] type=feat|fix|refactor|docs owner=<name>
#   summary: <one line>
#   impact: <sim/live/common>
#   compatibility: <backward-compatible|breaking>
#
# Update log:
# - [2026-05-10] type=docs owner=copilot
#   summary: added standardized file header and switched launcher target to r76 simulator.
#   impact: sim
#   compatibility: backward-compatible

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 YYYYMMDD [extra-args]"
  echo "Example: $0 20260422 --codes 003490 018880"
  exit 1
fi

DATE_ARG="$1"
shift

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found"
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import pandas" >/dev/null 2>&1; then
  echo "ERROR: pandas is not installed in the current environment"
  echo "Run: ${PYTHON_BIN} -m pip install -r ${REPO_ROOT}/requirements.txt"
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/r007_trade_simulate_by_date.py" --date "${DATE_ARG}" "$@"

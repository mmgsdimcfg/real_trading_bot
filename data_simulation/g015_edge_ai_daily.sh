#!/usr/bin/env bash
# Edge AI daily learning loop (run after g001 has finished collecting the day, e.g. 21:30 on weekdays):
#   1) g013: add the new day(s) to the dataset + label yesterday's/today's live shadow signals
#   2) g014: walk-forward re-validation, threshold re-selection, promotion check, publish current_* model
# r003 (r011) hot-reloads runtime/edge_ai/current_meta.json within 60 s - no bot restart needed.
#
# crontab example:  30 21 * * 1-5  /home/xgraph/git/auto_trading_src/data_simulation/g015_edge_ai_daily.sh
#
# Update log:
# - [2026-10-26] type=feat owner=claude
#     summary: 신규 - Edge AI 일일 재학습(g013 증분 + g014 검증/배포) 실행 스크립트.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${EDGE_AI_PYTHON:-$HOME/venv/bin/python}"
RUNTIME="${AUTO_TRADING_RUNTIME:-$(cd "$HERE/../.." && pwd)/runtime}"
LOG_DIR="$RUNTIME/edge_ai/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/daily_$(date +%Y%m%d).log"
exec 9>"$LOG_DIR/.daily.lock"
flock -n 9 || { echo "another g015 run is active" >>"$LOG"; exit 0; }
{
  echo "=== $(date '+%F %T') g013 dataset ==="
  nice -n 10 "$PY" "$HERE/g013_edge_ai_dataset.py" --workers "${EDGE_AI_WORKERS:-2}" --live
  echo "=== $(date '+%F %T') g014 train ==="
  nice -n 10 "$PY" "$HERE/g014_edge_ai_train.py" --threads "${EDGE_AI_WORKERS:-2}"
  echo "=== $(date '+%F %T') done ==="
} >>"$LOG" 2>&1

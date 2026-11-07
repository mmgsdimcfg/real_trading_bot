#!/usr/bin/env bash
# Edge AI daily learning loop (run after g001 has finished collecting the day, e.g. 21:30 on weekdays):
#   1) g013: add the new day(s) to the dataset + label yesterday's/today's live shadow signals
#   2) g014: walk-forward re-validation, threshold re-selection, promotion check, publish current_* model
# r003 (r011) hot-reloads runtime/edge_ai/current_meta.json within 60 s - no bot restart needed.
#
# crontab example:  30 21 * * 1-5  EDGE_AI_WORKERS=1 /data/srv/autobot/real_trading_bot/autobot/data_simulation/g015_edge_ai_daily.sh
# (EDGE_AI_WORKERS=1 on the 4 GB Pi - earlyoom kills Pool workers at 2.) g013 skips a day until g001 wrote its
# done marker, so running before g001 finishes is safe (the day is picked up by the next run).
# Failures (non-zero exit of g013/g014, no published model) are mailed by g024 --edge-ai.
#
# Update log:
# - [2026-11-02] type=fix owner=claude
#     summary: g013/g014 종료 코드를 기록하고 끝에 g024 --edge-ai로 실패 메일 알림(기존: set -e로 조용히 중단).
#       g013이 실패해도 g014는 기존 데이터로 진행.
# - [2026-10-26] type=feat owner=claude
#     summary: 신규 - Edge AI 일일 재학습(g013 증분 + g014 검증/배포) 실행 스크립트.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${EDGE_AI_PYTHON:-$HOME/venv/bin/python}"
RUNTIME="${AUTO_TRADING_RUNTIME:-$(cd "$HERE/../../.." && pwd)/runtime}"
LOG_DIR="$RUNTIME/edge_ai/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/daily_$(date +%Y%m%d).log"
exec 9>"$LOG_DIR/.daily.lock"
flock -n 9 || { echo "another g015 run is active" >>"$LOG"; exit 0; }
rc_dataset=0
rc_train=0
{
  echo "=== $(date '+%F %T') g013 dataset ==="
  nice -n 10 "$PY" "$HERE/g013_edge_ai_dataset.py" --workers "${EDGE_AI_WORKERS:-2}" --live || rc_dataset=$?
  echo "=== $(date '+%F %T') g014 train (g013 rc=$rc_dataset) ==="
  nice -n 10 "$PY" "$HERE/g014_edge_ai_train.py" --threads "${EDGE_AI_WORKERS:-2}" || rc_train=$?
  echo "=== $(date '+%F %T') done (g014 rc=$rc_train) ==="
} >>"$LOG" 2>&1
"$PY" "$HERE/g024_ops_check.py" --edge-ai --rc-dataset "$rc_dataset" --rc-train "$rc_train" --log "$LOG" >>"$LOG" 2>&1 || true

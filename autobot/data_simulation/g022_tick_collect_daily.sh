#!/usr/bin/env bash
# Daily g011 tick collector, independent of r003 (60/120-tick history for the g003 triple-trigger replay).
#   - Subscribes the r004 watchlist first, then fills up to 40 codes with the newest g002 scan candidates.
#   - Skips when a g011 is already running (r003's supervisor reuses this one - r004 codes come first, so the
#     watched codes are all subscribed), on a holiday, and outside [open-15m, close) of the r014 calendar
#     (08:45~15:30 normally, 09:45~16:30 on CSAT day). Re-running it is safe: g011 appends to the day's files,
#     holds its own single-instance lock, and r009 drops duplicate prints by ACML_VOL.
#   - g011's stop time defaults to the calendar close + 1 min (TICK_COLLECTOR_UNTIL=HH:MM overrides).
#
# crontab (weekdays; the 5-minute repeats restart g011 if it died; hour 16 covers late-close days):
#   50,55 8 * * 1-5   /data/srv/autobot/real_trading_bot/autobot/data_simulation/g022_tick_collect_daily.sh
#   */5 9-16 * * 1-5  /data/srv/autobot/real_trading_bot/autobot/data_simulation/g022_tick_collect_daily.sh
# Do not wrap it in another `flock`: that lock fd would be inherited by g011 and block every later run.
#
# Update log:
# - [2026-11-02] type=fix owner=claude
#     summary: 기동 구간을 r014 특수일 시간표에서 계산(휴장일 skip, 수능일 09:45~16:30). --until 기본값은 g011이
#       시간표로 결정. g011 자체 단일 실행 잠금 추가에 맞춰 설명 갱신.
# - [2026-11-02] type=feat owner=claude
#     summary: 신규 - r003 미기동일에도 틱을 쌓도록 g011을 cron으로 단독 기동(감시 종목 + g002 스캔 상위 후보, 40종목).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${TICK_COLLECTOR_PYTHON:-$HOME/venv/bin/python}"
RUNTIME="${AUTO_TRADING_RUNTIME:-$(cd "$HERE/../../.." && pwd)/runtime}"
CALENDAR="$HERE/../live_trading/r014_market_calendar.txt"
today=$(date +%Y%m%d)
# r014 calendar: "YYYYMMDD closed" or "YYYYMMDD HH:MM HH:MM" (same rules as r014_market_calendar.py)
special=$(awk -v d="$today" '{sub(/#.*/, "")} $1 == d {print $2, $3; exit}' "$CALENDAR" 2>/dev/null || true)
open_hm="09:00"; close_hm="15:30"
if [ -n "$special" ]; then
  set -- $special
  if [ "${1,,}" = "closed" ]; then
    exit 0
  fi
  if [[ "${1:-}" =~ ^[0-9]{1,2}:[0-9]{2}$ && "${2:-}" =~ ^[0-9]{1,2}:[0-9]{2}$ ]]; then
    open_hm="$1"; close_hm="$2"
  fi
fi
if [ "$(date +%u)" -ge 6 ]; then
  exit 0
fi
to_min() { local h=${1%%:*} m=${1##*:}; echo $((10#$h * 60 + 10#$m)); }
now_min=$(to_min "$(date +%H:%M)")
if [ "$now_min" -lt $(( $(to_min "$open_hm") - 15 )) ] || [ "$now_min" -ge "$(to_min "$close_hm")" ]; then
  exit 0
fi
LOG_DIR="$RUNTIME/log/$(date +%Y%m%d)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/g011_tick_collector.log"
exec 9>"$RUNTIME/log/.g022.lock"
flock -n 9 || exit 0
if pgrep -f "g011_tick_collector.py" >/dev/null; then
  exit 0
fi
UNTIL_ARGS=()
if [ -n "${TICK_COLLECTOR_UNTIL:-}" ]; then
  UNTIL_ARGS=(--until "$TICK_COLLECTOR_UNTIL")
fi
echo "$(date '+%F %T') g022 starting g011 --fill-from-scan ${UNTIL_ARGS[*]:-} (session $open_hm~$close_hm)" >>"$LOG"
cd "$HERE"
setsid "$PY" "$HERE/g011_tick_collector.py" --fill-from-scan "${UNTIL_ARGS[@]}" </dev/null >>"$LOG" 2>&1 9>&- &

#!/usr/bin/env bash
# Daily g011 tick collector, independent of r003 (60/120-tick history for the g003 triple-trigger replay).
#   - Subscribes the r004 watchlist first, then fills up to 40 codes with the newest g002 scan candidates.
#   - Skips when a g011 is already running (r003's supervisor reuses this one - r004 codes come first, so the
#     watched codes are all subscribed), and outside 08:45~15:30. Re-running it is safe: g011 appends to the
#     day's files and r009 drops duplicate prints by ACML_VOL.
#
# crontab (weekdays; the 5-minute repeats restart g011 if it died):
#   50,55 8 * * 1-5   /data/srv/autobot/auto_trading_src/data_simulation/g022_tick_collect_daily.sh
#   */5 9-15 * * 1-5  /data/srv/autobot/auto_trading_src/data_simulation/g022_tick_collect_daily.sh
#
# Update log:
# - [2026-11-02] type=feat owner=claude
#     summary: 신규 - r003 미기동일에도 틱을 쌓도록 g011을 cron으로 단독 기동(감시 종목 + g002 스캔 상위 후보, 40종목).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${TICK_COLLECTOR_PYTHON:-$HOME/venv/bin/python}"
RUNTIME="${AUTO_TRADING_RUNTIME:-$(cd "$HERE/../.." && pwd)/runtime}"
UNTIL="${TICK_COLLECTOR_UNTIL:-15:31}"
now=$(date +%H%M)
if [ "$now" -lt 0845 ] || [ "$now" -ge 1530 ]; then
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
echo "$(date '+%F %T') g022 starting g011 --fill-from-scan --until $UNTIL" >>"$LOG"
cd "$HERE"
setsid "$PY" "$HERE/g011_tick_collector.py" --fill-from-scan --until "$UNTIL" </dev/null >>"$LOG" 2>&1 9>&- &

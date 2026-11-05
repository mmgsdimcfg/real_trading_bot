#!/usr/bin/env bash
# r003 service wrapper (systemd --user autobot-r003.service -> this script -> venv python r003).
# Reads <runtime>/config/r003.env written by r016_control.py / AutoBotH as DATA (not sourced - no shell code runs):
#   R003_DRY_RUN=1            -> --dry-run (주문을 보내지 않고 로그만)
#   AUTO_TRADING_RISK_PROFILE -> conservative|neutral|aggressive (r001 risk profile, exported)
#   R003_DATE / R003_WATCHLIST_SOURCE -> --date / --watchlist-source
#   R003_EXTRA_ARGS           -> extra r003 arguments, split on spaces (no globbing)
# Manual run (same as the service): bash live_trading/r003_service.sh
set -u
set -f   # no globbing when splitting R003_EXTRA_ARGS
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${AUTO_TRADING_RUNTIME:-$(cd "$HERE/../.." && pwd)/runtime}"
ENV_FILE="$RUNTIME/config/r003.env"
R003_DRY_RUN="${R003_DRY_RUN:-0}"
R003_DATE="${R003_DATE:-}"
R003_WATCHLIST_SOURCE="${R003_WATCHLIST_SOURCE:-}"
R003_EXTRA_ARGS="${R003_EXTRA_ARGS:-}"
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        key="${line%%=*}"
        val="${line#*=}"
        val="${val%\"}"; val="${val#\"}"
        case "$val" in *[\`\$\;\|\&\<\>\(\)]*) echo "[r003_service] ignoring unsafe value for $key" >&2; continue ;; esac
        case "$key" in
            R003_DRY_RUN) R003_DRY_RUN="$val" ;;
            AUTO_TRADING_RISK_PROFILE) export AUTO_TRADING_RISK_PROFILE="$val" ;;
            R003_DATE) R003_DATE="$val" ;;
            R003_WATCHLIST_SOURCE) R003_WATCHLIST_SOURCE="$val" ;;
            R003_EXTRA_ARGS) R003_EXTRA_ARGS="$val" ;;
            *) echo "[r003_service] ignoring unknown key $key" >&2 ;;
        esac
    done < "$ENV_FILE"
fi
PY="${AUTOBOT_PYTHON:-$HOME/venv/bin/python}"
ARGS=()
[ "$R003_DRY_RUN" = "1" ] && ARGS+=(--dry-run)
[ -n "$R003_DATE" ] && ARGS+=(--date "$R003_DATE")
[ -n "$R003_WATCHLIST_SOURCE" ] && ARGS+=(--watchlist-source "$R003_WATCHLIST_SOURCE")
if [ -n "$R003_EXTRA_ARGS" ]; then
    read -r -a EXTRA <<< "$R003_EXTRA_ARGS"
    ARGS+=("${EXTRA[@]}")
fi
cd "$HERE" && exec "$PY" r003_trade_live_execute.py "${ARGS[@]}"

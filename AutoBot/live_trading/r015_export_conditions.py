# -*- coding: utf-8 -*-
"""R015 - 매수/매도 조건 목록을 JSON으로 내보낸다 (조건 다이어그램 뷰어용, 읽기 전용).

실행: ~/venv/bin/python live_trading/r015_export_conditions.py [--out path.json]
출력: {"generated_at", "entry_trigger_mode", "trading_strategy_mode", "buy": [...], "sell": [...]}
  - buy/sell 항목: {"order": 평가 순서(0부터), "no": 번호, "func": 함수명, "stage": STATE|TRIGGER|CONTEXT|PRE_ORDER / GUARD|UPDATE|EXIT|STATE,
                   "title", "description", "defines": [r001 상수명...], "values": {상수명: 값}}
  - 브로커 import 없음. r005/r006의 describe_* 와 active_* 를 그대로 쓰므로 코드가 바뀌면 출력도 바뀐다.

Update log:
- [2026-11-06] type=feat owner=claude
    summary: AutoBotH 차트 요청(P2-1) - 각 조건에 frame 필드(none|live|1m|3m|tick|daily, '+'로 복수) 추가. r003 의 조건별
      데이터 출처: TRIGGER _008 = 1분봉 MA5/MA20 + 60/120틱봉(triple), CONTEXT = 3분봉(buy_frame), _029 = 지수 ETF 1분봉,
      _018/_028 = 전일 종가(일봉), PRE_ORDER 나머지 = 현재가/호가(live). 매도: ATR 계열/반전/같은 봉 판정 = 3분봉, 나머지 live.
      최상위 frames 에 설명과 예열 범위(전일 1분봉 180개 -> 3분봉 60개, r001 PRIOR_DAY_WARMUP_MINUTES) 포함.
    impact: tooling
    compatibility: backward-compatible (필드 추가)
- [2026-11-04] type=feat owner=claude
    summary: 윈도우 조건 뷰어(ssh로 로그 수집) 구현 가이드용 - 조건 순서/설명/관련 상수 값을 JSON으로.
    impact: tooling
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib  # noqa: E402
import io  # noqa: E402

# r001 import가 stdout에 "[INFO] Applied risk profile ..."를 찍으므로 JSON만 나가도록 import 동안 stdout을 막는다.
with contextlib.redirect_stdout(io.StringIO()):
    import r001_define_config as r001  # noqa: E402
    import r005_buy_conditions as r005  # noqa: E402
    import r006_sell_conditions as r006  # noqa: E402


# 조건 -> 봇이 그 조건을 판정할 때 보는 데이터 프레임 (AutoBotH 가 노드에서 해당 차트를 바로 열 때 사용).
#   none  = 데이터 무관(상태/시간/계좌)   live = KIS 현재가·호가 폴링   1m = KIS 1분봉(분 시작 라벨)   3m = 1분봉 3개 집계(종료 라벨)
#   tick  = g011 틱 60/120 틱봉           daily = 일봉(전일 종가 등)
BUY_FRAME_BY_FUNC = {
    "_001_entry_time_window": "none", "_002_session_open_warmup": "none", "_003_trade_cooldown": "none",
    "_004_same_bar_signal_dedup": "3m", "_005_no_open_buy_exposure": "none", "_006_hard_stop_reentry_block": "none",
    "_007_daily_loss_limit": "none", "_008_max_open_positions": "none",
    "_008_triple_cross_trigger": "1m+tick", "_009_hybrid_1min_trigger": "1m",
    "_010_market_regime": "1m", "_011_context_frame_ready": "3m", "_012_min_entry_atr_volatility": "1m+3m",
    "_013_min_liquidity_safety": "3m", "_014_context_score_record": "3m",
    "_015_excessive_rise_from_prev_close": "live+daily", "_016_anti_chase_day_extension": "live+1m",
    "_017_opening_gap_volume_gate": "3m+daily", "_018_quote_status": "live", "_019_account_sync_fresh": "none",
    "_020_edge_ai_meta_filter": "3m+daily", "_021_entry_spread": "live", "_022_consecutive_poll_confirm": "live+3m",
    "_023_affordable_buy_qty": "live", "_024_fresh_live_price": "live", "_025_orderbook_ask_not_thin": "live",
}
SELL_FRAME_BY_FUNC = {
    "_001_stale_live_price_guard": "live", "_002_update_position_tracking": "live+3m",
    "_003_peak_next_bar_bearish_exit": "live+3m", "_004_hard_stop_loss": "live", "_005_staged_tp1_partial": "live",
    "_006_surge_ladder_tp": "live", "_007_profit_lock": "live", "_008_atr_take_profit": "live+3m",
    "_009_post_buy_entry_drop_guard": "live", "_010_breakeven_fail_guard": "live", "_011_no_trend_time_exit": "live",
    "_012_atr_stop_loss": "live+3m", "_013_trailing_stop": "live", "_014_same_bar_sell_dedup": "3m",
    "_015_shared_reversal_sell": "3m",
    "_s04_net_stop_loss": "live",   # [2026-11-12] AUTO_SELL_MODE="stop_only"
}
FRAMES_LEGEND = {
    "none": "데이터 무관(상태/시간/계좌)",
    "live": "KIS 현재가/호가 폴링(LIVE_PRICE_POLL_INTERVAL_SECONDS)",
    "1m": "KIS 1분봉(분 시작 라벨, g001 _10s raw_bar=1 행과 같음)",
    "3m": "1분봉 3개 집계(종료 라벨) = r003 buy_frame/sell frame; 지표는 r002.calculate_indicators_with_warmup",
    "tick": "g011 틱으로 만든 60/120 틱봉(r009 TickBarBuilder, r013 트리거)",
    "daily": "일봉(전일 종가 = KIS stck_sdpr 기준가, g001 _daily.csv)",
}


def _value(name: str):
    v = getattr(r001, name, None)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (tuple, list, set, frozenset)):
        return [x if isinstance(x, (int, float, str, bool)) else str(x) for x in v]
    if isinstance(v, dict):
        return {str(k): (x if isinstance(x, (int, float, str, bool)) else str(x)) for k, x in v.items()}
    return str(v)


def _rows(conditions, describe_rows, frame_map=None):
    desc = {d["func"]: d for d in describe_rows}
    frame_map = frame_map or {}
    out = []
    for order, c in enumerate(conditions):
        d = desc.get(c.func_name, {})
        defines = list(getattr(c, "related_defines", ()) or ())
        out.append({
            "order": order, "no": c.no, "func": c.func_name, "stage": getattr(c, "stage", getattr(c, "kind", "")), "title": c.title,
            "frame": frame_map.get(c.func_name, ""),
            "description": d.get("description", getattr(c, "description", "")),
            "defines": defines, "values": {n: _value(n) for n in defines},
        })
    return out


def build() -> dict:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "entry_trigger_mode": str(getattr(r001, "ENTRY_TRIGGER_MODE", "")),
        "trading_strategy_mode": str(getattr(r001, "TRADING_STRATEGY_MODE", "")),
        "auto_sell_mode": r006.auto_sell_mode(),   # [2026-11-12] "all" | "stop_only" | "off" - decides the sell list below
        "frames": FRAMES_LEGEND,
        "warmup": {"prior_day_1m_bars": int(getattr(r001, "PRIOR_DAY_WARMUP_MINUTES", 0) or 0),
                   "enabled": bool(getattr(r001, "ENABLE_PRIOR_DAY_INDICATOR_WARMUP", False)),
                   "note": "전일 원본 1분봉 마지막 N개(및 그 3분 집계)를 당일 봉 앞에 붙여 지표 계산 - 당일 행만 사용"},
        "buy": _rows(r005.active_buy_conditions(), r005.describe_buy_conditions(),
                     {**BUY_FRAME_BY_FUNC, "_009_hybrid_1min_trigger": "1m+tick" if str(getattr(r001, "ENTRY_TRIGGER_MODE", "")) == "triple" else "1m"}),
        "sell": _rows(r006.active_sell_conditions(), r006.describe_sell_conditions(), SELL_FRAME_BY_FUNC),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="export r005/r006 condition lists as JSON")
    ap.add_argument("--out", default="", help="write to this file instead of stdout")
    args = ap.parse_args()
    text = json.dumps(build(), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

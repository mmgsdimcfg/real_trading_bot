"""
g008_selection_reason_report.py

g002 스캔 결과(최종 선정 종목)마다 "왜 선정됐는가"를 한글로 풀어 쓴 리포트를 만든다.
산출물: data/{date}/_{date}_scanner_report_ko.md (g002 실행 시 자동 생성)

종목별 구성:
- 유형(강세 지속형/반등 기대형/완만한 상승형/횡보 후 반전 시도형)과 핵심 수치
- 점수 계산(가점 - 감점) - g002 calculate_candidate_score_breakdown 내역을 그대로 사용
  (리포트에서 점수를 다시 계산하지 않으므로 실제 채점과 어긋나지 않는다)
- 좋은 조건 / 감점 요인 / 점수에 반영 안 된 주의점 / 하드필터 통과 근거(반전 신호 종류 포함)

g002를 import하지 않는다(g002가 __main__으로 실행되므로 재import 시 모듈이 두 번 로드됨).
일봉 로더와 반전 신호 판정 함수는 g002가 인자로 넘겨준다.
"""

from __future__ import annotations

from typing import Callable

TREND_KO = {"up": "상승", "flat": "횡보", "down": "하락"}

REVERSAL_OVERRIDE_FLAGS = {
    "flat_trend_reversal_override": "횡보 추세 예외 통과",
    "down_trend_reversal_override": "하락 추세 예외 통과",
    "box_range_no_progress_reversal_override": "박스권 정체 예외 통과",
    "bearish_candle_dominant_reversal_override": "음봉 우세 예외 통과",
}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:+.1f}%"


def _num(x, default=0.0) -> float:
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


def _soft_flag_label(flag: str, row: dict) -> str:
    if flag in REVERSAL_OVERRIDE_FLAGS:
        return REVERSAL_OVERRIDE_FLAGS[flag]
    if flag.startswith("recent_pick_repeat_"):
        return "반복 선정 플래그"
    if flag.startswith("bearish_2in3d_"):
        return f"최근 3일 중 음봉 {flag.rsplit('_', 1)[-1]}개"
    high = row.get("high_52w_ratio")
    low = row.get("low_52w_ratio")
    labels = {
        "weak_signal_long_term_downtrend": "20일 회귀선 기울기 약한 하락",
        "bb_lower_not_uptrend": "볼린저 하단선 아직 상승 전환 안 됨",
        "low_up_days_warning": f"최근 5일 중 상승일 {row.get('up_days_in_5')}일뿐",
        "far_from_52w_high": f"52주 고가 대비 {_num(high) * 100:.0f}%로 멂(장기 하락 이력)",
        "near_52w_low_unconfirmed": f"52주 저가 대비 +{_num(low) * 100:.0f}%로 바닥권(반등 미확인)",
        "near_52w_high_override": "52주 신고가 근접(돌파 예외)",
        "near_52w_high": "52주 신고가 근접",
        "rsi_overbought": "RSI 과매수(75 초과)",
        "volume_declining": "거래량 감소",
        "low_volatility_atr": "변동성 하한 미달",
        "3d_close_downtrend": "3일 연속 종가 하락",
        "daily_5d_downtrend": "5일 연속 종가 하락",
        "prev_day_gap_risk": "전일 급등락 갭 리스크",
        "amount_below_market_avg": "거래대금 시장 평균 미달",
        "volume_below_market_price_adjusted_avg": "거래량 시장 평균 미달",
        "above_bb_upper": "볼린저 상단 위 마감(단기 과열)",
        "stoch_overbought": "스토캐스틱/윌리엄스 과매수",
    }
    return labels.get(flag, flag)


def _daily_features(df, has_volume_thrust_reversal: Callable, box_window: int, bearish_window: int) -> dict:
    """반전 신호 종류/최근 캔들/박스권·음봉 우세 수치. g002 _has_reversal_signal과 같은 정의를
    종류별로 나눠 판정한다(어느 신호로 예외 통과했는지 리포트에 쓰기 위해)."""
    feat = {"candles": "", "reversals": [], "last_vol_x": None, "bear_count": None,
            "box_net": None, "box_range": None}
    if df is None or len(df) == 0:
        return feat
    tail5 = df.tail(5)
    feat["candles"] = "".join(
        "양" if c > o else ("음" if c < o else "보") for o, c in zip(tail5["open"], tail5["close"])
    )
    if len(df) >= 2:
        last2 = df.tail(2)
        if bool((last2["close"].values > last2["open"].values).all()):
            feat["reversals"].append("2일 연속 양봉")
    closes = df["close"].dropna().tail(3).values
    if len(closes) >= 3 and closes[-1] > closes[-2] > closes[-3]:
        feat["reversals"].append("2일 연속 종가 상승")
    if has_volume_thrust_reversal(df):
        feat["reversals"].append("거래량 동반 V자 장대양봉")
    if len(df) >= 6:
        prior_avg = float(df["volume"].iloc[-6:-1].mean())
        if prior_avg > 0:
            feat["last_vol_x"] = float(df["volume"].iloc[-1]) / prior_avg
    recent_bear = df.tail(bearish_window)
    feat["bear_count"] = int((recent_bear["close"] < recent_bear["open"]).sum())
    if len(df) >= box_window:
        rec = df.tail(box_window)
        last_close = float(rec["close"].iloc[-1])
        first_close = float(rec["close"].iloc[0])
        if first_close > 0 and last_close > 0:
            feat["box_net"] = last_close / first_close - 1
            feat["box_range"] = (float(rec["high"].max()) - float(rec["low"].min())) / last_close
    return feat


def _classify(row: dict) -> str:
    cm = _num(row.get("close_ma20_ratio"))
    rt = _num(row.get("ret_20d"))
    trend = row.get("trend_state")
    if cm > 0.08 or rt > 0.25:
        return "강세 지속형 (이미 많이 오른 종목, 추격 주의)"
    if trend == "down" or cm < -0.02:
        return "반등 기대형 (조정 후 반전 신호)"
    if trend == "up":
        return "완만한 상승형 (추세 양호, 과열 없음)"
    return "횡보 후 반전 시도형"


def _render_stock(rank: int, row: dict, feat: dict, config) -> str:
    bd = row.get("score_breakdown") or {"gains": [], "penalties": []}
    gains = {g["key"]: g for g in bd["gains"]}
    pens = {p["key"]: p for p in bd["penalties"]}
    gain_total = sum(g["value"] for g in bd["gains"])
    pen_total = sum(p["value"] for p in bd["penalties"])

    price = _num(row.get("price"))
    atr = _num(row.get("atr_ratio"))
    amt_eok = _num(row.get("amount_ma20")) / 1e8
    rsi = row.get("rsi")
    cm = _num(row.get("close_ma20_ratio"))
    rt = _num(row.get("ret_20d"))
    pr = _num(row.get("prev_day_return"))
    high = _num(row.get("high_52w_ratio"))
    trend = row.get("trend_state")
    up5 = row.get("up_days_in_5")
    vtr = row.get("vol_trend_ratio")
    candles = feat["candles"]
    rev = " + ".join(feat["reversals"]) if feat["reversals"] else "없음"
    scored_flags = (pens.get("soft_flags") or {}).get("flags", [])
    later_flags = [f for f in row.get("soft_flags", []) if f not in scored_flags]

    g_atr = gains.get("atr", {"value": 0.0})["value"]
    g_amt = gains.get("amount", {"value": 0.0})["value"]
    g_rsi = gains.get("rsi", {"value": 0.0})["value"]
    g_vol = gains.get("vol_rel_strength", {"value": 0.0})
    vol_note = "데이터 부족 0" if g_vol.get("missing") else f"{g_vol['value']:.1f}"

    if pen_total < 2.5:
        rank_note = "감점이 거의 없어 최상위권"
    elif pen_total < 5.0:
        rank_note = "감점이 중간 수준"
    else:
        rank_note = "감점이 커서 하위권"

    # ---- 좋은 조건
    good = []
    atr_full = g_atr >= 15.95
    good.append(
        f"**변동성(ATR) {atr * 100:.1f}%** → {g_atr:.1f}/16점"
        + (f" (만점, 기준 {config.atr_ratio_min * 100:.1f}%의 {atr / config.atr_ratio_min:.0f}배)" if atr_full else "")
    )
    good.append(
        f"**20일 평균 거래대금 {amt_eok:,.0f}억** → {g_amt:.1f}/18점"
        + (" (만점)" if g_amt >= 17.95 else " (만점 미달, 상대적 약점)")
    )
    if rsi is not None and 40.0 <= rsi <= 60.0:
        good.append(f"**RSI {rsi:.1f}** → 8/8점 (과열 전 최적 구간 40~60)")
    elif rsi is not None:
        good.append(f"RSI {rsi:.1f} → {g_rsi:.1f}/8점")
    # [2026-10-10] g002 일봉 지표 기반 우상향 필수조건 + 지표 가점
    _sig_kr = {"macd_turn_up": "MACD 상승전환", "stoch_golden": "스토캐스틱 골든크로스", "di_plus_lead": "DI+ 우위",
               "rsi_50_68": "RSI 50~68", "volume_1_3x": "거래량 1.3배+", "obv_rising": "OBV 상승"}
    if row.get("pattern_tier"):
        _tier_txt = ("A: 당일 양봉으로 직전 5일 종가 고점 돌파" if row["pattern_tier"] == "A"
                     else "B: 당일 양봉, 최근 5일 음봉 1개 이하, 20일선 +10% 이내")
        _sigs = ", ".join(_sig_kr.get(x, x) for x in row.get("buy_signals") or [])
        _ma120 = "120일선 상승, " if row.get("ma120_slope20") is not None else "120일선 이력 부족(미확인), "
        good.insert(0, f"**상승 초입 패턴 {_tier_txt}** ({_ma120}눌림/바닥 출발) | 매수신호 {len(row.get('buy_signals') or [])}개: {_sigs}")
    _ind_items = [("adx_trend", "ADX 추세강도", 4), ("ma_alignment", "이평 정배열(5>20>60)", 3), ("obv_trend", "OBV 상승", 3),
                  ("macd_momentum", "MACD 히스토그램 증가", 2), ("stoch_bullish", "스토캐스틱 상승 교차", 2),
                  ("close_above_vwap", "종가가 당일 VWAP 위", 2)]
    _ind_got = [f"{label} {gains[k]['value']:.1f}/{mx}" for k, label, mx in _ind_items if k in gains and gains[k]["value"] > 0]
    if _ind_got:
        good.append("지표 가점: " + ", ".join(_ind_got))
    if 0 <= cm <= 0.08 and rt <= 0.25:
        good.append(f"20일선 위 {_pct(cm)}, 20일 수익률 {_pct(rt)}: 너무 오르지 않아 **과열 감점 없음**")
    elif -0.08 <= cm < 0:
        good.append(f"20일선 대비 {_pct(cm)}의 눌림 위치: 과열 감점 없음")
    if up5 is not None and up5 >= 4:
        good.append(f"최근 5일 중 {up5}일 상승: 단기 매수세 우위")
    if feat["last_vol_x"] is not None and feat["last_vol_x"] >= 1.5:
        good.append(f"당일 거래량이 직전 5일 평균의 {feat['last_vol_x']:.1f}배: 반등에 수급이 실렸다")
    if vtr is not None and vtr >= 1.3:
        good.append(f"최근 5일 거래량이 직전 5일의 {vtr:.1f}배로 증가 (점수 반영 없음)")
    if trend != "up" and feat["reversals"]:
        good.append(f"최근 5일 캔들 {candles}: 마지막에 반전 신호(**{rev}**) 발생")
    if 0.4 <= high <= 0.7:
        good.append(f"52주 고가의 {high * 100:.0f}% 수준: 위쪽 매물대까지 여유")
    if 0 < pr < 0.05 and not candles.endswith("음"):
        good.append(f"당일 {_pct(pr)} 상승 마감")

    # ---- 감점 요인 (채점 내역 그대로)
    bad = []
    if "overheat_ma20_gap" in pens:
        bad.append(f"20일선 이격 {_pct(cm)} (8% 초과) → -{pens['overheat_ma20_gap']['value']:.1f}점")
    if "overheat_ret_20d" in pens:
        bad.append(f"20일 수익률 {_pct(rt)} (25% 초과) → -{pens['overheat_ret_20d']['value']:.1f}점")
    if "overheat_prev_day_jump" in pens:
        bad.append(f"당일 급등 {_pct(pr)} (8% 초과) → -{pens['overheat_prev_day_jump']['value']:.1f}점")
    if "last_bearish" in pens:
        bad.append(f"당일 음봉 마감 (전일 대비 {_pct(pr)}, 시가보다 낮게 끝남) → -4.0점")
    if "near_52w_high" in pens:
        bad.append(f"52주 고가의 {high * 100:.0f}% (신고가 과열 구간) → -{pens['near_52w_high']['value']:.1f}점")
    if "prev_day_change_gap" in pens:
        bad.append(f"전일 등락 과대(갭 리스크) → -{pens['prev_day_change_gap']['value']:.1f}점")
    if "recent_pick_repeat" in pens:
        bad.append(
            f"최근 {config.recent_pick_penalty_lookback_days}일 내 {row.get('repeat_recent_days')}회 기선정(반복 선정)"
            f" → -{pens['recent_pick_repeat']['value']:.0f}점"
        )
    if rsi is not None and g_rsi < 7.99:
        side = "60 초과" if rsi > 60 else "40 미만"
        bad.append(f"RSI {rsi:.1f}: 최적 구간({side})을 벗어나 RSI 점수 {8 - g_rsi:.1f}점 손실")
    if g_amt < 17.95:
        bad.append(f"거래대금이 시장 기준 대비 부족해 거래대금 점수 {18 - g_amt:.1f}점 손실")
    if not atr_full:
        bad.append(f"변동성이 상대적으로 작아 ATR 점수 {16 - g_atr:.1f}점 손실")
    for flag in scored_flags:
        weight = 1.5 if flag.endswith("_reversal_override") else 0.7
        bad.append(f"{_soft_flag_label(flag, row)} → -{weight}점")

    # ---- 점수에 반영 안 된 주의점
    risk = []
    if pr >= 0.05:
        risk.append(f"당일 {_pct(pr)} 급등 마감 → 다음 날 갭/차익 매물 주의")
    if feat["last_vol_x"] is not None and feat["last_vol_x"] < 0.5 and trend != "up":
        risk.append(f"반등일 거래량이 직전 5일 평균의 {feat['last_vol_x']:.1f}배로 적어 반전 신뢰도가 낮음")
    if vtr is not None and vtr < 0.4:
        risk.append(f"최근 5일 거래량이 직전 5일의 {vtr:.2f}배로 급감 (관심 식는 중)")
    if candles.count("음") >= 4:
        gap_fade = " (갭 상승 후 밀리는 패턴)" if feat["reversals"] and candles.endswith("음") else ""
        risk.append(f"최근 5일 캔들 {candles}: 음봉 위주{gap_fade}")
    if high >= 0.9:
        risk.append(f"52주 고가의 {high * 100:.0f}%: 신고가 부근 매물")
    for flag in later_flags:
        if flag == "sector_cap_swapped_in":
            risk.append("업종 쏠림 제한으로 다른 종목 대신 교체 편입됨")
        elif flag == "fallback_selected":
            risk.append("적격 종목 부족으로 보충(fallback) 편입됨")
        else:
            risk.append(_soft_flag_label(flag, row))

    # ---- 하드필터 통과 근거
    hf = []
    if trend == "up":
        hf.append("MA5 > MA20 **상승 추세**로 추세 필터를 정상 통과했다.")
    else:
        hf.append(
            f"MA5/MA20 기준 **{TREND_KO.get(trend, trend)} 추세**라 원래는 탈락 대상이었으나, "
            f"최근 일봉의 반전 신호(**{rev}**)가 인정돼 예외로 통과했다 (-1.5점)."
        )
    if "box_range_no_progress_reversal_override" in scored_flags and feat["box_range"] is not None:
        hf.append(
            f"최근 10일 고저폭이 {feat['box_range'] * 100:.1f}%인데 순변동은 {_pct(feat['box_net'])}에 그쳐 "
            "'박스권 정체'에 해당했지만, 같은 반전 신호로 예외 통과했다 (-1.5점)."
        )
    if "bearish_candle_dominant_reversal_override" in scored_flags and feat["bear_count"] is not None:
        hf.append(
            f"최근 10일 중 음봉이 {feat['bear_count']}개(60% 이상)로 '음봉 우세'에 해당했지만, "
            "반전 신호로 예외 통과했다 (-1.5점)."
        )
    hf.append(
        f"가격·유동성(시장 평균 대비)·변동성 하한, 52주 고가 {config.max_52w_high_ratio * 100:.0f}% 미만"
        f"(현재 {high * 100:.0f}%), 전일 등락 {config.max_prev_day_change * 100:.0f}% 미만, "
        "관리/정지 아님 조건을 모두 충족했다."
    )

    lines = [
        f"### {rank}위. {row.get('name')} ({row.get('code')}) — {_num(row.get('score')):.2f}점",
        "",
        f"- **유형**: {_classify(row)}",
        (
            f"- **핵심 수치**: 종가 {price:,.0f}원 (당일 {_pct(pr)}) · ATR {atr * 100:.1f}% · "
            f"거래대금 {amt_eok:,.0f}억 · RSI {'-' if rsi is None else f'{rsi:.1f}'} · 20일선 {_pct(cm)} · "
            f"20일 {_pct(rt)} · 52주고가 대비 {high * 100:.0f}% · 최근 5일 캔들 {candles or '-'}"
        ),
        (
            f"- **점수 계산**: 가점 {gain_total:.1f} (ATR {g_atr:.1f} + 거래대금 {g_amt:.1f} + RSI {g_rsi:.1f}"
            f" + 거래량강도 {vol_note}) − 감점 {pen_total:.1f} = **{_num(row.get('score')):.2f}** → {rank_note}"
        ),
        "",
        "**✅ 선정된 이유 (좋은 조건)**",
        "",
        *[f"- {g}" for g in good],
        "",
        "**⛔ 감점 요인**",
        "",
        *([f"- {b}" for b in bad] or ["- 없음"]),
    ]
    if risk:
        lines += ["", "**⚠️ 점수에 반영 안 된 주의점**", "", *[f"- {r}" for r in risk]]
    lines += ["", "**필터 통과 근거**", "", *[f"- {h}" for h in hf], "", "---", ""]
    return "\n".join(lines)


def render_selection_reason_report(
    selected_rows: list[dict],
    *,
    target_label: str,
    summary: dict,
    config,
    load_daily_df: Callable[[str], object],
    has_volume_thrust_reversal: Callable,
    box_window: int = 10,
    bearish_window: int = 10,
) -> str:
    feats = []
    for row in selected_rows:
        try:
            df = load_daily_df(row["code"])
        except Exception:
            df = None
        feats.append(_daily_features(df, has_volume_thrust_reversal, box_window, bearish_window))

    vol_missing = sum(
        1 for row in selected_rows
        if any(g.get("missing") for g in (row.get("score_breakdown") or {}).get("gains", []))
    )

    out = [
        f"# 종목 선별 리포트 (종목별 선정 이유) — {target_label}",
        "",
        f"> g002 스캐너(`{config.name}`) 최종 선정 {len(selected_rows)}종목의 선정 이유. "
        f"적격 {summary.get('eligible_pool_count', '-')}종목 중 점수 순으로 선정.",
        "> 점수 내역은 실제 채점(`calculate_candidate_score_breakdown`) 결과를 그대로 옮겼다.",
        "",
        "## 읽는 법 — 점수가 매겨지는 방식",
        "",
        "| 구분 | 항목 | 배점 | 만점 조건 |",
        "| --- | --- | ---: | --- |",
        f"| 가점 | 변동성 ATR(일평균 변동폭/종가) | 16 | 약 {config.atr_ratio_min * 4.4 * 100:.1f}% 이상 |",
        "| 가점 | 20일 평균 거래대금(시장 기준 대비) | 18 | 시장 기준의 약 4배 이상 |",
        "| 가점 | RSI(10일) | 8 | 40~60 구간 |",
        "| 가점 | 거래량 상대강도(최근5일/이전20일) | 10 | 일봉 25개 필요 |",
        "| 감점 | 과열 | 최대 -12/-8/-6 | 20일선 이격 8% 초과 / 20일 상승률 25% 초과 / 당일 8% 초과 급등 |",
        "| 감점 | 당일 음봉 | -4 | |",
        f"| 감점 | 최근 {config.recent_pick_penalty_lookback_days}일 내 반복 선정 | 회당 -{config.recent_pick_penalty_per_day:.0f} | |",
        "| 감점 | 주의 플래그 | 개당 -0.7 / -1.5 | 추세·박스권·음봉 필터를 '반전 신호'로 예외 통과하면 -1.5 |",
        "",
        "- 반전 신호 = ① 2일 연속 양봉, ② 2일 연속 종가 상승, ③ 거래량 1.5배 이상 + 3% 이상 양봉 + "
        "전일 고가 돌파(V자 장대양봉) 중 하나 이상",
        "- 가점은 대부분 종목이 만점에 가까워 **순위 차이는 대부분 감점 요인에서 생긴다.**",
        "- 캔들 표기: 양 = 양봉, 음 = 음봉, 보 = 보합 (왼쪽이 5일 전, 오른쪽이 기준일)",
    ]
    if vol_missing:
        out.append(
            f"- ⚠️ 거래량 상대강도는 {vol_missing}/{len(selected_rows)}종목이 일봉 부족(25개 필요)으로 계산되지 않아 0점 처리됐다."
        )
    out += [
        "",
        "## 순위표",
        "",
        "| 순위 | 종목 | 점수 | 종가 | 당일 | ATR% | 거래대금(20일평균) | RSI | 20일선 이격 | 20일 수익률 | 52주고가 대비 | 추세 | 감점 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for rank, row in enumerate(selected_rows, start=1):
        bd = row.get("score_breakdown") or {"penalties": []}
        pen_total = sum(p["value"] for p in bd["penalties"])
        rsi = row.get("rsi")
        out.append(
            f"| {rank} | {row.get('name')}({row.get('code')}) | {_num(row.get('score')):.2f} | "
            f"{_num(row.get('price')):,.0f} | {_pct(row.get('prev_day_return'))} | {_num(row.get('atr_ratio')) * 100:.1f}% | "
            f"{_num(row.get('amount_ma20')) / 1e8:,.0f}억 | {'-' if rsi is None else f'{rsi:.1f}'} | "
            f"{_pct(row.get('close_ma20_ratio'))} | {_pct(row.get('ret_20d'))} | {_num(row.get('high_52w_ratio')) * 100:.0f}% | "
            f"{TREND_KO.get(row.get('trend_state'), row.get('trend_state'))} | -{pen_total:.1f} |"
        )
    out += ["", "", "## 종목별 선정 이유", ""]
    for rank, (row, feat) in enumerate(zip(selected_rows, feats), start=1):
        out.append(_render_stock(rank, row, feat, config))
    return "\n".join(out)

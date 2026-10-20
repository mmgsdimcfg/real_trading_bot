"""
g008_selection_reason_report.py

g002 스캔 결과(최종 선정 종목)마다 "왜 선정됐는가"를 풀어 쓴 리포트를 만든다.
산출물(g002 실행 시 자동 생성):
- data/{date}/_{date}_scanner_report_ko.md     (국문)
- data/{date}/_{date}_scanner_report_en.md  (영문, 같은 내용)

종목별 구성:
- 유형(강세 지속형/반등 기대형/완만한 상승형/횡보 후 반전 시도형)과 핵심 수치
- 점수 계산(가점 - 감점) - g002 calculate_candidate_score_breakdown 내역을 그대로 사용
  (리포트에서 점수를 다시 계산하지 않으므로 실제 채점과 어긋나지 않는다)
- 좋은 조건 / 감점 요인 / 점수에 반영 안 된 주의점 / 필터 통과 근거(반전 신호 종류 포함)
- 리포트 끝: 탈락 사유별 종목 수(Filter Counts)와 주의 플래그(Soft Flags) - 사유마다 뜻풀이 포함

g002를 import하지 않는다(g002가 __main__으로 실행되므로 재import 시 모듈이 두 번 로드됨).
일봉 로더와 반전 신호 판정 함수는 g002가 인자로 넘겨준다.

Update log:
- [2026-10-11] type=feat owner=claude
    summary: 국문/영문 두 언어로 렌더링(render_selection_reason_report(lang="ko"|"en")). 리포트 끝에 g002
      _scanner_report.md의 Filter Counts/Soft Flags를 사유별 뜻풀이와 함께 추가(사용자 요청). "읽는 법" 표를
      현재 선정 기준(상승 초입 패턴 필수 + 지표 가점)에 맞게 갱신.
    impact: scanner
    compatibility: backward-compatible (기본 lang="ko", 기존 호출 그대로 동작)
"""

from __future__ import annotations

import re
from typing import Callable

TREND = {
    "ko": {"up": "상승", "flat": "횡보", "down": "하락"},
    "en": {"up": "up", "flat": "flat", "down": "down"},
}

REVERSAL_OVERRIDE_FLAGS = {
    "flat_trend_reversal_override": ("횡보 추세 예외 통과", "flat trend passed by reversal exception"),
    "down_trend_reversal_override": ("하락 추세 예외 통과", "down trend passed by reversal exception"),
    "box_range_no_progress_reversal_override": ("박스권 정체 예외 통과", "box-range stall passed by reversal exception"),
    "bearish_candle_dominant_reversal_override": ("음봉 우세 예외 통과", "bearish-candle dominance passed by reversal exception"),
}

# [2026-10-20] g002 PATTERN_OVERRIDABLE_TREND_FAILS - 상승 초입 패턴(A/B) 종목은 레거시 추세 하드필터를 경고(-0.7점)로만 반영.
PATTERN_OVERRIDE_FLAGS = {
    "down_trend_pattern_override": ("하락 추세(MA5<MA20) - 패턴 종목이라 경고만", "down trend (MA5<MA20) - warning only for a pattern stock"),
    "flat_trend_pattern_override": ("횡보 추세 - 패턴 종목이라 경고만", "flat trend - warning only for a pattern stock"),
    "box_range_no_progress_pattern_override": ("박스권 정체 - 패턴 종목이라 경고만", "box-range stall - warning only for a pattern stock"),
    "bearish_candle_dominant_pattern_override": ("음봉 우세 - 패턴 종목이라 경고만", "bearish-candle dominance - warning only for a pattern stock"),
    "linreg_long_term_downtrend_pattern_override": ("20일 회귀선 하락 - 패턴 종목이라 경고만", "20-day regression downtrend - warning only for a pattern stock"),
}

BUY_SIGNAL_LABELS = {
    "macd_turn_up": ("MACD 상승전환", "MACD turning up"),
    "stoch_golden": ("스토캐스틱 골든크로스", "stochastic golden cross"),
    "di_plus_lead": ("DI+ 우위", "DI+ above DI-"),
    "rsi_50_68": ("RSI 50~68", "RSI 50-68"),
    "volume_1_3x": ("거래량 1.3배+", "volume 1.3x+"),
    "obv_rising": ("OBV 상승", "OBV rising"),
}

# 탈락 사유(하드필터) 뜻풀이: (국문, 영문)
FAIL_REASON_DESC = {
    "liquidity_below_market_dual": (
        "거래대금·거래량 모두 시장 기준(시장 평균의 70%) 미달 (20일 평균 거래대금 200억 이상, 또는 패턴 종목 50억 이상이면 면제)",
        "Both traded value and volume below the market benchmark (70% of market average); exempt at 20-day avg value >= 20bn KRW, or >= 5bn for pattern stocks"),
    "pattern_below_ma20": ("상승 초입 패턴: 종가가 20일선 이하", "Base-breakout pattern: close at or below MA20"),
    "pattern_below_long_ma": (
        "상승 초입 패턴: 종가가 60·120일선 중 낮은 쪽보다 2% 넘게 아래",
        "Base-breakout pattern: close more than 2% below the lower of MA60/MA120"),
    "pattern_ma20_falling": ("상승 초입 패턴: 20일선이 5일 동안 1% 넘게 하락", "Base-breakout pattern: MA20 fell more than 1% over 5 days"),
    "pattern_ma120_falling": ("상승 초입 패턴: 120일선 20일 변화율 -3% 이하(장기 하락)", "Base-breakout pattern: MA120 20-day change at or below -3% (long-term downtrend)"),
    "pattern_no_pullback_base": (
        "상승 초입 패턴: 직전 10일 동안 20일선 +5% 이내로 내려온 적 없음(눌림/바닥 없이 이미 올라간 상태)",
        "Base-breakout pattern: never within +5% of MA20 in the prior 10 days (already extended, no pullback/base)"),
    "pattern_overheated": ("상승 초입 패턴: 20일선 대비 +15% 초과 또는 RSI 70 초과(과열)", "Base-breakout pattern: more than +15% above MA20 or RSI above 70 (overheated)"),
    "pattern_few_buy_signals": ("상승 초입 패턴: 매수 신호 2개 미만", "Base-breakout pattern: fewer than 2 buy signals"),
    "pattern_no_bullish_close": ("상승 초입 패턴: 당일 음봉/보합 마감", "Base-breakout pattern: day closed bearish or flat"),
    "pattern_bearish_candles": ("상승 초입 패턴 B등급: 최근 5일 음봉 2개 이상", "Base-breakout tier B: 2 or more bearish candles in the last 5 days"),
    "pattern_b_extended": ("상승 초입 패턴 B등급: 돌파 없이 20일선 +10% 초과", "Base-breakout tier B: more than +10% above MA20 without a breakout"),
    "pattern_unknown": ("상승 초입 패턴: 일봉 이력 부족/대상일 일봉 없음으로 판정 불가", "Base-breakout pattern: cannot evaluate (short daily history or no target-day bar)"),
    "uptrend_unknown": ("우상향 판정 불가(일봉 이력 부족)", "Uptrend cannot be evaluated (short daily history)"),
    "bearish_candle_dominant": ("최근 10일 중 음봉 60% 이상", "60% or more bearish candles in the last 10 days"),
    "box_range_no_progress": ("최근 10일 변동폭 8% 이상인데 순변동 1% 미만(박스권 정체)", "Last 10 days: range >= 8% but net change < 1% (box-range stall)"),
    "big_bearish_candle": ("당일 장대음봉(몸통 2% 이상)", "Large bearish candle on the day (body >= 2%)"),
    "down_trend": ("하락 추세(5일선 < 20일선, 5일선 하락)", "Downtrend (MA5 < MA20 and MA5 falling)"),
    "flat_trend": ("추세 미확정(상승도 하락도 아님)", "No confirmed trend (neither up nor down)"),
    "linreg_long_term_downtrend": ("20일 회귀선 기울기 -0.3%/일 이하이고 R² 0.5 이상(뚜렷한 장기 하락)", "20-day regression slope <= -0.3%/day with R² >= 0.5 (clear long-term decline)"),
    "upperlimit_streak_crash": ("15일 내 상한가 2연속 후 -10% 이상 급락", "Two consecutive limit-up days followed by a -10% crash within 15 days"),
    "price_floor": ("주가 2,000원 미만", "Price below 2,000 KRW"),
    "price_ceiling": ("주가 100만원 초과", "Price above 1,000,000 KRW"),
    "preferred_or_spac_excluded": ("우선주 또는 스팩", "Preferred share or SPAC"),
    "admin_issue_or_halt_excluded": ("관리종목 또는 거래정지", "Administrative issue or trading halt"),
    "low_score": ("점수 15점 미만", "Score below 15"),
}

# 주의 플래그(감점만, 탈락 아님) 뜻풀이: (국문, 영문)
SOFT_FLAG_DESC = {
    "volume_declining": ("최근 5일 평균 거래량이 직전 20일 평균보다 적음", "Last-5-day average volume below the prior 20-day average"),
    "bb_lower_not_uptrend": ("볼린저밴드 하단선이 3일간 상승 전환 안 됨", "Bollinger lower band not rising over 3 days"),
    "near_52w_low_unconfirmed": ("52주 저가 대비 +30% 미만(바닥권, 반등 미확인)", "Less than +30% above the 52-week low (near the bottom, rebound unconfirmed)"),
    "low_up_days_warning": ("최근 5일 중 양봉 2일 미만", "Fewer than 2 bullish days in the last 5"),
    "far_from_52w_high": ("52주 고가의 40% 미만(장기 하락 이력)", "Below 40% of the 52-week high (long decline history)"),
    "3d_close_downtrend": ("3일 연속 종가 하락", "Three consecutive lower closes"),
    "daily_5d_downtrend": ("5일 연속 종가 하락", "Five consecutive lower closes"),
    "above_bb_upper": ("종가가 볼린저밴드 상단 위(단기 과열)", "Close above the upper Bollinger band (short-term overheated)"),
    "stoch_overbought": ("스토캐스틱 %K 90 이상 또는 Williams %R -10 이상(과매수)", "Stochastic %K >= 90 or Williams %R >= -10 (overbought)"),
    "rsi_overbought": ("RSI 75 초과(과매수)", "RSI above 75 (overbought)"),
    "weak_signal_long_term_downtrend": ("20일 회귀선 기울기 하락이지만 R² 낮음(약한 하락 신호)", "Negative 20-day regression slope with low R² (weak decline signal)"),
    "low_volatility_atr": ("ATR(변동성) 0.8% 미만", "ATR (volatility) below 0.8%"),
    "near_52w_high": ("52주 고가의 98% 이상(신고가 부근)", "At or above 98% of the 52-week high (near the high)"),
    "near_52w_high_override": ("52주 고가 근접이지만 추세·거래량 지지 신호 3개 이상(감점 축소)", "Near the 52-week high but with 3+ trend/volume support signals (reduced penalty)"),
    "prev_day_gap_risk": ("전일 등락률 ±20% 이상(갭 리스크)", "Previous-day move of ±20% or more (gap risk)"),
    "amount_below_market_avg": ("거래대금이 시장 기준 미달(거래량은 충족)", "Traded value below market benchmark (volume meets it)"),
    "volume_below_market_price_adjusted_avg": ("가격 환산 거래량이 시장 기준 미달(거래대금은 충족)", "Price-adjusted volume below market benchmark (traded value meets it)"),
    "liquidity_below_market_dual_but_absolute_safe": ("시장 기준 미달이지만 20일 평균 거래대금 200억 이상이라 통과", "Below market benchmark but passed: 20-day avg value >= 20bn KRW"),
    "liquidity_below_market_dual_pattern_exempt": ("시장 기준 미달이지만 상승 초입 패턴 + 20일 평균 거래대금 50억 이상이라 통과", "Below market benchmark but passed: base-breakout pattern with 20-day avg value >= 5bn KRW"),
    "sector_cap_swapped_in": ("업종 쏠림 제한(30%)으로 다른 종목 대신 교체 편입", "Swapped in by the 30% sector cap"),
    "fallback_selected": ("적격 종목 부족으로 보충 편입", "Added as fallback (not enough eligible stocks)"),
}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:+.1f}%"


def _num(x, default=0.0) -> float:
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


def _pick(lang: str, pair) -> str:
    return pair[0] if lang == "ko" else pair[1]


def _flag_desc(flag: str, lang: str) -> str:
    """Generic description of a soft flag / fail reason code (for the count tables)."""
    if flag in REVERSAL_OVERRIDE_FLAGS:
        return _pick(lang, REVERSAL_OVERRIDE_FLAGS[flag]) + (
            " (반전 신호로 하드필터 예외 통과, -1.5점)" if lang == "ko" else " (hard filter waived by a reversal signal, -1.5)")
    if flag in PATTERN_OVERRIDE_FLAGS:
        return _pick(lang, PATTERN_OVERRIDE_FLAGS[flag]) + (
            " (상승 초입 패턴 종목은 하드필터 대신 경고, -0.7점)" if lang == "ko" else " (pattern stock: hard filter downgraded to a warning, -0.7)")
    m = re.fullmatch(r"bearish_2in3d_(\d+)", flag)
    if m:
        return (f"최근 3일 중 음봉 {m.group(1)}개" if lang == "ko" else f"{m.group(1)} bearish candles in the last 3 days")
    m = re.fullmatch(r"recent_pick_repeat_(\d+)d", flag)
    if m:
        return (f"최근 선정 기간(4거래일) 중 {m.group(1)}회 선정됨(반복 선정 감점)" if lang == "ko"
                else f"Picked {m.group(1)} time(s) in the recent lookback (4 trading days; repeat-pick penalty)")
    if flag.startswith("uptrend_"):
        return "이전 우상향 필수조건(현재 미사용)" if lang == "ko" else "Former uptrend gate (no longer used)"
    for table in (FAIL_REASON_DESC, SOFT_FLAG_DESC):
        if flag in table:
            return _pick(lang, table[flag])
    return "-"


def _soft_flag_label(flag: str, row: dict, lang: str = "ko") -> str:
    """Per-stock soft flag text (uses the stock's own values where useful)."""
    ko = lang == "ko"
    if flag in REVERSAL_OVERRIDE_FLAGS:
        return _pick(lang, REVERSAL_OVERRIDE_FLAGS[flag])
    if flag in PATTERN_OVERRIDE_FLAGS:
        return _pick(lang, PATTERN_OVERRIDE_FLAGS[flag])
    if flag.startswith("recent_pick_repeat_"):
        return "반복 선정 플래그" if ko else "repeat-pick flag"
    if flag.startswith("bearish_2in3d_"):
        n = flag.rsplit("_", 1)[-1]
        return f"최근 3일 중 음봉 {n}개" if ko else f"{n} bearish candles in the last 3 days"
    high = _num(row.get("high_52w_ratio")) * 100
    low = _num(row.get("low_52w_ratio")) * 100
    up5 = row.get("up_days_in_5")
    specific = {
        "low_up_days_warning": (f"최근 5일 중 상승일 {up5}일뿐", f"only {up5} up days in the last 5"),
        "far_from_52w_high": (f"52주 고가 대비 {high:.0f}%로 멂(장기 하락 이력)", f"{high:.0f}% of the 52-week high (long decline history)"),
        "near_52w_low_unconfirmed": (f"52주 저가 대비 +{low:.0f}%로 바닥권(반등 미확인)", f"+{low:.0f}% above the 52-week low (near bottom, rebound unconfirmed)"),
        "near_52w_high_override": ("52주 신고가 근접(돌파 예외)", "near the 52-week high (breakout exception)"),
        "near_52w_high": ("52주 신고가 근접", "near the 52-week high"),
        "weak_signal_long_term_downtrend": ("20일 회귀선 기울기 약한 하락", "weak negative 20-day regression slope"),
        "bb_lower_not_uptrend": ("볼린저 하단선 아직 상승 전환 안 됨", "Bollinger lower band not yet rising"),
        "rsi_overbought": ("RSI 과매수(75 초과)", "RSI overbought (above 75)"),
        "volume_declining": ("거래량 감소", "declining volume"),
        "low_volatility_atr": ("변동성 하한 미달", "volatility below the minimum"),
        "3d_close_downtrend": ("3일 연속 종가 하락", "three consecutive lower closes"),
        "daily_5d_downtrend": ("5일 연속 종가 하락", "five consecutive lower closes"),
        "prev_day_gap_risk": ("전일 급등락 갭 리스크", "previous-day gap risk"),
        "amount_below_market_avg": ("거래대금 시장 평균 미달", "traded value below market average"),
        "volume_below_market_price_adjusted_avg": ("거래량 시장 평균 미달", "volume below market average"),
        "above_bb_upper": ("볼린저 상단 위 마감(단기 과열)", "closed above the upper Bollinger band (short-term overheated)"),
        "stoch_overbought": ("스토캐스틱/윌리엄스 과매수", "stochastic/Williams overbought"),
        "liquidity_below_market_dual_pattern_exempt": ("거래대금 시장 기준 미달(패턴 종목 50억+ 예외 통과)", "below market liquidity benchmark (pattern stock >= 5bn KRW exemption)"),
        "liquidity_below_market_dual_but_absolute_safe": ("거래대금 시장 기준 미달(200억+ 예외 통과)", "below market liquidity benchmark (>= 20bn KRW exemption)"),
    }
    if flag in specific:
        return _pick(lang, specific[flag])
    return flag


def _daily_features(df, has_volume_thrust_reversal: Callable, box_window: int, bearish_window: int) -> dict:
    """반전 신호 종류/최근 캔들/박스권·음봉 우세 수치. g002 _has_reversal_signal과 같은 정의를
    종류별로 나눠 판정한다(어느 신호로 예외 통과했는지 리포트에 쓰기 위해). 캔들은 U/D/=로 저장."""
    feat = {"candles": "", "reversals": [], "last_vol_x": None, "bear_count": None,
            "box_net": None, "box_range": None}
    if df is None or len(df) == 0:
        return feat
    tail5 = df.tail(5)
    feat["candles"] = "".join(
        "U" if c > o else ("D" if c < o else "=") for o, c in zip(tail5["open"], tail5["close"])
    )
    if len(df) >= 2:
        last2 = df.tail(2)
        if bool((last2["close"].values > last2["open"].values).all()):
            feat["reversals"].append("two_bull")
    closes = df["close"].dropna().tail(3).values
    if len(closes) >= 3 and closes[-1] > closes[-2] > closes[-3]:
        feat["reversals"].append("two_up_closes")
    if has_volume_thrust_reversal(df):
        feat["reversals"].append("volume_thrust")
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


REVERSAL_LABELS = {
    "two_bull": ("2일 연속 양봉", "two consecutive bullish candles"),
    "two_up_closes": ("2일 연속 종가 상승", "two consecutive higher closes"),
    "volume_thrust": ("거래량 동반 V자 장대양봉", "volume-backed V-shaped large bullish candle"),
}


def _candles(text: str, lang: str) -> str:
    if lang == "ko":
        return text.replace("U", "양").replace("D", "음").replace("=", "보")
    return text


def _classify(row: dict, lang: str) -> str:
    cm = _num(row.get("close_ma20_ratio"))
    rt = _num(row.get("ret_20d"))
    trend = row.get("trend_state")
    if cm > 0.08 or rt > 0.25:
        return _pick(lang, ("강세 지속형 (이미 많이 오른 종목, 추격 주의)", "Strong momentum (already up a lot - beware of chasing)"))
    if trend == "down" or cm < -0.02:
        return _pick(lang, ("반등 기대형 (조정 후 반전 신호)", "Rebound candidate (reversal signal after a pullback)"))
    if trend == "up":
        return _pick(lang, ("완만한 상승형 (추세 양호, 과열 없음)", "Steady riser (healthy trend, not overheated)"))
    return _pick(lang, ("횡보 후 반전 시도형", "Sideways, attempting a reversal"))


def _render_stock(rank: int, row: dict, feat: dict, config, lang: str = "ko") -> str:
    ko = lang == "ko"
    bd = row.get("score_breakdown") or {"gains": [], "penalties": []}
    gains = {g["key"]: g for g in bd["gains"]}
    pens = {p["key"]: p for p in bd["penalties"]}
    gain_total = sum(g["value"] for g in bd["gains"])
    pen_total = sum(p["value"] for p in bd["penalties"])

    price = _num(row.get("price"))
    atr = _num(row.get("atr_ratio"))
    amt_eok = _num(row.get("amount_ma20")) / 1e8
    amt_txt = f"{amt_eok:,.0f}억" if ko else f"{amt_eok / 10:,.1f}bn KRW"
    rsi = row.get("rsi")
    cm = _num(row.get("close_ma20_ratio"))
    rt = _num(row.get("ret_20d"))
    pr = _num(row.get("prev_day_return"))
    high = _num(row.get("high_52w_ratio"))
    trend = row.get("trend_state")
    up5 = row.get("up_days_in_5")
    vtr = row.get("vol_trend_ratio")
    candles = _candles(feat["candles"], lang)
    rev = " + ".join(_pick(lang, REVERSAL_LABELS[r]) for r in feat["reversals"]) if feat["reversals"] else ("없음" if ko else "none")
    scored_flags = (pens.get("soft_flags") or {}).get("flags", [])
    later_flags = [f for f in row.get("soft_flags", []) if f not in scored_flags]

    g_atr = gains.get("atr", {"value": 0.0})["value"]
    g_amt = gains.get("amount", {"value": 0.0})["value"]
    g_rsi = gains.get("rsi", {"value": 0.0})["value"]
    g_vol = gains.get("vol_rel_strength", {"value": 0.0})
    vol_note = ("데이터 부족 0" if ko else "no data 0") if g_vol.get("missing") else f"{g_vol['value']:.1f}"
    ind_keys = ("adx_trend", "ma_alignment", "obv_trend", "macd_momentum", "stoch_bullish", "close_above_vwap")
    g_ind = sum(gains[k]["value"] for k in ind_keys if k in gains)
    g_hot = gains.get("hot_sector_bonus", {"value": 0.0})["value"]  # [2026-10-15] g002 당일 강세 섹터 가산점
    hot_note = (f" + 강세섹터 {g_hot:.1f}" if ko else f" + hot sector {g_hot:.1f}") if g_hot > 0 else ""

    if pen_total < 2.5:
        rank_note = "감점이 거의 없음" if ko else "almost no penalties"
    elif pen_total < 5.0:
        rank_note = "감점이 중간 수준" if ko else "moderate penalties"
    else:
        rank_note = "감점이 큼" if ko else "large penalties"

    # ---- 좋은 조건 / reasons for selection
    good = []
    if row.get("pattern_tier"):
        tier_txt = (
            _pick(lang, ("A: 당일 양봉으로 직전 5일 종가 고점 돌파", "A: bullish candle breaking the prior 5-day closing high"))
            if row["pattern_tier"] == "A" else
            _pick(lang, ("B: 당일 양봉, 최근 5일 음봉 1개 이하, 20일선 +10% 이내",
                         "B: bullish candle, at most 1 bearish candle in 5 days, within +10% of MA20"))
        )
        sigs = row.get("buy_signals") or []
        sig_txt = ", ".join(_pick(lang, BUY_SIGNAL_LABELS[s]) if s in BUY_SIGNAL_LABELS else s for s in sigs)
        if ko:
            _s = row.get("ma120_slope20")
            ma120 = f"120일선 20일 변화 {_pct(_s)}, " if _s is not None else "120일선 이력 부족(미확인), "
            good.append(f"**상승 초입 패턴 {tier_txt}** ({ma120}눌림/바닥 출발) | 매수신호 {len(sigs)}개: {sig_txt}")
        else:
            _s = row.get("ma120_slope20")
            ma120 = f"MA120 20-day change {_pct(_s)}, " if _s is not None else "MA120 history too short (unchecked), "
            good.append(f"**Base-breakout pattern {tier_txt}** ({ma120}started from a pullback/base) | {len(sigs)} buy signals: {sig_txt}")
    atr_full = g_atr >= 15.95
    if ko:
        good.append(f"**변동성(ATR) {atr * 100:.1f}%** → {g_atr:.1f}/16점"
                    + (f" (만점, 기준 {config.atr_ratio_min * 100:.1f}%의 {atr / config.atr_ratio_min:.0f}배)" if atr_full else ""))
        good.append(f"**20일 평균 거래대금 {amt_txt}** → {g_amt:.1f}/18점" + (" (만점)" if g_amt >= 17.95 else " (만점 미달, 상대적 약점)"))
    else:
        good.append(f"**Volatility (ATR) {atr * 100:.1f}%** → {g_atr:.1f}/16"
                    + (f" (full marks, {atr / config.atr_ratio_min:.0f}x the {config.atr_ratio_min * 100:.1f}% minimum)" if atr_full else ""))
        good.append(f"**20-day avg traded value {amt_txt}** → {g_amt:.1f}/18" + (" (full marks)" if g_amt >= 17.95 else " (below full marks, relative weakness)"))
    if rsi is not None and 40.0 <= rsi <= 60.0:
        good.append(f"**RSI {rsi:.1f}** → 8/8" + ("점 (과열 전 최적 구간 40~60)" if ko else " (sweet spot 40-60, before overheating)"))
    elif rsi is not None:
        good.append(f"RSI {rsi:.1f} → {g_rsi:.1f}/8" + ("점" if ko else ""))
    ind_items = [
        ("adx_trend", ("ADX 추세강도", "ADX trend strength"), 4), ("ma_alignment", ("이평 정배열(5>20>60)", "MA alignment (5>20>60)"), 3),
        ("obv_trend", ("OBV 상승", "OBV rising"), 3), ("macd_momentum", ("MACD 히스토그램 증가", "MACD histogram rising"), 2),
        ("stoch_bullish", ("스토캐스틱 상승 교차", "stochastic bullish cross"), 2),
        ("close_above_vwap", ("종가가 당일 VWAP 위", "close above the day's VWAP"), 2),
    ]
    got = [f"{_pick(lang, label)} {gains[k]['value']:.1f}/{mx}" for k, label, mx in ind_items if k in gains and gains[k]["value"] > 0]
    if got:
        good.append(("지표 가점: " if ko else "Indicator points: ") + ", ".join(got))
    a3 = row.get("intraday_atr3_med_pct")  # [2026-10-15] g002 장중 3분봉 ATR 중앙값(%)
    a3_floor = row.get("intraday_atr3_floor_pct")  # 실제 적용된 선정 하한(필터 꺼짐/해제면 None)
    if a3 is not None:
        if a3_floor is not None:
            good.append(f"장중 3분봉 ATR 중앙값 {a3:.2f}% (선정 하한 {a3_floor:.2f}% 통과, r003 진입 기준 0.60%)" if ko
                        else f"Intraday 3-min ATR median {a3:.2f}% (passed the {a3_floor:.2f}% scanner floor; r003 entry gate 0.60%)")
        else:
            good.append(f"장중 3분봉 ATR 중앙값 {a3:.2f}% (선정 하한 미적용, r003 진입 기준 0.60%)" if ko
                        else f"Intraday 3-min ATR median {a3:.2f}% (scanner floor not applied; r003 entry gate 0.60%)")
    if g_hot > 0:
        sdr = row.get("sector_day_return")
        sdr_txt = _pct(sdr) if sdr is not None else "-"
        good.append(f"**당일 강세 섹터** ({row.get('sector_detail')}, 업종 평균 {sdr_txt}) → 가산점 +{g_hot:.1f}점" if ko
                    else f"**Hot sector of the day** ({row.get('sector_detail')}, sector average {sdr_txt}) → bonus +{g_hot:.1f}")
    if 0 <= cm <= 0.08 and rt <= 0.25:
        good.append(f"20일선 위 {_pct(cm)}, 20일 수익률 {_pct(rt)}: 너무 오르지 않아 **과열 감점 없음**" if ko
                    else f"{_pct(cm)} above MA20, 20-day return {_pct(rt)}: not stretched, **no overheating penalty**")
    elif -0.08 <= cm < 0:
        good.append(f"20일선 대비 {_pct(cm)}의 눌림 위치: 과열 감점 없음" if ko
                    else f"Pullback position at {_pct(cm)} vs MA20: no overheating penalty")
    if up5 is not None and up5 >= 4:
        good.append(f"최근 5일 중 {up5}일 상승: 단기 매수세 우위" if ko else f"{up5} up days in the last 5: short-term buyers in control")
    if feat["last_vol_x"] is not None and feat["last_vol_x"] >= 1.5:
        good.append(f"당일 거래량이 직전 5일 평균의 {feat['last_vol_x']:.1f}배: 상승에 수급이 실렸다" if ko
                    else f"Day volume {feat['last_vol_x']:.1f}x the prior 5-day average: the move had volume behind it")
    if vtr is not None and vtr >= 1.3:
        good.append(f"최근 5일 거래량이 직전 5일의 {vtr:.1f}배로 증가 (점수 반영 없음)" if ko
                    else f"Last-5-day volume {vtr:.1f}x the prior 5 days (not scored)")
    if trend != "up" and feat["reversals"]:
        good.append(f"최근 5일 캔들 {candles}: 마지막에 반전 신호(**{rev}**) 발생" if ko
                    else f"Last 5 candles {candles}: reversal signal at the end (**{rev}**)")
    if 0.4 <= high <= 0.7:
        good.append(f"52주 고가의 {high * 100:.0f}% 수준: 위쪽 매물대까지 여유" if ko
                    else f"At {high * 100:.0f}% of the 52-week high: room before overhead supply")
    if 0 < pr < 0.05 and not feat["candles"].endswith("D"):
        good.append(f"당일 {_pct(pr)} 상승 마감" if ko else f"Closed {_pct(pr)} on the day")

    # ---- 감점 요인 (채점 내역 그대로) / penalties
    bad = []
    pt = "점" if ko else ""
    if "overheat_ma20_gap" in pens:
        bad.append((f"20일선 이격 {_pct(cm)} (8% 초과)" if ko else f"{_pct(cm)} above MA20 (over 8%)") + f" → -{pens['overheat_ma20_gap']['value']:.1f}{pt}")
    if "overheat_ret_20d" in pens:
        bad.append((f"20일 수익률 {_pct(rt)} (25% 초과)" if ko else f"20-day return {_pct(rt)} (over 25%)") + f" → -{pens['overheat_ret_20d']['value']:.1f}{pt}")
    if "overheat_prev_day_jump" in pens:
        bad.append((f"당일 급등 {_pct(pr)} (8% 초과)" if ko else f"day jump {_pct(pr)} (over 8%)") + f" → -{pens['overheat_prev_day_jump']['value']:.1f}{pt}")
    if "last_bearish" in pens:
        bad.append((f"당일 음봉 마감 (전일 대비 {_pct(pr)}, 시가보다 낮게 끝남)" if ko else f"bearish close (day {_pct(pr)}, closed below the open)") + f" → -4.0{pt}")
    if "near_52w_high" in pens:
        bad.append((f"52주 고가의 {high * 100:.0f}% (신고가 과열 구간)" if ko else f"{high * 100:.0f}% of the 52-week high (near-high zone)") + f" → -{pens['near_52w_high']['value']:.1f}{pt}")
    if "prev_day_change_gap" in pens:
        bad.append(("전일 등락 과대(갭 리스크)" if ko else "excessive previous-day move (gap risk)") + f" → -{pens['prev_day_change_gap']['value']:.1f}{pt}")
    if "recent_pick_repeat" in pens:
        bad.append(
            (f"최근 {config.recent_pick_penalty_lookback_days}일 내 {row.get('repeat_recent_days')}회 기선정(반복 선정)" if ko
             else f"picked {row.get('repeat_recent_days')} time(s) in the last {config.recent_pick_penalty_lookback_days} days (repeat pick)")
            + f" → -{pens['recent_pick_repeat']['value']:.0f}{pt}"
        )
    if rsi is not None and g_rsi < 7.99:
        side = ("60 초과" if rsi > 60 else "40 미만") if ko else ("above 60" if rsi > 60 else "below 40")
        bad.append(f"RSI {rsi:.1f}: 최적 구간({side})을 벗어나 RSI 점수 {8 - g_rsi:.1f}점 손실" if ko
                   else f"RSI {rsi:.1f}: outside the sweet spot ({side}), {8 - g_rsi:.1f} RSI points lost")
    if g_amt < 17.95:
        bad.append(f"거래대금이 시장 기준 대비 부족해 거래대금 점수 {18 - g_amt:.1f}점 손실" if ko
                   else f"traded value short of the market benchmark: {18 - g_amt:.1f} value points lost")
    if not atr_full:
        bad.append(f"변동성이 상대적으로 작아 ATR 점수 {16 - g_atr:.1f}점 손실" if ko
                   else f"relatively low volatility: {16 - g_atr:.1f} ATR points lost")
    for flag in scored_flags:
        weight = 1.5 if flag.endswith("_reversal_override") else 0.7
        bad.append(f"{_soft_flag_label(flag, row, lang)} → -{weight}{pt}")

    # ---- 점수에 반영 안 된 주의점 / unscored risks
    risk = []
    if pr >= 0.05:
        risk.append(f"당일 {_pct(pr)} 급등 마감 → 다음 날 갭/차익 매물 주의" if ko
                    else f"closed up {_pct(pr)} → watch for a gap / profit-taking next day")
    if feat["last_vol_x"] is not None and feat["last_vol_x"] < 0.5 and trend != "up":
        risk.append(f"반등일 거래량이 직전 5일 평균의 {feat['last_vol_x']:.1f}배로 적어 반전 신뢰도가 낮음" if ko
                    else f"rebound-day volume only {feat['last_vol_x']:.1f}x the prior 5-day average: weak reversal")
    if vtr is not None and vtr < 0.4:
        risk.append(f"최근 5일 거래량이 직전 5일의 {vtr:.2f}배로 급감 (관심 식는 중)" if ko
                    else f"last-5-day volume collapsed to {vtr:.2f}x the prior 5 days (interest fading)")
    if feat["candles"].count("D") >= 4:
        if ko:
            gap_fade = " (갭 상승 후 밀리는 패턴)" if feat["reversals"] and feat["candles"].endswith("D") else ""
            risk.append(f"최근 5일 캔들 {candles}: 음봉 위주{gap_fade}")
        else:
            gap_fade = " (gap up then fade)" if feat["reversals"] and feat["candles"].endswith("D") else ""
            risk.append(f"last 5 candles {candles}: mostly bearish{gap_fade}")
    if high >= 0.9:
        risk.append(f"52주 고가의 {high * 100:.0f}%: 신고가 부근 매물" if ko else f"{high * 100:.0f}% of the 52-week high: supply near the high")
    for flag in later_flags:
        if flag == "sector_cap_swapped_in":
            risk.append("업종 쏠림 제한으로 다른 종목 대신 교체 편입됨" if ko else "swapped in by the sector cap")
        elif flag == "fallback_selected":
            if row.get("pattern_tier"):
                risk.append("적격 종목 부족으로 보충(fallback) 편입됨" if ko else "added as fallback (not enough eligible stocks)")
            else:
                risk.append("상승 초입 패턴 미충족 - 패턴 종목 부족으로 보충(fallback) 편입됨(MA20 이격 +10% 이내)" if ko
                            else "not a base-breakout pattern - added as fallback (MA20 gap within +10%)")
        else:
            risk.append(_soft_flag_label(flag, row, lang))

    # ---- 필터 통과 근거 / filters passed
    hf = []
    if row.get("pattern_tier"):
        hf.append("상승 초입 패턴 필수조건(120일선 20일 변화율 -3% 초과, 장기선·20일선 위, 눌림/바닥 출발, 과열 아님, 매수신호 2개 이상)을 모두 충족했다."
                  if ko else
                  "Met every base-breakout requirement (MA120 20-day change above -3%, above long MAs and MA20, started from a pullback/base, "
                  "not overheated, 2+ buy signals).")
    if trend == "up":
        hf.append("MA5 > MA20 **상승 추세**로 추세 필터를 정상 통과했다." if ko else "Passed the trend filter with **MA5 > MA20 (uptrend)**.")
    elif f"{trend}_trend_pattern_override" in scored_flags:
        hf.append(f"MA5/MA20 기준 **{TREND['ko'].get(trend, trend)} 추세**지만 상승 초입 패턴 종목이라 경고로만 반영했다 (-0.7점)."
                  if ko else
                  f"By MA5/MA20 this is a **{TREND['en'].get(trend, trend)} trend**, kept as a warning only for a pattern stock (-0.7).")
    else:
        if ko:
            hf.append(f"MA5/MA20 기준 **{TREND['ko'].get(trend, trend)} 추세**라 원래는 탈락 대상이었으나, "
                      f"최근 일봉의 반전 신호(**{rev}**)가 인정돼 예외로 통과했다 (-1.5점).")
        else:
            hf.append(f"By MA5/MA20 this is a **{TREND['en'].get(trend, trend)} trend** and would normally fail, "
                      f"but a recent reversal signal (**{rev}**) granted an exception (-1.5).")
    if "box_range_no_progress_reversal_override" in scored_flags and feat["box_range"] is not None:
        hf.append(f"최근 10일 고저폭이 {feat['box_range'] * 100:.1f}%인데 순변동은 {_pct(feat['box_net'])}에 그쳐 "
                  "'박스권 정체'에 해당했지만, 같은 반전 신호로 예외 통과했다 (-1.5점)." if ko else
                  f"10-day range {feat['box_range'] * 100:.1f}% with net change only {_pct(feat['box_net'])} "
                  "('box-range stall'), passed by the same reversal exception (-1.5).")
    if "bearish_candle_dominant_reversal_override" in scored_flags and feat["bear_count"] is not None:
        hf.append(f"최근 10일 중 음봉이 {feat['bear_count']}개(60% 이상)로 '음봉 우세'에 해당했지만, 반전 신호로 예외 통과했다 (-1.5점)."
                  if ko else
                  f"{feat['bear_count']} bearish candles in 10 days (60%+, 'bearish dominance'), passed by a reversal exception (-1.5).")
    if ko:
        hf.append(f"가격·유동성(시장 평균 대비)·변동성 하한, 52주 고가 {config.max_52w_high_ratio * 100:.0f}% 미만"
                  f"(현재 {high * 100:.0f}%), 전일 등락 {config.max_prev_day_change * 100:.0f}% 미만, 관리/정지 아님 조건을 모두 충족했다.")
    else:
        hf.append(f"Met the price, liquidity (vs market), and volatility floors, below {config.max_52w_high_ratio * 100:.0f}% of the "
                  f"52-week high (now {high * 100:.0f}%), previous-day move under {config.max_prev_day_change * 100:.0f}%, "
                  "and not under administrative issue or halt.")

    score = _num(row.get("score"))
    rsi_txt = "-" if rsi is None else f"{rsi:.1f}"
    if ko:
        head = f"### {rank}위. {row.get('name')} ({row.get('code')}) — {score:.2f}점"
        meta = [
            f"- **유형**: {_classify(row, lang)}",
            (f"- **핵심 수치**: 종가 {price:,.0f}원 (당일 {_pct(pr)}) · ATR {atr * 100:.1f}% · 거래대금 {amt_txt} · RSI {rsi_txt} · "
             f"20일선 {_pct(cm)} · 20일 {_pct(rt)} · 52주고가 대비 {high * 100:.0f}% · 최근 5일 캔들 {candles or '-'}"),
            (f"- **점수 계산**: 가점 {gain_total:.1f} (ATR {g_atr:.1f} + 거래대금 {g_amt:.1f} + RSI {g_rsi:.1f}"
             f" + 거래량강도 {vol_note} + 지표 {g_ind:.1f}{hot_note}) − 감점 {pen_total:.1f} = **{score:.2f}** → {rank_note}"),
        ]
        titles = ("**✅ 선정된 이유 (좋은 조건)**", "**⛔ 감점 요인**", "- 없음", "**⚠️ 점수에 반영 안 된 주의점**", "**필터 통과 근거**")
    else:
        head = f"### #{rank}. {row.get('name')} ({row.get('code')}) — {score:.2f} pts"
        meta = [
            f"- **Type**: {_classify(row, lang)}",
            (f"- **Key numbers**: close {price:,.0f} KRW (day {_pct(pr)}) · ATR {atr * 100:.1f}% · traded value {amt_txt} · RSI {rsi_txt} · "
             f"vs MA20 {_pct(cm)} · 20-day {_pct(rt)} · {high * 100:.0f}% of 52w high · last 5 candles {candles or '-'}"),
            (f"- **Score**: points {gain_total:.1f} (ATR {g_atr:.1f} + value {g_amt:.1f} + RSI {g_rsi:.1f}"
             f" + volume strength {vol_note} + indicators {g_ind:.1f}{hot_note}) − penalties {pen_total:.1f} = **{score:.2f}** → {rank_note}"),
        ]
        titles = ("**✅ Why it was picked**", "**⛔ Penalties**", "- none", "**⚠️ Risks not reflected in the score**", "**Filters passed**")

    lines = [head, "", *meta, "", titles[0], "", *[f"- {g}" for g in good], "", titles[1], "",
             *([f"- {b}" for b in bad] or [titles[2]])]
    if risk:
        lines += ["", titles[3], "", *[f"- {r}" for r in risk]]
    lines += ["", titles[4], "", *[f"- {h}" for h in hf], "", "---", ""]
    return "\n".join(lines)


def _render_counts(summary: dict, lang: str) -> list[str]:
    """g002 _scanner_report.md의 Filter Counts / Soft Flags를 뜻풀이와 함께 렌더링."""
    ko = lang == "ko"
    out: list[str] = []
    fail_counts = summary.get("fail_counts") or {}
    soft_counts = summary.get("soft_counts") or {}
    if fail_counts:
        if ko:
            out += ["", "## 탈락 사유별 종목 수 (Filter Counts)", "",
                    "> 하드필터에 걸려 선정에서 제외된 종목 수. 한 종목이 여러 사유에 동시에 걸릴 수 있어 합계는 전체 종목 수보다 크다.",
                    "", "| 탈락 사유 | 종목 수 | 코드 |", "| --- | ---: | --- |"]
        else:
            out += ["", "## Filter Counts (reasons for exclusion)", "",
                    "> Number of stocks excluded by each hard filter. A stock can fail several filters, so the total exceeds the universe size.",
                    "", "| reason | meaning | count |", "| --- | --- | ---: |"]
        for reason, count in fail_counts.items():
            out.append(f"| {_flag_desc(reason, lang)} | {count:,} | `{reason}` |" if ko
                       else f"| {reason} | {_flag_desc(reason, lang)} | {count:,} |")
    if soft_counts:
        if ko:
            out += ["", "## 주의 플래그별 종목 수 (Soft Flags)", "",
                    "> 탈락시키지는 않고 점수만 깎는 경고(개당 -0.7점, 반전 예외 통과는 -1.5점). 전체 스캔 종목 기준 집계.",
                    "", "| 주의 플래그 | 종목 수 | 코드 |", "| --- | ---: | --- |"]
        else:
            out += ["", "## Soft Flags (warnings)", "",
                    "> Warnings that only reduce the score (-0.7 each, -1.5 for a reversal exception). Counted over all scanned stocks.",
                    "", "| flag | meaning | count |", "| --- | --- | ---: |"]
        for flag, count in soft_counts.items():
            out.append(f"| {_flag_desc(flag, lang)} | {count:,} | `{flag}` |" if ko
                       else f"| {flag} | {_flag_desc(flag, lang)} | {count:,} |")
    return out


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
    lang: str = "ko",
) -> str:
    ko = lang == "ko"
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

    if ko:
        out = [
            f"# 종목 선별 리포트 (종목별 선정 이유) — {target_label}",
            "",
            f"> g002 스캐너(`{config.name}`) 최종 선정 {len(selected_rows)}종목의 선정 이유. "
            f"적격 {summary.get('eligible_pool_count', '-')}종목 중 패턴 등급 → 매수신호 개수 → 점수 순으로 선정.",
            "> 점수 내역은 실제 채점(`calculate_candidate_score_breakdown`) 결과를 그대로 옮겼다.",
            "",
            "## 읽는 법 — 선정 방식",
            "",
            "**1) 필수 조건: 상승 초입 패턴** (모두 만족해야 선정 대상)",
            "",
            "- 120일선 20일 변화율 -3% 초과(장기 하락 아님), 종가가 60·120일선 중 낮은 쪽 이상, 종가가 20일선 위",
            "- 20일선이 5일 동안 1% 넘게 떨어지지 않음, 직전 10일 중 종가가 20일선 +5% 이내였던 날이 있음(눌림/바닥 출발)",
            "- 20일선 대비 +15% 이하, RSI 70 이하(과열 아님), 매수신호 2개 이상"
            "(MACD 상승전환, 스토캐스틱 골든크로스, DI+ 우위, RSI 50~68, 거래량 1.3배+, OBV 상승)",
            "- **A등급**: 당일 양봉이 직전 5일 종가 고점 돌파 / **B등급**: 당일 양봉, 최근 5일 음봉 1개 이하, 20일선 +10% 이내",
            "- 거래대금이 시장 기준에 못 미쳐도 패턴 종목은 20일 평균 거래대금 50억 이상이면 통과",
            "",
            "**2) 점수** (같은 등급·신호 개수 안에서 순위를 가름)",
            "",
            "| 구분 | 항목 | 배점 | 만점 조건 |",
            "| --- | --- | ---: | --- |",
            f"| 가점 | 변동성 ATR(일평균 변동폭/종가) | 16 | 약 {config.atr_ratio_min * 4.4 * 100:.1f}% 이상 |",
            "| 가점 | 20일 평균 거래대금(시장 기준 대비) | 18 | 시장 기준의 약 4배 이상 |",
            "| 가점 | RSI(10일) | 8 | 40~60 구간 |",
            "| 가점 | 거래량 상대강도(최근5일/이전20일) | 10 | 일봉 25개 필요 |",
            "| 가점 | 일봉 지표 | 16 | ADX 추세강도 4, 이평 정배열 3, OBV 상승 3, MACD 히스토그램 증가 2, 스토캐스틱 상승 교차 2, 종가>VWAP 2 |",
            "| 감점 | 과열 | 최대 -12/-8/-6 | 20일선 이격 8% 초과 / 20일 상승률 25% 초과 / 당일 8% 초과 급등 |",
            "| 감점 | 당일 음봉 | -4 | |",
            f"| 감점 | 최근 {config.recent_pick_penalty_lookback_days}일 내 반복 선정 | 회당 -{config.recent_pick_penalty_per_day:.0f} | |",
            "| 감점 | 주의 플래그 | 개당 -0.7 / -1.5 | 추세·박스권·음봉 필터를 '반전 신호'로 예외 통과하면 -1.5 |",
            "",
            "- 반전 신호 = ① 2일 연속 양봉, ② 2일 연속 종가 상승, ③ 거래량 1.5배 이상 + 3% 이상 양봉 + "
            "전일 고가 돌파(V자 장대양봉) 중 하나 이상",
            "- 캔들 표기: 양 = 양봉, 음 = 음봉, 보 = 보합 (왼쪽이 5일 전, 오른쪽이 기준일)",
        ]
        if vol_missing:
            out.append(f"- ⚠️ 거래량 상대강도는 {vol_missing}/{len(selected_rows)}종목이 일봉 부족(25개 필요)으로 계산되지 않아 0점 처리됐다.")
        out += [
            "", "## 순위표", "",
            "| 순위 | 종목 | 등급 | 신호 | 점수 | 종가 | 당일 | ATR% | 거래대금(20일평균) | RSI | 20일선 이격 | 20일 수익률 | 52주고가 대비 | 추세 | 감점 |",
            "| ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
        ]
    else:
        out = [
            f"# Stock Selection Report (why each stock was picked) — {target_label}",
            "",
            f"> Reasons for the {len(selected_rows)} final picks of the g002 scanner (`{config.name}`), selected from "
            f"{summary.get('eligible_pool_count', '-')} eligible stocks by pattern tier → number of buy signals → score.",
            "> Score details are copied from the actual scoring (`calculate_candidate_score_breakdown`).",
            "",
            "## How to read — selection method",
            "",
            "**1) Requirement: base-breakout (early-uptrend) pattern** (all must hold)",
            "",
            "- MA120 20-day change above -3% (not in a long-term decline), close above the lower of MA60/MA120, close above MA20",
            "- MA20 has not fallen more than 1% over 5 days; in the prior 10 days the close was within +5% of MA20 at least once (started from a pullback/base)",
            "- At most +15% above MA20 and RSI at most 70 (not overheated); 2+ buy signals "
            "(MACD turning up, stochastic golden cross, DI+ above DI-, RSI 50-68, volume 1.3x+, OBV rising)",
            "- **Tier A**: bullish candle breaking the prior 5-day closing high / **Tier B**: bullish candle, at most 1 bearish candle in 5 days, within +10% of MA20",
            "- Pattern stocks below the market liquidity benchmark still pass with a 20-day average traded value of at least 5bn KRW",
            "",
            "**2) Score** (ranks stocks within the same tier and signal count)",
            "",
            "| type | item | points | full-marks condition |",
            "| --- | --- | ---: | --- |",
            f"| plus | volatility ATR (avg daily range / close) | 16 | about {config.atr_ratio_min * 4.4 * 100:.1f}% or more |",
            "| plus | 20-day avg traded value (vs market benchmark) | 18 | about 4x the benchmark |",
            "| plus | RSI (10-day) | 8 | 40-60 |",
            "| plus | relative volume (last 5 days / prior 20) | 10 | needs 25 daily bars |",
            "| plus | daily indicators | 16 | ADX trend 4, MA alignment 3, OBV rising 3, MACD histogram rising 2, stochastic bullish cross 2, close > VWAP 2 |",
            "| minus | overheating | up to -12/-8/-6 | over 8% above MA20 / 20-day gain over 25% / day jump over 8% |",
            "| minus | bearish close | -4 | |",
            f"| minus | picked in the last {config.recent_pick_penalty_lookback_days} days | -{config.recent_pick_penalty_per_day:.0f} each | |",
            "| minus | soft flags | -0.7 / -1.5 each | -1.5 when a trend/box/bearish filter was waived by a reversal signal |",
            "",
            "- Reversal signal = (1) two consecutive bullish candles, (2) two consecutive higher closes, or (3) volume 1.5x+ with a 3%+ "
            "bullish candle breaking the previous high (V-shaped large bullish candle)",
            "- Candles: U = bullish, D = bearish, = = flat (left is 5 days ago, right is the scan date)",
        ]
        if vol_missing:
            out.append(f"- ⚠️ Relative volume could not be computed for {vol_missing}/{len(selected_rows)} stocks (needs 25 daily bars) and scored 0.")
        out += [
            "", "## Ranking", "",
            "| rank | stock | tier | signals | score | close | day | ATR% | traded value (20d avg) | RSI | vs MA20 | 20-day return | vs 52w high | trend | penalties |",
            "| ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
        ]
    for rank, row in enumerate(selected_rows, start=1):
        bd = row.get("score_breakdown") or {"penalties": []}
        pen_total = sum(p["value"] for p in bd["penalties"])
        rsi = row.get("rsi")
        amt = _num(row.get("amount_ma20")) / 1e8
        amt_txt = f"{amt:,.0f}억" if ko else f"{amt / 10:,.1f}bn"
        out.append(
            f"| {rank} | {row.get('name')}({row.get('code')}) | {row.get('pattern_tier') or '-'} | {row.get('buy_signal_count') or 0} | "
            f"{_num(row.get('score')):.2f} | {_num(row.get('price')):,.0f} | {_pct(row.get('prev_day_return'))} | "
            f"{_num(row.get('atr_ratio')) * 100:.1f}% | {amt_txt} | {'-' if rsi is None else f'{rsi:.1f}'} | "
            f"{_pct(row.get('close_ma20_ratio'))} | {_pct(row.get('ret_20d'))} | {_num(row.get('high_52w_ratio')) * 100:.0f}% | "
            f"{TREND[lang].get(row.get('trend_state'), row.get('trend_state'))} | -{pen_total:.1f} |"
        )
    out += ["", "", "## 종목별 선정 이유" if ko else "## Reasons by stock", ""]
    for rank, (row, feat) in enumerate(zip(selected_rows, feats), start=1):
        out.append(_render_stock(rank, row, feat, config, lang))
    out += _render_counts(summary, lang)
    return "\n".join(out) + "\n"

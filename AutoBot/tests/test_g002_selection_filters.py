# -*- coding: utf-8 -*-
"""g002 selection filters: 2-day close decline exclusion + near-miss fallback (2026-11-01)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_simulation"))
sys.path.insert(0, str(ROOT / "live_trading"))

import g002_data_scan_trade_candidates as g002  # noqa: E402
import g008_selection_reason_report as g008  # noqa: E402


class TestTwoDayCloseDowntrend(unittest.TestCase):
    def test_290550_case(self):
        # 10/27 25,600 -> 10/28 25,200 -> 10/29 24,400
        self.assertTrue(g002._is_2d_close_downtrend([25600, 25200, 24400]))

    def test_only_last_three_closes(self):
        self.assertTrue(g002._is_2d_close_downtrend([20000, 21000, 25600, 25200, 24400]))
        self.assertFalse(g002._is_2d_close_downtrend([26000, 25600, 25200, 25300]))

    def test_not_consecutive(self):
        self.assertFalse(g002._is_2d_close_downtrend([25600, 25200, 25200]))  # flat today
        self.assertFalse(g002._is_2d_close_downtrend([25600, 25700, 24400]))  # one decline only
        self.assertFalse(g002._is_2d_close_downtrend([25200, 25200, 24400]))

    def test_short_or_missing(self):
        self.assertFalse(g002._is_2d_close_downtrend([25200, 24400]))
        self.assertFalse(g002._is_2d_close_downtrend([25600, None, 24400]))
        self.assertFalse(g002._is_2d_close_downtrend([]))

    def test_soft_flag_only(self):
        # 2026-11-11: 하드 제외 -> 소프트플래그 (85거래일 재현에서 D+1 차이 없음). 하드로 되돌리면 구제 경로는 여전히 없어야 한다.
        self.assertFalse(g002.EXCLUDE_2D_CLOSE_DOWNTREND)
        self.assertTrue(g002.SOFT_FLAG_2D_CLOSE_DOWNTREND)
        self.assertNotIn("2d_close_downtrend", g002.PATTERN_OVERRIDABLE_TREND_FAILS)
        self.assertNotIn("2d_close_downtrend", g002.FALLBACK_RELAXABLE_FAIL_REASONS)

    def test_report_soft_label(self):
        self.assertIn("2d_close_downtrend", g008.SOFT_FLAG_DESC)

    def test_near_miss_fallback_reasons(self):
        # 2026-11-01 메리츠형/롯데렌탈형 보충 허용 - 2d_close_downtrend 같은 비패턴 하드필터는 포함하지 않는다
        self.assertEqual(
            set(g002.FALLBACK_NEAR_MISS_PATTERN_FAILS), {"pattern_no_bullish_close", "pattern_bearish_candles"}
        )
        self.assertTrue(all(r.startswith("pattern_") for r in g002.FALLBACK_NEAR_MISS_PATTERN_FAILS))

    def test_report_label(self):
        self.assertIn("2d_close_downtrend", g008.FAIL_REASON_DESC)


def _feat(**kw):
    """Minimal indicator snapshot that passes _pattern_assessment as tier B unless overridden."""
    f = {
        "MA_5": 101.0, "MA_20": 100.0, "MA_60": 95.0, "MA_120": 90.0, "RSI": 55.0, "min_vs20_prev10": 0.01,
        "ma120_slope20": 0.01, "ma20_slope5": 0.005, "ma60_slope20": 0.02, "ret_20d": 0.08,
        "MACD": 1.0, "MACD_SIGNAL": 0.5, "MACD_HIST": 0.5, "macd_hist_prev": 0.3, "macd_x_age": None,
        "STOCH_K": 60.0, "STOCH_D": 55.0, "stoch_x_age": 1, "DI_PLUS": 25.0, "DI_MINUS": 15.0,
        "vol_ratio": 1.0, "obv_rising": False, "bull": True, "brk5": False, "bear5": 1, "close": 104.0, "open": 103.0,
    }
    f.update(kw)
    return f


class TestPatternTuning20261111(unittest.TestCase):
    """2026-11-11 selection tuning (85-day replay 2026-07-06..11-09 + r003 realized trades)."""

    def test_tier_b_allows_two_bearish_candles(self):
        self.assertEqual(g002.PATTERN_MAX_BEAR5, 2)
        fails, tier, _ = g002._pattern_assessment(104.0, _feat(bear5=2))
        self.assertEqual((fails, tier), ([], "B"))
        fails, tier, _ = g002._pattern_assessment(104.0, _feat(bear5=3))
        self.assertEqual((fails, tier), (["pattern_bearish_candles"], None))

    def test_pullback_base_within_8pct(self):
        self.assertAlmostEqual(g002.PATTERN_BASE_MAX_VS_MA20, 0.08)
        self.assertEqual(g002._pattern_assessment(104.0, _feat(min_vs20_prev10=0.07))[1], "B")
        self.assertIn("pattern_no_pullback_base", g002._pattern_assessment(104.0, _feat(min_vs20_prev10=0.09))[0])

    def test_overheated_by_ma60_slope_or_ret20d(self):
        self.assertIn("pattern_overheated", g002._pattern_assessment(104.0, _feat(ma60_slope20=0.07))[0])
        self.assertIn("pattern_overheated", g002._pattern_assessment(104.0, _feat(ret_20d=0.31))[0])
        self.assertEqual(g002._pattern_assessment(104.0, _feat(ma60_slope20=0.06, ret_20d=0.30))[1], "B")
        # missing values (short history) do not fail the pattern
        self.assertEqual(g002._pattern_assessment(104.0, _feat(ma60_slope20=None, ret_20d=None))[1], "B")

    def test_breakout_features_have_new_fields(self):
        import numpy as np
        import pandas as pd
        n = 30
        close = pd.Series(np.linspace(100, 130, n))
        df = pd.DataFrame({"close": close, "open": close - 0.5, "volume": 1000.0, "MA_20": close.rolling(20, min_periods=1).mean(),
                           "MA_60": close.rolling(5, min_periods=1).mean(), "OBV": close.cumsum(), "OBV_MA": close.cumsum(),
                           "MACD": 1.0, "MACD_SIGNAL": 0.5, "STOCH_K": 1.0, "STOCH_D": 0.5, "DI_PLUS": 1.0, "DI_MINUS": 0.5, "RSI": 55.0})
        f = g002._base_breakout_features(df)
        self.assertAlmostEqual(f["ret_20d"], float(close.iloc[-1] / close.iloc[-21] - 1.0))
        self.assertAlmostEqual(f["ma60_slope20"], float(df["MA_60"].iloc[-1] / df["MA_60"].iloc[-21] - 1.0))

    def test_rank_key_ignores_signal_count(self):
        self.assertFalse(g002.SELECTION_RANK_BY_SIGNAL_COUNT)
        a = {"pattern_tier_rank": 2, "buy_signal_count": 5, "score": 30.0, "amount_ma20": 1.0, "atr_ratio": 0.01}
        b = {"pattern_tier_rank": 2, "buy_signal_count": 2, "score": 31.0, "amount_ma20": 1.0, "atr_ratio": 0.01}
        self.assertGreater(g002._selection_rank_key(b), g002._selection_rank_key(a))

    def test_fallback_and_liquidity_constants(self):
        self.assertEqual(tuple(g002.FALLBACK_ALLOWED_TREND_STATES), ("up",))
        self.assertEqual(g002.LIQUIDITY_FALLBACK_MIN_AMOUNT, g002.PATTERN_LIQUIDITY_MIN_AMOUNT)
        self.assertEqual(g002.HOT_SECTOR_BONUS_RATIO, 0.0)

    def test_fallback_blocks_overheated(self):
        # Codex 검토(2026-11-11): 과열 상한은 패턴 탈락만 시키고 보충 경로로 되돌아올 수 있었다 -> 보충에서도 차단.
        self.assertIn("pattern_overheated", g002.FALLBACK_BLOCKED_PATTERN_FAILS)
        base = {"close_ma20_ratio": 0.04, "amount_ma20": 3e10}
        self.assertTrue(g002._fallback_relaxable_row({**base, "fail_reasons": ["pattern_no_bullish_close"]}))
        self.assertTrue(g002._fallback_relaxable_row({**base, "fail_reasons": ["low_score", "pattern_few_buy_signals"]}))
        self.assertFalse(g002._fallback_relaxable_row({**base, "fail_reasons": ["pattern_overheated"]}))
        self.assertFalse(g002._fallback_relaxable_row({**base, "fail_reasons": ["pattern_overheated", "liquidity_below_market_dual"]}))
        self.assertFalse(g002._fallback_relaxable_row({**base, "fail_reasons": ["pattern_no_bullish_close"], "close_ma20_ratio": 0.11}))
        self.assertFalse(g002._fallback_relaxable_row({**base, "fail_reasons": ["pattern_unknown"]}))
        self.assertFalse(g002._fallback_relaxable_row({**base, "fail_reasons": ["pattern_no_bullish_close", "down_trend"]}))
        # near-miss: one listed pattern fail + 5bn turnover
        self.assertTrue(g002._fallback_near_miss_row({**base, "fail_reasons": ["pattern_bearish_candles", "low_score"]}))
        self.assertFalse(g002._fallback_near_miss_row({**base, "fail_reasons": ["pattern_overheated"]}))
        self.assertFalse(g002._fallback_near_miss_row({**base, "fail_reasons": ["pattern_bearish_candles"], "amount_ma20": 4e9}))

    def test_breakout_feature_boundaries_with_real_indicators(self):
        import numpy as np
        import pandas as pd

        def frame(n):
            close = pd.Series(100.0 + np.arange(n) * 0.5)
            df = pd.DataFrame({"open": close - 0.2, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0})
            return g002._compute_daily_indicators(df)

        f20 = g002._base_breakout_features(frame(20))
        self.assertIsNone(f20["ret_20d"])           # needs 21 closes
        self.assertIsNone(f20["ma60_slope20"])      # MA_60 NaN below 60 rows
        f21 = g002._base_breakout_features(frame(21))
        self.assertAlmostEqual(f21["ret_20d"], 110.0 / 100.0 - 1.0)
        self.assertIsNone(f21["ma60_slope20"])
        f79 = g002._base_breakout_features(frame(79))
        self.assertIsNone(f79["ma60_slope20"])      # MA_60[-21] is NaN until 80 rows
        f80 = g002._base_breakout_features(frame(80))
        self.assertIsNotNone(f80["ma60_slope20"])
        self.assertGreater(f80["ma60_slope20"], 0)
        # caps disabled -> never overheated by these features
        old = (g002.PATTERN_MAX_MA60_SLOPE20, g002.PATTERN_MAX_RET_20D)
        try:
            g002.PATTERN_MAX_MA60_SLOPE20, g002.PATTERN_MAX_RET_20D = None, None
            self.assertEqual(g002._pattern_assessment(104.0, _feat(ma60_slope20=0.5, ret_20d=0.9))[1], "B")
        finally:
            g002.PATTERN_MAX_MA60_SLOPE20, g002.PATTERN_MAX_RET_20D = old

    def test_tie_bucket_ignores_signal_count(self):
        import random
        rows = [{"code": str(i), "pattern_tier_rank": 2, "buy_signal_count": i, "score": 30.0, "amount_ma20": 1.0 + i,
                 "atr_ratio": 0.02, "vol_ma20": 1.0, "trend_state": "up"} for i in range(5)]
        rows.sort(key=g002._selection_rank_key, reverse=True)
        picked = g002.pick_with_tie_randomization(rows, 2, random.Random(1))
        self.assertEqual(len(picked), 2)
        # all five share one (tier, 0, score) bucket -> sampled; tier precedence still wins over score
        top = {"code": "T", "pattern_tier_rank": 2, "buy_signal_count": 0, "score": 10.0, "amount_ma20": 1.0, "atr_ratio": 0.02, "vol_ma20": 1.0}
        low = {"code": "L", "pattern_tier_rank": 1, "buy_signal_count": 6, "score": 99.0, "amount_ma20": 1.0, "atr_ratio": 0.02, "vol_ma20": 1.0}
        self.assertGreater(g002._selection_rank_key(top), g002._selection_rank_key(low))
        old = g002.SELECTION_RANK_BY_SIGNAL_COUNT
        try:
            g002.SELECTION_RANK_BY_SIGNAL_COUNT = True
            self.assertEqual(g002._selection_rank_key(low)[1], 6)
        finally:
            g002.SELECTION_RANK_BY_SIGNAL_COUNT = old

    def test_two_day_decline_is_soft_flag(self):
        # 서버 일봉 3행(연속 하락)으로 evaluate_candidate 실행 - fail_reasons가 아니라 soft_flags에 기록돼야 한다.
        import pandas as pd
        dates = pd.to_datetime(["20261105", "20261106", "20261107"], format="%Y%m%d")
        hist = pd.DataFrame({"date": dates, "open": [100.0, 99.0, 98.0], "high": [101.0, 100.0, 99.0],
                             "low": [99.0, 98.0, 97.0], "close": [100.0, 99.0, 98.0], "volume": [1e6] * 3})
        hist["amount"] = hist["close"] * hist["volume"]
        daily = pd.DataFrame({"open": [98.5], "high": [99.0], "low": [97.0], "close": [98.0], "volume": [1e6]}, index=[dates[-1]])
        daily["amount"] = daily["close"] * daily["volume"]
        cfg = g002.BALANCED_CONFIG.__class__(**{**g002.BALANCED_CONFIG.__dict__, "min_listing_days": 1})
        cand = g002.evaluate_candidate("000001", "테스트", daily, cfg, daily_hist=hist)
        self.assertIn("2d_close_downtrend", cand["soft_flags"])
        self.assertNotIn("2d_close_downtrend", cand["fail_reasons"])


if __name__ == "__main__":
    unittest.main()

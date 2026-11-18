# -*- coding: utf-8 -*-
"""[2026-11-18] r002 clamp_cross_age_to_regular_open (r001 TRIPLE_1MIN_AGE_FROM_REGULAR_OPEN, r005 _008 / g003 parity).
Case: 214450 파마리서치 20261118 - 1-minute MA5>MA20 cross at 08:47 in the NXT pre-market, ABOVE_TOO_OLD (902 s > 900 s) at 09:02:20."""
from __future__ import annotations

import sys
import unittest
from datetime import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

from r002_strategy_core_shared import clamp_cross_age_to_regular_open as clamp, evaluate_triple_trigger  # noqa: E402

OPEN = time(9, 0)
T = pd.Timestamp


class TestClamp(unittest.TestCase):
    def test_live_defaults_20261118(self):
        import r001_define_config as r001
        self.assertTrue(r001.TRIPLE_1MIN_AGE_FROM_REGULAR_OPEN)
        self.assertEqual(r001.HYBRID_1MIN_TRIGGER_CANDLE_GAIN_MAX_PCT, 2.5)

    def test_pre_open_same_day_cross_aged_from_open(self):
        self.assertEqual(clamp(902.0, T("2026-11-18 09:02:20"), T("2026-11-18 08:47:18"), OPEN), 140.0)
        self.assertEqual(clamp(2300.0, T("2026-11-18 09:30:00"), T("2026-11-18 08:47:40"), OPEN), 1800.0)

    def test_prior_day_cross_keeps_real_age(self):
        age = (T("2026-11-18 09:02:20") - T("2026-11-17 15:10:00")).total_seconds()
        self.assertEqual(clamp(age, T("2026-11-18 09:02:20"), T("2026-11-17 15:10:00"), OPEN), age)

    def test_after_open_cross_and_pre_open_now_unchanged(self):
        self.assertEqual(clamp(300.0, T("2026-11-18 09:10:00"), T("2026-11-18 09:05:00"), OPEN), 300.0)
        self.assertEqual(clamp(120.0, T("2026-11-18 08:49:20"), T("2026-11-18 08:47:20"), OPEN), 120.0)

    def test_nan_and_none(self):
        self.assertTrue(pd.isna(clamp(float("nan"), T("2026-11-18 09:02"), T("2026-11-18 08:47"), OPEN)))
        self.assertEqual(clamp(500.0, T("2026-11-18 09:02"), None, OPEN), 500.0)

    def test_triple_above_too_old_flips(self):
        ma = {"ok": True, "cross_bar_time": T("2026-11-18 08:47:18"), "bars_since_cross": 15}
        crosses = {60: type("C", (), {"ok": True, "cross_time": T("2026-11-18 09:01:12")})()}
        now, cbt = T("2026-11-18 09:02:20"), ma["cross_bar_time"]
        old = evaluate_triple_trigger(ma, (now - cbt).total_seconds(), crosses, (60,), "above", "any", 240.0, 5, 20,
                                      above_max_age_s=900.0)
        self.assertTrue(any("ABOVE_TOO_OLD" in m for m in old["missing"]))
        new = evaluate_triple_trigger(ma, clamp((now - cbt).total_seconds(), now, cbt, OPEN), crosses, (60,), "above", "any",
                                      240.0, 5, 20, above_max_age_s=900.0)
        self.assertFalse(any("ABOVE_TOO_OLD" in m for m in new["missing"]))


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""g002 2-day consecutive close decline exclusion (2026-11-01, 20261029 290550 case)."""
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

    def test_hard_fail_not_rescued(self):
        self.assertTrue(g002.EXCLUDE_2D_CLOSE_DOWNTREND)
        self.assertNotIn("2d_close_downtrend", g002.PATTERN_OVERRIDABLE_TREND_FAILS)
        self.assertNotIn("2d_close_downtrend", g002.FALLBACK_RELAXABLE_FAIL_REASONS)

    def test_report_label(self):
        self.assertIn("2d_close_downtrend", g008.FAIL_REASON_DESC)


if __name__ == "__main__":
    unittest.main()

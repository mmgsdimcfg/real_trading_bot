# -*- coding: utf-8 -*-
"""2026-11-18 user decision: every loss exit at -3.5% (hard stop, post-buy drop guard, ATR stop floor).
Case: 100790 미래에셋벤처투자 20261118 09:18 signal, 14,910 assumed entry, day low 14,410 (-3.35%) at 09:23."""
from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r001_define_config as r001  # noqa: E402
import r006_sell_conditions as r006  # noqa: E402

ENTRY = 14910.0


def _tracked(price: float, atr: float) -> SimpleNamespace:
    ctx = SimpleNamespace(
        cur=pd.Series({"low": price, "ATR": atr}), price=price, pos={"highest_price": ENTRY}, entry_price=ENTRY,
        code="100790", current_dt=datetime(2026, 11, 18, 9, 23), pnl_pct=price / ENTRY - 1,
        state=SimpleNamespace(recent_price_samples={}),
    )
    with mock.patch.object(r006, "ENABLE_SURGE_LADDER_TP", False):
        r006._002_update_position_tracking(ctx)
    return ctx


class TestStopLoss35(unittest.TestCase):
    def test_config_values(self):
        self.assertEqual((r001.HARD_STOP_LOSS_PCT, r001.ATR_STOP_MIN_LOSS_PCT), (0.035, 0.035))
        for name in ("conservative", "neutral", "aggressive"):
            prof = json.loads((ROOT / "live_trading" / "risk_profiles" / f"{name}.json").read_text(encoding="utf-8"))
            self.assertEqual(prof["POST_BUY_BB_DROP_PCT"], 0.035, name)

    @mock.patch.object(r006, "ATR_STOP_MIN_LOSS_PCT", 0.035)
    @mock.patch.object(r006, "ATR_STOP_MULTIPLIER", 1.5)
    def test_atr_stop_never_closer_than_floor(self):
        ctx = _tracked(14410.0, atr=14410.0 * 0.006)  # ATR 0.6% x 1.5 = -0.9% -> floored to -3.5%
        self.assertAlmostEqual(ctx.atr_sl_pct, -0.035)
        self.assertAlmostEqual(ctx.atr_sl_price, ENTRY * 0.965)
        self.assertGreater(ctx.pnl_sl, ctx.atr_sl_pct)  # 100790 day low -3.35% does not reach the stop

    @mock.patch.object(r006, "ATR_STOP_MIN_LOSS_PCT", 0.035)
    @mock.patch.object(r006, "ATR_STOP_MULTIPLIER", 1.5)
    def test_wide_atr_kept(self):
        ctx = _tracked(14410.0, atr=14410.0 * 0.03)  # 3% x 1.5 = -4.5% stays
        self.assertAlmostEqual(ctx.atr_sl_pct, -0.045)

    @mock.patch.object(r006, "ATR_STOP_MIN_LOSS_PCT", 0.0)
    @mock.patch.object(r006, "ATR_STOP_MULTIPLIER", 1.5)
    def test_zero_floor_is_previous_behaviour(self):
        ctx = _tracked(14410.0, atr=14410.0 * 0.006)
        self.assertAlmostEqual(ctx.atr_sl_pct, -0.009)


if __name__ == "__main__":
    unittest.main()

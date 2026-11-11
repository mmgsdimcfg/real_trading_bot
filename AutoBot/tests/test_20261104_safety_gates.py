"""[2026-11-04] Codex safety items (partial-fill monitoring/cancel, unknown sell submission, account-sync freshness,
quote status gate, unrealized daily loss) and the index-ETF market regime gate."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r001_define_config as r001  # noqa: E402
import r002_strategy_core_shared as r002  # noqa: E402
import r005_buy_conditions as r005  # noqa: E402

try:
    import r003_trade_live_execute as r003  # noqa: E402
except Exception:  # broker SDK missing
    r003 = None


def _frame(closes):
    idx = pd.date_range("2026-11-04 09:01", periods=len(closes), freq="1min")
    return pd.DataFrame({"close": closes}, index=idx, dtype=float)


class TestMarketRegime(unittest.TestCase):
    def test_market_regime_ok(self):
        up = [100 + i for i in range(30)]
        self.assertEqual(r002.market_regime_ok(_frame(up), 5, 20)[0], True)
        down = [130 - i for i in range(30)]
        ok, detail = r002.market_regime_ok(_frame(down), 5, 20)
        self.assertFalse(ok)
        self.assertIn("NOT_ABOVE", detail)
        self.assertIsNone(r002.market_regime_ok(_frame(up[:10]), 5, 20)[0])
        self.assertIsNone(r002.market_regime_ok(None, 5, 20)[0])

    def test_market_regime_confirm_bars_tolerates_a_flicker(self):
        # [2026-11-11] an uptrend whose last bar alone pulls MA5 under MA20: blocked with confirm_bars=1, tolerated with 3,
        # blocked again once three trailing bars are below
        up = [100 + i for i in range(30)]
        flick = up + [60]
        self.assertFalse(r002.market_regime_ok(_frame(flick), 5, 20, 1)[0])
        ok, detail = r002.market_regime_ok(_frame(flick), 5, 20, 3)
        self.assertTrue(ok)
        self.assertIn("FLICKER_1bar_LT_3", detail)
        ok, detail = r002.market_regime_ok(_frame(up + [60, 60, 60]), 5, 20, 3)
        self.assertFalse(ok)
        self.assertIn("NOT_ABOVE", detail)
        self.assertEqual(r002.market_regime_ok(_frame(up), 5, 20, 3)[0], True)

    def test_load_symbol_market_map(self):
        d = tempfile.mkdtemp()
        p = Path(d) / "m.txt"
        p.write_text("code,name,market\n000020,동화약품,KOSPI\n# 000040,x,KOSPI  # auto\n159010,아스플로,KOSDAQ\nbad line\n", encoding="utf-8")
        self.assertEqual(r002.load_symbol_market_map(p), {"000020": "KOSPI", "159010": "KOSDAQ"})
        self.assertEqual(r002.load_symbol_market_map(Path(d) / "missing.txt"), {})

    def _ctx(self, regime_fn):
        logs = []
        return SimpleNamespace(norm_code="159010", code="159010", symbol_label="159010_x", log=logs.append,
                               state=SimpleNamespace(buy_confirm_state={"159010": {"count": 1}}),
                               services=SimpleNamespace(market_regime=regime_fn)), logs

    def test_r005_029(self):
        with mock.patch.object(r005, "ENABLE_MARKET_REGIME_GATE", True):
            ctx, logs = self._ctx(lambda c: (False, "KOSDAQ_229200_MA5_NOT_ABOVE_MA20"))
            self.assertFalse(r005._010_market_regime(ctx))
            self.assertIn("MARKET_REGIME_DOWN_KOSDAQ_229200_MA5_NOT_ABOVE_MA20", logs[-1])
            self.assertNotIn("159010", ctx.state.buy_confirm_state)
            ctx, _ = self._ctx(lambda c: (True, "ok"))
            self.assertTrue(r005._010_market_regime(ctx))
            with mock.patch.object(r005, "MARKET_REGIME_FAIL_OPEN", True):
                self.assertTrue(r005._010_market_regime(self._ctx(lambda c: (None, "no"))[0]))
            with mock.patch.object(r005, "MARKET_REGIME_FAIL_OPEN", False):
                self.assertFalse(r005._010_market_regime(self._ctx(lambda c: (None, "no"))[0]))
            self.assertTrue(r005._010_market_regime(self._ctx(None)[0]))
            self.assertTrue(r005._010_market_regime(self._ctx(lambda c: 1 / 0)[0]))  # exception -> fail-open
        with mock.patch.object(r005, "ENABLE_MARKET_REGIME_GATE", False):
            self.assertTrue(r005._010_market_regime(self._ctx(lambda c: (False, "x"))[0]))


class TestSafetyConditions(unittest.TestCase):
    def _ctx(self, **services):
        logs = []
        ctx = SimpleNamespace(norm_code="000001", code="000001", symbol_label="X", log=logs.append,
                              current_dt=datetime(2026, 11, 4, 10, 0, 0),
                              state=SimpleNamespace(buy_confirm_state={"000001": {"count": 1}}, traded_today=set()),
                              services=SimpleNamespace(**{"account_sync_age": None, "quote_status": None, **services}),
                              api=SimpleNamespace(live_state={"realized_pnl_krw": 0.0}, positions={}))
        return ctx, logs

    def test_030_account_sync_fresh(self):
        with mock.patch.object(r005, "ACCOUNT_SYNC_MAX_STALE_SECONDS", 600.0):
            self.assertTrue(r005._019_account_sync_fresh(self._ctx(account_sync_age=lambda: 30.0)[0]))
            ctx, logs = self._ctx(account_sync_age=lambda: 601.0)
            self.assertFalse(r005._019_account_sync_fresh(ctx))
            self.assertIn("ACCOUNT_SYNC_STALE", logs[-1])
            self.assertFalse(r005._019_account_sync_fresh(self._ctx(account_sync_age=lambda: None)[0]))
            self.assertTrue(r005._019_account_sync_fresh(self._ctx()[0]))  # not injected -> pass

    def test_031_quote_status(self):
        now = datetime(2026, 11, 4, 10, 0, 0)
        def st(**kw):
            base = {"vi_cls_code": "N", "mrkt_warn_cls_code": "00", "temp_stop_yn": "N", "iscd_stat_cls_code": "55", "at": now}
            base.update(kw); return base
        with mock.patch.object(r005, "ENABLE_QUOTE_STATUS_GATE", True):
            self.assertTrue(r005._018_quote_status(self._ctx(quote_status=lambda c: st())[0]))
            self.assertTrue(r005._018_quote_status(self._ctx(quote_status=lambda c: st(iscd_stat_cls_code="57"))[0]))
            for kw, tag in ((dict(vi_cls_code="Y"), "VI_ACTIVE"), (dict(temp_stop_yn="Y"), "TEMP_STOP"),
                            (dict(mrkt_warn_cls_code="02"), "MARKET_WARN_02"), (dict(iscd_stat_cls_code="58"), "STOCK_STATUS_58")):
                ctx, logs = self._ctx(quote_status=lambda c, kw=kw: st(**kw))
                self.assertFalse(r005._018_quote_status(ctx), tag)
                self.assertIn(tag, logs[-1])
            # stale status -> pass
            old = st(vi_cls_code="Y", at=now - timedelta(seconds=r001.QUOTE_STATUS_MAX_AGE_SECONDS + 1))
            self.assertTrue(r005._018_quote_status(self._ctx(quote_status=lambda c: old)[0]))
            self.assertTrue(r005._018_quote_status(self._ctx(quote_status=lambda c: None)[0]))

    def test_025_includes_unrealized(self):
        with mock.patch.object(r005, "ENABLE_DAILY_LOSS_LIMIT_GATE", True), mock.patch.object(r005, "DAILY_LOSS_LIMIT_KRW", 100_000), \
                mock.patch.object(r005, "DAILY_LOSS_LIMIT_INCLUDE_UNREALIZED", True):
            ctx, logs = self._ctx()
            ctx.api.live_state["realized_pnl_krw"] = -60_000
            ctx.api.positions = {"159010": {"quantity": 10, "buy_price": 30000.0, "current_price": 25000.0},  # -50,000
                                 "999999": {"quantity": 100, "buy_price": 100.0, "current_price": 1.0}}  # not bot's -> ignored
            ctx.state.traded_today = {"159010"}
            self.assertFalse(r005._007_daily_loss_limit(ctx))
            self.assertIn("unrealized=-50,000", logs[-1])
            ctx.state.traded_today = set()
            self.assertTrue(r005._007_daily_loss_limit(ctx))


@unittest.skipIf(r003 is None, "r003 import failed")
class TestR003OrderSafety(unittest.TestCase):
    def _api(self, positions, pending, dry_run=True):
        api = object.__new__(r003.TradingAPI)
        api.env_dv = "real"; api.dry_run = dry_run; api.cano = "1"; api.acnt_prdt_cd = "01"
        api.live_state = {"positions_meta": {}, "traded_today": set(positions)}
        api.positions = positions; api.pending_orders = pending
        api.buy_inflight_codes = set(positions); api.startup_position_codes = set()
        api.trade_lock_until = {}; api.last_sell_fill_price = {}
        api._last_sync_at = None; api._last_pending_poll_at = None; api._last_live_state_save_at = None
        api.sync_positions_from_account = lambda force=False: None
        api.persist_live_state = lambda *a, **k: None
        return api

    def test_account_sync_age(self):
        api = self._api({}, {})
        self.assertIsNone(api.account_sync_age_seconds())
        api._last_sync_at = datetime.now() - timedelta(seconds=42)
        self.assertGreaterEqual(api.account_sync_age_seconds(), 41.0)

    def test_sell_exception_registers_unknown_pending(self):
        code = "159010"
        api = self._api({code: {"quantity": 5, "buy_price": 29000.0, "buy_time": datetime(2026, 11, 4, 9, 8)}}, {}, dry_run=False)
        now = datetime(2026, 11, 4, 9, 15, 56)
        with mock.patch.object(r003.dsf, "order_cash", side_effect=RuntimeError("timeout")), \
                mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                mock.patch.object(r003, "log", lambda *a, **k: None):
            self.assertFalse(api.place_sell_order(code, 5, now, "HARD_STOP", False, price=28500.0, market_order=True))
        pend = api.pending_orders[code]
        self.assertEqual(pend["side"], "sell")
        self.assertTrue(pend["submission_unknown"])
        self.assertEqual(pend["unknown_since"], now)
        self.assertEqual(pend["order_no"], "")
        # a second sell while unknown is blocked (no duplicate submission)
        with mock.patch.object(r003.dsf, "order_cash", side_effect=AssertionError("must not be called")), \
                mock.patch.object(r003, "get_order_spec", return_value={"exchange": "KRX"}), \
                mock.patch.object(r003, "log", lambda *a, **k: None):
            self.assertFalse(api.place_sell_order(code, 5, now + timedelta(seconds=10), "HARD_STOP", False, price=28400.0))

    def test_partial_buy_remainder_cancelled_after_timeout(self):
        code = "010060"
        sub = datetime(2026, 11, 4, 9, 54, 10)
        positions = {code: {"quantity": 1, "buy_price": 185650.0, "buy_time": sub}}
        pending = {code: {"side": "buy", "quantity": 2, "submitted_at": sub, "order_no": "7", "order_org_no": "91",
                          "exchange": "KRX", "requested_price": 185700.0}}
        api = self._api(positions, pending)
        status = {"filled_qty": 1, "remaining_qty": 1, "rejected_qty": 0, "order_qty": 2, "cancel_yn": "", "avg_price": 185650.0, "order_no": "7"}
        api._fetch_today_order_status = lambda *a, **k: status
        with mock.patch.object(r003, "PARTIAL_BUY_CANCEL_AFTER_SECONDS", 60.0), mock.patch.object(r003, "log", lambda *a, **k: None):
            api.refresh_pending_orders(sub + timedelta(seconds=30))
            self.assertFalse(pending[code].get("cancel_inflight"))
            self.assertIn(code, api.pending_orders)
            api.refresh_pending_orders(sub + timedelta(seconds=61))
            self.assertTrue(pending[code].get("cancel_inflight"))
            self.assertFalse(pending[code].get("reprice_pending"))
            # cancel confirmed -> filled part becomes the position (remaining 0)
            status.update({"remaining_qty": 0, "cancel_yn": "Y"})
            api.refresh_pending_orders(sub + timedelta(seconds=70))
            self.assertNotIn(code, api.pending_orders)
            self.assertEqual(api.positions[code]["quantity"], 1)


if __name__ == "__main__":
    unittest.main()

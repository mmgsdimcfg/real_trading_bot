# -*- coding: utf-8 -*-
"""Regression test: a fully filled PARTIAL sell (TP1 40%) must keep the remaining position and its meta.

Bug (20261022 089890): refresh_pending_orders passed the order's rmn_qty (0 for a fully filled order) to
_confirm_pending_sell as the position's remaining qty, which popped the whole position and positions_meta.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="r003_test_runtime_")
os.environ["AUTO_TRADING_RUNTIME"] = _TMP  # best effort only: ineffective if another test module imported r001 first
sys.path.insert(0, str(ROOT / "live_trading"))

try:
    import r003_trade_live_execute as r003  # noqa: E402
except Exception as exc:  # broker SDK missing on this machine
    r003 = None
    _IMPORT_ERR = exc


def _make_api(positions: dict, pending: dict):
    api = object.__new__(r003.TradingAPI)
    api.env_dv = "real"
    api.dry_run = True
    api.live_state = {"positions_meta": {c: {"tp1_done": True, "entry_quantity": 4} for c in positions},
                      "traded_today": set(positions)}
    api.positions = positions
    api.pending_orders = pending
    api.buy_inflight_codes = set(positions)
    api.startup_position_codes = set()
    api.trade_lock_until = {}
    api.last_sell_fill_price = {}
    api._last_sync_at = None
    api._last_pending_poll_at = None
    api._last_live_state_save_at = None
    api.sync_positions_from_account = lambda force=False: None
    api.persist_live_state = lambda *a, **k: None
    return api


@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class TestPartialSellConfirm(unittest.TestCase):
    def _run(self, pre_qty: int, sell_qty: int, filled: int, rmn: int, cancel: str = ""):
        code = "089890"
        positions = {code: {"quantity": pre_qty, "buy_price": 35575.0, "buy_time": datetime(2026, 10, 22, 9, 5)}}
        pending = {code: {"side": "sell", "quantity": sell_qty, "pre_submit_qty": pre_qty, "order_no": "1",
                          "reason": "TP1_PARTIAL_40PCT", "buy_price": 35575.0, "requested_price": 36950.0,
                          "submitted_at": datetime(2026, 10, 22, 9, 8, 40), "exchange": "KRX"}}
        api = _make_api(positions, pending)
        status = {"filled_qty": filled, "remaining_qty": rmn, "rejected_qty": 0, "order_qty": sell_qty,
                  "cancel_yn": cancel, "avg_price": 36900.0, "order_no": "1"}
        api._fetch_today_order_status = lambda *a, **k: status
        api.refresh_pending_orders(datetime(2026, 10, 22, 9, 8, 41))
        return api, code

    def test_partial_tp_keeps_position(self):
        api, code = self._run(pre_qty=4, sell_qty=2, filled=2, rmn=0)
        self.assertIn(code, api.positions, "remaining 2 shares must stay tracked")
        self.assertIn(code, api.live_state["positions_meta"])
        self.assertIn(code, api.buy_inflight_codes)
        self.assertNotIn(code, api.pending_orders)
        self.assertEqual(api.last_sell_fill_price[code], 36900.0)
        self.assertTrue(api.live_state["positions_meta"][code]["tp1_done"])  # meta values preserved, not reset
        self.assertEqual(api.positions[code]["quantity"], 4)  # local qty is corrected by the next account sync
        # realized P&L accounted net of estimated costs: (36900-35575)*2 - costs
        gross = (36900 - 35575) * 2
        self.assertLess(api.live_state["realized_pnl_krw"], gross)
        self.assertGreater(api.live_state["realized_pnl_krw"], gross - 300)

    def test_realized_pnl_persist_roundtrip(self):
        state = {"positions_meta": {}, "traded_today": {"000001"}, "realized_pnl_krw": -12345.5}
        payload = r003._serialize_live_state(state)
        self.assertEqual(payload["realized_pnl_krw"], -12345.5)
        with mock.patch.object(r003, "_live_state_path", return_value=Path(_TMP) / "state.json"):
            (Path(_TMP) / "state.json").write_text(json.dumps(payload), encoding="utf-8")
            loaded = r003.load_live_state("20261023")
        self.assertEqual(loaded["realized_pnl_krw"], -12345.5)

    def test_full_sell_still_closes(self):
        api, code = self._run(pre_qty=4, sell_qty=4, filled=4, rmn=0)
        self.assertNotIn(code, api.positions)
        self.assertNotIn(code, api.live_state["positions_meta"])

    def test_cancelled_after_partial_fill_keeps_rest(self):
        api, code = self._run(pre_qty=4, sell_qty=4, filled=1, rmn=3, cancel="Y")
        self.assertIn(code, api.positions)



@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class TestImportCreatesNoLogFiles(unittest.TestCase):
    # 20261023: importing r003 under unittest discover wrote r003_*_{ts}.log pairs and fake 089890 sells
    # into the live runtime/log/<date>/ (RUNTIME_LOG_DIR already fixed by an earlier r001 import).
    def test_import_does_not_initialize_file_logging(self):
        self.assertIsNone(r003._LOG_CTX.get("trade_log"))
        self.assertEqual(r003._trade_log_target_paths(), [])
        self.assertFalse(any(hasattr(h, "baseFilename") for h in r003.trade_logger.handlers))

if __name__ == "__main__":
    unittest.main()

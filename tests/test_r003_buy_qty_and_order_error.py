# -*- coding: utf-8 -*-
"""Regression tests (2026-10-29, 105740/003160 11:54~12:00 UNKNOWN_ORDER_ERROR).

- get_affordable_buy_qty: an empty / failed inquire_psbl_order response must NOT fall back to the budget qty
  (fail-closed), and only the cash-based qty (nrcvb_buy_qty) is used, never max_buy_qty (includes margin).
- _submit_buy_order_cash: the KIS example function prints the rejection reason and returns an empty DataFrame;
  the reason must reach the BUY FAIL log and the result must still be judged a failure.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="r003_test_runtime_")
os.environ["AUTO_TRADING_RUNTIME"] = _TMP  # best effort only: ineffective if another test module imported r001 first
sys.path.insert(0, str(ROOT / "live_trading"))

try:
    import r003_trade_live_execute as r003  # noqa: E402
except Exception as exc:  # broker SDK missing on this machine
    r003 = None
    _IMPORT_ERR = exc

NOW = datetime(2026, 10, 29, 11, 59, 9)
KIS_ERR = ("-------------------------------\nError in response:  500  url= /uapi/domestic-stock/v1/trading/order-cash\n"
           "rt_cd :  1 / msg_cd :  APBK0952 / msg1 :  주문가능금액을 초과 했습니다\n-------------------------------")


def _api():
    api = object.__new__(r003.TradingAPI)
    api.env_dv = "real"
    api.dry_run = False
    api.cano = "00000000"
    api.acnt_prdt_cd = "01"
    return api


def _psbl_row(**kw):
    base = {"ord_psbl_cash": "193105", "nrcvb_buy_amt": "183580", "nrcvb_buy_qty": "3",
            "max_buy_amt": "183580", "max_buy_qty": "3", "psbl_qty_calc_unpr": "56200"}
    base.update(kw)
    return pd.DataFrame([base])


@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class TestAffordableQty(unittest.TestCase):
    def setUp(self):
        self.logs: list[str] = []
        for p in (mock.patch.object(r003, "log", self.logs.append),
                  mock.patch.object(r003, "get_order_spec", lambda now, nxt: {"exchange": "KRX", "ord_dvsn": "01"}),
                  mock.patch.object(r003.time, "sleep", lambda s: None)):
            p.start()
        self.addCleanup(mock.patch.stopall)

    def _qty(self, responses, price=43950.0):
        calls = []

        def fake(**kw):
            calls.append(kw)
            r = responses[min(len(calls), len(responses)) - 1]
            if isinstance(r, str):
                print(r)
                return pd.DataFrame()
            if isinstance(r, Exception):
                raise r
            return r
        with mock.patch.object(r003.dsf, "inquire_psbl_order", fake, create=True):
            qty = _api().get_affordable_buy_qty("003160", price, NOW, False)
        return qty, calls

    def test_empty_response_fails_closed_with_kis_reason(self):
        qty, calls = self._qty([KIS_ERR])
        self.assertEqual(qty, 0)  # was MAX_ORDER_AMOUNT_KRW // price = 11
        self.assertEqual(len(calls), 2)  # one retry
        self.assertIn("qty=0 (no order)", self.logs[-1])
        self.assertIn("msg_cd=APBK0952", self.logs[-1])

    def test_exception_fails_closed(self):
        qty, _ = self._qty([RuntimeError("timeout")])
        self.assertEqual(qty, 0)
        self.assertIn("RuntimeError: timeout", self.logs[-1])

    def test_retry_success(self):
        qty, calls = self._qty([KIS_ERR, _psbl_row()])
        self.assertEqual((qty, len(calls)), (3, 2))

    def test_cash_qty_not_margin_qty(self):
        qty, _ = self._qty([_psbl_row(nrcvb_buy_qty="3", max_buy_qty="11")])
        self.assertEqual(qty, 3)

    def test_zero_cash_qty_means_no_order(self):
        qty, _ = self._qty([_psbl_row(nrcvb_buy_qty="0", max_buy_qty="5")])
        self.assertEqual(qty, 0)

    def test_amount_fallback_when_qty_keys_missing(self):
        row = _psbl_row().drop(columns=["nrcvb_buy_qty", "max_buy_qty"])
        qty, _ = self._qty([row])
        self.assertEqual(qty, 4)  # 183,580 // 43,950

    def test_budget_cap_kept(self):
        qty, _ = self._qty([_psbl_row(nrcvb_buy_qty="100")])
        self.assertEqual(qty, int(r003.MAX_ORDER_AMOUNT_KRW / 43950))

    def test_market_order_query_uses_order_spec(self):
        _, calls = self._qty([_psbl_row()])
        self.assertEqual(calls[0]["ord_dvsn"], "01")


@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class TestBuyOrderError(unittest.TestCase):
    def _submit(self, printed: str, result):
        def fake(**kw):
            if printed:
                print(printed)
            return result
        with mock.patch.object(r003.dsf, "order_cash", fake):
            return _api()._submit_buy_order_cash("003160", 6, "01", "0", "KRX", 43950.0)

    def test_kis_reason_reaches_log(self):
        res = self._submit(KIS_ERR, pd.DataFrame())
        self.assertFalse(r003._order_succeeded(res))
        detail = r003._extract_order_error_detail(res)
        self.assertEqual(r003._short_order_error(detail), "APBK0952 주문가능금액을 초과 했습니다")

    def test_reason_without_rt_cd_is_still_failure(self):
        res = self._submit("Error Code : 500 | Internal Server Error", pd.DataFrame())
        self.assertFalse(r003._order_succeeded(res))
        self.assertIn("Internal Server Error", r003._extract_order_error_detail(res))

    def test_success_untouched(self):
        ok = pd.DataFrame([{"odno": "0013496600", "ord_tmd": "120109"}])
        res = self._submit("", ok)
        self.assertIs(res, ok)
        self.assertTrue(r003._order_succeeded(res))

    def test_empty_without_output_keeps_old_behavior(self):
        res = self._submit("", pd.DataFrame())
        self.assertFalse(r003._order_succeeded(res))
        self.assertEqual(r003._extract_order_error_detail(res), "UNKNOWN_ORDER_ERROR")


if __name__ == "__main__":
    unittest.main()

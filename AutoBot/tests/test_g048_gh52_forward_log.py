# -*- coding: utf-8 -*-
"""g048 GH52 forward log: month-end detection (data rows + KRX calendar for the newest row), formation (universe, stock filter by
name, top 30% by close / 252-day high), cohort scoring (next open + 1 tick, open after the 6th month end - 1 tick, unresolved = 0
recovery, open / waiting cohorts), live protection, the pre-registered acceptance (collecting / waiting / PASS / FAIL) and the CLI
(nothing logged on a non-month-end date)."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_simulation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))

import g042_hi52_signal_log as g042  # noqa: E402
import g048_gh52_forward_log as g048  # noqa: E402

N = 420
DATES = [d.strftime("%Y%m%d") for d in pd.bdate_range("2025-01-01", periods=N)]
Q = lambda *_: None  # noqa: E731
STOCKS = [f"{k}{k}{k}{k}{k}0" for k in range(1, 10)] + ["123450"]       # 10 stock codes


def write(dirp: Path, code: str, close, opn=None, amount=1e10, n=N):
    close = np.asarray(close, float)[:n]
    opn = close if opn is None else np.asarray(opn, float)[:n]
    df = pd.DataFrame({"date": DATES[:n], "open": opn, "high": close * 1.01, "low": close * 0.99, "close": close,
                       "volume": 1000, "amount": amount})
    df["source_date"] = [DATES[min(i + 1, N - 1)] for i in range(n)]
    dirp.mkdir(parents=True, exist_ok=True)
    df.to_csv(dirp / f"{code}.csv", index=False)


def run_main(*argv) -> tuple[int, str]:
    buf = io.StringIO()
    with mock.patch.object(sys, "argv", ["g048", *argv]), redirect_stdout(buf):
        rc = g048.main()
    return rc, buf.getvalue()


def month_end_rows():
    return [i for i in range(N - 1) if DATES[i][:6] != DATES[i + 1][:6]]


class G048Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old = os.environ.get("AUTO_TRADING_RUNTIME")
        os.environ["AUTO_TRADING_RUNTIME"] = str(self.root)
        self.h = self.root / "data" / "daily_history"
        for e in g042.ETFS:
            write(self.h, e, np.linspace(10000, 20000, N))
        write(self.h, g048.K200, np.linspace(30000, 40000, N))
        # stock k: rises to a peak at row 150 then falls by k%: a smaller k is closer to its 252-day high
        base = np.r_[np.linspace(5000, 10000, 151), np.full(N - 151, 10000.0)]
        for k, c in enumerate(STOCKS):
            x = base.copy()
            x[151:] = 10000 * (1 - 0.01 * (k + 1))
            write(self.h, c, x)
        write(self.h, "990010", base)                                # an ETF by name: never in the universe
        write(self.h, "990020", base, amount=1e8)                    # traded value too small
        self.out = self.root / "data" / "gh52_signals"
        self.names = mock.patch.object(g048, "stock_names", return_value={"990010": "KODEX 200선물", **{c: f"S{c}" for c in STOCKS}})
        self.names.start()
        self.cal = mock.patch.object(g048, "next_krx_day", side_effect=lambda d: (pd.Timestamp(d) + pd.offsets.BDay(1)).strftime("%Y%m%d"))
        self.cal.start()

    def tearDown(self):
        self.names.stop()
        self.cal.stop()
        if self.old is None:
            os.environ.pop("AUTO_TRADING_RUNTIME", None)
        else:
            os.environ["AUTO_TRADING_RUNTIME"] = self.old
        self.tmp.cleanup()

    def forms(self):
        return pd.read_csv(self.out / "formations.csv", dtype=str, keep_default_na=False).set_index("date")

    def ev(self):
        return pd.read_csv(self.out / "evaluation.csv", dtype=str, keep_default_na=False).set_index("date")

    def test_month_ends_use_the_next_row_and_the_calendar_for_the_newest(self):
        ends = g048.month_ends(DATES)
        self.assertEqual(ends, [DATES[i] for i in month_end_rows()])
        cut = DATES[: month_end_rows()[3] + 1]                      # the newest row is a month end by the calendar
        self.assertEqual(g048.month_ends(cut)[-1], cut[-1])
        self.assertNotIn(DATES[month_end_rows()[3] - 1], g048.month_ends(DATES[: month_end_rows()[3]]))

    def test_formation_universe_and_top_30_percent(self):
        P = g042.load_panel()
        ind = g042.indicators(P)
        i = next(i for i in month_end_rows() if i >= 200)               # the 252-day window still holds the row-150 peak
        d = DATES[i]
        self.assertEqual(g048.log_dates(P, ind, [d], "backfill", log=Q)[0], 1)
        row = self.forms().loc[d]
        univ = json.loads(row["universe"])
        self.assertEqual(sorted(univ), sorted(STOCKS))               # no index ETF / KOSPI ETF / named ETF / small value
        picks = [p["code"] for p in json.loads(row["picks"])]
        self.assertEqual(picks, STOCKS[:3])                          # 30% of 10, closest to the 252-day high first
        self.assertEqual((row["universe_n"], row["picks_n"], row["recorded"]), ("10", "3", "backfill"))
        self.assertEqual(g048.log_dates(P, ind, [DATES[i - 1]], "backfill", log=Q)[0], 0)                     # not a month end

    def test_closed_cohort_net_unresolved_and_benchmark(self):
        # first pick delisted before the exit, third pick has no open on the exit day (first open after it is used)
        me = month_end_rows()
        k0 = next(k for k, i in enumerate(me) if i >= 200)            # 200 rows of history for the 252-day high
        f = me[k0]
        e, x = f + 1, me[k0 + 6] + 1
        self.assertLess(x, N - 5)
        base = np.r_[np.linspace(5000, 10000, 151), np.full(N - 151, 10000.0)]
        x1 = base.copy(); x1[151:] = 9900
        write(self.h, STOCKS[0], x1, n=x - 5)
        x3 = base.copy(); x3[151:] = 9700
        o3 = x3.copy(); o3[x] = np.nan
        write(self.h, STOCKS[2], x3, opn=o3)
        P = g042.load_panel()
        ind = g042.indicators(P)
        g048.log_dates(P, ind, [DATES[f]], "backfill", log=Q)
        g048.evaluate(P, log=Q)
        r = self.ev().loc[DATES[f]]
        self.assertEqual((r["status"], r["entry_date"], r["exit_date"], r["unresolved"], r["traded"]),
                         ("closed", DATES[e], DATES[x], "1", "3"))

        def net(p_in, p_out, k):
            pin = (p_in + float(g042.tick(p_in))) * (1 + g042.FEE)
            return ((p_out - float(g042.tick(p_out))) * (1 - g042.FEE - g042.sell_tax(DATES[k])) / pin - 1) * 100

        expect = np.mean([-100.0, net(9800, 9800, x), net(9700, 9700, x + 1)])
        self.assertAlmostEqual(float(r["net_pct"]), expect, places=2)
        self.assertTrue(r["univ_pct"] and r["k200_pct"])
        self.assertAlmostEqual(float(r["excess_pct"]), float(r["net_pct"]) - float(r["univ_pct"]), places=2)

    def test_open_and_waiting_cohorts(self):
        P = g042.load_panel()
        ind = g042.indicators(P)
        me = month_end_rows()
        g048.log_dates(P, ind, [DATES[me[-2]]], "backfill", log=Q)
        g048.evaluate(P, log=Q)
        self.assertEqual(self.ev().loc[DATES[me[-2]], "status"], "open")
        self.assertEqual(self.ev().loc[DATES[me[-2]], "exit_date"], "")
        cut = DATES[: me[-1] + 1]
        for c in list(g042.ETFS) + [g048.K200] + STOCKS:
            df = pd.read_csv(self.h / f"{c}.csv", dtype={"date": str})
            df[df["date"].isin(cut)].to_csv(self.h / f"{c}.csv", index=False)
        P = g042.load_panel()
        ind = g042.indicators(P)
        g048.log_dates(P, ind, [cut[-1]], "backfill", log=Q)
        g048.evaluate(P, log=Q)
        self.assertEqual(self.ev().loc[cut[-1], "status"], "waiting_entry")

    def test_live_is_kept_and_cli_logs_only_month_ends(self):
        me = month_end_rows()
        cut = DATES[: me[-1] + 1]
        for c in list(g042.ETFS) + [g048.K200] + STOCKS:
            df = pd.read_csv(self.h / f"{c}.csv", dtype={"date": str})
            df[df["date"].isin(cut)].to_csv(self.h / f"{c}.csv", index=False)
        P = g042.load_panel()
        ind = g042.indicators(P)
        d = cut[-1]
        g048.log_dates(P, ind, [d], None, log=Q, now=datetime.strptime(d, "%Y%m%d").replace(hour=20))
        self.assertEqual(self.forms().at[d, "recorded"], "live")
        g048.log_dates(P, ind, [d], "backfill", log=Q)
        g048.log_dates(P, ind, [d], None, log=Q, now=datetime(2030, 1, 1))
        self.assertEqual(self.forms().at[d, "recorded"], "live")
        rc, out = run_main("--date", cut[-2], "--evaluate")
        self.assertEqual(rc, 0)
        self.assertIn("not the last trading day", out)
        self.assertEqual(len(self.forms()), 1)
        self.assertTrue((self.out / "report.md").is_file())
        rc, out = run_main("--date", "20250104")                       # Saturday
        self.assertEqual(rc, 0)


class AcceptanceTest(unittest.TestCase):
    """the sample = 36 calendar month ends from 2026-11; missing live months count 0; the final verdict is written once."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.final = Path(self.tmp.name) / "acceptance_final.json"
        self.months = g048.sample_months()

    def tearDown(self):
        self.tmp.cleanup()

    def ev(self, months, net=5.0, univ=1.0, status="closed", recorded="live"):
        rows = [{"date": m, "recorded": recorded, "status": status, "net_pct": str(net + (k % 3)), "univ_pct": str(univ),
                 "excess_pct": str(net + (k % 3) - univ)} for k, m in enumerate(months)]
        df = pd.DataFrame(rows)
        for c in g048.EVAL_COLS:
            if c not in df:
                df[c] = ""
        return df[g048.EVAL_COLS]

    def acc(self, ev, newest):
        return g048.acceptance(ev, newest, self.final)

    def test_sample_is_36_calendar_month_ends_from_november_2026(self):
        self.assertEqual(len(self.months), 36)
        self.assertEqual(self.months[0][:6], "202611")
        self.assertEqual(self.months[-1][:6], "202910")
        self.assertEqual(self.months[1], "20261230")                 # 12-31 is closed in the r014 calendar

    def test_collecting_waiting_missing_and_once_only(self):
        end = g048.next_krx_day(g048.cal_month_end(g048.add_months(self.months[-1][:6], 6)))
        self.assertEqual(self.acc(self.ev(self.months[:10]), "20270901")["verdict"], "collecting")
        self.assertFalse(self.final.exists())
        e = self.ev(self.months)
        e.loc[35, "status"] = "open"
        self.assertEqual(self.acc(e, "20291105")["verdict"], "waiting for the last 6-month windows")
        backfill = self.ev(self.months, recorded="backfill")                       # not live = every month missing
        r = self.acc(backfill, end)
        self.assertEqual((r["verdict"], len(r["missing"])), ("INVALID", 36))
        self.assertTrue(self.final.exists())
        r2 = self.acc(self.ev(self.months), end)                                  # recorded once: never recomputed
        self.assertEqual(r2["verdict"], "INVALID")
        self.assertIn("not recomputed", r2["note"])

    def test_pass_fail_and_missing_months_count_zero(self):
        end = g048.next_krx_day(g048.cal_month_end(g048.add_months(self.months[-1][:6], 6)))
        self.assertEqual(self.acc(self.ev(self.months), end)["verdict"], "PASS")
        self.final.unlink()
        self.assertEqual(self.acc(self.ev(self.months, net=0.5, univ=3.0), end)["verdict"], "FAIL")   # excess < 0
        self.final.unlink()
        r = self.acc(self.ev(self.months[:31]), end)                              # 5 missing months -> zeros, still judged
        self.assertEqual(len(r["missing"]), 5)
        self.assertIn(r["verdict"], ("PASS", "FAIL"))
        self.assertAlmostEqual(r["result"]["base"]["net_mean"], np.mean([5 + k % 3 for k in range(31)] + [0] * 5), places=6)


class StockFilterTest(unittest.TestCase):
    def test_names(self):
        for code, name, ok in (("037270", "YG PLUS", True), ("458650", "성우", True), ("138040", "메리츠금융지주", True),
                               ("369370", "블리츠웨이엔터테인먼트", True), ("206400", "아인스팩씨엔에스", True),
                               ("448100", "WON 200", False), ("385720", "TIME 코스피액티브", False), ("330590", "롯데리츠", False),
                               ("005935", "삼성전자우", False), ("000087", "하이트진로2우B", False), ("442580", "PLUS 글로벌HBM반도체", False),
                               ("472840", "IBK 200", False), ("005930", "삼성전자", True), ("0030R0", "대신밸류리츠", False)):
            self.assertEqual(g048.is_stock(code, name), ok, (code, name))


if __name__ == "__main__":
    unittest.main()

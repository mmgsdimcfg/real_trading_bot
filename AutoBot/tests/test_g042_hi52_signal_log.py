# -*- coding: utf-8 -*-
"""g042 HI52 signal log: signal / gate rules on synthetic daily files, live classification and protection, evaluation rules
(scoring v2 2026-11-20: sell - 1 tick, a bad window is kept and flagged, a delisted code is unresolved = 0 recovery), the
pre-registered acceptance (progress, sample close, verdict), input errors and the lock."""
from __future__ import annotations

import io
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
import r019_platform as r019  # noqa: E402

N = 320
DATES = [d.strftime("%Y%m%d") for d in pd.bdate_range("2025-01-01", periods=N)]
Q = lambda *_: None  # noqa: E731


def write(dirp: Path, code: str, close, amount=1e10, source_same_last=False, n=N):
    close = np.asarray(close, float)
    df = pd.DataFrame({"date": DATES[:n], "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                       "volume": 1000, "amount": amount})
    df["source_date"] = [DATES[min(i + 1, N - 1)] for i in range(n)]
    if source_same_last:
        df.loc[n - 1, "source_date"] = DATES[n - 1]
    dirp.mkdir(parents=True, exist_ok=True)
    df.to_csv(dirp / f"{code}.csv", index=False)


def run_main(*argv) -> tuple[int, str]:
    buf = io.StringIO()
    with mock.patch.object(sys, "argv", ["g042", *argv]), redirect_stdout(buf):
        rc = g042.main()
    return rc, buf.getvalue()


class G042Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old = os.environ.get("AUTO_TRADING_RUNTIME")
        os.environ["AUTO_TRADING_RUNTIME"] = str(self.root)
        self.h = h = self.root / "data" / "daily_history"
        up = np.linspace(10000, 20000, N)
        for e in g042.ETFS:
            write(h, e, up, source_same_last=True)
        c = np.full(N, 5000.0)
        c[-1] = 5600
        write(h, "111110", c)                       # new 250-day high on the last day
        write(h, "222220", c, amount=1e8)           # traded value too small
        write(h, "333330", c / 5)                   # below 2,000 won
        c4 = c.copy()
        c4[-30:-1] = 7600
        c4[-31] = 5000
        c4[-1] = 7700
        write(h, "444440", c4)                      # 52% jump 30 rows earlier -> excluded
        write(h, "555550", np.linspace(3000, 9000, N))   # new high every day
        self.out = self.root / "data" / "hi52_signals"

    def tearDown(self):
        if self.old is None:
            os.environ.pop("AUTO_TRADING_RUNTIME", None)
        else:
            os.environ["AUTO_TRADING_RUNTIME"] = self.old
        self.tmp.cleanup()

    def days(self):
        return pd.read_csv(self.out / "days.csv", dtype=str, keep_default_na=False).set_index("date")

    def test_signal_rules_gate_and_provisional(self):
        P = g042.load_panel()
        ind = g042.indicators(P)
        d = DATES[-1]
        self.assertEqual(ind["gate"].at[d], "UP")
        hits = set(ind["sig250"].loc[d][ind["sig250"].loc[d]].index)
        self.assertEqual(hits, {"111110", "555550"})                 # ETFs rise too but are never picks
        self.assertEqual(int(ind["sig500"].loc[d].sum()), 0)
        self.assertEqual(g042.log_dates(P, ind, [d], "live", log=Q), 1)
        row = self.days().loc[d]
        self.assertEqual(row["provisional"], "1")
        self.assertEqual(row["n_hi500"], "")                         # unknown (not enough rows), not zero
        sigs = pd.read_csv(self.out / "signals.csv", dtype=str)
        self.assertEqual(sorted(sigs["code"]), ["111110", "555550"])

    def test_classify_live_only_before_the_next_open(self):
        fri = "20261113"
        self.assertEqual(g042.classify(fri, fri, datetime(2026, 11, 13, 21, 0)), "live")
        self.assertEqual(g042.classify(fri, fri, datetime(2026, 11, 15, 23, 0)), "live")     # weekend
        self.assertEqual(g042.classify(fri, fri, datetime(2026, 11, 16, 8, 59)), "live")
        self.assertEqual(g042.classify(fri, fri, datetime(2026, 11, 16, 9, 0)), "late")
        self.assertEqual(g042.classify("20261112", fri, datetime(2026, 11, 13, 21, 0)), "late")  # not the newest day

    def test_live_is_never_replaced_by_late_or_backfill(self):
        P = g042.load_panel()
        ind = g042.indicators(P)
        d = DATES[-1]
        g042.log_dates(P, ind, [d], None, log=Q, now=datetime.strptime(d, "%Y%m%d").replace(hour=20))
        self.assertEqual(self.days().at[d, "recorded"], "live")
        g042.log_dates(P, ind, DATES[-5:], "backfill", log=Q)
        g042.log_dates(P, ind, [d], None, log=Q, now=datetime(2030, 1, 1))          # a late re-run
        days = self.days()
        self.assertEqual(days.at[d, "recorded"], "live")
        self.assertEqual(days.at[DATES[-2], "recorded"], "backfill")
        self.assertEqual(len(days), 5)
        g042.log_dates(P, ind, [DATES[-2]], None, log=Q)                              # past date -> late
        self.assertEqual(self.days().at[DATES[-2], "recorded"], "late")

    def test_evaluation_net_status_and_research_rules(self):
        c6 = np.linspace(3000, 9000, N)
        c6[N - 50] = c6[N - 51] * 1.5                 # a jump inside the holding window of a signal 60 rows before the end
        write(self.h, "666660", c6)
        write(self.h, "777770", np.linspace(3000, 9000, N)[: N - 40], n=N - 40)      # delisted 40 rows before the end
        P = g042.load_panel()
        ind = g042.indicators(P)
        i = N - 60
        g042.log_dates(P, ind, [DATES[i], DATES[-1]], "backfill", log=Q)
        g042.evaluate(P, ind, log=Q)
        ev = pd.read_csv(self.out / "evaluation.csv", dtype=str, keep_default_na=False)
        r = ev[(ev["date"] == DATES[i]) & (ev["code"] == "555550")].iloc[0]
        c = np.linspace(3000, 9000, N)
        p = c[i + 1] + float(g042.tick(c[i + 1]))
        x = c[i + 40] - float(g042.tick(c[i + 40]))                       # sell 1 tick below the close
        net = (x * (1 - g042.FEE - g042.sell_tax(DATES[i + 40])) - p * (1 + g042.FEE)) / (p * (1 + g042.FEE)) * 100
        self.assertEqual((r["status40"], r["exit40_date"]), ("closed", DATES[i + 40]))
        self.assertAlmostEqual(float(r["net40_pct"]), net, places=2)
        r6 = ev[(ev["date"] == DATES[i]) & (ev["code"] == "666660")].iloc[0]
        self.assertEqual((r6["status40"], r6["bad40"]), ("closed", "1"))                  # kept, flagged
        d7 = ev[(ev["date"] == DATES[i]) & (ev["code"] == "777770")].iloc[0]
        self.assertEqual((d7["status40"], float(d7["net40_pct"])), ("unresolved", -100.0))   # no close after the exit day
        self.assertEqual(ev[ev["date"] == DATES[-1]].iloc[0]["status40"], "waiting_entry")
        self.assertTrue((self.out / "report.md").is_file())
        # benchmark = the universe of DATES[i] by the same rules (incl. the bad window and the unresolved code)
        self.assertTrue(r["univ40_pct"] != "" and r["excess40_pct"] != "")
        self.assertAlmostEqual(float(r["excess40_pct"]), float(r["net40_pct"]) - float(r["univ40_pct"]), places=2)

    def test_relogged_old_date_is_evaluated_and_days_csv_is_the_only_source(self):
        rc, out = run_main("--backfill-from", DATES[-80], "--evaluate")
        self.assertEqual(rc, 0, out)
        rc, out = run_main("--date", DATES[-70])              # re-log an old date (late), then evaluate only
        self.assertEqual(rc, 0, out)
        rc, out = run_main("--evaluate")
        ev = pd.read_csv(self.out / "evaluation.csv", dtype=str, keep_default_na=False)
        self.assertIn(DATES[-70], set(ev["date"]))
        self.assertEqual(self.days().at[DATES[-70], "recorded"], "late")
        n = len(ev)
        # signals.csv is a derived view: losing it (a run that died after the days.csv commit) changes nothing
        (self.out / "signals.csv").unlink()
        rc, out = run_main("--evaluate")
        self.assertEqual(len(pd.read_csv(self.out / "evaluation.csv", dtype=str)), n)
        self.assertEqual(g042.status()["signals"], n)
        self.assertEqual(len(pd.read_csv(self.out / "signals.csv", dtype=str)), n)   # the evaluation rewrote the view

    def test_bad_move_boundary_is_exact(self):
        c = np.full(N, 2000.0)
        c[-2] = 1390                                          # exactly -30.5% -> not a bad move
        c[-1] = 1390
        write(self.h, "888880", c)
        P = g042.load_panel()
        ind = g042.indicators(P)
        self.assertFalse(bool(ind["bad"].at[DATES[-2], "888880"]))
        c[-2] = 1389
        write(self.h, "888880", c)
        ind = g042.indicators(g042.load_panel())
        self.assertTrue(bool(ind["bad"].at[DATES[-2], "888880"]))

    def test_start_counts_trading_rows(self):
        self.assertIsNone(g042._start_for([]))
        self.assertEqual(g042._start_for([DATES[300]]), DATES[0])
        with mock.patch.object(g042, "WARMUP_ROWS", 100):
            self.assertEqual(g042._start_for([DATES[300]]), DATES[200])

    def test_zero_signals_still_write_the_report(self):
        for code in ("111110", "222220", "333330", "444440", "555550"):
            (self.h / f"{code}.csv").unlink()
        rc, out = run_main("--date", DATES[-1], "--evaluate")
        self.assertEqual(rc, 0, out)
        self.assertIn("No signals yet", (self.out / "report.md").read_text(encoding="utf-8"))
        self.assertEqual(list(pd.read_csv(self.out / "evaluation.csv").columns), g042.EVAL_COLS)

    def test_exit_codes(self):
        rc, out = run_main("--date", "20250104")            # Saturday
        self.assertEqual(rc, 0, out)
        rc, out = run_main("--date", "20250104", "--evaluate")
        self.assertEqual(rc, 0, out)
        rc, out = run_main("--date", "20300101")            # weekday without ETF bars
        self.assertEqual(rc, 4, out)
        with r019.locked_file(self.out / ".g042.lock"):
            rc, out = run_main("--date", DATES[-1])
            self.assertEqual(rc, 3, out)
            rc, out = run_main("--date", "20250104", "--evaluate")   # weekend: 0 before the lock
            self.assertEqual(rc, 0, out)
        (self.h / f"{g042.ETFS[0]}.csv").unlink()
        rc, out = run_main("--date", DATES[-1])
        self.assertEqual(rc, 2, out)

    def test_panel_start(self):
        with mock.patch.object(g042, "HISTORY_ROWS", 100):
            self.assertEqual(len(g042.load_panel()["close"]), 100)
            self.assertEqual(len(g042.load_panel(start=DATES[50])["close"]), N - 50)
            self.assertEqual(len(g042.load_panel(start=DATES[-10])["close"]), 100)   # never fewer than HISTORY_ROWS

    def test_sell_tax_schedule(self):
        self.assertEqual(g042.sell_tax("20190531"), 0.0030)
        self.assertEqual(g042.sell_tax("20190603"), 0.0025)
        self.assertEqual(g042.sell_tax("20251231"), 0.0015)
        self.assertEqual(g042.sell_tax("20260102"), 0.0020)


class AcceptanceTest(unittest.TestCase):
    """[2026-11-20] pre-registered acceptance: progress, sample close, verdict"""

    def cal(self, n):
        return [d.strftime("%Y%m%d") for d in pd.bdate_range("2026-11-20", periods=n)]

    def frames(self, cal, net, univ, up_every=1, status="closed"):
        days = pd.DataFrame({"date": cal, "gate": ["UP" if k % up_every == 0 else "DOWN" for k in range(len(cal))],
                             "recorded": "live"})
        rows = [{"date": d, "lb": "250", "code": "111110", "gate": g, "recorded": "live", "status40": status,
                 "net40_pct": str(net), "univ40_pct": str(univ)} for d, g in zip(days["date"], days["gate"]) if g == "UP"]
        return pd.DataFrame(rows), days

    def test_collecting_until_all_minimums(self):
        cal = self.cal(400)
        ev, days = self.frames(cal, 2.0, 0.5)
        a = g042.acceptance(ev, days, cal)
        self.assertEqual((a["verdict"], a["trading_days"], a["up_days"], a["cohorts"]), ("collecting", 400, 400, 400))

    def test_verdicts(self):
        cal = self.cal(905)
        rng = np.random.default_rng(1)
        ev, days = self.frames(cal, 2.0, 0.5, up_every=3)
        ev["net40_pct"] = (2.0 + rng.normal(0, 1.0, len(ev))).astype(str)
        with mock.patch.object(g042, "BOOT_N", 300):
            a = g042.acceptance(ev, days, cal)
        self.assertEqual(a["sample_closed_at"], cal[899])
        self.assertEqual(a["verdict"], "PASS")
        ev2 = ev.copy()
        ev2["net40_pct"] = (0.6 + rng.normal(0, 1.0, len(ev2))).astype(str)        # excess ~ +0.1, minus 0.25 cost -> fails
        with mock.patch.object(g042, "BOOT_N", 300):
            self.assertEqual(g042.acceptance(ev2, days, cal)["verdict"], "FAIL")
        ev3 = ev.copy()
        ev3.loc[ev3[ev3["date"] <= cal[899]].index[-1], "status40"] = "open"        # a cohort inside the sample still open
        self.assertEqual(g042.acceptance(ev3, days, cal)["verdict"], "waiting for the last 40-day windows")

    def test_only_live_and_after_registration(self):
        cal = self.cal(905)
        ev, days = self.frames(cal, 2.0, 0.5)
        days["recorded"] = "backfill"
        ev["recorded"] = "backfill"
        a = g042.acceptance(ev, days, cal)
        self.assertEqual((a["up_days"], a["cohorts"], a["verdict"]), (0, 0, "collecting"))


if __name__ == "__main__":
    unittest.main()

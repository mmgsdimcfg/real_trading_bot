# -*- coding: utf-8 -*-
"""g049 HI52 paper account: next-open fills of the frozen plan (budget 1/10 of the prior close NAV, integer shares, tick +
commission), 40-row hold then the next open, 10-slot cap with a fixed hash order, RANDOM drawing the same count, gate DOWN = no
buys, a code that stops trading is written off after 20 rows, re-runs change nothing, and the once-only acceptance."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_simulation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))

import g042_hi52_signal_log as g042  # noqa: E402
import g049_hi52_paper_account as g049  # noqa: E402

N = 420
DATES = [d.strftime("%Y%m%d") for d in pd.bdate_range("2025-01-01", periods=N)]
Q = lambda *_: None  # noqa: E731
R0 = 300            # the first signal row (after REG)
REG = DATES[R0 - 3]


def write(dirp: Path, code: str, close, opn=None, n=N):
    close = np.asarray(close, float)[:n]
    opn = close if opn is None else np.asarray(opn, float)[:n]
    df = pd.DataFrame({"date": DATES[:n], "open": opn, "high": close, "low": close, "close": close, "volume": 1000, "amount": 1e10})
    df["source_date"] = [DATES[min(i + 1, N - 1)] for i in range(n)]
    dirp.mkdir(parents=True, exist_ok=True)
    df.to_csv(dirp / f"{code}.csv", index=False)


def step(at: int, lo=5000.0, hi=5600.0):
    x = np.full(N, lo)
    x[at:] = hi
    return x


class G049Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old = os.environ.get("AUTO_TRADING_RUNTIME")
        os.environ["AUTO_TRADING_RUNTIME"] = str(self.root)
        self.h = self.root / "data" / "daily_history"
        for e in g042.ETFS:
            write(self.h, e, np.linspace(10000, 20000, N))
        self.patches = [mock.patch.object(g049, "REGISTERED", REG),
                        mock.patch.object(g049.g048, "next_krx_day",
                                          side_effect=lambda d: (pd.Timestamp(d) + pd.offsets.BDay(1)).strftime("%Y%m%d"))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        if self.old is None:
            os.environ.pop("AUTO_TRADING_RUNTIME", None)
        else:
            os.environ["AUTO_TRADING_RUNTIME"] = self.old
        self.tmp.cleanup()

    def run_upto(self, k: int, now=None):
        P = g042.load_panel()
        ind = g042.indicators(P)
        return g049.process(P, ind, DATES[k], now=now, log=Q)

    def ledger(self):
        return pd.read_csv(g049.out_dir() / "ledger.csv", dtype=str, keep_default_na=False)

    def test_buy_next_open_hold_40_then_sell_next_open(self):
        write(self.h, "111110", step(R0))
        self.assertGreater(self.run_upto(R0 + 45), 0)
        led = self.ledger()
        b = led[(led["account"] == "HI52") & (led["side"] == "buy")].iloc[0]
        px = 5600 + float(g042.tick(5600))
        self.assertEqual((b["date"], b["code"], b["price"]), (DATES[R0 + 1], "111110", f"{px:.0f}"))
        self.assertEqual(int(b["shares"]), int(g049.CAPITAL / g049.SLOTS // (px * (1 + g049.FEE))))
        s = led[(led["account"] == "HI52") & (led["side"] == "sell")].iloc[0]
        self.assertEqual(s["date"], DATES[R0 + 1 + g049.HOLD])                       # due at the 40th held close, sold next open
        self.assertEqual(s["price"], f"{5600 - float(g042.tick(5600)):.0f}")
        self.assertEqual(s["reason"], "held 40 rows")
        r = led[(led["account"] == "RANDOM") & (led["side"] == "buy")]
        self.assertEqual(len(r), 1)                                                  # the same count from the universe
        plans = pd.read_csv(g049.out_dir() / "plans.csv", dtype=str, keep_default_na=False)
        p = plans[(plans["date"] == DATES[R0]) & (plans["account"] == "HI52")].iloc[0]
        self.assertEqual(json.loads(p["buys"]), [["111110", "5600"]])                 # code + the frozen reference close
        self.assertEqual(p["fill_date"], DATES[R0 + 1])
        self.assertEqual(p["rule"], g049.RULE_VERSION)

    def test_rerun_changes_nothing_and_live_plan(self):
        write(self.h, "111110", step(R0))
        now = datetime.strptime(DATES[R0 + 5], "%Y%m%d").replace(hour=20)
        P = g042.load_panel()
        newest_cut = R0 + 5
        self.run_upto(newest_cut, now=now)
        before = {f: (g049.out_dir() / f).read_text(encoding="utf-8") for f in ("ledger.csv", "nav.csv", "plans.csv", "state.json")}
        self.assertEqual(self.run_upto(newest_cut, now=now), 0)
        after = {f: (g049.out_dir() / f).read_text(encoding="utf-8") for f in before}
        self.assertEqual(before, after)
        plans = pd.read_csv(g049.out_dir() / "plans.csv", dtype=str, keep_default_na=False)
        self.assertTrue((plans["recorded"] == "late").all())                         # newest data row is not DATES[R0+5]
        del P

    def test_ten_slots_and_fixed_order(self):
        codes = [f"{k:05d}0" for k in range(10, 22)]                                  # 12 signals on the same day
        for c in codes:
            write(self.h, c, step(R0))
        self.run_upto(R0 + 3)
        led = self.ledger()
        b = led[(led["account"] == "HI52") & (led["side"] == "buy") & (led["result"] == "filled")]
        self.assertEqual(len(b), 10)
        order = sorted(codes, key=lambda c: g049.rank_key("HI52", DATES[R0], c))
        self.assertEqual(list(b["code"]), order[:10])
        st = json.loads((g049.out_dir() / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(len(st["accounts"]["HI52"]["positions"]), 10)

    def test_gate_down_means_no_buys(self):
        for e in g042.ETFS:
            write(self.h, e, np.linspace(20000, 10000, N))
        write(self.h, "111110", step(R0))
        self.run_upto(R0 + 5)
        self.assertFalse((g049.out_dir() / "ledger.csv").exists() and len(self.ledger()))

    def test_no_open_skip_and_write_off(self):
        c = step(R0)
        o = c.copy(); o[R0 + 1] = np.nan                                               # no open on the entry day -> skipped
        write(self.h, "111110", c, opn=o)
        c2 = step(R0)
        write(self.h, "222220", c2, n=R0 + 30)                                         # stops trading while held
        self.run_upto(R0 + 1 + g049.HOLD + 25)
        led = self.ledger()
        hi = led[led["account"] == "HI52"]
        self.assertIn("no open", list(hi.loc[hi["code"] == "111110", "reason"]))
        w = hi[(hi["code"] == "222220") & (hi["side"] == "sell")]
        self.assertEqual(list(w["result"]), ["written_off"])


    def test_crash_rows_after_the_commit_are_dropped(self):
        write(self.h, "111110", step(R0))
        self.run_upto(R0 + 3)
        nav = pd.read_csv(g049.out_dir() / "nav.csv", dtype=str)
        junk = nav.tail(2).copy()
        junk["date"] = DATES[R0 + 4]                                                 # written by a run that crashed before state.json
        pd.concat([nav, junk]).to_csv(g049.out_dir() / "nav.csv", index=False)
        self.run_upto(R0 + 6)
        nav = pd.read_csv(g049.out_dir() / "nav.csv", dtype=str)
        self.assertEqual(len(nav), len(nav.drop_duplicates(["date", "account"])))
        self.assertEqual(sorted(set(nav["date"]))[-1], DATES[R0 + 6])

    def test_missing_fill_day_is_noted_not_shifted_silently(self):
        for e in g042.ETFS:                                                           # the planned fill day has no index ETF row
            df = pd.read_csv(self.h / f"{e}.csv", dtype={"date": str})
            df[df["date"] != DATES[R0 + 1]].to_csv(self.h / f"{e}.csv", index=False)
        write(self.h, "111110", step(R0))
        self.run_upto(R0 + 10)
        b = self.ledger()
        b = b[(b["account"] == "HI52") & (b["side"] == "buy")].iloc[0]
        self.assertEqual(b["date"], DATES[R0 + 2])
        self.assertIn("missing in the data", b["reason"])


class AcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.p = mock.patch.object(g049, "out_dir", return_value=self.dir)
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def write_nav(self, n, hi_ret, rnd_ret, committed=None):
        rows, plans = [], []
        e = {"HI52": float(g049.CAPITAL), "RANDOM": float(g049.CAPITAL)}
        rng = np.random.default_rng(1)
        for k in range(n):
            d = f"2027{k:04d}"
            for a, mu in (("HI52", hi_ret), ("RANDOM", rnd_ret)):
                e[a] *= 1 + mu + rng.normal(0, 0.002)
                rows.append({"date": d, "account": a, "cash": 0, "value": 0, "nav": f"{e[a]:.0f}", "positions": 5, "exposure": 1,
                             "buy_value": 0})
                plans.append({"date": d, "account": a, "recorded": "live"})
        pd.DataFrame(rows, columns=g049.NAV_COLS).to_csv(self.dir / "nav.csv", index=False)
        pd.DataFrame(plans).reindex(columns=g049.PLAN_COLS).to_csv(self.dir / "plans.csv", index=False)
        led = [{"date": f"2027{k:04d}", "account": "HI52", "side": "buy", "code": "111110", "result": "filled" if k % 20 else "skipped"}
               for k in range(n)]
        pd.DataFrame(led).reindex(columns=g049.LEDGER_COLS).to_csv(self.dir / "ledger.csv", index=False)
        st = g049.load_state()
        st["last_processed"] = f"2027{(n if committed is None else committed) - 1:04d}"
        g049.save_state(st)

    def test_collecting_then_once(self):
        with mock.patch.object(g049, "JUDGE_ROWS", 300):
            self.write_nav(100, 0.001, 0.0)
            self.assertEqual(g049.acceptance()["verdict"], "collecting")
            self.write_nav(300, 0.001, 0.0, committed=299)                            # the 300th row is not committed yet
            self.assertEqual(g049.acceptance()["verdict"], "collecting")
            self.assertFalse((self.dir / "acceptance_final.json").exists())
            self.write_nav(300, 0.001, 0.0)
            r = g049.acceptance()
            self.assertEqual(r["verdict"], "PASS")
            self.assertTrue((self.dir / "acceptance_final.json").exists())
            self.write_nav(300, 0.0, 0.001)                                            # data changed later: verdict kept
            self.assertEqual(g049.acceptance()["verdict"], "PASS")
            (self.dir / "acceptance_final.json").unlink()
            self.assertEqual(g049.acceptance()["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()

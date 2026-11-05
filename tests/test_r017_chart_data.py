# -*- coding: utf-8 -*-
"""r017_chart_data (2026-11-06): bars from ticks (incremental cache == full), from 10s raw bars, day/week/month, indicators
through the r012/r002 path on a synthetic runtime."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

with contextlib.redirect_stdout(io.StringIO()):
    import r001_define_config as r001  # noqa: E402
    import r017_chart_data as r017  # noqa: E402
    from r009_tick_bars import TICK_FILE_COLUMNS  # noqa: E402

CODE, NAME, DAY, PREV = "005930", "삼성전자", "20261105", "20261104"


def _tick_lines(day: str, n: int, start=(9, 0, 0), base=70000, with_session=True):
    rows = []
    acml = 0
    t = datetime.strptime(day, "%Y%m%d").replace(hour=start[0], minute=start[1], second=start[2])
    for i in range(n):
        px = base + (i % 7) * 100
        qty = 1 + (i % 3)
        acml += qty
        hhmmss = t.strftime("%H%M%S")
        sess = "pre" if t.hour < 9 else ("close_auction" if (t.hour, t.minute) >= (15, 30) else "regular")
        row = {"recv_ts": t.strftime("%Y-%m-%dT%H:%M:%S.000"), "code": CODE, "bsop_date": day, "trade_time": hhmmss, "price": px,
               "cntg_vol": qty, "acml_vol": acml, "ccld_dvsn": 1, "seln_cntg_csnu": 0, "shnu_cntg_csnu": i + 1, "askp1": px + 100,
               "bidp1": px, "hour_cls_code": 0, "venue": "KRX", "session": sess}
        cols = TICK_FILE_COLUMNS if with_session else TICK_FILE_COLUMNS[:-2]
        rows.append(",".join(str(row[c]) for c in cols))
        t += timedelta(seconds=7)
    header = ",".join(TICK_FILE_COLUMNS if with_session else TICK_FILE_COLUMNS[:-2])
    return [header + "\n"] + [r + "\n" for r in rows]


def _write_10s(path: Path, day: str, minutes: int, base=70000):
    cols = ["datetime", "open", "high", "low", "close", "volume", "market", "amount", "raw_bar"]
    lines = [",".join(cols)]
    t = datetime.strptime(day, "%Y%m%d").replace(hour=9, minute=0)
    for m in range(minutes):
        o = base + m * 10
        for k in range(6):
            ts = t + timedelta(minutes=m, seconds=10 * k)
            raw = 1 if k == 0 else 0
            lines.append(f"{ts:%Y-%m-%d %H:%M:%S},{o},{o + 50},{o - 50},{o + 20},{100.0},J,{(o * 600) if raw else ''},{raw}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_daily(path: Path, last_day: str, n=30, base=70000):
    rows = ["date,open,high,low,close,volume,amount,MA_5,RSI"]
    d = datetime.strptime(last_day, "%Y%m%d")
    days = []
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    for i, day in enumerate(reversed(days)):
        px = base + i * 10
        rows.append(f"{day:%Y%m%d},{px},{px + 100},{px - 100},{px + 20},{1000 + i},{(px + 20) * (1000 + i)},{px},50.0")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.data = self.root / "data"
        for d in (DAY, PREV):
            (self.data / d).mkdir(parents=True)
            _write_10s(self.data / d / f"{CODE}_{NAME}_10s.txt", d, 390 if d == PREV else 30)
            _write_daily(self.data / d / f"{CODE}_{NAME}_daily.csv", d)
        (self.data / "ticks" / DAY).mkdir(parents=True)
        self.tick_path = self.data / "ticks" / DAY / f"{CODE}.csv"
        self.lines = _tick_lines(DAY, 700)
        self.tick_path.write_text("".join(self.lines), encoding="utf-8")
        self.patch = mock.patch.object(r001, "RUNTIME_ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.td.cleanup()

    def cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = r017.main(list(argv))
        return rc, json.loads(buf.getvalue())


class BarsTests(Base):
    def test_tick_bars_and_sessions(self):
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "tick", "--ticks", "60")
        self.assertEqual(rc, 0, out)
        self.assertEqual(out["count"], 12)                      # 700 regular prints / 60 = 11 full + 1 partial
        first = out["bars"][0]
        self.assertEqual(first[0], f"2026-11-05T09:00:00")
        self.assertEqual(first[5], sum(1 + (i % 3) for i in range(60)))
        self.assertEqual(out["sessions"], "regular+close_auction")
        self.assertEqual(out["next_since"], out["bars"][-1][0])

    def test_minute_bars_incremental_cache_matches_full(self):
        full = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "min")[1]["bars"]
        # rebuild the file piece by piece (with a torn trailing line) and compare with the one-shot aggregation
        for cut in (2, 150, 151, 400, 699, len(self.lines)):
            self.tick_path.write_text("".join(self.lines[:cut]) + (self.lines[cut][:12] if cut < len(self.lines) else ""), encoding="utf-8")
            rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "min")
            self.assertEqual(rc, 0)
        self.assertEqual(out["bars"], full)
        self.assertTrue((self.root / "cache" / "bars" / DAY / f"{CODE}_min1.json").is_file())
        # unchanged file -> served from cache
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "min")
        self.assertTrue(out["source"].endswith("(cache)"))
        self.assertEqual(out["bars"], full)
        # since-ts returns the tail only
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "min", "--since-ts", full[-2][0])
        self.assertEqual([b[0] for b in out["bars"]], [full[-2][0], full[-1][0]])
        self.assertEqual(out["warm"], [])

    def test_tick_bars_incremental_matches_full_and_replay_is_dropped(self):
        full = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "tick", "--ticks", "7")[1]["bars"]
        for cut in (1, 8, 9, 300, len(self.lines)):
            self.tick_path.write_text("".join(self.lines[:cut]), encoding="utf-8")
            self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "tick", "--ticks", "7")
        # a reconnect replays the last 20 prints: acml does not advance -> ignored
        with open(self.tick_path, "a", encoding="utf-8") as fh:
            fh.write("".join(self.lines[-20:]))
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "tick", "--ticks", "7")
        self.assertEqual(out["bars"], full)

    def test_v1_file_without_session_column_and_all_sessions(self):
        lines = _tick_lines(DAY, 50, start=(8, 30, 0), with_session=False) + _tick_lines(DAY, 50, start=(9, 0, 0), with_session=False)[1:]
        self.tick_path.write_text("".join(lines), encoding="utf-8")
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "min")
        self.assertEqual(out["bars"][0][0], "2026-11-05T09:00:00")      # pre-market prints excluded by time
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "min", "--all-sessions")
        self.assertEqual(out["bars"][0][0], "2026-11-05T08:30:00")

    def test_minute_bars_from_10s_when_no_tick_file_and_warmup(self):
        rc, out = self.cli("bars", "--date", PREV, "--code", CODE, "--kind", "min", "--n", "2")
        self.assertEqual(rc, 0, out)
        self.assertIn("_10s.txt", out["source"])
        self.assertEqual(out["bars"][-1][5], 600)                   # 100 per 10 s row x 6
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "min", "--warmup", "5")
        self.assertEqual(out["warm_date"], PREV)
        self.assertEqual(len(out["warm"]), 5)

    def test_day_week_month_from_folder_and_history(self):
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "day", "--n", "3")
        self.assertEqual(rc, 0, out)
        self.assertEqual(out["count"], 3)
        self.assertEqual(out["bars"][-1][0], "2026-11-05T00:00:00")
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "week")
        self.assertTrue(all(datetime.fromisoformat(b[0]).weekday() == 0 for b in out["bars"]), out["bars"])
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "month")
        self.assertTrue(all(b[0].endswith("-01T00:00:00") for b in out["bars"]))
        hist = self.data / "daily_history"
        hist.mkdir()
        _write_daily(hist / f"{CODE}.csv", DAY, n=500)
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "day")
        self.assertEqual(out["count"], 500)
        self.assertIn("daily_history", out["source"])

    def test_errors(self):
        rc, out = self.cli("bars", "--date", DAY, "--code", "000000", "--kind", "tick")
        self.assertEqual(rc, 2)
        self.assertFalse(out["ok"])
        rc, out = self.cli("bars", "--date", DAY, "--code", CODE, "--kind", "min", "--since-ts", "nonsense")
        self.assertEqual(rc, 2)


class IndicatorTests(Base):
    def test_3m_and_1m_with_warmup(self):
        rc, out = self.cli("indicators", "--date", DAY, "--code", CODE, "--frame", "3m", "--last", "3")
        self.assertEqual(rc, 0, out)
        self.assertTrue(out["warmup"].startswith("prior_day=20261104"), out["warmup"])
        self.assertEqual(out["columns"][:7], ["ts", "ts_end", "open", "high", "low", "close", "volume"])
        for c in ("BB_UPPER", "RSI", "MACD_HIST", "ADX", "DI_PLUS"):
            self.assertIn(c, out["columns"])
        row = out["rows"][-1]
        self.assertEqual(datetime.fromisoformat(row[1]) - datetime.fromisoformat(row[0]), timedelta(minutes=3))
        self.assertEqual(out["params"]["BB_PERIOD"], r001.BB_PERIOD)
        rc, out1 = self.cli("indicators", "--date", DAY, "--code", CODE, "--frame", "1m")
        self.assertEqual(len(out1["rows"]), 30)
        self.assertEqual(out1["rows"][0][0], "2026-11-05T09:00:00")
        self.assertEqual(out1["rows"][0][1], "2026-11-05T09:01:00")
        # same input + same function as the bot: recompute directly
        m1, m1_end, m3, _ = r017.minute_frames(DAY, CODE)
        w1, w3, _ = r017.warmup_frames(DAY, CODE)
        import r002_strategy_core_shared as r002
        ref = r002.calculate_indicators_with_warmup(m3, w3)
        self.assertAlmostEqual(float(ref["RSI"].iloc[-1]), out["rows"][-1][out["columns"].index("RSI")], places=5)

    def test_daily_and_errors(self):
        rc, out = self.cli("indicators", "--date", DAY, "--code", CODE, "--frame", "daily", "--last", "2")
        self.assertEqual(rc, 0)
        self.assertEqual(len(out["rows"]), 2)
        self.assertIn("RSI", out["columns"])
        rc, out = self.cli("indicators", "--date", "20250101", "--code", CODE, "--frame", "3m")
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()

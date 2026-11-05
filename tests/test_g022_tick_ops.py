# -*- coding: utf-8 -*-
"""Tests for the unattended tick / edge-AI jobs: r014 calendar, g011 lock / torn rows / idle window,
r009 torn-row skip, g013 g001-done gate, g024 checks, g022 calendar window (bash)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import date, datetime, time as dtime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

import r014_market_calendar as cal  # noqa: E402
import g011_tick_collector as g011  # noqa: E402
from r009_tick_bars import TICK_FILE_COLUMNS, build_tick_bars, iter_tick_file  # noqa: E402


def _cal_file(d: str, text: str) -> Path:
    p = Path(d) / "cal.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _rec(code="005930", hhmmss="090001", px="100", q="5", acml="5"):
    return {"code": code, "bsop_date": "20261102", "trade_time": hhmmss, "price": px, "cntg_vol": q,
            "acml_vol": acml, "ccld_dvsn": "1", "seln_cntg_csnu": "0", "shnu_cntg_csnu": "1",
            "askp1": "101", "bidp1": "100", "hour_cls_code": "0"}


class CalendarTest(unittest.TestCase):
    def test_default_special_closed_weekend_and_bad_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = _cal_file(d, "# c\n20261119 10:00 16:30  # csat\n20261225 CLOSED\n20261126 xx:yy 1\nbad line\n"
                             "20261119 09:00 15:30\n")
            self.assertEqual(cal.session_hours("20261119", p), (dtime(10, 0), dtime(16, 30)))
            self.assertIsNone(cal.session_hours(date(2026, 12, 25), p))
            self.assertEqual(cal.session_hours("20261126", p), (dtime(9, 0), dtime(15, 30)))  # malformed -> normal
            self.assertIsNone(cal.session_hours("20261225", p))  # CLOSED (case-insensitive)
            p2 = _cal_file(d, "20261119 bad\n20261119 10:00 16:30\n")  # malformed first line still wins (= awk)
            self.assertEqual(cal.session_hours("20261119", p2), (dtime(9, 0), dtime(15, 30)))
            self.assertIsNone(cal.session_hours(datetime(2026, 11, 7, 10, 0), p))  # Saturday
            self.assertEqual(cal.session_hours("20261102", Path(d) / "missing.txt"), (dtime(9, 0), dtime(15, 30)))

    def test_shipped_calendar_parses(self):
        self.assertIn("20261119", cal.load_calendar())


class G011Test(unittest.TestCase):
    def test_single_instance_lock(self):
        with tempfile.TemporaryDirectory() as d:
            first = g011.acquire_single_instance_lock(Path(d))
            self.assertIsNotNone(first)
            # a second process must fail while the first holds it
            code = ("import sys; sys.path[:0]=[%r,%r]; import g011_tick_collector as g; from pathlib import Path; "
                    "sys.exit(0 if g.acquire_single_instance_lock(Path(%r)) is None else 1)"
                    % (str(ROOT / "live_trading"), str(ROOT / "data_simulation"), d))
            self.assertEqual(subprocess.run([sys.executable, "-c", code], capture_output=True).returncode, 0)
            first.close()
            again = g011.acquire_single_instance_lock(Path(d))
            self.assertIsNotNone(again)
            again.close()

    def test_torn_row_is_closed_and_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            w = g011.TickWriter(Path(d))
            w.write("2026-11-02T09:00:01.000", _rec(acml="5"))
            w.close()
            path = Path(d) / "005930.csv"
            with open(path, "a", encoding="utf-8") as f:
                f.write("2026-11-02T09:00:02.000,005930,20261102,090002,101")  # crash mid-row
            w2 = g011.TickWriter(Path(d))
            w2.write("2026-11-02T09:00:03.000", _rec(hhmmss="090003", px="102", q="3", acml="8"))
            w2.close()
            self.assertEqual(w2.torn_rows, 1)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0].split(","), TICK_FILE_COLUMNS)
            self.assertEqual(len(lines), 4)  # header, row, torn, row (next print not glued to the torn one)
            b = build_tick_bars(iter_tick_file(path), "005930", 2)
            self.assertEqual(b.stats.accepted, 2)  # torn row is not read as a qty-0 / acml-0 print
            self.assertEqual(len(b.bars), 1)

    def test_session_classification_and_v2_columns(self):
        from datetime import time as dt_time
        cs = g011.classify_session
        self.assertEqual(cs("083005"), "pre")
        self.assertEqual(cs("090000"), "regular")
        self.assertEqual(cs("151959"), "regular")
        self.assertEqual(cs("153011"), "close_auction")
        self.assertEqual(cs("153959"), "close_auction")
        self.assertEqual(cs("154000"), "post")
        self.assertEqual(cs("160500", (dt_time(10, 0), dt_time(16, 30))), "regular")   # CSAT day
        self.assertEqual(cs("163005", (dt_time(10, 0), dt_time(16, 30))), "close_auction")
        self.assertEqual(cs("bad"), "")
        self.assertEqual(TICK_FILE_COLUMNS[-2:], ["venue", "session"])
        with tempfile.TemporaryDirectory() as d:
            w = g011.TickWriter(Path(d))
            w.write("2026-11-06T09:00:01.000", {**_rec(acml="5"), "venue": "KRX", "session": "regular"})
            w.close()
            lines = (Path(d) / "005930.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0].split(","), TICK_FILE_COLUMNS)
            self.assertTrue(lines[1].endswith(",KRX,regular"))
            b = build_tick_bars(iter_tick_file(Path(d) / "005930.csv"), "005930", 1)
            self.assertEqual(b.stats.accepted, 1)

    def test_writer_keeps_v1_width_of_existing_file(self):
        from r009_tick_bars import TICK_FILE_COLUMNS_V1
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "005930.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write(",".join(TICK_FILE_COLUMNS_V1) + "\n")
                f.write("2026-11-06T09:00:01.000,005930,20261102,090001,100,5,5,1,0,1,0,0,0\n")
            w = g011.TickWriter(Path(d))
            w.write("2026-11-06T09:00:02.000", {**_rec(hhmmss="090002", px="101", q="3", acml="8"), "venue": "KRX", "session": "regular"})
            w.close()
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines[0].split(",")), len(TICK_FILE_COLUMNS_V1))
            self.assertTrue(all(len(l.split(",")) == len(TICK_FILE_COLUMNS_V1) for l in lines[1:]), lines)
            b = build_tick_bars(iter_tick_file(path), "005930", 1)
            self.assertEqual(b.stats.accepted, 2)   # both rows readable, nothing dropped as a torn row

    def test_idle_window_and_timeout_shim(self):
        with mock.patch.object(g011, "session_hours", return_value=(dtime(10, 0), dtime(16, 30))):
            start, end = g011.idle_check_window(datetime(2026, 11, 19, 8, 0))
        self.assertEqual((start.time(), end.time()), (dtime(10, 2), dtime(16, 20)))
        with mock.patch.object(g011, "session_hours", return_value=None):
            self.assertIsNone(g011.idle_check_window(datetime(2026, 12, 25)))
        real = mock.Mock()
        shim = g011._RequestsWithTimeout(real)
        shim.post("u", data="x")
        self.assertEqual(real.post.call_args.kwargs["timeout"], g011.AUTH_TIMEOUT_SECONDS)
        shim.post("u", timeout=3)
        self.assertEqual(real.post.call_args.kwargs["timeout"], 3)
        self.assertIs(shim.exceptions, real.exceptions)


class _FakeWS:
    """Acks only the first subscribed code, then streams prints for it (the other code never gets an ACK)."""

    def __init__(self, codes):
        self.codes = codes
        self.sent = []
        self.n = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, msg):
        self.sent.append(msg)

    async def pong(self, raw):
        pass

    async def recv(self):
        import asyncio
        await asyncio.sleep(0.02)
        self.n += 1
        if self.n == 1:
            return json.dumps({"header": {"tr_id": "H0STCNT0", "tr_key": self.codes[0]},
                               "body": {"rt_cd": "0", "msg1": "SUBSCRIBE SUCCESS"}})
        f = [""] * g011.NCOL
        f[g011._IDX["MKSC_SHRN_ISCD"]] = self.codes[0]
        f[g011._IDX["STCK_CNTG_HOUR"]] = "100000"
        f[g011._IDX["STCK_PRPR"]] = "100"
        f[g011._IDX["CNTG_VOL"]] = "1"
        f[g011._IDX["ACML_VOL"]] = str(self.n)
        f[g011._IDX["BSOP_DATE"]] = "20261102"
        return "0|H0STCNT0|001|" + "^".join(f)


class G011AckTest(unittest.TestCase):
    def test_missing_ack_reconnects_then_keeps_connection(self):
        import asyncio
        import types
        codes = ["005930", "000660"]
        conns = []

        def connect(url, **kw):
            ws = _FakeWS(codes)
            conns.append(ws)
            return ws

        fake_ka = types.SimpleNamespace(auth_ws=lambda: None, requests=mock.Mock(),
                                        getTREnv=lambda: types.SimpleNamespace(my_url_ws="ws://x"))
        fake_ws_mod = types.SimpleNamespace(ccnl_krx=lambda tr, code: ({"code": code}, []))
        logs = []
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(sys.modules, {"websockets": types.SimpleNamespace(connect=connect),
                                              "kis_auth": fake_ka, "domestic_stock_functions_ws": fake_ws_mod}), \
                mock.patch.object(g011, "SUB_ACK_TIMEOUT_SECONDS", 0.3), \
                mock.patch.object(g011, "idle_check_window", return_value=None), \
                mock.patch.object(g011.random, "uniform", return_value=0.05):
            until = datetime.now() + __import__("datetime").timedelta(seconds=3.0)
            asyncio.run(g011.run_collector(codes, until, Path(d), logs.append))
        reconnects = [m for m in logs if m.startswith("DISCONNECTED") and "no subscribe ACK" in m]
        self.assertEqual(len(reconnects), g011.MAX_ACK_RECONNECTS, logs)
        self.assertIn("000660", reconnects[0])
        self.assertTrue(any(m.startswith("WARN still no subscribe ACK") for m in logs), logs)
        self.assertEqual(len(conns), g011.MAX_ACK_RECONNECTS + 1)  # last connection kept until `until`


class G013DoneGateTest(unittest.TestCase):
    def test_marker_rules(self):
        import g013_edge_ai_dataset as g013
        with tempfile.TemporaryDirectory() as d:
            legacy = Path(d) / "20261030"
            new = Path(d) / "20261103"
            for day in (legacy, new):
                day.mkdir()
                f = day / "000001_x_10s.txt"
                f.write_text("x", encoding="utf-8")
                old = time.time() - 3600
                os.utime(f, (old, old))
            self.assertTrue(g013._day_settled(legacy, 30))
            self.assertFalse(g013._day_settled(new, 30))  # no marker
            marker = new / g013.G001_DONE_MARKER
            for bad in ({"scope": "full", "failed": 2}, {"failed": 0}, {}, [], {"scope": "full", "failed": "bad"},
                        {"scope": "full", "failed": True}):
                marker.write_text(json.dumps(bad), encoding="utf-8")
                self.assertFalse(g013._day_settled(new, 30), bad)
            marker.write_text("{not json", encoding="utf-8")
            self.assertFalse(g013._day_settled(new, 30))
            marker.write_text(json.dumps({"scope": "full", "failed": 0}), encoding="utf-8")
            self.assertTrue(g013._day_settled(new, 30))
            import fcntl
            with open(new / ".g001.lock", "a") as lk:  # g001 still collecting (another open file description)
                fcntl.flock(lk, fcntl.LOCK_EX)
                self.assertFalse(g013._day_settled(new, 30))
            self.assertTrue(g013._day_settled(new, 30))
            os.utime(new / "000001_x_10s.txt")  # rewritten just now -> not settled yet
            self.assertFalse(g013._day_settled(new, 30))


class G013SourceChangeTest(unittest.TestCase):
    def test_source_state_detects_rewrite(self):
        import g013_edge_ai_dataset as g013
        with tempfile.TemporaryDirectory() as d:
            day = Path(d) / "20261103"
            day.mkdir()
            f = day / "000001_x_10s.txt"
            f.write_text("x", encoding="utf-8")
            (day / g013.G001_DONE_MARKER).write_text(json.dumps({"scope": "full", "failed": 0}), encoding="utf-8")
            before = g013._source_state(day)
            self.assertEqual(before, g013._source_state(day))
            (day / g013.G001_DONE_MARKER).unlink()  # g001 rerun starts
            self.assertNotEqual(before, g013._source_state(day))


class G024Test(unittest.TestCase):
    def test_tick_checks(self):
        import g024_ops_check as g024
        with tempfile.TemporaryDirectory() as d:
            tick_dir = Path(d) / "ticks" / "20261102"
            tick_dir.mkdir(parents=True)
            with mock.patch.object(g024, "RUNTIME_DATA_DIR", Path(d)), \
                    mock.patch.object(g024, "session_hours", return_value=(dtime(9, 0), dtime(15, 30))):
                problems, _ = g024.check_ticks(datetime(2026, 11, 2))
                self.assertTrue(any("틱 파일 없음" in p for p in problems))
                w = g011.TickWriter(tick_dir)
                w.write("2026-11-02T09:00:01.000", _rec())
                w.close()
                close_ts = datetime(2026, 11, 2, 15, 29).timestamp()
                os.utime(tick_dir / "005930.csv", (close_ts, close_ts))
                (tick_dir / "_collector.log").write_text(
                    "2026-11-02 08:50:00 START codes=1\n2026-11-02 08:50:00 CODES 005930\n"
                    "2026-11-02 08:50:02 SUBSCRIPTIONS ok=1 failed=0 []\n", encoding="utf-8")
                problems, info = g024.check_ticks(datetime(2026, 11, 2))
                self.assertEqual(problems, [], info)
                with open(tick_dir / "_collector.log", "a", encoding="utf-8") as f:  # 3 more requested, never printed
                    f.write("2026-11-02 10:00:00 CODES 005930,000660,035420,051910\n")
                problems, _ = g024.check_ticks(datetime(2026, 11, 2))
                self.assertTrue(any("파일 없음 포함" in p for p in problems), problems)
                (tick_dir / "_collector.log").write_text(
                    "2026-11-02 08:50:00 CODES 005930\n", encoding="utf-8")
                early = datetime(2026, 11, 2, 11, 0).timestamp()
                os.utime(tick_dir / "005930.csv", (early, early))
                problems, _ = g024.check_ticks(datetime(2026, 11, 2))
                self.assertTrue(any("수집 중단" in p for p in problems))
            with mock.patch.object(g024, "session_hours", return_value=None):
                problems, info = g024.check_ticks(datetime(2026, 12, 25))
                self.assertEqual(problems, [])
                self.assertIn("closed", info[0])

    def test_edge_ai_checks(self):
        import g024_ops_check as g024
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(g024, "EDGE_AI_DIR", Path(d)):
                problems, _ = g024.check_edge_ai(0, 1, None)
                self.assertEqual(len(problems), 2)  # g014 failed + no model
                (Path(d) / "current_meta.json").write_text("{}", encoding="utf-8")
                self.assertEqual(g024.check_edge_ai(0, 0, None)[0], [])


class G022WindowTest(unittest.TestCase):
    """Runs g022 with a fake date/pgrep/setsid on PATH and checks whether it would start g011."""

    def _run(self, today: str, hhmm: str, dow: int, cal_text: str) -> bool:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sim = d / "data_simulation"
            sim.mkdir()
            (d / "live_trading").mkdir()
            (d / "live_trading" / "r014_market_calendar.txt").write_text(cal_text, encoding="utf-8")
            script = sim / "g022_tick_collect_daily.sh"
            script.write_text((ROOT / "data_simulation" / "g022_tick_collect_daily.sh").read_text(encoding="utf-8"),
                              encoding="utf-8")
            binp = d / "bin"
            binp.mkdir()
            marker = d / "started"
            (binp / "date").write_text(
                f'#!/bin/bash\ncase "$1" in\n  +%Y%m%d) echo {today};;\n  +%H:%M) echo {hhmm};;\n'
                f'  +%u) echo {dow};;\n  *) /usr/bin/date "$@";;\nesac\n', encoding="utf-8")
            (binp / "pgrep").write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
            (binp / "setsid").write_text(f"#!/bin/bash\ntouch {marker}\n", encoding="utf-8")
            for f in binp.iterdir():
                f.chmod(0o755)
            env = dict(os.environ, PATH=f"{binp}:{os.environ['PATH']}", AUTO_TRADING_RUNTIME=str(d / "rt"))
            r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            time.sleep(0.2)
            return marker.exists()

    def test_window(self):
        cal_text = "20261119 10:00 16:30\n20261225 closed\n"
        self.assertTrue(self._run("20261102", "08:45", 1, cal_text))
        self.assertFalse(self._run("20261102", "08:44", 1, cal_text))
        self.assertFalse(self._run("20261102", "15:30", 1, cal_text))
        self.assertFalse(self._run("20261119", "09:00", 4, cal_text))   # CSAT: opens 10:00 -> window from 09:45
        self.assertTrue(self._run("20261119", "16:10", 4, cal_text))
        self.assertFalse(self._run("20261225", "10:00", 5, cal_text))   # holiday
        self.assertFalse(self._run("20261107", "10:00", 6, cal_text))   # Saturday


if __name__ == "__main__":
    unittest.main()

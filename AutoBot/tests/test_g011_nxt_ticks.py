# -*- coding: utf-8 -*-
"""g011 NXT executions (2026-11-12): frame parsing per TR, the time-of-day subscription plan under the 40-slot session
cap, separate nxt/ files, and the wider r016 collector window."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

import g011_tick_collector as g011  # noqa: E402
import r016_control as r016  # noqa: E402

DAY = datetime(2026, 11, 12)


def at(h, m=0):
    return DAY.replace(hour=h, minute=m)


def frame(tr, code, hhmmss, extra_cols=0):
    f = [""] * (g011.NCOL + extra_cols)
    f[g011._IDX["MKSC_SHRN_ISCD"]] = code
    f[g011._IDX["STCK_CNTG_HOUR"]] = hhmmss
    f[g011._IDX["STCK_PRPR"]] = "100"
    f[g011._IDX["CNTG_VOL"]] = "1"
    f[g011._IDX["ACML_VOL"]] = "10"
    f[g011._IDX["BSOP_DATE"]] = "20261112"
    return f"0|{tr}|001|" + "^".join(f)


class ParseTest(unittest.TestCase):
    def test_venue_and_session_per_tr(self):
        krx = g011.parse_frame(frame("H0STCNT0", "005930", "153005"))[0]
        self.assertEqual((krx["venue"], krx["session"]), ("KRX", "close_auction"))
        nxt = g011.parse_frame(frame("H0NXCNT0", "005930", "181715", extra_cols=1))   # NXT records carry one more column
        self.assertEqual((nxt[0]["code"], nxt[0]["venue"], nxt[0]["session"], nxt[0]["price"]), ("005930", "NXT", "post", "100"))
        self.assertEqual(g011.parse_frame(frame("H0NXCNT0", "005930", "081000", 1))[0]["session"], "pre")
        self.assertEqual(g011.parse_frame(frame("H0NXCNT0", "005930", "101000", 1))[0]["session"], "regular")
        self.assertEqual(g011.parse_frame(frame("H0UNCNT0", "005930", "101000")), [])   # other TRs are ignored


class PlanTest(unittest.TestCase):
    WIN = (at(8), at(20))
    KRX_END = at(15, 31)

    def plan(self, now, krx, nxt, cap=40):
        return g011.subscription_plan(now, krx, nxt, self.KRX_END, self.WIN, cap)

    def test_nxt_only_outside_krx_hours_and_both_while_slots_are_left(self):
        watch = [f"{i:06d}" for i in range(12)]
        krx = watch + [f"9{i:05d}" for i in range(16)]                                # watchlist + scan fill
        day = self.plan(at(10), krx, watch)
        self.assertEqual(sum(tr == "H0STCNT0" for tr, _ in day), 28)
        self.assertEqual([c for tr, c in day if tr == "H0NXCNT0"], watch)             # 12 slots were left
        evening = self.plan(at(18), krx, watch)
        self.assertEqual(evening, [("H0NXCNT0", c) for c in watch])                   # KRX ended at 15:31
        self.assertEqual(self.plan(at(20), krx, watch), [])                           # NXT window is [start, end)
        self.assertEqual(self.plan(at(7, 59), [], watch), [])

    def test_krx_keeps_priority_under_the_cap(self):
        krx = [f"{i:06d}" for i in range(35)]
        nxt = [f"{i:06d}" for i in range(20)]
        day = self.plan(at(10), krx, nxt)
        self.assertEqual(len(day), 40)
        self.assertEqual([c for tr, c in day if tr == "H0NXCNT0"], nxt[:5])           # watchlist order, only what fits
        self.assertEqual(len(self.plan(at(10), [f"{i:06d}" for i in range(40)], nxt)), 40)
        self.assertEqual(len(self.plan(at(18), krx, [f"{i:06d}" for i in range(45)])), 40)

    def test_no_nxt_window_is_the_old_behaviour(self):
        self.assertEqual(g011.subscription_plan(at(10), ["000001"], ["000001"], self.KRX_END, None), [("H0STCNT0", "000001")])


class _WS:
    """Fake session: answers every NXT subscribe, then sends one KRX and one NXT execution."""

    def __init__(self):
        self.sent, self.n = [], 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    async def pong(self, raw):
        pass

    async def recv(self):
        await asyncio.sleep(0.02)
        self.n += 1
        if self.n == 1:
            return json.dumps({"header": {"tr_id": "H0STCNT0", "tr_key": "000001"}, "body": {"rt_cd": "0", "msg1": "SUBSCRIBE SUCCESS"}})
        if self.n == 2:
            return json.dumps({"header": {"tr_id": "H0NXCNT0", "tr_key": "000002"}, "body": {"rt_cd": "1", "msg1": "invalid tr_key"}})
        return frame("H0STCNT0", "000001", "100000") if self.n % 2 else frame("H0NXCNT0", "000001", "100001", 1)


class RunTest(unittest.TestCase):
    def _run(self, krx_until_in, nxt_window, seconds=1.2, nxt_codes=("000001", "000002")):
        conns, logs = [], []

        def connect(url, **kw):
            conns.append(_WS())
            return conns[-1]

        fake_ka = types.SimpleNamespace(auth_ws=lambda: None, requests=mock.Mock(), getTREnv=lambda: types.SimpleNamespace(my_url_ws="ws://x"))
        fake_ws_mod = types.SimpleNamespace(ccnl_krx=lambda tr, code: ({"tr": "H0STCNT0", "type": tr, "code": code}, []),
                                            ccnl_nxt=lambda tr, code: ({"tr": "H0NXCNT0", "type": tr, "code": code}, []))
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(sys.modules, {"websockets": types.SimpleNamespace(connect=connect), "kis_auth": fake_ka,
                                              "domestic_stock_functions_ws": fake_ws_mod}), \
                mock.patch.object(g011, "idle_check_window", return_value=None), \
                mock.patch.object(g011, "SUBSCRIPTION_SYNC_SECONDS", 0.1), \
                mock.patch.object(g011.random, "uniform", return_value=0.05):
            now = datetime.now()
            win = (now + timedelta(seconds=nxt_window[0]), now + timedelta(seconds=nxt_window[1])) if nxt_window else None
            asyncio.run(g011.run_collector(["000001"], now + timedelta(seconds=krx_until_in), Path(d), logs.append,
                                           nxt_codes=list(nxt_codes), nxt_window=win))
            files = sorted(str(p.relative_to(d)).replace("\\", "/") for p in Path(d).rglob("*.csv"))
            rows = {f: (Path(d) / f).read_text(encoding="utf-8").splitlines() for f in files}
        return conns, logs, files, rows

    def test_both_feeds_go_to_their_own_files(self):
        conns, logs, files, rows = self._run(krx_until_in=5, nxt_window=(-1, 1.2))
        self.assertEqual(files, ["000001.csv", "nxt/000001.csv"])
        self.assertTrue(all(",KRX," in r for r in rows["000001.csv"][1:]))
        self.assertTrue(all(",NXT," in r for r in rows["nxt/000001.csv"][1:]))
        first = [(m["tr"], m["type"], m["code"]) for m in conns[0].sent[:3]]
        self.assertEqual(first, [("H0STCNT0", "1", "000001"), ("H0NXCNT0", "1", "000001"), ("H0NXCNT0", "1", "000002")])
        self.assertTrue(any(m.startswith("NXT SUBSCRIBE ERROR 000002") for m in logs), logs)
        self.assertTrue(any("nxt_rows=" in m for m in logs if m.startswith("STOPPED")), logs)

    def test_krx_is_unsubscribed_at_its_end_and_nxt_keeps_running(self):
        conns, logs, files, rows = self._run(krx_until_in=0.4, nxt_window=(-1, 1.2))
        sent = [(m["tr"], m["type"], m["code"]) for m in conns[0].sent]
        self.assertIn(("H0STCNT0", "0", "000001"), sent)                              # unsubscribed when the KRX end passed
        self.assertNotIn(("H0NXCNT0", "0", "000001"), sent)
        self.assertEqual(len(conns), 1)                                               # same session, no reconnect

    def test_nxt_starts_later_on_the_same_session(self):
        conns, logs, files, rows = self._run(krx_until_in=5, nxt_window=(0.4, 1.2))
        sent = [(m["tr"], m["type"], m["code"]) for m in conns[0].sent]
        self.assertEqual(sent[0], ("H0STCNT0", "1", "000001"))
        self.assertIn(("H0NXCNT0", "1", "000001"), sent[1:])
        self.assertEqual(len(conns), 1)

    def test_without_nxt_nothing_changes(self):
        conns, logs, files, rows = self._run(krx_until_in=0.6, nxt_window=None, nxt_codes=())
        self.assertEqual(files, ["000001.csv"])
        self.assertEqual([(m["tr"], m["type"]) for m in conns[0].sent], [("H0STCNT0", "1")])
        self.assertFalse(any("nxt_rows" in m or m.startswith("SUBSCRIBED") for m in logs), logs)


class WindowTest(unittest.TestCase):
    HOURS = (dt_time(9, 0), dt_time(15, 30))

    def _ok(self, when, nxt):
        with mock.patch("r014_market_calendar.session_hours", return_value=self.HOURS), \
                mock.patch.object(r016, "collector_nxt_hours", return_value=nxt):
            return r016.collector_window(when)[0]

    def test_krx_only_window_is_unchanged(self):
        self.assertFalse(self._ok(at(8, 44), None))
        self.assertTrue(self._ok(at(8, 45), None))
        self.assertFalse(self._ok(at(15, 30), None))

    def test_nxt_widens_it_to_the_nxt_sessions(self):
        nxt = (dt_time(8, 0), dt_time(20, 0))
        self.assertFalse(self._ok(at(7, 54), nxt))
        self.assertTrue(self._ok(at(7, 55), nxt))
        self.assertTrue(self._ok(at(18, 0), nxt))
        self.assertFalse(self._ok(at(20, 0), nxt))
        self.assertFalse(self._ok(datetime(2026, 11, 14, 10, 0), nxt))                # Saturday

    def test_config_switch(self):
        import r001_define_config as cfg
        with mock.patch.object(cfg, "TICK_COLLECT_NXT", False):
            self.assertIsNone(r016.collector_nxt_hours())
        with mock.patch.object(cfg, "TICK_COLLECT_NXT", True):
            self.assertEqual(r016.collector_nxt_hours(), (cfg.MORNING_NXT_START, cfg.AFTERNOON_NXT_END))


if __name__ == "__main__":
    unittest.main()

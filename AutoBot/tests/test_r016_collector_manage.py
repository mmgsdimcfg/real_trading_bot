# -*- coding: utf-8 -*-
"""r016 collector detail / extra / prune / stop (2026-11-16, AutoBotH 수집기 tab)."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

with contextlib.redirect_stdout(io.StringIO()):
    import r016_control as r016  # noqa: E402

HEADER = "recv_ts,bsop_date,trade_time,price,cntg_vol,acml_vol,ccld_dvsn,seln_cntg_csnu,shnu_cntg_csnu,askp1,bidp1,hour_cls_code,code,venue,session\n"


def tick_rows(times):
    return "".join(f"2026-11-16 {t[:2]}:{t[2:4]}:{t[4:]}.1,20261116,{t},100,1,1,1,1,1,101,100,0,005930,KRX,regular\n" for t in times)


class CollectorManageTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        (root / "live_trading").mkdir()
        self.paths = r016.ControlPaths(runtime_root=root / "runtime", live_trading_dir=root / "live_trading")
        self.paths.watchlist.write_text("# h\n005930,삼성전자\n000660,SK하이닉스\n", encoding="utf-8")
        self.now = datetime(2026, 11, 16, 10, 0, 0)
        day = self.paths.ticks_dir / "20261116"
        (day / "nxt").mkdir(parents=True)
        (day / "005930.csv").write_text(HEADER + tick_rows(["090001", "093000", "095959"]), encoding="utf-8")
        (day / "111111.csv").write_text(HEADER + tick_rows(["090010"]), encoding="utf-8")
        (day / "nxt" / "005930.csv").write_text(HEADER + tick_rows(["080100"]), encoding="utf-8")
        (day / "_collector.log").write_text(
            "2026-11-16 07:53:12 START codes=2 until=15:31 out=x\n2026-11-16 07:53:12 CODES 005930,000660\n"
            "2026-11-16 10:05:27 START codes=4 (watchlist 2 + extra 1 + scan 20261113 1) until=15:31 out=x\n"
            "2026-11-16 10:05:27 CODES 005930,000660,222222,111111\n"
            "2026-11-16 10:05:27 ROLES watch=005930,000660 extra=222222 scan=111111\n"
            "2026-11-16 10:05:27 NXT codes=2 window=08:00~20:00 out=x (KRX session: NXT only for held ['005930']; before ...)\n"
            "2026-11-16 10:05:30 SUBSCRIBED krx=4 nxt=1 (+5 -0)\n2026-11-16 10:05:30 CONNECTED url=ws requested=5\n", encoding="utf-8")
        for old in ("20261030", "20261102"):
            (self.paths.ticks_dir / old).mkdir()
            (self.paths.ticks_dir / old / "005930.csv").write_text(HEADER + tick_rows(["090001"]), encoding="utf-8")
        self.hist = mock.patch.object(r016, "append_history", lambda *a, **k: None)
        self.hist.start()

    def tearDown(self):
        self.hist.stop()
        self.td.cleanup()

    def test_detail_plan_roles_files_and_history(self):
        with mock.patch.object(r016, "_g011_processes", lambda paths: []):
            d = r016.collector_detail(self.paths, "20261116", 30, now=self.now)
        self.assertTrue(d["ok"])
        self.assertEqual(d["plan"]["codes"], ["005930", "000660", "222222", "111111"])      # the newest START's list
        self.assertEqual(d["plan"]["subscribed"]["krx"], 4)
        self.assertEqual(d["plan"]["nxt_keep"], ["005930"])
        by = {s["code"]: s for s in d["symbols"]}
        self.assertEqual([by[c]["role"] for c in ("005930", "000660", "222222", "111111")], ["watch", "watch", "extra", "scan"])
        self.assertEqual(by["005930"]["name"], "삼성전자")
        self.assertEqual((by["005930"]["krx"]["first"], by["005930"]["krx"]["last"]), ("090001", "095959"))
        self.assertEqual(by["005930"]["krx"]["rows_est"], 3)
        self.assertIsNotNone(by["005930"]["nxt"])
        self.assertIsNone(by["000660"]["krx"])
        self.assertTrue(by["000660"]["stale"])                                                # in session, planned, no file
        self.assertEqual([x["date"] for x in d["days"]], ["20261116", "20261102", "20261030"])
        self.assertEqual(d["days"][0]["krx"], 2)
        self.assertEqual(d["days"][0]["nxt"], 1)
        self.assertGreater(d["total_size"], 0)

    def test_extra_codes_roundtrip_and_validation(self):
        self.assertEqual(r016.read_tick_extra(self.paths)["codes"], [])
        out = r016.write_tick_extra(self.paths, ["5930", "000660", "000660"], "test")
        self.assertEqual(out["codes"], ["005930", "000660"])
        self.assertEqual(json.loads(r016.tick_extra_path(self.paths).read_text(encoding="utf-8"))["codes"], ["005930", "000660"])
        with self.assertRaises(ValueError):
            r016.write_tick_extra(self.paths, ["1234567"], "test")   # 7 characters
        with self.assertRaises(ValueError):
            r016.write_tick_extra(self.paths, ["00593-"], "test")
        with self.assertRaises(ValueError):
            r016.write_tick_extra(self.paths, [f"{i:06d}" for i in range(41)], "test")
        r016.write_tick_extra(self.paths, [], "test")
        self.assertEqual(r016.read_tick_extra(self.paths)["codes"], [])

    def test_prune_keeps_the_minimum_and_never_today(self):
        self.assertFalse(r016.collector_prune(self.paths, 5, False, "t", now=self.now)["ok"])   # below the 30-day floor
        with mock.patch.object(r016, "PRUNE_MIN_KEEP_DAYS", 1):
            plan = r016.collector_prune(self.paths, 1, False, "t", now=self.now)
            self.assertEqual([p["date"] for p in plan["delete"]], ["20261030"])                # today not counted, newest kept
            self.assertTrue((self.paths.ticks_dir / "20261030").exists())                       # preview only
            done = r016.collector_prune(self.paths, 1, True, "t", now=self.now)
            self.assertEqual(done["removed"], ["20261030"])
            self.assertFalse((self.paths.ticks_dir / "20261030").exists())
            self.assertTrue((self.paths.ticks_dir / "20261116").exists())

    def test_stop_without_a_collector(self):
        with mock.patch.object(r016, "_g011_processes", lambda paths: []):
            self.assertEqual(r016.collector_stop(self.paths, "t")["stopped"], [])

    def test_subscribed_list_comes_from_the_log(self):
        got = r016._collector_codes_of(["python", "g011.py", "--codes", "005930"], self.paths.ticks_dir / "20261116")
        self.assertEqual(got, {"005930", "000660", "222222", "111111"})


if __name__ == "__main__":
    unittest.main()

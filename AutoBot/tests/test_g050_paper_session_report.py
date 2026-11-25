# -*- coding: utf-8 -*-
"""[2026-11-25] g050 paper session report: trade pairing, groups, the 15:15 control, small-sample marks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

import g050_paper_session_report as g050  # noqa: E402

HEADER = ("recv_ts,code,bsop_date,trade_time,price,cntg_vol,acml_vol,ccld_dvsn,seln_cntg_csnu,shnu_cntg_csnu,askp1,bidp1,"
          "hour_cls_code,venue,session")


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pairs_groups_and_the_1515_control(self):
        day = "20261125"
        rows = [
            {"event": "ENTRY", "ts": "2026-11-25T08:20:05", "code": "000001", "entry": 10010, "session": "nxt_pre", "nxt": True},
            {"event": "EXIT", "ts": "2026-11-25T19:59:50", "code": "000001", "reason": "close_nxt", "net_pct": 1.0, "held_min": 700},
            {"event": "ENTRY", "ts": "2026-11-25T14:00:05", "code": "000002", "entry": 10010},              # old record: regular
            {"event": "EXIT", "ts": "2026-11-25T15:20:00", "code": "000002", "reason": "close_regular", "net_pct": -0.5,
             "held_min": 80, "estimated": True},
            {"event": "ENTRY", "ts": "2026-11-25T10:00:05", "code": "000003", "entry": 10010},              # still open
            "{torn",
        ]
        (self.d / f"edge_only_{day}.jsonl").write_text(
            "\n".join(x if isinstance(x, str) else json.dumps(x) for x in rows) + "\n", encoding="utf-8")
        (self.d / "ticks" / day).mkdir(parents=True)
        (self.d / "ticks" / day / "000002.csv").write_text(
            HEADER + f"\n2026-11-25T00:00:00,000002,{day},151400,10300,1,1,1,0,0,0,0,0,KRX,regular\n"
            f"2026-11-25T00:00:00,000002,{day},151600,9000,1,2,1,0,0,0,0,0,KRX,regular\n", encoding="utf-8")
        trades, open_n = g050.load_trades(self.d, "00000000", "99999999")
        self.assertEqual((len(trades), open_n), (2, 1))
        self.assertEqual([(t.session, t.nxt) for t in trades], [("nxt_pre", True), ("regular", False)])
        ctl, missing = g050.control_1515(trades, self.d / "ticks")
        self.assertEqual((len(ctl), missing), (1, 1))                                 # 000001 qualifies too, but has no tick file
        import r022_edge_only as r022
        self.assertEqual(ctl[0][0].code, "000002")
        self.assertAlmostEqual(ctl[0][1], r022.net_pct(10010, 10290))                 # last print <= 15:15 (10,300) - 1 tick
        text = g050.report(trades, open_n, self.d / "ticks", "00000000", "99999999")
        self.assertIn("| NXT 장전 | 1 |", text)
        self.assertIn("판단 보류", text)
        self.assertIn("아직 열린 보유 1건", text)
        self.assertIn("| 추정 체결 (마지막 가격으로 계산) | 1 |", text)
        self.assertIn("| NXT 장전 / NXT 보유 / close_nxt | 1 |", text)                 # cross table
        self.assertIn("| 15:15 이후 청산 | 1 |", text)                                 # NXT holding exited after 15:15

    def test_incomplete_exits_and_prints_before_the_entry(self):
        day = "20261126"
        rows = [
            {"event": "ENTRY", "ts": "2026-11-26T15:10:05", "code": "000001", "entry": 10010},
            {"event": "EXIT", "ts": "2026-11-26T15:20:00", "code": "000001", "reason": "close_regular", "net_pct": 0.1},
            {"event": "ENTRY", "ts": "2026-11-26T10:00:05", "code": "000002", "entry": 10010},
            {"event": "EXIT", "ts": "2026-11-26T11:00:00", "code": "000002", "reason": "tp"},      # no net_pct
        ]
        (self.d / f"edge_only_{day}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        (self.d / "ticks" / day).mkdir(parents=True)
        (self.d / "ticks" / day / "000001.csv").write_text(
            HEADER + f"\n2026-11-26T00:00:00,000001,{day},150000,10300,1,1,1,0,0,0,0,0,KRX,regular\n"   # before the buy
            f"2026-11-26T00:00:00,000001,{day},151005,10300,1,2,1,0,0,0,0,0,KRX,regular\n", encoding="utf-8")  # the buy's own second
        trades, _ = g050.load_trades(self.d, "00000000", "99999999")
        self.assertEqual(len(trades), 1)
        self.assertEqual(g050.INCOMPLETE["n"], 1)
        self.assertEqual(g050.control_1515(trades, self.d / "ticks"), ([], 1))


if __name__ == "__main__":
    unittest.main()

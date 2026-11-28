# -*- coding: utf-8 -*-
"""[2026-11-28] g061 s1 shadow log: the record of a date is written once, v0 picks are copied, watch-json groups by the next trading day."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_simulation"))
sys.path.insert(0, str(ROOT / "live_trading"))

import g061_s1_shadow_log as g061  # noqa: E402


class S1ShadowLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "20261127").mkdir()
        (self.root / "20261127" / "_20261127_picks.txt").write_text("# header\n005930,삼성전자\n000660,SK하이닉스\n", encoding="utf-8")
        ps = [mock.patch.object(g061.g002, "RUNTIME_DATA_DIR", self.root),
              mock.patch.object(g061, "OUT_DIR", self.root / "pick_shadow_s1"),
              mock.patch.object(g061.g002, "_filter_profile", return_value="v0"),
              mock.patch.object(g061, "scan_profile", side_effect=lambda d, prof: {"eligible": 3, "picks": [
                  {"rank": 1, "code": "000660", "name": "SK하이닉스", "score": 90.0, "price": 1, "amount_ma20": 1, "vol_ratio": 2.0, "tick_ratio": 0.001},
                  {"rank": 2, "code": "035420", "name": "NAVER", "score": 80.0, "price": 1, "amount_ma20": 1, "vol_ratio": 1.5, "tick_ratio": 0.002}]})]
        for p in ps:
            p.start(); self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_record_holds_s1_and_v0_picks_and_the_next_trading_day(self):
        self.assertEqual(g061.log_date("20261127"), 0)
        rec = json.loads(g061.shadow_path("20261127").read_text(encoding="utf-8"))
        self.assertEqual(rec["for_trading_day"], "20261130")   # Friday -> Monday
        self.assertEqual([p["code"] for p in rec["picks"]], ["000660", "035420"])
        self.assertEqual(rec["v0_picks"], [{"code": "005930", "name": "삼성전자"}, {"code": "000660", "name": "SK하이닉스"}])

    def test_a_logged_date_is_not_rewritten_without_force(self):
        g061.log_date("20261127")
        first = g061.shadow_path("20261127").read_text(encoding="utf-8")
        g061.scan_profile.side_effect = lambda d, prof: {"eligible": 0, "picks": []}
        self.assertEqual(g061.log_date("20261127"), 0)
        self.assertEqual(g061.shadow_path("20261127").read_text(encoding="utf-8"), first)
        self.assertEqual(g061.log_date("20261127", force=True), 0)
        self.assertEqual(json.loads(g061.shadow_path("20261127").read_text(encoding="utf-8"))["picks"], [])

    def test_missing_folder_and_failed_scan_return_2_without_a_file(self):
        self.assertEqual(g061.log_date("20261126"), 2)
        g061.scan_profile.side_effect = SystemExit("no data")
        self.assertEqual(g061.log_date("20261127"), 2)
        self.assertFalse(g061.shadow_path("20261127").exists())

    def test_live_s1_takes_s1_from_the_picks_file_and_scans_v0(self):
        """[2026-11-28] live = s1: picks (s1) = the picks file, v0_picks = the scan under v0"""
        g061.g002._filter_profile.return_value = "s1"
        self.assertEqual(g061.log_date("20261127"), 0)
        self.assertEqual(g061.scan_profile.call_args[0], ("20261127", "v0"))
        rec = json.loads(g061.shadow_path("20261127").read_text(encoding="utf-8"))
        self.assertEqual((rec["live_profile"], rec["shadow_profile"]), ("s1", "v0"))
        self.assertEqual([(p["rank"], p["code"]) for p in rec["picks"]], [(1, "005930"), (2, "000660")])
        self.assertEqual(rec["v0_picks"], [{"code": "000660", "name": "SK하이닉스"}, {"code": "035420", "name": "NAVER"}])

    def test_no_v0_picks_file_is_recorded_as_null(self):
        (self.root / "20261127" / "_20261127_picks.txt").unlink()
        g061.log_date("20261127")
        self.assertIsNone(json.loads(g061.shadow_path("20261127").read_text(encoding="utf-8"))["v0_picks"])

    def test_watch_json_is_keyed_by_the_trading_day(self):
        g061.log_date("20261127")
        out = self.root / "w.json"
        self.assertEqual(g061.write_watch_json(out), 0)
        self.assertEqual(json.loads(out.read_text(encoding="utf-8")), {"20261130": ["000660", "035420"]})
        self.assertEqual(json.loads((self.root / "w_v0.json").read_text(encoding="utf-8")), {"20261130": ["005930", "000660"]})


if __name__ == "__main__":
    unittest.main()

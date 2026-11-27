# -*- coding: utf-8 -*-
"""[2026-11-27] per-method target symbols composed from the scan (r031) and the r016 `targets` commands. Temp folders only."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))
import r016_control as r016  # noqa: E402
import r031_method_targets as r031  # noqa: E402

DAY = "20261127"


def picks(n):
    return [{"code": f"{i:06d}", "name": f"P{i}"} for i in range(1, n + 1)]


def ranked(n, start=1, fund_at=()):
    """rank order = code order; codes 1.. so the picks are part of the ranking like in a real scan"""
    return [{"rank": i, "code": f"{i:06d}", "name": ("KODEX %d" % i) if i in fund_at else f"S{i}", "score": 100.0 - i,
             "picked": False, "eligible": i % 10 == 0, "fail": "" if i % 10 == 0 else "down_trend"} for i in range(start, start + n)]


class ComposeTest(unittest.TestCase):
    def test_rule_30_edge_50_vwap_union_and_the_watch_order(self):
        c = r031.compose(picks(18), ranked(200), r031.default_config())
        self.assertEqual(len(c["a"]), 18)
        self.assertEqual(c["rule_extra"], [f"{i:06d}" for i in range(19, 31)])        # the next 12 of the ranking
        self.assertEqual(c["edge_extra"], [f"{i:06d}" for i in range(31, 63)])        # the 32 after those
        self.assertEqual({m: len(v) for m, v in c["lists"].items()}, {"rule": 30, "edge": 50, "vwap": 62})
        self.assertEqual(set(c["lists"]["rule"]) & set(c["edge_extra"]), set())
        self.assertEqual(set(c["lists"]["vwap"]), set(c["lists"]["rule"]) | set(c["lists"]["edge"]))
        self.assertEqual([w["code"] for w in c["watch"]], c["a"] + c["rule_extra"] + c["edge_extra"])
        self.assertEqual(c["no_tick"], [f"{i:06d}" for i in range(41, 63)])           # past the tick collector's 40
        self.assertEqual(c["rows"]["000001"]["src"], "A")
        self.assertEqual(c["rows"]["000020"], {"name": "S20", "src": "rule", "rank": 20, "score": 80.0, "eligible": True,
                                               "fail": "", "in": ["rule", "vwap"], "included": []})
        self.assertEqual(c["rows"]["000031"]["in"], ["edge", "vwap"])
        self.assertEqual(c["short"], {"rule": 0, "edge": 0})

    def test_a_larger_than_a_size_adds_nothing_and_a_short_ranking_is_reported(self):
        c = r031.compose(picks(40), ranked(45), r031.default_config())
        self.assertEqual(c["rule_extra"], [])                                          # A alone is already more than 30
        self.assertEqual(len(c["lists"]["rule"]), 40)
        self.assertEqual(len(c["edge_extra"]), 5)                                      # the ranking ran out
        self.assertEqual(c["short"], {"rule": 0, "edge": 5})

    def test_funds_are_skipped_only_when_asked(self):
        is_fund = lambda name: str(name).startswith("KODEX")
        c = r031.compose(picks(2), ranked(20, fund_at=(3, 4)), {**r031.default_config(), "sizes": {"rule": 4, "edge": 6}}, is_fund)
        self.assertEqual(c["rule_extra"], ["000005", "000006"])
        cfg = {**r031.default_config(), "sizes": {"rule": 4, "edge": 6}, "exclude_funds": False}
        self.assertEqual(r031.compose(picks(2), ranked(20, fund_at=(3, 4)), cfg, is_fund)["rule_extra"], ["000003", "000004"])

    def test_always_include_and_always_exclude_per_method(self):
        cfg = r031.default_config()
        cfg["sizes"] = {"rule": 4, "edge": 6}
        cfg["exclude"] = {"rule": ["000001"], "edge": [], "vwap": ["000003"]}
        cfg["include"] = {"rule": ["000900"], "edge": [], "vwap": ["000901"]}
        c = r031.compose(picks(2), ranked(20), cfg, names={"000900": "수동A"})
        self.assertEqual(c["lists"]["rule"], ["000002", "000003", "000004", "000900"])  # A's 000001 left out of the rule list only
        self.assertIn("000001", c["lists"]["edge"])
        self.assertNotIn("000003", c["lists"]["vwap"])
        self.assertIn("000901", c["lists"]["vwap"])
        self.assertEqual(c["rows"]["000900"]["src"], "manual")
        self.assertEqual(c["rows"]["000900"]["name"], "수동A")
        self.assertEqual(c["rows"]["000900"]["included"], ["rule"])
        self.assertEqual(c["excluded"], {"rule": ["000001"], "edge": [], "vwap": ["000003"]})
        codes = [w["code"] for w in c["watch"]]
        self.assertEqual(codes.index("000900"), 4)                                     # a rule include is watched before the Edge extras
        # a symbol excluded from every list is not watched at all
        cfg["exclude"] = {m: ["000001"] for m in r031.METHODS}
        self.assertNotIn("000001", [w["code"] for w in r031.compose(picks(2), ranked(20), cfg)["watch"]])

    def test_the_watchlist_limit(self):
        cfg = {**r031.default_config(), "sizes": {"rule": 80, "edge": 100}}
        c = r031.compose(picks(10), ranked(400), cfg)
        self.assertEqual(len(c["watch"]), r031.WATCH_MAX)
        self.assertTrue(c["dropped"])
        self.assertTrue(set(c["dropped"]).isdisjoint(c["lists"]["vwap"]))            # a dropped code is in no list


class ConfigTest(unittest.TestCase):
    def test_defaults_and_bad_values(self):
        self.assertEqual(r031.normalize_config(None), r031.default_config())
        cfg = r031.normalize_config({"auto": "yes", "sizes": {"rule": 0, "edge": 60}, "include": {"rule": ["5930", "005930", "005930"]},
                                     "exclude": {"rule": ["005930"]}, "junk": 1})
        self.assertFalse(cfg["auto"])                                                  # only a real true switches it on
        self.assertEqual(cfg["sizes"], {"rule": 30, "edge": 60})
        self.assertEqual(cfg["include"]["rule"], [])                                   # in both lists: excluded
        self.assertEqual(cfg["exclude"]["rule"], ["005930"])
        self.assertNotIn("junk", cfg)

    def test_merge(self):
        cfg = r031.default_config()
        new, why = r031.merge_config(cfg, {"auto": True, "sizes": {"edge": 40}, "exclude": {"vwap": ["005930"]}})
        self.assertIsNone(why)
        self.assertEqual((new["auto"], new["sizes"], new["exclude"]["vwap"]), (True, {"rule": 30, "edge": 40}, ["005930"]))
        # including a code again takes it off the exclude list of that method (the newer intent)
        again, _ = r031.merge_config(new, {"include": {"vwap": ["005930"]}})
        self.assertEqual((again["include"]["vwap"], again["exclude"]["vwap"]), (["005930"], []))
        for bad in ({"auto": 1}, {"sizes": {"rule": 0}}, {"sizes": {"vwap": 5}}, {"include": {"rule": ["5930"]}}, {"x": 1}, [1]):
            same, why = r031.merge_config(cfg, bad)
            self.assertIsNotNone(why, bad)
            self.assertEqual(same, cfg)

    def test_one_code_edits_work_on_the_stored_lists(self):
        cfg = r031.default_config()
        cfg, why = r031.merge_config(cfg, {"exclude_add": {"vwap": ["000001"]}})
        cfg, why = r031.merge_config(cfg, {"exclude_add": {"vwap": ["000002"]}})          # the first one is still there
        self.assertEqual(cfg["exclude"]["vwap"], ["000001", "000002"])
        cfg, why = r031.merge_config(cfg, {"include_add": {"vwap": ["000001"]}})          # the newer intent
        self.assertEqual((cfg["include"]["vwap"], cfg["exclude"]["vwap"]), (["000001"], ["000002"]))
        cfg, why = r031.merge_config(cfg, {"include_remove": {"vwap": ["000001"]}, "exclude_remove": {"vwap": ["000002"]}})
        self.assertEqual((cfg["include"]["vwap"], cfg["exclude"]["vwap"], why), ([], [], None))
        self.assertIsNotNone(r031.merge_config(cfg, {"exclude_add": {"x": ["000001"]}})[1])
        self.assertIsNotNone(r031.merge_config(cfg, {"exclude_add": {"vwap": ["1"]}})[1])

    def test_a_broken_file_never_composes(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(r031.load_config(Path(d))[1], "missing")
            saved = r031.save_config(Path(d), {**r031.default_config(), "auto": True}, "t")
            self.assertTrue(saved["auto"])
            self.assertEqual(r031.load_config(Path(d)), (saved, "ok"))
            r031.config_path(Path(d)).write_text("{ broken", encoding="utf-8")
            cfg, state = r031.load_config(Path(d))
            self.assertEqual((cfg["auto"], state), (False, "broken"))

    def test_scan_all_rows(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "s.md"
            f.write_text("| rank | picked | code | name | score | fail_reasons | eligible |\n| --- | --- | --- | --- | --- | --- | --- |\n"
                         "| 1 |  | 214150 | 클래시스 | 60.38 | a\\|b | N |\n| 2 | 🟢 | 122640 | 예스티 | 58.62 |  | Y |\n"
                         "| 3 |  | 122640 | dup | 1 |  | N |\n| x |  | 000000 | bad | 1 |  | N |\n", encoding="utf-8")
            rows = r031.parse_scan_all(f)
            self.assertEqual([(r["rank"], r["code"], r["name"], r["score"], r["picked"], r["eligible"], r["fail"]) for r in rows],
                             [(1, "214150", "클래시스", 60.38, False, False, "a/b"), (2, "122640", "예스티", 58.62, True, True, "")])
            self.assertEqual(r031.parse_scan_all(Path(d) / "none.md"), [])


class ControlTest(unittest.TestCase):
    """r016 targets get / config / apply on a temp runtime"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.paths = r016.ControlPaths(runtime_root=root / "runtime", live_trading_dir=root / "live_trading")
        (root / "live_trading").mkdir()
        day = self.paths.runtime_root / "data" / DAY
        day.mkdir(parents=True)
        (day / f"_{DAY}_picks.txt").write_text("\n".join(f"{i:06d},P{i}" for i in range(1, 4)) + "\n", encoding="utf-8")
        head = "| rank | picked | code | name | score | fail_reasons | eligible |\n| --- | --- | --- | --- | --- | --- | --- |\n"
        (day / f"_{DAY}_scan_all.md").write_text(head + "".join(f"| {i} |  | {i:06d} | S{i} | {100 - i} | x | N |\n" for i in range(1, 80)), encoding="utf-8")
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)
        self.p = mock.patch.object(r016, "collector_status", return_value={"running": False, "missing": []})
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_get_is_read_only_and_apply_writes_the_watchlist_and_the_three_lists(self):
        out = r016.targets_compose(self.paths, DAY)
        self.assertTrue(out["ok"])
        self.assertEqual(out["config_state"], "missing")
        self.assertEqual({m: len(v) for m, v in out["composition"]["lists"].items()}, {"rule": 30, "edge": 50, "vwap": 77})
        self.assertFalse(self.paths.watchlist.exists())
        self.assertFalse(self.paths.overrides.exists())

        res = r016.targets_apply(self.paths, "test", DAY)
        self.assertTrue(res["ok"], res.get("error"))
        wl = r016.read_watchlist(self.paths)
        self.assertEqual(len(wl["codes"]), 77)
        self.assertEqual(wl["codes"][0]["code"], "000001")                             # A first
        self.assertTrue(any(h.startswith(f"# scan_date={DAY}") for h in wl["header"]))
        stored = r016.read_overrides(self.paths)
        self.assertEqual(len(stored["RULE_TARGET_CODES"]), 30)
        self.assertEqual(len(stored["EDGE_TARGET_CODES"]), 50)
        self.assertEqual(len(stored["VWAP_TARGET_CODES"]), 77)
        self.assertEqual(stored["RULE_TARGET_CODES"], sorted(stored["RULE_TARGET_CODES"]))
        # the scan date of r004 is the default date afterwards
        self.assertEqual(r016.targets_compose(self.paths)["date"], DAY)

    def test_config_edits_are_kept_and_a_list_emptied_by_exclusions_is_not_applied(self):
        self.assertTrue(r016.targets_config(self.paths, {"exclude": {"vwap": ["000001"]}, "include": {"rule": ["123456"]}}, "t")["ok"])
        self.assertFalse(r016.targets_config(self.paths, {"sizes": {"rule": 0}}, "t")["ok"])
        c = r016.targets_compose(self.paths, DAY)["composition"]
        self.assertNotIn("000001", c["lists"]["vwap"])
        self.assertIn("123456", c["lists"]["rule"])
        r016.targets_config(self.paths, {"sizes": {"rule": 3, "edge": 3}, "exclude": {"rule": ["000001", "000002", "000003"]},
                                         "include": {"rule": []}}, "t")
        res = r016.targets_apply(self.paths, "t", DAY)
        self.assertFalse(res["ok"])
        self.assertIn("rule list is empty", res["error"])
        self.assertFalse(self.paths.watchlist.exists())                               # nothing was written

    def test_the_post_close_apply_copies_the_picks_unless_auto_is_on(self):
        out = r016.apply_picks_to_watchlist(self.paths, DAY, "t")
        self.assertTrue(out["ok"])
        self.assertEqual(len(r016.read_watchlist(self.paths)["codes"]), 3)            # as before: the picks only
        self.assertFalse(self.paths.overrides.exists())
        r016.targets_config(self.paths, {"auto": True}, "t")
        out = r016.apply_picks_to_watchlist(self.paths, DAY, "t")
        self.assertTrue(out["ok"])
        self.assertTrue(out["composed"])
        self.assertEqual(len(r016.read_watchlist(self.paths)["codes"]), 77)
        self.assertEqual(len(r016.read_overrides(self.paths)["RULE_TARGET_CODES"]), 30)

    def test_apply_refuses_a_composition_that_is_not_the_one_shown_and_the_jobs_tab_sees_it_applied(self):
        shown = r016.targets_compose(self.paths, DAY)["composition"]["digest"]
        r016.targets_config(self.paths, {"exclude_add": {"vwap": ["000005"]}}, "someone else")
        res = r016.targets_apply(self.paths, "t", DAY, expect=shown)
        self.assertFalse(res["ok"])
        self.assertFalse(self.paths.watchlist.exists())
        now = r016.targets_compose(self.paths, DAY)["composition"]["digest"]
        self.assertNotEqual(now, shown)
        r016.targets_config(self.paths, {"auto": True}, "t")
        self.assertTrue(r016.targets_apply(self.paths, "t", DAY, expect=now)["ok"])
        import r021_jobs as r021
        _pf, pk = r021.read_picks(r021.Paths(self.paths.runtime_root), DAY)
        cmp_ = r016.watchlist_vs_picks(self.paths, pk)
        self.assertEqual((cmp_["same_codes"], cmp_["applied"], cmp_["composed"]), (False, True, True))

    def test_picks_copied_while_older_lists_are_stored_say_which_picks_are_left_out(self):
        r016.write_overrides(self.paths, {"VWAP_TARGET_CODES": ["000001", "999999"]}, "t")
        out = r016.apply_picks_to_watchlist(self.paths, DAY, "t")
        self.assertTrue(out["ok"])
        self.assertEqual(out["picks_outside_targets"], {"vwap": ["000002", "000003"]})

    def test_no_picks_no_write(self):
        res = r016.targets_apply(self.paths, "t", "20261126")
        self.assertFalse(res["ok"])
        self.assertFalse(self.paths.watchlist.exists())


if __name__ == "__main__":
    unittest.main()

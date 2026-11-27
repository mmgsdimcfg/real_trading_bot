# -*- coding: utf-8 -*-
"""[2026-11-27] per-method target symbols (r001 RULE_ / EDGE_ / VWAP_TARGET_CODES): the list reader, what r016 `set` stores,
the paper runners' entry block and the real-order path of the side strategies. No broker call."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))
import r001_define_config as r001  # noqa: E402
import r016_control as r016  # noqa: E402
import r022_edge_only as r022  # noqa: E402

try:
    import r003_trade_live_execute as r003  # noqa: E402
except Exception:  # broker SDK missing on this machine
    r003 = None


class ReaderTest(unittest.TestCase):
    def test_the_default_is_no_list_and_everything_is_a_target(self):
        for m in ("rule", "edge", "vwap"):
            self.assertEqual(r001.method_target_codes(m), frozenset())
            self.assertTrue(r001.method_is_target(m, "005930"))

    def test_a_list_narrows_to_its_codes(self):
        ns = {"VWAP_TARGET_CODES": ("005930", "0001A0")}
        t = r001.method_target_codes("vwap", ns)
        self.assertEqual(t, frozenset({"005930", "0001A0"}))
        self.assertTrue(r001.method_is_target("vwap", "5930", t))          # zero-filled like every code in r003
        self.assertFalse(r001.method_is_target("vwap", "000660", t))
        self.assertEqual(r001.method_target_codes("rule", ns), frozenset())   # the other methods are not touched

    def test_a_value_that_cannot_be_read_closes_the_method_instead_of_opening_it(self):
        for bad in (("5930",), ("005930", 5930), "005930", ("00593a",), tuple("%06d" % i for i in range(r001.METHOD_TARGET_MAX + 1))):
            t = r001.method_target_codes("edge", {"EDGE_TARGET_CODES": bad})
            self.assertEqual(t, frozenset({r001.METHOD_TARGET_INVALID}), bad)
            self.assertFalse(r001.method_is_target("edge", "005930", t))

    def test_a_stored_list_that_was_not_applied_closes_the_method_too(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "live_overrides.json"
            base = {k: getattr(r001, k) for k in dir(r001) if k == k.upper() and not k.startswith("_")}
            # a wrong type for one key: that method closes, the others keep their (valid) lists
            f.write_text(json.dumps({"overrides": {"RULE_TARGET_CODES": "005930", "VWAP_TARGET_CODES": ["005930"]}}), encoding="utf-8")
            ns = dict(base)
            r001._apply_live_overrides(f, ns)
            self.assertEqual(r001.method_target_codes("rule", ns), frozenset({r001.METHOD_TARGET_INVALID}))
            self.assertEqual(r001.method_target_codes("vwap", ns), frozenset({"005930"}))
            self.assertEqual(r001.method_target_codes("edge", ns), frozenset())
            # a file that cannot be read: whether a list was stored is not known - every method closes
            f.write_text("{ broken", encoding="utf-8")
            ns = dict(base)
            r001._apply_live_overrides(f, ns)
            for m in ("rule", "edge", "vwap"):
                self.assertEqual(r001.method_target_codes(m, ns), frozenset({r001.METHOD_TARGET_INVALID}), m)
            # no file: nothing was stored
            ns = dict(base)
            r001._apply_live_overrides(Path(d) / "none.json", ns)
            self.assertEqual(r001.method_target_codes("rule", ns), frozenset())

    def test_the_keys_can_be_overridden_and_r016_stores_them_sorted_without_repeats(self):
        for key in r001.METHOD_TARGET_KEYS.values():
            self.assertTrue(r001.live_override_key_allowed(key, getattr(r001, key)), key)
        acc, rej = r016.validate_assignments({"VWAP_TARGET_CODES": ["237690", "005930", "237690"], "RULE_TARGET_CODES": []})
        self.assertEqual(rej, {})
        self.assertEqual(acc["VWAP_TARGET_CODES"], ["005930", "237690"])
        self.assertEqual(acc["RULE_TARGET_CODES"], [])
        acc, rej = r016.validate_assignments({"EDGE_TARGET_CODES": ["5930"], "VWAP_TARGET_CODES": [5930], "RULE_TARGET_CODES": "005930"})
        self.assertEqual(acc, {})
        self.assertEqual(set(rej), {"EDGE_TARGET_CODES", "VWAP_TARGET_CODES", "RULE_TARGET_CODES"})


class PaperRunnerTest(unittest.TestCase):
    def test_the_entry_block_names_the_list_before_the_fund_check(self):
        src = Path(r022.__file__).read_text(encoding="utf-8")
        self.assertEqual(src.count('block = "not_target"'), 2)             # the regular session and the NXT sessions
        self.assertEqual(r022.EdgeOnlyRunner.targets, frozenset())            # no list unless r003 sets one
        import r029_vwap_trade as r029
        self.assertEqual(r029.VwapPaperRunner.targets, frozenset())           # inherited: the same block


@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class LivePathTest(unittest.TestCase):
    NOW = datetime(2026, 11, 27, 10, 0, 0)

    def test_no_list_by_default(self):
        self.assertEqual(r003._METHOD_TARGETS, {"rule": frozenset(), "edge": frozenset(), "vwap": frozenset()})
        self.assertFalse(r003._NOT_TARGET_DECISION.approved)
        self.assertIsNone(r003._NOT_TARGET_DECISION.failed)

    def _try(self, strategy, targets):
        api = mock.Mock()
        api.is_edge_symbol.return_value = False
        api.has_pending_order.return_value = False
        api.edge_live_block_reason.return_value = "STOP_HERE"       # the first thing after the target check
        logs = []
        with mock.patch.dict(r003._METHOD_TARGETS, targets), mock.patch.object(r003, "log", logs.append), \
                mock.patch.object(r003, "is_regular_session", return_value=True), mock.patch.dict(r003._target_note_at, {}, clear=True):
            out = r003._edge_live_try_buy(api, {"enabled": True, "bar_seen": {}}, "005930", "삼성전자", "005930_삼성전자", self.NOW, False,
                                          10_000.0, "live", self.NOW, None, {}, set(), set(), set(), set(), None, None, {}, {},
                                          strategy=strategy)
        return out, api, logs

    def test_a_side_strategy_does_not_look_at_a_symbol_outside_its_list(self):
        out, api, logs = self._try(r003.r023.VWAP, {"vwap": frozenset({"000660"})})
        self.assertFalse(out)
        api.edge_live_block_reason.assert_not_called()                # returned at the target check
        self.assertTrue(any("[TARGET      ]" in m and "vwap" in m for m in logs))

    def test_the_other_methods_list_does_not_count_and_a_listed_symbol_goes_on(self):
        for targets in ({"edge": frozenset({"000660"})}, {"vwap": frozenset({"005930"})}, {}):
            out, api, logs = self._try(r003.r023.VWAP, targets)
            api.edge_live_block_reason.assert_called_once()            # went past the target check
            self.assertFalse(any("[TARGET      ]" in m for m in logs))

    def test_the_note_is_written_once_per_symbol_and_method_in_the_interval(self):
        logs = []
        with mock.patch.object(r003, "log", logs.append), mock.patch.dict(r003._target_note_at, {}, clear=True):
            for _ in range(3):
                r003._note_not_target("rule", "005930_삼성전자", "005930", self.NOW)
            r003._note_not_target("vwap", "005930_삼성전자", "005930", self.NOW)
        self.assertEqual(len(logs), 2)


@unittest.skipIf(r003 is None, "r003 import failed (broker SDK not available)")
class ActiveSlotTest(unittest.TestCase):
    """a symbol no method may newly buy does not keep an active watch slot"""
    NOW = datetime(2026, 11, 27, 10, 0, 0)

    def test_who_may_buy(self):
        on, off = {"enabled": True}, {"enabled": False}
        runner = mock.Mock(close_only=False, log_only=False, positions={})
        closing = mock.Mock(close_only=True, log_only=False, positions={})
        recording = mock.Mock(close_only=False, log_only=True, positions={})
        holding = mock.Mock(close_only=True, log_only=False, positions={"000009": {}})
        with mock.patch.dict(r003._METHOD_TARGETS, {"rule": frozenset({"000001"}), "edge": frozenset({"000002"}), "vwap": frozenset({"000003"})}),                 mock.patch.object(r003, "TRADING_STRATEGY_MODE", "normal"), mock.patch.object(r003, "RULE_BUY_ENABLED", True):
            c = r003._method_buy_candidate
            self.assertTrue(c("000001", off, off, None, None))                 # the rule trade's
            self.assertFalse(c("000002", off, off, None, None))                # Edge AI's, but no Edge AI path takes entries
            self.assertTrue(c("000002", on, off, None, None))                  # Edge AI real orders
            self.assertTrue(c("000002", off, off, runner, None))               # Edge AI paper runner
            self.assertFalse(c("000002", off, off, closing, None))             # a runner that only closes
            self.assertFalse(c("000002", off, off, recording, None))           # a runner that only records candidates
            self.assertTrue(c("000009", off, off, None, holding))              # an open paper position keeps its slot
            self.assertTrue(c("000003", off, on, None, None))
            self.assertTrue(c("000003", off, off, None, runner))
            self.assertFalse(c("000009", on, on, runner, runner))              # on nobody's list
        # rule buys off: the rule trade's (empty) list does not make every symbol a candidate
        with mock.patch.dict(r003._METHOD_TARGETS, {"rule": frozenset(), "edge": frozenset({"000002"}), "vwap": frozenset({"000003"})}), \
                mock.patch.object(r003, "TRADING_STRATEGY_MODE", "normal"):
            with mock.patch.object(r003, "RULE_BUY_ENABLED", False):
                self.assertFalse(r003._method_buy_candidate("000009", off, on, None, None))
                self.assertTrue(r003._method_buy_candidate("000003", off, on, None, None))
            with mock.patch.object(r003, "RULE_BUY_ENABLED", True):
                self.assertTrue(r003._method_buy_candidate("000009", off, on, None, None))
        with mock.patch.dict(r003._METHOD_TARGETS, {"rule": frozenset({"000001"}), "vwap": frozenset()}),                 mock.patch.object(r003, "TRADING_STRATEGY_MODE", "normal"), mock.patch.object(r003, "RULE_BUY_ENABLED", True):
            self.assertTrue(r003._method_buy_candidate("000009", {"enabled": False}, {"enabled": True}, None, None))   # VWAP has no list
            self.assertFalse(r003._method_buy_candidate("000009", {"enabled": False}, {"enabled": False}, None, None))

    def _run(self, active, backup, cand, reentry=None):
        import collections
        from types import SimpleNamespace
        api = SimpleNamespace(positions={}, has_buy_exposure=lambda c: False, live_state={"traded_today": set()})
        logs = []
        pool = collections.deque(backup)
        with mock.patch.object(r003, "log", logs.append), mock.patch.object(r003, "ACTIVE_WATCHLIST_SIZE", 2),                 mock.patch.object(r003, "_append_watchlist_change_history", lambda *a, **k: None),                 mock.patch.object(r003, "REENTRY_MODE", "live"):
            r003._rebalance_active_watchlist(
                current_dt=self.NOW, api=api, watch_map={}, active_set=active, backup_pool=pool,
                first_active_at={c: self.NOW for c in active}, hard_stop_today_codes=set(), gap_blocked_codes=set(),
                warn_state={}, frame_cache={}, reentry_state=reentry, is_candidate=cand)
        return pool, logs

    def test_a_non_target_leaves_its_slot_and_the_next_candidate_of_the_backup_pool_takes_it(self):
        active = {"000001", "000009"}
        pool, logs = self._run(active, ["000008", "000002", "000003"], lambda c: c in {"000001", "000002", "000003"})
        self.assertEqual(active, {"000001", "000002"})                         # 000009 out, 000008 passed over, 000002 promoted
        self.assertEqual(sorted(pool), ["000003", "000008", "000009"])         # nobody is lost: the non-targets wait in the pool
        self.assertEqual(pool[0], "000003")                                    # behind the candidates
        # ... and come back when a method may buy them again (a runner made anew at the day rollover)
        active2 = {"000001"}
        self._run(active2, list(pool), lambda c: True)
        self.assertEqual(len(active2), 2)
        # nothing but non-targets waiting: the pool is kept, nothing is promoted, the loop ends
        active3: set = set()
        pool3, _logs3 = self._run(active3, ["000008", "000009"], lambda c: False)
        self.assertEqual((active3, sorted(pool3)), (set(), ["000008", "000009"]))
        self.assertTrue(any("ACT DROPOUT" in m and "NOT_TARGET" in m for m in logs))

    def test_the_start_up_split_puts_candidates_in_the_active_slots_first(self):
        import collections
        ordered = ["000001", "000009", "000002", "000008", "000003", "000007"]
        active = {"000001", "000009"}                                   # the first two of the watchlist, one of them nobody's
        pool = collections.deque(["000002", "000008", "000003"])       # 000007 was taken out before (restored as bought)
        first = {"000001": datetime(2026, 11, 27, 8, 0), "000009": datetime(2026, 11, 27, 8, 0)}
        n = r003._split_watchlist_by_candidate(ordered, active, pool, first, lambda c: c in {"000001", "000002", "000003", "000007"},
                                               self.NOW, 2)
        self.assertEqual(n, 2)
        self.assertEqual(active, {"000001", "000002"})                 # watchlist order among the candidates
        self.assertEqual(list(pool), ["000003", "000009", "000008"])   # the waiting candidate first, then the others
        self.assertNotIn("000007", active | set(pool))                 # stays out
        self.assertEqual(first, {"000001": datetime(2026, 11, 27, 8, 0), "000002": self.NOW})   # kept / new / dropped
        # a symbol with an open paper position takes its slot before the other candidates (restored after a restart)
        a4, p4 = {"000001", "000002"}, collections.deque(["000003", "000008"])
        r003._split_watchlist_by_candidate(ordered, a4, p4, {}, lambda c: True, self.NOW, 2, first=lambda c: c == "000008")
        self.assertEqual((a4, list(p4)), ({"000008", "000001"}, ["000002", "000003"]))
        # every symbol a candidate: the same split as before
        a2, p2, f2 = {"000001", "000009"}, collections.deque(["000002"]), {}
        self.assertEqual(r003._split_watchlist_by_candidate(ordered, a2, p2, f2, lambda c: True, self.NOW, 2), 0)
        self.assertEqual((a2, list(p2)), ({"000001", "000009"}, ["000002"]))
        # nobody a candidate: the active set is empty, everything waits
        a3, p3 = {"000001", "000009"}, collections.deque(["000002"])
        r003._split_watchlist_by_candidate(ordered, a3, p3, {}, lambda c: False, self.NOW, 2)
        self.assertEqual((a3, list(p3)), (set(), ["000001", "000009", "000002"]))

    def test_without_a_list_nothing_changes(self):
        active = {"000001", "000009"}
        pool, logs = self._run(active, ["000008"], None)
        self.assertEqual(active, {"000001", "000009"})
        self.assertEqual(list(pool), ["000008"])
        self.assertFalse(any("NOT_TARGET" in m for m in logs))

    def test_a_sold_non_target_is_not_put_back(self):
        active: set = set()
        st = {"used": 0, "graduated": True, "bot": True, "filled": True}
        self._run(active, [], lambda c: False, reentry={"000009": st})
        self.assertEqual(active, set())
        self.assertTrue(st["graduated"])


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

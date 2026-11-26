# -*- coding: utf-8 -*-
"""Rule trade variant (2026-11-27): RULE_TRADE_VARIANT normal / extended - r001 _apply_rule_variant and the r016 value check."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("AUTO_TRADING_LIVE_OVERRIDES", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r001_define_config as r001  # noqa: E402
import r016_control as r016  # noqa: E402


class RuleVariantTest(unittest.TestCase):
    def _ns(self, variant):
        ns = {k: getattr(r001, k) for k in r001.RULE_EXTENDED_VALUES}
        ns.update(RULE_TRADE_VARIANT=variant, RULE_EXTENDED_VALUES=dict(r001.RULE_EXTENDED_VALUES))
        return ns

    def test_default_is_normal_and_nothing_is_written(self):
        self.assertEqual(r001.RULE_TRADE_VARIANT, "normal")
        self.assertEqual(r001.RULE_VARIANT_APPLIED, {})
        ns = self._ns("normal")
        before = dict(ns)
        self.assertEqual(r001._apply_rule_variant(ns), {})
        self.assertEqual({k: ns[k] for k in before}, before)

    def test_extended_writes_the_bundle_over_the_constants(self):
        ns = self._ns("Extended")
        ns["TICK_TRIGGER_CROSS_MAX_AGE_SECONDS"] = 300.0          # a value saved in the app loses to the bundle
        applied = r001._apply_rule_variant(ns)
        self.assertEqual(applied, r001.RULE_EXTENDED_VALUES)
        self.assertEqual(ns["TICK_TRIGGER_CROSS_MAX_AGE_SECONDS"], 60.0)
        self.assertEqual(ns["TICK_TRIGGER_BAR_SIZES"], (60,))
        self.assertIs(ns["TRIPLE_1MIN_BELOW_TURN_PASS"], False)
        self.assertEqual(ns["RULE_VARIANT_APPLIED"], applied)

    def test_every_bundle_key_is_a_constant_of_the_same_kind(self):
        for key, value in r001.RULE_EXTENDED_VALUES.items():
            self.assertTrue(hasattr(r001, key), key)
            self.assertIs(type(value), type(getattr(r001, key)), key)

    def test_unknown_variant_is_normal(self):
        ns = self._ns("turbo")
        self.assertEqual(r001._apply_rule_variant(ns), {})

    def test_the_bundle_itself_is_not_a_setting(self):
        self.assertFalse(r001.live_override_key_allowed("RULE_EXTENDED_VALUES", r001.RULE_EXTENDED_VALUES))
        self.assertFalse(r001.live_override_key_allowed("RULE_VARIANT_APPLIED", {}))
        self.assertTrue(r001.live_override_key_allowed("RULE_TRADE_VARIANT", "normal"))

    def test_r016_accepts_only_the_two_variants(self):
        ok, bad = r016.validate_assignments({"RULE_TRADE_VARIANT": "extended"})
        self.assertEqual((ok, bad), ({"RULE_TRADE_VARIANT": "extended"}, {}))
        ok, bad = r016.validate_assignments({"RULE_TRADE_VARIANT": "fast"})
        self.assertEqual(ok, {})
        self.assertIn("normal / extended", bad["RULE_TRADE_VARIANT"])


if __name__ == "__main__":
    unittest.main()

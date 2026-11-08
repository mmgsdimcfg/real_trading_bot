# -*- coding: utf-8 -*-
"""r001 live overrides (2026-11-05): <runtime>/config/live_overrides.json applied after the risk profile, with
key/type validation (AutoBotH remote settings)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

with contextlib.redirect_stdout(io.StringIO()):
    import r001_define_config as r001  # noqa: E402


def _ns():
    return {
        "ENABLE_X": True, "COUNT": 3, "PCT": 0.5, "MODE": "shadow", "ETF": {"KOSPI": "069500"},
        "SIZES": (60, 120), "TAGS": frozenset({"a"}), "MAYBE": None,
        "RUNTIME_ROOT": Path("/tmp/x"), "DATA_DIR_NAME": "data", "R004_WATCHLIST_FILENAME": "r004.txt",
        "_private": 1, "lower": 2, "LIVE_OVERRIDES_PATH": Path("/tmp/none"),
    }


class KeyAllowedTests(unittest.TestCase):
    def test_scalar_and_container_constants_allowed(self):
        ns = _ns()
        for key in ("ENABLE_X", "COUNT", "PCT", "MODE", "ETF", "SIZES", "TAGS", "MAYBE"):
            self.assertTrue(r001.live_override_key_allowed(key, ns[key]), key)

    def test_paths_dirs_filenames_private_lower_denied(self):
        ns = _ns()
        for key in ("RUNTIME_ROOT", "DATA_DIR_NAME", "R004_WATCHLIST_FILENAME", "_private", "lower", "LIVE_OVERRIDES_PATH"):
            self.assertFalse(r001.live_override_key_allowed(key, ns[key]), key)
        self.assertFalse(r001.live_override_key_allowed("FUNC", lambda: 1))
        self.assertFalse(r001.live_override_key_allowed("TICK_DATA_DIR", "x"))


class ValueCompatibleTests(unittest.TestCase):
    def test_bool_is_not_int_and_int_is_not_bool(self):
        self.assertTrue(r001.live_override_value_compatible(True, False))
        self.assertFalse(r001.live_override_value_compatible(True, 1))
        self.assertFalse(r001.live_override_value_compatible(3, True))

    def test_numeric_and_string(self):
        self.assertTrue(r001.live_override_value_compatible(3, 4))
        self.assertFalse(r001.live_override_value_compatible(3, 4.5))
        self.assertTrue(r001.live_override_value_compatible(0.5, 1))      # int -> float ok
        self.assertFalse(r001.live_override_value_compatible(0.5, "1"))
        self.assertTrue(r001.live_override_value_compatible("shadow", "gate"))
        self.assertFalse(r001.live_override_value_compatible("shadow", 1))

    def test_non_finite_numbers_rejected(self):
        self.assertFalse(r001.live_override_value_compatible(1.5, float("nan")))
        self.assertFalse(r001.live_override_value_compatible(1.5, float("inf")))
        self.assertFalse(r001.live_override_value_compatible((1.0, 2.0), [float("nan")]))
        self.assertFalse(r001.live_override_value_compatible({"a": 1.0}, {"a": float("inf")}))

    def test_container_elements_must_match_current_kind(self):
        self.assertFalse(r001.live_override_value_compatible(("02", "03"), [2, 3]))     # str codes stay str
        self.assertTrue(r001.live_override_value_compatible(("02", "03"), ["04"]))
        self.assertFalse(r001.live_override_value_compatible((60, 120), [1.5]))
        self.assertTrue(r001.live_override_value_compatible((60, 120), []))             # empty = rollback ok
        self.assertFalse(r001.live_override_value_compatible({"KOSPI": "069500"}, {"KOSPI": None}))
        pairs = ((0.10, 0.20),)
        self.assertTrue(r001.live_override_key_allowed("BB_UPPER_GAP_MIN_PCT_BY_TICK", pairs))
        self.assertTrue(r001.live_override_value_compatible(pairs, [[0.1, 0.3], [0.5, 1]]))
        self.assertTrue(r001.live_override_value_compatible(pairs, []))
        self.assertFalse(r001.live_override_value_compatible(pairs, [["a", 1]]))
        self.assertFalse(r001.live_override_value_compatible(pairs, [[[1]]]))
        self.assertEqual(r001._live_override_coerce(pairs, [[0.1, 0.3]]), ((0.1, 0.3),))

    def test_containers_and_none(self):
        self.assertTrue(r001.live_override_value_compatible((60, 120), [60]))
        self.assertFalse(r001.live_override_value_compatible((60, 120), [[60]]))
        self.assertTrue(r001.live_override_value_compatible({"a": "1"}, {"b": "2"}))
        self.assertFalse(r001.live_override_value_compatible({"a": "1"}, {"b": {"c": 1}}))
        self.assertTrue(r001.live_override_value_compatible(None, 5))
        self.assertFalse(r001.live_override_value_compatible(None, [1]))


class ApplyTests(unittest.TestCase):
    def _apply(self, payload, ns=None, env=None):
        ns = ns if ns is not None else _ns()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "live_overrides.json"
            if payload is not None:
                path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
            with mock.patch.dict(os.environ, env or {}, clear=False), contextlib.redirect_stdout(io.StringIO()):
                if env is not None and "AUTO_TRADING_LIVE_OVERRIDES" in env:
                    applied = r001._apply_live_overrides(namespace=ns)
                else:
                    applied = r001._apply_live_overrides(path=path, namespace=ns)
        return applied, ns

    def test_missing_file_is_noop(self):
        applied, ns = self._apply(None)
        self.assertEqual(applied, {})
        self.assertEqual(ns["LIVE_OVERRIDES_ERRORS"], [])
        self.assertTrue(ns["ENABLE_X"])

    def test_applies_valid_keys_keeps_container_types_and_records_base(self):
        applied, ns = self._apply({"ENABLE_X": False, "PCT": 1, "SIZES": [30], "TAGS": ["b"], "ETF": {"K": "1"}, "MAYBE": 7})
        self.assertEqual(set(applied), {"ENABLE_X", "PCT", "SIZES", "TAGS", "ETF", "MAYBE"})
        self.assertIs(ns["ENABLE_X"], False)
        self.assertIsInstance(ns["PCT"], float)
        self.assertEqual(ns["SIZES"], (30,))
        self.assertEqual(ns["TAGS"], frozenset({"b"}))
        self.assertEqual(ns["LIVE_OVERRIDES_BASE"]["PCT"], 0.5)
        self.assertEqual(ns["LIVE_OVERRIDES_BASE"]["ENABLE_X"], True)
        self.assertEqual(ns["LIVE_OVERRIDES_ERRORS"], [])

    def test_wrapped_format_with_metadata(self):
        applied, ns = self._apply({"updated_at": "x", "updated_by": "autoboth", "overrides": {"COUNT": 9}})
        self.assertEqual(applied, {"COUNT": 9})
        self.assertEqual(ns["COUNT"], 9)

    def test_rejects_unknown_denied_and_mismatched_keys_without_raising(self):
        applied, ns = self._apply({"NOPE": 1, "RUNTIME_ROOT": "/x", "COUNT": "3", "ENABLE_X": 1, "PCT": 0.9})
        self.assertEqual(applied, {"PCT": 0.9})
        self.assertEqual(ns["COUNT"], 3)
        self.assertEqual(ns["RUNTIME_ROOT"], Path("/tmp/x"))
        self.assertTrue(ns["ENABLE_X"])
        errs = "\n".join(ns["LIVE_OVERRIDES_ERRORS"])
        for frag in ("NOPE: unknown", "RUNTIME_ROOT: not overridable", "COUNT: type mismatch", "ENABLE_X: type mismatch"):
            self.assertIn(frag, errs)

    def test_broken_json_and_non_object_are_reported(self):
        applied, ns = self._apply("{not json")
        self.assertEqual(applied, {})
        self.assertTrue(ns["LIVE_OVERRIDES_ERRORS"][0].startswith("load failed"))
        applied, ns = self._apply([1, 2])
        self.assertEqual(applied, {})
        self.assertIn("invalid format", ns["LIVE_OVERRIDES_ERRORS"][0])

    def test_nan_literal_in_file_rejects_whole_file(self):
        applied, ns = self._apply('{"PCT": NaN, "COUNT": 1}')
        self.assertEqual(applied, {})
        self.assertIn("non-finite", ns["LIVE_OVERRIDES_ERRORS"][0])
        self.assertEqual(ns["COUNT"], 3)

    def test_env_off_disables(self):
        ns = _ns()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "o.json"
            path.write_text(json.dumps({"COUNT": 1}), encoding="utf-8")
            ns["LIVE_OVERRIDES_PATH"] = path
            for off in ("0", "off", ""):
                ns["COUNT"] = 3
                with mock.patch.dict(os.environ, {"AUTO_TRADING_LIVE_OVERRIDES": off}), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(r001._apply_live_overrides(namespace=ns), {})
                self.assertEqual(ns["COUNT"], 3)
            with mock.patch.dict(os.environ, {"AUTO_TRADING_LIVE_OVERRIDES": str(path)}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(r001._apply_live_overrides(namespace=ns), {"COUNT": 1})
            self.assertEqual(ns["LIVE_OVERRIDES_PATH"], path)

    def test_real_module_exposes_state(self):
        self.assertIsInstance(r001.LIVE_OVERRIDES_APPLIED, dict)
        self.assertIsInstance(r001.LIVE_OVERRIDES_ERRORS, list)
        self.assertTrue(str(r001.LIVE_OVERRIDES_PATH).endswith("live_overrides.json") or os.environ.get("AUTO_TRADING_LIVE_OVERRIDES"))


if __name__ == "__main__":
    unittest.main()

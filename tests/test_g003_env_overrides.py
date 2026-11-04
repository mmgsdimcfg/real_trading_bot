"""[2026-11-04] g003 replay-only reentry-policy env overrides (_env_int strictness, Codex review)."""
from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

import g003_trade_simulate_by_date as g003  # noqa: E402


class EnvIntTest(unittest.TestCase):
    def test_default_when_unset(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("G003_TEST_INT", None)
            self.assertEqual(g003._env_int("G003_TEST_INT", 5), 5)

    def test_valid_values(self):
        for raw, want in (("0", 0), (" 7 ", 7), ("12", 12)):
            with unittest.mock.patch.dict("os.environ", {"G003_TEST_INT": raw}):
                self.assertEqual(g003._env_int("G003_TEST_INT", 1), want)

    def test_invalid_values_exit(self):
        for raw in ("nan", "inf", "1.9", "-1", "", "abc"):
            with unittest.mock.patch.dict("os.environ", {"G003_TEST_INT": raw}):
                with self.assertRaises(SystemExit, msg=raw):
                    g003._env_int("G003_TEST_INT", 1)

    def test_reentry_overrides_default_to_r001(self):
        import r001_define_config as r001
        self.assertEqual(g003.REENTRY_MAX_PER_CODE, r001.REENTRY_MAX_PER_CODE)
        self.assertEqual(g003.REENTRY_MAX_PER_CODE_AFTER_PROFIT, r001.REENTRY_MAX_PER_CODE_AFTER_PROFIT)
        self.assertEqual(g003.HARD_STOP_BLOCK_REENTRY_TODAY, r001.HARD_STOP_BLOCK_REENTRY_TODAY)


if __name__ == "__main__":
    unittest.main()

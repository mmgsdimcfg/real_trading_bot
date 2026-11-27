# -*- coding: utf-8 -*-
"""Must stay the first test module by name: importing it sets the test environment (_test_env) before any other test module
imports r001 / r003. See _test_env.py."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_env  # noqa: E402,F401


class TestEnvTest(unittest.TestCase):
    def test_the_hosts_live_overrides_are_not_read_by_the_tests(self):
        self.assertIn("AUTO_TRADING_LIVE_OVERRIDES", os.environ)
        if os.environ["AUTO_TRADING_LIVE_OVERRIDES"].strip().lower() not in ("", "0", "off", "false", "no"):
            self.skipTest("AUTO_TRADING_LIVE_OVERRIDES was set by the caller")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))
        import r001_define_config as r001
        self.assertEqual(dict(getattr(r001, "LIVE_OVERRIDES_APPLIED", {})), {})

    def test_korean_text_can_be_printed(self):
        print("한글 출력 확인")          # raised UnicodeEncodeError on a cp1252 pipe before _test_env


if __name__ == "__main__":
    unittest.main()

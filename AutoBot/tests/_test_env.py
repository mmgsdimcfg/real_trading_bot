# -*- coding: utf-8 -*-
"""Test environment (2026-11-27): the tests never read the host's live settings and print Korean on any console.

Imported by test_000_env.py (the first module `python -m unittest discover -s tests` imports) and by conftest.py (pytest).

1. Live overrides off. Importing r003 applies <runtime>/config/live_overrides.json (what the user set in the app) to
   r001_define_config, and every later test that asserts a default then fails with the day's settings (2026-11-27: 23 failures from
   RULE_TRADE_VARIANT=extended, the Edge AI exits and VWAP_TRADE_MODE=paper). A value the caller exported is left alone, and a test
   that passes its own overrides path is not affected.
2. stdout / stderr in UTF-8. With the output piped, Windows Python uses the ANSI code page (cp1252 here) and a Korean print raises
   UnicodeEncodeError inside the code under test (g007 send_text).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("AUTO_TRADING_LIVE_OVERRIDES", "off")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:           # a replaced stream without reconfigure (a test runner's buffer): left as it is
        pass

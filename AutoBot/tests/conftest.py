# -*- coding: utf-8 -*-
"""pytest: the same test environment as `unittest discover` (see _test_env.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _test_env  # noqa: E402,F401

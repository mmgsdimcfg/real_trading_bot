# -*- coding: utf-8 -*-
"""r015 condition export: every active condition carries a frame tag (AutoBotH chart linking)."""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

with contextlib.redirect_stdout(io.StringIO()):
    import r015_export_conditions as r015  # noqa: E402

VALID = {"none", "live", "1m", "3m", "tick", "daily"}


class FrameTests(unittest.TestCase):
    def test_every_condition_has_a_known_frame(self):
        data = r015.build()
        self.assertIn("frames", data)
        self.assertEqual(set(data["frames"]), VALID)
        for side in ("buy", "sell"):
            for row in data[side]:
                self.assertTrue(row["frame"], row["func"])
                self.assertTrue(set(row["frame"].split("+")) <= VALID, row)
        buy = {r["func"]: r for r in data["buy"]}
        self.assertEqual(buy["_010_market_regime"]["frame"], "1m")
        self.assertEqual(buy["_011_context_frame_ready"]["frame"], "3m")
        if data["entry_trigger_mode"] == "triple":
            self.assertEqual(buy["_009_hybrid_1min_trigger"]["frame"], "1m+tick")
        self.assertGreater(data["warmup"]["prior_day_1m_bars"], 0)


if __name__ == "__main__":
    unittest.main()

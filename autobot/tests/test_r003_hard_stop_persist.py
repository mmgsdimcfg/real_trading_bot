"""[2026-11-04] hard_stop_today_codes persists across restarts (live_state["hard_stop_today"]) and the watchlist
re-entry path honours HARD_STOP_BLOCK_REENTRY_TODAY (previously it always blocked)."""
from __future__ import annotations

import collections
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r003_trade_live_execute as r003  # noqa: E402
import r005_buy_conditions as r005  # noqa: E402
import r006_sell_conditions as r006  # noqa: E402

_TMP = tempfile.mkdtemp()


class HardStopPersistTest(unittest.TestCase):
    def test_serialize_load_roundtrip(self):
        state = {"positions_meta": {}, "traded_today": set(), "hard_stop_today": {"159010", "032830"}}
        payload = r003._serialize_live_state(state)
        self.assertEqual(payload["hard_stop_today"], ["032830", "159010"])
        with mock.patch.object(r003, "_live_state_path", return_value=Path(_TMP) / "hs.json"):
            (Path(_TMP) / "hs.json").write_text(json.dumps(payload), encoding="utf-8")
            loaded = r003.load_live_state("20261104")
        self.assertEqual(loaded["hard_stop_today"], {"032830", "159010"})
        self.assertIsInstance(loaded["hard_stop_today"], set)
        # missing file / old file without the key -> empty set, never a crash
        with mock.patch.object(r003, "_live_state_path", return_value=Path(_TMP) / "none.json"):
            self.assertEqual(r003.load_live_state("20261104")["hard_stop_today"], set())
        with mock.patch.object(r003, "_live_state_path", return_value=Path(_TMP) / "old.json"):
            (Path(_TMP) / "old.json").write_text(json.dumps({"positions_meta": {}, "traded_today": []}), encoding="utf-8")
            self.assertEqual(r003.load_live_state("20261104")["hard_stop_today"], set())

    def test_register_hard_stop_persists_immediately(self):
        calls = []
        live_state = {"hard_stop_today": set()}
        api = SimpleNamespace(persist_live_state=lambda date_str=None: calls.append(date_str), live_state=live_state)
        ctx = SimpleNamespace(code="326030", state=SimpleNamespace(hard_stop_today_codes=live_state["hard_stop_today"]),
                              risk=r005.RiskState(), current_dt=datetime(2026, 11, 4, 11, 33), log=lambda m: None,
                              api=api, date_str="20261104")
        r006._register_hard_stop(ctx, "STOP_LOSS(ATR/HARD)")
        self.assertEqual(live_state["hard_stop_today"], {"326030"})
        self.assertEqual(calls, ["20261104"])

    def test_register_hard_stop_without_api_still_blocks(self):
        ctx = SimpleNamespace(code="326030", state=SimpleNamespace(hard_stop_today_codes=set()),
                              risk=r005.RiskState(), current_dt=datetime(2026, 11, 4, 11, 33), log=lambda m: None)
        r006._register_hard_stop(ctx, "STOP_LOSS(ATR/HARD)")
        self.assertEqual(ctx.state.hard_stop_today_codes, {"326030"})

    def test_rebalance_honours_block_flag(self):
        code = "159010"
        api = SimpleNamespace(positions={}, has_buy_exposure=lambda c: False, live_state={"traded_today": set()})
        for flag, expect in ((True, False), (False, True)):
            st = {"used": 0, "graduated": True, "bot": True, "filled": True, "last_exit_profit": False}
            active: set = set()
            with mock.patch.object(r003, "REENTRY_MODE", "live"), mock.patch.object(r003, "REENTRY_MAX_PER_CODE", 1), \
                    mock.patch.object(r003, "HARD_STOP_BLOCK_REENTRY_TODAY", flag), \
                    mock.patch.object(r003, "log", lambda *a, **k: None):
                r003._rebalance_active_watchlist(
                    current_dt=datetime(2026, 11, 4, 13, 40), api=api, watch_map={}, active_set=active,
                    backup_pool=collections.deque(), first_active_at={}, hard_stop_today_codes={code},
                    gap_blocked_codes=set(), warn_state={}, frame_cache={}, reentry_state={code: st})
            self.assertEqual(code in active, expect, flag)


if __name__ == "__main__":
    unittest.main()

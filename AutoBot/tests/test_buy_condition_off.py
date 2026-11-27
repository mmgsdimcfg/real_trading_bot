# -*- coding: utf-8 -*-
"""Per-condition applied / not applied of the rule trade (r001 BUY_CONDITION_OFF, 2026-11-27): the skip in r005 check_buy_conditions,
the locked conditions, the silent evaluation of the PREP conditions, the state the app shows (r005.buy_condition_state, r015 / r016)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("AUTO_TRADING_LIVE_OVERRIDES", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r001_define_config as r001  # noqa: E402
import r005_buy_conditions as r005  # noqa: E402
import r016_control as r016  # noqa: E402


def fake(no, result=True, on_check=None):
    """a stand-in for the catalog condition `no` (the caller puts it into _NORMAL_BY_NO: the skip checks object identity)"""
    calls = []

    def check(ctx):
        calls.append(no)
        if on_check:
            on_check(ctx)
        return result
    return r005.BuyCondition(no, f"_{no:03d}_fake", "STATE", f"cond {no}", "", (), check), calls


class Ctx:
    def __init__(self):
        self.code = "005930"
        self.logs = []
        self.services = SimpleNamespace(log=self.logs.append)
        self.state = SimpleNamespace(buy_confirm_state={})


def run(conds, off, ctx=None, **kw):
    ctx = ctx or Ctx()
    by_no = {c.no: c for c in conds}
    with mock.patch.object(r001, "BUY_CONDITION_OFF", tuple(off)), mock.patch.object(r005, "_NORMAL_BY_NO", by_no), \
            mock.patch.object(r005, "BUY_CONDITIONS", tuple(conds)), \
            mock.patch.object(r005, "active_buy_conditions", return_value=tuple(conds)), \
            mock.patch.object(r005, "replace", lambda obj, **ch: SimpleNamespace(**{**vars(obj), **ch})):
        return r005.check_buy_conditions(ctx, **kw), ctx


class SkipTest(unittest.TestCase):
    def test_an_off_condition_is_not_evaluated_and_cannot_block(self):
        (c8, n8), (c7, n7), (c10, n10) = fake(8), fake(7, result=False), fake(10)
        d, _ = run([c8, c7, c10], off=[7])
        self.assertTrue(d.approved)
        self.assertEqual((n8, n7, n10), ([8], [], [10]))
        d, _ = run([c8, c7, c10], off=[])
        self.assertEqual((d.approved, d.failed.no), (False, 7))

    def test_a_locked_condition_stays_applied_whatever_is_stored(self):
        (c9, n9), (c10, _n) = fake(9, result=False), fake(10)
        d, _ = run([c9, c10], off=[9, 23, 99])
        self.assertEqual((d.approved, d.failed.no, n9), (False, 9, [9]))
        with mock.patch.object(r001, "BUY_CONDITION_OFF", (9, 23, 99, 20)):
            self.assertEqual(sorted(r005.buy_condition_off()), [20])
            self.assertEqual(r005.buy_condition_off_ignored(), (9, 23, 99))

    def test_a_prep_condition_only_computes_the_values_later_conditions_read(self):
        """the REAL conditions 2 and 15 switched off: their functions are not called (no verdict, no log line, no shadow /
        gate record through the services, no confirm-state reset) - only the service calls for the ctx fields are made"""
        calls = []

        def svc(name, value):
            def f(*a):
                calls.append(name)
                return value
            return f
        opened = object()
        ctx = SimpleNamespace(
            code="005930", symbol_label="X", price=10_700.0, current_dt=None, nxt_tradeable=False, eval_mode="live",
            state=SimpleNamespace(buy_confirm_state={"005930": {"count": 1}}),
            services=SimpleNamespace(log=svc("log", None), record_gate_decision=svc("record_gate_decision", None),
                                     get_session_open_datetime=svc("get_session_open_datetime", opened),
                                     fetch_prev_close=svc("fetch_prev_close", 10_000.0),
                                     rise_from_prev_close=lambda price, prev: price / prev - 1.0))
        (c17, n17) = fake(17)
        real2, real15 = r005._NORMAL_BY_NO[2], r005._NORMAL_BY_NO[15]
        with mock.patch.object(r001, "BUY_CONDITION_OFF", (2, 15)), mock.patch.object(r005, "MAX_BUY_RISE_PCT_FROM_PREV_CLOSE", 0.05), \
                mock.patch.object(r005, "MAX_BUY_RISE_SHADOW_PCT", 0.01), mock.patch.object(r005, "STARTUP_WARMUP_SECONDS", 10 ** 9):
            d = r005.check_buy_conditions(ctx, (real2, real15, c17), apply_off=True)
        self.assertTrue(d.approved)                                   # +7 % over the close and inside the warm-up: neither blocks
        self.assertEqual(n17, [17])
        self.assertIs(ctx.session_open_dt, opened)
        self.assertEqual(ctx.prev_close, 10_000.0)
        self.assertAlmostEqual(ctx.rise_ratio, 0.07)
        self.assertEqual(calls, ["get_session_open_datetime", "fetch_prev_close"])      # no log, no gate / shadow record
        self.assertEqual(ctx.state.buy_confirm_state, {"005930": {"count": 1}})

    def test_switching_the_confirm_step_off_drops_its_count(self):
        (c21, _a), (c22, n22) = fake(21), fake(22, result=False)
        ctx = Ctx()
        ctx.state.buy_confirm_state["005930"] = {"count": 1}
        d, ctx = run([c21, c22], off=[22], ctx=ctx)
        self.assertEqual((d.approved, n22, ctx.state.buy_confirm_state), (True, [], {}))

    def test_another_list_is_not_touched_unless_asked(self):
        (c7, _n7), (c10, _n) = fake(7, result=False), fake(10)
        with mock.patch.object(r001, "BUY_CONDITION_OFF", (7,)), mock.patch.object(r005, "_NORMAL_BY_NO", {7: c7, 10: c10}), \
                mock.patch.object(r005, "BUY_CONDITIONS", (c7, c10)):
            self.assertFalse(r005.check_buy_conditions(Ctx(), (c7, c10)).approved)                  # Edge / VWAP lists: as they are
            self.assertTrue(r005.check_buy_conditions(Ctx(), (c7, c10), apply_off=True).approved)   # the time shadow: same policy
            other7, _ = fake(7, result=False)       # a strategy-specific condition with the same number is never skipped
            self.assertFalse(r005.check_buy_conditions(Ctx(), (other7, c10), apply_off=True).approved)

    def test_the_real_lists_share_the_catalog_objects_the_skip_checks(self):
        """the skip is by object identity with the catalog: the basic list reuses 1..8 / 23..25, its own trigger is not one"""
        for c in r005.BASIC_BUY_CONDITIONS:
            shared = r005._NORMAL_BY_NO.get(c.no) is c
            self.assertEqual(shared, c.no != 9, c.func_name)
        for lst in (r005.EDGE_BUY_CONDITIONS, r005.VWAP_BUY_CONDITIONS):
            self.assertFalse(r005._NORMAL_BY_NO.get(9) is next(c for c in lst if c.no == 9))
        self.assertTrue(all(r005._NORMAL_BY_NO[c.no] is c for c in r005.SHADOW_ENTRY_CONDITIONS))

    def test_the_edge_strategy_list_ignores_the_setting(self):
        with mock.patch.object(r001, "BUY_CONDITION_OFF", (2, 3, 6)), mock.patch.object(r005, "TRADING_STRATEGY_MODE", "edge"):
            self.assertEqual(r005.buy_condition_off(), frozenset())


class StateTest(unittest.TestCase):
    def state(self, no):
        return r005.buy_condition_state(r005._NORMAL_BY_NO[no])

    def test_defaults(self):
        self.assertEqual(r001.BUY_CONDITION_OFF, (20,))
        self.assertEqual([c.no for c in r005.BUY_CONDITIONS], list(range(1, 26)))
        st = self.state(20)
        self.assertEqual((st["apply"], st["off_by"], st["switchable"]), ("off", ["BUY_CONDITION_OFF"], True))
        st = self.state(9)
        self.assertEqual((st["apply"], st["switchable"]), ("on", False))
        self.assertIn("매수 신호", st["locked_reason"])
        self.assertEqual(self.state(14)["apply"], "record")
        self.assertEqual((self.state(2)["prep"], self.state(15)["prep"], self.state(7)["prep"]), (True, True, False))
        self.assertEqual(sorted(r005.LOCKED_BUY_CONDITIONS), [1, 3, 4, 5, 6, 9, 11, 14, 19, 23, 24, 25])
        # 3 / 6: the order function / the watchlist keep the same limit, so the list never claims they can be switched off
        self.assertEqual((self.state(3)["switchable"], self.state(6)["switchable"]), (False, False))

    def test_a_setting_inside_the_condition_that_makes_it_pass_is_reported(self):
        for no, name, value, text in ((7, "ENABLE_DAILY_LOSS_LIMIT_GATE", False, "ENABLE_DAILY_LOSS_LIMIT_GATE=False"),
                                      (8, "ENABLE_MAX_OPEN_POSITIONS_GATE", False, "ENABLE_MAX_OPEN_POSITIONS_GATE=False"),
                                      (17, "ENABLE_OPENING_GAP_VOLUME_GATE", False, "ENABLE_OPENING_GAP_VOLUME_GATE=False"),
                                      (18, "ENABLE_QUOTE_STATUS_GATE", False, "ENABLE_QUOTE_STATUS_GATE=False"),
                                      (22, "BUY_CONSECUTIVE_CONFIRM_COUNT", 1, "BUY_CONSECUTIVE_CONFIRM_COUNT<=1"),
                                      (2, "STARTUP_WARMUP_SECONDS", 0, "STARTUP_WARMUP_SECONDS<=0")):
            with mock.patch.object(r005, name, value):
                st = self.state(no)
                self.assertEqual((st["apply"], st["off_by"]), ("off", [text]), no)
        # two switches: off only when both are off
        with mock.patch.object(r005, "ENABLE_TICK_COST_GATE", False), mock.patch.object(r005, "ENABLE_SPREAD_GATE", True):
            self.assertEqual(self.state(21)["apply"], "on")
        with mock.patch.object(r005, "ENABLE_TICK_COST_GATE", False), mock.patch.object(r005, "ENABLE_SPREAD_GATE", False):
            self.assertEqual(self.state(21)["apply"], "off")
        # the option switch of the trigger does not switch the trigger off (the report that started this)
        with mock.patch.object(r005, "ENABLE_TRIPLE_CHASE_FROM_CROSS_GUARD", False):
            self.assertEqual(self.state(9)["apply"], "on")

    def test_nothing_is_offered_as_switchable_under_the_edge_strategy(self):
        with mock.patch.object(r005, "TRADING_STRATEGY_MODE", "edge"), mock.patch.object(r001, "BUY_CONDITION_OFF", (7,)):
            st = self.state(7)
            self.assertEqual((st["apply"], st["switchable"]), ("on", False))
            self.assertIn("edge", st["locked_reason"])

    def test_the_edge_filter_modes(self):
        with mock.patch.object(r001, "BUY_CONDITION_OFF", ()):
            for mode, apply in (("gate", "on"), ("auto", "on"), ("shadow", "record"), ("off", "off")):
                with mock.patch.object(r005, "EDGE_AI_MODE", mode):
                    self.assertEqual(self.state(20)["apply"], apply, mode)

    def test_each_reported_inactive_setting_really_makes_the_function_pass(self):
        """the functions behind the table: with the switch off they return True without reading the context"""
        ctx = object()      # any attribute access would raise
        for no, patches in ((7, {"ENABLE_DAILY_LOSS_LIMIT_GATE": False}), (8, {"ENABLE_MAX_OPEN_POSITIONS_GATE": False}),
                            (17, {"ENABLE_OPENING_GAP_VOLUME_GATE": False}),
                            (16, {"ENABLE_ANTI_CHASE_DAY_GATE": False})):
            with mock.patch.multiple(r005, **patches):
                self.assertIs(r005._NORMAL_BY_NO[no].check(ctx), True, no)


class ExportTest(unittest.TestCase):
    def test_r016_accepts_only_switchable_known_numbers(self):
        self.assertEqual(r016.validate_assignments({"BUY_CONDITION_OFF": [20, 22]}), ({"BUY_CONDITION_OFF": [20, 22]}, {}))
        self.assertEqual(r016.validate_assignments({"BUY_CONDITION_OFF": []}), ({"BUY_CONDITION_OFF": []}, {}))
        for bad, text in (([9], "cannot be switched off"), ([20, 25], "cannot be switched off"), ([3], "cannot be switched off"),
                          ([6], "cannot be switched off"), ([99], "unknown condition"),
                          ([True], "type mismatch"), ("20", "type mismatch")):
            ok, rej = r016.validate_assignments({"BUY_CONDITION_OFF": bad})
            self.assertEqual(ok, {}, bad)
            self.assertIn(text, rej["BUY_CONDITION_OFF"], bad)

    def test_a_hand_edited_file_is_judged_like_r001_reads_it(self):
        """[Codex round 2] the file with a locked number: r001 loads the key, r005 ignores that number and still switches 20 off -
        so the file check must not call the whole key rejected (it would show an applied value as "removed")"""
        ok, rej = r016.validate_assignments({"BUY_CONDITION_OFF": [20, 9]}, strict=False)
        self.assertEqual((ok, rej), ({"BUY_CONDITION_OFF": [20, 9]}, {}))
        with mock.patch.object(r001, "BUY_CONDITION_OFF", (20, 9)):
            self.assertEqual((sorted(r005.buy_condition_off()), r005.buy_condition_off_ignored()), ([20], (9,)))
        # an emptied list and a later refill both validate (the check is against the base value, not the stored one)
        for value in ([], [20], [20, 22]):
            self.assertEqual(r016.validate_assignments({"BUY_CONDITION_OFF": value})[1], {}, value)
        # r001's own loader accepts what r016 stores
        self.assertTrue(r001.live_override_value_compatible((20,), []))
        self.assertTrue(r001.live_override_value_compatible((20,), [20, 22]))
        self.assertEqual(r001._live_override_coerce((20,), []), ())

    def test_one_condition_is_switched_from_the_list_stored_now(self):
        """[Codex round 3] BUY_CONDITION_SWITCH: one number, the rest of the stored list kept, leftovers that are not honoured dropped"""
        sw = r016.buy_condition_switch
        self.assertEqual(sw((20,), {"no": 22, "apply": False}), ([20, 22], None))
        self.assertEqual(sw([20, 22], {"no": 20, "apply": True}), ([22], None))
        self.assertEqual(sw([20], {"no": 20, "apply": False}), ([20], None))                 # already off: unchanged
        self.assertEqual(sw([20, 9, 99, True], {"no": 22, "apply": False}), ([20, 22], None))   # a hand edit does not block it
        for bad in ({"no": 9, "apply": False}, {"no": 3, "apply": False}, {"no": 99, "apply": True}, {"no": "20", "apply": True},
                    {"no": 20, "apply": 1}, [20], None):
            new, why = sw((20,), bad)
            self.assertIsNone(new, bad)
            self.assertTrue(why, bad)

    def test_the_set_command_applies_a_switch_inside_the_lock(self):
        import contextlib
        import io
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config"
            cfg.mkdir()
            (cfg / "live_overrides.json").write_text(json.dumps({"overrides": {"BUY_CONDITION_OFF": [20, 9], "MAX_OPEN_POSITIONS": 7}}),
                                                     encoding="utf-8")

            paths = r016.ControlPaths(runtime_root=Path(tmp), live_trading_dir=Path(tmp))

            def run(*argv):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = r016.main(["--by", "test", *argv], paths=paths)
                return rc, json.loads(out.getvalue())
            rc, res = run("set", 'BUY_CONDITION_SWITCH={"no": 22, "apply": false}')
            self.assertEqual((rc, res["accepted"], res["rejected"]), (0, {"BUY_CONDITION_OFF": [20, 22]}, {}))
            stored = json.loads((cfg / "live_overrides.json").read_text(encoding="utf-8"))["overrides"]
            self.assertEqual((stored["BUY_CONDITION_OFF"], stored["MAX_OPEN_POSITIONS"]), ([20, 22], 7))
            rc, res = run("set", 'BUY_CONDITION_SWITCH={"no": 9, "apply": false}')
            self.assertEqual(rc, 2)
            self.assertIn("cannot be switched off", res["rejected"]["BUY_CONDITION_SWITCH"])
            self.assertEqual(json.loads((cfg / "live_overrides.json").read_text(encoding="utf-8"))["overrides"]["BUY_CONDITION_OFF"], [20, 22])

    def _run_set(self, tmp, stored_text, *argv):
        import contextlib
        import io
        import json
        cfg = Path(tmp) / "config"
        cfg.mkdir(exist_ok=True)
        (cfg / "live_overrides.json").write_text(stored_text, encoding="utf-8")
        paths = r016.ControlPaths(runtime_root=Path(tmp), live_trading_dir=Path(tmp))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = r016.main(["--by", "test", "set", *argv], paths=paths)
        return rc, json.loads(out.getvalue()), (cfg / "live_overrides.json").read_text(encoding="utf-8")

    def test_a_stored_list_r001_rejects_is_not_a_source_of_numbers(self):
        """[Codex round 4] [7, "bad"] is rejected as a whole by r001 (the bot runs with the base (20,)): switching 8 off must give
        [8, 20], not [7, 8] (which would switch 7 off and 20 back on at the next start)"""
        import json
        import tempfile
        self.assertFalse(r001.live_override_value_compatible((20,), [7, "bad"]))
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, text = self._run_set(tmp, json.dumps({"overrides": {"BUY_CONDITION_OFF": [7, "bad"]}}),
                                          'BUY_CONDITION_SWITCH={"no": 8, "apply": false}')
            self.assertEqual((rc, res["accepted"]), (0, {"BUY_CONDITION_OFF": [8, 20]}))
            self.assertEqual(json.loads(text)["overrides"]["BUY_CONDITION_OFF"], [8, 20])

    def test_an_unreadable_overrides_file_is_never_rewritten(self):
        """[Codex round 4] a file that cannot be parsed reads as {} - a set would rewrite it and lose every other stored value"""
        import tempfile
        broken = '{"overrides": {"MAX_OPEN_POSITIONS": 7, '
        with tempfile.TemporaryDirectory() as tmp:
            for argv in (('BUY_CONDITION_SWITCH={"no": 8, "apply": false}',), ("MAX_OPEN_POSITIONS=5",)):
                rc, res, text = self._run_set(tmp, broken, *argv)
                self.assertEqual((rc, res["ok"], res["accepted"]), (2, False, {}), argv)
                self.assertIn("cannot be read", res["error"])
                self.assertEqual(text, broken)                      # untouched
            # [Codex round 5] a key validation already rejected stays in the answer beside the keys the unreadable file blocks
            rc, res, text = self._run_set(tmp, broken, "MAX_OPEN_POSITIONS=5", "NO_SUCH_CONSTANT=1")
            self.assertEqual((rc, text), (2, broken))
            self.assertEqual(sorted(res["rejected"]), ["MAX_OPEN_POSITIONS", "NO_SUCH_CONSTANT"])
            self.assertIn("unknown constant", res["rejected"]["NO_SUCH_CONSTANT"])
        # one read: what was judged is what is merged (no second read that could come back empty)
        import inspect
        src = inspect.getsource(r016.main)
        block = src[src.index("if accepted or switch is not None:"):src.index("write_overrides(paths, overrides, args.by)")]
        self.assertIn("read_overrides_strict(paths)", block)
        self.assertNotIn("read_overrides(paths)", block)

    def test_the_rise_cap_off_with_the_shadow_threshold_on_is_record(self):
        c15 = r005._NORMAL_BY_NO[15]
        with mock.patch.object(r005, "MAX_BUY_RISE_PCT_FROM_PREV_CLOSE", 0.0), mock.patch.object(r005, "MAX_BUY_RISE_SHADOW_PCT", 0.15):
            self.assertEqual(r005.buy_condition_state(c15)["apply"], "record")
        with mock.patch.object(r005, "MAX_BUY_RISE_PCT_FROM_PREV_CLOSE", 0.0), mock.patch.object(r005, "MAX_BUY_RISE_SHADOW_PCT", 0.0):
            self.assertEqual(r005.buy_condition_state(c15)["apply"], "off")

    def test_the_condition_export_carries_the_state_of_every_condition(self):
        import r015_export_conditions as r015
        doc = r015.build()
        rows = {r["no"]: r for r in doc["buy"]}
        self.assertEqual(sorted(rows), list(range(1, 26)))                     # the off ones are listed too
        self.assertEqual((rows[20]["apply"], rows[20]["off_by"]), ("off", ["BUY_CONDITION_OFF"]))
        self.assertEqual((rows[9]["apply"], rows[9]["switchable"]), ("on", False))
        self.assertEqual((doc["buy_condition_off"], doc["buy_condition_off_ignored"]), ([20], []))

    def test_r003_snapshots_what_it_runs(self):
        try:
            import r003_trade_live_execute as r003
        except ImportError:
            self.skipTest("r003 import failed (broker SDK not available)")
        snap = r003._buy_condition_state_snapshot()
        # by function name: number 9 is another condition in the basic / edge lists
        self.assertEqual((snap["_020_edge_ai_meta_filter"], snap["_009_hybrid_1min_trigger"], snap["_014_context_score_record"]),
                         ("off", "on", "record"))

    def test_the_running_state_is_matched_by_function(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            paths = SimpleNamespace(overrides=Path(tmp) / "live_overrides.json")
            diff = {"pending_keys": [], "rejected_in_file": {}}
            snap = {"overrides_applied": {},
                    "buy_condition_state": {"_020_edge_ai_meta_filter": "on", "_b09_basic_1min_golden_cross": "on"}}
            with mock.patch.object(r016, "read_overrides", return_value={}):
                rows = {r["no"]: r for r in r016.conditions(paths, diff, snap)["buy"]}
                self.assertEqual(rows[20]["apply_running"], "on")              # the running bot still applies it
                self.assertEqual(rows[20]["apply"], "off")                     # the stored settings do not
                self.assertIsNone(rows[9]["apply_running"])                    # the running 9 is the basic trigger, not this row
                old = {r["no"]: r for r in r016.conditions(paths, diff, {"overrides_applied": {}})["buy"]}
                self.assertIsNone(old[20]["apply_running"])                    # an r003 from before this field
                none = {r["no"]: r for r in r016.conditions(paths, diff, None)["buy"]}
                self.assertIsNone(none[20]["apply_running"])                   # not running


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""Tests for the Edge AI meta-label filter (r011), its r005 hook (_028), g013 labels and g014 helpers."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

import r001_define_config as cfg  # noqa: E402
import r005_buy_conditions as r005  # noqa: E402
import r011_edge_ai as r011  # noqa: E402
import g013_edge_ai_dataset as g013  # noqa: E402
import g014_edge_ai_train as g014  # noqa: E402

CONFIG = r011.build_strategy_config(cfg)


def _bars(start: str, n: int, base: float = 10000.0, seed: int = 0) -> pd.DataFrame:
    """End-labeled synthetic 1m bars with a mild uptrend."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1min")
    close = base * np.cumprod(1 + rng.normal(0.0004, 0.002, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    vol = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


class TestFeatureParity(unittest.TestCase):
    def test_premarket_bars_do_not_change_features(self):
        regular = _bars("2026-10-22 09:01", 90)
        nxt = _bars("2026-10-22 08:01", 50, base=9000, seed=1)
        a = r011.prepare_session_1min(regular, "2026-10-22")
        b = r011.prepare_session_1min(pd.concat([nxt, regular]), "2026-10-22")
        pd.testing.assert_frame_equal(a, b, check_freq=False)
        now = a.index[-1]
        fa = r011.compute_features(a, r011.completed_3min(a), now, 10100.0, 9900.0, 9950.0, 10000.0, {}, CONFIG)
        fb = r011.compute_features(b, r011.completed_3min(b), now, 10100.0, 9900.0, 9950.0, 10000.0, {}, CONFIG)
        self.assertEqual(list(fa), list(r011.FEATURE_NAMES))
        np.testing.assert_allclose([fa[k] for k in fa], [fb[k] for k in fb], equal_nan=True)

    def test_completed_3min_excludes_partial_bucket(self):
        m1 = r011.prepare_session_1min(_bars("2026-10-22 09:01", 10), "2026-10-22")  # last bar 09:10
        m3 = r011.completed_3min(m1)
        self.assertEqual(m3.index[-1], pd.Timestamp("2026-10-22 09:09"))
        self.assertEqual(m3["volume"].iloc[0], m1["volume"].iloc[:3].sum())

    def test_features_use_only_past_bars(self):
        m1 = r011.prepare_session_1min(_bars("2026-10-22 09:01", 120), "2026-10-22")
        i = 80
        now = m1.index[i]
        cut = m1.iloc[: i + 1]
        f1 = r011.compute_features(cut, r011.completed_3min(cut), now, 10000.0, None, None, None, {}, CONFIG)
        # perturb the future: features at bar i must not move
        fut = m1.copy()
        fut.iloc[i + 1:, fut.columns.get_loc("close")] *= 1.5
        fut = r011.prepare_session_1min(fut[["open", "high", "low", "close", "volume"]], "2026-10-22").iloc[: i + 1]
        f2 = r011.compute_features(fut, r011.completed_3min(fut), now, 10000.0, None, None, None, {}, CONFIG)
        np.testing.assert_allclose([f1[k] for k in f1], [f2[k] for k in f2], equal_nan=True)

    def test_daily_features_ignore_trade_date_row(self):
        dates = pd.date_range("2026-07-01", periods=70, freq="B").strftime("%Y%m%d")
        d = pd.DataFrame({"date": dates, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                          "volume": 10.0, "amount": 1e9})
        trade_date = dates[-1]
        a = r011.daily_features_from_frame(d, trade_date)
        d.loc[d.index[-1], "close"] = 500.0  # same-day row must be invisible
        b = r011.daily_features_from_frame(d, trade_date)
        self.assertEqual(a, b)


class TestTripleBarrier(unittest.TestCase):
    ARGS = argparse.Namespace(tp=2.0, sl=1.7, horizon=60, slip_ticks=0)

    def _run(self, o, h, l, c, times=None):
        n = len(c)
        times = times or [pd.Timestamp("10:00").time()] * n
        return g013.triple_barrier(np.array(o, float), np.array(h, float), np.array(l, float), np.array(c, float),
                                   times, 0, self.ARGS)

    def test_stop_wins_same_bar_ambiguity(self):
        out = self._run([10000, 10000], [10000, 10300], [10000, 9800], [10000, 10100])
        self.assertEqual(out["exit_reason"], "sl")
        self.assertEqual(out["y"], 0)

    def test_take_profit_net_of_costs(self):
        out = self._run([10000, 10000, 10050], [10000, 10050, 10250], [10000, 9990, 10040], [10000, 10040, 10200])
        self.assertEqual(out["exit_reason"], "tp")
        self.assertAlmostEqual(out["net_ret"], (10200 - 10000 - 10000 * 0.00015 - 10200 * 0.00215) / 10000 * 100)
        self.assertEqual(out["y"], 1)

    def test_gap_through_stop_fills_at_open(self):
        out = self._run([10000, 10000, 9700], [10000, 10010, 9750], [10000, 9990, 9650], [10000, 10000, 9700])
        self.assertEqual(out["exit_px"], 9700)

    def test_no_next_bar(self):
        self.assertIsNone(self._run([1], [1], [1], [1]))


class TestTrainerHelpers(unittest.TestCase):
    def test_auc(self):
        self.assertAlmostEqual(g014.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(g014.auc(np.array([0, 1]), np.array([0.5, 0.5])), 0.5)

    def test_threshold_keeps_min_fraction_and_improves(self):
        rng = np.random.default_rng(0)
        p = rng.random(2000)
        net = np.where(p > 0.7, 0.5, -0.3) + rng.normal(0, 0.05, 2000)
        t = g014.choose_threshold(p, net, min_keep=0.2, min_n=50)
        kept = net[p >= t]
        self.assertGreaterEqual(len(kept), 400)
        self.assertGreater(kept.mean(), net.mean())

    def test_threshold_zero_when_no_signal(self):
        p = np.linspace(0, 1, 500)
        self.assertEqual(g014.choose_threshold(p, -p, 0.2, 50), 0.0)


def _write_model(model_dir: Path, promoted: bool, threshold: float = 0.5) -> None:
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, len(r011.FEATURE_NAMES))).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 4, "min_data_in_leaf": 5},
                        lgb.Dataset(x, y), 10)
    g014.publish(booster, {"model_id": "t", "feature_version": r011.FEATURE_VERSION,
                           "features": list(r011.FEATURE_NAMES), "threshold": threshold, "promoted": promoted},
                 model_dir)


class TestGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.frame = _bars("2026-10-22 09:01", 60)
        self.now = datetime(2026, 10, 22, 10, 0, 5)

    def tearDown(self):
        self.tmp.cleanup()

    def _gate(self, mode, override=0.0):
        return r011.EdgeAIGate(mode, self.dir, self.dir / "data", self.dir / "live", CONFIG, override)

    def test_no_model_fails_open(self):
        d = self._gate("gate").evaluate("000001", self.now, 10000.0, self.frame, 9900.0, 9950.0, 10000.0)
        self.assertTrue(d.allow)
        self.assertTrue(d.reason.startswith("fail_open"))

    def test_insufficient_bars_fail_open(self):
        d = self._gate("gate").evaluate("000001", self.now, 10000.0, self.frame.iloc[:2], None, None, None)
        self.assertTrue(d.allow)
        self.assertEqual(d.reason, "insufficient_bars")

    def test_modes(self):
        _write_model(self.dir, promoted=False)
        for mode, promoted, expect_allow in (("shadow", True, True), ("auto", False, True), ("auto", True, False),
                                             ("gate", False, False)):
            _write_model(self.dir, promoted=promoted)
            d = self._gate(mode, override=0.999).evaluate("000001", self.now, 10000.0, self.frame, None, None, None)
            self.assertIsNotNone(d.prob, mode)
            self.assertEqual(d.allow, expect_allow, (mode, promoted))
        logs = list((self.dir / "live").glob("edge_ai_*.jsonl"))
        self.assertTrue(logs)
        rec = json.loads(logs[0].read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("features", rec)

    def test_decision_cached_per_bar(self):
        _write_model(self.dir, promoted=True)
        gate = self._gate("auto")
        a = gate.evaluate("000001", self.now, 10000.0, self.frame, None, None, None)
        b = gate.evaluate("000001", self.now, 10500.0, self.frame, None, None, None)
        self.assertIs(a, b)

    def test_feature_list_mismatch_fails_open(self):
        _write_model(self.dir, promoted=True)
        meta = json.loads((self.dir / "current_meta.json").read_text())
        meta["features"] = meta["features"][:-1]
        (self.dir / "current_meta.json").write_text(json.dumps(meta))
        d = self._gate("gate").evaluate("000001", self.now, 10000.0, self.frame, None, None, None)
        self.assertTrue(d.allow)
        self.assertIn("feature list mismatch", d.reason)


class TestR005Hook(unittest.TestCase):
    def _ctx(self, session="regular"):
        logs = []
        services = SimpleNamespace(
            classify_buy_session=lambda now: session,
            fetch_day_ref_prices=mock.Mock(return_value=(9950.0, 10000.0)),
            get_frame_1min=lambda code, now, nxt: _bars("2026-10-22 09:01", 60),
        )
        ctx = SimpleNamespace(code="000001", symbol_label="X", current_dt=datetime(2026, 10, 22, 10, 0, 5),
                              nxt_tradeable=False, price=10000.0, prev_close=9900.0, day_open=None, day_vwap=None,
                              config=CONFIG, services=services, log=logs.append, trigger_info={"path": "cross"},
                              state=SimpleNamespace(buy_confirm_state={"000001": {"n": 1}}))
        return ctx, logs

    def test_registered_before_confirm_and_excluded_from_shadow_and_basic(self):
        order = [c.no for c in r005.BUY_CONDITIONS]
        self.assertLess(order.index(20), order.index(28))
        self.assertLess(order.index(28), order.index(21))
        self.assertNotIn(28, [c.no for c in r005.SHADOW_ENTRY_CONDITIONS])
        self.assertNotIn(28, [c.no for c in r005.BASIC_BUY_CONDITIONS])

    def test_off_and_non_regular_do_nothing(self):
        ctx, _ = self._ctx()
        with mock.patch.object(r005, "EDGE_AI_MODE", "off"):
            self.assertTrue(r005._028_edge_ai_meta_filter(ctx))
        ctx, _ = self._ctx(session="nxt_after")
        self.assertTrue(r005._028_edge_ai_meta_filter(ctx))
        ctx.services.fetch_day_ref_prices.assert_not_called()

    def test_block_clears_confirm_state(self):
        ctx, logs = self._ctx()
        gate = mock.Mock()
        gate.evaluate.return_value = r011.EdgeAIDecision(False, 0.3, 0.5, True, "block")
        with mock.patch.object(r005, "_edge_ai_gate", return_value=gate):
            self.assertFalse(r005._028_edge_ai_meta_filter(ctx))
        self.assertNotIn("000001", ctx.state.buy_confirm_state)
        self.assertTrue(any("EDGE_AI_LOW_PROB" in m for m in logs))
        self.assertEqual((ctx.day_open, ctx.day_vwap), (9950.0, 10000.0))
        self.assertEqual(gate.evaluate.call_args.kwargs["trigger_info"], {"path": "cross"})

    def test_exception_fails_open(self):
        ctx, logs = self._ctx()
        with mock.patch.object(r005, "_edge_ai_gate", side_effect=RuntimeError("boom")):
            self.assertTrue(r005._028_edge_ai_meta_filter(ctx))
        self.assertIn("000001", ctx.state.buy_confirm_state)
        self.assertTrue(any("fail-open" in m for m in logs))



class TestCodexRound1Fixes(unittest.TestCase):
    """Regression tests for the 2026-10-26 Codex round-1 findings."""

    def _features(self, m1, m3, price=10100.0):
        return r011.compute_features(m1, m3, m1.index[-1], price, 9900.0, 9950.0, 10000.0, {}, CONFIG,
                                     trigger_info={"path": "cross", "bars_since_cross": 1})

    def test_live_capped_cache_matches_offline_full_history(self):
        # (1) afternoon bar: live sees only r003's last 200 bars (50 NXT + 150 regular), offline the full session
        full = _bars("2026-10-22 09:01", 300)                                   # 09:01 .. 14:00
        nxt = _bars("2026-10-22 08:01", 50, base=9000, seed=1)
        live_cache = pd.concat([nxt, full]).iloc[-200:]
        raw = r011.session_raw_1min(full, "2026-10-22")
        m1_full = r011.calculate_indicators(raw)
        off = g013.offline_feature_frames(raw, m1_full, r011.completed_3min(m1_full), len(raw) - 1)
        live = r011.feature_frames(r011.session_raw_1min(live_cache, "2026-10-22"))
        fa, fb = self._features(*off), self._features(*live)
        np.testing.assert_allclose([fa[k] for k in fa], [fb[k] for k in fb], equal_nan=True)
        # and the window really matters: whole-session indicators give different 3m values
        fc = self._features(m1_full, r011.completed_3min(m1_full))
        self.assertNotAlmostEqual(fa["m3_ema20_60_pct"], fc["m3_ema20_60_pct"], places=6)

    def test_fast_indicators_match_r002(self):
        rng = np.random.default_rng(5)
        for n, flat in ((90, False), (30, True), (3, False), (200, False)):
            df = _bars("2026-10-22 09:01", n, seed=n).round(-1)
            if flat:
                df[["open", "high", "low", "close"]] = 10000.0
            df.iloc[::7, df.columns.get_loc("volume")] = 0.0
            if n == 200:
                df.iloc[50:80, :4] = df.iloc[50, 3]      # flat run inside a trend (constant BB window)
            df["volume"] = df["volume"] * rng.random(n).round(1)
            pd.testing.assert_frame_equal(r011.calculate_indicators(df), r011.fast_indicators(df),
                                          rtol=1e-9, check_freq=False, obj=f"n={n} flat={flat}")

    def test_prefix_shortcut_equals_recompute(self):
        raw = r011.session_raw_1min(_bars("2026-10-22 09:01", 200, seed=3), "2026-10-22")
        m1_full = r011.calculate_indicators(raw)
        m3_full = r011.completed_3min(m1_full)
        for i in (20, r011.FEATURE_HISTORY_BARS - 1, 150):
            a = g013.offline_feature_frames(raw, m1_full, m3_full, i)
            b = r011.feature_frames(raw.iloc[: i + 1])
            fa, fb = self._features(*a), self._features(*b)
            np.testing.assert_allclose([fa[k] for k in fa], [fb[k] for k in fb], equal_nan=True, err_msg=str(i))

    def test_restart_cache_without_history_fails_open(self):
        cache = _bars("2026-10-22 11:00", 30)        # bot restarted at 11:00: 30 rows only
        gate = r011.EdgeAIGate("gate", Path(tempfile.gettempdir()) / "edge_ai_none", Path("/nonexistent"),
                               Path(tempfile.mkdtemp()), CONFIG)
        d = gate.evaluate("000001", datetime(2026, 10, 22, 11, 30, 5), 10000.0, cache, None, None, None)
        self.assertTrue(d.allow)
        self.assertEqual(d.reason, "fail_open:insufficient_history")

    def test_completed_3min_drops_partial_leading_bucket(self):
        m1 = r011.calculate_indicators(_bars("2026-10-22 09:02", 10))   # 09:02..09:11, bucket 09:03 only 2 bars
        m3 = r011.completed_3min(m1)
        self.assertEqual(m3.index[0], pd.Timestamp("2026-10-22 09:06"))
        self.assertEqual(m3["volume"].iloc[0], m1["volume"].iloc[2:5].sum())   # 09:04..09:06

    def test_gap_above_take_profit_is_tp(self):
        # (4) Codex example: entry 10,000, next bar O=10,300 H=10,400 L=9,700 -> TP at the open
        args = argparse.Namespace(tp=2.0, sl=1.7, horizon=60, slip_ticks=0)
        o, h, l, c = (np.array(v, float) for v in ([10000, 10000, 10300], [10000, 10010, 10400],
                                                    [10000, 9990, 9700], [10000, 10000, 10000]))
        out = g013.triple_barrier(o, h, l, c, [pd.Timestamp("10:00").time()] * 3, 0, args)
        self.assertEqual((out["exit_reason"], out["exit_px"], out["y"]), ("tp", 10300, 1))

    def test_dataset_tag_covers_generation_settings(self):
        base = dict(tp=2.0, sl=1.7, horizon=60, slip_ticks=1, min_gap=5, min_turnover=3e9, min_price=1000,
                    max_codes=150, workers=3, dates=None)
        a = g013.dataset_tag(argparse.Namespace(**base))
        self.assertEqual(a, g013.dataset_tag(argparse.Namespace(**{**base, "workers": 1})))
        for k, v in (("slip_ticks", 2), ("min_gap", 3), ("max_codes", 0), ("min_turnover", 1e9)):
            self.assertNotEqual(a, g013.dataset_tag(argparse.Namespace(**{**base, k: v})), k)

    def test_publish_is_single_meta_pointer(self):
        # (3) no shared current_model.txt: meta points at an immutable versioned file
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_model(d, promoted=True)
            meta = json.loads((d / "current_meta.json").read_text())
            self.assertEqual(meta["model_file"], "models/model_t.txt")
            self.assertFalse((d / "current_model.txt").exists())
            model = r011.EdgeAIModel(d)
            p, m = model.predict_with_meta({k: 0.0 for k in r011.FEATURE_NAMES})
            self.assertTrue(r011.valid_prob(p))
            self.assertIs(m, model.meta)
            # a meta pointing outside model_dir or at a missing file is refused
            for bad in ("models/missing.txt", "../x.txt"):
                (d / "current_meta.json").write_text(json.dumps({**meta, "model_file": bad}))
                model = r011.EdgeAIModel(d)
                self.assertIsNone(model.predict({}))
                self.assertIn("model file missing", model.error)

    def test_invalid_threshold_or_prob_fails_open(self):
        # (6) NaN threshold in meta -> load refused -> fail-open (never a silent block)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_model(d, promoted=True)
            meta = json.loads((d / "current_meta.json").read_text())
            for bad in (float("nan"), 1.5, -0.1):
                (d / "current_meta.json").write_text(json.dumps({**meta, "threshold": bad}))
                gate = r011.EdgeAIGate("auto", d, d / "data", d / "live", CONFIG)
                dec = gate.evaluate("000001", datetime(2026, 10, 22, 10, 0, 5), 10000.0,
                                    _bars("2026-10-22 09:01", 60), None, None, None)
                self.assertTrue(dec.allow, bad)
                self.assertFalse(dec.enforced, bad)
            gate = r011.EdgeAIGate("gate", d, d / "data", d / "live", CONFIG)
            with mock.patch.object(gate.model, "predict_with_meta", return_value=(float("nan"), {"threshold": 0.5})):
                dec = gate.evaluate("000001", datetime(2026, 10, 22, 10, 0, 5), 10000.0,
                                    _bars("2026-10-22 09:01", 60), None, None, None)
            self.assertTrue(dec.allow)
            self.assertEqual(dec.reason, "fail_open:bad_prob")

    def test_walk_forward_embargo_and_no_test_leak(self):
        dates = [f"202607{d:02d}" for d in range(1, 31)]
        df = pd.DataFrame({"date": np.repeat(dates, 4), "y": 1, "net_ret": 0.1, "m3_ctx_pass": 1})
        for k in r011.FEATURE_NAMES:
            df[k] = 0.0
        seen = []

        def fake_fit(train, args, threads):
            seen.append(sorted(train["date"].unique()))
            return SimpleNamespace(predict=lambda x: np.full(len(x), 0.6)), 20, 0.5
        args = argparse.Namespace(min_train_days=10, embargo_days=2, test_days=5, train_window=0, threads=1)
        with mock.patch.object(g014, "fit_with_inner_val", side_effect=fake_fit):
            _oos, folds = g014.walk_forward(df, args)
        self.assertTrue(folds)
        for tr, f in zip(seen, folds):
            test_first = f["test"].split("~")[0]
            self.assertLess(dates.index(tr[-1]) + args.embargo_days, dates.index(test_first))


if __name__ == "__main__":
    unittest.main()

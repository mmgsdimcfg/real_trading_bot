# -*- coding: utf-8 -*-
"""Edge AI redesign (2026-11-27): the dataset label follows the live exits (g013.live_label), the trainer picks the dataset with
that label, the buy cut exists only with a positive expectancy (g014.choose_buy_threshold, NO_TRADE_THRESHOLD), the Edge AI
trades read their own model folder (r011 TRADE_MODEL_SUBDIR, g014 --role trade) while the rule-trade filter keeps its model and the candidate study rules (g056) read no future bar."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

import g013_edge_ai_dataset as g013  # noqa: E402
import g014_edge_ai_train as g014  # noqa: E402
import g056_candidate_study as g056  # noqa: E402
import r011_edge_ai as r011  # noqa: E402


class LiveLabelTest(unittest.TestCase):
    def test_overrides_above_zero_replace_the_defaults(self):
        cfg = SimpleNamespace(EDGE_ONLY_TP_PCT=3, EDGE_ONLY_SL_PCT=1.27, EDGE_ONLY_HORIZON_MIN=300)
        self.assertEqual(g013.live_label(cfg), {"tp": 3.0, "sl": 1.27, "horizon": 300})

    def test_zero_missing_or_bad_values_keep_the_defaults(self):
        self.assertEqual(g013.live_label(SimpleNamespace()), {"tp": 2.0, "sl": 1.77, "horizon": 400})
        cfg = SimpleNamespace(EDGE_ONLY_TP_PCT=0.0, EDGE_ONLY_SL_PCT="x", EDGE_ONLY_HORIZON_MIN=float("nan"))
        self.assertEqual(g013.live_label(cfg), {"tp": 2.0, "sl": 1.77, "horizon": 400})
        self.assertEqual(g013.live_label(SimpleNamespace(EDGE_ONLY_SL_PCT=-1))["sl"], 1.77)

    def test_the_trainer_picks_the_dataset_by_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = []
            for name, lab in (("a", {"tp": 2.0, "sl": 1.77, "horizon": 400}), ("b", {"tp": 3, "sl": 1.27, "horizon": 300}),
                              ("c", None)):
                d = root / name
                d.mkdir()
                if lab is not None:
                    (d / "params.json").write_text(json.dumps(lab), encoding="utf-8")
                dirs.append(d)
            self.assertEqual([d.name for d in g014.label_tags(dirs, {"tp": 3.0, "sl": 1.27, "horizon": 300})], ["b"])
            self.assertEqual([d.name for d in g014.label_tags(dirs, g013.DEFAULT_LABEL)], ["a"])      # the filter model
            self.assertEqual(g014.label_tags(dirs, {"tp": 5.0, "sl": 1.0, "horizon": 60}), [])


def val(day_nets: dict[str, list[tuple[float, float]]]):
    """{date: [(p, net), ..]} -> (p, net, dates)"""
    rows = [(p, n, d) for d, pairs in day_nets.items() for p, n in pairs]
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]))


class BuyThresholdTest(unittest.TestCase):
    def test_no_cut_when_nothing_is_profitable(self):
        rng = np.random.default_rng(0)
        days = {f"202611{d:02d}": [(float(p), -0.5 + float(e)) for p, e in zip(rng.random(200), rng.normal(0, 1, 200))]
                for d in range(1, 13)}
        self.assertIsNone(g014.choose_buy_threshold(*val(days), min_n=50))

    def test_a_cut_that_wins_on_every_day_is_found(self):
        # p >= 0.8 makes +1 %, the rest loses 1 %
        days = {f"202611{d:02d}": [(0.9, 1.0 + 0.01 * d)] * 20 + [(0.2, -1.0)] * 80 for d in range(1, 13)}
        t = g014.choose_buy_threshold(*val(days), min_n=50)
        self.assertIsNotNone(t)
        self.assertTrue(0.2 < t <= 0.9)

    def test_a_positive_mean_from_one_lucky_day_is_not_a_cut(self):
        days = {f"202611{d:02d}": [(0.9, -0.3)] * 20 + [(0.2, -1.0)] * 80 for d in range(1, 13)}
        days["20261101"] = [(0.9, 9.0)] * 20 + [(0.2, -1.0)] * 80          # mean of p >= 0.9 is +0.475, 11 of 12 days lose
        self.assertIsNone(g014.choose_buy_threshold(*val(days), min_n=50))

    def test_too_few_kept_trades_is_not_a_cut(self):
        days = {f"202611{d:02d}": [(0.9, 1.0)] * 2 + [(0.2, -1.0)] * 98 for d in range(1, 13)}
        self.assertIsNone(g014.choose_buy_threshold(*val(days), min_n=50))

    def test_the_no_trade_threshold_is_valid_and_unreachable(self):
        self.assertTrue(r011.valid_prob(r011.NO_TRADE_THRESHOLD))
        self.assertFalse(0.999999 >= r011.NO_TRADE_THRESHOLD)


def write_model(folder: Path, model_id: str, threshold: float, **extra) -> None:
    """a tiny real booster + meta, the way g014.publish lays it out"""
    import lightgbm as lgb
    rng = np.random.default_rng(1)
    x = rng.random((200, len(r011.FEATURE_NAMES)))
    booster = lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(x, (x[:, 0] > 0.5).astype(int)), 3)
    g014.publish(booster, {"model_id": model_id, "feature_version": r011.FEATURE_VERSION, "features": list(r011.FEATURE_NAMES),
                           "threshold": threshold, **extra}, folder)


class TradeModelFolderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_model(self.dir, "gate1", 0.33, promoted=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_without_a_trade_model_the_filter_model_is_read(self):
        m = r011.EdgeAIModel(self.dir, reload_check_seconds=0, prefer_subdir=r011.TRADE_MODEL_SUBDIR)
        m.maybe_reload()
        self.assertEqual(m.meta.get("model_id"), "gate1")

    def test_a_trade_model_published_later_is_picked_up_and_the_filter_keeps_its_own(self):
        trade = r011.EdgeAIModel(self.dir, reload_check_seconds=0, prefer_subdir=r011.TRADE_MODEL_SUBDIR)
        gate = r011.EdgeAIModel(self.dir, reload_check_seconds=0)
        trade.maybe_reload()
        write_model(self.dir / r011.TRADE_MODEL_SUBDIR, "trade1", r011.NO_TRADE_THRESHOLD, no_trade=True, role="trade")
        trade.maybe_reload()
        gate.maybe_reload()
        self.assertEqual((trade.meta.get("model_id"), trade.meta.get("threshold")), ("trade1", 1.0))
        self.assertEqual((gate.meta.get("model_id"), gate.meta.get("threshold")), ("gate1", 0.33))
        prob, meta = trade.predict_with_meta({k: 0.5 for k in r011.FEATURE_NAMES})
        self.assertTrue(r011.valid_prob(prob))
        self.assertLess(prob, meta["threshold"])                       # no probability reaches the no-trade cut

    def test_wiring(self):
        src = (ROOT / "live_trading" / "r005_buy_conditions.py").read_text(encoding="utf-8")
        gate = src[src.index("def _edge_ai_gate("):src.index("def _020_edge_ai_meta_filter(")]
        edge = src[src.index("def _edge_mode_gate("):src.index("def _edge_require_promoted(")]
        self.assertNotIn("prefer_subdir", gate)                        # the rule-trade filter: the model it always had
        self.assertIn("prefer_subdir=TRADE_MODEL_SUBDIR", edge)
        r022 = (ROOT / "live_trading" / "r022_edge_only.py").read_text(encoding="utf-8")
        self.assertIn("prefer_subdir=r011.TRADE_MODEL_SUBDIR", r022)


class PromotedOfTradeModelTest(unittest.TestCase):
    def test_r016_reads_the_trade_model_when_there_is_one(self):
        import r016_control as r016
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "edge_ai"
            base.mkdir()
            paths = SimpleNamespace(runtime_root=Path(tmp))
            self.assertIsNone(r016.edge_model_promoted(paths))
            (base / "current_meta.json").write_text(json.dumps({"promoted": True}), encoding="utf-8")
            self.assertIs(r016.edge_model_promoted(paths), True)
            (base / "trade").mkdir()
            (base / "trade" / "current_meta.json").write_text(json.dumps({"promoted": False}), encoding="utf-8")
            self.assertIs(r016.edge_model_promoted(paths), False)


def bars(closes, start="2026-11-02 09:01", vol=None, highs=None, lows=None, opens=None):
    idx = pd.date_range(start, periods=len(closes), freq="1min")
    c = np.array(closes, float)
    o = np.array(opens, float) if opens is not None else np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.array(highs, float) if highs is not None else np.maximum(o, c),
                         "low": np.array(lows, float) if lows is not None else np.minimum(o, c), "close": c,
                         "volume": np.array(vol, float) if vol is not None else np.full(len(c), 1000.0)}, index=idx)


class CandidateRulesTest(unittest.TestCase):
    def test_opening_range_breakout_is_the_first_close_above_the_first_30_minutes(self):
        closes = [100.0] * 30 + [100.0] * 10 + [101.0] + [100.0] * 30     # 09:01..09:30 range high 100, break at bar 40
        found = g056.candidates(bars(closes), float("nan"), False)
        self.assertEqual(found.get("orb"), 40)

    def test_rules_do_not_read_later_bars(self):
        rng = np.random.default_rng(3)
        closes = 100 + np.cumsum(rng.normal(0, 0.3, 300))
        vol = rng.integers(100, 5000, 300)
        full = bars(closes, vol=vol)
        found = g056.candidates(full, 95.0, True)
        self.assertTrue(found)
        for kind, i in found.items():
            cut = g056.candidates(full.iloc[: i + 2], 95.0, True)          # the event bar + the entry bar only
            self.assertEqual(cut.get(kind), i, kind)

    def test_pullback_needs_the_daily_uptrend_and_gaphold_the_gap(self):
        closes = [100.0] * 5 + [103.0] * 10 + [101.0] * 10 + [101.5, 102.0, 102.5, 103.5] + [103.5] * 40
        df = bars(closes)
        self.assertIn("pullback", g056.candidates(df, 99.0, True))
        self.assertNotIn("pullback", g056.candidates(df, 99.0, False))
        up = bars([103.0] + [103.5] * 60, opens=[103.0] + [103.5] * 60)
        self.assertEqual(g056.candidates(up, 100.0, False).get("gaphold"), 29)       # the bar labelled 09:30
        self.assertNotIn("gaphold", g056.candidates(up, 102.5, False))                # gap below +2 %

    def test_daily_trend(self):
        d = pd.DataFrame({"date": [f"2026{m:02d}{day:02d}" for m in (8, 9, 10) for day in range(1, 26)],
                          "close": np.linspace(100, 200, 75)})
        prev, up = g056.daily_trend(d, "20261102")
        self.assertEqual((prev, up), (200.0, True))
        self.assertEqual(g056.daily_trend(d.iloc[:30], "20261102")[1], False)       # under 60 days
        self.assertTrue(np.isnan(g056.daily_trend(None, "20261102")[0]))

    def test_split_days(self):
        days = [f"d{i:02d}" for i in range(10)]
        disc, hold = g056.split_days(days)
        self.assertEqual((len(disc), sorted(hold)), (7, ["d07", "d08", "d09"]))


if __name__ == "__main__":
    unittest.main()

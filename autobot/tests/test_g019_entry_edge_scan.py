# -*- coding: utf-8 -*-
"""g019 entry-family signals (2026-10-31, incl. Codex review cases)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_simulation"))
sys.path.insert(0, str(ROOT / "live_trading"))

import g019_entry_edge_scan as g019  # noqa: E402


def _arrays(closes, minute0=9 * 60 + 1, prev_high=np.nan):
    c = np.asarray(closes, float)
    n = len(c)
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c) + 1, np.minimum(o, c) - 1
    v = np.full(n, 100.0)
    minute = np.arange(n, dtype=np.int64) + minute0
    sma = lambda w: np.array([c[max(0, i - w + 1):i + 1].mean() if i >= w - 1 else np.nan for i in range(n)])
    ma5, ma20 = sma(5), sma(20)
    std = np.array([c[max(0, i - 19):i + 1].std() if i >= 19 else np.nan for i in range(n)])
    mid, up, lo = ma20, ma20 + 2 * std, ma20 - 2 * std
    vwap = np.cumsum(c * v) / np.cumsum(v)
    return [o, h, l, c, v, np.ones(n, bool), minute, ma5, ma20, mid, up, lo, np.full(n, 50.0), vwap, prev_high]


def _sig(arrs, seed=1):
    return g019.signals(*arrs, np.random.default_rng(seed))


class SignalTest(unittest.TestCase):
    def test_orb15_fires_at_0916(self):
        # flat 1000 through 09:15, break above the range on the 09:16 bar (index 15)
        closes = [1000.0] * 15 + [1010.0] * 30
        hits = [i for f, i in _sig(_arrays(closes)) if f == "orb15"]
        self.assertEqual(hits, [15])

    def test_random_is_causal(self):
        rng_closes = 1000 + np.cumsum(np.random.default_rng(3).normal(0, 2, 400))
        short = [i for f, i in _sig(_arrays(rng_closes[:200]), seed=5) if f == "random"]
        full = [i for f, i in _sig(_arrays(rng_closes), seed=5) if f == "random"]
        if short:  # target fell inside the short session -> the extended session must pick the same bar
            self.assertEqual(short, full)

    def test_no_lookahead_prefix_invariance(self):
        # every signal at bar i must be identical whether or not later bars exist
        closes = 1000 + np.cumsum(np.random.default_rng(11).normal(0.2, 3, 300))
        full = _arrays(closes, prev_high=1010.0)
        sig_full = _sig(full)
        cut = 150
        part = [a[:cut] if isinstance(a, np.ndarray) else a for a in full]
        # vwap/indicators of the prefix are unchanged because they are causal
        sig_part = _sig(part)
        lim = cut - 2   # signals() never evaluates the last bar (no next bar to enter)
        a = sorted((f, i) for f, i in sig_full if i <= lim and f != "random")
        b = sorted((f, i) for f, i in sig_part if i <= lim and f != "random")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

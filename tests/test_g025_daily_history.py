# -*- coding: utf-8 -*-
"""g025_daily_history (2026-11-06): merge date folders into per-symbol history, later folder wins, provisional rows."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(ROOT / "data_simulation"))

with contextlib.redirect_stdout(io.StringIO()):
    import g025_daily_history as g025  # noqa: E402


def _daily(path: Path, rows):
    path.write_text("date,open,high,low,close,volume,amount,MA_5\n" + "\n".join(
        f"{d},{o},{o+10},{o-10},{c},{v},{c*v},{o}" for d, o, c, v in rows) + "\n", encoding="utf-8")


class G025Tests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.env = mock.patch.dict(os.environ, {"AUTO_TRADING_RUNTIME": str(self.root)})
        self.env.start()
        d1, d2 = self.root / "data" / "20261104", self.root / "data" / "20261105"
        d1.mkdir(parents=True), d2.mkdir(parents=True)
        _daily(d1 / "005930_삼성전자_daily.csv", [("20261103", 100, 105, 10), ("20261104", 106, 108, 20)])   # 11/04 row provisional
        _daily(d2 / "005930_삼성전자_daily.csv", [("20261104", 106, 110, 25), ("20261105", 111, 112, 30)])   # final 11/04, provisional 11/05
        _daily(d2 / "000660_SK하이닉스_daily.csv", [("20261105", 500, 505, 7)])
        (d2 / "_g001_done.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.env.stop()
        self.td.cleanup()

    def test_rebuild_later_folder_wins_and_marks_source(self):
        n = g025.rebuild(log=lambda *_: None)
        self.assertEqual(n, 2)
        df = pd.read_csv(self.root / "data" / "daily_history" / "005930.csv", dtype={"date": str, "source_date": str})
        self.assertEqual(df["date"].tolist(), ["20261103", "20261104", "20261105"])
        self.assertEqual(df.set_index("date").loc["20261104", "close"], 110)       # next-day (final) row replaced the snapshot
        self.assertEqual(df.set_index("date").loc["20261104", "source_date"], "20261105")
        self.assertEqual(df.set_index("date").loc["20261105", "source_date"], "20261105")   # provisional: source == date

    def test_append_upserts_without_downgrading(self):
        g025.append_folder("20261105", log=lambda *_: None)
        g025.append_folder("20261104", log=lambda *_: None)     # older folder appended later must not override
        df = pd.read_csv(self.root / "data" / "daily_history" / "005930.csv", dtype={"date": str, "source_date": str})
        self.assertEqual(df.set_index("date").loc["20261104", "close"], 110)
        self.assertEqual(len(df), 3)
        self.assertEqual(g025.main(["--append"]), 0)

    def test_rebuild_is_per_symbol_and_corrects_provisional_rows(self):
        # 000660 exists only in a middle folder that a global sample would skip; its provisional row is corrected
        # by a later folder where it also appears
        for i, day in enumerate(("20261106", "20261107", "20261110", "20261111")):
            d = self.root / "data" / day
            d.mkdir()
            _daily(d / "005930_삼성전자_daily.csv", [(day, 100 + i, 101 + i, 1)])
        _daily(self.root / "data" / "20261107" / "000660_SK하이닉스_daily.csv", [("20261105", 500, 506, 7), ("20261107", 510, 511, 8)])
        _daily(self.root / "data" / "20261110" / "000660_SK하이닉스_daily.csv", [("20261107", 510, 515, 9), ("20261110", 520, 521, 3)])
        n = g025.rebuild(log=lambda *_: None, step=3)
        self.assertEqual(n, 2)
        df = pd.read_csv(self.root / "data" / "daily_history" / "000660.csv", dtype={"date": str, "source_date": str}).set_index("date")
        self.assertEqual(df.loc["20261105", "close"], 506)            # corrected by the next folder (final)
        self.assertEqual(df.loc["20261107", "close"], 515)            # provisional 11/07 replaced by 11/10 folder
        self.assertEqual(df.loc["20261110", "source_date"], "20261110")
        samsung = pd.read_csv(self.root / "data" / "daily_history" / "005930.csv", dtype={"date": str}).set_index("date")
        self.assertEqual(samsung.loc["20261111", "close"], 104)

    def test_cli_rebuild(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(g025.main(["--rebuild"]), 0)
        self.assertTrue((self.root / "data" / "daily_history" / "000660.csv").is_file())


if __name__ == "__main__":
    unittest.main()

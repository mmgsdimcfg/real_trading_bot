"""[2026-11-17] g040 daily backfill: paging backwards, resume, listing-date stop, window, universe parsing (no network)."""
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_simulation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))

import pandas as pd  # noqa: E402

import g040_daily_backfill as g040  # noqa: E402


def _days(first: str, last: str) -> list[str]:
    d, out = datetime.strptime(first, "%Y%m%d"), []
    while d <= datetime.strptime(last, "%Y%m%d"):
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


class FakeApi:
    """KIS-like: up to 100 rows, newest first, between date_from and date_to; listing date = first."""

    def __init__(self, first="20200302", last="20261116", fail_after=None):
        self.all = _days(first, last)
        self.calls = 0
        self.fail_after = fail_after

    def __call__(self, code, date_from, date_to, env_dv):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("rate limit")
        rows = [d for d in self.all if date_from <= d <= date_to][-100:][::-1]
        if not rows:
            return None
        return pd.DataFrame({"date": rows, "open": 1.0, "high": 1.0, "low": 1.0, "close": 10.0, "volume": 1, "amount": 1})


class NoWait:
    def wait(self):
        pass


class BackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self._log, g040.log = g040.log, lambda *a, **k: None   # keep the real runtime log clean

    def tearDown(self):
        g040.log = self._log
        self.tmp.cleanup()

    def test_pages_back_to_since(self):
        api, prog = FakeApi(), {}
        st = g040.backfill_code("005930", "20220101", "20261116", "real", NoWait(), self.d, prog, False, fetch=api)
        self.assertEqual(st, "done")
        df = pd.read_csv(self.d / "005930.csv", dtype={"date": str})
        self.assertEqual(df["date"].min(), "20220103")
        self.assertEqual(df["date"].max(), "20261116")
        self.assertTrue(df["date"].is_monotonic_increasing and df["date"].is_unique)
        self.assertTrue(prog["005930"]["done"])
        # done -> skipped without a call
        before = api.calls
        self.assertEqual(g040.backfill_code("005930", "20220101", "20261116", "real", NoWait(), self.d, prog, False, fetch=api), "skip")
        self.assertEqual(api.calls, before)

    def test_listing_date_short_page(self):
        api, prog = FakeApi(first="20250602"), {}
        self.assertEqual(g040.backfill_code("123456", "20180101", "20261116", "real", NoWait(), self.d, prog, False, fetch=api), "done")
        self.assertEqual(prog["123456"]["oldest"], "20250602")

    def test_resume_after_failure(self):
        prog = {}
        st = g040.backfill_code("000660", "20200101", "20261116", "real", NoWait(), self.d, prog, False, fetch=FakeApi(fail_after=3))
        self.assertEqual(st, "partial")
        self.assertFalse(prog["000660"]["done"])
        part_oldest = prog["000660"]["oldest"]
        api = FakeApi()
        self.assertEqual(g040.backfill_code("000660", "20200101", "20261116", "real", NoWait(), self.d, prog, False, fetch=api), "done")
        df = pd.read_csv(self.d / "000660.csv", dtype={"date": str})
        self.assertLess(df["date"].min(), part_oldest)
        self.assertEqual(df["date"].min(), "20200302")
        self.assertTrue(df["date"].is_unique)
        self.assertEqual(len(df), len(FakeApi().all))

    def test_window_and_codes(self):
        wed = datetime(2026, 11, 18)
        self.assertTrue(g040.in_window(wed.replace(hour=21), "20:10-07:40"))
        self.assertTrue(g040.in_window(wed.replace(hour=3), "20:10-07:40"))
        self.assertFalse(g040.in_window(wed.replace(hour=12), "20:10-07:40"))
        self.assertTrue(g040.in_window(datetime(2026, 11, 21, 12), "20:10-07:40"))   # Saturday
        f = self.d / "u.txt"
        f.write_text("code,name,market\n000020,A,KOSPI\n# 000040,B,KOSPI  # auto\n000020,A,KOSPI\n", encoding="utf-8")
        codes = g040.load_codes(f)
        self.assertIn("000040", codes)
        self.assertEqual(codes.count("000020"), 1)
        self.assertEqual(codes[:2], ["229200", "069500"])   # index ETFs first


if __name__ == "__main__":
    unittest.main()

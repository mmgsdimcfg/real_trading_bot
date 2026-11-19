# -*- coding: utf-8 -*-
"""[2026-11-19] Sessions outside the KRX regular one (user: 403870 HPSP does not trade on NXT but trades on the KRX after-hours
market 16:00 - 20:00; the AutoBotH chart stopped at the KRX close):
- r014 trading_hours: derived NXT pre / after and KRX after-hours sessions, CSAT-day overrides on the calendar line
- r018 nxt_flag (cptt_trad_tr_psbl_yn / nxt_tr_stop_yn, cached), minutes(after=True) through the KRX after-hours market
- r017 bars --source kis-ext / auto without a file: the venue by the NXT flag, hours + nxt on the answer
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, time, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_trading"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r014_market_calendar as r014  # noqa: E402
import r018_kis_market as r018  # noqa: E402
from test_r018_kis_market import Base  # noqa: E402


class TradingHoursTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cal = Path(self.tmp.name) / "cal.txt"
        self.cal.write_text(
            "20261119 10:00 16:30 nxt_pre=none nxt_after=16:40-20:00 krx_after=17:00-20:00  # CSAT\n"
            "20270104 10:00 15:30\n"
            "20261225 closed\n"
            "20261120 09:00 15:30 krx_after=bad nxt_pre=07:30-08:50\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_normal_day(self):
        h = r014.trading_hours("20261118", self.cal)
        self.assertEqual(h["regular"], (time(9, 0), time(15, 30)))
        self.assertEqual(h["nxt_pre"], (time(8, 0), time(8, 50)))
        self.assertEqual(h["nxt_after"], (time(15, 40), time(20, 0)))
        self.assertEqual(h["krx_after"], (time(16, 0), time(20, 0)))

    def test_csat_day_overrides_and_late_open_derivation(self):
        self.assertEqual(r014.trading_hours_json("20261119", self.cal),
                         {"regular": ["10:00", "16:30"], "nxt_pre": None, "nxt_after": ["16:40", "20:00"],
                          "krx_after": ["17:00", "20:00"]})
        h = r014.trading_hours("20270104", self.cal)          # late open only: no NXT pre-market, after sessions as usual
        self.assertIsNone(h["nxt_pre"])
        self.assertEqual(h["krx_after"], (time(16, 0), time(20, 0)))

    def test_closed_days_and_bad_tokens(self):
        self.assertIsNone(r014.trading_hours("20261225", self.cal))
        self.assertIsNone(r014.trading_hours("20261121", self.cal))   # Saturday
        h = r014.trading_hours("20261120", self.cal)
        self.assertEqual(h["krx_after"], (time(16, 0), time(20, 0)))   # bad token ignored
        self.assertEqual(h["nxt_pre"], (time(7, 30), time(8, 50)))
        # the old API is unchanged by the extra tokens
        self.assertEqual(r014.session_hours("20261119", self.cal), (time(10, 0), time(16, 30)))

    def test_real_calendar_file_parses(self):
        self.assertEqual(r014.trading_hours_json("20261119")["krx_after"], ["17:00", "20:00"])


def info(psbl: str, stop: str = "N"):
    def h(params):
        return {"output": {"prdt_abrv_name": "HPSP", "mket_id_cd": "KSQ", "cptt_trad_tr_psbl_yn": psbl, "nxt_tr_stop_yn": stop}}
    return h


class NxtFlagTest(Base):
    def test_flag_cached_for_a_week(self):
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 19, 17, 0)):
            c, s = self.client({"CTPF1002R": info("N")})
            self.assertIs(r018.nxt_flag("403870", c), False)
            self.assertEqual(len(s.calls), 1)
            self.assertIs(r018.nxt_flag("403870", c), False)        # cached
            self.assertEqual(len(s.calls), 1)
            self.assertIs(r018.nxt_flag("403870"), False)            # no client: the cache
            names = json.loads((self.runtime / "cache" / "kis" / "names.json").read_text(encoding="utf-8"))
            self.assertEqual(names["403870"]["name"], "HPSP")
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 27, 9, 0)):
            c2, s2 = self.client({"CTPF1002R": info("Y")})
            self.assertIs(r018.nxt_flag("403870", c2), True)         # 8 days later: asked again
            self.assertEqual(len(s2.calls), 1)

    def test_trading_halt_on_nxt_and_unknown(self):
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 19, 17, 0)):
            c, _ = self.client({"CTPF1002R": info("Y", stop="Y")})
            self.assertIs(r018.nxt_flag("214450", c), False)
            c2, _ = self.client({"CTPF1002R": info("")})
            self.assertIsNone(r018.nxt_flag("005930", c2))
            self.assertIsNone(r018.nxt_flag("005930"))

    def test_remember_name_keeps_the_flag(self):
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 19, 17, 0)):
            c, _ = self.client({"CTPF1002R": info("Y")})
            r018.nxt_flag("214450", c)
            r018.remember_name("214450", "파마리서치")
            self.assertIs(r018.nxt_flag("214450"), True)


def krx_minutes(date_str: str, traded_after: set, last_any: str = "200000"):
    """J minutes: every minute 09:00..15:30 traded; after the close KIS gives a row per minute up to last_any, volume only in
    traded_after (the others are flat rows at the last price)."""
    def h(params):
        assert params["FID_COND_MRKT_DIV_CODE"] == "J"
        hour = min(params["FID_INPUT_HOUR_1"], last_any)
        mins = []
        t = datetime.strptime(date_str + "090000", "%Y%m%d%H%M%S")
        while t.strftime("%H%M%S") <= hour:
            mins.append(t.strftime("%H%M%S"))
            t += timedelta(minutes=1)
        page = list(reversed(mins))[:30]

        def vol(m):
            return "5" if (m <= "153000" or m in traded_after) else "0"

        return {"output1": {"hts_kor_isnm": "HPSP"},
                "output2": [{"stck_bsop_date": date_str, "stck_cntg_hour": m, "stck_oprc": "10", "stck_hgpr": "11",
                             "stck_lwpr": "9", "stck_prpr": "10", "cntg_vol": vol(m)} for m in page]}
    return h


class AfterHoursMinutesTest(Base):
    def test_past_day_through_the_after_hours_market(self):
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 19, 9, 0)):
            c, s = self.client({"FHKST03010230": krx_minutes("20261118", {"160000", "195900"}, last_any="195900")})
            m = r018.minutes("403870", "20261118", client=c, after=True)
        hms = [r[0] for r in m["rows"]]
        self.assertEqual(hms[0], "090000")
        self.assertEqual(hms[-3:], ["153000", "160000", "195900"])     # 15:31..15:59 flat rows dropped
        self.assertTrue(m["complete"])
        self.assertTrue(m["source"].endswith(":after"))
        self.assertTrue((self.runtime / "cache" / "kis" / "min_ax" / "20261118" / "403870.json").is_file())
        self.assertFalse((self.runtime / "cache" / "kis" / "min" / "20261118" / "403870.json").exists())
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 19, 9, 0)):
            c2, s2 = self.client({})
            again = r018.minutes("403870", "20261118", client=c2, after=True)   # cached, filtered the same way
        self.assertEqual([r[0] for r in again["rows"]], hms)
        self.assertEqual(s2.calls, [])

    def test_default_minutes_stay_regular(self):
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 19, 9, 0)):
            c, _ = self.client({"FHKST03010230": krx_minutes("20261118", {"160000"}, last_any="195900")})
            m = r018.minutes("403870", "20261118", client=c)
        self.assertEqual(m["rows"][-1][0], "153000")


class R017ExtTest(Base):
    def run_bars(self, nxt, source, j_rows, un_rows, ja_rows, date="20261118"):
        import r017_chart_data as r017
        seen = []

        def fake_minutes(code, date_str, client=None, market="J", after=False):
            seen.append(market + ("+after" if after else ""))
            rows = un_rows if market == "UN" else ja_rows if after else j_rows
            if not rows:
                return None
            return {"rows": rows, "name": "X", "source": f"kis:{market}", "complete": True, "calls": 1}

        with mock.patch.object(r017, "_intraday_bars", return_value=None), mock.patch.object(r018, "minutes", fake_minutes), \
                mock.patch.object(r018, "nxt_flag", lambda code, client=None: nxt), mock.patch.object(r018, "shared_client", lambda: None):
            out = r017.bars(date, "403870", "min", source=source)
        return out, seen

    @staticmethod
    def rows(*hms):
        return [[h, 10, 11, 9, 10, 5] for h in hms]

    def test_ext_krx_only_symbol(self):
        out, seen = self.run_bars(False, "kis-ext", None, None, self.rows("090000", "153000", "160000", "195900"))
        self.assertEqual(seen, ["J+after"])
        self.assertEqual([b[0][11:16] for b in out["bars"]], ["16:00", "19:59"])
        self.assertIs(out["nxt"], False)
        self.assertEqual(out["hours"]["krx_after"], ["16:00", "20:00"])

    def test_ext_nxt_symbol_and_empty_answer(self):
        out, seen = self.run_bars(True, "kis-ext", None, self.rows("080000", "090000", "153000", "170000"), None)
        self.assertEqual(seen, ["UN"])
        self.assertEqual([b[0][11:16] for b in out["bars"]], ["08:00", "17:00"])
        self.assertIs(out["nxt"], True)
        out, _ = self.run_bars(False, "kis-ext", None, None, self.rows("090000", "153000"))
        self.assertEqual((out["ok"], out["count"]), (True, 0))

    def test_auto_without_a_file(self):
        out, seen = self.run_bars(False, "auto", None, None, self.rows("090000", "153000", "170000"))
        self.assertEqual((seen, out["count"]), (["J+after"], 3))
        out, seen = self.run_bars(True, "auto", self.rows("090000", "153000"), self.rows("080000", "090000", "170000"), None)
        self.assertEqual(seen, ["UN", "J"])
        self.assertEqual([b[0][11:16] for b in out["bars"]], ["08:00", "09:00", "15:30", "17:00"])

    def test_csat_day_regular_session(self):
        out, _ = self.run_bars(False, "kis-ext", None, None, self.rows("100000", "160000", "163000", "170000"), date="20261119")
        self.assertEqual([b[0][11:16] for b in out["bars"]], ["17:00"])
        self.assertEqual(out["hours"]["regular"], ["10:00", "16:30"])


if __name__ == "__main__":
    unittest.main()

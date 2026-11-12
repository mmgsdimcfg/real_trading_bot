"""r018 KIS read-only market data (AutoBotH quote / bars fallback) - mocked HTTP, temp KIS config + runtime."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

import r018_kis_market as r018  # noqa: E402


class FakeResp:
    def __init__(self, body: dict, status: int = 200, headers: dict | None = None):
        self._body = body
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeSession:
    """Routes GET by tr_id header to a handler(params) -> body dict (rt_cd 0 added)."""

    def __init__(self, handlers: dict):
        self.handlers = handlers
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, headers=None, params=None, timeout=None):
        tr_id = headers["tr_id"]
        self.calls.append((tr_id, dict(params)))
        h = self.handlers[tr_id]
        out = h(params)
        if isinstance(out, FakeResp):
            return out
        return FakeResp(dict({"rt_cd": "0", "msg_cd": "MCA00000", "msg1": "ok"}, **out))

    def post(self, *a, **k):  # pragma: no cover - must never be used
        raise AssertionError("POST must never be used")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.kis = root / "KIS"
        self.kis.mkdir()
        (self.kis / "kis_devlp.yaml").write_text(
            "prod: https://example.invalid:9443\nmy_app: APPKEY_SECRET\nmy_sec: APPSECRET_SECRET\nmy_agent: test\n", encoding="utf-8")
        self.runtime = root / "runtime"
        self.env = mock.patch.dict(os.environ, {"KIS_CONFIG_DIR": str(self.kis), "AUTO_TRADING_RUNTIME": str(self.runtime)})
        self.env.start()
        self.write_token("KIS20261106", "TOKEN_SECRET", datetime.now() + timedelta(days=30))
        # no real sleeping in the limiter
        self.sleep = mock.patch.object(r018.time, "sleep", lambda s: None)
        self.sleep.start()
        self.rate = mock.patch.object(r018, "_rate_slot", lambda deadline: None)
        self.rate.start()
        r018._SHARED = None

    def tearDown(self):
        self.rate.stop()
        self.sleep.stop()
        self.env.stop()
        self.tmp.cleanup()

    def write_token(self, name: str, token: str, valid: datetime):
        (self.kis / name).write_text(f"token: {token}\nvalid-date: {valid:%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")

    def client(self, handlers: dict) -> tuple[r018.Client, FakeSession]:
        s = FakeSession(handlers)
        return r018.Client(session=s), s


class TokenTest(Base):
    def test_picks_newest_valid_and_skips_empty_or_expired(self):
        (self.kis / "KIS20261107").write_text("", encoding="utf-8")  # 0-byte file (failed issuance)
        self.write_token("KIS20261105", "OLD", datetime.now() - timedelta(hours=1))
        self.assertEqual(r018.load_token(), "TOKEN_SECRET")

    def test_prefers_per_server_prod_file_and_ignores_vps(self):
        # kis_auth (prod/vps split): KIS<date> is a 0-byte placeholder, the real token sits in KIS<date>_prod
        (self.kis / "KIS20261111").write_text("", encoding="utf-8")
        self.write_token("KIS20261111_prod", "PROD_SECRET", datetime.now() + timedelta(hours=20))
        self.write_token("KIS20261111_vps", "PAPER_SECRET", datetime.now() + timedelta(hours=20))
        self.write_token("KIS20261112_vps", "PAPER_NEWER", datetime.now() + timedelta(hours=40))
        self.assertEqual(r018.load_token(), "PROD_SECRET")

    def test_no_valid_token_raises_and_never_issues(self):
        for p in self.kis.glob("KIS2*"):
            p.unlink()
        self.write_token("KIS20261105", "OLD", datetime.now() + timedelta(seconds=30))  # expires within 60 s
        with mock.patch.object(r018.requests, "post") as post, self.assertRaises(r018.KisError) as cm:
            r018.load_token()
        self.assertEqual(cm.exception.code, "token_not_ready")
        post.assert_not_called()

    def test_module_does_not_import_order_libraries(self):
        src = (ROOT / "live_trading" / "r018_kis_market.py").read_text(encoding="utf-8")
        imports = "\n".join(l for l in src.splitlines() if l.lstrip().startswith(("import ", "from ")))
        for bad in ("kis_auth", "domestic_stock_functions", "order", "balance"):
            self.assertNotIn(bad, imports)
        self.assertNotIn("requests.post", src)
        self.assertNotIn(".post(", src)


class GetTest(Base):
    def test_rejects_unknown_tr(self):
        c, _ = self.client({})
        with self.assertRaises(ValueError):
            c.get("order_cash", {})

    def test_egw00201_retries_once_then_raises(self):
        n = {"k": 0}

        def busy(params):
            n["k"] += 1
            return FakeResp({"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초당 거래건수를 초과하였습니다."}, status=500)

        c, s = self.client({"FHKST01010100": busy})
        with self.assertRaises(r018.KisError) as cm:
            c.get("inquire_price", {"FID_INPUT_ISCD": "005930"})
        self.assertEqual(cm.exception.code, "EGW00201")
        self.assertEqual(n["k"], 2)

    def test_error_message_has_no_secrets(self):
        c, _ = self.client({"FHKST01010100": lambda p: FakeResp({"rt_cd": "1", "msg_cd": "EGW00123", "msg1": "기간이 만료된 token"})})
        with self.assertRaises(r018.KisError) as cm:
            c.get("inquire_price", {})
        text = str(cm.exception)
        for secret in ("TOKEN_SECRET", "APPKEY_SECRET", "APPSECRET_SECRET"):
            self.assertNotIn(secret, text)


def price_body(params):
    return {"output": {"stck_prpr": "71200", "prdy_vrss": "800", "prdy_ctrt": "1.14", "stck_sdpr": "70400", "stck_oprc": "70600",
                       "stck_hgpr": "71500", "stck_lwpr": "70500", "acml_vol": "8123456", "acml_tr_pbmn": "578000000000",
                       "stck_mxpr": "91500", "stck_llam": "49300", "w52_hgpr": "88800", "w52_hgpr_date": "20260712",
                       "w52_lwpr": "49900", "w52_lwpr_date": "20251203", "per": "13.20", "pbr": "1.21", "hts_avls": "4250000",
                       "vi_cls_code": "N", "temp_stop_yn": "N"}}


def book_body(params):
    b = {"aspr_acpt_hour": "093101", "total_askp_rsqn": "210000", "total_bidp_rsqn": "190000"}
    for i in range(1, 11):
        b[f"askp{i}"] = str(71200 + 100 * i)
        b[f"askp_rsqn{i}"] = str(1000 * i)
        b[f"bidp{i}"] = str(71300 - 100 * i)
        b[f"bidp_rsqn{i}"] = str(2000 * i)
    return {"output1": b, "output2": {}}


def ccnl_body(params):
    return {"output": [{"stck_cntg_hour": "093101", "stck_prpr": "71200", "cntg_vol": "15", "tday_rltv": "103.2"},
                       {"stck_cntg_hour": "093100", "stck_prpr": "71100", "cntg_vol": "3", "tday_rltv": "103.1"}]}


def info_body(params):
    return {"output": {"prdt_abrv_name": "삼성전자", "mket_id_cd": "STK"}}


QUOTE_HANDLERS = {"FHKST01010100": price_body, "FHKST01010200": book_body, "FHKST01010300": ccnl_body, "CTPF1002R": info_body}


class QuoteTest(Base):
    def test_quote_shape_and_cache(self):
        c, s = self.client(QUOTE_HANDLERS)
        q = r018.quote("005930", trades=1, client=c)
        self.assertTrue(q["ok"])
        self.assertEqual((q["name"], q["market"], q["price"], q["prev_close"]), ("삼성전자", "KOSPI", 71200, 70400))
        self.assertEqual(q["orderbook"]["asks"][0], [71300, 1000])
        self.assertEqual(q["orderbook"]["bids"][0], [71200, 2000])
        self.assertEqual(q["orderbook"]["asof"], "09:31:01")
        self.assertEqual(q["trades"], [["09:31:01", 71200, 15, ""]])
        self.assertEqual(q["strength"], 103.2)
        self.assertFalse(q["vi"])
        self.assertEqual(len(s.calls), 4)
        # second call within 2 s -> cache, no HTTP
        c2, s2 = self.client(QUOTE_HANDLERS)
        q2 = r018.quote("005930", trades=30, client=c2)
        self.assertEqual(s2.calls, [])
        self.assertEqual(len(q2["trades"]), 2)
        # name cached -> later quotes skip search_stock_info
        cache = self.runtime / "cache" / "kis" / "quote" / "005930.json"
        data = json.loads(cache.read_text())
        data["_at"] = time.time() - 10
        cache.write_text(json.dumps(data))
        c3, s3 = self.client(QUOTE_HANDLERS)
        r018.quote("005930", client=c3)
        self.assertNotIn("CTPF1002R", [t for t, _ in s3.calls])
        self.assertNotIn("TOKEN_SECRET", cache.read_text())

    def test_partial_failure_keeps_price(self):
        h = dict(QUOTE_HANDLERS, FHKST01010200=lambda p: FakeResp({"rt_cd": "1", "msg_cd": "EGW00999", "msg1": "x"}))
        c, _ = self.client(h)
        q = r018.quote("005930", client=c)
        self.assertIsNone(q["orderbook"])
        self.assertTrue(q["partial_errors"])

    def test_invalid_code(self):
        with self.assertRaises(ValueError):
            r018.quote("12345")


class DailyTest(Base):
    def test_paging_and_cache(self):
        days = [(datetime(2026, 11, 6) - timedelta(days=i)).strftime("%Y%m%d") for i in range(150)]

        def handler(params):
            end = params["FID_INPUT_DATE_2"]
            page = [d for d in days if d <= end][:100]
            return {"output1": {"hts_kor_isnm": "삼성전자"},
                    "output2": [{"stck_bsop_date": d, "stck_oprc": "1", "stck_hgpr": "2", "stck_lwpr": "1", "stck_clpr": "2",
                                 "acml_vol": "10"} for d in page]}

        c, s = self.client({"FHKST03010100": handler})
        d = r018.daily("005930", client=c)
        self.assertEqual(len(d["rows"]), 150)
        self.assertEqual(d["rows"][0][0], days[-1])
        self.assertEqual(len(s.calls), 2)
        c2, s2 = self.client({"FHKST03010100": handler})
        self.assertEqual(r018.daily("005930", client=c2)["calls"], 0)
        self.assertEqual(s2.calls, [])

    def test_fresh_rules(self):
        now = datetime(2026, 11, 9, 11, 0)
        self.assertTrue(r018._daily_fresh((now - timedelta(minutes=5)).timestamp(), now))
        self.assertFalse(r018._daily_fresh((now - timedelta(minutes=11)).timestamp(), now))
        late = datetime(2026, 11, 9, 20, 0)
        self.assertTrue(r018._daily_fresh(datetime(2026, 11, 9, 18, 5).timestamp(), late))
        self.assertFalse(r018._daily_fresh(datetime(2026, 11, 9, 16, 40).timestamp(), late))
        self.assertFalse(r018._daily_fresh(datetime(2026, 11, 8, 20, 0).timestamp(), late))


def minute_handler(date_str: str, last: str = "100000", prev_day: bool = True):
    """30 rows per call ending at FID_INPUT_HOUR_1 (newest first), from 09:00 to `last`; plus a previous-day row."""
    mins = []
    t = datetime.strptime(date_str + "090000", "%Y%m%d%H%M%S")
    end = datetime.strptime(date_str + last, "%Y%m%d%H%M%S")
    while t <= end:
        mins.append(t.strftime("%H%M%S"))
        t += timedelta(minutes=1)

    def h(params):
        hour = params["FID_INPUT_HOUR_1"]
        page = [m for m in reversed(mins) if m <= hour][:30]
        rows = [{"stck_bsop_date": date_str, "stck_cntg_hour": m, "stck_oprc": "10", "stck_hgpr": "11", "stck_lwpr": "9",
                 "stck_prpr": "10", "cntg_vol": "5"} for m in page]
        if prev_day:
            rows.append({"stck_bsop_date": "20000101", "stck_cntg_hour": "152900", "stck_oprc": "1", "stck_hgpr": "1",
                         "stck_lwpr": "1", "stck_prpr": "1", "cntg_vol": "1"})
        return {"output1": {"hts_kor_isnm": "카카오"}, "output2": rows}
    return h


class MinutesTest(Base):
    def test_today_full_then_incremental(self):
        now = datetime(2026, 11, 9, 10, 0, 30)
        with mock.patch.object(r018, "_now", return_value=now):
            c, s = self.client({"FHKST03010200": minute_handler("20261109")})
            m = r018.minutes("035720", "20261109", client=c)
            self.assertEqual(len(m["rows"]), 61)            # 09:00..10:00, previous-day row dropped
            self.assertEqual(m["rows"][0][0], "090000")
            self.assertEqual(len(s.calls), 3)
            cpath = self.runtime / "cache" / "kis" / "min" / "20261109" / "035720.json"
            data = json.loads(cpath.read_text())
            self.assertTrue(data["complete_from_open"])
            data["fetched_at"] = time.time() - 60
            cpath.write_text(json.dumps(data))
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 9, 10, 5, 30)):
            c2, s2 = self.client({"FHKST03010200": minute_handler("20261109", last="100500")})
            m2 = r018.minutes("035720", "20261109", client=c2)
            self.assertEqual(len(s2.calls), 1)               # only the newest page
            self.assertEqual(m2["rows"][-1][0], "100500")
            self.assertEqual(len(m2["rows"]), 66)

    def test_incomplete_cache_refetches_from_open(self):
        now = datetime(2026, 11, 9, 10, 0, 30)
        cpath = self.runtime / "cache" / "kis" / "min" / "20261109" / "035720.json"
        cpath.parent.mkdir(parents=True)
        cpath.write_text(json.dumps({"fetched_at": time.time() - 60, "rows": [["095900", 1, 1, 1, 1, 1]], "complete_from_open": False}))
        with mock.patch.object(r018, "_now", return_value=now):
            c, s = self.client({"FHKST03010200": minute_handler("20261109")})
            m = r018.minutes("035720", "20261109", client=c)
        self.assertEqual(len(s.calls), 3)
        self.assertEqual(m["rows"][0][0], "090000")

    def test_past_day_cached_permanently(self):
        now = datetime(2026, 11, 9, 10, 0)
        with mock.patch.object(r018, "_now", return_value=now):
            c, s = self.client({"FHKST03010230": minute_handler("20261106", last="153000")})
            m = r018.minutes("035720", "20261106", client=c)
            self.assertTrue(m["complete"])
            self.assertEqual(m["rows"][-1][0], "153000")
            c2, s2 = self.client({})
            self.assertEqual(r018.minutes("035720", "20261106", client=c2)["calls"], 0)
            self.assertIsNone(r018.minutes("035720", "20261110"))


class UnifiedMinutesTest(Base):
    """[2026-11-12] minutes(market="UN"): KRX + NXT combined bars for the hours KRX is closed."""

    @staticmethod
    def handler(date_str: str, traded: set[str]):
        """Combined market: one row per minute from 08:00; minutes outside `traded` are flat rows with volume 0."""
        def h(params):
            assert params["FID_COND_MRKT_DIV_CODE"] == "UN"
            hour = params["FID_INPUT_HOUR_1"]
            mins = []
            t = datetime.strptime(date_str + "080000", "%Y%m%d%H%M%S")
            while t.strftime("%H%M%S") <= hour:
                mins.append(t.strftime("%H%M%S"))
                t += timedelta(minutes=1)
            page = list(reversed(mins))[:30]
            return {"output1": {"hts_kor_isnm": "OCI홀딩스"},
                    "output2": [{"stck_bsop_date": date_str, "stck_cntg_hour": m, "stck_oprc": "10", "stck_hgpr": "11",
                                 "stck_lwpr": "9", "stck_prpr": "10", "cntg_vol": "5" if m in traded else "0"} for m in page]}
        return h

    def test_pre_market_bars_before_the_krx_open(self):
        now = datetime(2026, 11, 12, 8, 27, 30)
        with mock.patch.object(r018, "_now", return_value=now):
            # KRX (J) has nothing before 09:00 - and is not even asked
            c0, s0 = self.client({})
            self.assertIsNone(r018.minutes("010060", "20261112", client=c0))
            self.assertEqual(s0.calls, [])
            c, s = self.client({"FHKST03010200": self.handler("20261112", {"080000", "080100", "082700"})})
            m = r018.minutes("010060", "20261112", client=c, market="UN")
        self.assertEqual([r[0] for r in m["rows"]], ["080000", "080100", "082700"])      # zero-volume minutes dropped
        self.assertEqual(m["source"], "kis:inquire_time_itemchartprice:UN")
        self.assertTrue((self.runtime / "cache" / "kis" / "min_un" / "20261112" / "010060.json").is_file())
        self.assertFalse((self.runtime / "cache" / "kis" / "min" / "20261112" / "010060.json").exists())

    def test_symbol_without_nxt_trades_returns_none(self):
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 12, 8, 27, 30)):
            c, _ = self.client({"FHKST03010200": self.handler("20261112", set())})
            self.assertIsNone(r018.minutes("006360", "20261112", client=c, market="UN"))

    def test_before_0800_and_bad_market(self):
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 12, 7, 50)):
            c, s = self.client({})
            self.assertIsNone(r018.minutes("010060", "20261112", client=c, market="UN"))
            self.assertEqual(s.calls, [])
        with self.assertRaises(ValueError):
            r018.minutes("010060", "20261112", market="NX")

    def test_r017_bars_fall_back_to_the_combined_market(self):
        import r017_chart_data as r017
        seen = []

        def fake_minutes(code, date_str, client=None, market="J"):
            seen.append(market)
            if market == "J":
                return None
            return {"rows": [["080000", 10, 11, 9, 10, 5]], "name": "OCI홀딩스", "source": "kis:inquire_time_itemchartprice:UN",
                    "complete": False, "calls": 1}

        with mock.patch.object(r017, "_intraday_bars", return_value=None), mock.patch.object(r018, "minutes", fake_minutes):
            out = r017.bars("20261112", "010060", "min")
            self.assertEqual((out["ok"], out["count"], out["source"]), (True, 1, "kis:inquire_time_itemchartprice:UN"))
            self.assertEqual(seen, ["J", "UN"])
            seen.clear()
            out = r017.bars("20261112", "010060", "min", source="kis-un")
            self.assertEqual((out["ok"], seen), (True, ["UN"]))
            seen.clear()
            out = r017.bars("20261112", "010060", "min", source="kis")          # gap filling stays on the KRX basis
            self.assertEqual((out["ok"], seen), (False, ["J"]))
        # a tick file that holds pre-open prints only gives no bar: KIS minutes are used as well
        import pandas as pd
        empty = (pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "n", "t_last"]), "ticks.csv")
        seen.clear()
        with mock.patch.object(r017, "_intraday_bars", return_value=empty), mock.patch.object(r018, "minutes", fake_minutes):
            out = r017.bars("20261112", "010060", "min")
            self.assertEqual((out["ok"], out["count"], seen), (True, 1, ["J", "UN"]))


class R016IntegrationTest(Base):
    def run16(self, argv):
        import r016_control as r016
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = r016.main(argv)
        return rc, json.loads(buf.getvalue())

    def test_quote_rc(self):
        orig = r018.Client
        with mock.patch.object(r018, "Client", lambda *a, **k: orig(session=FakeSession(QUOTE_HANDLERS))):
            rc, out = self.run16(["quote", "--code", "005930", "--trades", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(out["price"], 71200)
        rc, out = self.run16(["quote", "--code", "bad"])
        self.assertEqual(rc, 2)
        for p in self.kis.glob("KIS2*"):
            p.unlink()
        r018._SHARED = None  # a new r016 process
        (self.runtime / "cache" / "kis" / "quote" / "005930.json").unlink()
        rc, out = self.run16(["quote", "--code", "005930"])
        self.assertEqual((rc, out["kis_code"]), (3, "token_not_ready"))

    def test_bars_kis_fallback_and_tick_error(self):
        import r017_chart_data as r017
        days = [(datetime(2026, 11, 6) - timedelta(days=i)).strftime("%Y%m%d") for i in range(10)]

        def daily_h(params):
            return {"output1": {"hts_kor_isnm": "삼성전자"},
                    "output2": [{"stck_bsop_date": d, "stck_oprc": "1", "stck_hgpr": "2", "stck_lwpr": "1", "stck_clpr": "2",
                                 "acml_vol": "10"} for d in days if d <= params["FID_INPUT_DATE_2"]]}

        orig = r018.Client
        with mock.patch.object(r017, "day_bars", return_value=None), \
             mock.patch.object(r018, "Client", lambda *a, **k: orig(session=FakeSession({"FHKST03010100": daily_h}))):
            rc, out = self.run16(["bars", "--date", "20261106", "--code", "005930", "--kind", "day"])
        self.assertEqual(rc, 0)
        self.assertEqual(out["source"], "kis:inquire_daily_itemchartprice")
        self.assertEqual(out["name"], "삼성전자")
        self.assertEqual(out["columns"], ["t", "o", "h", "l", "c", "v"])
        self.assertEqual(out["count"], 10)
        with mock.patch.object(r017, "_intraday_bars", return_value=None):
            rc, out = self.run16(["bars", "--date", "20261106", "--code", "005930", "--kind", "tick"])
        self.assertEqual(rc, 2)
        self.assertIn("tick collector", out["error"])
        for p in self.kis.glob("KIS2*"):
            p.unlink()
        r018._SHARED = None  # a new r016 process
        with mock.patch.object(r017, "_intraday_bars", return_value=None):
            rc, out = self.run16(["bars", "--date", "20261106", "--code", "035720", "--kind", "min"])
        self.assertEqual((rc, out.get("kis_code")), (3, "token_not_ready"))


class CodexFindingsTest(Base):
    """Codex review 2026-11-07: deadline, interrupted incremental, cache failures, truncated daily, sessions, empty daily."""

    def test_expired_deadline_never_calls_http(self):
        c, s = self.client(QUOTE_HANDLERS)
        c.deadline = time.time() - 1
        with self.assertRaises(r018.KisError) as cm:
            c.get("inquire_price", {})
        self.assertEqual(cm.exception.code, "deadline")
        self.assertEqual(s.calls, [])

    def test_http_timeout_bounded_by_remaining_budget(self):
        seen = {}

        class S(FakeSession):
            def get(self, url, headers=None, params=None, timeout=None):
                seen["timeout"] = timeout
                return super().get(url, headers=headers, params=params, timeout=timeout)

        c = r018.Client(deadline_sec=2.0, session=S(QUOTE_HANDLERS))
        c.get("inquire_price", {})
        self.assertLessEqual(seen["timeout"][1], 2.0)

    def test_shared_client_is_one_per_process(self):
        self.assertIs(r018.shared_client(), r018.shared_client())

    def test_interrupted_incremental_does_not_keep_from_open(self):
        cpath = self.runtime / "cache" / "kis" / "min" / "20261109" / "035720.json"
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 9, 10, 0, 30)):
            c, _ = self.client({"FHKST03010200": minute_handler("20261109")})
            r018.minutes("035720", "20261109", client=c)
        data = json.loads(cpath.read_text())
        data["fetched_at"] = time.time() - 60
        cpath.write_text(json.dumps(data))
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 9, 12, 0, 30)):
            c2, s2 = self.client({"FHKST03010200": minute_handler("20261109", last="120000")})
            real = c2.time_left
            n = {"k": 0}

            def budget():  # first page fetched, then the budget runs out
                n["k"] += 1
                return real() if n["k"] <= 3 else 0.0
            c2.time_left = budget
            r018.minutes("035720", "20261109", client=c2)
        self.assertFalse(json.loads(cpath.read_text())["complete_from_open"])
        data = json.loads(cpath.read_text())
        data["fetched_at"] = time.time() - 60
        cpath.write_text(json.dumps(data))
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 9, 12, 0, 40)):
            c3, s3 = self.client({"FHKST03010200": minute_handler("20261109", last="120000")})
            m = r018.minutes("035720", "20261109", client=c3)
        self.assertEqual(len(m["rows"]), 181)  # 09:00..12:00 without a hole
        self.assertTrue(json.loads(cpath.read_text())["complete_from_open"])

    def test_cache_write_failure_still_returns_quote(self):
        with mock.patch.object(r018.tempfile, "mkstemp", side_effect=PermissionError("ro")):
            c, _ = self.client(QUOTE_HANDLERS)
            q = r018.quote("005930", client=c)
        self.assertEqual(q["price"], 71200)

    def test_rate_storage_failure_blocks_call(self):
        self.rate.stop()
        try:
            with mock.patch.object(r018, "_rate_slot_inner", side_effect=PermissionError("ro")):
                c, s = self.client(QUOTE_HANDLERS)
                with self.assertRaises(r018.KisError) as cm:
                    c.get("inquire_price", {})
            self.assertEqual(cm.exception.code, "rate_storage")
            self.assertEqual(s.calls, [])
        finally:
            self.rate.start()

    def test_truncated_daily_is_refetched(self):
        days = [(datetime(2026, 11, 6) - timedelta(days=i)).strftime("%Y%m%d") for i in range(150)]

        def handler(params):
            end = params["FID_INPUT_DATE_2"]
            return {"output1": {}, "output2": [{"stck_bsop_date": d, "stck_oprc": "1", "stck_hgpr": "2", "stck_lwpr": "1",
                                                "stck_clpr": "2", "acml_vol": "1"} for d in days if d <= end][:100]}

        c, _ = self.client({"FHKST03010100": handler})
        c.time_left = lambda: 2.5  # enough for one page, not for paging on
        d = r018.daily("005930", client=c)
        self.assertFalse(d["complete"])
        self.assertEqual(len(d["rows"]), 100)
        c2, s2 = self.client({"FHKST03010100": handler})
        d2 = r018.daily("005930", client=c2)
        self.assertTrue(d2["complete"])
        self.assertEqual(len(d2["rows"]), 150)

    def test_session_hours_from_calendar(self):
        with mock.patch("r014_market_calendar.session_hours", return_value=(datetime(2000, 1, 1, 10, 0).time(), datetime(2000, 1, 1, 16, 30).time())):
            self.assertEqual(r018._session_hours("20261119"), ("100000", "163000"))
        with mock.patch("r014_market_calendar.session_hours", return_value=None):
            self.assertIsNone(r018.minutes("035720", "20261225"))

    def test_before_open_today_returns_none_without_http(self):
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 9, 8, 40)):
            c, s = self.client({})
            self.assertIsNone(r018.minutes("035720", "20261109", client=c))
        self.assertEqual(s.calls, [])

    def test_empty_filtered_daily_is_rc2(self):
        import r017_chart_data as r017
        with mock.patch.object(r017, "day_bars", return_value=None), \
             mock.patch.object(r017, "kis_day_bars", return_value=(r017.pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "n", "t_last"]), "kis:x", None)):
            out = r017.bars("20200101", "005930", "day")
        self.assertFalse(out["ok"])


class CodexSecondPassTest(Base):
    """Codex 2nd review 2026-11-07: absolute deadline, partial pages on HTTP timeout, empty pages, legacy caches, error text."""

    def test_hard_deadline_interrupts_slow_streaming_call(self):
        import r016_control as r016

        def slow(params):
            for _ in range(200):          # bytes keep arriving: no read timeout would fire
                time.sleep(0.01)
            return price_body(params)

        self.sleep.stop()                 # real sleeping here
        try:
            orig = r018.Client
            with mock.patch.object(r016, "COMMAND_HARD_DEADLINE_SEC", 0.3), \
                 mock.patch.object(r018, "Client", lambda *a, **k: orig(session=FakeSession(dict(QUOTE_HANDLERS, FHKST01010100=slow)))):
                buf = io.StringIO()
                t0 = time.time()
                with contextlib.redirect_stdout(buf):
                    rc = r016.main(["quote", "--code", "005930"])
            self.assertLess(time.time() - t0, 1.5)
            out = json.loads(buf.getvalue())
            self.assertEqual((rc, out["kis_code"]), (3, "deadline"))
        finally:
            self.sleep.start()

    def test_page_two_http_timeout_keeps_page_one(self):
        days = [(datetime(2026, 11, 6) - timedelta(days=i)).strftime("%Y%m%d") for i in range(150)]
        n = {"k": 0}

        class S(FakeSession):
            def get(self, url, headers=None, params=None, timeout=None):
                n["k"] += 1
                if n["k"] == 2:
                    raise r018.requests.ReadTimeout("slow")
                return super().get(url, headers=headers, params=params, timeout=timeout)

        def handler(params):
            end = params["FID_INPUT_DATE_2"]
            return {"output1": {}, "output2": [{"stck_bsop_date": d, "stck_oprc": "1", "stck_hgpr": "2", "stck_lwpr": "1",
                                                "stck_clpr": "2", "acml_vol": "1"} for d in days if d <= end][:100]}

        c = r018.Client(session=S({"FHKST03010100": handler}))
        d = r018.daily("005930", client=c)
        self.assertEqual((len(d["rows"]), d["complete"]), (100, False))

    def test_empty_first_page_does_not_mark_continuity(self):
        cpath = self.runtime / "cache" / "kis" / "min" / "20261106" / "035720.json"
        cpath.parent.mkdir(parents=True)
        cpath.write_text(json.dumps({"fetched_at": 0, "rows": [["120000", 1, 1, 1, 1, 1]], "complete": False,
                                     "complete_from_open": False}))
        with mock.patch.object(r018, "_now", return_value=datetime(2026, 11, 9, 10, 0)):
            c, _ = self.client({"FHKST03010230": lambda p: {"output1": {}, "output2": []}})
            m = r018.minutes("035720", "20261106", client=c)
        self.assertFalse(m["complete"])
        self.assertFalse(json.loads(cpath.read_text())["complete_from_open"])

    def test_legacy_daily_cache_without_flag_is_refetched(self):
        cpath = self.runtime / "cache" / "kis" / "daily" / "005930.json"
        cpath.parent.mkdir(parents=True)
        cpath.write_text(json.dumps({"fetched_at": time.time(), "rows": [["20261106", 1, 2, 1, 2, 1]], "name": None}))
        c, s = self.client({"FHKST03010100": lambda p: {"output1": {}, "output2": [
            {"stck_bsop_date": "20261106", "stck_oprc": "1", "stck_hgpr": "2", "stck_lwpr": "1", "stck_clpr": "2", "acml_vol": "1"}]}})
        r018.daily("005930", client=c)
        self.assertEqual(len(s.calls), 1)

    def test_chart_exception_text_is_not_echoed(self):
        import r016_control as r016
        import r017_chart_data as r017
        with mock.patch.object(r017, "run_command", side_effect=RuntimeError("SECRET_SENTINEL")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = r016.main(["bars", "--date", "20261106", "--code", "005930", "--kind", "day"])
        self.assertEqual(rc, 3)
        self.assertNotIn("SECRET_SENTINEL", buf.getvalue())

    def test_quote_skips_optional_requests_when_budget_low(self):
        c, s = self.client(QUOTE_HANDLERS)
        real = c.time_left
        n = {"k": 0}

        def budget():
            n["k"] += 1
            return real() if n["k"] <= 4 else 1.0   # price + name fetched, then low budget
        c.time_left = budget
        q = r018.quote("005930", client=c)
        self.assertEqual(q["price"], 71200)
        self.assertIsNone(q["orderbook"])
        self.assertEqual(len(q["partial_errors"]), 2)


class RateLimitTest(unittest.TestCase):
    def test_spacing_and_per_second_cap(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"AUTO_TRADING_RUNTIME": d}):
            t0 = time.time()
            for _ in range(4):
                r018._rate_slot(time.time() + 10)
            elapsed = time.time() - t0
            # 4 calls at max 2/s with 0.3 s gap -> at least ~1.0 s
            self.assertGreaterEqual(elapsed, 0.95)
            with self.assertRaises(r018.KisError):
                r018._rate_slot(time.time() + 0.05)  # cannot get a slot before the deadline


if __name__ == "__main__":
    unittest.main()

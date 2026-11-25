# -*- coding: utf-8 -*-
"""[2026-11-19] r022 paper trading of the tick collector's symbols (user: include the collector symbols in the paper trading):
CollectorTickFeed (tick file -> confirmed end-labeled 1m bars + last price, incremental reads, torn line, reconnect replays,
regular session only), per-group limits (watch / collector counted separately), "group" on the records, restart restore,
poll_collector (exclude the watched symbols, throttle, keeps checking exits of held collector symbols)."""
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r022_edge_only as r022  # noqa: E402
from test_r022_edge_only import DAY, Base, frame  # noqa: E402

HEADER = ("recv_ts,code,bsop_date,trade_time,price,cntg_vol,acml_vol,ccld_dvsn,seln_cntg_csnu,shnu_cntg_csnu,askp1,bidp1,"
          "hour_cls_code,venue,session")


def row(code: str, hms: str, price: float, qty: int, acml: int, session: str = "regular") -> str:
    return f"2026-11-12T00:00:00.000,{code},{DAY},{hms},{price},{qty},{acml},1,0,0,0,0,0,KRX,{session}"


class FeedTest(Base):
    def setUp(self):
        super().setUp()
        self.ticks = self.out / "ticks" / DAY
        self.ticks.mkdir(parents=True)

    def write(self, code: str, lines: list[str], header: bool = True, end: str = "\n") -> None:
        p = self.ticks / f"{code}.csv"
        with p.open("a", encoding="utf-8", newline="") as fh:
            fh.write(("" if not header else HEADER + "\n") + "\n".join(lines) + end)

    def test_closing_auction_print_is_the_auction_price_not_a_regular_price(self):
        # [2026-11-25] paper regular close: filled at the 15:30 closing auction print (session close_auction)
        self.write("010060", [row("010060", "151959", 10000, 3, 3)])
        feed = r022.CollectorTickFeed(self.ticks, DAY, 2.0)
        self.assertIsNone(feed.auction_price("010060"))
        self.write("010060", [row("010060", "153001", 10150, 900, 903, "close_auction"),
                              row("010060", "153001", 10150, 100, 1003, "close_auction"),
                              row("010060", "154000", 10200, 5, 1008, "post")], header=False)
        self.assertEqual(feed.auction_price("010060"), 10150)
        _, price = feed.poll("010060", datetime(2026, 11, 12, 15, 41))
        self.assertEqual(price, 10000)                                        # the last regular price is unchanged
        self.assertEqual(feed.last_print("010060"), (10000, datetime(2026, 11, 12, 15, 19, 59)))
        self.assertIsNone(feed.auction_price("000000"))                       # no file

    def test_bars_are_confirmed_end_labeled_and_handed_out_once(self):
        self.write("010060", [row("010060", "085959", 9990, 5, 5, "pre"), row("010060", "090001", 10000, 3, 8),
                              row("010060", "090030", 10050, 2, 10), row("010060", "090100", 10020, 4, 14)])
        feed = r022.CollectorTickFeed(self.ticks, DAY, 2.0)
        f, price = feed.poll("010060", datetime(2026, 11, 12, 9, 1, 1))     # 09:01:01 - 2 s -> the 09:01 bar is not final
        self.assertIsNone(f)
        self.assertEqual(price, 10020)
        f, price = feed.poll("010060", datetime(2026, 11, 12, 9, 1, 3))
        self.assertEqual([str(t.time()) for t in f.index], ["09:01:00"])     # 09:00:xx prints -> the bar ending 09:01
        self.assertEqual(f.iloc[0].tolist(), [10000, 10050, 10000, 10050, 5])   # the pre-open print is not in it
        f, _ = feed.poll("010060", datetime(2026, 11, 12, 9, 1, 30))
        self.assertIsNone(f)                                                  # nothing new completed
        f, _ = feed.poll("010060", datetime(2026, 11, 12, 9, 2, 5))
        self.assertEqual(len(f), 2)

    def test_incremental_torn_line_and_replays(self):
        self.write("010060", [row("010060", "090001", 10000, 3, 3)], end="\n" + row("010060", "090010", 10100, 1, 4)[:20])
        feed = r022.CollectorTickFeed(self.ticks, DAY)
        _, price = feed.poll("010060", datetime(2026, 11, 12, 9, 0, 30))
        self.assertEqual(price, 10000)                                        # the torn line waits
        with (self.ticks / "010060.csv").open("a", encoding="utf-8", newline="") as fh:
            fh.write(row("010060", "090010", 10100, 1, 4)[20:] + "\n")
            fh.write(row("010060", "090001", 10000, 3, 3) + "\n")             # replayed after a reconnect
            fh.write(row("010060", "090020", 10050, 2, 6) + "\n")
        f, price = feed.poll("010060", datetime(2026, 11, 12, 9, 1, 5))
        self.assertEqual(price, 10050)
        self.assertEqual(f.iloc[0].tolist(), [10000, 10100, 10000, 10050, 6])

    def test_codes(self):
        self.write("010060", [row("010060", "090001", 10000, 3, 3)])
        (self.ticks / "_collector.log").write_text("x", encoding="utf-8")
        (self.ticks / "nxt").mkdir()
        self.assertEqual(r022.CollectorTickFeed(self.ticks, DAY).codes(), ["010060"])
        self.assertEqual(r022.CollectorTickFeed(self.out / "none", DAY).codes(), [])


class GroupTest(Base):
    def test_limits_count_per_group(self):
        r = self.runner(max_positions=1, max_entries=1, collector_max_positions=2, collector_max_entries=2)
        r.on_symbol("000001", "a", self.at("09:30"), frame(30), 10300.0)
        r.on_symbol("000002", "b", self.at("09:30"), frame(30), 10300.0)                       # watch full
        r.on_symbol("000003", "c", self.at("09:30"), frame(30), 10300.0, group="collector")
        r.on_symbol("000004", "d", self.at("09:30"), frame(30), 10300.0, group="collector")
        r.on_symbol("000005", "e", self.at("09:30"), frame(30), 10300.0, group="collector")   # collector full
        self.assertEqual(sorted(r.positions), ["000001", "000003", "000004"])
        rec = self.records()
        self.assertEqual([(x["code"], x["group"]) for x in rec],
                         [("000001", "watch"), ("000003", "collector"), ("000004", "collector")])
        self.assertEqual((r.entries_today, r.entries["collector"]), (1, 2))
        self.assertIn("[수집기]", [x for x in self.logs if "000003" in x][0])

    def test_collector_off_by_default(self):
        r = self.runner()
        r.on_symbol("000003", "c", self.at("09:30"), frame(30), 10300.0, group="collector")
        self.assertEqual(self.records(), [])

    def test_exit_record_and_restore_keep_the_group(self):
        r = self.runner(collector_max_positions=2, collector_max_entries=2)
        r.on_symbol("000003", "c", self.at("09:30"), frame(30), 10300.0, group="collector")
        r2 = self.runner(collector_max_positions=2, collector_max_entries=2)            # r003 restart
        self.assertEqual(r2.positions["000003"]["group"], "collector")
        self.assertEqual((r2.entries_today, r2.entries["collector"]), (0, 1))
        r2.on_symbol("000003", "c", self.at("09:40"), None, 99999.0, group="collector")   # target hit
        self.assertEqual(self.records()[-1]["event"], "EXIT")
        self.assertEqual(self.records()[-1]["group"], "collector")
        # an old record without group restores as watch
        f = self.out / f"edge_only_{DAY}.jsonl"
        f.write_text(json.dumps({"event": "ENTRY", "ts": "2026-11-12T09:30:05", "code": "000009", "name": "z", "entry": 100.0,
                                 "tp_px": 102.0, "sl_px": 98.0, "horizon": 60}) + "\n", encoding="utf-8")
        r3 = self.runner()
        self.assertEqual((r3.positions["000009"]["group"], r3.entries_today), ("watch", 1))


class StatusTest(Base):
    """[2026-11-19] edge_only_status_<date>.json for the app's 가상 매매 tab."""

    def status(self):
        return json.loads((self.out / f"edge_only_status_{DAY}.json").read_text(encoding="utf-8"))

    def test_stages_groups_and_positions(self):
        r = self.runner(prob=0.6, collector_max_positions=2, collector_max_entries=2)
        r.on_symbol("000001", "a", self.at("09:30"), frame(30), 10300.0)                          # bought
        r.on_symbol("000002", "b", self.at("09:30"), frame(30, last_close=9000.0), 9000.0)         # below the BB mid
        r.on_symbol("000003", "c", self.at("09:29"), frame(2, start="09:28"), 10000.0, group="collector")   # too few bars
        self.assertTrue(r.write_status(self.at("09:30"), force=True))
        st = self.status()
        by = {x["code"]: x for x in st["symbols"]}
        self.assertEqual((by["000001"]["stage"], by["000002"]["stage"], by["000003"]["stage"]), ("bought", "below_mid", "wait_bars"))
        self.assertEqual(by["000001"]["prob"], 0.6)
        self.assertEqual(by["000001"]["position"]["entry"], 10310.0)
        self.assertEqual(by["000003"]["group"], "collector")
        self.assertEqual(by["000001"]["bar"], "09:30")
        self.assertEqual(st["groups"]["watch"], {"open": 1, "max_positions": 3, "entries": 1, "max_entries": 10, "enabled": True})
        self.assertEqual(st["groups"]["collector"]["max_entries"], 2)
        self.assertEqual(st["threshold"], 0.45)

    def test_below_threshold_and_held(self):
        r = self.runner(prob=0.40)
        r.on_symbol("000001", "a", self.at("09:30"), frame(30), 10300.0)
        r.write_status(self.at("09:30"), force=True)
        self.assertEqual((self.status()["symbols"][0]["stage"], self.status()["symbols"][0]["prob"]), ("below_threshold", 0.4))
        r2 = self.runner(prob=0.6)
        r2.on_symbol("000005", "e", self.at("09:30"), frame(30), 10300.0)
        r2.on_symbol("000005", "e", self.at("09:31"), frame(31), 10310.0)
        r2.write_status(self.at("09:31"), force=True)
        self.assertEqual({x["code"]: x["stage"] for x in self.status()["symbols"]}["000005"], "held")

    def test_throttle_and_stale_symbols_leave(self):
        r = self.runner(prob=0.1)
        r.on_symbol("000001", "a", self.at("09:30"), frame(30), 10300.0)
        self.assertTrue(r.write_status(self.at("09:30")))
        self.assertFalse(r.write_status(self.at("09:30", 8)))                 # within 5 s
        r.on_symbol("000002", "b", self.at("09:45"), frame(30, start="09:16"), 10300.0)
        self.assertTrue(r.write_status(self.at("09:45")))
        self.assertEqual([x["code"] for x in self.status()["symbols"]], ["000002"])   # 000001 not seen for 15 min


class LazyModelTest(Base):
    def test_threshold_loads_the_model_first(self):
        """[2026-11-19] fix: with min_prob 0 the threshold comes from the model meta, which is loaded lazily - reading it before
        any prediction gave None and nothing was ever bought."""
        class LazyModel:
            def __init__(self):
                self.meta = {}
                self.calls = 0

            def maybe_reload(self):
                self.meta = {"threshold": 0.3, "model_id": "lazy", "label": {"tp": 2.0, "sl": 1.7, "horizon": 60}}

            def predict_with_meta(self, feats):
                self.calls += 1
                return 0.6, self.meta

        model = LazyModel()
        r = self.runner(model=model, min_prob=0.0)
        r.on_symbol("010060", "x", self.at("09:30"), frame(30), 10300.0)
        self.assertEqual(model.calls, 1)
        self.assertEqual(self.records()[0]["threshold"], 0.3)


class PollTest(Base):
    class Feed:
        def __init__(self, data):
            self.data = data
            self.polled = []

        def codes(self):
            return sorted(self.data)

        def poll(self, code, now):
            self.polled.append(code)
            return self.data.get(code, (None, None))

    def test_excludes_watched_throttles_and_names(self):
        prev = self.out / "20261111"
        prev.mkdir()
        (prev / "000003_시선AI_daily.csv").write_text("date,close\n20261111,1\n", encoding="utf-8")
        r = self.runner(collector_max_positions=5, collector_max_entries=5)
        feed = self.Feed({"000001": (frame(30), 10300.0), "000003": (frame(30), 10300.0)})
        n = r.poll_collector(feed, self.at("09:30"), exclude={"000001"}, every_s=5)
        self.assertEqual((n, feed.polled), (1, ["000003"]))
        self.assertEqual(self.records()[0]["name"], "시선AI")
        self.assertEqual(r.poll_collector(feed, self.at("09:30", 8), exclude={"000001"}, every_s=5), 0)   # throttled

    def test_held_collector_symbol_keeps_its_exit_checks(self):
        r = self.runner(collector_max_positions=5, collector_max_entries=5)
        r.on_symbol("000003", "c", self.at("09:30"), frame(30), 10300.0, group="collector")
        feed = self.Feed({})                                  # its tick file is gone from the listing
        feed.data_hidden = {"000003": (None, 9000.0)}
        feed.poll = lambda code, now: feed.data_hidden.get(code, (None, None))
        r.poll_collector(feed, self.at("09:40"), exclude=set(), every_s=0)
        self.assertEqual(self.records()[-1]["reason"], "sl")


class MissedDaysTest(Base):
    """[2026-11-25] paper positions left open on an earlier day (r003 stopped before that day's close)"""

    def entry(self, day, code, nxt=False, price=10000.0, ts="09:30:05"):
        return {"event": "ENTRY", "ts": f"{day[:4]}-{day[4:6]}-{day[6:]}T{ts}", "code": code, "name": "x", "entry": price + 10,
                "price": price, "tp_px": 99999, "sl_px": 1, "horizon": 60, "nxt": nxt}

    def day_file(self, day, rows):
        (self.out / f"edge_only_{day}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def ticks(self, day, code, rows, nxt=False):
        d = self.out / "ticks" / day / ("nxt" if nxt else "")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{code}.csv").write_text(HEADER + "\n" + "\n".join(
            row(code, hms, p, 1, i + 1, s).replace(DAY, day) for i, (hms, p, s) in enumerate(rows)) + "\n", encoding="utf-8")

    def exits(self, day):
        rows = []
        for x in (self.out / f"edge_only_{day}.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(x))
            except ValueError:
                continue                                                           # the torn line of a test
        return [r for r in rows if r["event"] == "EXIT"]

    def test_regular_auction_nxt_last_print_and_fallbacks(self):
        d = "20261120"
        self.day_file(d, [self.entry(d, "000001"), self.entry(d, "000002"), self.entry(d, "000003", nxt=True),
                          self.entry(d, "000004"), self.entry(d, "000005"), {"event": "EXIT", "code": "000005"}])
        self.ticks(d, "000001", [("151900", 10100, "regular"), ("153001", 10200, "close_auction")])
        self.ticks(d, "000002", [("151900", 10100, "regular"), ("152500", 9000, "regular")])   # after 15:20: not used
        self.ticks(d, "000003", [("195900", 10300, "post"), ("195955", 9000, "post")], nxt=True)   # after 19:59:50: not used
        logs = []
        n = r022.close_missed_days(self.out, self.out / "ticks", "20261125", logs.append, now=datetime(2026, 11, 25, 8, 0))
        self.assertEqual(n, 4)
        got = {e["code"]: e for e in self.exits(d) if e.get("reason") == "close_missed"}
        self.assertEqual((got["000001"]["exit"], got["000001"]["fill"], got["000001"].get("estimated")), (10200, "auction", None))
        self.assertEqual(got["000001"]["ts"], "2026-11-20T15:20:00")
        self.assertEqual((got["000002"]["exit"], got["000002"]["fill"], got["000002"]["estimated"]), (10090, "last", True))
        self.assertEqual((got["000003"]["exit"], got["000003"]["ts"], got["000003"].get("estimated")), (10290, "2026-11-20T19:59:50", None))
        self.assertEqual((got["000004"]["fill"], got["000004"]["estimated"], got["000004"]["exit"]), ("entry_record", True, 9990))
        self.assertEqual(got["000001"]["recorded_at"], "2026-11-25T08:00:00")
        # idempotent; today's and older-than-7-days files are not touched
        self.assertEqual(r022.close_missed_days(self.out, self.out / "ticks", "20261125", logs.append), 0)

    def test_torn_last_line_out_of_order_and_stale_nxt_prints(self):
        # [Codex review] a torn last line does not swallow the EXIT; the latest print by trade time wins; an NXT print more than
        # 180 s before 19:59:50 is used but marked estimated
        d = "20261121"
        self.day_file(d, [self.entry(d, "000001"), self.entry(d, "000002", nxt=True)])
        with (self.out / f"edge_only_{d}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write('{"event": "CANDID')                                          # torn
        self.ticks(d, "000001", [("151900", 10100, "regular"), ("151000", 9000, "regular")])    # arrival order != trade time
        self.ticks(d, "000002", [("195000", 10300, "post")], nxt=True)
        self.assertEqual(r022.close_missed_days(self.out, self.out / "ticks", "20261125", lambda m: None), 2)
        got = {e["code"]: e for e in self.exits(d)}
        self.assertEqual(got["000001"]["exit"], 10090)                              # 15:19:00 print, not the later-read 15:10
        self.assertEqual((got["000002"]["exit"], got["000002"]["estimated"]), (10290, True))
        self.assertEqual(r022.close_missed_days(self.out, self.out / "ticks", "20261125", lambda m: None), 0)

    def test_a_line_torn_inside_a_korean_character_is_skipped(self):
        d = "20261122"
        self.day_file(d, [self.entry(d, "000001")])
        with (self.out / f"edge_only_{d}.jsonl").open("ab") as fh:
            fh.write('{"event": "CANDIDATE", "name": "한'.encode("utf-8")[:-1])     # cut inside the last character
        self.assertEqual(r022.close_missed_days(self.out, self.out / "ticks", "20261125", lambda m: None), 1)
        self.assertEqual(r022.open_codes_in(self.out / f"edge_only_{d}.jsonl"), set())

    def test_today_and_old_days_are_left(self):
        self.day_file("20261125", [self.entry("20261125", "000001")])
        self.day_file("20261101", [self.entry("20261101", "000002")])
        logs = []
        self.assertEqual(r022.close_missed_days(self.out, self.out / "ticks", "20261125", logs.append), 0)
        self.assertEqual(self.exits("20261125") + self.exits("20261101"), [])
        self.assertTrue(any("older than 7 days" in x for x in logs))


if __name__ == "__main__":
    unittest.main()

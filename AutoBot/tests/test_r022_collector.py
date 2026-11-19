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


if __name__ == "__main__":
    unittest.main()

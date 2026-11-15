# -*- coding: utf-8 -*-
"""r026 adaptive exit shadow runner: tick-file tail, virtual fills (post-order data only), journal + checkpoint restore,
(code, entry time) copies, increase-only rebase, INCOMPLETE finalize, bounded files, threaded service isolation."""
import json
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live_trading"))

from r009_tick_bars import TICK_FILE_COLUMNS  # noqa: E402
from r024_adaptive_exit import ExitConfig  # noqa: E402
import r026_exit_shadow as r026  # noqa: E402
from r026_exit_shadow import ShadowExitRunner, ShadowExitService, TickFileTail, load_rvol_profile  # noqa: E402

DATE = "20261116"
T0 = datetime(2026, 11, 16, 10, 0, 0)
GROSS = ExitConfig(pnl_basis="gross")
KEY = "000001@2026-11-16T10:00:00"


def at(sec):
    return T0 + timedelta(seconds=sec)


class TickWriter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.acml = 1000
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(TICK_FILE_COLUMNS) + "\n")

    def add(self, sec, price, qty=10, side=1, partial=False):
        self.acml += qty
        ts = at(sec)
        row = {"recv_ts": ts.isoformat(), "code": "000001", "bsop_date": DATE, "trade_time": ts.strftime("%H%M%S"),
               "price": price, "cntg_vol": qty, "acml_vol": self.acml, "ccld_dvsn": side, "seln_cntg_csnu": "",
               "shnu_cntg_csnu": "", "askp1": price + 10, "bidp1": price, "hour_cls_code": "0", "venue": "KRX",
               "session": "regular"}
        line = ",".join(str(row[c]) for c in TICK_FILE_COLUMNS)
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            f.write(line if partial else line + "\n")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ticks = self.tmp / "ticks"
        self.data = self.tmp / "data"
        self.out = self.tmp / "live_runtime"
        self.data.mkdir()
        self.logs = []
        self.writer = TickWriter(self.ticks / DATE / "000001.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def runner(self, cfg=GROSS):
        return ShadowExitRunner(self.ticks, self.data, self.out, DATE, self.logs.append, cfg)

    @staticmethod
    def pos(qty=100, entry=10000, entry_at=T0):
        return {"buy_price": entry, "quantity": qty, "entry_buy_time": entry_at}

    def jsonl(self, name):
        p = self.out / name
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []

    def trades(self):
        return self.jsonl(f"exit_shadow_trades_{DATE}.jsonl")

    def step(self, r, sec, px, observe=True, **kw):
        self.writer.add(sec, px, **kw)
        if observe:
            r.observe("000001", "TEST", at(sec), self.pos(), 100, float(px))
        r.tick(at(sec))


class TailTests(Base):
    def test_partial_line_waits_and_truncation_restarts(self):
        tail = TickFileTail(self.writer.path)
        self.writer.add(0, 10000)
        self.writer.add(1, 10010, partial=True)
        self.assertEqual([t.price for t in tail.poll()], [10000.0])
        with open(self.writer.path, "a", encoding="utf-8") as f:
            f.write("\n")
        self.assertEqual([t.price for t in tail.poll()], [10010.0])
        TickWriter(self.writer.path)                          # rewritten smaller (new header only)
        self.assertEqual(tail.poll(), [])
        self.assertGreaterEqual(tail.resets, 1)

    def test_missing_file(self):
        self.assertEqual(TickFileTail(self.tmp / "nope.csv").poll(), [])

    def test_first_read_is_bounded(self):
        for s in range(400):
            self.writer.add(s, 10000 + s)
        tail = TickFileTail(self.writer.path, max_initial_bytes=2000)
        got = tail.poll()
        self.assertGreater(len(got), 0)
        self.assertLess(len(got), 30)                          # only the last ~2 KB
        self.assertEqual(got[-1].price, 10399.0)               # and it ends at the newest row
        self.writer.add(400, 20000)
        self.assertEqual([t.price for t in tail.poll()], [20000.0])


class RunnerTests(Base):
    def test_ladder_then_trailing_virtual_exits(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.assertIn(KEY, r.positions)
        for s in range(1, 60):
            self.step(r, s, 10000 + min(s, 35) * 10)          # climbs to 10,350
        # all prints buyer-initiated -> trade strength 200 -> m_cttr 1.15: A at 10,280, B at 10,220
        for s in range(60, 66):
            self.step(r, s, 10240)
        for s in range(66, 74):
            self.step(r, s, 10200)
        t = self.trades()
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["status"], "CLOSED")
        kinds = [(f["kind"], f["qty"]) for f in t[0]["fills"]]
        self.assertEqual(kinds, [("LADDER", 40), ("LADDER", 30), ("TRAIL_A", 21), ("TRAIL_B", 9)])
        self.assertTrue(all(d.get("shadow") for d in self.jsonl(f"exit_engine_{DATE}.jsonl")))

    def test_market_fill_needs_data_after_the_order(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 9790, observe=False)                  # STOP decided at 9,790
        vp = r.positions[KEY]
        self.assertEqual(vp.order["kind"], "STOP")
        self.assertEqual(vp.fills, [])                         # not filled on the decision snapshot
        r.tick(at(30))                                         # no new print, no REST: still unfilled
        self.assertEqual(vp.fills, [])
        self.step(r, 31, 9780, observe=False)
        self.assertEqual(self.trades()[0]["fills"][0]["price"], 9770)   # next print 9,780 - 1 tick

    def test_limit_needs_trade_above_after_order(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 10200, observe=False)                 # +2 %: ladder limit at 10,200
        vp = r.positions[KEY]
        self.assertEqual(vp.order["type"], "limit")
        self.step(r, 2, 10200, observe=False)                 # touch only - not a fill
        self.assertIsNotNone(vp.order)
        self.step(r, 3, 10210, observe=False)
        self.assertIsNone(vp.order)
        self.assertEqual(vp.fills[0]["price"], 10200)

    def test_limit_times_out_to_market_on_fresh_data_only(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 10200, observe=False)
        vp = r.positions[KEY]
        for s in range(2, 9):
            self.step(r, s, 10190, observe=False)
        self.assertIsNotNone(vp.order)
        r.tick(at(20))                                         # 19 s old, but no print after the 9 s mark yet
        self.assertIsNotNone(vp.order)
        self.step(r, 21, 10190, observe=False)
        self.assertEqual((vp.fills[0]["note"], vp.fills[0]["price"]), ("limit->market", 10180))

    def test_halt_blocks_fills(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 9790, observe=False)
        r.observe("000001", "TEST", at(2), self.pos(), 100, 9790.0, {"temp_stop_yn": "Y"})
        self.writer.add(3, 9780)
        r.tick(at(3))
        self.assertEqual(r.positions[KEY].fills, [])

    def test_stale_copy_ends_incomplete_not_filled(self):
        # real position sold at 10:00 -> REST stops; no tick file data -> the 15:15 FORCE must not fill on old prices
        r = self.runner()
        (self.ticks / DATE / "000001.csv").unlink()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        r.tick(at(1))
        r.tick(datetime(2026, 11, 16, 15, 15, 1))
        self.assertIsNotNone(r.positions[KEY].order)            # FORCE ordered but unfilled
        r.tick(datetime(2026, 11, 16, 15, 30, 0))
        t = self.trades()[0]
        self.assertEqual((t["status"], t["reason"], t["sold_qty"]), ("INCOMPLETE", "SESSION_END", 0))
        self.assertEqual(r.positions, {})

    def test_bot_quantity_only(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(qty=150), 100, 10000.0)
        self.assertEqual(r.positions[KEY].state.q0, 100)

    def test_vi_status_from_quote(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0, {"vi_cls_code": "Y", "temp_stop_yn": "N"})
        self.assertTrue(r.positions[KEY].rest.vi_active)
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0, {"vi_cls_code": "N"})
        self.assertFalse(r.positions[KEY].rest.vi_active)

    def test_errors_never_escape(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, None, 100, 10000.0)
        r.observe("000001", "TEST", T0, {"buy_price": "x", "quantity": 1}, 1, 10000.0)
        r.positions["bad"] = object()
        r.tick(T0)
        self.assertGreaterEqual(r.errors, 1)

    def test_ai_is_off_in_shadow(self):
        self.assertEqual(self.runner(ExitConfig(ai_mode="live")).cfg.ai_mode, "off")


class CopyIdentityTests(Base):
    def test_closed_copy_not_reopened_and_rebuy_opens_new_copy(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 9790, observe=False)
        self.step(r, 2, 9780, observe=False)
        self.assertEqual(r.positions, {})
        r.observe("000001", "TEST", at(3), self.pos(), 100, 9780.0)     # real position still held
        self.assertEqual(r.positions, {})
        r.persist(at(3))
        r2 = self.runner()
        r2.observe("000001", "TEST", at(4), self.pos(), 100, 9780.0)
        self.assertEqual(r2.positions, {})
        rebuy_at = at(300)
        r2.observe("000001", "TEST", rebuy_at, self.pos(50, 9800, rebuy_at), 50, 9800.0)
        self.assertEqual([vp.state.q0 for vp in r2.positions.values()], [50])

    def test_old_copy_and_rebuy_copy_coexist(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        rebuy_at = at(300)
        r.observe("000001", "TEST", rebuy_at, self.pos(30, 10100, rebuy_at), 30, 10100.0)
        self.assertEqual(sorted(vp.state.q0 for vp in r.positions.values()), [30, 100])
        self.assertEqual(r.positions[KEY].state.entry_price, 10000)

    def test_add_after_partial_virtual_sell_adds_only_the_increase(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 10200, observe=False)                 # ladder 40 ordered
        self.step(r, 2, 10210, observe=False)                 # filled -> virtual 60 left
        vp = r.positions[KEY]
        self.assertEqual(vp.state.held_qty, 60)
        r.observe("000001", "TEST", at(3), {"buy_price": 10009.09, "quantity": 110, "entry_buy_time": T0}, 110, 10100.0)
        self.assertEqual(vp.state.held_qty, 70)               # 60 + 10, not 110
        self.assertEqual(vp.bought_qty, 110)
        self.assertAlmostEqual(vp.state.entry_price, (10000 * 60 + 10100 * 10) / 70, places=0)

    def test_add_during_virtual_order_waits(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 10200, observe=False)                 # ladder order in flight
        vp = r.positions[KEY]
        r.observe("000001", "TEST", at(2), {"buy_price": 10009.09, "quantity": 110, "entry_buy_time": T0}, 110, 10200.0)
        self.assertEqual((vp.pending_add, vp.state.held_qty), (10, 100))
        self.step(r, 3, 10210, observe=False)                 # fill -> pending add applied
        self.assertEqual((vp.pending_add, vp.state.held_qty, vp.bought_qty), (0, 70, 110))


class RestartTests(Base):
    def test_checkpoint_after_every_event(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 10200, observe=False)
        r2 = self.runner()                                     # no explicit persist(): events already checkpointed
        self.assertEqual(r2.positions[KEY].order, r.positions[KEY].order)

    def test_journal_prevents_reopening_after_lost_checkpoint(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        stale = (self.out / f"exit_shadow_{DATE}.json").read_text(encoding="utf-8")
        self.step(r, 1, 9790, observe=False)
        self.step(r, 2, 9780, observe=False)                  # CLOSED written to the trades journal
        (self.out / f"exit_shadow_{DATE}.json").write_text(stale, encoding="utf-8")   # checkpoint "lost"
        r2 = self.runner()
        self.assertEqual(r2.positions, {})
        self.assertIn(KEY, r2.closed_keys)

    def test_restored_limit_does_not_fill_on_pre_order_highs(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.writer.add(1, 10350)                              # high BEFORE the order
        self.step(r, 2, 10300, observe=False)                  # ladder limit placed after that high
        order = dict(r.positions[KEY].order)
        self.assertEqual(order["type"], "limit")
        r2 = self.runner()                                     # restart: day file re-read from the start
        self.writer.add(5, int(order["limit"]))                # only a touch after the order
        r2.tick(at(5))
        self.assertIsNotNone(r2.positions[KEY].order)

    def test_restored_position_keeps_entry_based_peak(self):
        for s in range(0, 10):
            self.writer.add(s, 11000)
        entry_at = at(50)
        k = f"000001@{entry_at.isoformat()}"
        r = self.runner()
        r.observe("000001", "TEST", entry_at, self.pos(100, 10000, entry_at), 100, 10000.0)
        self.writer.add(60, 10050)
        r2 = self.runner()
        r2.tick(at(61))
        self.assertEqual(r2.positions[k].state.peak_price, 10050)

    def test_pre_entry_highs_do_not_become_the_peak(self):
        for s in range(0, 60):
            self.writer.add(s, 11000)
        for s in range(100, 105):
            self.writer.add(s, 10010)
        r = self.runner()
        entry_at = at(100)
        r.observe("000001", "TEST", entry_at, self.pos(100, 10000, entry_at), 100, 10010.0)
        r.tick(at(105))
        st = r.positions[f"000001@{entry_at.isoformat()}"].state
        self.assertEqual(st.peak_price, 10010)
        self.assertFalse(st.be_armed)


class FilesTests(Base):
    def test_decision_log_cap_keeps_events(self):
        old = r026.MAX_DECISION_BYTES
        r026.MAX_DECISION_BYTES = 1
        try:
            r = self.runner()
            r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
            for s in range(1, 5):
                self.step(r, s, 10010, observe=False)
            events = [d.get("event") for d in self.jsonl(f"exit_engine_{DATE}.jsonl")]
            self.assertIn("OPEN", events)
            self.assertNotIn(None, events)                     # heartbeats dropped
            self.assertTrue(any("decision log reached" in x for x in self.logs))
        finally:
            r026.MAX_DECISION_BYTES = old

    def test_old_decision_logs_removed(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "exit_engine_20260901.jsonl").write_text("{}\n", encoding="utf-8")
        (self.out / f"exit_engine_{DATE}.jsonl").write_text("", encoding="utf-8")
        (self.out / "exit_shadow_trades_20260901.jsonl").write_text("{}\n", encoding="utf-8")
        self.runner()
        self.assertFalse((self.out / "exit_engine_20260901.jsonl").exists())
        self.assertTrue((self.out / "exit_shadow_trades_20260901.jsonl").exists())   # results are kept


class ServiceTests(Base):
    def _wait(self, cond, timeout=3.0):
        deadline = time.time() + timeout
        while not cond() and time.time() < deadline:
            time.sleep(0.02)

    def test_trading_thread_only_enqueues_and_close_checkpoints(self):
        r = self.runner()
        svc = ShadowExitService(r, interval_s=0.05, clock=lambda: at(1))
        try:
            svc.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
            self._wait(lambda: bool(r.positions))
            self.assertIn(KEY, r.positions)
        finally:
            self.assertTrue(svc.close(finalize_reason=None, timeout_s=3))
        self.assertTrue((self.out / f"exit_shadow_{DATE}.json").exists())
        self.assertEqual(self.trades(), [])                    # kept open for a restart

    def test_full_queue_drops_instead_of_blocking(self):
        r = self.runner()
        svc = ShadowExitService(r, interval_s=60, queue_max=1, clock=lambda: at(1))
        svc._stop.set()                                         # worker stops draining
        svc._thread.join(2)
        t0 = time.time()
        for _ in range(100):
            svc.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.assertLess(time.time() - t0, 0.5)
        self.assertGreaterEqual(svc.dropped, 99)

    def test_date_change_finalizes_open_copies(self):
        r = self.runner()
        svc = ShadowExitService(r, interval_s=0.05, clock=lambda: at(1))
        svc.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self._wait(lambda: bool(r.positions))
        self.assertTrue(svc.close(finalize_reason="DATE_CHANGE", timeout_s=3))
        t = self.trades()
        self.assertEqual((t[0]["status"], t[0]["reason"]), ("INCOMPLETE", "DATE_CHANGE"))


class RvolLoaderTests(Base):
    def _write_10s(self, day: str, vol_per_min: float):
        d = self.data / day
        d.mkdir()
        lines = ["datetime,open,high,low,close,volume,raw_bar"]
        for m in range(1, 31):
            t = datetime(2026, 11, 1, 9, 0) + timedelta(minutes=m)
            lines.append(f"{t.isoformat()},1,1,1,1,{vol_per_min / 6},1")
            for k in range(1, 6):
                lines.append(f"{(t + timedelta(seconds=10 * k)).isoformat()},1,1,1,1,{vol_per_min / 6},0")
        (d / "000001_TEST_10s.txt").write_text("\n".join(lines), encoding="utf-8")

    def test_profile_from_g001_files(self):
        for i in range(12):
            self._write_10s(f"202611{i + 1:02d}", 600.0)
        p = load_rvol_profile(self.data, "000001", DATE)
        self.assertEqual((p.source, p.days), ("slot", 12))
        self.assertAlmostEqual(p.per_minute[540], 600.0)       # 09:01 bar = minute 09:00, volume x 6 restored

    def test_too_little_history(self):
        for i in range(3):
            self._write_10s(f"202611{i + 1:02d}", 600.0)
        self.assertIsNone(load_rvol_profile(self.data, "000001", DATE))
        self.assertIsNone(load_rvol_profile(self.tmp / "missing", "000001", DATE))


class WiringReReviewTests(Base):
    """Codex wiring re-review 2026-11-15 (adaptive_exit_codex_wiring_rereview_20261115.md)."""

    def test_rr1_missing_entry_time_one_copy(self):
        r = self.runner()
        p = {"quantity": 100, "buy_price": 10000, "entry_buy_time": None, "buy_time": None}
        r.observe("000001", "TEST", T0, p, 100, 10000.0)
        r.observe("000001", "TEST", at(5), p, 100, 10000.0)
        self.assertEqual(len(r.positions), 1)
        r2 = self.runner()                                     # survives a restart
        r2.observe("000001", "TEST", at(10), p, 100, 10000.0)
        self.assertEqual(len(r2.positions), 1)

    def test_rr2_real_decrease_then_rebuy_is_added(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        vp = r.positions[KEY]
        r.observe("000001", "TEST", at(1), {"buy_price": 10000, "quantity": 60, "entry_buy_time": T0}, 60, 10000.0)
        self.assertEqual(vp.state.held_qty, 100)               # real sells never change the virtual copy
        r.observe("000001", "TEST", at(2), {"buy_price": 10014.29, "quantity": 70, "entry_buy_time": T0}, 70, 10100.0)
        self.assertEqual((vp.state.held_qty, vp.bought_qty), (110, 110))

    def test_rr3_add_during_full_exit_order_reopens_not_closed(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 9790, observe=False)                  # STOP 100 ordered (unfilled)
        r.observe("000001", "TEST", at(2), {"buy_price": 9981.82, "quantity": 110, "entry_buy_time": T0}, 110, 9790.0)
        self.step(r, 3, 9780, observe=False)                  # STOP fills 100 -> 10 bought meanwhile remain
        self.assertEqual(self.trades(), [])                    # not CLOSED
        vp = r.positions[KEY]
        self.assertEqual((vp.state.held_qty, vp.state.closed, vp.bought_qty), (10, False, 110))
        self.assertEqual(vp.state.closing_kind, "")            # fresh state: no inherited full-exit decision

    def test_rr4_service_stop_after_session_end_finalizes(self):
        import threading
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        late = datetime(2026, 11, 16, 15, 30, 1)
        gate = threading.Event()

        def factory():                                         # worker is held until the trading thread stops it
            gate.wait(3)
            return r
        svc = ShadowExitService(factory, interval_s=3600, clock=lambda: late, log=self.logs.append)
        svc._stop.set()                                        # market-end break happens before any regular tick
        gate.set()
        self.assertTrue(svc.close(finalize_reason=None, timeout_s=3))
        self.assertEqual(self.trades()[0]["reason"], "SESSION_END")

    def test_rr5_factory_runs_on_worker_and_zero_timeout_close_returns_at_once(self):
        import threading
        built_on = []

        def factory():
            built_on.append(threading.current_thread().name)
            return self.runner()
        svc = ShadowExitService(factory, interval_s=0.05, clock=lambda: at(1), log=self.logs.append)
        self.assertTrue(svc.ready.wait(3))
        self.assertEqual(built_on, ["exit-shadow"])
        t0 = time.time()
        svc.close(finalize_reason="DATE_CHANGE", timeout_s=0)
        self.assertLess(time.time() - t0, 0.1)
        svc._thread.join(3)

    def test_rr5_failed_factory_disables_quietly(self):
        def factory():
            raise OSError("disk")
        svc = ShadowExitService(factory, clock=lambda: at(1), log=self.logs.append)
        svc._thread.join(3)
        svc.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)   # still never raises
        self.assertTrue(any("init failed" in x for x in self.logs))

    def test_rr6_later_add_does_not_reprice_earlier_sales(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 10200, observe=False)
        self.step(r, 2, 10210, observe=False)                 # 40 sold at 10,200 (cost 10,000)
        r.observe("000001", "TEST", at(3), {"buy_price": 10090.91, "quantity": 110, "entry_buy_time": T0}, 110, 11000.0)
        r.finalize(at(4), "TEST")
        t = self.trades()[0]
        f, tax = 0.00015, 0.002
        expect = (10200 * 40 * (1 - f - tax)) / (10000 * 40 * (1 + f)) * 100 - 100
        self.assertAlmostEqual(t["net_pct_sold"], expect, places=3)

    def test_rr7_run_costs_used_in_results(self):
        r = self.runner(ExitConfig(pnl_basis="gross", fee_pct=0.1, tax_pct=0.3))
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 10200, observe=False)
        self.step(r, 2, 10210, observe=False)
        fill = r.positions[KEY].fills[0]
        expect = (10200 * (1 - 0.001 - 0.003) / (10000 * 1.001) - 1) * 100
        self.assertAlmostEqual(fill["net_pct"], expect, places=3)

    def test_stale_halt_flag_stops_blocking(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.step(r, 1, 9790, observe=False)
        r.observe("000001", "TEST", at(2), self.pos(), 100, 9790.0, {"temp_stop_yn": "Y"})
        self.writer.add(3, 9780)
        r.tick(at(3))
        self.assertEqual(r.positions[KEY].fills, [])           # fresh halt: blocked
        self.writer.add(200, 9770)
        r.tick(at(200))                                        # halt flag 198 s old, prints flowing again
        self.assertEqual(self.trades()[0]["fills"][0]["kind"], "STOP")


class FinalGateTests(Base):
    def test_expired_halt_does_not_freeze_new_decisions(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0, {"temp_stop_yn": "Y"})
        self.writer.add(1, 10000)
        r.tick(at(1))
        self.assertEqual(r.positions[KEY].order, None)
        self.writer.add(100, 9790)                             # REST never refreshed again; halt flag 100 s old
        r.tick(at(100))
        vp = r.positions[KEY]
        self.assertIsNotNone(vp.order)                         # engine NOT frozen: STOP decided without a pending order
        self.assertEqual(vp.order["kind"], "STOP")

    def test_fresh_halt_freezes_engine(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0)
        self.writer.add(1, 9790)
        r.observe("000001", "TEST", at(1), self.pos(), 100, 9790.0, {"temp_stop_yn": "Y"})
        r.tick(at(1))
        self.assertIsNone(r.positions[KEY].order)              # fresh halt: no decision

    def test_expired_vi_flag_cleared(self):
        r = self.runner()
        r.observe("000001", "TEST", T0, self.pos(), 100, 10000.0, {"vi_cls_code": "Y"})
        vp = r.positions[KEY]
        self.assertTrue(r._effective_rest(vp, at(10)).vi_active)
        self.assertFalse(r._effective_rest(vp, at(100)).vi_active)


if __name__ == "__main__":
    unittest.main()

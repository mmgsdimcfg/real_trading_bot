# -*- coding: utf-8 -*-
"""G032 adaptive exit replay - real bot entries replayed through the SAME r026 shadow runner (r024 engine, r025 features)
on the collected g011 tick files, compared with the real exit and with a stop_only baseline.

Design: adaptive_exit_design_20261115.md section 12 (validation plan) / 20. This is the "tick replay" stage 1:
- entries   = real round trips from the buy_sell logs (g026.real_buys), one position per (code, buy_time), only when the
              day's tick file exists (ticks are collected since 2026-10-30, so the sample is SMALL - no tuning on it)
- replay    = r026.ShadowExitRunner with its tick-file tail swapped for an in-memory feed released by receive time;
              1 evaluation / s
- fills     = r026 virtual fills (market = price - 1 tick, limit = only when traded above, 9 s -> market)
- baselines = real exit (log fills, net of costs), stop_only_live (the live policy: net -2 % stop, no scheduled exit,
              remainder marked at the last print) and stop2_eod1515 (same stop + an ASSUMED 15:15 exit)
- REST      = emulated only while the REAL position was held (until its last logged sell), stamped with the receive
              time of the newest visible print - after the real exit the copy runs on the tick file alone, like live

  python g032_adaptive_exit_replay.py [--dates 20261112 20261113] [--config '{"trail_arm_pct": 1.2}'] [--gross]
Output: <runtime>/data/sim/g032/<run>/trades.csv + summary.txt (+ the runner's jsonl files)

Update log:
- [2026-11-15] type=feat owner=claude
    summary: 신규 - 적응형 청산 엔진 틱 리플레이(실매매 진입 x r026 shadow 러너 재사용) + 실제/stop_only 비교.
    impact: sim only
    compatibility: backward-compatible
- [2026-11-15] type=fix owner=claude
    summary: Codex 연결 검토 #8~#10 - 실제 손익 비용 차감(net), 기준선 분리(stop_only_live = 실제 정책 / stop2_eod1515 = 가정),
      REST 흉내는 실제 보유 중에만 + 수신 시각 보존, r026 키 변경 대응.
    impact: sim only
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from r001_define_config import RUNTIME_DATA_DIR, TICK_DATA_DIR  # noqa: E402
from r009_tick_bars import iter_tick_file  # noqa: E402
from r024_adaptive_exit import ExitConfig, pnl_pct, tick_down  # noqa: E402
from r026_exit_shadow import ShadowExitRunner  # noqa: E402

END = dt_time(15, 20)
STOP_ONLY_NET = -2.0
FORCE = dt_time(15, 15)
NET = ExitConfig()


class ReplayTail:
    """Drop-in for r026 TickFileTail: releases the day's prints whose receive time <= the replay clock."""

    def __init__(self, ticks: list):
        self.ticks = sorted(ticks, key=lambda t: (t.recv or t.ts))
        self.i = 0
        self.now: datetime | None = None
        self.resets = 0

    def poll(self):
        out = []
        while self.i < len(self.ticks) and (self.ticks[self.i].recv or self.ticks[self.i].ts) <= self.now:
            out.append(self.ticks[self.i])
            self.i += 1
        return out


def stop_baselines(ticks: list, entry: float, start: datetime) -> dict:
    """Two baselines on the same tick path (wiring review #9):
    - stop_only_live : what AUTO_SELL_MODE="stop_only" really does - net -2 % stop, otherwise NO scheduled exit; the
                       remainder is marked at the day's last print (unrealised, flagged mtm=True)
    - stop2_eod1515  : the same stop plus an ASSUMED 15:15 exit (not the live policy)"""
    out = {}
    after = [t for t in ticks if t.ts >= start]
    if not after:
        return {"stop_only_live": (entry, "NO_DATA", True), "stop2_eod1515": (entry, "NO_DATA", True)}
    stop = next((t for t in after if pnl_pct(NET, entry, t.price) <= STOP_ONLY_NET), None)
    if stop is not None:
        out["stop_only_live"] = (tick_down(stop.price), "NET_STOP", False)
    else:
        out["stop_only_live"] = (after[-1].price, "MTM_LAST_PRINT", True)
    eod = next((t for t in after if t.ts.time() >= FORCE), None)
    if stop is not None and (eod is None or stop.ts <= eod.ts):
        out["stop2_eod1515"] = (tick_down(stop.price), "NET_STOP", False)
    elif eod is not None:
        out["stop2_eod1515"] = (tick_down(eod.price), "FORCE_1515", False)
    else:
        out["stop2_eod1515"] = (after[-1].price, "MTM_LAST_PRINT", True)
    return out


def replay_one(row, cfg: ExitConfig, out_dir: Path, log) -> dict | None:
    code, buy_time = row["code"], row["buy_time"].to_pydatetime()
    real_end = row["last_sell_time"].to_pydatetime()
    date_str = buy_time.strftime("%Y%m%d")
    path = TICK_DATA_DIR / date_str / f"{code}.csv"
    if not path.exists():
        return None
    ticks = [t for t in iter_tick_file(path) if t.ts.time() <= END]
    if not ticks:
        return None
    entry, qty = float(row["buy_price"]), int(row["buy_qty"])
    runner = ShadowExitRunner(TICK_DATA_DIR, RUNTIME_DATA_DIR, out_dir, date_str, log, cfg, cleanup=False)
    pos = {"buy_price": entry, "quantity": qty, "entry_buy_time": buy_time}
    runner.observe(code, row["name"], buy_time, pos, qty, entry)
    if len(runner.positions) != 1:
        return None
    vp = next(iter(runner.positions.values()))
    feed = ReplayTail(ticks)
    vp.tail = feed
    recv_keys = [t.recv or t.ts for t in feed.ticks]
    now = buy_time.replace(microsecond=0) + timedelta(seconds=1)
    end = datetime.combine(buy_time.date(), END)
    while now <= end and runner.positions:
        feed.now = now
        # r003 observes the symbol (REST price) only while the REAL position is held (review #10): the quote carries
        # the receive time of the newest print visible now - never a synthetic "fresh" timestamp
        if now <= real_end:
            k = bisect.bisect_right(recv_keys, now)
            if k:
                last = feed.ticks[k - 1]
                runner.observe(code, row["name"], now, pos, qty, last.price, price_ts=last.recv or last.ts)
        runner.tick(now)
        now += timedelta(seconds=1)
    if runner.positions:
        runner.finalize(now, "REPLAY_END")
    tpath = out_dir / f"exit_shadow_trades_{date_str}.jsonl"
    trades = [json.loads(x) for x in tpath.read_text(encoding="utf-8").splitlines()] if tpath.exists() else []
    base = stop_baselines(ticks, entry, buy_time)
    peak = max([t.price for t in ticks if t.ts >= buy_time] or [entry])
    rec = {"date": date_str, "code": code, "name": row["name"], "buy_time": buy_time.isoformat(), "entry": entry,
           "qty": qty, "real_net_pct": round(float(row["real_net_pct"]), 4), "real_reason": row["reason"],
           "peak_net_pct": round(pnl_pct(NET, entry, peak), 4)}
    for name, (px, why, mtm) in base.items():
        rec[f"{name}_net_pct"] = round(pnl_pct(NET, entry, px), 4)
        rec[f"{name}_reason"] = why
        rec[f"{name}_mtm"] = mtm
    t = trades[-1] if trades else None
    if t and t["status"] == "CLOSED":
        rec.update({"engine_net_pct": t["net_pct_sold"], "engine_status": "CLOSED",
                    "engine_fills": "+".join(f"{f['kind']}:{f['qty']}" for f in t["fills"]),
                    "engine_last_exit": t["fills"][-1]["ts"][11:19] if t["fills"] else ""})
    else:
        rec.update({"engine_net_pct": None, "engine_status": t["status"] + ":" + t["reason"] if t else "NONE",
                    "engine_fills": "+".join(f"{f['kind']}:{f['qty']}" for f in (t or {}).get("fills", [])),
                    "engine_last_exit": ""})
    return rec


def _real_round_trips(real: pd.DataFrame) -> pd.DataFrame:
    """One row per (code, buy_time). Net % after fee on both sides and the sell tax (review #8): the logged pnl_krw is
    (sell - buy) x qty without costs."""
    f, tax = NET.fee_pct / 100.0, NET.tax_pct / 100.0
    real = real.assign(
        proceeds_net=real["sell_price"] * real["sell_qty"] * (1 - f - tax),
        cost_net=real["buy_price"] * real["sell_qty"] * (1 + f))
    g = real.groupby(["code", "buy_time"], as_index=False).agg(
        name=("name", "first"), buy_price=("buy_price", "first"), buy_qty=("buy_qty", "max"),
        proceeds_net=("proceeds_net", "sum"), cost_net=("cost_net", "sum"), reason=("reason", "last"),
        last_sell_time=("sell_time", "max"))
    g["real_net_pct"] = (g["proceeds_net"] / g["cost_net"] - 1.0) * 100.0
    return g


def main() -> None:
    import g026_timing_study as g026
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--config", default="{}", help="ExitConfig overrides as JSON")
    ap.add_argument("--gross", action="store_true", help="pnl_basis=gross (default net)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    overrides = json.loads(args.config)
    if args.gross:
        overrides["pnl_basis"] = "gross"
    cfg = ExitConfig.from_mapping(overrides)
    cfg.validate()
    global NET  # every column (real, baselines, engine) uses the run's own fee / tax (wiring re-review #7)
    NET = ExitConfig(pnl_basis="net", fee_pct=cfg.fee_pct, tax_pct=cfg.tax_pct)
    real = g026.real_buys()
    if real.empty:
        print("no real trades")
        return
    tick_days = {p.name for p in TICK_DATA_DIR.iterdir() if p.is_dir()}
    real = real[real["date"].isin(tick_days)]
    if args.dates:
        real = real[real["date"].isin(args.dates)]
    g = _real_round_trips(real)
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else RUNTIME_DATA_DIR / "sim" / "g032" / run
    out_root.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    rows = []
    for k, (_, row) in enumerate(g.sort_values("buy_time").iterrows()):
        rec = replay_one(row, cfg, out_root / "positions" / f"p{k:03d}_{row['code']}", logs.append)
        if rec:
            rows.append(rec)
            print(f"{rec['date']} {rec['code']} real={rec['real_net_pct']:+.2f} stop_live="
                  f"{rec['stop_only_live_net_pct']:+.2f}{'(mtm)' if rec['stop_only_live_mtm'] else ''} "
                  f"stop2_1515={rec['stop2_eod1515_net_pct']:+.2f} engine={rec['engine_net_pct']} "
                  f"{rec['engine_status']} {rec['engine_fills']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "trades.csv", index=False, encoding="utf-8-sig")
    lines = [f"g032 adaptive exit replay {run} | positions={len(df)} | config={overrides}",
             "WARNING: ticks exist only since 2026-10-30 - this is a wiring / sanity replay, NOT evidence for tuning.",
             "All columns are NET of fee (both sides) and sell tax. stop_only_live = the live AUTO_SELL_MODE policy "
             "(no scheduled exit; open remainder marked at the last print = unrealised). stop2_eod1515 = the same stop "
             "+ an ASSUMED 15:15 exit."]
    if not df.empty:
        n_mtm = int(df["stop_only_live_mtm"].sum())
        lines.append(f"stop_only_live marked-to-market (not realised) positions: {n_mtm}")
        lines.append(f"engine status: {df['engine_status'].value_counts().to_dict()}")
        for col in ("real_net_pct", "stop_only_live_net_pct", "stop2_eod1515_net_pct", "engine_net_pct"):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            lines.append(f"{col:24s} n={len(s):3d} mean={s.mean():+.3f}% median={s.median():+.3f}% "
                         f"win={(s > 0).mean() * 100:5.1f}% sum={s.sum():+.2f}%")
        both = df.dropna(subset=["engine_net_pct"])
        for ref in ("real_net_pct", "stop_only_live_net_pct", "stop2_eod1515_net_pct"):
            d = both["engine_net_pct"] - both[ref]
            if len(d):
                lines.append(f"engine - {ref:22s} (paired) n={len(d)} mean={d.mean():+.3f}%p median={d.median():+.3f}%p")
    (out_root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"out: {out_root}")


if __name__ == "__main__":
    main()

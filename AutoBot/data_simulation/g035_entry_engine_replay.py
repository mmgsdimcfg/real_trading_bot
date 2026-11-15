# -*- coding: utf-8 -*-
"""G035 entry engine replay (stage 4 of the entry-engine review, 2026-11-15).

Runs the LIVE r028 EntryEngine code (not a re-implementation) on the g011 tick files, one second at a time, with the
same fill / exit model as g034, and compares it with the current trigger (g034 E_live rows of the same days).

- feed      prints released by receive time; prior-day tick file = warm-up (like the live shadow runner)
- decision  EntryEngine.evaluate every second; a symbol is blocked while the replayed position is open
- fill      "market": the first print received LAT_S after the decision, at its ask1 (g034 model)
            "limit" : r028 OrderExecutor - limit at the best ask seen at the decision, filled when a later print trades at
                      or below the limit (가정: full fill, no queue), reprice / cancel per the yaml order section
- exit      r024 adaptive exit engine replay (g034 engine_exit), NET of fee and tax
- variants  base yaml + one-parameter changes (sensitivity)

  python g035_entry_engine_replay.py [--workers 14] [--variants base,vwap_off,...] [--g034 <dir with trades.csv>]
Output: <runtime>/data/sim/g035/<run>/trades.csv + latency.csv + report.md

Update log
- [2026-11-15] type=feat owner=claude
    summary: 신규 - r028 진입 엔진 틱 리플레이(기존 트리거 대비, 민감도, 지정가 체결 모델, 워크포워드).
    impact: sim only
    compatibility: backward-compatible
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))
sys.path.insert(0, str(HERE))

from r001_define_config import RUNTIME_DATA_DIR, TICK_DATA_DIR  # noqa: E402
from r028_entry_engine import EntryEngine, EntryParams, LatencyLogger, OrderExecutor, Print  # noqa: E402
import g034_entry_bar_study as g034  # noqa: E402

LAT_S = g034.LAT_S
VARIANTS: dict[str, dict] = {
    "base": {},
    # every filter off + market fill = the g034 F3/SEQ rule: checks that r028 reproduces the stage-2 signals
    "parity_g034_F3SEQ": {"_fill": "market", "filter": {"vwap_gap_min_pct": -100.0, "vwap_gap_max_pct": 100.0,
                                                        "end": "14:50:01", "chase_max_pct": 100.0, "rsi_max": 101.0}},
    "market_fill": {"_fill": "market"},
    "vwap_off": {"filter": {"vwap_gap_min_pct": -100.0, "vwap_gap_max_pct": 100.0}},
    "end_1450": {"filter": {"end": "14:50:00"}},
    "chase_0.5": {"filter": {"chase_max_pct": 0.5}},
    "chase_0.15": {"filter": {"chase_max_pct": 0.15}},
    "w5": {"signal": {"window_bars": 5}},
    "w15": {"signal": {"window_bars": 15}},
    "combo_ma": {"signal": {"combo": "MA"}},
    "regime_1m": {"bars": {"regime_minutes": 1}},
    "valid_30": {"entry": {"valid_s": 30.0}},
    "valid_180": {"entry": {"valid_s": 180.0}},
    "intrabar": {"signal": {"intrabar": True}},
    "rsi_max_70": {"filter": {"rsi_max": 70.0}},
}


def _prints(t: pd.DataFrame) -> list[Print]:
    return [Print(ts=r.ts.to_pydatetime(), recv=r.recv.to_pydatetime(), price=float(r.price), qty=int(r.qty), side=int(r.side),
                  acml_vol=int(r.acml), ask1=float(r.ask), bid1=float(r.bid)) for r in t.itertuples(index=False)]


def _prior_prints(day: str, code: str) -> list[Print]:
    days = sorted(p.name for p in TICK_DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit() and p.name < day)
    for d in reversed(days[-3:]):
        f = TICK_DATA_DIR / d / f"{code}.csv"
        if f.exists():
            t = g034.load_ticks(f)
            if len(t):
                return _prints(t.tail(20000))
    return []


def _limit_fill(rt, price, ask, t_dec: float, ref: float, ex: OrderExecutor, code: str, day: str) -> tuple[int, float] | None:
    """Index / price of the limit fill, or None (cancelled). Uses r028 OrderExecutor for reprice / cancel."""
    i0 = int(np.searchsorted(rt, t_dec, side="right")) - 1
    if i0 < 0:
        return None
    a0 = ask[i0] if ask[i0] > 0 else price[i0]
    base = datetime.strptime(day, "%Y%m%d")
    plan = ex.submit(code, 1, base + timedelta(seconds=t_dec + LAT_S), float(a0), ref)
    if plan is None:
        return None
    j = int(np.searchsorted(rt, t_dec + LAT_S, side="right"))
    while j < len(rt):
        now = base + timedelta(seconds=float(rt[j]))
        act, _px = ex.step(code, now, float(ask[j]) if ask[j] > 0 else float("nan"))
        if act == "CANCEL":
            return None
        if price[j] <= plan.price:
            ex.on_fill(code, 1)
            return j, float(plan.price)
        j += 1
    ex.active.pop(code, None)
    return None


def run_code_day(task) -> tuple[list[dict], list[dict]]:
    day, code, variants = task
    t = g034.load_ticks(TICK_DATA_DIR / day / f"{code}.csv")
    if len(t) < 2000:
        return [], []
    prints = _prints(t)
    prior = _prior_prints(day, code)
    ticks = g034.to_tick_objects(t, code)
    recv_keys = [x.recv for x in ticks]
    rt, price, ask = t["rt"].to_numpy(), t["price"].to_numpy(), t["ask"].to_numpy()
    turnover = g034.prev_turnover(day, code)
    base_dt = datetime.strptime(day, "%Y%m%d")
    tmp = Path(tempfile.mkdtemp(prefix=f"g035_{day}_{code}_"))
    cache: dict = {}
    trades, lat_rows = [], []
    try:
        for name in variants:
            over = {k: v for k, v in VARIANTS[name].items() if not k.startswith("_")}
            fill_mode = VARIANTS[name].get("_fill", "limit")
            params = EntryParams.load(overrides=over)
            eng = EntryEngine(params)
            ex = OrderExecutor(params)
            lat = LatencyLogger(None)
            eng.warm(code, prior)
            busy_until = -1.0
            k = 0
            seen_cand = None
            start, end = int(rt[0]) + 1, g034.SESSION_END_S
            for sec in range(start, end + 1):
                now = base_dt + timedelta(seconds=sec)
                while k < len(prints) and rt[k] <= sec:
                    eng.on_print(code, prints[k])
                    k += 1
                st = eng.codes.get(code)
                c = st.cand if st else None
                if c is not None and seen_cand != (c.series, c.t1):
                    seen_cand = (c.series, c.t1)
                    lat.engine_start(code, c.t0, c.p0, c.t1, c.p1, series=c.series)
                for d in eng.evaluate(now, code, blocked=lambda _c: "POSITION" if sec < busy_until else None):
                    if d.action != "BUY":
                        lat.engine_end(code, "EXPIRED")
                        continue
                    lat.engine_decision(code, d.at, d.price, d.reason)
                    ref = d.p0 if d.p0 == d.p0 else d.p1
                    if fill_mode == "market":
                        i = int(np.searchsorted(rt, sec + LAT_S, side="right"))
                        if i >= len(rt) or rt[i] - sec > 120:
                            lat.order_submit(code, base_dt + timedelta(seconds=sec + LAT_S), d.price)
                            lat._flush(lat.order_open.pop(code), "NO_PRINT")
                            continue
                        fill = float(ask[i]) if ask[i] > 0 and ask[i] >= price[i] else float(price[i] + g034.tick_size(price[i]))
                    else:
                        r = _limit_fill(rt, price, ask, float(sec), ref, ex, code, day)
                        if r is None:
                            lat.order_submit(code, base_dt + timedelta(seconds=sec + LAT_S), d.price)
                            lat._flush(lat.order_open.pop(code), "CANCELLED")
                            continue
                        i, fill = r
                    lat.order_submit(code, base_dt + timedelta(seconds=sec + LAT_S), d.price)
                    entry_dt = base_dt + timedelta(seconds=float(rt[i]))
                    lat.order_fill(code, entry_dt, fill)
                    key = (round(float(rt[i]), 3), fill)
                    if key not in cache:
                        net, exit_dt, kinds, mtm = g034.engine_exit(day, code, ticks, recv_keys, entry_dt, fill, tmp)
                        j10 = int(np.searchsorted(rt, rt[i] + 600, side="right"))
                        j30 = int(np.searchsorted(rt, rt[i] + 1800, side="right"))
                        seg = price[i:max(j30, i + 1)]
                        cache[key] = (net, exit_dt, kinds, mtm, (price[max(i, j10 - 1)] / fill - 1) * 100,
                                      (price[max(i, j30 - 1)] / fill - 1) * 100, (seg.max() / fill - 1) * 100,
                                      (seg.min() / fill - 1) * 100)
                    net, exit_dt, kinds, mtm, r10, r30, mfe, mae = cache[key]
                    trades.append({"date": day, "code": code, "variant": name, "sig_t": sec, "sig_px": ref, "fill_t": float(rt[i]),
                                   "fill": fill, "slip_pct": (fill / ref - 1) * 100 if ref > 0 else np.nan, "net": net,
                                   "exit_t": exit_dt.isoformat(timespec="seconds"), "exit_kinds": kinds, "mtm": mtm,
                                   "ret10": r10, "ret30": r30, "mfe30": mfe, "mae30": mae, "turnover_prev": turnover,
                                   "series": d.series, "t0_s": (d.t0 - base_dt).total_seconds() if d.t0 else np.nan,
                                   "t1_s": (d.t1 - base_dt).total_seconds(), **{f"f_{k2}": v for k2, v in d.features.items()}})
                    busy_until = (exit_dt - base_dt).total_seconds()
            for row in lat.rows:
                lat_rows.append({"date": day, "variant": name, **row})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return trades, lat_rows


def report(d: pd.DataFrame, live: pd.DataFrame, lat: pd.DataFrame, days: list[str]) -> str:
    n_days = len(days)
    L = [f"# g035 r028 진입 엔진 리플레이 ({days[0]} ~ {days[-1]}, {n_days}거래일)", "",
         "r028 코드 그대로 1초 단위 리플레이. 청산 r024 엔진, 순손익 % (수수료 양방향 + 세금). base = r028_entry_engine.yaml "
         "(지정가 체결 모델: 매도1호가 지정가, 이후 그 가격 이하 체결이 나오면 전량 체결로 가정). "
         "E_live = 현행 트리거(g034, 시장가 체결 모델). 주문 직전 게이트(_011~_025)는 양쪽 모두 미적용.", "",
         "## 1. 기존 vs 개선 (같은 데이터)", ""] + g034.HEAD
    L.append(g034._row("E_live", "현행 트리거 (g034)", g034.stats(live, n_days)))
    for name in VARIANTS:
        g = d[d["variant"] == name]
        if len(g):
            L.append(g034._row(name, "r028 " + (str(VARIANTS[name]) if VARIANTS[name] else "yaml 기본값"), g034.stats(g, n_days)))
    split = days[: max(1, min(n_days - 1, int(round(n_days * 2 / 3))))]
    L += ["", f"## 2. 학습 / 검증 (학습 {split[0]}~{split[-1]}, 검증 이후)", "",
          "| 구성 | 학습 거래 | 학습 기대값 | 검증 거래 | 검증 기대값 |", "|---|---|---|---|---|"]
    for name, g in [("E_live", live)] + [(n, d[d["variant"] == n]) for n in VARIANTS]:
        if g.empty:
            continue
        a, b = g[g["date"].isin(split)], g[~g["date"].isin(split)]
        L.append(f"| {name} | {len(a)} | {g034._q(a['net'])} | {len(b)} | {g034._q(b['net'])} |")
    L += ["", "## 3. 날짜별 기대값 (base vs E_live)", "", "| 날짜 | base 거래 | base | E_live 거래 | E_live |", "|---|---|---|---|---|"]
    b = d[d["variant"] == "base"]
    for day in days:
        x, y = b[b["date"] == day], live[live["date"] == day]
        L.append(f"| {day} | {len(x)} | {g034._q(x['net'])} | {len(y)} | {g034._q(y['net'])} |")
    if not lat.empty:
        lb = lat[(lat["variant"] == "base") & (lat["kind"] == "order") & (lat["outcome"] == "FILLED")]
        L += ["", "## 4. 지연 기록 (base, 체결된 시도) - LatencyLogger 출력", "", "| 구간 | 초 (평균 / 중앙값) |", "|---|---|"]
        for col in [c for c in lat.columns if c.endswith("_s")]:
            L.append(f"| {col[:-2]} | {g034._q(lb[col])} |")
        if "first_to_fill_pct" in lb:
            L.append(f"| 첫 신호 가격 -> 체결가 (%) | {g034._q(lb['first_to_fill_pct'])} |")
        L.append("")
        out = lat[lat["variant"] == "base"]["outcome"].value_counts()
        L.append("- 시도 결과: " + ", ".join(f"{k} {v}" for k, v in out.items()))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--g034", default=str(RUNTIME_DATA_DIR / "sim" / "g034" / "w10_20261115"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v in VARIANTS]
    days = sorted(p.name for p in TICK_DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit()
                  and (not args.dates or p.name in args.dates))
    tasks = [(d, f.stem, variants) for d in days for f in sorted((TICK_DATA_DIR / d).glob("*.csv"))]
    if args.limit:
        tasks = tasks[: args.limit]
    out = Path(args.out) if args.out else RUNTIME_DATA_DIR / "sim" / "g035" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    trades, lats = [], []
    with Pool(args.workers) as pool:
        for k, (tr, la) in enumerate(pool.imap_unordered(run_code_day, tasks), 1):
            trades += tr
            lats += la
            if k % 10 == 0 or k == len(tasks):
                print(f"{k}/{len(tasks)} symbol-days, trades {len(trades)}", flush=True)
    d, lat = pd.DataFrame(trades), pd.DataFrame(lats)
    d.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    lat.to_csv(out / "latency.csv", index=False, encoding="utf-8-sig")
    g = pd.read_csv(Path(args.g034) / "trades.csv", dtype={"date": str, "code": str})
    used = sorted({str(x) for x in d["date"].unique()}) if not d.empty else days
    d["date"] = d["date"].astype(str)
    live = g[(g["config"] == "E_live") & g["date"].isin(used)]
    if args.limit:
        codes = {(str(a), str(b).zfill(6)) for a, b in zip(d["date"], d["code"])}
        live = live[[(a, str(b).zfill(6)) in codes for a, b in zip(live["date"], live["code"])]]
    (out / "report.md").write_text(report(d, live, lat, used), encoding="utf-8")
    print(f"out: {out}")


if __name__ == "__main__":
    main()

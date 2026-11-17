# -*- coding: utf-8 -*-
"""g039 - triple trigger: let 1-minute MA5 < MA20 pass when the MA5 slope turns up and MACD improves? (2026-11-17)

Question (user): the triple trigger rejects 1MIN_MA5_MA20_NOT_ABOVE. Should a tick cross pass while 1-minute MA5 <= MA20 if the
1-minute MA5 slope has turned up and MACD is improving? Also: which items of the user's "bottom rebound" table (MA5 rising,
MACD rising, RSI 30~50 rising, stochastic %K rising and > %D, -DI easing or +DI improving, BB lower touch then rebound) add
anything when merged into the current conditions?

Events / timing / exits are g037's (same tick files, same 60/120-tick MA5 x MA20 golden cross, decision = bar avail + 15 s,
one event per symbol per 5 min, entry first print + half tick, exits 10m/30m/60m/EOD/TP1.5-SL1-60m, costs 0.23 %).
1-minute features use the last 1-minute bar completed before the decision (bar end + g034 delay), with the live bot's indicator
definitions (r002 calculate_indicators: MACD 5/12/4, stochastic K10/D5, ADX/DI 7, RSI 14, BB 20/2) on prior-day + today's bars.

Run: python data_simulation/g039_ma5_turn_study.py [--out DIR]

Update log
- [2026-11-17] type=feat owner=claude
    summary: new - MA5-turn / MACD-improve pass rule for 1-minute MA5 <= MA20 tick crosses, plus the user's bottom-rebound items
    impact: sim only
    compatibility: new file
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live_trading"))

import g034_entry_bar_study as g034  # noqa: E402
import g036_signal_grid as g036  # noqa: E402
import g037_pullback_study as g037  # noqa: E402
from r002_strategy_core_shared import calculate_indicators  # noqa: E402

TICK_ROOT = g036.TICK_ROOT
EXITS = g037.EXITS


def _minute_frame(date: str, code: str, t: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, int]:
    """today's 1-minute bars (from ticks) with prior-day bars prepended for warm-up; returns (indicator frame, avail, offset)."""
    m1 = g034.minute_bars_from_ticks(t, 1)
    prior = None
    if hasattr(g034, "prior_minute_bars"):
        prior = g034.prior_minute_bars(date, code, 1)
    if prior is None or not len(prior):
        pc = g034.prior_minute_closes(date, code, 1)
        pc = pc[-120:] if pc is not None and len(pc) else np.empty(0)
        prior = pd.DataFrame({"open": pc, "high": pc, "low": pc, "close": pc, "volume": np.ones(len(pc))})
    else:
        prior = prior[["open", "high", "low", "close", "volume"]].tail(120)
    cols = ["open", "high", "low", "close", "volume"]
    frame = pd.concat([prior[cols].reset_index(drop=True), m1[cols].reset_index(drop=True)], ignore_index=True)
    ind = calculate_indicators(frame)
    return ind, m1["avail"].to_numpy(float), len(prior)


def symbol_day(date: str, path: Path) -> list[dict]:
    t = g034.load_ticks(path)
    if t.empty or len(t) < 500:
        return []
    code = path.stem
    rt, px = t["rt"].to_numpy(float), t["price"].to_numpy(float)
    ind, m_avail, off = _minute_frame(date, code, t)
    if len(m_avail) < 5:
        return []
    ma5 = ind["close"].rolling(5).mean().to_numpy()
    ma20 = ind["close"].rolling(20).mean().to_numpy()
    macd, hist = ind["MACD"].to_numpy(), ind["MACD_HIST"].to_numpy()
    rsi, k_, d_ = ind["RSI"].to_numpy(), ind["STOCH_K"].to_numpy(), ind["STOCH_D"].to_numpy()
    dip, dim = ind["DI_PLUS"].to_numpy(), ind["DI_MINUS"].to_numpy()
    bbl, bbm, lo, cl = ind["BB_LOWER"].to_numpy(), ind["BB_MIDDLE"].to_numpy(), ind["low"].to_numpy(), ind["close"].to_numpy()

    prior = g037._prior_ticks(date, path.name)
    events = []
    for n in g037.SIZES:
        b = g034.tick_bars(t, n)
        if len(b) < 3:
            continue
        pc = np.empty(0)
        if prior is not None and len(prior) >= n:
            pc = g034.tick_bars(prior, n)["close"].to_numpy(float)[-120:]
        closes = np.r_[pc, b["close"].to_numpy(float)]
        f, s = g037._ma(closes, g037.FAST)[len(pc):], g037._ma(closes, g037.SLOW)[len(pc):]
        up = f > s
        avail, bcl = b["avail"].to_numpy(float), b["close"].to_numpy(float)
        first = 1 if len(pc) else g037.SLOW
        for i in range(max(first, 1), len(b)):
            if up[i] and not up[i - 1] and not np.isnan(s[i - 1]):
                events.append({"n": n, "avail": float(avail[i]), "cross_close": float(bcl[i])})
    events.sort(key=lambda e: e["avail"])
    out, last_taken = [], -1e9
    for e in events:
        d = e["avail"] + g037.DECISION_DELAY_S
        if d < g037.ENTRY_FIRST_S or d > g037.ENTRY_LAST_S or d - last_taken < g037.MIN_GAP_S:
            continue
        kk = int(np.searchsorted(m_avail, d, side="right")) - 1        # last 1-minute bar completed before D (today index)
        if kk < 2:
            continue
        k = off + kk                                                   # index in the warm-up + today frame
        if np.isnan(ma20[k]) or np.isnan(ma5[k - 3]):
            continue
        last_taken = d
        ei = int(np.searchsorted(rt, d, side="left"))
        if ei >= len(rt):
            continue
        e_raw = float(px[ei])
        rec = {"date": date, "code": code, "n": e["n"], "t": d,
               "above": bool(ma5[k] > ma20[k]),
               "ma5_up": bool(ma5[k] > ma5[k - 1]),
               # "turned up": rising now, and it was falling (or flat) on at least one of the previous 3 bars
               "ma5_turn": bool(ma5[k] > ma5[k - 1] and np.nanmin(np.diff(ma5[k - 4:k])) <= 0),
               "macd_up": bool(macd[k] > macd[k - 1]),
               "hist_up": bool(hist[k] > hist[k - 1]),
               "rsi": float(rsi[k]), "rsi_up": bool(rsi[k] > rsi[k - 1]),
               "stoch_ok": bool(k_[k] > k_[k - 1] and k_[k] > d_[k]),
               "di_ok": bool(dim[k] < dim[k - 1] or dip[k] > dip[k - 1]),
               "bb_rebound": bool(np.nanmin(lo[k - 4:k + 1] - bbl[k - 4:k + 1]) <= 0 and cl[k] > cl[k - 1] and cl[k] < bbm[k]),
               "above_mid": bool(e_raw > bbm[k]),
               "gap20": float((cl[k] / ma20[k] - 1) * 100),
               "chase_ok": bool(e_raw <= e["cross_close"] * 1.005)}
        atr = ind["ATR"].to_numpy()[k] / cl[k] * 100 if cl[k] > 0 else np.nan
        rec["atr_ok"] = bool(not (atr == atr) or atr >= 0.5)
        epx = e_raw + 0.5 * g037._tick_size(e_raw)
        for name, h in (("10m", 600), ("30m", 1800), ("60m", 3600), ("EOD", 10 ** 9)):
            fx = g037._exit_fill(rt, px, ei, min(rt[ei] + h, g037.EOD_S))
            if fx is None or (h == 10 ** 9 and fx[0] < g037.EOD_S - 300):
                rec[f"net_{name}"] = rec[f"gro_{name}"] = np.nan
                continue
            raw = fx[1]
            xpx = raw - 0.5 * g037._tick_size(raw)
            rec[f"net_{name}"] = (xpx * (1 - g037.FEE - g037.TAX) - epx * (1 + g037.FEE)) / epx * 100.0
            rec[f"gro_{name}"] = (raw / e_raw - 1.0) * 100.0
        tp, sl = epx * 1.015, epx * 0.99
        end_t = min(rt[ei] + 3600, g037.EOD_S)
        stop = ei + int(np.searchsorted(rt[ei:], end_t, side="left"))
        raw = np.nan
        if stop < len(rt):
            seg = px[ei + 1:stop + 1]
            hit = np.flatnonzero((seg >= tp) | (seg <= sl))
            raw = float(seg[hit[0]]) if len(hit) else float(px[stop])
        if raw == raw:
            xpx = raw - 0.5 * g037._tick_size(raw)
            rec["net_TP1.5/SL1/60m"] = (xpx * (1 - g037.FEE - g037.TAX) - epx * (1 + g037.FEE)) / epx * 100.0
            rec["gro_TP1.5/SL1/60m"] = (raw / e_raw - 1.0) * 100.0
        else:
            rec["net_TP1.5/SL1/60m"] = rec["gro_TP1.5/SL1/60m"] = np.nan
        out.append(rec)
    return out


def _task(a):
    return symbol_day(*a)


def row(name: str, s: pd.DataFrame, train_days) -> str:
    cells = " | ".join(f"{s[f'net_{x}'].mean():+.3f} / {s[f'gro_{x}'].mean():+.3f} / {g037._cluster_t(s, f'net_{x}'):.2f}" for x in EXITS)
    tr = s[s["date"].isin(train_days)]["net_30m"].mean()
    te = s[~s["date"].isin(train_days)]["net_30m"].mean()
    win = (s["net_TP1.5/SL1/60m"] > 0).mean() * 100 if len(s) else np.nan
    return f"| {name} | {len(s)} | {win:.0f}% | {cells} | {tr:+.3f} / {te:+.3f} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(g036.g026.g013.DATA_ROOT / "sim" / "g039" / "run"))
    ap.add_argument("--exclude", nargs="*", default=["20261117"], help="days to skip (incomplete live day)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(p.name for p in TICK_ROOT.iterdir() if p.is_dir() and p.name.isdigit() and p.name not in args.exclude)
    tasks = [(d, f) for d in days for f in sorted((TICK_ROOT / d).glob("*.csv"))]
    with Pool() as pool:
        recs = [r for part in pool.map(_task, tasks, chunksize=4) for r in part]
    df = pd.DataFrame(recs)
    df.to_csv(out / "events.csv", index=False, encoding="utf-8-sig")
    train_days = days[: max(1, len(days) * 2 // 3)]
    live = df["atr_ok"] & df["chase_ok"]                 # live side filters other than the 1-minute BB-mid guard
    head = ["| 규칙 | 거래 | TP1.5/SL1 승률 | " + " | ".join(f"{x} 순 / 비용전 / t" for x in EXITS) + " | 30m 순 학습 / 검증 |",
            "|---|---|---|" + "---|" * len(EXITS) + "---|"]
    below = df[~df["above"] & live]
    above = df[df["above"] & live]
    L = [f"# g039 1분 MA5<MA20 통과 규칙 연구 ({days[0]}~{days[-1]}, {len(days)}일)", "",
         f"틱 크로스 사건 {len(df)}건, ATR·추격 필터 통과 {int(live.sum())}건 (1분 위 {len(above)} / 아래 {len(below)}). "
         f"학습 {train_days[0]}~{train_days[-1]}, 검증 나머지. 칸 = 순손익 % / 비용 전 % / 날짜 클러스터 t", ""]
    L += ["## 1. 현재 통과(1분 MA5>MA20) vs 제안 통과(1분 MA5<=MA20 + MA5 상승 전환 + MACD 개선)", ""] + head
    prop = below["ma5_turn"] & below["macd_up"]
    for name, s in (("현재 통과: 1분 위 (BB중심선 가드 전)", above), ("현재 통과: 1분 위 + BB중심선 위", above[above["above_mid"]]),
                    ("1분 아래 전체 (현재 막힘)", below),
                    ("제안: 아래 + MA5 상승 전환 + MACD 상승", below[prop]),
                    ("제안 + BB중심선 위 (현재 가드 유지 시 실제 통과분)", below[prop & below["above_mid"]]),
                    ("제안 중 BB중심선 아래 (가드에 막힘)", below[prop & ~below["above_mid"]]),
                    ("아래 + MA5 상승(전환 무관) + MACD 상승", below[below["ma5_up"] & below["macd_up"]]),
                    ("아래 + MA5 상승 전환 + 히스토그램 상승", below[below["ma5_turn"] & below["hist_up"]]),
                    ("아래 + 나머지 (제안에서도 막힘)", below[~prop])):
        L.append(row(name, s, train_days))
    L += ["", "## 2. 첨부 '저점 반등' 항목을 각 경로에 더했을 때 (해당 항목 True 인 사건만)", ""]
    items = {"MA5 상승": "ma5_up", "MACD 상승": "macd_up", "RSI 30~50 상승": None, "스토캐스틱 %K 상승 & %K>%D": "stoch_ok",
             "-DI 하락 또는 +DI 상승": "di_ok", "BB 하단 접근 후 반등(중심선 아래)": "bb_rebound"}
    for label, base in (("현재 통과 경로 (1분 위)", above), ("제안 경로 (아래 + 전환 + MACD)", below[prop])):
        L += [f"### {label}", ""] + head + [row("기준 (항목 없음)", base, train_days)]
        for name, col in items.items():
            m = (base["rsi"].between(30, 50) & base["rsi_up"]) if col is None else base[col]
            L.append(row(f"+ {name}", base[m], train_days))
            L.append(row(f"  (반대: {name} 아님)", base[~m], train_days))
        L.append("")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

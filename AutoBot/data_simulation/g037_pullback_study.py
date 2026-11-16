# -*- coding: utf-8 -*-
"""g037 - 1분 MA5>MA20 추세 중 "눌림 후 틱 재크로스" 진입 연구 (2026-11-16).

질문(사용자): 1분봉 MA5 > MA20 상태에서 잠깐 조정 후 재상승하는 구간이 단타 진입 기회인가? 현재 3중 트리거는 1분 상태가
크로스 후 TRIPLE_1MIN_ABOVE_MAX_AGE_SECONDS(900초) 를 넘으면 막는데(ABOVE_TOO_OLD), 그 이후의 진입은 실제로 나쁜가?

방법 (사건 연구, 청산 엔진과 분리)
- 사건 = 60틱 / 120틱봉 MA5 > MA20 골든크로스(봉 완성 시각 avail). 틱봉 MA 는 전일 틱으로 워밍업.
- 판정 시각 D = avail + DECISION_DELAY_S (폴링 간격 절반 근사). 1분 상태는 D 이전에 완성된 1분봉(거래소 시각 기준, 전일 종가로 MA 워밍업).
- 1분 MA5 > MA20 인 사건만 진입 후보. 1분 "위" 구간의 시작 봉(크로스 봉) 종료 시각부터 D 까지 = 추세 나이.
  나이 구간: 0~15분(현재 규칙 통과) / 15~30 / 30~60 / 60분+ / 전일부터(오늘 크로스 없음 = live 의 NO_CROSS_IN_FRAME, 현재 통과).
- 눌림 깊이: 직전 틱 "위" 구간의 고점(1분 추세 시작 이후) 대비 이번 틱 "아래" 구간 저점 하락률, 1분 MA20 터치 여부.
- live 와 같은 보조 필터(표시용 플래그): 현재가 > 1분 BB 중심선(MA20), 1분 ATR% >= 0.5 (GC_VOLATILITY), 크로스 대비 추격 <= 0.5%.
- 체결: D 이후 첫 틱 + 반 틱 매수, 청산 시각 이후 첫 틱 - 반 틱 매도, 수수료 0.015% x 2 + 세금 0.20%.
  청산 10분 / 30분 / 60분 / 15:15, 익절 1.5% 손절 1% 최대 60분(틱 경로, 동시 터치 없음 - 틱 단위).
- 같은 종목 사건 간격 MIN_GAP_S 이상(60/120틱 크로스가 몰리는 것 중복 제거), 진입 09:05~14:50.
- 통계: 평균(비용 전 / 후), 승률, 날짜 클러스터 t, 앞 7일(학습) / 뒤 4일(검증).

한계: 틱 파일 11거래일 · 감시 종목만, 적응형 청산(r024/r027)·나머지 게이트는 재현하지 않음. 탐색용.

실행: python data_simulation/g037_pullback_study.py [--out DIR]

Update log
- [2026-11-16] type=feat owner=claude
    summary: 신규 - 1분 추세 나이 / 눌림 깊이별 틱 재크로스 진입 성과 (ABOVE_TOO_OLD 900초 제한 검증)
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

TICK_ROOT = g036.TICK_ROOT
FEE, TAX = 0.00015, 0.0020
SIZES = (60, 120)
FAST, SLOW = 5, 20
DECISION_DELAY_S = 15.0
MIN_GAP_S = 300.0
ENTRY_FIRST_S, ENTRY_LAST_S, EOD_S = (9 * 60 + 5) * 60, (14 * 60 + 50) * 60, (15 * 60 + 15) * 60
EXITS = ("10m", "30m", "60m", "EOD", "TP1.5/SL1/60m")
AGE_BINS = ((0, 15), (15, 30), (30, 60), (60, 10 ** 6))
AGE_LABEL = ("0~15분", "15~30분", "30~60분", "60분+", "전일부터")
DEPTH_LABEL = ("<0.3%", "0.3~0.8%", ">=0.8%")
ATR_N = 14


def _tick_size(p: float) -> float:
    return float(g036._krx_tick(np.array([p]))[0])


def _ma(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x).rolling(w, min_periods=w).mean().to_numpy()
    return s


def _prior_ticks(date: str, name: str) -> pd.DataFrame | None:
    for d in sorted((p.name for p in TICK_ROOT.iterdir() if p.is_dir() and p.name.isdigit() and p.name < date), reverse=True)[:3]:
        f = TICK_ROOT / d / name
        if f.exists():
            t = g034.load_ticks(f)
            if not t.empty:
                return t
    return None


def _exit_fill(rt: np.ndarray, px: np.ndarray, start: int, t_exit: float) -> tuple[float, float] | None:
    """first print at / after t_exit (receive clock) from index start; None when the data ends first."""
    j = start + int(np.searchsorted(rt[start:], t_exit, side="left"))
    if j >= len(rt):
        return None
    return float(rt[j]), float(px[j])


def symbol_day(date: str, path: Path) -> list[dict]:
    t = g034.load_ticks(path)
    if t.empty or len(t) < 500:
        return []
    code = path.stem
    rt, px = t["rt"].to_numpy(float), t["price"].to_numpy(float)
    # ---- 1-minute bars (exchange clock) + prior-day closes for the MAs
    m1 = g034.minute_bars_from_ticks(t, 1)
    if len(m1) < 5:
        return []
    prior_c = g034.prior_minute_closes(date, code, 1)
    prior_c = prior_c[-120:] if prior_c is not None and len(prior_c) else np.empty(0)
    c_all = np.r_[prior_c, m1["close"].to_numpy(float)]
    off = len(prior_c)
    ma5_1, ma20_1 = _ma(c_all, FAST)[off:], _ma(c_all, SLOW)[off:]
    above_all = (_ma(c_all, FAST) > _ma(c_all, SLOW))
    # start of the "above" run that each bar belongs to (index in the combined series; -1 = from the first bar)
    run_start = np.full(len(c_all), -1)
    for i in range(len(c_all)):
        if above_all[i]:
            run_start[i] = run_start[i - 1] if i > 0 and above_all[i - 1] else i
    m_avail = m1["avail"].to_numpy(float)           # end + delay (receive clock ~ exchange clock)
    m_end = m_avail - g034.TIME_BAR_DELAY_S
    hi, lo, cl = m1["high"].to_numpy(float), m1["low"].to_numpy(float), m1["close"].to_numpy(float)
    prev_c = np.r_[np.nan, cl[:-1]]
    tr = np.nanmax(np.vstack([hi - lo, np.abs(hi - prev_c), np.abs(lo - prev_c)]), axis=0)
    atr_pct = pd.Series(tr).rolling(ATR_N, min_periods=ATR_N).mean().to_numpy() / cl * 100.0

    prior = _prior_ticks(date, path.name)
    events: list[dict] = []
    for n in SIZES:
        b = g034.tick_bars(t, n)
        if len(b) < 3:
            continue
        pc = np.empty(0)
        if prior is not None and len(prior) >= n:
            pc = g034.tick_bars(prior, n)["close"].to_numpy(float)[-120:]
        closes = np.r_[pc, b["close"].to_numpy(float)]
        f, s = _ma(closes, FAST)[len(pc):], _ma(closes, SLOW)[len(pc):]
        up = f > s
        avail, bhi, blo, bcl = b["avail"].to_numpy(float), b["high"].to_numpy(float), b["low"].to_numpy(float), b["close"].to_numpy(float)
        first = 1 if len(pc) else SLOW
        for i in range(max(first, 1), len(b)):
            if not (up[i] and not up[i - 1]) or np.isnan(s[i - 1]):
                continue
            # the "below" phase just before this cross and the "above" phase before it
            j = i - 1
            while j > 0 and not up[j - 1]:
                j -= 1
            trough = float(blo[j:i + 1].min())
            events.append({"n": n, "avail": float(avail[i]), "cross_close": float(bcl[i]), "dip_start": float(avail[j - 1]) if j > 0 else float(avail[0]),
                           "trough": trough, "hi_before": (bhi[:j], avail[:j])})
    if not events:
        return []
    events.sort(key=lambda e: e["avail"])
    out = []
    last_taken = -1e9
    for e in events:
        d = e["avail"] + DECISION_DELAY_S
        if d < ENTRY_FIRST_S or d > ENTRY_LAST_S or d - last_taken < MIN_GAP_S:
            continue
        k = int(np.searchsorted(m_avail, d, side="right")) - 1          # last 1-minute bar completed before D
        if k < 0 or np.isnan(ma20_1[k]):
            continue
        ci = off + k
        above = bool(above_all[ci])
        last_taken = d
        rec = {"date": date, "code": code, "n": e["n"], "t": d, "above": above}
        if above:
            rs = run_start[ci]
            if rs < off:                                                 # the run started before today's first bar
                rec["age_bin"] = 4
                age_s = np.nan
                run_t0 = m_end[0] - 60
            else:
                run_t0 = m_end[rs - off]
                age_s = d - run_t0
                rec["age_bin"] = next(ix for ix, (a, b_) in enumerate(AGE_BINS) if a * 60 <= age_s < b_ * 60)
            rec["age_min"] = age_s / 60.0
            hb, ha = e["hi_before"]
            sel = hb[ha >= run_t0]
            peak = float(sel.max()) if len(sel) else np.nan
            depth = (peak - e["trough"]) / peak * 100.0 if peak == peak and peak > 0 else np.nan
            rec["depth"] = depth
            rec["depth_bin"] = -1 if depth != depth else (0 if depth < 0.3 else 1 if depth < 0.8 else 2)
            rec["touch_ma20"] = bool(e["trough"] <= ma20_1[k] * 1.001)
        # entry
        ei = int(np.searchsorted(rt, d, side="left"))
        if ei >= len(rt):
            continue
        e_raw = float(px[ei])
        rec["above_mid"] = bool(e_raw > ma20_1[k])
        rec["atr_ok"] = bool(not (atr_pct[k] == atr_pct[k]) or atr_pct[k] >= 0.5)
        rec["chase_ok"] = bool(e_raw <= e["cross_close"] * 1.005)
        epx = e_raw + 0.5 * _tick_size(e_raw)
        for name, h in (("10m", 600), ("30m", 1800), ("60m", 3600), ("EOD", 10 ** 9)):
            fx = _exit_fill(rt, px, ei, min(rt[ei] + h, EOD_S))
            if fx is None or (h == 10 ** 9 and fx[0] < EOD_S - 300):
                rec[f"net_{name}"] = rec[f"gro_{name}"] = np.nan
                continue
            raw = fx[1]
            xpx = raw - 0.5 * _tick_size(raw)
            rec[f"net_{name}"] = (xpx * (1 - FEE - TAX) - epx * (1 + FEE)) / epx * 100.0
            rec[f"gro_{name}"] = (raw / e_raw - 1.0) * 100.0
        # take profit 1.5 % / stop 1 % on the print path, at most 60 minutes
        tp, sl = epx * 1.015, epx * 0.99
        end_t = min(rt[ei] + 3600, EOD_S)
        stop = ei + int(np.searchsorted(rt[ei:], end_t, side="left"))
        raw = np.nan
        if stop < len(rt):
            seg = px[ei + 1:stop + 1]
            hit = np.flatnonzero((seg >= tp) | (seg <= sl))
            raw = float(seg[hit[0]]) if len(hit) else float(px[stop])
        if raw == raw:
            xpx = raw - 0.5 * _tick_size(raw)
            rec["net_TP1.5/SL1/60m"] = (xpx * (1 - FEE - TAX) - epx * (1 + FEE)) / epx * 100.0
            rec["gro_TP1.5/SL1/60m"] = (raw / e_raw - 1.0) * 100.0
        else:
            rec["net_TP1.5/SL1/60m"] = rec["gro_TP1.5/SL1/60m"] = np.nan
        out.append(rec)
    return out


def _task(a):
    return symbol_day(*a)


def _cluster_t(df: pd.DataFrame, col: str) -> float:
    x = df[["date", col]].dropna()
    if len(x) < 3 or x["date"].nunique() < 3:
        return np.nan
    mu = x[col].mean()
    g = x.groupby("date")[col].agg(lambda s: (s - mu).sum())
    se = np.sqrt((g ** 2).sum()) / len(x)
    return mu / se if se > 0 else np.nan


def table(df: pd.DataFrame, by: str, labels, exits, train_days) -> str:
    rows = ["| 구간 | 거래 | " + " | ".join(f"{x} 순 / 비용전 / t" for x in exits) + " | 학습 / 검증 (" + exits[1] + " 순) |",
            "|---|---|" + "---|" * len(exits) + "---|"]
    for key in sorted(df[by].dropna().unique()):
        s = df[df[by] == key]
        lab = labels[int(key)] if isinstance(labels, (list, tuple)) else labels(key)
        cells = []
        for x in exits:
            cells.append(f"{s[f'net_{x}'].mean():+.3f} / {s[f'gro_{x}'].mean():+.3f} / {_cluster_t(s, f'net_{x}'):.2f}")
        tr = s[s["date"].isin(train_days)][f"net_{exits[1]}"].mean()
        te = s[~s["date"].isin(train_days)][f"net_{exits[1]}"].mean()
        rows.append(f"| {lab} | {len(s)} | " + " | ".join(cells) + f" | {tr:+.3f} / {te:+.3f} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(g036.g026.g013.DATA_ROOT / "sim" / "g037" / "run"))
    ap.add_argument("--exclude", nargs="*", default=["20261116"], help="days to skip (incomplete live day)")
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
    ab = df[df["above"]].copy()
    live = ab[ab["above_mid"] & ab["atr_ok"] & ab["chase_ok"]].copy()
    exits = list(EXITS)
    lines = [f"# g037 눌림 후 틱 재크로스 연구 ({days[0]}~{days[-1]}, {len(days)}일)", "",
             f"사건 {len(df)}건 (틱 크로스, 종목 간격 {MIN_GAP_S:.0f}s) · 1분 MA5>MA20 {len(ab)}건 · live 보조 필터 통과 {len(live)}건",
             f"학습 {train_days[0]}~{train_days[-1]} / 검증 나머지. 칸 = 순손익 평균 % / 비용 전 평균 % / 날짜 클러스터 t", ""]
    lines += ["## 1. 1분 상태별 (1분 MA5<MA20 = 현재 규칙에서 막힘)", ""]
    df["above_i"] = df["above"].astype(int)
    lines += [table(df, "above_i", ("1분 MA5<MA20", "1분 MA5>MA20"), exits, train_days), ""]
    for name, d in (("전체 (1분 위)", ab), ("live 보조 필터 통과", live)):
        lines += [f"## 2. 1분 추세 나이별 - {name}", "", "0~15분과 전일부터는 현재 규칙 통과, 15분 이상은 ABOVE_TOO_OLD 로 막힘", ""]
        lines += [table(d, "age_bin", AGE_LABEL, exits, train_days), ""]
        old = d[d["age_bin"].isin([1, 2, 3])]
        lines += [f"### 15분 이상 구간의 눌림 깊이별 - {name}", ""]
        lines += [table(old[old["depth_bin"] >= 0], "depth_bin", DEPTH_LABEL, exits, train_days), ""]
        old = old.copy()
        old["touch_i"] = old["touch_ma20"].astype(int)
        lines += [f"### 15분 이상 구간, 눌림 저점이 1분 MA20 에 닿았는가 - {name}", ""]
        lines += [table(old, "touch_i", ("MA20 위에서 반등", "MA20 터치"), exits, train_days), ""]
    # rule variants (live filter)
    lines += ["## 3. 규칙 변형 (live 보조 필터 통과 사건 기준)", "", "| 규칙 | 거래 | " + " | ".join(f"{x} 순 / t" for x in exits) + " |",
              "|---|---|" + "---|" * len(exits)]
    var = {
        "현재 (900초)": live["age_bin"].isin([0, 4]),
        "1800초": live["age_bin"].isin([0, 1, 4]),
        "3600초": live["age_bin"].isin([0, 1, 2, 4]),
        "나이 제한 없음": live["age_bin"].notna(),
        "현재 + 15분 이후 MA20 터치 눌림 허용": live["age_bin"].isin([0, 4]) | (live["age_bin"].isin([1, 2, 3]) & live["touch_ma20"]),
        "15분 이후만 (늘어나는 진입)": live["age_bin"].isin([1, 2, 3]),
    }
    for k, m in var.items():
        s = live[m]
        lines.append(f"| {k} | {len(s)} | " + " | ".join(f"{s[f'net_{x}'].mean():+.3f} / {_cluster_t(s, f'net_{x}'):.2f}" for x in exits) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

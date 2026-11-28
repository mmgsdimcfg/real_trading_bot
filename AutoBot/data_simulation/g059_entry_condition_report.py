# -*- coding: utf-8 -*-
"""g059 - report on the g058 entry-condition table (read-only).

Entries of a condition set = the first passing bar of a symbol-day, then not again for GAP_MIN minutes (the same for every set, so
only the conditions differ). For each set: entries, days, mean net % when held to 15:20 (H), under take-profit 3 / stop 1 and 2 / 2,
the best cell of the 15 x 15 grid, how many of its 225 cells are positive, the grid mean, and the hold mean by period
(A = before 2026-09, B = to 2026-10-29, C = the tick days from 2026-10-30).
Sections: each strategy as it runs, leave-one-out of the pooled set and of each strategy, combinations, one strategy + one
condition of another, each condition alone, and the grids of the main sets.
Net % = after commission 0.015 % x 2 and sell tax 0.20 %; sold 1 tick under the stop / the close, at the target for a take-profit
(the bar's open when it gaps over); a bar that reaches both = the stop. The best cell of a grid is in-sample - read the positive-cell
count and the periods, not the best cell.

usage: g059_entry_condition_report.py <g058 out dir> [bars]      (bars = a table built with --mode bars)

Update log:
- [2026-11-28] type=fix owner=claude
    summary: Codex 3차 검토 - 시가가 익절선 이상으로 열린 봉은 익절 우선, 매도 1틱은 매도가의 호가 단위, 진입 봉 없는 행 제외,
      Edge 확률 통과율은 점수 있는 봉 기준, E_scored 는 제외 비교에서 뺌, 반등 CSV 대조 문구 정정.
    impact: sim (읽기 전용)
    compatibility: g058 표에 entry_ok 열 필요
- [2026-11-28] type=feat owner=claude
    summary: 신규 - g058 조건표 집계 - 조건 하나씩 제외 / 추가 / 단독, 익절 x 손절 1~15% 격자, 기간별 비교.
    impact: sim (읽기 전용: 주문 / 봇 상태 / runtime 쓰기 없음. 결과는 --out 폴더에만 쓴다)
    compatibility: backward-compatible (신규 파일)
"""
import sys, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "live_trading"))
sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]
C, TAX, KRW, GAP_MIN = 0.015, 0.20, 500_000, 30
L = np.arange(1, 16)

rows = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(D, "rows_*.pkl")))], ignore_index=True)
reb = pd.concat([pd.read_csv(f, dtype={"day": str, "code": str}) for f in sorted(glob.glob(os.path.join(D, "rebound_*.csv")))], ignore_index=True)
reb["t"] = pd.to_datetime(reb["t"]); reb["code"] = reb["code"].str.zfill(6)
df = rows.merge(reb.rename(columns={"plunge": "A_plunge", "rsi": "A_rsi", "macd": "A_macd", "dmi": "A_dmi", "ma": "A_ma", "angle": "A_angle"}),
                on=["day", "code", "t"], how="left")
n_reb = int(df["hit"].notna().sum())
n_extra = int((~reb.set_index(["day", "code", "t"]).index.isin(rows.set_index(["day", "code", "t"]).index)).sum())
print(f"table rows matched in the rebound csv {n_reb}/{len(rows)} (csv rows that are no table row: {n_extra})")
if rows.duplicated(["day", "code", "t"]).any() or reb.duplicated(["day", "code", "t"]).any() or len(df) != len(rows) or n_reb != len(rows):
    raise SystemExit("rebound csv does not match the table row for row (run AutoBotHtsH/scripts/rebound-conds.ts on this folder)")
n_noeod = int(df["eod"].isna().sum())
df = df[df["eod"].notna()]
print(f"rows without a 15:20 close left out: {n_noeod}")
if "entry_ok" not in df.columns:
    raise SystemExit("table built before 2026-11-28 round 3 (no entry_ok column): run g058 again")
# round 4 - 1: such a row is no entry. Bars mode: it is no candidate at all (its fill price would be a later bar's open).
# Tick mode: the signal itself was real, so it still holds the 30-minute spacing - only its result is not counted.
print(f"rows whose entry bar is missing (never counted as an entry): {int((~df.entry_ok).sum())}")
for k in ("A_plunge", "A_rsi", "A_macd", "A_dmi", "A_ma", "A_angle", "hit", "required", "edge"):
    df[k] = df[k].fillna(0).astype(int).astype(bool)
df["A_score"] = df["score"].fillna(0) >= 3
df["A_noedge"] = ~df["edge"]
df["A_any"] = df["hit"]                                   # classic or one of the three extra setups
df["A_classic"] = df["hit"] & df["setup"].isna()
df["V_tick"] = df["V_tick"] & df["V_tchase"]
df["E_prob"] = df["E_scored"] & (df["E_p"] >= df["E_thr"])
print("rows", len(df), "days", df.day.nunique(), "symbol-days", df.groupby(["day", "code"]).ngroups, "| with tick coverage", int(df.tick_cov.sum()))
BARS = len(sys.argv) > 2 and sys.argv[2] == "bars"
if not BARS:
    df = df[df.tick_cov]
df = df.sort_values(["day", "code", "t"]).reset_index(drop=True)
PER = np.where(df["day"].values < "20260901", 0, np.where(df["day"].values < "20261030", 1, 2))

RULE = ["R_1m", "R_tick", "R_guard", "R_mid", "R_drop", "R_chase", "R_regime", "R_atr1", "R_atr3", "R_liq", "R_antichase", "R_spread"]
VWAP = ["V_hist", "V_above", "V_trend", "V_gap", "V_cap", "V_drop"]          # the running setting: tick condition off
VTICK = [] if BARS else ["V_tick"]
if BARS:
    RULE = [c for c in RULE if c not in ("R_tick", "R_chase", "R_spread")]
EDGE = ["E_mid", "E_trig", "E_scored", "E_prob"]   # E_scored: the sample the model has a number for (not a rule of the trade)
ALERT = ["A_plunge", "A_rsi", "A_macd", "A_dmi", "A_ma", "A_angle", "A_score", "A_noedge"]
NAME = {"R_1m": "규칙: 1분 MA5>MA20(15분 이내) 또는 MA5 반등", "R_tick": "규칙: 60/120틱 골든크로스(4분 이내)", "R_guard": "규칙: 1분 캔들/BB 이격 추격 가드",
        "R_mid": "규칙: 현재가 > 1분 BB 중심선", "R_drop": "규칙: 확정 종가 대비 -0.4% 이내", "R_chase": "규칙: 틱 크로스 대비 +0.5% 이내",
        "R_regime": "규칙: 지수 ETF 1분 MA5>MA20", "R_atr1": "규칙: 1분 ATR 0.5% 이상", "R_atr3": "규칙: 3분 ATR 하한", "R_liq": "규칙: 최소 유동성",
        "R_antichase": "규칙: 당일 과열 차단(시가+8%/VWAP+3%/전일+23%)", "R_spread": "규칙: 호가 스프레드 2틱 이하",
        "V_hist": "VWAP: 당일 1분봉 21개 이상", "V_above": "VWAP: 종가·현재가 > VWAP", "V_trend": "VWAP: MA5 또는 20이평 상승", "V_gap": "VWAP: 시가 갭 1% 이상", "V_cap": "VWAP: VWAP 대비 +2% 이내",
        "V_drop": "VWAP: 확정 종가 대비 -0.4% 이내", "V_tick": "VWAP: 60틱 MA5/15 크로스(60초 이내, 옵션)",
        "E_mid": "Edge: 종가 > 1분 BB 중심선", "E_trig": "Edge: 1분 하이브리드 트리거", "E_scored": "Edge: (표본) 모델 점수가 있는 봉", "E_prob": "Edge: 모델 확률 >= 기준",
        "A_plunge": "알림 A: 급락 선행", "A_rsi": "알림 B: RSI 상향 돌파", "A_macd": "알림 C: MACD 상향 돌파", "A_dmi": "알림 D: DI+ 상향 돌파",
        "A_ma": "알림 E: MA5/MA20 골든크로스", "A_angle": "알림 F: MA5 각도", "A_score": "알림: 가점 3점 이상", "A_noedge": "알림: 장 시작/마감 15분 제외"}

TP_I = np.array(df["tp_i"].tolist()); SL_I = np.array(df["sl_i"].tolist())
TP_O = np.array(df["tp_o"].tolist()); SL_O = np.array(df["sl_o"].tolist())
ENT = df["entry"].to_numpy(float); EOD = df["eod"].to_numpy(float); TK = df["tick"].to_numpy(float)


def entries(mask: np.ndarray) -> np.ndarray:
    """row indexes of the entries: the first passing bar of a symbol-day, then not again for GAP_MIN minutes"""
    out = []; last = {}
    idx = np.flatnonzero(mask)
    d, c, t = df["day"].values, df["code"].values, df["t"].values
    ok = df["entry_ok"].to_numpy(bool)
    for i in idx:
        if BARS and not ok[i]:
            continue
        k = (d[i], c[i])
        if k in last and (t[i] - last[k]) < np.timedelta64(GAP_MIN, "m"):
            continue
        last[k] = t[i]; out.append(i)
    return np.array([i for i in out if ok[i]], dtype=int)


def _tick(p):
    return np.select([p < 2_000, p < 5_000, p < 20_000, p < 50_000, p < 200_000, p < 500_000], [1, 5, 10, 50, 100, 500], 1_000)


def up_grid(p):
    """the first quotable price at or over p (round 4 - 2: a level is seldom on the tick grid)"""
    p = np.asarray(p, float); t = _tick(p)
    return np.ceil(p / t - 1e-9) * t


def dn_grid(p):
    """the first quotable price at or under p"""
    p = np.asarray(p, float); t = _tick(p)
    return np.floor(p / t + 1e-9) * t


def down1(p):
    """one KRX tick under the price p (the tick of the band the sale is in - round 3 - 4: it was the entry price's tick)"""
    p = np.asarray(p, float)
    return p - np.select([p <= 2_000, p <= 5_000, p <= 20_000, p <= 50_000, p <= 200_000, p <= 500_000], [1, 5, 10, 50, 100, 500], 1_000)


def net(ix: np.ndarray, u: int, s: int) -> np.ndarray:
    """net % of each entry under take-profit u % / stop s % (0 = none), sold at 15:20 otherwise. A bar that holds both = the stop,
    unless it opens at or over the target (then the target came first - round 3 - 2)."""
    e = ENT[ix]; n = len(ix)
    ti = TP_I[ix, u - 1] if u else np.full(n, 9999); si = SL_I[ix, s - 1] if s else np.full(n, 9999)
    px = down1(EOD[ix])
    open_tp = (ti < 9999) & (ti == si) & (TP_O[ix, u - 1] >= e * (1 + u / 100) - 1e-6) if u else np.zeros(n, dtype=bool)
    hit_s = (si < 9999) & (si <= ti) & ~open_tp
    hit_t = (ti < 9999) & ~hit_s
    if s:
        px = np.where(hit_s, down1(np.minimum(SL_O[ix, s - 1], dn_grid(e * (1 - s / 100)))), px)
    if u:
        px = np.where(hit_t, np.maximum(TP_O[ix, u - 1], up_grid(e * (1 + u / 100))), px)
    g = px / e - 1
    return (g - (2 * C + TAX) / 100 - g * (C + TAX) / 100) * 100


def grid(ix: np.ndarray):
    m = np.full((15, 15), np.nan); w = np.full((15, 15), np.nan)
    if len(ix):
        for u in L:
            for s in L:
                r = net(ix, u, s); m[u - 1, s - 1] = r.mean(); w[u - 1, s - 1] = (r > 0).mean() * 100
    return m, w


def line(lbl: str, mask: np.ndarray, show=True) -> dict:
    ix = entries(mask)
    o = {"label": lbl, "n": len(ix), "days": int(df["day"].values[ix].size and len(set(df["day"].values[ix])))}
    if len(ix):
        m, w = grid(ix)
        bu, bs = np.unravel_index(np.nanargmax(m), m.shape)
        hold = net(ix, 0, 0)
        ref = net(ix, 3, 1); ref2 = net(ix, 2, 2)
        dsum = pd.Series(ref * KRW / 100).groupby(df["day"].values[ix]).sum()
        o.update(hold=hold.mean(), hold_win=(hold > 0).mean() * 100, tp3sl1=ref.mean(), tp3sl1_win=(ref > 0).mean() * 100, tp2sl2=ref2.mean(), tp2sl2_win=(ref2 > 0).mean() * 100,
                 best=m[bu, bs], best_tp=bu + 1, best_sl=bs + 1, best_win=w[bu, bs], cells_pos=int((m > 0).sum()), grid_mean=float(np.nanmean(m)),
                 days_pos=int((dsum > 0).sum()), top=(pd.Series(hold).groupby(df['day'].values[ix]).sum().abs().max() / max(1e-9, pd.Series(hold).groupby(df['day'].values[ix]).sum().abs().sum()) * 100))
        o["_m"], o["_w"] = m, w
    if show:
        if len(ix):
            print(f"  {lbl:44s} n={o['n']:4d} d={o['days']:2d} H{o['hold']:+.2f}({o['hold_win']:.0f}) 3/1 {o['tp3sl1']:+.2f}({o['tp3sl1_win']:.0f}) 2/2 {o['tp2sl2']:+.2f}({o['tp2sl2_win']:.0f}) best {o['best_tp']}/{o['best_sl']} {o['best']:+.2f}({o['best_win']:.0f}) pos {o['cells_pos']} avg {o['grid_mean']:+.2f} | " + " ".join(f"{'ABC'[q]}:{int((PER[ix]==q).sum())}/{(hold[PER[ix]==q].mean() if (PER[ix]==q).any() else float('nan')):+.2f}" for q in range(3)) + f" d+{o['days_pos']} top1 {o['top']:.0f}%")
        else:
            print(f"  {lbl:44s} n=   0")
    return o


def AND(cols) -> np.ndarray:
    m = np.ones(len(df), dtype=bool)
    for c in cols:
        m &= df[c].to_numpy(bool)
    return m


def loo(title: str, cols: list[str]) -> None:
    print(f"\n=== {title} ({len([c for c in cols if c != 'E_scored'])}개 조건) ===")
    line("전체 조건 적용", AND(cols))
    for c in cols:
        if c == "E_scored":      # the sample the model has a number for, not a rule that can be left out
            continue
        line(f"제외: {NAME[c]}", AND([x for x in cols if x != c]))


print("\n--- 조건별 통과율(봉 기준, %) ---")
for c in RULE + VWAP + VTICK + EDGE + ALERT:
    if c == "E_prob":        # among the scored bars (the unscored ones are not fails)
        sc_ = df["E_scored"].to_numpy(bool)
        print(f"  {NAME[c] + ' (점수 있는 봉 중)':50s} {(df.loc[sc_, c].mean() * 100 if sc_.any() else float('nan')):5.1f}")
        continue
    print(f"  {NAME[c]:50s} {df[c].mean()*100:5.1f}")

print("\n=== 매매 방식별 현재 조건 그대로 ===")
base = {"규칙매매(시장자료 조건 12개)": RULE, "VWAP 매매(현재: 갭 1%+, 틱 끔)": VWAP, "VWAP 매매(틱 조건 켬)": VWAP + VTICK, "Edge AI 매매": EDGE, "알림(바닥 반등 초입, 필수 6 + 가점)": ALERT}
res = {}
for k, cols in base.items():
    res[k] = line(k, AND(cols))
res["알림(추가 유형 포함 전체)"] = line("알림(추가 유형 포함 전체)", df["A_any"].to_numpy(bool))
for s in ("vTurn", "pullback", "vwapHold"):
    line(f"알림 유형 {s}", (df["hit"] & (df["setup"] == s)).to_numpy(bool))

# the tick-off VWAP set against r029.vwap_signal itself (the running r001 values)
if not bool(getattr(__import__("r001_define_config"), "VWAP_TRADE_TICK_TRIGGER", False)):
    print(f"\nVWAP 조건 묶음 vs r029.vwap_signal 불일치 봉: {int((AND(VWAP) != df['V_live_now'].to_numpy(bool)).sum())} / {len(df)}")
# the model on the bars that have a score only (a bar without one is not a fail)
print("\n=== Edge AI: 점수가 있는 봉만 (1분 트리거 통과) ===")
sc = df["E_scored"].to_numpy(bool) & df["E_trig"].to_numpy(bool)
line("점수 있음: 확률 >= 기준", sc & df["E_prob"].to_numpy(bool))
line("점수 있음: 확률 < 기준", sc & ~df["E_prob"].to_numpy(bool))
line("점수 있음: 전체", sc)

loo("종합: 규칙 + VWAP + Edge AI + 알림 전체", RULE + VWAP + VTICK + EDGE + ALERT)
loo("종합(알림 제외): 규칙 + VWAP + Edge AI", RULE + VWAP + VTICK + EDGE)
loo("규칙매매", RULE)
loo("VWAP 매매(틱 조건 포함)", VWAP + VTICK)
loo("Edge AI 매매", EDGE)
loo("알림(바닥 반등 초입)", ALERT)

print("\n=== 조합 탐색 ===")
for lbl, cols in (("갭1% + 1분ATR0.5%", ["V_gap", "R_atr1"]), ("갭1% + 1분ATR + VWAP 위", ["V_gap", "R_atr1", "V_above"]), ("VWAP(현재) + 1분ATR", VWAP + ["R_atr1"]),
                  ("VWAP(현재) + 1분ATR + 지수", VWAP + ["R_atr1", "R_regime"]), ("규칙 + 갭1%", RULE + ["V_gap"]), ("Edge + 갭1%", EDGE + ["V_gap"]),
                  ("Edge + 갭1% + 1분ATR", EDGE + ["V_gap", "R_atr1"]), ("Edge트리거 + 갭1% + 1분ATR", ["E_trig", "V_gap", "R_atr1"]),
                  ("갭1% + 1분ATR + 지수", ["V_gap", "R_atr1", "R_regime"]), ("갭1% + 1분ATR + 알림A 급락", ["V_gap", "R_atr1", "A_plunge"]),
                  ("1분ATR + 지수", ["R_atr1", "R_regime"]), ("갭1% + 지수", ["V_gap", "R_regime"]), ("1분ATR + VWAP 위", ["R_atr1", "V_above"]), ("갭1% + VWAP 아래", None)):
    m = AND(cols) if cols else (df["V_gap"] & ~df["V_above"]).to_numpy(bool)
    o = line(lbl, m)
    if o.get("n") and lbl in ("갭1% + 1분ATR0.5%", "VWAP(현재) + 1분ATR"):
        res[lbl] = o

print("\n=== 한 방식의 조건 + 다른 방식 조건 1개 추가 ===")
ALLC = RULE + VWAP + VTICK + EDGE + ALERT
for k, cols in (("VWAP(현재)", VWAP), ("Edge AI", EDGE), ("규칙", RULE)):
    print(f"\n-- {k} + 1 --")
    line("기준", AND(cols))
    for c in ALLC:
        if c not in cols:
            line(f"추가: {NAME[c]}", AND(cols + [c]))

print("\n=== 조건 하나만 (그 조건이 켜지는 첫 봉에 진입) ===")
for c in ALLC:
    line(NAME[c], df[c].to_numpy(bool))
line("모든 봉(기준선: 조건 없음)", np.ones(len(df), dtype=bool))

# the grids of the main sets
for k, o in res.items():
    if o.get("n"):
        pd.DataFrame(o["_m"], index=[f"tp{u}" for u in L], columns=[f"sl{s}" for s in L]).round(3).to_csv(os.path.join(D, f"grid_mean_{list(res).index(k)}.csv"), encoding="utf-8-sig")
        print(f"\n--- 익절(행) x 손절(열) 평균 순수익 % : {k} (n={o['n']}) ---")
        print(pd.DataFrame(o["_m"], index=[f"익{u}" for u in L], columns=[f"손{s}" for s in L]).round(2).to_string())

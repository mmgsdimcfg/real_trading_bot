# -*- coding: utf-8 -*-
"""g060 - the "open gap >= 1 % and 1-minute ATR >= 0.5 %" entry on the g058 table (read-only): exits, amounts at 500,000 KRW a
trade, what the entries look like by feature (gap size, ATR, time, place in the day's range, volume ...), one more condition added,
and larger gap floors. Every line carries the three periods (before 2026-09 / to 2026-10-29 / from 2026-10-30) - a cut that is
positive in one period only is not a finding. Result 2026-11-28: the base is -0.80 % / -0.89 % / +0.94 % by period.

usage: g060_gap_atr_study.py <g058 out dir> [bars]

Update log:
- [2026-11-28] type=fix owner=claude
    summary: Codex 3차 검토 - g059 와 같은 청산 규칙 / 호가 단위 / 진입 봉 필터, 하루 한도는 시각 -> 종목코드 순, 거래없음은 건수 기준.
    impact: sim (읽기 전용)
    compatibility: g058 표에 entry_ok 열 필요
- [2026-11-28] type=feat owner=claude
    summary: 신규 - 시가 갭 + 1분 ATR 진입의 청산 / 금액 / 구간별 / 조건 추가 / 갭 하한별 분석.
    impact: sim (읽기 전용: 주문 / 봇 상태 / runtime 쓰기 없음. 결과는 --out 폴더에만 쓴다)
    compatibility: backward-compatible (신규 파일)
"""
import sys, glob, os
from datetime import time as dt_time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]
BARS = len(sys.argv) > 2 and sys.argv[2] == "bars"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "live_trading"))
import r001_define_config as cfg  # noqa: E402
DATA = os.path.join(str(cfg.RUNTIME_ROOT), "data")
C, TAX, KRW, GAP_MIN = 0.015, 0.20, 500_000, 30
L = np.arange(1, 16)

rows = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(D, "rows_*.pkl")))], ignore_index=True)
reb = pd.concat([pd.read_csv(f, dtype={"day": str, "code": str}) for f in sorted(glob.glob(os.path.join(D, "rebound_*.csv")))], ignore_index=True)
reb["t"] = pd.to_datetime(reb["t"]); reb["code"] = reb["code"].str.zfill(6)
df = rows.merge(reb[["day", "code", "t", "plunge", "score", "hit"]], on=["day", "code", "t"], how="left")
if rows.duplicated(["day", "code", "t"]).any() or reb.duplicated(["day", "code", "t"]).any() or len(df) != len(rows) or int(df["hit"].notna().sum()) != len(rows):
    raise SystemExit("rebound csv does not match the table row for row (run AutoBotHtsH/scripts/rebound-conds.ts on this folder)")
if not BARS:
    df = df[df.tick_cov]
print("rows without a 15:20 close left out:", int(df["eod"].isna().sum()))
df = df[df["eod"].notna()]
if "entry_ok" not in df.columns:
    raise SystemExit("table built before 2026-11-28 round 3 (no entry_ok column): run g058 again")
# as g059: no entry is counted on such a row; in tick mode it still holds the 30-minute spacing
print("rows whose entry bar is missing (never counted as an entry):", int((~df.entry_ok).sum()))
df = df.sort_values(["day", "code", "t"]).reset_index(drop=True)
df["E_prob"] = df["E_scored"] & (df["E_p"] >= df["E_thr"])
df["A_plunge"] = df["plunge"].fillna(0).astype(bool)
PERN = ["7-8월", "9월-10/29", "10/30-11월"]
df["per"] = np.where(df["day"] < "20260901", 0, np.where(df["day"] < "20261030", 1, 2))

TP_I = np.array(df["tp_i"].tolist()); SL_I = np.array(df["sl_i"].tolist()); TP_O = np.array(df["tp_o"].tolist()); SL_O = np.array(df["sl_o"].tolist())
ENT = df["entry"].to_numpy(float); EOD = df["eod"].to_numpy(float); TK = df["tick"].to_numpy(float)


def entries(mask, gap_min=GAP_MIN, once=False):
    out = []; last = {}
    d, c, t = df["day"].values, df["code"].values, df["t"].values
    ok = df["entry_ok"].to_numpy(bool)
    for i in np.flatnonzero(mask):
        if BARS and not ok[i]:
            continue
        k = (d[i], c[i])
        if k in last and (once or (t[i] - last[k]) < np.timedelta64(gap_min, "m")):
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


def net(ix, u, s):
    """as g059.net: a bar holding both = the stop unless it opens at or over the target; sold one tick (of the sale's band) lower"""
    e = ENT[ix]; n = len(ix)
    ti = TP_I[ix, u - 1] if u else np.full(n, 9999); si = SL_I[ix, s - 1] if s else np.full(n, 9999)
    px = down1(EOD[ix])
    open_tp = (ti < 9999) & (ti == si) & (TP_O[ix, u - 1] >= e * (1 + u / 100) - 1e-6) if u else np.zeros(n, dtype=bool)
    hs = (si < 9999) & (si <= ti) & ~open_tp; ht = (ti < 9999) & ~hs
    if s: px = np.where(hs, down1(np.minimum(SL_O[ix, s - 1], dn_grid(e * (1 - s / 100)))), px)
    if u: px = np.where(ht, np.maximum(TP_O[ix, u - 1], up_grid(e * (1 + u / 100))), px)
    g = px / e - 1
    return (g - (2 * C + TAX) / 100 - g * (C + TAX) / 100) * 100


def stat(lbl, ix, u=0, s=0, width=40):
    if not len(ix):
        print(f"  {lbl:{width}s} n=0"); return
    r = net(ix, u, s); per = df["per"].values[ix]; day = df["day"].values[ix]
    dd = pd.Series(r * KRW / 100).groupby(day).sum()
    seg = " ".join(f"[{PERN[q]} {int((per==q).sum())}건 {r[per==q].mean():+.2f}%]" if (per == q).any() else f"[{PERN[q]} 0건]" for q in range(3))
    print(f"  {lbl:{width}s} n={len(ix):4d} 승률{(r>0).mean()*100:3.0f}% 평균{r.mean():+.2f}% 중앙{np.median(r):+.2f}% 합계{r.sum()*KRW/100:+11,.0f}원 수익일{int((dd>0).sum())}/{len(dd)} {seg}")


base_mask = (df["V_gap"] & df["R_atr1"]).to_numpy(bool)
B = entries(base_mask)
print("rows", len(df), "days", df.day.nunique(), "| base entries", len(B))

print("\n=== 1. 기준 조건(시가 갭 1%+ & 1분 ATR 0.5%+) - 청산 방식별 ===")
stat("종가까지 보유(익절·손절 없음)", B)
for u, s in ((0, 3), (0, 5), (0, 8), (5, 3), (5, 5), (8, 5), (10, 5), (10, 8), (15, 8), (3, 1), (3, 2), (2, 2)):
    stat(f"익절 {u or '없음'}% / 손절 {s or '없음'}%", B, u, s)

print("\n=== 2. 익절 x 손절 격자 (평균 순수익 %, 기간별로 따로) ===")
for q in (None, 0, 1, 2):
    ix = B if q is None else B[df["per"].values[B] == q]
    if not len(ix): continue
    m = np.array([[net(ix, u, s).mean() for s in L] for u in L])
    bu, bs = np.unravel_index(np.argmax(m), m.shape)
    print(f"-- {'전체' if q is None else PERN[q]} (n={len(ix)}) 최고 익{bu+1}/손{bs+1} {m[bu,bs]:+.2f}% | 플러스 칸 {(m>0).sum()}/225 | 보유 {net(ix,0,0).mean():+.2f}%")
    if q is None:
        print(pd.DataFrame(m[[0, 1, 2, 4, 6, 9, 14]][:, [0, 1, 2, 4, 6, 9, 14]], index=[f"익{u}" for u in (1, 2, 3, 5, 7, 10, 15)], columns=[f"손{s}" for s in (1, 2, 3, 5, 7, 10, 15)]).round(2).to_string())

print("\n=== 3. 금액 추산 (1회 50만원, 종가까지 보유) ===")
r = net(B, 0, 0); k = r * KRW / 100
bd = pd.DataFrame({"day": df["day"].values[B], "t": df["t"].values[B], "code": df["code"].values[B], "krw": k, "per": df["per"].values[B]})
print("  (금액 = 건당 50만원을 모두 투입했다고 본 추산. 하루 한도는 시각 순, 같은 시각이면 종목코드 순으로 채움 - 실제 봇의 선택 순서와 다를 수 있음)")
alld = sorted(df["day"].unique())
for cap in (None, 10, 5, 3):
    x = bd if cap is None else bd.sort_values(["t", "code"]).groupby("day").head(cap)
    dsum = x.groupby("day")["krw"].sum().reindex(alld, fill_value=0.0)
    dcnt = x.groupby("day").size().reindex(alld, fill_value=0)
    print(f"  하루 최대 {cap or '무제한':>4} 건: 거래 {len(x):4d}건 | 합계 {x.krw.sum():+11,.0f}원 | 거래일({len(alld)}일) 평균 {dsum.mean():+9,.0f}원/일 | 최악일 {dsum.min():+9,.0f} 최고일 {dsum.max():+9,.0f} | 수익일 {(dsum>0).sum()} 손실일 {(dsum<0).sum()} 거래없음 {(dcnt==0).sum()}"
          f" | 하루 최대 투입 {x.groupby('day').size().max()*KRW:,.0f}원")
    if cap in (None, 5):
        mm = x.groupby(x["day"].str[:6])["krw"].agg(["size", "sum"])
        print("     월별: " + "  ".join(f"{i} {int(v['size'])}건 {v['sum']:+,.0f}" for i, v in mm.iterrows()))
print("  수익 분포(건당 %): " + " ".join(f"{p}%={np.percentile(r,p):+.1f}" for p in (5, 25, 50, 75, 95)), f"| 최악 {r.min():+.1f} 최고 {r.max():+.1f}")
# how much of the total the best trades / days carry
srt = np.sort(k)[::-1]
print(f"  상위 5건 합 {srt[:5].sum():+,.0f}원 / 전체 {k.sum():+,.0f}원 | 상위 10% 거래 제외 시 평균 {np.sort(r)[: int(len(r)*0.9)].mean():+.2f}%")
ds = bd.groupby("day")["krw"].sum().sort_values(ascending=False)
print("  상위 3일: " + ", ".join(f"{d} {v:+,.0f}" for d, v in ds.head(3).items()) + " | 하위 3일: " + ", ".join(f"{d} {v:+,.0f}" for d, v in ds.tail(3).items()))

# ---- features at the entry bar, from the day's 1-minute bars ----
_c = {}
def bars(day, code):
    k = (day, code)
    if k not in _c:
        h = glob.glob(os.path.join(DATA, day, f"{code}_*_10s.txt")); m = None
        if h:
            raw = pd.read_csv(h[0], encoding="utf-8-sig", usecols=["datetime", "open", "high", "low", "close", "volume", "raw_bar"])
            raw = raw[raw["raw_bar"] == 1]
            raw.index = pd.to_datetime(raw.pop("datetime")) + pd.Timedelta(minutes=1)
            m = raw[(raw.index.time > dt_time(9, 0)) & (raw.index.time <= dt_time(15, 20))]
        _c[k] = m
    return _c[k]

days_all = sorted(d for d in os.listdir(DATA) if d.isdigit() and len(d) == 8)
def feats(i):
    day, code, T, price = df.at[i, "day"], df.at[i, "code"], df.at[i, "t"], df.at[i, "price"]
    o = {}
    m = bars(day, code)
    if m is not None and len(m):
        b = m[m.index <= T]
        if len(b) >= 15:
            pc = b["close"].shift(1)
            tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
            o["atr_pct"] = float(tr.iloc[-14:].mean() / b["close"].iat[-1] * 100)
            op = float(b["open"].iat[0]); hi = float(b["high"].max()); lo = float(b["low"].min())
            o["from_open"] = (price / op - 1) * 100
            o["range_pos"] = (price - lo) / (hi - lo) * 100 if hi > lo else np.nan
            o["from_high"] = (price / hi - 1) * 100
            o["ret5"] = (b["close"].iat[-1] / b["close"].iat[-6] - 1) * 100 if len(b) >= 6 else np.nan
            o["ret15"] = (b["close"].iat[-1] / b["close"].iat[-16] - 1) * 100 if len(b) >= 16 else np.nan
            v = b["volume"].astype(float)
            o["vol_burst"] = float(v.iloc[-3:].mean() / v.iloc[:-3].mean()) if len(v) > 6 and v.iloc[:-3].mean() > 0 else np.nan
            o["turnover_eok"] = float((b["close"] * v * 6).sum() / 1e8)
            prev = [d for d in days_all if d < day]
            pm = bars(prev[-1], code) if prev else None
            if pm is not None and len(pm):
                pv = float(pm["volume"].sum())
                o["vol_vs_prev"] = float(v.sum() / pv) if pv > 0 else np.nan
    for etf, nm in (("069500", "kospi_from_open"),):
        e = bars(day, etf)
        if e is not None and len(e):
            eb = e[e.index <= T]
            if len(eb):
                o[nm] = (float(eb["close"].iat[-1]) / float(eb["open"].iat[0]) - 1) * 100
    return o

F = pd.DataFrame([feats(i) for i in B], index=B)
F["open_gap"] = df["open_gap"].values[B]; F["vwap_gap"] = df["vwap_gap"].values[B]; F["price"] = df["price"].values[B]
F["minute"] = (pd.DatetimeIndex(df["t"].values[B]).hour * 60 + pd.DatetimeIndex(df["t"].values[B]).minute).astype(float)
F["tick_pct"] = df["tick"].values[B] / df["price"].values[B] * 100
F["E_p"] = df["E_p"].values[B]
seq = pd.Series(1, index=B).groupby([df["day"].values[B], df["code"].values[B]]).cumsum()
F["nth_entry"] = seq.values.astype(float)
F["r"] = r; F["per"] = df["per"].values[B]

print("\n=== 4. 기준 진입을 구간으로 나눠 본 종가 보유 평균 % (건수) - 기간별 일관성 ===")
LBL = {"open_gap": "시가 갭 %", "atr_pct": "1분 ATR %(당일 14봉 단순 평균 - 진입 판정의 ATR 과 다름)", "minute": "진입 시각", "vwap_gap": "VWAP 대비 %", "from_open": "시가 대비 %", "range_pos": "당일 범위 내 위치 %(0=저가,100=고가)",
       "from_high": "당일 고가 대비 %", "ret5": "직전 5분 수익률 %", "ret15": "직전 15분 수익률 %", "vol_burst": "직전 3분 거래량 / 당일 평균", "vol_vs_prev": "누적 거래량 / 전일 전체",
       "turnover_eok": "당일 누적 거래대금(억원)", "kospi_from_open": "KODEX200 시가 대비 %", "price": "주가(원)", "tick_pct": "호가단위/주가 %", "nth_entry": "그 종목 당일 몇 번째 진입", "E_p": "Edge AI 확률"}
CUTS = {"open_gap": [1, 2, 3, 5, 100], "minute": [545, 570, 600, 660, 780, 900], "vwap_gap": [-100, -1, 0, 1, 2, 100], "from_open": [-100, -2, 0, 2, 5, 100],
        "range_pos": [-1e9, 25, 50, 75, 90, 1e9], "nth_entry": [0.5, 1.5, 2.5, 100], "kospi_from_open": [-100, -1, -0.3, 0.3, 1, 100], "price": [0, 5000, 20000, 50000, 200000, 1e9],
        "tick_pct": [0, 0.1, 0.2, 0.3, 1]}
for col in LBL:
    if col not in F or F[col].notna().sum() < 30: continue
    x = F[F[col].notna()]
    cuts = list(CUTS.get(col) or sorted(set(np.quantile(x[col], [0, .25, .5, .75, 1]))))
    if len(cuts) < 3: continue
    cuts[0] = min(cuts[0], float(x[col].min())) - 1e-9; cuts[-1] = max(cuts[-1], float(x[col].max())) + 1e-9
    g = pd.cut(x[col], cuts, include_lowest=True)
    print(f"-- {LBL[col]}")
    for b_, y in x.groupby(g, observed=True):
        seg = " ".join(f"[{PERN[q]} {int((y.per==q).sum())}건 {y.r[y.per==q].mean():+.2f}]" if (y.per == q).any() else f"[{PERN[q]} -]" for q in range(3))
        print(f"   {str(b_):22s} n={len(y):4d} 승률{(y.r>0).mean()*100:3.0f}% 평균{y.r.mean():+.2f}% {seg}")

print("\n=== 5. 기준 + 조건 1개 추가 (종가 보유) ===")
ADD = {"R_regime": "지수 ETF 1분 MA5>MA20", "V_above": "VWAP 위", "V_trend": "MA5/20이평 상승", "R_1m": "1분 MA5>MA20", "R_mid": "현재가>BB 중심선", "R_liq": "최소 유동성",
       "R_atr3": "3분 ATR 하한", "R_antichase": "과열 차단", "E_trig": "Edge 1분 트리거", "E_prob": "Edge 확률>=기준", "A_plunge": "알림A 급락 선행", "V_cap": "VWAP +2% 이내"}
stat("기준", B)
for c, nm in ADD.items():
    stat(f"+ {nm}", entries(base_mask & df[c].to_numpy(bool)))
    # a bar without a model score is not a "fail": the opposite of E_prob is taken among the scored bars only
    opp = (df["E_scored"].to_numpy(bool) & ~df[c].to_numpy(bool)) if c == "E_prob" else ~df[c].to_numpy(bool)
    stat(f"+ NOT {nm}", entries(base_mask & opp))
print("-- 진입 방식")
stat("종목당 하루 1회만(첫 신호)", entries(base_mask, once=True))
stat("재진입 간격 60분", entries(base_mask, 60))
stat("재진입 간격 10분", entries(base_mask, 10))
F.to_csv(os.path.join(D, "boost_entries.csv"), encoding="utf-8-sig")

print("\n=== 6. 시가 갭 하한별 (종가 보유) ===")
gap_v = df["open_gap"].to_numpy(float); atr_b = df["R_atr1"].to_numpy(bool)
for lo in (2, 3, 5, 7):
    stat(f"갭 {lo}%+", entries(gap_v >= lo))
    stat(f"갭 {lo}%+ & 1분 ATR", entries((gap_v >= lo) & atr_b))
    stat(f"갭 {lo}%+ & 1분 ATR, 종목당 하루 1회", entries((gap_v >= lo) & atr_b, once=True))

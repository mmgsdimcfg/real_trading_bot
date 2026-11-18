# -*- coding: utf-8 -*-
"""g042_hi52_signal_log.py - HI52 swing signal log (forward test, no orders) + evaluation report.

Why: runtime/research/08_daily_swing (entry_signals_long.py, hi52_refine.py, 2018-06..2026-11) found one entry with an edge over
random stock picks: a new 250-day (or 500-day) closing high on an up day, while both index ETFs are in an uptrend, held ~40 days.
The edge is small and parameter-sensitive (about 260 portfolio trades in 8.5 years), so before any trading module it is recorded
forward, every trading day, and scored later against the same universe bought at random.

Signal at the close of D (same as the research):
  universe  6-digit code (not the index ETFs), close >= 2,000, 20-day average traded value (amount) >= 5 bn KRW, no day move or
            open gap whose absolute size exceeds 30.5% in the last 60 rows (split / data error)
  HI<lb>    close_D >= the highest close of the prior lb rows (at least 80% of lb rows present) and close_D > close_(D-1); lb = 250, 500
  gate      both index ETFs (r001 MARKET_REGIME_ETF 069500 / 229200): close_D > MA60_D and MA60_D > MA60 10 rows earlier
            ("UP"; the strategy only buys on UP days, signals of other days are logged too for comparison)
  Rows are trading days = dates where both index ETFs have a bar.
Planned trade (scored for hold 20 and 40): buy the open of the next trading day + 1 tick (no fill when that open is at +29.5% or
there is no bar), sell the close of the hold-th trading day. As in the research: a trade whose holding window has a move or open
gap beyond 30.5% is excluded; a code without a close on the planned exit day (suspended / delisted) exits at its last close inside
the window, on that day. Net % after commission 0.015% x 2 + the sell tax of the exit date. Benchmark = the whole universe of D
bought and sold by the same rules (equal weight = random picks); excess = net - benchmark.
Data: <runtime>/data/daily_backfill (g040, 2018.., optional - needed for lb 500) + <runtime>/data/daily_history (g025, rows after the
newest backfill row). A daily_history row with source_date == date is provisional (16:00 snapshot, corrected by the next folder).

Recorded kind of a logged date (a forward test only counts "live"):
  live      D is the newest trading day in the data and the run happens before the next weekday's 09:00 (local time) - the
            decision could still have been traded
  late      any other run for a single date (re-run of a past date, or after the next open) - computed with data that may have
            been corrected later
  backfill  --backfill-from
  A live date is never replaced by a late / backfill run.
Files (<runtime>/data/hi52_signals/, written under the lock .g042.lock):
  days.csv        THE log (one atomic file replace = one commit): date, gate, etf, n_hi250, n_hi500, universe, provisional,
                  recorded, logged_at, signals (JSON list of {lb, code, name, close, prior_high, amt20_bn, ret60_pct, provisional})
  signals.csv     a flat view of the signals in days.csv (date, lb, code, name, close, prior_high, amt20_bn, ret60_pct, gate,
                  provisional, recorded), rewritten from days.csv by every logging / evaluation run - never read back
  evaluation.csv  per signal: entry, exit / status per hold (closed | open | waiting_entry | no_bar | limit_up_open |
                  excluded_bad), net %, universe %, excess %. Recomputed from the current data on every run (a provisional
                  close that g025 corrects later is picked up by the next run).
  report.md       summary per recorded / lb / gate and by month
Run:
  python g042_hi52_signal_log.py --date 20261118 --evaluate     log one date (r021 postclose step "hi52") + rewrite the report
  python g042_hi52_signal_log.py                                 log the newest date in the data + evaluate
  python g042_hi52_signal_log.py --backfill-from 20250901        log every trading date from there (recorded=backfill)
  python g042_hi52_signal_log.py --evaluate                      report only
  python g042_hi52_signal_log.py --status                        one JSON line: last logged date, rows, last report time
Exit codes: 0 ok (also: --date on a weekend - nothing is done) / 2 no usable input (no daily files, an index ETF file missing) / 3 another g042 is running /
4 D is a weekday without index ETF bars (holiday, or the daily history of D was not written).
No KIS calls, no r003 / r004 change; read-only on daily_history / daily_backfill.

Update log:
- [2026-11-18] type=feat owner=claude
    summary: 신규 - HI52(52주 신고가) 스윙 신호 전진 기록과 평가 리포트 (사용자 요청, 주문 없음). Codex 검토 반영: live 판정
      (최신 거래일 + 다음 개장 전만), 잠금과 커밋 순서, 보유 구간 이상 변동 제외·마지막 종가 청산(연구와 동일), 입력 누락 종료 코드,
      신호 0건 리포트. 2/3차 검토: 기록 원본은 days.csv 한 파일(날짜 행에 신호 JSON, 원자적 교체 한 번 = 커밋, signals.csv 는
      파생 보기), 평가는 매번 전체 재계산(캐시 제거), 지표 준비 구간은 지수 ETF 거래일 600행, 30.5% 경계는 정수 비교,
      주말 --date 는 잠금 전에 rc 0.
    impact: tooling (r021 postclose 의 hi52 단계에서 실행, 매매 판정 불변)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live_trading"))
with contextlib.redirect_stdout(io.StringIO()):
    import r001_define_config as r001  # noqa: E402
import r019_platform as r019  # noqa: E402

READ_COLS = ["date", "open", "close", "amount"]
LOOKBACKS = (250, 500)
HOLDS = (20, 40)
MIN_CLOSE = 2000
MIN_AMT20 = 5e9
FEE = 0.00015
BAD_MOVE = 0.305
LIMIT_UP_OPEN = 1.295
ETFS = tuple(str(c) for c in dict(r001.MARKET_REGIME_ETF).values())
HISTORY_ROWS = 800          # rows per code when nothing older is needed
WARMUP_ROWS = 600           # index ETF trading rows loaded before the oldest date logged / evaluated (lb 500 + 60 + slack)
OPEN_HOUR = 9               # a date is "live" until this hour of the next weekday
RANK = {"live": 2, "late": 1, "backfill": 1}
SIG_COLS = ["date", "lb", "code", "name", "close", "prior_high", "amt20_bn", "ret60_pct", "gate", "provisional", "recorded"]
SIG_FIELDS = ["lb", "code", "name", "close", "prior_high", "amt20_bn", "ret60_pct", "provisional"]     # stored per date in days.csv
DAY_COLS = ["date", "gate", "etf"] + [f"n_hi{lb}" for lb in LOOKBACKS] + ["universe", "provisional", "recorded", "logged_at",
                                                                           "signals"]
EVAL_COLS = ["date", "lb", "code", "name", "gate", "recorded", "entry_date", "entry_px"] + [
    f"{k}{h}{s}" for h in HOLDS for k, s in (("exit", "_date"), ("exit", "_px"), ("net", "_pct"), ("univ", "_pct"),
                                              ("excess", "_pct"), ("status", ""))]


class InputError(SystemExit):
    pass


def runtime_root() -> Path:
    return Path(os.environ.get("AUTO_TRADING_RUNTIME") or r001.RUNTIME_ROOT)


def out_dir() -> Path:
    return runtime_root() / "data" / "hi52_signals"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def sell_tax(ymd: str) -> float:
    """securities transaction tax + rural special tax on the sell date (KOSPI and KOSDAQ totals are the same)."""
    for lim, t in (("20190603", 0.0030), ("20210101", 0.0025), ("20230101", 0.0023), ("20240101", 0.0020),
                   ("20250101", 0.0018), ("20260101", 0.0015)):
        if ymd < lim:
            return t
    return 0.0020


def tick(p):
    p = np.asarray(p, float)
    t = np.full(p.shape, 1000.0)
    for lim, tk in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        t = np.where((p < lim) & (t == 1000.0), tk, t)
    return t


def classify(d: str, newest: str, now: datetime | None = None) -> str:
    """live only for the newest trading day before the next weekday's OPEN_HOUR (holidays are ignored - stricter)."""
    if d != newest:
        return "late"
    now = now or datetime.now()
    nxt = datetime.strptime(d, "%Y%m%d") + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return "live" if now < nxt.replace(hour=OPEN_HOUR) else "late"


# ---------------------------------------------------------------- data
def _read(path: Path, with_source: bool) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, dtype={"date": str, "source_date": str}, encoding="utf-8",
                         usecols=lambda c: c in READ_COLS or (with_source and c == "source_date"))
    except Exception:
        return None
    if "date" not in df.columns or "close" not in df.columns:
        return None
    df = df[df["date"].astype(str).str.fullmatch(r"\d{8}")].copy()
    for c in READ_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    df["prov"] = (df["source_date"] == df["date"]) if with_source and "source_date" in df.columns else False
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()[READ_COLS[1:] + ["prov"]]


def load_panel(start: str | None = None) -> dict[str, pd.DataFrame]:
    """{open, close, amount, prov: DataFrame trading days x codes} from daily_backfill + daily_history (history adds the rows after
    the newest backfill row). Per code: every row from `start` on, but never fewer than its last HISTORY_ROWS rows."""
    data = runtime_root() / "data"
    hist = {p.stem: p for p in (data / "daily_history").glob("*.csv")}
    back = {p.stem: p for p in (data / "daily_backfill").glob("*.csv")} if (data / "daily_backfill").is_dir() else {}
    missing = [e for e in ETFS if e not in hist and e not in back]
    if missing:
        raise InputError(f"[g042] index ETF daily file missing: {missing} (daily_history / daily_backfill)")
    frames = {}
    for code in sorted(set(hist) | set(back)):
        if not (len(code) == 6 and code.isalnum()):
            continue
        b = _read(back[code], False) if code in back else None
        h = _read(hist[code], True) if code in hist else None
        if b is not None and len(b) and h is not None and len(h):
            df = pd.concat([b, h[h.index > b.index.max()]])
        else:
            df = b if b is not None and len(b) else h
        if df is None or not len(df):
            continue
        tail = df.tail(HISTORY_ROWS)
        df = df[df.index >= start] if start and (df.index >= start).sum() > len(tail) else tail
        if len(df):
            frames[code] = df
    if any(e not in frames for e in ETFS):
        raise InputError(f"[g042] no usable index ETF rows {ETFS} under {data}")
    panel = pd.concat(frames, axis=1).sort_index()
    out = {k: panel.xs(k, axis=1, level=1) for k in READ_COLS[1:] + ["prov"]}
    for k in ("open", "close"):
        out[k] = out[k].astype(float).where(out[k] > 0)
    out["amount"] = out["amount"].astype(float)
    out["prov"] = out["prov"].eq(True)
    cal = out["close"][list(ETFS)].notna().all(axis=1)       # trading days = both index ETFs have a bar
    if not cal.any():
        raise InputError("[g042] the index ETF files have no common dates")
    return {k: v.loc[cal] for k, v in out.items()}


def names() -> dict[str, str]:
    p = Path(__file__).resolve().parent / "g004_universe_symbols_master.txt"
    out = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.lstrip("# ").split("#")[0].strip()
            parts = s.split(",")
            if len(parts) >= 2 and len(parts[0]) == 6:
                out[parts[0]] = parts[1].strip()
    except Exception:
        pass
    return out


def indicators(P: dict[str, pd.DataFrame]) -> dict:
    C, O, A = P["close"], P["open"], P["amount"]
    prev = C.shift(1)
    lim = prev * round(BAD_MOVE * 1000)          # |x - prev| / prev > 0.305 without float rounding at the boundary
    bad = ((C - prev).abs() * 1000 > lim) | ((O - prev).abs() * 1000 > lim)
    ind = {"bad": bad, "bad60": bad.astype(float).rolling(60, min_periods=1).max() > 0,
           "amt20": A.rolling(20, min_periods=20).mean(), "ret60": C / C.shift(60) - 1}
    pick = np.array([c.isdigit() and c not in ETFS for c in C.columns])     # the index ETFs are the gate, not a pick
    ind["univ"] = (C >= MIN_CLOSE) & (ind["amt20"] >= MIN_AMT20) & C.notna() & ~ind["bad60"] & pick[None, :]
    for lb in LOOKBACKS:
        ind[f"hip{lb}"] = C.shift(1).rolling(lb, min_periods=int(lb * 0.8)).max()
        ind[f"sig{lb}"] = (C >= ind[f"hip{lb}"]) & (C > prev) & ind["univ"]
    ic = C[list(ETFS)]
    m60 = ic.rolling(60, min_periods=60).mean()
    m60p = m60.shift(10)
    up = (ic > m60) & (m60 > m60p)
    known = m60.notna().all(axis=1) & m60p.notna().all(axis=1)
    ind["gate"] = pd.Series(np.where(~known, "NA", np.where(up.all(axis=1), "UP", "DOWN")), index=C.index)
    icv, mv, mpv = ic.to_numpy(), m60.to_numpy(), m60p.to_numpy()
    ind["etf"] = pd.Series([" / ".join(f"{e} c={icv[i, j]:,.0f} ma60={mv[i, j]:,.0f} ma60_10={mpv[i, j]:,.0f}"
                                       if np.isfinite(mv[i, j]) and np.isfinite(mpv[i, j]) else f"{e} n/a"
                                       for j, e in enumerate(ETFS)) for i in range(len(C.index))], index=C.index)
    return ind


# ---------------------------------------------------------------- log files
def _read_csv(p: Path, cols: list[str]) -> pd.DataFrame:
    """the file as strings; a missing file is empty. An unreadable existing file raises (it must never be treated as empty and
    then overwritten)."""
    if not p.is_file():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(p, dtype=str, encoding="utf-8", keep_default_na=False)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def _write_csv(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, p)


def log_dates(P, ind, dates: list[str], recorded: str | None, log=print, now: datetime | None = None) -> int:
    """recorded None = classify each date (live / late); "backfill" for --backfill-from. A date already logged with a higher rank
    (live) is kept."""
    nm = names()
    C = P["close"]
    newest = C.index[-1]
    days_p = out_dir() / "days.csv"
    days = _read_csv(days_p, DAY_COLS)
    have = dict(zip(days["date"], days["recorded"]))
    new_days, done = [], []
    for d in dates:
        if d not in C.index:
            continue
        rec = recorded or classify(d, newest, now)
        if RANK.get(have.get(d, ""), 0) > RANK[rec]:
            log(f"[g042] {d}: already logged as {have[d]} - kept")
            continue
        gate = str(ind["gate"].at[d])
        prov = bool(P["prov"].loc[d, list(ETFS)].any())
        counts, day_sigs = {}, []
        for lb in LOOKBACKS:
            row = ind[f"sig{lb}"].loc[d]
            hits = list(row[row].index)
            counts[lb] = str(len(hits)) if ind[f"hip{lb}"].loc[d].notna().any() else ""   # "" = not enough rows, unknown
            for c in hits:
                r60 = ind["ret60"].at[d, c]
                day_sigs.append({"lb": str(lb), "code": c, "name": nm.get(c, ""), "close": f"{C.at[d, c]:.0f}",
                                 "prior_high": f"{ind[f'hip{lb}'].at[d, c]:.0f}",
                                 "amt20_bn": f"{ind['amt20'].at[d, c] / 1e9:.1f}",
                                 "ret60_pct": f"{r60 * 100:.1f}" if pd.notna(r60) else "",
                                 "provisional": str(int(bool(P["prov"].at[d, c])))})
        new_days.append({"date": d, "gate": gate, "etf": ind["etf"].at[d], **{f"n_hi{lb}": counts[lb] for lb in LOOKBACKS},
                         "universe": str(int(ind["univ"].loc[d].sum())), "provisional": str(int(prov)), "recorded": rec,
                         "logged_at": now_text(), "signals": json.dumps(day_sigs, ensure_ascii=False)})
        done.append(d)
        log(f"[g042] {d} gate={gate} " + " ".join(f"HI{lb}={counts[lb] or '-'}" for lb in LOOKBACKS) +
            f" provisional={int(prov)} ({rec})")
    if not done:
        return 0
    days = pd.concat([days[~days["date"].isin(done)], pd.DataFrame(new_days, columns=DAY_COLS)]).sort_values("date")
    _write_csv(days, days_p)                                           # the commit (one atomic replace)
    _write_csv(signals_of(days), out_dir() / "signals.csv")           # derived view only
    return len(done)


# ---------------------------------------------------------------- evaluation
class Scorer:
    """trade rules shared by the signal and its benchmark (research: entry_signals_long.py / hi52_refine.py)."""

    def __init__(self, P, ind):
        self.cal = list(P["close"].index)
        self.pos = {d: i for i, d in enumerate(self.cal)}
        self.C, self.O = P["close"].to_numpy(), P["open"].to_numpy()
        self.Cff = P["close"].ffill().to_numpy()
        self.bad = ind["bad"].to_numpy()
        self.univ = ind["univ"].to_numpy()
        self.col = {c: j for j, c in enumerate(P["close"].columns)}
        self.tax = np.array([sell_tax(d) for d in self.cal])
        self.bench = {}

    def _window(self, i, hold):
        """entry row, last row, status for the window of a signal at row i."""
        e = i + 1
        x = e + hold - 1
        return e, min(x, len(self.cal) - 1), "closed" if x < len(self.cal) else "open"

    def trade(self, i, hold, j):
        n = len(self.cal)
        if i + 1 >= n:
            return {"status": "waiting_entry"}
        e, x, st = self._window(i, hold)
        o, pc = self.O[e, j], self.Cff[i, j]
        if not np.isfinite(o):
            return {"status": "no_bar", "entry_date": self.cal[e]}
        if np.isfinite(pc) and o >= pc * LIMIT_UP_OPEN:
            return {"status": "limit_up_open", "entry_date": self.cal[e]}
        p = float(o + tick(o))
        if self.bad[e:x + 1, j].any():
            return {"status": "excluded_bad", "entry_date": self.cal[e], "entry_px": p}
        cw = self.C[e:x + 1, j]
        valid = np.flatnonzero(np.isfinite(cw))
        if not len(valid):
            return {"status": "no_bar", "entry_date": self.cal[e]}
        k = e + valid[-1]                       # planned exit, or the last close inside the window (suspended / delisted)
        xp = float(self.C[k, j])
        net = (xp * (1 - FEE - self.tax[k]) - p * (1 + FEE)) / (p * (1 + FEE)) * 100
        return {"status": st, "entry_date": self.cal[e], "entry_px": p, "exit_date": self.cal[k], "exit_px": xp, "net": net}

    def universe(self, i, hold):
        key = (i, hold)
        if key in self.bench:
            return self.bench[key]
        n = len(self.cal)
        if i + 1 >= n:
            self.bench[key] = np.nan
            return np.nan
        e, x, _ = self._window(i, hold)
        o, pc = self.O[e], self.Cff[i]
        ok = self.univ[i] & np.isfinite(o) & ~(o >= pc * LIMIT_UP_OPEN) & ~self.bad[e:x + 1].any(axis=0)
        cw = self.C[e:x + 1][:, ok]
        valid = np.isfinite(cw)
        has = valid.any(axis=0)
        last = cw.shape[0] - 1 - valid[::-1].argmax(axis=0)
        xp = cw[last, np.arange(cw.shape[1])]
        p = o[ok] + tick(o[ok])
        r = (xp * (1 - FEE - self.tax[e + last]) - p * (1 + FEE)) / (p * (1 + FEE)) * 100
        r = r[has & np.isfinite(r)]
        self.bench[key] = float(r.mean()) if len(r) else np.nan
        return self.bench[key]


def signals_of(days: pd.DataFrame) -> pd.DataFrame:
    """the flat signal table from days.csv (the only source)."""
    rows = []
    for d in days.itertuples(index=False):
        for x in json.loads(d.signals or "[]"):
            rows.append({**{k: str(x.get(k, "")) for k in SIG_FIELDS}, "date": d.date, "gate": d.gate, "recorded": d.recorded})
    return pd.DataFrame(rows, columns=SIG_COLS)


def evaluate(P, ind, log=print) -> None:
    days = _read_csv(out_dir() / "days.csv", DAY_COLS)
    sigs = signals_of(days)
    _write_csv(sigs, out_dir() / "signals.csv")                      # keep the derived view in step with days.csv
    sc = Scorer(P, ind)
    rows, skipped = [], 0
    for s in sigs.itertuples(index=False):
        if s.date not in sc.pos or s.code not in sc.col:
            skipped += 1
            continue
        i, j = sc.pos[s.date], sc.col[s.code]
        row = {"date": s.date, "lb": s.lb, "code": s.code, "name": s.name, "gate": s.gate, "recorded": s.recorded,
               "entry_date": "", "entry_px": ""}
        for h in HOLDS:
            t = sc.trade(i, h, j)
            b = sc.universe(i, h)
            net = t.get("net")
            row["entry_date"] = row["entry_date"] or t.get("entry_date", "")
            row["entry_px"] = row["entry_px"] or (f"{t['entry_px']:.0f}" if t.get("entry_px") else "")
            row.update({f"exit{h}_date": t.get("exit_date", ""), f"exit{h}_px": f"{t['exit_px']:.0f}" if "exit_px" in t else "",
                        f"net{h}_pct": f"{net:.3f}" if net is not None else "",
                        f"univ{h}_pct": f"{b:.3f}" if np.isfinite(b) else "",
                        f"excess{h}_pct": f"{net - b:.3f}" if net is not None and np.isfinite(b) else "",
                        f"status{h}": t["status"]})
        rows.append(row)
    ev = pd.DataFrame(rows, columns=EVAL_COLS).sort_values(["date", "lb", "code"])
    _write_csv(ev, out_dir() / "evaluation.csv")
    if skipped:
        log(f"[g042] evaluate: {skipped} signals outside the loaded data")
    write_report(ev, sc.cal[-1], days, log)


def write_report(ev: pd.DataFrame, last: str, days: pd.DataFrame, log=print) -> None:
    lines = [f"# HI52 signal log - report ({now_text()}, data up to {last})", "",
             "Planned trade: buy the next open + 1 tick, sell the close of the 20th / 40th holding day; net % after commission and "
             "the sell tax of the exit date. Benchmark = the whole universe of that day traded by the same rules (random picks); "
             "excess = net - benchmark. The strategy is gate UP only. Only `live` rows are a forward test (recorded before the "
             "next open); `late` / `backfill` rows were computed afterwards.", ""]
    if len(days):
        dl = days[days["recorded"] == "live"]
        lines += [f"Logged dates: {len(days)} (live {len(dl)}, first live {dl['date'].min() if len(dl) else '-'}, "
                  f"last {days['date'].max()}). Gate UP on {int((days['gate'] == 'UP').sum())} of them; last gate "
                  f"{days.sort_values('date')['gate'].iloc[-1]}.", ""]
    if not len(ev):
        lines += ["No signals yet."]
    else:
        for h in HOLDS:
            lines += [f"## Hold {h} days", "",
                      "| recorded | lb | gate | closed | mean net % | mean excess % | win % | open (latest net %) | excluded |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for (rec, lb, gate), g in ev.groupby(["recorded", "lb", "gate"]):
                st = g[f"status{h}"]
                c = g[st == "closed"]
                net = pd.to_numeric(c[f"net{h}_pct"], errors="coerce")
                exc = pd.to_numeric(c[f"excess{h}_pct"], errors="coerce")
                o = pd.to_numeric(g.loc[st == "open", f"net{h}_pct"], errors="coerce")
                nx = int(st.isin(["excluded_bad", "no_bar", "limit_up_open"]).sum())
                cl = f"{len(c)} | {net.mean():.2f} | {exc.mean():.2f} | {(net > 0).mean() * 100:.0f}" if len(c) else "0 | - | - | -"
                op = f"{len(o)} ({o.mean():.2f})" if len(o) else "0"
                lines.append(f"| {rec} | {lb} | {gate} | {cl} | {op} | {nx} |")
            lines.append("")
        g = ev[(ev["gate"] == "UP") & (ev["status40"] == "closed")].copy()
        if len(g):
            g["month"] = g["date"].str[:6]
            lines += ["## Strategy (gate UP, hold 40, closed) by signal month", "",
                      "| month | recorded | lb | n | mean net % | mean excess % |", "|---|---|---|---|---|---|"]
            for (m, rec, lb), x in g.groupby(["month", "recorded", "lb"]):
                lines.append(f"| {m} | {rec} | {lb} | {len(x)} | {pd.to_numeric(x['net40_pct']).mean():.2f} | "
                             f"{pd.to_numeric(x['excess40_pct'], errors='coerce').mean():.2f} |")
            lines.append("")
        lines += ["Research reference (2018-06..2026-11, gate UP, hold 40, per-signal-day mean): HI250 net about +1.4% (2018-2022) / "
                  "+6.4% (2023-2026), random picks -0.6% / +3.3%. A forward excess near zero after a few months means the edge "
                  "did not carry over."]
    p = out_dir() / "report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".md.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    log(f"[g042] report -> {p} ({len(ev)} signals scored)")


def status() -> dict:
    days = _read_csv(out_dir() / "days.csv", DAY_COLS)
    sigs = signals_of(days)
    rp = out_dir() / "report.md"
    live = days[days["recorded"] == "live"]
    return {"dir": str(out_dir()), "days": len(days), "last_date": days["date"].max() if len(days) else None,
            "last_live": live["date"].max() if len(live) else None,
            "last_gate": days.sort_values("date")["gate"].iloc[-1] if len(days) else None, "signals": len(sigs),
            "report_at": datetime.fromtimestamp(rp.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%S") if rp.is_file() else None}


def _etf_dates() -> list[str]:
    """trading days = dates where every index ETF has a bar (backfill + history), sorted."""
    data = runtime_root() / "data"
    sets = []
    for e in ETFS:
        ds = set()
        for folder in ("daily_backfill", "daily_history"):
            f = data / folder / f"{e}.csv"
            if f.is_file():
                with contextlib.suppress(Exception):
                    ds |= set(pd.read_csv(f, dtype={"date": str}, usecols=["date", "close"]).dropna()["date"])
        sets.append(ds)
    return sorted(set.intersection(*sets)) if sets else []


def _start_for(dates: list[str]) -> str | None:
    """first date to load: WARMUP_ROWS index-ETF trading days before the oldest date logged now or already logged (every logged
    signal is evaluated again on each run)."""
    need = [d for d in dates if d]
    days = _read_csv(out_dir() / "days.csv", DAY_COLS)
    if len(days):
        need.append(days["date"].min())
    if not need:
        return None
    cal = _etf_dates()
    first = min(need)
    k = sum(1 for d in cal if d < first)
    return cal[max(0, k - WARMUP_ROWS)] if cal else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="log this trading date (YYYYMMDD); default: the newest date in the data")
    ap.add_argument("--backfill-from", help="log every trading date from this one to the newest (recorded=backfill)")
    ap.add_argument("--evaluate", action="store_true", help="rewrite evaluation.csv / report.md")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.status:
        print(json.dumps(status(), ensure_ascii=False))
        return 0
    if a.date and datetime.strptime(a.date, "%Y%m%d").weekday() >= 5:
        print(f"[g042] {a.date} is a weekend day - nothing done")
        return 0
    lock = r019.acquire_instance_lock(out_dir() / ".g042.lock")
    if lock is None:
        print("[g042] another g042 run holds the lock - nothing done")
        return 3
    try:
        only_eval = a.evaluate and not a.date and not a.backfill_from
        try:
            P = load_panel(start=_start_for([a.date, a.backfill_from]))
        except InputError as exc:
            print(exc.code if isinstance(exc.code, str) else exc)
            return 2
        ind = indicators(P)
        cal = list(P["close"].index)
        print(f"[g042] data {cal[0]}..{cal[-1]} ({len(cal)} trading days, {P['close'].shape[1]} codes)", flush=True)
        rc = 0
        if a.backfill_from:
            n = log_dates(P, ind, [d for d in cal if d >= a.backfill_from], "backfill")
            print(f"[g042] backfill: {n} dates logged")
        elif not only_eval:
            d = a.date or cal[-1]
            if d not in cal:
                print(f"[g042] {d}: no index ETF bars for that weekday (newest {cal[-1]}) - a holiday, or the daily history of "
                      f"{d} was not written (g025)")
                rc = 4
            else:
                log_dates(P, ind, [d], None)
        if a.evaluate or not (a.date or a.backfill_from):
            evaluate(P, ind)
        return rc
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

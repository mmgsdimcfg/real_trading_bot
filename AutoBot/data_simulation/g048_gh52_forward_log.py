# -*- coding: utf-8 -*-
"""g048_gh52_forward_log.py - George-Hwang 52-week-high (GH52) monthly portfolio forward log (no orders) + evaluation report.

Why: runtime/research/13_public_strategies (2026-11-20) applied published trading rules with their PUBLISHED parameters to
2018-06..2026-11 daily data. Only the 52-week-high proximity rule of George & Hwang (2004) beat the equal-weight universe in both
periods after a Holm correction (excess about +13%p / yr, 2023..: +19%p). It is the same effect as g042 (HI52), but with parameters
fixed by others - so it is recorded forward with a pre-registered rule (see REGISTERED) before any trading module.

Formation at the close of the LAST KRX TRADING DAY of a month (D):
  universe  the g042 universe (6-digit code, close >= 2,000, 20-day mean traded value >= 5 bn KRW, no move / open gap beyond
            30.5% in 60 rows) and a common stock: code ending in 0 (KRX: preferred shares end in 5 / 7 / 9 / K..), not an ETF
            (brand prefix + space, or a fund word after a space), ETN, SPAC (..호스팩) or REIT (..리츠 at the end)
  score     close_D / the highest close of the last 252 rows (at least 200 rows present)
  picks     the top 30% by score (ties: code order), equal weight; the universe of D is stored too (the benchmark)
Month ends / trade days: from REGISTERED on by the KRX calendar (r014: weekday, not "closed") - D is a month end when the next KRX
day is in another month, the entry is the next KRX day, the exit is the KRX day after the 6th following calendar month end. A
planned day missing in the data is a data gap (never shifted to another row). Before REGISTERED (backfill) the data rows decide
(r014 holds no old holidays).
Planned trade (scored per cohort): buy the open of the entry day + 1 tick (no open or an open at +29.5%: not bought), sell the open
of the exit day - 1 tick; no open on that day: the first open (else close) after it; nothing within SETTLE_ROWS rows: unresolved =
0 recovery (-100%) - until then the cohort is "settling". No pick bought at all = the cohort stays in cash (net 0). A cohort whose
exit day is not reached is marked at the newest close (status open). Net % after commission 0.015% x 2 and the sell tax of the
exit date. Benchmark = the stored universe of D by the same rules (equal weight); excess = net - benchmark.
Reference (not judged): KODEX 200 (069500) bought and sold at the same opens (ETF tick 5 KRW, no sell tax).
Recorded kind (only "live" counts): live = D is the newest data row, the run is after D's KRX close and before the next KRX day's
open (r014 times); late = any other single-date run; backfill = --backfill-from. A live formation is never replaced (not even by a
live re-run).
Files (<runtime>/data/gh52_signals/, under the lock .g048.lock):
  formations.csv     THE log (atomic replace): date, universe_n, picks_n, provisional, recorded, logged_at, picks (JSON list of
                     {code, name, close, high252, score}), universe (JSON list of codes)
  evaluation.csv     per cohort: planned entry / exit, status (waiting_entry | open | settling | closed | data_gap), traded picks,
                     unresolved, net %, universe %, excess %, KODEX 200 % - recomputed from the current data on every run
  acceptance_final.json  written ONCE when the verdict is reached; never recomputed afterwards
  report.md          acceptance progress, missing months, cohort table, summary per recorded kind
Pre-registered acceptance (REGISTERED, user approval 2026-11-21, judged ONCE, never tuned afterwards):
  sample = the ACCEPT_COHORTS (36) consecutive calendar month ends from the first one on / after REGISTERED (2026-11 .. 2029-10).
  A month without a LIVE cohort counts net 0 and excess 0 (conservative); more than MAX_MISSING (6) missing months = INVALID.
  The verdict waits until every sample cohort is closed (settled). PASS = the one-sided 95% lower bounds (circular block bootstrap
  over the 36-month sequence, BOOT_BLOCK 6 = the overlap, BOOT_N draws, seed 42) of the mean net AND the mean excess are both > 0,
  also after an extra round trip of EXTRA_COST_PCT per cohort.
  Power (research gh52_power.txt): 36 cohorts detect the 2018-2026 mean effect (+3.6%p / 6 months, sd 5.8) with ~80%; half of it
  would need ~140 - a FAIL at 36 does not prove the effect is zero.
Run:
  python g048_gh52_forward_log.py --date 20261130 --evaluate   r021 postclose step "gh52" (logs only on a month end) + report
  python g048_gh52_forward_log.py --backfill-from 20180601     log every month end from there (recorded=backfill)
  python g048_gh52_forward_log.py --evaluate                   report only
  python g048_gh52_forward_log.py --status                     one JSON line
Exit codes: 0 ok (also: D is not a month end / a weekend - nothing logged) / 2 no usable input / 3 another g048 is running /
4 D is a weekday without index ETF bars (holiday, or the daily history of D was not written) / 5 D is a month end but no cohort
could be formed (empty universe) - the month would be missing, re-run before the next open.
No KIS calls, no r003 / r004 change; read-only on daily_history / daily_backfill.

Update log:
- [2026-11-21] type=fix owner=claude
    summary: Codex 2차 - 청산가 탐색은 청산일부터 SETTLE_ROWS 행까지만(늦게 들어온 가격이 0원 처리를 되살리지 않음), 비교 유니버스도
      정산돼야 closed, 데이터에 없는 예정일은 그 뒤 GAP_ROWS(5)행이 생기면 다음 행으로(note 기록, 미등록 휴장일), 그 전에는 data_gap.
    impact: tooling
    compatibility: evaluation.csv 열 추가(note)
- [2026-11-21] type=fix owner=claude
    summary: Codex 코드 검토 반영 - live 기록은 live 재실행도 덮어쓰지 않음, live 창 = D 장 마감 이후 ~ 다음 KRX 개장 전(r014 시각),
      등록일 이후 월말/진입/청산일은 KRX 달력으로 정하고 데이터 누락은 data_gap(다른 행으로 밀지 않음), 미해결 종목은 청산일 뒤
      20행까지 settling, 체결 0건 코호트 = 현금(0%), 판정 표본 = 2026-11 부터 달력 월말 36개(live 없는 달은 0, 6개 초과 누락 INVALID),
      최종 판정은 acceptance_final.json 에 한 번만 기록, 월말에 코호트를 못 만들면 rc 5, 종목 필터를 코드 끝자리(우선주)·브랜드
      접두어(ETF)·이름 끝(리츠/스팩) 기준으로 (YG PLUS, 성우, 메리츠금융지주 오제외 / WON 200 통과 수정).
    impact: tooling. 주문/감시목록 영향 없음
    compatibility: formations.csv 형식 불변, evaluation.csv 열 추가(planned_entry, planned_exit)
- [2026-11-21] type=feat owner=claude
    summary: 신규 - George-Hwang 52주 고점 근접 월간 포트폴리오 전진 기록과 사전 등록 판정 (사용자 승인: 표본 36개, r021 새 단계 gh52).
      공개 파라미터(252일 고점 대비 종가 상위 30%, 6개월 중첩 보유) 그대로. 주문 없음.
    impact: tooling (새 출력 폴더 <runtime>/data/gh52_signals). 매매/감시목록 영향 없음
    compatibility: backward-compatible (새 파일)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live_trading"))
import g042_hi52_signal_log as g042  # noqa: E402
import r014_market_calendar as r014  # noqa: E402
import r019_platform as r019  # noqa: E402

LB, LB_MIN = 252, 200
PICK = 0.30
HOLD_MONTHS = 6
K200, ETF_TICK = "069500", 5.0
FEE = g042.FEE
WARMUP_ROWS = 300
SETTLE_ROWS = 20
GAP_ROWS = 5
REGISTERED = "20261121"
ACCEPT_COHORTS, MAX_MISSING = 36, 6
BOOT_BLOCK, BOOT_N, EXTRA_COST_PCT = 6, 10000, 0.25
BRAND_RE = re.compile(r"^(KODEX|TIGER|KBSTAR|RISE|ACE|SOL|HANARO|ARIRANG|PLUS|KOSEF|TIMEFOLIO|WON|1Q|UNICORN|마이티|히어로즈|TRUSTON|"
                      r"에셋플러스|파워|BNK|KCGI|FOCUS|TREX|VITA|ITF|마이다스|DAISHIN343|DAISHIN|KIWOOM|KoAct|HK|MASTER|QV|TIME|"
                      r"MIDAS|IBK|DS|아이엠에셋|더제이|N2|TRUE|신한|미래에셋|삼성|KB)\s", re.I)
FUNDWORD_RE = re.compile(r"\s.*(액티브|레버리지|인버스|선물|채권|합성|커버드|코스피|코스닥|나스닥|S&P|KRX|200|150)")
OTHER_RE = re.compile(r"ETN|리츠$|스팩(\d+호)?$|\d+호스팩")
FORM_COLS = ["date", "universe_n", "picks_n", "provisional", "recorded", "logged_at", "picks", "universe"]
EVAL_COLS = ["date", "recorded", "picks_n", "traded", "unresolved", "planned_entry", "planned_exit", "entry_date", "exit_date",
             "status", "net_pct", "univ_pct", "excess_pct", "k200_pct", "note"]


def out_dir() -> Path:
    return g042.runtime_root() / "data" / "gh52_signals"


def stock_names() -> dict[str, str]:
    """g004 names, completed with the newest daily folders (<runtime>/data/YYYYMMDD/<code>_<name>_daily.csv)."""
    nm = dict(g042.names())
    data = g042.runtime_root() / "data"
    with contextlib.suppress(Exception):
        folders = sorted(p for p in data.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name))[-5:]
        for f in reversed(folders):
            for p in f.glob("*_daily.csv"):
                code, _, name = p.name[:-len("_daily.csv")].partition("_")
                if len(code) == 6 and code not in nm and name:
                    nm[code] = name
    return nm


def is_stock(code: str, name: str) -> bool:
    name = name or ""
    return (code.isdigit() and code.endswith("0") and code not in g042.ETFS and code != K200 and not BRAND_RE.search(name)
            and not FUNDWORD_RE.search(name) and not OTHER_RE.search(name))


# ---------------------------------------------------------------- KRX calendar
def _is_krx_day(x: datetime) -> bool:
    return x.weekday() < 5 and r014.session_hours(x) is not None


def next_krx_day(d: str) -> str:
    """the next weekday that the r014 calendar does not mark closed."""
    x = datetime.strptime(d, "%Y%m%d") + timedelta(days=1)
    for _ in range(30):
        if _is_krx_day(x):
            break
        x += timedelta(days=1)
    return x.strftime("%Y%m%d")


def cal_month_end(yyyymm: str) -> str:
    """the last KRX day of a calendar month."""
    first_next = (datetime.strptime(yyyymm + "01", "%Y%m%d") + timedelta(days=32)).replace(day=1)
    x = first_next - timedelta(days=1)
    while not _is_krx_day(x):
        x -= timedelta(days=1)
    return x.strftime("%Y%m%d")


def add_months(yyyymm: str, n: int) -> str:
    y, m = int(yyyymm[:4]), int(yyyymm[4:]) + n
    y += (m - 1) // 12
    return f"{y:04d}{(m - 1) % 12 + 1:02d}"


def month_ends(cal: list[str]) -> list[str]:
    """month-end rows of the data: from REGISTERED on by the KRX calendar, before it by the next data row."""
    out = []
    for k, d in enumerate(cal):
        if d >= REGISTERED or k == len(cal) - 1:
            if next_krx_day(d)[:6] != d[:6]:
                out.append(d)
        elif cal[k + 1][:6] != d[:6]:
            out.append(d)
    return out


def _hhmm(day: str, which: int):
    s = r014.session_hours(datetime.strptime(day, "%Y%m%d"))
    return s[which] if s else None


def classify(d: str, newest: str, now: datetime | None = None) -> str:
    """live = d is the newest row and now is after d's KRX close and before the next KRX day's open."""
    if d != newest:
        return "late"
    now = now or datetime.now()
    close, nxt = _hhmm(d, 1), next_krx_day(d)
    opn = _hhmm(nxt, 0)
    if close is None or opn is None:
        return "late"
    start = datetime.combine(datetime.strptime(d, "%Y%m%d").date(), close)
    end = datetime.combine(datetime.strptime(nxt, "%Y%m%d").date(), opn)
    return "live" if start <= now < end else "late"


def calendar_horizon() -> str | None:
    """the newest dated line of the r014 calendar file (holidays after it are unknown)."""
    with contextlib.suppress(Exception):
        ds = [ln[:8] for ln in r014.CALENDAR_PATH.read_text(encoding="utf-8").splitlines() if re.match(r"\d{8}\s", ln)]
        return max(ds) if ds else None
    return None


# ---------------------------------------------------------------- formation
def form(P, ind, d: str, nm: dict) -> dict:
    C = P["close"]
    i = C.index.get_loc(d)
    hist = C.iloc[max(0, i - LB + 1): i + 1]
    hi = hist.max()
    cnt = hist.notna().sum()
    score = (C.loc[d] / hi).where(cnt >= LB_MIN)
    u = ind["univ"].loc[d] & score.notna()
    codes = [c for c in u[u].index if is_stock(c, nm.get(c, ""))]
    order = sorted(codes, key=lambda c: (-float(score[c]), c))
    k = max(1, int(round(len(order) * PICK))) if order else 0
    return {"universe": codes, "picks": [{"code": c, "name": nm.get(c, ""), "close": f"{C.at[d, c]:.0f}", "high252": f"{hi[c]:.0f}",
                                          "score": f"{score[c]:.4f}"} for c in order[:k]]}


def log_dates(P, ind, dates: list[str], recorded: str | None, log=print, now: datetime | None = None) -> tuple[int, list[str]]:
    """returns (logged count, month ends that could not be formed)."""
    cal = list(P["close"].index)
    newest = cal[-1]
    ends = set(month_ends(cal))
    nm = stock_names()
    fp = out_dir() / "formations.csv"
    forms = g042._read_csv(fp, FORM_COLS)
    have = dict(zip(forms["date"], forms["recorded"]))
    new, done, failed = [], [], []
    for d in dates:
        if d not in ends:
            continue
        rec = recorded or classify(d, newest, now)
        if have.get(d) == "live" or g042.RANK.get(have.get(d, ""), 0) > g042.RANK[rec]:
            log(f"[g048] {d}: already logged as {have[d]} - kept")
            continue
        f = form(P, ind, d, nm)
        if not f["universe"]:
            log(f"[g048] {d}: empty universe (fewer than {LB_MIN} rows of history?) - not logged")
            failed.append(d)
            continue
        prov = bool(P["prov"].loc[d, [c for c in f["universe"] if c in P["prov"].columns]].any())
        new.append({"date": d, "universe_n": str(len(f["universe"])), "picks_n": str(len(f["picks"])), "provisional": str(int(prov)),
                    "recorded": rec, "logged_at": g042.now_text(), "picks": json.dumps(f["picks"], ensure_ascii=False),
                    "universe": json.dumps(f["universe"])})
        done.append(d)
        log(f"[g048] {d} month end: universe {len(f['universe'])}, picks {len(f['picks'])} (top {PICK:.0%} of close / 252-day high), "
            f"provisional={int(prov)} ({rec})")
    if done:
        forms = pd.concat([forms[~forms["date"].isin(done)], pd.DataFrame(new, columns=FORM_COLS)]).sort_values("date")
        g042._write_csv(forms, fp)
    return len(done), failed


# ---------------------------------------------------------------- evaluation
class Scorer:
    def __init__(self, P):
        self.cal = list(P["close"].index)
        self.pos = {d: i for i, d in enumerate(self.cal)}
        self.C, self.O = P["close"].to_numpy(float), P["open"].to_numpy(float)
        self.Cff = P["close"].ffill().to_numpy(float)
        self.col = {c: j for j, c in enumerate(P["close"].columns)}
        self.tax = np.array([g042.sell_tax(d) for d in self.cal])
        self.ends = [self.pos[d] for d in month_ends(self.cal)]

    def plan(self, d: str):
        """planned entry / exit dates (strings) of a cohort formed at d."""
        if d >= REGISTERED:
            exit_me = cal_month_end(add_months(d[:6], HOLD_MONTHS))
            return next_krx_day(d), next_krx_day(exit_me)
        i = self.pos[d]
        e = self.cal[i + 1] if i + 1 < len(self.cal) else ""
        later = [k for k in self.ends if k > i]
        x = self.cal[later[HOLD_MONTHS - 1] + 1] if len(later) >= HOLD_MONTHS and later[HOLD_MONTHS - 1] + 1 < len(self.cal) else ""
        return e, x

    def basket(self, i: int, codes: list[str], e: int, x: int | None, etf: bool = False):
        """(mean net %, bought, unresolved, settled) of the codes bought at the open of row e and sold at the open of row x
        (None: marked at the newest close). Nothing bought: (nan, 0, 0, True)."""
        js = np.array([self.col[c] for c in codes if c in self.col], int)
        if not len(js):
            return np.nan, 0, 0, True
        o = self.O[e, js]
        ok = np.isfinite(o) & ~(o >= self.Cff[i, js] * g042.LIMIT_UP_OPEN)
        js, o = js[ok], o[ok]
        if not len(js):
            return np.nan, 0, 0, True
        tk = np.full(len(o), ETF_TICK) if etf else g042.tick(o)
        p = (o + tk) * (1 + FEE)
        n = len(self.cal)
        if x is None:
            xp = self.Cff[n - 1, js]
            tax = np.zeros(len(js)) if etf else np.full(len(js), self.tax[n - 1])
            r = ((xp - (np.full(len(js), ETF_TICK) if etf else g042.tick(xp))) * (1 - FEE - tax) / p - 1) * 100
            return float(np.nanmean(r)), len(js), 0, True
        xp = np.full(len(js), np.nan)
        tax = np.zeros(len(js))
        lim = x + SETTLE_ROWS + 1                          # a price later than SETTLE_ROWS rows never revives a write-off
        for m, j in enumerate(js):
            # [Codex round 3] the FIRST row with any price decides (its open, else its close): a later row never changes it
            anyp = np.flatnonzero(np.isfinite(self.O[x:lim, j]) | np.isfinite(self.C[x:lim, j]))
            if not len(anyp):
                continue
            k = x + anyp[0]
            xp[m] = self.O[k, j] if np.isfinite(self.O[k, j]) else self.C[k, j]
            tax[m] = 0.0 if etf else self.tax[k]
        have = np.isfinite(xp)
        unres = int((~have).sum())
        settled = unres == 0 or (n - 1 - x) >= SETTLE_ROWS
        r = np.full(len(js), -100.0)
        xt = np.full(have.sum(), ETF_TICK) if etf else g042.tick(xp[have])
        r[have] = ((xp[have] - xt) * (1 - FEE - tax[have]) / p[have] - 1) * 100
        return float(r.mean()), len(js), unres, settled


def evaluate(P, log=print) -> pd.DataFrame:
    forms = g042._read_csv(out_dir() / "formations.csv", FORM_COLS)
    sc = Scorer(P)
    newest = sc.cal[-1]
    rows = []

    def resolve(day: str):
        """row of a planned KRX day; missing in the data: the next row once GAP_ROWS later rows exist (an unlisted holiday),
        else None (data gap - wait)."""
        if day in sc.pos:
            return sc.pos[day], ""
        later = [k for k, x in enumerate(sc.cal) if x > day]
        if len(later) >= GAP_ROWS:
            return later[0], f"{day} missing -> {sc.cal[later[0]]}"
        return None, ""

    for f in forms.itertuples(index=False):
        row = {c: "" for c in EVAL_COLS}
        row.update({"date": f.date, "recorded": f.recorded, "picks_n": f.picks_n})
        if f.date not in sc.pos:
            row["status"] = "data_gap"
            rows.append(row)
            continue
        i = sc.pos[f.date]
        pe, px = sc.plan(f.date)
        row.update({"planned_entry": pe, "planned_exit": px})
        if not pe or pe > newest:
            row["status"] = "waiting_entry"
            rows.append(row)
            continue
        e, note_e = resolve(pe)
        if e is None:
            row["status"] = "data_gap"
            rows.append(row)
            continue
        x, note_x = None, ""
        if px and px <= newest:
            x, note_x = resolve(px)
            if x is None:
                row["status"] = "data_gap"
                rows.append(row)
                continue
        picks = [p["code"] for p in json.loads(f.picks or "[]")]
        univ = json.loads(f.universe or "[]")
        net, traded, unres, settled = sc.basket(i, picks, e, x)
        ub, _, _, u_settled = sc.basket(i, univ, e, x)
        settled = settled and u_settled                               # the benchmark must be final too
        kb, _, _, _ = sc.basket(i, [K200], e, x, etf=True)
        if traded == 0:
            net = 0.0                                                    # nothing bought: the cohort stays in cash
        ub = ub if np.isfinite(ub) else 0.0
        status = "open" if x is None else ("closed" if settled else "settling")
        row.update({"traded": str(traded), "unresolved": str(unres), "entry_date": sc.cal[e],
                    "exit_date": sc.cal[x] if x is not None else "", "note": "; ".join(n for n in (note_e, note_x) if n),
                    "status": status, "net_pct": f"{net:.3f}", "univ_pct": f"{ub:.3f}", "excess_pct": f"{net - ub:.3f}",
                    "k200_pct": f"{kb:.3f}" if np.isfinite(kb) else ""})
        rows.append(row)
    ev = pd.DataFrame(rows, columns=EVAL_COLS)
    g042._write_csv(ev, out_dir() / "evaluation.csv")
    write_report(ev, sc.cal, log)
    return ev


def lower_bound(x: np.ndarray, block: int = BOOT_BLOCK, n_boot: int = BOOT_N, seed: int = 42) -> float:
    """one-sided 95% lower bound of the mean, circular block bootstrap over the cohort sequence."""
    x = np.asarray(x, float)
    T = len(x)
    if T == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    idx = ((rng.integers(0, T, (n_boot, nb))[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :T]) % T
    return float(np.percentile(x[idx].mean(axis=1), 5))


def sample_months() -> list[str]:
    """the 36 calendar month ends of the pre-registered sample."""
    first = REGISTERED[:6] if cal_month_end(REGISTERED[:6]) >= REGISTERED else add_months(REGISTERED[:6], 1)
    return [cal_month_end(add_months(first, k)) for k in range(ACCEPT_COHORTS)]


def acceptance(ev: pd.DataFrame, newest: str | None = None, final_path: Path | None = None) -> dict:
    final_path = final_path or (out_dir() / "acceptance_final.json")
    if final_path.is_file():
        with contextlib.suppress(Exception):
            out = json.loads(final_path.read_text(encoding="utf-8"))
            out["note"] = f"final verdict recorded {out.get('decided_at', '')} - not recomputed"
            return out
    months = sample_months()
    out = {"registered": REGISTERED, "sample": f"{months[0]}..{months[-1]}", "rule": f"{ACCEPT_COHORTS} calendar month ends from "
           f"{months[0]}; a month without a live cohort = net 0 / excess 0 (more than {MAX_MISSING} missing = INVALID); judged "
           f"once all are closed; pass = one-sided 95% lower bounds of mean net and mean excess > 0, also with +{EXTRA_COST_PCT}% "
           f"round trip ({BOOT_BLOCK}-cohort blocks, {BOOT_N} draws)"}
    live = ev[ev["recorded"] == "live"].set_index("date")
    newest = newest or ""
    net, exc, missing, pending = [], [], [], 0
    for m in months:
        if m in live.index:
            r = live.loc[m]
            if r["status"] != "closed":
                pending += 1
                net.append(np.nan); exc.append(np.nan)
                continue
            net.append(float(r["net_pct"])); exc.append(float(r["excess_pct"]))
        else:
            if newest and m < newest and next_krx_day(m) <= newest:
                missing.append(m)                                         # its live window has passed
            elif not newest or m >= newest:
                pending += 1
            net.append(0.0); exc.append(0.0)
    in_s = live[live.index.isin(months)]
    out.update({"live_cohorts": len(in_s), "closed": int((in_s["status"] == "closed").sum()), "missing": missing})
    if len(missing) > MAX_MISSING:
        out["verdict"] = "INVALID"
    elif pending or not newest or newest < next_krx_day(months[-1]):
        out["verdict"] = "collecting" if out["live_cohorts"] + len(missing) < ACCEPT_COHORTS else "waiting for the last 6-month windows"
        return out
    else:
        net_a, exc_a = np.array(net), np.array(exc)
        res = {}
        for label, extra in (("base", 0.0), ("plus_cost", EXTRA_COST_PCT)):
            n, x = net_a - extra, exc_a - extra
            res[label] = {"net_mean": float(n.mean()), "excess_mean": float(x.mean()), "net_lb": lower_bound(n), "excess_lb": lower_bound(x)}
        out["result"] = res
        out["verdict"] = "PASS" if all(v["net_lb"] > 0 and v["excess_lb"] > 0 for v in res.values()) else "FAIL"
    out["decided_at"] = g042.now_text()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, final_path)
    return out


def write_report(ev: pd.DataFrame, cal: list[str], log=print) -> None:
    acc = acceptance(ev, cal[-1])
    hz = calendar_horizon()
    lines = [f"# GH52 (George-Hwang 52-week high) forward log - report ({g042.now_text()}, data up to {cal[-1]})", "",
             "Monthly cohort formed at the last KRX trading day's close: top 30% of the universe by close / 252-day high, equal weight; "
             "bought at the next KRX day's open + 1 tick, sold at the open of the KRX day after the 6th following month end - 1 tick "
             "(no open: first price after it; none within 20 rows: 0 recovery). Benchmark = the stored universe of the formation "
             "day by the same rules; KODEX 200 at the same opens is a reference only. Only `live` cohorts are a forward test.", "",
             "## Pre-registered acceptance (judged once)", "", f"- rule: {acc.get('rule', '')}",
             f"- sample {acc.get('sample', '')}: live cohorts {acc.get('live_cohorts', 0)} / {ACCEPT_COHORTS}, missing months "
             f"{len(acc.get('missing', []))} {acc.get('missing', [])}; verdict: **{acc['verdict']}** {acc.get('note', '')}"]
    for k, v in acc.get("result", {}).items():
        lines.append(f"- {k}: mean net {v['net_mean']:.2f}% (LB {v['net_lb']:.2f}), mean excess {v['excess_mean']:.2f}%p "
                     f"(LB {v['excess_lb']:.2f})")
    lines += ["- power: 36 cohorts detect the 2018-2026 effect (+3.6%p / 6 months) with ~80%; half of it needs ~140 cohorts.",
              f"- KRX calendar (r014) dated up to {hz or '-'}: month ends / trade days after it assume no unlisted holiday - keep "
              f"r014_market_calendar.txt updated (KRX notices).", ""]
    if not len(ev):
        lines.append("No cohorts yet.")
    else:
        lines += ["## Summary by recorded kind (closed cohorts)", "",
                  "| recorded | cohorts | closed | mean net % | mean universe % | mean excess %p | excess > 0 % | mean KODEX200 % |",
                  "|---|---|---|---|---|---|---|---|"]
        for rec, g in ev.groupby("recorded"):
            c = g[g["status"] == "closed"]
            num = {k: pd.to_numeric(c[k], errors="coerce") for k in ("net_pct", "univ_pct", "excess_pct", "k200_pct")}
            cl = (f"{num['net_pct'].mean():.2f} | {num['univ_pct'].mean():.2f} | {num['excess_pct'].mean():.2f} | "
                  f"{(num['excess_pct'] > 0).mean() * 100:.0f} | {num['k200_pct'].mean():.2f}") if len(c) else "- | - | - | - | -"
            lines.append(f"| {rec} | {len(g)} | {len(c)} | {cl} |")
        lines += ["", "## Cohorts (newest first)", "",
                  "| formed | recorded | picks | status | entry | exit | net % | universe % | excess %p | KODEX200 % | unresolved |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in ev.sort_values("date", ascending=False).itertuples(index=False):
            lines.append(f"| {r.date} | {r.recorded} | {r.picks_n} | {r.status} | {r.planned_entry} | {r.planned_exit} | {r.net_pct} | "
                         f"{r.univ_pct} | {r.excess_pct} | {r.k200_pct} | {r.unresolved} |")
        lines += ["", "Research reference (runtime/research/13_public_strategies, 2018-06..2026-11): 95 monthly cohorts, mean excess "
                  "+3.61%p / 6 months (sd 5.84, > 0 in 71%), 2023..: +6.64%p (> 0 in 92%). KODEX 200 over the same windows was "
                  "higher on average (+13.9% vs +5.6% net) - the test is against the stock universe, not the index."]
    p = out_dir() / "report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".md.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    log(f"[g048] report -> {p} ({len(ev)} cohorts, acceptance {acc['verdict']})")


def status() -> dict:
    forms = g042._read_csv(out_dir() / "formations.csv", FORM_COLS)
    live = forms[forms["recorded"] == "live"]
    rp = out_dir() / "report.md"
    return {"dir": str(out_dir()), "cohorts": len(forms), "last_date": forms["date"].max() if len(forms) else None,
            "last_live": live["date"].max() if len(live) else None,
            "report_at": datetime.fromtimestamp(rp.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%S") if rp.is_file() else None,
            "acceptance_registered": REGISTERED, "acceptance_cohorts": ACCEPT_COHORTS, "calendar_horizon": calendar_horizon()}


def _start_for(dates: list[str]) -> str | None:
    need = [d for d in dates if d]
    forms = g042._read_csv(out_dir() / "formations.csv", FORM_COLS)
    if len(forms):
        need.append(forms["date"].min())
    if not need:
        return None
    cal = g042._etf_dates()
    first = min(need)
    k = sum(1 for d in cal if d < first)
    return cal[max(0, k - WARMUP_ROWS)] if cal else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="the job date (YYYYMMDD): a cohort is logged only when it is a month end")
    ap.add_argument("--backfill-from", help="log every month end from this date to the newest (recorded=backfill)")
    ap.add_argument("--evaluate", action="store_true", help="rewrite evaluation.csv / report.md")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.status:
        print(json.dumps(status(), ensure_ascii=False))
        return 0
    if a.date and datetime.strptime(a.date, "%Y%m%d").weekday() >= 5:
        print(f"[g048] {a.date} is a weekend day - nothing done")
        return 0
    lock = r019.acquire_instance_lock(out_dir() / ".g048.lock")
    if lock is None:
        print("[g048] another g048 run holds the lock - nothing done")
        return 3
    try:
        try:
            P = g042.load_panel(start=_start_for([a.date, a.backfill_from]))
        except g042.InputError as exc:
            print(exc.code if isinstance(exc.code, str) else exc)
            return 2
        ind = g042.indicators(P)
        cal = list(P["close"].index)
        print(f"[g048] data {cal[0]}..{cal[-1]} ({len(cal)} trading days, {P['close'].shape[1]} codes)", flush=True)
        hz = calendar_horizon()
        if hz and hz < (datetime.now() + timedelta(days=45)).strftime("%Y%m%d"):
            print(f"[g048] WARNING: the r014 KRX calendar is dated only up to {hz} - add the coming holidays (month ends may be wrong)")
        rc = 0
        if a.backfill_from:
            n, _ = log_dates(P, ind, [d for d in cal if d >= a.backfill_from], "backfill")
            print(f"[g048] backfill: {n} month ends logged")
        elif a.date or not a.evaluate:
            d = a.date or cal[-1]
            if d not in cal:
                print(f"[g048] {d}: no index ETF bars for that weekday (newest {cal[-1]}) - a holiday, or the daily history of {d} "
                      f"was not written (g025)")
                rc = 4
            elif d not in month_ends(cal):
                print(f"[g048] {d}: not the last trading day of its month - no cohort (next KRX day {next_krx_day(d)})")
            else:
                _, failed = log_dates(P, ind, [d], None)
                if failed:
                    rc = 5
        if a.evaluate or not (a.date or a.backfill_from):
            evaluate(P)
        return rc
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

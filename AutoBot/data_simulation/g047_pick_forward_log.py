# -*- coding: utf-8 -*-
"""g047_pick_forward_log.py - forward log of the g002 picks with the market gate and their 1/5/10/20-day results (no orders).

Why: AutoBot/selection_filter_redesign_20261119.md chapter 10/11 - the picks beat the tradable universe more the longer they are held,
but the 2026-06-29..11-17 data had the market gate DOWN on every day (KODEX 200 -46% from the June high), so "market gate UP +
multi-day hold" and "tick >= 0.20% excluded" could not be validated. Like the HI52 log (g042), this records every day from now on and
scores it later against the same universe, so the question can be answered once gate-UP days exist.

Logged per date D (the picks file g002 wrote: <runtime>/data/D/_D_picks.txt, "code,name" lines):
  gate      the g042 market gate of D: both index ETFs (r001 MARKET_REGIME_ETF): close_D > MA60_D and MA60_D > MA60 10 rows earlier
  per pick  close_D, tick_pct = KRX tick(close_D) / close_D (known on D - the tick group "<0.20%" / ">=0.20%"), fallback (replay
            rows only - the picks file does not say)
Scored (the g042 Scorer - same rules as the HI52 log): buy the open of D+1 + 1 tick (no fill when that open is at +29.5%), sell the
close of the k-th trading day (k = 1 same-day close, 5, 10, 20) - 1 tick. [2026-11-20, g042 scoring v2] A window with a move / gap
beyond 30.5% is kept (flagged bad); a code without a close on the planned day sells at its first close after it, none yet =
unresolved (0 recovery, -100%, counted with the closed ones). Net % after commission 0.015% x 2 + the sell tax of the exit date.
Benchmark = the g042 universe of D (close >= 2,000, 20-day turnover >= 5 bn) bought and sold by the same rules; excess = net -
benchmark.

Recorded kind (a forward test only counts "live"):
  live     D is the newest trading day in the data and the run happens before the next weekday's 09:00 (g042 classify)
  late     any other run for one date
  replay   --import-replay: the g043 point-in-time replay picks (research/10_selection/rows_<tag>.jsonl.gz) - computed later
  A date logged with a higher kind is never replaced (live > late > replay).
Files (<runtime>/data/pick_forward/, written under the lock .g047.lock):
  days.csv        THE log: date, gate, etf, n_picks, source, recorded, logged_at, picks (JSON list of {code, name, close, tick_pct,
                  fallback})
  evaluation.csv  per pick: entry, exit / net / universe / excess / status per hold - recomputed from the current data on every run
  report.md       per hold: recorded x gate x tick group (closed n, mean net, mean excess, win %, open)
Run:
  python g047_pick_forward_log.py --date 20261119 --evaluate          log one date (r021 postclose step "picklog") + the report
  python g047_pick_forward_log.py --import-replay rows_v0.jsonl.gz     log the replay picks (recorded=replay) + the report
  python g047_pick_forward_log.py --evaluate                           report only
  python g047_pick_forward_log.py --status                             one JSON line
Exit codes: 0 ok (also: weekend --date, nothing done) / 2 no usable input (no daily files, no picks file for D) / 3 another g047 is
running / 4 D is a weekday without index ETF bars (holiday, or the daily history of D was not written).
No KIS calls, no r003 / r004 change; read-only on daily_history / daily_backfill / the picks files.

Update log:
- [2026-11-20] type=fix owner=claude
    summary: g042 채점 v2 를 따름(Scorer 공유) - 매도 1틱 차감, 이상 변동 구간 유지(bad 표시), 청산일 종가 없으면 다음 첫 종가,
      끝까지 없으면 unresolved(0원 회수, 청산 집계에 포함). 리포트의 excluded 는 no_bar / limit_up_open 만.
    impact: tooling (리포트 수치 재계산)
    compatibility: evaluation.csv 형식 그대로 (상태 excluded_bad 없어짐, unresolved 추가)
- [2026-11-19] type=feat owner=claude
    summary: 신규 - g002 선정 종목 전진 기록(시장 필터 상태, 1틱 비율) + 1/5/10/20일 성과 평가 리포트 (사용자 요청, 주문 없음).
      채점은 g042 Scorer 재사용. g043 재현 선정은 --import-replay 로 recorded=replay 기록.
    impact: tooling (r021 postclose 의 picklog 단계에서 실행, 매매 판정 불변)
    compatibility: backward-compatible (신규 파일)
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import g042_hi52_signal_log as g042

HOLDS = (1, 5, 10, 20)
TICK_SPLIT = 0.0020
RANK = {"live": 3, "late": 2, "replay": 1}
PICK_FIELDS = ["code", "name", "close", "tick_pct", "fallback"]
DAY_COLS = ["date", "gate", "etf", "n_picks", "source", "recorded", "logged_at", "picks"]
EVAL_COLS = ["date", "code", "name", "gate", "recorded", "close", "tick_pct", "tick_group", "fallback", "entry_date", "entry_px"] + [
    f"{k}{h}{s}" for h in HOLDS for k, s in (("exit", "_date"), ("net", "_pct"), ("univ", "_pct"), ("excess", "_pct"),
                                              ("status", ""))]


def out_dir() -> Path:
    return g042.runtime_root() / "data" / "pick_forward"


def tick_pct(close: float) -> float | None:
    return float(g042.tick(close) / close) if close and np.isfinite(close) and close > 0 else None


def tick_group(tp: str) -> str:
    try:
        return ">=0.20%" if float(tp) >= TICK_SPLIT else "<0.20%"
    except ValueError:
        return "unknown"


def read_picks_file(d: str) -> tuple[list[tuple[str, str]], Path] | None:
    p = g042.runtime_root() / "data" / d / f"_{d}_picks.txt"
    if not p.is_file():
        return None
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if parts and len(parts[0]) == 6 and parts[0].isalnum():
            out.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return out, p


def _day_row(P, ind, d: str, picks: list[dict], source: str, rec: str) -> dict:
    C = P["close"]
    rows = []
    for x in picks:
        c = C.at[d, x["code"]] if x["code"] in C.columns else np.nan
        tp = tick_pct(c)
        rows.append({"code": x["code"], "name": x.get("name", ""), "close": f"{c:.0f}" if np.isfinite(c) else "",
                     "tick_pct": f"{tp:.5f}" if tp is not None else "", "fallback": x.get("fallback", "")})
    return {"date": d, "gate": str(ind["gate"].at[d]), "etf": ind["etf"].at[d], "n_picks": str(len(rows)), "source": source,
            "recorded": rec, "logged_at": g042.now_text(), "picks": json.dumps(rows, ensure_ascii=False)}


def log_rows(new: list[dict], log=print) -> int:
    days_p = out_dir() / "days.csv"
    days = g042._read_csv(days_p, DAY_COLS)
    have = dict(zip(days["date"], days["recorded"]))
    keep = []
    for r in new:
        if RANK.get(have.get(r["date"], ""), 0) > RANK[r["recorded"]]:
            log(f"[g047] {r['date']}: already logged as {have[r['date']]} - kept")
            continue
        keep.append(r)
    if not keep:
        return 0
    done = {r["date"] for r in keep}
    days = pd.concat([days[~days["date"].isin(done)], pd.DataFrame(keep, columns=DAY_COLS)]).sort_values("date")
    g042._write_csv(days, days_p)  # the commit (one atomic replace)
    return len(keep)


def log_date(P, ind, d: str, log=print, now: datetime | None = None) -> int:
    got = read_picks_file(d)
    if got is None:
        log(f"[g047] {d}: no picks file (<runtime>/data/{d}/_{d}_picks.txt) - run the scan step (g002) first")
        return 2
    picks, path = got
    rec = g042.classify(d, P["close"].index[-1], now)
    row = _day_row(P, ind, d, [{"code": c, "name": n} for c, n in picks], path.name, rec)
    if log_rows([row], log):
        log(f"[g047] {d} gate={row['gate']} picks={row['n_picks']} ({rec})")
    return 0


def import_replay(P, ind, rows_file: str, log=print) -> int:
    p = Path(rows_file)
    if not p.is_file():
        p = g042.runtime_root() / "research" / "10_selection" / rows_file
    if not p.is_file():
        log(f"[g047] replay rows not found: {rows_file}")
        return 2
    by: dict[str, list[dict]] = {}
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("picked"):
                by.setdefault(r["date"], []).append({"code": r["code"], "name": r.get("name", ""),
                                                     "fallback": "1" if r.get("fallback") else "0"})
    cal = set(P["close"].index)
    rows = [_day_row(P, ind, d, picks, p.name, "replay") for d, picks in sorted(by.items()) if d in cal]
    n = log_rows(rows, log)
    log(f"[g047] replay {p.name}: {n} of {len(by)} dates logged")
    return 0


def evaluate(P, ind, log=print) -> None:
    days = g042._read_csv(out_dir() / "days.csv", DAY_COLS)
    sc = g042.Scorer(P, ind)
    out, skipped = [], 0
    for d in days.itertuples(index=False):
        for x in json.loads(d.picks or "[]"):
            if d.date not in sc.pos or x["code"] not in sc.col:
                skipped += 1
                continue
            i, j = sc.pos[d.date], sc.col[x["code"]]
            row = {"date": d.date, "code": x["code"], "name": x.get("name", ""), "gate": d.gate, "recorded": d.recorded,
                   "close": x.get("close", ""), "tick_pct": x.get("tick_pct", ""), "tick_group": tick_group(x.get("tick_pct", "")),
                   "fallback": x.get("fallback", ""), "entry_date": "", "entry_px": ""}
            for h in HOLDS:
                t = sc.trade(i, h, j)
                b = sc.universe(i, h)
                net = t.get("net")
                row["entry_date"] = row["entry_date"] or t.get("entry_date", "")
                row["entry_px"] = row["entry_px"] or (f"{t['entry_px']:.0f}" if t.get("entry_px") else "")
                row.update({f"exit{h}_date": t.get("exit_date", ""), f"net{h}_pct": f"{net:.3f}" if net is not None else "",
                            f"univ{h}_pct": f"{b:.3f}" if np.isfinite(b) else "",
                            f"excess{h}_pct": f"{net - b:.3f}" if net is not None and np.isfinite(b) else "",
                            f"status{h}": t["status"]})
            out.append(row)
    ev = pd.DataFrame(out, columns=EVAL_COLS).sort_values(["date", "code"])
    g042._write_csv(ev, out_dir() / "evaluation.csv")
    if skipped:
        log(f"[g047] evaluate: {skipped} picks outside the loaded data")
    write_report(ev, sc.cal[-1], days, log)


def write_report(ev: pd.DataFrame, last: str, days: pd.DataFrame, log=print) -> None:
    lines = [f"# g002 pick forward log - report ({g042.now_text()}, data up to {last})", "",
             "Planned trade: buy the next open + 1 tick, sell the close of the k-th holding day (k = 1: same day); net % after "
             "commission and the sell tax of the exit date. Benchmark = the g042 universe of that day traded by the same rules; "
             "excess = net - benchmark. Only `live` rows are a forward test; `replay` = g043 point-in-time replay picks.",
             "Questions (selection_filter_redesign_20261119.md ch.11): (1) gate UP + hold 5/10/20 positive and above the universe? "
             "(2) tick >= 0.20% worse than < 0.20%?", ""]
    if len(days):
        dl = days[days["recorded"] == "live"]
        lines += [f"Logged dates: {len(days)} (live {len(dl)}, replay {int((days['recorded'] == 'replay').sum())}, last "
                  f"{days['date'].max()}). Gate UP on {int((days['gate'] == 'UP').sum())} of them; last gate "
                  f"{days.sort_values('date')['gate'].iloc[-1]}.", ""]
    if not len(ev):
        lines += ["No picks yet."]
    else:
        for h in HOLDS:
            lines += [f"## Hold {h} day{'s' if h > 1 else ''}", "",
                      "| recorded | gate | tick | closed | mean net % | mean excess % | win % | open (latest net %) | excluded |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for (rec, gate, tg), g in ev.groupby(["recorded", "gate", "tick_group"]):
                st = g[f"status{h}"]
                c = g[st.isin(["closed", "unresolved"])]                 # [2026-11-20] unresolved = 0 recovery
                net = pd.to_numeric(c[f"net{h}_pct"], errors="coerce")
                exc = pd.to_numeric(c[f"excess{h}_pct"], errors="coerce")
                o = pd.to_numeric(g.loc[st == "open", f"net{h}_pct"], errors="coerce")
                nx = int(st.isin(["no_bar", "limit_up_open"]).sum())
                cl = f"{len(c)} | {net.mean():.2f} | {exc.mean():.2f} | {(net > 0).mean() * 100:.0f}" if len(c) else "0 | - | - | -"
                op = f"{len(o)} ({o.mean():.2f})" if len(o) else "0"
                lines.append(f"| {rec} | {gate} | {tg} | {cl} | {op} | {nx} |")
            lines.append("")
    p = out_dir() / "report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".md.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    log(f"[g047] report -> {p} ({len(ev)} picks scored)")


def status() -> dict:
    days = g042._read_csv(out_dir() / "days.csv", DAY_COLS)
    live = days[days["recorded"] == "live"]
    rp = out_dir() / "report.md"
    return {"dir": str(out_dir()), "days": len(days), "last_date": days["date"].max() if len(days) else None,
            "last_live": live["date"].max() if len(live) else None,
            "last_gate": days.sort_values("date")["gate"].iloc[-1] if len(days) else None,
            "report_at": datetime.fromtimestamp(rp.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%S") if rp.is_file() else None}


def _start_for(dates: list[str]) -> str | None:
    """first date to load: g042.WARMUP_ROWS index-ETF trading days before the oldest date logged now or already logged."""
    need = [d for d in dates if d]
    days = g042._read_csv(out_dir() / "days.csv", DAY_COLS)
    if len(days):
        need.append(days["date"].min())
    if not need:
        return None
    cal = g042._etf_dates()
    k = sum(1 for d in cal if d < min(need))
    return cal[max(0, k - g042.WARMUP_ROWS)] if cal else None


def _replay_first(rows_file: str) -> str | None:
    p = Path(rows_file)
    if not p.is_file():
        p = g042.runtime_root() / "research" / "10_selection" / rows_file
    if not p.is_file():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return min(json.loads(line)["date"] for line in fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="log the picks of this trading date (YYYYMMDD)")
    ap.add_argument("--import-replay", help="log the picks of a g043 replay rows file (recorded=replay)")
    ap.add_argument("--evaluate", action="store_true", help="rewrite evaluation.csv / report.md")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.status:
        print(json.dumps(status(), ensure_ascii=False))
        return 0
    if a.date and datetime.strptime(a.date, "%Y%m%d").weekday() >= 5:
        print(f"[g047] {a.date} is a weekend day - nothing done")
        return 0
    lock = g042.r019.acquire_instance_lock(out_dir() / ".g047.lock")
    if lock is None:
        print("[g047] another g047 run holds the lock - nothing done")
        return 3
    try:
        try:
            P = g042.load_panel(start=_start_for([a.date, _replay_first(a.import_replay) if a.import_replay else None]))
        except g042.InputError as exc:
            print(exc.code if isinstance(exc.code, str) else exc)
            return 2
        ind = g042.indicators(P)
        cal = list(P["close"].index)
        print(f"[g047] data {cal[0]}..{cal[-1]} ({len(cal)} trading days, {P['close'].shape[1]} codes)", flush=True)
        rc = 0
        if a.import_replay:
            rc = import_replay(P, ind, a.import_replay)
        if a.date:
            if a.date not in cal:
                print(f"[g047] {a.date}: no index ETF bars for that weekday (newest {cal[-1]}) - a holiday, or the daily history "
                      f"of {a.date} was not written (g025)")
                rc = 4
            else:
                rc = log_date(P, ind, a.date) or rc
        if a.evaluate or a.import_replay or not a.date:
            evaluate(P, ind)
        return rc
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

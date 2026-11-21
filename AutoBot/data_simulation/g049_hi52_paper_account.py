# -*- coding: utf-8 -*-
"""g049_hi52_paper_account.py - forward PAPER ACCOUNT of the HI52 swing rules (no orders, no KIS calls) + report.

Why: g042 scores every HI52 signal; a real account holds at most 10 positions, buys in some order, keeps cash and pays the opens it
gets. runtime/research/11_swing_validation showed that one account's result depends a lot on which signals fill (CAGR +4.7..+17.8%
over 2018-2026 by order alone), and Codex (2026-11-20/21) asked for an account-level forward record before any pilot decision.
14_swing_rules (2026-11-21) compared 2 buy gates x 7 sell rules (time, 6 months, chandelier 3xATR, O'Neil -8/+25, MA50, market
exits): none beat these rules, so they are frozen here (RULE_VERSION) and must not be changed while the test runs.

Rules (RULE_VERSION "hi52-t40-v1"):
  buy signal  g042 HI250 at the close of D (universe close >= 2,000, 20-day mean value >= 5 bn, no |move| > 30.5% in 60 rows,
              new 250-row closing high on an up day) while the g042 gate is UP (both index ETFs above a rising MA60)
  order       all candidates of D ranked by sha256(account|D|code) (fixed, neutral); buy at the open of the next row while a slot
              is free: budget per slot = NAV at the close of D / 10, limited by the cash left after that open's sells; integer
              shares at open + 1 tick + commission; skipped when the open is missing or at +29.5% of the prior close, or a code is held
  sell        at the close of the 40th held row (entry row = 1) the position is due; it sells at the next open - 1 tick
              (commission + the sell tax of that day); no open: the order waits; no price at all for 20 rows: written off at 0
  RANDOM      the benchmark account: on every day with n HI52 candidates (gate UP) it ranks the WHOLE universe of D by the same hash
              (account "RANDOM") and takes the first n as its candidates; everything else identical
Each evening (r021 postclose step "hi52acct", after the daily history): every data row after the last processed one is processed in
order: (1) the plan of the previous row is filled at this row's open, (2) NAV at the close, (3) the next plan (due sells, ranked
buy candidates, budget) is FROZEN in plans.csv with recorded = live (newest row, between the KRX close and the next open - g048)
or late. Re-running a processed row changes nothing (state.json last_processed).
Files (<runtime>/data/hi52_paper/, lock .g049.lock): state.json (cash / positions / pending sells per account, last_processed),
plans.csv, ledger.csv (fills / skips with reasons, raw open, fill price, fee, tax), nav.csv (per row: cash, value, NAV, positions,
exposure, buy turnover), acceptance_final.json (once), report.md.
Robustness (Codex round 2): every plan stores its KRX fill day (r014); a fill day missing in the data stops processing until it
arrives (after GAP_ROWS later rows it is taken as an unlisted holiday and noted); the +29.5% check uses the reference close frozen
in the plan; a due position with no close for WRITE_OFF_ROWS rows is written off at the next open (counted at the closes, no
look-ahead); csv rows after state.json's last_processed (a crash) are dropped and re-done; both accounts' plans are classified with
one commit time.
Pre-registered acceptance (REGISTERED 20261121, judged ONCE when the first JUDGE_ROWS processed rows are done, not tuned):
  (a) one-sided 95% lower bound (circular block bootstrap, 60-row blocks, 10,000 draws, seed 42) of the mean daily return
      difference HI52 - RANDOM > 0; (b) the same after charging an extra 0.25% of every buy's value; (c) HI52 account MDD > -45%;
  (d) buy fill rate (filled / attempted with a free slot, ledger rows of the judged window) >= 90%; (e) late plans <= 10% of the rows. PASS needs all five.
  Pilot recommendation = g042 PASS and g049 PASS (written in the report; nothing trades automatically).
Run:
  python g049_hi52_paper_account.py --date 20261123     process up to that row (r021) + report
  python g049_hi52_paper_account.py --report            report only
  python g049_hi52_paper_account.py --status            one JSON line
Exit codes: 0 ok (weekend / nothing new) / 2 no usable input / 3 another g049 is running / 4 D has no index ETF bars.

Update log:
- [2026-11-21] type=feat owner=claude
    summary: 신규 - HI52 스윙 규칙(HI250 + 지수 MA60 상승 게이트, 10슬롯, 40거래일 보유 후 다음 시가 매도, 손절 없음)의 모의계좌 전진
      기록과 같은 규칙의 무작위 비교 계좌, 사전 등록 판정(900거래일). 매일 저녁 다음 시가 주문 목록 동결. 주문/KIS 호출 없음.
    impact: tooling (새 출력 폴더 <runtime>/data/hi52_paper). 매매/감시목록 영향 없음
    compatibility: backward-compatible (새 파일)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "live_trading"))
import g042_hi52_signal_log as g042  # noqa: E402
import g048_gh52_forward_log as g048  # noqa: E402
import r019_platform as r019  # noqa: E402

RULE_VERSION = "hi52-t40-v1"
REGISTERED = "20261121"
CAPITAL = 10_000_000
SLOTS, HOLD, WRITE_OFF_ROWS, GAP_ROWS = 10, 40, 20, 5
FEE = g042.FEE
ACCOUNTS = ("HI52", "RANDOM")
JUDGE_ROWS, BLOCK, BOOT_N, EXTRA_BUY_COST = 900, 60, 10000, 0.0025
MAX_MDD, MIN_FILL, MAX_LATE = -45.0, 0.90, 0.10
WARMUP_ROWS = 320
PLAN_COLS = ["date", "account", "recorded", "rule", "gate", "fill_date", "nav_close", "budget_slot", "sells", "buys", "n_candidates", "logged_at"]
LEDGER_COLS = ["date", "account", "side", "code", "shares", "open", "price", "fee", "tax", "amount", "result", "reason"]
NAV_COLS = ["date", "account", "cash", "value", "nav", "positions", "exposure", "buy_value"]


def out_dir() -> Path:
    return g042.runtime_root() / "data" / "hi52_paper"


def rank_key(account: str, d: str, code: str) -> str:
    return hashlib.sha256(f"{account}|{d}|{code}".encode()).hexdigest()


def _append(rows: list[dict], cols: list[str], p: Path) -> None:
    if not rows:
        return
    old = g042._read_csv(p, cols)
    g042._write_csv(pd.concat([old, pd.DataFrame(rows, columns=cols).astype(str)]), p)


def load_state() -> dict:
    p = out_dir() / "state.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"rule": RULE_VERSION, "registered": REGISTERED, "last_processed": None,
            "accounts": {a: {"cash": float(CAPITAL), "positions": {}} for a in ACCOUNTS}}


def save_state(st: dict) -> None:
    p = out_dir() / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def candidates(ind, d: str) -> dict[str, list[str]]:
    """ranked buy candidates of D per account (empty when the gate is not UP)."""
    if str(ind["gate"].at[d]) != "UP":
        return {a: [] for a in ACCOUNTS}
    row = ind["sig250"].loc[d]
    hits = list(row[row].index)
    univ = ind["univ"].loc[d]
    pool = list(univ[univ].index)
    out = {"HI52": sorted(hits, key=lambda c: rank_key("HI52", d, c))}
    out["RANDOM"] = sorted(pool, key=lambda c: rank_key("RANDOM", d, c))[:len(hits)]
    return out


def _truncate_uncommitted(last: str | None) -> None:
    """rows written after the last committed row (a crash between the csv writes and state.json) are dropped."""
    for name, cols in (("plans.csv", PLAN_COLS), ("ledger.csv", LEDGER_COLS), ("nav.csv", NAV_COLS)):
        p = out_dir() / name
        if not p.is_file():
            continue
        df = g042._read_csv(p, cols)
        keep = df[df["date"] <= last] if last else df.iloc[0:0]
        if len(keep) != len(df):
            g042._write_csv(keep, p)


def process(P, ind, upto: str, now: datetime | None = None, log=print) -> int:
    """process every data row after last_processed up to `upto` in order; returns the number of rows processed."""
    st = load_state()
    _truncate_uncommitted(st["last_processed"])
    cal = list(P["close"].index)
    newest = cal[-1]
    O, C = P["open"], P["close"]
    Cff = C.ffill()
    last = st["last_processed"]
    todo = [d for d in cal if d > REGISTERED and d <= upto and (last is None or d > last)]
    if not todo:
        return 0
    plans_p, ledger_p, nav_p = out_dir() / "plans.csv", out_dir() / "ledger.csv", out_dir() / "nav.csv"
    plans = g042._read_csv(plans_p, PLAN_COLS)
    new_plans, new_ledger, new_nav = [], [], []
    done = 0

    def px_at(frame, d, c):
        return float(frame.at[d, c]) if c in frame.columns and np.isfinite(frame.at[d, c]) else np.nan

    for d in todo:
        i = cal.index(d)
        prevd = cal[i - 1]
        prev_plans = {r["account"]: r for r in new_plans if r["date"] == prevd} or \
            {r["account"]: r for r in plans[plans["date"] == prevd].to_dict("records")}
        note = ""
        if prev_plans:
            fill = next(iter(prev_plans.values()))["fill_date"]
            if fill and fill < d:                                    # the planned KRX day is missing in the data
                if sum(1 for x in cal if x > fill) < GAP_ROWS:
                    log(f"[g049] {d}: the plan of {prevd} fills on {fill}, which is not in the data yet - waiting (data gap)")
                    break
                note = f"planned {fill} missing in the data ({GAP_ROWS}+ later rows) - filled at {d}"
        for a in ACCOUNTS:
            acc = st["accounts"][a]
            pos = acc["positions"]
            plan = prev_plans.get(a)
            sells = json.loads(plan["sells"]) if plan else []
            buys = json.loads(plan["buys"]) if plan else []
            budget = float(plan["budget_slot"]) if plan else 0.0
            # 1) open of d: sells first, then buys
            for c in sells:
                p = pos.get(c)
                if p is None:
                    continue
                o = px_at(O, d, c)
                if not np.isfinite(o):
                    if p.get("noprice", 0) >= WRITE_OFF_ROWS:
                        new_ledger.append({"date": d, "account": a, "side": "sell", "code": c, "shares": p["shares"], "open": "",
                                           "price": 0, "fee": 0, "tax": 0, "amount": 0, "result": "written_off",
                                           "reason": f"no price for {p['noprice']} rows"})
                        del pos[c]
                    else:
                        p["carry"] = True
                    continue
                px = float(o - g042.tick(o))
                tax = g042.sell_tax(d)
                amt = p["shares"] * px
                fee = amt * FEE
                acc["cash"] += amt - fee - amt * tax
                new_ledger.append({"date": d, "account": a, "side": "sell", "code": c, "shares": p["shares"], "open": f"{o:.0f}",
                                   "price": f"{px:.0f}", "fee": f"{fee:.0f}", "tax": f"{amt * tax:.0f}",
                                   "amount": f"{amt - fee - amt * tax:.0f}", "result": "filled",
                                   "reason": f"held {p['held']} rows" + (f"; {note}" if note else "")})
                del pos[c]
            buy_value = 0.0
            for c, ref in buys:
                if len(pos) >= SLOTS:
                    break
                if c in pos:
                    continue
                o = px_at(O, d, c)
                reason = ""
                if not np.isfinite(o):
                    reason = "no open"
                elif ref and ref != "nan" and o >= float(ref) * g042.LIMIT_UP_OPEN:
                    reason = "open at +29.5%"
                if not reason:
                    px = float(o + g042.tick(o))
                    sh = int(min(budget, acc["cash"]) // (px * (1 + FEE)))
                    if sh < 1:
                        reason = "cash"
                if reason:
                    new_ledger.append({"date": d, "account": a, "side": "buy", "code": c, "shares": 0,
                                       "open": f"{o:.0f}" if np.isfinite(o) else "", "price": "", "fee": "", "tax": "", "amount": "",
                                       "result": "skipped", "reason": reason})
                    continue
                amt = sh * px
                fee = amt * FEE
                acc["cash"] -= amt + fee
                buy_value += amt
                pos[c] = {"shares": sh, "entry": d, "price": px, "held": 0, "noprice": 0}
                new_ledger.append({"date": d, "account": a, "side": "buy", "code": c, "shares": sh, "open": f"{o:.0f}",
                                   "price": f"{px:.0f}", "fee": f"{fee:.0f}", "tax": 0, "amount": f"{-(amt + fee):.0f}",
                                   "result": "filled", "reason": note})
            # 2) close of d: NAV, held rows, rows without a price, due sells
            value = 0.0
            due = []
            for c, p in pos.items():
                p["held"] += 1
                cl = px_at(C, d, c)
                p["noprice"] = 0 if np.isfinite(cl) else p.get("noprice", 0) + 1
                mark = px_at(Cff, d, c)
                value += p["shares"] * (mark if np.isfinite(mark) else p["price"])
                if p["held"] >= HOLD or p.get("carry"):
                    due.append(c)
            nav = acc["cash"] + value
            new_nav.append({"date": d, "account": a, "cash": f"{acc['cash']:.0f}", "value": f"{value:.0f}", "nav": f"{nav:.0f}",
                            "positions": len(pos), "exposure": f"{value / nav:.4f}" if nav > 0 else "0",
                            "buy_value": f"{buy_value:.0f}"})
            # 3) the plan for the next open (frozen: candidates, their reference close, budget, fill day)
            cand = [[c, f"{px_at(Cff, d, c):.0f}"] for c in candidates(ind, d)[a]]
            new_plans.append({"date": d, "account": a, "recorded": "", "rule": RULE_VERSION, "gate": str(ind["gate"].at[d]),
                              "fill_date": g048.next_krx_day(d), "nav_close": f"{nav:.0f}", "budget_slot": f"{nav / SLOTS:.0f}",
                              "sells": json.dumps(due), "buys": json.dumps(cand), "n_candidates": len(cand), "logged_at": ""})
        st["last_processed"] = d
        done += 1
        log(f"[g049] {d}: " + ", ".join(f"{a} NAV {float(new_nav[-len(ACCOUNTS) + k]['nav']):,.0f} pos "
                                        f"{len(st['accounts'][a]['positions'])}" for k, a in enumerate(ACCOUNTS))
            + f" (gate {ind['gate'].at[d]})")
    if not done:
        return 0
    commit = now or datetime.now()                                    # one commit time for both accounts
    for r in new_plans:
        r["recorded"] = g048.classify(r["date"], newest, commit)
        r["logged_at"] = commit.strftime("%Y-%m-%dT%H:%M:%S")
    _append(new_plans, PLAN_COLS, plans_p)
    _append(new_ledger, LEDGER_COLS, ledger_p)
    _append(new_nav, NAV_COLS, nav_p)
    save_state(st)                                                     # the commit: rows after last_processed are dropped on the next run
    return done


def _mdd(nav: np.ndarray) -> float:
    eq = np.r_[CAPITAL, nav]
    return float((eq / np.maximum.accumulate(eq) - 1).min() * 100)


def _lb(x: np.ndarray, block: int = BLOCK, n: int = BOOT_N, seed: int = 42) -> float:
    T = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    idx = ((rng.integers(0, T, (n, nb))[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]) % T
    return float(np.percentile(x[idx].mean(axis=1), 5))


def acceptance(final_path: Path | None = None) -> dict:
    final_path = final_path or (out_dir() / "acceptance_final.json")
    if final_path.is_file():
        with contextlib.suppress(Exception):
            out = json.loads(final_path.read_text(encoding="utf-8"))
            out["note"] = f"final verdict recorded {out.get('decided_at', '')} - not recomputed"
            return out
    last = load_state()["last_processed"] or ""                      # [Codex round 3] committed rows only
    nav = g042._read_csv(out_dir() / "nav.csv", NAV_COLS)
    nav = nav[nav["date"] <= last]
    plans = g042._read_csv(out_dir() / "plans.csv", PLAN_COLS)
    plans = plans[plans["date"] <= last]
    rows = sorted(set(nav["date"]))
    out = {"registered": REGISTERED, "rule": RULE_VERSION, "rows": len(rows), "judge_rows": JUDGE_ROWS}
    if len(rows) < JUDGE_ROWS:
        out["verdict"] = "collecting"
        return out
    rows = rows[:JUDGE_ROWS]
    n = {a: nav[(nav["account"] == a) & nav["date"].isin(rows)].set_index("date").loc[rows] for a in ACCOUNTS}
    eq = {a: n[a]["nav"].astype(float).to_numpy() for a in ACCOUNTS}
    r = {a: np.diff(np.r_[CAPITAL, eq[a]]) / np.r_[CAPITAL, eq[a]][:-1] for a in ACCOUNTS}
    bv = n["HI52"]["buy_value"].astype(float).to_numpy() / np.r_[CAPITAL, eq["HI52"]][:-1]
    diff = r["HI52"] - r["RANDOM"]
    pl = plans[plans["date"].isin(rows)]
    led = g042._read_csv(out_dir() / "ledger.csv", LEDGER_COLS)
    led = led[led["date"] <= last]
    bl = led[led["date"].isin(rows) & (led["account"] == "HI52") & (led["side"] == "buy")]
    res = {"diff_lb": _lb(diff), "diff_lb_cost": _lb(diff - EXTRA_BUY_COST * bv), "mdd": _mdd(eq["HI52"]),
           "fill_rate": float((bl["result"] == "filled").mean()) if len(bl) else 1.0,
           "late_share": float((pl["recorded"] != "live").mean()) if len(pl) else 0.0,
           "diff_mean_pct_yr": float(diff.mean() * 250 * 100)}
    ok = (res["diff_lb"] > 0 and res["diff_lb_cost"] > 0 and res["mdd"] > MAX_MDD and res["fill_rate"] >= MIN_FILL
          and res["late_share"] <= MAX_LATE)
    out.update({"result": res, "verdict": "PASS" if ok else "FAIL", "decided_at": g042.now_text(), "last_row": rows[-1]})
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, final_path)
    return out


def write_report(log=print) -> None:
    st = load_state()
    acc = acceptance()
    nav = g042._read_csv(out_dir() / "nav.csv", NAV_COLS)
    plans = g042._read_csv(out_dir() / "plans.csv", PLAN_COLS)
    ledger = g042._read_csv(out_dir() / "ledger.csv", LEDGER_COLS)
    nm = g042.names()
    lines = [f"# HI52 paper account ({g042.now_text()}, processed up to {st['last_processed'] or '-'}, rule {RULE_VERSION})", "",
             "Rules: HI250 signal while the g042 gate is UP, candidates ranked by a fixed hash, buy at the next open (1/10 of the prior "
             "close NAV per slot, 10 slots, integer shares), sell at the open after the 40th held row; no stop. RANDOM = the same "
             "account drawing the same number of codes from the universe. No orders are sent.", "",
             "## Pre-registered acceptance (judged once)", "",
             f"- registered {REGISTERED}: rows {acc.get('rows', '-')} / {JUDGE_ROWS}; verdict **{acc['verdict']}** {acc.get('note', '')}",
             f"- pass = all of: LB95 of mean daily return HI52 - RANDOM > 0 ({BLOCK}-row blocks), the same with +{EXTRA_BUY_COST:.2%} per "
             f"buy, HI52 MDD > {MAX_MDD:.0f}%, buy fill rate >= {MIN_FILL:.0%}, late plans <= {MAX_LATE:.0%}",
             "- pilot recommendation only when g042 AND g049 pass (nothing trades automatically)."]
    if "result" in acc:
        lines.append("- result: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in acc["result"].items()))
    lines.append("")
    for a in ACCOUNTS:
        n = nav[nav["account"] == a]
        s = st["accounts"][a]
        if len(n):
            e = n["nav"].astype(float).to_numpy()
            lb = ledger[(ledger["account"] == a) & (ledger["side"] == "buy")]
            lines.append(f"- {a}: NAV {e[-1]:,.0f} ({(e[-1] / CAPITAL - 1) * 100:+.2f}%), MDD {_mdd(e):.2f}%, positions "
                         f"{len(s['positions'])}, buy fills {int((lb['result'] == 'filled').sum())} / attempts {len(lb)}")
        else:
            lines.append(f"- {a}: not started (first row after {REGISTERED})")
    lines.append("")
    if st["last_processed"]:
        lines += ["## Orders for the next open (frozen plan)", ""]
        for a in ACCOUNTS:
            p = plans[(plans["date"] == st["last_processed"]) & (plans["account"] == a)]
            if len(p):
                p = p.iloc[0]
                sells = json.loads(p["sells"]); buys = json.loads(p["buys"])
                free = SLOTS - len(st["accounts"][a]["positions"]) + len(sells)
                sl = ", ".join(c + " " + nm.get(c, "") for c in sells) or "-"
                bl = ", ".join(c + " " + nm.get(c, "") for c, _ in buys[:15]) or "-"
                lines.append(f"- {a} ({p['recorded']}, gate {p['gate']}, fills {p['fill_date']}): sell [{sl}]; buy up to {free} of [{bl}] "
                             f"(budget {float(p['budget_slot']):,.0f} each)")
        lines += ["", "## Positions", "", "| account | code | name | shares | entry | price | held rows |", "|---|---|---|---|---|---|---|"]
        for a in ACCOUNTS:
            for c, p in sorted(st["accounts"][a]["positions"].items()):
                lines.append(f"| {a} | {c} | {nm.get(c, '')} | {p['shares']} | {p['entry']} | {p['price']:,.0f} | {p['held']} |")
        lines += ["", "## Recent ledger", "", "| date | account | side | code | shares | price | result | reason |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in ledger.tail(30).itertuples(index=False):
            lines.append(f"| {r.date} | {r.account} | {r.side} | {r.code} | {r.shares} | {r.price} | {r.result} | {r.reason} |")
    p = out_dir() / "report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".md.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    log(f"[g049] report -> {p} (acceptance {acc['verdict']})")


def status() -> dict:
    st = load_state()
    return {"dir": str(out_dir()), "rule": RULE_VERSION, "last_processed": st["last_processed"],
            "positions": {a: len(st["accounts"][a]["positions"]) for a in ACCOUNTS}, "acceptance_registered": REGISTERED}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="process every data row up to this date (YYYYMMDD); default the newest")
    ap.add_argument("--report", action="store_true", help="report only")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.status:
        print(json.dumps(status(), ensure_ascii=False))
        return 0
    if a.date and datetime.strptime(a.date, "%Y%m%d").weekday() >= 5:
        print(f"[g049] {a.date} is a weekend day - nothing done")
        return 0
    lock = r019.acquire_instance_lock(out_dir() / ".g049.lock")
    if lock is None:
        print("[g049] another g049 run holds the lock - nothing done")
        return 3
    try:
        rc = 0
        if not a.report:
            cal = g042._etf_dates()
            k = sum(1 for d in cal if d <= REGISTERED)
            try:
                P = g042.load_panel(start=cal[max(0, k - WARMUP_ROWS)] if cal else None)
            except g042.InputError as exc:
                print(exc.code if isinstance(exc.code, str) else exc)
                return 2
            ind = g042.indicators(P)
            dates = list(P["close"].index)
            d = a.date or dates[-1]
            if d not in dates:
                print(f"[g049] {d}: no index ETF bars for that weekday (newest {dates[-1]})")
                rc = 4
            else:
                n = process(P, ind, d)
                print(f"[g049] processed {n} rows (rule {RULE_VERSION})" if n else f"[g049] nothing new up to {d} (rows after "
                      f"{REGISTERED} only)")
        write_report()
        return rc
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

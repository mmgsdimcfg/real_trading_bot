# -*- coding: utf-8 -*-
"""g050_paper_session_report.py - session report of the Edge AI paper trades (r022 edge_only_<date>.jsonl). No orders, no KIS calls.

Why (2026-11-25, trade_modes_plan_20261125.md section 8 C, Codex review): the Edge AI model is trained on regular-session data up to
15:15 (g013). Since 2026-11-25 the paper runner keeps NXT holdings until 19:59:50 and also buys in the NXT pre / after sessions, so
their results must be read apart from the regular ones before anything is concluded.

What it reads: every ENTRY / EXIT pair (in order per code) of runtime/data/live_runtime/edge_only_<date>.jsonl in the date range.
Older records: no "session" = regular, no "nxt" = false. Open positions (no EXIT) are counted, not scored.

Tables (net % after commission and sell tax as recorded by r022):
  1. by entry session (regular / nxt_pre / nxt_after)        2. by holding kind (NXT holding / regular-only)
  3. by exit reason                                           4. estimated fills (closed with the last price seen)
  5. the 15:15 control: trades bought before 15:15 and still held after it, re-priced at 15:15 from the day's KRX tick file (last
     regular print <= 15:15:00 - 1 tick, same costs) next to their real result. Trades without a print are excluded and counted.
A cell with fewer than MIN_N trades is marked "판단 보류" (not enough to judge).

Run:
  python g050_paper_session_report.py                      all days
  python g050_paper_session_report.py --from 20261125 --to 20261231
Output: <runtime>/research/paper_session/paper_session_<from>_<to>.md (+ the same text on stdout).

Update log:
- [2026-11-25] type=feat owner=claude
    summary: 신규 - 가상 매매 세션별 성과 보고서(진입 세션 / NXT 보유 / 청산 사유 / 추정 체결 / 15:15 대조군), 표본 30건 미만 판단 보류.
    impact: research only
    compatibility: new file
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live_trading"))

MIN_N = 30
CONTROL_T = "151500"


@dataclass
class Trade:
    day: str
    code: str
    group: str
    session: str
    nxt: bool
    entry: float
    entry_ts: datetime
    exit_ts: datetime
    reason: str
    net: float
    held: float
    estimated: bool


INCOMPLETE = {"n": 0}     # [Codex review] EXITs without the required values (not scored, counted in the report)


def load_trades(live_dir: Path, d_from: str, d_to: str) -> tuple[list[Trade], int]:
    trades: list[Trade] = []
    open_n = 0
    INCOMPLETE["n"] = 0
    for path in sorted(Path(live_dir).glob("edge_only_*.jsonl")):
        day = path.stem[len("edge_only_"):]
        if not (len(day) == 8 and day.isdigit()) or day < d_from or day > d_to:
            continue
        open_: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            code = str(rec.get("code") or "")
            if rec.get("event") == "ENTRY":
                open_[code] = rec
            elif rec.get("event") == "EXIT" and code in open_:
                e = open_.pop(code)
                try:
                    if rec.get("net_pct") is None or not rec.get("reason"):
                        raise ValueError("missing net_pct / reason")
                    trades.append(Trade(
                        day=day, code=code, group=e.get("group") or "watch", session=e.get("session") or "regular",
                        nxt=e.get("nxt") is True, entry=float(e["entry"]), entry_ts=datetime.fromisoformat(e["ts"]),
                        exit_ts=datetime.fromisoformat(rec["ts"]), reason=str(rec["reason"]),
                        net=float(rec["net_pct"]), held=float(rec.get("held_min") or 0.0),
                        estimated=rec.get("estimated") is True))
                except (KeyError, ValueError, TypeError):
                    INCOMPLETE["n"] += 1
                    continue
        open_n += len(open_)
    return trades, open_n


def stats(xs: list[Trade]) -> dict:
    n = len(xs)
    if not n:
        return {"n": 0}
    nets = [t.net for t in xs]
    return {"n": n, "win": sum(1 for x in nets if x > 0) / n * 100, "mean": sum(nets) / n, "sum": sum(nets),
            "held": sum(t.held for t in xs) / n, "est": sum(1 for t in xs if t.estimated) / n * 100}


def row(label: str, s: dict) -> str:
    if not s["n"]:
        return f"| {label} | 0 | - | - | - | - | - | - |"
    verdict = "판단 보류" if s["n"] < MIN_N else ""
    return (f"| {label} | {s['n']} | {s['win']:.1f}% | {s['mean']:+.2f}% | {s['sum']:+.2f}% | {s['held']:.0f} | {s['est']:.0f}% "
            f"| {verdict} |")


HEAD = "| 구분 | 건수 | 승률 | 평균 순손익 | 합계 | 평균 보유(분) | 추정 체결 | 비고 |\n|---|---|---|---|---|---|---|---|"


def control_1515(trades: list[Trade], ticks_root: Path) -> tuple[list[tuple[Trade, float]], int]:
    """(trade, net if closed at 15:15) for trades bought before 15:15 and exited after it; and how many had no regular print
    between the entry and 15:15 (excluded - a print from before the buy is not a price it could have sold at)"""
    from r022_edge_only import last_print_until, net_pct, tick_down
    out, missing = [], 0
    for t in trades:
        if not (t.entry_ts.strftime("%H%M%S") < CONTROL_T < t.exit_ts.strftime("%H%M%S") and t.entry_ts.date() == t.exit_ts.date()):
            continue
        lp = last_print_until(Path(ticks_root) / t.day / f"{t.code}.csv", t.day, CONTROL_T, ("regular",),
                              since_hhmmss=(t.entry_ts + timedelta(seconds=1)).strftime("%H%M%S"))   # after the buy, not its own second
        if not lp:
            missing += 1
            continue
        out.append((t, net_pct(t.entry, tick_down(lp[0]))))
    return out, missing


def report(trades: list[Trade], open_n: int, ticks_root: Path, d_from: str, d_to: str) -> str:
    days = sorted({t.day for t in trades})
    lines = [f"# 가상 매매 세션별 성과 ({d_from} ~ {d_to})", "",
             f"- 청산된 거래 {len(trades)}건, 거래일 {len(days)}일 ({days[0] if days else '-'} ~ {days[-1] if days else '-'}), "
             f"아직 열린 보유 {open_n}건, 값이 빠진 청산 기록 {INCOMPLETE['n']}건 (둘 다 집계 제외)",
             f"- 순손익 = 수수료·매도세 포함(r022 기록). 칸의 건수가 {MIN_N}건 미만이면 판단 보류.",
             "- 모델은 15:15 까지의 정규장 데이터로 학습됨 - NXT 세션·NXT 보유는 학습 범위 밖", ""]
    lines += ["## 1. 진입 세션별", "", HEAD]
    for s in ("regular", "nxt_pre", "nxt_after"):
        lines.append(row({"regular": "정규장", "nxt_pre": "NXT 장전", "nxt_after": "NXT 장후"}[s], stats([t for t in trades if t.session == s])))
    lines += ["", "## 2. 보유 종류별", "", HEAD,
              row("NXT 보유 (19:59:50 청산)", stats([t for t in trades if t.nxt])),
              row("정규장 보유 (15:20 / 예전 15:15)", stats([t for t in trades if not t.nxt]))]
    lines += ["", "## 3. 청산 사유별", "", HEAD]
    for r in sorted({t.reason for t in trades}):
        lines.append(row(r, stats([t for t in trades if t.reason == r])))
    lines += ["", "## 4. 추정 체결", "", HEAD,
              row("추정 체결 (마지막 가격으로 계산)", stats([t for t in trades if t.estimated])),
              row("실제 체결가 기준", stats([t for t in trades if not t.estimated]))]
    # [Codex review] the cross table the plan asked for: entry session x holding kind x exit reason (cells with trades only)
    lines += ["", "## 5. 진입 세션 x 보유 종류 x 청산 사유", "", HEAD]
    ses_ko = {"regular": "정규장", "nxt_pre": "NXT 장전", "nxt_after": "NXT 장후"}
    for s in ("regular", "nxt_pre", "nxt_after"):
        for nx in (True, False):
            for r in sorted({t.reason for t in trades}):
                cell = [t for t in trades if t.session == s and t.nxt == nx and t.reason == r]
                if cell:
                    lines.append(row(f"{ses_ko[s]} / {'NXT 보유' if nx else '정규장 보유'} / {r}", stats(cell)))
    # NXT holdings: closed before 15:15 (inside the model's horizon) vs after it (the after-market the model never saw)
    nxt_h = [t for t in trades if t.nxt]
    lines += ["", "## 6. NXT 보유: 15:15 이전 청산 vs 이후 청산", "", HEAD,
              row("15:15 이전 청산", stats([t for t in nxt_h if t.exit_ts.strftime("%H%M%S") <= CONTROL_T])),
              row("15:15 이후 청산", stats([t for t in nxt_h if t.exit_ts.strftime("%H%M%S") > CONTROL_T]))]
    ctl, missing = control_1515(trades, ticks_root)
    lines += ["", "## 7. 15:15 대조군 (15:15 전에 사서 15:15 넘게 보유한 거래, 같은 거래를 15:15 에 팔았다면)", "",
              f"- 비교 가능 {len(ctl)}건, 매수 뒤 15:15 까지 체결 기록이 없어 제외 {missing}건",
              "- 15:15 가격 = 매수 뒤 15:15 까지의 마지막 정규장 체결 - 1틱, 비용은 r022 와 같음"]
    for label, part in (("NXT 보유", [(t, a) for t, a in ctl if t.nxt]), ("정규장 보유", [(t, a) for t, a in ctl if not t.nxt])):
        if not part:
            lines += ["", f"- {label}: 비교 가능한 거래 없음"]
            continue
        n = len(part)
        real = [t.net for t, _ in part]
        alt = [a for _, a in part]
        better = sum(1 for (t, a) in part if t.net > a)
        mark = "판단 보류" if n < MIN_N else ""
        lines += ["", f"### {label}", "", "| 구분 | 건수 | 평균 순손익 | 합계 | 비고 |", "|---|---|---|---|---|",
                  f"| 실제 청산 | {n} | {sum(real) / n:+.2f}% | {sum(real):+.2f}% | {mark} |",
                  f"| 15:15 에 청산했다면 | {n} | {sum(alt) / n:+.2f}% | {sum(alt):+.2f}% | {mark} |",
                  "", f"- 15:15 이후 보유가 더 나았던 거래 {better}/{n}건"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="d_from", default="00000000")
    ap.add_argument("--to", dest="d_to", default="99999999")
    ap.add_argument("--live-dir", default=None)
    ap.add_argument("--ticks", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import r001_define_config as cfg
    live_dir = Path(a.live_dir) if a.live_dir else Path(cfg.RUNTIME_DATA_DIR) / "live_runtime"   # = r003 LIVE_RUNTIME_DIR
    ticks = Path(a.ticks) if a.ticks else Path(cfg.TICK_DATA_DIR)
    trades, open_n = load_trades(live_dir, a.d_from, a.d_to)
    text = report(trades, open_n, ticks, a.d_from, a.d_to)
    out = Path(a.out) if a.out else Path(cfg.RUNTIME_ROOT) / "research" / "paper_session" / f"paper_session_{a.d_from}_{a.d_to}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    sys.stdout.write(f"\n-> {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Experiment orchestration runner for g003_trade_simulate_by_date.py.

[2026-09-13] Runs 5 variants (baseline, A, B, C, D - see data_simulation/
experiment_results_A_D.md for the full write-up) of g003 as subprocesses,
once per (date, variant) pair, over a fixed RANKED/SELECTED top-8-by-score
subset of stocks for 5 recent trading days (read from each date's
_<date>_ranked.txt at the time this script was written - see DATASET below).
Parses per-run trade stats from g003's stdout and aggregates them into a
markdown report.

This script does NOT modify r001/r002/r003/g003 behavior itself - it only
sets environment variables that the (already-added) opt-in overrides in
g003/r002 read. Every variant's env vars are additive on top of an
unmodified baseline subprocess environment; the baseline run itself sets no
experiment env vars at all, so it reproduces current/live default behavior.

Usage:
    python3 g006_experiment_runner.py [--timeout SECONDS]

Update log:
- [2026-09-13] type=feat owner=claude
    summary: initial version - runs baseline/A/B/C/D across 5 dates x 8 codes
      (40 symbol-days), parses trade stats + (for A) fill/cancel/slippage
      stats, writes experiment_results_A_D.md.
    impact: sim (analysis tooling only, no effect on live/backtest strategy code)
    compatibility: n/a (new file)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
G003_PATH = SCRIPT_DIR / "g003_trade_simulate_by_date.py"
REPORT_PATH = SCRIPT_DIR / "experiment_results_A_D.md"

# Re-verified directly against data/<date>/_<date>_ranked.txt (top 8 rows by
# the scanner's `score` column, descending) before use - see final report for
# confirmation that no transcription errors were found.
DATASET: dict[str, list[str]] = {
    "20260907": ["001820", "093370", "052690", "388050", "078930", "004310", "064550", "064760"],
    "20260908": ["388050", "053260", "459550", "052690", "183300", "030530", "098120", "024060"],
    "20260909": ["351320", "036540", "053260", "336260", "069540", "030530", "000990", "302430"],
    "20260910": ["003670", "162300", "092870", "036540", "232140", "131970", "452190", "000990"],
    "20260911": ["092870", "437730", "108490", "413630", "198440", "001820", "003350", "417840"],
}

# [2026-09-13] B/C/D removed after the 5-day sweep showed all three underperformed
# baseline (-16.6%/-11.4%/-18.8% PnL - see experiment_results_A_D.md) and their
# override plumbing was reverted from g003/r001/r002 accordingly (user decision:
# keep the buy-condition logic as-is). Only baseline/A remain runnable here - A's
# result was a no-op vs baseline (not harmful, just inconclusive on this sample),
# and its execution-delay-proxy code was kept in g003 as a standalone analysis tool.
VARIANTS: list[tuple[str, dict[str, str]]] = [
    ("baseline", {}),
    ("A", {"R76_SIM_EXEC_MODEL": "passive_limit_chase"}),
]

VARIANT_LABELS = {
    "baseline": "Baseline (변경 없음)",
    "A": "A - 체결 공격성/지연비용 추정 (실행 프록시)",
    "B": "B - 재확인 대기 축소 (CONFIRM_COUNT=1)",
    "C": "C - 개장 가드 완화 (OPENING_GUARD_MINUTES=0)",
    "D": "D - VWAP 필수 게이트화",
}

SELL_LINE_RE = re.compile(r"매도\s*건수\s*:\s*(\d+)건\s*\(익절\s*(\d+)건/\s*손절·기타\s*(\d+)건\)")
WINRATE_RE = re.compile(r"승률\s*:\s*([\d.]+)%")
PNL_RE = re.compile(r"총\s*손익\s*:\s*([+-][\d,]+)\s*KRW")
EXEC_SIM_RE = re.compile(
    r"EXEC_SIM_RESULT code=(\S+) date=(\S+) signal_ts=([\d-]+ [\d:]+) "
    r"signal_price=([\d.]+) status=(FILLED|CANCELLED) fill_price=(\S+) "
    r"delay_s=(\d+) slippage_pct=(\S+)"
)


@dataclass
class RunResult:
    date: str
    variant: str
    codes: list[str]
    ok: bool
    elapsed_s: float
    signals_attempted: int = 0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_krw: float = 0.0
    exec_filled: int = 0
    exec_cancelled: int = 0
    exec_slippage_delayed: list[float] = field(default_factory=list)  # L2/L3 fills only
    stdout_tail: str = ""


def run_one(date: str, codes: list[str], variant: str, env_overrides: dict[str, str], timeout: int) -> RunResult:
    env = dict(os.environ)
    env.pop("R76_SIM_EXEC_MODEL", None)
    env.pop("R76_SIM_BUY_CONSECUTIVE_CONFIRM_COUNT", None)
    env.pop("R76_SIM_OPENING_GUARD_MINUTES", None)
    env.pop("R76_SIM_ENABLE_VWAP_MANDATORY_GATE", None)
    env.update(env_overrides)

    cmd = [
        sys.executable, str(G003_PATH),
        "--date", date,
        "--codes", *codes,
        "--summary",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(SCRIPT_DIR), env=env,
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0
        output = proc.stdout + "\n" + proc.stderr
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - t0
        # [2026-09-13 fix] exc.stdout/stderr can come back as bytes even with
        # text=True when the timeout fires mid-communicate() - normalize before
        # concatenating (previously crashed the whole sweep with
        # "TypeError: can't concat str to bytes" on the very first slow run).
        def _to_text(x) -> str:
            if x is None:
                return ""
            if isinstance(x, bytes):
                return x.decode("utf-8", errors="replace")
            return x
        output = _to_text(exc.stdout) + "\n" + _to_text(exc.stderr)
        ok = False

    result = RunResult(date=date, variant=variant, codes=codes, ok=ok, elapsed_s=elapsed)
    result.signals_attempted = output.count("[BUY SIGNAL]")

    m = SELL_LINE_RE.search(output)
    if m:
        result.trades = int(m.group(1))
        result.wins = int(m.group(2))
        result.losses = int(m.group(3))
    m = WINRATE_RE.search(output)
    if m:
        result.win_rate = float(m.group(1))
    m = PNL_RE.search(output)
    if m:
        result.total_pnl_krw = float(m.group(1).replace(",", ""))

    if variant == "A":
        for em in EXEC_SIM_RE.finditer(output):
            status = em.group(5)
            delay_s = int(em.group(7))
            slippage_raw = em.group(8)
            if status == "FILLED":
                result.exec_filled += 1
                if delay_s in (20, 30) and slippage_raw != "NA":
                    try:
                        result.exec_slippage_delayed.append(float(slippage_raw))
                    except ValueError:
                        pass
            else:
                result.exec_cancelled += 1

    result.stdout_tail = output[-4000:]
    return result


def main() -> int:
    timeout = 1500
    if "--timeout" in sys.argv:
        idx = sys.argv.index("--timeout")
        timeout = int(sys.argv[idx + 1])

    all_results: list[RunResult] = []
    total_runs = len(DATASET) * len(VARIANTS)
    run_no = 0
    for date, codes in DATASET.items():
        for variant, env_overrides in VARIANTS:
            run_no += 1
            print(f"[{run_no}/{total_runs}] Running date={date} variant={variant} codes={codes} ...", flush=True)
            try:
                result = run_one(date, codes, variant, env_overrides, timeout)
            except Exception as exc:  # noqa: BLE001 - [2026-09-13 fix] one bad run
                # must never kill the other 24: a single subprocess/parsing
                # failure previously propagated all the way out of main() and
                # silently aborted the entire sweep after run 2/25.
                elapsed = time.time()
                print(f"    -> FAILED (unexpected exception, skipping) | {type(exc).__name__}: {exc}")
                result = RunResult(date=date, variant=variant, codes=codes, ok=False, elapsed_s=0.0)
                all_results.append(result)
                continue
            status = "OK" if result.ok else "FAILED/TIMEOUT"
            print(
                f"    -> {status} in {result.elapsed_s:.1f}s | signals={result.signals_attempted} "
                f"trades={result.trades} wins={result.wins} losses={result.losses} "
                f"win_rate={result.win_rate:.1f}% pnl={result.total_pnl_krw:+,.0f} KRW"
                + (f" | exec_filled={result.exec_filled} exec_cancelled={result.exec_cancelled}" if variant == "A" else "")
            )
            if not result.ok:
                print("    --- tail of output ---")
                print(result.stdout_tail)
            all_results.append(result)
            write_report(all_results)

    print(f"\nReport written to {REPORT_PATH}")
    return 0


def write_report(all_results: list[RunResult]) -> None:
    by_variant: dict[str, list[RunResult]] = {}
    for r in all_results:
        by_variant.setdefault(r.variant, []).append(r)

    lines: list[str] = []
    lines.append("# Experiment Results A-D (2026-09-13)")
    lines.append("")
    lines.append(
        "Backtest sweep over 5 recent trading days x top-8-ranked-by-score codes each "
        "(40 symbol-days total), comparing baseline g003 behavior against 4 experimental "
        "toggles (A/B/C/D). All runs use `--summary` and identical `--capital` defaults; "
        "only the environment-variable overrides listed per variant differ."
    )
    lines.append("")
    lines.append("## Dataset (re-verified against `_<date>_ranked.txt`, top 8 by `score`)")
    lines.append("")
    for date, codes in DATASET.items():
        lines.append(f"- `{date}`: {' '.join(codes)}")
    lines.append("")

    lines.append("## Summary table (aggregated across all 5 dates / 40 symbol-days)")
    lines.append("")
    lines.append("| Variant | Total signals attempted | Trades taken | Wins | Losses | Win rate | Total PnL (KRW) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    variant_order = ["baseline", "A", "B", "C", "D"]
    agg: dict[str, dict[str, float]] = {}
    for variant in variant_order:
        runs = by_variant.get(variant, [])
        signals = sum(r.signals_attempted for r in runs)
        trades = sum(r.trades for r in runs)
        wins = sum(r.wins for r in runs)
        losses = sum(r.losses for r in runs)
        pnl = sum(r.total_pnl_krw for r in runs)
        win_rate = (wins / trades * 100.0) if trades else 0.0
        agg[variant] = {
            "signals": signals, "trades": trades, "wins": wins, "losses": losses,
            "win_rate": win_rate, "pnl": pnl,
        }
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {signals} | {trades} | {wins} | {losses} | "
            f"{win_rate:.1f}% | {pnl:+,.0f} |"
        )
    lines.append("")

    # Experiment A specific fill/cancel/slippage stats
    a_runs = by_variant.get("A", [])
    a_filled = sum(r.exec_filled for r in a_runs)
    a_cancelled = sum(r.exec_cancelled for r in a_runs)
    a_slippages: list[float] = []
    for r in a_runs:
        a_slippages.extend(r.exec_slippage_delayed)
    a_avg_slip = (sum(a_slippages) / len(a_slippages)) if a_slippages else None

    lines.append("## Experiment A detail: fill vs cancel, slippage")
    lines.append("")
    lines.append(f"- FILLED signals: {a_filled}")
    lines.append(f"- CANCELLED signals (chase exceeded BUY_ORDER_REPRICE_MAX_CHASE_PCT, or data ran out): {a_cancelled}")
    total_a = a_filled + a_cancelled
    fill_rate = (a_filled / total_a * 100.0) if total_a else 0.0
    lines.append(f"- Fill rate: {fill_rate:.1f}%")
    if a_avg_slip is not None:
        lines.append(
            f"- Average slippage_pct among FILLED-with-delay trades (L2/L3 reprice tiers, "
            f"delay_s in {{20,30}}), n={len(a_slippages)}: {a_avg_slip:+.4f}%"
        )
    else:
        lines.append("- No FILLED-with-delay (L2/L3) trades observed - all fills were immediate (L1, delay_s=10).")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")

    def _delta(variant: str, key: str) -> float:
        return agg[variant][key] - agg["baseline"][key]

    lines.append(
        f"- **Baseline**: {agg['baseline']['trades']:.0f} trades, "
        f"{agg['baseline']['win_rate']:.1f}% win rate, {agg['baseline']['pnl']:+,.0f} KRW total PnL "
        f"across {agg['baseline']['signals']:.0f} attempted signals - reference point for all deltas below."
    )
    lines.append(
        f"- **A (execution-delay proxy)**: {fill_rate:.1f}% of attempted signals FILLED "
        f"({a_cancelled} of {total_a} CANCELLED because the reprice chase exceeded "
        f"BUY_ORDER_REPRICE_MAX_CHASE_PCT or the day's data ran out), trades taken "
        f"{agg['A']['trades']:.0f} ({_delta('A','trades'):+.0f} vs baseline), PnL "
        f"{agg['A']['pnl']:+,.0f} KRW ({_delta('A','pnl'):+,.0f} vs baseline)"
        + (f", avg slippage on delayed fills {a_avg_slip:+.4f}%" if a_avg_slip is not None else "")
        + " - this is the key number for the hypothesis that passive-order execution delay "
        "costs real money: a high cancel rate and/or clearly negative average slippage would "
        "support it, while a high fill rate with near-zero slippage would refute it."
    )
    lines.append(
        f"- **B (confirm-count=1)**: signals attempted {agg['B']['signals']:.0f} "
        f"({_delta('B','signals'):+.0f}), trades {agg['B']['trades']:.0f} ({_delta('B','trades'):+.0f}), "
        f"win rate {agg['B']['win_rate']:.1f}% ({_delta('B','win_rate'):+.1f}pp), PnL "
        f"{agg['B']['pnl']:+,.0f} KRW ({_delta('B','pnl'):+,.0f} vs baseline) - acting on the first "
        "confirmation instead of waiting for a second should fire earlier/more often; whether that "
        "helps or hurts PnL here indicates whether the second confirmation bar was filtering out "
        "false signals or just adding costly delay."
    )
    lines.append(
        f"- **C (opening guard disabled)**: signals attempted {agg['C']['signals']:.0f} "
        f"({_delta('C','signals'):+.0f}), trades {agg['C']['trades']:.0f} ({_delta('C','trades'):+.0f}), "
        f"win rate {agg['C']['win_rate']:.1f}% ({_delta('C','win_rate'):+.1f}pp), PnL "
        f"{agg['C']['pnl']:+,.0f} KRW ({_delta('C','pnl'):+,.0f} vs baseline) - removes the stricter "
        "score>=12 requirement in the first 15 minutes after 09:00, so more early-session entries "
        "should qualify; the PnL delta shows whether that stricter opening-window filter was "
        "protective or just overly conservative."
    )
    lines.append(
        f"- **D (VWAP mandatory gate)**: signals attempted {agg['D']['signals']:.0f} "
        f"({_delta('D','signals'):+.0f}), trades {agg['D']['trades']:.0f} ({_delta('D','trades'):+.0f}), "
        f"win rate {agg['D']['win_rate']:.1f}% ({_delta('D','win_rate'):+.1f}pp), PnL "
        f"{agg['D']['pnl']:+,.0f} KRW ({_delta('D','pnl'):+,.0f} vs baseline) - promoting VWAP from a "
        "+1/+2 bonus-score component to a hard mandatory gate should reject some signals that "
        "previously passed on other score components alone; fewer trades with a higher win rate "
        "would support making VWAP mandatory, fewer trades with a similar or lower win rate would not."
    )
    lines.append("")

    lines.append("## Code changes (file:line ranges)")
    lines.append("")
    lines.append(
        "- `live_trading/r001_define_config.py` (~line 513-519): added two new constants "
        "immediately after `BB_UPPER_GAP_MIN_PCT` - `ENABLE_VWAP_MANDATORY_GATE = False` and "
        "`VWAP_MANDATORY_GATE_MIN_PCT = 0.0`. Pure addition, no existing line changed; default "
        "preserves current behavior exactly."
    )
    lines.append(
        "- `live_trading/r002_strategy_core_shared.py`: (1) import block ~line 322-325, added "
        "`ENABLE_VWAP_MANDATORY_GATE`/`VWAP_MANDATORY_GATE_MIN_PCT` to the existing "
        "`from r001_define_config import (...)` statement. (2) ~line 1156-1170, new function "
        "`_gate_vwap_mandatory(ctx)` placed directly after `_gate_min_liquidity_safety` - returns "
        "`(True, ctx, \"SKIP_VWAP_GATE_DISABLED\")` immediately when the flag is off, otherwise "
        "requires `close > VWAP*(1+VWAP_MANDATORY_GATE_MIN_PCT)`. (3) ~line 1229-1236, appended a "
        "new `BuyGateCondition(\"vwap_mandatory_break\", ...)` as the 10th (last) entry of "
        "`BUY_GATE_CONDITIONS`, after `min_liquidity_safety` - no reordering of the other 9 gates. "
        "Because `HYBRID_3MIN_CONTEXT_GATES` (~line 1288) is built as a list comprehension over "
        "`BUY_GATE_CONDITIONS` at module-import time, the new gate is automatically included there "
        "too (it is not one of the two excluded gate names)."
    )
    lines.append(
        "- `data_simulation/g003_trade_simulate_by_date.py`: (1) ~line 175-176, added "
        "`BUY_ORDER_REPRICE_AFTER_SECONDS`/`BUY_ORDER_REPRICE_MAX_CHASE_PCT` to the r001 import "
        "block, and added `import r002_strategy_core_shared as _r002_mod` immediately before the "
        "existing `from r002_strategy_core_shared import (...)` block. (2) ~line 400-408, added "
        "`_env_int(name, default)` helper following the existing `_env_bool`/`_env_float` pattern. "
        "(3) ~line 455-478 (Experiment overrides block, right after `SIMULATE_10S_GRID_DEFAULT`): "
        "Experiment B overrides g003's own module-level `BUY_CONSECUTIVE_CONFIRM_COUNT` directly "
        "(env `R76_SIM_BUY_CONSECUTIVE_CONFIRM_COUNT`); Experiment C and D instead mutate "
        "`_r002_mod.OPENING_GUARD_MINUTES` / `_r002_mod.ENABLE_VWAP_MANDATORY_GATE` (env "
        "`R76_SIM_OPENING_GUARD_MINUTES` / `R76_SIM_ENABLE_VWAP_MANDATORY_GATE`) because those two "
        "names are read as r002's OWN module-global bindings inside "
        "`run_buy_condition_pipeline_comment()`/`run_3min_context_pipeline()` and "
        "`_gate_vwap_mandatory()` respectively - reassigning g003's own same-named globals would "
        "have had zero effect on r002's behavior (verified by reading the actual import/read "
        "sites, not assumed). `R76_SIM_EXEC_MODEL` (default `\"instant\"`) is read as a plain "
        "string. (4) ~line 2384-2429 (data loading loop): added `exec_sim_raw_frames: dict[str, "
        "pd.DataFrame] = {}`, populated with the pristine per-code `raw_df` *before* the "
        "`upsample_price_frame_to_10s()` call - kept deliberately separate from the pre-existing "
        "loop-local variable literally named `raw_frame` (~line 2481, which is actually "
        "`price_frames.get(code)`, i.e. the ffilled/upsampled frame when "
        "`SIMULATE_10S_GRID_DEFAULT=True`, the default) to avoid silently reading duplicated "
        "ffilled low/high values instead of genuine intrabar lows. (5) ~line 2126-2199, new "
        "helper functions `_raw_bar_after()` and `_simulate_passive_limit_fill()` implementing the "
        "Experiment A proxy algorithm exactly as specified (L1 check at t1+10s, L2 reprice check "
        "at t2+10s, L3 final chase-or-cancel check, using `BUY_ORDER_REPRICE_AFTER_SECONDS` and "
        "`BUY_ORDER_REPRICE_MAX_CHASE_PCT` imported from r001, not hardcoded). (6) ~line 3253-3283, "
        "the original `sim.buy(code, ..., price, ts, session, reason)` call is now wrapped in an "
        "`if R76_SIM_EXEC_MODEL == \"passive_limit_chase\":` branch that runs the proxy, logs one "
        "`EXEC_SIM_RESULT ...` line per attempted signal, and calls `sim.buy()` with the proxy's "
        "fill price/time instead of the signal tick - falling through to the original unchanged "
        "call when the env var is unset/`\"instant\"`."
    )
    lines.append("")

    lines.append("## Confirmations")
    lines.append("")
    lines.append(
        "- **r001 pre-existing default values were unchanged.** Only two brand-new constants were "
        "added (`ENABLE_VWAP_MANDATORY_GATE = False`, `VWAP_MANDATORY_GATE_MIN_PCT = 0.0`); no "
        "existing r001 line was edited. `git diff` on r001/r002 for this task's changes shows pure "
        "additions only (verified explicitly by extracting and reviewing the diff hunks before "
        "running anything)."
    )
    lines.append(
        "- **`r003_trade_live_execute.py` (the live executor) was never opened for writing and "
        "never run** at any point in this work - all experiments run exclusively through the "
        "backtest simulator `g003_trade_simulate_by_date.py` (no real orders)."
    )
    lines.append(
        "- **Every new toggle defaults to reproducing current/live behavior when its env var is "
        "unset.** `ENABLE_VWAP_MANDATORY_GATE` defaults to `False` in r001 and the new "
        "`_gate_vwap_mandatory` gate short-circuits to always-pass in that state; `R76_SIM_EXEC_MODEL` "
        "defaults to `\"instant\"`, which takes the original unmodified `sim.buy(...)` code path; "
        "`R76_SIM_BUY_CONSECUTIVE_CONFIRM_COUNT`/`R76_SIM_OPENING_GUARD_MINUTES` default to the "
        "existing r001 values (2 and 15 respectively) when unset. This was verified concretely for "
        "Experiment D with a smoke test: `g003 --date 20260907 --codes 001820 --summary` was run "
        "once immediately before adding the r002 gate function/list entry and once immediately "
        "after (with no env override in either run), and the two full trade logs were byte-for-byte "
        "identical except for the auto-incrementing log-file name/index that g003 stamps into its "
        "own output on every run (`20260907_simulate_00N_result.txt`) - i.e. a confirmed no-op on "
        "default behavior."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **Experiment A is a proxy, not a certified replay of live execution.** No historical "
        "bid/ask orderbook data exists for this backtest dataset (a known, pre-existing, permanent "
        "gap documented in r002/g003's own changelog comments), so the passive-limit/reprice/cancel "
        "state machine cannot be replayed literally. The proxy uses the RAW (non-ffilled) 10-second "
        "bar's `low` as a stand-in for \"would a resting bid at this price have traded\" - this is a "
        "reasonable approximation given the constraint, but it is not equivalent to true orderbook-"
        "level fill simulation, and its precision should not be oversold."
    )
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

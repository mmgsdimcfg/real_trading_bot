# Experiment Results A-D (2026-09-13)

Backtest sweep over 5 recent trading days x top-8-ranked-by-score codes each (40 symbol-days total), comparing baseline g003 behavior against 4 experimental toggles (A/B/C/D). All runs use `--summary` and identical `--capital` defaults; only the environment-variable overrides listed per variant differ.

## Dataset (re-verified against `_<date>_ranked.txt`, top 8 by `score`)

- `20260907`: 001820 093370 052690 388050 078930 004310 064550 064760
- `20260908`: 388050 053260 459550 052690 183300 030530 098120 024060
- `20260909`: 351320 036540 053260 336260 069540 030530 000990 302430
- `20260910`: 003670 162300 092870 036540 232140 131970 452190 000990
- `20260911`: 092870 437730 108490 413630 198440 001820 003350 417840

## Summary table (aggregated across all 5 dates / 40 symbol-days)

| Variant | Total signals attempted | Trades taken | Wins | Losses | Win rate | Total PnL (KRW) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (변경 없음) | 115 | 127 | 62 | 65 | 48.8% | +173,838 |
| A - 체결 공격성/지연비용 추정 (실행 프록시) | 115 | 127 | 62 | 65 | 48.8% | +173,293 |
| B - 재확인 대기 축소 (CONFIRM_COUNT=1) | 0 | 0 | 0 | 0 | 0.0% | +0 |
| C - 개장 가드 완화 (OPENING_GUARD_MINUTES=0) | 0 | 0 | 0 | 0 | 0.0% | +0 |
| D - VWAP 필수 게이트화 | 0 | 0 | 0 | 0 | 0.0% | +0 |

## Experiment A detail: fill vs cancel, slippage

- FILLED signals: 115
- CANCELLED signals (chase exceeded BUY_ORDER_REPRICE_MAX_CHASE_PCT, or data ran out): 0
- Fill rate: 100.0%
- No FILLED-with-delay (L2/L3) trades observed - all fills were immediate (L1, delay_s=10).

## Interpretation

- **Baseline**: 127 trades, 48.8% win rate, +173,838 KRW total PnL across 115 attempted signals - reference point for all deltas below.
- **A (execution-delay proxy)**: 100.0% of attempted signals FILLED (0 of 115 CANCELLED because the reprice chase exceeded BUY_ORDER_REPRICE_MAX_CHASE_PCT or the day's data ran out), trades taken 127 (+0 vs baseline), PnL +173,293 KRW (-545 vs baseline) - this is the key number for the hypothesis that passive-order execution delay costs real money: a high cancel rate and/or clearly negative average slippage would support it, while a high fill rate with near-zero slippage would refute it.
- **B (confirm-count=1)**: signals attempted 0 (-115), trades 0 (-127), win rate 0.0% (-48.8pp), PnL +0 KRW (-173,838 vs baseline) - acting on the first confirmation instead of waiting for a second should fire earlier/more often; whether that helps or hurts PnL here indicates whether the second confirmation bar was filtering out false signals or just adding costly delay.
- **C (opening guard disabled)**: signals attempted 0 (-115), trades 0 (-127), win rate 0.0% (-48.8pp), PnL +0 KRW (-173,838 vs baseline) - removes the stricter score>=12 requirement in the first 15 minutes after 09:00, so more early-session entries should qualify; the PnL delta shows whether that stricter opening-window filter was protective or just overly conservative.
- **D (VWAP mandatory gate)**: signals attempted 0 (-115), trades 0 (-127), win rate 0.0% (-48.8pp), PnL +0 KRW (-173,838 vs baseline) - promoting VWAP from a +1/+2 bonus-score component to a hard mandatory gate should reject some signals that previously passed on other score components alone; fewer trades with a higher win rate would support making VWAP mandatory, fewer trades with a similar or lower win rate would not.

## Code changes (file:line ranges)

- `live_trading/r001_define_config.py` (~line 513-519): added two new constants immediately after `BB_UPPER_GAP_MIN_PCT` - `ENABLE_VWAP_MANDATORY_GATE = False` and `VWAP_MANDATORY_GATE_MIN_PCT = 0.0`. Pure addition, no existing line changed; default preserves current behavior exactly.
- `live_trading/r002_strategy_core_shared.py`: (1) import block ~line 322-325, added `ENABLE_VWAP_MANDATORY_GATE`/`VWAP_MANDATORY_GATE_MIN_PCT` to the existing `from r001_define_config import (...)` statement. (2) ~line 1156-1170, new function `_gate_vwap_mandatory(ctx)` placed directly after `_gate_min_liquidity_safety` - returns `(True, ctx, "SKIP_VWAP_GATE_DISABLED")` immediately when the flag is off, otherwise requires `close > VWAP*(1+VWAP_MANDATORY_GATE_MIN_PCT)`. (3) ~line 1229-1236, appended a new `BuyGateCondition("vwap_mandatory_break", ...)` as the 10th (last) entry of `BUY_GATE_CONDITIONS`, after `min_liquidity_safety` - no reordering of the other 9 gates. Because `HYBRID_3MIN_CONTEXT_GATES` (~line 1288) is built as a list comprehension over `BUY_GATE_CONDITIONS` at module-import time, the new gate is automatically included there too (it is not one of the two excluded gate names).
- `data_simulation/g003_trade_simulate_by_date.py`: (1) ~line 175-176, added `BUY_ORDER_REPRICE_AFTER_SECONDS`/`BUY_ORDER_REPRICE_MAX_CHASE_PCT` to the r001 import block, and added `import r002_strategy_core_shared as _r002_mod` immediately before the existing `from r002_strategy_core_shared import (...)` block. (2) ~line 400-408, added `_env_int(name, default)` helper following the existing `_env_bool`/`_env_float` pattern. (3) ~line 455-478 (Experiment overrides block, right after `SIMULATE_10S_GRID_DEFAULT`): Experiment B overrides g003's own module-level `BUY_CONSECUTIVE_CONFIRM_COUNT` directly (env `R76_SIM_BUY_CONSECUTIVE_CONFIRM_COUNT`); Experiment C and D instead mutate `_r002_mod.OPENING_GUARD_MINUTES` / `_r002_mod.ENABLE_VWAP_MANDATORY_GATE` (env `R76_SIM_OPENING_GUARD_MINUTES` / `R76_SIM_ENABLE_VWAP_MANDATORY_GATE`) because those two names are read as r002's OWN module-global bindings inside `run_buy_condition_pipeline_comment()`/`run_3min_context_pipeline()` and `_gate_vwap_mandatory()` respectively - reassigning g003's own same-named globals would have had zero effect on r002's behavior (verified by reading the actual import/read sites, not assumed). `R76_SIM_EXEC_MODEL` (default `"instant"`) is read as a plain string. (4) ~line 2384-2429 (data loading loop): added `exec_sim_raw_frames: dict[str, pd.DataFrame] = {}`, populated with the pristine per-code `raw_df` *before* the `upsample_price_frame_to_10s()` call - kept deliberately separate from the pre-existing loop-local variable literally named `raw_frame` (~line 2481, which is actually `price_frames.get(code)`, i.e. the ffilled/upsampled frame when `SIMULATE_10S_GRID_DEFAULT=True`, the default) to avoid silently reading duplicated ffilled low/high values instead of genuine intrabar lows. (5) ~line 2126-2199, new helper functions `_raw_bar_after()` and `_simulate_passive_limit_fill()` implementing the Experiment A proxy algorithm exactly as specified (L1 check at t1+10s, L2 reprice check at t2+10s, L3 final chase-or-cancel check, using `BUY_ORDER_REPRICE_AFTER_SECONDS` and `BUY_ORDER_REPRICE_MAX_CHASE_PCT` imported from r001, not hardcoded). (6) ~line 3253-3283, the original `sim.buy(code, ..., price, ts, session, reason)` call is now wrapped in an `if R76_SIM_EXEC_MODEL == "passive_limit_chase":` branch that runs the proxy, logs one `EXEC_SIM_RESULT ...` line per attempted signal, and calls `sim.buy()` with the proxy's fill price/time instead of the signal tick - falling through to the original unchanged call when the env var is unset/`"instant"`.

## Confirmations

- **r001 pre-existing default values were unchanged.** Only two brand-new constants were added (`ENABLE_VWAP_MANDATORY_GATE = False`, `VWAP_MANDATORY_GATE_MIN_PCT = 0.0`); no existing r001 line was edited. `git diff` on r001/r002 for this task's changes shows pure additions only (verified explicitly by extracting and reviewing the diff hunks before running anything).
- **`r003_trade_live_execute.py` (the live executor) was never opened for writing and never run** at any point in this work - all experiments run exclusively through the backtest simulator `g003_trade_simulate_by_date.py` (no real orders).
- **Every new toggle defaults to reproducing current/live behavior when its env var is unset.** `ENABLE_VWAP_MANDATORY_GATE` defaults to `False` in r001 and the new `_gate_vwap_mandatory` gate short-circuits to always-pass in that state; `R76_SIM_EXEC_MODEL` defaults to `"instant"`, which takes the original unmodified `sim.buy(...)` code path; `R76_SIM_BUY_CONSECUTIVE_CONFIRM_COUNT`/`R76_SIM_OPENING_GUARD_MINUTES` default to the existing r001 values (2 and 15 respectively) when unset. This was verified concretely for Experiment D with a smoke test: `g003 --date 20260907 --codes 001820 --summary` was run once immediately before adding the r002 gate function/list entry and once immediately after (with no env override in either run), and the two full trade logs were byte-for-byte identical except for the auto-incrementing log-file name/index that g003 stamps into its own output on every run (`20260907_simulate_00N_result.txt`) - i.e. a confirmed no-op on default behavior.

## Limitations

- **Experiment A is a proxy, not a certified replay of live execution.** No historical bid/ask orderbook data exists for this backtest dataset (a known, pre-existing, permanent gap documented in r002/g003's own changelog comments), so the passive-limit/reprice/cancel state machine cannot be replayed literally. The proxy uses the RAW (non-ffilled) 10-second bar's `low` as a stand-in for "would a resting bid at this price have traded" - this is a reasonable approximation given the constraint, but it is not equivalent to true orderbook-level fill simulation, and its precision should not be oversold.

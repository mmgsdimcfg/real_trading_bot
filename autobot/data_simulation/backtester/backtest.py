# -*- coding: utf-8 -*-
"""Main backtesting loop."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from shared.config import (
    ALLOW_REBUY_SAME_CODE,
    MIN_BARS_REQUIRED,
    REGULAR_END,
    REGULAR_FORCE_EXIT,
    REGULAR_NEW_ENTRY_CUTOFF,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    TRADE_COOLDOWN_MINUTES,
    TRAILING_STOP_FROM_PEAK,
)
from shared.indicators import calculate_indicators
from shared.strategy import check_buy_condition, check_sell_condition
from .data_loader import DataLoader
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


def classify_buy_session(ts: pd.Timestamp) -> str:
    """Classify which trading session the timestamp belongs to."""
    current_time = ts.time()
    from shared.config import AFTERNOON_NXT_START, AFTERNOON_NXT_END, MORNING_NXT_START, MORNING_NXT_END
    
    if MORNING_NXT_START <= current_time <= MORNING_NXT_END:
        return "morning_nxt"
    if AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END:
        return "afternoon_nxt"
    return "regular"


def can_trade_now(ts: pd.Timestamp) -> bool:
    """Check if trading is allowed at this timestamp."""
    from shared.config import REGULAR_START, REGULAR_END, MORNING_NXT_START, MORNING_NXT_END, AFTERNOON_NXT_START, AFTERNOON_NXT_END
    
    current_time = ts.time()
    return (
        (REGULAR_START <= current_time <= REGULAR_END)
        or (MORNING_NXT_START <= current_time <= MORNING_NXT_END)
        or (AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_END)
    )


def is_new_entry_allowed(ts: pd.Timestamp) -> bool:
    """Check if new position entry is allowed at this timestamp."""
    from shared.config import REGULAR_START, REGULAR_NEW_ENTRY_CUTOFF, MORNING_NXT_START, MORNING_NXT_END, AFTERNOON_NXT_START, AFTERNOON_NXT_NEW_ENTRY_CUTOFF
    
    current_time = ts.time()
    if REGULAR_START <= current_time < REGULAR_NEW_ENTRY_CUTOFF:
        return True
    if (MORNING_NXT_START <= current_time <= MORNING_NXT_END) or (AFTERNOON_NXT_START <= current_time <= AFTERNOON_NXT_NEW_ENTRY_CUTOFF):
        return True
    return False


def run_forced_liquidations(portfolio: Portfolio, frames: dict[str, pd.DataFrame], ts: pd.Timestamp, target_date) -> None:
    """Run scheduled end-of-day liquidations."""
    current_time = ts.time()
    
    if current_time >= REGULAR_FORCE_EXIT:
        for code in list(portfolio.positions.keys()):
            pos = portfolio.positions.get(code)
            if pos is None or pos.buy_time.date() != target_date:
                continue
            
            frame = frames.get(code)
            if frame is None or ts not in frame.index:
                continue
            
            price = float(frame.loc[:ts].iloc[-1]["close"])
            portfolio.sell(code, price, ts, "EOD_REGULAR", classify_buy_session(ts))


def run_backtest(
    date_str: str,
    data_root: Path,
    codes: list[str] | None = None,
    names: dict[str, str] | None = None,
    initial_capital: float = 10_000_000,
) -> dict:
    """Run complete backtest for a single date.
    
    Args:
        date_str: Date in YYYYMMDD format
        data_root: Root directory for data
        codes: Optional list of specific codes to test
        names: Optional dict of {code: name}
        initial_capital: Starting capital in KRW
    
    Returns:
        Dictionary with backtest results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Backtest Date: {date_str} [R73 MA5-BB Cross Strategy]")
    logger.info(f"Initial Capital: {initial_capital:,.0f} KRW")
    logger.info(f"{'='*60}\n")

    # Load data
    loader = DataLoader(data_root)
    frames_dict = loader.load_date_folder(date_str)
    
    if not frames_dict:
        logger.error(f"No data found for {date_str}")
        return {"error": "No data"}

    # Filter by specific codes if provided
    if codes:
        target_codes = {c.zfill(6) for c in codes}
        frames_dict = {c: df for c, df in frames_dict.items() if c in target_codes}
        logger.info(f"Filtered to {len(frames_dict)} codes")

    if not frames_dict:
        logger.error("No matching codes after filtering")
        return {"error": "No matching codes"}

    # Calculate indicators for all frames
    for code, frame in frames_dict.items():
        frames_dict[code] = calculate_indicators(frame)

    # Get all timestamps to replay
    target_date = datetime.strptime(date_str, "%Y%m%d").date()
    all_times = sorted({
        ts for df in frames_dict.values()
        for ts in df.index
        if ts.date() == target_date
    })

    if not all_times:
        logger.error(f"No bars found for {date_str}")
        return {"error": "No bars for target date"}

    # Initialize portfolio
    portfolio = Portfolio(initial_capital)
    selected_names = dict(names or {})

    logger.info(f"Starting simulation with {len(all_times)} timestamps\n")

    # Main loop
    for ts in all_times:
        # Run forced liquidations
        run_forced_liquidations(portfolio, frames_dict, ts, target_date)

        # Process each code
        for code, frame in frames_dict.items():
            if ts not in frame.index:
                continue

            if not can_trade_now(ts):
                continue

            available = frame[frame.index <= ts]
            if len(available) < MIN_BARS_REQUIRED:
                continue

            price = float(available.iloc[-1]["close"])
            session = classify_buy_session(ts)

            # Check position management (TP/SL/Trail)
            pos = portfolio.positions.get(code)
            if pos is not None:
                profit_pct = price / pos.buy_price - 1.0
                pos.highest_price = max(pos.highest_price, price)

                # Take profit
                if profit_pct >= TAKE_PROFIT_PERCENT:
                    portfolio.sell(
                        code, price, ts,
                        f"TAKE_PROFIT_{TAKE_PROFIT_PERCENT*100:.1f}%",
                        session
                    )
                    continue

                # Stop loss
                if profit_pct <= STOP_LOSS_PERCENT:
                    portfolio.sell(
                        code, price, ts,
                        f"STOP_LOSS_{abs(STOP_LOSS_PERCENT)*100:.1f}%",
                        session
                    )
                    continue

                # Trailing stop
                dd_from_peak = (price / pos.highest_price - 1.0) if pos.highest_price > 0 else 0.0
                if dd_from_peak <= -TRAILING_STOP_FROM_PEAK:
                    portfolio.sell(
                        code, price, ts,
                        f"TRAILING_STOP_{TRAILING_STOP_FROM_PEAK*100:.1f}%",
                        session
                    )
                    continue

                # Check sell signal
                should_sell, sell_reason = check_sell_condition(available)
                if should_sell:
                    portfolio.sell(code, price, ts, sell_reason, session)
                continue

            # Try to buy
            if not is_new_entry_allowed(ts):
                continue
            if portfolio.in_cooldown(code, ts):
                continue
            if (not ALLOW_REBUY_SAME_CODE) and code in portfolio.completed_codes:
                continue

            # Check buy signal
            should_buy, buy_reason = check_buy_condition(available, ts)
            if should_buy:
                qty = int(500_000 / price)  # MAX_ORDER_AMOUNT_KRW / price
                if qty > 0 and qty * price <= portfolio.cash:
                    portfolio.buy(
                        code,
                        selected_names.get(code, code),
                        price,
                        qty,
                        ts,
                        session,
                        buy_reason
                    )

    # Force close all positions at end of day
    final_close_time = pd.Timestamp(datetime.combine(target_date, REGULAR_END))
    for code in list(portfolio.positions.keys()):
        frame = frames_dict.get(code)
        if frame is not None and final_close_time in frame.index:
            price = float(frame.loc[:final_close_time].iloc[-1]["close"])
            session = portfolio.positions[code].buy_session
            portfolio.sell(code, price, final_close_time, "EOD_CLOSE", session)

    # Calculate final stats
    last_prices = {}
    for code, frame in frames_dict.items():
        same_day = frame[frame.index.date == target_date]
        if not same_day.empty:
            last_prices[code] = float(same_day.iloc[-1]["close"])

    final_value = portfolio.portfolio_value(last_prices)
    stats = portfolio.get_stats(final_value)
    stats["date"] = date_str
    stats["trades"] = portfolio.trade_log

    # Log results
    logger.info(f"\n{'='*60}")
    logger.info("BACKTEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Initial Capital: {stats['initial_capital']:,.0f} KRW")
    logger.info(f"Final Value:     {stats['final_value']:,.0f} KRW")
    logger.info(f"Total P&L:       {stats['total_pnl']:+,.0f} KRW ({stats['total_pnl_pct']:+.2f}%)")
    logger.info(f"Realized P&L:    {stats['realized_pnl']:+,.0f} KRW")
    logger.info(f"Win Rate:        {stats['win_rate']:.1f}% ({stats['win_count']}/{stats['sell_count']})")
    logger.info(f"Traded Codes:    {stats['traded_codes']}")
    logger.info(f"{'='*60}\n")

    return stats

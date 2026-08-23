# -*- coding: utf-8 -*-
"""Portfolio and position management for backtesting."""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""

    code: str
    name: str
    buy_price: float
    quantity: int
    buy_time: pd.Timestamp
    buy_session: str
    highest_price: float = field(default=0.0)

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.buy_price


@dataclass
class TradeRecord:
    """Records a single buy/sell action."""

    code: str
    name: str
    action: str  # "BUY" or "SELL"
    bar_time: pd.Timestamp
    price: float
    quantity: int
    reason: str
    session: str
    pnl_pct: float | None = None
    pnl_krw: float | None = None

    def __repr__(self) -> str:
        pnl_str = f" | pnl={self.pnl_pct:.2f}%" if self.pnl_pct is not None else ""
        return (
            f"{self.bar_time:%H:%M} | {self.action:4s} | {self.code}({self.name}) | "
            f"qty={self.quantity} | price={self.price:,.0f}{pnl_str} | {self.reason}"
        )


class Portfolio:
    """Manages cash, positions, and trade history."""

    def __init__(self, initial_capital: float):
        """Initialize portfolio.
        
        Args:
            initial_capital: Starting cash in KRW
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_log: list[TradeRecord] = []
        self.cooldown_until: dict[str, pd.Timestamp] = {}
        self.completed_codes: set[str] = set()

    def in_cooldown(self, code: str, now: pd.Timestamp) -> bool:
        """Check if code is in trade cooldown."""
        until = self.cooldown_until.get(code)
        return until is not None and now < until

    def set_cooldown(self, code: str, now: pd.Timestamp, minutes: int = 3) -> None:
        """Set cooldown period after trade."""
        self.cooldown_until[code] = now + pd.Timedelta(minutes=minutes)

    def buy(
        self,
        code: str,
        name: str,
        price: float,
        quantity: int,
        now: pd.Timestamp,
        session: str,
        reason: str,
    ) -> bool:
        """Execute buy order.
        
        Returns:
            True if successful, False otherwise
        """
        if self.in_cooldown(code, now) or price <= 0 or code in self.positions:
            return False

        if quantity <= 0 or price * quantity > self.cash:
            return False

        self.cash -= price * quantity
        self.positions[code] = Position(code, name, price, quantity, now, session)
        self.set_cooldown(code, now)

        rec = TradeRecord(code, name, "BUY", now, price, quantity, reason, session)
        self.trade_log.append(rec)
        logger.info(f"  {rec}")
        return True

    def sell(
        self,
        code: str,
        price: float,
        now: pd.Timestamp,
        reason: str,
        session: str,
    ) -> bool:
        """Execute sell order.
        
        Returns:
            True if successful, False otherwise
        """
        pos = self.positions.get(code)
        if pos is None or pos.quantity <= 0 or price <= 0:
            return False

        proceeds = price * pos.quantity
        self.cash += proceeds

        rec = TradeRecord(code, pos.name, "SELL", now, price, pos.quantity, reason, session)
        rec.pnl_pct = (price / pos.buy_price - 1.0) * 100.0
        rec.pnl_krw = (price - pos.buy_price) * pos.quantity
        self.trade_log.append(rec)
        logger.info(f"  {rec}")

        del self.positions[code]
        self.completed_codes.add(code)
        self.set_cooldown(code, now)
        return True

    def update_position_prices(self, current_prices: dict[str, float]) -> None:
        """Update highest_price for open positions."""
        for code, pos in self.positions.items():
            current_price = current_prices.get(code, pos.buy_price)
            pos.highest_price = max(pos.highest_price, current_price)

    def portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total = self.cash
        for code, pos in self.positions.items():
            total += current_prices.get(code, pos.buy_price) * pos.quantity
        return total

    def realized_pnl(self) -> float:
        """Calculate total realized P&L from sold positions."""
        sell_trades = [r for r in self.trade_log if r.action == "SELL" and r.pnl_krw is not None]
        return sum(r.pnl_krw for r in sell_trades)

    def get_stats(self, final_value: float) -> dict:
        """Get summary statistics."""
        sell_trades = [r for r in self.trade_log if r.action == "SELL" and r.pnl_krw is not None]
        wins = [r for r in sell_trades if r.pnl_krw > 0]
        losses = [r for r in sell_trades if r.pnl_krw <= 0]

        total_pnl = final_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital * 100.0) if self.initial_capital > 0 else 0.0
        realized_pnl = self.realized_pnl()
        win_rate = (len(wins) / len(sell_trades) * 100.0) if sell_trades else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "realized_pnl": realized_pnl,
            "cash": self.cash,
            "sell_count": len(sell_trades),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": win_rate,
            "traded_codes": len(set(r.code for r in self.trade_log)),
        }

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Direction(int, Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def validate(self) -> None:
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError(f"non-positive OHLC at {self.timestamp.isoformat()}")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError(f"inconsistent OHLC at {self.timestamp.isoformat()}")
        if self.volume < 0:
            raise ValueError(f"negative volume at {self.timestamp.isoformat()}")


@dataclass(frozen=True)
class Signal:
    direction: Direction
    confidence: float
    reason: str
    stop_loss_pct: float
    take_profit_pct: float
    risk_multiplier: float = 1.0
    max_holding_bars: int = 0


@dataclass
class Position:
    direction: Direction
    qty: float
    entry_price: float
    entry_time: datetime
    stop_price: float
    take_profit_price: float
    entry_fee: float
    peak_price: float
    trough_price: float
    bars_held: int = 0
    max_holding_bars: int = 0

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price

    def unrealized_pnl(self, mark_price: float) -> float:
        return self.direction.value * self.qty * (mark_price - self.entry_price)


@dataclass(frozen=True)
class Trade:
    direction: Direction
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str

    @property
    def return_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return self.net_pnl / (self.entry_price * self.qty)


@dataclass(frozen=True)
class EquityPoint:
    timestamp: datetime
    equity: float
    drawdown_pct: float

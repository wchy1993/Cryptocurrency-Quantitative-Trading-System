from __future__ import annotations

from datetime import date

from .config import RiskConfig
from .models import Candle, Signal


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.peak_equity = config.initial_equity
        self.day: date | None = None
        self.day_start_equity = config.initial_equity
        self.cooldown_remaining = 0

    def on_bar(self, candle: Candle, equity: float) -> None:
        self.peak_equity = max(self.peak_equity, equity)
        current_day = candle.timestamp.date()
        if self.day != current_day:
            self.day = current_day
            self.day_start_equity = equity
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def can_enter(self, equity: float) -> tuple[bool, str]:
        if equity <= 0:
            return False, "equity_depleted"
        if self.cooldown_remaining > 0:
            return False, "cooldown"
        if equity <= self.day_start_equity * (1.0 - self.config.max_daily_loss_pct):
            return False, "daily_loss_limit"
        if equity <= self.peak_equity * (1.0 - self.config.max_drawdown_pct):
            return False, "max_drawdown_limit"
        return True, "ok"

    def size_position(self, equity: float, price: float, signal: Signal) -> tuple[float, str]:
        if price <= 0:
            return 0.0, "bad_price"
        if signal.stop_loss_pct <= 0:
            return 0.0, "missing_stop"

        risk_amount = equity * self.config.risk_per_trade_pct * max(0.0, min(1.0, signal.risk_multiplier))
        loss_per_unit = price * signal.stop_loss_pct
        qty_by_risk = risk_amount / loss_per_unit

        max_notional_by_leverage = equity * self.config.max_leverage
        max_notional_by_policy = equity * self.config.max_position_notional_pct
        max_notional = min(max_notional_by_leverage, max_notional_by_policy)
        qty_by_notional = max_notional / price

        qty = max(0.0, min(qty_by_risk, qty_by_notional))
        if qty * price < self.config.min_order_notional:
            return 0.0, "below_min_notional"
        return qty, "ok"

    def on_trade_closed(self, net_pnl: float) -> None:
        if net_pnl < 0:
            self.cooldown_remaining = max(self.cooldown_remaining, self.config.cooldown_bars_after_loss)

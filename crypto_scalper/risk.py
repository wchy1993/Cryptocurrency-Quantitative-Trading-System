from __future__ import annotations

from datetime import date

from .config import RiskConfig
from .models import Candle, Signal


def signal_risk_weight(confidence: float, risk_multiplier: float) -> float:
    """Map signal quality to risk weight with high-conviction acceleration.

    Low quality signals stay as probe-size entries. Above 0.75 confidence the
    weight accelerates so the best long/short signals contribute materially more
    profit, while the final result is still capped by position and drawdown limits.
    """
    score = max(0.0, min(1.0, confidence))
    base = max(0.0, min(1.5, risk_multiplier))
    if score < 0.55:
        conviction_multiplier = 0.45 + score
    elif score < 0.75:
        conviction_multiplier = 1.0 + (score - 0.55) / 0.20 * 0.25
    else:
        conviction_multiplier = 1.25 + (score - 0.75) / 0.25 * 0.55
    return max(0.0, min(1.8, base * conviction_multiplier))


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.peak_equity = config.initial_equity
        self.day: date | None = None
        self.day_start_equity = config.initial_equity
        self.cooldown_remaining = 0
        self.consecutive_losses = 0

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

    def current_drawdown_pct(self, equity: float) -> float:
        if self.peak_equity <= 0:
            return 1.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

    def drawdown_size_multiplier(self, equity: float) -> float:
        drawdown = self.current_drawdown_pct(equity)
        limit = self.config.max_drawdown_pct
        if limit <= 0:
            return 1.0
        pressure = drawdown / limit
        if pressure >= 1.0:
            return 0.0
        if pressure >= 0.90:
            return 0.20
        if pressure >= 0.75:
            return 0.35
        if pressure >= 0.50:
            return 0.60
        return 1.0

    def size_position(
        self,
        equity: float,
        price: float,
        signal: Signal,
        existing_notional: float = 0.0,
        entry_fraction: float | None = None,
    ) -> tuple[float, str]:
        if price <= 0:
            return 0.0, "bad_price"
        if signal.stop_loss_pct <= 0:
            return 0.0, "missing_stop"

        # Position rule:
        # 1. Signal confidence is capped to 0..1 and comes from non-duplicated signal groups.
        # 2. risk_multiplier carries long/short bias and any reduced-risk signal flags.
        # 3. signal_risk_weight keeps weak signals small, but accelerates 0.75+ confidence signals.
        # 4. A drawdown brake reduces new risk as total drawdown approaches max_drawdown_pct.
        # 5. Existing notional is subtracted for scale-ins, and max_position_notional_pct is the total cap.
        risk_weight = signal_risk_weight(signal.confidence, signal.risk_multiplier)
        risk_amount = equity * self.config.risk_per_trade_pct * risk_weight * self.drawdown_size_multiplier(equity)
        if risk_amount <= 0:
            return 0.0, "risk_brake"
        loss_per_unit = price * signal.stop_loss_pct
        qty_by_risk = risk_amount / loss_per_unit

        max_notional_by_leverage = equity * self.config.max_leverage
        max_notional_by_policy = equity * self.config.max_position_notional_pct
        max_notional = min(max_notional_by_leverage, max_notional_by_policy)
        target_notional = min(qty_by_risk * price, max_notional)
        remaining_notional = max(0.0, target_notional - max(0.0, existing_notional))
        if entry_fraction is not None:
            remaining_notional = min(remaining_notional, target_notional * max(0.0, min(1.0, entry_fraction)))

        qty = max(0.0, remaining_notional / price)
        if qty * price < self.config.min_order_notional:
            return 0.0, "below_min_notional"
        return qty, "ok"

    def on_trade_closed(self, net_pnl: float) -> None:
        if net_pnl < 0:
            self.consecutive_losses += 1
            loss_cooldown = self.config.cooldown_bars_after_loss * min(self.consecutive_losses, 3)
            self.cooldown_remaining = max(self.cooldown_remaining, loss_cooldown)
        else:
            self.consecutive_losses = 0

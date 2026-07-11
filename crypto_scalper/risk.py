from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR
from typing import Any, Iterable

from .binance_client import SymbolRules
from .config import RiskConfig
from .models import Candle, Direction, Signal


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


@dataclass(frozen=True)
class BacktestExecutionConfig:
    mode: str = "conservative"
    market_slippage_bps: float = 2.0
    stop_slippage_bps: float = 5.0
    take_profit_slippage_bps: float = 2.0
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0005
    funding_enabled: bool = False
    funding_default_rate: float = 0.0
    funding_rates_by_symbol: dict[str, tuple[FundingRate, ...]] | None = None
    dynamic_slippage_enabled: bool = False
    impact_coefficient_bps: float = 25.0
    impact_exponent: float = 0.5
    max_bar_participation_rate: float = 0.003
    min_partial_fill_ratio: float = 0.10


@dataclass(frozen=True)
class ExecutionFill:
    raw_price: float
    price: float
    fee: float
    fee_rate: float
    slippage_cost: float
    order_type: str
    liquidity: str
    participation_rate: float = 0.0


@dataclass(frozen=True)
class FundingRate:
    timestamp: datetime
    rate: float


@dataclass
class BacktestExecutionStats:
    same_bar_tp_sl_conflict_count: int = 0
    limit_touch_count: int = 0
    limit_filled_count: int = 0


def execution_config_from_live_config(
    config: Any,
    cost_experiment: str | None = None,
    mode: str | None = None,
) -> BacktestExecutionConfig:
    risk = getattr(config, "risk", config)
    estimated_fee_bps = float(getattr(risk, "estimated_fee_bps", getattr(risk, "fee_bps", 5.0)))
    estimated_slippage_bps = float(getattr(risk, "estimated_slippage_bps", getattr(risk, "slippage_bps", 2.0)))
    taker_fee_rate = float(getattr(risk, "taker_fee_rate", estimated_fee_bps / 10_000.0))
    maker_fee_rate = float(getattr(risk, "maker_fee_rate", min(taker_fee_rate, 0.0002)))
    market_slippage = float(getattr(risk, "market_slippage_bps", estimated_slippage_bps))
    stop_slippage = float(getattr(risk, "stop_slippage_bps", max(estimated_slippage_bps * 2.0, estimated_slippage_bps)))
    take_profit_slippage = float(getattr(risk, "take_profit_slippage_bps", estimated_slippage_bps))
    selected_mode = str(mode or getattr(risk, "backtest_mode", "conservative") or "conservative").lower()
    experiment = str(cost_experiment or getattr(risk, "cost_experiment", "full_cost") or "full_cost").lower()
    funding_enabled = bool(getattr(risk, "funding_enabled", False))
    funding_default_rate = float(getattr(risk, "funding_default_rate", 0.0))
    dynamic_slippage_enabled = bool(getattr(risk, "dynamic_slippage_enabled", False))
    impact_coefficient_bps = float(getattr(risk, "impact_coefficient_bps", 25.0))
    impact_exponent = float(getattr(risk, "impact_exponent", 0.5))
    max_bar_participation_rate = float(getattr(risk, "max_bar_participation_rate", 0.003))
    min_partial_fill_ratio = float(getattr(risk, "min_partial_fill_ratio", 0.10))

    if experiment == "no_cost":
        maker_fee_rate = taker_fee_rate = 0.0
        market_slippage = stop_slippage = take_profit_slippage = 0.0
        funding_enabled = False
    elif experiment == "fee_only":
        market_slippage = stop_slippage = take_profit_slippage = 0.0
        funding_enabled = False
    elif experiment.startswith("fee_slippage_") and experiment.endswith("bps"):
        value = experiment.removeprefix("fee_slippage_").removesuffix("bps")
        try:
            bps = float(value)
        except ValueError:
            bps = estimated_slippage_bps
        market_slippage = bps
        take_profit_slippage = bps
        stop_slippage = max(bps * 2.0, bps)
        funding_enabled = False
    elif experiment == "full_cost":
        pass
    elif experiment in {"conservative", "pessimistic", "optimistic"}:
        selected_mode = experiment

    if selected_mode == "pessimistic":
        selected_mode = "conservative"
    if selected_mode not in {"conservative", "optimistic", "neutral"}:
        selected_mode = "conservative"

    return BacktestExecutionConfig(
        mode=selected_mode,
        market_slippage_bps=max(0.0, market_slippage),
        stop_slippage_bps=max(0.0, stop_slippage),
        take_profit_slippage_bps=max(0.0, take_profit_slippage),
        maker_fee_rate=max(0.0, maker_fee_rate),
        taker_fee_rate=max(0.0, taker_fee_rate),
        funding_enabled=funding_enabled,
        funding_default_rate=funding_default_rate,
        dynamic_slippage_enabled=dynamic_slippage_enabled,
        impact_coefficient_bps=max(0.0, impact_coefficient_bps),
        impact_exponent=max(0.1, impact_exponent),
        max_bar_participation_rate=max(0.0, max_bar_participation_rate),
        min_partial_fill_ratio=max(0.0, min(1.0, min_partial_fill_ratio)),
    )


def market_entry_fill(
    config: BacktestExecutionConfig,
    rules: SymbolRules,
    direction: Direction,
    quantity: float,
    raw_open_price: float,
    bar_quote_volume: float | None = None,
) -> ExecutionFill:
    side = "buy" if direction == Direction.LONG else "sell"
    return _fill_with_slippage(config, rules, direction, side, quantity, raw_open_price, config.market_slippage_bps, "market", "taker", bar_quote_volume)


def market_exit_fill(
    config: BacktestExecutionConfig,
    rules: SymbolRules,
    direction: Direction,
    quantity: float,
    raw_price: float,
    order_type: str = "market",
    bar_quote_volume: float | None = None,
) -> ExecutionFill:
    side = "sell" if direction == Direction.LONG else "buy"
    slip_bps = config.market_slippage_bps
    normalized = order_type.lower()
    if "stop" in normalized:
        slip_bps = config.stop_slippage_bps
    elif "take_profit" in normalized or "take-profit" in normalized:
        slip_bps = config.take_profit_slippage_bps
    return _fill_with_slippage(config, rules, direction, side, quantity, raw_price, slip_bps, order_type, "taker", bar_quote_volume)


def capacity_limited_quantity(
    config: BacktestExecutionConfig,
    rules: SymbolRules,
    quantity: float,
    raw_price: float,
    bar_quote_volume: float,
) -> tuple[float, float, str]:
    requested = max(0.0, quantity)
    if requested <= 0:
        return 0.0, 0.0, "zero_quantity"
    if not config.dynamic_slippage_enabled or config.max_bar_participation_rate <= 0 or bar_quote_volume <= 0:
        return requested, 1.0, "full_fill"
    capacity_notional = bar_quote_volume * config.max_bar_participation_rate
    filled = conservative_quantity(rules, min(requested, capacity_notional / max(raw_price, 1e-12)))
    fill_ratio = filled / requested if requested > 0 else 0.0
    if fill_ratio < config.min_partial_fill_ratio:
        return 0.0, fill_ratio, "capacity_rejected"
    return filled, min(1.0, fill_ratio), "full_fill" if fill_ratio >= 0.999999 else "partial_fill"


def limit_order_filled(rules: SymbolRules, side: str, candle: Candle, limit_price: float) -> tuple[bool, bool]:
    tick = float(rules.price_tick)
    side = side.lower()
    if side == "buy":
        touched = candle.low <= limit_price
        filled = candle.low < limit_price - tick
    elif side == "sell":
        touched = candle.high >= limit_price
        filled = candle.high > limit_price + tick
    else:
        raise ValueError(f"unsupported side: {side}")
    return touched, filled


def limit_fill(
    config: BacktestExecutionConfig,
    rules: SymbolRules,
    direction: Direction,
    side: str,
    quantity: float,
    limit_price: float,
) -> ExecutionFill:
    rounded = conservative_price(rules, limit_price, side)
    notional = abs(quantity * rounded)
    fee = notional * config.maker_fee_rate
    return ExecutionFill(
        raw_price=limit_price,
        price=rounded,
        fee=fee,
        fee_rate=config.maker_fee_rate,
        slippage_cost=0.0,
        order_type="limit",
        liquidity="maker",
    )


def conservative_price(rules: SymbolRules, price: float, side: str) -> float:
    rounding = ROUND_CEILING if side.lower() == "buy" else ROUND_FLOOR
    return float(_round_decimal(price, rules.price_tick, rounding))


def conservative_quantity(rules: SymbolRules, quantity: float) -> float:
    return float(_round_decimal(quantity, rules.quantity_step, ROUND_DOWN))


def validate_order_size(rules: SymbolRules, quantity: float, price: float, min_notional: float = 0.0) -> str:
    if quantity <= 0:
        return "below_min_quantity"
    if Decimal(str(quantity)) < rules.min_quantity:
        return "below_min_quantity"
    notional = quantity * price
    required_notional = max(float(rules.min_notional), float(min_notional))
    if notional < required_notional:
        return "below_exchange_min_notional"
    return "ok"


def funding_cashflow(direction: Direction, notional: float, funding_rates: Iterable[float]) -> float:
    return sum(-direction.value * abs(notional) * float(rate) for rate in funding_rates)


def funding_rates_between(rates: Iterable[FundingRate], entry_time: datetime, exit_time: datetime) -> list[float]:
    return [item.rate for item in rates if entry_time < item.timestamp <= exit_time]


def _fill_with_slippage(
    config: BacktestExecutionConfig,
    rules: SymbolRules,
    direction: Direction,
    side: str,
    quantity: float,
    raw_price: float,
    slippage_bps: float,
    order_type: str,
    liquidity: str,
    bar_quote_volume: float | None = None,
) -> ExecutionFill:
    requested_notional = abs(quantity * raw_price)
    participation = 0.0
    effective_slippage_bps = max(0.0, slippage_bps)
    if config.dynamic_slippage_enabled and bar_quote_volume is not None and bar_quote_volume > 0:
        participation = requested_notional / bar_quote_volume
        effective_slippage_bps += config.impact_coefficient_bps * (max(0.0, participation) ** config.impact_exponent)
    slip = effective_slippage_bps / 10_000.0
    slipped = raw_price * (1.0 + slip) if side == "buy" else raw_price * (1.0 - slip)
    executed = conservative_price(rules, slipped, side)
    fee_rate = config.taker_fee_rate if liquidity == "taker" else config.maker_fee_rate
    notional = abs(quantity * executed)
    fee = notional * fee_rate
    raw_reference = raw_price
    if side == "buy":
        slip_cost = max(0.0, (executed - raw_reference) * quantity)
    else:
        slip_cost = max(0.0, (raw_reference - executed) * quantity)
    return ExecutionFill(
        raw_price=raw_price,
        price=executed,
        fee=fee,
        fee_rate=fee_rate,
        slippage_cost=slip_cost,
        order_type=order_type,
        liquidity=liquidity,
        participation_rate=participation,
    )


def _round_decimal(value: float, step: Decimal, rounding: str) -> Decimal:
    decimal_value = Decimal(str(value))
    decimal_step = Decimal(str(step))
    if decimal_step <= 0:
        return decimal_value
    return (decimal_value / decimal_step).to_integral_value(rounding=rounding) * decimal_step

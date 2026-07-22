from __future__ import annotations

import bisect
import math
from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from .indicators import atr
from .live_portfolio_backtest import _inferred_symbol_rules
from .models import Candle, Direction
from .mtf_structure_derivatives import (
    FeatureBundle,
    FundingPoint,
    RawSetup,
    SignalFilterConfig,
    TrendState,
    select_setups,
)
from .risk import (
    BacktestExecutionConfig,
    conservative_quantity,
    market_entry_fill,
    market_exit_fill,
)
from .volatility_breakout_optimize import (
    CompactSeries,
    load_compact_execution_series,
    minute_datetime,
    minute_token,
)


@dataclass(frozen=True)
class ExitConfig:
    min_stop_atr: float = 0.75
    max_stop_atr: float = 3.0
    stop_buffer_atr: float = 0.25
    max_stop_pct: float = 0.08
    tp1_r: float = 1.0
    tp1_fraction: float = 0.50
    tp2_r: float = 2.5
    tp2_fraction: float = 0.25
    breakeven_after_tp1: bool = True
    breakeven_offset_r: float = 0.0
    locked_r_after_tp2: float = 0.75
    trailing_activation_r: float = 2.0
    trailing_atr_multiple: float = 2.0
    trend_exit_mode: str = "hybrid"
    trend_exit_confirm_bars: int = 2
    max_holding_minutes: int = 2_880

    def validate(self) -> None:
        if self.min_stop_atr <= 0.0 or self.max_stop_atr < self.min_stop_atr:
            raise ValueError("invalid ATR stop bounds")
        if self.max_stop_pct <= 0.0:
            raise ValueError("max_stop_pct must be positive")
        if self.tp1_r <= 0.0 or self.tp2_r <= self.tp1_r:
            raise ValueError("tp2_r must exceed positive tp1_r")
        if not 0.0 < self.tp1_fraction < 1.0:
            raise ValueError("tp1_fraction must be in (0, 1)")
        if not 0.0 <= self.tp2_fraction < 1.0:
            raise ValueError("tp2_fraction must be in [0, 1)")
        if self.tp1_fraction + self.tp2_fraction >= 1.0:
            raise ValueError("partial exits must leave a runner")
        if self.trend_exit_mode not in {"fast", "slow", "structure", "hybrid"}:
            raise ValueError("unsupported trend_exit_mode")
        if self.trend_exit_confirm_bars <= 0 or self.max_holding_minutes <= 0:
            raise ValueError("exit confirmations and holding time must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioConfig:
    initial_equity: float = 200.0
    risk_per_trade_pct: float = 0.02
    max_trade_risk_pct: float = 0.05
    max_open_positions: int = 2
    max_daily_trades: int = 4
    symbol_cooldown_minutes: int = 360
    max_notional_multiple: float = 6.0
    hard_drawdown_stop_pct: float = 0.50
    compound: bool = True

    def validate(self) -> None:
        if self.initial_equity <= 0.0:
            raise ValueError("initial_equity must be positive")
        if not 0.0 < self.risk_per_trade_pct <= self.max_trade_risk_pct:
            raise ValueError("risk_per_trade_pct must be positive and capped")
        if self.max_open_positions <= 0 or self.max_daily_trades <= 0:
            raise ValueError("position and daily trade limits must be positive")
        if self.max_notional_multiple <= 0.0:
            raise ValueError("max_notional_multiple must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtrTimeline:
    available_minutes: tuple[int, ...]
    values: tuple[float, ...]

    def value_at(self, minute: int, fallback: float) -> float:
        index = bisect.bisect_right(self.available_minutes, minute) - 1
        if index < 0:
            return fallback
        value = self.values[index]
        return value if math.isfinite(value) and value > 0.0 else fallback


@dataclass(frozen=True)
class UnitExitLeg:
    minute: int
    phase: int
    fraction: float
    raw_price: float
    fill_price: float
    gross_pnl_per_unit: float
    fee_per_unit: float
    funding_per_unit: float
    cashflow_per_unit: float
    slippage_per_unit: float
    reason: str


@dataclass(frozen=True)
class UnitOutcome:
    setup: RawSetup
    entry_minute: int
    raw_entry_price: float
    entry_price: float
    entry_fee_per_unit: float
    entry_slippage_per_unit: float
    initial_stop_price: float
    initial_risk_price: float
    unit_risk_cash: float
    legs: tuple[UnitExitLeg, ...]
    max_mfe_r: float
    max_mae_r: float

    @property
    def exit_minute(self) -> int:
        return self.legs[-1].minute

    @property
    def exit_reason(self) -> str:
        return self.legs[-1].reason

    @property
    def net_pnl_per_unit(self) -> float:
        return -self.entry_fee_per_unit + sum(item.cashflow_per_unit for item in self.legs)

    @property
    def total_fees_per_unit(self) -> float:
        return self.entry_fee_per_unit + sum(item.fee_per_unit for item in self.legs)

    @property
    def total_funding_per_unit(self) -> float:
        return sum(item.funding_per_unit for item in self.legs)

    @property
    def total_slippage_per_unit(self) -> float:
        return self.entry_slippage_per_unit + sum(item.slippage_per_unit for item in self.legs)


@dataclass
class ActiveTrade:
    outcome: UnitOutcome
    quantity: float
    remaining_fraction: float = 1.0
    next_leg: int = 0
    realized_exit_cash: float = 0.0


@dataclass(frozen=True)
class TradeResult:
    event_id: str
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    quantity: float
    entry_price: float
    initial_stop_price: float
    net_pnl: float
    fees: float
    funding: float
    slippage: float
    exit_reason: str
    max_mfe_r: float
    max_mae_r: float
    quality_score: float
    leg_count: int

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["entry_time"] = self.entry_time.isoformat()
        value["exit_time"] = self.exit_time.isoformat()
        return value


@dataclass(frozen=True)
class BacktestResult:
    start: datetime
    end: datetime
    resolution: str
    initial_equity: float
    final_equity: float
    net_profit: float
    return_pct: float
    profit_factor: float
    win_rate_pct: float
    max_drawdown_pct: float
    trade_count: int
    long_count: int
    short_count: int
    gross_profit: float
    gross_loss: float
    fees: float
    funding: float
    slippage: float
    rejected: dict[str, int]
    by_symbol: dict[str, dict[str, float]]
    by_exit_reason: dict[str, dict[str, float]]
    by_month: dict[str, dict[str, float]]
    trades: tuple[TradeResult, ...] = field(repr=False)

    def as_dict(self, include_trades: bool = False) -> dict[str, Any]:
        value = asdict(self)
        value["start"] = self.start.isoformat()
        value["end"] = self.end.isoformat()
        if not include_trades:
            value.pop("trades", None)
        else:
            value["trades"] = [item.as_dict() for item in self.trades]
        return value


def default_execution_config(bundle: FeatureBundle) -> BacktestExecutionConfig:
    return BacktestExecutionConfig(
        mode="conservative",
        market_slippage_bps=2.0,
        stop_slippage_bps=5.0,
        take_profit_slippage_bps=2.0,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        funding_enabled=True,
        funding_rates_by_symbol=None,
    )


def build_atr_timelines(bundle: FeatureBundle, period: int = 14) -> dict[str, AtrTimeline]:
    output: dict[str, AtrTimeline] = {}
    for symbol, rows in bundle.candles_15m.items():
        values = atr(list(rows), period)
        output[symbol] = AtrTimeline(
            available_minutes=tuple(minute_token(item.timestamp) + 15 for item in rows),
            values=tuple(values),
        )
    return output


def build_15m_execution_series(
    bundle: FeatureBundle,
    start: datetime,
    end: datetime,
) -> dict[str, CompactSeries]:
    output: dict[str, CompactSeries] = {}
    for symbol, rows in bundle.candles_15m.items():
        selected = [item for item in rows if start <= item.timestamp < end]
        output[symbol] = CompactSeries(
            array("q", (minute_token(item.timestamp) for item in selected)),
            array("d", (item.open for item in selected)),
            array("d", (item.high for item in selected)),
            array("d", (item.low for item in selected)),
            array("d", (item.close for item in selected)),
            array("d", (item.volume for item in selected)),
        )
    return output


def load_1m_execution_series(
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
    data_dir: str = "data/binance_1m_365d_top100",
) -> dict[str, CompactSeries]:
    root = Path(data_dir)
    output: dict[str, CompactSeries] = {}
    for symbol in symbols:
        matches = sorted(root.glob(f"{symbol}_1m_*.csv"))
        if not matches:
            raise FileNotFoundError(f"missing 1m execution data for {symbol} in {root}")
        output[symbol] = load_compact_execution_series(matches[-1], start, end)
    return output


def inferred_rules(bundle: FeatureBundle) -> dict[str, Any]:
    return {
        symbol: _inferred_symbol_rules(symbol, list(bundle.candles_15m[symbol]))
        for symbol in bundle.symbols
    }


def simulate_unit_outcome(
    setup: RawSetup,
    series: CompactSeries,
    states: dict[int, TrendState],
    atr_timeline: AtrTimeline,
    funding: tuple[FundingPoint, ...],
    rules: Any,
    exit_config: ExitConfig,
    execution: BacktestExecutionConfig,
) -> UnitOutcome | None:
    exit_config.validate()
    entry_minute = minute_token(setup.signal_available_time)
    entry_index = series.index_at(entry_minute)
    if entry_index is None:
        return None
    raw_entry = float(series.opens[entry_index])
    direction = setup.direction
    entry_fill = market_entry_fill(
        execution,
        rules,
        direction,
        1.0,
        raw_entry,
        float(series.volumes[entry_index]) * raw_entry,
    )
    structure_stop = setup.sweep_price - direction.value * exit_config.stop_buffer_atr * setup.atr_15m
    structure_distance = direction.value * (entry_fill.price - structure_stop)
    min_distance = exit_config.min_stop_atr * setup.atr_15m
    max_distance = min(
        exit_config.max_stop_atr * setup.atr_15m,
        exit_config.max_stop_pct * entry_fill.price,
    )
    max_distance = max(min_distance, max_distance)
    stop_distance = min(max_distance, max(min_distance, structure_distance))
    raw_stop = entry_fill.price - direction.value * stop_distance
    if raw_stop <= 0.0:
        return None
    unit_stop = market_exit_fill(execution, rules, direction, 1.0, raw_stop, "stop_market")
    unit_stop_net = (
        direction.value * (unit_stop.price - entry_fill.price)
        - entry_fill.fee
        - unit_stop.fee
    )
    unit_risk_cash = -unit_stop_net
    if unit_risk_cash <= 0.0:
        return None
    risk_price = abs(entry_fill.price - raw_stop)
    tp1 = entry_fill.price + direction.value * risk_price * exit_config.tp1_r
    tp2 = entry_fill.price + direction.value * risk_price * exit_config.tp2_r
    current_stop = raw_stop
    best_price = entry_fill.price
    worst_price = entry_fill.price
    remaining = 1.0
    tp1_done = False
    tp2_done = False
    invalid_count = 0
    legs: list[UnitExitLeg] = []

    funding_times = [item.timestamp for item in funding]
    funding_rates = [item.rate for item in funding]

    def add_leg(
        fraction: float,
        index: int,
        raw_price: float,
        order_type: str,
        reason: str,
        phase: int,
    ) -> None:
        nonlocal remaining
        fraction = min(remaining, max(0.0, fraction))
        if fraction <= 1e-12:
            return
        fill = market_exit_fill(
            execution,
            rules,
            direction,
            fraction,
            raw_price,
            order_type,
            float(series.volumes[index]) * max(raw_price, 1e-12),
        )
        exit_time = minute_datetime(int(series.minutes[index]))
        left = bisect.bisect_right(funding_times, setup.signal_available_time)
        right = bisect.bisect_right(funding_times, exit_time)
        rate_sum = sum(funding_rates[left:right]) if execution.funding_enabled else 0.0
        funding_cash = -direction.value * entry_fill.price * fraction * rate_sum
        gross = direction.value * fraction * (fill.price - entry_fill.price)
        cashflow = gross - fill.fee + funding_cash
        legs.append(
            UnitExitLeg(
                minute=int(series.minutes[index]),
                phase=phase,
                fraction=fraction,
                raw_price=raw_price,
                fill_price=fill.price,
                gross_pnl_per_unit=gross,
                fee_per_unit=fill.fee,
                funding_per_unit=funding_cash,
                cashflow_per_unit=cashflow,
                slippage_per_unit=fill.slippage_cost,
                reason=reason,
            )
        )
        remaining = max(0.0, remaining - fraction)

    for index in range(entry_index, len(series.minutes)):
        minute = int(series.minutes[index])
        raw_open = float(series.opens[index])
        high = float(series.highs[index])
        low = float(series.lows[index])
        close = float(series.closes[index])
        if index > entry_index:
            state = states.get(minute)
            if state is not None:
                invalid_count = (
                    invalid_count + 1
                    if state.invalid(direction, exit_config.trend_exit_mode)
                    else 0
                )
            if minute - entry_minute >= exit_config.max_holding_minutes:
                add_leg(remaining, index, raw_open, "market", "max_holding", 0)
            elif invalid_count >= exit_config.trend_exit_confirm_bars:
                add_leg(remaining, index, raw_open, "market", "trend_end", 0)
            if remaining <= 1e-12:
                break

        gap_stop = (
            raw_open <= current_stop
            if direction == Direction.LONG
            else raw_open >= current_stop
        )
        if gap_stop:
            add_leg(remaining, index, raw_open, "stop_market", "atr_stop", 0)
            break

        target_done_this_bar = False
        if not tp1_done:
            gap_target = raw_open >= tp1 if direction == Direction.LONG else raw_open <= tp1
            if gap_target:
                add_leg(exit_config.tp1_fraction, index, raw_open, "take_profit_market", "tp1", 0)
                tp1_done = True
                target_done_this_bar = True
        elif not tp2_done:
            gap_target = raw_open >= tp2 if direction == Direction.LONG else raw_open <= tp2
            if gap_target:
                add_leg(exit_config.tp2_fraction, index, raw_open, "take_profit_market", "tp2", 0)
                tp2_done = True
                target_done_this_bar = True

        if remaining <= 1e-12:
            break
        intrabar_stop = low <= current_stop if direction == Direction.LONG else high >= current_stop
        if intrabar_stop:
            add_leg(remaining, index, current_stop, "stop_market", "atr_stop", 1)
            break

        if not target_done_this_bar:
            if not tp1_done:
                target_hit = high >= tp1 if direction == Direction.LONG else low <= tp1
                if target_hit:
                    add_leg(exit_config.tp1_fraction, index, tp1, "take_profit_market", "tp1", 1)
                    tp1_done = True
                    target_done_this_bar = True
            elif not tp2_done:
                target_hit = high >= tp2 if direction == Direction.LONG else low <= tp2
                if target_hit:
                    add_leg(exit_config.tp2_fraction, index, tp2, "take_profit_market", "tp2", 1)
                    tp2_done = True
                    target_done_this_bar = True

        best_price = max(best_price, high) if direction == Direction.LONG else min(best_price, low)
        worst_price = min(worst_price, low) if direction == Direction.LONG else max(worst_price, high)
        if tp1_done and exit_config.breakeven_after_tp1:
            breakeven = entry_fill.price + direction.value * risk_price * exit_config.breakeven_offset_r
            current_stop = max(current_stop, breakeven) if direction == Direction.LONG else min(current_stop, breakeven)
        if tp2_done:
            locked = entry_fill.price + direction.value * risk_price * exit_config.locked_r_after_tp2
            current_stop = max(current_stop, locked) if direction == Direction.LONG else min(current_stop, locked)
        favorable_r = direction.value * (best_price - entry_fill.price) / max(risk_price, 1e-12)
        if favorable_r >= exit_config.trailing_activation_r:
            current_atr = atr_timeline.value_at(minute, setup.atr_15m)
            trail = best_price - direction.value * exit_config.trailing_atr_multiple * current_atr
            current_stop = max(current_stop, trail) if direction == Direction.LONG else min(current_stop, trail)

        if index == len(series.minutes) - 1 and remaining > 1e-12:
            add_leg(remaining, index, close, "market", "end_of_test", 2)

    if remaining > 1e-9 or not legs:
        return None
    mfe = direction.value * (best_price - entry_fill.price) / max(risk_price, 1e-12)
    mae = -direction.value * (worst_price - entry_fill.price) / max(risk_price, 1e-12)
    return UnitOutcome(
        setup=setup,
        entry_minute=entry_minute,
        raw_entry_price=raw_entry,
        entry_price=entry_fill.price,
        entry_fee_per_unit=entry_fill.fee,
        entry_slippage_per_unit=entry_fill.slippage_cost,
        initial_stop_price=raw_stop,
        initial_risk_price=risk_price,
        unit_risk_cash=unit_risk_cash,
        legs=tuple(legs),
        max_mfe_r=max(0.0, mfe),
        max_mae_r=max(0.0, mae),
    )


def run_backtest(
    bundle: FeatureBundle,
    signal_config: SignalFilterConfig,
    exit_config: ExitConfig,
    portfolio_config: PortfolioConfig,
    start: datetime,
    end: datetime,
    *,
    resolution: str = "15m",
    execution_series: dict[str, CompactSeries] | None = None,
    execution: BacktestExecutionConfig | None = None,
    atr_timelines: dict[str, AtrTimeline] | None = None,
    rules: dict[str, Any] | None = None,
    include_trades: bool = True,
) -> BacktestResult:
    exit_config.validate()
    portfolio_config.validate()
    if resolution not in {"15m", "1m"}:
        raise ValueError("resolution must be 15m or 1m")
    execution = execution or default_execution_config(bundle)
    atr_timelines = atr_timelines or build_atr_timelines(bundle)
    rules = rules or inferred_rules(bundle)
    if execution_series is None:
        if resolution != "15m":
            raise ValueError("1m execution_series must be loaded explicitly")
        execution_series = build_15m_execution_series(bundle, start, end)
    selected = select_setups(bundle.raw_setups, signal_config, start, end)
    outcomes: dict[str, UnitOutcome] = {}
    rejected: Counter[str] = Counter()
    for setup in selected:
        outcome = simulate_unit_outcome(
            setup,
            execution_series[setup.symbol],
            bundle.trend_states[setup.symbol],
            atr_timelines[setup.symbol],
            bundle.funding[setup.symbol],
            rules[setup.symbol],
            exit_config,
            execution,
        )
        if outcome is None:
            rejected["no_execution_path"] += 1
        else:
            outcomes[setup.event_id] = outcome

    by_entry: dict[int, list[UnitOutcome]] = defaultdict(list)
    for outcome in outcomes.values():
        by_entry[outcome.entry_minute].append(outcome)
    for rows in by_entry.values():
        rows.sort(key=lambda item: (-item.setup.quality_score, item.setup.symbol, item.setup.event_id))

    reference = execution_series[bundle.symbols[0]]
    start_minute = minute_token(start)
    end_minute = minute_token(end)
    timeline = [int(value) for value in reference.minutes if start_minute <= value < end_minute]
    cash = portfolio_config.initial_equity
    peak_equity = cash
    max_drawdown = 0.0
    hard_stop = False
    active: dict[str, ActiveTrade] = {}
    trades: list[TradeResult] = []
    last_exit: dict[str, int] = {}
    daily_entries: Counter[date] = Counter()

    def mark_equity(minute: int, use_open: bool = False) -> float:
        value = cash
        for item in active.values():
            series = execution_series[item.outcome.setup.symbol]
            index = series.index_at(minute)
            if index is None:
                index = series.last_index_at_or_before(minute)
            if index is None:
                continue
            price = float(series.opens[index] if use_open else series.closes[index])
            remaining_qty = item.quantity * item.remaining_fraction
            unrealized = item.outcome.setup.direction.value * remaining_qty * (
                price - item.outcome.entry_price
            )
            exit_fee_estimate = abs(remaining_qty * price) * execution.taker_fee_rate
            value += unrealized - exit_fee_estimate
        return value

    def realize(minute: int, phases: set[int], include_past: bool = False) -> None:
        nonlocal cash
        completed: list[str] = []
        for event_id, item in list(active.items()):
            while item.next_leg < len(item.outcome.legs):
                leg = item.outcome.legs[item.next_leg]
                due = leg.minute < minute if include_past else False
                due = due or (leg.minute == minute and leg.phase in phases)
                if not due:
                    break
                leg_cash = item.quantity * leg.cashflow_per_unit
                cash += leg_cash
                item.realized_exit_cash += leg_cash
                item.remaining_fraction = max(0.0, item.remaining_fraction - leg.fraction)
                item.next_leg += 1
            if item.next_leg >= len(item.outcome.legs):
                outcome = item.outcome
                net = -item.quantity * outcome.entry_fee_per_unit + item.realized_exit_cash
                trades.append(
                    TradeResult(
                        event_id=event_id,
                        symbol=outcome.setup.symbol,
                        direction=outcome.setup.direction.name,
                        entry_time=minute_datetime(outcome.entry_minute),
                        exit_time=minute_datetime(outcome.exit_minute),
                        quantity=item.quantity,
                        entry_price=outcome.entry_price,
                        initial_stop_price=outcome.initial_stop_price,
                        net_pnl=net,
                        fees=item.quantity * outcome.total_fees_per_unit,
                        funding=item.quantity * outcome.total_funding_per_unit,
                        slippage=item.quantity * outcome.total_slippage_per_unit,
                        exit_reason=outcome.exit_reason,
                        max_mfe_r=outcome.max_mfe_r,
                        max_mae_r=outcome.max_mae_r,
                        quality_score=outcome.setup.quality_score,
                        leg_count=len(outcome.legs),
                    )
                )
                last_exit[outcome.setup.symbol] = outcome.exit_minute
                completed.append(event_id)
        for event_id in completed:
            active.pop(event_id, None)

    for minute in timeline:
        realize(minute, set(), include_past=True)
        realize(minute, {0})
        entry_equity = mark_equity(minute, use_open=True)
        current_drawdown = (peak_equity - entry_equity) / peak_equity if peak_equity > 0 else 1.0
        if current_drawdown >= portfolio_config.hard_drawdown_stop_pct:
            hard_stop = True
        for outcome in by_entry.get(minute, ()):
            symbol = outcome.setup.symbol
            if hard_stop:
                rejected["hard_drawdown_stop"] += 1
                continue
            if len(active) >= portfolio_config.max_open_positions:
                rejected["max_open_positions"] += 1
                continue
            if any(item.outcome.setup.symbol == symbol for item in active.values()):
                rejected["symbol_already_open"] += 1
                continue
            previous_exit = last_exit.get(symbol)
            if previous_exit is not None and minute - previous_exit < portfolio_config.symbol_cooldown_minutes:
                rejected["symbol_cooldown"] += 1
                continue
            trade_day = minute_datetime(minute).date()
            if daily_entries[trade_day] >= portfolio_config.max_daily_trades:
                rejected["max_daily_trades"] += 1
                continue
            entry_equity = mark_equity(minute, use_open=True)
            if entry_equity <= 0.0:
                rejected["equity_depleted"] += 1
                continue
            risk_base = entry_equity if portfolio_config.compound else portfolio_config.initial_equity
            risk_pct = min(portfolio_config.risk_per_trade_pct, portfolio_config.max_trade_risk_pct)
            requested = risk_base * risk_pct / outcome.unit_risk_cash
            current_notional = sum(
                item.quantity * item.remaining_fraction * item.outcome.entry_price
                for item in active.values()
            )
            notional_headroom = max(
                0.0,
                entry_equity * portfolio_config.max_notional_multiple - current_notional,
            )
            requested = min(requested, notional_headroom / max(outcome.entry_price, 1e-12))
            quantity = conservative_quantity(rules[symbol], requested)
            min_notional = max(5.0, float(rules[symbol].min_notional))
            if quantity <= 0.0 or quantity * outcome.entry_price < min_notional:
                rejected["below_min_order"] += 1
                continue
            cash -= quantity * outcome.entry_fee_per_unit
            active[outcome.setup.event_id] = ActiveTrade(outcome=outcome, quantity=quantity)
            daily_entries[trade_day] += 1
        realize(minute, {1, 2})
        equity = mark_equity(minute)
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0.0:
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)

    if timeline:
        realize(timeline[-1] + 1, {0, 1, 2}, include_past=True)
    trades.sort(key=lambda item: (item.exit_time, item.symbol, item.event_id))
    gross_profit = sum(max(0.0, item.net_pnl) for item in trades)
    gross_loss = sum(min(0.0, item.net_pnl) for item in trades)
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0.0 else (math.inf if gross_profit > 0 else 0.0)
    final_equity = cash
    grouped_symbol = _group_trade_stats(trades, lambda item: item.symbol)
    grouped_exit = _group_trade_stats(trades, lambda item: item.exit_reason)
    grouped_month = _group_trade_stats(trades, lambda item: item.exit_time.strftime("%Y-%m"))
    return BacktestResult(
        start=start,
        end=end,
        resolution=resolution,
        initial_equity=portfolio_config.initial_equity,
        final_equity=final_equity,
        net_profit=final_equity - portfolio_config.initial_equity,
        return_pct=(final_equity / portfolio_config.initial_equity - 1.0) * 100.0,
        profit_factor=profit_factor,
        win_rate_pct=(sum(item.net_pnl > 0.0 for item in trades) / len(trades) * 100.0) if trades else 0.0,
        max_drawdown_pct=max_drawdown * 100.0,
        trade_count=len(trades),
        long_count=sum(item.direction == "LONG" for item in trades),
        short_count=sum(item.direction == "SHORT" for item in trades),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        fees=sum(item.fees for item in trades),
        funding=sum(item.funding for item in trades),
        slippage=sum(item.slippage for item in trades),
        rejected=dict(rejected),
        by_symbol=grouped_symbol,
        by_exit_reason=grouped_exit,
        by_month=grouped_month,
        trades=tuple(trades) if include_trades else (),
    )


def _group_trade_stats(
    trades: Iterable[TradeResult],
    key_function: Any,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[TradeResult]] = defaultdict(list)
    for trade in trades:
        grouped[str(key_function(trade))].append(trade)
    return {
        key: {
            "trades": len(rows),
            "net_profit": sum(item.net_pnl for item in rows),
            "win_rate_pct": sum(item.net_pnl > 0.0 for item in rows) / len(rows) * 100.0,
        }
        for key, rows in sorted(grouped.items())
    }

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Iterable

from .binance_client import SymbolRules
from .models import Direction
from .risk import BacktestExecutionConfig, conservative_quantity
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    OpenPosition,
    PortfolioSearchConfig,
    _candidate_sort_key,
    _entry_position,
    _exit_position,
    _mark_equity,
    _market_filter_reject_reason,
    _position_current_r,
    _raw_breakeven_stop,
    _raw_stop_for_full_cost_r,
    _raw_target_for_full_cost_r,
    minute_datetime,
    minute_token,
    summarize_result,
)


EXIT_PROTECTION_STRATEGY_NAME = "dual_thrust_volatility_breakout_v3_exit_protection"
EXIT_PROTECTION_VERSION = "v3_independent_exit_protection_research"


@dataclass(frozen=True)
class ExitProtectionConfig:
    """Exit overlay for research; all-zero values reproduce frozen v2."""

    breakeven_trigger_r: float = 0.0
    profit_giveback_activation_r: float = 0.0
    profit_giveback_r: float = 0.0
    partial_take_profit_r: float = 0.0
    partial_take_profit_fraction: float = 0.0
    move_stop_to_breakeven_after_partial: bool = False

    def validate(self) -> None:
        if min(
            self.breakeven_trigger_r,
            self.profit_giveback_activation_r,
            self.profit_giveback_r,
            self.partial_take_profit_r,
            self.partial_take_profit_fraction,
        ) < 0.0:
            raise ValueError("exit-protection parameters cannot be negative")
        if self.partial_take_profit_fraction >= 1.0:
            raise ValueError("partial_take_profit_fraction must be below 1")
        partial_enabled = self.partial_take_profit_r > 0.0 or self.partial_take_profit_fraction > 0.0
        if partial_enabled and not (
            self.partial_take_profit_r > 0.0
            and 0.0 < self.partial_take_profit_fraction < 1.0
        ):
            raise ValueError("partial take profit requires a positive R target and fraction in (0, 1)")
        giveback_enabled = self.profit_giveback_activation_r > 0.0 or self.profit_giveback_r > 0.0
        if giveback_enabled and not (
            self.profit_giveback_activation_r > 0.0 and self.profit_giveback_r > 0.0
        ):
            raise ValueError("profit giveback requires positive activation and giveback R")

    @property
    def partial_enabled(self) -> bool:
        return self.partial_take_profit_r > 0.0 and self.partial_take_profit_fraction > 0.0


@dataclass
class ProtectedPosition:
    position: OpenPosition
    original_quantity: float
    original_risk_budget: float
    partial_target_price: float
    partial_taken: bool = False
    realized_legs: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "stop_loss"


def _protected_position(
    position: OpenPosition,
    config: ExitProtectionConfig,
    execution: BacktestExecutionConfig,
) -> ProtectedPosition:
    partial_target = 0.0
    if config.partial_enabled:
        partial_target = _raw_target_for_full_cost_r(
            position.candidate.signal.direction,
            position.entry_price,
            position.entry_fee / position.quantity,
            position.unit_risk,
            config.partial_take_profit_r,
            execution,
        )
    return ProtectedPosition(
        position=position,
        original_quantity=position.quantity,
        original_risk_budget=position.risk_budget,
        partial_target_price=partial_target,
    )


def _raise_stop(
    protected: ProtectedPosition,
    raw_stop: float,
    reason: str,
) -> None:
    position = protected.position
    direction = position.candidate.signal.direction
    improves = raw_stop > position.stop_price if direction == Direction.LONG else raw_stop < position.stop_price
    if improves:
        position.stop_price = raw_stop
        protected.stop_reason = reason


def _take_partial(
    protected: ProtectedPosition,
    minute: int,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
    config: ExitProtectionConfig,
) -> float:
    position = protected.position
    requested = protected.original_quantity * config.partial_take_profit_fraction
    partial_quantity = conservative_quantity(rules, requested)
    remaining = position.quantity - partial_quantity
    if (
        partial_quantity <= 0.0
        or remaining <= 0.0
        or remaining * position.entry_price < max(5.0, float(rules.min_notional))
    ):
        return 0.0

    ratio = partial_quantity / position.quantity
    leg_position = replace(
        position,
        quantity=partial_quantity,
        entry_fee=position.entry_fee * ratio,
        entry_slippage=position.entry_slippage * ratio,
        risk_budget=position.risk_budget * ratio,
    )
    trade, cash_delta = _exit_position(
        leg_position,
        protected.partial_target_price,
        "partial_take_profit",
        minute,
        execution,
        rules,
    )
    trade["leg_type"] = "partial_take_profit"
    protected.realized_legs.append(trade)
    position.quantity -= partial_quantity
    position.entry_fee -= leg_position.entry_fee
    position.entry_slippage -= leg_position.entry_slippage
    position.risk_budget -= leg_position.risk_budget
    protected.partial_taken = True
    return cash_delta


def _aggregate_trade(
    protected: ProtectedPosition,
    final_leg: dict[str, Any],
) -> dict[str, Any]:
    if not protected.realized_legs:
        result = dict(final_leg)
        result.update(
            {
                "partial_exit_count": 0,
                "partial_realized_net_pnl": 0.0,
                "partial_legs": [],
            }
        )
        return result

    legs = [*protected.realized_legs, final_leg]
    result = dict(final_leg)
    for key in ("gross_pnl", "fee", "slippage", "funding", "net_pnl"):
        result[key] = sum(float(leg[key]) for leg in legs)
    result["quantity"] = protected.original_quantity
    result["notional"] = protected.original_quantity * protected.position.entry_price
    result["risk_usdt"] = protected.original_risk_budget
    result["pnl_r"] = result["net_pnl"] / max(protected.original_risk_budget, 1e-12)
    result["mfe_r"] = protected.position.max_mfe_r
    result["mae_r"] = protected.position.max_mae_r
    result["partial_exit_count"] = len(protected.realized_legs)
    result["partial_realized_net_pnl"] = sum(
        float(leg["net_pnl"]) for leg in protected.realized_legs
    )
    result["partial_legs"] = protected.realized_legs
    return result


def _close_protected_position(
    protected: ProtectedPosition,
    raw_exit: float,
    reason: str,
    minute: int,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
) -> tuple[dict[str, Any], float]:
    final_leg, cash_delta = _exit_position(
        protected.position,
        raw_exit,
        reason,
        minute,
        execution,
        rules,
    )
    return _aggregate_trade(protected, final_leg), cash_delta


def _process_protected_bar(
    protected: ProtectedPosition,
    minute: int,
    series: CompactSeries,
    index: int,
    signal_config: VolatilityBreakoutConfig,
    exit_config: ExitProtectionConfig,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
) -> tuple[dict[str, Any] | None, float]:
    position = protected.position
    direction = position.candidate.signal.direction
    open_price = float(series.opens[index])
    high = float(series.highs[index])
    low = float(series.lows[index])

    gap_stop = open_price <= position.stop_price if direction == Direction.LONG else open_price >= position.stop_price
    if gap_stop:
        reason = f"{protected.stop_reason}_gap"
        trade, cash_delta = _close_protected_position(
            protected, open_price, reason, minute, execution, rules
        )
        return trade, cash_delta

    holding_minutes = minute - position.entry_minute
    current_open_r = _position_current_r(position, open_price, execution, rules)
    if (
        signal_config.fail_fast_minutes > 0
        and holding_minutes >= signal_config.fail_fast_minutes
        and position.max_mfe_r < signal_config.fail_fast_min_mfe_r
        and current_open_r <= signal_config.fail_fast_max_current_r
    ):
        trade, cash_delta = _close_protected_position(
            protected, open_price, "fail_fast", minute, execution, rules
        )
        return trade, cash_delta

    if holding_minutes >= signal_config.max_holding_minutes:
        extension_enabled = signal_config.extended_holding_minutes > signal_config.max_holding_minutes
        if extension_enabled and not position.extension_qualified:
            position.extension_qualified = (
                current_open_r >= signal_config.extension_min_current_r
                and position.max_mfe_r >= signal_config.extension_min_mfe_r
            )
        if not position.extension_qualified:
            trade, cash_delta = _close_protected_position(
                protected, open_price, "time_stop", minute, execution, rules
            )
            return trade, cash_delta
        if holding_minutes >= signal_config.extended_holding_minutes:
            trade, cash_delta = _close_protected_position(
                protected, open_price, "extended_time_stop", minute, execution, rules
            )
            return trade, cash_delta

    stop_hit = low <= position.stop_price if direction == Direction.LONG else high >= position.stop_price
    target_hit = high >= position.take_profit_price if direction == Direction.LONG else low <= position.take_profit_price
    if stop_hit:
        trade, cash_delta = _close_protected_position(
            protected, position.stop_price, protected.stop_reason, minute, execution, rules
        )
        return trade, cash_delta
    if target_hit:
        trade, cash_delta = _close_protected_position(
            protected, position.take_profit_price, "take_profit", minute, execution, rules
        )
        return trade, cash_delta

    cash_delta = 0.0
    if exit_config.partial_enabled and not protected.partial_taken:
        partial_hit = (
            high >= protected.partial_target_price
            if direction == Direction.LONG
            else low <= protected.partial_target_price
        )
        if partial_hit:
            cash_delta += _take_partial(protected, minute, execution, rules, exit_config)

    favorable = high if direction == Direction.LONG else low
    adverse = low if direction == Direction.LONG else high
    if direction == Direction.LONG:
        position.best_price = max(position.best_price, favorable)
        position.worst_price = min(position.worst_price, adverse)
    else:
        position.best_price = min(position.best_price, favorable)
        position.worst_price = max(position.worst_price, adverse)
    position.max_mfe_r = max(
        position.max_mfe_r,
        _position_current_r(position, favorable, execution, rules),
    )
    position.max_mae_r = min(
        position.max_mae_r,
        _position_current_r(position, adverse, execution, rules),
    )

    if (
        exit_config.profit_giveback_activation_r > 0.0
        and position.max_mfe_r >= exit_config.profit_giveback_activation_r
    ):
        locked_r = max(0.0, position.max_mfe_r - exit_config.profit_giveback_r)
        _raise_stop(
            protected,
            _raw_stop_for_full_cost_r(
                direction,
                position.entry_price,
                position.entry_fee / position.quantity,
                position.unit_risk,
                locked_r,
                execution,
            ),
            "profit_giveback_stop",
        )

    if exit_config.breakeven_trigger_r > 0.0 and position.max_mfe_r >= exit_config.breakeven_trigger_r:
        _raise_stop(
            protected,
            _raw_breakeven_stop(
                direction,
                position.entry_price,
                position.entry_fee / position.quantity,
                execution,
            ),
            "breakeven_stop",
        )

    if protected.partial_taken and exit_config.move_stop_to_breakeven_after_partial:
        _raise_stop(
            protected,
            _raw_breakeven_stop(
                direction,
                position.entry_price,
                position.entry_fee / position.quantity,
                execution,
            ),
            "partial_breakeven_stop",
        )
    return None, cash_delta


def simulate_exit_protected_portfolio(
    candidates: dict[int, list[Candidate]],
    symbols: Iterable[str],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    signal_config: VolatilityBreakoutConfig,
    portfolio_config: PortfolioSearchConfig,
    exit_config: ExitProtectionConfig,
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
) -> dict[str, Any]:
    exit_config.validate()
    if start >= end:
        raise ValueError("start must be before end")

    start_minute = minute_token(start)
    end_minute = minute_token(end)
    cash = float(initial_equity)
    positions: dict[str, ProtectedPosition] = {}
    trades: list[dict[str, Any]] = []
    cooldown_until: dict[str, int] = {}
    daily_entries: dict[str, int] = defaultdict(int)
    rejected: dict[str, int] = defaultdict(int)
    peak_equity = cash
    max_drawdown = 0.0
    max_drawdown_duration = 0
    drawdown_start: int | None = None
    hard_stopped = False
    symbol_set = frozenset(symbols)

    def marked_equity(minute: int) -> float:
        return _mark_equity(
            cash,
            {symbol: protected.position for symbol, protected in positions.items()},
            execution_data,
            minute,
            execution,
        )

    for minute in range(start_minute, end_minute):
        for symbol, protected in list(positions.items()):
            series = execution_data[symbol]
            index = series.index_at(minute)
            if index is None:
                continue
            trade, cash_delta = _process_protected_bar(
                protected,
                minute,
                series,
                index,
                signal_config,
                exit_config,
                execution,
                rules[symbol],
            )
            cash += cash_delta
            if trade is None:
                continue
            trades.append(trade)
            positions.pop(symbol, None)
            cooldown_until[symbol] = minute + portfolio_config.symbol_cooldown_minutes

        equity_before_entry = marked_equity(minute)
        day_key = minute_datetime(minute).date().isoformat()
        rows = sorted(
            candidates.get(minute, ()),
            key=lambda row: _candidate_sort_key(row, portfolio_config.ranking_mode),
        )
        if not hard_stopped:
            for candidate in rows:
                symbol = candidate.signal.symbol
                if symbol not in symbol_set or symbol not in execution_data:
                    continue
                market_reject = _market_filter_reject_reason(candidate, portfolio_config)
                if market_reject is not None:
                    rejected[market_reject] += 1
                    continue
                if symbol in positions:
                    rejected["symbol_already_open"] += 1
                    continue
                if cooldown_until.get(symbol, -1) > minute:
                    rejected["symbol_cooldown"] += 1
                    continue
                if len(positions) >= portfolio_config.max_open_positions:
                    rejected["position_limit"] += 1
                    continue
                if daily_entries[day_key] >= portfolio_config.max_daily_trades:
                    rejected["daily_trade_limit"] += 1
                    continue
                opened = _entry_position(
                    candidate,
                    execution_data[symbol],
                    rules[symbol],
                    signal_config,
                    portfolio_config,
                    execution,
                    equity_before_entry if portfolio_config.compound else initial_equity,
                )
                if opened is None:
                    rejected["sizing_or_data"] += 1
                    continue
                position, entry_fee = opened
                protected = _protected_position(position, exit_config, execution)
                positions[symbol] = protected
                cash -= entry_fee
                daily_entries[day_key] += 1

                index = execution_data[symbol].index_at(minute)
                if index is not None:
                    trade, cash_delta = _process_protected_bar(
                        protected,
                        minute,
                        execution_data[symbol],
                        index,
                        signal_config,
                        exit_config,
                        execution,
                        rules[symbol],
                    )
                    cash += cash_delta
                    if trade is not None:
                        trades.append(trade)
                        positions.pop(symbol, None)
                        cooldown_until[symbol] = minute + portfolio_config.symbol_cooldown_minutes
                if len(positions) >= portfolio_config.max_open_positions:
                    break

        equity = marked_equity(minute)
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0.0 else 1.0
        if drawdown > 0.0:
            drawdown_start = minute if drawdown_start is None else drawdown_start
            max_drawdown_duration = max(max_drawdown_duration, minute - drawdown_start)
        else:
            drawdown_start = None
        max_drawdown = max(max_drawdown, drawdown)
        if drawdown >= portfolio_config.hard_drawdown_stop_pct or equity <= 0.0:
            hard_stopped = True

    force_minute = end_minute - 1
    for symbol, protected in list(positions.items()):
        series = execution_data[symbol]
        index = series.last_index_at_or_before(force_minute)
        if index is None:
            continue
        trade, cash_delta = _close_protected_position(
            protected,
            float(series.closes[index]),
            "end_of_backtest",
            int(series.minutes[index]),
            execution,
            rules[symbol],
        )
        cash += cash_delta
        trades.append(trade)

    result = summarize_result(
        initial_equity,
        cash,
        trades,
        max_drawdown,
        max_drawdown_duration,
        sum(len(rows) for rows in candidates.values()),
        dict(rejected),
        hard_stopped,
        signal_config,
        portfolio_config,
    )
    result.update(
        {
            "strategy": EXIT_PROTECTION_STRATEGY_NAME,
            "strategy_version": EXIT_PROTECTION_VERSION,
            "exit_protection_config": asdict(exit_config),
            "partial_exit_count": sum(trade["partial_exit_count"] for trade in trades),
            "trades_with_partial_exit": sum(
                int(trade["partial_exit_count"] > 0) for trade in trades
            ),
            "cash_reconciliation_error": (
                cash - initial_equity - sum(float(trade["net_pnl"]) for trade in trades)
            ),
        }
    )
    return result

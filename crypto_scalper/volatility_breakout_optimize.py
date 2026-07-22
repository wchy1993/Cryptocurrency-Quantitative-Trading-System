from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
from array import array
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .binance_client import SymbolRules
from .data import load_candles_csv, parse_timestamp
from .live_config import load_live_config
from .live_execution_backtest import _resample_to_timeframe
from .live_portfolio_backtest import _inferred_symbol_rules
from .models import Candle, Direction
from .realistic_data import load_funding_rate_directory
from .risk import (
    BacktestExecutionConfig,
    conservative_quantity,
    execution_config_from_live_config,
    funding_cashflow,
    funding_rates_between,
    market_entry_fill,
    market_exit_fill,
)
from .volatility_breakout import (
    VOLATILITY_BREAKOUT_STRATEGY_NAME,
    VOLATILITY_BREAKOUT_VERSION,
    DualThrustSignal,
    VolatilityBreakoutConfig,
    build_dual_thrust_signals,
)


EPOCH = datetime(1970, 1, 1)

UNIVERSE_30: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "TRXUSDT", "DOGEUSDT", "XLMUSDT", "ADAUSDT", "XMRUSDT",
    "LINKUSDT", "BCHUSDT", "HBARUSDT", "LTCUSDT", "AVAXUSDT",
    "SUIUSDT", "1000SHIBUSDT", "NEARUSDT", "TAOUSDT", "WLDUSDT",
    "DOTUSDT", "UNIUSDT", "ICPUSDT", "1000PEPEUSDT", "ETCUSDT",
    "AAVEUSDT", "ENAUSDT", "ATOMUSDT", "ALGOUSDT", "POLUSDT",
)

UNIVERSE_50: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "ZECUSDT", "HYPEUSDT",
    "XRPUSDT", "DOGEUSDT", "BNBUSDT", "SIRENUSDT", "1000PEPEUSDT",
    "TAOUSDT", "NEARUSDT", "WLDUSDT", "SUIUSDT", "ADAUSDT",
    "TONUSDT", "AVAXUSDT", "ENAUSDT", "LINKUSDT", "FILUSDT",
    "XLMUSDT", "BCHUSDT", "ONDOUSDT", "TRUMPUSDT", "AAVEUSDT",
    "DOTUSDT", "PENGUUSDT", "LTCUSDT", "TRXUSDT", "FETUSDT",
    "UNIUSDT", "INJUSDT", "DASHUSDT", "BUSDT", "ARBUSDT",
    "1000SHIBUSDT", "APEUSDT", "XMRUSDT", "APTUSDT", "WIFUSDT",
    "ICPUSDT", "ETCUSDT", "VIRTUALUSDT", "HBARUSDT", "RENDERUSDT",
    "CRVUSDT", "JTOUSDT", "OPUSDT", "CHZUSDT", "1000BONKUSDT",
)


@dataclass(frozen=True)
class PortfolioSearchConfig:
    risk_per_trade_pct: float = 0.02
    max_trade_risk_pct: float = 0.10
    max_open_positions: int = 1
    max_daily_trades: int = 8
    symbol_cooldown_minutes: int = 360
    max_notional_multiple: float = 9.0
    hard_drawdown_stop_pct: float = 0.60
    compound: bool = True
    ranking_mode: str = "quality_desc"
    long_risk_multiplier: float = 1.0
    short_risk_multiplier: float = 1.0
    min_directional_btc_return_4h: float = -999.0
    max_directional_btc_return_4h: float = 999.0
    min_directional_eth_return_4h: float = -999.0
    max_directional_eth_return_4h: float = 999.0
    min_directional_breadth: float = 0.0
    max_directional_breadth: float = 1.0


@dataclass(frozen=True)
class Candidate:
    signal: DualThrustSignal
    entry_minute: int
    btc_return_4h: float | None = None
    eth_return_4h: float | None = None
    breadth_above_ema21: float | None = None


@dataclass
class CompactSeries:
    minutes: array
    opens: array
    highs: array
    lows: array
    closes: array
    volumes: array

    def index_at(self, minute: int) -> int | None:
        index = bisect.bisect_left(self.minutes, minute)
        if index >= len(self.minutes) or self.minutes[index] != minute:
            return None
        return index

    def last_index_at_or_before(self, minute: int) -> int | None:
        index = bisect.bisect_right(self.minutes, minute) - 1
        return index if index >= 0 else None


@dataclass
class OpenPosition:
    candidate: Candidate
    quantity: float
    raw_entry_price: float
    entry_price: float
    entry_fee: float
    entry_slippage: float
    entry_minute: int
    stop_price: float
    initial_stop_price: float
    take_profit_price: float
    unit_risk: float
    risk_budget: float
    best_price: float
    worst_price: float
    max_mfe_r: float = 0.0
    max_mae_r: float = 0.0
    extension_qualified: bool = False


def minute_token(timestamp: datetime) -> int:
    return int((timestamp - EPOCH).total_seconds() // 60)


def minute_datetime(value: int) -> datetime:
    return EPOCH + timedelta(minutes=int(value))


def load_compact_execution_series(
    path: Path,
    start: datetime,
    end: datetime,
) -> CompactSeries:
    minutes = array("q")
    opens = array("d")
    highs = array("d")
    lows = array("d")
    closes = array("d")
    volumes = array("d")
    start_text = start.isoformat(timespec="seconds")
    end_text = end.isoformat(timespec="seconds")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            token = row["timestamp"]
            if token < start_text:
                continue
            if token >= end_text:
                break
            timestamp = parse_timestamp(token)
            minutes.append(minute_token(timestamp))
            opens.append(float(row["open"]))
            highs.append(float(row["high"]))
            lows.append(float(row["low"]))
            closes.append(float(row["close"]))
            volumes.append(float(row["volume"]))
    return CompactSeries(minutes, opens, highs, lows, closes, volumes)


def load_research_data(
    symbols: Iterable[str],
    signal_data_dir: str,
    execution_data_dir: str,
    start: datetime,
    end: datetime,
) -> tuple[dict[str, list[Candle]], dict[str, CompactSeries], dict[str, SymbolRules]]:
    symbols = tuple(symbols)
    signal_root = Path(signal_data_dir)
    execution_root = Path(execution_data_dir)
    warmup_start = start - timedelta(days=21)
    signal_data: dict[str, list[Candle]] = {}
    execution_data: dict[str, CompactSeries] = {}
    rules: dict[str, SymbolRules] = {}
    for number, symbol in enumerate(symbols, start=1):
        signal_paths = sorted(signal_root.glob(f"{symbol}_15m_*.csv"))
        execution_paths = sorted(execution_root.glob(f"{symbol}_1m_*.csv"))
        if not signal_paths or not execution_paths:
            continue
        candles = load_candles_csv(signal_paths[-1], start=warmup_start, end=end)
        series = load_compact_execution_series(execution_paths[-1], start, end)
        if len(candles) < 500 or len(series.minutes) < 1_000:
            continue
        signal_data[symbol] = candles
        execution_data[symbol] = series
        rules[symbol] = _inferred_symbol_rules(symbol, candles)
        if number % 10 == 0:
            print(f"loaded {number}/{len(symbols)} symbols", flush=True)
    return signal_data, execution_data, rules


def signal_candles_for_timeframe(candles: list[Candle], minutes: int) -> list[Candle]:
    if minutes == 15:
        return candles
    if minutes not in {30, 60}:
        raise ValueError(f"unsupported signal timeframe: {minutes}")
    return _resample_to_timeframe(candles, "15m", f"{minutes}m")


def build_candidates(
    symbols: Iterable[str],
    signal_data: dict[str, list[Candle]],
    execution_data: dict[str, CompactSeries],
    config: VolatilityBreakoutConfig,
    start: datetime,
    end: datetime,
) -> dict[int, list[Candidate]]:
    output: dict[int, list[Candidate]] = defaultdict(list)
    for symbol in symbols:
        if symbol not in signal_data or symbol not in execution_data:
            continue
        candles = signal_candles_for_timeframe(signal_data[symbol], config.timeframe_minutes)
        signals = build_dual_thrust_signals(symbol, candles, config, start, end)
        series = execution_data[symbol]
        for signal in signals:
            entry_minute = minute_token(signal.signal_available_time)
            if series.index_at(entry_minute) is None:
                continue
            output[entry_minute].append(Candidate(signal=signal, entry_minute=entry_minute))
    for rows in output.values():
        rows.sort(key=lambda row: (-row.signal.quality_score, row.signal.symbol))
    return dict(output)


def _raw_target_for_full_cost_r(
    direction: Direction,
    entry_price: float,
    entry_fee_per_unit: float,
    unit_risk: float,
    target_r: float,
    execution: BacktestExecutionConfig,
) -> float:
    target_net = max(0.0, target_r) * unit_risk
    slip = execution.take_profit_slippage_bps / 10_000.0
    fee = execution.taker_fee_rate
    if direction == Direction.LONG:
        denominator = max((1.0 - slip) * (1.0 - fee), 1e-12)
        return (target_net + entry_price + entry_fee_per_unit) / denominator
    denominator = max((1.0 + slip) * (1.0 + fee), 1e-12)
    return max(1e-12, (entry_price - entry_fee_per_unit - target_net) / denominator)


def _raw_breakeven_stop(
    direction: Direction,
    entry_price: float,
    entry_fee_per_unit: float,
    execution: BacktestExecutionConfig,
) -> float:
    slip = execution.stop_slippage_bps / 10_000.0
    fee = execution.taker_fee_rate
    if direction == Direction.LONG:
        return (entry_price + entry_fee_per_unit) / max((1.0 - slip) * (1.0 - fee), 1e-12)
    return max(1e-12, (entry_price - entry_fee_per_unit) / max((1.0 + slip) * (1.0 + fee), 1e-12))


def _raw_stop_for_full_cost_r(
    direction: Direction,
    entry_price: float,
    entry_fee_per_unit: float,
    unit_risk: float,
    locked_r: float,
    execution: BacktestExecutionConfig,
) -> float:
    target_net = locked_r * unit_risk
    slip = execution.stop_slippage_bps / 10_000.0
    fee = execution.taker_fee_rate
    if direction == Direction.LONG:
        denominator = max((1.0 - slip) * (1.0 - fee), 1e-12)
        return max(1e-12, (target_net + entry_price + entry_fee_per_unit) / denominator)
    denominator = max((1.0 + slip) * (1.0 + fee), 1e-12)
    return max(1e-12, (entry_price - entry_fee_per_unit - target_net) / denominator)


def _funding_cashflow(
    execution: BacktestExecutionConfig,
    symbol: str,
    direction: Direction,
    notional: float,
    entry_minute: int,
    exit_minute: int,
) -> float:
    if not execution.funding_enabled:
        return 0.0
    rates = (execution.funding_rates_by_symbol or {}).get(symbol, ())
    values = funding_rates_between(rates, minute_datetime(entry_minute), minute_datetime(exit_minute))
    return funding_cashflow(direction, notional, values)


def _entry_position(
    candidate: Candidate,
    series: CompactSeries,
    rules: SymbolRules,
    signal_config: VolatilityBreakoutConfig,
    portfolio_config: PortfolioSearchConfig,
    execution: BacktestExecutionConfig,
    equity: float,
) -> tuple[OpenPosition, float] | None:
    index = series.index_at(candidate.entry_minute)
    if index is None or equity <= 0.0:
        return None
    raw_entry = series.opens[index]
    direction = candidate.signal.direction
    raw_stop = raw_entry - direction.value * candidate.signal.atr_value * signal_config.stop_atr_multiple
    if raw_stop <= 0.0:
        return None

    unit_entry_fill = market_entry_fill(execution, rules, direction, 1.0, raw_entry)
    unit_stop_fill = market_exit_fill(execution, rules, direction, 1.0, raw_stop, "stop_market")
    entry_fee_per_unit = unit_entry_fill.fee
    unit_stop_net = (
        direction.value * (unit_stop_fill.price - unit_entry_fill.price)
        - entry_fee_per_unit
        - unit_stop_fill.fee
    )
    unit_risk = -unit_stop_net
    if unit_risk <= 0.0:
        return None

    side_risk_multiplier = (
        portfolio_config.long_risk_multiplier
        if direction == Direction.LONG
        else portfolio_config.short_risk_multiplier
    )
    effective_risk_pct = min(
        max(0.0, portfolio_config.max_trade_risk_pct),
        max(0.0, portfolio_config.risk_per_trade_pct) * max(0.0, side_risk_multiplier),
    )
    risk_budget = equity * effective_risk_pct
    requested_quantity = risk_budget / unit_risk
    max_notional = equity * max(0.0, portfolio_config.max_notional_multiple)
    requested_quantity = min(requested_quantity, max_notional / max(unit_entry_fill.price, 1e-12))
    quantity = conservative_quantity(rules, requested_quantity)
    if quantity <= 0.0 or quantity * unit_entry_fill.price < max(5.0, float(rules.min_notional)):
        return None

    entry_fill = market_entry_fill(
        execution,
        rules,
        direction,
        quantity,
        raw_entry,
        series.volumes[index] * raw_entry,
    )
    stop_fill = market_exit_fill(execution, rules, direction, quantity, raw_stop, "stop_market")
    actual_unit_risk = -(
        direction.value * (stop_fill.price - entry_fill.price)
        - entry_fill.fee / quantity
        - stop_fill.fee / quantity
    )
    if actual_unit_risk <= 0.0:
        return None
    target = _raw_target_for_full_cost_r(
        direction,
        entry_fill.price,
        entry_fill.fee / quantity,
        actual_unit_risk,
        signal_config.take_profit_r,
        execution,
    )
    return (
        OpenPosition(
            candidate=candidate,
            quantity=quantity,
            raw_entry_price=raw_entry,
            entry_price=entry_fill.price,
            entry_fee=entry_fill.fee,
            entry_slippage=entry_fill.slippage_cost,
            entry_minute=candidate.entry_minute,
            stop_price=raw_stop,
            initial_stop_price=raw_stop,
            take_profit_price=target,
            unit_risk=actual_unit_risk,
            risk_budget=quantity * actual_unit_risk,
            best_price=entry_fill.price,
            worst_price=entry_fill.price,
        ),
        entry_fill.fee,
    )


def _position_current_r(
    position: OpenPosition,
    raw_exit: float,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
) -> float:
    fill = market_exit_fill(
        execution,
        rules,
        position.candidate.signal.direction,
        1.0,
        raw_exit,
        "market",
    )
    net_per_unit = (
        position.candidate.signal.direction.value * (fill.price - position.entry_price)
        - position.entry_fee / position.quantity
        - fill.fee
    )
    return net_per_unit / max(position.unit_risk, 1e-12)


def _exit_position(
    position: OpenPosition,
    raw_exit: float,
    reason: str,
    exit_minute: int,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
) -> tuple[dict[str, Any], float]:
    direction = position.candidate.signal.direction
    order_type = "market"
    if "stop" in reason:
        order_type = "stop_market"
    elif "take_profit" in reason:
        order_type = "take_profit_market"
    fill = market_exit_fill(execution, rules, direction, position.quantity, raw_exit, order_type)
    raw_gross = direction.value * position.quantity * (raw_exit - position.raw_entry_price)
    execution_gross = direction.value * position.quantity * (fill.price - position.entry_price)
    funding = _funding_cashflow(
        execution,
        position.candidate.signal.symbol,
        direction,
        position.quantity * position.entry_price,
        position.entry_minute,
        exit_minute,
    )
    fees = position.entry_fee + fill.fee
    slippage = position.entry_slippage + fill.slippage_cost
    # Entry fee was removed from cash when the position opened. The cash delta
    # returned here therefore contains execution PnL, exit fee, and funding.
    cash_delta = execution_gross - fill.fee + funding
    total_net = raw_gross - fees - slippage + funding
    if not math.isclose(execution_gross - fees + funding, total_net, rel_tol=1e-8, abs_tol=1e-8):
        total_net = execution_gross - fees + funding
    actual_risk = max(position.risk_budget, 1e-12)
    signal = position.candidate.signal
    trade = {
        "event_id": signal.event_id,
        "symbol": signal.symbol,
        "side": direction.name,
        "entry_time": minute_datetime(position.entry_minute).isoformat(),
        "exit_time": minute_datetime(exit_minute).isoformat(),
        "raw_entry_price": position.raw_entry_price,
        "entry_price": position.entry_price,
        "raw_exit_price": raw_exit,
        "exit_price": fill.price,
        "quantity": position.quantity,
        "notional": position.quantity * position.entry_price,
        "initial_stop": position.initial_stop_price,
        "final_stop": position.stop_price,
        "take_profit": position.take_profit_price,
        "risk_usdt": actual_risk,
        "gross_pnl": raw_gross,
        "fee": fees,
        "slippage": slippage,
        "funding": funding,
        "net_pnl": total_net,
        "pnl_r": total_net / actual_risk,
        "mfe_r": position.max_mfe_r,
        "mae_r": position.max_mae_r,
        "holding_minutes": exit_minute - position.entry_minute,
        "exit_reason": reason,
        "quality_score": signal.quality_score,
        "volume_ratio": signal.volume_ratio,
        "body_atr": signal.body_atr,
        "trend_alignment_atr": signal.trend_alignment_atr,
        "range_atr": signal.range_atr,
        "breakout_extension_atr": signal.breakout_extension_atr,
        "extension_qualified": position.extension_qualified,
        "btc_return_4h": position.candidate.btc_return_4h,
        "eth_return_4h": position.candidate.eth_return_4h,
        "breadth_above_ema21": position.candidate.breadth_above_ema21,
    }
    return trade, cash_delta


def _process_position_bar(
    position: OpenPosition,
    minute: int,
    series: CompactSeries,
    index: int,
    signal_config: VolatilityBreakoutConfig,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
) -> tuple[dict[str, Any], float] | None:
    direction = position.candidate.signal.direction
    open_price = series.opens[index]
    high = series.highs[index]
    low = series.lows[index]

    gap_stop = open_price <= position.stop_price if direction == Direction.LONG else open_price >= position.stop_price
    if gap_stop:
        return _exit_position(position, open_price, "stop_loss_gap", minute, execution, rules)
    holding_minutes = minute - position.entry_minute
    current_open_r = _position_current_r(position, open_price, execution, rules)
    if (
        signal_config.fail_fast_minutes > 0
        and holding_minutes >= signal_config.fail_fast_minutes
        and position.max_mfe_r < signal_config.fail_fast_min_mfe_r
        and current_open_r <= signal_config.fail_fast_max_current_r
    ):
        return _exit_position(position, open_price, "fail_fast", minute, execution, rules)
    if holding_minutes >= signal_config.max_holding_minutes:
        extension_enabled = signal_config.extended_holding_minutes > signal_config.max_holding_minutes
        if extension_enabled and not position.extension_qualified:
            position.extension_qualified = (
                current_open_r >= signal_config.extension_min_current_r
                and position.max_mfe_r >= signal_config.extension_min_mfe_r
            )
        if not position.extension_qualified:
            return _exit_position(position, open_price, "time_stop", minute, execution, rules)
        if holding_minutes >= signal_config.extended_holding_minutes:
            return _exit_position(position, open_price, "extended_time_stop", minute, execution, rules)

    stop_hit = low <= position.stop_price if direction == Direction.LONG else high >= position.stop_price
    target_hit = high >= position.take_profit_price if direction == Direction.LONG else low <= position.take_profit_price
    if stop_hit:
        return _exit_position(position, position.stop_price, "stop_loss", minute, execution, rules)
    if target_hit:
        return _exit_position(position, position.take_profit_price, "take_profit", minute, execution, rules)

    favorable = high if direction == Direction.LONG else low
    adverse = low if direction == Direction.LONG else high
    if direction == Direction.LONG:
        position.best_price = max(position.best_price, favorable)
        position.worst_price = min(position.worst_price, adverse)
    else:
        position.best_price = min(position.best_price, favorable)
        position.worst_price = max(position.worst_price, adverse)
    position.max_mfe_r = max(position.max_mfe_r, _position_current_r(position, favorable, execution, rules))
    position.max_mae_r = min(position.max_mae_r, _position_current_r(position, adverse, execution, rules))

    if (
        signal_config.profit_giveback_activation_r > 0.0
        and signal_config.profit_giveback_r > 0.0
        and position.max_mfe_r >= signal_config.profit_giveback_activation_r
    ):
        locked_r = max(0.0, position.max_mfe_r - signal_config.profit_giveback_r)
        profit_floor = _raw_stop_for_full_cost_r(
            direction,
            position.entry_price,
            position.entry_fee / position.quantity,
            position.unit_risk,
            locked_r,
            execution,
        )
        if direction == Direction.LONG:
            position.stop_price = max(position.stop_price, profit_floor)
        else:
            position.stop_price = min(position.stop_price, profit_floor)

    if signal_config.breakeven_trigger_r > 0.0 and position.max_mfe_r >= signal_config.breakeven_trigger_r:
        breakeven = _raw_breakeven_stop(
            direction,
            position.entry_price,
            position.entry_fee / position.quantity,
            execution,
        )
        if direction == Direction.LONG:
            position.stop_price = max(position.stop_price, breakeven)
        else:
            position.stop_price = min(position.stop_price, breakeven)

    if (
        signal_config.trailing_activation_r > 0.0
        and signal_config.trailing_atr_multiple > 0.0
        and position.max_mfe_r >= signal_config.trailing_activation_r
    ):
        distance = signal_config.trailing_atr_multiple * position.candidate.signal.atr_value
        trailing = position.best_price - direction.value * distance
        if direction == Direction.LONG:
            position.stop_price = max(position.stop_price, trailing)
        else:
            position.stop_price = min(position.stop_price, trailing)
    return None


def _mark_equity(
    cash: float,
    positions: dict[str, OpenPosition],
    execution_data: dict[str, CompactSeries],
    minute: int,
    execution: BacktestExecutionConfig,
) -> float:
    equity = cash
    slip = execution.market_slippage_bps / 10_000.0
    for symbol, position in positions.items():
        series = execution_data[symbol]
        index = series.last_index_at_or_before(minute)
        if index is None:
            continue
        raw = series.closes[index]
        direction = position.candidate.signal.direction
        exit_price = raw * (1.0 - slip) if direction == Direction.LONG else raw * (1.0 + slip)
        gross = direction.value * position.quantity * (exit_price - position.entry_price)
        exit_fee = abs(position.quantity * exit_price) * execution.taker_fee_rate
        equity += gross - exit_fee
    return equity


def simulate_portfolio(
    candidates: dict[int, list[Candidate]],
    symbols: Iterable[str],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    signal_config: VolatilityBreakoutConfig,
    portfolio_config: PortfolioSearchConfig,
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
    skip_event_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    start_minute = minute_token(start)
    end_minute = minute_token(end)
    cash = float(initial_equity)
    positions: dict[str, OpenPosition] = {}
    trades: list[dict[str, Any]] = []
    cooldown_until: dict[str, int] = {}
    daily_entries: dict[str, int] = defaultdict(int)
    peak_equity = cash
    max_drawdown = 0.0
    drawdown_start: int | None = None
    max_drawdown_duration = 0
    hard_stopped = False
    symbol_set = frozenset(symbols)
    candidate_count = sum(len(rows) for rows in candidates.values())
    rejected = defaultdict(int)

    for minute in range(start_minute, end_minute):
        for symbol, position in list(positions.items()):
            series = execution_data[symbol]
            index = series.index_at(minute)
            if index is None:
                continue
            closed = _process_position_bar(
                position,
                minute,
                series,
                index,
                signal_config,
                execution,
                rules[symbol],
            )
            if closed is None:
                continue
            trade, cash_delta = closed
            cash += cash_delta
            trades.append(trade)
            positions.pop(symbol, None)
            cooldown_until[symbol] = minute + portfolio_config.symbol_cooldown_minutes

        equity_before_entry = _mark_equity(cash, positions, execution_data, minute, execution)
        day_key = minute_datetime(minute).date().isoformat()
        rows = sorted(candidates.get(minute, ()), key=lambda row: _candidate_sort_key(row, portfolio_config.ranking_mode))
        if not hard_stopped and rows:
            for candidate in rows:
                symbol = candidate.signal.symbol
                if candidate.signal.event_id in skip_event_ids:
                    rejected["excluded_event"] += 1
                    continue
                if symbol not in execution_data or symbol not in symbol_set:
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
                positions[symbol] = position
                cash -= entry_fee
                daily_entries[day_key] += 1

                index = execution_data[symbol].index_at(minute)
                if index is not None:
                    closed = _process_position_bar(
                        position,
                        minute,
                        execution_data[symbol],
                        index,
                        signal_config,
                        execution,
                        rules[symbol],
                    )
                    if closed is not None:
                        trade, cash_delta = closed
                        cash += cash_delta
                        trades.append(trade)
                        positions.pop(symbol, None)
                        cooldown_until[symbol] = minute + portfolio_config.symbol_cooldown_minutes
                if len(positions) >= portfolio_config.max_open_positions:
                    break

        equity = _mark_equity(cash, positions, execution_data, minute, execution)
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
    for symbol, position in list(positions.items()):
        series = execution_data[symbol]
        index = series.last_index_at_or_before(force_minute)
        if index is None:
            continue
        trade, cash_delta = _exit_position(
            position,
            series.closes[index],
            "end_of_backtest",
            int(series.minutes[index]),
            execution,
            rules[symbol],
        )
        cash += cash_delta
        trades.append(trade)
        positions.pop(symbol, None)

    return summarize_result(
        initial_equity,
        cash,
        trades,
        max_drawdown,
        max_drawdown_duration,
        candidate_count,
        dict(rejected),
        hard_stopped,
        signal_config,
        portfolio_config,
    )


def summarize_result(
    initial_equity: float,
    final_equity: float,
    trades: list[dict[str, Any]],
    max_drawdown: float,
    max_drawdown_duration: int,
    candidate_count: int,
    rejected: dict[str, int],
    hard_stopped: bool,
    signal_config: VolatilityBreakoutConfig,
    portfolio_config: PortfolioSearchConfig,
) -> dict[str, Any]:
    wins = [trade for trade in trades if trade["net_pnl"] > 0.0]
    losses = [trade for trade in trades if trade["net_pnl"] <= 0.0]
    net_profit = final_equity - initial_equity
    gross_profit = sum(trade["net_pnl"] for trade in wins)
    gross_loss = abs(sum(trade["net_pnl"] for trade in losses))
    raw_gross_profit = sum(max(0.0, trade["gross_pnl"]) for trade in trades)
    fees = sum(trade["fee"] for trade in trades)
    slippage = sum(trade["slippage"] for trade in trades)
    funding = sum(trade["funding"] for trade in trades)
    average_win = statistics.mean(trade["net_pnl"] for trade in wins) if wins else 0.0
    average_loss = statistics.mean(trade["net_pnl"] for trade in losses) if losses else 0.0
    monthly: dict[str, dict[str, Any]] = {}
    by_symbol: dict[str, dict[str, Any]] = {}
    by_side: dict[str, dict[str, Any]] = {}
    by_exit_reason: dict[str, dict[str, Any]] = {}
    for trade in trades:
        _aggregate_bucket(monthly, trade["exit_time"][:7], trade)
        _aggregate_bucket(by_symbol, trade["symbol"], trade)
        _aggregate_bucket(by_side, trade["side"], trade)
        _aggregate_bucket(by_exit_reason, trade["exit_reason"], trade)
    for buckets in (monthly, by_symbol, by_side, by_exit_reason):
        for row in buckets.values():
            _finalize_bucket(row)

    sorted_wins = sorted((trade["net_pnl"] for trade in wins), reverse=True)
    top5 = sum(sorted_wins[:5])
    return {
        "strategy": VOLATILITY_BREAKOUT_STRATEGY_NAME,
        "strategy_version": VOLATILITY_BREAKOUT_VERSION,
        "signal_config": signal_config.as_dict(),
        "portfolio_config": asdict(portfolio_config),
        "candidate_count": candidate_count,
        "trade_count": len(trades),
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "net_profit": net_profit,
        "return_pct": net_profit / initial_equity if initial_equity else 0.0,
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else (math.inf if gross_profit > 0.0 else 0.0),
        "expectancy_usdt": net_profit / len(trades) if trades else 0.0,
        "expectancy_r": statistics.mean(trade["pnl_r"] for trade in trades) if trades else 0.0,
        "average_win": average_win,
        "average_loss": average_loss,
        "average_win_loss_ratio": average_win / abs(average_loss) if average_loss < 0.0 else 0.0,
        "max_drawdown_pct": max_drawdown,
        "max_drawdown_duration_minutes": max_drawdown_duration,
        "fee": fees,
        "slippage": slippage,
        "funding": funding,
        "full_cost": fees + slippage - funding,
        "cost_to_raw_gross_profit_ratio": (fees + slippage - funding) / raw_gross_profit if raw_gross_profit > 0.0 else math.inf,
        "top5_net_profit": top5,
        "top5_profit_contribution": top5 / net_profit if net_profit > 0.0 else math.inf,
        "hard_drawdown_stopped": hard_stopped,
        "positive_months": sum(1 for row in monthly.values() if row["net_pnl"] > 0.0),
        "negative_months": sum(1 for row in monthly.values() if row["net_pnl"] < 0.0),
        "rejected": rejected,
        "by_month": monthly,
        "by_symbol": by_symbol,
        "by_side": by_side,
        "by_exit_reason": by_exit_reason,
        "trades": trades,
    }


def _aggregate_bucket(output: dict[str, dict[str, Any]], key: str, trade: dict[str, Any]) -> None:
    row = output.setdefault(
        key,
        {"trade_count": 0, "wins": 0, "net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0},
    )
    row["trade_count"] += 1
    row["wins"] += int(trade["net_pnl"] > 0.0)
    row["net_pnl"] += trade["net_pnl"]
    if trade["net_pnl"] > 0.0:
        row["gross_profit"] += trade["net_pnl"]
    else:
        row["gross_loss"] += abs(trade["net_pnl"])


def _finalize_bucket(row: dict[str, Any]) -> None:
    count = row["trade_count"]
    row["win_rate"] = row.pop("wins") / count if count else 0.0
    row["profit_factor"] = (
        row["gross_profit"] / row["gross_loss"]
        if row["gross_loss"] > 0.0
        else (math.inf if row["gross_profit"] > 0.0 else 0.0)
    )


def compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_count", "trade_count", "initial_equity", "final_equity", "net_profit",
        "return_pct", "win_rate", "profit_factor", "expectancy_usdt", "expectancy_r",
        "average_win", "average_loss", "average_win_loss_ratio", "max_drawdown_pct",
        "max_drawdown_duration_minutes", "fee", "slippage", "funding", "full_cost",
        "cost_to_raw_gross_profit_ratio", "top5_profit_contribution", "hard_drawdown_stopped",
        "positive_months", "negative_months", "by_month", "by_side", "by_exit_reason",
    )
    return {key: result[key] for key in keys}


def _entry_cache_key(config: VolatilityBreakoutConfig) -> str:
    fields = config.as_dict()
    for key in (
        "stop_atr_multiple", "take_profit_r", "breakeven_trigger_r",
        "trailing_activation_r", "trailing_atr_multiple", "fail_fast_minutes",
        "fail_fast_min_mfe_r", "fail_fast_max_current_r", "extended_holding_minutes",
        "extension_min_current_r", "extension_min_mfe_r", "profit_giveback_activation_r",
        "profit_giveback_r", "max_holding_minutes",
    ):
        fields.pop(key, None)
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _candidate_sort_key(candidate: Candidate, mode: str) -> tuple[float, str]:
    signal = candidate.signal
    if mode == "quality_desc":
        score = -signal.quality_score
    elif mode == "range_asc":
        score = signal.range_atr
    elif mode == "breakout_extension_desc":
        score = -signal.breakout_extension_atr
    elif mode == "extension_to_range_desc":
        score = -(signal.breakout_extension_atr / max(signal.range_atr, 1e-12))
    elif mode == "trend_alignment_desc":
        score = -signal.trend_alignment_atr
    elif mode == "volume_asc":
        score = signal.volume_ratio
    elif mode == "directional_breadth_desc":
        breadth = candidate.breadth_above_ema21
        if breadth is None:
            score = math.inf
        else:
            directional = breadth if signal.direction == Direction.LONG else 1.0 - breadth
            score = -directional
    elif mode == "directional_btc_4h_desc":
        value = candidate.btc_return_4h
        score = math.inf if value is None else -(signal.direction.value * value)
    else:
        raise ValueError(f"unsupported volatility breakout ranking mode: {mode}")
    return score, signal.symbol


def _market_filter_reject_reason(
    candidate: Candidate,
    config: PortfolioSearchConfig,
) -> str | None:
    direction = candidate.signal.direction.value
    if candidate.btc_return_4h is None:
        if config.min_directional_btc_return_4h > -999.0 or config.max_directional_btc_return_4h < 999.0:
            return "market_context_missing"
    else:
        directional_btc = direction * candidate.btc_return_4h
        if directional_btc < config.min_directional_btc_return_4h:
            return "btc_4h_too_adverse"
        if directional_btc > config.max_directional_btc_return_4h:
            return "btc_4h_overextended"
    if candidate.eth_return_4h is None:
        if config.min_directional_eth_return_4h > -999.0 or config.max_directional_eth_return_4h < 999.0:
            return "market_context_missing"
    else:
        directional_eth = direction * candidate.eth_return_4h
        if directional_eth < config.min_directional_eth_return_4h:
            return "eth_4h_too_adverse"
        if directional_eth > config.max_directional_eth_return_4h:
            return "eth_4h_overextended"
    if candidate.breadth_above_ema21 is None:
        if config.min_directional_breadth > 0.0 or config.max_directional_breadth < 1.0:
            return "market_context_missing"
    else:
        directional_breadth = (
            candidate.breadth_above_ema21
            if candidate.signal.direction == Direction.LONG
            else 1.0 - candidate.breadth_above_ema21
        )
        if directional_breadth < config.min_directional_breadth:
            return "breadth_too_adverse"
        if directional_breadth > config.max_directional_breadth:
            return "breadth_overextended"
    return None


def _selection_score(result: dict[str, Any]) -> tuple[float, float, float, int]:
    if result["trade_count"] < 20 or result["hard_drawdown_stopped"]:
        return (-math.inf, -math.inf, -math.inf, result["trade_count"])
    if result["profit_factor"] <= 1.0 or result["max_drawdown_pct"] > 0.50:
        return (-math.inf, -math.inf, -math.inf, result["trade_count"])
    return (
        result["net_profit"],
        result["profit_factor"],
        -result["max_drawdown_pct"],
        result["trade_count"],
    )


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if _selection_score(row["result"])[0] != -math.inf]
    pool = valid or rows
    return max(pool, key=lambda row: (_selection_score(row["result"]), row["result"]["net_profit"]))


def run_experiment_set(
    stage: str,
    variants: Iterable[tuple[str, VolatilityBreakoutConfig, PortfolioSearchConfig]],
    universe: tuple[str, ...],
    signal_data: dict[str, list[Candle]],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
    candidate_cache: dict[tuple[tuple[str, ...], str], dict[int, list[Candidate]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, (name, signal_config, portfolio_config) in enumerate(variants, start=1):
        key = (universe, _entry_cache_key(signal_config))
        candidates = candidate_cache.get(key)
        if candidates is None:
            candidates = build_candidates(universe, signal_data, execution_data, signal_config, start, end)
            candidate_cache[key] = candidates
        result = simulate_portfolio(
            candidates,
            universe,
            execution_data,
            rules,
            signal_config,
            portfolio_config,
            execution,
            start,
            end,
            initial_equity,
        )
        rows.append(
            {
                "stage": stage,
                "name": name,
                "signal_config": signal_config.as_dict(),
                "portfolio_config": asdict(portfolio_config),
                "result": compact_summary(result),
            }
        )
        if number % 10 == 0 or number == 1:
            print(
                f"{len(universe)} symbols {stage} {number}: {name} "
                f"trades={result['trade_count']} net={result['net_profit']:.3f} "
                f"pf={result['profit_factor']:.3f} dd={result['max_drawdown_pct']:.2%}",
                flush=True,
            )
    return rows


def optimize_universe(
    universe: tuple[str, ...],
    signal_data: dict[str, list[Candle]],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
) -> dict[str, Any]:
    candidate_cache: dict[tuple[tuple[str, ...], str], dict[int, list[Candidate]]] = {}
    base_signal = VolatilityBreakoutConfig()
    base_portfolio = PortfolioSearchConfig()
    stages: dict[str, list[dict[str, Any]]] = {}

    stages["stage0_baseline"] = run_experiment_set(
        "stage0_baseline",
        [("classic_30m_n4_k050", base_signal, base_portfolio)],
        universe, signal_data, execution_data, rules, execution, start, end, initial_equity, candidate_cache,
    )

    stage1_variants = []
    for timeframe in (15, 30, 60):
        for lookback in (2, 3, 5, 7):
            for k_value in (0.25, 0.35, 0.45, 0.55, 0.70):
                config = replace(
                    base_signal,
                    timeframe_minutes=timeframe,
                    lookback_days=lookback,
                    long_k=k_value,
                    short_k=k_value,
                )
                stage1_variants.append((f"tf{timeframe}_n{lookback}_k{k_value:.2f}", config, base_portfolio))
    stages["stage1_range_entry"] = run_experiment_set(
        "stage1_range_entry", stage1_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage1_range_entry"])
    selected_signal = VolatilityBreakoutConfig(**best["signal_config"])

    stage2_variants = []
    for side in ("both", "long", "short"):
        for trend in (-999.0, 0.0, 0.25):
            for volume in (0.0, 0.8, 1.2):
                config = replace(
                    selected_signal,
                    allow_long=side != "short",
                    allow_short=side != "long",
                    min_trend_alignment_atr=trend,
                    min_volume_ratio=volume,
                )
                stage2_variants.append((f"{side}_trend{trend:g}_vol{volume:.1f}", config, base_portfolio))
    stages["stage2_direction_quality"] = run_experiment_set(
        "stage2_direction_quality", stage2_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage2_direction_quality"])
    selected_signal = VolatilityBreakoutConfig(**best["signal_config"])

    quality_variants = []
    for body, close_position in ((0.0, 0.0), (0.10, 0.55), (0.20, 0.60), (0.30, 0.65), (0.40, 0.70)):
        for extension in (999.0, 1.0, 0.5):
            config = replace(
                selected_signal,
                min_body_atr=body,
                min_directional_close_position=close_position,
                max_breakout_extension_atr=extension,
            )
            quality_variants.append((f"body{body:.2f}_close{close_position:.2f}_ext{extension:g}", config, base_portfolio))
    stages["stage2b_breakout_quality"] = run_experiment_set(
        "stage2b_breakout_quality", quality_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage2b_breakout_quality"])
    selected_signal = VolatilityBreakoutConfig(**best["signal_config"])

    exit_variants = []
    for stop in (0.75, 1.0, 1.25, 1.50, 2.0, 2.5):
        for target in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
            config = replace(selected_signal, stop_atr_multiple=stop, take_profit_r=target)
            exit_variants.append((f"stop{stop:.2f}_tp{target:.1f}", config, base_portfolio))
    stages["stage3_stop_target"] = run_experiment_set(
        "stage3_stop_target", exit_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage3_stop_target"])
    selected_signal = VolatilityBreakoutConfig(**best["signal_config"])

    management_variants = []
    for holding in (360, 720, 1440, 2880):
        management_variants.append((f"hold{holding}", replace(selected_signal, max_holding_minutes=holding), base_portfolio))
    for trigger in (0.50, 0.75, 1.0):
        management_variants.append((f"breakeven{trigger:.2f}", replace(selected_signal, breakeven_trigger_r=trigger), base_portfolio))
    for activation, trail in ((1.0, 1.0), (1.0, 1.5), (1.5, 1.5), (2.0, 2.0), (2.0, 2.5)):
        management_variants.append(
            (
                f"trail_activate{activation:.1f}_atr{trail:.1f}",
                replace(
                    selected_signal,
                    take_profit_r=20.0,
                    trailing_activation_r=activation,
                    trailing_atr_multiple=trail,
                    max_holding_minutes=2880,
                ),
                base_portfolio,
            )
        )
    stages["stage3b_management"] = run_experiment_set(
        "stage3b_management", management_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage3b_management"] + [select_best(stages["stage3_stop_target"])])
    selected_signal = VolatilityBreakoutConfig(**best["signal_config"])

    risk_variants = []
    for max_positions in (1, 2, 3):
        for risk in (0.02, 0.035, 0.05, 0.075, 0.10):
            portfolio = replace(base_portfolio, risk_per_trade_pct=risk, max_open_positions=max_positions)
            risk_variants.append((f"positions{max_positions}_risk{risk:.3f}", selected_signal, portfolio))
    stages["stage4_risk"] = run_experiment_set(
        "stage4_risk", risk_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage4_risk"])
    selected_signal = VolatilityBreakoutConfig(**best["signal_config"])
    selected_portfolio = PortfolioSearchConfig(**best["portfolio_config"])
    final_candidates = candidate_cache[(universe, _entry_cache_key(selected_signal))]
    final_result = simulate_portfolio(
        final_candidates,
        universe,
        execution_data,
        rules,
        selected_signal,
        selected_portfolio,
        execution,
        start,
        end,
        initial_equity,
    )
    return {
        "universe_size": len(universe),
        "symbols": list(universe),
        "selected_signal_config": selected_signal.as_dict(),
        "selected_portfolio_config": asdict(selected_portfolio),
        "stages": stages,
        "final_result": final_result,
    }


def _shift_candidates(
    candidates: dict[int, list[Candidate]],
    delay_minutes: int,
    execution_data: dict[str, CompactSeries],
) -> dict[int, list[Candidate]]:
    if delay_minutes <= 0:
        return candidates
    shifted: dict[int, list[Candidate]] = defaultdict(list)
    for rows in candidates.values():
        for candidate in rows:
            minute = candidate.entry_minute + delay_minutes
            series = execution_data.get(candidate.signal.symbol)
            if series is None or series.index_at(minute) is None:
                continue
            shifted[minute].append(replace(candidate, entry_minute=minute))
    for rows in shifted.values():
        rows.sort(key=lambda row: (-row.signal.quality_score, row.signal.symbol))
    return dict(shifted)


def _without_symbols(
    candidates: dict[int, list[Candidate]],
    excluded: frozenset[str],
) -> dict[int, list[Candidate]]:
    return {
        minute: [candidate for candidate in rows if candidate.signal.symbol not in excluded]
        for minute, rows in candidates.items()
        if any(candidate.signal.symbol not in excluded for candidate in rows)
    }


def refine_universe(
    universe: tuple[str, ...],
    frozen_signal: VolatilityBreakoutConfig,
    frozen_portfolio: PortfolioSearchConfig,
    signal_data: dict[str, list[Candle]],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
) -> dict[str, Any]:
    candidate_cache: dict[tuple[tuple[str, ...], str], dict[int, list[Candidate]]] = {}
    alpha_portfolio = replace(frozen_portfolio, risk_per_trade_pct=0.02, max_open_positions=1)
    stages: dict[str, list[dict[str, Any]]] = {}

    lookbacks = sorted({max(2, frozen_signal.lookback_days - 1), frozen_signal.lookback_days, frozen_signal.lookback_days + 1})
    k_values = sorted({round(max(0.10, frozen_signal.long_k + offset), 2) for offset in (-0.10, -0.05, 0.0, 0.05, 0.10)})
    entry_variants = []
    for lookback in lookbacks:
        for long_k in k_values:
            for short_k in k_values:
                config = replace(frozen_signal, lookback_days=lookback, long_k=long_k, short_k=short_k)
                entry_variants.append((f"n{lookback}_kl{long_k:.2f}_ks{short_k:.2f}", config, alpha_portfolio))
    stages["stage5_local_range"] = run_experiment_set(
        "stage5_local_range", entry_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage5_local_range"])
    selected_signal = VolatilityBreakoutConfig(**best["signal_config"])

    if selected_signal.take_profit_r >= 4.0:
        targets = (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 30.0, 40.0, 60.0, 100.0)
    else:
        targets = (1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 3.0)
    stops = sorted({max(0.50, selected_signal.stop_atr_multiple + offset) for offset in (-0.50, -0.25, 0.0, 0.25, 0.50)})
    exit_variants = [
        (f"stop{stop:.2f}_tp{target:.2f}", replace(selected_signal, stop_atr_multiple=stop, take_profit_r=target), alpha_portfolio)
        for stop in stops
        for target in targets
    ]
    stages["stage6_local_exit"] = run_experiment_set(
        "stage6_local_exit", exit_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage6_local_exit"])
    selected_signal = VolatilityBreakoutConfig(**best["signal_config"])

    holding_variants = [
        (f"hold{holding}", replace(selected_signal, max_holding_minutes=holding), alpha_portfolio)
        for holding in (480, 720, 960, 1200, 1440, 1920)
    ]
    stages["stage6b_local_holding"] = run_experiment_set(
        "stage6b_local_holding", holding_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage6b_local_holding"])
    selected_signal = VolatilityBreakoutConfig(**best["signal_config"])

    portfolio_variants = []
    for daily in (3, 5, 8, 12):
        for cooldown in (60, 120, 180, 240, 360, 720, 1440):
            portfolio = replace(alpha_portfolio, max_daily_trades=daily, symbol_cooldown_minutes=cooldown)
            portfolio_variants.append((f"daily{daily}_cooldown{cooldown}", selected_signal, portfolio))
    stages["stage7_frequency"] = run_experiment_set(
        "stage7_frequency", portfolio_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage7_frequency"])
    selected_portfolio = PortfolioSearchConfig(**best["portfolio_config"])

    risk_variants = []
    for risk in (0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075):
        risk_variants.append(
            (f"risk{risk:.3f}", selected_signal, replace(selected_portfolio, risk_per_trade_pct=risk))
        )
    stages["stage8_local_risk"] = run_experiment_set(
        "stage8_local_risk", risk_variants, universe, signal_data, execution_data, rules,
        execution, start, end, initial_equity, candidate_cache,
    )
    best = select_best(stages["stage8_local_risk"])
    selected_portfolio = PortfolioSearchConfig(**best["portfolio_config"])
    candidates = candidate_cache[(universe, _entry_cache_key(selected_signal))]
    final_result = simulate_portfolio(
        candidates, universe, execution_data, rules, selected_signal, selected_portfolio,
        execution, start, end, initial_equity,
    )

    stress: dict[str, Any] = {}
    stress["fixed_risk_no_compounding"] = compact_summary(
        simulate_portfolio(
            candidates, universe, execution_data, rules, selected_signal,
            replace(selected_portfolio, compound=False), execution, start, end, initial_equity,
        )
    )
    for delay in (1, 2, 5):
        stress[f"entry_delay_{delay}m"] = compact_summary(
            simulate_portfolio(
                _shift_candidates(candidates, delay, execution_data), universe, execution_data, rules,
                selected_signal, selected_portfolio, execution, start, end, initial_equity,
            )
        )
    for multiplier in (1.5, 2.0):
        stressed_execution = replace(
            execution,
            market_slippage_bps=execution.market_slippage_bps * multiplier,
            stop_slippage_bps=execution.stop_slippage_bps * multiplier,
            take_profit_slippage_bps=execution.take_profit_slippage_bps * multiplier,
            taker_fee_rate=execution.taker_fee_rate * multiplier,
        )
        stress[f"cost_{multiplier:.1f}x"] = compact_summary(
            simulate_portfolio(
                candidates, universe, execution_data, rules, selected_signal, selected_portfolio,
                stressed_execution, start, end, initial_equity,
            )
        )
    ranked_trades = sorted(final_result["trades"], key=lambda trade: trade["net_pnl"], reverse=True)
    for count in (1, 3, 5):
        excluded = frozenset(trade["event_id"] for trade in ranked_trades[:count])
        stress[f"exclude_top_{count}_path"] = compact_summary(
            simulate_portfolio(
                candidates, universe, execution_data, rules, selected_signal, selected_portfolio,
                execution, start, end, initial_equity, skip_event_ids=excluded,
            )
        )
    top_symbol = max(final_result["by_symbol"], key=lambda key: final_result["by_symbol"][key]["net_pnl"])
    stress["exclude_top_symbol"] = {
        "excluded_symbol": top_symbol,
        "result": compact_summary(
            simulate_portfolio(
                _without_symbols(candidates, frozenset({top_symbol})), universe, execution_data, rules,
                selected_signal, selected_portfolio, execution, start, end, initial_equity,
            )
        ),
    }
    return {
        "selected_signal_config": selected_signal.as_dict(),
        "selected_portfolio_config": asdict(selected_portfolio),
        "stages": stages,
        "final_result": final_result,
        "stress_tests": stress,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Dual Thrust Volatility Breakout - 3 Month Full-Cost Research",
        "",
        f"- Period: `{report['period']['start']}` to `{report['period']['end']}`",
        f"- Initial equity: `{report['initial_equity']:.2f}U`",
        "- Execution: closed signal bar, next 1m open, conservative same-bar path, full cost",
        "- Status: historical optimization; not untouched OOS and not a live recommendation",
        "",
        "| Universe | Trades | Final equity | Net profit | Return | PF | Win rate | Max DD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("30", "50"):
        result = report["universes"][key]["final_result"]
        lines.append(
            f"| {key} | {result['trade_count']} | {result['final_equity']:.2f}U | "
            f"{result['net_profit']:+.2f}U | {result['return_pct']:.2%} | "
            f"{result['profit_factor']:.3f} | {result['win_rate']:.2%} | "
            f"{result['max_drawdown_pct']:.2%} |"
        )
    for key in ("30", "50"):
        universe = report["universes"][key]
        signal = universe["selected_signal_config"]
        portfolio = universe["selected_portfolio_config"]
        stress = universe.get("stress_tests", {})
        lines.extend(
            [
                "",
                f"## {key}-Symbol Frozen Historical Maximum",
                "",
                f"- Signal: `{signal['timeframe_minutes']}m`, N=`{signal['lookback_days']}`, "
                f"K long/short=`{signal['long_k']}/{signal['short_k']}`, stop=`{signal['stop_atr_multiple']} ATR`, "
                f"TP=`{signal['take_profit_r']}R`, max hold=`{signal['max_holding_minutes']}m`",
                f"- Portfolio: risk=`{portfolio['risk_per_trade_pct']:.2%}`, max positions=`{portfolio['max_open_positions']}`, "
                f"daily entries=`{portfolio['max_daily_trades']}`, cooldown=`{portfolio['symbol_cooldown_minutes']}m`",
            ]
        )
        for stress_name in (
            "fixed_risk_no_compounding", "entry_delay_1m", "cost_2.0x",
            "exclude_top_1_path", "exclude_top_symbol",
        ):
            if stress_name not in stress:
                continue
            row = stress[stress_name].get("result", stress[stress_name])
            lines.append(
                f"- Stress `{stress_name}`: net `{row['net_profit']:+.2f}U`, PF `{(row['profit_factor'] or 0.0):.3f}`, "
                f"DD `{row['max_drawdown_pct']:.2%}`"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The selected parameters maximize net profit inside a bounded staged search while requiring PF > 1, at least 20 trades, and max drawdown <= 50%.",
            "Because all three months were inspected during selection, these numbers are in-sample historical research. Freeze the selected config before any new shadow/dry-run assessment.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize an independent Dual Thrust volatility breakout strategy")
    parser.add_argument("--signal-data-dir", default="data/binance_15m_365d_top100")
    parser.add_argument("--execution-data-dir", default="data/binance_1m_365d_top100")
    parser.add_argument("--funding-data-dir", default="data/binance_funding_365d_top100")
    parser.add_argument("--cost-config", default="config.gui.mtf-momentum-reset-stage21.json")
    parser.add_argument("--start", default="2026-03-06T00:00:00")
    parser.add_argument("--end", default="2026-06-06T00:00:00")
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument("--output", default="reports/volatility_breakout_3m_30_vs_50.json")
    parser.add_argument("--summary", default="reports/volatility_breakout_3m_30_vs_50.md")
    parser.add_argument("--config30", default="config.volatility-breakout.optimized-30.json")
    parser.add_argument("--config50", default="config.volatility-breakout.optimized-50.json")
    parser.add_argument("--manifest", default="config.volatility-breakout.3m-optimized-manifest.json")
    parser.add_argument("--refine-only", action="store_true")
    parser.add_argument("--refine-universe", choices=("30", "50", "both"), default="both")
    return parser


def _load_runtime_inputs(
    args: argparse.Namespace,
) -> tuple[
    datetime,
    datetime,
    dict[str, list[Candle]],
    dict[str, CompactSeries],
    dict[str, SymbolRules],
    BacktestExecutionConfig,
    dict[str, Any],
]:
    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    if start >= end:
        raise ValueError("start must be before end")
    union = tuple(dict.fromkeys((*UNIVERSE_30, *UNIVERSE_50)))
    signal_data, execution_data, rules = load_research_data(
        union, args.signal_data_dir, args.execution_data_dir, start, end
    )
    missing30 = sorted(set(UNIVERSE_30) - set(signal_data) | (set(UNIVERSE_30) - set(execution_data)))
    missing50 = sorted(set(UNIVERSE_50) - set(signal_data) | (set(UNIVERSE_50) - set(execution_data)))
    if missing30 or missing50:
        raise RuntimeError(f"incomplete universe data: 30={missing30}, 50={missing50}")
    live_config = load_live_config(args.cost_config)
    execution = execution_config_from_live_config(live_config, cost_experiment="full_cost", mode="conservative")
    funding = load_funding_rate_directory(args.funding_data_dir, union)
    execution = replace(execution, funding_enabled=True, funding_default_rate=0.0, funding_rates_by_symbol=funding)
    metadata = {
        "symbols": union,
        "funding": funding,
        "funding_missing": sorted(set(union) - set(funding)),
    }
    print(
        f"data ready symbols={len(union)} funding={len(funding)} "
        f"missing_funding={len(metadata['funding_missing'])}",
        flush=True,
    )
    return start, end, signal_data, execution_data, rules, execution, metadata


def run_optimization(args: argparse.Namespace) -> dict[str, Any]:
    start, end, signal_data, execution_data, rules, execution, metadata = _load_runtime_inputs(args)
    funding = metadata["funding"]
    funding_missing = metadata["funding_missing"]

    result30 = optimize_universe(
        UNIVERSE_30, signal_data, execution_data, rules, execution, start, end, args.initial_equity
    )
    result50 = optimize_universe(
        UNIVERSE_50, signal_data, execution_data, rules, execution, start, end, args.initial_equity
    )
    report = {
        "strategy_name": VOLATILITY_BREAKOUT_STRATEGY_NAME,
        "strategy_version": VOLATILITY_BREAKOUT_VERSION,
        "research_status": "historical_in_sample_optimization_not_untouched_oos",
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "initial_equity": args.initial_equity,
        "cost_model": {
            "source_config": args.cost_config,
            "source_config_sha256": sha256_file(args.cost_config),
            "mode": execution.mode,
            "market_slippage_bps": execution.market_slippage_bps,
            "stop_slippage_bps": execution.stop_slippage_bps,
            "take_profit_slippage_bps": execution.take_profit_slippage_bps,
            "taker_fee_rate": execution.taker_fee_rate,
            "maker_fee_rate": execution.maker_fee_rate,
            "funding_enabled": execution.funding_enabled,
            "funding_data_dir": args.funding_data_dir,
            "funding_symbols": len(funding),
            "funding_missing_symbols": funding_missing,
        },
        "execution_rules": {
            "signal_confirmation": "closed_15m_30m_or_60m_bar",
            "entry": "next_1m_open",
            "old_positions_exit_before_new_entries": True,
            "same_bar_conflict": "stop_first",
            "gap_stop": "worse_open_price",
            "point_in_time_data": True,
        },
        "selection_rule": {
            "primary": "maximum_net_profit",
            "minimum_trades": 20,
            "minimum_profit_factor": 1.0,
            "maximum_drawdown_pct": 0.50,
            "note": "bounded staged grid, not an assertion of global optimum",
        },
        "universes": {"30": result30, "50": result50},
    }
    _write_json(Path(args.output), report)
    _write_markdown(Path(args.summary), report)

    for size, target, result in (
        (30, Path(args.config30), result30),
        (50, Path(args.config50), result50),
    ):
        _write_json(
            target,
            {
                "strategy_name": VOLATILITY_BREAKOUT_STRATEGY_NAME,
                "strategy_version": VOLATILITY_BREAKOUT_VERSION,
                "status": "frozen_historical_research_not_live",
                "period": report["period"],
                "universe_size": size,
                "symbols": result["symbols"],
                "signal": result["selected_signal_config"],
                "portfolio": result["selected_portfolio_config"],
                "cost_model": report["cost_model"],
                "execution_rules": report["execution_rules"],
            },
        )

    mtf_assets = (
        "config.gui.mtf-momentum-reset-stage21.json",
        "config.mtf-htf.momentum-reset-stage21-gui-dryrun-manifest.json",
        "reports/mtf_momentum_reset_full12m_stage21_path.json",
        "reports/mtf_momentum_reset_full12m_stage21_summary.md",
    )
    manifest = {
        "strategy_name": VOLATILITY_BREAKOUT_STRATEGY_NAME,
        "status": "historical_research_frozen",
        "report": args.output,
        "summary": args.summary,
        "configs": [args.config30, args.config50],
        "implementation_files": [
            "crypto_scalper/volatility_breakout.py",
            "crypto_scalper/volatility_breakout_optimize.py",
            "tests/test_volatility_breakout.py",
        ],
        "hashes": {
            path: sha256_file(path)
            for path in (
                args.output, args.summary, args.config30, args.config50,
                "crypto_scalper/volatility_breakout.py",
                "crypto_scalper/volatility_breakout_optimize.py",
                "tests/test_volatility_breakout.py",
                *mtf_assets,
            )
            if Path(path).exists()
        },
        "preserved_mtf_assets": list(mtf_assets),
    }
    _write_json(Path(args.manifest), manifest)
    return report


def run_refinement(args: argparse.Namespace) -> dict[str, Any]:
    report_path = Path(args.output)
    if not report_path.exists():
        raise FileNotFoundError(f"base optimization report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    start, end, signal_data, execution_data, rules, execution, metadata = _load_runtime_inputs(args)
    selected_universes = {
        "30": (("30", UNIVERSE_30),),
        "50": (("50", UNIVERSE_50),),
        "both": (("30", UNIVERSE_30), ("50", UNIVERSE_50)),
    }[args.refine_universe]
    for key, universe in selected_universes:
        base = report["universes"][key]
        signal = VolatilityBreakoutConfig(**base["selected_signal_config"])
        portfolio = PortfolioSearchConfig(**base["selected_portfolio_config"])
        refined = refine_universe(
            universe, signal, portfolio, signal_data, execution_data, rules, execution,
            start, end, args.initial_equity,
        )
        base["pre_refinement_final_result"] = compact_summary(base["final_result"])
        base["refinement_stages"] = refined["stages"]
        base["selected_signal_config"] = refined["selected_signal_config"]
        base["selected_portfolio_config"] = refined["selected_portfolio_config"]
        base["final_result"] = refined["final_result"]
        base["stress_tests"] = refined["stress_tests"]
    report["refinement"] = {
        "completed": True,
        "local_parameter_search": True,
        "stress_tests": [
            "fixed_risk_no_compounding", "entry_delay_1m", "entry_delay_2m", "entry_delay_5m",
            "cost_1.5x", "cost_2.0x", "exclude_top_1_path", "exclude_top_3_path",
            "exclude_top_5_path", "exclude_top_symbol",
        ],
    }
    _write_json(report_path, report)
    _write_markdown(Path(args.summary), report)
    for size, target in (("30", Path(args.config30)), ("50", Path(args.config50))):
        result = report["universes"][size]
        _write_json(
            target,
            {
                "strategy_name": VOLATILITY_BREAKOUT_STRATEGY_NAME,
                "strategy_version": VOLATILITY_BREAKOUT_VERSION,
                "status": "frozen_historical_research_not_live",
                "period": report["period"],
                "universe_size": int(size),
                "symbols": result["symbols"],
                "signal": result["selected_signal_config"],
                "portfolio": result["selected_portfolio_config"],
                "cost_model": report["cost_model"],
                "execution_rules": report["execution_rules"],
                "stress_tests": result["stress_tests"],
            },
        )
    mtf_assets = (
        "config.gui.mtf-momentum-reset-stage21.json",
        "config.mtf-htf.momentum-reset-stage21-gui-dryrun-manifest.json",
        "reports/mtf_momentum_reset_full12m_stage21_path.json",
        "reports/mtf_momentum_reset_full12m_stage21_summary.md",
    )
    _write_json(
        Path(args.manifest),
        {
            "strategy_name": VOLATILITY_BREAKOUT_STRATEGY_NAME,
            "status": "historical_research_frozen_after_local_refinement",
            "report": args.output,
            "summary": args.summary,
            "configs": [args.config30, args.config50],
            "implementation_files": [
                "crypto_scalper/volatility_breakout.py",
                "crypto_scalper/volatility_breakout_optimize.py",
                "tests/test_volatility_breakout.py",
            ],
            "hashes": {
                path: sha256_file(path)
                for path in (
                    args.output, args.summary, args.config30, args.config50,
                    "crypto_scalper/volatility_breakout.py",
                    "crypto_scalper/volatility_breakout_optimize.py",
                    "tests/test_volatility_breakout.py",
                    *mtf_assets,
                )
                if Path(path).exists()
            },
            "preserved_mtf_assets": list(mtf_assets),
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_refinement(args) if args.refine_only else run_optimization(args)
    concise = {
        size: compact_summary(report["universes"][size]["final_result"])
        for size in ("30", "50")
    }
    print(json.dumps(_json_safe(concise), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

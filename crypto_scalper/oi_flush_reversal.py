from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .binance_client import SymbolRules
from .data import load_candles_csv, parse_timestamp
from .indicators import atr, ema
from .live_config import load_live_config
from .live_portfolio_backtest import _load_symbol_data
from .models import Candle, Direction, Signal
from .risk import (
    BacktestExecutionConfig,
    conservative_quantity,
    execution_config_from_live_config,
    funding_cashflow,
    market_entry_fill,
    market_exit_fill,
    signal_risk_weight,
    validate_order_size,
)


BINANCE_FUTURES_DATA_URL = "https://fapi.binance.com/futures/data"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


@dataclass(frozen=True)
class OiFeature:
    timestamp: datetime
    available_time: datetime
    sum_open_interest_value: float
    oi_chg_15m: float | None
    oi_chg_30m: float | None


@dataclass(frozen=True)
class TakerFeature:
    timestamp: datetime
    available_time: datetime
    buy_vol: float
    sell_vol: float
    buy_sell_ratio: float
    sell_imbalance_5m: float
    sell_imbalance_15m: float | None


@dataclass(frozen=True)
class FundingFeature:
    timestamp: datetime
    funding_rate: float


@dataclass(frozen=True)
class OiFlushFeature:
    oi_chg_15m: float
    oi_chg_30m: float
    sell_imbalance_5m: float
    sell_imbalance_15m: float
    funding_rate: float
    btc_ret_15m: float
    btc_state: str
    funding_bucket: str
    oi_drop_bucket: str
    sell_imbalance_bucket: str


@dataclass
class PendingFlush:
    symbol: str
    flush_low: float
    flush_time: datetime
    flush_index: int
    atr_pct: float
    atr_value: float
    funding_rate: float
    oi_chg_15m: float
    oi_chg_30m: float
    sell_imbalance_15m: float
    btc_ret_15m: float
    btc_state: str


@dataclass
class PendingEntry:
    symbol: str
    signal: Signal
    flush: PendingFlush
    signal_time: datetime
    signal_available_time: datetime
    earliest_execution_index: int


@dataclass
class OiFlushPosition:
    symbol: str
    quantity: float
    entry_price: float
    raw_entry_price: float
    entry_time: datetime
    entry_index: int
    stop_price: float
    take_profit_price: float
    stop_loss_pct: float
    take_profit_pct: float
    entry_fee: float
    entry_slippage_cost: float
    signal_time: datetime
    signal_available_time: datetime
    flush: PendingFlush
    notional: float
    mfe: float = 0.0
    mae: float = 0.0
    best_price: float = 0.0
    worst_price: float = 0.0


@dataclass
class OiFlushStats:
    pending_count: int = 0
    pending_expired_count: int = 0
    entry_count: int = 0
    fail_fast_count: int = 0
    time_stop_count: int = 0
    low_broken_count: int = 0
    skipped_missing_features: int = 0
    skipped_daily_limit: int = 0
    skipped_cooldown: int = 0
    skipped_position_limit: int = 0


@dataclass(frozen=True)
class Experiment:
    name: str
    overrides: dict[str, Any]
    top30_only: bool = False


def run_oi_flush_experiments(
    config_path: str,
    price_data_dir: str,
    feature_data_dir: str,
    funding_data_dir: str,
    initial_equity: float | None = None,
    include_trades: bool = False,
    download_missing: bool = False,
    sleep_seconds: float = 0.08,
) -> dict[str, Any]:
    experiments = [
        Experiment("A_base", {}),
        Experiment("B_oi_drop_4pct", {"oi_flush_oi_drop_15m": -0.040}),
        Experiment("C_sell_imbalance_068", {"oi_flush_sell_imbalance_15m": 0.68}),
        Experiment("D_btc_ret_gt_minus_0003", {"oi_flush_btc_min_ret_15m": -0.003}),
        Experiment("E_funding_lte_0", {"oi_flush_max_funding_rate": 0.0}),
        Experiment("F_close_position_070", {"oi_flush_min_close_position": 0.70}),
        Experiment("G_tp_22_time_stop_60", {"oi_flush_take_profit_r": 2.2, "oi_flush_time_stop_minutes": 60}),
        Experiment("H_high_liquidity_top30_only", {}, True),
    ]
    output: dict[str, Any] = {"experiments": []}
    for experiment in experiments:
        no_cost = run_oi_flush_backtest(
            config_path,
            price_data_dir,
            feature_data_dir,
            funding_data_dir,
            initial_equity=initial_equity,
            include_trades=include_trades,
            cost_experiment="no_cost",
            strategy_overrides=experiment.overrides,
            top30_only=experiment.top30_only,
            download_missing=download_missing,
            sleep_seconds=sleep_seconds,
        )
        full_cost = run_oi_flush_backtest(
            config_path,
            price_data_dir,
            feature_data_dir,
            funding_data_dir,
            initial_equity=initial_equity,
            include_trades=include_trades,
            cost_experiment="full_cost",
            strategy_overrides=experiment.overrides,
            top30_only=experiment.top30_only,
            download_missing=False,
            sleep_seconds=sleep_seconds,
        )
        output["experiments"].append(
            {
                "name": experiment.name,
                "overrides": experiment.overrides,
                "top30_only": experiment.top30_only,
                "no_cost": _compact_result(no_cost),
                "full_cost": _compact_result(full_cost),
            }
        )
    return output


def run_oi_flush_backtest(
    config_path: str,
    price_data_dir: str,
    feature_data_dir: str,
    funding_data_dir: str,
    initial_equity: float | None = None,
    include_trades: bool = False,
    cost_experiment: str = "full_cost",
    strategy_overrides: dict[str, Any] | None = None,
    top30_only: bool = False,
    download_missing: bool = False,
    sleep_seconds: float = 0.08,
) -> dict[str, Any]:
    config = load_live_config(config_path)
    strategy = config.strategy
    if strategy_overrides:
        strategy = replace(strategy, **strategy_overrides)
        config = replace(config, strategy=strategy)
    config = _single_strategy_config(config, top30_only=top30_only)
    if not getattr(config.strategy, "oi_flush_reversal_enabled", True):
        raise RuntimeError("oi_flush_reversal_enabled is false")

    candles_by_symbol = _load_symbol_data(price_data_dir, tuple(config.trading.symbols), "1m")
    if not candles_by_symbol:
        raise RuntimeError(f"no 1m price data loaded from {price_data_dir}")
    symbols = tuple(symbol for symbol in config.trading.symbols if symbol in candles_by_symbol)
    if top30_only:
        symbols = symbols[:30]
    candles_by_symbol = {symbol: candles_by_symbol[symbol] for symbol in symbols}
    btc_candles = candles_by_symbol.get("BTCUSDT")
    if not btc_candles:
        raise RuntimeError("BTCUSDT 1m data is required for oi_flush_reversal_long")

    start = min(candle.timestamp for candles in candles_by_symbol.values() for candle in candles[:1])
    end = max(candles[-1].timestamp for candles in candles_by_symbol.values() if candles)
    if download_missing:
        download_oi_flush_feature_data(
            symbols,
            feature_data_dir,
            funding_data_dir,
            start,
            end + timedelta(minutes=1),
            sleep_seconds=sleep_seconds,
        )

    feature_series = {
        symbol: build_oi_flush_feature_series(symbol, feature_data_dir, funding_data_dir)
        for symbol in symbols
    }
    usable_symbols = tuple(
        symbol
        for symbol in symbols
        if feature_series[symbol]["oi"] and feature_series[symbol]["taker"]
    )
    candles_by_symbol = {symbol: candles_by_symbol[symbol] for symbol in usable_symbols}
    if not candles_by_symbol:
        return _empty_summary(initial_equity or config.risk.starting_capital_usdt, "missing_oi_or_taker_features")

    common_length = min(len(candles) for candles in candles_by_symbol.values())
    btc_ret_15m = _btc_ret_by_time(btc_candles)
    indicator_cache = {
        symbol: _price_indicator_cache(candles)
        for symbol, candles in candles_by_symbol.items()
    }
    execution_config = execution_config_from_live_config(config, cost_experiment=cost_experiment, mode="conservative")
    include_funding_cost = cost_experiment == "full_cost"

    starting_equity = initial_equity if initial_equity is not None else config.risk.starting_capital_usdt
    cash = starting_equity
    peak_equity = starting_equity
    equity_curve = [starting_equity]
    position: OiFlushPosition | None = None
    flush_pending: dict[str, PendingFlush] = {}
    entry_pending: list[PendingEntry] = []
    trades: list[dict[str, Any]] = []
    stats = OiFlushStats()
    daily_entries: dict[str, int] = {}
    cooldown_until: dict[str, datetime] = {}
    no_cost_mode = cost_experiment == "no_cost"
    rules = {symbol: _inferred_symbol_rules(symbol, candles_by_symbol[symbol], no_cost_mode=no_cost_mode) for symbol in candles_by_symbol}

    timestamps = [candle.timestamp for candle in next(iter(candles_by_symbol.values()))[:common_length]]
    for index, timestamp in enumerate(timestamps):
        if position is not None:
            candle = candles_by_symbol[position.symbol][index]
            cash, closed = _manage_oi_flush_position(
                config,
                execution_config,
                include_funding_cost,
                cash,
                position,
                candle,
                index,
                feature_series[position.symbol]["funding"],
                rules[position.symbol],
            )
            if closed:
                trades.append(closed)
                if closed["exit_reason"] == "oi_flush_fail_fast":
                    stats.fail_fast_count += 1
                elif closed["exit_reason"] == "oi_flush_time_stop":
                    stats.time_stop_count += 1
                elif closed["exit_reason"] == "oi_flush_low_broken":
                    stats.low_broken_count += 1
                cooldown_until[position.symbol] = timestamp + timedelta(hours=int(config.strategy.oi_flush_symbol_cooldown_hours))
                position = None

        if position is None and entry_pending:
            remaining_entries = []
            for pending in entry_pending:
                if pending.earliest_execution_index > index:
                    remaining_entries.append(pending)
                    continue
                if position is not None:
                    remaining_entries.append(pending)
                    continue
                candle = candles_by_symbol[pending.symbol][index]
                opened = _open_oi_flush_position(
                    config,
                    execution_config,
                    cash,
                    pending,
                    candle,
                    index,
                    rules[pending.symbol],
                )
                if opened is None:
                    continue
                position, entry_cash_delta = opened
                cash += entry_cash_delta
                stats.entry_count += 1
                day_key = timestamp.date().isoformat()
                daily_entries[day_key] = daily_entries.get(day_key, 0) + 1
            entry_pending = remaining_entries

        mark_equity = cash
        if position is not None:
            mark = candles_by_symbol[position.symbol][index].close
            mark_equity += position.quantity * (mark - position.entry_price)
        peak_equity = max(peak_equity, mark_equity)
        equity_curve.append(mark_equity)

        if position is not None or entry_pending:
            continue
        if mark_equity <= 0:
            continue
        if _max_drawdown_pct(equity_curve) >= max(0.0, float(config.risk.max_drawdown_pct)) * 100.0 and config.risk.max_drawdown_pct > 0:
            continue

        day_key = timestamp.date().isoformat()
        if daily_entries.get(day_key, 0) >= int(config.strategy.oi_flush_max_daily_trades):
            stats.skipped_daily_limit += 1
            continue

        for symbol in usable_symbols:
            if cooldown_until.get(symbol) and timestamp < cooldown_until[symbol]:
                stats.skipped_cooldown += 1
                continue
            candles = candles_by_symbol[symbol]
            if index >= len(candles):
                continue
            feature = _feature_at(feature_series[symbol], timestamp, btc_ret_15m.get(timestamp))
            if feature is None:
                stats.skipped_missing_features += 1
                continue
            candle = candles[index]
            indicators = indicator_cache[symbol]
            if index < 30:
                continue
            pending_flush = flush_pending.get(symbol)
            if pending_flush and index - pending_flush.flush_index > int(config.strategy.oi_flush_pending_bars):
                stats.pending_expired_count += 1
                flush_pending.pop(symbol, None)
                pending_flush = None
            if pending_flush:
                entry = _confirm_oi_flush_entry(config, pending_flush, candle, indicators, feature, index)
                if entry:
                    entry_pending.append(
                        PendingEntry(
                            symbol=symbol,
                            signal=entry,
                            flush=pending_flush,
                            signal_time=timestamp,
                            signal_available_time=timestamp + timedelta(minutes=1),
                            earliest_execution_index=index + 1,
                        )
                    )
                    flush_pending.pop(symbol, None)
                    break

            if symbol in flush_pending:
                continue
            new_flush = _detect_oi_flush(config, symbol, candles, indicators, feature, index)
            if new_flush:
                flush_pending[symbol] = new_flush
                stats.pending_count += 1

    if position is not None:
        last_candle = candles_by_symbol[position.symbol][common_length - 1]
        cash, closed = _close_oi_flush_position(
            execution_config,
            include_funding_cost,
            cash,
            position,
            last_candle.close,
            "end_of_data",
            last_candle.timestamp,
            feature_series[position.symbol]["funding"],
            rules[position.symbol],
        )
        trades.append(closed)
        position = None

    final_equity = cash
    equity_curve.append(final_equity)
    summary = _summary(
        starting_equity,
        final_equity,
        equity_curve,
        trades,
        stats,
        feature_data_dir,
        funding_data_dir,
        usable_symbols,
        cost_experiment,
        include_trades=include_trades,
    )
    return summary


def _single_strategy_config(config: Any, top30_only: bool = False) -> Any:
    strategy = replace(
        config.strategy,
        allow_short=False,
        rsi_reversal_enabled=False,
        ordinary_breakout_enabled=False,
        pullback_reclaim_enabled=False,
        fast_breakout_enabled=False,
        startup_breakout_enabled=False,
        oi_flush_reversal_enabled=True,
    )
    trading_symbols = tuple(config.trading.symbols[:30] if top30_only else config.trading.symbols)
    trading = replace(
        config.trading,
        symbols=trading_symbols,
        entry_symbols=trading_symbols,
        max_open_positions=1,
        super_volume_extra_slot_enabled=False,
        max_new_entries_per_cycle=1,
    )
    filters = replace(config.filters, extreme_reversal_entry_enabled=False, pre_cross_entry_enabled=False)
    return replace(config, strategy=strategy, trading=trading, filters=filters)


def _price_indicator_cache(candles: list[Candle]) -> dict[str, list[float]]:
    closes = [candle.close for candle in candles]
    return {
        "atr": atr(candles, 14),
        "ema": ema(closes, 9),
        "ema_9": ema(closes, 9),
        "ema_by_period": {},
    }


def _detect_oi_flush(
    config: Any,
    symbol: str,
    candles: list[Candle],
    indicators: dict[str, list[float]],
    feature: OiFlushFeature,
    index: int,
) -> PendingFlush | None:
    strategy = config.strategy
    candle = candles[index]
    previous = candles[index - 15].close
    if previous <= 0 or candle.close <= 0:
        return None
    ret_15m = candle.close / previous - 1.0
    atr_value = max(indicators["atr"][index], 1e-12)
    atr_pct = atr_value / candle.close
    drop_threshold = max(float(strategy.oi_flush_price_drop_min_pct), float(strategy.oi_flush_price_drop_atr_mult) * atr_pct)
    if ret_15m > -drop_threshold:
        return None
    if feature.oi_chg_15m > float(strategy.oi_flush_oi_drop_15m):
        return None
    if feature.sell_imbalance_15m < float(strategy.oi_flush_sell_imbalance_15m):
        return None
    if feature.btc_ret_15m <= float(strategy.oi_flush_btc_min_ret_15m):
        return None
    if feature.funding_rate > float(strategy.oi_flush_max_funding_rate):
        return None
    return PendingFlush(
        symbol=symbol,
        flush_low=candle.low,
        flush_time=candle.timestamp,
        flush_index=index,
        atr_pct=atr_pct,
        atr_value=atr_value,
        funding_rate=feature.funding_rate,
        oi_chg_15m=feature.oi_chg_15m,
        oi_chg_30m=feature.oi_chg_30m,
        sell_imbalance_15m=feature.sell_imbalance_15m,
        btc_ret_15m=feature.btc_ret_15m,
        btc_state=feature.btc_state,
    )


def _confirm_oi_flush_entry(
    config: Any,
    pending: PendingFlush,
    candle: Candle,
    indicators: dict[str, list[float]],
    feature: OiFlushFeature,
    index: int,
) -> Signal | None:
    strategy = config.strategy
    no_new_low = candle.low >= pending.flush_low * 0.998
    sell_exhausted = feature.sell_imbalance_5m <= float(strategy.oi_flush_sell_exhaustion_5m)
    reclaim_period = max(2, int(strategy.oi_flush_reclaim_ema_period))
    ema_values = indicators.get(f"ema_{reclaim_period}")
    if ema_values is None:
        return None
    ema_reclaim = candle.close > ema_values[index]
    candle_range = max(candle.high - candle.low, 1e-12)
    close_position = (candle.close - candle.low) / candle_range
    strong_close = close_position >= float(strategy.oi_flush_min_close_position)
    if not (no_new_low and sell_exhausted and ema_reclaim and strong_close):
        return None
    risk_multiplier = max(0.35, min(0.60, float(getattr(strategy, "oi_flush_risk_multiplier", 0.40))))
    return Signal(
        Direction.LONG,
        confidence=0.62,
        reason="long_oi_flush_reversal",
        stop_loss_pct=0.001,
        take_profit_pct=0.0017,
        risk_multiplier=risk_multiplier,
        max_holding_bars=max(1, int(strategy.oi_flush_time_stop_minutes)),
    )


def _open_oi_flush_position(
    config: Any,
    execution_config: BacktestExecutionConfig,
    cash: float,
    pending: PendingEntry,
    candle: Candle,
    index: int,
    rules: SymbolRules,
) -> tuple[OiFlushPosition, float] | None:
    strategy = config.strategy
    entry_price_estimate = candle.open
    structural_stop = min(
        pending.flush.flush_low * 0.998,
        entry_price_estimate - pending.flush.atr_value * float(strategy.oi_flush_stop_atr_mult),
    )
    stop_loss_pct = (entry_price_estimate - structural_stop) / max(entry_price_estimate, 1e-12)
    stop_loss_pct = max(0.0015, min(float(strategy.oi_flush_max_stop_pct), stop_loss_pct))
    take_profit_pct = max(stop_loss_pct * float(strategy.oi_flush_take_profit_r), stop_loss_pct * 1.05)
    signal = replace(pending.signal, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
    notional, reason = _size_notional(config, cash, entry_price_estimate, signal)
    if reason != "ok" or notional <= 0:
        return None
    quantity = conservative_quantity(rules, notional / entry_price_estimate)
    if validate_order_size(rules, quantity, entry_price_estimate, config.risk.min_order_notional_usdt) != "ok":
        return None
    fill = market_entry_fill(execution_config, rules, Direction.LONG, quantity, candle.open)
    stop_price = fill.price * (1.0 - stop_loss_pct)
    take_profit_price = fill.price * (1.0 + take_profit_pct)
    position = OiFlushPosition(
        symbol=pending.symbol,
        quantity=quantity,
        entry_price=fill.price,
        raw_entry_price=candle.open,
        entry_time=candle.timestamp,
        entry_index=index,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        entry_fee=fill.fee,
        entry_slippage_cost=fill.slippage_cost,
        signal_time=pending.signal_time,
        signal_available_time=pending.signal_available_time,
        flush=pending.flush,
        notional=abs(quantity * fill.price),
        best_price=fill.price,
        worst_price=fill.price,
    )
    return position, -fill.fee


def _size_notional(config: Any, equity: float, price: float, signal: Signal) -> tuple[float, str]:
    if equity < config.risk.min_available_balance_usdt:
        return 0.0, "available_balance_too_low"
    if price <= 0 or signal.stop_loss_pct <= 0:
        return 0.0, "bad_price_or_stop"
    leverage = max(1, int(config.trading.leverage))
    risk_weight = signal_risk_weight(signal.confidence, signal.risk_multiplier)
    risk_notional = equity * config.risk.risk_per_trade_pct * risk_weight / signal.stop_loss_pct
    symbol_cap = equity * config.risk.max_symbol_margin_pct * leverage
    account_cap = equity * config.risk.max_account_margin_usage_pct * leverage
    policy_cap = config.risk.max_position_notional_usdt if config.risk.max_position_notional_usdt > 0 else float("inf")
    notional = min(risk_notional, symbol_cap, account_cap, policy_cap)
    notional *= max(0.0, min(1.0, config.trading.initial_entry_fraction))
    if notional < config.risk.min_order_notional_usdt:
        return 0.0, "below_min_notional"
    return notional, "ok"


def _inferred_symbol_rules(symbol: str, candles: list[Candle], no_cost_mode: bool = False) -> SymbolRules:
    if no_cost_mode:
        return SymbolRules(symbol, "0.000001", "0.000001", "0.000000001", "0")
    price = statistics.median(candle.close for candle in candles[: min(len(candles), 1440)] if candle.close > 0)
    if price >= 1000:
        tick = "0.1"
    elif price >= 100:
        tick = "0.01"
    elif price >= 1:
        tick = "0.001"
    elif price >= 0.1:
        tick = "0.0001"
    elif price >= 0.01:
        tick = "0.00001"
    else:
        tick = "0.000001"
    return SymbolRules(symbol, "0.001", "0.001", tick, "5")


def _manage_oi_flush_position(
    config: Any,
    execution_config: BacktestExecutionConfig,
    include_funding_cost: bool,
    cash: float,
    position: OiFlushPosition,
    candle: Candle,
    index: int,
    funding_rates: list[FundingFeature],
    rules: SymbolRules,
) -> tuple[float, dict[str, Any] | None]:
    position.best_price = max(position.best_price, candle.high)
    position.worst_price = min(position.worst_price, candle.low)
    position.mfe = max(position.mfe, (position.best_price - position.entry_price) * position.quantity)
    position.mae = min(position.mae, (position.worst_price - position.entry_price) * position.quantity)
    stop_hit = candle.low <= position.stop_price
    take_profit_hit = candle.high >= position.take_profit_price
    reason = ""
    raw_exit = 0.0
    if stop_hit:
        reason = "stop_loss"
        raw_exit = position.stop_price
    elif take_profit_hit:
        reason = "take_profit"
        raw_exit = position.take_profit_price
    elif candle.low < position.flush.flush_low:
        reason = "oi_flush_low_broken"
        raw_exit = candle.close
    else:
        held_minutes = max(0, index - position.entry_index)
        if held_minutes >= int(config.strategy.oi_flush_fail_fast_minutes):
            required_mfe = position.notional * position.stop_loss_pct * float(config.strategy.oi_flush_fail_fast_min_r)
            if position.mfe < required_mfe:
                reason = "oi_flush_fail_fast"
                raw_exit = candle.close
        if not reason and held_minutes >= int(config.strategy.oi_flush_time_stop_minutes):
            reason = "oi_flush_time_stop"
            raw_exit = candle.close
    if not reason:
        return cash, None
    return _close_oi_flush_position(
        execution_config,
        include_funding_cost,
        cash,
        position,
        raw_exit,
        reason,
        candle.timestamp,
        funding_rates,
        rules,
    )


def _close_oi_flush_position(
    execution_config: BacktestExecutionConfig,
    include_funding_cost: bool,
    cash: float,
    position: OiFlushPosition,
    raw_exit_price: float,
    reason: str,
    exit_time: datetime,
    funding_rates: list[FundingFeature],
    rules: SymbolRules,
) -> tuple[float, dict[str, Any]]:
    order_type = "stop_market" if reason == "stop_loss" else "take_profit_market" if reason == "take_profit" else "market"
    fill = market_exit_fill(execution_config, rules, Direction.LONG, position.quantity, raw_exit_price, order_type)
    raw_gross_pnl = position.quantity * (raw_exit_price - position.raw_entry_price)
    execution_gross_pnl = position.quantity * (fill.price - position.entry_price)
    fee = position.entry_fee + fill.fee
    slippage = position.entry_slippage_cost + fill.slippage_cost
    funding_rates_values = [
        item.funding_rate
        for item in funding_rates
        if position.entry_time < item.timestamp <= exit_time
    ]
    funding = funding_cashflow(Direction.LONG, position.notional, funding_rates_values) if include_funding_cost else 0.0
    net_pnl = raw_gross_pnl - fee - slippage + funding
    cash += execution_gross_pnl - fill.fee + funding
    hold_minutes = max(0.0, (exit_time - position.entry_time).total_seconds() / 60.0)
    trade = {
        "symbol": position.symbol,
        "strategy": "oi_flush_reversal_long",
        "strategy_bucket": "oi_flush_reversal_long",
        "side": "LONG",
        "direction": "LONG",
        "entry_time": position.entry_time.isoformat(),
        "entry_price": position.entry_price,
        "raw_entry_price": position.raw_entry_price,
        "exit_time": exit_time.isoformat(),
        "exit_price": fill.price,
        "raw_exit_price": raw_exit_price,
        "qty": position.quantity,
        "quantity": position.quantity,
        "notional": position.notional,
        "gross_pnl": raw_gross_pnl,
        "execution_gross_pnl": execution_gross_pnl,
        "fee": fee,
        "fees": fee,
        "slippage": slippage,
        "slippage_cost": slippage,
        "funding": funding,
        "net_pnl": net_pnl,
        "mfe": position.mfe,
        "mae": position.mae,
        "hold_minutes": hold_minutes,
        "exit_reason": reason,
        "entry_order_type": "market",
        "exit_order_type": fill.order_type,
        "entry_liquidity": "taker",
        "exit_liquidity": fill.liquidity,
        "signal_time": position.signal_time.isoformat(),
        "signal_available_time": position.signal_available_time.isoformat(),
        "flush_time": position.flush.flush_time.isoformat(),
        "flush_low": position.flush.flush_low,
        "funding_rate": position.flush.funding_rate,
        "oi_chg_15m": position.flush.oi_chg_15m,
        "oi_chg_30m": position.flush.oi_chg_30m,
        "sell_imbalance_15m": position.flush.sell_imbalance_15m,
        "btc_ret_15m": position.flush.btc_ret_15m,
        "btc_state": position.flush.btc_state,
        "funding_bucket": _funding_bucket(position.flush.funding_rate),
        "oi_drop_bucket": _oi_drop_bucket(position.flush.oi_chg_15m),
        "sell_imbalance_bucket": _sell_imbalance_bucket(position.flush.sell_imbalance_15m),
        "return_pct": net_pnl / max(position.notional, 1e-12) * 100.0,
    }
    return cash, trade


def build_oi_flush_feature_series(symbol: str, feature_data_dir: str, funding_data_dir: str) -> dict[str, Any]:
    oi_rows = build_oi_features(load_oi_csv(feature_data_dir, symbol))
    taker_rows = build_taker_features(load_taker_csv(feature_data_dir, symbol))
    funding_rows = load_funding_csv(funding_data_dir, symbol)
    return {
        "oi": oi_rows,
        "oi_available_times": [item.available_time for item in oi_rows],
        "taker": taker_rows,
        "taker_available_times": [item.available_time for item in taker_rows],
        "funding": funding_rows,
        "funding_times": [item.timestamp for item in funding_rows],
    }


def load_oi_csv(data_dir: str, symbol: str) -> list[dict[str, Any]]:
    return _load_feature_csv(
        data_dir,
        symbol,
        ("*oi_5m*.csv", "*open_interest_5m*.csv", "*openInterest_5m*.csv"),
        required_any=("sumOpenInterestValue", "sum_open_interest_value"),
    )


def load_taker_csv(data_dir: str, symbol: str) -> list[dict[str, Any]]:
    return _load_feature_csv(
        data_dir,
        symbol,
        ("*taker_5m*.csv", "*taker_buy_sell_5m*.csv", "*takerlongshort_5m*.csv"),
        required_any=("buyVol", "buy_vol"),
    )


def _load_feature_csv(data_dir: str, symbol: str, patterns: tuple[str, ...], required_any: tuple[str, ...]) -> list[dict[str, Any]]:
    root = Path(data_dir)
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(f"{symbol}_{pattern}"))
    if not matches:
        return []
    path = sorted(matches)[-1]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not any(name in fields for name in required_any):
            return []
        rows = [dict(row) for row in reader]
    rows.sort(key=lambda row: _row_time(row))
    return rows


def load_funding_csv(data_dir: str, symbol: str) -> list[FundingFeature]:
    root = Path(data_dir)
    matches = sorted(root.glob(f"{symbol}_funding_*.csv"))
    if not matches:
        return []
    with matches[-1].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            timestamp = parse_timestamp(row.get("funding_time") or row.get("timestamp") or row.get("time") or "")
            rate = float(row.get("funding_rate") or row.get("fundingRate") or 0.0)
            rows.append(FundingFeature(timestamp, rate))
    rows.sort(key=lambda item: item.timestamp)
    return rows


def build_oi_features(rows: list[dict[str, Any]]) -> list[OiFeature]:
    values = []
    for row in sorted(rows, key=_row_time):
        value = float(row.get("sumOpenInterestValue") or row.get("sum_open_interest_value") or 0.0)
        values.append((_row_time(row), value))
    output: list[OiFeature] = []
    for index, (timestamp, value) in enumerate(values):
        chg_15 = _pct_change(value, values[index - 3][1]) if index >= 3 else None
        chg_30 = _pct_change(value, values[index - 6][1]) if index >= 6 else None
        output.append(OiFeature(timestamp, timestamp + timedelta(minutes=5), value, chg_15, chg_30))
    return output


def build_taker_features(rows: list[dict[str, Any]]) -> list[TakerFeature]:
    items = []
    for row in sorted(rows, key=_row_time):
        buy = float(row.get("buyVol") or row.get("buy_vol") or 0.0)
        sell = float(row.get("sellVol") or row.get("sell_vol") or 0.0)
        ratio = float(row.get("buySellRatio") or row.get("buy_sell_ratio") or (buy / sell if sell > 0 else 0.0))
        total = buy + sell
        imbalance = sell / total if total > 0 else 0.5
        items.append((_row_time(row), buy, sell, ratio, imbalance))
    output: list[TakerFeature] = []
    for index, (timestamp, buy, sell, ratio, imbalance) in enumerate(items):
        imbalance_15 = None
        if index >= 2:
            window = items[index - 2:index + 1]
            buy_sum = sum(item[1] for item in window)
            sell_sum = sum(item[2] for item in window)
            total = buy_sum + sell_sum
            imbalance_15 = sell_sum / total if total > 0 else 0.5
        output.append(TakerFeature(timestamp, timestamp + timedelta(minutes=5), buy, sell, ratio, imbalance, imbalance_15))
    return output


def _feature_at(series: dict[str, Any], timestamp: datetime, btc_ret_15m: float | None) -> OiFlushFeature | None:
    oi_rows: list[OiFeature] = series["oi"]
    taker_rows: list[TakerFeature] = series["taker"]
    oi_index = bisect.bisect_right(series["oi_available_times"], timestamp) - 1
    taker_index = bisect.bisect_right(series["taker_available_times"], timestamp) - 1
    if oi_index < 0 or taker_index < 0 or btc_ret_15m is None:
        return None
    oi = oi_rows[oi_index]
    taker = taker_rows[taker_index]
    if oi.oi_chg_15m is None or oi.oi_chg_30m is None or taker.sell_imbalance_15m is None:
        return None
    funding_rate = _latest_funding_rate(series["funding"], series["funding_times"], timestamp)
    return OiFlushFeature(
        oi_chg_15m=oi.oi_chg_15m,
        oi_chg_30m=oi.oi_chg_30m,
        sell_imbalance_5m=taker.sell_imbalance_5m,
        sell_imbalance_15m=taker.sell_imbalance_15m,
        funding_rate=funding_rate,
        btc_ret_15m=btc_ret_15m,
        btc_state=_btc_state(btc_ret_15m),
        funding_bucket=_funding_bucket(funding_rate),
        oi_drop_bucket=_oi_drop_bucket(oi.oi_chg_15m),
        sell_imbalance_bucket=_sell_imbalance_bucket(taker.sell_imbalance_15m),
    )


def _latest_funding_rate(rows: list[FundingFeature], timestamps: list[datetime], timestamp: datetime) -> float:
    if not rows:
        return 0.0
    index = bisect.bisect_right(timestamps, timestamp) - 1
    if index < 0:
        return 0.0
    return rows[index].funding_rate


def _btc_ret_by_time(candles: list[Candle]) -> dict[datetime, float]:
    output = {}
    for index, candle in enumerate(candles):
        if index < 15 or candles[index - 15].close <= 0:
            continue
        output[candle.timestamp] = candle.close / candles[index - 15].close - 1.0
    return output


def _summary(
    initial_equity: float,
    final_equity: float,
    equity_curve: list[float],
    trades: list[dict[str, Any]],
    stats: OiFlushStats,
    feature_data_dir: str,
    funding_data_dir: str,
    symbols: Iterable[str],
    cost_experiment: str,
    include_trades: bool,
) -> dict[str, Any]:
    net_pnl = final_equity - initial_equity
    output = {
        "strategy": "oi_flush_reversal_long",
        "cost_experiment": cost_experiment,
        "feature_data_dir": feature_data_dir,
        "funding_data_dir": funding_data_dir,
        "symbols": list(symbols),
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "net_pnl": net_pnl,
        "net_return_pct": net_pnl / initial_equity * 100.0 if initial_equity > 0 else 0.0,
        "max_drawdown_pct": _max_drawdown_pct(equity_curve),
        "trade_count": len(trades),
        "total_trades": len(trades),
        **_trade_metrics(trades),
        "pending_count": stats.pending_count,
        "pending_expired_count": stats.pending_expired_count,
        "entry_count": stats.entry_count,
        "fail_fast_count": stats.fail_fast_count,
        "time_stop_count": stats.time_stop_count,
        "low_broken_count": stats.low_broken_count,
        "skipped_missing_features": stats.skipped_missing_features,
        "skipped_daily_limit": stats.skipped_daily_limit,
        "skipped_cooldown": stats.skipped_cooldown,
        "by_strategy": [_group_metrics("oi_flush_reversal_long", trades, "strategy")],
        "by_symbol": _group_by(trades, "symbol"),
        "by_month": _group_by_month(trades),
        "by_btc_state": _group_by(trades, "btc_state"),
        "by_funding_bucket": _group_by(trades, "funding_bucket"),
        "by_oi_drop_bucket": _group_by(trades, "oi_drop_bucket"),
        "by_sell_imbalance_bucket": _group_by(trades, "sell_imbalance_bucket"),
    }
    if include_trades:
        output["trades"] = trades
    return output


def _empty_summary(initial_equity: float, reason: str) -> dict[str, Any]:
    stats = OiFlushStats()
    output = _summary(initial_equity, initial_equity, [initial_equity], [], stats, "", "", [], "full_cost", False)
    output["skip_reason"] = reason
    return output


def _trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    gross = sum(float(t.get("gross_pnl", 0.0)) for t in trades)
    fee = sum(float(t.get("fee", 0.0)) for t in trades)
    slippage = sum(float(t.get("slippage_cost", t.get("slippage", 0.0))) for t in trades)
    funding = sum(float(t.get("funding", 0.0)) for t in trades)
    net = sum(float(t.get("net_pnl", 0.0)) for t in trades)
    wins = [float(t["net_pnl"]) for t in trades if float(t.get("net_pnl", 0.0)) > 0]
    losses = [float(t["net_pnl"]) for t in trades if float(t.get("net_pnl", 0.0)) <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "gross_pnl": gross,
        "fee": fee,
        "slippage": slippage,
        "slippage_cost": slippage,
        "funding": funding,
        "net_pnl": net,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": len(wins) / len(trades) * 100.0 if trades else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else None,
        "expectancy": net / len(trades) if trades else 0.0,
        "expectancy_per_trade": net / len(trades) if trades else 0.0,
        "avg_mfe": statistics.mean(float(t.get("mfe", 0.0)) for t in trades) if trades else 0.0,
        "avg_mae": statistics.mean(float(t.get("mae", 0.0)) for t in trades) if trades else 0.0,
    }


def _group_by(trades: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups = sorted({str(trade.get(field, "")) for trade in trades})
    return [_group_metrics(group, [trade for trade in trades if str(trade.get(field, "")) == group], field) for group in groups]


def _group_by_month(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({str(trade.get("exit_time", ""))[:7] for trade in trades})
    return [_group_metrics(month, [trade for trade in trades if str(trade.get("exit_time", ""))[:7] == month], "month") for month in groups]


def _group_metrics(name: str, trades: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return {"name": name, field: name, "trade_count": len(trades), **_trade_metrics(trades)}


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "initial_equity",
        "final_equity",
        "net_pnl",
        "net_return_pct",
        "max_drawdown_pct",
        "trade_count",
        "win_rate_pct",
        "profit_factor",
        "expectancy",
        "gross_pnl",
        "fee",
        "slippage",
        "funding",
        "pending_count",
        "pending_expired_count",
        "entry_count",
        "fail_fast_count",
        "time_stop_count",
        "low_broken_count",
    )
    return {key: result.get(key) for key in keys}


def _max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    max_dd = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd * 100.0


def _pct_change(current: float, previous: float) -> float | None:
    if previous <= 0:
        return None
    return current / previous - 1.0


def _row_time(row: dict[str, Any]) -> datetime:
    raw = row.get("timestamp") or row.get("time") or row.get("funding_time")
    if raw is None:
        raise ValueError("feature row missing timestamp")
    return parse_timestamp(str(raw))


def _btc_state(ret_15m: float) -> str:
    if ret_15m <= -0.008:
        return "btc_crash"
    if ret_15m < -0.003:
        return "btc_weak"
    if ret_15m <= 0.003:
        return "btc_neutral"
    return "btc_strong"


def _funding_bucket(rate: float) -> str:
    if rate <= 0:
        return "funding_lte_0"
    if rate <= 0.0002:
        return "funding_0_to_2bps"
    return "funding_gt_2bps"


def _oi_drop_bucket(value: float) -> str:
    if value <= -0.06:
        return "oi_drop_lte_-6pct"
    if value <= -0.04:
        return "oi_drop_-4_to_-6pct"
    if value <= -0.025:
        return "oi_drop_-2p5_to_-4pct"
    return "oi_drop_gt_-2p5pct"


def _sell_imbalance_bucket(value: float) -> str:
    if value >= 0.75:
        return "sell_imbalance_gte_075"
    if value >= 0.68:
        return "sell_imbalance_068_075"
    if value >= 0.62:
        return "sell_imbalance_062_068"
    return "sell_imbalance_lt_062"


def download_oi_flush_feature_data(
    symbols: Iterable[str],
    feature_data_dir: str,
    funding_data_dir: str,
    start: datetime,
    end: datetime,
    sleep_seconds: float = 0.08,
    overwrite: bool = False,
) -> None:
    feature_root = Path(feature_data_dir)
    funding_root = Path(funding_data_dir)
    feature_root.mkdir(parents=True, exist_ok=True)
    funding_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for symbol in symbols:
        start_tag = start.strftime("%Y%m%d")
        end_tag = end.strftime("%Y%m%d")
        oi_path = feature_root / f"{symbol}_oi_5m_{start_tag}_{end_tag}.csv"
        taker_path = feature_root / f"{symbol}_taker_5m_{start_tag}_{end_tag}.csv"
        funding_path = funding_root / f"{symbol}_funding_{start_tag}_{end_tag}.csv"
        try:
            if overwrite or not oi_path.exists():
                rows = _download_futures_data("openInterestHist", symbol, "5m", start, end, sleep_seconds=sleep_seconds)
                _write_rows(oi_path, rows, ("timestamp", "sumOpenInterest", "sumOpenInterestValue"))
            if overwrite or not taker_path.exists():
                rows = _download_futures_data("takerlongshortRatio", symbol, "5m", start, end, sleep_seconds=sleep_seconds)
                _write_rows(taker_path, rows, ("timestamp", "buySellRatio", "buyVol", "sellVol"))
            if overwrite or not funding_path.exists():
                rows = _download_funding(symbol, start, end, sleep_seconds=sleep_seconds)
                _write_rows(funding_path, rows, ("timestamp", "funding_rate", "mark_price"))
        except Exception as exc:
            failures.append(f"{symbol}: {exc}")
            continue
    if failures:
        (feature_root / "download_failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")


def _download_futures_data(
    endpoint: str,
    symbol: str,
    period: str,
    start: datetime,
    end: datetime,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    rows: list[dict[str, Any]] = []
    seen = set()
    cursor_end_ms = end_ms
    step_ms = 5 * 60_000
    while cursor_end_ms > start_ms:
        params = {"symbol": symbol, "period": period, "startTime": start_ms, "endTime": cursor_end_ms, "limit": 500}
        request = Request(f"{BINANCE_FUTURES_DATA_URL}/{endpoint}?{urlencode(params)}", headers={"User-Agent": "crypto-scalper/0.1"})
        with urlopen(request, timeout=20) as response:
            batch = json.loads(response.read().decode("utf-8"))
        if not batch:
            break
        min_ts = cursor_end_ms
        new_rows = 0
        for row in batch:
            timestamp_ms = int(row.get("timestamp", 0))
            if timestamp_ms < start_ms or timestamp_ms > end_ms:
                continue
            if timestamp_ms in seen:
                continue
            seen.add(timestamp_ms)
            min_ts = min(min_ts, timestamp_ms)
            row = dict(row)
            row["timestamp"] = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None).isoformat()
            rows.append(row)
            new_rows += 1
        if new_rows <= 0 or min_ts <= start_ms:
            break
        cursor_end_ms = min_ts - step_ms
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def _download_funding(symbol: str, start: datetime, end: datetime, sleep_seconds: float) -> list[dict[str, Any]]:
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    rows: list[dict[str, Any]] = []
    seen = set()
    while start_ms < end_ms:
        params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
        request = Request(f"{BINANCE_FUNDING_URL}?{urlencode(params)}", headers={"User-Agent": "crypto-scalper/0.1"})
        with urlopen(request, timeout=20) as response:
            batch = json.loads(response.read().decode("utf-8"))
        if not batch:
            break
        last_ts = start_ms
        for row in batch:
            timestamp_ms = int(row.get("fundingTime", 0))
            if timestamp_ms in seen:
                continue
            seen.add(timestamp_ms)
            last_ts = max(last_ts, timestamp_ms)
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None).isoformat(),
                    "funding_rate": row.get("fundingRate", "0"),
                    "mark_price": row.get("markPrice", "0"),
                }
            )
        next_ms = last_ts + 1
        if next_ms <= start_ms:
            break
        start_ms = next_ms
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _to_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return int(value.timestamp() * 1000)


def main() -> int:
    parser = argparse.ArgumentParser(description="OI flush reversal long-only backtest")
    parser.add_argument("--config", default="config.live.optimized_super_volume.json")
    parser.add_argument("--price-data-dir", default="data/binance_1m_3m_top100")
    parser.add_argument("--feature-data-dir", default="data/binance_oi_taker_5m")
    parser.add_argument("--funding-data-dir", default="data/binance_30m_365d")
    parser.add_argument("--initial-equity", type=float, default=None)
    parser.add_argument("--include-trades", action="store_true")
    parser.add_argument("--cost-experiment", default="full_cost", choices=("no_cost", "full_cost"))
    parser.add_argument("--run-experiments", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.08)
    args = parser.parse_args()
    if args.run_experiments:
        result = run_oi_flush_experiments(
            args.config,
            args.price_data_dir,
            args.feature_data_dir,
            args.funding_data_dir,
            initial_equity=args.initial_equity,
            include_trades=args.include_trades,
            download_missing=args.download_missing,
            sleep_seconds=args.sleep_seconds,
        )
    else:
        result = run_oi_flush_backtest(
            args.config,
            args.price_data_dir,
            args.feature_data_dir,
            args.funding_data_dir,
            initial_equity=args.initial_equity,
            include_trades=args.include_trades,
            cost_experiment=args.cost_experiment,
            download_missing=args.download_missing,
            sleep_seconds=args.sleep_seconds,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

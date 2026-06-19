from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .binance_client import SymbolRules
from .data import parse_timestamp
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

STRATEGY_NAME = "oi_flush_reversal_long"
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
    sell_imbalance_5m: float
    sell_imbalance_15m: float | None


@dataclass(frozen=True)
class FundingFeature:
    timestamp: datetime
    funding_rate: float


@dataclass(frozen=True)
class OiFlushFeature:
    oi_feature_time: datetime
    taker_feature_time: datetime
    funding_time: datetime | None
    feature_age_seconds: float
    oi_chg_15m: float
    oi_chg_30m: float
    sell_imbalance_5m: float
    sell_imbalance_15m: float
    funding_rate: float
    btc_ret_15m: float
    btc_ret_1h: float
    btc_state: str
    funding_bucket: str
    oi_drop_bucket: str
    sell_imbalance_bucket: str


@dataclass
class PendingFlush:
    symbol: str
    flush_time: datetime
    flush_index: int
    flush_low: float
    flush_close: float
    atr_pct: float
    atr_value: float
    oi_chg_15m: float
    oi_chg_30m: float
    sell_imbalance_15m: float
    latest_sell_imbalance_5m: float
    funding_rate: float
    btc_ret_15m: float
    btc_ret_1h: float
    btc_state: str
    oi_feature_time: datetime
    taker_feature_time: datetime
    funding_time: datetime | None
    feature_age_seconds: float
    expire_index: int
    reason: str = "oi_flush_pending"


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
    breakeven_active: bool = False


@dataclass
class OiFlushStats:
    pending_count: int = 0
    pending_expired_count: int = 0
    entry_count: int = 0
    rejected_stop_too_wide_count: int = 0
    rejected_stop_invalid_count: int = 0
    rejected_btc_count: int = 0
    rejected_funding_count: int = 0
    rejected_stale_feature_count: int = 0
    rejected_missing_features_count: int = 0
    fail_fast_count: int = 0
    low_broken_count: int = 0
    time_stop_count: int = 0
    breakeven_triggered_count: int = 0
    runner_activated_count: int = 0
    skipped_daily_limit: int = 0
    skipped_cooldown: int = 0
    skipped_position_limit: int = 0
    consecutive_loss_pause_count: int = 0


@dataclass(frozen=True)
class Experiment:
    name: str
    overrides: dict[str, Any]


EXPERIMENTS = (
    Experiment("A_baseline", {"oi_flush_oi_drop_15m": -0.035, "oi_flush_sell_imbalance_15m": 0.65, "oi_flush_sell_exhaustion_5m": 0.60, "oi_flush_take_profit_r": 2.0}),
    Experiment("B_stronger_sell_pressure", {"oi_flush_oi_drop_15m": -0.035, "oi_flush_sell_imbalance_15m": 0.68, "oi_flush_sell_exhaustion_5m": 0.60, "oi_flush_take_profit_r": 2.0}),
    Experiment("C_stronger_oi_flush", {"oi_flush_oi_drop_15m": -0.050, "oi_flush_sell_imbalance_15m": 0.65, "oi_flush_sell_exhaustion_5m": 0.60, "oi_flush_take_profit_r": 2.0}),
    Experiment("D_funding_lte_0", {"oi_flush_oi_drop_15m": -0.035, "oi_flush_sell_imbalance_15m": 0.65, "oi_flush_sell_exhaustion_5m": 0.60, "oi_flush_max_funding_rate": 0.0, "oi_flush_take_profit_r": 2.0}),
    Experiment("E_shorter_delay", {"oi_flush_oi_drop_15m": -0.035, "oi_flush_sell_imbalance_15m": 0.65, "oi_flush_sell_exhaustion_5m": 0.60, "oi_flush_min_entry_delay_bars": 3, "oi_flush_take_profit_r": 2.0}),
    Experiment("F_longer_delay", {"oi_flush_oi_drop_15m": -0.035, "oi_flush_sell_imbalance_15m": 0.65, "oi_flush_sell_exhaustion_5m": 0.60, "oi_flush_min_entry_delay_bars": 7, "oi_flush_take_profit_r": 2.0}),
    Experiment("G_stricter_exhaustion", {"oi_flush_oi_drop_15m": -0.035, "oi_flush_sell_imbalance_15m": 0.65, "oi_flush_sell_exhaustion_5m": 0.55, "oi_flush_take_profit_r": 2.0}),
    Experiment("H_tp_1_5r", {"oi_flush_oi_drop_15m": -0.035, "oi_flush_sell_imbalance_15m": 0.65, "oi_flush_sell_exhaustion_5m": 0.60, "oi_flush_take_profit_r": 1.5}),
    Experiment("I_tp_2_5r", {"oi_flush_oi_drop_15m": -0.035, "oi_flush_sell_imbalance_15m": 0.65, "oi_flush_sell_exhaustion_5m": 0.60, "oi_flush_take_profit_r": 2.5}),
    Experiment("J_conservative_live", {"oi_flush_oi_drop_15m": -0.050, "oi_flush_sell_imbalance_15m": 0.68, "oi_flush_sell_exhaustion_5m": 0.55, "oi_flush_btc_min_ret_15m": -0.006, "oi_flush_max_stop_pct": 0.010, "oi_flush_max_daily_trades": 1, "oi_flush_symbol_cooldown_hours": 12, "oi_flush_take_profit_r": 2.0}),
)


def run_oi_flush_experiments(
    config_path: str,
    price_data_dir: str,
    feature_data_dir: str,
    funding_data_dir: str,
    initial_equity: float | None = None,
    include_trades: bool = False,
    download_missing: bool = False,
    sleep_seconds: float = 0.08,
    trade_start: str | None = None,
    trade_end: str | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    results = []
    best: dict[str, Any] | None = None
    selected = [experiment for experiment in EXPERIMENTS if experiment_name is None or experiment.name == experiment_name]
    if experiment_name is not None and not selected:
        raise ValueError(f"unknown experiment: {experiment_name}")
    for experiment in selected:
        result = run_oi_flush_backtest(
            config_path,
            price_data_dir,
            feature_data_dir,
            funding_data_dir,
            initial_equity=initial_equity,
            include_trades=include_trades,
            cost_experiment="full_cost",
            strategy_overrides=experiment.overrides,
            download_missing=download_missing if experiment.name == "A_baseline" else False,
            sleep_seconds=sleep_seconds,
            trade_start=trade_start,
            trade_end=trade_end,
        )
        compact = {"name": experiment.name, "overrides": experiment.overrides, **_compact_result(result)}
        results.append(compact)
        pf = compact.get("profit_factor") or 0.0
        if best is None or (pf, compact.get("net_pnl", 0.0)) > ((best.get("profit_factor") or 0.0), best.get("net_pnl", 0.0)):
            best = compact
    return {"strategy": STRATEGY_NAME, "cost_mode": "full_cost", "experiments": results, "best": best}


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
    trade_start: str | None = None,
    trade_end: str | None = None,
) -> dict[str, Any]:
    if cost_experiment != "full_cost":
        raise ValueError("oi_flush_reversal_long diagnostics are full-cost only")
    config = load_live_config(config_path)
    if strategy_overrides:
        config = replace(config, strategy=replace(config.strategy, **strategy_overrides))
    config = _single_strategy_config(config, top30_only=top30_only)

    candles_by_symbol = _load_symbol_data(price_data_dir, tuple(config.trading.symbols), "1m")
    if not candles_by_symbol:
        raise RuntimeError(f"no 1m price data loaded from {price_data_dir}")
    start_filter = parse_timestamp(trade_start) if trade_start else None
    end_filter = parse_timestamp(trade_end) if trade_end else None
    if start_filter or end_filter:
        filtered: dict[str, list[Candle]] = {}
        for symbol, candles in candles_by_symbol.items():
            selected = [
                candle
                for candle in candles
                if (start_filter is None or candle.timestamp >= start_filter)
                and (end_filter is None or candle.timestamp <= end_filter)
            ]
            if selected:
                filtered[symbol] = selected
        candles_by_symbol = filtered
    symbols = tuple(symbol for symbol in config.trading.symbols if symbol in candles_by_symbol)
    if top30_only:
        symbols = symbols[:30]
    candles_by_symbol = {symbol: candles_by_symbol[symbol] for symbol in symbols}
    btc_candles = candles_by_symbol.get("BTCUSDT")
    if not btc_candles:
        raise RuntimeError("BTCUSDT 1m data is required for oi_flush_reversal_long")

    start = min(candles[0].timestamp for candles in candles_by_symbol.values() if candles)
    end = max(candles[-1].timestamp for candles in candles_by_symbol.values() if candles)
    if download_missing:
        download_oi_flush_feature_data(symbols, feature_data_dir, funding_data_dir, start, end + timedelta(minutes=1), sleep_seconds=sleep_seconds)

    feature_series = {symbol: build_oi_flush_feature_series(symbol, feature_data_dir, funding_data_dir) for symbol in symbols}
    usable_symbols = tuple(symbol for symbol in symbols if feature_series[symbol]["oi"] and feature_series[symbol]["taker"])
    candles_by_symbol = {symbol: candles_by_symbol[symbol] for symbol in usable_symbols}
    if not candles_by_symbol:
        return _empty_summary(initial_equity or config.risk.starting_capital_usdt, "missing_oi_or_taker_features")

    common_length = min(len(candles) for candles in candles_by_symbol.values())
    btc_ret_15m, btc_ret_1h = _btc_ret_by_time(btc_candles)
    indicator_cache = {symbol: _price_indicator_cache(candles) for symbol, candles in candles_by_symbol.items()}
    execution_config = execution_config_from_live_config(config, cost_experiment="full_cost", mode="conservative")

    starting_equity = initial_equity if initial_equity is not None else config.risk.starting_capital_usdt
    cash = starting_equity
    equity_curve = [starting_equity]
    position: OiFlushPosition | None = None
    flush_pending: dict[str, PendingFlush] = {}
    entry_pending: list[PendingEntry] = []
    trades: list[dict[str, Any]] = []
    stats = OiFlushStats()
    daily_entries: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}
    cooldown_until: dict[str, datetime] = {}
    loss_streak = 0
    paused_day: str | None = None
    rules = {symbol: _inferred_symbol_rules(symbol, candles_by_symbol[symbol]) for symbol in candles_by_symbol}
    timestamps = [candle.timestamp for candle in next(iter(candles_by_symbol.values()))[:common_length]]

    for index, timestamp in enumerate(timestamps):
        day_key = timestamp.date().isoformat()
        if paused_day and day_key != paused_day:
            paused_day = None
            loss_streak = 0

        if position is not None:
            candle = candles_by_symbol[position.symbol][index]
            cash, closed = _manage_oi_flush_position(config, execution_config, cash, position, candle, index, feature_series[position.symbol]["funding"], rules[position.symbol], stats)
            if closed:
                trades.append(closed)
                daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + closed["net_pnl"]
                cooldown_until[position.symbol] = timestamp + timedelta(hours=int(config.strategy.oi_flush_symbol_cooldown_hours))
                if closed["net_pnl"] <= 0:
                    loss_streak += 1
                    if loss_streak >= 2:
                        paused_day = day_key
                        stats.consecutive_loss_pause_count += 1
                else:
                    loss_streak = 0
                position = None

        if position is None and entry_pending:
            remaining_entries: list[PendingEntry] = []
            for pending in entry_pending:
                if pending.earliest_execution_index > index:
                    remaining_entries.append(pending)
                    continue
                if position is not None:
                    remaining_entries.append(pending)
                    continue
                if daily_entries.get(day_key, 0) >= int(config.strategy.oi_flush_max_daily_trades):
                    stats.skipped_daily_limit += 1
                    continue
                candle = candles_by_symbol[pending.symbol][index]
                opened = _open_oi_flush_position(config, execution_config, cash, pending, candle, index, rules[pending.symbol], stats)
                if opened is None:
                    continue
                position, entry_cash_delta = opened
                cash += entry_cash_delta
                stats.entry_count += 1
                daily_entries[day_key] = daily_entries.get(day_key, 0) + 1
            entry_pending = remaining_entries

        mark_equity = cash
        if position is not None:
            mark = candles_by_symbol[position.symbol][index].close
            mark_equity += position.quantity * (mark - position.entry_price)
        equity_curve.append(mark_equity)

        if position is not None or entry_pending:
            continue
        if mark_equity <= 0:
            continue
        if paused_day == day_key:
            continue
        if daily_entries.get(day_key, 0) >= int(config.strategy.oi_flush_max_daily_trades):
            stats.skipped_daily_limit += 1
            continue
        if daily_pnl.get(day_key, 0.0) <= -starting_equity * float(config.risk.max_daily_loss_pct):
            continue

        detect_this_bar = timestamp.minute % 5 == 0
        for symbol in usable_symbols:
            if symbol == "BTCUSDT":
                continue
            has_pending = symbol in flush_pending
            if not has_pending and not detect_this_bar:
                continue
            if cooldown_until.get(symbol) and timestamp < cooldown_until[symbol]:
                stats.skipped_cooldown += 1
                continue
            candles = candles_by_symbol[symbol]
            if index >= len(candles) or index < 61:
                continue
            candle = candles[index]
            indicators = indicator_cache[symbol]
            atr_value = max(indicators["atr"][index], 1e-12)
            atr_pct = atr_value / max(candle.close, 1e-12)
            ret_15m = candle.close / max(candles[index - 15].close, 1e-12) - 1.0
            price_threshold = max(float(config.strategy.oi_flush_price_drop_min_pct), float(config.strategy.oi_flush_price_drop_atr_mult) * atr_pct)
            if ret_15m > -price_threshold:
                continue

            feature = _feature_at(feature_series[symbol], timestamp, btc_ret_15m.get(timestamp), btc_ret_1h.get(timestamp), float(getattr(config.strategy, "oi_flush_feature_max_age_seconds", 600.0)))
            if feature is None:
                stats.rejected_missing_features_count += 1
                continue

            pending_flush = flush_pending.get(symbol)
            if pending_flush:
                expired, expire_reason = _pending_expired(config, pending_flush, candle, feature, index)
                if expired:
                    stats.pending_expired_count += 1
                    flush_pending.pop(symbol, None)
                    pending_flush = None
                else:
                    entry_signal = _confirm_oi_flush_entry(config, pending_flush, candle, indicators, feature, index, stats)
                    if entry_signal:
                        entry_pending.append(PendingEntry(symbol, entry_signal, pending_flush, timestamp, timestamp + timedelta(minutes=1), index + 1))
                        flush_pending.pop(symbol, None)
                        break
            if symbol in flush_pending:
                continue
            if not detect_this_bar:
                continue
            new_flush = _detect_oi_flush(config, symbol, candles, indicators, feature, index)
            if new_flush:
                flush_pending[symbol] = new_flush
                stats.pending_count += 1

    if position is not None:
        last_candle = candles_by_symbol[position.symbol][common_length - 1]
        cash, closed = _close_oi_flush_position(execution_config, cash, position, last_candle.close, "end_of_data", last_candle.timestamp, feature_series[position.symbol]["funding"], rules[position.symbol])
        trades.append(closed)

    final_equity = cash
    equity_curve.append(final_equity)
    return _summary(starting_equity, final_equity, equity_curve, trades, stats, feature_data_dir, funding_data_dir, usable_symbols, include_trades)


def _single_strategy_config(config: Any, top30_only: bool = False) -> Any:
    strategy = replace(
        config.strategy,
        allow_short=False,
        rsi_reversal_enabled=False,
        ordinary_breakout_enabled=False,
        pullback_reclaim_enabled=False,
        fast_breakout_enabled=False,
        startup_breakout_enabled=False,
        super_volume_breakout_enabled=False,
        oi_flush_reversal_enabled=True,
    )
    symbols = tuple(config.trading.symbols[:30] if top30_only else config.trading.symbols)
    trading = replace(config.trading, symbols=symbols, entry_symbols=symbols, max_open_positions=1, max_new_entries_per_cycle=1, super_volume_extra_slot_enabled=False, max_scale_ins_per_symbol=0, scale_in_entry_fraction=0.0)
    filters = replace(config.filters, enabled=False, extreme_reversal_entry_enabled=False, pre_cross_entry_enabled=False)
    return replace(config, strategy=strategy, trading=trading, filters=filters)


def _price_indicator_cache(candles: list[Candle]) -> dict[str, list[float]]:
    closes = [c.close for c in candles]
    return {"atr": atr(candles, 14), "ema_9": ema(closes, 9)}


def _detect_oi_flush(config: Any, symbol: str, candles: list[Candle], indicators: dict[str, list[float]], feature: OiFlushFeature, index: int) -> PendingFlush | None:
    s = config.strategy
    candle = candles[index]
    ret_15m = candle.close / max(candles[index - 15].close, 1e-12) - 1.0
    atr_value = max(indicators["atr"][index], 1e-12)
    atr_pct = atr_value / max(candle.close, 1e-12)
    threshold = max(float(s.oi_flush_price_drop_min_pct), float(s.oi_flush_price_drop_atr_mult) * atr_pct)
    if ret_15m > -threshold:
        return None
    if feature.oi_chg_15m > float(s.oi_flush_oi_drop_15m):
        return None
    if feature.sell_imbalance_15m < float(s.oi_flush_sell_imbalance_15m):
        return None
    if feature.btc_ret_15m <= float(s.oi_flush_btc_min_ret_15m):
        return None
    if feature.funding_rate > float(s.oi_flush_max_funding_rate):
        return None
    return PendingFlush(
        symbol=symbol,
        flush_time=candle.timestamp,
        flush_index=index,
        flush_low=candle.low,
        flush_close=candle.close,
        atr_pct=atr_pct,
        atr_value=atr_value,
        oi_chg_15m=feature.oi_chg_15m,
        oi_chg_30m=feature.oi_chg_30m,
        sell_imbalance_15m=feature.sell_imbalance_15m,
        latest_sell_imbalance_5m=feature.sell_imbalance_5m,
        funding_rate=feature.funding_rate,
        btc_ret_15m=feature.btc_ret_15m,
        btc_ret_1h=feature.btc_ret_1h,
        btc_state=feature.btc_state,
        oi_feature_time=feature.oi_feature_time,
        taker_feature_time=feature.taker_feature_time,
        funding_time=feature.funding_time,
        feature_age_seconds=feature.feature_age_seconds,
        expire_index=index + int(s.oi_flush_pending_bars),
    )


def _pending_expired(config: Any, pending: PendingFlush, candle: Candle, feature: OiFlushFeature, index: int) -> tuple[bool, str]:
    s = config.strategy
    if index > pending.expire_index:
        return True, "expired_time"
    if candle.low < pending.flush_low * 0.997:
        return True, "expired_low_broken"
    if feature.btc_ret_15m <= float(s.oi_flush_btc_min_ret_15m):
        return True, "expired_btc"
    if feature.funding_rate > float(s.oi_flush_max_funding_rate):
        return True, "expired_funding"
    if feature.oi_chg_15m > 0.005:
        return True, "expired_oi_rebuild"
    return False, ""


def _confirm_oi_flush_entry(config: Any, pending: PendingFlush, candle: Candle, indicators: dict[str, list[float]], feature: OiFlushFeature, index: int, stats: OiFlushStats) -> Signal | None:
    s = config.strategy
    if index - pending.flush_index < int(getattr(s, "oi_flush_min_entry_delay_bars", 5)):
        return None
    if candle.low < pending.flush_low * 0.997:
        return None
    if feature.sell_imbalance_5m > float(s.oi_flush_sell_exhaustion_5m):
        return None
    if feature.btc_ret_15m <= float(s.oi_flush_btc_min_ret_15m):
        stats.rejected_btc_count += 1
        return None
    if feature.funding_rate > float(s.oi_flush_max_funding_rate):
        stats.rejected_funding_count += 1
        return None
    ema_values = indicators.get("ema_9")
    if not ema_values or candle.close <= ema_values[index]:
        return None
    candle_range = max(candle.high - candle.low, 1e-12)
    if (candle.close - candle.low) / candle_range < float(s.oi_flush_min_close_position):
        return None
    stop_price = min(pending.flush_low * 0.997, candle.close - pending.atr_value * float(s.oi_flush_stop_atr_mult))
    stop_loss_pct = (candle.close - stop_price) / max(candle.close, 1e-12)
    if stop_loss_pct <= 0:
        stats.rejected_stop_invalid_count += 1
        return None
    if stop_loss_pct > float(s.oi_flush_max_stop_pct):
        stats.rejected_stop_too_wide_count += 1
        return None
    return Signal(Direction.LONG, 0.62, STRATEGY_NAME, stop_loss_pct, stop_loss_pct * float(s.oi_flush_take_profit_r), risk_multiplier=float(s.oi_flush_risk_multiplier), max_holding_bars=int(s.oi_flush_time_stop_minutes))


def _open_oi_flush_position(config: Any, execution_config: BacktestExecutionConfig, cash: float, pending: PendingEntry, candle: Candle, index: int, rules: SymbolRules, stats: OiFlushStats) -> tuple[OiFlushPosition, float] | None:
    entry = candle.open
    stop_price_raw = min(pending.flush.flush_low * 0.997, entry - pending.flush.atr_value * float(config.strategy.oi_flush_stop_atr_mult))
    stop_loss_pct = (entry - stop_price_raw) / max(entry, 1e-12)
    if stop_loss_pct <= 0:
        stats.rejected_stop_invalid_count += 1
        return None
    if stop_loss_pct > float(config.strategy.oi_flush_max_stop_pct):
        stats.rejected_stop_too_wide_count += 1
        return None
    signal = replace(pending.signal, stop_loss_pct=stop_loss_pct, take_profit_pct=stop_loss_pct * float(config.strategy.oi_flush_take_profit_r))
    notional, reason = _size_notional(config, cash, entry, signal)
    if reason != "ok" or notional <= 0:
        return None
    qty = conservative_quantity(rules, notional / entry)
    if validate_order_size(rules, qty, entry, config.risk.min_order_notional_usdt) != "ok":
        return None
    fill = market_entry_fill(execution_config, rules, Direction.LONG, qty, entry)
    position = OiFlushPosition(
        symbol=pending.symbol,
        quantity=qty,
        entry_price=fill.price,
        raw_entry_price=entry,
        entry_time=candle.timestamp,
        entry_index=index,
        stop_price=fill.price * (1 - stop_loss_pct),
        take_profit_price=fill.price * (1 + signal.take_profit_pct),
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=signal.take_profit_pct,
        entry_fee=fill.fee,
        entry_slippage_cost=fill.slippage_cost,
        signal_time=pending.signal_time,
        signal_available_time=pending.signal_available_time,
        flush=pending.flush,
        notional=abs(qty * fill.price),
        best_price=fill.price,
        worst_price=fill.price,
    )
    return position, -fill.fee


def _manage_oi_flush_position(config: Any, execution_config: BacktestExecutionConfig, cash: float, position: OiFlushPosition, candle: Candle, index: int, funding_rates: list[FundingFeature], rules: SymbolRules, stats: OiFlushStats) -> tuple[float, dict[str, Any] | None]:
    position.best_price = max(position.best_price, candle.high)
    position.worst_price = min(position.worst_price, candle.low)
    position.mfe = max(position.mfe, (position.best_price - position.entry_price) * position.quantity)
    position.mae = min(position.mae, (position.worst_price - position.entry_price) * position.quantity)
    risk_cash = max(1e-12, position.notional * position.stop_loss_pct)
    if not position.breakeven_active and position.mfe >= risk_cash:
        position.stop_price = max(position.stop_price, position.entry_price * (1.0 + float(getattr(config.strategy, "oi_flush_breakeven_offset_pct", 0.001))))
        position.breakeven_active = True
        stats.breakeven_triggered_count += 1
    reason = ""
    raw_exit = 0.0
    if candle.low <= position.stop_price:
        reason = "oi_flush_breakeven_stop" if position.breakeven_active else "stop_loss"
        raw_exit = position.stop_price
    elif candle.high >= position.take_profit_price:
        reason = "take_profit"
        raw_exit = position.take_profit_price
    elif candle.low < position.flush.flush_low:
        reason = "oi_flush_low_broken"
        raw_exit = candle.close
        stats.low_broken_count += 1
    else:
        held = max(0, index - position.entry_index)
        if held >= int(config.strategy.oi_flush_fail_fast_minutes):
            if position.mfe < risk_cash * float(config.strategy.oi_flush_fail_fast_min_r):
                reason = "oi_flush_fail_fast"
                raw_exit = candle.close
                stats.fail_fast_count += 1
        if not reason and held >= int(config.strategy.oi_flush_time_stop_minutes):
            reason = "oi_flush_time_stop"
            raw_exit = candle.close
            stats.time_stop_count += 1
    if not reason:
        return cash, None
    return _close_oi_flush_position(execution_config, cash, position, raw_exit, reason, candle.timestamp, funding_rates, rules)


def _close_oi_flush_position(execution_config: BacktestExecutionConfig, cash: float, position: OiFlushPosition, raw_exit_price: float, reason: str, exit_time: datetime, funding_rates: list[FundingFeature], rules: SymbolRules) -> tuple[float, dict[str, Any]]:
    order_type = "stop_market" if reason in {"stop_loss", "oi_flush_breakeven_stop"} else "take_profit_market" if reason == "take_profit" else "market"
    fill = market_exit_fill(execution_config, rules, Direction.LONG, position.quantity, raw_exit_price, order_type)
    raw_gross = position.quantity * (raw_exit_price - position.raw_entry_price)
    execution_gross = position.quantity * (fill.price - position.entry_price)
    fee = position.entry_fee + fill.fee
    slippage = position.entry_slippage_cost + fill.slippage_cost
    funding_values = [item.funding_rate for item in funding_rates if position.entry_time < item.timestamp <= exit_time]
    funding = funding_cashflow(Direction.LONG, position.notional, funding_values)
    net = raw_gross - fee - slippage + funding
    cash += execution_gross - fill.fee + funding
    hold = max(0.0, (exit_time - position.entry_time).total_seconds() / 60.0)
    trade = {
        "symbol": position.symbol,
        "strategy": STRATEGY_NAME,
        "strategy_bucket": STRATEGY_NAME,
        "side": "LONG",
        "direction": "LONG",
        "entry_time": position.entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "entry_price": position.entry_price,
        "exit_price": fill.price,
        "raw_entry_price": position.raw_entry_price,
        "raw_exit_price": raw_exit_price,
        "qty": position.quantity,
        "quantity": position.quantity,
        "notional": position.notional,
        "gross_pnl": raw_gross,
        "execution_gross_pnl": execution_gross,
        "fee": fee,
        "fees": fee,
        "slippage": slippage,
        "slippage_cost": slippage,
        "funding": funding,
        "net_pnl": net,
        "mfe": position.mfe,
        "mae": position.mae,
        "mfe_r": position.mfe / max(1e-12, position.notional * position.stop_loss_pct),
        "mae_r": position.mae / max(1e-12, position.notional * position.stop_loss_pct),
        "hold_minutes": hold,
        "exit_reason": reason,
        "entry_order_type": "market",
        "exit_order_type": fill.order_type,
        "entry_liquidity": "taker",
        "exit_liquidity": fill.liquidity,
        "signal_time": position.signal_time.isoformat(),
        "signal_available_time": position.signal_available_time.isoformat(),
        "flush_time": position.flush.flush_time.isoformat(),
        "flush_low": position.flush.flush_low,
        "oi_feature_time": position.flush.oi_feature_time.isoformat(),
        "taker_feature_time": position.flush.taker_feature_time.isoformat(),
        "funding_time": position.flush.funding_time.isoformat() if position.flush.funding_time else "",
        "feature_age_seconds": position.flush.feature_age_seconds,
        "entry_delay_minutes": max(0.0, (position.entry_time - position.flush.flush_time).total_seconds() / 60.0),
        "funding_rate": position.flush.funding_rate,
        "oi_chg_15m": position.flush.oi_chg_15m,
        "oi_chg_30m": position.flush.oi_chg_30m,
        "sell_imbalance_15m": position.flush.sell_imbalance_15m,
        "latest_sell_imbalance_5m": position.flush.latest_sell_imbalance_5m,
        "btc_ret_15m": position.flush.btc_ret_15m,
        "btc_ret_1h": position.flush.btc_ret_1h,
        "btc_state": position.flush.btc_state,
        "funding_bucket": _funding_bucket(position.flush.funding_rate),
        "oi_drop_bucket": _oi_drop_bucket(position.flush.oi_chg_15m),
        "sell_imbalance_bucket": _sell_imbalance_bucket(position.flush.sell_imbalance_15m),
        "entry_delay_bucket": _entry_delay_bucket(max(0.0, (position.entry_time - position.flush.flush_time).total_seconds() / 60.0)),
        "stop_pct_bucket": _stop_pct_bucket(position.stop_loss_pct),
        "return_pct": net / max(position.notional, 1e-12) * 100.0,
    }
    return cash, trade


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
    notional = min(risk_notional, symbol_cap, account_cap, policy_cap) * max(0.0, min(1.0, config.trading.initial_entry_fraction))
    if notional < config.risk.min_order_notional_usdt:
        return 0.0, "below_min_notional"
    return notional, "ok"


def build_oi_flush_feature_series(symbol: str, feature_data_dir: str, funding_data_dir: str) -> dict[str, Any]:
    oi_rows = build_oi_features(load_oi_csv(feature_data_dir, symbol))
    taker_rows = build_taker_features(load_taker_csv(feature_data_dir, symbol))
    funding_rows = load_funding_csv(funding_data_dir, symbol)
    return {"oi": oi_rows, "oi_available_times": [x.available_time for x in oi_rows], "taker": taker_rows, "taker_available_times": [x.available_time for x in taker_rows], "funding": funding_rows, "funding_times": [x.timestamp for x in funding_rows]}


def load_oi_csv(data_dir: str, symbol: str) -> list[dict[str, Any]]:
    return _load_feature_csv(data_dir, symbol, ("*oi_5m*.csv", "*open_interest_5m*.csv", "*openInterest_5m*.csv"), ("sumOpenInterestValue", "sum_open_interest_value"))


def load_taker_csv(data_dir: str, symbol: str) -> list[dict[str, Any]]:
    return _load_feature_csv(data_dir, symbol, ("*taker_5m*.csv", "*taker_buy_sell_5m*.csv", "*takerlongshort_5m*.csv"), ("buyVol", "buy_vol"))


def _load_feature_csv(data_dir: str, symbol: str, patterns: tuple[str, ...], required_any: tuple[str, ...]) -> list[dict[str, Any]]:
    root = Path(data_dir)
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(f"{symbol}_{pattern}"))
    if not matches:
        return []
    with sorted(matches)[-1].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not any(name in fields for name in required_any):
            return []
        rows = [dict(row) for row in reader]
    rows.sort(key=_row_time)
    return rows


def load_funding_csv(data_dir: str, symbol: str) -> list[FundingFeature]:
    matches = sorted(Path(data_dir).glob(f"{symbol}_funding_*.csv"))
    if not matches:
        return []
    rows = []
    with matches[-1].open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(FundingFeature(parse_timestamp(row.get("funding_time") or row.get("timestamp") or row.get("time") or ""), float(row.get("funding_rate") or row.get("fundingRate") or 0.0)))
    rows.sort(key=lambda item: item.timestamp)
    return rows


def build_oi_features(rows: list[dict[str, Any]]) -> list[OiFeature]:
    values = [(_row_time(row), float(row.get("sumOpenInterestValue") or row.get("sum_open_interest_value") or 0.0)) for row in sorted(rows, key=_row_time)]
    out = []
    for i, (ts, value) in enumerate(values):
        out.append(OiFeature(ts, ts + timedelta(minutes=5), value, _pct_change(value, values[i - 3][1]) if i >= 3 else None, _pct_change(value, values[i - 6][1]) if i >= 6 else None))
    return out


def build_taker_features(rows: list[dict[str, Any]]) -> list[TakerFeature]:
    items = []
    for row in sorted(rows, key=_row_time):
        buy = float(row.get("buyVol") or row.get("buy_vol") or 0.0)
        sell = float(row.get("sellVol") or row.get("sell_vol") or 0.0)
        total = buy + sell
        items.append((_row_time(row), buy, sell, sell / total if total > 0 else 0.5))
    out = []
    for i, (ts, buy, sell, imbalance) in enumerate(items):
        im15 = None
        if i >= 2:
            window = items[i - 2:i + 1]
            buy_sum = sum(x[1] for x in window)
            sell_sum = sum(x[2] for x in window)
            im15 = sell_sum / (buy_sum + sell_sum) if buy_sum + sell_sum > 0 else 0.5
        out.append(TakerFeature(ts, ts + timedelta(minutes=5), buy, sell, imbalance, im15))
    return out


def _feature_at(series: dict[str, Any], timestamp: datetime, btc_ret_15m: float | None, btc_ret_1h: float | None, max_age_seconds: float) -> OiFlushFeature | None:
    oi_i = bisect.bisect_right(series["oi_available_times"], timestamp) - 1
    taker_i = bisect.bisect_right(series["taker_available_times"], timestamp) - 1
    if oi_i < 0 or taker_i < 0 or btc_ret_15m is None or btc_ret_1h is None:
        return None
    oi = series["oi"][oi_i]
    taker = series["taker"][taker_i]
    if oi.oi_chg_15m is None or oi.oi_chg_30m is None or taker.sell_imbalance_15m is None:
        return None
    age = max((timestamp - oi.available_time).total_seconds(), (timestamp - taker.available_time).total_seconds())
    if age > max_age_seconds:
        return None
    funding_time, funding_rate = _latest_funding(series["funding"], series["funding_times"], timestamp)
    return OiFlushFeature(oi.available_time, taker.available_time, funding_time, age, oi.oi_chg_15m, oi.oi_chg_30m, taker.sell_imbalance_5m, taker.sell_imbalance_15m, funding_rate, btc_ret_15m, btc_ret_1h, _btc_state(btc_ret_15m, btc_ret_1h), _funding_bucket(funding_rate), _oi_drop_bucket(oi.oi_chg_15m), _sell_imbalance_bucket(taker.sell_imbalance_15m))


def _latest_funding(rows: list[FundingFeature], times: list[datetime], timestamp: datetime) -> tuple[datetime | None, float]:
    if not rows:
        return None, 0.0
    i = bisect.bisect_right(times, timestamp) - 1
    if i < 0:
        return None, 0.0
    return rows[i].timestamp, rows[i].funding_rate


def _btc_ret_by_time(candles: list[Candle]) -> tuple[dict[datetime, float], dict[datetime, float]]:
    ret15, ret1h = {}, {}
    for i, c in enumerate(candles):
        if i >= 15 and candles[i - 15].close > 0:
            ret15[c.timestamp] = c.close / candles[i - 15].close - 1.0
        if i >= 60 and candles[i - 60].close > 0:
            ret1h[c.timestamp] = c.close / candles[i - 60].close - 1.0
    return ret15, ret1h


def _summary(initial: float, final: float, equity_curve: list[float], trades: list[dict[str, Any]], stats: OiFlushStats, feature_dir: str, funding_dir: str, symbols: Iterable[str], include_trades: bool) -> dict[str, Any]:
    net = final - initial
    out = {"strategy": STRATEGY_NAME, "cost_experiment": "full_cost", "feature_data_dir": feature_dir, "funding_data_dir": funding_dir, "symbols": list(symbols), "initial_equity": initial, "final_equity": final, "net_pnl": net, "net_return_pct": net / initial * 100.0 if initial > 0 else 0.0, "max_drawdown_pct": _max_drawdown_pct(equity_curve), "trade_count": len(trades), "total_trades": len(trades), **_trade_metrics(trades), **asdict(stats), "by_symbol": _group_by(trades, "symbol"), "by_month": _group_by_month(trades), "by_exit_reason": _group_by(trades, "exit_reason"), "by_sell_imbalance_bucket": _group_by(trades, "sell_imbalance_bucket"), "by_oi_drop_bucket": _group_by(trades, "oi_drop_bucket"), "by_funding_bucket": _group_by(trades, "funding_bucket"), "by_btc_state": _group_by(trades, "btc_state"), "by_entry_delay_bucket": _group_by(trades, "entry_delay_bucket"), "by_stop_pct_bucket": _group_by(trades, "stop_pct_bucket")}
    if include_trades:
        out["trades"] = trades
    return out


def _empty_summary(initial: float, reason: str) -> dict[str, Any]:
    out = _summary(initial, initial, [initial], [], OiFlushStats(), "", "", [], False)
    out["skip_reason"] = reason
    return out


def _trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    gross = sum(float(t.get("gross_pnl", 0.0)) for t in trades)
    fee = sum(float(t.get("fee", 0.0)) for t in trades)
    slippage = sum(float(t.get("slippage_cost", 0.0)) for t in trades)
    funding = sum(float(t.get("funding", 0.0)) for t in trades)
    net = sum(float(t.get("net_pnl", 0.0)) for t in trades)
    wins = [float(t["net_pnl"]) for t in trades if float(t.get("net_pnl", 0.0)) > 0]
    losses = [float(t["net_pnl"]) for t in trades if float(t.get("net_pnl", 0.0)) <= 0]
    gross_loss = abs(sum(losses))
    return {"gross_pnl": gross, "fee": fee, "slippage": slippage, "slippage_cost": slippage, "funding": funding, "wins": len(wins), "losses": len(losses), "win_rate_pct": len(wins) / len(trades) * 100.0 if trades else 0.0, "profit_factor": sum(wins) / gross_loss if gross_loss > 0 else None, "expectancy": net / len(trades) if trades else 0.0, "avg_mfe": statistics.mean(float(t.get("mfe", 0.0)) for t in trades) if trades else 0.0, "avg_mae": statistics.mean(float(t.get("mae", 0.0)) for t in trades) if trades else 0.0, "avg_hold_minutes": statistics.mean(float(t.get("hold_minutes", 0.0)) for t in trades) if trades else 0.0}


def _group_by(trades: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [_group_metrics(key, [t for t in trades if str(t.get(field, "")) == key], field) for key in sorted({str(t.get(field, "")) for t in trades})]


def _group_by_month(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_group_metrics(month, [t for t in trades if str(t.get("exit_time", ""))[:7] == month], "month") for month in sorted({str(t.get("exit_time", ""))[:7] for t in trades})]


def _group_metrics(name: str, trades: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return {field: name, "name": name, "trade_count": len(trades), **_trade_metrics(trades)}


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("initial_equity", "final_equity", "net_pnl", "net_return_pct", "max_drawdown_pct", "trade_count", "win_rate_pct", "profit_factor", "expectancy", "gross_pnl", "fee", "slippage", "funding", "pending_count", "pending_expired_count", "entry_count", "rejected_stop_too_wide_count", "rejected_btc_count", "rejected_funding_count", "rejected_stale_feature_count", "fail_fast_count", "low_broken_count", "time_stop_count", "breakeven_triggered_count", "runner_activated_count")
    return {k: result.get(k) for k in keys}


def _max_drawdown_pct(curve: list[float]) -> float:
    peak = curve[0] if curve else 0.0
    out = 0.0
    for equity in curve:
        peak = max(peak, equity)
        if peak > 0:
            out = max(out, (peak - equity) / peak)
    return out * 100.0


def _pct_change(cur: float, prev: float) -> float | None:
    return None if prev <= 0 else cur / prev - 1.0


def _row_time(row: dict[str, Any]) -> datetime:
    return parse_timestamp(str(row.get("timestamp") or row.get("time") or row.get("funding_time") or ""))


def _btc_state(ret15: float, ret1h: float) -> str:
    if ret15 <= -0.008 or ret1h <= -0.02:
        return "btc_crash"
    if ret15 < -0.003:
        return "btc_weak"
    if ret15 <= 0.003:
        return "btc_neutral"
    return "btc_strong"


def _funding_bucket(x: float) -> str:
    if x <= 0:
        return "funding_lte_0"
    if x <= 0.0002:
        return "funding_0_to_2bps"
    return "funding_gt_2bps"


def _oi_drop_bucket(x: float) -> str:
    if x <= -0.06:
        return "oi_drop_lte_-6pct"
    if x <= -0.04:
        return "oi_drop_-4_to_-6pct"
    if x <= -0.025:
        return "oi_drop_-2p5_to_-4pct"
    return "oi_drop_gt_-2p5pct"


def _sell_imbalance_bucket(x: float) -> str:
    if x >= 0.75:
        return "sell_imbalance_gte_075"
    if x >= 0.68:
        return "sell_imbalance_068_075"
    if x >= 0.62:
        return "sell_imbalance_062_068"
    return "sell_imbalance_lt_062"


def _entry_delay_bucket(x: float) -> str:
    if x <= 5:
        return "delay_lte_5m"
    if x <= 10:
        return "delay_5_10m"
    return "delay_gt_10m"


def _stop_pct_bucket(x: float) -> str:
    if x <= 0.010:
        return "stop_lte_1pct"
    if x <= 0.012:
        return "stop_1_1p2pct"
    if x <= 0.015:
        return "stop_1p2_1p5pct"
    return "stop_gt_1p5pct"


def _inferred_symbol_rules(symbol: str, candles: list[Candle]) -> SymbolRules:
    closes = [c.close for c in candles[: min(len(candles), 1440)] if c.close > 0]
    price = statistics.median(closes) if closes else 1.0
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


def download_oi_flush_feature_data(symbols: Iterable[str], feature_data_dir: str, funding_data_dir: str, start: datetime, end: datetime, sleep_seconds: float = 0.08, overwrite: bool = False) -> None:
    feature_root = Path(feature_data_dir); funding_root = Path(funding_data_dir)
    feature_root.mkdir(parents=True, exist_ok=True); funding_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for symbol in symbols:
        tag1, tag2 = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        oi_path = feature_root / f"{symbol}_oi_5m_{tag1}_{tag2}.csv"
        taker_path = feature_root / f"{symbol}_taker_5m_{tag1}_{tag2}.csv"
        funding_path = funding_root / f"{symbol}_funding_{tag1}_{tag2}.csv"
        try:
            if overwrite or not oi_path.exists():
                _write_rows(oi_path, _download_futures_data("openInterestHist", symbol, "5m", start, end, sleep_seconds), ("timestamp", "sumOpenInterest", "sumOpenInterestValue"))
            if overwrite or not taker_path.exists():
                _write_rows(taker_path, _download_futures_data("takerlongshortRatio", symbol, "5m", start, end, sleep_seconds), ("timestamp", "buySellRatio", "buyVol", "sellVol"))
            if overwrite or not funding_path.exists():
                _write_rows(funding_path, _download_funding(symbol, start, end, sleep_seconds), ("timestamp", "funding_rate", "mark_price"))
        except Exception as exc:
            failures.append(f"{symbol}: {exc}")
    if failures:
        (feature_root / "download_failures.txt").write_text("\n".join(failures) + "\n")


def _download_futures_data(endpoint: str, symbol: str, period: str, start: datetime, end: datetime, sleep_seconds: float) -> list[dict[str, Any]]:
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    rows, seen, cursor_end_ms = [], set(), end_ms
    while cursor_end_ms > start_ms:
        params = {"symbol": symbol, "period": period, "startTime": start_ms, "endTime": cursor_end_ms, "limit": 500}
        with urlopen(Request(f"{BINANCE_FUTURES_DATA_URL}/{endpoint}?{urlencode(params)}", headers={"User-Agent": "crypto-scalper/0.1"}), timeout=20) as response:
            batch = json.loads(response.read().decode("utf-8"))
        if not batch:
            break
        min_ts, new_rows = cursor_end_ms, 0
        for row in batch:
            ts = int(row.get("timestamp", 0))
            if ts < start_ms or ts > end_ms or ts in seen:
                continue
            seen.add(ts); min_ts = min(min_ts, ts); new_rows += 1
            row = dict(row); row["timestamp"] = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).replace(tzinfo=None).isoformat(); rows.append(row)
        if new_rows <= 0 or min_ts <= start_ms:
            break
        cursor_end_ms = min_ts - 300_000
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _download_funding(symbol: str, start: datetime, end: datetime, sleep_seconds: float) -> list[dict[str, Any]]:
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    rows, seen = [], set()
    while start_ms < end_ms:
        params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
        with urlopen(Request(f"{BINANCE_FUNDING_URL}?{urlencode(params)}", headers={"User-Agent": "crypto-scalper/0.1"}), timeout=20) as response:
            batch = json.loads(response.read().decode("utf-8"))
        if not batch:
            break
        last = start_ms
        for row in batch:
            ts = int(row.get("fundingTime", 0))
            if ts in seen:
                continue
            seen.add(ts); last = max(last, ts)
            rows.append({"timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).replace(tzinfo=None).isoformat(), "funding_rate": row.get("fundingRate", "0"), "mark_price": row.get("markPrice", "0")})
        if last + 1 <= start_ms:
            break
        start_ms = last + 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _to_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return int(value.timestamp() * 1000)


def main() -> int:
    parser = argparse.ArgumentParser(description="OI flush reversal long-only full-cost backtest")
    parser.add_argument("--config", default="config.live.json")
    parser.add_argument("--price-data-dir", default="data/binance_1m_3m_top100")
    parser.add_argument("--feature-data-dir", default="data/binance_oi_taker_5m")
    parser.add_argument("--funding-data-dir", default="data/binance_oi_flush_funding")
    parser.add_argument("--initial-equity", type=float, default=None)
    parser.add_argument("--include-trades", action="store_true")
    parser.add_argument("--run-experiments", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.08)
    parser.add_argument("--trade-start", default=None)
    parser.add_argument("--trade-end", default=None)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()
    if args.run_experiments:
        result = run_oi_flush_experiments(args.config, args.price_data_dir, args.feature_data_dir, args.funding_data_dir, initial_equity=args.initial_equity, include_trades=args.include_trades, download_missing=args.download_missing, sleep_seconds=args.sleep_seconds, trade_start=args.trade_start, trade_end=args.trade_end, experiment_name=args.experiment)
    else:
        result = run_oi_flush_backtest(args.config, args.price_data_dir, args.feature_data_dir, args.funding_data_dir, initial_equity=args.initial_equity, include_trades=args.include_trades, download_missing=args.download_missing, sleep_seconds=args.sleep_seconds, trade_start=args.trade_start, trade_end=args.trade_end)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

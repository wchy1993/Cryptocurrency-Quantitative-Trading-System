from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .binance_client import SymbolRules
from .data import interval_to_milliseconds, load_candles_csv, parse_timestamp
from .indicators import atr, ema, kdj, macd, rsi
from .live_config import LiveAppConfig, load_live_config
from .macro_events import MacroEvent, load_macro_events
from .market_filters import TimeframeSignal
from .live_trader import (
    AccountSnapshot,
    BinanceAutoTrader,
    BtcMarketState,
    EntryCandidate,
    LivePosition,
    SimPosition,
    _candidate_requires_extra_slot,
    _entry_execution_guard_reason_for_candles,
    _entry_position_limit,
    _entry_quality_guard_reason,
    _fast_breakout_signal_for_candles,
    _indicator_long_reclaim_guard_reason_for_candles,
    _indicator_side_float,
    _indicator_side_holding_bars,
    _effective_entry_leverage,
    _is_trend_entry_reason,
    _ordinary_breakout_adjusted_signal,
    _scale_fraction,
    _super_volume_extra_slot_candidate_allowed,
    _trend_reference_adjusted_signal_for_candles,
)
from .models import Candle, Direction, Signal
from .mtf_4h_rsi_regime import MTF_REASON_TOKEN
from .risk import (
    BacktestExecutionConfig,
    BacktestExecutionStats,
    execution_config_from_live_config,
    funding_cashflow,
    funding_rates_between,
    capacity_limited_quantity,
    market_entry_fill,
    market_exit_fill,
    validate_order_size,
)
from .strategy import VolatilityBreakoutScalper


@dataclass
class PortfolioPosition:
    symbol: str
    direction: Direction
    quantity: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    entry_fee: float
    entry_index: int
    max_holding_bars: int
    best_price: float
    leverage: int = 0
    entry_reason: str = ""
    bars_held: int = 0
    scale_ins: int = 0
    entry_time: Any = None
    raw_entry_price: float = 0.0
    entry_slippage_cost: float = 0.0
    entry_order_type: str = "market"
    entry_liquidity: str = "taker"
    signal_time: Any = None
    signal_available_time: Any = None
    mfe: float = 0.0
    mae: float = 0.0
    entry_participation_rate: float = 0.0
    capacity_fill_ratio: float = 1.0
    liquidity_reference_quote_volume: float = 0.0
    initial_stop_price: float = 0.0
    risk_budget_usdt: float = 0.0
    campaign_risk_budget_usdt: float = 0.0
    initial_leg_price_risk_usdt: float = 0.0
    initial_leg_full_cost_risk_usdt: float = 0.0
    initial_leg_actual_risk_fraction: float = 0.0
    capacity_clipped_initial_risk_fraction: float = 0.0
    stop_execution_price_estimate: float = 0.0
    estimated_stop_exit_fee_usdt: float = 0.0
    estimated_stop_exit_slippage_usdt: float = 0.0
    cmipr_max_campaign_executable_r: float = 0.0
    cmipr_min_campaign_executable_r: float = 0.0
    cmipr_max_initial_leg_executable_r: float = 0.0
    cmipr_min_initial_leg_executable_r: float = 0.0
    cmipr_executable_mfe_usdt: float = 0.0
    cmipr_executable_mae_usdt: float = 0.0
    cmipr_max_executable_r: float = 0.0
    cmipr_last_addon_index: int = -1
    cmipr_initial_quantity: float = 0.0
    cmipr_addon_1_quantity: float = 0.0
    cmipr_addon_2_quantity: float = 0.0
    reversal_v2_max_executable_r: float = 0.0
    mtper_max_campaign_executable_r: float = 0.0
    mtper_min_campaign_executable_r: float = 0.0
    mtper_max_initial_leg_executable_r: float = 0.0
    mtper_min_initial_leg_executable_r: float = 0.0
    mtper_executable_mfe_usdt: float = 0.0
    mtper_executable_mae_usdt: float = 0.0
    mtpc_max_executable_r: float = 0.0
    mtpc_min_executable_r: float = 0.0
    mtpc_max_campaign_executable_r: float = 0.0
    mtpc_min_campaign_executable_r: float = 0.0
    mtpc_max_initial_leg_executable_r: float = 0.0
    mtpc_min_initial_leg_executable_r: float = 0.0
    mtpc_executable_mfe_usdt: float = 0.0
    mtpc_executable_mae_usdt: float = 0.0
    strategy_metadata: dict[str, Any] | None = None

    def unrealized_pnl(self, mark_price: float) -> float:
        return self.direction.value * self.quantity * (mark_price - self.entry_price)


@dataclass(frozen=True)
class MarketRegime:
    weak: bool
    breadth_pct: float
    avg_return_pct: float


class HistoricalClient:
    api_key = "backtest"
    api_secret = "backtest"

    def __init__(self, candles_by_symbol: dict[str, list[Candle]], base_timeframe: str, filter_timeframes: tuple[str, ...]) -> None:
        self.base_timeframe = base_timeframe
        self.base_candles = candles_by_symbol
        self.current_index = 0
        self._rules = {
            symbol.upper(): _inferred_symbol_rules(symbol.upper(), candles)
            for symbol, candles in candles_by_symbol.items()
        }
        self.resample_factors: dict[str, int] = {}
        self.resampled: dict[str, dict[str, list[Candle]]] = {}
        for timeframe in (base_timeframe, *filter_timeframes):
            if timeframe in self.resampled:
                continue
            factor = _resample_factor(base_timeframe, timeframe)
            self.resample_factors[timeframe] = factor
            self.resampled[timeframe] = (
                candles_by_symbol
                if factor == 1
                else {symbol: _resample(candles, factor) for symbol, candles in candles_by_symbol.items()}
            )

    def klines(self, symbol: str, interval: str, limit: int = 200) -> list[Candle]:
        symbol = symbol.upper()
        if interval not in self.resampled:
            raise ValueError(f"unsupported historical interval: {interval}")
        factor = self.resample_factors[interval]
        candles = self.resampled[interval][symbol]
        if factor == 1:
            end = min(len(candles), self.current_index + 1)
            return candles[max(0, end - limit):end]

        closed_count = min(len(candles), max(0, self.current_index + 1) // factor)
        return candles[max(0, closed_count - limit):closed_count]

    def symbol_rules(self, symbol: str) -> SymbolRules:
        symbol = symbol.upper()
        return self._rules.get(symbol, _inferred_symbol_rules(symbol, self.base_candles.get(symbol, [])))


def _inferred_symbol_rules(symbol: str, candles: list[Candle]) -> SymbolRules:
    sampled_closes = [
        candle.close
        for candle in candles[: min(len(candles), 1440)]
        if candle.close > 0
    ]
    price = statistics.median(sampled_closes) if sampled_closes else 1.0
    if price >= 1000:
        tick = "0.1"
        quantity_step = "0.001"
    elif price >= 100:
        tick = "0.01"
        quantity_step = "0.001"
    elif price >= 1:
        tick = "0.001"
        quantity_step = "0.01"
    elif price >= 0.1:
        tick = "0.0001"
        quantity_step = "0.1"
    elif price >= 0.01:
        tick = "0.00001"
        quantity_step = "1"
    else:
        tick = "0.000001"
        quantity_step = "1"
    return SymbolRules(symbol, quantity_step, quantity_step, tick, "5")


def run_portfolio_backtest(
    config_path: str,
    data_dir: str,
    initial_equity: float | None = None,
    macro_data_dir: str | None = None,
    entry_timing_data_dir: str | None = None,
    execution_timing_data_dir: str | None = None,
    include_trades: bool = False,
    trade_start: Any = None,
    trade_end: Any = None,
) -> dict[str, Any]:
    config = load_live_config(config_path)
    candles_by_symbol = _load_symbol_data(data_dir, tuple(config.trading.symbols), config.trading.timeframe)
    if not candles_by_symbol:
        raise RuntimeError(f"no data loaded from {data_dir}")
    entry_timing_candles_by_symbol = None
    if entry_timing_data_dir and getattr(config.strategy, "entry_timing_filter_enabled", False):
        entry_timing_candles_by_symbol = _load_symbol_data(
            entry_timing_data_dir,
            tuple(config.trading.symbols),
            str(getattr(config.strategy, "entry_timing_timeframe", "15m")),
        )
    execution_timing_candles_by_symbol = None
    if execution_timing_data_dir and getattr(config.strategy, "entry_execution_filter_enabled", False):
        execution_timing_candles_by_symbol = _load_symbol_data(
            execution_timing_data_dir,
            tuple(config.trading.symbols),
            str(getattr(config.strategy, "entry_execution_timeframe", "5m")),
        )
    return run_portfolio_backtest_config(
        config,
        candles_by_symbol,
        initial_equity,
        macro_data_dir=macro_data_dir,
        entry_timing_candles_by_symbol=entry_timing_candles_by_symbol,
        execution_timing_candles_by_symbol=execution_timing_candles_by_symbol,
        include_trades=include_trades,
        trade_start=trade_start,
        trade_end=trade_end,
    )


def run_portfolio_backtest_config(
    config: LiveAppConfig,
    candles_by_symbol: dict[str, list[Candle]],
    initial_equity: float | None = None,
    macro_data_dir: str | None = None,
    entry_timing_candles_by_symbol: dict[str, list[Candle]] | None = None,
    execution_timing_candles_by_symbol: dict[str, list[Candle]] | None = None,
    include_trades: bool = False,
    trade_start: Any = None,
    trade_end: Any = None,
) -> dict[str, Any]:
    symbols = tuple(symbol for symbol in config.trading.symbols if symbol in candles_by_symbol)
    if not symbols:
        raise RuntimeError("no configured symbols have loaded candles")
    candles_by_symbol = {symbol: candles_by_symbol[symbol] for symbol in symbols}
    candles_by_symbol = _align_candles_by_common_timestamps(candles_by_symbol)
    common_length = min(len(candles) for candles in candles_by_symbol.values())
    historical_filter_timeframes = tuple(config.filters.timeframes)
    if getattr(config.strategy, "low_base_volume_ignition_enabled", False):
        for timeframe in ("15m", "30m", "1h", "4h"):
            if timeframe not in historical_filter_timeframes:
                historical_filter_timeframes += (timeframe,)
    client = HistoricalClient(candles_by_symbol, config.trading.timeframe, historical_filter_timeframes)
    trader = BinanceAutoTrader(config, client, logger=lambda _message: None)
    signal_cache = _build_signal_cache(config, candles_by_symbol, common_length)
    reversal_cache = _build_indicator_reversal_cache(config, candles_by_symbol, common_length)
    mtf_cache = _build_mtf_cache(config, client, tuple(candles_by_symbol))
    market_regime_cache = _build_market_regime_cache(config, candles_by_symbol, common_length)
    btc_market_state_cache = _build_btc_market_state_cache(config, candles_by_symbol, common_length)
    entry_timing_candles_by_symbol = {
        symbol: candles
        for symbol, candles in (entry_timing_candles_by_symbol or {}).items()
        if symbol in symbols and candles
    }
    entry_timing_timestamps_by_symbol = {
        symbol: [candle.timestamp for candle in candles]
        for symbol, candles in entry_timing_candles_by_symbol.items()
    }
    execution_timing_candles_by_symbol = {
        symbol: candles
        for symbol, candles in (execution_timing_candles_by_symbol or {}).items()
        if symbol in symbols and candles
    }
    execution_timing_timestamps_by_symbol = {
        symbol: [candle.timestamp for candle in candles]
        for symbol, candles in execution_timing_candles_by_symbol.items()
    }
    macro_state = _build_macro_backtest_state(config, macro_data_dir)
    starting_equity = initial_equity if initial_equity is not None else config.risk.starting_capital_usdt
    trader._peak_equity = starting_equity
    cash = starting_equity
    peak_equity = starting_equity
    day_start_equity = starting_equity
    current_day = None
    current_week = None
    week_start_equity = starting_equity
    week_peak_equity = starting_equity
    positions: dict[str, PortfolioPosition] = {}
    reentry_block_until: dict[str, int] = {}
    trades: list[dict[str, Any]] = []
    equity_curve: list[float] = []
    equity_timeline: list[tuple[Any, float]] = []
    monthly_stats: dict[str, dict[str, Any]] = {}
    indicator_reversal_loss_streak = 0
    indicator_reversal_pause_until = -1

    warmup = max(config.strategy.slow_ema, config.strategy.channel_period, config.strategy.volume_period, 96) + 8
    bar_seconds = max(1.0, interval_to_milliseconds(config.trading.timeframe) / 1000.0)
    next_entry_scan_second = 0.0
    summary_first_candle = next(iter(candles_by_symbol.values()))[0]
    for index in range(warmup, common_length - 1):
        bar_number = index - warmup
        bar_start_second = bar_number * bar_seconds
        bar_end_second = bar_start_second + bar_seconds
        entry_scan_cycles, next_entry_scan_second = _entry_scan_cycles_for_bar(
            bar_start_second,
            bar_end_second,
            next_entry_scan_second,
            max(1.0, float(config.trading.entry_scan_seconds)),
        )
        client.current_index = index
        trader._mtf_candle_cache.clear()
        timestamp = next(iter(candles_by_symbol.values()))[index].timestamp
        equity = _mark_equity(cash, positions, candles_by_symbol, index)
        if trade_start is not None and timestamp < trade_start:
            summary_first_candle = next(iter(candles_by_symbol.values()))[index]
            equity_curve.append(equity)
            equity_timeline.append((timestamp, equity))
            continue
        if trade_end is not None and timestamp > trade_end:
            break
        if trade_start is not None and summary_first_candle.timestamp < trade_start <= timestamp:
            summary_first_candle = next(iter(candles_by_symbol.values()))[index]
        macro_event = _macro_event_for_bar(macro_state, timestamp, bar_seconds)
        if current_day != timestamp.date():
            current_day = timestamp.date()
            day_start_equity = equity
        week_key = _week_key(timestamp)
        if current_week != week_key:
            current_week = week_key
            week_start_equity = equity
            week_peak_equity = equity
        else:
            week_peak_equity = max(week_peak_equity, equity)
        peak_equity = max(peak_equity, equity)
        trader._peak_equity = peak_equity

        if macro_event is not None and positions:
            closed_before = len(trades)
            for symbol in list(positions):
                cash = _close_position(
                    config,
                    cash,
                    positions,
                    trades,
                    symbol,
                    candles_by_symbol[symbol][index].open,
                    f"macro_event_pre_flatten_{macro_event.event_type}",
                    index,
                    candles_by_symbol[symbol][index].timestamp,
                )
                _mark_reentry_cooldown(config, reentry_block_until, symbol, index)
            _record_monthly_closed_trades(monthly_stats, timestamp, trades[closed_before:])
            indicator_reversal_loss_streak, indicator_reversal_pause_until = _update_indicator_reversal_pause_state(
                config,
                indicator_reversal_loss_streak,
                indicator_reversal_pause_until,
                trades[closed_before:],
                index,
            )

        if macro_event is not None:
            closed_before = len(trades)
            cash = _execute_macro_event_trade(
                config,
                cash,
                trades,
                monthly_stats,
                macro_state,
                macro_event,
            )
            _record_monthly_closed_trades(monthly_stats, macro_event.timestamp, trades[closed_before:])
            indicator_reversal_loss_streak, indicator_reversal_pause_until = _update_indicator_reversal_pause_state(
                config,
                indicator_reversal_loss_streak,
                indicator_reversal_pause_until,
                trades[closed_before:],
                index,
            )

        closed_before = len(trades)
        cash = _manage_positions(
            trader,
            config,
            cash,
            positions,
            reentry_block_until,
            trades,
            candles_by_symbol,
            index,
            client,
            signal_cache,
            reversal_cache,
            mtf_cache,
            market_regime_cache,
            btc_market_state_cache,
        )
        _record_monthly_closed_trades(monthly_stats, timestamp, trades[closed_before:])
        indicator_reversal_loss_streak, indicator_reversal_pause_until = _update_indicator_reversal_pause_state(
            config,
            indicator_reversal_loss_streak,
            indicator_reversal_pause_until,
            trades[closed_before:],
            index,
        )
        equity = _mark_equity(cash, positions, candles_by_symbol, index)
        peak_equity = max(peak_equity, equity)
        week_peak_equity = max(week_peak_equity, equity)
        trader._peak_equity = peak_equity
        equity_curve.append(equity)
        equity_timeline.append((timestamp, equity))
        _record_monthly_equity(monthly_stats, timestamp, equity)

        if equity <= day_start_equity * (1.0 - config.risk.max_daily_loss_pct):
            continue
        if _soft_drawdown_stop_hit(config, equity, peak_equity):
            continue
        if _starting_capital_drawdown_stop_hit(config, starting_equity, equity):
            continue
        if _weekly_profit_drawdown_stop_hit(config, starting_equity, week_start_equity, week_peak_equity, equity):
            continue
        if _macro_lockout_active(macro_state, timestamp):
            continue
        if entry_scan_cycles <= 0:
            continue

        bar_candidates: list[EntryCandidate] | None = None
        for _scan_cycle in range(entry_scan_cycles):
            account = _account_snapshot(config, _mark_equity(cash, positions, candles_by_symbol, index), positions, candles_by_symbol, index)
            if account.initial_margin_usage_pct >= config.risk.max_account_margin_usage_pct:
                break

            position_limit = _entry_position_limit(config)
            available_slots = max(0, position_limit - len(positions))
            cycle_slots = min(available_slots, max(1, config.trading.max_new_entries_per_cycle))
            if cycle_slots <= 0:
                break

            if bar_candidates is None:
                entry_symbols = set(config.trading.entry_symbols or config.trading.symbols)
                bar_candidates = []
                for symbol in symbols:
                    if symbol in positions or symbol not in entry_symbols:
                        continue
                    if reentry_block_until.get(symbol, -1) >= index:
                        continue
                    signal_index = index - 1
                    main_signal = signal_cache[symbol][signal_index]
                    reversal_signal = reversal_cache[symbol][signal_index]
                    if main_signal.direction == Direction.FLAT and reversal_signal.direction == Direction.FLAT:
                        continue
                    if (
                        main_signal.direction == Direction.FLAT
                        and reversal_signal.direction != Direction.FLAT
                        and _indicator_reversal_pause_active(config, index, indicator_reversal_pause_until)
                    ):
                        continue
                    candidate = _cached_entry_candidate(
                        trader,
                        config,
                        client,
                        candles_by_symbol,
                        signal_cache,
                        reversal_cache,
                        mtf_cache,
                        market_regime_cache,
                        btc_market_state_cache,
                        symbol,
                        signal_index,
                        entry_timing_candles_by_symbol=entry_timing_candles_by_symbol,
                        entry_timing_timestamps_by_symbol=entry_timing_timestamps_by_symbol,
                        execution_timing_candles_by_symbol=execution_timing_candles_by_symbol,
                        execution_timing_timestamps_by_symbol=execution_timing_timestamps_by_symbol,
                        decision_timestamp=timestamp,
                    )
                    if candidate:
                        bar_candidates.append(candidate)
                bar_candidates.sort(key=lambda item: item.rank_score, reverse=True)
            candidates = [
                candidate
                for candidate in bar_candidates
                if candidate.symbol not in positions and reentry_block_until.get(candidate.symbol, -1) < index
            ]

            opened = 0
            scan_start_positions = len(positions)
            for candidate in candidates:
                if opened >= cycle_slots:
                    break
                if _candidate_requires_extra_slot(config, scan_start_positions + opened):
                    if not _super_volume_extra_slot_candidate_allowed(config, candidate):
                        continue
                account = _account_snapshot(config, _mark_equity(cash, positions, candles_by_symbol, index), positions, candles_by_symbol, index)
                execution_price = candles_by_symbol[candidate.symbol][index].open
                execution_candle = replace(
                    candidate.candle,
                    timestamp=timestamp,
                    open=execution_price,
                    high=execution_price,
                    low=execution_price,
                    close=execution_price,
                )
                quantity_text, reason = trader._size_order(candidate.symbol, execution_price, candidate.signal, account)
                quantity = float(quantity_text)
                if reason != "ok" or quantity <= 0:
                    continue
                cash = _open_position(config, cash, positions, candidate.symbol, candidate.signal, execution_candle, quantity, index)
                _record_monthly_open(monthly_stats, timestamp, candidate.signal.direction)
                opened += 1
            if opened <= 0:
                continue
            _record_monthly_equity(monthly_stats, timestamp, _mark_equity(cash, positions, candles_by_symbol, index))

    last_index = common_length - 1
    if trade_end is not None:
        timestamps = [candle.timestamp for candle in next(iter(candles_by_symbol.values()))]
        last_index = max(0, bisect.bisect_right(timestamps, trade_end) - 1)
    last_candle = next(iter(candles_by_symbol.values()))[last_index]
    for symbol in list(positions):
        candle = candles_by_symbol[symbol][last_index]
        closed_before = len(trades)
        cash = _close_position(config, cash, positions, trades, symbol, candle.close, "end_of_data", last_index, candle.timestamp)
        _record_monthly_closed_trades(monthly_stats, last_candle.timestamp, trades[closed_before:])
    final_equity = cash
    equity_curve.append(final_equity)
    equity_timeline.append((last_candle.timestamp, final_equity))
    _record_monthly_equity(monthly_stats, last_candle.timestamp, final_equity)
    return _summary(
        starting_equity,
        final_equity,
        equity_curve,
        trades,
        summary_first_candle,
        last_candle,
        monthly_stats,
        include_trades=include_trades,
        equity_timeline=equity_timeline,
    )


def _build_signal_cache(config: Any, candles_by_symbol: dict[str, list[Candle]], common_length: int) -> dict[str, list[Signal]]:
    cache: dict[str, list[Signal]] = {}
    for symbol, candles in candles_by_symbol.items():
        strategy = VolatilityBreakoutScalper(config.strategy)
        strategy.prepare(candles)
        cache[symbol] = [strategy.signal(index, candles) for index in range(common_length)]
    return cache


def _align_candles_by_common_timestamps(candles_by_symbol: dict[str, list[Candle]]) -> dict[str, list[Candle]]:
    if not candles_by_symbol:
        return {}

    common_timestamps: set[Any] | None = None
    candles_by_timestamp: dict[str, dict[Any, Candle]] = {}
    for symbol, candles in candles_by_symbol.items():
        by_timestamp = {candle.timestamp: candle for candle in candles}
        candles_by_timestamp[symbol] = by_timestamp
        timestamps = set(by_timestamp)
        common_timestamps = timestamps if common_timestamps is None else common_timestamps & timestamps

    ordered_timestamps = sorted(common_timestamps or set())
    if not ordered_timestamps:
        raise RuntimeError("configured symbols have no overlapping candle timestamps")

    return {
        symbol: [by_timestamp[timestamp] for timestamp in ordered_timestamps]
        for symbol, by_timestamp in candles_by_timestamp.items()
    }


def _entry_scan_cycles_for_bar(
    bar_start_second: float,
    bar_end_second: float,
    next_scan_second: float,
    scan_interval_seconds: float,
) -> tuple[int, float]:
    interval = max(1.0, scan_interval_seconds)
    cycles = 0
    while next_scan_second < bar_end_second:
        if next_scan_second >= bar_start_second:
            cycles += 1
        next_scan_second += interval
    return cycles, next_scan_second


def _build_indicator_reversal_cache(config: Any, candles_by_symbol: dict[str, list[Candle]], common_length: int) -> dict[str, list[Signal]]:
    cache: dict[str, list[Signal]] = {}
    filter_config = config.filters
    if not filter_config.extreme_reversal_entry_enabled:
        flat = Signal(Direction.FLAT, 0.0, "indicator_reversal_disabled", 0.0, 0.0)
        return {symbol: [flat] * common_length for symbol in candles_by_symbol}

    minimum = max(
        filter_config.rsi_period + 2,
        filter_config.macd_slow + filter_config.macd_signal + 3,
        filter_config.kdj_period + 2,
        config.strategy.atr_period + 2,
    )
    warmup_flat = Signal(Direction.FLAT, 0.0, "indicator_warmup", 0.0, 0.0)
    no_signal = Signal(Direction.FLAT, 0.0, "indicator_no_extreme", 0.0, 0.0)
    for symbol, candles in candles_by_symbol.items():
        closes = [candle.close for candle in candles]
        rsi_values = rsi(closes, filter_config.rsi_period)
        macd_line, macd_signal_line, macd_histogram = macd(
            closes,
            filter_config.macd_fast,
            filter_config.macd_slow,
            filter_config.macd_signal,
        )
        k_values, d_values, _ = kdj(candles, filter_config.kdj_period)
        atr_values = atr(candles, config.strategy.atr_period)
        slow_values = ema(closes, config.strategy.slow_ema)
        indicator_size_multiplier = max(
            0.0,
            float(getattr(config.strategy, "indicator_reversal_size_multiplier", 1.0)),
        )
        signals: list[Signal] = []
        for index in range(common_length):
            if index + 1 < minimum:
                signals.append(warmup_flat)
                continue
            candle = candles[index]
            atr_pct = max(atr_values[index] / max(candle.close, 1e-12), 0.0001)

            current_rsi = rsi_values[index]
            previous_rsi = rsi_values[index - 1]
            current_macd = macd_line[index]
            current_signal = macd_signal_line[index]
            previous_hist = macd_histogram[index - 1]
            current_hist = macd_histogram[index]
            current_k = k_values[index]
            current_d = d_values[index]
            previous_k = k_values[index - 1]
            previous_d = d_values[index - 1]

            lookback_start = max(1, index + 1 - max(1, filter_config.reversal_cross_lookback_bars))
            long_cross = False
            short_cross = False
            for cross_index in range(lookback_start, index + 1):
                if (
                    macd_line[cross_index - 1] <= macd_signal_line[cross_index - 1]
                    and macd_line[cross_index] > macd_signal_line[cross_index]
                    and (macd_line[cross_index] < 0 or macd_signal_line[cross_index] < 0)
                ):
                    long_cross = True
                if (
                    macd_line[cross_index - 1] >= macd_signal_line[cross_index - 1]
                    and macd_line[cross_index] < macd_signal_line[cross_index]
                    and (macd_line[cross_index] > 0 or macd_signal_line[cross_index] > 0)
                ):
                    short_cross = True

            long_pre_cross = (
                filter_config.pre_cross_entry_enabled
                and current_macd < current_signal
                and current_hist > previous_hist
                and (current_macd < 0 or current_signal < 0)
                and (current_rsi <= filter_config.long_extreme_rsi or min(current_k, current_d) <= filter_config.long_extreme_kdj)
                and (current_rsi >= previous_rsi or current_k > previous_k)
            )
            short_pre_cross = (
                filter_config.pre_cross_entry_enabled
                and current_macd > current_signal
                and current_hist < previous_hist
                and (current_macd > 0 or current_signal > 0)
                and (current_rsi >= filter_config.short_extreme_rsi or max(current_k, current_d) >= filter_config.short_extreme_kdj)
                and (current_rsi <= previous_rsi or current_k < previous_k)
            )

            if long_cross:
                if config.strategy.long_risk_bias <= 0:
                    signals.append(no_signal)
                else:
                    strict_guard_reason = _indicator_confirmed_cross_required_extreme_guard_reason(
                        config,
                        Direction.LONG,
                        current_rsi,
                        current_k,
                        current_d,
                    )
                    if strict_guard_reason:
                        signals.append(Signal(Direction.FLAT, 0.0, strict_guard_reason, 0.0, 0.0))
                        continue
                    context_guard_reason = _indicator_confirmed_cross_context_guard_reason(
                        config,
                        Direction.LONG,
                        closes,
                        rsi_values,
                        k_values,
                        d_values,
                        slow_values,
                        atr_values,
                        index,
                    )
                    if context_guard_reason:
                        signals.append(Signal(Direction.FLAT, 0.0, context_guard_reason, 0.0, 0.0))
                        continue
                    guard_reason = _indicator_trend_guard_reason(config, Direction.LONG, closes, slow_values, atr_values, index)
                    if guard_reason:
                        signals.append(Signal(Direction.FLAT, 0.0, guard_reason, 0.0, 0.0))
                        continue
                    long_stop_pct = max(
                        atr_pct * _indicator_side_float(config.strategy, Direction.LONG, "stop_loss_atr", config.strategy.stop_loss_atr),
                        0.0008,
                    )
                    long_take_profit_pct = max(
                        atr_pct * _indicator_side_float(config.strategy, Direction.LONG, "take_profit_atr", config.strategy.take_profit_atr),
                        long_stop_pct * 1.05,
                    )
                    long_size_multiplier = _indicator_side_float(
                        config.strategy,
                        Direction.LONG,
                        "size_multiplier",
                        indicator_size_multiplier,
                    )
                    long_risk_multiplier = _indicator_side_float(
                        config.strategy,
                        Direction.LONG,
                        "confirmed_cross_risk_multiplier",
                        filter_config.confirmed_cross_risk_multiplier,
                    )
                    signals.append(Signal(
                        Direction.LONG,
                        0.7,
                        f"indicator_long_macd_golden_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                        long_stop_pct,
                        long_take_profit_pct,
                        risk_multiplier=long_risk_multiplier * config.strategy.long_risk_bias * long_size_multiplier,
                        max_holding_bars=_indicator_side_holding_bars(config.strategy, Direction.LONG),
                    ))
            elif short_cross:
                if config.strategy.short_risk_bias <= 0:
                    signals.append(no_signal)
                else:
                    strict_guard_reason = _indicator_confirmed_cross_required_extreme_guard_reason(
                        config,
                        Direction.SHORT,
                        current_rsi,
                        current_k,
                        current_d,
                    )
                    if strict_guard_reason:
                        signals.append(Signal(Direction.FLAT, 0.0, strict_guard_reason, 0.0, 0.0))
                        continue
                    context_guard_reason = _indicator_confirmed_cross_context_guard_reason(
                        config,
                        Direction.SHORT,
                        closes,
                        rsi_values,
                        k_values,
                        d_values,
                        slow_values,
                        atr_values,
                        index,
                    )
                    if context_guard_reason:
                        signals.append(Signal(Direction.FLAT, 0.0, context_guard_reason, 0.0, 0.0))
                        continue
                    guard_reason = _indicator_trend_guard_reason(config, Direction.SHORT, closes, slow_values, atr_values, index)
                    if guard_reason:
                        signals.append(Signal(Direction.FLAT, 0.0, guard_reason, 0.0, 0.0))
                        continue
                    close_position_multiplier, close_position_reason = _indicator_short_close_position_adjustment(config, candle)
                    if close_position_multiplier <= 0:
                        signals.append(Signal(Direction.FLAT, 0.0, close_position_reason or "indicator_short_blocked_high_close_position", 0.0, 0.0))
                        continue
                    short_stop_pct = max(
                        atr_pct * _indicator_side_float(config.strategy, Direction.SHORT, "stop_loss_atr", config.strategy.stop_loss_atr),
                        0.0008,
                    )
                    short_take_profit_pct = max(
                        atr_pct * _indicator_side_float(config.strategy, Direction.SHORT, "take_profit_atr", config.strategy.take_profit_atr),
                        short_stop_pct * 1.05,
                    )
                    short_size_multiplier = _indicator_side_float(
                        config.strategy,
                        Direction.SHORT,
                        "size_multiplier",
                        indicator_size_multiplier,
                    )
                    short_risk_multiplier = _indicator_side_float(
                        config.strategy,
                        Direction.SHORT,
                        "confirmed_cross_risk_multiplier",
                        filter_config.confirmed_cross_risk_multiplier,
                    )
                    signals.append(Signal(
                        Direction.SHORT,
                        0.7,
                        (
                            f"indicator_short_macd_dead_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
                            + (f" {close_position_reason}" if close_position_reason else "")
                        ),
                        short_stop_pct,
                        short_take_profit_pct,
                        risk_multiplier=short_risk_multiplier * config.strategy.short_risk_bias * short_size_multiplier * close_position_multiplier,
                        max_holding_bars=_indicator_side_holding_bars(config.strategy, Direction.SHORT),
                    ))
            elif long_pre_cross:
                if config.strategy.long_risk_bias <= 0:
                    signals.append(no_signal)
                else:
                    guard_reason = _indicator_trend_guard_reason(config, Direction.LONG, closes, slow_values, atr_values, index)
                    if guard_reason:
                        signals.append(Signal(Direction.FLAT, 0.0, guard_reason, 0.0, 0.0))
                        continue
                    long_stop_pct = max(
                        atr_pct * _indicator_side_float(config.strategy, Direction.LONG, "stop_loss_atr", config.strategy.stop_loss_atr),
                        0.0008,
                    )
                    long_take_profit_pct = max(
                        atr_pct * _indicator_side_float(config.strategy, Direction.LONG, "take_profit_atr", config.strategy.take_profit_atr),
                        long_stop_pct * 1.05,
                    )
                    long_size_multiplier = _indicator_side_float(
                        config.strategy,
                        Direction.LONG,
                        "size_multiplier",
                        indicator_size_multiplier,
                    )
                    long_pre_cross_multiplier = _indicator_side_float(
                        config.strategy,
                        Direction.LONG,
                        "pre_cross_risk_multiplier",
                        filter_config.pre_cross_risk_multiplier,
                    )
                    long_holding_bars = _indicator_side_holding_bars(config.strategy, Direction.LONG)
                    signals.append(Signal(
                        Direction.LONG,
                        0.45,
                        f"indicator_long_pre_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                        long_stop_pct,
                        long_take_profit_pct,
                        risk_multiplier=long_pre_cross_multiplier * config.strategy.long_risk_bias * long_size_multiplier,
                        max_holding_bars=max(1, min(long_holding_bars, max(6, config.strategy.max_holding_bars // 2))),
                    ))
            elif short_pre_cross:
                if config.strategy.short_risk_bias <= 0:
                    signals.append(no_signal)
                else:
                    guard_reason = _indicator_trend_guard_reason(config, Direction.SHORT, closes, slow_values, atr_values, index)
                    if guard_reason:
                        signals.append(Signal(Direction.FLAT, 0.0, guard_reason, 0.0, 0.0))
                        continue
                    close_position_multiplier, close_position_reason = _indicator_short_close_position_adjustment(config, candle)
                    if close_position_multiplier <= 0:
                        signals.append(Signal(Direction.FLAT, 0.0, close_position_reason or "indicator_short_blocked_high_close_position", 0.0, 0.0))
                        continue
                    short_stop_pct = max(
                        atr_pct * _indicator_side_float(config.strategy, Direction.SHORT, "stop_loss_atr", config.strategy.stop_loss_atr),
                        0.0008,
                    )
                    short_take_profit_pct = max(
                        atr_pct * _indicator_side_float(config.strategy, Direction.SHORT, "take_profit_atr", config.strategy.take_profit_atr),
                        short_stop_pct * 1.05,
                    )
                    short_size_multiplier = _indicator_side_float(
                        config.strategy,
                        Direction.SHORT,
                        "size_multiplier",
                        indicator_size_multiplier,
                    )
                    short_pre_cross_multiplier = _indicator_side_float(
                        config.strategy,
                        Direction.SHORT,
                        "pre_cross_risk_multiplier",
                        filter_config.pre_cross_risk_multiplier,
                    )
                    short_holding_bars = _indicator_side_holding_bars(config.strategy, Direction.SHORT)
                    signals.append(Signal(
                        Direction.SHORT,
                        0.45,
                        (
                            f"indicator_short_pre_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
                            + (f" {close_position_reason}" if close_position_reason else "")
                        ),
                        short_stop_pct,
                        short_take_profit_pct,
                        risk_multiplier=short_pre_cross_multiplier * config.strategy.short_risk_bias * short_size_multiplier * close_position_multiplier,
                        max_holding_bars=max(1, min(short_holding_bars, max(6, config.strategy.max_holding_bars // 2))),
                    ))
            else:
                signals.append(no_signal)
        cache[symbol] = signals
    return cache


def _indicator_trend_guard_reason(
    config: Any,
    direction: Direction,
    closes: list[float],
    slow_values: list[float],
    atr_values: list[float],
    index: int,
) -> str | None:
    strategy = config.strategy
    if not getattr(strategy, "indicator_trend_guard_enabled", False):
        return None
    minimum = max(strategy.slow_ema, getattr(strategy, "indicator_trend_guard_lookback_bars", 12) + 1)
    if index + 1 < minimum:
        return None

    lookback = max(1, min(getattr(strategy, "indicator_trend_guard_lookback_bars", 12), index))
    current_close = closes[index]
    current_slow = slow_values[index]
    atr_value = max(atr_values[index], 1e-12)
    slope_atr = (slow_values[index] - slow_values[index - lookback]) / atr_value
    buffer = max(0.0, getattr(strategy, "indicator_trend_guard_buffer_atr", 0.10)) * atr_value
    slope_threshold = max(0.0, getattr(strategy, "indicator_trend_guard_slope_atr", 0.0))

    if direction == Direction.SHORT and current_close > current_slow + buffer and slope_atr >= slope_threshold:
        return f"indicator_short_blocked_30m_uptrend close={current_close:.6g} slow={current_slow:.6g} slope_atr={slope_atr:.2f}"
    if direction == Direction.LONG and current_close < current_slow - buffer and slope_atr <= -slope_threshold:
        return f"indicator_long_blocked_30m_downtrend close={current_close:.6g} slow={current_slow:.6g} slope_atr={slope_atr:.2f}"
    return None


def _indicator_short_close_position_adjustment(config: Any, candle: Candle) -> tuple[float, str | None]:
    threshold = float(getattr(config.strategy, "indicator_short_max_close_position", 0.0))
    if threshold <= 0:
        return 1.0, None
    candle_range = max(candle.high - candle.low, 1e-12)
    close_position = (candle.close - candle.low) / candle_range
    if close_position <= threshold:
        return 1.0, None
    multiplier = max(0.0, min(1.0, float(getattr(config.strategy, "indicator_short_high_close_risk_multiplier", 1.0))))
    reason = f"indicator_short_high_close_risk close_pos={close_position:.2f}>{threshold:.2f} mult={multiplier:.2f}"
    return multiplier, reason


def _indicator_confirmed_cross_context_guard_reason(
    config: Any,
    direction: Direction,
    closes: list[float],
    rsi_values: list[float],
    k_values: list[float],
    d_values: list[float],
    slow_values: list[float],
    atr_values: list[float],
    index: int,
) -> str | None:
    strategy = config.strategy
    if not getattr(strategy, "indicator_confirmed_cross_extreme_guard_enabled", False):
        return None
    lookback = max(1, min(int(getattr(strategy, "indicator_confirmed_extreme_lookback_bars", 4)), index + 1))
    start = index + 1 - lookback
    recent_rsi = rsi_values[start:index + 1]
    recent_kd = [value for pair in zip(k_values[start:index + 1], d_values[start:index + 1]) for value in pair]

    if direction == Direction.LONG:
        has_extreme = (
            min(recent_rsi) <= float(getattr(strategy, "indicator_confirmed_long_max_rsi", 42.0))
            or min(recent_kd) <= float(getattr(strategy, "indicator_confirmed_long_max_kdj", 35.0))
        )
        if has_extreme:
            return None
        conflict_reason = _indicator_confirmed_counter_trend_conflict_reason(config, direction, closes, slow_values, atr_values, index)
        if conflict_reason:
            return (
                f"indicator_long_blocked_no_cold_context "
                f"rsi_min={min(recent_rsi):.1f} kdj_min={min(recent_kd):.1f} {conflict_reason}"
            )
        return None

    if direction == Direction.SHORT:
        has_extreme = (
            max(recent_rsi) >= float(getattr(strategy, "indicator_confirmed_short_min_rsi", 58.0))
            or max(recent_kd) >= float(getattr(strategy, "indicator_confirmed_short_min_kdj", 65.0))
        )
        if has_extreme:
            return None
        conflict_reason = _indicator_confirmed_counter_trend_conflict_reason(config, direction, closes, slow_values, atr_values, index)
        if conflict_reason:
            return (
                f"indicator_short_blocked_no_hot_context "
                f"rsi_max={max(recent_rsi):.1f} kdj_max={max(recent_kd):.1f} {conflict_reason}"
            )
        return None

    return None


def _indicator_confirmed_cross_required_extreme_guard_reason(
    config: Any,
    direction: Direction,
    current_rsi: float,
    current_k: float,
    current_d: float,
) -> str | None:
    strategy = config.strategy
    if not getattr(strategy, "indicator_confirmed_cross_extreme_required_enabled", False):
        return None

    if direction == Direction.LONG:
        max_rsi = float(getattr(strategy, "indicator_confirmed_cross_long_max_rsi", 45.0))
        max_kdj = float(getattr(strategy, "indicator_confirmed_cross_long_max_kdj", 35.0))
        if current_rsi <= max_rsi or min(current_k, current_d) <= max_kdj:
            return None
        return (
            f"indicator_long_blocked_not_cold_enough "
            f"rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
        )

    if direction == Direction.SHORT:
        min_rsi = float(getattr(strategy, "indicator_confirmed_cross_short_min_rsi", 60.0))
        min_kdj = float(getattr(strategy, "indicator_confirmed_cross_short_min_kdj", 70.0))
        if current_rsi >= min_rsi or max(current_k, current_d) >= min_kdj:
            return None
        return (
            f"indicator_short_blocked_not_hot_enough "
            f"rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
        )

    return None


def _indicator_confirmed_counter_trend_conflict_reason(
    config: Any,
    direction: Direction,
    closes: list[float],
    slow_values: list[float],
    atr_values: list[float],
    index: int,
) -> str | None:
    strategy = config.strategy
    if not getattr(strategy, "indicator_confirmed_trend_fallback_enabled", True):
        return None
    minimum = max(strategy.slow_ema, 2)
    if index + 1 < minimum or index <= 0:
        return None
    lookback = max(1, min(getattr(strategy, "indicator_trend_guard_lookback_bars", 12), index))
    current_close = closes[index]
    current_slow = slow_values[index]
    atr_value = max(atr_values[index], 1e-12)
    slope_atr = (slow_values[index] - slow_values[index - lookback]) / atr_value
    buffer = max(0.0, getattr(strategy, "indicator_confirmed_trend_buffer_atr", 0.15)) * atr_value
    slope_threshold = max(0.0, getattr(strategy, "indicator_confirmed_trend_slope_atr", 0.05))
    if direction == Direction.LONG and current_close < current_slow - buffer and slope_atr <= -slope_threshold:
        return f"against_30m_downtrend close={current_close:.6g} slow={current_slow:.6g} slope_atr={slope_atr:.2f}"
    if direction == Direction.SHORT and current_close > current_slow + buffer and slope_atr >= slope_threshold:
        return f"against_30m_uptrend close={current_close:.6g} slow={current_slow:.6g} slope_atr={slope_atr:.2f}"
    return None


def _build_mtf_cache(config: Any, client: HistoricalClient, symbols: tuple[str, ...]) -> dict[tuple[str, str], list[TimeframeSignal | None]]:
    output: dict[tuple[str, str], list[TimeframeSignal | None]] = {}
    if not config.filters.enabled:
        return output
    for symbol in symbols:
        for timeframe in config.filters.timeframes:
            candles = client.resampled[timeframe][symbol]
            closes = [candle.close for candle in candles]
            rsi_values = rsi(closes, config.filters.rsi_period)
            macd_values, macd_signal_values, macd_histogram_values = macd(
                closes,
                config.filters.macd_fast,
                config.filters.macd_slow,
                config.filters.macd_signal,
            )
            k_values, d_values, j_values = kdj(candles, config.filters.kdj_period)
            snapshots: list[TimeframeSignal | None] = []
            for index, candle in enumerate(candles):
                if index < trader_mtf_warmup(config):
                    snapshots.append(None)
                    continue
                snapshots.append(TimeframeSignal(
                    timeframe=timeframe,
                    close=candle.close,
                    rsi=rsi_values[index],
                    previous_rsi=rsi_values[index - 1],
                    macd=macd_values[index],
                    macd_signal=macd_signal_values[index],
                    macd_histogram=macd_histogram_values[index],
                    previous_macd_histogram=macd_histogram_values[index - 1],
                    k=k_values[index],
                    d=d_values[index],
                    j=j_values[index],
                    previous_k=k_values[index - 1],
                    previous_d=d_values[index - 1],
                ))
            output[(symbol, timeframe)] = snapshots
    return output


def _build_market_regime_cache(config: Any, candles_by_symbol: dict[str, list[Candle]], common_length: int) -> list[MarketRegime]:
    strategy = config.strategy
    if not (
        getattr(strategy, "weak_market_long_filter_enabled", False)
        or getattr(strategy, "strong_market_short_filter_enabled", False)
    ):
        return [MarketRegime(False, 1.0, 0.0) for _ in range(common_length)]

    weak_lookback = int(getattr(strategy, "weak_market_lookback_bars", 48))
    strong_lookback = int(getattr(strategy, "strong_market_lookback_bars", weak_lookback))
    lookback = max(1, weak_lookback, strong_lookback)
    breadth_threshold = max(0.0, min(1.0, float(getattr(strategy, "weak_market_breadth_threshold", 0.42))))
    avg_return_threshold = float(getattr(strategy, "weak_market_avg_return_threshold", -0.012))
    closes_by_symbol = {symbol: [candle.close for candle in candles] for symbol, candles in candles_by_symbol.items()}
    slow_by_symbol = {symbol: ema(closes, strategy.slow_ema) for symbol, closes in closes_by_symbol.items()}
    output: list[MarketRegime] = []

    for index in range(common_length):
        if index < max(lookback, strategy.slow_ema):
            output.append(MarketRegime(False, 1.0, 0.0))
            continue

        valid = 0
        constructive = 0
        returns = []
        for symbol, closes in closes_by_symbol.items():
            if index >= len(closes) or index - lookback < 0:
                continue
            previous = closes[index - lookback]
            current = closes[index]
            if previous <= 0:
                continue
            valid += 1
            returns.append(current / previous - 1.0)
            if current > previous and current > slow_by_symbol[symbol][index]:
                constructive += 1

        breadth = 1.0 if valid <= 0 else constructive / valid
        avg_return = 0.0 if not returns else sum(returns) / len(returns)
        weak = breadth <= breadth_threshold and avg_return <= avg_return_threshold
        output.append(MarketRegime(weak, breadth, avg_return * 100.0))
    return output


def _build_btc_market_state_cache(config: Any, candles_by_symbol: dict[str, list[Candle]], common_length: int) -> list[BtcMarketState]:
    strategy = config.strategy
    if not getattr(strategy, "btc_market_filter_enabled", False):
        return [BtcMarketState(Direction.FLAT, 0.0, 0.0, "btc_market_disabled") for _ in range(common_length)]

    symbol = str(getattr(strategy, "btc_market_symbol", "BTCUSDT")).upper()
    candles = candles_by_symbol.get(symbol)
    if not candles:
        return [BtcMarketState(Direction.FLAT, 0.0, 0.0, "btc_market_no_data") for _ in range(common_length)]

    primary = _build_btc_market_state_cache_for_timeframe(
        config,
        candles,
        common_length,
        str(getattr(strategy, "btc_market_timeframe", config.trading.timeframe)),
        max(1, int(getattr(strategy, "btc_market_lookback_bars", 12))),
    )
    if not getattr(strategy, "btc_market_confirmation_enabled", False):
        return primary

    confirmation = _build_btc_market_state_cache_for_timeframe(
        config,
        candles,
        common_length,
        str(getattr(strategy, "btc_market_confirmation_timeframe", "8h")),
        max(1, int(getattr(strategy, "btc_market_confirmation_lookback_bars", 2))),
    )
    return [
        _combine_btc_market_states(primary_state, confirmation_state)
        for primary_state, confirmation_state in zip(primary, confirmation)
    ]


def _build_btc_market_state_cache_for_timeframe(
    config: Any,
    base_candles: list[Candle],
    common_length: int,
    timeframe: str,
    lookback: int,
) -> list[BtcMarketState]:
    strategy = config.strategy
    try:
        factor = _resample_factor(config.trading.timeframe, timeframe)
    except ValueError:
        factor = 1
    candles = base_candles if factor == 1 else _resample(base_candles, factor)
    closes = [candle.close for candle in candles]
    ema_period = max(2, int(getattr(strategy, "btc_market_ema_period", strategy.slow_ema)))
    slow_values = ema(closes, ema_period)
    atr_values = atr(candles, strategy.atr_period)
    bull_return = float(getattr(strategy, "btc_bull_return_threshold", 0.006))
    bear_return = float(getattr(strategy, "btc_bear_return_threshold", -0.006))
    slope_threshold = float(getattr(strategy, "btc_market_slope_atr_threshold", 0.20))
    warmup = max(lookback, ema_period, strategy.atr_period)
    timeframe_states: list[BtcMarketState] = []

    for index in range(len(candles)):
        if index >= len(candles) or index < warmup:
            timeframe_states.append(BtcMarketState(Direction.FLAT, 0.0, 0.0, f"btc_{timeframe}_warmup"))
            continue

        current = closes[index]
        previous = closes[index - lookback]
        atr_value = atr_values[index]
        return_pct = current / previous - 1.0 if previous > 0 else 0.0
        slope_atr = (slow_values[index] - slow_values[index - lookback]) / atr_value if atr_value > 0 else 0.0
        if return_pct >= bull_return and slope_atr >= slope_threshold and current >= slow_values[index]:
            direction = Direction.LONG
            label = "btc_bull"
        elif return_pct <= bear_return and slope_atr <= -slope_threshold and current <= slow_values[index]:
            direction = Direction.SHORT
            label = "btc_bear"
        else:
            direction = Direction.FLAT
            label = "btc_neutral"
        timeframe_states.append(BtcMarketState(direction, return_pct * 100.0, slope_atr, f"{label}_{timeframe} ret={return_pct * 100:.2f}% slope_atr={slope_atr:.2f}"))

    output: list[BtcMarketState] = []
    for index in range(common_length):
        if factor <= 1:
            state_index = min(index, len(timeframe_states) - 1)
        else:
            closed_count = min(len(timeframe_states), max(0, index + 1) // factor)
            state_index = closed_count - 1
        if state_index < 0:
            output.append(BtcMarketState(Direction.FLAT, 0.0, 0.0, f"btc_{timeframe}_warmup"))
        else:
            output.append(timeframe_states[state_index])
    return output


def _combine_btc_market_states(primary: BtcMarketState, confirmation: BtcMarketState) -> BtcMarketState:
    if primary.direction == Direction.FLAT:
        return primary
    if confirmation.direction == primary.direction:
        return BtcMarketState(
            primary.direction,
            primary.return_pct,
            primary.slope_atr,
            f"{primary.reason}; confirm={confirmation.reason}",
        )
    return BtcMarketState(
        Direction.FLAT,
        primary.return_pct,
        primary.slope_atr,
        f"btc_mixed primary={primary.reason}; confirm={confirmation.reason}",
    )


def trader_mtf_warmup(config: Any) -> int:
    return max(
        config.filters.rsi_period + 2,
        config.filters.macd_slow + config.filters.macd_signal + 2,
        config.filters.kdj_period + 2,
    )


def _cached_mtf_allows(
    trader: BinanceAutoTrader,
    config: Any,
    client: HistoricalClient,
    mtf_cache: dict[tuple[str, str], list[TimeframeSignal | None]],
    symbol: str,
    index: int,
    direction: Direction,
) -> tuple[bool, str]:
    if not config.filters.enabled:
        return True, "disabled"
    frames: list[TimeframeSignal] = []
    for timeframe in config.filters.timeframes:
        factor = client.resample_factors[timeframe]
        closed_count = max(0, index + 1) // factor
        snapshot_index = closed_count - 1
        snapshots = mtf_cache.get((symbol, timeframe), [])
        if snapshot_index < 0 or snapshot_index >= len(snapshots):
            return False, f"{timeframe}_candles_insufficient"
        snapshot = snapshots[snapshot_index]
        if snapshot is None:
            return False, f"{timeframe}_candles_insufficient"
        frames.append(snapshot)
    return trader._mtf_filter.evaluate(direction, frames)


def _cached_entry_candidate(
    trader: BinanceAutoTrader,
    config: Any,
    client: HistoricalClient,
    candles_by_symbol: dict[str, list[Candle]],
    signal_cache: dict[str, list[Signal]],
    reversal_cache: dict[str, list[Signal]],
    mtf_cache: dict[tuple[str, str], list[TimeframeSignal | None]],
    market_regime_cache: list[MarketRegime],
    btc_market_state_cache: list[BtcMarketState],
    symbol: str,
    index: int,
    entry_timing_candles_by_symbol: dict[str, list[Candle]] | None = None,
    entry_timing_timestamps_by_symbol: dict[str, list[Any]] | None = None,
    execution_timing_candles_by_symbol: dict[str, list[Candle]] | None = None,
    execution_timing_timestamps_by_symbol: dict[str, list[Any]] | None = None,
    decision_timestamp: Any = None,
) -> EntryCandidate | None:
    if getattr(config.strategy, "low_base_volume_ignition_enabled", False):
        candidate = trader._low_base_volume_ignition_entry_candidate(symbol)
        if candidate is not None or getattr(config.strategy, "low_base_volume_ignition_disable_legacy", False):
            return candidate

    start = max(0, index + 1 - config.trading.kline_limit)
    candles = candles_by_symbol[symbol][start:index + 1]
    if len(candles) < VolatilityBreakoutScalper(config.strategy).warmup_bars:
        return None

    signal = signal_cache[symbol][index]
    if signal.direction == Direction.FLAT:
        signal = reversal_cache[symbol][index]
        if signal.direction == Direction.FLAT:
            return None

    return _filtered_entry_candidate_from_signal(
        trader,
        config,
        client,
        mtf_cache,
        market_regime_cache,
        btc_market_state_cache,
        symbol,
        index,
        signal,
        candles,
        candles[-1],
        entry_timing_candles_by_symbol=entry_timing_candles_by_symbol,
        entry_timing_timestamps_by_symbol=entry_timing_timestamps_by_symbol,
        execution_timing_candles_by_symbol=execution_timing_candles_by_symbol,
        execution_timing_timestamps_by_symbol=execution_timing_timestamps_by_symbol,
        decision_timestamp=decision_timestamp,
    )


def _filtered_entry_candidate_from_signal(
    trader: BinanceAutoTrader,
    config: Any,
    client: HistoricalClient,
    mtf_cache: dict[tuple[str, str], list[TimeframeSignal | None]],
    market_regime_cache: list[MarketRegime],
    btc_market_state_cache: list[BtcMarketState],
    symbol: str,
    index: int,
    signal: Signal,
    rank_candles: list[Candle],
    candidate_candle: Candle,
    entry_timing_candles_by_symbol: dict[str, list[Candle]] | None = None,
    entry_timing_timestamps_by_symbol: dict[str, list[Any]] | None = None,
    execution_timing_candles_by_symbol: dict[str, list[Candle]] | None = None,
    execution_timing_timestamps_by_symbol: dict[str, list[Any]] | None = None,
    decision_timestamp: Any = None,
) -> EntryCandidate | None:
    reference_guard_reason = _indicator_reference_guard_reason(config, client, symbol, signal)
    if reference_guard_reason:
        return None

    short_guard_reason = trader._indicator_short_guard_reason(symbol, signal, rank_candles)
    if short_guard_reason:
        trader._record_short_guard_reject(short_guard_reason)
        return None

    allowed, filter_reason = _cached_mtf_allows(trader, config, client, mtf_cache, symbol, index, signal.direction)
    if not allowed:
        return None

    rank_score, momentum_pct, volume_ratio = trader._entry_rank_metrics(signal, rank_candles)
    regime = market_regime_cache[index] if 0 <= index < len(market_regime_cache) else MarketRegime(False, 1.0, 0.0)
    signal = _weak_market_adjusted_signal(config, signal, rank_score, regime)
    if signal is None:
        return None
    rank_score, momentum_pct, volume_ratio = trader._entry_rank_metrics(signal, rank_candles)
    signal = _strong_market_adjusted_signal(config, signal, rank_score, regime)
    if signal is None:
        return None
    rank_score, momentum_pct, volume_ratio = trader._entry_rank_metrics(signal, rank_candles)
    btc_state = btc_market_state_cache[index] if 0 <= index < len(btc_market_state_cache) else BtcMarketState(Direction.FLAT, 0.0, 0.0, "btc_market_missing")
    signal = _btc_market_adjusted_signal(config, signal, rank_score, momentum_pct, volume_ratio, btc_state)
    if signal is None:
        return None
    rank_score, momentum_pct, volume_ratio = trader._entry_rank_metrics(signal, rank_candles)
    signal = _trend_reference_adjusted_signal(config, client, symbol, signal)
    if signal is None:
        return None
    rank_score, momentum_pct, volume_ratio = trader._entry_rank_metrics(signal, rank_candles)
    signal = _ordinary_breakout_adjusted_signal(config, signal, rank_score)
    if signal is None:
        return None
    rank_score, momentum_pct, volume_ratio = trader._entry_rank_metrics(signal, rank_candles)
    candidate = EntryCandidate(symbol, signal, candidate_candle, rank_score, momentum_pct, volume_ratio, filter_reason)
    if _entry_quality_guard_reason(config, candidate):
        return None
    timing_guard_reason = _entry_timing_guard_reason(
        config,
        candidate,
        entry_timing_candles_by_symbol,
        entry_timing_timestamps_by_symbol,
        decision_timestamp,
    )
    if timing_guard_reason:
        return None
    execution_guard_reason = _execution_timing_guard_reason(
        config,
        candidate,
        execution_timing_candles_by_symbol,
        execution_timing_timestamps_by_symbol,
        decision_timestamp,
    )
    if execution_guard_reason:
        return None
    return candidate


def _cached_fast_breakout_candidate(
    trader: BinanceAutoTrader,
    config: Any,
    client: HistoricalClient,
    mtf_cache: dict[tuple[str, str], list[TimeframeSignal | None]],
    market_regime_cache: list[MarketRegime],
    btc_market_state_cache: list[BtcMarketState],
    symbol: str,
    index: int,
    fast_candles_by_symbol: dict[str, list[Candle]] | None,
    fast_timestamps_by_symbol: dict[str, list[Any]] | None,
    decision_timestamp: Any,
    entry_timing_candles_by_symbol: dict[str, list[Candle]] | None = None,
    entry_timing_timestamps_by_symbol: dict[str, list[Any]] | None = None,
    execution_timing_candles_by_symbol: dict[str, list[Candle]] | None = None,
    execution_timing_timestamps_by_symbol: dict[str, list[Any]] | None = None,
) -> EntryCandidate | None:
    if not getattr(config.strategy, "fast_breakout_enabled", False):
        return None
    if not fast_candles_by_symbol or not fast_timestamps_by_symbol or decision_timestamp is None:
        return None

    candles = fast_candles_by_symbol.get(symbol)
    timestamps = fast_timestamps_by_symbol.get(symbol)
    if not candles or not timestamps:
        return None

    timeframe = str(getattr(config.strategy, "fast_breakout_timeframe", "5m"))
    closed_before = decision_timestamp - timedelta(milliseconds=interval_to_milliseconds(timeframe))
    end = bisect.bisect_right(timestamps, closed_before)
    if end <= 0:
        return None
    start = max(0, end - max(100, int(getattr(config.strategy, "fast_breakout_channel_period", 18)) + int(getattr(config.strategy, "fast_breakout_volume_period", 24)) + 20))
    fast_candles = candles[start:end]
    signal = _fast_breakout_signal_for_candles(config, fast_candles)
    if signal.direction == Direction.FLAT:
        return None

    return _filtered_entry_candidate_from_signal(
        trader,
        config,
        client,
        mtf_cache,
        market_regime_cache,
        btc_market_state_cache,
        symbol,
        index,
        signal,
        fast_candles,
        fast_candles[-1],
        entry_timing_candles_by_symbol=entry_timing_candles_by_symbol,
        entry_timing_timestamps_by_symbol=entry_timing_timestamps_by_symbol,
        execution_timing_candles_by_symbol=execution_timing_candles_by_symbol,
        execution_timing_timestamps_by_symbol=execution_timing_timestamps_by_symbol,
        decision_timestamp=decision_timestamp,
    )


def _entry_timing_guard_reason(
    config: Any,
    candidate: EntryCandidate,
    candles_by_symbol: dict[str, list[Candle]] | None,
    timestamps_by_symbol: dict[str, list[Any]] | None,
    decision_timestamp: Any,
) -> str | None:
    strategy = config.strategy
    if not getattr(strategy, "entry_timing_filter_enabled", False):
        return None
    if not candles_by_symbol or not timestamps_by_symbol or decision_timestamp is None:
        return None

    candles = candles_by_symbol.get(candidate.symbol)
    timestamps = timestamps_by_symbol.get(candidate.symbol)
    if not candles or not timestamps:
        return None

    timeframe = str(getattr(strategy, "entry_timing_timeframe", "15m"))
    closed_before = decision_timestamp - timedelta(milliseconds=interval_to_milliseconds(timeframe))
    end = bisect.bisect_right(timestamps, closed_before)
    if end <= 0:
        return None
    fast_period = max(2, int(getattr(strategy, "entry_timing_rsi_fast_period", 6)))
    mid_period = max(fast_period, int(getattr(strategy, "entry_timing_rsi_mid_period", 12)))
    start = max(0, end - max(80, mid_period + 10))
    return _entry_timing_guard_reason_for_candles(config, candidate, candles[start:end])


def _entry_timing_guard_reason_for_candles(config: Any, candidate: EntryCandidate, candles: list[Candle]) -> str | None:
    strategy = config.strategy
    fast_period = max(2, int(getattr(strategy, "entry_timing_rsi_fast_period", 6)))
    mid_period = max(fast_period, int(getattr(strategy, "entry_timing_rsi_mid_period", 12)))
    if len(candles) < mid_period + 3:
        return None

    reclaim_reason = _indicator_long_reclaim_guard_reason_for_candles(config, candidate, candles, str(getattr(strategy, "entry_timing_timeframe", "15m")))
    if reclaim_reason:
        return reclaim_reason

    window = candles[-max(80, mid_period + 10):]
    closes = [candle.close for candle in window]
    latest = closes[-1]
    signal_close = candidate.candle.close
    if latest <= 0 or signal_close <= 0:
        return None

    direction = candidate.signal.direction
    chase_pct = (latest / signal_close - 1.0) * direction.value
    max_chase = max(0.0, float(getattr(strategy, "entry_timing_max_chase_pct", 0.010)))
    if chase_pct > max_chase:
        return "15m_chase"

    reversal_pct = (latest / window[-2].close - 1.0) * direction.value if window[-2].close > 0 else 0.0
    max_reversal = max(0.0, float(getattr(strategy, "entry_timing_reversal_pct", 0.006)))
    if reversal_pct < -max_reversal:
        return "15m_reversal"

    fast_rsi = rsi(closes, fast_period)[-1]
    mid_rsi = rsi(closes, mid_period)[-1]
    if direction == Direction.LONG:
        if (
            fast_rsi >= float(getattr(strategy, "entry_timing_long_rsi_fast_ceiling", 82.0))
            and mid_rsi >= float(getattr(strategy, "entry_timing_long_rsi_mid_ceiling", 76.0))
        ):
            return "15m_long_rsi_hot"
    elif direction == Direction.SHORT:
        if (
            fast_rsi <= float(getattr(strategy, "entry_timing_short_rsi_fast_floor", 18.0))
            and mid_rsi <= float(getattr(strategy, "entry_timing_short_rsi_mid_floor", 24.0))
        ):
            return "15m_short_rsi_cold"
    return None


def _execution_timing_guard_reason(
    config: Any,
    candidate: EntryCandidate,
    candles_by_symbol: dict[str, list[Candle]] | None,
    timestamps_by_symbol: dict[str, list[Any]] | None,
    decision_timestamp: Any,
) -> str | None:
    strategy = config.strategy
    if not getattr(strategy, "entry_execution_filter_enabled", False):
        return None
    if getattr(strategy, "entry_execution_filter_trend_only", True) and not _is_trend_entry_reason(candidate.signal.reason):
        return None
    if not candles_by_symbol or not timestamps_by_symbol or decision_timestamp is None:
        return None

    candles = candles_by_symbol.get(candidate.symbol)
    timestamps = timestamps_by_symbol.get(candidate.symbol)
    if not candles or not timestamps:
        return None

    timeframe = str(getattr(strategy, "entry_execution_timeframe", "5m"))
    closed_before = decision_timestamp - timedelta(milliseconds=interval_to_milliseconds(timeframe))
    end = bisect.bisect_right(timestamps, closed_before)
    if end <= 0:
        return None
    fast_period = max(2, int(getattr(strategy, "entry_execution_rsi_fast_period", 6)))
    mid_period = max(fast_period, int(getattr(strategy, "entry_execution_rsi_mid_period", 12)))
    start = max(0, end - max(80, mid_period + 10))
    return _entry_execution_guard_reason_for_candles(config, candidate, candles[start:end])


def _trend_reference_adjusted_signal(config: Any, client: HistoricalClient, symbol: str, signal: Signal) -> Signal | None:
    strategy = config.strategy
    if not getattr(strategy, "trend_reference_filter_enabled", False):
        return signal
    if signal.direction == Direction.FLAT or not _is_trend_entry_reason(signal.reason):
        return signal

    timeframe = str(getattr(strategy, "trend_reference_timeframe", "1h"))
    if timeframe not in client.resampled:
        return signal
    lookback = max(1, int(getattr(strategy, "trend_reference_lookback_bars", 6)))
    limit = max(strategy.slow_ema + lookback + 5, strategy.atr_period + lookback + 5, 120)
    candles = client.klines(symbol, timeframe, limit)
    adjusted, _reason = _trend_reference_adjusted_signal_for_candles(config, signal, candles)
    return adjusted


def _indicator_reference_guard_reason(config: Any, client: HistoricalClient, symbol: str, signal: Signal) -> str | None:
    strategy = config.strategy
    if not getattr(strategy, "indicator_reference_guard_enabled", False):
        return None
    if signal.direction == Direction.FLAT or not signal.reason.lower().startswith("indicator_"):
        return None
    if signal.direction == Direction.LONG and not getattr(strategy, "indicator_reference_guard_long_enabled", True):
        return None
    if signal.direction == Direction.SHORT and not getattr(strategy, "indicator_reference_guard_short_enabled", True):
        return None

    timeframe = str(getattr(strategy, "indicator_reference_timeframe", "1h"))
    if timeframe not in client.resampled:
        return None
    lookback = max(1, int(getattr(strategy, "indicator_reference_lookback_bars", 12)))
    limit = max(
        strategy.slow_ema + lookback + 5,
        strategy.atr_period + lookback + 5,
        config.filters.rsi_period + 5,
        config.filters.kdj_period + 5,
        120,
    )
    candles = client.klines(symbol, timeframe, limit)
    minimum = max(strategy.slow_ema, strategy.atr_period, lookback + 1)
    if len(candles) < minimum:
        return None

    closes = [candle.close for candle in candles]
    slow_values = ema(closes, strategy.slow_ema)
    atr_values = atr(candles, strategy.atr_period)
    rsi_values = rsi(closes, config.filters.rsi_period)
    k_values, d_values, _ = kdj(candles, config.filters.kdj_period)

    current_close = closes[-1]
    current_slow = slow_values[-1]
    atr_value = max(atr_values[-1], 1e-12)
    slope_lookback = min(lookback, len(slow_values) - 1)
    slope_atr = (slow_values[-1] - slow_values[-1 - slope_lookback]) / atr_value
    buffer = max(0.0, float(getattr(strategy, "indicator_reference_buffer_atr", 0.10))) * atr_value
    slope_threshold = max(0.0, float(getattr(strategy, "indicator_reference_slope_atr", 0.05)))
    current_rsi = rsi_values[-1]
    current_k = k_values[-1]
    current_d = d_values[-1]
    extreme_override = bool(getattr(strategy, "indicator_reference_extreme_override_enabled", True))

    if signal.direction == Direction.SHORT:
        overhot = (
            current_rsi >= float(getattr(strategy, "indicator_reference_short_extreme_rsi", 62.0))
            or max(current_k, current_d) >= float(getattr(strategy, "indicator_reference_short_extreme_kdj", 70.0))
        )
        if extreme_override and overhot:
            return None
        if current_close > current_slow + buffer and slope_atr >= slope_threshold:
            return (
                f"{timeframe}_bullish_against_short close={current_close:.6g} "
                f"slow={current_slow:.6g} slope_atr={slope_atr:.2f} "
                f"rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
            )
    elif signal.direction == Direction.LONG:
        oversold = (
            current_rsi <= float(getattr(strategy, "indicator_reference_long_extreme_rsi", 38.0))
            or min(current_k, current_d) <= float(getattr(strategy, "indicator_reference_long_extreme_kdj", 30.0))
        )
        if extreme_override and oversold:
            return None
        if current_close < current_slow - buffer and slope_atr <= -slope_threshold:
            return (
                f"{timeframe}_bearish_against_long close={current_close:.6g} "
                f"slow={current_slow:.6g} slope_atr={slope_atr:.2f} "
                f"rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
            )
    return None


def _weak_market_adjusted_signal(config: Any, signal: Signal, rank_score: float, regime: MarketRegime) -> Signal | None:
    strategy = config.strategy
    if signal.direction != Direction.LONG or not getattr(strategy, "weak_market_long_filter_enabled", False) or not regime.weak:
        return signal

    if getattr(strategy, "weak_market_long_super_volume_only", True) and "super_volume" not in signal.reason:
        return None
    min_rank = max(0.0, float(getattr(strategy, "weak_market_long_min_rank_score", 4.8)))
    if rank_score < min_rank:
        return None

    multiplier = max(0.0, min(1.0, float(getattr(strategy, "weak_market_long_risk_multiplier", 0.55))))
    return Signal(
        signal.direction,
        signal.confidence,
        f"{signal.reason}_weak_market_guard breadth={regime.breadth_pct:.2f} avg={regime.avg_return_pct:.2f}%",
        signal.stop_loss_pct,
        signal.take_profit_pct,
        risk_multiplier=signal.risk_multiplier * multiplier,
        max_holding_bars=signal.max_holding_bars,
    )


def _strong_market_adjusted_signal(config: Any, signal: Signal, rank_score: float, regime: MarketRegime) -> Signal | None:
    strategy = config.strategy
    if signal.direction != Direction.SHORT or not getattr(strategy, "strong_market_short_filter_enabled", False):
        return signal

    strong = (
        regime.breadth_pct >= max(0.0, min(1.0, float(getattr(strategy, "strong_market_breadth_threshold", 0.58))))
        and regime.avg_return_pct >= float(getattr(strategy, "strong_market_avg_return_threshold", 0.006)) * 100.0
    )
    if not strong:
        return signal

    min_rank = max(0.0, float(getattr(strategy, "strong_market_short_min_rank_score", 6.2)))
    is_exceptional = "super_volume" in signal.reason or "startup_breakout" in signal.reason
    if not is_exceptional and rank_score < min_rank:
        return None

    multiplier = max(0.0, min(1.0, float(getattr(strategy, "strong_market_short_risk_multiplier", 0.45))))
    return Signal(
        signal.direction,
        signal.confidence,
        f"{signal.reason}_strong_market_short_guard breadth={regime.breadth_pct:.2f} avg={regime.avg_return_pct:.2f}%",
        signal.stop_loss_pct,
        signal.take_profit_pct,
        risk_multiplier=signal.risk_multiplier * multiplier,
        max_holding_bars=signal.max_holding_bars,
    )


def _btc_market_adjusted_signal(
    config: Any,
    signal: Signal,
    rank_score: float,
    momentum_pct: float,
    volume_ratio: float,
    state: BtcMarketState,
) -> Signal | None:
    strategy = config.strategy
    if not getattr(strategy, "btc_market_filter_enabled", False):
        return signal
    if not _btc_market_filter_applies(config, signal):
        return signal
    if state.direction == Direction.FLAT or state.direction == signal.direction:
        return signal

    is_exceptional = "super_volume" in signal.reason or "startup_breakout" in signal.reason
    if getattr(strategy, "btc_counter_trend_super_volume_only", True) and not is_exceptional:
        return None
    if rank_score < float(getattr(strategy, "btc_counter_trend_min_rank_score", 6.2)):
        return None
    if momentum_pct < float(getattr(strategy, "btc_counter_trend_min_momentum_pct", 0.025)):
        return None
    if volume_ratio < float(getattr(strategy, "btc_counter_trend_min_volume_ratio", 2.2)):
        return None

    multiplier = max(0.0, min(1.0, float(getattr(strategy, "btc_counter_trend_risk_multiplier", 0.45))))
    return replace(
        signal,
        risk_multiplier=signal.risk_multiplier * multiplier,
        reason=f"{signal.reason}_btc_counter_guard {state.reason}",
    )


def _btc_market_filter_applies(config: Any, signal: Signal) -> bool:
    strategy = config.strategy
    if not getattr(strategy, "btc_market_filter_trend_only", True):
        return True
    reason = signal.reason.lower()
    trend_like = any(
        token in reason
        for token in ("breakout", "breakdown", "pullback", "startup_breakout", "super_volume")
    )
    holding_bars = signal.max_holding_bars or config.strategy.max_holding_bars
    long_holding = holding_bars >= max(1, int(getattr(strategy, "btc_market_filter_min_holding_bars", 18)))
    return trend_like or long_holding


def _load_symbol_data(
    data_dir: str,
    symbols: tuple[str, ...],
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, list[Candle]]:
    root = Path(data_dir)
    loaded = {}
    for symbol in symbols:
        matches = sorted(root.glob(f"{symbol}_{timeframe}_*.csv"))
        if matches:
            loaded[symbol] = load_candles_csv(matches[-1], start=start, end=end)
    return loaded


@dataclass
class MacroBacktestState:
    enabled: bool
    events: list[MacroEvent]
    candles: list[Candle]
    timestamps: list[Any]
    traded_ids: set[str]


def _build_macro_backtest_state(config: Any, macro_data_dir: str | None) -> MacroBacktestState:
    macro = getattr(config, "macro_events", None)
    if macro is None or not macro.enabled:
        return MacroBacktestState(False, [], [], [], set())
    try:
        events = [event for event in load_macro_events(macro.events_path) if event.event_type in set(macro.event_types)]
    except FileNotFoundError:
        return MacroBacktestState(False, [], [], [], set())

    root = Path(macro_data_dir or "data/binance_1m_365d_macro")
    matches = sorted(root.glob(f"{macro.primary_symbol}_1m_*.csv"))
    if not matches:
        return MacroBacktestState(False, [], [], [], set())
    candles = load_candles_csv(matches[-1])
    return MacroBacktestState(True, events, candles, [candle.timestamp for candle in candles], set())


def _macro_event_for_bar(macro_state: MacroBacktestState, timestamp: Any, bar_seconds: float) -> MacroEvent | None:
    if not macro_state.enabled:
        return None
    bar_end = timestamp.timestamp() + bar_seconds
    for event in macro_state.events:
        event_seconds = event.timestamp.timestamp()
        if timestamp.timestamp() <= event_seconds < bar_end:
            event_id = _macro_event_id(event)
            if event_id not in macro_state.traded_ids:
                return event
    return None


def _macro_lockout_active(macro_state: MacroBacktestState, timestamp: Any) -> bool:
    if not macro_state.enabled:
        return False
    current = timestamp.timestamp()
    for event in macro_state.events:
        if event.timestamp.timestamp() <= current < event.timestamp.timestamp() + 3600:
            return True
    return False


def _macro_event_id(event: MacroEvent) -> str:
    return f"{event.event_type}:{event.reference}:{event.timestamp.isoformat()}"


def _execute_macro_event_trade(
    config: Any,
    cash: float,
    trades: list[dict[str, Any]],
    monthly_stats: dict[str, dict[str, Any]],
    macro_state: MacroBacktestState,
    event: MacroEvent,
) -> float:
    if not macro_state.enabled:
        return cash
    event_id = _macro_event_id(event)
    if event_id in macro_state.traded_ids:
        return cash
    macro_state.traded_ids.add(event_id)

    direction = _macro_event_direction(config, event)
    if direction == Direction.FLAT:
        return cash
    entry_time = event.timestamp.timestamp() + max(0, config.macro_events.post_event_entry_delay_seconds)
    entry_index = bisect.bisect_left([timestamp.timestamp() for timestamp in macro_state.timestamps], entry_time)
    if entry_index >= len(macro_state.candles):
        return cash
    expected = event.timestamp.timestamp() + max(0, config.macro_events.post_event_entry_delay_seconds)
    if abs(macro_state.candles[entry_index].timestamp.timestamp() - expected) > 300:
        return cash

    entry_candle = macro_state.candles[entry_index]
    leverage = max(1, int(config.macro_events.leverage))
    notional = cash * max(0.0, config.macro_events.margin_pct) * leverage
    if config.risk.max_position_notional_usdt > 0:
        notional = min(notional, config.risk.max_position_notional_usdt)
    if notional < config.risk.min_order_notional_usdt:
        return cash

    execution_config = execution_config_from_live_config(config)
    rules = SymbolRules(config.macro_events.primary_symbol, "0.001", "0.001", "0.01", "5")
    entry_fill = market_entry_fill(execution_config, rules, direction, notional / entry_candle.open, entry_candle.open)
    entry_price = entry_fill.price
    quantity = notional / entry_price
    entry_fill = market_entry_fill(execution_config, rules, direction, quantity, entry_candle.open)
    entry_price = entry_fill.price
    entry_fee = entry_fill.fee
    cash -= entry_fee
    if direction == Direction.LONG:
        stop_price = entry_price * (1.0 - config.macro_events.stop_loss_pct)
        take_profit_price = entry_price * (1.0 + config.macro_events.take_profit_pct)
    else:
        stop_price = entry_price * (1.0 + config.macro_events.stop_loss_pct)
        take_profit_price = entry_price * (1.0 - config.macro_events.take_profit_pct)

    max_minutes = max(1, math.ceil(config.macro_events.max_holding_seconds / 60.0))
    exit_index = min(len(macro_state.candles) - 1, entry_index + max_minutes)
    exit_price = macro_state.candles[exit_index].close
    exit_reason = "macro_event_time_stop"
    mfe = 0.0
    mae = 0.0
    for scan_index in range(entry_index, exit_index + 1):
        candle = macro_state.candles[scan_index]
        if direction == Direction.LONG:
            mfe = max(mfe, max(0.0, (candle.high - entry_price) * quantity))
            mae = min(mae, min(0.0, (candle.low - entry_price) * quantity))
        else:
            mfe = max(mfe, max(0.0, (entry_price - candle.low) * quantity))
            mae = min(mae, min(0.0, (entry_price - candle.high) * quantity))
        if direction == Direction.LONG:
            if candle.low <= stop_price:
                exit_index = scan_index
                exit_price = stop_price
                exit_reason = "macro_event_stop_loss"
                break
            if candle.high >= take_profit_price:
                exit_index = scan_index
                exit_price = take_profit_price
                exit_reason = "macro_event_take_profit"
                break
        else:
            if candle.high >= stop_price:
                exit_index = scan_index
                exit_price = stop_price
                exit_reason = "macro_event_stop_loss"
                break
            if candle.low <= take_profit_price:
                exit_index = scan_index
                exit_price = take_profit_price
                exit_reason = "macro_event_take_profit"
                break

    exit_fill = market_exit_fill(execution_config, rules, direction, quantity, exit_price, _exit_order_type(exit_reason))
    executed_exit = exit_fill.price
    raw_gross_pnl = direction.value * quantity * (exit_price - entry_candle.open)
    execution_gross_pnl = direction.value * quantity * (executed_exit - entry_price)
    fee = entry_fee + exit_fill.fee
    slippage_cost = entry_fill.slippage_cost + exit_fill.slippage_cost
    funding = 0.0
    net_pnl = raw_gross_pnl - fee - slippage_cost + funding
    if "cross_sectional_momentum_ignition_pyramid" in str(position.entry_reason):
        position.cmipr_executable_mfe_usdt = max(position.cmipr_executable_mfe_usdt, net_pnl)
        position.cmipr_executable_mae_usdt = min(position.cmipr_executable_mae_usdt, net_pnl)
    cash += execution_gross_pnl - exit_fill.fee + funding
    exit_candle = macro_state.candles[exit_index]
    hold_minutes = _hold_minutes(entry_candle.timestamp, exit_candle.timestamp)
    entry_reason = (
        f"macro_event_{event.event_type} reference={event.reference} "
        f"actual={event.actual:g}{event.unit} forecast={event.forecast:g}{event.unit}"
    )
    _record_monthly_open(monthly_stats, entry_candle.timestamp, direction)
    trades.append(
        {
            "symbol": config.macro_events.primary_symbol,
            "strategy": "macro_event",
            "side": direction.name,
            "direction": direction.name,
            "entry_time": entry_candle.timestamp.isoformat(),
            "exit_time": exit_candle.timestamp.isoformat(),
            "entry_price": entry_price,
            "exit_price": executed_exit,
            "raw_entry_price": entry_candle.open,
            "raw_exit_price": exit_price,
            "qty": quantity,
            "quantity": quantity,
            "notional": abs(quantity * entry_price),
            "entry_fee": entry_fee,
            "exit_fee": exit_fill.fee,
            "fee": fee,
            "fees": fee,
            "gross_pnl": raw_gross_pnl,
            "execution_gross_pnl": execution_gross_pnl,
            "slippage_cost": slippage_cost,
            "funding": funding,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / max(notional, 1e-12),
            "net_bps": net_pnl / max(notional, 1e-12) * 10_000.0,
            "entry_reason": entry_reason,
            "strategy_bucket": "macro_event",
            "reason": exit_reason,
            "exit_reason": exit_reason,
            "mfe": mfe,
            "mae": mae,
            "hold_minutes": hold_minutes,
            "entry_order_type": entry_fill.order_type,
            "exit_order_type": exit_fill.order_type,
            "entry_liquidity": entry_fill.liquidity,
            "exit_liquidity": exit_fill.liquidity,
            "signal_time": event.timestamp.isoformat(),
            "signal_available_time": event.timestamp.isoformat(),
            "skip_reason": "",
            "bars": exit_index - entry_index,
            "scale_ins": 0,
        }
    )
    _record_monthly_equity(monthly_stats, exit_candle.timestamp, cash)
    return cash


def _macro_event_direction(config: Any, event: MacroEvent) -> Direction:
    surprise = event.actual - event.forecast
    if event.event_type == "NFP":
        threshold = abs(config.macro_events.nfp_min_surprise_k)
    elif event.event_type == "CPI_YOY":
        threshold = abs(config.macro_events.cpi_min_surprise_pct)
    else:
        return Direction.FLAT
    if abs(surprise) < threshold:
        return Direction.FLAT
    return Direction.SHORT if surprise > 0 else Direction.LONG


def _resample(candles: list[Candle], factor: int) -> list[Candle]:
    if factor <= 0:
        raise ValueError("resample factor must be positive")
    output = []
    for start in range(0, len(candles), factor):
        chunk = candles[start:start + factor]
        if len(chunk) < factor:
            break
        output.append(
            Candle(
                timestamp=chunk[-1].timestamp,
                open=chunk[0].open,
                high=max(candle.high for candle in chunk),
                low=min(candle.low for candle in chunk),
                close=chunk[-1].close,
                volume=sum(candle.volume for candle in chunk),
            )
        )
    return output


def _resample_factor(base_timeframe: str, target_timeframe: str) -> int:
    base_ms = interval_to_milliseconds(base_timeframe)
    target_ms = interval_to_milliseconds(target_timeframe)
    if target_ms < base_ms:
        raise ValueError(f"{target_timeframe} is shorter than base timeframe {base_timeframe}")
    factor, remainder = divmod(target_ms, base_ms)
    if remainder:
        raise ValueError(f"{target_timeframe} is not an even multiple of {base_timeframe}")
    return factor


def _mark_equity(cash: float, positions: dict[str, PortfolioPosition], candles_by_symbol: dict[str, list[Candle]], index: int) -> float:
    equity = cash
    for symbol, position in positions.items():
        equity += position.unrealized_pnl(candles_by_symbol[symbol][index].close)
    return equity


def _soft_drawdown_stop_hit(config: Any, equity: float, peak_equity: float) -> bool:
    if config.risk.soft_drawdown_min_size_multiplier > 0:
        return False
    stop_at = max(0.0, config.risk.soft_drawdown_stop_pct)
    if stop_at <= 0 or peak_equity <= 0:
        return False
    return equity <= peak_equity * (1.0 - stop_at)


def _starting_capital_drawdown_stop_hit(config: Any, starting_equity: float, equity: float) -> bool:
    stop_at = max(0.0, getattr(config.risk, "starting_capital_drawdown_stop_pct", config.risk.max_drawdown_pct))
    if stop_at <= 0 or starting_equity <= 0:
        return False
    return equity <= starting_equity * (1.0 - stop_at)


def _weekly_profit_drawdown_stop_hit(
    config: Any,
    starting_equity: float,
    week_start_equity: float,
    week_peak_equity: float,
    equity: float,
) -> bool:
    stop_at = max(0.0, getattr(config.risk, "weekly_profit_drawdown_stop_pct", 0.0))
    if stop_at <= 0 or week_peak_equity <= 0:
        return False
    if week_peak_equity <= starting_equity and week_start_equity <= starting_equity:
        return False
    return equity <= week_peak_equity * (1.0 - stop_at)


def _week_key(timestamp: Any) -> tuple[int, int]:
    iso = timestamp.isocalendar()
    return int(iso[0]), int(iso[1])


def _month_key(timestamp: Any) -> str:
    return f"{timestamp.year:04d}-{timestamp.month:02d}"


def _new_monthly_side() -> dict[str, float]:
    return {
        "opened": 0.0,
        "closed": 0.0,
        "wins": 0.0,
        "losses": 0.0,
        "pnl": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
    }


def _new_monthly_stats(equity: float) -> dict[str, Any]:
    return {
        "start_equity": equity,
        "end_equity": equity,
        "opened": 0.0,
        "closed": 0.0,
        "long": _new_monthly_side(),
        "short": _new_monthly_side(),
    }


def _monthly_bucket(monthly_stats: dict[str, dict[str, Any]], timestamp: Any, equity: float = 0.0) -> dict[str, Any]:
    return monthly_stats.setdefault(_month_key(timestamp), _new_monthly_stats(equity))


def _record_monthly_equity(monthly_stats: dict[str, dict[str, Any]], timestamp: Any, equity: float) -> None:
    stats = _monthly_bucket(monthly_stats, timestamp, equity)
    if stats["start_equity"] == 0.0 and stats["end_equity"] == 0.0:
        stats["start_equity"] = equity
    stats["end_equity"] = equity


def _record_monthly_open(monthly_stats: dict[str, dict[str, Any]], timestamp: Any, direction: Direction) -> None:
    stats = _monthly_bucket(monthly_stats, timestamp)
    side = "long" if direction == Direction.LONG else "short"
    stats["opened"] += 1
    stats[side]["opened"] += 1


def _record_monthly_closed_trades(monthly_stats: dict[str, dict[str, Any]], timestamp: Any, closed_trades: list[dict[str, Any]]) -> None:
    if not closed_trades:
        return
    stats = _monthly_bucket(monthly_stats, timestamp)
    for trade in closed_trades:
        side = "long" if trade["direction"] == "LONG" else "short"
        net_pnl = float(trade["net_pnl"])
        side_stats = stats[side]
        stats["closed"] += 1
        side_stats["closed"] += 1
        side_stats["pnl"] += net_pnl
        if net_pnl > 0:
            side_stats["wins"] += 1
            side_stats["gross_profit"] += net_pnl
        else:
            side_stats["losses"] += 1
            side_stats["gross_loss"] += abs(net_pnl)


def _indicator_reversal_pause_active(config: Any, index: int, pause_until: int) -> bool:
    if not getattr(config.strategy, "indicator_reversal_loss_pause_enabled", False):
        return False
    return index < pause_until


def _update_indicator_reversal_pause_state(
    config: Any,
    loss_streak: int,
    pause_until: int,
    closed_trades: list[dict[str, Any]],
    index: int,
) -> tuple[int, int]:
    if not getattr(config.strategy, "indicator_reversal_loss_pause_enabled", False):
        return loss_streak, pause_until

    trigger_losses = max(1, int(getattr(config.strategy, "indicator_reversal_loss_pause_losses", 2)))
    pause_bars = max(1, int(getattr(config.strategy, "indicator_reversal_loss_pause_bars", 8)))
    for trade in closed_trades:
        bucket = str(trade.get("strategy_bucket", _strategy_bucket(str(trade.get("entry_reason", "")))))
        if bucket != "indicator_reversal":
            continue
        net_pnl = float(trade.get("net_pnl", 0.0))
        if net_pnl > 0:
            loss_streak = 0
            continue
        loss_streak += 1
        if loss_streak >= trigger_losses:
            pause_until = max(pause_until, index + pause_bars)
            loss_streak = 0
    return loss_streak, pause_until


def _manage_positions(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    reentry_block_until: dict[str, int],
    trades: list[dict[str, Any]],
    candles_by_symbol: dict[str, list[Candle]],
    index: int,
    client: HistoricalClient,
    signal_cache: dict[str, list[Signal]],
    reversal_cache: dict[str, list[Signal]],
    mtf_cache: dict[tuple[str, str], list[TimeframeSignal | None]],
    market_regime_cache: list[MarketRegime],
    btc_market_state_cache: list[BtcMarketState],
) -> float:
    for symbol in list(positions):
        position = positions[symbol]
        if index <= position.entry_index:
            continue
        candle = candles_by_symbol[symbol][index]
        position.bars_held += 1
        _update_position_excursion(position, candle)
        exit_price = None
        reason = ""
        if position.direction == Direction.LONG:
            if candle.low <= position.stop_price:
                exit_price = position.stop_price
                reason = "stop_loss"
            elif candle.high >= position.take_profit_price:
                exit_price = position.take_profit_price
                reason = "take_profit"
        else:
            if candle.high >= position.stop_price:
                exit_price = position.stop_price
                reason = "stop_loss"
            elif candle.low <= position.take_profit_price:
                exit_price = position.take_profit_price
                reason = "take_profit"

        if exit_price is None and trader._managed_exit_allowed(position):
            _update_profit_protection(trader, config, position, candle)
            account = _account_snapshot(config, _mark_equity(cash, positions, candles_by_symbol, index), positions, candles_by_symbol, index)
            sim = SimPosition(
                symbol=position.symbol,
                direction=position.direction,
                quantity=position.quantity,
                entry_price=position.entry_price,
                stop_price=position.stop_price,
                take_profit_price=position.take_profit_price,
                max_holding_bars=position.max_holding_bars,
                entry_time=candle.timestamp,
                last_checked_time=candle.timestamp,
                best_price=position.best_price,
                bars_held=position.bars_held,
                entry_reason=position.entry_reason,
            )
            profit_reason = trader._profit_exit_reason(sim, candles_by_symbol[symbol][:index + 1], current_candle=candle)
            if profit_reason:
                exit_price = candle.close
                reason = profit_reason
            else:
                account_loss_reason = trader._account_loss_exit_reason(sim, candle.close, account)
                if account_loss_reason:
                    exit_price = candle.close
                    reason = account_loss_reason
                else:
                    fail_fast_reason = trader._indicator_long_fail_fast_exit_reason(sim, candle.close)
                    if fail_fast_reason:
                        exit_price = candle.close
                        reason = fail_fast_reason
                    else:
                        trend_loss_reason = trader._trend_loss_exit_reason(sim, candle.close)
                        if trend_loss_reason:
                            exit_price = candle.close
                            reason = trend_loss_reason
                        else:
                            stale_reason = trader._stale_position_exit_reason(sim, candle.close)
                            if stale_reason:
                                exit_price = candle.close
                                reason = stale_reason

        if exit_price is None and position.max_holding_bars > 0 and position.bars_held >= position.max_holding_bars:
            exit_price = candle.close
            reason = "time_stop"
        if exit_price is not None:
            cash = _close_position(config, cash, positions, trades, symbol, exit_price, reason, index, candle.timestamp)
            _mark_reentry_cooldown(config, reentry_block_until, symbol, index)
            continue

        cash = _maybe_scale_in_position(
            trader,
            config,
            cash,
            positions,
            symbol,
            candles_by_symbol,
            index,
            client,
            signal_cache,
            reversal_cache,
            mtf_cache,
            market_regime_cache,
            btc_market_state_cache,
        )
    return cash


def _mark_reentry_cooldown(config: Any, reentry_block_until: dict[str, int], symbol: str, index: int) -> None:
    cooldown_seconds = max(0, config.trading.symbol_reentry_cooldown_seconds)
    if cooldown_seconds <= 0:
        return
    cooldown_bars = max(1, math.ceil(cooldown_seconds / _interval_seconds(config.trading.timeframe)))
    reentry_block_until[symbol] = max(reentry_block_until.get(symbol, -1), index + cooldown_bars)


def _interval_seconds(timeframe: str) -> float:
    return interval_to_milliseconds(timeframe) / 1000.0


def _maybe_scale_in_position(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    symbol: str,
    candles_by_symbol: dict[str, list[Candle]],
    index: int,
    client: HistoricalClient,
    signal_cache: dict[str, list[Signal]],
    reversal_cache: dict[str, list[Signal]],
    mtf_cache: dict[tuple[str, str], list[TimeframeSignal | None]],
    market_regime_cache: list[MarketRegime],
    btc_market_state_cache: list[BtcMarketState],
) -> float:
    position = positions[symbol]
    trading = config.trading
    if trading.max_scale_ins_per_symbol <= 0 or trading.scale_in_entry_fraction <= 0:
        return cash
    if position.scale_ins >= trading.max_scale_ins_per_symbol:
        return cash

    candle = candles_by_symbol[symbol][index]
    notional = abs(position.quantity * candle.close)
    if notional <= 0:
        return cash
    if str(getattr(trading, "scale_in_profit_basis", "gross")).lower() == "full_cost":
        profit_pct = _position_executable_net_profit_pct(config, position, candle, client.symbol_rules(symbol))
    else:
        profit_pct = position.unrealized_pnl(candle.close) / notional
    if profit_pct < trading.scale_in_min_profit_pct:
        return cash

    candles = candles_by_symbol[symbol][:index + 1]
    is_mtf_position = MTF_REASON_TOKEN in str(position.entry_reason)
    if is_mtf_position:
        signal = trader._continuation_signal(
            position.direction,
            candles,
            "mtf_winner_addon_continuation",
        )
    else:
        candidate = _cached_entry_candidate(
            trader,
            config,
            client,
            candles_by_symbol,
            signal_cache,
            reversal_cache,
            mtf_cache,
            market_regime_cache,
            btc_market_state_cache,
            symbol,
            index,
        )
        if candidate is not None and candidate.signal.direction == position.direction:
            signal = candidate.signal
        else:
            supported, support_reason = _cached_mtf_allows(trader, config, client, mtf_cache, symbol, index, position.direction)
            if not supported:
                return cash
            signal = trader._continuation_signal(
                position.direction,
                candles,
                f"portfolio_profit_continuation mtf={support_reason}",
            )

    if not is_mtf_position and signal.direction != position.direction:
        supported, support_reason = _cached_mtf_allows(trader, config, client, mtf_cache, symbol, index, position.direction)
        if not supported:
            return cash
        signal = trader._continuation_signal(
            position.direction,
            candles,
            f"portfolio_profit_continuation mtf={support_reason}",
        )

    confirmed, _reason = trader._scale_in_confirmation_reason(position.direction, candles, loss_scale=False)
    if not confirmed:
        return cash

    signal_quality = max(0.25, min(1.0, signal.risk_multiplier))
    entry_fraction = _scale_fraction(trading.scale_in_entry_fraction, position.scale_ins) * signal_quality
    account = _account_snapshot(config, _mark_equity(cash, positions, candles_by_symbol, index), positions, candles_by_symbol, index)
    existing_position = account.positions.get(symbol)
    if existing_position is None:
        return cash

    quantity_text, reason = trader._size_order(
        symbol,
        candle.close,
        signal,
        account,
        existing_position=existing_position,
        entry_fraction=entry_fraction,
    )
    quantity = float(quantity_text)
    if reason != "ok" or quantity <= 0:
        return cash
    return _add_to_position(
        config,
        cash,
        position,
        signal,
        candle,
        quantity,
        rules=client.symbol_rules(symbol),
    )


def _position_executable_net_profit_pct(
    config: Any,
    position: PortfolioPosition,
    candle: Candle,
    rules: SymbolRules,
) -> float:
    notional = abs(position.quantity * position.entry_price)
    if notional <= 0.0:
        return 0.0
    execution_config = execution_config_from_live_config(config)
    bar_quote_volume = abs(candle.volume * candle.close)
    fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        position.quantity,
        candle.close,
        "market",
        bar_quote_volume,
    )
    execution_gross = position.direction.value * position.quantity * (fill.price - position.entry_price)
    funding = _funding_for_position(execution_config, position, candle.timestamp)
    return (execution_gross - position.entry_fee - fill.fee + funding) / notional


def _update_profit_protection(trader: BinanceAutoTrader, config: Any, position: PortfolioPosition, candle: Candle) -> None:
    previous_stop = position.stop_price
    lock_pct = max(config.trading.breakeven_lock_pct, trader._minimum_profit_exit_pct())
    if position.direction == Direction.LONG:
        position.best_price = max(position.best_price, candle.high)
        peak_profit = (position.best_price - position.entry_price) / position.entry_price
        if peak_profit >= config.trading.breakeven_trigger_pct:
            position.stop_price = max(position.stop_price, position.entry_price * (1.0 + lock_pct))
        if peak_profit >= config.trading.trailing_activation_pct:
            position.stop_price = max(position.stop_price, position.best_price * (1.0 - config.trading.trailing_pullback_pct))
    else:
        position.best_price = min(position.best_price, candle.low)
        peak_profit = (position.entry_price - position.best_price) / position.entry_price
        if peak_profit >= config.trading.breakeven_trigger_pct:
            position.stop_price = min(position.stop_price, position.entry_price * (1.0 - lock_pct))
        if peak_profit >= config.trading.trailing_activation_pct:
            position.stop_price = min(position.stop_price, position.best_price * (1.0 + config.trading.trailing_pullback_pct))


def _account_snapshot(config: Any, equity: float, positions: dict[str, PortfolioPosition], candles_by_symbol: dict[str, list[Candle]], index: int) -> AccountSnapshot:
    rows = []
    initial_margin = 0.0
    maintenance_margin = 0.0
    total_unrealized = 0.0
    position_map = {}
    for symbol, position in positions.items():
        mark = candles_by_symbol[symbol][index].close
        notional = abs(position.quantity * mark)
        unrealized = position.unrealized_pnl(mark)
        leverage = position.leverage or config.trading.leverage
        initial_margin += notional / max(leverage, 1)
        maintenance_margin += notional * max(
            0.0,
            float(getattr(config.risk, "estimated_maintenance_margin_rate", 0.005)),
        )
        total_unrealized += unrealized
        live = LivePosition(symbol, "BACKTEST", position.direction, position.quantity, position.entry_price, mark, notional, unrealized, leverage, config.trading.margin_type, None)
        rows.append(live)
        position_map[symbol] = live
    return AccountSnapshot(equity, equity, max(0.0, equity - initial_margin), initial_margin, maintenance_margin, total_unrealized, position_map, tuple(rows), "portfolio_backtest")


def _open_position(
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    symbol: str,
    signal: Any,
    candle: Candle,
    quantity: float,
    index: int,
    leverage_override: int | None = None,
    raw_entry_price: float | None = None,
    entry_time: Any = None,
    signal_time: Any = None,
    signal_available_time: Any = None,
    execution_config: BacktestExecutionConfig | None = None,
    rules: SymbolRules | None = None,
) -> float:
    execution_config = execution_config or execution_config_from_live_config(config)
    rules = rules or SymbolRules(symbol, "0.001", "0.001", "0.01", "5")
    raw_price = candle.close if raw_entry_price is None else raw_entry_price
    requested_quantity = quantity
    bar_quote_volume = abs(candle.volume * raw_price)
    quantity, capacity_fill_ratio, _ = capacity_limited_quantity(
        execution_config, rules, quantity, raw_price, bar_quote_volume
    )
    if quantity <= 0:
        return cash
    fill = market_entry_fill(execution_config, rules, signal.direction, quantity, raw_price, bar_quote_volume)
    size_reason = validate_order_size(rules, quantity, fill.price, getattr(config.risk, "min_order_notional_usdt", 0.0))
    if size_reason != "ok":
        return cash
    entry_price = fill.price
    notional = abs(quantity * entry_price)
    entry_fee = fill.fee
    cash -= entry_fee
    if signal.direction == Direction.LONG:
        stop_price = entry_price * (1.0 - signal.stop_loss_pct)
        take_profit_price = entry_price * (1.0 + signal.take_profit_pct)
    else:
        stop_price = entry_price * (1.0 + signal.stop_loss_pct)
        take_profit_price = entry_price * (1.0 - signal.take_profit_pct)
    positions[symbol] = PortfolioPosition(
        symbol,
        signal.direction,
        quantity,
        entry_price,
        stop_price,
        take_profit_price,
        entry_fee,
        index,
        signal.max_holding_bars or config.strategy.max_holding_bars,
        entry_price,
        _effective_entry_leverage(config, symbol, signal, leverage_override),
        signal.reason,
        entry_time=entry_time or candle.timestamp,
        raw_entry_price=fill.raw_price,
        entry_slippage_cost=fill.slippage_cost,
        entry_order_type=fill.order_type,
        entry_liquidity=fill.liquidity,
        signal_time=signal_time,
        signal_available_time=signal_available_time,
        entry_participation_rate=fill.participation_rate,
        capacity_fill_ratio=quantity / requested_quantity if requested_quantity > 0 else capacity_fill_ratio,
        liquidity_reference_quote_volume=bar_quote_volume,
        initial_stop_price=stop_price,
        cmipr_initial_quantity=quantity if "cross_sectional_momentum_ignition_pyramid" in str(signal.reason) else 0.0,
    )
    return cash


def _add_to_position(
    config: Any,
    cash: float,
    position: PortfolioPosition,
    signal: Any,
    candle: Candle,
    quantity: float,
    execution_config: BacktestExecutionConfig | None = None,
    rules: SymbolRules | None = None,
) -> float:
    execution_config = execution_config or execution_config_from_live_config(config)
    rules = rules or SymbolRules(position.symbol, "0.001", "0.001", "0.01", "5")
    requested_quantity = quantity
    bar_quote_volume = abs(candle.volume * candle.close)
    quantity, capacity_fill_ratio, _ = capacity_limited_quantity(
        execution_config, rules, quantity, candle.close, bar_quote_volume
    )
    if quantity <= 0:
        return cash
    fill = market_entry_fill(execution_config, rules, signal.direction, quantity, candle.close, bar_quote_volume)
    size_reason = validate_order_size(rules, quantity, fill.price, getattr(config.risk, "min_order_notional_usdt", 0.0))
    if size_reason != "ok":
        return cash
    entry_price = fill.price
    entry_fee = fill.fee
    cash -= entry_fee
    old_quantity = position.quantity
    old_raw_entry = position.raw_entry_price or position.entry_price
    total_quantity = old_quantity + quantity
    if total_quantity <= 0:
        return cash

    average_entry = (position.entry_price * old_quantity + entry_price * quantity) / total_quantity
    average_raw_entry = (old_raw_entry * old_quantity + fill.raw_price * quantity) / total_quantity
    if signal.direction == Direction.LONG:
        position.stop_price = max(position.stop_price, entry_price * (1.0 - signal.stop_loss_pct))
        position.take_profit_price = min(position.take_profit_price, entry_price * (1.0 + signal.take_profit_pct))
        position.best_price = max(position.best_price, candle.high)
    else:
        position.stop_price = min(position.stop_price, entry_price * (1.0 + signal.stop_loss_pct))
        position.take_profit_price = max(position.take_profit_price, entry_price * (1.0 - signal.take_profit_pct))
        position.best_price = min(position.best_price, candle.low)

    position.quantity = total_quantity
    position.entry_price = average_entry
    position.raw_entry_price = average_raw_entry
    position.entry_fee += entry_fee
    position.entry_slippage_cost += fill.slippage_cost
    position.scale_ins += 1
    position.capacity_fill_ratio = min(position.capacity_fill_ratio, quantity / requested_quantity if requested_quantity > 0 else capacity_fill_ratio)
    position.entry_participation_rate = max(position.entry_participation_rate, fill.participation_rate)
    position.liquidity_reference_quote_volume = max(position.liquidity_reference_quote_volume, bar_quote_volume)
    return cash


def _close_position(
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    trades: list[dict[str, Any]],
    symbol: str,
    exit_price: float,
    reason: str,
    index: int,
    exit_time: Any = None,
    execution_config: BacktestExecutionConfig | None = None,
    rules: SymbolRules | None = None,
) -> float:
    position = positions.pop(symbol)
    execution_config = execution_config or execution_config_from_live_config(config)
    rules = rules or SymbolRules(symbol, "0.001", "0.001", "0.01", "5")
    exit_order_type = _exit_order_type(reason)
    fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        position.quantity,
        exit_price,
        exit_order_type,
        position.liquidity_reference_quote_volume or None,
    )
    executed_exit = fill.price
    raw_entry = position.raw_entry_price or position.entry_price
    raw_gross_pnl = position.direction.value * position.quantity * (exit_price - raw_entry)
    execution_gross_pnl = position.direction.value * position.quantity * (executed_exit - position.entry_price)
    fee = position.entry_fee + fill.fee
    slippage_cost = position.entry_slippage_cost + fill.slippage_cost
    funding = _funding_for_position(execution_config, position, exit_time)
    net_pnl = raw_gross_pnl - fee - slippage_cost + funding
    cash += execution_gross_pnl - fill.fee + funding
    notional = abs(position.quantity * position.entry_price)
    hold_minutes = _hold_minutes(position.entry_time, exit_time)
    strategy_bucket = _strategy_bucket(position.entry_reason)
    trades.append(
        {
            "symbol": symbol,
            "strategy": strategy_bucket,
            "side": position.direction.name,
            "direction": position.direction.name,
            "entry_time": position.entry_time.isoformat() if hasattr(position.entry_time, "isoformat") else position.entry_time,
            "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else exit_time,
            "entry_price": position.entry_price,
            "exit_price": executed_exit,
            "raw_entry_price": raw_entry,
            "raw_exit_price": exit_price,
            "qty": position.quantity,
            "quantity": position.quantity,
            "notional": notional,
            "entry_fee": position.entry_fee,
            "exit_fee": fill.fee,
            "fee": fee,
            "fees": fee,
            "gross_pnl": raw_gross_pnl,
            "execution_gross_pnl": execution_gross_pnl,
            "slippage_cost": slippage_cost,
            "funding": funding,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / max(notional, 1e-12),
            "net_bps": net_pnl / max(notional, 1e-12) * 10_000.0,
            "entry_reason": position.entry_reason,
            "strategy_bucket": strategy_bucket,
            "reason": reason,
            "exit_reason": reason,
            "bars": index - position.entry_index,
            "scale_ins": position.scale_ins,
            "mfe": position.mfe,
            "mae": position.mae,
            "hold_minutes": hold_minutes,
            "avg_hold_minutes": hold_minutes,
            "entry_order_type": position.entry_order_type,
            "exit_order_type": fill.order_type,
            "entry_liquidity": position.entry_liquidity,
            "exit_liquidity": fill.liquidity,
            "signal_time": position.signal_time.isoformat() if hasattr(position.signal_time, "isoformat") else position.signal_time,
            "signal_available_time": position.signal_available_time.isoformat() if hasattr(position.signal_available_time, "isoformat") else position.signal_available_time,
            "skip_reason": "",
            "entry_participation_rate": position.entry_participation_rate,
            "exit_participation_rate": fill.participation_rate,
            "capacity_fill_ratio": position.capacity_fill_ratio,
            "risk_budget_usdt": position.risk_budget_usdt,
            "campaign_risk_budget_usdt": position.campaign_risk_budget_usdt,
            "initial_leg_price_risk_usdt": position.initial_leg_price_risk_usdt,
            "initial_leg_full_cost_risk_usdt": position.initial_leg_full_cost_risk_usdt,
            "initial_leg_actual_risk_fraction": position.initial_leg_actual_risk_fraction,
            "capacity_clipped_initial_risk_fraction": position.capacity_clipped_initial_risk_fraction,
            "stop_execution_price_estimate": position.stop_execution_price_estimate,
            "estimated_stop_exit_fee_usdt": position.estimated_stop_exit_fee_usdt,
            "estimated_stop_exit_slippage_usdt": position.estimated_stop_exit_slippage_usdt,
            "initial_stop_price": position.initial_stop_price,
            "max_executable_r": max(
                position.cmipr_max_campaign_executable_r,
                position.reversal_v2_max_executable_r,
                position.mtper_max_campaign_executable_r,
                position.mtpc_max_executable_r,
            ),
            "max_executable_r_basis": (
                "initial_leg"
                if "multi_timeframe_trend_pullback_continuation" in str(position.entry_reason)
                else
                "campaign"
                if "cross_sectional_momentum_ignition_pyramid" in str(position.entry_reason)
                or "multi_timeframe_pre_cross_exhaustion_reversal" in str(position.entry_reason)
                else "strategy_specific"
            ),
            "max_executable_campaign_r": position.cmipr_max_campaign_executable_r,
            "min_executable_campaign_r": position.cmipr_min_campaign_executable_r,
            "max_executable_initial_leg_r": position.cmipr_max_initial_leg_executable_r,
            "min_executable_initial_leg_r": position.cmipr_min_initial_leg_executable_r,
            "executable_mfe_usdt": position.cmipr_executable_mfe_usdt,
            "executable_mae_usdt": position.cmipr_executable_mae_usdt,
            "mtper_max_executable_campaign_r": position.mtper_max_campaign_executable_r,
            "mtper_min_executable_campaign_r": position.mtper_min_campaign_executable_r,
            "mtper_max_executable_initial_leg_r": position.mtper_max_initial_leg_executable_r,
            "mtper_min_executable_initial_leg_r": position.mtper_min_initial_leg_executable_r,
            "mtper_executable_mfe_usdt": position.mtper_executable_mfe_usdt,
            "mtper_executable_mae_usdt": position.mtper_executable_mae_usdt,
            "mtpc_max_executable_r": position.mtpc_max_executable_r,
            "mtpc_min_executable_r": position.mtpc_min_executable_r,
            "mtpc_max_executable_campaign_r": position.mtpc_max_campaign_executable_r,
            "mtpc_min_executable_campaign_r": position.mtpc_min_campaign_executable_r,
            "mtpc_max_executable_initial_leg_r": position.mtpc_max_initial_leg_executable_r,
            "mtpc_min_executable_initial_leg_r": position.mtpc_min_initial_leg_executable_r,
            "mtpc_executable_mfe_usdt": position.mtpc_executable_mfe_usdt,
            "mtpc_executable_mae_usdt": position.mtpc_executable_mae_usdt,
            "raw_mfe_usdt": position.mfe,
            "raw_mae_usdt": position.mae,
            "net_campaign_r": (
                net_pnl / max(position.campaign_risk_budget_usdt, 1e-12)
                if position.campaign_risk_budget_usdt > 0.0 else None
            ),
            "net_initial_leg_r": (
                net_pnl / max(position.initial_leg_full_cost_risk_usdt, 1e-12)
                if position.initial_leg_full_cost_risk_usdt > 0.0 else None
            ),
            "initial_quantity": position.cmipr_initial_quantity,
            "addon_1_quantity": position.cmipr_addon_1_quantity,
            "addon_2_quantity": position.cmipr_addon_2_quantity,
            "strategy_metadata": dict(position.strategy_metadata or {}),
        }
    )
    return cash


def _exit_order_type(reason: str) -> str:
    normalized = reason.lower()
    if "stop_loss" in normalized or "liquidation" in normalized or normalized in {
        "mtf_breakeven_stop",
        "mtf_giveback_stop",
        "mtf_atr_stop",
    }:
        return "stop_market"
    if "take_profit" in normalized:
        return "take_profit_market"
    return "market"


def _hold_minutes(entry_time: Any, exit_time: Any) -> float:
    if hasattr(entry_time, "timestamp") and hasattr(exit_time, "timestamp"):
        return max(0.0, (exit_time - entry_time).total_seconds() / 60.0)
    return 0.0


def _funding_for_position(execution_config: BacktestExecutionConfig, position: PortfolioPosition, exit_time: Any) -> float:
    if not execution_config.funding_enabled:
        return 0.0
    if not hasattr(position.entry_time, "timestamp") or not hasattr(exit_time, "timestamp"):
        return 0.0
    histories = execution_config.funding_rates_by_symbol or {}
    history = histories.get(position.symbol, ())
    rates = funding_rates_between(history, position.entry_time, exit_time)
    if not rates and execution_config.funding_default_rate:
        rates = [execution_config.funding_default_rate]
    return funding_cashflow(position.direction, abs(position.quantity * position.entry_price), rates)


def _update_position_excursion(position: PortfolioPosition, candle: Candle) -> None:
    if position.direction == Direction.LONG:
        mfe = max(0.0, (candle.high - position.entry_price) * position.quantity)
        mae = min(0.0, (candle.low - position.entry_price) * position.quantity)
    else:
        mfe = max(0.0, (position.entry_price - candle.low) * position.quantity)
        mae = min(0.0, (position.entry_price - candle.high) * position.quantity)
    position.mfe = max(position.mfe, mfe)
    position.mae = min(position.mae, mae)


def _summary(
    starting_equity: float,
    final_equity: float,
    equity_curve: list[float],
    trades: list[dict[str, Any]],
    first: Candle,
    last: Candle,
    monthly_stats: dict[str, dict[str, Any]] | None = None,
    include_trades: bool = False,
    execution_stats: BacktestExecutionStats | None = None,
    equity_timeline: list[tuple[Any, float]] | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    wins = [trade for trade in trades if trade["net_pnl"] > 0]
    losses = [trade for trade in trades if trade["net_pnl"] <= 0]
    gross_profit = sum(trade["net_pnl"] for trade in wins)
    gross_loss = abs(sum(trade["net_pnl"] for trade in losses))
    drawdown_analysis = _drawdown_analysis(starting_equity, equity_curve, trades, equity_timeline)
    max_drawdown = float(drawdown_analysis["max_drawdown_pct"]) / 100.0
    period_days = (last.timestamp - first.timestamp).total_seconds() / 86_400.0
    net_return = (final_equity / starting_equity - 1.0) * 100.0
    monthly = net_return / (period_days / 30.4375) if period_days > 0 else 0.0
    payload = {
        "initial_equity": starting_equity,
        "final_equity": final_equity,
        "gross_pnl": sum(float(trade.get("gross_pnl", 0.0)) for trade in trades),
        "fee": sum(float(trade.get("fee", trade.get("fees", 0.0))) for trade in trades),
        "slippage_cost": sum(float(trade.get("slippage_cost", 0.0)) for trade in trades),
        "funding": sum(float(trade.get("funding", 0.0)) for trade in trades),
        "net_pnl": sum(float(trade.get("net_pnl", 0.0)) for trade in trades),
        "net_return_pct": net_return,
        "monthly_return_pct": monthly,
        "max_drawdown_pct": max_drawdown * 100.0,
        "trade_count": len(trades),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": 0.0 if not trades else len(wins) / len(trades) * 100.0,
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "avg_trade_pnl": 0.0 if not trades else sum(trade["net_pnl"] for trade in trades) / len(trades),
        "avg_trade_return_pct": 0.0 if not trades else sum(trade["return_pct"] for trade in trades) / len(trades) * 100.0,
        "scale_ins": sum(trade["scale_ins"] for trade in trades),
        "long": _side_summary(trades, "LONG"),
        "short": _side_summary(trades, "SHORT"),
        "strategy_buckets": _strategy_bucket_summary(trades),
        "by_strategy_side": _drawdown_group_summary(trades),
        "entry_reasons": [] if compact else _entry_reason_summary(trades),
        "monthly": _monthly_summary(monthly_stats or {}),
        "same_bar_tp_sl_conflict_count": 0 if execution_stats is None else execution_stats.same_bar_tp_sl_conflict_count,
        "limit_touch_count": 0 if execution_stats is None else execution_stats.limit_touch_count,
        "limit_filled_count": 0 if execution_stats is None else execution_stats.limit_filled_count,
        "drawdown_analysis": drawdown_analysis,
        "enhanced_summary": {} if compact else _enhanced_summary(trades, execution_stats),
    }
    if include_trades:
        payload["trades"] = trades
    return payload


def _drawdown_analysis(
    starting_equity: float,
    equity_curve: list[float],
    trades: list[dict[str, Any]],
    equity_timeline: list[tuple[Any, float]] | None = None,
) -> dict[str, Any]:
    if equity_timeline:
        points = [(timestamp, float(equity)) for timestamp, equity in equity_timeline]
    else:
        points = [(None, float(equity)) for equity in equity_curve]
    if not points:
        points = [(None, starting_equity)]

    peak_time = points[0][0]
    peak_equity = max(starting_equity, points[0][1])
    max_peak_time = peak_time
    max_peak_equity = peak_equity
    trough_time = points[0][0]
    trough_equity = points[0][1]
    max_drawdown = 0.0

    for timestamp, equity in points:
        if equity > peak_equity:
            peak_equity = equity
            peak_time = timestamp
        drawdown = 0.0 if peak_equity <= 0 else (peak_equity - equity) / peak_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_peak_time = peak_time
            max_peak_equity = peak_equity
            trough_time = timestamp
            trough_equity = equity

    interval_trades = _trades_closed_between(trades, max_peak_time, trough_time)
    return {
        "max_drawdown_pct": max_drawdown * 100.0,
        "peak_time": _format_optional_time(max_peak_time),
        "peak_equity": max_peak_equity,
        "trough_time": _format_optional_time(trough_time),
        "trough_equity": trough_equity,
        "drawdown_usdt": max_peak_equity - trough_equity,
        "peak_to_trough_minutes": _minutes_between(max_peak_time, trough_time),
        "closed_trades_between_peak_and_trough": {
            "trade_count": len(interval_trades),
            "net_pnl": sum(float(trade.get("net_pnl", 0.0)) for trade in interval_trades),
            "by_strategy_side": _drawdown_group_summary(interval_trades),
            "worst_trades": _worst_drawdown_trades(interval_trades, limit=10),
        },
    }


def _trades_closed_between(trades: list[dict[str, Any]], start: Any, end: Any) -> list[dict[str, Any]]:
    if start is None or end is None:
        return []
    output = []
    for trade in trades:
        exit_time = _parse_trade_time(trade.get("exit_time"))
        if exit_time is not None and start <= exit_time <= end:
            output.append(trade)
    return output


def _drawdown_group_summary(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"trades": 0.0, "wins": 0.0, "losses": 0.0, "net_pnl": 0.0})
    for trade in trades:
        strategy = str(trade.get("strategy_bucket") or trade.get("strategy") or "")
        side = str(trade.get("direction") or trade.get("side") or "")
        key = (strategy, side)
        net_pnl = float(trade.get("net_pnl", 0.0))
        grouped[key]["trades"] += 1
        grouped[key]["net_pnl"] += net_pnl
        if net_pnl > 0:
            grouped[key]["wins"] += 1
        else:
            grouped[key]["losses"] += 1
    return [
        {
            "strategy": strategy,
            "side": side,
            "trades": int(values["trades"]),
            "wins": int(values["wins"]),
            "losses": int(values["losses"]),
            "win_rate_pct": 0.0 if values["trades"] <= 0 else values["wins"] / values["trades"] * 100.0,
            "net_pnl": values["net_pnl"],
        }
        for (strategy, side), values in sorted(grouped.items(), key=lambda item: item[1]["net_pnl"])
    ]


def _worst_drawdown_trades(trades: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    output = []
    for trade in sorted(trades, key=lambda item: float(item.get("net_pnl", 0.0)))[:limit]:
        output.append(
            {
                "symbol": trade.get("symbol", ""),
                "strategy": trade.get("strategy_bucket") or trade.get("strategy") or "",
                "side": trade.get("direction") or trade.get("side") or "",
                "entry_time": trade.get("entry_time", ""),
                "exit_time": trade.get("exit_time", ""),
                "net_pnl": float(trade.get("net_pnl", 0.0)),
                "exit_reason": trade.get("exit_reason") or trade.get("reason") or "",
                "entry_reason": trade.get("entry_reason", ""),
            }
        )
    return output


def _parse_trade_time(value: Any) -> Any:
    if value is None or hasattr(value, "year"):
        return value
    if isinstance(value, str) and value:
        try:
            return parse_timestamp(value)
        except ValueError:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
    return None


def _format_optional_time(value: Any) -> str:
    if value is None:
        return ""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _minutes_between(start: Any, end: Any) -> float:
    if start is None or end is None or not hasattr(end, "__sub__"):
        return 0.0
    try:
        return max(0.0, (end - start).total_seconds() / 60.0)
    except TypeError:
        return 0.0


def _monthly_summary(monthly_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for month in sorted(monthly_stats):
        stats = monthly_stats[month]
        start_equity = stats["start_equity"]
        end_equity = stats["end_equity"]
        pnl = end_equity - start_equity
        output.append(
            {
                "month": month,
                "start_equity": start_equity,
                "end_equity": end_equity,
                "pnl": pnl,
                "return_pct": 0.0 if start_equity <= 0 else pnl / start_equity * 100.0,
                "opened": int(stats.get("opened", 0.0)),
                "closed": int(stats.get("closed", 0.0)),
                "long": _monthly_side_summary(stats["long"]),
                "short": _monthly_side_summary(stats["short"]),
            }
        )
    return output


def _monthly_side_summary(stats: dict[str, float]) -> dict[str, Any]:
    closed = int(stats["closed"])
    gross_loss = stats["gross_loss"]
    return {
        "opened": int(stats["opened"]),
        "closed": closed,
        "wins": int(stats["wins"]),
        "losses": int(stats["losses"]),
        "pnl": stats["pnl"],
        "win_rate_pct": 0.0 if closed <= 0 else stats["wins"] / closed * 100.0,
        "profit_factor": None if gross_loss == 0 else stats["gross_profit"] / gross_loss,
    }


def _side_summary(trades: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    side = [trade for trade in trades if trade["direction"] == direction]
    wins = [trade for trade in side if trade["net_pnl"] > 0]
    losses = [trade for trade in side if trade["net_pnl"] <= 0]
    gross_profit = sum(trade["net_pnl"] for trade in wins)
    gross_loss = abs(sum(trade["net_pnl"] for trade in losses))
    return {
        "trades": len(side),
        "net_pnl": sum(trade["net_pnl"] for trade in side),
        "win_rate_pct": 0.0 if not side else len(wins) / len(side) * 100.0,
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
    }


def _enhanced_summary(trades: list[dict[str, Any]], execution_stats: BacktestExecutionStats | None = None) -> dict[str, Any]:
    return {
        "overall": _enhanced_group_summary("overall", trades, execution_stats=execution_stats),
        "by_strategy": _group_by(trades, ("strategy",)),
        "by_strategy_side": _group_by(trades, ("strategy", "side")),
        "by_strategy_symbol": _group_by(trades, ("strategy", "symbol")),
        "by_strategy_side_symbol": _group_by(trades, ("strategy", "side", "symbol")),
        "by_hour": _group_by_time(trades, "hour"),
        "by_weekday": _group_by_time(trades, "weekday"),
        "long_only": _enhanced_group_summary("long_only", [trade for trade in trades if _trade_side(trade) == "LONG"]),
        "short_only": _enhanced_group_summary("short_only", [trade for trade in trades if _trade_side(trade) == "SHORT"]),
    }


def _group_by(trades: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for trade in trades:
        key = tuple(str(trade.get(field, "")) for field in fields)
        groups.setdefault(key, []).append(trade)
    output = []
    for key in sorted(groups):
        row = _enhanced_group_summary("|".join(key), groups[key])
        for field, value in zip(fields, key):
            row[field] = value
        output.append(row)
    return output


def _group_by_time(trades: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        parsed = _parse_trade_time(trade.get("entry_time"))
        if parsed is None:
            key = "unknown"
        elif kind == "hour":
            key = f"{parsed.hour:02d}"
        else:
            key = str(parsed.weekday())
        groups.setdefault(key, []).append(trade)
    return [_enhanced_group_summary(key, groups[key]) for key in sorted(groups)]


def _enhanced_group_summary(
    name: str,
    trades: list[dict[str, Any]],
    execution_stats: BacktestExecutionStats | None = None,
) -> dict[str, Any]:
    wins = [trade for trade in trades if float(trade.get("net_pnl", 0.0)) > 0]
    losses = [trade for trade in trades if float(trade.get("net_pnl", 0.0)) <= 0]
    gross_profit = sum(float(trade.get("net_pnl", 0.0)) for trade in wins)
    gross_loss = abs(sum(float(trade.get("net_pnl", 0.0)) for trade in losses))
    avg_win = 0.0 if not wins else gross_profit / len(wins)
    avg_loss = 0.0 if not losses else sum(float(trade.get("net_pnl", 0.0)) for trade in losses) / len(losses)
    hold_minutes = [float(trade.get("hold_minutes", 0.0)) for trade in trades]
    notionals = [float(trade.get("notional", 0.0)) for trade in trades]
    total_notional = sum(notionals)
    total_net = sum(float(trade.get("net_pnl", 0.0)) for trade in trades)
    total_mfe = sum(float(trade.get("mfe", 0.0)) for trade in trades)
    maker_fills = sum(1 for trade in trades for field in ("entry_liquidity", "exit_liquidity") if trade.get(field) == "maker")
    taker_fills = sum(1 for trade in trades for field in ("entry_liquidity", "exit_liquidity") if trade.get(field) == "taker")
    total_fills = max(1, maker_fills + taker_fills)
    return {
        "name": name,
        "strategy": "" if name == "overall" else name,
        "side": "",
        "symbol": "",
        "trade_count": len(trades),
        "trades": len(trades),
        "win_rate": 0.0 if not trades else len(wins) / len(trades) * 100.0,
        "win_rate_pct": 0.0 if not trades else len(wins) / len(trades) * 100.0,
        "gross_pnl": sum(float(trade.get("gross_pnl", 0.0)) for trade in trades),
        "fee": sum(float(trade.get("fee", trade.get("fees", 0.0))) for trade in trades),
        "slippage_cost": sum(float(trade.get("slippage_cost", 0.0)) for trade in trades),
        "funding": sum(float(trade.get("funding", 0.0)) for trade in trades),
        "net_pnl": total_net,
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss_ratio": None if avg_loss == 0 else avg_win / abs(avg_loss),
        "max_win": max((float(trade.get("net_pnl", 0.0)) for trade in trades), default=0.0),
        "max_loss": min((float(trade.get("net_pnl", 0.0)) for trade in trades), default=0.0),
        "avg_hold_minutes": 0.0 if not hold_minutes else sum(hold_minutes) / len(hold_minutes),
        "median_hold_minutes": 0.0 if not hold_minutes else statistics.median(hold_minutes),
        "avg_mfe": 0.0 if not trades else total_mfe / len(trades),
        "avg_mae": 0.0 if not trades else sum(float(trade.get("mae", 0.0)) for trade in trades) / len(trades),
        "mfe_to_final_pnl_ratio": None if total_net == 0 else total_mfe / abs(total_net),
        "same_bar_tp_sl_conflict_count": 0 if execution_stats is None else execution_stats.same_bar_tp_sl_conflict_count,
        "limit_touch_count": 0 if execution_stats is None else execution_stats.limit_touch_count,
        "limit_filled_count": 0 if execution_stats is None else execution_stats.limit_filled_count,
        "maker_fill_rate": maker_fills / total_fills * 100.0,
        "taker_ratio": taker_fills / total_fills * 100.0,
        "net_bps": 0.0 if total_notional <= 0 else total_net / total_notional * 10_000.0,
        "expectancy_per_trade": 0.0 if not trades else total_net / len(trades),
    }


def _trade_side(trade: dict[str, Any]) -> str:
    return str(trade.get("side", trade.get("direction", "")))


def _parse_trade_time(value: Any) -> Any:
    if hasattr(value, "hour"):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _strategy_bucket(entry_reason: str) -> str:
    reason = entry_reason.lower()
    if "multi_timeframe_trend_pullback_continuation" in reason:
        return "multi_timeframe_trend_pullback_continuation"
    if "multi_timeframe_pre_cross_exhaustion_reversal" in reason:
        return "multi_timeframe_pre_cross_exhaustion_reversal"
    if "cross_sectional_momentum_ignition_pyramid" in reason:
        return "cross_sectional_momentum_ignition_pyramid"
    if "vbp_" in reason or "volume_breakout_pullback" in reason:
        return "volume_breakout_pullback"
    if "mtf_4h_rsi_regime_pullback" in reason:
        return "mtf_4h_rsi_regime_pullback"
    if "macro_event" in reason:
        return "macro_event"
    if "fast_breakout" in reason:
        return "fast_breakout"
    if "startup_breakout" in reason:
        return "startup_breakout"
    if "super_volume" in reason:
        return "super_volume_startup"
    if "spike" in reason:
        return "spike_reversal"
    if "indicator_" in reason:
        return "indicator_reversal"
    if "rsi_reversal" in reason:
        return "rsi_reversal"
    if "breakout" in reason or "breakdown" in reason:
        return "breakout"
    if "pullback" in reason:
        return "pullback_reclaim"
    return "other"


def _strategy_bucket_summary(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = sorted({str(trade.get("strategy_bucket", _strategy_bucket(str(trade.get("entry_reason", ""))))) for trade in trades})
    return [_trade_group_summary(bucket, [trade for trade in trades if trade.get("strategy_bucket") == bucket]) for bucket in buckets]


def _entry_reason_summary(trades: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    reasons = sorted({str(trade.get("entry_reason", "")) for trade in trades})
    summaries = [_trade_group_summary(reason, [trade for trade in trades if trade.get("entry_reason") == reason]) for reason in reasons]
    summaries.sort(key=lambda item: abs(float(item["net_pnl"])), reverse=True)
    return summaries[:limit]


def _trade_group_summary(name: str, trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [trade for trade in trades if trade["net_pnl"] > 0]
    losses = [trade for trade in trades if trade["net_pnl"] <= 0]
    gross_profit = sum(trade["net_pnl"] for trade in wins)
    gross_loss = abs(sum(trade["net_pnl"] for trade in losses))
    long_trades = [trade for trade in trades if trade["direction"] == "LONG"]
    short_trades = [trade for trade in trades if trade["direction"] == "SHORT"]
    return {
        "name": name,
        "trades": len(trades),
        "net_pnl": sum(trade["net_pnl"] for trade in trades),
        "win_rate_pct": 0.0 if not trades else len(wins) / len(trades) * 100.0,
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "long_trades": len(long_trades),
        "long_pnl": sum(trade["net_pnl"] for trade in long_trades),
        "short_trades": len(short_trades),
        "short_pnl": sum(trade["net_pnl"] for trade in short_trades),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.live.json")
    parser.add_argument("--data-dir", default="data/binance_15m_90d")
    parser.add_argument("--entry-timing-data-dir", default=None)
    parser.add_argument("--execution-timing-data-dir", default=None)
    parser.add_argument("--initial-equity", type=float, default=None)
    parser.add_argument("--include-trades", action="store_true")
    parser.add_argument("--trade-start", default=None, help="UTC ISO timestamp; data before this is used only for warmup")
    parser.add_argument("--trade-end", default=None, help="UTC ISO timestamp")
    args = parser.parse_args()
    trade_start = parse_timestamp(args.trade_start) if args.trade_start else None
    trade_end = parse_timestamp(args.trade_end) if args.trade_end else None
    print(
        json.dumps(
            run_portfolio_backtest(
                args.config,
                args.data_dir,
                args.initial_equity,
                entry_timing_data_dir=args.entry_timing_data_dir,
                execution_timing_data_dir=args.execution_timing_data_dir,
                include_trades=args.include_trades,
                trade_start=trade_start,
                trade_end=trade_end,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

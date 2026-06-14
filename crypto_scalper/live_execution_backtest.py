from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from .data import interval_to_milliseconds, parse_timestamp
from .live_config import load_live_config
from .live_portfolio_backtest import (
    HistoricalClient,
    PortfolioPosition,
    _account_snapshot,
    _build_btc_market_state_cache,
    _build_indicator_reversal_cache,
    _build_market_regime_cache,
    _build_mtf_cache,
    _build_signal_cache,
    _cached_entry_candidate,
    _cached_fast_breakout_candidate,
    _close_position,
    _load_symbol_data,
    _mark_equity,
    _maybe_scale_in_position,
    _open_position,
    _record_monthly_closed_trades,
    _record_monthly_equity,
    _record_monthly_open,
    _summary,
    _update_position_excursion,
)
from .live_trader import (
    BinanceAutoTrader,
    EntryCandidate,
    SimPosition,
    _candidate_requires_extra_slot,
    _entry_position_limit,
    _super_volume_extra_slot_candidate_allowed,
)
from .models import Candle, Direction, Signal
from .mtf_4h_rsi_regime import (
    MTF_REASON_TOKEN,
    Mtf4hRsiRegimePullbackStrategy,
    closed_candles_for_decision,
    funding_at,
    load_auxiliary_features,
    mtf_report_from_summary,
    oi_change_at,
)
from .risk import BacktestExecutionConfig, BacktestExecutionStats, execution_config_from_live_config


@dataclass
class PendingEntry:
    candidate: Any
    signal_index: int
    signal_time: Any
    signal_available_time: Any
    decision_time: Any
    earliest_execution_index: int


def run_execution_backtest(
    config_path: str,
    execution_data_dir: str,
    initial_equity: float | None = None,
    include_trades: bool = False,
    compact: bool = False,
    progress: bool = False,
    trade_start: Any = None,
    trade_end: Any = None,
    cost_experiment: str | None = None,
    backtest_mode: str | None = None,
    mtf_candidate_cache: dict[tuple[str, Any], EntryCandidate | None] | None = None,
) -> dict[str, Any]:
    config = load_live_config(config_path)
    execution_timeframe = "1m"
    execution_candles = _load_symbol_data(execution_data_dir, tuple(config.trading.symbols), execution_timeframe)
    if not execution_candles:
        raise RuntimeError(f"no 1m execution data loaded from {execution_data_dir}")

    symbols = tuple(symbol for symbol in config.trading.symbols if symbol in execution_candles)
    config = replace(
        config,
        trading=replace(
            config.trading,
            symbols=symbols,
            entry_symbols=tuple(symbol for symbol in (config.trading.entry_symbols or symbols) if symbol in symbols),
        ),
    )
    signal_candles = {
        symbol: _resample_to_timeframe(candles, execution_timeframe, config.trading.timeframe)
        for symbol, candles in execution_candles.items()
        if symbol in symbols
    }
    signal_candles = {symbol: candles for symbol, candles in signal_candles.items() if candles}
    symbols = tuple(symbol for symbol in symbols if symbol in signal_candles)
    config = replace(
        config,
        trading=replace(
            config.trading,
            symbols=symbols,
            entry_symbols=tuple(symbol for symbol in (config.trading.entry_symbols or symbols) if symbol in symbols),
        ),
    )
    if not symbols:
        raise RuntimeError("1m data could not be resampled into signal candles")

    timing_candles = None
    if getattr(config.strategy, "entry_timing_filter_enabled", False):
        timing_candles = {
            symbol: _resample_to_timeframe(candles, execution_timeframe, str(config.strategy.entry_timing_timeframe))
            for symbol, candles in execution_candles.items()
            if symbol in symbols
        }
    execution_timing_candles = None
    if getattr(config.strategy, "entry_execution_filter_enabled", False):
        execution_timing_candles = {
            symbol: _resample_to_timeframe(candles, execution_timeframe, str(config.strategy.entry_execution_timeframe))
            for symbol, candles in execution_candles.items()
            if symbol in symbols
        }
    fast_breakout_candles = None
    if getattr(config.strategy, "fast_breakout_enabled", False):
        fast_breakout_candles = {
            symbol: _resample_to_timeframe(candles, execution_timeframe, str(config.strategy.fast_breakout_timeframe))
            for symbol, candles in execution_candles.items()
            if symbol in symbols
        }
    mtf_candles = None
    if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False):
        mtf_candles = {
            timeframe: {
                symbol: _resample_to_timeframe(candles, execution_timeframe, timeframe)
                for symbol, candles in execution_candles.items()
                if symbol in symbols
            }
            for timeframe in _mtf_required_timeframes(config)
        }
    return run_execution_backtest_config(
        config,
        execution_candles,
        signal_candles,
        initial_equity=initial_equity,
        entry_timing_candles_by_symbol=timing_candles,
        execution_timing_candles_by_symbol=execution_timing_candles,
        fast_breakout_candles_by_symbol=fast_breakout_candles,
        mtf_candles_by_timeframe=mtf_candles,
        include_trades=include_trades,
        compact=compact,
        progress=progress,
        trade_start=trade_start,
        trade_end=trade_end,
        cost_experiment=cost_experiment,
        backtest_mode=backtest_mode,
        mtf_candidate_cache=mtf_candidate_cache,
    )


def run_execution_backtest_config(
    config: Any,
    execution_candles_by_symbol: dict[str, list[Candle]],
    signal_candles_by_symbol: dict[str, list[Candle]],
    initial_equity: float | None = None,
    entry_timing_candles_by_symbol: dict[str, list[Candle]] | None = None,
    execution_timing_candles_by_symbol: dict[str, list[Candle]] | None = None,
    fast_breakout_candles_by_symbol: dict[str, list[Candle]] | None = None,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]] | None = None,
    include_trades: bool = False,
    compact: bool = False,
    progress: bool = False,
    trade_start: Any = None,
    trade_end: Any = None,
    cost_experiment: str | None = None,
    backtest_mode: str | None = None,
    mtf_candidate_cache: dict[tuple[str, Any], EntryCandidate | None] | None = None,
) -> dict[str, Any]:
    symbols = tuple(
        symbol
        for symbol in config.trading.symbols
        if symbol in execution_candles_by_symbol and symbol in signal_candles_by_symbol
    )
    execution_candles_by_symbol = {symbol: execution_candles_by_symbol[symbol] for symbol in symbols}
    signal_candles_by_symbol = {symbol: signal_candles_by_symbol[symbol] for symbol in symbols}
    if entry_timing_candles_by_symbol:
        entry_timing_candles_by_symbol = {
            symbol: entry_timing_candles_by_symbol[symbol]
            for symbol in symbols
            if symbol in entry_timing_candles_by_symbol and entry_timing_candles_by_symbol[symbol]
        }
    else:
        entry_timing_candles_by_symbol = {}
    entry_timing_timestamps_by_symbol = {
        symbol: [candle.timestamp for candle in candles]
        for symbol, candles in entry_timing_candles_by_symbol.items()
    }
    execution_timing_candles_by_symbol = {
        symbol: execution_timing_candles_by_symbol[symbol]
        for symbol in symbols
        if execution_timing_candles_by_symbol and symbol in execution_timing_candles_by_symbol and execution_timing_candles_by_symbol[symbol]
    }
    execution_timing_timestamps_by_symbol = {
        symbol: [candle.timestamp for candle in candles]
        for symbol, candles in execution_timing_candles_by_symbol.items()
    }
    fast_breakout_candles_by_symbol = {
        symbol: fast_breakout_candles_by_symbol[symbol]
        for symbol in symbols
        if fast_breakout_candles_by_symbol and symbol in fast_breakout_candles_by_symbol and fast_breakout_candles_by_symbol[symbol]
    }
    fast_breakout_timestamps_by_symbol = {
        symbol: [candle.timestamp for candle in candles]
        for symbol, candles in fast_breakout_candles_by_symbol.items()
    }
    if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False):
        if mtf_candles_by_timeframe is None:
            mtf_candles_by_timeframe = {
                timeframe: {
                    symbol: _resample_to_timeframe(candles, "1m", timeframe)
                    for symbol, candles in execution_candles_by_symbol.items()
                    if symbol in symbols
                }
                for timeframe in _mtf_required_timeframes(config)
            }
        else:
            mtf_candles_by_timeframe = {
                timeframe: {
                    symbol: candles_by_symbol[symbol]
                    for symbol in symbols
                    if symbol in candles_by_symbol and candles_by_symbol[symbol]
                }
                for timeframe, candles_by_symbol in mtf_candles_by_timeframe.items()
            }
    else:
        mtf_candles_by_timeframe = {}
    mtf_timestamps_by_timeframe = {
        timeframe: {
            symbol: [candle.timestamp for candle in candles]
            for symbol, candles in candles_by_symbol.items()
        }
        for timeframe, candles_by_symbol in mtf_candles_by_timeframe.items()
    }
    mtf_aux_features = (
        load_auxiliary_features(
            symbols,
            str(getattr(config.strategy, "mtf_oi_data_dir", "data/binance_oi_taker_5m")),
            str(getattr(config.strategy, "mtf_funding_data_dir", "data/binance_oi_flush_funding")),
        )
        if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False)
        else {}
    )

    client = HistoricalClient(signal_candles_by_symbol, config.trading.timeframe, tuple(config.filters.timeframes))
    trader = BinanceAutoTrader(config, client)
    execution_config = execution_config_from_live_config(config, cost_experiment=cost_experiment, mode=backtest_mode)
    execution_stats = BacktestExecutionStats()
    common_signal_length = min(len(candles) for candles in signal_candles_by_symbol.values())
    common_execution_length = min(len(candles) for candles in execution_candles_by_symbol.values())
    legacy_disabled = bool(getattr(config.strategy, "mtf_disable_legacy_strategies", False)) and bool(getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False))
    if legacy_disabled:
        flat = Signal(Direction.FLAT, 0.0, "legacy_disabled_for_mtf", 0.0, 0.0)
        signal_cache = {symbol: [flat] * common_signal_length for symbol in signal_candles_by_symbol}
        reversal_cache = {symbol: [flat] * common_signal_length for symbol in signal_candles_by_symbol}
    else:
        signal_cache = _build_signal_cache(config, signal_candles_by_symbol, common_signal_length)
        reversal_cache = _build_indicator_reversal_cache(config, signal_candles_by_symbol, common_signal_length)
    mtf_cache = _build_mtf_cache(config, client, symbols)
    market_regime_cache = _build_market_regime_cache(config, signal_candles_by_symbol, common_signal_length)
    btc_market_state_cache = _build_btc_market_state_cache(config, signal_candles_by_symbol, common_signal_length)

    execution_timestamps = [candle.timestamp for candle in next(iter(execution_candles_by_symbol.values()))]
    signal_timestamps = [candle.timestamp for candle in next(iter(signal_candles_by_symbol.values()))]
    signal_ms = interval_to_milliseconds(config.trading.timeframe)
    warmup = max(config.strategy.slow_ema, config.strategy.channel_period, config.strategy.volume_period, 96) + 8
    first_signal_time = signal_timestamps[min(warmup, len(signal_timestamps) - 1)]
    first_trade_time = first_signal_time + timedelta(milliseconds=signal_ms)
    if trade_start is not None:
        first_trade_time = max(first_trade_time, trade_start)

    starting_equity = initial_equity if initial_equity is not None else config.risk.starting_capital_usdt
    cash = starting_equity
    positions: dict[str, PortfolioPosition] = {}
    trades: list[dict[str, Any]] = []
    monthly_stats: dict[str, dict[str, Any]] = {}
    equity_curve = [starting_equity]
    equity_timeline: list[tuple[Any, float]] = []
    reentry_block_until: dict[str, Any] = {}
    pending_entries: list[PendingEntry] = []
    indicator_reversal_loss_streak = 0
    indicator_reversal_pause_until_time: Any = None
    next_entry_scan_time = first_trade_time
    summary_first_candle = next(iter(execution_candles_by_symbol.values()))[0]
    last_execution_index = 0
    mtf_reject_stats: dict[str, int] = {}
    mtf_daily_entry_counts: dict[Any, int] = {}
    mtf_symbol_cooldown_until: dict[str, Any] = {}
    if mtf_candidate_cache is None:
        mtf_candidate_cache = {}
    progress_started_at = time.monotonic()
    next_progress_at = progress_started_at
    progress_total = max(1, common_execution_length)

    for execution_index in range(common_execution_length):
        timestamp = execution_timestamps[execution_index]
        if progress:
            now = time.monotonic()
            if now >= next_progress_at or execution_index == common_execution_length - 1:
                pct = min(100.0, max(0.0, (execution_index + 1) / progress_total * 100.0))
                elapsed = max(0.0, now - progress_started_at)
                print(
                    f"[backtest] {pct:5.1f}% time={timestamp} "
                    f"trades={len(trades)} open={len(positions)} elapsed={elapsed:,.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                next_progress_at = now + 30.0
        if timestamp < first_trade_time:
            summary_first_candle = next(iter(execution_candles_by_symbol.values()))[execution_index]
            continue
        if trade_end is not None and timestamp > trade_end:
            break
        last_execution_index = execution_index

        if not positions and not pending_entries and timestamp < next_entry_scan_time:
            continue

        signal_index = _closed_signal_index(signal_timestamps, timestamp, signal_ms)
        if signal_index < warmup or signal_index >= common_signal_length:
            continue
        client.current_index = signal_index
        trader._mtf_candle_cache.clear()

        closed_before = len(trades)
        cash = _manage_positions_1m(
            trader,
            config,
            cash,
            positions,
            reentry_block_until,
            trades,
            execution_candles_by_symbol,
            signal_candles_by_symbol,
            execution_index,
            signal_index,
            execution_config,
            execution_stats,
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
        )
        _record_monthly_closed_trades(monthly_stats, timestamp, trades[closed_before:])
        cash = _fill_pending_entries_1m(
            trader,
            config,
            cash,
            positions,
            pending_entries,
            monthly_stats,
            execution_candles_by_symbol,
            execution_index,
            signal_index,
            timestamp,
            client,
            execution_config,
        )
        equity = _mark_equity(cash, positions, execution_candles_by_symbol, execution_index)
        equity_curve.append(equity)
        equity_timeline.append((timestamp, equity))
        _record_monthly_equity(monthly_stats, timestamp, equity)

        if timestamp < next_entry_scan_time:
            continue
        scan_seconds = max(1, int(config.trading.entry_scan_seconds))
        while next_entry_scan_time <= timestamp:
            next_entry_scan_time += timedelta(seconds=scan_seconds)

        closed_before = len(trades)
        cash = _run_scale_ins_1m(
            trader,
            config,
            cash,
            positions,
            execution_candles_by_symbol,
            signal_candles_by_symbol,
            execution_index,
            signal_index,
            client,
            signal_cache,
            reversal_cache,
            mtf_cache,
            market_regime_cache,
            btc_market_state_cache,
        )
        _record_monthly_closed_trades(monthly_stats, timestamp, trades[closed_before:])
        indicator_reversal_loss_streak, pause_signal_index = _update_indicator_reversal_pause_state_by_time(
            config,
            indicator_reversal_loss_streak,
            indicator_reversal_pause_until_time,
            trades[closed_before:],
            timestamp,
        )
        indicator_reversal_pause_until_time = pause_signal_index

        if len(positions) + len(pending_entries) >= _entry_position_limit(config):
            continue
        if _drawdown_stopped(config, starting_equity, equity_curve):
            continue

        candidates = _entry_candidates_for_scan(
            trader,
            config,
            client,
            signal_candles_by_symbol,
            signal_cache,
            reversal_cache,
            mtf_cache,
            market_regime_cache,
            btc_market_state_cache,
            entry_timing_candles_by_symbol,
            entry_timing_timestamps_by_symbol,
            execution_timing_candles_by_symbol,
            execution_timing_timestamps_by_symbol,
            fast_breakout_candles_by_symbol,
            fast_breakout_timestamps_by_symbol,
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
            mtf_aux_features,
            mtf_reject_stats,
            mtf_daily_entry_counts,
            mtf_symbol_cooldown_until,
            mtf_candidate_cache,
            positions,
            reentry_block_until,
            signal_index,
            timestamp,
            indicator_reversal_pause_until_time,
        )
        pending_symbols = {entry.candidate.symbol for entry in pending_entries}
        candidates = [candidate for candidate in candidates if candidate.symbol not in pending_symbols]
        opened = 0
        scan_start_positions = len(positions) + len(pending_entries)
        cycle_slots = min(
            max(0, _entry_position_limit(config) - len(positions) - len(pending_entries)),
            max(1, int(config.trading.max_new_entries_per_cycle)),
        )
        for candidate in candidates:
            if opened >= cycle_slots:
                break
            if _candidate_requires_extra_slot(config, scan_start_positions + opened):
                if not _super_volume_extra_slot_candidate_allowed(config, candidate):
                    continue
            signal_time, signal_available_time = _candidate_signal_times(config, candidate, signal_timestamps[signal_index])
            pending_entries.append(
                PendingEntry(
                    candidate=candidate,
                    signal_index=signal_index,
                    signal_time=signal_time,
                    signal_available_time=signal_available_time,
                    decision_time=timestamp,
                    earliest_execution_index=execution_index + 1,
                )
            )
            if _is_mtf_candidate(candidate):
                mtf_daily_entry_counts[timestamp.date()] = mtf_daily_entry_counts.get(timestamp.date(), 0) + 1
                cooldown_hours = max(0, int(getattr(config.strategy, "mtf_symbol_cooldown_hours", 12)))
                if cooldown_hours > 0:
                    mtf_symbol_cooldown_until[candidate.symbol] = timestamp + timedelta(hours=cooldown_hours)
            opened += 1
        if opened > 0:
            _record_monthly_equity(monthly_stats, timestamp, _mark_equity(cash, positions, execution_candles_by_symbol, execution_index))

    last_index = min(common_execution_length - 1, last_execution_index)
    last_candle = next(iter(execution_candles_by_symbol.values()))[last_index]
    final_signal_index = max(0, _closed_signal_index(signal_timestamps, last_candle.timestamp, signal_ms))
    for symbol in list(positions):
        candle = execution_candles_by_symbol[symbol][last_index]
        closed_before = len(trades)
        cash = _close_position(
            config,
            cash,
            positions,
            trades,
            symbol,
            candle.close,
            "end_of_data",
            final_signal_index,
            candle.timestamp,
            execution_config=execution_config,
            rules=client.symbol_rules(symbol),
        )
        _record_monthly_closed_trades(monthly_stats, candle.timestamp, trades[closed_before:])
    final_equity = cash
    equity_curve.append(final_equity)
    equity_timeline.append((last_candle.timestamp, final_equity))
    _record_monthly_equity(monthly_stats, last_candle.timestamp, final_equity)
    payload = _summary(
        starting_equity,
        final_equity,
        equity_curve,
        trades,
        summary_first_candle,
        last_candle,
        monthly_stats,
        include_trades=include_trades,
        execution_stats=execution_stats,
        equity_timeline=equity_timeline,
        compact=compact,
    )
    if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False):
        mtf_payload = dict(payload)
        mtf_payload["trades"] = trades
        payload["mtf_report"] = mtf_report_from_summary(mtf_payload, mtf_reject_stats)
    return payload


def _fill_pending_entries_1m(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    pending_entries: list[PendingEntry],
    monthly_stats: dict[str, dict[str, Any]],
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
    signal_index: int,
    timestamp: Any,
    client: HistoricalClient,
    execution_config: BacktestExecutionConfig,
) -> float:
    if not pending_entries:
        return cash
    remaining: list[PendingEntry] = []
    filled = 0
    max_fills = max(1, int(config.trading.max_new_entries_per_cycle))
    for pending in pending_entries:
        candidate = pending.candidate
        if pending.earliest_execution_index > execution_index:
            remaining.append(pending)
            continue
        if filled >= max_fills:
            remaining.append(pending)
            continue
        if candidate.symbol in positions:
            continue
        if len(positions) >= _entry_position_limit(config):
            remaining.append(pending)
            continue
        candle = execution_candles_by_symbol[candidate.symbol][execution_index]
        execution_candle = replace(
            candle,
            high=candle.open,
            low=candle.open,
            close=candle.open,
        )
        account = _account_snapshot(
            config,
            _mark_equity(cash, positions, execution_candles_by_symbol, execution_index),
            positions,
            execution_candles_by_symbol,
            execution_index,
        )
        quantity_text, reason = trader._size_order(candidate.symbol, execution_candle.close, candidate.signal, account)
        quantity = float(quantity_text)
        if reason != "ok" or quantity <= 0:
            continue
        rules = client.symbol_rules(candidate.symbol)
        before = candidate.symbol in positions
        cash = _open_position(
            config,
            cash,
            positions,
            candidate.symbol,
            candidate.signal,
            execution_candle,
            quantity,
            pending.signal_index,
            raw_entry_price=candle.open,
            entry_time=timestamp,
            signal_time=pending.signal_time,
            signal_available_time=pending.signal_available_time,
            execution_config=execution_config,
            rules=rules,
        )
        if not before and candidate.symbol in positions:
            _record_monthly_open(monthly_stats, timestamp, candidate.signal.direction)
            filled += 1
    pending_entries[:] = remaining
    return cash


def _candidate_signal_times(config: Any, candidate: Any, fallback_signal_time: Any) -> tuple[Any, Any]:
    if _is_mtf_candidate(candidate):
        signal_time = candidate.candle.timestamp
        trigger_timeframe = _mtf_trigger_timeframe(config)
        return signal_time, signal_time + timedelta(milliseconds=interval_to_milliseconds(trigger_timeframe))
    signal_ms = interval_to_milliseconds(config.trading.timeframe)
    return fallback_signal_time, fallback_signal_time + timedelta(milliseconds=signal_ms)


def _is_mtf_candidate(candidate: Any) -> bool:
    return MTF_REASON_TOKEN in str(candidate.signal.reason)


def _mtf_required_timeframes(config: Any) -> tuple[str, ...]:
    timeframes = {
        _mtf_trigger_timeframe(config),
        "30m",
        "1h",
        "4h",
        _mtf_regime_timeframe(config),
    }
    if getattr(config.strategy, "mtf_secondary_2h_enabled", False):
        timeframes.add("2h")
    return tuple(sorted(timeframes, key=interval_to_milliseconds))


def _mtf_trigger_timeframe(config: Any) -> str:
    return _valid_mtf_timeframe(getattr(config.strategy, "mtf_trigger_timeframe", "15m"), "15m")


def _mtf_regime_timeframe(config: Any) -> str:
    return _valid_mtf_timeframe(getattr(config.strategy, "mtf_regime_timeframe", "4h"), "4h")


def _valid_mtf_timeframe(value: Any, default: str) -> str:
    timeframe = str(value or default).strip().lower()
    try:
        interval_to_milliseconds(timeframe)
    except ValueError:
        return default
    return timeframe


def _closed_signal_index(signal_timestamps: list[Any], timestamp: Any, signal_ms: int) -> int:
    return bisect.bisect_right(signal_timestamps, timestamp - timedelta(milliseconds=signal_ms)) - 1


def _entry_candidates_for_scan(
    trader: BinanceAutoTrader,
    config: Any,
    client: HistoricalClient,
    signal_candles_by_symbol: dict[str, list[Candle]],
    signal_cache: dict[str, list[Any]],
    reversal_cache: dict[str, list[Any]],
    mtf_cache: dict[tuple[str, str], list[Any]],
    market_regime_cache: list[Any],
    btc_market_state_cache: list[Any],
    entry_timing_candles_by_symbol: dict[str, list[Candle]],
    entry_timing_timestamps_by_symbol: dict[str, list[Any]],
    execution_timing_candles_by_symbol: dict[str, list[Candle]],
    execution_timing_timestamps_by_symbol: dict[str, list[Any]],
    fast_breakout_candles_by_symbol: dict[str, list[Candle]],
    fast_breakout_timestamps_by_symbol: dict[str, list[Any]],
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
    mtf_aux_features: dict[str, dict[str, Any]],
    mtf_reject_stats: dict[str, int],
    mtf_daily_entry_counts: dict[Any, int],
    mtf_symbol_cooldown_until: dict[str, Any],
    mtf_candidate_cache: dict[tuple[str, Any], EntryCandidate | None],
    positions: dict[str, PortfolioPosition],
    reentry_block_until: dict[str, Any],
    signal_index: int,
    timestamp: Any,
    indicator_reversal_pause_until_time: Any,
) -> list[Any]:
    candidates = []
    entry_symbols = set(config.trading.entry_symbols or config.trading.symbols)
    legacy_disabled = bool(getattr(config.strategy, "mtf_disable_legacy_strategies", False)) and bool(getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False))
    for symbol in config.trading.symbols:
        if symbol in positions or symbol not in entry_symbols:
            continue
        if reentry_block_until.get(symbol) and timestamp < reentry_block_until[symbol]:
            continue
        candidate = None
        if not legacy_disabled:
            main_signal = signal_cache[symbol][signal_index]
            reversal_signal = reversal_cache[symbol][signal_index]
            normal_signal_available = main_signal.direction != Direction.FLAT or reversal_signal.direction != Direction.FLAT
            if (
                main_signal.direction == Direction.FLAT
                and reversal_signal.direction != Direction.FLAT
                and indicator_reversal_pause_until_time is not None
                and timestamp < indicator_reversal_pause_until_time
            ):
                continue
            if normal_signal_available:
                candidate = _cached_entry_candidate(
                    trader,
                    config,
                    client,
                    signal_candles_by_symbol,
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
            if candidate is None:
                candidate = _cached_fast_breakout_candidate(
                    trader,
                    config,
                    client,
                    mtf_cache,
                    market_regime_cache,
                    btc_market_state_cache,
                    symbol,
                    signal_index,
                    fast_breakout_candles_by_symbol,
                    fast_breakout_timestamps_by_symbol,
                    timestamp,
                    entry_timing_candles_by_symbol=entry_timing_candles_by_symbol,
                    entry_timing_timestamps_by_symbol=entry_timing_timestamps_by_symbol,
                    execution_timing_candles_by_symbol=execution_timing_candles_by_symbol,
                    execution_timing_timestamps_by_symbol=execution_timing_timestamps_by_symbol,
                )
        if candidate is None:
            candidate = _cached_mtf_4h_rsi_candidate(
                trader,
                config,
                symbol,
                timestamp,
                mtf_candles_by_timeframe,
                mtf_timestamps_by_timeframe,
                mtf_aux_features,
                mtf_reject_stats,
                mtf_daily_entry_counts,
                mtf_symbol_cooldown_until,
                mtf_candidate_cache,
                positions,
            )
        if candidate:
            candidates.append(candidate)
    candidates.sort(key=lambda item: item.rank_score, reverse=True)
    return candidates


def _cached_mtf_4h_rsi_candidate(
    trader: BinanceAutoTrader,
    config: Any,
    symbol: str,
    timestamp: Any,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
    mtf_aux_features: dict[str, dict[str, Any]],
    mtf_reject_stats: dict[str, int],
    mtf_daily_entry_counts: dict[Any, int],
    mtf_symbol_cooldown_until: dict[str, Any],
    mtf_candidate_cache: dict[tuple[str, Any], EntryCandidate | None],
    positions: dict[str, PortfolioPosition],
) -> EntryCandidate | None:
    if not getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False):
        return None
    if not _mtf_symbol_allowed(config, symbol):
        _count_mtf_reject(mtf_reject_stats, "mtf_symbol_mode_filtered")
        return None
    if _mtf_open_positions(positions) >= max(1, int(getattr(config.strategy, "mtf_max_open_positions", 1))):
        _count_mtf_reject(mtf_reject_stats, "mtf_max_open_positions")
        return None
    if mtf_symbol_cooldown_until.get(symbol) and timestamp < mtf_symbol_cooldown_until[symbol]:
        _count_mtf_reject(mtf_reject_stats, "mtf_symbol_cooldown")
        return None
    max_daily = max(1, int(getattr(config.strategy, "mtf_max_daily_trades", 2)))
    if mtf_daily_entry_counts.get(timestamp.date(), 0) >= max_daily:
        _count_mtf_reject(mtf_reject_stats, "mtf_daily_trade_limit")
        return None

    trigger_timeframe = _mtf_trigger_timeframe(config)
    regime_timeframe = _mtf_regime_timeframe(config)
    try:
        candles_trigger = _mtf_closed(mtf_candles_by_timeframe, mtf_timestamps_by_timeframe, trigger_timeframe, symbol, timestamp, 180)
        candles_30m = _mtf_closed(mtf_candles_by_timeframe, mtf_timestamps_by_timeframe, "30m", symbol, timestamp, 140)
        candles_1h = _mtf_closed(mtf_candles_by_timeframe, mtf_timestamps_by_timeframe, "1h", symbol, timestamp, 140)
        candles_regime = _mtf_closed(mtf_candles_by_timeframe, mtf_timestamps_by_timeframe, regime_timeframe, symbol, timestamp, 160)
        btc_1h = _mtf_closed(mtf_candles_by_timeframe, mtf_timestamps_by_timeframe, "1h", "BTCUSDT", timestamp, 80)
        btc_4h = _mtf_closed(mtf_candles_by_timeframe, mtf_timestamps_by_timeframe, "4h", "BTCUSDT", timestamp, 80)
    except KeyError:
        _count_mtf_reject(mtf_reject_stats, "mtf_missing_timeframe_data")
        return None
    if not candles_trigger or not candles_30m or not candles_1h or not candles_regime or not btc_1h or not btc_4h:
        _count_mtf_reject(mtf_reject_stats, "mtf_warmup")
        return None
    cache_key = (symbol, trigger_timeframe, candles_trigger[-1].timestamp)
    if cache_key in mtf_candidate_cache:
        return mtf_candidate_cache[cache_key]

    features = mtf_aux_features.get(symbol, {})
    strategy = Mtf4hRsiRegimePullbackStrategy(config)
    decision = strategy.build_signal(
        symbol,
        candles_trigger,
        candles_30m,
        candles_1h,
        candles_regime,
        btc_1h,
        btc_4h,
        oi_change_at(features, timestamp),
        funding_at(features, timestamp, float(getattr(config.risk, "funding_default_rate", 0.0))),
        candles_regime=candles_regime,
        candles_trigger=candles_trigger,
        regime_timeframe=regime_timeframe,
        trigger_timeframe=trigger_timeframe,
    )
    if (
        decision.signal is None
        and getattr(config.strategy, "mtf_secondary_2h_enabled", False)
        and regime_timeframe != "2h"
    ):
        try:
            candles_secondary_regime = _mtf_closed(mtf_candles_by_timeframe, mtf_timestamps_by_timeframe, "2h", symbol, timestamp, 160)
        except KeyError:
            candles_secondary_regime = []
        if candles_secondary_regime:
            decision = strategy.build_signal(
                symbol,
                candles_trigger,
                candles_30m,
                candles_1h,
                candles_secondary_regime,
                btc_1h,
                btc_4h,
                oi_change_at(features, timestamp),
                funding_at(features, timestamp, float(getattr(config.risk, "funding_default_rate", 0.0))),
                candles_regime=candles_secondary_regime,
                candles_trigger=candles_trigger,
                regime_timeframe="2h",
                trigger_timeframe=trigger_timeframe,
            )
    if decision.signal is None or decision.candle is None:
        _count_mtf_reject(mtf_reject_stats, decision.reject_reason or "mtf_no_signal")
        mtf_candidate_cache[cache_key] = None
        return None
    rank_score, momentum_pct, volume_ratio = trader._entry_rank_metrics(decision.signal, decision.rank_candles)
    candidate = EntryCandidate(symbol, decision.signal, decision.candle, rank_score, momentum_pct, volume_ratio, "mtf_4h_rsi_regime")
    mtf_candidate_cache[cache_key] = candidate
    return candidate


def _mtf_closed(
    candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
    timeframe: str,
    symbol: str,
    timestamp: Any,
    limit: int,
) -> list[Candle]:
    return closed_candles_for_decision(
        candles_by_timeframe[timeframe][symbol],
        timestamps_by_timeframe[timeframe][symbol],
        timestamp,
        timeframe,
        limit,
    )


def _count_mtf_reject(stats: dict[str, int], reason: str) -> None:
    if not reason:
        return
    stats[reason] = stats.get(reason, 0) + 1


def _mtf_open_positions(positions: dict[str, PortfolioPosition]) -> int:
    return sum(1 for position in positions.values() if MTF_REASON_TOKEN in str(position.entry_reason))


def _mtf_symbol_allowed(config: Any, symbol: str) -> bool:
    mode = str(getattr(config.strategy, "mtf_symbols_mode", "configured")).lower()
    symbols = tuple(config.trading.symbols)
    if mode == "top30":
        return symbol in set(symbols[:30])
    if mode == "top50":
        return symbol in set(symbols[:50])
    return True


def _manage_positions_1m(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    reentry_block_until: dict[str, Any],
    trades: list[dict[str, Any]],
    execution_candles_by_symbol: dict[str, list[Candle]],
    signal_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
    signal_index: int,
    execution_config: BacktestExecutionConfig,
    execution_stats: BacktestExecutionStats,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]] | None = None,
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]] | None = None,
) -> float:
    for symbol in list(positions):
        position = positions[symbol]
        candle = execution_candles_by_symbol[symbol][execution_index]
        position.bars_held = max(0, signal_index - position.entry_index)
        _update_position_excursion(position, candle)
        exit_price = None
        reason = ""
        if position.direction == Direction.LONG:
            position.best_price = max(position.best_price, candle.high)
            stop_hit = candle.low <= position.stop_price
            take_profit_hit = candle.high >= position.take_profit_price
            if stop_hit and take_profit_hit:
                execution_stats.same_bar_tp_sl_conflict_count += 1
            if stop_hit and (not take_profit_hit or execution_config.mode != "optimistic"):
                exit_price = position.stop_price
                reason = "stop_loss_1m"
            elif take_profit_hit:
                exit_price = position.take_profit_price
                reason = "take_profit_1m"
        else:
            position.best_price = min(position.best_price, candle.low)
            stop_hit = candle.high >= position.stop_price
            take_profit_hit = candle.low <= position.take_profit_price
            if stop_hit and take_profit_hit:
                execution_stats.same_bar_tp_sl_conflict_count += 1
            if stop_hit and (not take_profit_hit or execution_config.mode != "optimistic"):
                exit_price = position.stop_price
                reason = "stop_loss_1m"
            elif take_profit_hit:
                exit_price = position.take_profit_price
                reason = "take_profit_1m"

        if exit_price is None and trader._managed_exit_allowed(position):
            sim = SimPosition(
                symbol=position.symbol,
                direction=position.direction,
                quantity=position.quantity,
                entry_price=position.entry_price,
                stop_price=position.stop_price,
                take_profit_price=position.take_profit_price,
                max_holding_bars=position.max_holding_bars,
                entry_time=position.entry_time,
                last_checked_time=candle.timestamp,
                best_price=position.best_price,
                bars_held=position.bars_held,
                entry_reason=position.entry_reason,
            )
            signal_candles = signal_candles_by_symbol[symbol]
            base_signal_candle = signal_candles[signal_index]
            current_signal_candle = replace(
                base_signal_candle,
                close=candle.close,
                high=max(base_signal_candle.high, candle.high),
                low=min(base_signal_candle.low, candle.low),
            )
            recent_start = max(0, signal_index - 120)
            recent_signal_candles = signal_candles[recent_start:signal_index + 1]
            profit_reason = trader._profit_exit_reason(sim, recent_signal_candles, current_candle=current_signal_candle)
            if profit_reason:
                exit_price = candle.close
                reason = profit_reason
            else:
                account = _account_snapshot(config, _mark_equity(cash, positions, execution_candles_by_symbol, execution_index), positions, execution_candles_by_symbol, execution_index)
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

        if exit_price is None:
            mtf_reason = _mtf_exit_reason_for_position(
                config,
                position,
                candle,
                mtf_candles_by_timeframe or {},
                mtf_timestamps_by_timeframe or {},
            )
            if mtf_reason:
                exit_price = candle.close
                reason = mtf_reason

        if exit_price is None and position.max_holding_bars > 0 and position.bars_held >= position.max_holding_bars:
            exit_price = candle.close
            reason = "time_stop"
        if exit_price is not None:
            cash = _close_position(
                config,
                cash,
                positions,
                trades,
                symbol,
                exit_price,
                reason,
                signal_index,
                candle.timestamp,
                execution_config=execution_config,
                rules=trader.client.symbol_rules(symbol),
            )
            _mark_reentry_cooldown_time(config, reentry_block_until, symbol, candle.timestamp)
    return cash


def _mtf_exit_reason_for_position(
    config: Any,
    position: PortfolioPosition,
    candle: Candle,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
) -> str | None:
    if MTF_REASON_TOKEN not in str(position.entry_reason):
        return None
    hold_minutes = 0.0
    if hasattr(position.entry_time, "timestamp"):
        hold_minutes = max(0.0, (candle.timestamp - position.entry_time).total_seconds() / 60.0)
    strategy = config.strategy

    try:
        one_h = _mtf_closed(mtf_candles_by_timeframe, mtf_timestamps_by_timeframe, "1h", position.symbol, candle.timestamp, 80)
    except KeyError:
        one_h = []
    if one_h and Mtf4hRsiRegimePullbackStrategy(config).one_h_confirm_lost(position.direction, one_h):
        return "mtf_1h_confirm_lost"

    risk_cash = abs(position.entry_price - position.stop_price) * position.quantity
    fail_fast_minutes = max(1, int(getattr(strategy, "mtf_fail_fast_minutes", 90)))
    if hold_minutes >= fail_fast_minutes and risk_cash > 0:
        min_mfe = risk_cash * max(0.0, float(getattr(strategy, "mtf_fail_fast_min_r", 0.5)))
        if position.mfe < min_mfe:
            return "mtf_fail_fast"

    max_holding_minutes = max(1, int(getattr(strategy, "mtf_max_holding_minutes", 720)))
    if hold_minutes >= max_holding_minutes:
        return "mtf_time_stop"
    return None


def _run_scale_ins_1m(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    execution_candles_by_symbol: dict[str, list[Candle]],
    signal_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
    signal_index: int,
    client: HistoricalClient,
    signal_cache: dict[str, list[Any]],
    reversal_cache: dict[str, list[Any]],
    mtf_cache: dict[tuple[str, str], list[Any]],
    market_regime_cache: list[Any],
    btc_market_state_cache: list[Any],
) -> float:
    for symbol in list(positions):
        # Keep scale-in decisions on the 30m signal layer, matching the portfolio
        # backtest; mark-to-market remains on the 1m execution layer.
        cash = _maybe_scale_in_position(
            trader,
            config,
            cash,
            positions,
            symbol,
            signal_candles_by_symbol,
            signal_index,
            client,
            signal_cache,
            reversal_cache,
            mtf_cache,
            market_regime_cache,
            btc_market_state_cache,
        )
    return cash


def _update_indicator_reversal_pause_state_by_time(
    config: Any,
    loss_streak: int,
    pause_until: Any,
    closed_trades: list[dict[str, Any]],
    timestamp: Any,
) -> tuple[int, Any]:
    if not getattr(config.strategy, "indicator_reversal_loss_pause_enabled", False):
        return loss_streak, pause_until
    trigger_losses = max(1, int(getattr(config.strategy, "indicator_reversal_loss_pause_losses", 2)))
    pause_bars = max(1, int(getattr(config.strategy, "indicator_reversal_loss_pause_bars", 8)))
    pause_seconds = pause_bars * interval_to_milliseconds(config.trading.timeframe) / 1000.0
    for trade in closed_trades:
        if str(trade.get("strategy_bucket", "")) != "indicator_reversal":
            continue
        if float(trade.get("net_pnl", 0.0)) > 0:
            loss_streak = 0
            continue
        loss_streak += 1
        if loss_streak >= trigger_losses:
            pause_until = timestamp + timedelta(seconds=pause_seconds)
            loss_streak = 0
    return loss_streak, pause_until


def _mark_reentry_cooldown_time(config: Any, reentry_block_until: dict[str, Any], symbol: str, timestamp: Any) -> None:
    cooldown_seconds = max(0, int(config.trading.symbol_reentry_cooldown_seconds))
    if cooldown_seconds <= 0:
        return
    reentry_block_until[symbol] = max(
        reentry_block_until.get(symbol, timestamp),
        timestamp + timedelta(seconds=cooldown_seconds),
    )


def _drawdown_stopped(config: Any, starting_equity: float, equity_curve: list[float]) -> bool:
    if not equity_curve:
        return False
    equity = equity_curve[-1]
    peak = max(equity_curve)
    soft_stop = max(0.0, config.risk.soft_drawdown_stop_pct)
    if soft_stop > 0 and peak > 0 and equity <= peak * (1.0 - soft_stop):
        return True
    starting_stop = max(0.0, getattr(config.risk, "starting_capital_drawdown_stop_pct", config.risk.max_drawdown_pct))
    return starting_stop > 0 and starting_equity > 0 and equity <= starting_equity * (1.0 - starting_stop)


def _resample_to_timeframe(candles: list[Candle], base_timeframe: str, target_timeframe: str) -> list[Candle]:
    base_ms = interval_to_milliseconds(base_timeframe)
    target_ms = interval_to_milliseconds(target_timeframe)
    if target_ms < base_ms or target_ms % base_ms:
        raise ValueError(f"{target_timeframe} is not an even multiple of {base_timeframe}")
    factor = target_ms // base_ms
    if target_ms % 60_000:
        raise ValueError(f"{target_timeframe} must align to whole minutes")

    target_minutes = target_ms // 60_000
    output: list[Candle] = []
    current_bucket: tuple[int, int] | None = None
    bucket_open_time = None
    bucket_open = 0.0
    bucket_high = 0.0
    bucket_low = 0.0
    bucket_close = 0.0
    bucket_volume = 0.0
    bucket_count = 0

    def flush_bucket() -> None:
        nonlocal bucket_open_time, bucket_open, bucket_high, bucket_low, bucket_close, bucket_volume, bucket_count
        if bucket_open_time is None or bucket_count < factor:
            return
        output.append(
            Candle(
                timestamp=bucket_open_time,
                open=bucket_open,
                high=bucket_high,
                low=bucket_low,
                close=bucket_close,
                volume=bucket_volume,
            )
        )

    for candle in candles:
        minute_of_day = candle.timestamp.hour * 60 + candle.timestamp.minute
        bucket = (candle.timestamp.toordinal(), minute_of_day // target_minutes)
        if current_bucket is None:
            current_bucket = bucket
            bucket_open_time = candle.timestamp
            bucket_open = candle.open
            bucket_high = candle.high
            bucket_low = candle.low
            bucket_close = candle.close
            bucket_volume = candle.volume
            bucket_count = 1
            continue
        if bucket != current_bucket:
            flush_bucket()
            current_bucket = bucket
            bucket_open_time = candle.timestamp
            bucket_open = candle.open
            bucket_high = candle.high
            bucket_low = candle.low
            bucket_close = candle.close
            bucket_volume = candle.volume
            bucket_count = 1
            continue
        bucket_high = max(bucket_high, candle.high)
        bucket_low = min(bucket_low, candle.low)
        bucket_close = candle.close
        bucket_volume += candle.volume
        bucket_count += 1
    flush_bucket()
    return output


def _merge_chunk(chunk: list[Candle]) -> Candle:
    return Candle(
        timestamp=chunk[0].timestamp,
        open=chunk[0].open,
        high=max(candle.high for candle in chunk),
        low=min(candle.low for candle in chunk),
        close=chunk[-1].close,
        volume=sum(candle.volume for candle in chunk),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="1m execution replay for the live portfolio strategy")
    parser.add_argument("--config", default="config.live.optimized_super_volume.json")
    parser.add_argument("--execution-data-dir", default="data/binance_1m_live_replay")
    parser.add_argument("--initial-equity", type=float, default=None)
    parser.add_argument("--include-trades", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Skip heavy detailed report groups; keeps monthly, strategy, and drawdown summaries.")
    parser.add_argument("--no-progress", action="store_true", help="Disable stderr progress updates.")
    parser.add_argument("--trade-start", default=None, help="UTC ISO timestamp; earlier data is warmup only")
    parser.add_argument("--trade-end", default=None, help="UTC ISO timestamp")
    parser.add_argument(
        "--cost-experiment",
        default=None,
        choices=(
            "no_cost",
            "fee_only",
            "fee_slippage_1bps",
            "fee_slippage_3bps",
            "fee_slippage_5bps",
            "full_cost",
            "pessimistic",
            "optimistic",
        ),
    )
    parser.add_argument("--backtest-mode", default=None, choices=("conservative", "pessimistic", "optimistic", "neutral"))
    args = parser.parse_args()
    trade_start = parse_timestamp(args.trade_start) if args.trade_start else None
    trade_end = parse_timestamp(args.trade_end) if args.trade_end else None
    print(
        json.dumps(
            run_execution_backtest(
                args.config,
                args.execution_data_dir,
                initial_equity=args.initial_equity,
                include_trades=args.include_trades,
                compact=args.compact,
                progress=not args.no_progress,
                trade_start=trade_start,
                trade_end=trade_end,
                cost_experiment=args.cost_experiment,
                backtest_mode=args.backtest_mode,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from .alpha_diagnostics import AlphaCandidateDiagnostics
from .data import interval_to_milliseconds, parse_timestamp
from .live_config import load_live_config
from .live_portfolio_backtest import (
    HistoricalClient,
    PortfolioPosition,
    _account_snapshot,
    _add_to_position,
    _build_btc_market_state_cache,
    _build_indicator_reversal_cache,
    _build_market_regime_cache,
    _build_mtf_cache,
    _build_signal_cache,
    _cached_entry_candidate,
    _cached_fast_breakout_candidate,
    _close_position,
    _funding_for_position,
    _hold_minutes,
    _load_symbol_data,
    _mark_equity,
    _maybe_scale_in_position,
    _open_position,
    _record_monthly_closed_trades,
    _record_monthly_equity,
    _record_monthly_open,
    _summary,
    _strategy_bucket,
    _update_position_excursion,
)
from .live_trader import (
    BinanceAutoTrader,
    EntryCandidate,
    SimPosition,
    _candidate_requires_extra_slot,
    _entry_position_limit,
    _sma_values,
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
from .risk import BacktestExecutionConfig, BacktestExecutionStats, execution_config_from_live_config, market_exit_fill
from .risk import market_entry_fill
from .indicators import atr, ema, macd
from .realistic_data import load_funding_rate_directory
from .regime_score import RegimeScoreEngine
from .reversal_alpha import ReversalAlphaEngine
from .cmipr import CMIPR_REASON_TOKEN, CmiprEngine, CmiprState, audit_derivative_coverage


_POINT_IN_TIME_UNIVERSE: dict[Any, frozenset[str]] = {}


@dataclass
class PendingEntry:
    candidate: Any
    signal_index: int
    signal_time: Any
    signal_available_time: Any
    decision_time: Any
    earliest_execution_index: int


@dataclass
class PendingCmiprAddon:
    symbol: str
    signal: Signal
    fraction: float
    proposed_stop: float
    signal_time: Any
    earliest_execution_index: int
    addon_number: int


@dataclass
class VbpBreakoutState:
    symbol: str
    breakout_index: int
    breakout_time: Any
    breakout_level: float
    consolidation_bottom: float
    pullback_target: float
    breakout_volume: float
    breakout_close: float
    tp2_price: float
    touched_pullback: bool = False
    pullback_touch_index: int | None = None
    confirmation_index: int | None = None
    confirmation_price: float | None = None
    confirmation_time: Any = None
    breakout_atr: float = 0.0
    confirmation_body_atr: float = 0.0
    confirmation_close_position: float = 0.0
    confirmation_wick_ratio: float = 0.0
    confirmation_volume_ratio: float = 0.0
    pullback_low: float = 0.0
    pullback_broke_breakout_level: bool = False
    alpha_event_id: str | None = None


@dataclass
class VbpSymbolFeatures:
    rvol_1h: list[float]
    rvol_1m: list[float]
    quote_prefix: list[float]
    consolidation_ok: list[bool]
    consolidation_top: list[float]
    consolidation_bottom: list[float]
    daily_high_ok: list[bool]


@dataclass
class VbpRuntimeControl:
    loss_streak: int = 0
    pause_until: Any = None
    loss_reduce_until: Any = None
    loss_quality_until: Any = None
    peak_equity: float = 0.0
    symbol_cooldown_until: dict[str, Any] | None = None
    daily_pnl: dict[Any, float] | None = None
    monthly_pnl: dict[str, float] | None = None
    monthly_peak_equity: dict[str, float] | None = None
    weekly_pnl: dict[str, float] | None = None
    weekly_start_equity: dict[str, float] | None = None
    weekly_peak_equity: dict[str, float] | None = None
    symbol_recent_pnl: dict[str, list[float]] | None = None

    def __post_init__(self) -> None:
        if self.symbol_cooldown_until is None:
            self.symbol_cooldown_until = {}
        if self.daily_pnl is None:
            self.daily_pnl = {}
        if self.monthly_pnl is None:
            self.monthly_pnl = {}
        if self.monthly_peak_equity is None:
            self.monthly_peak_equity = {}
        if self.weekly_pnl is None:
            self.weekly_pnl = {}
        if self.weekly_start_equity is None:
            self.weekly_start_equity = {}
        if self.weekly_peak_equity is None:
            self.weekly_peak_equity = {}
        if self.symbol_recent_pnl is None:
            self.symbol_recent_pnl = {}


@dataclass(frozen=True)
class VbpCompressionMetrics:
    atr_percentile: float
    range_atr: float
    volume_contraction: float
    prior_move_atr: float
    atr_value: float = 0.0

    def diagnostic_fields(self) -> dict[str, float]:
        return {
            "pre_breakout_atr_percentile": self.atr_percentile,
            "pre_breakout_range_compression_atr": self.range_atr,
            "pre_breakout_volume_contraction": self.volume_contraction,
            "pre_breakout_prior_move_atr": self.prior_move_atr,
        }


@dataclass(frozen=True)
class VbpBreakoutMetrics:
    distance_atr: float
    body_atr: float
    close_position: float
    upper_wick_ratio: float
    volume_ratio: float

    def diagnostic_fields(self) -> dict[str, float]:
        return {
            "breakout_distance_atr": self.distance_atr,
            "breakout_body_atr": self.body_atr,
            "breakout_close_position": self.close_position,
            "upper_wick_ratio": self.upper_wick_ratio,
            "breakout_volume_ratio": self.volume_ratio,
        }


@dataclass(frozen=True)
class VbpPullbackMetrics:
    depth_atr: float
    depth_to_breakout: float
    bars: int
    broke_breakout_level: bool


@dataclass(frozen=True)
class VbpConfirmationMetrics:
    body_atr: float
    close_position: float
    upper_wick_ratio: float


@dataclass
class PortfolioRuntimeControl:
    weekly_start_equity: dict[str, float] | None = None
    weekly_peak_equity: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.weekly_start_equity is None:
            self.weekly_start_equity = {}
        if self.weekly_peak_equity is None:
            self.weekly_peak_equity = {}


@dataclass
class CmiprAccountRuntime:
    loss_streak: int = 0
    pause_until: Any = None
    daily_pnl: dict[Any, float] | None = None

    def __post_init__(self) -> None:
        if self.daily_pnl is None:
            self.daily_pnl = {}


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
    execution_candles, alignment_report = _align_execution_candles_by_utc_timestamp(execution_candles)

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
    cmipr_enabled = bool(getattr(getattr(config, "cmipr", None), "enabled", False))
    if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False) or cmipr_enabled:
        mtf_candles = {
            timeframe: {
                symbol: _resample_to_timeframe(candles, execution_timeframe, timeframe)
                for symbol, candles in execution_candles.items()
                if symbol in symbols
            }
            for timeframe in _mtf_required_timeframes(config)
        }
    result = run_execution_backtest_config(
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
    result["time_alignment"] = alignment_report
    return result


def _align_execution_candles_by_utc_timestamp(
    candles_by_symbol: dict[str, list[Candle]],
) -> tuple[dict[str, list[Candle]], dict[str, Any]]:
    """Build one explicit UTC clock shared by every execution series.

    The engine is index-based internally, so indexes are safe only after every
    symbol has been reordered against the same timestamp sequence.  A minute
    missing from any symbol is excluded from the portfolio clock rather than
    forward-filled or borrowed from a neighbouring candle.
    """
    if not candles_by_symbol:
        return {}, {"mode": "utc_intersection", "symbols": 0, "aligned_minutes": 0}
    original_counts = {symbol: len(candles) for symbol, candles in candles_by_symbol.items()}
    first_symbol = next(iter(candles_by_symbol))
    common_timestamps = {candle.timestamp for candle in candles_by_symbol[first_symbol]}
    for symbol, candles in candles_by_symbol.items():
        if symbol == first_symbol:
            continue
        common_timestamps.intersection_update(candle.timestamp for candle in candles)
        if not common_timestamps:
            raise RuntimeError(f"no common UTC execution timestamps remain after aligning {symbol}")
    timeline = sorted(common_timestamps)
    aligned: dict[str, list[Candle]] = {}
    for symbol, candles in candles_by_symbol.items():
        by_timestamp = {candle.timestamp: candle for candle in candles}
        aligned[symbol] = [by_timestamp[timestamp] for timestamp in timeline]
    return aligned, {
        "mode": "utc_intersection",
        "symbols": len(aligned),
        "aligned_minutes": len(timeline),
        "first_timestamp": timeline[0].isoformat() if timeline else None,
        "last_timestamp": timeline[-1].isoformat() if timeline else None,
        "original_min_minutes": min(original_counts.values()),
        "original_max_minutes": max(original_counts.values()),
        "dropped_minutes_from_shortest_series": min(original_counts.values()) - len(timeline),
    }


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
    cmipr_feature_cache: dict[tuple[Any, ...], Any] | None = None,
) -> dict[str, Any]:
    symbols = tuple(
        symbol
        for symbol in config.trading.symbols
        if symbol in execution_candles_by_symbol and symbol in signal_candles_by_symbol
    )
    global _POINT_IN_TIME_UNIVERSE
    _POINT_IN_TIME_UNIVERSE = _build_point_in_time_universe(config, execution_candles_by_symbol)
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
    cmipr_enabled = bool(getattr(getattr(config, "cmipr", None), "enabled", False))
    if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False) or cmipr_enabled:
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
        if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False) or cmipr_enabled
        else {}
    )

    low_base_ignition_enabled = bool(getattr(config.strategy, "low_base_volume_ignition_enabled", False))
    vbp_enabled = _vbp_enabled(config)
    historical_filter_timeframes = tuple(config.filters.timeframes)
    if low_base_ignition_enabled:
        for timeframe in ("15m", "30m", "1h", "4h"):
            if timeframe not in historical_filter_timeframes:
                historical_filter_timeframes += (timeframe,)
    if low_base_ignition_enabled:
        client = HistoricalClient(execution_candles_by_symbol, "1m", historical_filter_timeframes)
    else:
        client = HistoricalClient(signal_candles_by_symbol, config.trading.timeframe, historical_filter_timeframes)
    trader = BinanceAutoTrader(config, client)
    execution_config = execution_config_from_live_config(config, cost_experiment=cost_experiment, mode=backtest_mode)
    if execution_config.funding_enabled and getattr(config.risk, "funding_data_dir", ""):
        execution_config = replace(
            execution_config,
            funding_rates_by_symbol=load_funding_rate_directory(config.risk.funding_data_dir, symbols),
        )
    execution_stats = BacktestExecutionStats()
    common_signal_length = min(len(candles) for candles in signal_candles_by_symbol.values())
    common_execution_length = min(len(candles) for candles in execution_candles_by_symbol.values())
    legacy_disabled = (
        bool(getattr(config.strategy, "mtf_disable_legacy_strategies", False))
        and bool(getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False))
    ) or (cmipr_enabled and bool(getattr(config.cmipr, "disable_legacy_strategies", True)))
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
    pending_cmipr_addons: list[PendingCmiprAddon] = []
    indicator_reversal_loss_streak: dict[Direction, int] = {Direction.LONG: 0, Direction.SHORT: 0}
    indicator_reversal_pause_until_time: dict[Direction, Any] = {Direction.LONG: None, Direction.SHORT: None}
    next_entry_scan_time = first_trade_time
    summary_first_candle = next(iter(execution_candles_by_symbol.values()))[0]
    last_execution_index = 0
    mtf_reject_stats: dict[str, int] = {}
    mtf_daily_entry_counts: dict[Any, int] = {}
    mtf_symbol_cooldown_until: dict[str, Any] = {}
    low_base_daily_entry_counts: dict[Any, int] = {}
    vbp_watch_until: dict[str, Any] = {}
    vbp_breakouts: dict[str, VbpBreakoutState] = {}
    vbp_stats: dict[str, int] = {}
    cmipr_engine = CmiprEngine(
        config,
        mtf_candles_by_timeframe,
        mtf_aux_features,
        shared_feature_cache=cmipr_feature_cache,
    ) if cmipr_enabled else None
    cmipr_coverage = audit_derivative_coverage(config, symbols, trade_start, trade_end) if cmipr_enabled else None
    if cmipr_coverage is not None and not cmipr_coverage.eligible:
        raise RuntimeError(cmipr_coverage.reason)
    cmipr_stats: dict[str, int] = {}
    cmipr_account_runtime = CmiprAccountRuntime()
    alpha_diagnostics_enabled = bool(getattr(config.risk, "alpha_diagnostics_enabled", False))
    regime_score_config = getattr(config, "regime_score", None)
    regime_score_engine = None
    if alpha_diagnostics_enabled and regime_score_config is not None and bool(getattr(regime_score_config, "enabled", False)):
        regime_score_engine = RegimeScoreEngine(
            regime_score_config,
            {
                timeframe: {
                    symbol: _resample_to_timeframe(candles, "1m", timeframe)
                    for symbol, candles in execution_candles_by_symbol.items()
                }
                for timeframe in ("15m", "30m", "1h")
            },
        )
    reversal_alpha_config = getattr(config, "reversal_alpha", None)
    reversal_alpha_engine = None
    if alpha_diagnostics_enabled and reversal_alpha_config is not None and bool(getattr(reversal_alpha_config, "enabled", False)):
        reversal_alpha_engine = ReversalAlphaEngine(
            reversal_alpha_config,
            {
                timeframe: {
                    symbol: _resample_to_timeframe(candles, "1m", timeframe)
                    for symbol, candles in execution_candles_by_symbol.items()
                }
                for timeframe in ("5m", "15m", "30m", "1h")
            },
        )
    alpha_diagnostics = AlphaCandidateDiagnostics(
        alpha_diagnostics_enabled,
        full_round_trip_cost_pct=(
            2.0 * execution_config.taker_fee_rate
            + (execution_config.market_slippage_bps + execution_config.take_profit_slippage_bps) / 10_000.0
        ),
        stop_round_trip_cost_pct=(
            2.0 * execution_config.taker_fee_rate
            + (execution_config.market_slippage_bps + execution_config.stop_slippage_bps) / 10_000.0
        ),
        regime_score_engine=regime_score_engine,
        reversal_alpha_engine=reversal_alpha_engine,
    )
    portfolio_control_stats: dict[str, int] = {}
    portfolio_symbol_cooldown_until: dict[str, Any] = {}
    portfolio_runtime = PortfolioRuntimeControl()
    vbp_control = VbpRuntimeControl()
    vbp_feature_cache = _build_vbp_feature_cache(config, execution_candles_by_symbol) if vbp_enabled else {}
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
        client.current_index = execution_index if low_base_ignition_enabled else signal_index
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
            vbp_stats,
            cmipr_engine,
            cmipr_stats,
        )
        _record_monthly_closed_trades(monthly_stats, timestamp, trades[closed_before:])
        if cmipr_enabled:
            _update_cmipr_account_runtime(
                config,
                cmipr_account_runtime,
                trades[closed_before:],
                timestamp,
                cmipr_stats,
            )
        if vbp_enabled:
            _update_vbp_control_after_trades(config, vbp_control, trades[closed_before:], timestamp, starting_equity, vbp_stats)
        _update_portfolio_control_after_trades(
            config,
            portfolio_symbol_cooldown_until,
            trades[closed_before:],
            timestamp,
            portfolio_control_stats,
        )
        cash = _fill_cmipr_pending_addons_1m(
            trader,
            config,
            cash,
            positions,
            pending_cmipr_addons,
            execution_candles_by_symbol,
            execution_index,
            timestamp,
            execution_config,
            cmipr_engine,
            cmipr_stats,
        )
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
            portfolio_symbol_cooldown_until,
            portfolio_control_stats,
            portfolio_runtime,
            cmipr_engine,
            cmipr_stats,
        )
        equity = _mark_equity(cash, positions, execution_candles_by_symbol, execution_index)
        trader._peak_equity = max(getattr(trader, "_peak_equity", starting_equity), equity)
        equity_curve.append(equity)
        equity_timeline.append((timestamp, equity))
        _record_monthly_equity(monthly_stats, timestamp, equity)

        if timestamp < next_entry_scan_time:
            continue
        scan_seconds = max(1, int(config.trading.entry_scan_seconds))
        if low_base_ignition_enabled:
            scan_seconds = max(scan_seconds, interval_to_milliseconds("30m") // 1000)
        while next_entry_scan_time <= timestamp:
            next_entry_scan_time += timedelta(seconds=scan_seconds)

        closed_before = len(trades)
        if cmipr_enabled:
            _queue_cmipr_addons_1m(
                config,
                positions,
                pending_cmipr_addons,
                execution_candles_by_symbol,
                execution_index,
                timestamp,
                execution_config,
                client,
                mtf_candles_by_timeframe,
                mtf_timestamps_by_timeframe,
                cmipr_engine,
                cmipr_stats,
            )
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
        if cmipr_enabled and _cmipr_new_entries_paused(
            config,
            cmipr_account_runtime,
            timestamp,
            starting_equity,
        ):
            continue
        if low_base_ignition_enabled:
            daily_limit = max(1, int(getattr(config.strategy, "low_base_daily_entry_limit", 2)))
            if low_base_daily_entry_counts.get(timestamp.date(), 0) >= daily_limit:
                trader.low_base_ignition_stats["reject_daily_entry_limit"] = trader.low_base_ignition_stats.get("reject_daily_entry_limit", 0) + 1
                continue

        if vbp_enabled:
            opened_before = len(positions)
            cash = _run_vbp_scan_1m(
                trader,
                config,
                cash,
                positions,
                monthly_stats,
                execution_candles_by_symbol,
                execution_index,
                signal_index,
                timestamp,
                client,
                execution_config,
                vbp_watch_until,
                vbp_breakouts,
                vbp_feature_cache,
                vbp_control,
                starting_equity,
                equity,
                vbp_stats,
                portfolio_symbol_cooldown_until,
                portfolio_control_stats,
                portfolio_runtime,
                alpha_diagnostics,
            )
            if len(positions) > opened_before:
                _record_monthly_equity(monthly_stats, timestamp, _mark_equity(cash, positions, execution_candles_by_symbol, execution_index))
            if len(positions) + len(pending_entries) >= _entry_position_limit(config):
                continue

        if cmipr_enabled and cmipr_engine is not None:
            occupied = set(positions) | {entry.candidate.symbol for entry in pending_entries}
            allowed = set(_POINT_IN_TIME_UNIVERSE.get(timestamp.date(), frozenset())) if bool(config.risk.point_in_time_universe_enabled) else set(cmipr_engine.symbols)
            candidates = cmipr_engine.scan(timestamp, occupied, allowed)
        else:
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
                alpha_diagnostics,
            )
        pending_symbols = {entry.candidate.symbol for entry in pending_entries}
        candidates = [candidate for candidate in candidates if candidate.symbol not in pending_symbols]
        if cmipr_enabled:
            max_same_direction = max(1, int(config.cmipr.risk_control.max_same_direction_positions))
            candidates = [
                candidate
                for candidate in candidates
                if sum(1 for position in positions.values() if position.direction == candidate.signal.direction) < max_same_direction
            ]
        candidates = _portfolio_filter_candidates_for_scan(
            config,
            candidates,
            positions,
            pending_entries,
            execution_candles_by_symbol,
            execution_index,
            timestamp,
            portfolio_symbol_cooldown_until,
            portfolio_control_stats,
            equity,
            portfolio_runtime,
        )
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
                    earliest_execution_index=(
                        execution_index
                        + 1
                        + (
                            max(0, int(config.cmipr.entry.extra_execution_delay_minutes))
                            if CMIPR_REASON_TOKEN in str(candidate.signal.reason)
                            else 0
                        )
                    ),
                )
            )
            if cmipr_engine is not None and CMIPR_REASON_TOKEN in str(candidate.signal.reason):
                cmipr_engine.mark_order_pending(candidate.symbol)
            if _is_mtf_candidate(candidate):
                mtf_daily_entry_counts[timestamp.date()] = mtf_daily_entry_counts.get(timestamp.date(), 0) + 1
                cooldown_hours = max(0, int(getattr(config.strategy, "mtf_symbol_cooldown_hours", 12)))
                if cooldown_hours > 0:
                    mtf_symbol_cooldown_until[candidate.symbol] = timestamp + timedelta(hours=cooldown_hours)
            if "low_base_volume_ignition_long" in str(candidate.signal.reason):
                low_base_daily_entry_counts[timestamp.date()] = low_base_daily_entry_counts.get(timestamp.date(), 0) + 1
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
    short_guard_stats = getattr(trader, "short_guard_stats", None)
    if short_guard_stats:
        payload["short_guard_stats"] = dict(sorted(short_guard_stats.items()))
    low_base_stats = getattr(trader, "low_base_ignition_stats", None)
    if low_base_stats:
        payload["low_base_ignition_stats"] = dict(sorted(low_base_stats.items()))
    if vbp_stats:
        payload["vbp_stats"] = dict(sorted(vbp_stats.items()))
    if alpha_diagnostics.enabled:
        payload["alpha_candidate_diagnostics"] = alpha_diagnostics.finalize(execution_candles_by_symbol, trades)
    if portfolio_control_stats:
        payload["portfolio_control_stats"] = dict(sorted(portfolio_control_stats.items()))
    if cmipr_engine is not None and cmipr_coverage is not None:
        payload["cmipr_report"] = _cmipr_report(
            config,
            trades,
            cmipr_engine.report(),
            cmipr_coverage.as_dict(),
            cmipr_stats,
        )
    if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False):
        mtf_payload = dict(payload)
        mtf_payload["trades"] = trades
        payload["mtf_report"] = mtf_report_from_summary(mtf_payload, mtf_reject_stats)
    return payload


def _cmipr_report(
    config: Any,
    trades: list[dict[str, Any]],
    engine_report: dict[str, Any],
    coverage: dict[str, Any],
    stats: dict[str, int],
) -> dict[str, Any]:
    cmipr_trades = [trade for trade in trades if trade.get("strategy") == CMIPR_REASON_TOKEN]
    by_addons: dict[str, list[dict[str, Any]]] = {"no_addon": [], "one_addon": [], "two_addons": []}
    for trade in cmipr_trades:
        scale_ins = int(trade.get("scale_ins", 0) or 0)
        key = "no_addon" if scale_ins <= 0 else "one_addon" if scale_ins == 1 else "two_addons"
        by_addons[key].append(trade)
    return {
        **engine_report,
        "derivatives_coverage": coverage,
        "execution_stats": dict(sorted(stats.items())),
        "historical_test_policy": {
            "historical_test_start": config.cmipr.research.historical_test_start,
            "historical_test_end": config.cmipr.research.historical_test_end,
            "is_untouched_final_holdout": False,
            "final_acceptance_source": config.cmipr.research.final_acceptance_source,
        },
        "by_addon_count": {
            key: _cmipr_trade_metrics(rows)
            for key, rows in by_addons.items()
        },
    }


def _cmipr_trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [trade for trade in trades if float(trade.get("net_pnl", 0.0)) > 0]
    losses = [trade for trade in trades if float(trade.get("net_pnl", 0.0)) <= 0]
    gross_profit = sum(float(trade.get("net_pnl", 0.0)) for trade in wins)
    gross_loss = abs(sum(float(trade.get("net_pnl", 0.0)) for trade in losses))
    return {
        "trades": len(trades),
        "net_pnl": sum(float(trade.get("net_pnl", 0.0)) for trade in trades),
        "win_rate_pct": len(wins) / len(trades) * 100.0 if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "average_net_pnl": sum(float(trade.get("net_pnl", 0.0)) for trade in trades) / len(trades) if trades else 0.0,
    }


def _cmipr_trade_risk_budget(config: Any, current_equity: float) -> float:
    mode = str(config.cmipr.research.sizing_mode).strip().lower()
    if mode == "fixed_risk_usdt":
        return max(0.0, float(config.cmipr.research.fixed_trade_risk_usdt))
    if mode == "fixed_equity":
        equity = max(0.0, float(config.cmipr.research.fixed_equity_usdt))
    else:
        equity = max(0.0, current_equity)
    return equity * max(0.0, float(config.risk.risk_per_trade_pct))


def _cmipr_sizing_account(config: Any, account: Any, signal: Signal) -> Any:
    if CMIPR_REASON_TOKEN not in str(signal.reason):
        return account
    mode = str(config.cmipr.research.sizing_mode).strip().lower()
    if mode == "compounding":
        return account
    if mode == "fixed_risk_usdt":
        risk_pct = max(float(config.risk.risk_per_trade_pct), 1e-12)
        sizing_equity = float(config.cmipr.research.fixed_trade_risk_usdt) / risk_pct
    elif mode == "fixed_equity":
        sizing_equity = float(config.cmipr.research.fixed_equity_usdt)
    else:
        raise ValueError(f"unknown CMIPR sizing_mode: {mode}")
    sizing_equity = max(0.0, sizing_equity)
    return replace(
        account,
        equity=sizing_equity,
        wallet_balance=sizing_equity,
        available_balance=max(0.0, min(account.available_balance, sizing_equity - account.initial_margin)),
    )


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
    portfolio_symbol_cooldown_until: dict[str, Any] | None = None,
    portfolio_control_stats: dict[str, int] | None = None,
    portfolio_runtime: PortfolioRuntimeControl | None = None,
    cmipr_engine: CmiprEngine | None = None,
    cmipr_stats: dict[str, int] | None = None,
) -> float:
    if not pending_entries:
        return cash
    portfolio_symbol_cooldown_until = portfolio_symbol_cooldown_until or {}
    portfolio_control_stats = portfolio_control_stats if portfolio_control_stats is not None else {}
    portfolio_runtime = portfolio_runtime or PortfolioRuntimeControl()
    remaining: list[PendingEntry] = []
    filled = 0
    max_fills = max(1, int(config.trading.max_new_entries_per_cycle))
    for pending in pending_entries:
        candidate = pending.candidate
        if not _point_in_time_symbol_allowed(config, candidate.symbol, timestamp):
            if CMIPR_REASON_TOKEN in str(candidate.signal.reason):
                _count_stat(cmipr_stats if cmipr_stats is not None else {}, "initial_reject_point_in_time_universe")
            continue
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
        bucket = _portfolio_bucket_for_candidate(candidate)
        current_equity = _mark_equity(cash, positions, execution_candles_by_symbol, execution_index)
        portfolio_allowed, portfolio_reason, portfolio_multiplier = _portfolio_entry_decision(
            config,
            candidate.symbol,
            bucket,
            positions,
            remaining,
            execution_candles_by_symbol,
            execution_index,
            timestamp,
            portfolio_symbol_cooldown_until,
            current_equity,
            portfolio_runtime,
        )
        if not portfolio_allowed:
            _portfolio_count(portfolio_control_stats, portfolio_reason)
            if CMIPR_REASON_TOKEN in str(candidate.signal.reason):
                _count_stat(cmipr_stats if cmipr_stats is not None else {}, f"initial_reject_{portfolio_reason}")
            continue
        if portfolio_multiplier <= 0:
            _portfolio_count(portfolio_control_stats, "portfolio_zero_risk")
            continue
        if portfolio_multiplier < 0.999:
            adjusted_signal = replace(
                candidate.signal,
                risk_multiplier=candidate.signal.risk_multiplier * portfolio_multiplier,
            )
            candidate = replace(candidate, signal=adjusted_signal)
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
        pre_entry_equity = account.equity
        sizing_account = _cmipr_sizing_account(config, account, candidate.signal)
        quantity_text, reason = trader._size_order(candidate.symbol, execution_candle.close, candidate.signal, sizing_account)
        quantity = float(quantity_text)
        if reason != "ok" or quantity <= 0:
            if CMIPR_REASON_TOKEN in str(candidate.signal.reason):
                _count_stat(cmipr_stats if cmipr_stats is not None else {}, f"initial_reject_size_{reason}")
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
            position = positions[candidate.symbol]
            if _is_cmipr_position(position):
                side_multiplier = candidate.signal.risk_multiplier / max(float(config.cmipr.entry.initial_risk_fraction), 1e-12)
                position.risk_budget_usdt = _cmipr_trade_risk_budget(config, pre_entry_equity) * min(1.0, max(0.0, side_multiplier))
                position.initial_stop_price = position.stop_price
                if cmipr_engine is not None:
                    if position.capacity_fill_ratio < 0.999:
                        cmipr_engine.mark_partial_fill(candidate.symbol)
                    cmipr_engine.mark_protection_pending(candidate.symbol)
                    cmipr_engine.mark_protected(candidate.symbol)
                _count_stat(cmipr_stats if cmipr_stats is not None else {}, "initial_fill_count")
            _record_monthly_open(monthly_stats, timestamp, candidate.signal.direction)
            filled += 1
        elif CMIPR_REASON_TOKEN in str(candidate.signal.reason):
            _count_stat(cmipr_stats if cmipr_stats is not None else {}, "initial_reject_execution_capacity_or_exchange_rules")
    pending_entries[:] = remaining
    return cash


def _candidate_signal_times(config: Any, candidate: Any, fallback_signal_time: Any) -> tuple[Any, Any]:
    if CMIPR_REASON_TOKEN in str(candidate.signal.reason):
        trigger_timeframe = _reason_tag(candidate.signal.reason, "trigger_tf", "15m")
        signal_time = candidate.candle.timestamp
        return signal_time, signal_time + timedelta(milliseconds=interval_to_milliseconds(trigger_timeframe))
    if _is_mtf_candidate(candidate):
        signal_time = candidate.candle.timestamp
        trigger_timeframe = _mtf_trigger_timeframe(config)
        return signal_time, signal_time + timedelta(milliseconds=interval_to_milliseconds(trigger_timeframe))
    signal_ms = interval_to_milliseconds(config.trading.timeframe)
    return fallback_signal_time, fallback_signal_time + timedelta(milliseconds=signal_ms)


def _reason_tag(reason: str, key: str, default: str) -> str:
    marker = f"{key}="
    if marker not in str(reason):
        return default
    return str(reason).split(marker, 1)[1].split()[0].strip().rstrip(",") or default


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
    if bool(getattr(getattr(config, "cmipr", None), "enabled", False)):
        timeframes.update(("5m", "15m", "30m", "1h", "4h"))
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
    indicator_reversal_pause_until_time: dict[Direction, Any],
    alpha_diagnostics: AlphaCandidateDiagnostics,
) -> list[Any]:
    candidates = []
    entry_symbols = set(config.trading.entry_symbols or config.trading.symbols)
    legacy_disabled = bool(getattr(config.strategy, "mtf_disable_legacy_strategies", False)) and bool(getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False))
    low_base_only = bool(getattr(config.strategy, "low_base_volume_ignition_enabled", False)) and bool(getattr(config.strategy, "low_base_volume_ignition_disable_legacy", False))
    for symbol in config.trading.symbols:
        if symbol in positions or symbol not in entry_symbols:
            continue
        if reentry_block_until.get(symbol) and timestamp < reentry_block_until[symbol]:
            continue
        candidate = None
        if not legacy_disabled:
            main_signal = signal_cache[symbol][signal_index]
            reversal_signal = reversal_cache[symbol][signal_index]
            if reversal_signal.direction != Direction.FLAT:
                alpha_diagnostics.record_reversal(
                    symbol,
                    reversal_signal,
                    signal_candles_by_symbol[symbol],
                    signal_index,
                    decision_time=timestamp,
                )
            normal_signal_available = low_base_only or main_signal.direction != Direction.FLAT or reversal_signal.direction != Direction.FLAT
            if (
                not low_base_only
                and
                main_signal.direction == Direction.FLAT
                and reversal_signal.direction != Direction.FLAT
                and indicator_reversal_pause_until_time.get(reversal_signal.direction) is not None
                and timestamp < indicator_reversal_pause_until_time[reversal_signal.direction]
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
            if "indicator_" in str(candidate.signal.reason):
                alpha_diagnostics.mark_reversal_accepted(
                    symbol,
                    signal_candles_by_symbol[symbol][signal_index].timestamp,
                    candidate.signal.direction,
                    candidate.rank_score,
                    candidate.filter_reason,
                )
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
    vbp_stats: dict[str, int] | None = None,
    cmipr_engine: CmiprEngine | None = None,
    cmipr_stats: dict[str, int] | None = None,
) -> float:
    for symbol in list(positions):
        position = positions[symbol]
        candle = execution_candles_by_symbol[symbol][execution_index]
        position.bars_held = max(0, signal_index - position.entry_index)
        _update_position_excursion(position, candle)
        if _is_cmipr_position(position):
            cash, closed = _manage_cmipr_position_1m(
                trader,
                config,
                cash,
                positions,
                trades,
                position,
                candle,
                execution_index,
                signal_index,
                execution_config,
                mtf_candles_by_timeframe or {},
                mtf_timestamps_by_timeframe or {},
                cmipr_stats if cmipr_stats is not None else {},
            )
            if closed:
                _mark_reentry_cooldown_time(config, reentry_block_until, symbol, candle.timestamp)
                if cmipr_engine is not None:
                    cmipr_engine.mark_closed(symbol, candle.timestamp)
            continue
        if _is_vbp_position(position) and position.direction == Direction.LONG:
            position.best_price = max(position.best_price, candle.close)
            cash = _update_vbp_partial_exit(
                config,
                cash,
                position,
                trades,
                candle,
                signal_index,
                execution_config,
                trader.client.symbol_rules(symbol),
                vbp_stats if vbp_stats is not None else {},
            )
            recent_1m_candles = execution_candles_by_symbol[symbol][max(0, execution_index - 40):execution_index + 1]
            vbp_exit_reason = _vbp_dynamic_exit_reason(config, position, candle, vbp_stats if vbp_stats is not None else {}, recent_1m_candles)
            if vbp_exit_reason:
                cash = _close_position(
                    config,
                    cash,
                    positions,
                    trades,
                    symbol,
                    candle.close,
                    vbp_exit_reason,
                    signal_index,
                    candle.timestamp,
                    execution_config=execution_config,
                    rules=trader.client.symbol_rules(symbol),
                )
                continue
        if _is_low_base_position(position) and position.direction == Direction.LONG:
            _update_low_base_dynamic_stop(
                trader,
                config,
                position,
                candle,
            )
        exit_price = None
        reason = ""
        vbp_runner_after_tp1 = (
            _is_vbp_position(position)
            and bool(getattr(config.vbp_strategy.exit, "runner_after_tp1_enabled", False))
        )
        if position.direction == Direction.LONG:
            if _is_vbp_position(position):
                position.best_price = max(position.best_price, candle.close)
            else:
                position.best_price = max(position.best_price, candle.high)
            stop_hit = candle.low <= position.stop_price
            take_profit_hit = (not vbp_runner_after_tp1) and candle.high >= position.take_profit_price
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
            take_profit_hit = (not vbp_runner_after_tp1) and candle.low <= position.take_profit_price
            if stop_hit and take_profit_hit:
                execution_stats.same_bar_tp_sl_conflict_count += 1
            if stop_hit and (not take_profit_hit or execution_config.mode != "optimistic"):
                exit_price = position.stop_price
                reason = "stop_loss_1m"
            elif take_profit_hit:
                exit_price = position.take_profit_price
                reason = "take_profit_1m"

        if exit_price is None and _is_low_base_position(position) and position.direction == Direction.LONG:
            low_base_reason = _low_base_exit_reason(
                trader,
                config,
                position,
                candle,
                execution_candles_by_symbol,
                execution_index,
            )
            if low_base_reason:
                exit_price = candle.close
                reason = low_base_reason

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
            if _is_low_base_position(position) and ("stop_loss" in reason or "fail_fast" in reason or "fake_breakout" in reason):
                cooldown_bars = max(0, int(getattr(config.strategy, "low_base_fake_breakout_cooldown_bars_15m", 48)))
                if cooldown_bars > 0:
                    reentry_block_until[symbol] = candle.timestamp + timedelta(minutes=15 * cooldown_bars)
                    trader.low_base_ignition_stats["fake_breakout_cooldown_count"] = trader.low_base_ignition_stats.get("fake_breakout_cooldown_count", 0) + 1
            else:
                _mark_reentry_cooldown_time(config, reentry_block_until, symbol, candle.timestamp)
    return cash


def _is_cmipr_position(position: PortfolioPosition) -> bool:
    return CMIPR_REASON_TOKEN in str(position.entry_reason)


def _manage_cmipr_position_1m(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    trades: list[dict[str, Any]],
    position: PortfolioPosition,
    candle: Candle,
    execution_index: int,
    signal_index: int,
    execution_config: BacktestExecutionConfig,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
    stats: dict[str, int],
) -> tuple[float, bool]:
    symbol = position.symbol
    stop_hit = candle.low <= position.stop_price if position.direction == Direction.LONG else candle.high >= position.stop_price
    if stop_hit:
        cash = _close_position(
            config,
            cash,
            positions,
            trades,
            symbol,
            position.stop_price,
            "cmipr_stop_loss_1m",
            signal_index,
            candle.timestamp,
            execution_config=execution_config,
            rules=trader.client.symbol_rules(symbol),
        )
        _count_stat(stats, "stop_loss_count")
        return cash, True

    current_r = _cmipr_executable_current_r(
        config,
        position,
        candle.close,
        candle.timestamp,
        execution_config,
        trader.client.symbol_rules(symbol),
    )
    position.cmipr_max_executable_r = max(position.cmipr_max_executable_r, current_r)
    hold_minutes = _hold_minutes(position.entry_time, candle.timestamp)
    exit_cfg = config.cmipr.exit
    breakout_level = _reason_float(position.entry_reason, "breakout_level", position.entry_price)
    lost_level = candle.close < breakout_level if position.direction == Direction.LONG else candle.close > breakout_level
    fail_fast_minutes = max(1, int(exit_cfg.fail_fast_bars_5m)) * 5
    reason = ""
    if hold_minutes >= fail_fast_minutes and position.cmipr_max_executable_r < float(exit_cfg.fail_fast_min_mfe_r):
        reason = "cmipr_fail_fast_no_extension"
    elif lost_level and hold_minutes >= 5:
        reason = "cmipr_fail_fast_lost_breakout"
    elif hold_minutes >= max(1, int(exit_cfg.max_holding_minutes)):
        reason = "cmipr_time_stop"
    elif not bool(exit_cfg.runner_enabled) and current_r >= float(exit_cfg.fixed_take_profit_r):
        reason = "cmipr_fixed_r_take_profit"
    else:
        giveback = _cmipr_allowed_giveback_r(exit_cfg, position.cmipr_max_executable_r)
        if position.cmipr_max_executable_r >= float(exit_cfg.runner_activation_r) and current_r < position.cmipr_max_executable_r - giveback:
            reason = "cmipr_max_profit_giveback"

    if not reason and bool(exit_cfg.runner_enabled) and position.cmipr_max_executable_r >= float(exit_cfg.runner_activation_r):
        candles_15m = _mtf_closed(
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
            "15m",
            symbol,
            candle.timestamp,
            80,
        )
        if len(candles_15m) >= 24:
            closes = [item.close for item in candles_15m]
            ema_period = 9 if str(exit_cfg.trailing_type).lower() == "ema9_15m" else 21
            trailing_ema = ema(closes, ema_period)[-1]
            if position.direction == Direction.LONG and candles_15m[-1].close < trailing_ema:
                reason = "cmipr_runner_ema_exit"
            elif position.direction == Direction.SHORT and candles_15m[-1].close > trailing_ema:
                reason = "cmipr_runner_ema_exit"

    if reason:
        cash = _close_position(
            config,
            cash,
            positions,
            trades,
            symbol,
            candle.close,
            reason,
            signal_index,
            candle.timestamp,
            execution_config=execution_config,
            rules=trader.client.symbol_rules(symbol),
        )
        _count_stat(stats, reason)
        return cash, True

    # Stop changes become effective only for the next 1m bar. The old stop was
    # checked first, preserving adverse same-bar ordering.
    if position.cmipr_max_executable_r >= float(exit_cfg.breakeven_trigger_r):
        buffer_pct = max(0.0, float(exit_cfg.breakeven_cost_buffer_pct))
        if position.direction == Direction.LONG:
            position.stop_price = max(position.stop_price, position.entry_price * (1.0 + buffer_pct))
        else:
            position.stop_price = min(position.stop_price, position.entry_price * (1.0 - buffer_pct))
    return cash, False


def _cmipr_executable_current_r(
    config: Any,
    position: PortfolioPosition,
    raw_exit_price: float,
    exit_time: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> float:
    fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        position.quantity,
        raw_exit_price,
        "market",
        position.liquidity_reference_quote_volume or None,
    )
    raw_entry = position.raw_entry_price or position.entry_price
    gross = position.direction.value * position.quantity * (raw_exit_price - raw_entry)
    funding = _funding_for_position(execution_config, position, exit_time)
    net = gross - position.entry_fee - position.entry_slippage_cost - fill.fee - fill.slippage_cost + funding
    risk_budget = position.risk_budget_usdt
    if risk_budget <= 0:
        risk_budget = max(abs(position.entry_price - position.initial_stop_price) * position.quantity, 1e-12)
    return net / max(risk_budget, 1e-12)


def _cmipr_allowed_giveback_r(exit_config: Any, max_r: float) -> float:
    if max_r >= float(exit_config.giveback_high_mfe_r):
        return max(0.05, float(exit_config.giveback_r_high))
    if max_r >= float(exit_config.giveback_mid_mfe_r):
        return max(0.05, float(exit_config.giveback_r_mid))
    return max(0.05, float(exit_config.giveback_r_low))


def _reason_float(reason: str, key: str, default: float = 0.0) -> float:
    marker = f"{key}="
    if marker not in str(reason):
        return default
    try:
        return float(str(reason).split(marker, 1)[1].split()[0].strip().rstrip(","))
    except (TypeError, ValueError):
        return default


def _count_stat(stats: dict[str, int], key: str) -> None:
    stats[key] = stats.get(key, 0) + 1


def _is_low_base_position(position: PortfolioPosition) -> bool:
    return "low_base_volume_ignition_long" in str(position.entry_reason)


def _vbp_enabled(config: Any) -> bool:
    return bool(getattr(getattr(config, "vbp_strategy", None), "enabled", False))


def _vbp_symbols(config: Any) -> tuple[str, ...]:
    vbp_symbols = tuple(getattr(getattr(config, "vbp_strategy", None), "enabled_symbols", ()) or ())
    if not vbp_symbols:
        return tuple(config.trading.symbols)
    trading_symbols = set(config.trading.symbols)
    entry_symbols = set(config.trading.entry_symbols or config.trading.symbols)
    return tuple(symbol for symbol in vbp_symbols if symbol in trading_symbols and symbol in entry_symbols)


def _point_in_time_symbol_allowed(config: Any, symbol: str, timestamp: Any) -> bool:
    if not bool(getattr(config.risk, "point_in_time_universe_enabled", False)):
        return True
    day = timestamp.date() if hasattr(timestamp, "date") else None
    return day is not None and symbol in _POINT_IN_TIME_UNIVERSE.get(day, frozenset())


def _build_point_in_time_universe(config: Any, candles_by_symbol: dict[str, list[Candle]]) -> dict[Any, frozenset[str]]:
    if not bool(getattr(config.risk, "point_in_time_universe_enabled", False)):
        return {}
    top_n = max(1, int(getattr(config.risk, "point_in_time_universe_top_n", 100)))
    lookback_days = max(1, int(getattr(config.risk, "universe_lookback_days", 1)))
    warmup_days = max(0, int(getattr(config.risk, "new_symbol_warmup_days", 20)))
    daily: dict[Any, dict[str, float]] = {}
    first_day: dict[str, Any] = {}
    for symbol, candles in candles_by_symbol.items():
        for candle in candles:
            day = candle.timestamp.date()
            first_day.setdefault(symbol, day)
            bucket = daily.setdefault(day, {})
            bucket[symbol] = bucket.get(symbol, 0.0) + max(0.0, candle.volume * candle.close)
    days = sorted(daily)
    result: dict[Any, frozenset[str]] = {}
    for index, day in enumerate(days):
        prior_days = days[max(0, index - lookback_days):index]
        scores: dict[str, float] = {}
        for prior_day in prior_days:
            for symbol, quote_volume in daily[prior_day].items():
                if (day - first_day[symbol]).days < warmup_days:
                    continue
                scores[symbol] = scores.get(symbol, 0.0) + quote_volume
        ranked = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))[:top_n]
        result[day] = frozenset(ranked)
    return result


def _is_vbp_position(position: PortfolioPosition) -> bool:
    return "vbp_" in str(position.entry_reason)


def _vbp_count(stats: dict[str, int], key: str) -> None:
    stats[key] = stats.get(key, 0) + 1


def _vbp_reason_float(entry_reason: str, key: str, default: float = 0.0) -> float:
    marker = f"{key}="
    if marker not in entry_reason:
        return default
    raw = entry_reason.split(marker, 1)[1].split()[0].strip().rstrip(",")
    try:
        return float(raw)
    except ValueError:
        return default


def _vbp_open_positions(positions: dict[str, PortfolioPosition]) -> int:
    return sum(1 for position in positions.values() if _is_vbp_position(position))


def _update_vbp_control_after_trades(
    config: Any,
    control: VbpRuntimeControl,
    closed_trades: list[dict[str, Any]],
    timestamp: Any,
    starting_equity: float,
    stats: dict[str, int],
) -> None:
    risk = config.vbp_strategy.risk_control
    risk_enabled = bool(risk.enabled)
    quality_enabled = bool(getattr(risk, "consecutive_loss_quality_filter_enabled", False))
    if not risk_enabled and not quality_enabled:
        return
    for trade in closed_trades:
        if trade.get("strategy") != "volume_breakout_pullback":
            continue
        pnl = float(trade.get("net_pnl", 0.0) or 0.0)
        if risk_enabled and control.daily_pnl is not None:
            control.daily_pnl[timestamp.date()] = control.daily_pnl.get(timestamp.date(), 0.0) + pnl
        if trade.get("exit_reason") == "vbp_tp1_partial":
            continue
        month_key = timestamp.strftime("%Y-%m") if hasattr(timestamp, "strftime") else str(timestamp)[:7]
        if risk_enabled and control.monthly_pnl is not None:
            control.monthly_pnl[month_key] = control.monthly_pnl.get(month_key, 0.0) + pnl
        week_key = _week_key(timestamp)
        if risk_enabled and control.weekly_pnl is not None:
            control.weekly_pnl[week_key] = control.weekly_pnl.get(week_key, 0.0) + pnl
        if risk_enabled and control.symbol_recent_pnl is not None:
            symbol = str(trade.get("symbol"))
            recent = control.symbol_recent_pnl.setdefault(symbol, [])
            recent.append(pnl)
            window = max(1, int(risk.symbol_recent_trade_window))
            del recent[:-window]
            if bool(risk.symbol_performance_guard_enabled) and len(recent) >= max(1, int(risk.symbol_recent_min_trades)):
                gains = sum(value for value in recent if value > 0)
                losses = abs(sum(value for value in recent if value < 0))
                pf = gains / losses if losses > 0 else float("inf")
                if sum(recent) <= float(risk.symbol_recent_net_pnl_min) or pf < float(risk.symbol_recent_pf_min):
                    cooldown_minutes = max(1, int(risk.symbol_performance_cooldown_minutes))
                    if control.symbol_cooldown_until is not None:
                        control.symbol_cooldown_until[symbol] = timestamp + timedelta(minutes=cooldown_minutes)
                    _vbp_count(stats, "symbol_performance_cooldown_count")
        if pnl < 0:
            control.loss_streak += 1
            if quality_enabled:
                quality_losses = max(1, int(getattr(risk, "consecutive_loss_quality_losses", 2)))
                if control.loss_streak >= quality_losses:
                    quality_minutes = max(1, int(getattr(risk, "consecutive_loss_quality_minutes", 240)))
                    quality_until = timestamp + timedelta(minutes=quality_minutes)
                    if control.loss_quality_until is None or quality_until > control.loss_quality_until:
                        control.loss_quality_until = quality_until
                    _vbp_count(stats, "consecutive_loss_quality_mode_count")
            if risk_enabled and bool(getattr(risk, "consecutive_loss_reduce_enabled", False)):
                reduce_losses = max(1, int(getattr(risk, "consecutive_loss_reduce_losses", 1)))
                if control.loss_streak >= reduce_losses:
                    reduce_minutes = max(1, int(getattr(risk, "consecutive_loss_reduce_minutes", 1440)))
                    reduce_until = timestamp + timedelta(minutes=reduce_minutes)
                    if control.loss_reduce_until is None or reduce_until > control.loss_reduce_until:
                        control.loss_reduce_until = reduce_until
                    _vbp_count(stats, "consecutive_loss_reduce_count")
            if risk_enabled and control.symbol_cooldown_until is not None:
                cooldown_minutes = max(0, int(risk.symbol_loss_cooldown_minutes))
                if cooldown_minutes > 0:
                    control.symbol_cooldown_until[str(trade.get("symbol"))] = timestamp + timedelta(minutes=cooldown_minutes)
                    _vbp_count(stats, "symbol_loss_cooldown_count")
            limit = max(1, int(risk.consecutive_loss_limit))
            if risk_enabled and control.loss_streak >= limit:
                control.pause_until = timestamp + timedelta(minutes=max(1, int(risk.consecutive_loss_pause_minutes)))
                control.loss_streak = 0
                _vbp_count(stats, "consecutive_loss_pause_count")
        else:
            control.loss_streak = 0
        if risk_enabled and control.daily_pnl is not None:
            daily_limit = -abs(float(risk.daily_loss_stop_pct)) * starting_equity
            if control.daily_pnl.get(timestamp.date(), 0.0) <= daily_limit:
                _vbp_count(stats, "daily_loss_stop_count")


def _vbp_symbol_on_cooldown(control: VbpRuntimeControl, symbol: str, timestamp: Any) -> bool:
    if not control.symbol_cooldown_until:
        return False
    until = control.symbol_cooldown_until.get(symbol)
    return until is not None and timestamp < until


def _vbp_frequency_rejects(
    config: Any,
    positions: dict[str, PortfolioPosition],
    candles: list[Candle],
    execution_index: int,
    symbol: str,
) -> bool:
    risk = config.vbp_strategy.risk_control
    if not bool(risk.enabled) or not bool(getattr(risk, "frequency_control_enabled", False)):
        return False
    max_24h_return = float(risk.max_24h_return_pct)
    if max_24h_return > 0 and _vbp_return(candles, execution_index, 1440) >= max_24h_return:
        return True
    alt_limit = int(risk.correlated_alt_max_positions)
    if alt_limit > 0 and symbol not in {"BTCUSDT", "ETHUSDT"}:
        alt_positions = sum(
            1
            for position in positions.values()
            if _is_vbp_position(position) and position.symbol not in {"BTCUSDT", "ETHUSDT"}
        )
        if alt_positions >= alt_limit:
            return True
    return False


def _portfolio_count(stats: dict[str, int], key: str) -> None:
    stats[key] = stats.get(key, 0) + 1


def _portfolio_control_enabled(config: Any) -> bool:
    control = getattr(config, "portfolio_control", None)
    return bool(control is not None and getattr(control, "enabled", False))


def _portfolio_bucket_from_reason(reason: str) -> str:
    lowered = str(reason or "").lower()
    if "vbp_" in lowered or "volume_breakout_pullback" in lowered:
        return "vbp"
    if "indicator_" in lowered or "macd_golden_cross" in lowered or "macd_dead_cross" in lowered:
        return "indicator"
    return "other"


def _portfolio_bucket_for_candidate(candidate: EntryCandidate) -> str:
    return _portfolio_bucket_from_reason(candidate.signal.reason)


def _portfolio_open_count(
    positions: dict[str, PortfolioPosition],
    bucket: str,
    pending_entries: list[PendingEntry] | None = None,
) -> int:
    count = sum(1 for position in positions.values() if _portfolio_bucket_from_reason(getattr(position, "entry_reason", "")) == bucket)
    if pending_entries:
        count += sum(1 for entry in pending_entries if _portfolio_bucket_for_candidate(entry.candidate) == bucket)
    return count


def _portfolio_altcoin_count(
    positions: dict[str, PortfolioPosition],
    pending_entries: list[PendingEntry] | None = None,
) -> int:
    majors = {"BTCUSDT", "ETHUSDT"}
    count = sum(1 for position in positions.values() if position.symbol not in majors)
    if pending_entries:
        count += sum(1 for entry in pending_entries if entry.candidate.symbol not in majors)
    return count


def _portfolio_symbol_on_cooldown(symbol_cooldown_until: dict[str, Any], symbol: str, timestamp: Any) -> bool:
    until = symbol_cooldown_until.get(symbol)
    return until is not None and timestamp < until


def _portfolio_btc_weak_multiplier(
    config: Any,
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
) -> float:
    if not _portfolio_control_enabled(config):
        return 1.0
    control = config.portfolio_control
    if not bool(getattr(control, "btc_weak_risk_reduction_enabled", False)):
        return 1.0
    btc = execution_candles_by_symbol.get("BTCUSDT", [])
    if execution_index >= len(btc):
        return 1.0
    weak_1h = _vbp_return(btc, execution_index, 60) <= float(control.btc_weak_1h_return_pct)
    weak_4h = _vbp_return(btc, execution_index, 240) <= float(control.btc_weak_4h_return_pct)
    if weak_1h or weak_4h:
        return max(0.0, min(1.0, float(control.btc_weak_risk_multiplier)))
    return 1.0


def _week_key(timestamp: Any) -> str:
    if hasattr(timestamp, "isocalendar"):
        iso = timestamp.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    return str(timestamp)[:10]


def _portfolio_weekly_risk_multiplier(
    config: Any,
    runtime: PortfolioRuntimeControl | None,
    timestamp: Any,
    current_equity: float | None,
) -> tuple[float, str]:
    if runtime is None or current_equity is None:
        return 1.0, "portfolio_weekly_base"
    if not _portfolio_control_enabled(config):
        return 1.0, "portfolio_weekly_disabled"
    control = config.portfolio_control
    if not bool(getattr(control, "weekly_drawdown_control_enabled", False)):
        return 1.0, "portfolio_weekly_disabled"
    key = _week_key(timestamp)
    assert runtime.weekly_start_equity is not None
    assert runtime.weekly_peak_equity is not None
    runtime.weekly_start_equity.setdefault(key, current_equity)
    runtime.weekly_peak_equity[key] = max(runtime.weekly_peak_equity.get(key, current_equity), current_equity)
    start = max(runtime.weekly_start_equity.get(key, current_equity), 1e-12)
    peak = max(runtime.weekly_peak_equity.get(key, current_equity), 1e-12)
    drawdown = max(0.0, 1.0 - current_equity / peak)
    loss = max(0.0, 1.0 - current_equity / start)
    if drawdown >= float(getattr(control, "weekly_drawdown_stop_pct", 0.0)) > 0:
        return 0.0, "portfolio_weekly_drawdown_stop"
    if loss >= float(getattr(control, "weekly_loss_stop_pct", 0.0)) > 0:
        return 0.0, "portfolio_weekly_loss_stop"
    multiplier = 1.0
    if drawdown >= float(getattr(control, "weekly_drawdown_reduce_pct", 0.0)) > 0:
        multiplier = min(multiplier, max(0.0, min(1.0, float(getattr(control, "weekly_drawdown_risk_multiplier", 0.5)))))
    if loss >= float(getattr(control, "weekly_loss_reduce_pct", 0.0)) > 0:
        multiplier = min(multiplier, max(0.0, min(1.0, float(getattr(control, "weekly_loss_risk_multiplier", 0.5)))))
    if multiplier < 0.999:
        return multiplier, "portfolio_weekly_risk_reduced"
    return 1.0, "portfolio_weekly_base"


def _portfolio_entry_decision(
    config: Any,
    symbol: str,
    bucket: str,
    positions: dict[str, PortfolioPosition],
    pending_entries: list[PendingEntry],
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
    timestamp: Any,
    symbol_cooldown_until: dict[str, Any],
    current_equity: float | None = None,
    portfolio_runtime: PortfolioRuntimeControl | None = None,
) -> tuple[bool, str, float]:
    if not _portfolio_control_enabled(config):
        return True, "portfolio_ok", 1.0
    control = config.portfolio_control
    if _portfolio_symbol_on_cooldown(symbol_cooldown_until, symbol, timestamp):
        return False, "portfolio_symbol_cooldown", 0.0
    if bool(control.prevent_same_symbol_overlap) and symbol in positions:
        return False, "portfolio_same_symbol_overlap", 0.0
    total_open = len(positions) + len(pending_entries)
    max_open = max(0, int(control.max_open_positions))
    if max_open > 0 and total_open >= max_open:
        return False, "portfolio_max_open_positions", 0.0
    max_alt = max(0, int(control.max_altcoin_positions))
    if max_alt > 0 and symbol not in {"BTCUSDT", "ETHUSDT"} and _portfolio_altcoin_count(positions, pending_entries) >= max_alt:
        return False, "portfolio_max_altcoin_positions", 0.0
    if bucket == "vbp":
        max_bucket = max(0, int(control.max_vbp_positions))
        if max_bucket > 0 and _portfolio_open_count(positions, "vbp", pending_entries) >= max_bucket:
            return False, "portfolio_max_vbp_positions", 0.0
        multiplier = float(control.vbp_risk_multiplier)
    elif bucket == "indicator":
        max_bucket = max(0, int(control.max_indicator_positions))
        if max_bucket > 0 and _portfolio_open_count(positions, "indicator", pending_entries) >= max_bucket:
            return False, "portfolio_max_indicator_positions", 0.0
        multiplier = float(control.indicator_risk_multiplier)
    else:
        multiplier = 1.0
    weekly_multiplier, weekly_reason = _portfolio_weekly_risk_multiplier(config, portfolio_runtime, timestamp, current_equity)
    if weekly_multiplier <= 0:
        return False, weekly_reason, 0.0
    multiplier *= weekly_multiplier
    if bucket == "vbp":
        multiplier *= _portfolio_btc_weak_multiplier(config, execution_candles_by_symbol, execution_index)
    return True, "portfolio_ok", max(0.0, multiplier)


def _portfolio_filter_candidates_for_scan(
    config: Any,
    candidates: list[EntryCandidate],
    positions: dict[str, PortfolioPosition],
    pending_entries: list[PendingEntry],
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
    timestamp: Any,
    symbol_cooldown_until: dict[str, Any],
    stats: dict[str, int],
    current_equity: float,
    portfolio_runtime: PortfolioRuntimeControl,
) -> list[EntryCandidate]:
    if not _portfolio_control_enabled(config):
        return candidates
    filtered: list[EntryCandidate] = []
    projected_pending = list(pending_entries)
    for candidate in candidates:
        bucket = _portfolio_bucket_for_candidate(candidate)
        allowed, reason, multiplier = _portfolio_entry_decision(
            config,
            candidate.symbol,
            bucket,
            positions,
            projected_pending,
            execution_candles_by_symbol,
            execution_index,
            timestamp,
            symbol_cooldown_until,
            current_equity,
            portfolio_runtime,
        )
        if not allowed:
            _portfolio_count(stats, reason)
            continue
        if multiplier <= 0:
            _portfolio_count(stats, "portfolio_zero_risk")
            continue
        if multiplier < 0.999:
            _portfolio_count(stats, "portfolio_risk_reduced")
            adjusted_signal = replace(candidate.signal, risk_multiplier=candidate.signal.risk_multiplier * multiplier)
            candidate = replace(candidate, signal=adjusted_signal)
        filtered.append(candidate)
        projected_pending.append(PendingEntry(candidate, 0, None, None, timestamp, execution_index + 1))
    return filtered


def _update_portfolio_control_after_trades(
    config: Any,
    symbol_cooldown_until: dict[str, Any],
    closed_trades: list[dict[str, Any]],
    timestamp: Any,
    stats: dict[str, int],
) -> None:
    if not _portfolio_control_enabled(config):
        return
    control = config.portfolio_control
    base_minutes = max(0, int(control.symbol_cooldown_minutes))
    loss_minutes = max(base_minutes, int(control.symbol_loss_cooldown_minutes))
    for trade in closed_trades:
        if trade.get("exit_reason") == "vbp_tp1_partial":
            continue
        symbol = str(trade.get("symbol") or "")
        if not symbol:
            continue
        minutes = base_minutes
        if float(trade.get("net_pnl", 0.0) or 0.0) < 0:
            minutes = loss_minutes
            _portfolio_count(stats, "portfolio_loss_symbol_cooldown")
        elif minutes > 0:
            _portfolio_count(stats, "portfolio_symbol_cooldown")
        if minutes > 0:
            symbol_cooldown_until[symbol] = max(
                symbol_cooldown_until.get(symbol, timestamp),
                timestamp + timedelta(minutes=minutes),
            )


def _vbp_risk_allows(config: Any, control: VbpRuntimeControl, timestamp: Any, starting_equity: float) -> tuple[bool, str]:
    risk = config.vbp_strategy.risk_control
    if not bool(risk.enabled):
        return True, "ok"
    if control.pause_until is not None and timestamp < control.pause_until:
        return False, "reject_vbp_loss_pause"
    daily_pnl = control.daily_pnl.get(timestamp.date(), 0.0) if control.daily_pnl else 0.0
    if daily_pnl <= -abs(float(risk.daily_loss_stop_pct)) * starting_equity:
        return False, "reject_vbp_daily_loss_stop"
    return True, "ok"


def _vbp_dynamic_exposure(
    config: Any,
    control: VbpRuntimeControl,
    timestamp: Any,
    starting_equity: float,
    equity: float,
) -> tuple[int, float, str]:
    vbp = config.vbp_strategy
    risk = vbp.risk_control
    base_max_positions = max(1, int(vbp.position.max_positions))
    base_size_multiplier = float(vbp.position.size_multiplier)
    if _vbp_quality_mode_active(config, control, timestamp):
        return (
            max(1, min(base_max_positions, int(getattr(risk, "consecutive_loss_quality_max_positions", 1)))),
            min(base_size_multiplier, float(getattr(risk, "consecutive_loss_quality_size_multiplier", 0.50))),
            "consecutive_loss_quality_mode",
        )
    if not bool(risk.enabled) or not bool(risk.adaptive_exposure_enabled):
        return base_max_positions, base_size_multiplier, "base"
    if control.peak_equity <= 0:
        control.peak_equity = equity
    control.peak_equity = max(control.peak_equity, equity)
    drawdown = 1.0 - equity / max(control.peak_equity, 1e-12)
    week_key = _week_key(timestamp)
    if control.weekly_start_equity is not None:
        control.weekly_start_equity.setdefault(week_key, equity)
    if control.weekly_peak_equity is not None:
        control.weekly_peak_equity[week_key] = max(control.weekly_peak_equity.get(week_key, equity), equity)
    weekly_start = control.weekly_start_equity.get(week_key, equity) if control.weekly_start_equity else equity
    weekly_peak = control.weekly_peak_equity.get(week_key, equity) if control.weekly_peak_equity else equity
    weekly_drawdown = 1.0 - equity / max(weekly_peak, 1e-12)
    weekly_loss = max(0.0, 1.0 - equity / max(weekly_start, 1e-12))
    month_key = timestamp.strftime("%Y-%m") if hasattr(timestamp, "strftime") else str(timestamp)[:7]
    if control.monthly_peak_equity is not None:
        control.monthly_peak_equity[month_key] = max(control.monthly_peak_equity.get(month_key, equity), equity)
    monthly_peak = control.monthly_peak_equity.get(month_key, equity) if control.monthly_peak_equity else equity
    monthly_drawdown = 1.0 - equity / max(monthly_peak, 1e-12)
    month_pnl = control.monthly_pnl.get(month_key, 0.0) if control.monthly_pnl else 0.0
    day_pnl = control.daily_pnl.get(timestamp.date(), 0.0) if control.daily_pnl and hasattr(timestamp, "date") else 0.0
    if bool(getattr(risk, "weekly_drawdown_control_enabled", False)):
        if weekly_drawdown >= float(getattr(risk, "weekly_drawdown_stop_pct", 0.0)) > 0:
            return 0, 0.0, "weekly_drawdown_stop"
        if weekly_loss >= float(getattr(risk, "weekly_loss_stop_pct", 0.0)) > 0:
            return 0, 0.0, "weekly_loss_stop"
        if (
            weekly_drawdown >= float(getattr(risk, "weekly_drawdown_one_position_pct", 0.0)) > 0
            or weekly_loss >= float(getattr(risk, "weekly_loss_one_position_pct", 0.0)) > 0
        ):
            return 1, min(base_size_multiplier, float(risk.reduced_size_multiplier)), "weekly_one_position"
        if (
            weekly_drawdown >= float(getattr(risk, "weekly_drawdown_half_size_pct", 0.0)) > 0
            or weekly_loss >= float(getattr(risk, "weekly_loss_reduce_pct", 0.0)) > 0
        ):
            return base_max_positions, min(base_size_multiplier, base_size_multiplier * 0.5), "weekly_half_size"
    if bool(getattr(risk, "consecutive_loss_reduce_enabled", False)):
        if control.loss_reduce_until is not None and timestamp < control.loss_reduce_until:
            return (
                max(1, min(base_max_positions, int(getattr(risk, "consecutive_loss_reduced_max_positions", 1)))),
                min(base_size_multiplier, float(getattr(risk, "consecutive_loss_reduced_size_multiplier", 0.65))),
                "consecutive_loss_reduced",
            )
    if bool(getattr(risk, "monthly_drawdown_control_enabled", False)):
        if monthly_drawdown >= float(risk.monthly_drawdown_stop_pct):
            return 0, 0.0, "monthly_stop"
        if monthly_drawdown >= float(risk.monthly_drawdown_one_position_pct):
            return 1, min(base_size_multiplier, float(risk.reduced_size_multiplier)), "monthly_one_position"
        if monthly_drawdown >= float(risk.monthly_drawdown_half_size_pct):
            return base_max_positions, min(base_size_multiplier, base_size_multiplier * 0.5), "monthly_half_size"
    reduce = (
        drawdown >= float(risk.drawdown_reduce_pct)
        or month_pnl <= -abs(float(risk.monthly_loss_reduce_pct)) * starting_equity
        or day_pnl <= -abs(float(risk.daily_loss_reduce_pct)) * starting_equity
    )
    if not reduce:
        return base_max_positions, base_size_multiplier, "base"
    return (
        max(1, min(base_max_positions, int(risk.reduced_max_positions))),
        min(base_size_multiplier, float(risk.reduced_size_multiplier)),
        "reduced",
    )


def _vbp_quality_mode_active(config: Any, control: VbpRuntimeControl, timestamp: Any) -> bool:
    risk = config.vbp_strategy.risk_control
    if not bool(getattr(risk, "consecutive_loss_quality_filter_enabled", False)):
        return False
    return control.loss_quality_until is not None and timestamp < control.loss_quality_until


def _vbp_candle_close_position(candle: Candle) -> float:
    return (candle.close - candle.low) / max(candle.high - candle.low, 1e-12)


def _vbp_market_allows(
    config: Any,
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
) -> tuple[bool, str]:
    market = config.vbp_strategy.market_filter
    if not bool(market.enabled):
        return True, "ok"
    btc = execution_candles_by_symbol.get("BTCUSDT", [])
    if execution_index >= len(btc):
        return False, "reject_vbp_missing_btc"
    btc_ret_15m = _vbp_return(btc, execution_index, 15)
    if btc_ret_15m <= float(market.btc_15m_drop_block):
        return False, "reject_vbp_btc_15m_drop"
    btc_ret_1h = _vbp_return(btc, execution_index, 60)
    if btc_ret_1h <= float(market.btc_1h_drop_block):
        return False, "reject_vbp_btc_1h_drop"
    if bool(market.btc_1h_ema_bear_block_enabled) and execution_index >= 60:
        one_h = _resample_to_timeframe(btc[max(0, execution_index - 180):execution_index + 1], "1m", "1h")
        if len(one_h) >= 25:
            closes = [item.close for item in one_h]
            ema9 = ema(closes, 9)
            ema21 = ema(closes, 21)
            if closes[-1] < ema21[-1] and ema9[-1] < ema21[-1]:
                return False, "reject_vbp_btc_1h_ema_bear"
    if bool(market.breadth_enabled):
        breadth = _vbp_market_breadth(execution_candles_by_symbol, execution_index)
        if breadth["up_15m"] < float(market.breadth_min_15m_up_pct):
            return False, "reject_vbp_breadth_15m"
        if breadth["up_1h"] < float(market.breadth_min_1h_up_pct):
            return False, "reject_vbp_breadth_1h"
        if breadth["above_ema21"] < float(market.breadth_min_above_ema21_pct):
            return False, "reject_vbp_breadth_ema21"
    return True, "ok"


def _vbp_return(candles: list[Candle], index: int, bars: int) -> float:
    if index < bars or index >= len(candles):
        return 0.0
    previous = candles[index - bars].close
    return candles[index].close / max(previous, 1e-12) - 1.0


def _vbp_market_breadth(candles_by_symbol: dict[str, list[Candle]], index: int) -> dict[str, float]:
    up_15m = 0
    up_1h = 0
    above_ema21 = 0
    total = 0
    for candles in candles_by_symbol.values():
        if index >= len(candles) or index < 60:
            continue
        total += 1
        if candles[index].close > candles[index - 15].close:
            up_15m += 1
        if candles[index].close > candles[index - 60].close:
            up_1h += 1
        closes = [item.close for item in candles[index - 20:index + 1]]
        ema21 = ema(closes, 21)
        if closes[-1] > ema21[-1]:
            above_ema21 += 1
    denominator = max(total, 1)
    return {
        "up_15m": up_15m / denominator,
        "up_1h": up_1h / denominator,
        "above_ema21": above_ema21 / denominator,
    }


def _vbp_relative_strength_allows(
    config: Any,
    execution_candles_by_symbol: dict[str, list[Candle]],
    symbol: str,
    index: int,
) -> tuple[bool, str]:
    entry = config.vbp_strategy.entry
    if not bool(getattr(entry, "relative_strength_enabled", False)):
        return True, "ok"
    lookback = max(1, int(getattr(entry, "relative_strength_lookback_minutes", 60)))
    candles = execution_candles_by_symbol.get(symbol, [])
    btc = execution_candles_by_symbol.get("BTCUSDT", [])
    if index >= len(candles) or index >= len(btc) or index < lookback:
        return False, "reject_vbp_relative_strength_missing"
    symbol_ret = candles[index].close / max(candles[index - lookback].close, 1e-12) - 1.0
    btc_ret = btc[index].close / max(btc[index - lookback].close, 1e-12) - 1.0
    min_vs_btc = float(getattr(entry, "relative_strength_min_vs_btc_pct", 0.0))
    if symbol_ret - btc_ret < min_vs_btc:
        return False, "reject_vbp_relative_strength_btc"

    min_vs_market = float(getattr(entry, "relative_strength_min_vs_market_pct", -1.0))
    max_rank_pct = float(getattr(entry, "relative_strength_max_rank_pct", 1.0))
    if min_vs_market <= -0.999 and max_rank_pct >= 0.999:
        return True, "ok"

    returns: list[float] = []
    for market_symbol in _vbp_symbols(config):
        market_candles = execution_candles_by_symbol.get(market_symbol, [])
        if index < lookback or index >= len(market_candles):
            continue
        previous = market_candles[index - lookback].close
        if previous > 0:
            returns.append(market_candles[index].close / previous - 1.0)
    if len(returns) < 10:
        return False, "reject_vbp_relative_strength_market_missing"
    returns.sort()
    median_ret = returns[len(returns) // 2]
    if symbol_ret - median_ret < min_vs_market:
        return False, "reject_vbp_relative_strength_market"
    rank_pct = sum(1 for value in returns if value > symbol_ret) / max(1, len(returns))
    if rank_pct > max_rank_pct:
        return False, "reject_vbp_relative_strength_rank"
    return True, "ok"


def rank_surge_detector(
    volumes_now: dict[str, float],
    volumes_previous: dict[str, float],
    rank_jump: int = 50,
) -> set[str]:
    current_rank = {
        symbol: rank
        for rank, symbol in enumerate(sorted(volumes_now, key=lambda item: volumes_now[item], reverse=True), start=1)
    }
    previous_rank = {
        symbol: rank
        for rank, symbol in enumerate(sorted(volumes_previous, key=lambda item: volumes_previous[item], reverse=True), start=1)
    }
    return {
        symbol
        for symbol, rank in current_rank.items()
        if previous_rank.get(symbol, len(previous_rank) + 1) - rank >= rank_jump
    }


def rvol_calculator(candles: list[Candle], index: int, lookback_days: int = 20) -> float:
    if index < 60:
        return 0.0
    current_volume = _vbp_quote_volume(candles[index - 59:index + 1])
    lookback_minutes = max(60, int(lookback_days) * 24 * 60)
    start = max(0, index - lookback_minutes)
    previous = candles[start:index - 59]
    if len(previous) < 60:
        return 0.0
    hourly_volumes = [
        _vbp_quote_volume(previous[i:i + 60])
        for i in range(0, len(previous) - 59, 60)
    ]
    average = sum(hourly_volumes) / max(len(hourly_volumes), 1)
    return current_volume / max(average, 1e-12)


def in_consolidation_zone(
    candles: list[Candle],
    index: int,
    n_bars: int,
    threshold_pct: float = 0.02,
) -> tuple[bool, float, float]:
    if index < n_bars:
        return False, 0.0, 0.0
    window = candles[index - n_bars:index]
    top = max(candle.high for candle in window)
    bottom = min(candle.low for candle in window)
    close = candles[index].close
    if top <= bottom or close <= 0:
        return False, top, bottom
    near_top = (top - close) / close <= threshold_pct
    inside = bottom <= close <= top * (1.0 + threshold_pct)
    return inside and near_top, top, bottom


def funding_rate_filter(funding_rate: float, max_rate: float = 0.0001) -> bool:
    return funding_rate <= max_rate


def not_at_daily_high(
    candles: list[Candle],
    index: int,
    lookback_days: int = 90,
    zone_pct: float = 0.9,
) -> bool:
    lookback = max(1, int(lookback_days) * 1440)
    start = max(0, index - lookback)
    window = candles[start:index + 1]
    if len(window) < 7 * 1440:
        return False
    high = max(candle.high for candle in window)
    return candles[index].close < high * zone_pct


def _run_vbp_scan_1m(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    monthly_stats: dict[str, dict[str, Any]],
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
    signal_index: int,
    timestamp: Any,
    client: HistoricalClient,
    execution_config: BacktestExecutionConfig,
    watch_until: dict[str, Any],
    breakouts: dict[str, VbpBreakoutState],
    feature_cache: dict[str, VbpSymbolFeatures],
    control: VbpRuntimeControl,
    starting_equity: float,
    equity: float,
    stats: dict[str, int],
    portfolio_symbol_cooldown_until: dict[str, Any],
    portfolio_control_stats: dict[str, int],
    portfolio_runtime: PortfolioRuntimeControl,
    alpha_diagnostics: AlphaCandidateDiagnostics,
) -> float:
    vbp = config.vbp_strategy
    market_allowed, market_reason = _vbp_market_allows(config, execution_candles_by_symbol, execution_index)
    if not market_allowed:
        _vbp_count(stats, market_reason)
        return cash
    risk_allowed, risk_reason = _vbp_risk_allows(config, control, timestamp, starting_equity)
    if not risk_allowed:
        _vbp_count(stats, risk_reason)
        return cash
    effective_max_positions, effective_size_multiplier, exposure_mode = _vbp_dynamic_exposure(config, control, timestamp, starting_equity, equity)
    if effective_max_positions <= 0 or effective_size_multiplier <= 0:
        _vbp_count(stats, f"reject_vbp_{exposure_mode}")
        return cash
    if exposure_mode == "reduced":
        _vbp_count(stats, "adaptive_exposure_reduced_count")
    elif exposure_mode != "base":
        _vbp_count(stats, f"adaptive_exposure_{exposure_mode}_count")
    if _vbp_open_positions(positions) >= effective_max_positions:
        _vbp_count(stats, "reject_max_vbp_positions")
        return cash
    if len(positions) >= _entry_position_limit(config):
        _vbp_count(stats, "reject_global_position_limit")
        return cash

    volumes_now, volumes_previous = _vbp_rank_volumes(feature_cache, execution_index, int(vbp.universe.rank_window_minutes))
    rank_surges = rank_surge_detector(volumes_now, volumes_previous, int(vbp.universe.rank_jump_threshold))
    opened = 0
    max_new = max(1, int(config.trading.max_new_entries_per_cycle))
    entry_symbols = set(config.trading.entry_symbols or config.trading.symbols)
    vbp_symbols = set(_vbp_symbols(config))
    candidates = []
    for symbol in _vbp_symbols(config):
        if not _point_in_time_symbol_allowed(config, symbol, timestamp):
            continue
        if symbol not in vbp_symbols or symbol not in entry_symbols or symbol not in execution_candles_by_symbol:
            continue
        if _vbp_symbol_on_cooldown(control, symbol, timestamp):
            _vbp_count(stats, "reject_symbol_cooldown")
            continue
        if symbol in positions:
            continue
        candles = execution_candles_by_symbol[symbol]
        if execution_index >= len(candles):
            continue
        if _vbp_frequency_rejects(config, positions, candles, execution_index, symbol):
            _vbp_count(stats, "reject_vbp_frequency_quality")
            continue
        rvol_1h = _vbp_cached_feature(feature_cache, symbol, "rvol_1h", execution_index)
        if symbol in rank_surges or rvol_1h >= float(vbp.universe.rvol_entry_threshold):
            watch_until[symbol] = timestamp + timedelta(minutes=max(1, int(vbp.universe.watchlist_ttl_minutes)))
            _vbp_count(stats, "watchlist_added")
        if watch_until.get(symbol) and timestamp <= watch_until[symbol]:
            score = volumes_now.get(symbol, 0.0) * max(rvol_1h, 1.0)
            candidates.append((score, symbol, rvol_1h))
    candidates.sort(reverse=True)

    for _, symbol, rvol_1h in candidates:
        if opened >= max_new:
            break
        if _vbp_open_positions(positions) >= effective_max_positions:
            break
        if len(positions) >= _entry_position_limit(config):
            break
        cash, did_open = _vbp_process_symbol(
            trader,
            config,
            cash,
            positions,
            monthly_stats,
            execution_candles_by_symbol,
            execution_index,
            signal_index,
            timestamp,
            client,
            execution_config,
            breakouts,
            stats,
            symbol,
            rvol_1h,
            feature_cache,
            effective_size_multiplier,
            control,
            portfolio_symbol_cooldown_until,
            portfolio_control_stats,
            portfolio_runtime,
            alpha_diagnostics,
        )
        if did_open:
            opened += 1
    return cash


def _vbp_process_symbol(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    monthly_stats: dict[str, dict[str, Any]],
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
    signal_index: int,
    timestamp: Any,
    client: HistoricalClient,
    execution_config: BacktestExecutionConfig,
    breakouts: dict[str, VbpBreakoutState],
    stats: dict[str, int],
    symbol: str,
    rvol_1h: float,
    feature_cache: dict[str, VbpSymbolFeatures],
    effective_size_multiplier: float,
    control: VbpRuntimeControl,
    portfolio_symbol_cooldown_until: dict[str, Any],
    portfolio_control_stats: dict[str, int],
    portfolio_runtime: PortfolioRuntimeControl,
    alpha_diagnostics: AlphaCandidateDiagnostics,
) -> tuple[float, bool]:
    vbp = config.vbp_strategy
    candles = execution_candles_by_symbol[symbol]
    candle = candles[execution_index]
    pending = breakouts.get(symbol)
    if pending is not None:
        return _vbp_process_pending(
            trader,
            config,
            cash,
            positions,
            monthly_stats,
            execution_candles_by_symbol,
            execution_index,
            signal_index,
            timestamp,
            client,
            execution_config,
            breakouts,
            stats,
            pending,
            effective_size_multiplier,
            control,
            portfolio_symbol_cooldown_until,
            portfolio_control_stats,
            portfolio_runtime,
            alpha_diagnostics,
        )

    structure = vbp.structure_filter
    features = feature_cache.get(symbol)
    ok_zone = bool(features and execution_index < len(features.consolidation_ok) and features.consolidation_ok[execution_index])
    zone_top = features.consolidation_top[execution_index] if features and execution_index < len(features.consolidation_top) else 0.0
    zone_bottom = features.consolidation_bottom[execution_index] if features and execution_index < len(features.consolidation_bottom) else 0.0
    if not ok_zone:
        _vbp_count(stats, "reject_not_consolidating_near_top")
        return cash, False
    funding_rate = float(getattr(config.risk, "funding_default_rate", 0.0))
    if not funding_rate_filter(funding_rate, float(structure.funding_rate_max)):
        _vbp_count(stats, "reject_funding_rate")
        return cash, False
    daily_ok = bool(features and execution_index < len(features.daily_high_ok) and features.daily_high_ok[execution_index])
    if not daily_ok:
        _vbp_count(stats, "reject_daily_high_zone")
        return cash, False
    rs_allowed, rs_reason = _vbp_relative_strength_allows(config, execution_candles_by_symbol, symbol, execution_index)
    if not rs_allowed:
        _vbp_count(stats, rs_reason)
        return cash, False

    candle_rvol = _vbp_cached_feature(feature_cache, symbol, "rvol_1m", execution_index)
    quality_mode = _vbp_quality_mode_active(config, control, timestamp)
    rvol_threshold = float(vbp.universe.rvol_trigger_threshold)
    if quality_mode:
        rvol_threshold *= max(1.0, float(getattr(vbp.risk_control, "consecutive_loss_quality_rvol_multiplier", 1.25)))
    if candle_rvol < rvol_threshold:
        _vbp_count(stats, "reject_no_breakout_rvol")
        return cash, False
    if quality_mode:
        min_close_position = float(getattr(vbp.risk_control, "consecutive_loss_quality_min_close_position", 0.65))
        if _vbp_candle_close_position(candle) < min_close_position:
            _vbp_count(stats, "reject_vbp_quality_close_position")
            return cash, False
    previous_close = candles[execution_index - 1].close if execution_index > 0 else candle.open
    if not (previous_close <= zone_top and candle.close > zone_top):
        _vbp_count(stats, "reject_no_zone_breakout")
        return cash, False

    compression = _vbp_compression_metrics(candles, execution_index, zone_top, zone_bottom, structure)
    breakout = _vbp_breakout_metrics(candle, zone_top, candle_rvol, compression.atr_value)
    alpha_event_id = alpha_diagnostics.record_vbp_breakout(
        symbol,
        candles,
        execution_index,
        zone_top,
        zone_bottom,
        candle_rvol,
        decision_time=timestamp,
        compression_metrics=compression.diagnostic_fields(),
        breakout_metrics=breakout.diagnostic_fields(),
    )
    compression_allowed, compression_reason = _vbp_compression_allows(structure, compression)
    if not compression_allowed:
        alpha_diagnostics.mark_vbp(alpha_event_id, "rejected", compression_reason)
        _vbp_count(stats, compression_reason)
        return cash, False
    breakout_allowed, breakout_reason = _vbp_breakout_quality_allows(vbp.entry, breakout)
    if not breakout_allowed:
        alpha_diagnostics.mark_vbp(alpha_event_id, "rejected", breakout_reason)
        _vbp_count(stats, breakout_reason)
        return cash, False

    pullback_target = zone_top
    if bool(vbp.entry.use_vwap_as_pullback_target):
        pullback_target = max(zone_top, _vbp_vwap(candles[max(0, execution_index - int(structure.consolidation_bars)):execution_index + 1]))
    tp2_price = _vbp_tp2_price(candles, execution_index, candle.close, zone_bottom, float(vbp.exit.tp1_rr_ratio))
    breakouts[symbol] = VbpBreakoutState(
        symbol=symbol,
        breakout_index=execution_index,
        breakout_time=timestamp,
        breakout_level=zone_top,
        consolidation_bottom=zone_bottom,
        pullback_target=pullback_target,
        breakout_volume=candle.volume,
        breakout_close=candle.close,
        tp2_price=tp2_price,
        breakout_atr=compression.atr_value,
        pullback_low=candle.close,
        alpha_event_id=alpha_event_id,
    )
    _vbp_count(stats, "breakout_detected")
    return cash, False


def _vbp_compression_metrics(
    candles: list[Candle],
    index: int,
    zone_top: float,
    zone_bottom: float,
    config: Any,
) -> VbpCompressionMetrics:
    atr_lookback = max(20, int(getattr(config, "compression_atr_lookback_bars", 80)))
    recent_volume_bars = max(1, int(getattr(config, "compression_volume_recent_bars", 12)))
    baseline_volume_bars = max(1, int(getattr(config, "compression_volume_baseline_bars", 48)))
    prior_move_bars = max(1, int(getattr(config, "compression_prior_move_lookback_bars", 30)))
    history_bars = max(atr_lookback + 30, recent_volume_bars + baseline_volume_bars, prior_move_bars + 2)
    previous = candles[max(0, index - history_bars):index]
    atr_values = atr(previous, 14)
    atr_value = max(atr_values[-1] if atr_values else 0.0, 1e-12)
    atr_history = atr_values[max(0, len(atr_values) - atr_lookback - 1):-1]
    atr_percentile = sum(value <= atr_value for value in atr_history) / max(1, len(atr_history))

    recent_volume = previous[-recent_volume_bars:]
    baseline_end = max(0, len(previous) - len(recent_volume))
    baseline_start = max(0, baseline_end - baseline_volume_bars)
    baseline_volume = previous[baseline_start:baseline_end]
    recent_average = sum(candle.volume for candle in recent_volume) / max(1, len(recent_volume))
    baseline_average = sum(candle.volume for candle in baseline_volume) / max(1, len(baseline_volume))
    volume_contraction = recent_average / max(baseline_average, 1e-12)

    if len(previous) > prior_move_bars:
        prior_move = max(0.0, previous[-1].close - previous[-1 - prior_move_bars].close) / atr_value
    else:
        prior_move = 0.0
    return VbpCompressionMetrics(
        atr_percentile=atr_percentile,
        range_atr=max(0.0, zone_top - zone_bottom) / atr_value,
        volume_contraction=volume_contraction,
        prior_move_atr=prior_move,
        atr_value=atr_value,
    )


def _vbp_compression_allows(config: Any, metrics: VbpCompressionMetrics) -> tuple[bool, str]:
    if not bool(getattr(config, "compression_quality_enabled", False)):
        return True, "ok"
    if metrics.atr_percentile > float(getattr(config, "compression_atr_percentile_max", 1.0)):
        return False, "reject_compression_atr_percentile"
    if metrics.range_atr > float(getattr(config, "compression_range_atr_max", 999.0)):
        return False, "reject_compression_range_atr"
    if metrics.volume_contraction > float(getattr(config, "compression_volume_contraction_max", 999.0)):
        return False, "reject_compression_volume_contraction"
    if metrics.prior_move_atr > float(getattr(config, "compression_prior_move_max_atr", 999.0)):
        return False, "reject_compression_prior_extension"
    return True, "ok"


def _vbp_breakout_metrics(
    candle: Candle,
    breakout_level: float,
    volume_ratio: float,
    atr_value: float,
) -> VbpBreakoutMetrics:
    normalized_atr = max(atr_value, 1e-12)
    candle_range = max(candle.high - candle.low, 1e-12)
    return VbpBreakoutMetrics(
        distance_atr=max(0.0, candle.close - breakout_level) / normalized_atr,
        body_atr=abs(candle.close - candle.open) / normalized_atr,
        close_position=(candle.close - candle.low) / candle_range,
        upper_wick_ratio=(candle.high - max(candle.open, candle.close)) / candle_range,
        volume_ratio=volume_ratio,
    )


def _vbp_breakout_quality_allows(config: Any, metrics: VbpBreakoutMetrics) -> tuple[bool, str]:
    if not bool(getattr(config, "breakout_quality_enabled", False)):
        return True, "ok"
    if metrics.distance_atr < float(getattr(config, "breakout_distance_atr_min", 0.0)):
        return False, "reject_breakout_distance_too_small"
    if metrics.distance_atr > float(getattr(config, "breakout_distance_atr_max", 999.0)):
        return False, "reject_breakout_distance_too_large"
    if metrics.body_atr < float(getattr(config, "breakout_body_atr_min", 0.0)):
        return False, "reject_breakout_body_atr"
    if metrics.close_position < float(getattr(config, "breakout_close_position_min", 0.0)):
        return False, "reject_breakout_close_position"
    if metrics.upper_wick_ratio > float(getattr(config, "breakout_upper_wick_ratio_max", 1.0)):
        return False, "reject_breakout_upper_wick"
    if metrics.volume_ratio < float(getattr(config, "breakout_volume_ratio_min", 0.0)):
        return False, "reject_breakout_volume_ratio"
    return True, "ok"


def _vbp_pending_timing_phase(pending: VbpBreakoutState, execution_index: int) -> str:
    if pending.confirmation_index is not None:
        if execution_index == pending.confirmation_index + 1:
            return "execute"
        if execution_index > pending.confirmation_index + 1:
            return "execution_missed"
        return "wait_confirmation_close"
    if pending.pullback_touch_index is None:
        return "wait_pullback"
    if execution_index <= pending.pullback_touch_index:
        return "wait_after_pullback"
    return "confirm"


def _vbp_pullback_metrics(pending: VbpBreakoutState, execution_index: int) -> VbpPullbackMetrics:
    depth = max(0.0, pending.breakout_close - pending.pullback_low)
    breakout_distance = max(pending.breakout_close - pending.breakout_level, 1e-12)
    return VbpPullbackMetrics(
        depth_atr=depth / max(pending.breakout_atr, 1e-12),
        depth_to_breakout=depth / breakout_distance,
        bars=max(0, execution_index - pending.breakout_index),
        broke_breakout_level=pending.pullback_broke_breakout_level,
    )


def _vbp_pullback_quality_action(config: Any, metrics: VbpPullbackMetrics) -> tuple[str, str]:
    if not bool(getattr(config, "pullback_quality_enabled", False)):
        return "allow", "ok"
    if metrics.depth_atr < float(getattr(config, "pullback_depth_atr_min", 0.0)):
        return "wait", "wait_pullback_depth_too_shallow"
    if metrics.depth_atr > float(getattr(config, "pullback_depth_atr_max", 999.0)):
        return "reject", "reject_pullback_depth_atr"
    if metrics.depth_to_breakout > float(getattr(config, "pullback_depth_to_breakout_max", 999.0)):
        return "reject", "reject_pullback_depth_ratio"
    if metrics.bars > int(getattr(config, "pullback_bars_max", 999)):
        return "reject", "reject_pullback_duration"
    if bool(getattr(config, "pullback_require_hold_breakout_level", False)) and metrics.broke_breakout_level:
        return "reject", "reject_pullback_broke_breakout_level"
    return "allow", "ok"


def _vbp_confirmation_metrics(pending: VbpBreakoutState, candle: Candle) -> VbpConfirmationMetrics:
    candle_range = max(candle.high - candle.low, 1e-12)
    return VbpConfirmationMetrics(
        body_atr=abs(candle.close - candle.open) / max(pending.breakout_atr, 1e-12),
        close_position=_vbp_candle_close_position(candle),
        upper_wick_ratio=(candle.high - max(candle.open, candle.close)) / candle_range,
    )


def _vbp_confirmation_quality_allows(config: Any, metrics: VbpConfirmationMetrics) -> tuple[bool, str]:
    if not bool(getattr(config, "confirmation_quality_enabled", False)):
        return True, "ok"
    if metrics.body_atr < float(getattr(config, "confirmation_body_atr_min", 0.0)):
        return False, "wait_confirmation_body_atr"
    if metrics.close_position < float(getattr(config, "confirmation_close_position_min", 0.0)):
        return False, "wait_confirmation_close_position"
    if metrics.upper_wick_ratio > float(getattr(config, "confirmation_upper_wick_ratio_max", 1.0)):
        return False, "wait_confirmation_upper_wick"
    return True, "ok"


def _vbp_process_pending(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    monthly_stats: dict[str, dict[str, Any]],
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
    signal_index: int,
    timestamp: Any,
    client: HistoricalClient,
    execution_config: BacktestExecutionConfig,
    breakouts: dict[str, VbpBreakoutState],
    stats: dict[str, int],
    pending: VbpBreakoutState,
    effective_size_multiplier: float,
    control: VbpRuntimeControl,
    portfolio_symbol_cooldown_until: dict[str, Any],
    portfolio_control_stats: dict[str, int],
    portfolio_runtime: PortfolioRuntimeControl,
    alpha_diagnostics: AlphaCandidateDiagnostics,
) -> tuple[float, bool]:
    vbp = config.vbp_strategy
    candles = execution_candles_by_symbol[pending.symbol]
    candle = candles[execution_index]
    age = execution_index - pending.breakout_index
    if age <= 0:
        return cash, False
    timing_phase = _vbp_pending_timing_phase(pending, execution_index)
    if timing_phase == "execution_missed":
        alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", "confirmation_execution_missed")
        breakouts.pop(pending.symbol, None)
        _vbp_count(stats, "confirmation_execution_missed")
        return cash, False
    execute_ready = timing_phase == "execute"
    if not execute_ready:
        alpha_diagnostics.update_vbp_pullback(
            pending.alpha_event_id,
            candle,
            age,
            pending.breakout_close,
            pending.breakout_volume,
        )
    if age > int(vbp.entry.timeout_bars) and not execute_ready:
        alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", "pending_timeout")
        breakouts.pop(pending.symbol, None)
        _vbp_count(stats, "pending_timeout")
        return cash, False

    quality_mode = _vbp_quality_mode_active(config, control, timestamp)
    if not execute_ready:
        pending.pullback_low = min(pending.pullback_low or candle.low, candle.low)
        pending.pullback_broke_breakout_level = bool(
            pending.pullback_broke_breakout_level or candle.close < pending.breakout_level
        )
        if candle.close < pending.consolidation_bottom:
            alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", "pending_failed_back_inside")
            breakouts.pop(pending.symbol, None)
            _vbp_count(stats, "pending_failed_back_inside")
            return cash, False

        pullback_volume_ratio = float(vbp.entry.pullback_volume_ratio)
        if quality_mode:
            pullback_volume_ratio = min(
                pullback_volume_ratio,
                float(getattr(vbp.risk_control, "consecutive_loss_quality_pullback_volume_ratio", 0.30)),
            )
        pullback_volume_ok = candle.volume <= pending.breakout_volume * pullback_volume_ratio
        touched_target = candle.low <= pending.pullback_target
        if touched_target and not pullback_volume_ok:
            alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", "pending_reject_high_volume_pullback")
            breakouts.pop(pending.symbol, None)
            _vbp_count(stats, "pending_reject_high_volume_pullback")
            return cash, False
        if touched_target and pending.pullback_touch_index is None:
            pending.touched_pullback = True
            pending.pullback_touch_index = execution_index
            _vbp_count(stats, "pullback_touched")
            return cash, False
        if _vbp_pending_timing_phase(pending, execution_index) != "confirm":
            return cash, False
        pullback_metrics = _vbp_pullback_metrics(pending, execution_index)
        pullback_action, pullback_reason = _vbp_pullback_quality_action(vbp.entry, pullback_metrics)
        if pullback_action == "wait":
            _vbp_count(stats, pullback_reason)
            return cash, False
        if pullback_action == "reject":
            alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", pullback_reason)
            breakouts.pop(pending.symbol, None)
            _vbp_count(stats, pullback_reason)
            return cash, False
        if not (candle.close > candle.open and candle.close >= pending.pullback_target):
            _vbp_count(stats, "pending_wait_bull_reclaim")
            return cash, False
        confirmation_metrics = _vbp_confirmation_metrics(pending, candle)
        confirmation_allowed, confirmation_reason = _vbp_confirmation_quality_allows(
            vbp.entry,
            confirmation_metrics,
        )
        if not confirmation_allowed:
            _vbp_count(stats, confirmation_reason)
            return cash, False
        if quality_mode:
            min_close_position = float(getattr(vbp.risk_control, "consecutive_loss_quality_min_close_position", 0.65))
            if _vbp_candle_close_position(candle) < min_close_position:
                alpha_diagnostics.mark_vbp(
                    pending.alpha_event_id,
                    "waiting_confirmation",
                    "pending_reject_vbp_quality_close_position",
                )
                _vbp_count(stats, "pending_reject_vbp_quality_close_position")
                return cash, False
        rs_allowed, rs_reason = _vbp_relative_strength_allows(
            config,
            execution_candles_by_symbol,
            pending.symbol,
            execution_index,
        )
        if not rs_allowed:
            alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", rs_reason)
            breakouts.pop(pending.symbol, None)
            _vbp_count(stats, rs_reason)
            return cash, False
        pending.confirmation_index = execution_index
        pending.confirmation_price = candle.close
        pending.confirmation_time = candle.timestamp + timedelta(minutes=1)
        pending.confirmation_body_atr = confirmation_metrics.body_atr
        pending.confirmation_close_position = confirmation_metrics.close_position
        pending.confirmation_wick_ratio = confirmation_metrics.upper_wick_ratio
        pending.confirmation_volume_ratio = candle.volume / max(pending.breakout_volume, 1e-12)
        alpha_diagnostics.mark_vbp(
            pending.alpha_event_id,
            "pullback_confirmed",
            confirmation_time=pending.confirmation_time.isoformat(),
            confirmation_body_atr=pending.confirmation_body_atr,
            confirmation_close_position=pending.confirmation_close_position,
            confirmation_wick_ratio=pending.confirmation_wick_ratio,
            confirmation_volume_ratio=pending.confirmation_volume_ratio,
        )
        _vbp_count(stats, "confirmation_closed")
        _vbp_count(stats, "entry_deferred_to_next_open")
        return cash, False

    if candle.open < pending.consolidation_bottom:
        alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", "pending_gap_below_structure")
        breakouts.pop(pending.symbol, None)
        _vbp_count(stats, "pending_gap_below_structure")
        return cash, False

    if pending.confirmation_index is None or pending.confirmation_price is None or pending.confirmation_time is None:
        alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", "reject_missing_confirmation_price")
        breakouts.pop(pending.symbol, None)
        _vbp_count(stats, "reject_missing_confirmation_price")
        return cash, False

    # The confirmation candle is closed. Execution is now allowed at the next
    # 1m open; the current candle's high/low/close are not signal inputs.
    confirmed_price = pending.confirmation_price
    entry_price = max(candle.open, confirmed_price)
    entry_chase_atr = (entry_price - pending.breakout_level) / max(pending.breakout_atr, 1e-12)
    if entry_chase_atr > float(getattr(vbp.entry, "max_entry_chase_atr", 999.0)):
        alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", "reject_entry_chase_atr")
        breakouts.pop(pending.symbol, None)
        _vbp_count(stats, "reject_entry_chase_atr")
        return cash, False

    portfolio_allowed, portfolio_reason, portfolio_multiplier = _portfolio_entry_decision(
        config,
        pending.symbol,
        "vbp",
        positions,
        [],
        execution_candles_by_symbol,
        execution_index,
        timestamp,
        portfolio_symbol_cooldown_until,
        _mark_equity(cash, positions, execution_candles_by_symbol, execution_index),
        portfolio_runtime,
    )
    if not portfolio_allowed:
        alpha_diagnostics.mark_vbp(
            pending.alpha_event_id,
            "blocked_by_portfolio",
            portfolio_reason,
        )
        _portfolio_count(portfolio_control_stats, portfolio_reason)
        _vbp_count(stats, f"reject_{portfolio_reason}")
        return cash, False

    # For a long entry, waiting must not award a better price than the close
    # that confirmed the signal.  Adverse gaps are retained; favorable gaps
    # are ignored before the normal pessimistic slippage model is applied.
    stop_price = max(pending.consolidation_bottom, entry_price * (1.0 - float(vbp.exit.stop_loss_pct)))
    stop_loss_pct = (entry_price - stop_price) / max(entry_price, 1e-12)
    if stop_loss_pct <= 0:
        alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", "reject_invalid_stop")
        breakouts.pop(pending.symbol, None)
        _vbp_count(stats, "reject_invalid_stop")
        return cash, False
    risk = entry_price - stop_price
    tp1_price = entry_price + risk * float(vbp.exit.tp1_rr_ratio)
    tp2_price = max(pending.tp2_price, tp1_price + risk)
    take_profit_pct = max((tp2_price - entry_price) / max(entry_price, 1e-12), stop_loss_pct * float(vbp.exit.tp1_rr_ratio))
    reason = (
        "vbp_volume_breakout_pullback "
        f"level={pending.breakout_level:.8g} bottom={pending.consolidation_bottom:.8g} "
        f"target={pending.pullback_target:.8g} "
        f"tp1={tp1_price:.8g} tp1_ratio={float(vbp.exit.tp1_close_ratio):.3f} stop={stop_loss_pct * 100:.3f}%"
        + (" vbp_quality_mode=1" if quality_mode else "")
        + (f" alpha_event_id={pending.alpha_event_id}" if pending.alpha_event_id else "")
    )
    signal = Signal(
        Direction.LONG,
        0.75,
        reason,
        stop_loss_pct,
        take_profit_pct,
        risk_multiplier=float(effective_size_multiplier) * portfolio_multiplier,
        max_holding_bars=max(1, int(vbp.entry.timeout_bars) * 4),
    )
    account = _account_snapshot(
        config,
        _mark_equity(cash, positions, execution_candles_by_symbol, execution_index),
        positions,
        execution_candles_by_symbol,
        execution_index,
    )
    quantity_text, size_reason = trader._size_order(pending.symbol, entry_price, signal, account)
    quantity = float(quantity_text)
    if size_reason != "ok" or quantity <= 0:
        alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", f"reject_size_{size_reason}")
        _vbp_count(stats, f"reject_size_{size_reason}")
        return cash, False
    execution_candle = replace(candle, open=entry_price, high=entry_price, low=entry_price, close=entry_price)
    before = pending.symbol in positions
    cash = _open_position(
        config,
        cash,
        positions,
        pending.symbol,
        signal,
        execution_candle,
        quantity,
        signal_index,
        raw_entry_price=entry_price,
        entry_time=timestamp,
        signal_time=pending.breakout_time,
        signal_available_time=timestamp,
        execution_config=execution_config,
        rules=client.symbol_rules(pending.symbol),
    )
    breakouts.pop(pending.symbol, None)
    if not before and pending.symbol in positions:
        alpha_diagnostics.mark_vbp(
            pending.alpha_event_id,
            "traded",
            entry_time=timestamp.isoformat(),
            anchor_timestamp=timestamp.isoformat(),
            anchor_price=entry_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            target_to_full_cost_ratio=take_profit_pct / max(alpha_diagnostics.full_round_trip_cost_pct, 1e-12),
            stop_to_full_cost_ratio=stop_loss_pct / max(alpha_diagnostics.stop_round_trip_cost_pct, 1e-12),
            confirmation_body_atr=pending.confirmation_body_atr,
            confirmation_close_position=pending.confirmation_close_position,
            confirmation_wick_ratio=pending.confirmation_wick_ratio,
            confirmation_volume_ratio=pending.confirmation_volume_ratio,
            entry_chase_distance_atr=entry_chase_atr,
        )
        _record_monthly_open(monthly_stats, timestamp, Direction.LONG)
        _vbp_count(stats, "entry_count")
        risk = config.vbp_strategy.risk_control
        if bool(risk.enabled) and bool(getattr(risk, "frequency_control_enabled", False)):
            cooldown_minutes = max(0, int(risk.symbol_entry_cooldown_minutes))
            if cooldown_minutes > 0:
                if control.symbol_cooldown_until is not None:
                    control.symbol_cooldown_until[pending.symbol] = timestamp + timedelta(minutes=cooldown_minutes)
                _vbp_count(stats, "symbol_entry_cooldown_count")
        return cash, True
    _vbp_count(stats, "reject_open_failed")
    alpha_diagnostics.mark_vbp(pending.alpha_event_id, "rejected", "reject_open_failed")
    return cash, False


def _update_vbp_partial_exit(
    config: Any,
    cash: float,
    position: PortfolioPosition,
    trades: list[dict[str, Any]],
    candle: Candle,
    signal_index: int,
    execution_config: BacktestExecutionConfig,
    rules: Any,
    stats: dict[str, int],
) -> float:
    if " vbp_tp1_done=1" in position.entry_reason:
        return cash
    tp1_price = _vbp_reason_float(str(position.entry_reason), "tp1", 0.0)
    if tp1_price <= 0 or candle.high < tp1_price:
        return cash
    close_ratio = max(0.0, min(1.0, _vbp_reason_float(str(position.entry_reason), "tp1_ratio", 0.5)))
    if close_ratio <= 0 or close_ratio >= 1.0:
        return cash
    cash = _close_vbp_partial_position(
        config,
        cash,
        position,
        trades,
        tp1_price,
        close_ratio,
        "vbp_tp1_partial",
        signal_index,
        candle.timestamp,
        execution_config,
        rules,
    )
    if bool(config.vbp_strategy.exit.trailing_stop_after_tp1):
        position.stop_price = max(position.stop_price, position.entry_price)
    position.entry_reason += " vbp_tp1_done=1"
    _vbp_count(stats, "tp1_partial_count")
    return cash


def _vbp_dynamic_exit_reason(
    config: Any,
    position: PortfolioPosition,
    candle: Candle,
    stats: dict[str, int],
    recent_1m_candles: list[Candle] | None = None,
) -> str | None:
    risk = config.vbp_strategy.risk_control
    if not bool(risk.enabled):
        return None
    exit_config = config.vbp_strategy.exit
    current_profit = (candle.close - position.entry_price) / max(position.entry_price, 1e-12)
    peak_profit = (position.best_price - position.entry_price) / max(position.entry_price, 1e-12)
    pullback = max(0.0, (position.best_price - candle.close) / max(position.entry_price, 1e-12))
    minimum_profit = _vbp_minimum_profit_exit_pct(config)
    tp1_done = " vbp_tp1_done=1" in position.entry_reason

    if bool(getattr(exit_config, "peak_giveback_enabled", True)):
        trigger = max(0.0, float(getattr(exit_config, "peak_giveback_trigger_pct", 0.008)))
        floor = float(getattr(exit_config, "peak_giveback_floor_pct", 0.0015))
        retrace = max(0.0, float(getattr(exit_config, "peak_giveback_retrace_pct", 0.005)))
        pre_tp1_trigger = max(0.0, float(getattr(exit_config, "peak_giveback_pre_tp1_trigger_pct", 0.020)))
        current_profit_ok = current_profit >= minimum_profit
        post_tp1_giveback = tp1_done and peak_profit >= trigger and (current_profit <= floor or pullback >= retrace)
        pre_tp1_large_giveback = (
            not tp1_done
            and peak_profit >= pre_tp1_trigger
            and pullback >= retrace
        )
        if current_profit_ok and (post_tp1_giveback or pre_tp1_large_giveback):
            _vbp_count(stats, "peak_giveback_exit_count")
            tag = "vbp_peak_giveback" if tp1_done else "vbp_peak_giveback_pre_tp1"
            return (
                f"{tag} peak={peak_profit * 100:.3f}% "
                f"now={current_profit * 100:.3f}% pullback={pullback * 100:.3f}%"
            )

    if bool(getattr(exit_config, "large_bear_exit_enabled", True)) and recent_1m_candles:
        min_peak = float(getattr(exit_config, "large_bear_min_peak_profit_pct", 0.006))
        min_current = float(getattr(exit_config, "large_bear_min_current_profit_pct", -0.001))
        if peak_profit >= min_peak and current_profit >= min_current:
            bear_reason = _vbp_large_bearish_candle_reason(recent_1m_candles, exit_config)
            if bear_reason:
                _vbp_count(stats, "large_bear_exit_count")
                return (
                    f"vbp_high_volume_bear_exit {bear_reason} "
                    f"peak={peak_profit * 100:.3f}% now={current_profit * 100:.3f}%"
                )

    risk_price = _vbp_risk_price(position)
    mfe_r = position.mfe / max(risk_price * position.quantity, 1e-12)
    if bool(risk.breakeven_enabled) and " vbp_be=1" not in position.entry_reason:
        if mfe_r >= float(risk.breakeven_trigger_r):
            breakeven_stop = position.entry_price * (1.0 + float(risk.breakeven_offset_pct))
            if position.stop_price < breakeven_stop:
                position.stop_price = min(breakeven_stop, candle.close)
                position.entry_reason += " vbp_be=1"
                _vbp_count(stats, "breakeven_triggered_count")
    if not bool(risk.fail_fast_enabled):
        return None
    hold_minutes = _hold_minutes(position.entry_time, candle.timestamp)
    if hold_minutes >= max(1, int(risk.fail_fast_minutes)) and mfe_r < float(risk.fail_fast_min_mfe_r):
        _vbp_count(stats, "fail_fast_count")
        return f"vbp_fail_fast_mfe mfe_r={mfe_r:.2f}"
    if bool(risk.fail_fast_lost_level_enabled):
        level = _vbp_reason_float(str(position.entry_reason), "level", 0.0)
        if level > 0 and candle.close < level:
            _vbp_count(stats, "fail_fast_lost_level_count")
            return "vbp_fail_fast_lost_breakout_level"
    if bool(risk.fail_fast_lost_vwap_enabled):
        target = _vbp_reason_float(str(position.entry_reason), "target", 0.0)
        if target > 0 and candle.close < target:
            _vbp_count(stats, "fail_fast_lost_vwap_count")
            return "vbp_fail_fast_lost_vwap"
    return None


def _vbp_minimum_profit_exit_pct(config: Any) -> float:
    risk = config.risk
    round_trip_fee = max(0.0, float(getattr(risk, "estimated_fee_bps", 0.0))) * 2.0 / 10_000.0
    slippage = max(0.0, float(getattr(risk, "estimated_slippage_bps", 0.0))) / 10_000.0
    net_buffer = max(0.0, float(getattr(risk, "min_profit_after_cost_pct", 0.0)))
    return round_trip_fee + slippage + net_buffer


def _vbp_large_bearish_candle_reason(candles: list[Candle], exit_config: Any) -> str | None:
    lookback = max(3, int(getattr(exit_config, "large_bear_lookback_bars", 20)))
    if len(candles) < lookback + 1:
        return None
    candle = candles[-1]
    previous = candles[-lookback - 1:-1]
    average_volume = sum(max(item.volume, 0.0) for item in previous) / max(1, len(previous))
    volume_multiplier = candle.volume / max(average_volume, 1e-12)
    candle_range = max(candle.high - candle.low, 1e-12)
    close_position = (candle.close - candle.low) / candle_range
    body_pct = abs(candle.close - candle.open) / max(candle.open, 1e-12)
    if candle.close >= candle.open:
        return None
    if volume_multiplier < float(getattr(exit_config, "large_bear_volume_multiplier", 2.0)):
        return None
    if close_position > float(getattr(exit_config, "large_bear_max_close_position", 0.35)):
        return None
    if body_pct < float(getattr(exit_config, "large_bear_min_body_pct", 0.0015)):
        return None
    return (
        f"vol={volume_multiplier:.2f}x close_pos={close_position:.2f} "
        f"body={body_pct * 100:.3f}%"
    )


def _vbp_risk_price(position: PortfolioPosition) -> float:
    stop_pct = _vbp_reason_float(str(position.entry_reason), "stop", 0.0) / 100.0
    if stop_pct > 0:
        return max(position.entry_price * stop_pct, 1e-12)
    return max(abs(position.entry_price - position.stop_price), 1e-12)


def _close_vbp_partial_position(
    config: Any,
    cash: float,
    position: PortfolioPosition,
    trades: list[dict[str, Any]],
    exit_price: float,
    close_ratio: float,
    reason: str,
    index: int,
    exit_time: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> float:
    close_quantity = position.quantity * close_ratio
    if close_quantity <= 0 or close_quantity >= position.quantity:
        return cash
    fill = market_exit_fill(execution_config, rules, position.direction, close_quantity, exit_price, "take_profit_market")
    executed_exit = fill.price
    raw_entry = position.raw_entry_price or position.entry_price
    raw_gross_pnl = position.direction.value * close_quantity * (exit_price - raw_entry)
    execution_gross_pnl = position.direction.value * close_quantity * (executed_exit - position.entry_price)
    entry_fee = position.entry_fee * close_ratio
    entry_slippage = position.entry_slippage_cost * close_ratio
    fee = entry_fee + fill.fee
    slippage_cost = entry_slippage + fill.slippage_cost
    net_pnl = raw_gross_pnl - fee - slippage_cost
    cash += execution_gross_pnl - fill.fee
    position.quantity -= close_quantity
    position.entry_fee -= entry_fee
    position.entry_slippage_cost -= entry_slippage
    notional = abs(close_quantity * position.entry_price)
    hold_minutes = _hold_minutes(position.entry_time, exit_time)
    strategy_bucket = _strategy_bucket(position.entry_reason)
    trades.append(
        {
            "symbol": position.symbol,
            "strategy": strategy_bucket,
            "side": position.direction.name,
            "direction": position.direction.name,
            "entry_time": position.entry_time.isoformat() if hasattr(position.entry_time, "isoformat") else position.entry_time,
            "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else exit_time,
            "entry_price": position.entry_price,
            "exit_price": executed_exit,
            "raw_entry_price": raw_entry,
            "raw_exit_price": exit_price,
            "qty": close_quantity,
            "quantity": close_quantity,
            "notional": notional,
            "entry_fee": entry_fee,
            "exit_fee": fill.fee,
            "fee": fee,
            "fees": fee,
            "gross_pnl": raw_gross_pnl,
            "execution_gross_pnl": execution_gross_pnl,
            "slippage_cost": slippage_cost,
            "funding": 0.0,
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
        }
    )
    return cash


def _vbp_rank_volumes(
    feature_cache: dict[str, VbpSymbolFeatures],
    index: int,
    window_minutes: int,
) -> tuple[dict[str, float], dict[str, float]]:
    now: dict[str, float] = {}
    previous: dict[str, float] = {}
    window = max(1, int(window_minutes))
    for symbol, features in feature_cache.items():
        if index + 1 >= len(features.quote_prefix):
            continue
        now_start = max(0, index - window + 1)
        prev_end = max(0, now_start)
        prev_start = max(0, prev_end - window)
        now[symbol] = features.quote_prefix[index + 1] - features.quote_prefix[now_start]
        previous[symbol] = features.quote_prefix[prev_end] - features.quote_prefix[prev_start]
    return now, previous


def _build_vbp_feature_cache(config: Any, candles_by_symbol: dict[str, list[Candle]]) -> dict[str, VbpSymbolFeatures]:
    cache: dict[str, VbpSymbolFeatures] = {}
    vbp = config.vbp_strategy
    lookback_minutes = max(60, int(vbp.universe.rvol_lookback_days) * 24 * 60)
    consolidation_bars = max(1, int(vbp.structure_filter.consolidation_bars))
    consolidation_threshold = float(vbp.structure_filter.consolidation_threshold_pct)
    daily_lookback = max(1, int(vbp.structure_filter.daily_high_lookback_days) * 1440)
    daily_zone = float(vbp.structure_filter.daily_high_zone_pct)
    allowed_symbols = set(_vbp_symbols(config))
    for symbol, candles in candles_by_symbol.items():
        if symbol not in allowed_symbols:
            continue
        quote = [max(candle.volume, 0.0) * max(candle.close, 0.0) for candle in candles]
        prefix = [0.0]
        for value in quote:
            prefix.append(prefix[-1] + value)
        rvol_1h: list[float] = []
        rvol_1m: list[float] = []
        highs = [candle.high for candle in candles]
        lows = [candle.low for candle in candles]
        consolidation_top = _rolling_max_previous(highs, consolidation_bars)
        consolidation_bottom = _rolling_min_previous(lows, consolidation_bars)
        daily_high = _rolling_max_inclusive(highs, daily_lookback)
        consolidation_ok: list[bool] = []
        daily_high_ok: list[bool] = []
        for index in range(len(candles)):
            current_start = max(0, index - 59)
            current_1h = prefix[index + 1] - prefix[current_start]
            baseline_end = max(0, current_start)
            baseline_start = max(0, baseline_end - lookback_minutes)
            baseline_volume = prefix[baseline_end] - prefix[baseline_start]
            baseline_minutes = max(1, baseline_end - baseline_start)
            baseline_1h = baseline_volume / baseline_minutes * 60.0
            rvol_1h.append(current_1h / max(baseline_1h, 1e-12))
            one_min_start = max(0, index - 20)
            one_min_count = max(1, index - one_min_start)
            one_min_avg = (prefix[index] - prefix[one_min_start]) / one_min_count if index > one_min_start else 0.0
            rvol_1m.append(quote[index] / max(one_min_avg, 1e-12) if one_min_avg > 0 else 0.0)
            top = consolidation_top[index]
            bottom = consolidation_bottom[index]
            close = candles[index].close
            near_top = top > 0 and (top - close) / max(close, 1e-12) <= consolidation_threshold
            inside = bottom <= close <= top * (1.0 + consolidation_threshold) if top > bottom else False
            consolidation_ok.append(index >= consolidation_bars and inside and near_top)
            daily_high_ok.append(index >= 7 * 1440 and close < daily_high[index] * daily_zone)
        cache[symbol] = VbpSymbolFeatures(
            rvol_1h=rvol_1h,
            rvol_1m=rvol_1m,
            quote_prefix=prefix,
            consolidation_ok=consolidation_ok,
            consolidation_top=consolidation_top,
            consolidation_bottom=consolidation_bottom,
            daily_high_ok=daily_high_ok,
        )
    return cache


def _vbp_cached_feature(cache: dict[str, VbpSymbolFeatures], symbol: str, name: str, index: int) -> float:
    features = cache.get(symbol)
    values = getattr(features, name, None) if features else None
    if not values or index >= len(values):
        return 0.0
    return values[index]


def _rolling_max_previous(values: list[float], window: int) -> list[float]:
    output: list[float] = []
    queue: deque[int] = deque()
    for index in range(len(values)):
        while queue and queue[0] < index - window:
            queue.popleft()
        output.append(values[queue[0]] if queue else 0.0)
        while queue and values[queue[-1]] <= values[index]:
            queue.pop()
        queue.append(index)
    return output


def _rolling_min_previous(values: list[float], window: int) -> list[float]:
    output: list[float] = []
    queue: deque[int] = deque()
    for index in range(len(values)):
        while queue and queue[0] < index - window:
            queue.popleft()
        output.append(values[queue[0]] if queue else 0.0)
        while queue and values[queue[-1]] >= values[index]:
            queue.pop()
        queue.append(index)
    return output


def _rolling_max_inclusive(values: list[float], window: int) -> list[float]:
    output: list[float] = []
    queue: deque[int] = deque()
    for index in range(len(values)):
        while queue and queue[0] < index - window + 1:
            queue.popleft()
        while queue and values[queue[-1]] <= values[index]:
            queue.pop()
        queue.append(index)
        output.append(values[queue[0]] if queue else 0.0)
    return output


def _vbp_quote_volume(candles: list[Candle]) -> float:
    return sum(max(candle.volume, 0.0) * max(candle.close, 0.0) for candle in candles)


def _vbp_1m_rvol(candles: list[Candle], index: int, lookback: int) -> float:
    if index <= 0:
        return 0.0
    start = max(0, index - max(1, lookback))
    previous = candles[start:index]
    if not previous:
        return 0.0
    average = sum(candle.volume for candle in previous) / len(previous)
    return candles[index].volume / max(average, 1e-12)


def _vbp_vwap(candles: list[Candle]) -> float:
    volume = sum(max(candle.volume, 0.0) for candle in candles)
    if volume <= 0:
        return candles[-1].close if candles else 0.0
    return sum(((candle.high + candle.low + candle.close) / 3.0) * max(candle.volume, 0.0) for candle in candles) / volume


def _vbp_tp2_price(
    candles: list[Candle],
    index: int,
    entry_price: float,
    stop_price: float,
    tp1_rr: float,
) -> float:
    lookback = 20 * 1440
    start = max(0, index - lookback)
    recent_high = max((candle.high for candle in candles[start:index + 1]), default=entry_price)
    risk = max(entry_price - stop_price, entry_price * 0.005)
    if recent_high > entry_price + risk * tp1_rr:
        return recent_high
    return entry_price + risk * max(tp1_rr * 1.5, 2.0)


def _low_base_reason_float(entry_reason: str, key: str, default: float = 0.0) -> float:
    marker = f"{key}="
    if marker not in entry_reason:
        return default
    raw = entry_reason.split(marker, 1)[1].split()[0].strip().rstrip(",")
    try:
        return float(raw)
    except ValueError:
        return default


def _low_base_risk_cash(position: PortfolioPosition) -> float:
    stop_pct = _low_base_reason_float(str(position.entry_reason), "stop", 0.0) / 100.0
    risk_per_unit = position.entry_price * stop_pct if stop_pct > 0 else abs(position.entry_price - position.stop_price)
    return max(1e-12, risk_per_unit * position.quantity)


def _low_base_closed_15m(
    execution_candles_by_symbol: dict[str, list[Candle]],
    symbol: str,
    execution_index: int,
    limit: int,
) -> list[Candle]:
    candles_1m = execution_candles_by_symbol.get(symbol, [])[: execution_index + 1]
    if not candles_1m:
        return []
    return _resample_to_timeframe(candles_1m, "1m", "15m")[-limit:]


def _update_low_base_dynamic_stop(
    trader: BinanceAutoTrader,
    config: Any,
    position: PortfolioPosition,
    candle: Candle,
) -> None:
    if not bool(getattr(config.strategy, "low_base_breakeven_enabled", False)):
        return
    risk_cash = _low_base_risk_cash(position)
    mfe_r = position.mfe / risk_cash
    trigger_r = float(getattr(config.strategy, "low_base_breakeven_trigger_r", 1.0))
    if mfe_r < trigger_r:
        return
    offset = float(getattr(config.strategy, "low_base_breakeven_offset_pct", 0.001))
    breakeven_stop = position.entry_price * (1.0 + offset)
    if position.stop_price < breakeven_stop:
        position.stop_price = min(breakeven_stop, candle.close)
        if " be=1" not in position.entry_reason:
            position.entry_reason += " be=1"
            trader.low_base_ignition_stats["breakeven_triggered_count"] = trader.low_base_ignition_stats.get("breakeven_triggered_count", 0) + 1


def _low_base_exit_reason(
    trader: BinanceAutoTrader,
    config: Any,
    position: PortfolioPosition,
    candle: Candle,
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
) -> str | None:
    strategy = config.strategy
    risk_cash = _low_base_risk_cash(position)
    mfe_r = position.mfe / risk_cash
    hold_minutes = _hold_minutes(position.entry_time, candle.timestamp)
    candles_15m = _low_base_closed_15m(execution_candles_by_symbol, position.symbol, execution_index, 80)
    btc_15m = _low_base_closed_15m(execution_candles_by_symbol, "BTCUSDT", execution_index, 12)

    if bool(getattr(strategy, "low_base_fail_fast_enabled", False)):
        fail_minutes = max(1, int(getattr(strategy, "low_base_fail_fast_minutes", 30)))
        min_mfe_r = float(getattr(strategy, "low_base_fail_fast_min_mfe_r", 0.30))
        level = _low_base_reason_float(str(position.entry_reason), "level", 0.0)
        if hold_minutes >= fail_minutes and mfe_r < min_mfe_r:
            trader.low_base_ignition_stats["fail_fast_count"] = trader.low_base_ignition_stats.get("fail_fast_count", 0) + 1
            return f"low_base_fail_fast_mfe mfe_r={mfe_r:.2f}"
        if bool(getattr(strategy, "low_base_fail_fast_exit_if_lost_level", True)) and level > 0 and candle.close < level * 0.998:
            trader.low_base_ignition_stats["fail_fast_count"] = trader.low_base_ignition_stats.get("fail_fast_count", 0) + 1
            return "low_base_fail_fast_lost_breakout_level"
        if candles_15m and len(candles_15m) >= 25:
            closes = [item.close for item in candles_15m]
            ma7 = _sma_values(closes, 7)
            ma25 = _sma_values(closes, 25)
            if hold_minutes >= 15 and candles_15m[-1].close < min(ma7[-1], ma25[-1]):
                trader.low_base_ignition_stats["fail_fast_count"] = trader.low_base_ignition_stats.get("fail_fast_count", 0) + 1
                return "low_base_fail_fast_lost_15m_ma"
        if btc_15m and len(btc_15m) >= 5:
            btc_drop = btc_15m[-1].close / max(btc_15m[-5].close, 1e-12) - 1.0
            if btc_drop <= float(getattr(strategy, "low_base_btc_15m_drop_block", -0.008)):
                trader.low_base_ignition_stats["fail_fast_count"] = trader.low_base_ignition_stats.get("fail_fast_count", 0) + 1
                return "low_base_fail_fast_btc_drop"

    if not bool(getattr(strategy, "low_base_runner_enabled", False)):
        return None
    activate_r = float(getattr(strategy, "low_base_runner_activate_r", 1.5))
    if mfe_r >= activate_r and " runner=1" not in position.entry_reason:
        position.entry_reason += " runner=1"
        trader.low_base_ignition_stats["runner_activated_count"] = trader.low_base_ignition_stats.get("runner_activated_count", 0) + 1
    if " runner=1" not in position.entry_reason or not candles_15m or len(candles_15m) < 35:
        return None
    closes = [item.close for item in candles_15m]
    ma7 = _sma_values(closes, 7)
    ema21 = ema(closes, 21)
    _, _, hist = macd(closes, config.filters.macd_fast, config.filters.macd_slow, config.filters.macd_signal)
    fade_bars = max(2, int(getattr(strategy, "low_base_runner_macd_fade_bars", 2)))
    if candles_15m[-1].close < ma7[-1]:
        return "low_base_runner_ma7_exit"
    if candles_15m[-1].close < ema21[-1]:
        return "low_base_runner_ema21_exit"
    if len(hist) > fade_bars + 1 and all(hist[-i] < hist[-i - 1] for i in range(1, fade_bars + 1)):
        return "low_base_runner_macd_fade_exit"
    if btc_15m and len(btc_15m) >= 5:
        btc_drop = btc_15m[-1].close / max(btc_15m[-5].close, 1e-12) - 1.0
        if btc_drop <= float(getattr(strategy, "low_base_btc_15m_drop_block", -0.008)):
            return "low_base_runner_btc_weak_exit"
    latest = candles_15m[-1]
    latest_range = max(latest.high - latest.low, 1e-12)
    upper_wick = (latest.high - max(latest.open, latest.close)) / latest_range
    volume_avg = sum(item.volume for item in candles_15m[-25:-1]) / 24
    if upper_wick > float(getattr(strategy, "low_base_upper_wick_max_ratio", 0.45)) and latest.volume > volume_avg * 1.5:
        return "low_base_runner_volume_upper_wick_exit"
    return None


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


def _queue_cmipr_addons_1m(
    config: Any,
    positions: dict[str, PortfolioPosition],
    pending: list[PendingCmiprAddon],
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
    timestamp: Any,
    execution_config: BacktestExecutionConfig,
    client: HistoricalClient,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
    engine: CmiprEngine | None,
    stats: dict[str, int],
) -> None:
    pyramid = config.cmipr.pyramid
    if not bool(pyramid.enabled) or int(pyramid.max_addons) <= 0:
        return
    pending_symbols = {item.symbol for item in pending}
    for symbol, position in positions.items():
        if not _is_cmipr_position(position) or symbol in pending_symbols:
            continue
        if position.scale_ins >= min(2, int(pyramid.max_addons)):
            continue
        bars_between = max(1, int(pyramid.min_bars_between_addons)) * 5
        reference_index = position.cmipr_last_addon_index if position.cmipr_last_addon_index >= 0 else position.entry_index
        if execution_index - reference_index < bars_between:
            continue
        candle = execution_candles_by_symbol[symbol][execution_index]
        current_r = _cmipr_executable_current_r(
            config,
            position,
            candle.close,
            timestamp,
            execution_config,
            client.symbol_rules(symbol),
        )
        trigger = float(pyramid.addon_1_trigger_r if position.scale_ins == 0 else pyramid.addon_2_trigger_r)
        if current_r < trigger or current_r <= 0:
            continue
        timeframe = str(pyramid.confirmation_timeframe)
        candles = _mtf_closed(mtf_candles_by_timeframe, mtf_timestamps_by_timeframe, timeframe, symbol, timestamp, 40)
        if len(candles) < 20:
            _count_stat(stats, "addon_reject_warmup")
            continue
        latest = candles[-1]
        previous = candles[-2]
        if position.direction == Direction.LONG:
            confirmation = latest.close > previous.high and latest.close > latest.open
        else:
            confirmation = latest.close < previous.low and latest.close < latest.open
        if not confirmation or _volume_ratio_live(candles, 12) < float(pyramid.min_confirmation_volume_ratio):
            continue
        atr_value = atr(candles, 14)[-1]
        structure = min(item.low for item in candles[-3:]) if position.direction == Direction.LONG else max(item.high for item in candles[-3:])
        proposed_stop = structure - config.cmipr.entry.stop_atr_buffer * atr_value if position.direction == Direction.LONG else structure + config.cmipr.entry.stop_atr_buffer * atr_value
        noise_distance = position.direction.value * (candle.close - proposed_stop) / max(atr_value, 1e-12)
        if noise_distance < float(pyramid.min_new_stop_noise_atr):
            _count_stat(stats, "addon_reject_stop_inside_noise")
            continue
        # A genuinely newer structure must improve the stop. A breakeven stop
        # moved only to free risk does not qualify as an add-on structure.
        improves_structure = proposed_stop > position.stop_price if position.direction == Direction.LONG else proposed_stop < position.stop_price
        if not improves_structure:
            _count_stat(stats, "addon_reject_no_new_structure")
            continue
        stop_pct = abs(candle.close - proposed_stop) / max(candle.close, 1e-12)
        addon_number = position.scale_ins + 1
        fraction = float(pyramid.addon_1_risk_fraction if addon_number == 1 else pyramid.addon_2_risk_fraction)
        signal = Signal(
            position.direction,
            0.55,
            f"{CMIPR_REASON_TOKEN} addon={addon_number} full_cost_current_r={current_r:.6f}",
            stop_pct,
            max(stop_pct * 20.0, 0.25),
            risk_multiplier=1.0,
            max_holding_bars=position.max_holding_bars,
        )
        delay = 1 + max(0, int(pyramid.extra_execution_delay_minutes))
        pending.append(PendingCmiprAddon(symbol, signal, fraction, proposed_stop, latest.timestamp, execution_index + delay, addon_number))
        if engine is not None:
            engine.runtime[symbol].state = CmiprState.ADDON_1_ARMED if addon_number == 1 else CmiprState.ADDON_2_ARMED
        _count_stat(stats, f"addon_{addon_number}_confirmed")


def _fill_cmipr_pending_addons_1m(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    pending: list[PendingCmiprAddon],
    execution_candles_by_symbol: dict[str, list[Candle]],
    execution_index: int,
    timestamp: Any,
    execution_config: BacktestExecutionConfig,
    engine: CmiprEngine | None,
    stats: dict[str, int],
) -> float:
    remaining: list[PendingCmiprAddon] = []
    for addon in pending:
        if addon.earliest_execution_index > execution_index:
            remaining.append(addon)
            continue
        position = positions.get(addon.symbol)
        if position is None or not _is_cmipr_position(position) or position.direction != addon.signal.direction:
            _count_stat(stats, "addon_cancel_position_missing")
            continue
        candle = execution_candles_by_symbol[addon.symbol][execution_index]
        execution_candle = replace(candle, high=candle.open, low=candle.open, close=candle.open)
        current_r = _cmipr_executable_current_r(
            config,
            position,
            candle.open,
            timestamp,
            execution_config,
            trader.client.symbol_rules(addon.symbol),
        )
        trigger = float(config.cmipr.pyramid.addon_1_trigger_r if addon.addon_number == 1 else config.cmipr.pyramid.addon_2_trigger_r)
        if current_r < trigger or current_r <= 0:
            _count_stat(stats, "addon_cancel_current_r_lost")
            continue
        atr_candles = execution_candles_by_symbol[addon.symbol][max(0, execution_index - 120):execution_index + 1]
        atr_value = atr(atr_candles, 14)[-1]
        noise_distance = position.direction.value * (candle.open - addon.proposed_stop) / max(atr_value, 1e-12)
        if noise_distance < float(config.cmipr.pyramid.min_new_stop_noise_atr):
            _count_stat(stats, "addon_cancel_stop_inside_noise")
            continue
        account = _account_snapshot(
            config,
            _mark_equity(cash, positions, execution_candles_by_symbol, execution_index),
            positions,
            execution_candles_by_symbol,
            execution_index,
        )
        existing = account.positions.get(addon.symbol)
        signal = replace(addon.signal, stop_loss_pct=abs(candle.open - addon.proposed_stop) / max(candle.open, 1e-12))
        sizing_account = _cmipr_sizing_account(config, account, signal)
        existing = sizing_account.positions.get(addon.symbol)
        quantity_text, size_reason = trader._size_order(
            addon.symbol,
            candle.open,
            signal,
            sizing_account,
            existing_position=existing,
            entry_fraction=addon.fraction,
        )
        quantity = float(quantity_text)
        if size_reason != "ok" or quantity <= 0:
            _count_stat(stats, f"addon_reject_size_{size_reason}")
            continue
        rules = trader.client.symbol_rules(addon.symbol)
        quantity = _cmipr_quantity_within_original_risk_budget(
            config,
            position,
            signal,
            execution_candle,
            quantity,
            addon.proposed_stop,
            timestamp,
            execution_config,
            rules,
        )
        if quantity <= 0:
            _count_stat(stats, "addon_reject_full_cost_risk_budget")
            continue
        old_quantity = position.quantity
        cash = _add_to_position(
            config,
            cash,
            position,
            signal,
            execution_candle,
            quantity,
            execution_config=execution_config,
            rules=rules,
        )
        added_quantity = max(0.0, position.quantity - old_quantity)
        if added_quantity <= 0:
            _count_stat(stats, "addon_unfilled")
            continue
        position.stop_price = addon.proposed_stop
        position.cmipr_last_addon_index = execution_index
        if addon.addon_number == 1:
            position.cmipr_addon_1_quantity += added_quantity
        else:
            position.cmipr_addon_2_quantity += added_quantity
        if engine is not None:
            engine.mark_protection_pending(addon.symbol)
            engine.mark_protected(addon.symbol)
            engine.runtime[addon.symbol].state = CmiprState.ADDON_1_POSITION if addon.addon_number == 1 else CmiprState.FULL_POSITION
        _count_stat(stats, f"addon_{addon.addon_number}_fill_count")
    pending[:] = remaining
    return cash


def _cmipr_quantity_within_original_risk_budget(
    config: Any,
    position: PortfolioPosition,
    signal: Signal,
    candle: Candle,
    requested_quantity: float,
    proposed_stop: float,
    timestamp: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> float:
    budget = max(0.0, position.risk_budget_usdt)
    if budget <= 0:
        return 0.0
    quantity = requested_quantity
    for _ in range(10):
        risk = _cmipr_worst_full_cost_risk(
            position,
            signal,
            candle,
            quantity,
            proposed_stop,
            timestamp,
            execution_config,
            rules,
        )
        if risk <= budget + 1e-9:
            return quantity
        ratio = budget / max(risk, 1e-12)
        quantity = float(rules.round_quantity(quantity * max(0.0, min(0.98, ratio * 0.98))))
        if quantity <= 0:
            return 0.0
    return 0.0


def _cmipr_worst_full_cost_risk(
    position: PortfolioPosition,
    signal: Signal,
    candle: Candle,
    new_quantity: float,
    proposed_stop: float,
    timestamp: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> float:
    bar_quote_volume = abs(candle.volume * candle.close)
    entry_fill = market_entry_fill(execution_config, rules, signal.direction, new_quantity, candle.close, bar_quote_volume)
    total_quantity = position.quantity + new_quantity
    stop_fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        total_quantity,
        proposed_stop,
        "stop_market",
        max(position.liquidity_reference_quote_volume, bar_quote_volume) or None,
    )
    existing_raw_entry = position.raw_entry_price or position.entry_price
    gross_at_stop = position.direction.value * (
        position.quantity * (proposed_stop - existing_raw_entry)
        + new_quantity * (proposed_stop - entry_fill.raw_price)
    )
    accrued_funding = min(0.0, _funding_for_position(execution_config, position, timestamp))
    net_at_stop = (
        gross_at_stop
        - position.entry_fee
        - position.entry_slippage_cost
        - entry_fill.fee
        - entry_fill.slippage_cost
        - stop_fill.fee
        - stop_fill.slippage_cost
        + accrued_funding
    )
    return max(0.0, -net_at_stop)


def _volume_ratio_live(candles: list[Candle], period: int) -> float:
    if len(candles) < period + 1:
        return 0.0
    average = sum(item.volume for item in candles[-period - 1:-1]) / period
    return candles[-1].volume / max(average, 1e-12)


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
        if _is_cmipr_position(positions[symbol]):
            continue
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
    loss_streak: dict[Direction, int],
    pause_until: dict[Direction, Any],
    closed_trades: list[dict[str, Any]],
    timestamp: Any,
) -> tuple[dict[Direction, int], dict[Direction, Any]]:
    if not (
        getattr(config.strategy, "indicator_reversal_loss_pause_enabled", False)
        or getattr(config.strategy, "indicator_long_loss_pause_enabled", False)
        or getattr(config.strategy, "indicator_short_loss_pause_enabled", False)
    ):
        return loss_streak, pause_until
    for trade in closed_trades:
        if str(trade.get("strategy_bucket", "")) != "indicator_reversal":
            continue
        direction = _trade_direction(trade)
        if direction not in (Direction.LONG, Direction.SHORT):
            continue
        if not _indicator_reversal_side_pause_enabled(config, direction):
            continue
        trigger_losses = _indicator_reversal_side_pause_losses(config, direction)
        pause_bars = _indicator_reversal_side_pause_bars(config, direction)
        pause_seconds = pause_bars * interval_to_milliseconds(config.trading.timeframe) / 1000.0
        if float(trade.get("net_pnl", 0.0)) > 0:
            loss_streak[direction] = 0
            continue
        loss_streak[direction] = loss_streak.get(direction, 0) + 1
        if loss_streak[direction] >= trigger_losses:
            pause_until[direction] = timestamp + timedelta(seconds=pause_seconds)
            loss_streak[direction] = 0
    return loss_streak, pause_until


def _trade_direction(trade: dict[str, Any]) -> Direction | None:
    value = str(trade.get("direction") or trade.get("side") or "").upper()
    if value == "LONG":
        return Direction.LONG
    if value == "SHORT":
        return Direction.SHORT
    return None


def _indicator_reversal_side_pause_enabled(config: Any, direction: Direction) -> bool:
    side = "long" if direction == Direction.LONG else "short"
    side_attr = f"indicator_{side}_loss_pause_enabled"
    if hasattr(config.strategy, side_attr):
        return bool(getattr(config.strategy, side_attr))
    return bool(getattr(config.strategy, "indicator_reversal_loss_pause_enabled", False))


def _indicator_reversal_side_pause_losses(config: Any, direction: Direction) -> int:
    side = "long" if direction == Direction.LONG else "short"
    return max(1, int(getattr(config.strategy, f"indicator_{side}_loss_pause_losses", getattr(config.strategy, "indicator_reversal_loss_pause_losses", 2))))


def _indicator_reversal_side_pause_bars(config: Any, direction: Direction) -> int:
    side = "long" if direction == Direction.LONG else "short"
    return max(1, int(getattr(config.strategy, f"indicator_{side}_loss_pause_bars", getattr(config.strategy, "indicator_reversal_loss_pause_bars", 8))))


def _mark_reentry_cooldown_time(config: Any, reentry_block_until: dict[str, Any], symbol: str, timestamp: Any) -> None:
    cooldown_seconds = max(0, int(config.trading.symbol_reentry_cooldown_seconds))
    if cooldown_seconds <= 0:
        return
    reentry_block_until[symbol] = max(
        reentry_block_until.get(symbol, timestamp),
        timestamp + timedelta(seconds=cooldown_seconds),
    )


def _update_cmipr_account_runtime(
    config: Any,
    runtime: CmiprAccountRuntime,
    closed_trades: list[dict[str, Any]],
    timestamp: Any,
    stats: dict[str, int],
) -> None:
    risk = config.cmipr.risk_control
    for trade in closed_trades:
        if trade.get("strategy") != CMIPR_REASON_TOKEN:
            continue
        pnl = float(trade.get("net_pnl", 0.0) or 0.0)
        runtime.daily_pnl[timestamp.date()] = runtime.daily_pnl.get(timestamp.date(), 0.0) + pnl
        if pnl < 0:
            runtime.loss_streak += 1
            if runtime.loss_streak >= max(1, int(risk.consecutive_loss_limit)):
                runtime.pause_until = timestamp + timedelta(minutes=max(1, int(risk.consecutive_loss_pause_minutes)))
                runtime.loss_streak = 0
                _count_stat(stats, "consecutive_loss_pause_count")
        else:
            runtime.loss_streak = 0


def _cmipr_new_entries_paused(
    config: Any,
    runtime: CmiprAccountRuntime,
    timestamp: Any,
    starting_equity: float,
) -> bool:
    if runtime.pause_until is not None and timestamp < runtime.pause_until:
        return True
    limit = abs(float(config.cmipr.risk_control.daily_loss_stop_pct)) * starting_equity
    return limit > 0 and runtime.daily_pnl.get(timestamp.date(), 0.0) <= -limit


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

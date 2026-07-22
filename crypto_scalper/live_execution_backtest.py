from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import lru_cache
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
    mtf_min_rank_score_for_direction,
    mtf_report_from_summary,
    oi_change_at,
)
from .mtf_candidate_quality import mtf_candidate_quality
from .risk import (
    BacktestExecutionConfig,
    BacktestExecutionStats,
    capacity_limited_quantity,
    execution_config_from_live_config,
    market_exit_fill,
)
from .risk import market_entry_fill
from .indicators import atr, ema, macd
from .realistic_data import load_funding_rate_directory
from .regime_score import RegimeScoreEngine
from .reversal_alpha import REVERSAL_V2_REASON_TOKEN, ReversalAlphaEngine
from .cmipr import CMIPR_REASON_TOKEN, CmiprEngine, CmiprState, audit_derivative_coverage
from .mtper import MTPER_REASON_TOKEN, MtperEngine, MtperState
from .mtpc import MTPC_REASON_TOKEN, MtpcEngine, MtpcState
from .cmipr_r_basis import (
    CAMPAIGN_R_BASIS,
    CMIPR_R_DEFINITION_VERSION,
    INITIAL_LEG_R_BASIS,
    executable_net_pnl,
    initial_leg_risk,
    net_pnl_r,
    normalize_r_basis,
)


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
    mtf_candidate_cache: dict[tuple[Any, ...], EntryCandidate | None] | None = None,
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
    mtper_enabled = bool(getattr(getattr(config, "mtper", None), "enabled", False))
    mtpc_enabled = bool(getattr(getattr(config, "mtpc", None), "enabled", False))
    reversal_v2_enabled = bool(getattr(getattr(config, "reversal_alpha", None), "enabled", False))
    if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False) or cmipr_enabled or mtper_enabled or mtpc_enabled or reversal_v2_enabled:
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
    mtf_candidate_cache: dict[tuple[Any, ...], EntryCandidate | None] | None = None,
    cmipr_feature_cache: dict[tuple[Any, ...], Any] | None = None,
    mtf_aux_features_override: dict[str, dict[str, Any]] | None = None,
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
    mtper_enabled = bool(getattr(getattr(config, "mtper", None), "enabled", False))
    mtpc_enabled = bool(getattr(getattr(config, "mtpc", None), "enabled", False))
    reversal_v2_enabled = bool(getattr(getattr(config, "reversal_alpha", None), "enabled", False))
    if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False) or cmipr_enabled or mtper_enabled or mtpc_enabled or reversal_v2_enabled:
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
    cmipr_only = cmipr_enabled and bool(getattr(config.cmipr, "disable_legacy_strategies", True))
    mtper_only = mtper_enabled and bool(getattr(config.mtper, "disable_legacy_strategies", True))
    mtpc_only = mtpc_enabled and bool(getattr(config.mtpc, "disable_legacy_strategies", True))
    cmipr_core_model = cmipr_only and str(config.cmipr.research.model_variant).strip().lower() == "core"
    mtf_aux_features = mtf_aux_features_override if mtf_aux_features_override is not None else (
        load_auxiliary_features(
            symbols,
            str(getattr(config.strategy, "mtf_oi_data_dir", "data/binance_oi_taker_5m")),
            str(getattr(config.strategy, "mtf_funding_data_dir", "data/binance_oi_flush_funding")),
        )
        if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False) or (cmipr_enabled and not cmipr_core_model)
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
        client = HistoricalClient(
            signal_candles_by_symbol,
            config.trading.timeframe,
            () if cmipr_only or mtper_only or mtpc_only else historical_filter_timeframes,
        )
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
    ) or (cmipr_enabled and bool(getattr(config.cmipr, "disable_legacy_strategies", True))) or mtper_only or mtpc_only
    if legacy_disabled:
        signal_cache = {}
        reversal_cache = {}
        mtf_cache = {}
        market_regime_cache = []
        btc_market_state_cache = []
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
    mtper_engine = MtperEngine(config, mtf_candles_by_timeframe) if mtper_enabled else None
    mtper_stats: dict[str, int] = {}
    mtpc_engine = MtpcEngine(config, mtf_candles_by_timeframe) if mtpc_enabled else None
    mtpc_stats: dict[str, int] = {}
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
    if reversal_alpha_config is not None and bool(getattr(reversal_alpha_config, "enabled", False)):
        reversal_alpha_engine = ReversalAlphaEngine(
            reversal_alpha_config,
            {
                timeframe: mtf_candles_by_timeframe.get(timeframe, {})
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
    reversal_v2_stats: dict[str, int] = {}
    reversal_v2_consumed_events: set[str] = set()
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
            mtper_engine,
            mtper_stats,
            mtpc_engine,
            mtpc_stats,
            reversal_v2_stats,
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
            mtper_engine,
            mtper_stats,
            mtpc_engine,
            mtpc_stats,
            reversal_v2_stats,
            mtf_reject_stats,
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

        if mtpc_enabled and mtpc_engine is not None:
            occupied = set(positions) | {entry.candidate.symbol for entry in pending_entries}
            allowed = (
                set(_POINT_IN_TIME_UNIVERSE.get(timestamp.date(), frozenset()))
                if bool(config.risk.point_in_time_universe_enabled)
                else set(mtpc_engine.symbols)
            )
            mtpc_candidates = mtpc_engine.scan(timestamp, occupied, allowed)
            if (
                bool(getattr(config.mtpc, "combine_with_mtf", False))
                and bool(getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False))
            ):
                mtf_candidates = _entry_candidates_for_scan(
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
                    reversal_alpha_engine,
                    reversal_v2_consumed_events,
                    reversal_v2_stats,
                )
                candidates = _merge_candidate_sleeves(mtf_candidates, mtpc_candidates)
            else:
                candidates = mtpc_candidates
        elif mtper_enabled and mtper_engine is not None:
            occupied = set(positions) | {entry.candidate.symbol for entry in pending_entries}
            allowed = (
                set(_POINT_IN_TIME_UNIVERSE.get(timestamp.date(), frozenset()))
                if bool(config.risk.point_in_time_universe_enabled)
                else set(mtper_engine.symbols)
            )
            candidates = mtper_engine.scan(timestamp, occupied, allowed)
        elif cmipr_enabled and cmipr_engine is not None:
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
                reversal_alpha_engine,
                reversal_v2_consumed_events,
                reversal_v2_stats,
            )
        if bool(config.risk.point_in_time_universe_enabled):
            candidates = [
                candidate
                for candidate in candidates
                if _point_in_time_symbol_allowed(config, candidate.symbol, timestamp)
            ]
        pending_symbols = {entry.candidate.symbol for entry in pending_entries}
        candidates = [candidate for candidate in candidates if candidate.symbol not in pending_symbols]
        if cmipr_enabled:
            max_same_direction = max(1, int(config.cmipr.risk_control.max_same_direction_positions))
            candidates = [
                candidate
                for candidate in candidates
                if sum(1 for position in positions.values() if position.direction == candidate.signal.direction) < max_same_direction
            ]
        if mtper_enabled:
            max_same_direction = max(1, int(config.mtper.risk_control.max_same_direction_campaigns))
            candidates = [
                candidate
                for candidate in candidates
                if sum(1 for position in positions.values() if position.direction == candidate.signal.direction) < max_same_direction
            ]
        if mtpc_enabled:
            max_same_direction = max(1, int(config.mtpc.risk_control.max_same_direction_positions))
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
                            else (
                                max(0, int(config.mtper.entry.extra_execution_delay_minutes))
                                if MTPER_REASON_TOKEN in str(candidate.signal.reason)
                                else (
                                    max(0, int(config.mtpc.pullback.extra_execution_delay_minutes))
                                    if MTPC_REASON_TOKEN in str(candidate.signal.reason)
                                    else (
                                        max(0, int(getattr(config.strategy, "mtf_extra_execution_delay_minutes", 0)))
                                        if MTF_REASON_TOKEN in str(candidate.signal.reason)
                                        else 0
                                    )
                                )
                            )
                        )
                    ),
                )
            )
            if REVERSAL_V2_REASON_TOKEN in str(candidate.signal.reason):
                event_id = str(candidate.metadata.get("event_id", ""))
                if event_id:
                    reversal_v2_consumed_events.add(event_id)
                _count_stat(reversal_v2_stats, "pending_count")
            if cmipr_engine is not None and CMIPR_REASON_TOKEN in str(candidate.signal.reason):
                cmipr_engine.mark_order_pending(candidate.symbol)
            if mtper_engine is not None and MTPER_REASON_TOKEN in str(candidate.signal.reason):
                mtper_engine.mark_order_pending(candidate.symbol)
                _count_stat(mtper_stats, "pending_count")
            if mtpc_engine is not None and MTPC_REASON_TOKEN in str(candidate.signal.reason):
                mtpc_engine.mark_order_pending(candidate.symbol, timestamp)
                _count_stat(mtpc_stats, "pending_count")
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
    if reversal_v2_stats:
        payload["reversal_v2_stats"] = dict(sorted(reversal_v2_stats.items()))
    if cmipr_engine is not None and cmipr_coverage is not None:
        engine_report = cmipr_engine.report()
        campaign_diagnostics = cmipr_engine.finalize_campaign_diagnostics(
            execution_candles_by_symbol,
            trades,
            execution_config,
            {symbol: client.symbol_rules(symbol) for symbol in execution_candles_by_symbol},
        )
        if campaign_diagnostics:
            engine_report["convex_campaign_diagnostics"] = campaign_diagnostics
        payload["cmipr_report"] = _cmipr_report(
            config,
            trades,
            engine_report,
            cmipr_coverage.as_dict(),
            cmipr_stats,
            execution_config,
            summary_first_candle.timestamp,
            last_candle.timestamp,
        )
    if mtper_engine is not None:
        payload["mtper_report"] = _mtper_report(
            config,
            trades,
            mtper_engine.report(),
            mtper_stats,
            execution_config,
            summary_first_candle.timestamp,
            last_candle.timestamp,
        )
    if mtpc_engine is not None:
        payload["mtpc_report"] = _mtpc_report(
            config,
            trades,
            mtpc_engine.report(),
            mtpc_stats,
            execution_config,
            summary_first_candle.timestamp,
            last_candle.timestamp,
        )
    if getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False):
        mtf_payload = dict(payload)
        mtf_payload["trades"] = trades
        payload["mtf_report"] = mtf_report_from_summary(mtf_payload, mtf_reject_stats)
    return payload


def _mtpc_report(
    config: Any,
    trades: list[dict[str, Any]],
    engine_report: dict[str, Any],
    stats: dict[str, int],
    execution_config: BacktestExecutionConfig,
    data_start: Any,
    data_end: Any,
) -> dict[str, Any]:
    legs = [trade for trade in trades if trade.get("strategy") == MTPC_REASON_TOKEN]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in legs:
        metadata = trade.get("strategy_metadata") or {}
        event_id = str(metadata.get("event_id") or _reason_tag(str(trade.get("entry_reason", "")), "event_id", ""))
        if not event_id:
            event_id = f"legacy:{trade.get('symbol')}:{trade.get('entry_time')}"
        grouped.setdefault(event_id, []).append(trade)
    campaigns: list[dict[str, Any]] = []
    for event_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: str(item.get("exit_time", "")))
        first = ordered[0]
        last = ordered[-1]
        metadata = dict(last.get("strategy_metadata") or first.get("strategy_metadata") or {})
        risk_budget = max(float(item.get("campaign_risk_budget_usdt", 0.0) or 0.0) for item in ordered)
        initial_risk = max(float(item.get("initial_leg_full_cost_risk_usdt", 0.0) or 0.0) for item in ordered)
        net = sum(float(item.get("net_pnl", 0.0)) for item in ordered)
        campaigns.append(
            {
                "campaign_id": event_id,
                "event_id": event_id,
                "setup_id": metadata.get("setup_id"),
                "symbol": first.get("symbol"),
                "side": first.get("side"),
                "entry_time": first.get("entry_time"),
                "exit_time": last.get("exit_time"),
                "exit_reason": last.get("exit_reason"),
                "net_pnl": net,
                "gross_pnl": sum(float(item.get("gross_pnl", 0.0)) for item in ordered),
                "fee": sum(float(item.get("fee", item.get("fees", 0.0))) for item in ordered),
                "slippage": sum(float(item.get("slippage_cost", 0.0)) for item in ordered),
                "funding": sum(float(item.get("funding", 0.0)) for item in ordered),
                "campaign_risk_usdt": risk_budget,
                "initial_leg_full_cost_risk_usdt": initial_risk,
                "initial_leg_actual_risk_fraction": initial_risk / max(risk_budget, 1e-12),
                "pnl_r": net / max(risk_budget, 1e-12),
                "initial_leg_pnl_r": net / max(initial_risk, 1e-12),
                "mfe_r": max(float(item.get("mtpc_max_executable_initial_leg_r", item.get("mtpc_max_executable_r", 0.0)) or 0.0) for item in ordered),
                "mae_r": min(float(item.get("mtpc_min_executable_initial_leg_r", item.get("mtpc_min_executable_r", 0.0)) or 0.0) for item in ordered),
                "campaign_mfe_r": max(float(item.get("mtpc_max_executable_campaign_r", 0.0) or 0.0) for item in ordered),
                "campaign_mae_r": min(float(item.get("mtpc_min_executable_campaign_r", 0.0) or 0.0) for item in ordered),
                "rank_percentile": metadata.get("rank_percentile"),
                "impulse_quality": metadata.get("impulse_quality"),
                "pullback_quality": metadata.get("pullback_quality"),
                "confirmation_quality": metadata.get("confirmation_quality"),
                "trigger_type": metadata.get("trigger_type"),
                "target_to_cost_ratio": metadata.get("target_to_cost_ratio"),
                "leg_count": len(ordered),
            }
        )
    return {
        **engine_report,
        "cost_model_version": "full_cost_path_aware_conservative_v1",
        "cost_model_hash": _cmipr_cost_model_hash(execution_config),
        "data_range": {
            "start": data_start.isoformat() if hasattr(data_start, "isoformat") else str(data_start),
            "end": data_end.isoformat() if hasattr(data_end, "isoformat") else str(data_end),
        },
        "execution_stats": dict(sorted(stats.items())),
        "strategy_summary": _mtper_campaign_metrics(campaigns),
        "initial_leg_r_summary": _mtper_campaign_metrics(
            [{**campaign, "pnl_r": campaign["initial_leg_pnl_r"]} for campaign in campaigns]
        ),
        "by_month": _mtper_group_campaigns(campaigns, "exit_time", month=True),
        "by_symbol": _mtper_group_campaigns(campaigns, "symbol"),
        "by_exit_reason": _mtper_group_campaigns(campaigns, "exit_reason"),
        "by_trigger": _mtper_group_campaigns(campaigns, "trigger_type"),
        "campaigns": campaigns,
        "risk_policy": {
            "configured_trade_risk_pct": float(config.mtpc.risk_control.trade_risk_pct),
            "max_trade_risk_pct": float(config.mtpc.risk_control.max_trade_risk_pct),
            "max_open_positions": int(config.mtpc.risk_control.max_open_positions),
            "addon_enabled": False,
            "exit_r_basis": str(config.mtpc.exit.r_basis),
        },
    }


def _mtper_report(
    config: Any,
    trades: list[dict[str, Any]],
    engine_report: dict[str, Any],
    stats: dict[str, int],
    execution_config: BacktestExecutionConfig,
    data_start: Any,
    data_end: Any,
) -> dict[str, Any]:
    legs = [trade for trade in trades if trade.get("strategy") == MTPER_REASON_TOKEN]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in legs:
        metadata = trade.get("strategy_metadata") or {}
        campaign_id = str(metadata.get("campaign_id") or _reason_tag(str(trade.get("entry_reason", "")), "campaign_id", ""))
        if not campaign_id:
            campaign_id = f"legacy:{trade.get('symbol')}:{trade.get('entry_time')}"
        grouped.setdefault(campaign_id, []).append(trade)
    campaigns: list[dict[str, Any]] = []
    for campaign_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: str(item.get("exit_time", "")))
        first = ordered[0]
        last = ordered[-1]
        metadata = dict(last.get("strategy_metadata") or first.get("strategy_metadata") or {})
        risk_budget = max(float(item.get("campaign_risk_budget_usdt", 0.0) or 0.0) for item in ordered)
        initial_risk = max(float(item.get("initial_leg_full_cost_risk_usdt", 0.0) or 0.0) for item in ordered)
        net = sum(float(item.get("net_pnl", 0.0)) for item in ordered)
        campaigns.append(
            {
                "campaign_id": campaign_id,
                "setup_id": metadata.get("setup_id"),
                "symbol": first.get("symbol"),
                "side": first.get("side"),
                "setup_time": metadata.get("setup_time"),
                "initial_entry_time": first.get("entry_time"),
                "second_entry_time": metadata.get("second_entry_time"),
                "initial_entry_price": first.get("entry_price"),
                "second_entry_price": metadata.get("second_entry_price"),
                "average_entry_price": first.get("entry_price"),
                "hard_stop": first.get("initial_stop_price", metadata.get("structural_stop_price")),
                "total_quantity": sum(float(item.get("qty", 0.0)) for item in ordered),
                "campaign_risk_usdt": risk_budget,
                "actual_worst_case_loss": initial_risk,
                "first_target_time": next((item.get("exit_time") for item in ordered if item.get("exit_reason") == "mtper_mean_target_1"), None),
                "second_target_time": next((item.get("exit_time") for item in ordered if item.get("exit_reason") == "mtper_mean_target_2"), None),
                "trend_conversion_time": metadata.get("trend_conversion_time"),
                "final_exit_time": last.get("exit_time"),
                "final_exit_reason": last.get("exit_reason"),
                "gross_pnl": sum(float(item.get("gross_pnl", 0.0)) for item in ordered),
                "fee": sum(float(item.get("fee", item.get("fees", 0.0))) for item in ordered),
                "slippage": sum(float(item.get("slippage_cost", 0.0)) for item in ordered),
                "funding": sum(float(item.get("funding", 0.0)) for item in ordered),
                "net_pnl": net,
                "pnl_r": net / max(risk_budget, 1e-12),
                "initial_leg_pnl_r": net / max(initial_risk, 1e-12),
                "mfe_r": max(float(item.get("mtper_max_executable_campaign_r", 0.0) or 0.0) for item in ordered),
                "mae_r": min(float(item.get("mtper_min_executable_campaign_r", 0.0) or 0.0) for item in ordered),
                "second_entry_mode": metadata.get("second_entry_mode", "none"),
                "actual_4h_cross_confirmed": bool(metadata.get("trend_converted", False)),
                "setup_invalidated": False,
                "trigger_type": metadata.get("trigger_type"),
                "extreme_score": metadata.get("extreme_score"),
                "ema_gap_atr": metadata.get("ema_gap_atr"),
                "target_to_cost_ratio": metadata.get("target_to_cost_ratio"),
                "liquidation_price_estimate": metadata.get("liquidation_price_estimate"),
                "liquidation_buffer_ok": metadata.get("liquidation_buffer_ok"),
                "leg_count": len(ordered),
            }
        )
    metrics = _mtper_campaign_metrics(campaigns)
    return {
        **engine_report,
        "cost_model_version": "full_cost_path_aware_conservative_v1",
        "cost_model_hash": _cmipr_cost_model_hash(execution_config),
        "data_range": {
            "start": data_start.isoformat() if hasattr(data_start, "isoformat") else str(data_start),
            "end": data_end.isoformat() if hasattr(data_end, "isoformat") else str(data_end),
        },
        "execution_stats": dict(sorted(stats.items())),
        "strategy_summary": metrics,
        "campaign_risk_report": {
            "configured_campaign_risk_pct": float(config.mtper.risk_control.campaign_risk_pct),
            "max_campaign_risk_pct": float(config.mtper.risk_control.max_campaign_risk_pct),
            "initial_risk_fraction": float(config.mtper.entry.initial_risk_fraction),
            "actual_worst_case_within_budget_count": sum(
                float(row["actual_worst_case_loss"]) <= float(row["campaign_risk_usdt"]) + 1e-9
                for row in campaigns
            ),
        },
        "liquidation_buffer_report": {
            "checked_campaigns": len(campaigns),
            "all_passed": all(row.get("liquidation_buffer_ok") is True for row in campaigns) if campaigns else True,
        },
        "four_hour_pre_cross_report": {
            "setup_count": len(engine_report.get("setups", [])),
            "formal_cross_setup_count": sum(bool(row.get("formal_cross")) for row in engine_report.get("setups", [])),
        },
        "extreme_score_report": _mtper_numeric_summary(
            [row.get("extreme_score") for row in engine_report.get("pre_cross_candidates", [])]
        ),
        "two_hour_exhaustion_report": _mtper_setup_check_summary(engine_report.get("setups", []), "two_hour_checks"),
        "one_hour_permission_report": _mtper_setup_check_summary(engine_report.get("setups", []), "one_hour_checks"),
        "fifteen_minute_trigger_report": _mtper_group_campaigns(campaigns, "trigger_type"),
        "no_second_entry_report": metrics if not bool(config.mtper.second_entry.enabled) else None,
        "defensive_scale_in_report": None,
        "winner_addon_report": None,
        "target_ladder_report": _mtper_group_legs(legs, "exit_reason"),
        "trend_conversion_report": {
            "count": sum(bool(row.get("actual_4h_cross_confirmed")) for row in campaigns),
            "metrics": _mtper_campaign_metrics([row for row in campaigns if row.get("actual_4h_cross_confirmed")]),
        },
        "long_short_comparison": _mtper_group_campaigns(campaigns, "side"),
        "by_month": _mtper_group_campaigns(campaigns, "final_exit_time", month=True),
        "by_symbol": _mtper_group_campaigns(campaigns, "symbol"),
        "campaigns": campaigns,
    }


def _mtper_campaign_metrics(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [row for row in campaigns if float(row.get("net_pnl", 0.0)) > 0.0]
    losses = [row for row in campaigns if float(row.get("net_pnl", 0.0)) <= 0.0]
    gross_profit = sum(float(row.get("net_pnl", 0.0)) for row in wins)
    gross_loss = abs(sum(float(row.get("net_pnl", 0.0)) for row in losses))
    total = sum(float(row.get("net_pnl", 0.0)) for row in campaigns)
    return {
        "campaign_count": len(campaigns),
        "net_pnl": total,
        "win_rate_pct": len(wins) / len(campaigns) * 100.0 if campaigns else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else (None if gross_profit <= 0.0 else float("inf")),
        "expectancy": total / len(campaigns) if campaigns else 0.0,
        "average_winner": gross_profit / len(wins) if wins else 0.0,
        "average_loser": -gross_loss / len(losses) if losses else 0.0,
        "average_pnl_r": statistics.fmean(float(row.get("pnl_r", 0.0)) for row in campaigns) if campaigns else 0.0,
        "fee": sum(float(row.get("fee", 0.0)) for row in campaigns),
        "slippage": sum(float(row.get("slippage", 0.0)) for row in campaigns),
        "funding": sum(float(row.get("funding", 0.0)) for row in campaigns),
    }


def _mtper_group_campaigns(
    campaigns: list[dict[str, Any]],
    key: str,
    month: bool = False,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in campaigns:
        raw = row.get(key)
        group = str(raw)[:7] if month and raw else str(raw or "unknown")
        grouped.setdefault(group, []).append(row)
    return {group: _mtper_campaign_metrics(rows) for group, rows in sorted(grouped.items())}


def _mtper_group_legs(legs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in legs:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {
        group: {
            "count": len(rows),
            "net_pnl": sum(float(row.get("net_pnl", 0.0)) for row in rows),
        }
        for group, rows in sorted(grouped.items())
    }


def _mtper_numeric_summary(values: list[Any]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "min": finite[0],
        "median": statistics.median(finite),
        "mean": statistics.fmean(finite),
        "max": finite[-1],
    }


def _mtper_setup_check_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return _mtper_numeric_summary(values)


def _cmipr_report(
    config: Any,
    trades: list[dict[str, Any]],
    engine_report: dict[str, Any],
    coverage: dict[str, Any],
    stats: dict[str, int],
    execution_config: BacktestExecutionConfig,
    data_start: Any,
    data_end: Any,
) -> dict[str, Any]:
    cmipr_trades = [trade for trade in trades if trade.get("strategy") == CMIPR_REASON_TOKEN]
    by_addons: dict[str, list[dict[str, Any]]] = {"no_addon": [], "one_addon": [], "two_addons": []}
    for trade in cmipr_trades:
        scale_ins = int(trade.get("scale_ins", 0) or 0)
        key = "no_addon" if scale_ins <= 0 else "one_addon" if scale_ins == 1 else "two_addons"
        by_addons[key].append(trade)
    return {
        **engine_report,
        "r_definition_version": CMIPR_R_DEFINITION_VERSION,
        "cost_model_version": "full_cost_path_aware_conservative_v1",
        "cost_model_hash": _cmipr_cost_model_hash(execution_config),
        "universe_version": _cmipr_universe_hash(),
        "data_range": {
            "start": data_start.isoformat() if hasattr(data_start, "isoformat") else str(data_start),
            "end": data_end.isoformat() if hasattr(data_end, "isoformat") else str(data_end),
        },
        "random_seed": None,
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
        "r_basis_audit": _cmipr_r_basis_audit(config, cmipr_trades),
    }


def _cmipr_cost_model_hash(execution_config: BacktestExecutionConfig) -> str:
    payload = {
        "mode": execution_config.mode,
        "market_slippage_bps": execution_config.market_slippage_bps,
        "stop_slippage_bps": execution_config.stop_slippage_bps,
        "take_profit_slippage_bps": execution_config.take_profit_slippage_bps,
        "maker_fee_rate": execution_config.maker_fee_rate,
        "taker_fee_rate": execution_config.taker_fee_rate,
        "funding_enabled": execution_config.funding_enabled,
        "funding_default_rate": execution_config.funding_default_rate,
        "dynamic_slippage_enabled": execution_config.dynamic_slippage_enabled,
        "impact_coefficient_bps": execution_config.impact_coefficient_bps,
        "impact_exponent": execution_config.impact_exponent,
        "max_bar_participation_rate": execution_config.max_bar_participation_rate,
        "min_partial_fill_ratio": execution_config.min_partial_fill_ratio,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cmipr_universe_hash() -> str:
    payload = [
        (day.isoformat() if hasattr(day, "isoformat") else str(day), sorted(symbols))
        for day, symbols in sorted(_POINT_IN_TIME_UNIVERSE.items(), key=lambda item: str(item[0]))
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cmipr_r_basis_audit(config: Any, trades: list[dict[str, Any]]) -> dict[str, Any]:
    take_basis = normalize_r_basis(config.cmipr.exit.take_profit_r_basis)
    fail_fast_basis = normalize_r_basis(config.cmipr.exit.fail_fast_r_basis)
    take_threshold = float(config.cmipr.exit.fixed_take_profit_r)
    fail_fast_threshold = float(config.cmipr.exit.fail_fast_min_mfe_r)
    rows: list[dict[str, Any]] = []
    for trade in trades:
        campaign_risk = float(trade.get("campaign_risk_budget_usdt", trade.get("risk_budget_usdt", 0.0)) or 0.0)
        initial_leg_risk_usdt = float(trade.get("initial_leg_full_cost_risk_usdt", 0.0) or 0.0)
        actual_fraction = initial_leg_risk_usdt / campaign_risk if campaign_risk > 0.0 else 0.0
        tp_initial_leg_r = take_threshold if take_basis == INITIAL_LEG_R_BASIS else take_threshold / max(actual_fraction, 1e-12)
        fail_fast_initial_leg_r = fail_fast_threshold if fail_fast_basis == INITIAL_LEG_R_BASIS else fail_fast_threshold / max(actual_fraction, 1e-12)
        rows.append(
            {
                "event_id": _reason_tag(str(trade.get("entry_reason", "")), "event_id", ""),
                "symbol": str(trade.get("symbol", "")),
                "entry_time": trade.get("entry_time"),
                "exit_time": trade.get("exit_time"),
                "campaign_risk_budget_usdt": campaign_risk,
                "initial_leg_price_risk_usdt": float(trade.get("initial_leg_price_risk_usdt", 0.0) or 0.0),
                "initial_leg_full_cost_risk_usdt": initial_leg_risk_usdt,
                "initial_leg_actual_risk_fraction": actual_fraction,
                "capacity_fill_ratio": float(trade.get("capacity_fill_ratio", 1.0) or 0.0),
                "capacity_clipped_initial_risk_fraction": float(trade.get("capacity_clipped_initial_risk_fraction", 0.0) or 0.0),
                "stop_execution_price_estimate": float(trade.get("stop_execution_price_estimate", 0.0) or 0.0),
                "fixed_tp_initial_leg_r_equivalent": tp_initial_leg_r,
                "fail_fast_initial_leg_r_equivalent": fail_fast_initial_leg_r,
                "expected_stop_initial_leg_r": -1.0 if initial_leg_risk_usdt > 0.0 else None,
                "max_executable_initial_leg_r": trade.get("max_executable_initial_leg_r"),
                "max_executable_campaign_r": trade.get("max_executable_campaign_r"),
                "net_initial_leg_r": trade.get("net_initial_leg_r"),
                "net_campaign_r": trade.get("net_campaign_r"),
                "max_executable_r_legacy_basis": trade.get("max_executable_r_basis"),
            }
        )
    distribution_fields = (
        "campaign_risk_budget_usdt",
        "initial_leg_price_risk_usdt",
        "initial_leg_full_cost_risk_usdt",
        "initial_leg_actual_risk_fraction",
        "capacity_clipped_initial_risk_fraction",
        "fixed_tp_initial_leg_r_equivalent",
        "fail_fast_initial_leg_r_equivalent",
        "max_executable_initial_leg_r",
        "max_executable_campaign_r",
        "net_initial_leg_r",
        "net_campaign_r",
    )
    return {
        "trade_count": len(rows),
        "take_profit_r_basis": take_basis,
        "fail_fast_r_basis": fail_fast_basis,
        "legacy_max_executable_r_basis": "campaign",
        "capacity_clipped_trade_count": sum(float(row["capacity_fill_ratio"]) < 0.999 for row in rows),
        "distributions": {
            field: _numeric_distribution([row.get(field) for row in rows])
            for field in distribution_fields
        },
        "rows": rows,
    }


def _numeric_distribution(values: list[Any]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not ordered:
        return {"count": 0, "min": None, "p10": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None}

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "median": statistics.median(ordered),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": ordered[-1],
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


def _initialize_cmipr_risk_basis(
    config: Any,
    position: PortfolioPosition,
    campaign_risk_budget_usdt: float,
    planned_initial_risk_fraction: float,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> None:
    audit = initial_leg_risk(
        position,
        campaign_risk_budget_usdt,
        planned_initial_risk_fraction,
        execution_config,
        rules,
    )
    position.risk_budget_usdt = audit.campaign_risk_budget_usdt
    position.campaign_risk_budget_usdt = audit.campaign_risk_budget_usdt
    position.initial_leg_price_risk_usdt = audit.initial_leg_price_risk_usdt
    position.initial_leg_full_cost_risk_usdt = audit.initial_leg_full_cost_risk_usdt
    position.initial_leg_actual_risk_fraction = audit.initial_leg_actual_risk_fraction
    position.capacity_clipped_initial_risk_fraction = audit.capacity_clipped_initial_risk_fraction
    position.stop_execution_price_estimate = audit.stop_execution_price_estimate
    position.estimated_stop_exit_fee_usdt = audit.estimated_stop_exit_fee_usdt
    position.estimated_stop_exit_slippage_usdt = audit.estimated_stop_exit_slippage_usdt

    immediate_net = executable_net_pnl(
        position,
        position.raw_entry_price or position.entry_price,
        execution_config,
        rules,
    )
    campaign_r = net_pnl_r(position, immediate_net, CAMPAIGN_R_BASIS)
    initial_leg_r = net_pnl_r(position, immediate_net, INITIAL_LEG_R_BASIS)
    position.cmipr_max_campaign_executable_r = campaign_r
    position.cmipr_min_campaign_executable_r = campaign_r
    position.cmipr_max_initial_leg_executable_r = initial_leg_r
    position.cmipr_min_initial_leg_executable_r = initial_leg_r
    position.cmipr_executable_mfe_usdt = immediate_net


def _cmipr_initial_quantity_within_campaign_budget(
    signal: Signal,
    candle: Candle,
    requested_quantity: float,
    campaign_risk_budget_usdt: float,
    execution_config: BacktestExecutionConfig,
    rules: Any,
    exact_stop_price: float | None = None,
) -> float:
    """Cap initial quantity using the same full-cost stop path used by the R audit."""
    budget = max(0.0, float(campaign_risk_budget_usdt))
    quantity = max(0.0, float(requested_quantity))
    if budget <= 0.0 or quantity <= 0.0:
        return 0.0
    raw_price = float(candle.open)
    bar_quote_volume = abs(float(candle.volume) * raw_price)
    for _ in range(12):
        filled_quantity, _, _ = capacity_limited_quantity(
            execution_config,
            rules,
            quantity,
            raw_price,
            bar_quote_volume,
        )
        if filled_quantity <= 0.0:
            return 0.0
        entry_fill = market_entry_fill(
            execution_config,
            rules,
            signal.direction,
            filled_quantity,
            raw_price,
            bar_quote_volume,
        )
        if exact_stop_price is not None:
            raw_stop = float(exact_stop_price)
        elif signal.direction == Direction.LONG:
            raw_stop = entry_fill.price * (1.0 - signal.stop_loss_pct)
        else:
            raw_stop = entry_fill.price * (1.0 + signal.stop_loss_pct)
        stop_fill = market_exit_fill(
            execution_config,
            rules,
            signal.direction,
            filled_quantity,
            raw_stop,
            "stop_market",
            bar_quote_volume or None,
        )
        price_risk = max(
            0.0,
            signal.direction.value * filled_quantity * (entry_fill.price - stop_fill.price),
        )
        full_cost_risk = price_risk + entry_fill.fee + stop_fill.fee
        if full_cost_risk <= budget + 1e-9:
            return quantity
        ratio = budget / max(full_cost_risk, 1e-12)
        quantity = float(rules.round_quantity(quantity * max(0.0, min(0.995, ratio * 0.995))))
        if quantity <= 0.0:
            return 0.0
    return 0.0
    position.cmipr_executable_mae_usdt = immediate_net
    position.cmipr_max_executable_r = campaign_r


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
    mtper_engine: MtperEngine | None = None,
    mtper_stats: dict[str, int] | None = None,
    mtpc_engine: MtpcEngine | None = None,
    mtpc_stats: dict[str, int] | None = None,
    reversal_v2_stats: dict[str, int] | None = None,
    mtf_reject_stats: dict[str, int] | None = None,
) -> float:
    if not pending_entries:
        return cash
    portfolio_symbol_cooldown_until = portfolio_symbol_cooldown_until or {}
    portfolio_control_stats = portfolio_control_stats if portfolio_control_stats is not None else {}
    portfolio_runtime = portfolio_runtime or PortfolioRuntimeControl()
    reversal_v2_stats = reversal_v2_stats if reversal_v2_stats is not None else {}
    mtper_stats = mtper_stats if mtper_stats is not None else {}
    mtpc_stats = mtpc_stats if mtpc_stats is not None else {}
    mtf_reject_stats = mtf_reject_stats if mtf_reject_stats is not None else {}
    remaining: list[PendingEntry] = []
    filled = 0
    max_fills = max(1, int(config.trading.max_new_entries_per_cycle))
    for pending in pending_entries:
        candidate = pending.candidate
        mtper_pending = MTPER_REASON_TOKEN in str(candidate.signal.reason)
        mtper_symbol = candidate.symbol
        mtpc_pending = MTPC_REASON_TOKEN in str(candidate.signal.reason)
        mtpc_symbol = candidate.symbol
        if not _point_in_time_symbol_allowed(config, candidate.symbol, timestamp):
            if CMIPR_REASON_TOKEN in str(candidate.signal.reason):
                _count_stat(cmipr_stats if cmipr_stats is not None else {}, "initial_reject_point_in_time_universe")
            if MTPER_REASON_TOKEN in str(candidate.signal.reason):
                _count_stat(mtper_stats, "initial_reject_point_in_time_universe")
                if mtper_engine is not None:
                    mtper_engine.mark_cancelled(mtper_symbol, timestamp, "point_in_time_universe")
            if mtpc_pending:
                _count_stat(mtpc_stats, "initial_reject_point_in_time_universe")
                if mtpc_engine is not None:
                    mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, "point_in_time_universe")
            continue
        if pending.earliest_execution_index > execution_index:
            remaining.append(pending)
            continue
        if mtper_pending and pending.earliest_execution_index < execution_index:
            _count_stat(mtper_stats, "initial_reject_missed_next_1m_open")
            if mtper_engine is not None:
                mtper_engine.mark_cancelled(mtper_symbol, timestamp, "missed_next_1m_open")
            continue
        if mtpc_pending and pending.earliest_execution_index < execution_index:
            _count_stat(mtpc_stats, "initial_reject_missed_next_1m_open")
            if mtpc_engine is not None:
                mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, "missed_next_1m_open")
            continue
        if filled >= max_fills:
            remaining.append(pending)
            continue
        if candidate.symbol in positions:
            if mtper_pending and mtper_engine is not None:
                mtper_engine.mark_cancelled(mtper_symbol, timestamp, "symbol_already_open")
            if mtpc_pending and mtpc_engine is not None:
                mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, "symbol_already_open")
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
            if MTPER_REASON_TOKEN in str(candidate.signal.reason):
                _count_stat(mtper_stats, f"initial_reject_{portfolio_reason}")
                if mtper_engine is not None:
                    mtper_engine.mark_cancelled(mtper_symbol, timestamp, portfolio_reason)
            if mtpc_pending:
                _count_stat(mtpc_stats, f"initial_reject_{portfolio_reason}")
                if mtpc_engine is not None:
                    mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, portfolio_reason)
            continue
        if portfolio_multiplier <= 0:
            _portfolio_count(portfolio_control_stats, "portfolio_zero_risk")
            if mtper_pending and mtper_engine is not None:
                mtper_engine.mark_cancelled(mtper_symbol, timestamp, "portfolio_zero_risk")
            if mtpc_pending and mtpc_engine is not None:
                mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, "portfolio_zero_risk")
            continue
        if portfolio_multiplier < 0.999:
            adjusted_signal = replace(
                candidate.signal,
                risk_multiplier=candidate.signal.risk_multiplier * portfolio_multiplier,
            )
            candidate = replace(candidate, signal=adjusted_signal)
        candle = execution_candles_by_symbol[candidate.symbol][execution_index]
        candidate = _reversal_v2_adjust_candidate_for_fill(
            config,
            candidate,
            candle.open,
            execution_config,
            reversal_v2_stats,
        )
        if candidate is None:
            continue
        candidate = _mtf_adjust_candidate_for_fill(
            config,
            candidate,
            candle.open,
            execution_config,
            mtf_reject_stats,
        )
        if candidate is None:
            continue
        adjusted_mtper_candidate = _mtper_adjust_candidate_for_fill(
            config,
            candidate,
            candle.open,
            execution_config,
            mtper_stats,
        )
        if adjusted_mtper_candidate is None:
            if mtper_pending and mtper_engine is not None:
                mtper_engine.mark_cancelled(mtper_symbol, timestamp, "fill_revalidation")
            continue
        candidate = adjusted_mtper_candidate
        adjusted_mtpc_candidate = _mtpc_adjust_candidate_for_fill(
            config,
            candidate,
            candle.open,
            execution_config,
            mtpc_stats,
        )
        if adjusted_mtpc_candidate is None:
            if mtpc_pending and mtpc_engine is not None:
                mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, "fill_revalidation")
            continue
        candidate = adjusted_mtpc_candidate
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
        sizing_account = _mtpc_sizing_account(
            config,
            _mtper_sizing_account(config, _cmipr_sizing_account(config, account, candidate.signal), candidate.signal),
            candidate.signal,
        )
        quantity_text, reason = trader._size_order(candidate.symbol, execution_candle.close, candidate.signal, sizing_account)
        quantity = float(quantity_text)
        if reason != "ok" or quantity <= 0:
            if CMIPR_REASON_TOKEN in str(candidate.signal.reason):
                _count_stat(cmipr_stats if cmipr_stats is not None else {}, f"initial_reject_size_{reason}")
            if mtper_pending:
                _count_stat(mtper_stats, f"initial_reject_size_{reason}")
                if mtper_engine is not None:
                    mtper_engine.mark_cancelled(mtper_symbol, timestamp, f"size_{reason}")
            if mtpc_pending:
                _count_stat(mtpc_stats, f"initial_reject_size_{reason}")
                if mtpc_engine is not None:
                    mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, f"size_{reason}")
            continue
        rules = client.symbol_rules(candidate.symbol)
        campaign_risk_budget = None
        if CMIPR_REASON_TOKEN in str(candidate.signal.reason):
            side_multiplier = candidate.signal.risk_multiplier / max(float(config.cmipr.entry.initial_risk_fraction), 1e-12)
            campaign_risk_budget = _cmipr_trade_risk_budget(config, pre_entry_equity) * min(1.0, max(0.0, side_multiplier))
            capped_quantity = _cmipr_initial_quantity_within_campaign_budget(
                candidate.signal,
                execution_candle,
                quantity,
                campaign_risk_budget,
                execution_config,
                rules,
            )
            if capped_quantity <= 0.0:
                _count_stat(cmipr_stats if cmipr_stats is not None else {}, "initial_reject_full_cost_campaign_risk_cap")
                continue
            if capped_quantity + 1e-12 < quantity:
                _count_stat(cmipr_stats if cmipr_stats is not None else {}, "initial_quantity_capped_by_full_cost_campaign_risk")
            quantity = capped_quantity
        if MTPER_REASON_TOKEN in str(candidate.signal.reason):
            initial_fraction = max(float(config.mtper.entry.initial_risk_fraction), 1e-12)
            side_multiplier = candidate.signal.risk_multiplier / initial_fraction
            campaign_risk_budget = _mtper_trade_risk_budget(config, pre_entry_equity) * min(1.0, max(0.0, side_multiplier))
            initial_budget = campaign_risk_budget * initial_fraction
            capped_quantity = _cmipr_initial_quantity_within_campaign_budget(
                candidate.signal,
                execution_candle,
                quantity,
                initial_budget,
                execution_config,
                rules,
            )
            if capped_quantity <= 0.0:
                _count_stat(mtper_stats, "initial_reject_full_cost_campaign_risk_cap")
                if mtper_engine is not None:
                    mtper_engine.mark_cancelled(mtper_symbol, timestamp, "full_cost_risk_cap")
                continue
            if capped_quantity + 1e-12 < quantity:
                _count_stat(mtper_stats, "initial_quantity_capped_by_full_cost_risk")
            quantity = capped_quantity
        if MTPC_REASON_TOKEN in str(candidate.signal.reason):
            campaign_risk_budget = _mtpc_trade_risk_budget(config, pre_entry_equity)
            if not _mtpc_total_open_risk_allows_entry(
                config,
                positions,
                pre_entry_equity,
                campaign_risk_budget,
            ):
                _count_stat(mtpc_stats, "initial_reject_max_total_open_risk")
                if mtpc_engine is not None:
                    mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, "max_total_open_risk")
                continue
            capacity_quantity, _, _ = capacity_limited_quantity(
                execution_config,
                rules,
                quantity,
                execution_candle.open,
                abs(execution_candle.volume * execution_candle.open),
            )
            if capacity_quantity <= 0.0:
                _count_stat(mtpc_stats, "initial_reject_execution_capacity")
                if mtpc_engine is not None:
                    mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, "execution_capacity")
                continue
            exact_stop = float(candidate.metadata.get("structural_stop_price", 0.0) or 0.0)
            capped_quantity = _cmipr_initial_quantity_within_campaign_budget(
                candidate.signal,
                execution_candle,
                quantity,
                campaign_risk_budget,
                execution_config,
                rules,
                exact_stop_price=exact_stop if exact_stop > 0.0 else None,
            )
            if capped_quantity <= 0.0:
                _count_stat(mtpc_stats, "initial_reject_full_cost_risk_cap")
                if mtpc_engine is not None:
                    mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, "full_cost_risk_cap")
                continue
            if capped_quantity + 1e-12 < quantity:
                _count_stat(mtpc_stats, "initial_quantity_capped_by_full_cost_risk")
            quantity = capped_quantity
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
            position.strategy_metadata = dict(candidate.metadata)
            if _is_cmipr_position(position):
                assert campaign_risk_budget is not None
                position.initial_stop_price = position.stop_price
                _initialize_cmipr_risk_basis(
                    config,
                    position,
                    campaign_risk_budget,
                    candidate.signal.risk_multiplier,
                    execution_config,
                    rules,
                )
                if cmipr_engine is not None:
                    if position.capacity_fill_ratio < 0.999:
                        cmipr_engine.mark_partial_fill(candidate.symbol)
                    cmipr_engine.mark_protection_pending(candidate.symbol)
                    cmipr_engine.mark_protected(candidate.symbol)
                _count_stat(cmipr_stats if cmipr_stats is not None else {}, "initial_fill_count")
            if _is_mtper_position(position):
                assert campaign_risk_budget is not None
                _initialize_mtper_position(
                    config,
                    position,
                    candidate,
                    campaign_risk_budget,
                    execution_config,
                    rules,
                )
                if mtper_engine is not None:
                    mtper_engine.mark_filled(
                        candidate.symbol,
                        position.quantity,
                        position.quantity / max(position.capacity_fill_ratio, 1e-12),
                    )
                    mtper_engine.mark_protected(candidate.symbol)
                _count_stat(mtper_stats, "initial_fill_count")
            if _is_mtpc_position(position):
                assert campaign_risk_budget is not None
                _initialize_mtpc_position(
                    config,
                    position,
                    candidate,
                    campaign_risk_budget,
                    execution_config,
                    rules,
                )
                if mtpc_engine is not None:
                    mtpc_engine.mark_filled(candidate.symbol, position.capacity_fill_ratio < 0.999, timestamp)
                    mtpc_engine.mark_protected(candidate.symbol, timestamp)
                _count_stat(mtpc_stats, "initial_fill_count")
            if _is_reversal_v2_position(position):
                _initialize_reversal_v2_position(position, candidate)
                _count_stat(reversal_v2_stats, "entry_count")
            if MTF_REASON_TOKEN in str(position.entry_reason):
                _initialize_mtf_position(config, position, candidate)
            _record_monthly_open(monthly_stats, timestamp, candidate.signal.direction)
            filled += 1
        elif CMIPR_REASON_TOKEN in str(candidate.signal.reason):
            _count_stat(cmipr_stats if cmipr_stats is not None else {}, "initial_reject_execution_capacity_or_exchange_rules")
        elif MTPER_REASON_TOKEN in str(candidate.signal.reason):
            _count_stat(mtper_stats, "initial_reject_execution_capacity_or_exchange_rules")
            if mtper_engine is not None:
                mtper_engine.mark_cancelled(mtper_symbol, timestamp, "execution_capacity_or_exchange_rules")
        elif MTPC_REASON_TOKEN in str(candidate.signal.reason):
            _count_stat(mtpc_stats, "initial_reject_execution_capacity_or_exchange_rules")
            if mtpc_engine is not None:
                mtpc_engine.mark_cancelled(mtpc_symbol, timestamp, "execution_capacity_or_exchange_rules")
    pending_entries[:] = remaining
    return cash


def _reversal_v2_adjust_candidate_for_fill(
    config: Any,
    candidate: EntryCandidate,
    raw_entry_price: float,
    execution_config: BacktestExecutionConfig,
    stats: dict[str, int],
) -> EntryCandidate | None:
    if REVERSAL_V2_REASON_TOKEN not in str(candidate.signal.reason):
        return candidate
    alpha = config.reversal_alpha
    side = candidate.signal.direction.value
    setup_atr = max(float(candidate.metadata.get("setup_atr", 0.0)), 1e-12)
    trigger_close = max(float(candidate.metadata.get("trigger_close", 0.0)), 1e-12)
    structural_stop = float(candidate.metadata.get("structural_stop_price", 0.0))
    chase_atr = side * (raw_entry_price - trigger_close) / setup_atr
    if chase_atr > float(alpha.max_entry_chase_atr):
        _count_stat(stats, "reject_entry_chase")
        return None
    stop_distance = side * (raw_entry_price - structural_stop)
    if stop_distance <= 0:
        _count_stat(stats, "reject_entry_beyond_stop")
        return None
    stop_atr = stop_distance / setup_atr
    stop_pct = stop_distance / max(raw_entry_price, 1e-12)
    if stop_atr < float(alpha.min_stop_atr) or stop_pct < float(alpha.min_stop_pct):
        _count_stat(stats, "reject_fill_stop_too_tight")
        return None
    if stop_atr > float(alpha.max_stop_atr) or stop_pct > float(alpha.max_stop_pct):
        _count_stat(stats, "reject_fill_stop_too_wide")
        return None
    take_profit_r = max(0.1, float(alpha.take_profit_r))
    target_pct = stop_pct * take_profit_r
    round_trip_cost_pct = (
        2.0 * float(execution_config.taker_fee_rate)
        + (
            float(execution_config.market_slippage_bps)
            + float(execution_config.take_profit_slippage_bps)
        )
        / 10_000.0
    )
    target_to_cost = target_pct / max(round_trip_cost_pct, 1e-12)
    if target_to_cost < float(alpha.min_target_to_cost_ratio):
        _count_stat(stats, "reject_cost_edge")
        return None
    signal = replace(
        candidate.signal,
        stop_loss_pct=stop_pct,
        take_profit_pct=target_pct,
    )
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "entry_chase_atr": chase_atr,
            "fill_stop_atr": stop_atr,
            "fill_stop_pct": stop_pct,
            "target_to_cost_ratio": target_to_cost,
        }
    )
    return replace(candidate, signal=signal, metadata=metadata)


def _mtper_trade_risk_budget(config: Any, current_equity: float) -> float:
    risk_pct = min(
        max(0.0, float(config.mtper.risk_control.campaign_risk_pct)),
        max(0.0, float(config.mtper.risk_control.max_campaign_risk_pct)),
    )
    return max(0.0, float(current_equity)) * risk_pct


def _mtper_sizing_account(config: Any, account: Any, signal: Signal) -> Any:
    if MTPER_REASON_TOKEN not in str(signal.reason):
        return account
    global_risk_pct = max(float(config.risk.risk_per_trade_pct), 1e-12)
    campaign_risk_pct = min(
        max(0.0, float(config.mtper.risk_control.campaign_risk_pct)),
        max(0.0, float(config.mtper.risk_control.max_campaign_risk_pct)),
    )
    if abs(global_risk_pct - campaign_risk_pct) <= 1e-12:
        return account
    sizing_equity = max(0.0, float(account.equity) * campaign_risk_pct / global_risk_pct)
    return replace(
        account,
        equity=sizing_equity,
        wallet_balance=sizing_equity,
        available_balance=max(0.0, min(account.available_balance, sizing_equity - account.initial_margin)),
    )


def _mtper_adjust_candidate_for_fill(
    config: Any,
    candidate: EntryCandidate,
    raw_entry_price: float,
    execution_config: BacktestExecutionConfig,
    stats: dict[str, int],
) -> EntryCandidate | None:
    if MTPER_REASON_TOKEN not in str(candidate.signal.reason):
        return candidate
    side = candidate.signal.direction.value
    metadata = dict(candidate.metadata)
    trigger_atr = max(float(metadata.get("trigger_atr_15m", 0.0)), 1e-12)
    trigger_close = max(float(metadata.get("trigger_close", 0.0)), 1e-12)
    structural_stop = float(metadata.get("structural_stop_price", 0.0))
    chase_atr = side * (raw_entry_price - trigger_close) / trigger_atr
    if chase_atr > float(config.mtper.entry.max_entry_chase_atr):
        _count_stat(stats, "reject_entry_chase")
        return None
    stop_distance = side * (raw_entry_price - structural_stop)
    if stop_distance <= 0:
        _count_stat(stats, "reject_entry_beyond_hard_stop")
        return None
    stop_atr = stop_distance / trigger_atr
    stop_pct = stop_distance / max(raw_entry_price, 1e-12)
    entry_cfg = config.mtper.entry
    if stop_atr < float(entry_cfg.min_stop_atr_15m) or stop_pct < float(entry_cfg.min_stop_pct):
        _count_stat(stats, "reject_fill_stop_too_tight")
        return None
    if stop_atr > float(entry_cfg.max_stop_atr_15m) or stop_pct > float(entry_cfg.max_stop_pct):
        _count_stat(stats, "reject_fill_stop_too_wide")
        return None
    leverage = max(1, int(getattr(config.trading, "leverage", 1)))
    maintenance_rate = max(0.0, float(getattr(config.risk, "estimated_maintenance_margin_rate", 0.005)))
    buffer_pct = max(0.0, float(config.mtper.risk_control.liquidation_buffer_pct))
    if candidate.signal.direction == Direction.LONG:
        liquidation_estimate = raw_entry_price * (1.0 - 1.0 / leverage + maintenance_rate)
        liquidation_safe = structural_stop > liquidation_estimate + raw_entry_price * buffer_pct
    else:
        liquidation_estimate = raw_entry_price * (1.0 + 1.0 / leverage - maintenance_rate)
        liquidation_safe = structural_stop < liquidation_estimate - raw_entry_price * buffer_pct
    if not liquidation_safe:
        _count_stat(stats, "reject_liquidation_before_hard_stop")
        return None
    targets = []
    for index, key in enumerate(("target_1_price", "target_2_price", "target_3_price"), start=1):
        configured = float(metadata.get(key, raw_entry_price + side * stop_distance * 0.5 * index))
        minimum = raw_entry_price + side * stop_distance * 0.5 * index
        target = configured if side * (configured - minimum) >= 0 else minimum
        targets.append(target)
    if candidate.signal.direction == Direction.LONG:
        targets.sort()
    else:
        targets.sort(reverse=True)
    target_distance = side * (targets[1] - raw_entry_price)
    round_trip_cost_pct = (
        2.0 * float(execution_config.taker_fee_rate)
        + (float(execution_config.market_slippage_bps) + float(execution_config.take_profit_slippage_bps)) / 10_000.0
    )
    target_to_cost = target_distance / max(raw_entry_price * round_trip_cost_pct, 1e-12)
    if target_to_cost < float(entry_cfg.min_target_to_cost_ratio):
        _count_stat(stats, "reject_fill_target_to_cost")
        return None
    signal = replace(
        candidate.signal,
        stop_loss_pct=stop_pct,
        take_profit_pct=side * (targets[2] - raw_entry_price) / max(raw_entry_price, 1e-12),
    )
    metadata.update(
        {
            "entry_chase_atr": chase_atr,
            "fill_stop_atr_15m": stop_atr,
            "fill_stop_pct": stop_pct,
            "target_1_price": targets[0],
            "target_2_price": targets[1],
            "target_3_price": targets[2],
            "target_to_cost_ratio": target_to_cost,
            "liquidation_price_estimate": liquidation_estimate,
            "liquidation_buffer_ok": liquidation_safe,
        }
    )
    return replace(candidate, signal=signal, metadata=metadata)


def _initialize_mtper_position(
    config: Any,
    position: PortfolioPosition,
    candidate: EntryCandidate,
    campaign_risk_budget_usdt: float,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> None:
    metadata = dict(candidate.metadata)
    metadata["initial_quantity"] = position.quantity
    metadata["target_1_done"] = False
    metadata["target_2_done"] = False
    metadata["trend_converted"] = False
    structural_stop = float(metadata.get("structural_stop_price", position.stop_price))
    position.stop_price = structural_stop
    position.initial_stop_price = structural_stop
    position.take_profit_price = float(metadata.get("target_3_price", position.take_profit_price))
    position.strategy_metadata = metadata
    audit = initial_leg_risk(
        position,
        campaign_risk_budget_usdt,
        float(config.mtper.entry.initial_risk_fraction),
        execution_config,
        rules,
    )
    position.risk_budget_usdt = audit.campaign_risk_budget_usdt
    position.campaign_risk_budget_usdt = audit.campaign_risk_budget_usdt
    position.initial_leg_price_risk_usdt = audit.initial_leg_price_risk_usdt
    position.initial_leg_full_cost_risk_usdt = audit.initial_leg_full_cost_risk_usdt
    position.initial_leg_actual_risk_fraction = audit.initial_leg_actual_risk_fraction
    position.capacity_clipped_initial_risk_fraction = audit.capacity_clipped_initial_risk_fraction
    position.stop_execution_price_estimate = audit.stop_execution_price_estimate
    position.estimated_stop_exit_fee_usdt = audit.estimated_stop_exit_fee_usdt
    position.estimated_stop_exit_slippage_usdt = audit.estimated_stop_exit_slippage_usdt
    immediate_net = executable_net_pnl(
        position,
        position.raw_entry_price or position.entry_price,
        execution_config,
        rules,
    )
    campaign_r = net_pnl_r(position, immediate_net, CAMPAIGN_R_BASIS)
    initial_leg_r = net_pnl_r(position, immediate_net, INITIAL_LEG_R_BASIS)
    position.mtper_max_campaign_executable_r = campaign_r
    position.mtper_min_campaign_executable_r = campaign_r
    position.mtper_max_initial_leg_executable_r = initial_leg_r
    position.mtper_min_initial_leg_executable_r = initial_leg_r
    position.mtper_executable_mfe_usdt = immediate_net
    position.mtper_executable_mae_usdt = immediate_net


def _mtpc_trade_risk_budget(config: Any, current_equity: float) -> float:
    risk_pct = min(
        max(0.0, float(config.mtpc.risk_control.trade_risk_pct)),
        max(0.0, float(config.mtpc.risk_control.max_trade_risk_pct)),
    )
    return max(0.0, float(current_equity)) * risk_pct


def _mtpc_open_campaign_risk_usdt(positions: dict[str, PortfolioPosition]) -> float:
    return sum(
        max(0.0, float(position.campaign_risk_budget_usdt or position.risk_budget_usdt))
        for position in positions.values()
        if _is_mtpc_position(position)
    )


def _mtpc_total_open_risk_allows_entry(
    config: Any,
    positions: dict[str, PortfolioPosition],
    current_equity: float,
    new_campaign_risk_usdt: float,
) -> bool:
    limit_pct = max(0.0, float(config.mtpc.risk_control.max_total_open_risk_pct))
    if limit_pct <= 0.0:
        return False
    limit_usdt = max(0.0, float(current_equity)) * limit_pct
    projected = _mtpc_open_campaign_risk_usdt(positions) + max(0.0, float(new_campaign_risk_usdt))
    return projected <= limit_usdt + 1e-9


def _mtpc_exit_uses_initial_leg_r(config: Any) -> bool:
    basis = str(getattr(config.mtpc.exit, "r_basis", "initial_leg")).strip().lower()
    if basis not in {"initial_leg", "campaign"}:
        raise ValueError(f"unsupported MTPC exit R basis: {basis}")
    return basis == "initial_leg"


def _mtpc_sizing_account(config: Any, account: Any, signal: Signal) -> Any:
    if MTPC_REASON_TOKEN not in str(signal.reason):
        return account
    global_risk_pct = max(float(config.risk.risk_per_trade_pct), 1e-12)
    mtpc_risk_pct = min(
        max(0.0, float(config.mtpc.risk_control.trade_risk_pct)),
        max(0.0, float(config.mtpc.risk_control.max_trade_risk_pct)),
    )
    if abs(global_risk_pct - mtpc_risk_pct) <= 1e-12:
        return account
    sizing_equity = max(0.0, float(account.equity) * mtpc_risk_pct / global_risk_pct)
    return replace(
        account,
        equity=sizing_equity,
        wallet_balance=sizing_equity,
        available_balance=max(0.0, min(account.available_balance, sizing_equity - account.initial_margin)),
    )


def _mtpc_adjust_candidate_for_fill(
    config: Any,
    candidate: EntryCandidate,
    raw_entry_price: float,
    execution_config: BacktestExecutionConfig,
    stats: dict[str, int],
) -> EntryCandidate | None:
    if MTPC_REASON_TOKEN not in str(candidate.signal.reason):
        return candidate
    side = candidate.signal.direction.value
    metadata = dict(candidate.metadata)
    trigger_atr = max(float(metadata.get("trigger_atr_15m", 0.0)), 1e-12)
    trigger_close = max(float(metadata.get("trigger_close", 0.0)), 1e-12)
    structural_stop = float(metadata.get("structural_stop_price", 0.0))
    chase_atr = side * (raw_entry_price - trigger_close) / trigger_atr
    if chase_atr > float(config.mtpc.pullback.max_entry_chase_atr):
        _count_stat(stats, "reject_entry_chase")
        return None
    stop_distance = side * (raw_entry_price - structural_stop)
    if stop_distance <= 0:
        _count_stat(stats, "reject_entry_beyond_structural_stop")
        return None
    stop_atr = stop_distance / trigger_atr
    stop_pct = stop_distance / max(raw_entry_price, 1e-12)
    entry_cfg = config.mtpc.pullback
    if stop_atr < float(entry_cfg.min_stop_atr) or stop_pct < float(entry_cfg.min_stop_pct):
        _count_stat(stats, "reject_fill_stop_too_tight")
        return None
    if stop_atr > float(entry_cfg.max_stop_atr) or stop_pct > float(entry_cfg.max_stop_pct):
        _count_stat(stats, "reject_fill_stop_too_wide")
        return None
    leverage = max(1, int(getattr(config.trading, "leverage", 1)))
    maintenance_rate = max(0.0, float(getattr(config.risk, "estimated_maintenance_margin_rate", 0.005)))
    buffer_pct = max(0.0, float(config.mtpc.risk_control.liquidation_buffer_pct))
    if candidate.signal.direction == Direction.LONG:
        liquidation_estimate = raw_entry_price * (1.0 - 1.0 / leverage + maintenance_rate)
        liquidation_safe = structural_stop > liquidation_estimate + raw_entry_price * buffer_pct
    else:
        liquidation_estimate = raw_entry_price * (1.0 + 1.0 / leverage - maintenance_rate)
        liquidation_safe = structural_stop < liquidation_estimate - raw_entry_price * buffer_pct
    if not liquidation_safe:
        _count_stat(stats, "reject_liquidation_before_stop")
        return None
    target_1_r = max(0.1, float(config.mtpc.exit.take_profit_1_r))
    target_2_r = max(target_1_r, float(config.mtpc.exit.take_profit_2_r))
    target_1 = raw_entry_price + side * stop_distance * target_1_r
    target_2 = raw_entry_price + side * stop_distance * target_2_r
    round_trip_cost_pct = (
        2.0 * float(execution_config.taker_fee_rate)
        + (
            float(execution_config.market_slippage_bps)
            + float(execution_config.take_profit_slippage_bps)
        )
        / 10_000.0
    )
    target_to_cost = side * (target_1 - raw_entry_price) / max(
        raw_entry_price * round_trip_cost_pct, 1e-12
    )
    if target_to_cost < float(entry_cfg.min_target_to_cost_ratio):
        _count_stat(stats, "reject_fill_target_to_cost")
        return None
    signal = replace(
        candidate.signal,
        stop_loss_pct=stop_pct,
        take_profit_pct=side * (target_2 - raw_entry_price) / max(raw_entry_price, 1e-12),
    )
    metadata.update(
        {
            "entry_chase_atr": chase_atr,
            "fill_stop_atr_15m": stop_atr,
            "fill_stop_pct": stop_pct,
            "target_1_price": target_1,
            "target_2_price": target_2,
            "target_to_cost_ratio": target_to_cost,
            "liquidation_price_estimate": liquidation_estimate,
            "liquidation_buffer_ok": True,
        }
    )
    return replace(candidate, signal=signal, metadata=metadata)


def _initialize_mtpc_position(
    config: Any,
    position: PortfolioPosition,
    candidate: EntryCandidate,
    campaign_risk_budget_usdt: float,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> None:
    metadata = dict(candidate.metadata)
    metadata["initial_quantity"] = position.quantity
    metadata["target_1_done"] = False
    structural_stop = float(metadata.get("structural_stop_price", position.stop_price))
    position.stop_price = structural_stop
    position.initial_stop_price = structural_stop
    position.take_profit_price = float(metadata.get("target_2_price", position.take_profit_price))
    position.strategy_metadata = metadata
    audit = initial_leg_risk(
        position,
        campaign_risk_budget_usdt,
        1.0,
        execution_config,
        rules,
    )
    position.risk_budget_usdt = audit.campaign_risk_budget_usdt
    position.campaign_risk_budget_usdt = audit.campaign_risk_budget_usdt
    position.initial_leg_price_risk_usdt = audit.initial_leg_price_risk_usdt
    position.initial_leg_full_cost_risk_usdt = audit.initial_leg_full_cost_risk_usdt
    position.initial_leg_actual_risk_fraction = audit.initial_leg_actual_risk_fraction
    position.capacity_clipped_initial_risk_fraction = audit.capacity_clipped_initial_risk_fraction
    position.stop_execution_price_estimate = audit.stop_execution_price_estimate
    position.estimated_stop_exit_fee_usdt = audit.estimated_stop_exit_fee_usdt
    position.estimated_stop_exit_slippage_usdt = audit.estimated_stop_exit_slippage_usdt
    immediate_net = executable_net_pnl(
        position,
        position.raw_entry_price or position.entry_price,
        execution_config,
        rules,
    )
    campaign_r = net_pnl_r(position, immediate_net, CAMPAIGN_R_BASIS)
    initial_leg_r = net_pnl_r(position, immediate_net, INITIAL_LEG_R_BASIS)
    position.mtpc_max_campaign_executable_r = campaign_r
    position.mtpc_min_campaign_executable_r = campaign_r
    position.mtpc_max_initial_leg_executable_r = initial_leg_r
    position.mtpc_min_initial_leg_executable_r = initial_leg_r
    selected_r = initial_leg_r if _mtpc_exit_uses_initial_leg_r(config) else campaign_r
    position.mtpc_max_executable_r = selected_r
    position.mtpc_min_executable_r = selected_r
    position.mtpc_executable_mfe_usdt = immediate_net
    position.mtpc_executable_mae_usdt = immediate_net


def _mtf_adjust_candidate_for_fill(
    config: Any,
    candidate: EntryCandidate,
    raw_entry_price: float,
    execution_config: BacktestExecutionConfig,
    stats: dict[str, int],
) -> EntryCandidate | None:
    if MTF_REASON_TOKEN not in str(candidate.signal.reason):
        return candidate
    strategy = config.strategy
    side = candidate.signal.direction.value
    trigger_atr = max(float(candidate.metadata.get("trigger_atr", 0.0)), 1e-12)
    trigger_close = max(float(candidate.metadata.get("trigger_close", 0.0)), 1e-12)
    structural_stop = float(candidate.metadata.get("structural_stop_price", 0.0))
    chase_atr = side * (raw_entry_price - trigger_close) / trigger_atr
    if chase_atr > float(getattr(strategy, "mtf_max_entry_chase_atr", 999.0)):
        _count_mtf_reject(stats, "mtf_entry_chase")
        return None
    stop_distance = side * (raw_entry_price - structural_stop)
    if stop_distance <= 0:
        _count_mtf_reject(stats, "mtf_entry_beyond_stop")
        return None
    stop_pct = stop_distance / max(raw_entry_price, 1e-12)
    min_stop = max(0.0, float(getattr(strategy, "mtf_min_stop_pct", 0.0)))
    max_stop = max(0.0, float(getattr(strategy, "mtf_max_stop_pct", 0.0)))
    if min_stop > 0 and stop_pct < min_stop:
        _count_mtf_reject(stats, "mtf_fill_stop_too_tight")
        return None
    if max_stop > 0 and stop_pct > max_stop:
        _count_mtf_reject(stats, "mtf_fill_stop_too_wide")
        return None
    take_profit_r = max(0.1, float(getattr(strategy, "mtf_take_profit_r", 2.0)))
    target_pct = stop_pct * take_profit_r
    conservative_cost_pct = (
        2.0 * float(execution_config.taker_fee_rate)
        + (
            float(execution_config.market_slippage_bps)
            + max(
                float(execution_config.stop_slippage_bps),
                float(execution_config.take_profit_slippage_bps),
            )
        )
        / 10_000.0
    )
    target_to_cost = target_pct / max(conservative_cost_pct, 1e-12)
    if target_to_cost < float(getattr(strategy, "mtf_min_target_to_cost_ratio", 0.0)):
        _count_mtf_reject(stats, "mtf_cost_edge")
        return None
    signal = replace(
        candidate.signal,
        stop_loss_pct=stop_pct,
        take_profit_pct=target_pct,
    )
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "entry_chase_atr": chase_atr,
            "fill_stop_pct": stop_pct,
            "target_to_cost_ratio": target_to_cost,
        }
    )
    return replace(candidate, signal=signal, metadata=metadata)


def _initialize_mtf_position(config: Any, position: PortfolioPosition, candidate: EntryCandidate) -> None:
    side = position.direction.value
    structural_stop = float(candidate.metadata.get("structural_stop_price", position.stop_price))
    risk_distance = side * (position.entry_price - structural_stop)
    if risk_distance <= 0:
        return
    take_profit_r = max(0.1, float(candidate.metadata.get("take_profit_r", 2.0)))
    exit_mode = _mtf_exit_mode(config)
    if exit_mode == "partial_runner":
        take_profit_r = max(
            0.1,
            float(getattr(config.strategy, "mtf_partial_take_profit_r", take_profit_r)),
        )
    position.stop_price = structural_stop
    position.initial_stop_price = structural_stop
    position.take_profit_price = position.entry_price + side * risk_distance * take_profit_r
    position.risk_budget_usdt = risk_distance * position.quantity
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "mtf_exit_mode": exit_mode,
            "mtf_initial_quantity": position.quantity,
            "mtf_partial_take_profit_r": take_profit_r,
            "mtf_partial_take_profit_done": False,
        }
    )
    position.strategy_metadata = metadata


def _initialize_reversal_v2_position(position: PortfolioPosition, candidate: EntryCandidate) -> None:
    side = position.direction.value
    structural_stop = float(candidate.metadata.get("structural_stop_price", position.stop_price))
    risk_distance = side * (position.entry_price - structural_stop)
    if risk_distance <= 0:
        return
    position.stop_price = structural_stop
    position.initial_stop_price = structural_stop
    position.take_profit_price = position.entry_price + side * risk_distance * max(
        0.1,
        float(candidate.metadata.get("take_profit_r", 1.5)),
    )
    position.risk_budget_usdt = risk_distance * position.quantity
    position.strategy_metadata = dict(candidate.metadata)


def _candidate_signal_times(config: Any, candidate: Any, fallback_signal_time: Any) -> tuple[Any, Any]:
    if REVERSAL_V2_REASON_TOKEN in str(candidate.signal.reason):
        signal_time = candidate.candle.timestamp
        return signal_time, signal_time + timedelta(milliseconds=interval_to_milliseconds("5m"))
    if CMIPR_REASON_TOKEN in str(candidate.signal.reason):
        trigger_timeframe = _reason_tag(candidate.signal.reason, "trigger_tf", "15m")
        signal_time = candidate.candle.timestamp
        return signal_time, signal_time + timedelta(milliseconds=interval_to_milliseconds(trigger_timeframe))
    if MTPER_REASON_TOKEN in str(candidate.signal.reason):
        trigger_timeframe = str(candidate.metadata.get("trigger_timeframe", _reason_tag(candidate.signal.reason, "trigger_tf", "15m")))
        signal_time = candidate.candle.timestamp
        return signal_time, signal_time + timedelta(milliseconds=interval_to_milliseconds(trigger_timeframe))
    if MTPC_REASON_TOKEN in str(candidate.signal.reason):
        trigger_timeframe = str(candidate.metadata.get("trigger_timeframe", "5m"))
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
    if bool(getattr(getattr(config, "mtper", None), "enabled", False)):
        timeframes.update(("15m", "30m", "1h", "2h", "4h"))
    if bool(getattr(getattr(config, "mtpc", None), "enabled", False)):
        timeframes.update(("5m", "15m", "1h", "4h"))
    if bool(getattr(getattr(config, "reversal_alpha", None), "enabled", False)):
        timeframes.update(("5m", "15m", "30m", "1h"))
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
    mtf_candidate_cache: dict[tuple[Any, ...], EntryCandidate | None],
    positions: dict[str, PortfolioPosition],
    reentry_block_until: dict[str, Any],
    signal_index: int,
    timestamp: Any,
    indicator_reversal_pause_until_time: dict[Direction, Any],
    alpha_diagnostics: AlphaCandidateDiagnostics,
    reversal_alpha_engine: ReversalAlphaEngine | None = None,
    reversal_v2_consumed_events: set[str] | None = None,
    reversal_v2_stats: dict[str, int] | None = None,
) -> list[Any]:
    candidates = []
    reversal_v2_consumed_events = reversal_v2_consumed_events if reversal_v2_consumed_events is not None else set()
    reversal_v2_stats = reversal_v2_stats if reversal_v2_stats is not None else {}
    entry_symbols = set(config.trading.entry_symbols or config.trading.symbols)
    legacy_disabled = bool(getattr(config.strategy, "mtf_disable_legacy_strategies", False)) and bool(getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False))
    low_base_only = bool(getattr(config.strategy, "low_base_volume_ignition_enabled", False)) and bool(getattr(config.strategy, "low_base_volume_ignition_disable_legacy", False))
    for symbol in config.trading.symbols:
        if symbol in positions or symbol not in entry_symbols:
            continue
        if reentry_block_until.get(symbol) and timestamp < reentry_block_until[symbol]:
            continue
        candidate = None
        if reversal_alpha_engine is not None and reversal_alpha_engine.execution_enabled:
            candidate = _reversal_v2_candidate(
                reversal_alpha_engine,
                config,
                symbol,
                timestamp,
                reversal_v2_consumed_events,
                reversal_v2_stats,
            )
            if candidate is not None:
                candidates.append(candidate)
            continue
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
    if bool(getattr(config.strategy, "mtf_quality_sleeve_enabled", False)):
        candidates.sort(
            key=lambda item: (
                not bool(item.metadata.get("mtf_quality_sleeve", False)),
                item.rank_score,
            ),
            reverse=True,
        )
    else:
        candidates.sort(key=lambda item: item.rank_score, reverse=True)
    return candidates


def _merge_candidate_sleeves(
    primary: list[EntryCandidate],
    secondary: list[EntryCandidate],
) -> list[EntryCandidate]:
    """Keep sleeve priority deterministic and prevent same-symbol competition."""
    merged: list[EntryCandidate] = []
    seen_symbols: set[str] = set()
    for candidate in (*primary, *secondary):
        if candidate.symbol in seen_symbols:
            continue
        seen_symbols.add(candidate.symbol)
        merged.append(candidate)
    return merged


def _reversal_v2_candidate(
    engine: ReversalAlphaEngine,
    config: Any,
    symbol: str,
    timestamp: Any,
    consumed_events: set[str],
    stats: dict[str, int],
) -> EntryCandidate | None:
    best: EntryCandidate | None = None
    for direction in engine.allowed_directions():
        snapshot = engine.evaluate(symbol, timestamp, direction)
        if snapshot is None:
            _count_stat(stats, "reject_warmup_or_missing_data")
            continue
        _count_stat(stats, "evaluated_count")
        if snapshot.event_id in consumed_events:
            _count_stat(stats, "reject_event_consumed")
            continue
        if not snapshot.eligible:
            reason = snapshot.reject_reasons[0] if snapshot.reject_reasons else "not_eligible"
            _count_stat(stats, f"reject_{reason}")
            continue
        decision = engine.decision_from_snapshot(snapshot, direction)
        if decision is None:
            continue
        _count_stat(stats, "eligible_count")
        signal = decision.signal
        momentum_pct = (
            direction.value
            * (decision.candle.close / max(decision.candle.open, 1e-12) - 1.0)
            * 100.0
        )
        volume_ratio = float(snapshot.features.get("trigger_volume_ratio", 0.0))
        candidate = EntryCandidate(
            symbol,
            signal,
            decision.candle,
            snapshot.quality_score * 100.0,
            momentum_pct,
            volume_ratio,
            "reversal_v2_closed_30m_setup_closed_5m_trigger",
            metadata={
                **snapshot.features,
                "event_id": snapshot.event_id,
                "structural_stop_price": snapshot.structural_stop_price,
                "setup_atr": snapshot.setup_atr,
                "trigger_close": decision.candle.close,
                "take_profit_r": float(getattr(config.reversal_alpha, "take_profit_r", 1.5)),
                "quality_score": snapshot.quality_score,
            },
        )
        if best is None or candidate.rank_score > best.rank_score:
            best = candidate
    return best


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
    mtf_candidate_cache: dict[tuple[Any, ...], EntryCandidate | None],
    positions: dict[str, PortfolioPosition],
) -> EntryCandidate | None:
    if not getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False):
        return None
    if not _mtf_symbol_allowed(config, symbol):
        _count_mtf_reject(mtf_reject_stats, "mtf_symbol_mode_filtered")
        return None
    open_positions = _mtf_open_positions(positions)
    base_schedule_block = ""
    if open_positions >= max(1, int(getattr(config.strategy, "mtf_max_open_positions", 1))):
        base_schedule_block = "mtf_max_open_positions"
    elif mtf_symbol_cooldown_until.get(symbol) and timestamp < mtf_symbol_cooldown_until[symbol]:
        base_schedule_block = "mtf_symbol_cooldown"
    max_daily = max(1, int(getattr(config.strategy, "mtf_max_daily_trades", 2)))
    if not base_schedule_block and mtf_daily_entry_counts.get(timestamp.date(), 0) >= max_daily:
        base_schedule_block = "mtf_daily_trade_limit"
    quality_sleeve_enabled = bool(getattr(config.strategy, "mtf_quality_sleeve_enabled", False))
    if base_schedule_block and not quality_sleeve_enabled:
        _count_mtf_reject(mtf_reject_stats, base_schedule_block)
        return None

    trigger_timeframe = _mtf_trigger_timeframe(config)
    regime_timeframe = _mtf_regime_timeframe(config)
    try:
        trigger_timestamps = mtf_timestamps_by_timeframe[trigger_timeframe][symbol]
    except KeyError:
        _count_mtf_reject(mtf_reject_stats, "mtf_missing_timeframe_data")
        return None
    trigger_end = bisect.bisect_right(
        trigger_timestamps,
        timestamp - timedelta(milliseconds=interval_to_milliseconds(trigger_timeframe)),
    )
    if trigger_end <= 0:
        _count_mtf_reject(mtf_reject_stats, "mtf_warmup")
        return None
    cache_key = (
        _mtf_signal_config_hash(config.strategy),
        symbol,
        trigger_timeframe,
        trigger_timestamps[trigger_end - 1],
    )
    if cache_key in mtf_candidate_cache:
        cached_candidate = mtf_candidate_cache[cache_key]
        if not quality_sleeve_enabled or cached_candidate is None:
            return cached_candidate
        return _mtf_apply_quality_sleeve(
            config,
            cached_candidate,
            base_schedule_block,
            timestamp,
            symbol,
            open_positions,
            mtf_daily_entry_counts,
            mtf_symbol_cooldown_until,
            mtf_reject_stats,
        )

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
        oi_change_at(
            features,
            timestamp,
            int(getattr(config.strategy, "mtf_oi_max_age_minutes", 15)),
        ),
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
                oi_change_at(
                    features,
                    timestamp,
                    int(getattr(config.strategy, "mtf_oi_max_age_minutes", 15)),
                ),
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
    metadata = dict(decision.metadata)
    metadata.update(
        {
            "rank_score": rank_score,
            "directional_momentum_pct": momentum_pct,
            "entry_rank_volume_ratio": volume_ratio,
        }
    )
    raw_candidate = EntryCandidate(
        symbol,
        decision.signal,
        decision.candle,
        rank_score,
        momentum_pct,
        volume_ratio,
        "mtf_4h_rsi_regime",
        metadata=metadata,
    )
    if quality_sleeve_enabled:
        mtf_candidate_cache[cache_key] = raw_candidate
        return _mtf_apply_quality_sleeve(
            config,
            raw_candidate,
            base_schedule_block,
            timestamp,
            symbol,
            open_positions,
            mtf_daily_entry_counts,
            mtf_symbol_cooldown_until,
            mtf_reject_stats,
        )
    if rank_score < mtf_min_rank_score_for_direction(config.strategy, decision.signal.direction):
        _count_mtf_reject(mtf_reject_stats, "mtf_rank_low")
        mtf_candidate_cache[cache_key] = None
        return None
    mtf_candidate_cache[cache_key] = raw_candidate
    return raw_candidate


def _mtf_apply_quality_sleeve(
    config: Any,
    candidate: EntryCandidate,
    base_schedule_block: str,
    timestamp: Any,
    symbol: str,
    open_positions: int,
    daily_entry_counts: dict[Any, int],
    symbol_cooldown_until: dict[str, Any],
    reject_stats: dict[str, int],
) -> EntryCandidate | None:
    rank_score = float(candidate.rank_score)
    base_rank_pass = rank_score >= mtf_min_rank_score_for_direction(config.strategy, candidate.signal.direction)
    base_reject_reason = base_schedule_block or ("mtf_rank_low" if not base_rank_pass else "")
    if not base_reject_reason:
        return candidate
    metadata = dict(candidate.metadata)
    conservative_cost_pct = (
        2.0 * float(getattr(config.risk, "taker_fee_rate", float(getattr(config.risk, "estimated_fee_bps", 5.0)) / 10_000.0))
        + (
            float(getattr(config.risk, "market_slippage_bps", getattr(config.risk, "estimated_slippage_bps", 2.0)))
            + max(
                float(getattr(config.risk, "stop_slippage_bps", 5.0)),
                float(getattr(config.risk, "take_profit_slippage_bps", 2.0)),
            )
        )
        / 10_000.0
    )
    metadata["target_to_cost_ratio"] = candidate.signal.take_profit_pct / max(conservative_cost_pct, 1e-12)
    quality = mtf_candidate_quality(metadata, candidate.signal.direction)
    metadata.update(quality.as_dict())
    trigger_timeframe = str(metadata.get("trigger_timeframe", _mtf_trigger_timeframe(config)))
    signal_available_time = candidate.candle.timestamp + timedelta(
        milliseconds=interval_to_milliseconds(trigger_timeframe)
    )
    signal_age_minutes = max(0.0, (timestamp - signal_available_time).total_seconds() / 60.0)
    sleeve_allowed, sleeve_reject = _mtf_quality_sleeve_allows(
        config,
        candidate.signal.direction,
        rank_score,
        quality.score,
        timestamp,
        symbol,
        open_positions,
        daily_entry_counts,
        symbol_cooldown_until,
        signal_age_minutes,
    )
    if not sleeve_allowed:
        _count_mtf_reject(reject_stats, base_reject_reason)
        if sleeve_reject:
            _count_mtf_reject(reject_stats, sleeve_reject)
        return None
    mode = str(getattr(config.strategy, "mtf_quality_sleeve_ranking_mode", "rank")).lower()
    metadata.update(
        {
            "mtf_quality_sleeve": True,
            "mtf_quality_sleeve_base_reject_reason": base_reject_reason,
            "mtf_quality_sleeve_selected_score": rank_score if mode == "rank" else quality.score,
        }
    )
    signal = replace(
        candidate.signal,
        reason=(
            f"{candidate.signal.reason} quality_sleeve=1 "
            f"quality_score={quality.score:.3f} base_reject={base_reject_reason}"
        ),
        risk_multiplier=(
            candidate.signal.risk_multiplier
            * max(0.0, float(getattr(config.strategy, "mtf_quality_sleeve_risk_multiplier", 1.0)))
        ),
    )
    _count_mtf_reject(reject_stats, "mtf_quality_sleeve_candidate")
    return replace(candidate, signal=signal, metadata=metadata)


def _mtf_quality_sleeve_allows(
    config: Any,
    direction: Direction,
    rank_score: float,
    quality_score: float,
    timestamp: Any,
    symbol: str,
    open_positions: int,
    daily_entry_counts: dict[Any, int],
    symbol_cooldown_until: dict[str, Any],
    signal_age_minutes: float = 0.0,
) -> tuple[bool, str]:
    strategy = config.strategy
    if not bool(getattr(strategy, "mtf_quality_sleeve_enabled", False)):
        return False, ""
    if direction is Direction.LONG and not bool(getattr(strategy, "mtf_quality_sleeve_allow_long", False)):
        return False, "mtf_quality_sleeve_side_filtered"
    if direction is Direction.SHORT and not bool(getattr(strategy, "mtf_quality_sleeve_allow_short", True)):
        return False, "mtf_quality_sleeve_side_filtered"
    if open_positions >= max(1, int(getattr(strategy, "mtf_quality_sleeve_max_open_positions", 1))):
        return False, "mtf_quality_sleeve_max_open_positions"
    sleeve_daily = max(1, int(getattr(strategy, "mtf_quality_sleeve_max_daily_trades", 4)))
    if daily_entry_counts.get(timestamp.date(), 0) >= sleeve_daily:
        return False, "mtf_quality_sleeve_daily_trade_limit"
    max_signal_age = max(0, int(getattr(strategy, "mtf_quality_sleeve_max_signal_age_minutes", 0)))
    if signal_age_minutes > max_signal_age + 1e-9:
        return False, "mtf_quality_sleeve_stale_signal"
    base_cooldown_hours = max(0, int(getattr(strategy, "mtf_symbol_cooldown_hours", 12)))
    sleeve_cooldown_hours = max(0, int(getattr(strategy, "mtf_quality_sleeve_symbol_cooldown_hours", 0)))
    base_cooldown_until = symbol_cooldown_until.get(symbol)
    if base_cooldown_until is not None and base_cooldown_hours > 0 and sleeve_cooldown_hours > 0:
        last_entry_time = base_cooldown_until - timedelta(hours=base_cooldown_hours)
        if timestamp < last_entry_time + timedelta(hours=sleeve_cooldown_hours):
            return False, "mtf_quality_sleeve_symbol_cooldown"
    mode = str(getattr(strategy, "mtf_quality_sleeve_ranking_mode", "rank")).strip().lower()
    selected_score = rank_score if mode == "rank" else quality_score
    threshold_field = (
        "mtf_quality_sleeve_long_min_score"
        if direction is Direction.LONG
        else "mtf_quality_sleeve_short_min_score"
    )
    if selected_score < float(getattr(strategy, threshold_field, 999.0)):
        return False, "mtf_quality_sleeve_score_low"
    return True, ""


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


@lru_cache(maxsize=128)
def _mtf_signal_config_hash(strategy: Any) -> int:
    normalized = replace(
        strategy,
        mtf_profit_protection_enabled=False,
        mtf_move_stop_to_breakeven_r=0.0,
        mtf_breakeven_extra_bps=0.0,
        mtf_trailing_start_r=0.0,
        mtf_trailing_mode="none",
        mtf_profit_giveback_r=0.0,
        mtf_trailing_atr15m_mult=0.0,
        mtf_exit_mode="fixed_tp",
        mtf_partial_take_profit_r=0.0,
        mtf_partial_take_profit_fraction=0.0,
        mtf_fail_fast_minutes=0,
        mtf_fail_fast_min_r=0.0,
        mtf_exit_on_30m_confirm_lost=False,
        mtf_30m_exit_confirm_bars=0,
        mtf_30m_exit_require_macd_adverse=False,
        mtf_exit_on_1h_confirm_lost=False,
        mtf_extra_execution_delay_minutes=0,
        mtf_max_open_positions=1,
        mtf_max_daily_trades=1,
        mtf_symbol_cooldown_hours=0,
    )
    return hash(normalized)


def _count_mtf_reject(stats: dict[str, int], reason: str) -> None:
    if not reason:
        return
    normalized = str(reason).split()[0]
    stats[normalized] = stats.get(normalized, 0) + 1


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
    mtper_engine: MtperEngine | None = None,
    mtper_stats: dict[str, int] | None = None,
    mtpc_engine: MtpcEngine | None = None,
    mtpc_stats: dict[str, int] | None = None,
    reversal_v2_stats: dict[str, int] | None = None,
) -> float:
    for symbol in list(positions):
        position = positions[symbol]
        candle = execution_candles_by_symbol[symbol][execution_index]
        position.bars_held = max(0, signal_index - position.entry_index)
        previous_mfe = position.mfe
        _update_position_excursion(position, candle)
        if MTF_REASON_TOKEN in str(position.entry_reason):
            _update_mtf_profit_protection(
                config,
                position,
                previous_mfe,
                execution_config,
                candle,
                mtf_candles_by_timeframe or {},
                mtf_timestamps_by_timeframe or {},
            )
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
        if _is_mtper_position(position):
            cash, closed = _manage_mtper_position_1m(
                trader,
                config,
                cash,
                positions,
                trades,
                position,
                candle,
                signal_index,
                execution_config,
                execution_stats,
                mtf_candles_by_timeframe or {},
                mtf_timestamps_by_timeframe or {},
                mtper_stats if mtper_stats is not None else {},
            )
            if closed:
                _mark_reentry_cooldown_time(config, reentry_block_until, symbol, candle.timestamp)
                if mtper_engine is not None:
                    mtper_engine.mark_closed(symbol, candle.timestamp, str(trades[-1].get("exit_reason", "closed")))
            continue
        if _is_mtpc_position(position):
            cash, closed = _manage_mtpc_position_1m(
                trader,
                config,
                cash,
                positions,
                trades,
                position,
                candle,
                signal_index,
                execution_config,
                execution_stats,
                mtf_candles_by_timeframe or {},
                mtf_timestamps_by_timeframe or {},
                mtpc_stats if mtpc_stats is not None else {},
            )
            if closed:
                _mark_reentry_cooldown_time(config, reentry_block_until, symbol, candle.timestamp)
                if mtpc_engine is not None:
                    mtpc_engine.mark_closed(symbol, candle.timestamp, str(trades[-1].get("exit_reason", "closed")))
            continue
        if _is_reversal_v2_position(position):
            cash, closed = _manage_reversal_v2_position_1m(
                config,
                cash,
                positions,
                trades,
                position,
                candle,
                signal_index,
                execution_config,
                execution_stats,
                trader.client.symbol_rules(symbol),
                reversal_v2_stats if reversal_v2_stats is not None else {},
            )
            if closed:
                _mark_reentry_cooldown_time(config, reentry_block_until, symbol, candle.timestamp)
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
        mtf_take_profit_action = _mtf_take_profit_action(config, position)
        take_profit_enabled = (
            not vbp_runner_after_tp1
            and mtf_take_profit_action != "disabled"
        )
        if position.direction == Direction.LONG:
            if _is_vbp_position(position):
                position.best_price = max(position.best_price, candle.close)
            else:
                position.best_price = max(position.best_price, candle.high)
            stop_hit = candle.low <= position.stop_price
            take_profit_hit = take_profit_enabled and candle.high >= position.take_profit_price
            if stop_hit and take_profit_hit:
                execution_stats.same_bar_tp_sl_conflict_count += 1
            if stop_hit and (not take_profit_hit or execution_config.mode != "optimistic"):
                exit_price = position.stop_price
                reason = _mtf_stop_exit_reason(position) or "stop_loss_1m"
            elif take_profit_hit:
                if mtf_take_profit_action == "partial":
                    cash = _close_mtf_partial_take_profit(
                        config,
                        cash,
                        position,
                        trades,
                        position.take_profit_price,
                        signal_index,
                        candle.timestamp,
                        execution_config,
                        trader.client.symbol_rules(symbol),
                    )
                else:
                    exit_price = position.take_profit_price
                    reason = "take_profit_1m"
        else:
            position.best_price = min(position.best_price, candle.low)
            stop_hit = candle.high >= position.stop_price
            take_profit_hit = take_profit_enabled and candle.low <= position.take_profit_price
            if stop_hit and take_profit_hit:
                execution_stats.same_bar_tp_sl_conflict_count += 1
            if stop_hit and (not take_profit_hit or execution_config.mode != "optimistic"):
                exit_price = position.stop_price
                reason = _mtf_stop_exit_reason(position) or "stop_loss_1m"
            elif take_profit_hit:
                if mtf_take_profit_action == "partial":
                    cash = _close_mtf_partial_take_profit(
                        config,
                        cash,
                        position,
                        trades,
                        position.take_profit_price,
                        signal_index,
                        candle.timestamp,
                        execution_config,
                        trader.client.symbol_rules(symbol),
                    )
                else:
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


def _is_mtper_position(position: PortfolioPosition) -> bool:
    return MTPER_REASON_TOKEN in str(position.entry_reason)


def _is_mtpc_position(position: PortfolioPosition) -> bool:
    return MTPC_REASON_TOKEN in str(position.entry_reason)


def _mtf_exit_mode(config: Any) -> str:
    mode = str(getattr(config.strategy, "mtf_exit_mode", "fixed_tp")).strip().lower()
    aliases = {
        "fixed": "fixed_tp",
        "partial": "partial_runner",
        "runner_only": "runner",
        "pure_runner": "runner",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in {"fixed_tp", "partial_runner", "runner"} else "fixed_tp"


def _mtf_take_profit_action(config: Any, position: PortfolioPosition) -> str:
    if MTF_REASON_TOKEN not in str(position.entry_reason):
        return "fixed"
    mode = _mtf_exit_mode(config)
    if mode == "runner":
        return "disabled"
    if mode == "partial_runner":
        if bool((position.strategy_metadata or {}).get("mtf_partial_take_profit_done", False)):
            return "disabled"
        return "partial"
    return "fixed"


def _close_mtf_partial_take_profit(
    config: Any,
    cash: float,
    position: PortfolioPosition,
    trades: list[dict[str, Any]],
    exit_price: float,
    index: int,
    exit_time: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> float:
    metadata = dict(position.strategy_metadata or {})
    initial_quantity = max(float(metadata.get("mtf_initial_quantity", position.quantity)), position.quantity)
    fraction = min(
        0.95,
        max(0.05, float(getattr(config.strategy, "mtf_partial_take_profit_fraction", 0.25))),
    )
    close_quantity = min(position.quantity * 0.95, initial_quantity * fraction)
    if close_quantity <= 0.0 or close_quantity >= position.quantity:
        return cash

    original_quantity = position.quantity
    close_ratio = close_quantity / original_quantity
    fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        close_quantity,
        exit_price,
        "take_profit_market",
        position.liquidity_reference_quote_volume or None,
    )
    raw_entry = position.raw_entry_price or position.entry_price
    raw_gross = position.direction.value * close_quantity * (exit_price - raw_entry)
    execution_gross = position.direction.value * close_quantity * (fill.price - position.entry_price)
    allocated_entry_fee = position.entry_fee * close_ratio
    allocated_entry_slippage = position.entry_slippage_cost * close_ratio
    funding = _funding_for_position(execution_config, position, exit_time) * close_ratio
    fee = allocated_entry_fee + fill.fee
    slippage = allocated_entry_slippage + fill.slippage_cost
    net_pnl = raw_gross - fee - slippage + funding
    cash += execution_gross - fill.fee + funding

    partial_mfe = position.mfe * close_ratio
    partial_mae = position.mae * close_ratio
    remaining_ratio = 1.0 - close_ratio
    position.quantity -= close_quantity
    position.entry_fee -= allocated_entry_fee
    position.entry_slippage_cost -= allocated_entry_slippage
    position.mfe *= remaining_ratio
    position.mae *= remaining_ratio
    metadata.update(
        {
            "mtf_partial_take_profit_done": True,
            "mtf_partial_take_profit_time": (
                exit_time.isoformat() if hasattr(exit_time, "isoformat") else exit_time
            ),
            "mtf_partial_take_profit_quantity": close_quantity,
            "mtf_partial_take_profit_net_pnl": net_pnl,
        }
    )
    position.strategy_metadata = metadata

    notional = abs(close_quantity * position.entry_price)
    strategy_bucket = _strategy_bucket(position.entry_reason)
    trades.append(
        {
            "symbol": position.symbol,
            "strategy": strategy_bucket,
            "strategy_bucket": strategy_bucket,
            "side": position.direction.name,
            "direction": position.direction.name,
            "entry_time": position.entry_time.isoformat() if hasattr(position.entry_time, "isoformat") else position.entry_time,
            "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else exit_time,
            "entry_price": position.entry_price,
            "exit_price": fill.price,
            "raw_entry_price": raw_entry,
            "raw_exit_price": exit_price,
            "qty": close_quantity,
            "quantity": close_quantity,
            "notional": notional,
            "entry_fee": allocated_entry_fee,
            "exit_fee": fill.fee,
            "fee": fee,
            "fees": fee,
            "gross_pnl": raw_gross,
            "execution_gross_pnl": execution_gross,
            "slippage_cost": slippage,
            "funding": funding,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / max(notional, 1e-12),
            "net_bps": net_pnl / max(notional, 1e-12) * 10_000.0,
            "entry_reason": position.entry_reason,
            "reason": "mtf_partial_take_profit",
            "exit_reason": "mtf_partial_take_profit",
            "bars": index - position.entry_index,
            "scale_ins": position.scale_ins,
            "mfe": partial_mfe,
            "mae": partial_mae,
            "hold_minutes": _hold_minutes(position.entry_time, exit_time),
            "avg_hold_minutes": _hold_minutes(position.entry_time, exit_time),
            "entry_order_type": position.entry_order_type,
            "exit_order_type": fill.order_type,
            "entry_liquidity": position.entry_liquidity,
            "exit_liquidity": fill.liquidity,
            "signal_time": position.signal_time.isoformat() if hasattr(position.signal_time, "isoformat") else position.signal_time,
            "signal_available_time": position.signal_available_time.isoformat() if hasattr(position.signal_available_time, "isoformat") else position.signal_available_time,
            "entry_participation_rate": position.entry_participation_rate,
            "exit_participation_rate": fill.participation_rate,
            "capacity_fill_ratio": position.capacity_fill_ratio,
            "risk_budget_usdt": position.risk_budget_usdt,
            "initial_stop_price": position.initial_stop_price,
            "strategy_metadata": dict(metadata),
            "skip_reason": "",
        }
    )
    return cash


def _manage_mtpc_position_1m(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    trades: list[dict[str, Any]],
    position: PortfolioPosition,
    candle: Candle,
    signal_index: int,
    execution_config: BacktestExecutionConfig,
    execution_stats: BacktestExecutionStats,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
    stats: dict[str, int],
) -> tuple[float, bool]:
    symbol = position.symbol
    rules = trader.client.symbol_rules(symbol)
    metadata = dict(position.strategy_metadata or {})
    target_1 = float(metadata.get("target_1_price", position.take_profit_price))
    target_2 = float(metadata.get("target_2_price", position.take_profit_price))
    is_long = position.direction == Direction.LONG
    stop_hit = candle.low <= position.stop_price if is_long else candle.high >= position.stop_price
    first_target_hit = (
        candle.high >= target_1 if is_long else candle.low <= target_1
    ) and not bool(metadata.get("target_1_done", False))
    if stop_hit and first_target_hit:
        execution_stats.same_bar_tp_sl_conflict_count += 1
    # The protective stop that existed before this bar has priority.
    if stop_hit:
        cash = _close_position(
            config,
            cash,
            positions,
            trades,
            symbol,
            position.stop_price,
            "mtpc_stop_loss_1m",
            signal_index,
            candle.timestamp,
            execution_config=execution_config,
            rules=rules,
        )
        _count_stat(stats, "stop_loss_count")
        return cash, True

    favorable_price = candle.high if is_long else candle.low
    favorable_net = executable_net_pnl(position, favorable_price, execution_config, rules)
    current_net = executable_net_pnl(position, candle.close, execution_config, rules)
    favorable_campaign_r = net_pnl_r(position, favorable_net, CAMPAIGN_R_BASIS)
    current_campaign_r = net_pnl_r(position, current_net, CAMPAIGN_R_BASIS)
    favorable_initial_leg_r = net_pnl_r(position, favorable_net, INITIAL_LEG_R_BASIS)
    current_initial_leg_r = net_pnl_r(position, current_net, INITIAL_LEG_R_BASIS)
    position.mtpc_max_campaign_executable_r = max(
        position.mtpc_max_campaign_executable_r, favorable_campaign_r
    )
    position.mtpc_min_campaign_executable_r = min(
        position.mtpc_min_campaign_executable_r, current_campaign_r
    )
    position.mtpc_max_initial_leg_executable_r = max(
        position.mtpc_max_initial_leg_executable_r, favorable_initial_leg_r
    )
    position.mtpc_min_initial_leg_executable_r = min(
        position.mtpc_min_initial_leg_executable_r, current_initial_leg_r
    )
    if _mtpc_exit_uses_initial_leg_r(config):
        favorable_r = favorable_initial_leg_r
        current_r = current_initial_leg_r
    else:
        favorable_r = favorable_campaign_r
        current_r = current_campaign_r
    position.mtpc_max_executable_r = max(position.mtpc_max_executable_r, favorable_r)
    position.mtpc_min_executable_r = min(position.mtpc_min_executable_r, current_r)
    position.mtpc_executable_mfe_usdt = max(position.mtpc_executable_mfe_usdt, favorable_net)
    position.mtpc_executable_mae_usdt = min(position.mtpc_executable_mae_usdt, current_net)
    exit_cfg = config.mtpc.exit

    if first_target_hit:
        fraction = max(0.0, min(1.0, float(exit_cfg.take_profit_1_fraction)))
        initial_quantity = max(float(metadata.get("initial_quantity", position.quantity)), position.quantity)
        close_quantity = min(position.quantity, initial_quantity * fraction)
        if close_quantity >= position.quantity - 1e-12:
            cash = _close_position(
                config,
                cash,
                positions,
                trades,
                symbol,
                target_1,
                "mtpc_take_profit_1_full",
                signal_index,
                candle.timestamp,
                execution_config=execution_config,
                rules=rules,
            )
            _count_stat(stats, "take_profit_1_full_count")
            return cash, True
        if close_quantity > 0.0:
            cash = _close_mtpc_partial_position(
                cash,
                position,
                trades,
                target_1,
                close_quantity,
                "mtpc_take_profit_1_partial",
                signal_index,
                candle.timestamp,
                execution_config,
                rules,
            )
            metadata["target_1_done"] = True
            metadata["target_1_time"] = candle.timestamp.isoformat()
            _count_stat(stats, "take_profit_1_partial_count")

    reason = ""
    if bool(metadata.get("target_1_done", False)):
        if bool(exit_cfg.runner_enabled):
            runner = _mtf_closed(
                mtf_candles_by_timeframe,
                mtf_timestamps_by_timeframe,
                str(exit_cfg.runner_timeframe),
                symbol,
                candle.timestamp,
                max(40, int(exit_cfg.runner_ema_period) + 10),
            )
            if len(runner) >= int(exit_cfg.runner_ema_period) + 3:
                trailing = ema([item.close for item in runner], int(exit_cfg.runner_ema_period))[-1]
                if (is_long and runner[-1].close < trailing) or (
                    not is_long and runner[-1].close > trailing
                ):
                    reason = "mtpc_runner_ema_exit"
        elif (is_long and candle.high >= target_2) or (not is_long and candle.low <= target_2):
            cash = _close_position(
                config,
                cash,
                positions,
                trades,
                symbol,
                target_2,
                "mtpc_take_profit_2",
                signal_index,
                candle.timestamp,
                execution_config=execution_config,
                rules=rules,
            )
            _count_stat(stats, "take_profit_2_count")
            return cash, True

    if not reason and bool(exit_cfg.structural_fail_fast_enabled):
        reason = _mtpc_structural_exit_reason(
            position,
            candle.timestamp,
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
        )
    hold_minutes = _hold_minutes(position.entry_time, candle.timestamp)
    if not reason and hold_minutes >= max(1, int(exit_cfg.time_nonresponse_minutes)):
        failed_checks = _mtpc_nonresponse_failed_checks(
            position,
            candle,
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
        )
        if (
            position.mtpc_max_executable_r < float(exit_cfg.time_nonresponse_min_mfe_r)
            and failed_checks >= max(1, int(exit_cfg.time_nonresponse_min_failed_checks))
        ):
            reason = "mtpc_time_nonresponse"
            _count_stat(stats, f"nonresponse_failed_checks_{failed_checks}")
    if not reason and hold_minutes >= max(1, int(exit_cfg.max_holding_minutes)):
        reason = "mtpc_max_holding_time"
    if (
        not reason
        and bool(exit_cfg.giveback_enabled)
        and position.mtpc_max_executable_r >= float(exit_cfg.giveback_activation_r)
        and current_r < position.mtpc_max_executable_r - float(exit_cfg.allowed_giveback_r)
    ):
        reason = "mtpc_profit_giveback"
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
            rules=rules,
        )
        _count_stat(stats, reason)
        return cash, True

    if bool(exit_cfg.breakeven_enabled) and position.mtpc_max_executable_r >= float(exit_cfg.breakeven_trigger_r):
        cost_buffer = max(0.0, float(exit_cfg.breakeven_cost_buffer_pct))
        if is_long:
            proposed = max(position.stop_price, position.entry_price * (1.0 + cost_buffer))
            crossed_proposed = proposed >= candle.close
            improves_stop = proposed > position.stop_price
        else:
            proposed = min(position.stop_price, position.entry_price * (1.0 - cost_buffer))
            crossed_proposed = proposed <= candle.close
            improves_stop = proposed < position.stop_price
        if crossed_proposed:
            cash = _close_position(
                config,
                cash,
                positions,
                trades,
                symbol,
                candle.close,
                "mtpc_breakeven_reversal",
                signal_index,
                candle.timestamp,
                execution_config=execution_config,
                rules=rules,
            )
            _count_stat(stats, "breakeven_reversal_count")
            return cash, True
        if improves_stop:
            position.stop_price = proposed
            metadata["breakeven_armed"] = True
            _count_stat(stats, "breakeven_armed_count")
    position.strategy_metadata = metadata
    return cash, False


def _mtpc_structural_exit_reason(
    position: PortfolioPosition,
    timestamp: Any,
    candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
) -> str:
    metadata = position.strategy_metadata or {}
    is_long = position.direction == Direction.LONG
    pullback_extreme = float(
        metadata.get("pullback_low" if is_long else "pullback_high")
        or metadata.get("pullback_extreme")
        or position.initial_stop_price
    )
    breakout_level = float(metadata.get("breakout_level") or position.entry_price)
    c5 = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "5m", position.symbol, timestamp, 40)
    c15 = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "15m", position.symbol, timestamp, 50)
    if len(c5) >= 3:
        if is_long and c5[-1].close < pullback_extreme:
            return "mtpc_structural_fail_pullback_low"
        if not is_long and c5[-1].close > pullback_extreme:
            return "mtpc_structural_fail_pullback_high"
    if len(c15) >= 25:
        ema21 = ema([item.close for item in c15], 21)[-1]
        if is_long and c15[-1].close < breakout_level and c15[-1].close < ema21:
            return "mtpc_structural_fail_breakout_and_ema21"
        if not is_long and c15[-1].close > breakout_level and c15[-1].close > ema21:
            return "mtpc_structural_fail_breakout_and_ema21"
    return ""


def _mtpc_nonresponse_failed_checks(
    position: PortfolioPosition,
    candle: Candle,
    candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
) -> int:
    is_long = position.direction == Direction.LONG
    checks = int(
        candle.close <= position.entry_price if is_long else candle.close >= position.entry_price
    )
    c5 = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "5m", position.symbol, candle.timestamp, 40)
    c15 = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "15m", position.symbol, candle.timestamp, 50)
    c1 = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "1h", position.symbol, candle.timestamp, 60)
    if len(c5) >= 25:
        e9 = ema([item.close for item in c5], 9)[-1]
        _, _, hist5 = macd([item.close for item in c5])
        checks += int(c5[-1].close <= e9 if is_long else c5[-1].close >= e9)
        checks += int(hist5[-1] <= hist5[-2] if is_long else hist5[-1] >= hist5[-2])
    if len(c15) >= 25:
        e21 = ema([item.close for item in c15], 21)[-1]
        checks += int(c15[-1].close <= e21 if is_long else c15[-1].close >= e21)
        checks += int(
            c15[-1].close <= c15[-2].close if is_long else c15[-1].close >= c15[-2].close
        )
    if len(c1) >= 30:
        _, _, hist1 = macd([item.close for item in c1])
        checks += int(hist1[-1] <= hist1[-2] if is_long else hist1[-1] >= hist1[-2])
    return checks


def _manage_mtper_position_1m(
    trader: BinanceAutoTrader,
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    trades: list[dict[str, Any]],
    position: PortfolioPosition,
    candle: Candle,
    signal_index: int,
    execution_config: BacktestExecutionConfig,
    execution_stats: BacktestExecutionStats,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
    stats: dict[str, int],
) -> tuple[float, bool]:
    symbol = position.symbol
    rules = trader.client.symbol_rules(symbol)
    stop_hit = candle.low <= position.stop_price if position.direction == Direction.LONG else candle.high >= position.stop_price
    metadata = position.strategy_metadata if position.strategy_metadata is not None else {}
    target_1 = float(metadata.get("target_1_price", position.take_profit_price))
    target_2 = float(metadata.get("target_2_price", position.take_profit_price))
    target_3 = float(metadata.get("target_3_price", position.take_profit_price))
    target_hit = (
        candle.high >= target_1 if position.direction == Direction.LONG else candle.low <= target_1
    ) and not bool(metadata.get("target_1_done", False))
    if stop_hit and target_hit:
        execution_stats.same_bar_tp_sl_conflict_count += 1
    # The old protective stop is always evaluated before same-bar targets.
    if stop_hit:
        cash = _close_position(
            config,
            cash,
            positions,
            trades,
            symbol,
            position.stop_price,
            "mtper_hard_stop_1m",
            signal_index,
            candle.timestamp,
            execution_config=execution_config,
            rules=rules,
        )
        _count_stat(stats, "hard_stop_count")
        return cash, True

    executable_net = executable_net_pnl(position, candle.close, execution_config, rules)
    campaign_r = net_pnl_r(position, executable_net, CAMPAIGN_R_BASIS)
    initial_leg_r = net_pnl_r(position, executable_net, INITIAL_LEG_R_BASIS)
    position.mtper_max_campaign_executable_r = max(position.mtper_max_campaign_executable_r, campaign_r)
    position.mtper_min_campaign_executable_r = min(position.mtper_min_campaign_executable_r, campaign_r)
    position.mtper_max_initial_leg_executable_r = max(position.mtper_max_initial_leg_executable_r, initial_leg_r)
    position.mtper_min_initial_leg_executable_r = min(position.mtper_min_initial_leg_executable_r, initial_leg_r)
    position.mtper_executable_mfe_usdt = max(position.mtper_executable_mfe_usdt, executable_net)
    position.mtper_executable_mae_usdt = min(position.mtper_executable_mae_usdt, executable_net)

    exit_cfg = config.mtper.exit
    initial_quantity = max(float(metadata.get("initial_quantity", position.quantity)), position.quantity)
    if not bool(metadata.get("target_1_done", False)):
        hit = candle.high >= target_1 if position.direction == Direction.LONG else candle.low <= target_1
        if hit:
            close_quantity = min(position.quantity, initial_quantity * max(0.0, float(exit_cfg.target_1_fraction)))
            if 0.0 < close_quantity < position.quantity:
                cash = _close_mtper_partial_position(
                    config,
                    cash,
                    position,
                    trades,
                    target_1,
                    close_quantity,
                    "mtper_mean_target_1",
                    signal_index,
                    candle.timestamp,
                    execution_config,
                    rules,
                )
                metadata["target_1_done"] = True
                metadata["target_1_time"] = candle.timestamp.isoformat()
                _count_stat(stats, "target_1_count")
            elif close_quantity >= position.quantity:
                cash = _close_position(
                    config,
                    cash,
                    positions,
                    trades,
                    symbol,
                    target_1,
                    "mtper_mean_target_1_full",
                    signal_index,
                    candle.timestamp,
                    execution_config=execution_config,
                    rules=rules,
                )
                _count_stat(stats, "target_1_full_count")
                return cash, True

    if bool(metadata.get("target_1_done", False)) and not bool(metadata.get("target_2_done", False)):
        hit = candle.high >= target_2 if position.direction == Direction.LONG else candle.low <= target_2
        if hit:
            close_quantity = min(position.quantity, initial_quantity * max(0.0, float(exit_cfg.target_2_fraction)))
            if 0.0 < close_quantity < position.quantity:
                cash = _close_mtper_partial_position(
                    config,
                    cash,
                    position,
                    trades,
                    target_2,
                    close_quantity,
                    "mtper_mean_target_2",
                    signal_index,
                    candle.timestamp,
                    execution_config,
                    rules,
                )
                metadata["target_2_done"] = True
                metadata["target_2_time"] = candle.timestamp.isoformat()
                _count_stat(stats, "target_2_count")
            elif close_quantity >= position.quantity:
                cash = _close_position(
                    config,
                    cash,
                    positions,
                    trades,
                    symbol,
                    target_2,
                    "mtper_mean_target_2_full",
                    signal_index,
                    candle.timestamp,
                    execution_config=execution_config,
                    rules=rules,
                )
                _count_stat(stats, "target_2_full_count")
                return cash, True

    if bool(exit_cfg.trend_conversion_enabled) and not bool(metadata.get("trend_converted", False)):
        if _mtper_trend_conversion_confirmed(
            config,
            position,
            candle.timestamp,
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
        ):
            metadata["trend_converted"] = True
            metadata["trend_conversion_time"] = candle.timestamp.isoformat()
            _count_stat(stats, "trend_conversion_count")

    reason = _mtper_structural_exit_reason(
        config,
        position,
        candle.timestamp,
        mtf_candles_by_timeframe,
        mtf_timestamps_by_timeframe,
    )
    hold_hours = _hold_minutes(position.entry_time, candle.timestamp) / 60.0
    if not reason and hold_hours >= max(1, int(exit_cfg.time_nonresponse_hours)):
        failed_checks = _mtper_nonresponse_failed_checks(
            position,
            candle,
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
        )
        if (
            position.mtper_max_initial_leg_executable_r < float(exit_cfg.time_nonresponse_min_mfe_r)
            and failed_checks >= max(1, int(exit_cfg.time_nonresponse_min_failed_checks))
        ):
            reason = "mtper_time_nonresponse"
            _count_stat(stats, f"nonresponse_failed_checks_{failed_checks}")
    if not reason and hold_hours >= max(1, int(exit_cfg.max_holding_hours)):
        reason = "mtper_max_holding_time"

    if not reason and bool(metadata.get("trend_converted", False)):
        reason = _mtper_runner_exit_reason(
            config,
            position,
            candle.timestamp,
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
        )
    elif not reason:
        hit_target_3 = candle.high >= target_3 if position.direction == Direction.LONG else candle.low <= target_3
        if hit_target_3:
            reason = "mtper_mean_target_3"

    if not reason and position.mtper_max_campaign_executable_r >= float(exit_cfg.giveback_trigger_r):
        giveback = _mtper_allowed_giveback(exit_cfg, position.mtper_max_campaign_executable_r)
        if campaign_r < position.mtper_max_campaign_executable_r - giveback:
            reason = "mtper_max_profit_giveback"

    if reason:
        exit_price = target_3 if reason == "mtper_mean_target_3" else candle.close
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
            rules=rules,
        )
        _count_stat(stats, reason)
        return cash, True

    # Profit protection becomes active on the next 1m bar because the old stop
    # was checked before this update.
    if position.mtper_max_initial_leg_executable_r >= float(exit_cfg.breakeven_trigger_r):
        buffer = max(0.0, float(exit_cfg.breakeven_cost_buffer_pct))
        if position.direction == Direction.LONG:
            position.stop_price = max(position.stop_price, position.entry_price * (1.0 + buffer))
        else:
            position.stop_price = min(position.stop_price, position.entry_price * (1.0 - buffer))
        metadata["breakeven_armed"] = True
    position.strategy_metadata = metadata
    return cash, False


def _close_mtpc_partial_position(
    cash: float,
    position: PortfolioPosition,
    trades: list[dict[str, Any]],
    exit_price: float,
    close_quantity: float,
    reason: str,
    index: int,
    exit_time: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> float:
    if close_quantity <= 0.0 or close_quantity >= position.quantity:
        return cash
    original_quantity = position.quantity
    ratio = close_quantity / original_quantity
    fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        close_quantity,
        exit_price,
        "take_profit_market",
        position.liquidity_reference_quote_volume or None,
    )
    raw_entry = position.raw_entry_price or position.entry_price
    raw_gross = position.direction.value * close_quantity * (exit_price - raw_entry)
    execution_gross = position.direction.value * close_quantity * (fill.price - position.entry_price)
    entry_fee = position.entry_fee * ratio
    entry_slippage = position.entry_slippage_cost * ratio
    funding = _funding_for_position(execution_config, position, exit_time) * ratio
    fee = entry_fee + fill.fee
    slippage = entry_slippage + fill.slippage_cost
    net_pnl = raw_gross - fee - slippage + funding
    cash += execution_gross - fill.fee + funding
    position.quantity -= close_quantity
    position.entry_fee -= entry_fee
    position.entry_slippage_cost -= entry_slippage
    notional = abs(close_quantity * position.entry_price)
    trades.append(
        {
            "symbol": position.symbol,
            "strategy": MTPC_REASON_TOKEN,
            "strategy_bucket": MTPC_REASON_TOKEN,
            "side": position.direction.name,
            "direction": position.direction.name,
            "entry_time": position.entry_time.isoformat() if hasattr(position.entry_time, "isoformat") else position.entry_time,
            "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else exit_time,
            "entry_price": position.entry_price,
            "exit_price": fill.price,
            "raw_entry_price": raw_entry,
            "raw_exit_price": exit_price,
            "qty": close_quantity,
            "quantity": close_quantity,
            "notional": notional,
            "entry_fee": entry_fee,
            "exit_fee": fill.fee,
            "fee": fee,
            "fees": fee,
            "gross_pnl": raw_gross,
            "execution_gross_pnl": execution_gross,
            "slippage_cost": slippage,
            "funding": funding,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / max(notional, 1e-12),
            "net_bps": net_pnl / max(notional, 1e-12) * 10_000.0,
            "entry_reason": position.entry_reason,
            "reason": reason,
            "exit_reason": reason,
            "bars": index - position.entry_index,
            "scale_ins": position.scale_ins,
            "mfe": position.mfe * ratio,
            "mae": position.mae * ratio,
            "hold_minutes": _hold_minutes(position.entry_time, exit_time),
            "avg_hold_minutes": _hold_minutes(position.entry_time, exit_time),
            "entry_order_type": position.entry_order_type,
            "exit_order_type": fill.order_type,
            "entry_liquidity": position.entry_liquidity,
            "exit_liquidity": fill.liquidity,
            "signal_time": position.signal_time.isoformat() if hasattr(position.signal_time, "isoformat") else position.signal_time,
            "signal_available_time": position.signal_available_time.isoformat() if hasattr(position.signal_available_time, "isoformat") else position.signal_available_time,
            "campaign_risk_budget_usdt": position.campaign_risk_budget_usdt,
            "initial_leg_full_cost_risk_usdt": position.initial_leg_full_cost_risk_usdt,
            "net_campaign_r": net_pnl / max(position.campaign_risk_budget_usdt, 1e-12),
            "net_initial_leg_r": net_pnl / max(position.initial_leg_full_cost_risk_usdt, 1e-12),
            "mtpc_max_executable_r": position.mtpc_max_executable_r,
            "mtpc_min_executable_r": position.mtpc_min_executable_r,
            "mtpc_max_executable_campaign_r": position.mtpc_max_campaign_executable_r,
            "mtpc_min_executable_campaign_r": position.mtpc_min_campaign_executable_r,
            "mtpc_max_executable_initial_leg_r": position.mtpc_max_initial_leg_executable_r,
            "mtpc_min_executable_initial_leg_r": position.mtpc_min_initial_leg_executable_r,
            "strategy_metadata": dict(position.strategy_metadata or {}),
            "skip_reason": "",
        }
    )
    return cash


def _close_mtper_partial_position(
    config: Any,
    cash: float,
    position: PortfolioPosition,
    trades: list[dict[str, Any]],
    exit_price: float,
    close_quantity: float,
    reason: str,
    index: int,
    exit_time: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> float:
    if close_quantity <= 0.0 or close_quantity >= position.quantity:
        return cash
    original_quantity = position.quantity
    ratio = close_quantity / original_quantity
    fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        close_quantity,
        exit_price,
        "take_profit_market",
        position.liquidity_reference_quote_volume or None,
    )
    raw_entry = position.raw_entry_price or position.entry_price
    raw_gross = position.direction.value * close_quantity * (exit_price - raw_entry)
    execution_gross = position.direction.value * close_quantity * (fill.price - position.entry_price)
    entry_fee = position.entry_fee * ratio
    entry_slippage = position.entry_slippage_cost * ratio
    funding = _funding_for_position(execution_config, position, exit_time) * ratio
    fee = entry_fee + fill.fee
    slippage = entry_slippage + fill.slippage_cost
    net_pnl = raw_gross - fee - slippage + funding
    cash += execution_gross - fill.fee + funding
    position.quantity -= close_quantity
    position.entry_fee -= entry_fee
    position.entry_slippage_cost -= entry_slippage
    notional = abs(close_quantity * position.entry_price)
    metadata = dict(position.strategy_metadata or {})
    trades.append(
        {
            "symbol": position.symbol,
            "strategy": MTPER_REASON_TOKEN,
            "strategy_bucket": MTPER_REASON_TOKEN,
            "side": position.direction.name,
            "direction": position.direction.name,
            "entry_time": position.entry_time.isoformat() if hasattr(position.entry_time, "isoformat") else position.entry_time,
            "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else exit_time,
            "entry_price": position.entry_price,
            "exit_price": fill.price,
            "raw_entry_price": raw_entry,
            "raw_exit_price": exit_price,
            "qty": close_quantity,
            "quantity": close_quantity,
            "notional": notional,
            "entry_fee": entry_fee,
            "exit_fee": fill.fee,
            "fee": fee,
            "fees": fee,
            "gross_pnl": raw_gross,
            "execution_gross_pnl": execution_gross,
            "slippage_cost": slippage,
            "funding": funding,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / max(notional, 1e-12),
            "net_bps": net_pnl / max(notional, 1e-12) * 10_000.0,
            "entry_reason": position.entry_reason,
            "reason": reason,
            "exit_reason": reason,
            "bars": index - position.entry_index,
            "scale_ins": position.scale_ins,
            "mfe": position.mfe * ratio,
            "mae": position.mae * ratio,
            "hold_minutes": _hold_minutes(position.entry_time, exit_time),
            "avg_hold_minutes": _hold_minutes(position.entry_time, exit_time),
            "entry_order_type": position.entry_order_type,
            "exit_order_type": fill.order_type,
            "entry_liquidity": position.entry_liquidity,
            "exit_liquidity": fill.liquidity,
            "signal_time": position.signal_time.isoformat() if hasattr(position.signal_time, "isoformat") else position.signal_time,
            "signal_available_time": position.signal_available_time.isoformat() if hasattr(position.signal_available_time, "isoformat") else position.signal_available_time,
            "campaign_risk_budget_usdt": position.campaign_risk_budget_usdt,
            "initial_leg_full_cost_risk_usdt": position.initial_leg_full_cost_risk_usdt,
            "net_campaign_r": net_pnl / max(position.campaign_risk_budget_usdt, 1e-12),
            "net_initial_leg_r": net_pnl / max(position.initial_leg_full_cost_risk_usdt, 1e-12),
            "strategy_metadata": metadata,
            "skip_reason": "",
        }
    )
    return cash


def _mtper_trend_conversion_confirmed(
    config: Any,
    position: PortfolioPosition,
    timestamp: Any,
    candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
) -> bool:
    symbol = position.symbol
    c4h = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "4h", symbol, timestamp, 80)
    c2h = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "2h", symbol, timestamp, 50)
    c1h = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "1h", symbol, timestamp, 50)
    if min(len(c4h), len(c2h), len(c1h)) < 30:
        return False
    fast = ema([item.close for item in c4h], int(config.mtper.pre_cross.ema_fast_period))
    slow = ema([item.close for item in c4h], int(config.mtper.pre_cross.ema_slow_period))
    side = position.direction.value
    cross_confirmed = side * (fast[-1] - slow[-1]) > 0
    ema2 = ema([item.close for item in c2h], 21)[-1]
    ema1 = ema([item.close for item in c1h], 21)
    _, _, hist1 = macd([item.close for item in c1h])
    return (
        cross_confirmed
        and side * (c2h[-1].close - ema2) > 0
        and side * (ema1[-1] - ema1[-3]) > 0
        and side * (hist1[-1] - hist1[-2]) >= 0
    )


def _mtper_structural_exit_reason(
    config: Any,
    position: PortfolioPosition,
    timestamp: Any,
    candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
) -> str:
    if not bool(config.mtper.exit.structural_fail_fast_enabled):
        return ""
    symbol = position.symbol
    side = position.direction.value
    c15 = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "15m", symbol, timestamp, 40)
    c1h = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "1h", symbol, timestamp, 40)
    c2h = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "2h", symbol, timestamp, 40)
    if len(c15) >= 22:
        ema9 = ema([item.close for item in c15], 9)[-1]
        trigger_level = float((position.strategy_metadata or {}).get("trigger_close", position.entry_price))
        if side * (c15[-1].close - trigger_level) < 0 and side * (c15[-1].close - ema9) < 0:
            return "mtper_structural_fail_fast_15m_reclaim_lost"
    if len(c1h) >= 30:
        _, _, hist1 = macd([item.close for item in c1h])
        if side * (hist1[-1] - hist1[-2]) < 0 and side * (c1h[-1].close - c1h[-2].close) < 0:
            return "mtper_structural_fail_fast_1h_momentum"
    if len(c2h) >= 30:
        _, _, hist2 = macd([item.close for item in c2h])
        adverse_new_extreme = c2h[-1].low < min(item.low for item in c2h[-4:-1]) if position.direction == Direction.LONG else c2h[-1].high > max(item.high for item in c2h[-4:-1])
        if adverse_new_extreme and side * (hist2[-1] - hist2[-2]) < 0:
            return "mtper_structural_fail_fast_2h_reacceleration"
    return ""


def _mtper_nonresponse_failed_checks(
    position: PortfolioPosition,
    candle: Candle,
    candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
) -> int:
    symbol = position.symbol
    side = position.direction.value
    checks = int(side * (candle.close - position.entry_price) <= 0)
    c15 = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "15m", symbol, candle.timestamp, 40)
    c1h = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, "1h", symbol, candle.timestamp, 40)
    if len(c15) >= 22:
        ema9 = ema([item.close for item in c15], 9)[-1]
        checks += int(side * (c15[-1].close - ema9) <= 0)
        checks += int(side * (c15[-1].close - c15[-2].close) <= 0)
    if len(c1h) >= 30:
        _, _, hist = macd([item.close for item in c1h])
        checks += int(side * (hist[-1] - hist[-2]) <= 0)
        checks += int(side * (c1h[-1].close - c1h[-2].close) <= 0)
    return checks


def _mtper_runner_exit_reason(
    config: Any,
    position: PortfolioPosition,
    timestamp: Any,
    candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
) -> str:
    mode = str(config.mtper.exit.trend_trailing_mode).strip().lower()
    symbol = position.symbol
    side = position.direction.value
    if mode == "1h_ema21":
        timeframe = "1h"
    else:
        timeframe = "2h"
    candles = _mtf_closed(candles_by_timeframe, timestamps_by_timeframe, timeframe, symbol, timestamp, 60)
    if len(candles) < 30:
        return ""
    ema21 = ema([item.close for item in candles], 21)[-1]
    if side * (candles[-1].close - ema21) < 0:
        return f"mtper_trend_runner_{timeframe}_ema21_exit"
    _, _, hist = macd([item.close for item in candles])
    if side * (hist[-1] - hist[-2]) < 0 and side * (hist[-2] - hist[-3]) < 0:
        return f"mtper_trend_runner_{timeframe}_macd_fade"
    return ""


def _mtper_allowed_giveback(exit_cfg: Any, max_mfe_r: float) -> float:
    if max_mfe_r >= float(exit_cfg.giveback_high_mfe_r):
        return max(0.0, float(exit_cfg.giveback_high_r))
    if max_mfe_r >= float(exit_cfg.giveback_mid_mfe_r):
        return max(0.0, float(exit_cfg.giveback_mid_r))
    return max(0.0, float(exit_cfg.giveback_low_r))


def _is_reversal_v2_position(position: PortfolioPosition) -> bool:
    return REVERSAL_V2_REASON_TOKEN in str(position.entry_reason)


def _manage_reversal_v2_position_1m(
    config: Any,
    cash: float,
    positions: dict[str, PortfolioPosition],
    trades: list[dict[str, Any]],
    position: PortfolioPosition,
    candle: Candle,
    signal_index: int,
    execution_config: BacktestExecutionConfig,
    execution_stats: BacktestExecutionStats,
    rules: Any,
    stats: dict[str, int],
) -> tuple[float, bool]:
    side = position.direction.value
    stop_hit = candle.low <= position.stop_price if position.direction == Direction.LONG else candle.high >= position.stop_price
    take_profit_hit = candle.high >= position.take_profit_price if position.direction == Direction.LONG else candle.low <= position.take_profit_price
    if stop_hit and take_profit_hit:
        execution_stats.same_bar_tp_sl_conflict_count += 1
    if stop_hit and (not take_profit_hit or execution_config.mode != "optimistic"):
        cash = _close_position(
            config,
            cash,
            positions,
            trades,
            position.symbol,
            position.stop_price,
            "reversal_v2_stop_loss_1m",
            signal_index,
            candle.timestamp,
            execution_config=execution_config,
            rules=rules,
        )
        _count_stat(stats, "exit_stop_loss")
        return cash, True
    if take_profit_hit:
        cash = _close_position(
            config,
            cash,
            positions,
            trades,
            position.symbol,
            position.take_profit_price,
            "reversal_v2_take_profit_1m",
            signal_index,
            candle.timestamp,
            execution_config=execution_config,
            rules=rules,
        )
        _count_stat(stats, "exit_take_profit")
        return cash, True

    favorable_price = candle.high if position.direction == Direction.LONG else candle.low
    favorable_r = _reversal_v2_executable_r(position, favorable_price, execution_config, rules)
    current_r = _reversal_v2_executable_r(position, candle.close, execution_config, rules)
    position.reversal_v2_max_executable_r = max(position.reversal_v2_max_executable_r, favorable_r)
    max_r = position.reversal_v2_max_executable_r
    alpha = config.reversal_alpha
    hold_minutes = _hold_minutes(position.entry_time, candle.timestamp)

    exit_reason = ""
    fail_fast_minutes = max(0, int(alpha.fail_fast_minutes))
    if fail_fast_minutes > 0 and hold_minutes >= fail_fast_minutes and max_r < float(alpha.fail_fast_min_mfe_r):
        exit_reason = "reversal_v2_fail_fast"
    elif max(0, int(alpha.time_stop_minutes)) > 0 and hold_minutes >= int(alpha.time_stop_minutes):
        exit_reason = "reversal_v2_time_stop"
    elif (
        bool(alpha.giveback_enabled)
        and max_r >= float(alpha.giveback_activation_r)
        and current_r < max_r - float(alpha.allowed_giveback_r)
    ):
        exit_reason = "reversal_v2_profit_giveback"

    if exit_reason:
        cash = _close_position(
            config,
            cash,
            positions,
            trades,
            position.symbol,
            candle.close,
            exit_reason,
            signal_index,
            candle.timestamp,
            execution_config=execution_config,
            rules=rules,
        )
        _count_stat(stats, f"exit_{exit_reason.replace('reversal_v2_', '', 1)}")
        return cash, True

    if max_r >= float(alpha.breakeven_trigger_r):
        cost_buffer = max(0.0, float(alpha.breakeven_cost_buffer_pct))
        breakeven_stop = position.entry_price * (1.0 + side * cost_buffer)
        if position.direction == Direction.LONG:
            proposed_stop = max(position.stop_price, breakeven_stop)
            stop_is_beyond_close = proposed_stop >= candle.close
        else:
            proposed_stop = min(position.stop_price, breakeven_stop)
            stop_is_beyond_close = proposed_stop <= candle.close
        if stop_is_beyond_close:
            cash = _close_position(
                config,
                cash,
                positions,
                trades,
                position.symbol,
                candle.close,
                "reversal_v2_breakeven_reversal",
                signal_index,
                candle.timestamp,
                execution_config=execution_config,
                rules=rules,
            )
            _count_stat(stats, "exit_breakeven_reversal")
            return cash, True
        if proposed_stop != position.stop_price:
            position.stop_price = proposed_stop
            _count_stat(stats, "breakeven_armed")
    return cash, False


def _reversal_v2_executable_r(
    position: PortfolioPosition,
    raw_exit_price: float,
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
    raw_gross = position.direction.value * position.quantity * (raw_exit_price - raw_entry)
    net = raw_gross - position.entry_fee - position.entry_slippage_cost - fill.fee - fill.slippage_cost
    risk_cash = max(abs(position.entry_price - position.initial_stop_price) * position.quantity, 1e-12)
    return net / risk_cash


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

    current_net_pnl, current_campaign_r, current_initial_leg_r = _cmipr_executable_r_values(
        config,
        position,
        candle.close,
        candle.timestamp,
        execution_config,
        trader.client.symbol_rules(symbol),
    )
    position.cmipr_max_campaign_executable_r = max(position.cmipr_max_campaign_executable_r, current_campaign_r)
    position.cmipr_min_campaign_executable_r = min(position.cmipr_min_campaign_executable_r, current_campaign_r)
    position.cmipr_max_initial_leg_executable_r = max(position.cmipr_max_initial_leg_executable_r, current_initial_leg_r)
    position.cmipr_min_initial_leg_executable_r = min(position.cmipr_min_initial_leg_executable_r, current_initial_leg_r)
    position.cmipr_executable_mfe_usdt = max(position.cmipr_executable_mfe_usdt, current_net_pnl)
    position.cmipr_executable_mae_usdt = min(position.cmipr_executable_mae_usdt, current_net_pnl)
    position.cmipr_max_executable_r = position.cmipr_max_campaign_executable_r
    hold_minutes = _hold_minutes(position.entry_time, candle.timestamp)
    exit_cfg = config.cmipr.exit
    fail_fast_basis = normalize_r_basis(exit_cfg.fail_fast_r_basis)
    take_profit_basis = normalize_r_basis(exit_cfg.take_profit_r_basis)
    breakeven_basis = normalize_r_basis(exit_cfg.breakeven_r_basis)
    runner_basis = normalize_r_basis(exit_cfg.runner_activation_r_basis)
    giveback_basis = normalize_r_basis(exit_cfg.giveback_r_basis)
    max_by_basis = {
        CAMPAIGN_R_BASIS: position.cmipr_max_campaign_executable_r,
        INITIAL_LEG_R_BASIS: position.cmipr_max_initial_leg_executable_r,
    }
    current_by_basis = {
        CAMPAIGN_R_BASIS: current_campaign_r,
        INITIAL_LEG_R_BASIS: current_initial_leg_r,
    }
    breakout_level = _reason_float(position.entry_reason, "breakout_level", position.entry_price)
    lost_level = candle.close < breakout_level if position.direction == Direction.LONG else candle.close > breakout_level
    fail_fast_minutes = max(1, int(exit_cfg.fail_fast_bars_5m)) * 5
    reason = ""
    fail_fast_mode = str(exit_cfg.fail_fast_mode or "current").strip().lower()
    if fail_fast_mode == "current":
        if hold_minutes >= fail_fast_minutes and max_by_basis[fail_fast_basis] < float(exit_cfg.fail_fast_min_mfe_r):
            reason = "cmipr_fail_fast_no_extension"
        elif lost_level and hold_minutes >= 5:
            reason = "cmipr_fail_fast_lost_breakout"
    else:
        hard_invalidated = _cmipr_hard_invalidation_reason(
            position,
            candle,
            breakout_level,
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
        )
        if hard_invalidated:
            reason = hard_invalidated
        elif fail_fast_mode.startswith("conditional"):
            conditional_minutes = max(1, int(exit_cfg.conditional_fail_fast_minutes))
            conditional_min_mfe = float(exit_cfg.conditional_fail_fast_min_mfe_r)
            if hold_minutes >= conditional_minutes and max_by_basis[fail_fast_basis] < conditional_min_mfe:
                failed_checks = _cmipr_conditional_fail_fast_failed_checks(
                    position,
                    candle,
                    mtf_candles_by_timeframe,
                    mtf_timestamps_by_timeframe,
                )
                if failed_checks >= max(1, int(exit_cfg.conditional_min_failed_checks)):
                    reason = "cmipr_fail_fast_conditional_no_extension"
                    _count_stat(stats, f"conditional_fail_fast_failed_checks_{failed_checks}")
        elif fail_fast_mode not in {"structural_only", "hard_invalidation_only", "no_time"}:
            raise ValueError(f"unsupported CMIPR fail_fast_mode: {exit_cfg.fail_fast_mode}")
    if not reason and hold_minutes >= max(1, int(exit_cfg.max_holding_minutes)):
        reason = "cmipr_time_stop"
    elif not reason and not bool(exit_cfg.runner_enabled) and current_by_basis[take_profit_basis] >= float(exit_cfg.fixed_take_profit_r):
        reason = "cmipr_fixed_r_take_profit"
    elif not reason:
        max_giveback_r = max_by_basis[giveback_basis]
        giveback = _cmipr_allowed_giveback_r(exit_cfg, max_giveback_r)
        if max_by_basis[runner_basis] >= float(exit_cfg.runner_activation_r) and current_by_basis[giveback_basis] < max_giveback_r - giveback:
            reason = "cmipr_max_profit_giveback"

    if not reason and bool(exit_cfg.runner_enabled) and max_by_basis[runner_basis] >= float(exit_cfg.runner_activation_r):
        trailing_type = str(exit_cfg.trailing_type).lower()
        if trailing_type not in {
            "ema9_5m",
            "ema21_5m",
            "ema9_15m",
            "ema21_15m",
            "chandelier_5m",
            "chandelier_15m",
        }:
            trailing_type = "ema21_15m"
        trailing_timeframe = "5m" if trailing_type.endswith("_5m") else "15m"
        trailing_candles = _mtf_closed(
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
            trailing_timeframe,
            symbol,
            candle.timestamp,
            80,
        )
        if trailing_type.startswith("ema") and len(trailing_candles) >= 24:
            closes = [item.close for item in trailing_candles]
            ema_period = 9 if trailing_type.startswith("ema9_") else 21
            trailing_ema = ema(closes, ema_period)[-1]
            if position.direction == Direction.LONG and trailing_candles[-1].close < trailing_ema:
                reason = "cmipr_runner_ema_exit"
            elif position.direction == Direction.SHORT and trailing_candles[-1].close > trailing_ema:
                reason = "cmipr_runner_ema_exit"
        elif trailing_type.startswith("chandelier_") and len(trailing_candles) >= 24:
            atr_value = atr(trailing_candles, 14)[-1]
            peak_price = position.entry_price + position.direction.value * position.mfe / max(position.quantity, 1e-12)
            multiple = max(0.25, float(exit_cfg.chandelier_atr_mult))
            proposed_stop = peak_price - position.direction.value * atr_value * multiple
            if position.direction == Direction.LONG:
                position.stop_price = max(position.stop_price, proposed_stop)
            else:
                position.stop_price = min(position.stop_price, proposed_stop)
            _count_stat(stats, "cmipr_runner_chandelier_armed")

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
    if max_by_basis[breakeven_basis] >= float(exit_cfg.breakeven_trigger_r):
        buffer_pct = max(0.0, float(exit_cfg.breakeven_cost_buffer_pct))
        if position.direction == Direction.LONG:
            position.stop_price = max(position.stop_price, position.entry_price * (1.0 + buffer_pct))
        else:
            position.stop_price = min(position.stop_price, position.entry_price * (1.0 - buffer_pct))
    return cash, False


def _cmipr_hard_invalidation_reason(
    position: PortfolioPosition,
    candle: Candle,
    breakout_level: float,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
) -> str:
    closed_5m = _mtf_closed(
        mtf_candles_by_timeframe,
        mtf_timestamps_by_timeframe,
        "5m",
        position.symbol,
        candle.timestamp,
        3,
    )
    if not closed_5m:
        return ""
    latest = closed_5m[-1]
    lost = latest.close < breakout_level if position.direction == Direction.LONG else latest.close > breakout_level
    return "cmipr_hard_invalidation_lost_breakout" if lost else ""


def _cmipr_conditional_fail_fast_failed_checks(
    position: PortfolioPosition,
    candle: Candle,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
) -> int:
    symbol_5m = _mtf_closed(
        mtf_candles_by_timeframe,
        mtf_timestamps_by_timeframe,
        "5m",
        position.symbol,
        candle.timestamp,
        24,
    )
    if len(symbol_5m) < 10:
        return 5
    latest = symbol_5m[-1]
    previous = symbol_5m[-4:-1]
    closes = [item.close for item in symbol_5m]
    _, _, histogram = macd(closes)
    if position.direction == Direction.LONG:
        higher_structure = latest.low > min(item.low for item in previous)
        momentum_improving = histogram[-1] > histogram[-2]
        ema_held = latest.close >= ema(closes, 9)[-1]
    else:
        higher_structure = latest.high < max(item.high for item in previous)
        momentum_improving = histogram[-1] < histogram[-2]
        ema_held = latest.close <= ema(closes, 9)[-1]
    average_volume = sum(item.volume for item in previous) / len(previous)
    volume_reexpanded = latest.volume >= average_volume * 1.05

    btc_5m = _mtf_closed(
        mtf_candles_by_timeframe,
        mtf_timestamps_by_timeframe,
        "5m",
        "BTCUSDT",
        candle.timestamp,
        3,
    )
    relative_strength_improving = False
    if len(btc_5m) >= 2 and len(symbol_5m) >= 2:
        symbol_return = symbol_5m[-1].close / max(symbol_5m[-2].close, 1e-12) - 1.0
        btc_return = btc_5m[-1].close / max(btc_5m[-2].close, 1e-12) - 1.0
        relative_strength_improving = (
            symbol_return >= btc_return if position.direction == Direction.LONG else symbol_return <= btc_return
        )
    checks = (
        higher_structure,
        volume_reexpanded,
        momentum_improving,
        relative_strength_improving,
        ema_held,
    )
    return sum(not passed for passed in checks)


def _cmipr_executable_r_values(
    config: Any,
    position: PortfolioPosition,
    raw_exit_price: float,
    exit_time: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> tuple[float, float, float]:
    funding = _funding_for_position(execution_config, position, exit_time)
    net = executable_net_pnl(position, raw_exit_price, execution_config, rules, funding)
    return (
        net,
        net_pnl_r(position, net, CAMPAIGN_R_BASIS),
        net_pnl_r(position, net, INITIAL_LEG_R_BASIS),
    )


def _cmipr_executable_campaign_r(
    config: Any,
    position: PortfolioPosition,
    raw_exit_price: float,
    exit_time: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> float:
    return _cmipr_executable_r_values(
        config,
        position,
        raw_exit_price,
        exit_time,
        execution_config,
        rules,
    )[1]


def _cmipr_executable_initial_leg_r(
    config: Any,
    position: PortfolioPosition,
    raw_exit_price: float,
    exit_time: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> float:
    return _cmipr_executable_r_values(
        config,
        position,
        raw_exit_price,
        exit_time,
        execution_config,
        rules,
    )[2]


def _cmipr_executable_current_r(
    config: Any,
    position: PortfolioPosition,
    raw_exit_price: float,
    exit_time: Any,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> float:
    """Backward-compatible alias. CMIPR callers must use explicit basis helpers."""
    return _cmipr_executable_campaign_r(
        config,
        position,
        raw_exit_price,
        exit_time,
        execution_config,
        rules,
    )


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
    data_start_day = days[0] if days else None
    result: dict[Any, frozenset[str]] = {}
    for index, day in enumerate(days):
        prior_days = days[max(0, index - lookback_days):index]
        scores: dict[str, float] = {}
        for prior_day in prior_days:
            for symbol, quote_volume in daily[prior_day].items():
                first_seen = first_day[symbol]
                if first_seen != data_start_day and (day - first_seen).days < warmup_days:
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
        thirty_m = _mtf_closed(
            mtf_candles_by_timeframe,
            mtf_timestamps_by_timeframe,
            "30m",
            position.symbol,
            candle.timestamp,
            80,
        )
    except KeyError:
        thirty_m = []
    if (
        bool(getattr(strategy, "mtf_exit_on_30m_confirm_lost", False))
        and thirty_m
        and Mtf4hRsiRegimePullbackStrategy(config).thirty_minute_confirm_lost(
            position.direction,
            thirty_m,
        )
    ):
        return "mtf_30m_confirm_lost"

    try:
        one_h = _mtf_closed(mtf_candles_by_timeframe, mtf_timestamps_by_timeframe, "1h", position.symbol, candle.timestamp, 80)
    except KeyError:
        one_h = []
    if (
        bool(getattr(strategy, "mtf_exit_on_1h_confirm_lost", True))
        and one_h
        and Mtf4hRsiRegimePullbackStrategy(config).one_h_confirm_lost(position.direction, one_h)
    ):
        return "mtf_1h_confirm_lost"

    initial_stop = position.initial_stop_price or position.stop_price
    risk_cash = abs(position.entry_price - initial_stop) * position.quantity
    fail_fast_minutes = max(1, int(getattr(strategy, "mtf_fail_fast_minutes", 90)))
    if hold_minutes >= fail_fast_minutes and risk_cash > 0:
        min_mfe = risk_cash * max(0.0, float(getattr(strategy, "mtf_fail_fast_min_r", 0.5)))
        if position.mfe < min_mfe:
            return "mtf_fail_fast"

    max_holding_minutes = max(1, int(getattr(strategy, "mtf_max_holding_minutes", 720)))
    if hold_minutes >= max_holding_minutes:
        return "mtf_time_stop"
    return None


def _update_mtf_profit_protection(
    config: Any,
    position: PortfolioPosition,
    previous_mfe: float,
    execution_config: BacktestExecutionConfig,
    candle: Candle,
    mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    mtf_timestamps_by_timeframe: dict[str, dict[str, list[Any]]],
) -> None:
    strategy = config.strategy
    if not bool(getattr(strategy, "mtf_profit_protection_enabled", False)):
        return
    initial_stop = position.initial_stop_price or position.stop_price
    risk_distance = abs(position.entry_price - initial_stop)
    risk_cash = risk_distance * position.quantity
    if risk_cash <= 0 or position.quantity <= 0:
        return
    previous_mfe_r = previous_mfe / risk_cash
    side = position.direction.value
    protected_stop = position.stop_price
    stop_mode = ""

    breakeven_r = max(0.0, float(getattr(strategy, "mtf_move_stop_to_breakeven_r", 1.0)))
    if previous_mfe_r >= breakeven_r:
        projected_exit_fee = abs(position.quantity * position.entry_price) * float(execution_config.taker_fee_rate)
        projected_stop_slippage = (
            abs(position.quantity * position.entry_price)
            * float(execution_config.stop_slippage_bps)
            / 10_000.0
        )
        extra_cost = (
            abs(position.quantity * position.entry_price)
            * max(0.0, float(getattr(strategy, "mtf_breakeven_extra_bps", 0.0)))
            / 10_000.0
        )
        full_cost_cash = (
            position.entry_fee
            + position.entry_slippage_cost
            + projected_exit_fee
            + projected_stop_slippage
            + extra_cost
        )
        breakeven_stop = _mtf_favorable_stop(
            position,
            protected_stop,
            position.entry_price + side * full_cost_cash / position.quantity,
        )
        if breakeven_stop != protected_stop:
            protected_stop = breakeven_stop
            stop_mode = "breakeven"

    trailing_start_r = max(0.0, float(getattr(strategy, "mtf_trailing_start_r", 1.5)))
    trailing_mode = str(getattr(strategy, "mtf_trailing_mode", "none")).strip().lower()
    if previous_mfe_r >= trailing_start_r and trailing_mode in {"giveback", "atr", "hybrid"}:
        if trailing_mode in {"giveback", "hybrid"}:
            giveback_r = max(0.05, float(getattr(strategy, "mtf_profit_giveback_r", 0.75)))
            locked_r = max(0.0, previous_mfe_r - giveback_r)
            giveback_stop = _mtf_favorable_stop(
                position,
                protected_stop,
                position.entry_price + side * locked_r * risk_distance,
            )
            if giveback_stop != protected_stop:
                protected_stop = giveback_stop
                stop_mode = "giveback"
        if trailing_mode in {"atr", "hybrid"}:
            try:
                candles_15m = _mtf_closed(
                    mtf_candles_by_timeframe,
                    mtf_timestamps_by_timeframe,
                    "15m",
                    position.symbol,
                    candle.timestamp,
                    80,
                )
            except KeyError:
                candles_15m = []
            if len(candles_15m) >= 20:
                atr_value = max(atr(candles_15m, 14)[-1], 1e-12)
                multiple = max(0.1, float(getattr(strategy, "mtf_trailing_atr15m_mult", 1.0)))
                reference = candles_15m[-1].close - side * atr_value * multiple
                atr_stop = _mtf_favorable_stop(position, protected_stop, reference)
                if atr_stop != protected_stop:
                    protected_stop = atr_stop
                    stop_mode = "atr"

    if position.direction == Direction.LONG and protected_stop > candle.open:
        protected_stop = candle.open
    elif position.direction == Direction.SHORT and protected_stop < candle.open:
        protected_stop = candle.open
    position.stop_price = protected_stop
    if stop_mode:
        metadata = dict(position.strategy_metadata or {})
        metadata["mtf_stop_mode"] = stop_mode
        position.strategy_metadata = metadata


def _mtf_favorable_stop(position: PortfolioPosition, current: float, candidate: float) -> float:
    if position.direction == Direction.LONG:
        return max(current, candidate)
    return min(current, candidate)


def _mtf_stop_exit_reason(position: PortfolioPosition) -> str | None:
    if MTF_REASON_TOKEN not in str(position.entry_reason):
        return None
    mode = str((position.strategy_metadata or {}).get("mtf_stop_mode", ""))
    if mode in {"breakeven", "giveback", "atr"}:
        return f"mtf_{mode}_stop"
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
        _, current_campaign_r, current_initial_leg_r = _cmipr_executable_r_values(
            config,
            position,
            candle.close,
            timestamp,
            execution_config,
            client.symbol_rules(symbol),
        )
        addon_basis = normalize_r_basis(pyramid.addon_trigger_r_basis)
        current_r = current_initial_leg_r if addon_basis == INITIAL_LEG_R_BASIS else current_campaign_r
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
        _, current_campaign_r, current_initial_leg_r = _cmipr_executable_r_values(
            config,
            position,
            candle.open,
            timestamp,
            execution_config,
            trader.client.symbol_rules(addon.symbol),
        )
        addon_basis = normalize_r_basis(config.cmipr.pyramid.addon_trigger_r_basis)
        current_r = current_initial_leg_r if addon_basis == INITIAL_LEG_R_BASIS else current_campaign_r
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

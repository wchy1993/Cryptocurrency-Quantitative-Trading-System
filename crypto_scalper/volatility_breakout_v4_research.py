from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from array import array
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from .binance_client import SymbolRules
from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
    CombinedPortfolioConfig,
)
from .combined_volatility_trend_grid_v4_backtest import (
    simulate_combined_v4_portfolio,
)
from .data import parse_timestamp
from .indicators import atr, ema
from .live_config import load_live_config
from .live_execution_backtest import _resample_to_timeframe
from .live_portfolio_backtest import _inferred_symbol_rules
from .models import Candle, Direction
from .realistic_data import load_funding_rate_directory
from .risk import BacktestExecutionConfig, FundingRate, execution_config_from_live_config
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import (
    GridPortfolioConfig,
    build_grid_research_timeline,
    compact_grid_summary,
    simulate_grid_portfolio,
)
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import (
    ExitProtectionConfig,
    simulate_exit_protected_portfolio,
)
from .volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    PortfolioSearchConfig,
    UNIVERSE_50,
    build_candidates,
    compact_summary,
    minute_datetime,
    minute_token,
    sha256_file,
    _shift_candidates,
)


V4_RESEARCH_NAME = "dual_thrust_volatility_breakout_v4_regime_entry_exit"
V4_RESEARCH_VERSION = "v4_independent_3m_live_like_research"


@dataclass(frozen=True)
class V4RegimeConfig:
    """Point-in-time market-state gates applied before portfolio selection."""

    min_directional_btc_return_4h: float = -999.0
    max_directional_btc_return_4h: float = 999.0
    min_directional_eth_return_4h: float = -999.0
    max_directional_eth_return_4h: float = 999.0
    min_directional_breadth: float = 0.0
    max_directional_breadth: float = 1.0
    min_directional_breadth_change_4h: float = -999.0
    min_directional_symbol_return_4h: float = -999.0
    max_directional_symbol_return_4h: float = 999.0
    min_directional_symbol_efficiency_12h: float = -999.0
    min_market_efficiency_12h: float = 0.0
    min_directional_symbol_ema55_atr: float = -999.0
    max_directional_symbol_ema55_atr: float = 999.0
    min_regime_score: float = -999.0
    max_regime_score: float = 999.0

    def validate(self) -> None:
        pairs = (
            (self.min_directional_btc_return_4h, self.max_directional_btc_return_4h),
            (self.min_directional_eth_return_4h, self.max_directional_eth_return_4h),
            (self.min_directional_breadth, self.max_directional_breadth),
            (self.min_directional_symbol_return_4h, self.max_directional_symbol_return_4h),
            (self.min_directional_symbol_ema55_atr, self.max_directional_symbol_ema55_atr),
            (self.min_regime_score, self.max_regime_score),
        )
        if any(low > high for low, high in pairs):
            raise ValueError("v4 regime minimum cannot exceed maximum")
        if not 0.0 <= self.min_directional_breadth <= 1.0:
            raise ValueError("min_directional_breadth must be in [0, 1]")
        if not 0.0 <= self.max_directional_breadth <= 1.0:
            raise ValueError("max_directional_breadth must be in [0, 1]")
        if not 0.0 <= self.min_market_efficiency_12h <= 1.0:
            raise ValueError("min_market_efficiency_12h must be in [0, 1]")


@dataclass(frozen=True)
class V4MarketSnapshot:
    available_minute: int
    symbol: str
    btc_return_4h: float
    eth_return_4h: float
    breadth_above_ema21: float
    breadth_change_4h: float
    symbol_return_4h: float
    symbol_efficiency_12h: float
    market_efficiency_12h: float
    symbol_ema55_atr: float


@dataclass(frozen=True)
class V4Variant:
    name: str
    signal: VolatilityBreakoutConfig
    portfolio: PortfolioSearchConfig
    regime: V4RegimeConfig
    exit: ExitProtectionConfig
    family_key: str


def _latest_csv_paths(
    roots: Sequence[str | Path], symbol: str, interval: str
) -> list[Path]:
    paths: list[Path] = []
    for root_value in roots:
        root = Path(root_value)
        matches = sorted(root.glob(f"{symbol}_{interval}_*.csv"))
        if matches:
            paths.append(matches[-1])
    return paths


def _read_stitched_1m_rows(
    paths: Sequence[Path],
    start: datetime,
    end: datetime,
) -> dict[int, tuple[float, float, float, float, float]]:
    """Read overlapping official Binance CSVs; later roots win on duplicates."""

    output: dict[int, tuple[float, float, float, float, float]] = {}
    start_text = start.isoformat(timespec="seconds")
    end_text = end.isoformat(timespec="seconds")
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                timestamp_text = row["timestamp"]
                if timestamp_text < start_text:
                    continue
                if timestamp_text >= end_text:
                    break
                timestamp = parse_timestamp(timestamp_text)
                output[minute_token(timestamp)] = (
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                )
    return output


def _series_from_rows(
    rows: dict[int, tuple[float, float, float, float, float]],
    start: datetime,
    end: datetime,
) -> CompactSeries:
    start_minute = minute_token(start)
    end_minute = minute_token(end)
    tokens = sorted(token for token in rows if start_minute <= token < end_minute)
    return CompactSeries(
        minutes=array("q", tokens),
        opens=array("d", (rows[token][0] for token in tokens)),
        highs=array("d", (rows[token][1] for token in tokens)),
        lows=array("d", (rows[token][2] for token in tokens)),
        closes=array("d", (rows[token][3] for token in tokens)),
        volumes=array("d", (rows[token][4] for token in tokens)),
    )


def _candles_from_rows(
    rows: dict[int, tuple[float, float, float, float, float]],
) -> list[Candle]:
    return [
        Candle(minute_datetime(token), *rows[token])
        for token in sorted(rows)
    ]


def _coverage_audit(series: CompactSeries, start: datetime, end: datetime) -> dict[str, Any]:
    expected = minute_token(end) - minute_token(start)
    observed = len(series.minutes)
    maximum_gap = 0
    gap_count = 0
    previous: int | None = None
    for token in series.minutes:
        token = int(token)
        if previous is not None and token != previous + 1:
            gap_count += 1
            maximum_gap = max(maximum_gap, token - previous - 1)
        previous = token
    return {
        "expected_minutes": expected,
        "observed_minutes": observed,
        "missing_minutes": expected - observed,
        "coverage_ratio": observed / expected if expected else 0.0,
        "gap_count": gap_count,
        "maximum_gap_minutes": maximum_gap,
        "first_minute": minute_datetime(int(series.minutes[0])).isoformat() if observed else None,
        "last_minute": minute_datetime(int(series.minutes[-1])).isoformat() if observed else None,
    }


def load_stitched_research_data(
    symbols: Iterable[str],
    one_minute_roots: Sequence[str | Path],
    start: datetime,
    end: datetime,
    *,
    warmup_days: int = 21,
) -> tuple[
    dict[str, list[Candle]],
    dict[str, CompactSeries],
    dict[str, SymbolRules],
    dict[str, dict[str, Any]],
]:
    """Load one canonical 1m path and derive closed 15m signal bars from it."""

    symbols = tuple(symbol.upper() for symbol in symbols)
    warmup_start = start - timedelta(days=warmup_days)
    signal_data: dict[str, list[Candle]] = {}
    execution_data: dict[str, CompactSeries] = {}
    rules: dict[str, SymbolRules] = {}
    audit: dict[str, dict[str, Any]] = {}
    for number, symbol in enumerate(symbols, start=1):
        paths = _latest_csv_paths(one_minute_roots, symbol, "1m")
        if not paths:
            audit[symbol] = {"error": "missing_source_file"}
            continue
        rows = _read_stitched_1m_rows(paths, warmup_start, end)
        series = _series_from_rows(rows, start, end)
        one_minute_candles = _candles_from_rows(rows)
        fifteen_minute = _resample_to_timeframe(one_minute_candles, "1m", "15m")
        coverage = _coverage_audit(series, start, end)
        coverage["source_files"] = [str(path) for path in paths]
        coverage["signal_15m_candles"] = len(fifteen_minute)
        audit[symbol] = coverage
        if len(fifteen_minute) < 500 or len(series.minutes) < 1_000:
            continue
        signal_data[symbol] = fifteen_minute
        execution_data[symbol] = series
        rules[symbol] = _inferred_symbol_rules(symbol, fifteen_minute)
        if number % 10 == 0 or number == len(symbols):
            print(f"stitched data {number}/{len(symbols)} symbols", flush=True)
    return signal_data, execution_data, rules, audit


def load_stitched_funding(
    symbols: Iterable[str], funding_roots: Sequence[str | Path]
) -> dict[str, tuple[FundingRate, ...]]:
    merged: dict[str, dict[datetime, FundingRate]] = defaultdict(dict)
    symbols = tuple(symbol.upper() for symbol in symbols)
    for root in funding_roots:
        for symbol, rows in load_funding_rate_directory(root, symbols).items():
            for row in rows:
                merged[symbol][row.timestamp] = row
    return {
        symbol: tuple(rows[timestamp] for timestamp in sorted(rows))
        for symbol, rows in merged.items()
        if rows
    }


def load_v4_runtime_inputs(
    symbols: Iterable[str],
    one_minute_roots: Sequence[str | Path],
    funding_roots: Sequence[str | Path],
    cost_config: str | Path,
    start: datetime,
    end: datetime,
) -> tuple[
    dict[str, list[Candle]],
    dict[str, CompactSeries],
    dict[str, SymbolRules],
    BacktestExecutionConfig,
    dict[str, Any],
]:
    symbols = tuple(symbols)
    signal_data, execution_data, rules, audit = load_stitched_research_data(
        symbols, one_minute_roots, start, end
    )
    missing = sorted(set(symbols) - set(signal_data) | (set(symbols) - set(execution_data)))
    if missing:
        raise RuntimeError(f"incomplete stitched universe data: {missing}")
    funding = load_stitched_funding(symbols, funding_roots)
    live_config = load_live_config(cost_config)
    execution = execution_config_from_live_config(
        live_config, cost_experiment="full_cost", mode="conservative"
    )
    execution = replace(
        execution,
        funding_enabled=True,
        funding_default_rate=0.0,
        funding_rates_by_symbol=funding,
    )
    metadata = {
        "coverage": audit,
        "minimum_coverage_ratio": min(row["coverage_ratio"] for row in audit.values()),
        "maximum_missing_minutes": max(row["missing_minutes"] for row in audit.values()),
        "funding_symbols": sorted(funding),
        "funding_missing_symbols": sorted(set(symbols) - set(funding)),
    }
    return signal_data, execution_data, rules, execution, metadata


def _efficiency_ratio(closes: list[float], index: int, lookback: int) -> float:
    displacement = closes[index] - closes[index - lookback]
    path = sum(abs(closes[row] - closes[row - 1]) for row in range(index - lookback + 1, index + 1))
    return displacement / max(path, 1e-12)


def build_v4_market_context(
    universe: Iterable[str],
    signal_data: dict[str, list[Candle]],
) -> dict[int, dict[str, V4MarketSnapshot]]:
    """Build only point-in-time features available at each completed 60m close."""

    universe = tuple(universe)
    local: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for symbol in universe:
        source = signal_data.get(symbol)
        if not source:
            continue
        candles = _resample_to_timeframe(source, "15m", "60m")
        closes = [candle.close for candle in candles]
        ema21 = ema(closes, 21)
        ema55 = ema(closes, 55)
        atr14 = atr(candles, 14)
        for index in range(55, len(candles)):
            available = minute_token(candles[index].timestamp + timedelta(minutes=60))
            local[available][symbol] = {
                "return_4h": closes[index] / max(closes[index - 4], 1e-12) - 1.0,
                "efficiency_12h": _efficiency_ratio(closes, index, 12),
                "above_ema21": float(closes[index] > ema21[index]),
                "ema55_atr": (closes[index] - ema55[index]) / max(atr14[index], 1e-12),
            }

    breadth_by_minute: dict[int, float] = {}
    market_efficiency_by_minute: dict[int, float] = {}
    for minute, rows in local.items():
        if len(rows) < max(5, len(universe) // 2):
            continue
        breadth_by_minute[minute] = statistics.mean(row["above_ema21"] for row in rows.values())
        market_efficiency_by_minute[minute] = statistics.mean(
            abs(row["efficiency_12h"]) for row in rows.values()
        )

    output: dict[int, dict[str, V4MarketSnapshot]] = {}
    for minute, rows in local.items():
        breadth = breadth_by_minute.get(minute)
        market_efficiency = market_efficiency_by_minute.get(minute)
        btc = rows.get("BTCUSDT")
        eth = rows.get("ETHUSDT")
        if breadth is None or market_efficiency is None or btc is None or eth is None:
            continue
        breadth_change = breadth - breadth_by_minute.get(minute - 240, breadth)
        output[minute] = {
            symbol: V4MarketSnapshot(
                available_minute=minute,
                symbol=symbol,
                btc_return_4h=btc["return_4h"],
                eth_return_4h=eth["return_4h"],
                breadth_above_ema21=breadth,
                breadth_change_4h=breadth_change,
                symbol_return_4h=row["return_4h"],
                symbol_efficiency_12h=row["efficiency_12h"],
                market_efficiency_12h=market_efficiency,
                symbol_ema55_atr=row["ema55_atr"],
            )
            for symbol, row in rows.items()
        }
    return output


def _context_minute(entry_minute: int) -> int:
    return entry_minute - entry_minute % 60


def enrich_candidates_v4(
    candidates: dict[int, list[Candidate]],
    context: dict[int, dict[str, V4MarketSnapshot]],
) -> dict[int, list[Candidate]]:
    output: dict[int, list[Candidate]] = {}
    for minute, rows in candidates.items():
        snapshots = context.get(_context_minute(minute), {})
        enriched: list[Candidate] = []
        for candidate in rows:
            snapshot = snapshots.get(candidate.signal.symbol)
            if snapshot is None:
                enriched.append(candidate)
                continue
            enriched.append(
                replace(
                    candidate,
                    btc_return_4h=snapshot.btc_return_4h,
                    eth_return_4h=snapshot.eth_return_4h,
                    breadth_above_ema21=snapshot.breadth_above_ema21,
                )
            )
        if enriched:
            output[minute] = enriched
    return output


def _regime_score(snapshot: V4MarketSnapshot, direction: Direction) -> float:
    side = float(direction.value)

    def clipped(value: float, scale: float) -> float:
        return max(-2.0, min(2.0, value / scale))

    directional_breadth = (
        snapshot.breadth_above_ema21
        if direction == Direction.LONG
        else 1.0 - snapshot.breadth_above_ema21
    )
    return (
        0.18 * clipped(side * snapshot.btc_return_4h, 0.01)
        + 0.14 * clipped(side * snapshot.eth_return_4h, 0.012)
        + 0.18 * clipped(directional_breadth - 0.5, 0.20)
        + 0.12 * clipped(side * snapshot.breadth_change_4h, 0.10)
        + 0.18 * clipped(side * snapshot.symbol_return_4h, 0.02)
        + 0.12 * clipped(side * snapshot.symbol_efficiency_12h, 0.25)
        + 0.08 * clipped(side * snapshot.symbol_ema55_atr, 2.0)
    )


def _passes_signal_filters(signal: Any, config: VolatilityBreakoutConfig) -> bool:
    if signal.direction == Direction.LONG and not config.allow_long:
        return False
    if signal.direction == Direction.SHORT and not config.allow_short:
        return False
    return (
        config.min_trend_alignment_atr <= signal.trend_alignment_atr <= config.max_trend_alignment_atr
        and config.min_volume_ratio <= signal.volume_ratio <= config.max_volume_ratio
        and config.min_body_atr <= signal.body_atr <= config.max_body_atr
        and signal.directional_close_position >= config.min_directional_close_position
        and config.min_range_atr <= signal.range_atr <= config.max_range_atr
        and config.min_breakout_extension_atr
        <= signal.breakout_extension_atr
        <= config.max_breakout_extension_atr
    )


def _passes_regime(
    candidate: Candidate,
    snapshot: V4MarketSnapshot | None,
    config: V4RegimeConfig,
) -> bool:
    if snapshot is None:
        return config == V4RegimeConfig()
    side = float(candidate.signal.direction.value)
    directional_btc = side * snapshot.btc_return_4h
    directional_eth = side * snapshot.eth_return_4h
    directional_breadth = (
        snapshot.breadth_above_ema21
        if candidate.signal.direction == Direction.LONG
        else 1.0 - snapshot.breadth_above_ema21
    )
    directional_symbol_return = side * snapshot.symbol_return_4h
    directional_efficiency = side * snapshot.symbol_efficiency_12h
    directional_alignment = side * snapshot.symbol_ema55_atr
    score = _regime_score(snapshot, candidate.signal.direction)
    return (
        config.min_directional_btc_return_4h
        <= directional_btc
        <= config.max_directional_btc_return_4h
        and config.min_directional_eth_return_4h
        <= directional_eth
        <= config.max_directional_eth_return_4h
        and config.min_directional_breadth <= directional_breadth <= config.max_directional_breadth
        and side * snapshot.breadth_change_4h >= config.min_directional_breadth_change_4h
        and config.min_directional_symbol_return_4h
        <= directional_symbol_return
        <= config.max_directional_symbol_return_4h
        and directional_efficiency >= config.min_directional_symbol_efficiency_12h
        and snapshot.market_efficiency_12h >= config.min_market_efficiency_12h
        and config.min_directional_symbol_ema55_atr
        <= directional_alignment
        <= config.max_directional_symbol_ema55_atr
        and config.min_regime_score <= score <= config.max_regime_score
    )


def filter_candidates_v4(
    candidates: dict[int, list[Candidate]],
    signal_config: VolatilityBreakoutConfig,
    regime_config: V4RegimeConfig,
    context: dict[int, dict[str, V4MarketSnapshot]],
) -> dict[int, list[Candidate]]:
    """Apply entry and regime filters, then reproduce the per-symbol daily cap."""

    regime_config.validate()
    counts: dict[tuple[str, str], int] = defaultdict(int)
    output: dict[int, list[Candidate]] = {}
    for minute in sorted(candidates):
        selected: list[Candidate] = []
        snapshots = context.get(_context_minute(minute), {})
        for candidate in candidates[minute]:
            signal = candidate.signal
            if not _passes_signal_filters(signal, signal_config):
                continue
            if not _passes_regime(candidate, snapshots.get(signal.symbol), regime_config):
                continue
            day = minute_datetime(minute).date().isoformat()
            key = (signal.symbol, day)
            if counts[key] >= signal_config.max_signals_per_symbol_day:
                continue
            counts[key] += 1
            selected.append(candidate)
        if selected:
            output[minute] = selected
    return output


def fold_net_profit(
    result: dict[str, Any], fold_boundaries: Sequence[tuple[datetime, datetime]]
) -> list[float]:
    output: list[float] = []
    for start, end in fold_boundaries:
        output.append(
            sum(
                float(trade["net_pnl"])
                for trade in result["trades"]
                if start <= parse_timestamp(trade["exit_time"]) < end
            )
        )
    return output


def result_quality(
    result: dict[str, Any], fold_boundaries: Sequence[tuple[datetime, datetime]]
) -> dict[str, Any]:
    folds = fold_net_profit(result, fold_boundaries)
    positive_folds = sum(value > 0.0 for value in folds)
    return {
        "fold_net_profit": folds,
        "positive_folds": positive_folds,
        "minimum_fold_net_profit": min(folds) if folds else 0.0,
        "median_fold_net_profit": statistics.median(folds) if folds else 0.0,
        "all_folds_positive": bool(folds) and positive_folds == len(folds),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _family_key(config: VolatilityBreakoutConfig) -> str:
    return (
        f"tf{config.timeframe_minutes}_lb{config.lookback_days}_"
        f"lk{config.long_k:g}_sk{config.short_k:g}"
    )


def build_v4_families() -> dict[str, VolatilityBreakoutConfig]:
    """Bounded structural families; thresholds are applied after raw crossings."""

    specs: list[tuple[int, int, float, float]] = []
    for lookback in (3, 5, 7):
        for long_k, short_k in (
            (0.50, 0.50),
            (0.60, 0.60),
            (0.70, 0.60),
            (0.70, 0.70),
            (0.90, 0.80),
        ):
            specs.append((60, lookback, long_k, short_k))
    for lookback in (3, 5):
        for long_k, short_k in ((0.60, 0.60), (0.70, 0.60), (0.80, 0.80)):
            specs.append((30, lookback, long_k, short_k))

    output: dict[str, VolatilityBreakoutConfig] = {}
    for timeframe, lookback, long_k, short_k in specs:
        config = VolatilityBreakoutConfig(
            timeframe_minutes=timeframe,
            lookback_days=lookback,
            long_k=long_k,
            short_k=short_k,
            allow_long=True,
            allow_short=True,
            atr_period=14,
            trend_ema_period=48,
            min_trend_alignment_atr=-999.0,
            min_volume_ratio=0.0,
            min_body_atr=0.0,
            min_directional_close_position=0.0,
            min_range_atr=0.0,
            max_range_atr=999.0,
            min_breakout_extension_atr=-999.0,
            max_breakout_extension_atr=999.0,
            max_volume_ratio=999.0,
            max_body_atr=999.0,
            max_trend_alignment_atr=999.0,
            # Build all valid daily crossings, then apply the researched cap.
            max_signals_per_symbol_day=24,
            stop_atr_multiple=1.0,
            take_profit_r=20.0,
            max_holding_minutes=720,
        )
        output[_family_key(config)] = config
    return output


def _neutral_portfolio(**changes: Any) -> PortfolioSearchConfig:
    config = PortfolioSearchConfig(
        risk_per_trade_pct=0.02,
        max_trade_risk_pct=0.10,
        max_open_positions=1,
        max_daily_trades=5,
        symbol_cooldown_minutes=120,
        max_notional_multiple=9.0,
        hard_drawdown_stop_pct=0.60,
        compound=True,
        ranking_mode="quality_desc",
        long_risk_multiplier=1.0,
        short_risk_multiplier=1.0,
    )
    return replace(config, **changes)


def _entry_signature(variant: V4Variant) -> str:
    signal = variant.signal.as_dict()
    for key in (
        "stop_atr_multiple",
        "take_profit_r",
        "breakeven_trigger_r",
        "trailing_activation_r",
        "trailing_atr_multiple",
        "fail_fast_minutes",
        "fail_fast_min_mfe_r",
        "fail_fast_max_current_r",
        "extended_holding_minutes",
        "extension_min_current_r",
        "extension_min_mfe_r",
        "profit_giveback_activation_r",
        "profit_giveback_r",
        "max_holding_minutes",
    ):
        signal.pop(key, None)
    portfolio = asdict(variant.portfolio)
    for key in ("risk_per_trade_pct", "long_risk_multiplier", "short_risk_multiplier", "compound"):
        portfolio.pop(key, None)
    return json.dumps(
        {
            "family": variant.family_key,
            "signal": signal,
            "portfolio": portfolio,
            "regime": asdict(variant.regime),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _deduplicate_variants(rows: Iterable[V4Variant]) -> list[V4Variant]:
    output: list[V4Variant] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(
            {
                "entry": _entry_signature(row),
                "signal_exit": {
                    key: value
                    for key, value in row.signal.as_dict().items()
                    if key
                    in {
                        "stop_atr_multiple",
                        "take_profit_r",
                        "fail_fast_minutes",
                        "fail_fast_min_mfe_r",
                        "fail_fast_max_current_r",
                        "max_holding_minutes",
                    }
                },
                "exit": asdict(row.exit),
                "risk": row.portfolio.risk_per_trade_pct,
                "long_multiplier": row.portfolio.long_risk_multiplier,
                "short_multiplier": row.portfolio.short_risk_multiplier,
                "compound": row.portfolio.compound,
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def build_entry_search_variants(
    families: dict[str, VolatilityBreakoutConfig],
    *,
    budget: int,
    seed: int,
) -> list[V4Variant]:
    rng = random.Random(seed)
    fixed_exit = ExitProtectionConfig(
        breakeven_trigger_r=1.5,
        profit_giveback_activation_r=3.0,
        profit_giveback_r=1.5,
        partial_take_profit_r=1.5,
        partial_take_profit_fraction=0.25,
        move_stop_to_breakeven_after_partial=False,
    )
    family_keys = tuple(families)
    rows: list[V4Variant] = []

    # Human-readable anchors are kept in every run, including the current shape.
    anchors = (
        (
            "current_shape_new_exit",
            "tf60_lb5_lk0.7_sk0.6",
            {},
            V4RegimeConfig(max_directional_btc_return_4h=0.02),
        ),
        (
            "trend_impulse",
            "tf60_lb5_lk0.7_sk0.6",
            {
                "min_trend_alignment_atr": 0.25,
                "min_volume_ratio": 1.0,
                "min_body_atr": 0.25,
                "min_directional_close_position": 0.65,
                "min_breakout_extension_atr": 0.0,
                "max_breakout_extension_atr": 2.0,
                "max_signals_per_symbol_day": 2,
            },
            V4RegimeConfig(
                min_directional_breadth=0.50,
                min_directional_symbol_return_4h=0.0,
                min_directional_symbol_efficiency_12h=0.0,
            ),
        ),
        (
            "non_exhausted_trend",
            "tf60_lb3_lk0.6_sk0.6",
            {
                "min_trend_alignment_atr": 0.0,
                "max_trend_alignment_atr": 3.0,
                "min_volume_ratio": 0.75,
                "max_volume_ratio": 4.0,
                "min_directional_close_position": 0.60,
                "max_body_atr": 2.5,
                "min_breakout_extension_atr": -0.10,
                "max_breakout_extension_atr": 1.25,
                "max_signals_per_symbol_day": 1,
            },
            V4RegimeConfig(
                min_directional_breadth=0.45,
                max_directional_breadth=0.85,
                min_market_efficiency_12h=0.10,
                min_regime_score=0.0,
            ),
        ),
    )
    for name, key, changes, regime in anchors:
        family = families[key]
        signal_changes = {
            "max_signals_per_symbol_day": 2,
            "stop_atr_multiple": 1.0,
            "take_profit_r": 20.0,
            "fail_fast_minutes": 120,
            "fail_fast_min_mfe_r": 0.10,
            "fail_fast_max_current_r": -0.50,
            "max_holding_minutes": 720,
        }
        signal_changes.update(changes)
        rows.append(
            V4Variant(
                name=name,
                family_key=key,
                signal=replace(family, **signal_changes),
                portfolio=_neutral_portfolio(),
                regime=regime,
                exit=fixed_exit,
            )
        )

    # Cover every structural family before randomized combinations. This avoids
    # a sparse random draw accidentally eliminating a viable family.
    for family_key, family in families.items():
        runner_signal = replace(
            family,
            max_signals_per_symbol_day=2,
            stop_atr_multiple=0.75,
            take_profit_r=60.0,
            fail_fast_minutes=120,
            fail_fast_min_mfe_r=0.10,
            fail_fast_max_current_r=-0.50,
            max_holding_minutes=960,
        )
        protected_signal = replace(
            family,
            max_signals_per_symbol_day=2,
            stop_atr_multiple=1.0,
            take_profit_r=20.0,
            fail_fast_minutes=120,
            fail_fast_min_mfe_r=0.10,
            fail_fast_max_current_r=-0.50,
            max_holding_minutes=720,
        )
        rows.extend(
            (
                V4Variant(
                    name=f"{family_key}_runner_unfiltered",
                    family_key=family_key,
                    signal=runner_signal,
                    portfolio=_neutral_portfolio(),
                    regime=V4RegimeConfig(),
                    exit=ExitProtectionConfig(),
                ),
                V4Variant(
                    name=f"{family_key}_protected_unfiltered",
                    family_key=family_key,
                    signal=protected_signal,
                    portfolio=_neutral_portfolio(),
                    regime=V4RegimeConfig(),
                    exit=fixed_exit,
                ),
                V4Variant(
                    name=f"{family_key}_runner_short",
                    family_key=family_key,
                    signal=replace(runner_signal, allow_long=False),
                    portfolio=_neutral_portfolio(),
                    regime=V4RegimeConfig(),
                    exit=ExitProtectionConfig(),
                ),
                V4Variant(
                    name=f"{family_key}_runner_quality",
                    family_key=family_key,
                    signal=replace(
                        runner_signal,
                        min_trend_alignment_atr=0.0,
                        min_volume_ratio=0.75,
                        min_directional_close_position=0.60,
                        max_breakout_extension_atr=2.0,
                    ),
                    portfolio=_neutral_portfolio(),
                    regime=V4RegimeConfig(min_directional_breadth=0.45),
                    exit=ExitProtectionConfig(),
                ),
            )
        )

    side_choices = (
        (True, True),
        (True, True),
        (True, True),
        (True, False),
        (False, True),
    )
    for number in range(max(0, budget - len(rows))):
        family_key = rng.choice(family_keys)
        family = families[family_key]
        allow_long, allow_short = rng.choice(side_choices)
        minimum_trend = rng.choice((-999.0, -999.0, -0.50, 0.0, 0.25, 0.50, 0.75))
        maximum_trend = rng.choice((2.0, 3.0, 5.0, 999.0, 999.0, 999.0))
        if maximum_trend < minimum_trend:
            maximum_trend = 999.0
        minimum_volume = rng.choice((0.0, 0.0, 0.0, 0.75, 1.0, 1.25, 1.50))
        maximum_volume = rng.choice((2.5, 4.0, 8.0, 999.0, 999.0, 999.0))
        if maximum_volume < minimum_volume:
            maximum_volume = 999.0
        minimum_body = rng.choice((0.0, 0.0, 0.0, 0.20, 0.40, 0.70))
        maximum_body = rng.choice((1.5, 2.5, 4.0, 999.0, 999.0, 999.0))
        if maximum_body < minimum_body:
            maximum_body = 999.0
        minimum_range = rng.choice((0.0, 0.0, 0.0, 1.0, 2.0, 3.0))
        maximum_range = rng.choice((4.0, 6.0, 8.0, 12.0, 999.0, 999.0, 999.0))
        if maximum_range < minimum_range:
            maximum_range = 999.0
        minimum_extension = rng.choice((-999.0, -999.0, -0.25, 0.0, 0.10, 0.25))
        maximum_extension = rng.choice((0.75, 1.25, 2.0, 4.0, 999.0, 999.0, 999.0))
        if maximum_extension < minimum_extension:
            maximum_extension = 999.0

        # Defaults appear repeatedly so the random budget does not force every gate on.
        regime = V4RegimeConfig(
            min_directional_btc_return_4h=rng.choice((-999.0, -999.0, -999.0, -0.005, 0.0, 0.0025)),
            max_directional_btc_return_4h=rng.choice((0.015, 0.03, 0.06, 999.0, 999.0, 999.0, 999.0)),
            min_directional_eth_return_4h=rng.choice((-999.0, -999.0, -999.0, -0.005, 0.0, 0.0025)),
            max_directional_eth_return_4h=rng.choice((0.02, 0.04, 0.08, 999.0, 999.0, 999.0, 999.0)),
            min_directional_breadth=rng.choice((0.0, 0.0, 0.0, 0.0, 0.40, 0.50, 0.60)),
            max_directional_breadth=rng.choice((0.75, 0.85, 0.95, 1.0, 1.0, 1.0, 1.0)),
            min_directional_breadth_change_4h=rng.choice((-999.0, -999.0, -999.0, -0.05, 0.0, 0.05)),
            min_directional_symbol_return_4h=rng.choice((-999.0, -999.0, -999.0, -0.01, 0.0, 0.005)),
            max_directional_symbol_return_4h=rng.choice((0.03, 0.06, 0.12, 999.0, 999.0, 999.0, 999.0)),
            min_directional_symbol_efficiency_12h=rng.choice((-999.0, -999.0, -999.0, -999.0, -0.10, 0.0, 0.15)),
            min_market_efficiency_12h=rng.choice((0.0, 0.0, 0.0, 0.0, 0.10, 0.15, 0.20)),
            min_directional_symbol_ema55_atr=rng.choice((-999.0, -999.0, -999.0, -0.50, 0.0, 0.50)),
            max_directional_symbol_ema55_atr=rng.choice((2.0, 3.0, 5.0, 999.0, 999.0, 999.0, 999.0)),
            min_regime_score=rng.choice((-999.0, -999.0, -999.0, -999.0, -0.25, 0.0, 0.25)),
            max_regime_score=rng.choice((1.0, 1.5, 2.0, 999.0, 999.0, 999.0, 999.0)),
        )
        if (
            regime.min_directional_breadth > regime.max_directional_breadth
            or regime.min_directional_btc_return_4h > regime.max_directional_btc_return_4h
            or regime.min_directional_eth_return_4h > regime.max_directional_eth_return_4h
            or regime.min_directional_symbol_return_4h > regime.max_directional_symbol_return_4h
            or regime.min_directional_symbol_ema55_atr > regime.max_directional_symbol_ema55_atr
            or regime.min_regime_score > regime.max_regime_score
        ):
            continue
        signal = replace(
            family,
            allow_long=allow_long,
            allow_short=allow_short,
            min_trend_alignment_atr=minimum_trend,
            max_trend_alignment_atr=maximum_trend,
            min_volume_ratio=minimum_volume,
            max_volume_ratio=maximum_volume,
            min_body_atr=minimum_body,
            max_body_atr=maximum_body,
            min_directional_close_position=rng.choice((0.0, 0.0, 0.50, 0.60, 0.70, 0.80)),
            min_range_atr=minimum_range,
            max_range_atr=maximum_range,
            min_breakout_extension_atr=minimum_extension,
            max_breakout_extension_atr=maximum_extension,
            max_signals_per_symbol_day=rng.choice((1, 1, 2, 3)),
            stop_atr_multiple=1.0,
            take_profit_r=20.0,
            fail_fast_minutes=120,
            fail_fast_min_mfe_r=0.10,
            fail_fast_max_current_r=-0.50,
            max_holding_minutes=720,
        )
        use_runner = rng.random() < 0.45
        if use_runner:
            signal = replace(
                signal,
                stop_atr_multiple=0.75,
                take_profit_r=60.0,
                max_holding_minutes=960,
            )
            stage_exit = ExitProtectionConfig()
        else:
            stage_exit = fixed_exit
        rows.append(
            V4Variant(
                name=f"entry_{number + 1:04d}",
                family_key=family_key,
                signal=signal,
                portfolio=_neutral_portfolio(
                    ranking_mode=rng.choice(
                        (
                            "quality_desc",
                            "quality_desc",
                            "trend_alignment_desc",
                            "breakout_extension_desc",
                            "directional_breadth_desc",
                            "directional_btc_4h_desc",
                        )
                    )
                ),
                regime=regime,
                exit=stage_exit,
            )
        )
    return _deduplicate_variants(rows)


def build_exit_search_specs(*, budget: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    specs: list[dict[str, Any]] = [
        {
            "name": "runner_no_overlay",
            "stop": 0.75,
            "tp": 60.0,
            "hold": 960,
            "fail_fast": (120, 0.10, -0.50),
            "exit": ExitProtectionConfig(),
        },
        {
            "name": "partial1_be1_runner",
            "stop": 1.0,
            "tp": 20.0,
            "hold": 720,
            "fail_fast": (120, 0.10, -0.50),
            "exit": ExitProtectionConfig(
                breakeven_trigger_r=1.0,
                partial_take_profit_r=1.0,
                partial_take_profit_fraction=0.25,
            ),
        },
        {
            "name": "partial1p5_giveback",
            "stop": 1.0,
            "tp": 20.0,
            "hold": 960,
            "fail_fast": (240, 0.25, -0.25),
            "exit": ExitProtectionConfig(
                breakeven_trigger_r=1.5,
                profit_giveback_activation_r=2.0,
                profit_giveback_r=1.0,
                partial_take_profit_r=1.5,
                partial_take_profit_fraction=0.25,
            ),
        },
    ]
    fail_fast_choices = (
        (0, 0.0, 0.0),
        (120, 0.10, -0.50),
        (120, 0.25, -0.25),
        (240, 0.25, -0.25),
        (360, 0.50, 0.0),
    )
    overlay_modes = (
        "none",
        "breakeven",
        "giveback",
        "partial",
        "partial_be",
        "partial_giveback",
    )
    while len(specs) < budget:
        stop = rng.choice((0.60, 0.75, 1.0, 1.25, 1.50))
        tp = rng.choice((3.0, 5.0, 8.0, 12.0, 20.0, 60.0))
        hold = rng.choice((240, 480, 720, 960, 1440))
        mode = rng.choice(overlay_modes)
        breakeven = 0.0
        giveback_activation = 0.0
        giveback_r = 0.0
        partial_r = 0.0
        partial_fraction = 0.0
        move_after_partial = False
        if mode == "breakeven":
            breakeven = rng.choice((0.75, 1.0, 1.5, 2.0))
        elif mode == "giveback":
            giveback_activation = rng.choice((1.0, 1.5, 2.0, 3.0))
            giveback_r = rng.choice((0.50, 0.75, 1.0, 1.50))
        elif mode in {"partial", "partial_be", "partial_giveback"}:
            partial_r = rng.choice((0.75, 1.0, 1.5, 2.0))
            partial_fraction = rng.choice((0.25, 0.33, 0.50))
            if partial_r >= tp:
                continue
            if mode == "partial_be":
                move_after_partial = True
            if mode == "partial_giveback":
                giveback_activation = rng.choice((1.5, 2.0, 3.0))
                giveback_r = rng.choice((0.50, 1.0, 1.50))
        exit_config = ExitProtectionConfig(
            breakeven_trigger_r=breakeven,
            profit_giveback_activation_r=giveback_activation,
            profit_giveback_r=giveback_r,
            partial_take_profit_r=partial_r,
            partial_take_profit_fraction=partial_fraction,
            move_stop_to_breakeven_after_partial=move_after_partial,
        )
        exit_config.validate()
        specs.append(
            {
                "name": f"exit_{len(specs) + 1:03d}",
                "stop": stop,
                "tp": tp,
                "hold": hold,
                "fail_fast": rng.choice(fail_fast_choices),
                "exit": exit_config,
            }
        )
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        key = json.dumps(
            {
                "stop": spec["stop"],
                "tp": spec["tp"],
                "hold": spec["hold"],
                "fail_fast": spec["fail_fast"],
                "exit": asdict(spec["exit"]),
            },
            sort_keys=True,
        )
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def apply_exit_spec(variant: V4Variant, spec: dict[str, Any], suffix: str) -> V4Variant:
    minutes, minimum_mfe, maximum_current = spec["fail_fast"]
    signal = replace(
        variant.signal,
        stop_atr_multiple=float(spec["stop"]),
        take_profit_r=float(spec["tp"]),
        breakeven_trigger_r=0.0,
        trailing_activation_r=0.0,
        trailing_atr_multiple=0.0,
        fail_fast_minutes=int(minutes),
        fail_fast_min_mfe_r=float(minimum_mfe),
        fail_fast_max_current_r=float(maximum_current),
        extended_holding_minutes=0,
        extension_min_current_r=0.0,
        extension_min_mfe_r=0.0,
        profit_giveback_activation_r=0.0,
        profit_giveback_r=0.0,
        max_holding_minutes=int(spec["hold"]),
    )
    return replace(
        variant,
        name=f"{variant.name}__{suffix}",
        signal=signal,
        exit=spec["exit"],
    )


def _public_variant(row: dict[str, Any]) -> dict[str, Any]:
    variant: V4Variant = row["variant"]
    return {
        "name": variant.name,
        "family_key": variant.family_key,
        "signal": variant.signal.as_dict(),
        "portfolio": asdict(variant.portfolio),
        "regime": asdict(variant.regime),
        "exit": asdict(variant.exit),
        "result": compact_summary(row["result"]),
        "fold_quality": row["quality"],
    }


def _champion_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    result = row["result"]
    eligible = result["trade_count"] >= 15 and not result["hard_drawdown_stopped"]
    return (
        eligible,
        result["net_profit"],
        result["profit_factor"],
        -result["max_drawdown_pct"],
        result["trade_count"],
    )


def _robust_pool(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    strict = [
        row
        for row in rows
        if row["result"]["trade_count"] >= 20
        and not row["result"]["hard_drawdown_stopped"]
        and row["result"]["profit_factor"] > 1.10
        and row["result"]["max_drawdown_pct"] <= 0.40
        and row["quality"]["all_folds_positive"]
        and row["result"]["top5_profit_contribution"] <= 0.80
    ]
    if strict:
        return strict
    relaxed = [
        row
        for row in rows
        if row["result"]["trade_count"] >= 15
        and not row["result"]["hard_drawdown_stopped"]
        and row["result"]["profit_factor"] > 1.0
        and row["result"]["max_drawdown_pct"] <= 0.50
        and row["quality"]["positive_folds"] >= 2
        and row["quality"]["fold_net_profit"][-1] > 0.0
    ]
    return relaxed or list(rows)


def _select_champion(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=_champion_sort_key)


def _select_robust(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pool = _robust_pool(rows)
    return max(
        pool,
        key=lambda row: (
            row["result"]["net_profit"],
            row["quality"]["minimum_fold_net_profit"],
            row["result"]["profit_factor"],
            -row["result"]["max_drawdown_pct"],
        ),
    )


def _top_entry_rows(rows: Sequence[dict[str, Any]], count_each: int) -> list[dict[str, Any]]:
    champion_rows = sorted(rows, key=_champion_sort_key, reverse=True)[:count_each]
    robust_rows = sorted(
        _robust_pool(rows),
        key=lambda row: (
            row["result"]["net_profit"],
            row["quality"]["minimum_fold_net_profit"],
        ),
        reverse=True,
    )[:count_each]
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in (*champion_rows, *robust_rows):
        key = _entry_signature(row["variant"])
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def _run_variants(
    variants: Sequence[V4Variant],
    raw_by_family: dict[str, dict[int, list[Candidate]]],
    context: dict[int, dict[str, V4MarketSnapshot]],
    symbols: tuple[str, ...],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
    fold_boundaries: Sequence[tuple[datetime, datetime]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    candidate_cache: dict[str, dict[int, list[Candidate]]] = {}
    output: list[dict[str, Any]] = []
    for number, variant in enumerate(variants, start=1):
        signature = _entry_signature(variant)
        candidates = candidate_cache.get(signature)
        if candidates is None:
            candidates = filter_candidates_v4(
                raw_by_family[variant.family_key],
                variant.signal,
                variant.regime,
                context,
            )
            candidate_cache[signature] = candidates
        result = simulate_exit_protected_portfolio(
            candidates,
            symbols,
            execution_data,
            rules,
            variant.signal,
            variant.portfolio,
            variant.exit,
            execution,
            start,
            end,
            initial_equity,
        )
        output.append(
            {
                "variant": variant,
                "candidates": candidates,
                "result": result,
                "quality": result_quality(result, fold_boundaries),
            }
        )
        if number == 1 or number % 25 == 0 or number == len(variants):
            print(
                f"{label} {number}/{len(variants)} {variant.name} "
                f"trades={result['trade_count']} net={result['net_profit']:+.2f} "
                f"pf={result['profit_factor']:.3f} dd={result['max_drawdown_pct']:.2%}",
                flush=True,
            )
    return output


def _risk_refinement_variants(rows: Sequence[dict[str, Any]]) -> list[V4Variant]:
    seeds = (_select_champion(rows)["variant"], _select_robust(rows)["variant"])
    output: list[V4Variant] = []
    for seed_number, seed in enumerate(seeds, start=1):
        for risk in (0.01, 0.02, 0.03, 0.044):
            for long_multiplier, short_multiplier in (
                (1.0, 1.0),
                (1.30, 1.0),
                (1.0, 1.30),
                (1.20, 0.80),
                (0.80, 1.20),
            ):
                for ranking in (
                    "quality_desc",
                    "trend_alignment_desc",
                    "directional_breadth_desc",
                    "directional_btc_4h_desc",
                ):
                    output.append(
                        replace(
                            seed,
                            name=(
                                f"risk_seed{seed_number}_r{risk:g}_"
                                f"l{long_multiplier:g}_s{short_multiplier:g}_{ranking}"
                            ),
                            portfolio=replace(
                                seed.portfolio,
                                risk_per_trade_pct=risk,
                                long_risk_multiplier=long_multiplier,
                                short_risk_multiplier=short_multiplier,
                                ranking_mode=ranking,
                            ),
                        )
                    )
    return _deduplicate_variants(output)


def _variant_from_current_config(
    path: str | Path,
) -> tuple[V4Variant, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    signal = VolatilityBreakoutConfig(**payload["balanced_signal"])
    source_portfolio = PortfolioSearchConfig(**payload["balanced_portfolio"])
    regime = V4RegimeConfig(
        min_directional_btc_return_4h=source_portfolio.min_directional_btc_return_4h,
        max_directional_btc_return_4h=source_portfolio.max_directional_btc_return_4h,
        min_directional_eth_return_4h=source_portfolio.min_directional_eth_return_4h,
        max_directional_eth_return_4h=source_portfolio.max_directional_eth_return_4h,
        min_directional_breadth=source_portfolio.min_directional_breadth,
        max_directional_breadth=source_portfolio.max_directional_breadth,
    )
    portfolio = replace(
        source_portfolio,
        min_directional_btc_return_4h=-999.0,
        max_directional_btc_return_4h=999.0,
        min_directional_eth_return_4h=-999.0,
        max_directional_eth_return_4h=999.0,
        min_directional_breadth=0.0,
        max_directional_breadth=1.0,
    )
    return (
        V4Variant(
            name="current_v2_balanced_baseline",
            family_key=_family_key(signal),
            signal=signal,
            portfolio=portfolio,
            regime=regime,
            exit=ExitProtectionConfig(),
        ),
        payload,
    )


def _simulate_stress_suite(
    row: dict[str, Any],
    symbols: tuple[str, ...],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
) -> dict[str, Any]:
    variant: V4Variant = row["variant"]
    candidates = row["candidates"]

    def run(
        candidate_override: dict[int, list[Candidate]],
        portfolio_override: PortfolioSearchConfig,
        execution_override: BacktestExecutionConfig,
    ) -> dict[str, Any]:
        result = simulate_exit_protected_portfolio(
            candidate_override,
            symbols,
            execution_data,
            rules,
            variant.signal,
            portfolio_override,
            variant.exit,
            execution_override,
            start,
            end,
            initial_equity,
        )
        return compact_summary(result)

    stressed_execution = replace(
        execution,
        market_slippage_bps=execution.market_slippage_bps * 1.5,
        stop_slippage_bps=execution.stop_slippage_bps * 1.5,
        take_profit_slippage_bps=execution.take_profit_slippage_bps * 1.5,
        taker_fee_rate=execution.taker_fee_rate * 1.5,
    )
    return {
        "fixed_risk_no_compounding": run(
            candidates, replace(variant.portfolio, compound=False), execution
        ),
        "entry_delay_1m": run(
            _shift_candidates(candidates, 1, execution_data),
            variant.portfolio,
            execution,
        ),
        "cost_1p5x": run(candidates, variant.portfolio, stressed_execution),
    }


def _write_trades_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "strategy",
        "event_id",
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "quantity",
        "net_pnl",
        "pnl_r",
        "mfe_r",
        "mae_r",
        "exit_reason",
        "partial_exit_count",
        "partial_realized_net_pnl",
        "fee",
        "slippage",
        "funding",
    )
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _metric_line(label: str, result: dict[str, Any]) -> str:
    pf = result["profit_factor"]
    pf_text = "inf" if not math.isfinite(pf) else f"{pf:.3f}"
    return (
        f"- {label}: `{result['trade_count']}` trades, "
        f"`{result['net_profit']:+.2f}U`, PF `{pf_text}`, "
        f"win `{result['win_rate']:.2%}`, DD `{result['max_drawdown_pct']:.2%}`."
    )


def _write_summary(path: str | Path, report: dict[str, Any]) -> None:
    baseline = report["baseline_current_v2"]["standalone"]
    grid = report["unchanged_grid"]["standalone"]
    champion = report["finalists"]["historical_profit_champion"]
    robust = report["finalists"]["robust_candidate"]
    lines = [
        "# Volatility Breakout v4 — 近三个月独立重设计回测",
        "",
        f"- 区间：`{report['period']['start']}` 至 `{report['period']['end']}`（右端不含）。",
        f"- 初始权益：`{report['initial_equity']:.2f}U`；50 币；全局最多两仓。",
        "- 信号只使用已收盘数据，下一根 1m 开盘成交；同根冲突止损优先。",
        "- 已计手续费、滑点、Funding；逐分钟盯市计算回撤。",
        "- APT / Dynamic Trend Grid 的配置与引擎未修改。",
        "- v4 是历史研究版本，未写入 GUI，也不是未见样本保证。",
        "",
        "## 核心结果",
        "",
        _metric_line("当前 v2 Breakout（同区间基线）", baseline),
        _metric_line("冻结 Grid 单独", grid),
        _metric_line("v4 纯历史利润冠军单独", champion["standalone"]),
        _metric_line("v4 稳定性候选单独", robust["standalone"]),
        _metric_line("当前 v2 + Grid，共享两仓", report["baseline_current_v2"]["combined"]),
        _metric_line("利润冠军 + Grid，共享两仓", champion["combined"]),
        _metric_line("稳定性候选 + Grid，共享两仓", robust["combined"]),
        "",
        "## 选择说明",
        "",
        "- 利润冠军：在本次受限参数预算中，以三个月全期净利润最大为主，属于样本内冠军。",
        "- 稳定性候选：要求交易数量、PF、回撤、30 日分段和利润集中度同时过门槛，再在合格项中取净利润最大。",
        "- 三个 30 日分段仍来自同一历史区间，因此只能降低过拟合风险，不能当作真正未来 OOS。",
        "",
        "## Grid 冻结校验",
        "",
        f"- 配置 SHA256（前/后）：`{report['unchanged_grid']['sha256_before']}` / `{report['unchanged_grid']['sha256_after']}`。",
        f"- 配置不变：`{report['unchanged_grid']['hash_unchanged']}`。",
        "",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v4_research(args: argparse.Namespace) -> dict[str, Any]:
    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    if start >= end:
        raise ValueError("start must be before end")
    symbols = tuple(UNIVERSE_50)
    fold_boundaries = (
        (start, start + timedelta(days=30)),
        (start + timedelta(days=30), start + timedelta(days=61)),
        (start + timedelta(days=61), end),
    )
    if fold_boundaries[-1][1] != end:
        raise ValueError("default v4 research expects a 91-day three-fold period")

    print("loading stitched 1m price path and full-cost inputs", flush=True)
    signal_data, execution_data, rules, execution, metadata = load_v4_runtime_inputs(
        symbols,
        args.one_minute_roots,
        args.funding_roots,
        args.cost_config,
        start,
        end,
    )
    print(
        f"data ready symbols={len(symbols)} min_coverage={metadata['minimum_coverage_ratio']:.6f} "
        f"funding_missing={len(metadata['funding_missing_symbols'])}",
        flush=True,
    )
    context = build_v4_market_context(symbols, signal_data)
    print(f"market context snapshots={len(context)}", flush=True)

    current_variant, current_payload = _variant_from_current_config(args.current_breakout_config)
    print("building current-v2 baseline candidates", flush=True)
    current_raw = enrich_candidates_v4(
        build_candidates(
            symbols,
            signal_data,
            execution_data,
            current_variant.signal,
            start,
            end,
        ),
        context,
    )
    current_candidates = filter_candidates_v4(
        current_raw,
        current_variant.signal,
        current_variant.regime,
        context,
    )
    current_result = simulate_exit_protected_portfolio(
        current_candidates,
        symbols,
        execution_data,
        rules,
        current_variant.signal,
        current_variant.portfolio,
        current_variant.exit,
        execution,
        start,
        end,
        args.initial_equity,
    )
    current_quality = result_quality(current_result, fold_boundaries)
    print(
        f"baseline trades={current_result['trade_count']} net={current_result['net_profit']:+.2f} "
        f"pf={current_result['profit_factor']:.3f} dd={current_result['max_drawdown_pct']:.2%}",
        flush=True,
    )

    families = build_v4_families()
    raw_by_family: dict[str, dict[int, list[Candidate]]] = {}
    for number, (key, family) in enumerate(families.items(), start=1):
        raw = build_candidates(
            symbols, signal_data, execution_data, family, start, end
        )
        raw_by_family[key] = enrich_candidates_v4(raw, context)
        print(
            f"raw family {number}/{len(families)} {key} "
            f"candidates={sum(len(rows) for rows in raw.values())}",
            flush=True,
        )

    entry_variants = build_entry_search_variants(
        families, budget=args.entry_budget, seed=args.seed
    )
    stage1 = _run_variants(
        entry_variants,
        raw_by_family,
        context,
        symbols,
        execution_data,
        rules,
        execution,
        start,
        end,
        args.initial_equity,
        fold_boundaries,
        label="entry-regime",
    )
    top_entries = _top_entry_rows(stage1, args.top_entry_count)
    exit_specs = build_exit_search_specs(budget=args.exit_budget, seed=args.seed + 1)
    exit_variants = [
        apply_exit_spec(row["variant"], spec, spec["name"])
        for row in top_entries
        for spec in exit_specs
    ]
    stage2 = _run_variants(
        _deduplicate_variants(exit_variants),
        raw_by_family,
        context,
        symbols,
        execution_data,
        rules,
        execution,
        start,
        end,
        args.initial_equity,
        fold_boundaries,
        label="exit",
    )
    stage3_variants = _risk_refinement_variants(stage2)
    stage3 = _run_variants(
        stage3_variants,
        raw_by_family,
        context,
        symbols,
        execution_data,
        rules,
        execution,
        start,
        end,
        args.initial_equity,
        fold_boundaries,
        label="risk-ranking",
    )
    final_pool = [*stage1, *stage2, *stage3]
    champion_row = _select_champion(final_pool)
    robust_row = _select_robust(final_pool)
    print(
        f"selected champion={champion_row['variant'].name} "
        f"net={champion_row['result']['net_profit']:+.2f}; "
        f"robust={robust_row['variant'].name} "
        f"net={robust_row['result']['net_profit']:+.2f}",
        flush=True,
    )

    grid_path = Path(args.grid_config)
    grid_hash_before = sha256_file(grid_path)
    grid_payload = json.loads(grid_path.read_text(encoding="utf-8"))
    grid_signal = TrendGridConfig(
        **grid_payload.get("validation_selected_signal", grid_payload["signal"])
    )
    grid_signal.validate()
    grid_portfolio = GridPortfolioConfig(
        **grid_payload.get("validation_selected_portfolio", grid_payload["portfolio"])
    )
    print("building unchanged Grid timeline once", flush=True)
    grid_candidates, grid_snapshots = build_grid_research_timeline(
        symbols,
        signal_data,
        execution_data,
        grid_signal,
        start,
        end,
    )
    standalone_grid = simulate_grid_portfolio(
        grid_candidates,
        grid_snapshots,
        symbols,
        execution_data,
        rules,
        grid_signal,
        grid_portfolio,
        execution,
        start,
        end,
        args.initial_equity,
    )
    combined_config = CombinedPortfolioConfig(
        max_open_positions=2,
        max_gross_notional_multiple=9.0,
        hard_drawdown_stop_pct=0.60,
        allow_same_symbol_across_strategies=False,
        entry_priority=(BREAKOUT_KEY, GRID_KEY),
    )

    def combined_for(row: dict[str, Any]) -> dict[str, Any]:
        variant: V4Variant = row["variant"]
        return simulate_combined_v4_portfolio(
            row["candidates"],
            grid_candidates,
            grid_snapshots,
            symbols,
            execution_data,
            rules,
            variant.signal,
            variant.portfolio,
            variant.exit,
            grid_signal,
            grid_portfolio,
            combined_config,
            execution,
            start,
            end,
            args.initial_equity,
        )

    baseline_row = {
        "variant": current_variant,
        "candidates": current_candidates,
        "result": current_result,
        "quality": current_quality,
    }
    print("running exact shared-account combinations", flush=True)
    baseline_combined = combined_for(baseline_row)
    champion_combined = combined_for(champion_row)
    robust_combined = combined_for(robust_row)
    reversed_combined_config = replace(
        combined_config,
        entry_priority=(GRID_KEY, BREAKOUT_KEY),
    )
    robust_variant: V4Variant = robust_row["variant"]
    robust_reversed = simulate_combined_v4_portfolio(
        robust_row["candidates"],
        grid_candidates,
        grid_snapshots,
        symbols,
        execution_data,
        rules,
        robust_variant.signal,
        robust_variant.portfolio,
        robust_variant.exit,
        grid_signal,
        grid_portfolio,
        reversed_combined_config,
        execution,
        start,
        end,
        args.initial_equity,
    )

    champion_stress = _simulate_stress_suite(
        champion_row,
        symbols,
        execution_data,
        rules,
        execution,
        start,
        end,
        args.initial_equity,
    )
    robust_stress = _simulate_stress_suite(
        robust_row,
        symbols,
        execution_data,
        rules,
        execution,
        start,
        end,
        args.initial_equity,
    )
    grid_hash_after = sha256_file(grid_path)

    report = {
        "strategy_name": V4_RESEARCH_NAME,
        "strategy_version": V4_RESEARCH_VERSION,
        "research_status": (
            "historical_bounded_optimization_with_30d_stability_folds_not_untouched_future_oos"
        ),
        "gui_modified": False,
        "grid_modified": False,
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "folds": [
            {"start": fold_start.isoformat(), "end": fold_end.isoformat()}
            for fold_start, fold_end in fold_boundaries
        ],
        "initial_equity": args.initial_equity,
        "symbols": list(symbols),
        "data": {
            "one_minute_roots": list(args.one_minute_roots),
            "funding_roots": list(args.funding_roots),
            "coverage": metadata,
        },
        "cost_model": {
            "source_config": args.cost_config,
            "source_config_sha256": sha256_file(args.cost_config),
            "mode": execution.mode,
            "market_slippage_bps": execution.market_slippage_bps,
            "stop_slippage_bps": execution.stop_slippage_bps,
            "take_profit_slippage_bps": execution.take_profit_slippage_bps,
            "maker_fee_rate": execution.maker_fee_rate,
            "taker_fee_rate": execution.taker_fee_rate,
            "funding_enabled": execution.funding_enabled,
        },
        "execution_rules": {
            "canonical_price_source": "official Binance 1m CSV stitched and deduplicated",
            "signal_bars": "15m derived from the same 1m path; 30m/60m closed-bar resampling",
            "entry": "next_1m_open_after_signal_close",
            "same_bar_conflict": "gap_stop_then_stop_before_target_or_partial",
            "protective_stop_update": "after_bar_effective_next_minute",
            "old_positions_exit_before_new_entries": True,
            "mark_to_market_drawdown_each_minute": True,
            "shared_account_max_positions": 2,
            "aggregate_committed_notional_cap": 9.0,
        },
        "selection": {
            "entry_regime_budget": len(stage1),
            "exit_budget_per_selected_entry": len(exit_specs),
            "selected_entry_count": len(top_entries),
            "exit_variant_count": len(stage2),
            "risk_ranking_variant_count": len(stage3),
            "random_seed": args.seed,
            "historical_champion_rule": "maximum full-period net profit with >=15 trades and no hard stop",
            "robust_rule": (
                ">=20 trades, PF>1.10, DD<=40%, all 30d folds positive, "
                "top5 contribution<=80%; relaxed rule disclosed by result if no strict candidate"
            ),
            "robust_strict_pool_count": len(
                [row for row in final_pool if row in _robust_pool(final_pool)]
            ) if _robust_pool(final_pool) else 0,
        },
        "baseline_current_v2": {
            "source_config": args.current_breakout_config,
            "source_sha256": sha256_file(args.current_breakout_config),
            "variant": _public_variant(baseline_row),
            "standalone": current_result,
            "combined": baseline_combined,
        },
        "unchanged_grid": {
            "source_config": args.grid_config,
            "sha256_before": grid_hash_before,
            "sha256_after": grid_hash_after,
            "hash_unchanged": grid_hash_before == grid_hash_after,
            "signal": grid_signal.as_dict(),
            "portfolio": asdict(grid_portfolio),
            "standalone": standalone_grid,
        },
        "finalists": {
            "historical_profit_champion": {
                "variant": _public_variant(champion_row),
                "standalone": champion_row["result"],
                "combined": champion_combined,
                "stress": champion_stress,
            },
            "robust_candidate": {
                "variant": _public_variant(robust_row),
                "standalone": robust_row["result"],
                "combined": robust_combined,
                "priority_reversed_combined": robust_reversed,
                "stress": robust_stress,
            },
        },
        "search_leaderboard": {
            "entry_regime_top20": [
                _public_variant(row)
                for row in sorted(stage1, key=_champion_sort_key, reverse=True)[:20]
            ],
            "exit_top20": [
                _public_variant(row)
                for row in sorted(stage2, key=_champion_sort_key, reverse=True)[:20]
            ],
            "risk_ranking_top20": [
                _public_variant(row)
                for row in sorted(stage3, key=_champion_sort_key, reverse=True)[:20]
            ],
        },
    }
    _write_json(args.output, report)
    _write_summary(args.summary, report)
    _write_trades_csv(
        args.trades,
        report["finalists"]["robust_candidate"]["combined"]["trades"],
    )

    champion_variant: V4Variant = champion_row["variant"]
    robust_variant = robust_row["variant"]
    config_payload = {
        "strategy_name": V4_RESEARCH_NAME,
        "status": "historical_research_not_live_not_gui",
        "period": report["period"],
        "symbols": list(symbols),
        "selected_profile": "robust_candidate",
        "robust_candidate": {
            "signal": robust_variant.signal.as_dict(),
            "portfolio": asdict(robust_variant.portfolio),
            "regime": asdict(robust_variant.regime),
            "exit": asdict(robust_variant.exit),
            "standalone": compact_summary(robust_row["result"]),
            "combined": {
                key: robust_combined[key]
                for key in (
                    "trade_count",
                    "net_profit",
                    "return_pct",
                    "profit_factor",
                    "win_rate",
                    "max_drawdown_pct",
                )
            },
        },
        "historical_profit_champion": {
            "signal": champion_variant.signal.as_dict(),
            "portfolio": asdict(champion_variant.portfolio),
            "regime": asdict(champion_variant.regime),
            "exit": asdict(champion_variant.exit),
            "standalone": compact_summary(champion_row["result"]),
            "combined": {
                key: champion_combined[key]
                for key in (
                    "trade_count",
                    "net_profit",
                    "return_pct",
                    "profit_factor",
                    "win_rate",
                    "max_drawdown_pct",
                )
            },
        },
        "unchanged_grid_reference": {
            "path": args.grid_config,
            "sha256": grid_hash_after,
        },
        "gui_modified": False,
    }
    _write_json(args.config_output, config_payload)
    artifacts = (
        args.output,
        args.summary,
        args.trades,
        args.config_output,
        "crypto_scalper/volatility_breakout_v4_research.py",
        "crypto_scalper/combined_volatility_trend_grid_v4_backtest.py",
        args.grid_config,
        args.current_breakout_config,
    )
    _write_json(
        args.manifest,
        {
            "strategy_name": V4_RESEARCH_NAME,
            "status": "independent_historical_research_grid_and_gui_preserved",
            "report": args.output,
            "summary": args.summary,
            "trades": args.trades,
            "config": args.config_output,
            "grid_modified": False,
            "gui_modified": False,
            "hashes": {
                str(path): sha256_file(path)
                for path in artifacts
                if Path(path).exists()
            },
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Research an independent live-like Volatility Breakout v4 while "
            "keeping Dynamic Trend Grid frozen"
        )
    )
    parser.add_argument("--start", default="2026-04-19T00:00:00")
    parser.add_argument("--end", default="2026-07-19T00:00:00")
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument(
        "--one-minute-roots",
        nargs="+",
        default=(
            "data/binance_1m_3m_to_20260622_top100",
            "data/binance_1m_v3_exit_holdout_20260522_20260719",
        ),
    )
    parser.add_argument(
        "--funding-roots",
        nargs="+",
        default=(
            "data/binance_funding_365d_top100",
            "data/binance_funding_v3_exit_holdout_20260612_20260719",
        ),
    )
    parser.add_argument(
        "--cost-config", default="config.gui.mtf-momentum-reset-stage21.json"
    )
    parser.add_argument(
        "--current-breakout-config",
        default="config.volatility-breakout.v2-optimized-50.json",
    )
    parser.add_argument(
        "--grid-config", default="config.trend-grid.v2-optimized-50.json"
    )
    parser.add_argument("--entry-budget", type=int, default=360)
    parser.add_argument("--exit-budget", type=int, default=48)
    parser.add_argument("--top-entry-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--output", default="reports/volatility_breakout_v4_regime_exit_3m.json"
    )
    parser.add_argument(
        "--summary", default="reports/volatility_breakout_v4_regime_exit_3m.md"
    )
    parser.add_argument(
        "--trades", default="reports/volatility_breakout_v4_regime_exit_3m_trades.csv"
    )
    parser.add_argument(
        "--config-output", default="config.volatility-breakout.v4-regime-exit-50.json"
    )
    parser.add_argument(
        "--manifest", default="config.volatility-breakout.v4-regime-exit-50-manifest.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_v4_research(args)
    concise = {
        "baseline_current_v2": compact_summary(
            report["baseline_current_v2"]["standalone"]
        ),
        "unchanged_grid": compact_grid_summary(
            report["unchanged_grid"]["standalone"]
        ),
        "historical_profit_champion": {
            "standalone": compact_summary(
                report["finalists"]["historical_profit_champion"]["standalone"]
            ),
            "combined": {
                key: report["finalists"]["historical_profit_champion"]["combined"][key]
                for key in (
                    "trade_count",
                    "net_profit",
                    "profit_factor",
                    "max_drawdown_pct",
                )
            },
        },
        "robust_candidate": {
            "standalone": compact_summary(
                report["finalists"]["robust_candidate"]["standalone"]
            ),
            "combined": {
                key: report["finalists"]["robust_candidate"]["combined"][key]
                for key in (
                    "trade_count",
                    "net_profit",
                    "profit_factor",
                    "max_drawdown_pct",
                )
            },
        },
    }
    print(json.dumps(_json_safe(concise), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

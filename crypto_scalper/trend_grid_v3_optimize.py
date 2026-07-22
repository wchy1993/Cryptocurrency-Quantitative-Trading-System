from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .models import Direction
from .trend_grid import TREND_GRID_STRATEGY_NAME, TrendGridConfig
from .trend_grid_optimize import (
    GridCandidate,
    GridPortfolioConfig,
    TrendGridSnapshot,
    _scaled_execution,
    _shift_candidates,
    build_grid_research_timeline,
    compact_grid_summary,
    sha256_file,
    simulate_grid_portfolio,
)
from .volatility_breakout_optimize import UNIVERSE_50, CompactSeries, minute_datetime
from .volatility_breakout_v4_research import (
    V4MarketSnapshot,
    build_v4_market_context,
    load_v4_runtime_inputs,
)


TREND_GRID_V3_RESEARCH_NAME = "dynamic_trend_following_grid_v3_regime_adaptive"


@dataclass(frozen=True)
class GridMarketOverlay:
    """Point-in-time market filter and cross-sectional ranking for Grid entries."""

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
    # Zero means the overlay preserves the signal engine's original daily cap.
    max_signals_per_symbol_day: int = 0
    ranking_mode: str = "quality_desc"

    def validate(self) -> None:
        if self.ranking_mode not in {
            "quality_desc",
            "alignment_desc",
            "slope_desc",
            "extension_asc",
            "regime_desc",
            "blended_desc",
        }:
            raise ValueError(f"unsupported Grid ranking mode: {self.ranking_mode}")
        if self.max_signals_per_symbol_day < 0:
            raise ValueError("max_signals_per_symbol_day cannot be negative")
        if self.min_directional_breadth > self.max_directional_breadth:
            raise ValueError("invalid directional breadth range")


def _context_minute(minute: int) -> int:
    return minute - minute % 60


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


def _passes_overlay(
    candidate: GridCandidate,
    snapshot: V4MarketSnapshot | None,
    overlay: GridMarketOverlay,
) -> bool:
    if snapshot is None:
        return overlay == GridMarketOverlay()
    direction = candidate.signal.direction
    side = float(direction.value)
    directional_breadth = (
        snapshot.breadth_above_ema21
        if direction == Direction.LONG
        else 1.0 - snapshot.breadth_above_ema21
    )
    return (
        overlay.min_directional_btc_return_4h
        <= side * snapshot.btc_return_4h
        <= overlay.max_directional_btc_return_4h
        and overlay.min_directional_eth_return_4h
        <= side * snapshot.eth_return_4h
        <= overlay.max_directional_eth_return_4h
        and overlay.min_directional_breadth
        <= directional_breadth
        <= overlay.max_directional_breadth
        and side * snapshot.breadth_change_4h
        >= overlay.min_directional_breadth_change_4h
        and overlay.min_directional_symbol_return_4h
        <= side * snapshot.symbol_return_4h
        <= overlay.max_directional_symbol_return_4h
        and side * snapshot.symbol_efficiency_12h
        >= overlay.min_directional_symbol_efficiency_12h
        and snapshot.market_efficiency_12h >= overlay.min_market_efficiency_12h
        and overlay.min_directional_symbol_ema55_atr
        <= side * snapshot.symbol_ema55_atr
        <= overlay.max_directional_symbol_ema55_atr
        and overlay.min_regime_score
        <= _regime_score(snapshot, direction)
        <= overlay.max_regime_score
    )


def _ranking_value(
    candidate: GridCandidate,
    snapshot: V4MarketSnapshot | None,
    mode: str,
) -> float:
    signal = candidate.signal
    directional_slope = signal.direction.value * signal.fast_slope_atr
    if mode == "alignment_desc":
        return signal.alignment_atr
    if mode == "slope_desc":
        return directional_slope
    if mode == "extension_asc":
        return -signal.extension_atr
    regime = _regime_score(snapshot, signal.direction) if snapshot is not None else 0.0
    if mode == "regime_desc":
        return regime
    if mode == "blended_desc":
        return (
            0.45 * signal.quality_score
            + 0.20 * min(signal.alignment_atr, 3.0)
            + 0.15 * min(max(directional_slope, -1.0), 2.0)
            + 0.20 * regime
        )
    return signal.quality_score


def apply_market_overlay(
    candidates: dict[int, list[GridCandidate]],
    context: dict[int, dict[str, V4MarketSnapshot]],
    overlay: GridMarketOverlay,
) -> dict[int, list[GridCandidate]]:
    overlay.validate()
    daily_counts: dict[tuple[str, str], int] = defaultdict(int)
    output: dict[int, list[GridCandidate]] = {}
    for minute in sorted(candidates):
        snapshots = context.get(_context_minute(minute), {})
        ranked: list[tuple[float, GridCandidate]] = []
        for candidate in candidates[minute]:
            snapshot = snapshots.get(candidate.signal.symbol)
            if not _passes_overlay(candidate, snapshot, overlay):
                continue
            day = minute_datetime(minute).date().isoformat()
            count_key = (candidate.signal.symbol, day)
            if (
                overlay.max_signals_per_symbol_day > 0
                and daily_counts[count_key] >= overlay.max_signals_per_symbol_day
            ):
                continue
            if overlay.max_signals_per_symbol_day > 0:
                daily_counts[count_key] += 1
            ranked.append((_ranking_value(candidate, snapshot, overlay.ranking_mode), candidate))
        if ranked:
            ranked.sort(key=lambda row: (-row[0], row[1].signal.symbol))
            output[minute] = [candidate for _, candidate in ranked]
    return output


class _TimelineCache:
    def __init__(self, maximum: int = 8) -> None:
        self.maximum = maximum
        self.values: OrderedDict[
            str,
            tuple[
                dict[int, list[GridCandidate]],
                dict[str, dict[int, TrendGridSnapshot]],
            ],
        ] = OrderedDict()

    @staticmethod
    def key(config: TrendGridConfig) -> str:
        fields = config.as_dict()
        for name in (
            "grid_spacing_atr",
            "grid_levels",
            "grid_target_spacing",
            "deeper_level_size_multiplier",
            "initial_entry_enabled",
            "hard_stop_atr_multiple",
            "hard_stop_slow_ema_buffer_atr",
            "regime_exit_mode",
            "regime_exit_confirm_bars",
            "max_campaign_minutes",
            "max_cycles_per_level",
            "max_total_entries",
            "pause_new_fills_on_fast_breach",
            "campaign_loss_limit_r",
            "campaign_take_profit_r",
            "profit_lock_activation_r",
            "profit_giveback_r",
        ):
            fields.pop(name, None)
        return json.dumps(fields, sort_keys=True, separators=(",", ":"))

    def get(
        self,
        config: TrendGridConfig,
        universe: tuple[str, ...],
        signal_data: dict[str, Any],
        execution_data: dict[str, CompactSeries],
        start: datetime,
        end: datetime,
    ) -> tuple[
        dict[int, list[GridCandidate]],
        dict[str, dict[int, TrendGridSnapshot]],
    ]:
        key = self.key(config)
        cached = self.values.pop(key, None)
        if cached is None:
            cached = build_grid_research_timeline(
                universe, signal_data, execution_data, config, start, end
            )
            if len(self.values) >= self.maximum:
                self.values.popitem(last=False)
        self.values[key] = cached
        return cached


def _compact_row(
    name: str,
    signal: TrendGridConfig,
    overlay: GridMarketOverlay,
    portfolio: GridPortfolioConfig,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "signal": signal.as_dict(),
        "market_overlay": asdict(overlay),
        "portfolio": asdict(portfolio),
        "metrics": compact_grid_summary(result),
    }


def _row_key(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "signal": row["signal"],
            "market_overlay": row["market_overlay"],
            "portfolio": row["portfolio"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _score(row: dict[str, Any], drawdown_cap: float, initial_equity: float) -> float:
    result = row["metrics"]
    net = float(result["net_profit"])
    profit_factor = float(result["profit_factor"])
    drawdown = float(result["max_drawdown_pct"])
    valid = (
        result["trade_count"] >= 18
        and net > 0.0
        and profit_factor > 1.0
        and drawdown <= drawdown_cap + 1e-12
        and not result["hard_drawdown_stopped"]
    )
    if not valid:
        return -1e9 + net - 100.0 * max(0.0, drawdown - drawdown_cap)
    growth = math.log1p(net / max(initial_equity, 1e-12))
    pf_term = math.log(max(1.0, min(profit_factor, 5.0)))
    fold_term = 0.10 * int(result["positive_months"])
    concentration = min(float(result["top5_profit_contribution"]), 5.0)
    return 3.0 * growth + 1.35 * pf_term + fold_term - 1.35 * drawdown - 0.04 * concentration


def _ranked(
    rows: Iterable[dict[str, Any]],
    drawdown_cap: float,
    initial_equity: float,
    limit: int,
) -> list[dict[str, Any]]:
    unique = {_row_key(row): row for row in rows}
    return sorted(
        unique.values(),
        key=lambda row: _score(row, drawdown_cap, initial_equity),
        reverse=True,
    )[:limit]


def _evaluate(
    name: str,
    signal: TrendGridConfig,
    overlay: GridMarketOverlay,
    portfolio: GridPortfolioConfig,
    universe: tuple[str, ...],
    signal_data: dict[str, Any],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, Any],
    execution: Any,
    context: dict[int, dict[str, V4MarketSnapshot]],
    timeline_cache: _TimelineCache,
    overlay_cache: dict[str, dict[int, list[GridCandidate]]],
    start: datetime,
    end: datetime,
    initial_equity: float,
) -> dict[str, Any]:
    signal.validate()
    raw_candidates, snapshots = timeline_cache.get(
        signal, universe, signal_data, execution_data, start, end
    )
    overlay_key = (
        _TimelineCache.key(signal)
        + "|"
        + json.dumps(asdict(overlay), sort_keys=True, separators=(",", ":"))
    )
    candidates = overlay_cache.get(overlay_key)
    if candidates is None:
        candidates = apply_market_overlay(raw_candidates, context, overlay)
        overlay_cache[overlay_key] = candidates
    result = simulate_grid_portfolio(
        candidates,
        snapshots,
        universe,
        execution_data,
        rules,
        signal,
        portfolio,
        execution,
        start,
        end,
        initial_equity,
    )
    return _compact_row(name, signal, overlay, portfolio, result)


def _log_stage(stage: str, number: int, total: int, row: dict[str, Any]) -> None:
    if number == 1 or number % 10 == 0 or number == total:
        result = row["metrics"]
        print(
            f"{stage} {number}/{total}: trades={result['trade_count']} "
            f"net={result['net_profit']:+.2f} PF={result['profit_factor']:.3f} "
            f"DD={result['max_drawdown_pct']:.2%}",
            flush=True,
        )


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


def _random_entry_variant(
    source: TrendGridConfig,
    rng: random.Random,
) -> TrendGridConfig:
    minimum_alignment = rng.choice((0.0, 0.15, 0.30, 0.50, 0.75, 1.0))
    maximum_alignment = rng.choice((1.5, 2.0, 2.5, 3.0, 4.0, 999.0))
    maximum_alignment = max(maximum_alignment, minimum_alignment + 0.25)
    minimum_volume = rng.choice((0.0, 0.5, 0.7, 0.9, 1.1))
    maximum_volume = rng.choice((1.5, 2.0, 3.0, 5.0, 999.0))
    maximum_volume = max(maximum_volume, minimum_volume + 0.1)
    return replace(
        source,
        min_fast_slope_atr=rng.choice((0.0, 0.02, 0.05, 0.08, 0.12)),
        min_slow_slope_atr=rng.choice((-0.05, -0.02, 0.0, 0.02, 0.05)),
        min_alignment_atr=minimum_alignment,
        max_alignment_atr=maximum_alignment,
        max_entry_extension_atr=rng.choice((0.40, 0.65, 0.90, 1.20, 1.60, 2.0)),
        pullback_touch_atr=rng.choice((0.10, 0.20, 0.30, 0.45, 0.60)),
        min_directional_close_position=rng.choice((0.40, 0.50, 0.58, 0.65, 0.72)),
        min_volume_ratio=minimum_volume,
        max_volume_ratio=maximum_volume,
        reentry_interval_bars=rng.choice((2, 4, 6, 8, 12)),
        max_signals_per_symbol_day=rng.choice((2, 3, 4, 6, 8)),
    )


def _random_overlay(rng: random.Random) -> GridMarketOverlay:
    min_breadth = rng.choice((0.25, 0.35, 0.42, 0.48, 0.52, 0.58))
    max_breadth = rng.choice((0.70, 0.80, 0.90, 1.0))
    max_breadth = max(max_breadth, min_breadth + 0.05)
    return GridMarketOverlay(
        min_directional_btc_return_4h=rng.choice((-999.0, -0.02, -0.01, -0.003, 0.0)),
        max_directional_btc_return_4h=rng.choice((0.01, 0.02, 0.03, 0.05, 999.0)),
        min_directional_eth_return_4h=rng.choice((-999.0, -0.025, -0.012, -0.004, 0.0)),
        max_directional_eth_return_4h=rng.choice((0.015, 0.03, 0.05, 999.0)),
        min_directional_breadth=min_breadth,
        max_directional_breadth=max_breadth,
        min_directional_breadth_change_4h=rng.choice((-999.0, -0.12, -0.06, -0.02, 0.0)),
        min_directional_symbol_return_4h=rng.choice((-999.0, -0.03, -0.015, -0.005, 0.0)),
        max_directional_symbol_return_4h=rng.choice((0.02, 0.04, 0.08, 999.0)),
        min_directional_symbol_efficiency_12h=rng.choice((-999.0, -0.20, -0.05, 0.0, 0.08)),
        min_market_efficiency_12h=rng.choice((0.0, 0.05, 0.08, 0.12, 0.16)),
        min_directional_symbol_ema55_atr=rng.choice((-999.0, -0.50, 0.0, 0.50)),
        max_directional_symbol_ema55_atr=rng.choice((2.0, 3.0, 5.0, 999.0)),
        min_regime_score=rng.choice((-999.0, -0.50, -0.20, 0.0, 0.20, 0.40)),
        max_regime_score=rng.choice((0.75, 1.0, 1.5, 999.0)),
        max_signals_per_symbol_day=rng.choice((1, 2, 3, 4, 6)),
        ranking_mode=rng.choice(
            (
                "quality_desc",
                "alignment_desc",
                "slope_desc",
                "extension_asc",
                "regime_desc",
                "blended_desc",
            )
        ),
    )


def _random_geometry(source: TrendGridConfig, rng: random.Random) -> TrendGridConfig:
    spacing = rng.choice((0.40, 0.50, 0.60, 0.70, 0.85, 1.0))
    levels = rng.choice((2, 3, 4))
    stop = max(
        spacing * levels + rng.choice((0.20, 0.40, 0.75, 1.25)),
        rng.choice((3.0, 4.0, 5.0, 6.0)),
    )
    protection = rng.choice(("none", "loss", "take_profit", "giveback"))
    loss_limit = rng.choice((0.25, 0.35, 0.50, 0.75)) if protection == "loss" else 0.0
    take_profit = rng.choice((0.10, 0.15, 0.25, 0.40, 0.60)) if protection == "take_profit" else 0.0
    activation = rng.choice((0.15, 0.25, 0.40, 0.60)) if protection == "giveback" else 0.0
    giveback = rng.choice((0.08, 0.12, 0.20, 0.30)) if protection == "giveback" else 0.0
    return replace(
        source,
        grid_spacing_atr=spacing,
        grid_levels=levels,
        grid_target_spacing=rng.choice((0.65, 0.80, 1.0, 1.20, 1.50, 2.0)),
        deeper_level_size_multiplier=rng.choice((0.50, 0.70, 0.85, 1.0)),
        initial_entry_enabled=rng.choice((True, True, True, False)),
        hard_stop_atr_multiple=stop,
        hard_stop_slow_ema_buffer_atr=rng.choice((0.20, 0.35, 0.50, 0.75, 1.0)),
        regime_exit_mode=rng.choice(("fast_ema", "slow_ema", "ema_cross", "fast_or_cross")),
        regime_exit_confirm_bars=rng.choice((1, 2, 3)),
        max_campaign_minutes=rng.choice((720, 1_440, 2_880, 4_320)),
        max_cycles_per_level=rng.choice((1, 2, 3, 4)),
        max_total_entries=rng.choice((0, 2, 3, 4, 5, 7)),
        pause_new_fills_on_fast_breach=rng.choice((False, True)),
        campaign_loss_limit_r=loss_limit,
        campaign_take_profit_r=take_profit,
        profit_lock_activation_r=activation,
        profit_giveback_r=giveback,
    )


def _random_portfolio(source: GridPortfolioConfig, rng: random.Random) -> GridPortfolioConfig:
    risk = rng.choice((0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.10))
    return replace(
        source,
        risk_per_campaign_pct=risk,
        max_campaign_risk_pct=0.10,
        max_open_campaigns=rng.choice((1, 1, 2)),
        max_daily_campaigns=rng.choice((4, 6, 8, 12)),
        symbol_cooldown_minutes=rng.choice((120, 240, 360, 480, 720)),
        max_notional_multiple=rng.choice((5.0, 7.0, 9.0)),
        hard_drawdown_stop_pct=0.70,
        compound=True,
        long_risk_multiplier=rng.choice((0.50, 0.75, 1.0, 1.20)),
        short_risk_multiplier=rng.choice((0.50, 0.75, 1.0, 1.20)),
    )


def _run_variants(
    stage: str,
    variants: list[tuple[str, TrendGridConfig, GridMarketOverlay, GridPortfolioConfig]],
    common: tuple[Any, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, (name, signal, overlay, portfolio) in enumerate(variants, 1):
        row = _evaluate(name, signal, overlay, portfolio, *common)
        rows.append(row)
        _log_stage(stage, number, len(variants), row)
    return rows


def _final_stress_tests(
    selected: dict[str, Any],
    universe: tuple[str, ...],
    signal_data: dict[str, Any],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, Any],
    execution: Any,
    context: dict[int, dict[str, V4MarketSnapshot]],
    cache: _TimelineCache,
    start: datetime,
    end: datetime,
    initial_equity: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signal = TrendGridConfig(**selected["signal"])
    overlay = GridMarketOverlay(**selected["market_overlay"])
    portfolio = GridPortfolioConfig(**selected["portfolio"])
    raw_candidates, snapshots = cache.get(
        signal, universe, signal_data, execution_data, start, end
    )
    candidates = apply_market_overlay(raw_candidates, context, overlay)

    def simulate(
        local_candidates: dict[int, list[GridCandidate]],
        local_portfolio: GridPortfolioConfig = portfolio,
        local_execution: Any = execution,
        local_start: datetime = start,
        local_end: datetime = end,
        skip_symbols: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        return simulate_grid_portfolio(
            local_candidates,
            snapshots,
            universe,
            execution_data,
            rules,
            signal,
            local_portfolio,
            local_execution,
            local_start,
            local_end,
            initial_equity,
            skip_symbols=skip_symbols,
        )

    full = simulate(candidates)
    stress: dict[str, Any] = {
        "fixed_risk_no_compounding": compact_grid_summary(
            simulate(candidates, replace(portfolio, compound=False))
        ),
        "entry_delay_1m": compact_grid_summary(
            simulate(_shift_candidates(candidates, 1, execution_data))
        ),
        "cost_1_5x": compact_grid_summary(
            simulate(candidates, local_execution=_scaled_execution(execution, 1.5))
        ),
    }
    top_symbol = max(
        full["by_symbol"],
        key=lambda symbol: full["by_symbol"][symbol]["net_pnl"],
        default="",
    )
    if top_symbol:
        excluded = compact_grid_summary(
            simulate(candidates, skip_symbols=frozenset({top_symbol}))
        )
        excluded["excluded_symbol"] = top_symbol
        stress["exclude_top_symbol"] = excluded

    split1 = start + timedelta(days=30)
    split2 = start + timedelta(days=60)
    folds: dict[str, Any] = {}
    for name, fold_start, fold_end in (
        ("fold_1", start, split1),
        ("fold_2", split1, split2),
        ("fold_3", split2, end),
    ):
        folds[name] = compact_grid_summary(
            simulate(candidates, local_start=fold_start, local_end=fold_end)
        )
    stress["fresh_equity_time_folds"] = folds
    return full, stress


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    if end - start < timedelta(days=89):
        raise ValueError("v3 Grid optimization requires a full three-month period")
    data_start = start - timedelta(days=args.warmup_days)
    universe = tuple(UNIVERSE_50)
    signal_data, execution_data, rules, execution, metadata = load_v4_runtime_inputs(
        universe,
        args.one_minute_roots,
        args.funding_roots,
        args.cost_config,
        data_start,
        end,
    )
    if metadata["minimum_coverage_ratio"] < 0.999999 or metadata["maximum_missing_minutes"]:
        raise RuntimeError("Grid v3 requires gap-free stitched 1m data")
    if metadata["funding_missing_symbols"]:
        raise RuntimeError(
            f"Grid v3 requires complete funding data: {metadata['funding_missing_symbols']}"
        )
    context = build_v4_market_context(universe, signal_data)
    baseline_payload = json.loads(Path(args.baseline_config).read_text(encoding="utf-8"))
    baseline_signal = TrendGridConfig(**baseline_payload["signal"])
    baseline_portfolio = GridPortfolioConfig(**baseline_payload["portfolio"])
    baseline_overlay = GridMarketOverlay()
    timeline_cache = _TimelineCache(args.timeline_cache_size)
    overlay_cache: dict[str, dict[int, list[GridCandidate]]] = {}
    common = (
        universe,
        signal_data,
        execution_data,
        rules,
        execution,
        context,
        timeline_cache,
        overlay_cache,
        start,
        end,
        args.initial_equity,
    )

    baseline = _evaluate(
        "v2_same_period_baseline",
        baseline_signal,
        baseline_overlay,
        baseline_portfolio,
        *common,
    )
    drawdown_cap = float(baseline["metrics"]["max_drawdown_pct"])
    print(
        f"baseline net={baseline['metrics']['net_profit']:+.2f} "
        f"PF={baseline['metrics']['profit_factor']:.3f} "
        f"DD cap={drawdown_cap:.2%}",
        flush=True,
    )
    rng = random.Random(args.seed)
    stages: dict[str, list[dict[str, Any]]] = {"baseline": [baseline]}

    alpha_portfolio = replace(
        baseline_portfolio,
        risk_per_campaign_pct=0.04,
        max_campaign_risk_pct=0.10,
        max_open_campaigns=1,
        compound=True,
    )
    structural: list[
        tuple[str, TrendGridConfig, GridMarketOverlay, GridPortfolioConfig]
    ] = []
    for timeframe in (30, 60):
        for fast, slow in ((13, 55), (21, 55), (21, 99)):
            for mode in ("continuous", "fast_reclaim", "pullback_touch", "hybrid"):
                for side in ("both", "long", "short"):
                    signal = replace(
                        baseline_signal,
                        timeframe_minutes=timeframe,
                        fast_ema_period=fast,
                        slow_ema_period=slow,
                        entry_mode=mode,
                        allow_long=side != "short",
                        allow_short=side != "long",
                    )
                    structural.append(
                        (
                            f"tf{timeframe}_ema{fast}_{slow}_{mode}_{side}",
                            signal,
                            baseline_overlay,
                            alpha_portfolio,
                        )
                    )
    stage1 = _run_variants("structure", structural, common)
    stages["structure"] = _ranked(stage1, drawdown_cap, args.initial_equity, 20)
    structure_finalists = _ranked(stage1, drawdown_cap, args.initial_equity, 4)

    entry_variants: list[
        tuple[str, TrendGridConfig, GridMarketOverlay, GridPortfolioConfig]
    ] = []
    entry_portfolio = replace(alpha_portfolio, risk_per_campaign_pct=0.05)
    for finalist_number, source in enumerate(structure_finalists, 1):
        source_signal = TrendGridConfig(**source["signal"])
        entry_variants.append(
            (f"entry_source_{finalist_number}", source_signal, baseline_overlay, entry_portfolio)
        )
        for variant in range(args.entry_budget):
            entry_variants.append(
                (
                    f"entry_{finalist_number}_{variant + 1}",
                    _random_entry_variant(source_signal, rng),
                    baseline_overlay,
                    entry_portfolio,
                )
            )
    stage2 = _run_variants("entry", entry_variants, common)
    stages["entry"] = _ranked(stage2, drawdown_cap, args.initial_equity, 25)
    entry_finalists = _ranked(stage2, drawdown_cap, args.initial_equity, 4)

    overlay_variants: list[
        tuple[str, TrendGridConfig, GridMarketOverlay, GridPortfolioConfig]
    ] = []
    for finalist_number, source in enumerate(entry_finalists, 1):
        source_signal = TrendGridConfig(**source["signal"])
        overlay_variants.append(
            (f"overlay_source_{finalist_number}", source_signal, baseline_overlay, entry_portfolio)
        )
        for variant in range(args.overlay_budget):
            overlay_variants.append(
                (
                    f"overlay_{finalist_number}_{variant + 1}",
                    source_signal,
                    _random_overlay(rng),
                    entry_portfolio,
                )
            )
    stage3 = _run_variants("overlay", overlay_variants, common)
    stages["overlay"] = _ranked(stage3, drawdown_cap, args.initial_equity, 30)
    overlay_finalists = _ranked(stage3, drawdown_cap, args.initial_equity, 6)

    geometry_variants: list[
        tuple[str, TrendGridConfig, GridMarketOverlay, GridPortfolioConfig]
    ] = []
    geometry_portfolio = replace(entry_portfolio, risk_per_campaign_pct=0.06)
    for finalist_number, source in enumerate(overlay_finalists, 1):
        source_signal = TrendGridConfig(**source["signal"])
        source_overlay = GridMarketOverlay(**source["market_overlay"])
        geometry_variants.append(
            (
                f"geometry_source_{finalist_number}",
                source_signal,
                source_overlay,
                geometry_portfolio,
            )
        )
        for variant in range(args.geometry_budget):
            geometry_variants.append(
                (
                    f"geometry_{finalist_number}_{variant + 1}",
                    _random_geometry(source_signal, rng),
                    source_overlay,
                    geometry_portfolio,
                )
            )
    stage4 = _run_variants("geometry", geometry_variants, common)
    stages["geometry"] = _ranked(stage4, drawdown_cap, args.initial_equity, 35)
    geometry_finalists = _ranked(stage4, drawdown_cap, args.initial_equity, 8)

    portfolio_variants: list[
        tuple[str, TrendGridConfig, GridMarketOverlay, GridPortfolioConfig]
    ] = []
    for finalist_number, source in enumerate(geometry_finalists, 1):
        source_signal = TrendGridConfig(**source["signal"])
        source_overlay = GridMarketOverlay(**source["market_overlay"])
        source_portfolio = GridPortfolioConfig(**source["portfolio"])
        portfolio_variants.append(
            (
                f"portfolio_source_{finalist_number}",
                source_signal,
                source_overlay,
                source_portfolio,
            )
        )
        for variant in range(args.portfolio_budget):
            portfolio_variants.append(
                (
                    f"portfolio_{finalist_number}_{variant + 1}",
                    source_signal,
                    source_overlay,
                    _random_portfolio(source_portfolio, rng),
                )
            )
    stage5 = _run_variants("portfolio", portfolio_variants, common)
    stages["portfolio"] = _ranked(stage5, drawdown_cap, args.initial_equity, 40)

    eligible = [
        row
        for row in stage5
        if row["metrics"]["trade_count"] >= 18
        and row["metrics"]["net_profit"] > baseline["metrics"]["net_profit"]
        and row["metrics"]["profit_factor"] > baseline["metrics"]["profit_factor"]
        and row["metrics"]["max_drawdown_pct"] <= drawdown_cap + 1e-12
        and not row["metrics"]["hard_drawdown_stopped"]
    ]
    if not eligible:
        raise RuntimeError("no Grid v3 candidate improved baseline within the drawdown cap")
    balanced = max(
        eligible,
        key=lambda row: _score(row, drawdown_cap, args.initial_equity),
    )
    profit_champion = max(eligible, key=lambda row: row["metrics"]["net_profit"])
    pf_champion = max(
        (row for row in eligible if row["metrics"]["net_profit"] > 0.0),
        key=lambda row: row["metrics"]["profit_factor"],
    )
    final_result, stress = _final_stress_tests(
        balanced,
        universe,
        signal_data,
        execution_data,
        rules,
        execution,
        context,
        timeline_cache,
        start,
        end,
        args.initial_equity,
    )
    report = {
        "strategy_name": TREND_GRID_V3_RESEARCH_NAME,
        "status": "independent_grid_research_not_live_gui_unchanged",
        "period": {
            "warmup_start": data_start.isoformat(),
            "backtest_start": start.isoformat(),
            "backtest_end": end.isoformat(),
        },
        "universe_size": len(universe),
        "symbols": list(universe),
        "initial_equity": args.initial_equity,
        "data_quality": metadata,
        "cost_model": {
            "mode": "conservative_full_cost",
            "cost_config": args.cost_config,
            "market_slippage_bps": execution.market_slippage_bps,
            "stop_slippage_bps": execution.stop_slippage_bps,
            "take_profit_slippage_bps": execution.take_profit_slippage_bps,
            "maker_fee_rate": execution.maker_fee_rate,
            "taker_fee_rate": execution.taker_fee_rate,
            "funding_enabled": execution.funding_enabled,
        },
        "drawdown_constraint": {
            "source": "same_period_v2_baseline",
            "maximum_allowed_pct": drawdown_cap,
            "satisfied": final_result["max_drawdown_pct"] <= drawdown_cap + 1e-12,
        },
        "search": {
            "seed": args.seed,
            "structure_evaluations": len(stage1),
            "entry_evaluations": len(stage2),
            "overlay_evaluations": len(stage3),
            "geometry_evaluations": len(stage4),
            "portfolio_evaluations": len(stage5),
            "total_evaluations": len(stage1) + len(stage2) + len(stage3) + len(stage4) + len(stage5),
        },
        "baseline": baseline,
        "selected": {
            **{key: value for key, value in balanced.items() if key != "metrics"},
            "metrics": compact_grid_summary(final_result),
        },
        "profit_champion": profit_champion,
        "pf_champion": pf_champion,
        "stress_tests": stress,
        "stages": stages,
        "full_selected_result": final_result,
        "preserved": {
            "active_gui_config": "config.volatility-breakout.v2-balanced-50-shadow.json",
            "apt_grid": "unchanged",
            "other_strategy_source": "unchanged",
        },
    }
    _write_json(args.output, report)
    config_payload = {
        "strategy_name": TREND_GRID_V3_RESEARCH_NAME,
        "status": "historical_research_not_live",
        "period": report["period"],
        "symbols": list(universe),
        "signal": balanced["signal"],
        "market_overlay": balanced["market_overlay"],
        "portfolio": balanced["portfolio"],
        "baseline_metrics": baseline["metrics"],
        "selected_metrics": compact_grid_summary(final_result),
        "drawdown_cap_pct": drawdown_cap,
        "cost_model": report["cost_model"],
    }
    _write_json(args.config_output, config_payload)
    summary_lines = [
        "# Dynamic Trend-Following Grid v3 - Latest 3 Month Optimization",
        "",
        f"- Period: `{start.isoformat()}` to `{end.isoformat()}`",
        f"- Warmup: `{data_start.isoformat()}` to `{start.isoformat()}`",
        f"- Universe: `{len(universe)}` symbols",
        "- Execution: gap-free 1m, next-bar execution, conservative stop-first, full fees/slippage/funding",
        "- Selection: higher net and PF than same-period v2 while max drawdown cannot exceed the v2 baseline",
        "- Research note: this is an optimized historical window, not an untouched forward sample",
        "",
        "| Version | Campaigns | Net | PF | Win rate | Max DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (
        ("Grid v2 baseline", baseline["metrics"]),
        ("Grid v3 selected", compact_grid_summary(final_result)),
    ):
        summary_lines.append(
            f"| {name} | {metrics['trade_count']} | {metrics['net_profit']:+.2f}U | "
            f"{metrics['profit_factor']:.3f} | {metrics['win_rate']:.2%} | "
            f"{metrics['max_drawdown_pct']:.2%} |"
        )
    summary_lines.extend(
        [
            "",
            "## Selected parameters",
            "",
            f"- Signal: `{balanced['signal']}`",
            f"- Market overlay: `{balanced['market_overlay']}`",
            f"- Portfolio: `{balanced['portfolio']}`",
            "",
            "## Stress tests",
            "",
        ]
    )
    for name, result in stress.items():
        if name == "fresh_equity_time_folds":
            continue
        summary_lines.append(
            f"- `{name}`: {result['net_profit']:+.2f}U / PF "
            f"{(result['profit_factor'] or 0.0):.3f} / DD {result['max_drawdown_pct']:.2%}"
        )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    manifest = {
        "strategy_name": TREND_GRID_V3_RESEARCH_NAME,
        "status": "independent_research_artifacts",
        "config": args.config_output,
        "report": args.output,
        "summary": args.summary,
        "hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                "crypto_scalper/trend_grid_v3_optimize.py",
                "tests/test_trend_grid_v3_optimize.py",
                args.config_output,
                args.output,
                args.summary,
            )
        },
        "preserved": report["preserved"],
    }
    _write_json(args.manifest, manifest)
    print(
        f"selected net={final_result['net_profit']:+.2f} "
        f"PF={final_result['profit_factor']:.3f} "
        f"DD={final_result['max_drawdown_pct']:.2%}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent latest-three-month Dynamic Trend Grid v3 optimizer"
    )
    parser.add_argument("--start", default="2026-04-19T00:00:00")
    parser.add_argument("--end", default="2026-07-19T00:00:00")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=260721)
    parser.add_argument("--entry-budget", type=int, default=12)
    parser.add_argument("--overlay-budget", type=int, default=16)
    parser.add_argument("--geometry-budget", type=int, default=25)
    parser.add_argument("--portfolio-budget", type=int, default=20)
    parser.add_argument("--timeline-cache-size", type=int, default=8)
    parser.add_argument(
        "--one-minute-roots",
        nargs="+",
        default=(
            "data/binance_1m_365d_top100",
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
        "--cost-config",
        default="config.volatility-breakout.v2-balanced-50-shadow.json",
    )
    parser.add_argument(
        "--baseline-config", default="config.trend-grid.v2-optimized-50.json"
    )
    parser.add_argument(
        "--output", default="reports/trend_grid_v3_latest_3m_50.json"
    )
    parser.add_argument(
        "--summary", default="reports/trend_grid_v3_latest_3m_50.md"
    )
    parser.add_argument(
        "--config-output", default="config.trend-grid.v3-optimized-50.json"
    )
    parser.add_argument(
        "--manifest", default="config.trend-grid.v3-latest-3m-manifest.json"
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .combined_hybrid_v5_grid_v3_backtest import (
    _slice_signal_data,
    _write_json,
    build_frozen_configs,
)
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import (
    GridCandidate,
    GridPortfolioConfig,
    _scaled_execution,
    _shift_candidates,
    build_grid_research_timeline,
    compact_grid_summary,
    sha256_file,
    simulate_grid_portfolio,
)
from .trend_grid_v3_optimize import GridMarketOverlay, apply_market_overlay
from .trend_grid_v4 import (
    TREND_GRID_V4_NAME,
    GridV4EntryGate,
    filter_grid_v4_candidates,
)
from .volatility_breakout_optimize import UNIVERSE_50
from .volatility_breakout_v4_research import (
    build_v4_market_context,
    load_v4_runtime_inputs,
)


def _value(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _candidate_count(candidates: dict[int, list[GridCandidate]]) -> int:
    return sum(len(rows) for rows in candidates.values())


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    output = compact_grid_summary(result)
    trades = result.get("trades", ())
    output.update(
        {
            "largest_trade_net_profit": max(
                (_value(row.get("net_pnl")) for row in trades), default=0.0
            ),
            "largest_trade_r": max(
                (_value(row.get("pnl_r")) for row in trades), default=0.0
            ),
            "hard_stop_count": sum(
                row.get("exit_reason") == "hard_stop" for row in trades
            ),
        }
    )
    return output


def _gate_variants(seed: int, budget: int) -> list[GridV4EntryGate]:
    rows = [
        GridV4EntryGate(),
        GridV4EntryGate(min_quality_score=0.45),
        GridV4EntryGate(min_quality_score=0.60),
        GridV4EntryGate(min_quality_score=0.75),
        GridV4EntryGate(min_alignment_atr=0.60),
        GridV4EntryGate(min_alignment_atr=0.75, max_alignment_atr=3.0),
        GridV4EntryGate(max_extension_atr=0.45),
        GridV4EntryGate(min_directional_fast_slope_atr=0.08),
        GridV4EntryGate(min_directional_slow_slope_atr=0.08),
        GridV4EntryGate(
            min_quality_score=0.50,
            min_alignment_atr=0.50,
            max_extension_atr=0.55,
        ),
        GridV4EntryGate(
            min_market_efficiency_12h=0.20,
            min_directional_breadth=0.35,
            max_directional_breadth=0.65,
        ),
    ]
    # One-factor boundaries are deliberately evaluated before random
    # interactions.  They are easier to reason about, implement live, and
    # perturb around during robustness checks.
    rows.extend(
        GridV4EntryGate(min_quality_score=value)
        for value in (0.25, 0.30, 0.35, 0.40, 0.50, 0.55, 0.65)
    )
    rows.extend(
        GridV4EntryGate(max_quality_score=value)
        for value in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    )
    rows.extend(
        GridV4EntryGate(min_alignment_atr=value)
        for value in (0.30, 0.40, 0.50, 0.60, 0.80, 1.00)
    )
    rows.extend(
        GridV4EntryGate(max_alignment_atr=value)
        for value in (1.50, 2.00, 2.50, 3.00, 3.50)
    )
    rows.extend(
        GridV4EntryGate(max_extension_atr=value)
        for value in (0.40, 0.50, 0.55, 0.60)
    )
    rows.extend(
        GridV4EntryGate(min_extension_atr=value)
        for value in (0.05, 0.10, 0.20)
    )
    rows.extend(
        GridV4EntryGate(min_volume_ratio=value)
        for value in (0.75, 0.80, 0.90, 1.00)
    )
    rows.extend(
        GridV4EntryGate(max_volume_ratio=value)
        for value in (1.30, 1.50, 1.80)
    )
    rows.extend(
        GridV4EntryGate(min_directional_fast_slope_atr=value)
        for value in (0.02, 0.05, 0.08, 0.12)
    )
    rows.extend(
        GridV4EntryGate(min_directional_slow_slope_atr=value)
        for value in (0.06, 0.08, 0.12, 0.18)
    )
    rows.extend(
        GridV4EntryGate(min_market_efficiency_12h=value)
        for value in (0.12, 0.18, 0.20, 0.22, 0.24)
    )
    rows.extend(
        GridV4EntryGate(min_directional_breadth=value)
        for value in (0.20, 0.30, 0.35, 0.40)
    )
    rows.extend(
        GridV4EntryGate(max_directional_breadth=value)
        for value in (0.55, 0.60, 0.65, 0.75)
    )
    rows.extend(
        GridV4EntryGate(min_regime_score=value)
        for value in (-0.50, -0.25, 0.0)
    )
    rows.extend(
        GridV4EntryGate(max_regime_score=value)
        for value in (0.50, 0.75, 1.00, 1.25)
    )
    rng = random.Random(seed)
    while len(list(dict.fromkeys(rows))) < budget:
        min_alignment = rng.choice((-999.0, 0.30, 0.45, 0.60, 0.75, 1.0))
        max_alignment = rng.choice((1.5, 2.0, 2.5, 3.0, 4.0, 999.0))
        if max_alignment < min_alignment:
            continue
        min_breadth = rng.choice((0.0, 0.25, 0.35, 0.45, 0.55))
        max_breadth = rng.choice((0.55, 0.65, 0.70, 0.80, 1.0))
        if max_breadth < min_breadth:
            continue
        rows.append(
            GridV4EntryGate(
                min_quality_score=rng.choice(
                    (-999.0, 0.35, 0.45, 0.55, 0.65, 0.75, 0.90)
                ),
                max_quality_score=rng.choice((0.9, 1.1, 1.3, 999.0)),
                min_alignment_atr=min_alignment,
                max_alignment_atr=max_alignment,
                min_extension_atr=rng.choice((-999.0, 0.0, 0.10, 0.20)),
                max_extension_atr=rng.choice((0.35, 0.45, 0.55, 0.65, 999.0)),
                min_volume_ratio=rng.choice((0.0, 0.70, 0.80, 0.90, 1.0)),
                max_volume_ratio=rng.choice((1.3, 1.5, 1.8, 2.0, 999.0)),
                min_directional_fast_slope_atr=rng.choice(
                    (-999.0, 0.0, 0.04, 0.08, 0.12)
                ),
                min_directional_slow_slope_atr=rng.choice(
                    (-999.0, 0.05, 0.08, 0.12, 0.18)
                ),
                min_market_efficiency_12h=rng.choice((0.0, 0.16, 0.20, 0.24)),
                min_directional_breadth=min_breadth,
                max_directional_breadth=max_breadth,
                min_regime_score=rng.choice((-999.0, -0.50, -0.20, 0.0)),
                max_regime_score=rng.choice((0.50, 0.75, 1.0, 999.0)),
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _overlay_variants(
    source: GridMarketOverlay, seed: int, budget: int
) -> list[GridMarketOverlay]:
    rows = [source]
    rows.extend(
        replace(source, ranking_mode=value)
        for value in (
            "quality_desc",
            "alignment_desc",
            "slope_desc",
            "extension_asc",
            "regime_desc",
            "blended_desc",
        )
    )
    rows.extend(
        replace(source, max_directional_btc_return_4h=value)
        for value in (0.0075, 0.012, 0.015, 0.020)
    )
    rows.extend(
        replace(source, min_directional_btc_return_4h=value)
        for value in (-0.015, -0.012, -0.0075, -0.005)
    )
    rows.extend(
        replace(source, max_directional_eth_return_4h=value)
        for value in (0.010, 0.012, 0.020, 0.030)
    )
    rows.extend(
        replace(source, min_directional_breadth=value)
        for value in (0.20, 0.30, 0.35, 0.40)
    )
    rows.extend(
        replace(source, max_directional_breadth=value)
        for value in (0.55, 0.60, 0.65, 0.75, 0.80)
    )
    rows.extend(
        replace(source, min_directional_breadth_change_4h=value)
        for value in (-0.10, -0.05, -0.02, 0.0)
    )
    rows.extend(
        replace(source, min_directional_symbol_return_4h=value)
        for value in (-0.030, -0.020, -0.010, -0.005, 0.0)
    )
    rows.extend(
        replace(source, max_directional_symbol_return_4h=value)
        for value in (0.030, 0.050, 0.100, 0.120)
    )
    rows.extend(
        replace(source, min_directional_symbol_efficiency_12h=value)
        for value in (-0.10, 0.0, 0.05)
    )
    rows.extend(
        replace(source, min_market_efficiency_12h=value)
        for value in (0.12, 0.18, 0.20, 0.22, 0.24)
    )
    rows.extend(
        replace(source, min_directional_symbol_ema55_atr=value)
        for value in (-1.0, -0.25, 0.0)
    )
    rows.extend(
        replace(source, max_directional_symbol_ema55_atr=value)
        for value in (1.25, 1.50, 2.50, 3.00)
    )
    rows.extend(
        replace(source, min_regime_score=value)
        for value in (-0.50, -0.25, 0.0)
    )
    rows.extend(
        replace(source, max_regime_score=value)
        for value in (0.50, 0.75, 1.25)
    )
    rows = list(dict.fromkeys(rows))
    rng = random.Random(seed)
    while len(list(dict.fromkeys(rows))) < budget:
        min_breadth = rng.choice((0.20, 0.25, 0.30, 0.35, 0.40, 0.45))
        max_breadth = rng.choice((0.55, 0.60, 0.65, 0.70, 0.75, 0.80))
        if max_breadth < min_breadth:
            continue
        min_btc = rng.choice((-0.015, -0.012, -0.010, -0.0075, -0.005, 0.0))
        max_btc = rng.choice((0.005, 0.0075, 0.010, 0.012, 0.015, 0.02))
        if max_btc < min_btc:
            continue
        min_ema = rng.choice((-1.0, -0.75, -0.50, -0.25, 0.0))
        max_ema = rng.choice((1.0, 1.5, 2.0, 2.5, 3.0))
        if max_ema < min_ema:
            continue
        rows.append(
            replace(
                source,
                min_directional_btc_return_4h=min_btc,
                max_directional_btc_return_4h=max_btc,
                max_directional_eth_return_4h=rng.choice(
                    (0.008, 0.012, 0.015, 0.020, 0.030)
                ),
                min_directional_breadth=min_breadth,
                max_directional_breadth=max_breadth,
                min_directional_breadth_change_4h=rng.choice(
                    (-999.0, -0.10, -0.05, -0.02, 0.0, 0.02)
                ),
                min_directional_symbol_return_4h=rng.choice(
                    (-0.03, -0.02, -0.015, -0.01, -0.005, 0.0)
                ),
                max_directional_symbol_return_4h=rng.choice(
                    (0.03, 0.05, 0.08, 0.12)
                ),
                min_directional_symbol_efficiency_12h=rng.choice(
                    (-0.15, -0.10, -0.05, 0.0, 0.05)
                ),
                min_market_efficiency_12h=rng.choice(
                    (0.12, 0.16, 0.20, 0.24, 0.28)
                ),
                min_directional_symbol_ema55_atr=min_ema,
                max_directional_symbol_ema55_atr=max_ema,
                min_regime_score=rng.choice((-999.0, -0.50, -0.25, 0.0)),
                max_regime_score=rng.choice((0.50, 0.75, 1.0, 1.25)),
                ranking_mode=rng.choice(
                    (
                        "alignment_desc",
                        "quality_desc",
                        "slope_desc",
                        "extension_asc",
                        "regime_desc",
                        "blended_desc",
                    )
                ),
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _exit_variants(
    source: TrendGridConfig, seed: int, budget: int
) -> list[TrendGridConfig]:
    rows = [
        source,
        replace(source, campaign_loss_limit_r=0.60),
        replace(source, campaign_loss_limit_r=0.80),
        replace(source, campaign_loss_limit_r=1.0),
        replace(source, profit_lock_activation_r=0.40, profit_giveback_r=0.06),
        replace(source, profit_lock_activation_r=0.80, profit_giveback_r=0.12),
        replace(source, max_campaign_minutes=2_880),
        replace(source, grid_target_spacing=2.50),
    ]
    # The first broad pass found a useful interaction: tighter campaign loss
    # limits improved the older 1.75 target, while 1.85 raised both-window
    # profit but needed drawdown control.  Evaluate that interaction directly;
    # one-factor-at-a-time searches cannot discover it.
    for target in (1.75, 1.80, 1.85, 1.90, 1.95):
        for loss_limit in (0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00):
            rows.append(
                replace(
                    source,
                    grid_target_spacing=target,
                    campaign_loss_limit_r=loss_limit,
                    profit_lock_activation_r=0.0,
                    profit_giveback_r=0.0,
                )
            )
            rows.append(
                replace(
                    source,
                    grid_target_spacing=target,
                    campaign_loss_limit_r=loss_limit,
                )
            )
    # Grid v3's realised winners cluster near +0.45R, while several eventual
    # losers first reached +0.15R to +0.40R.  Search the live-computable
    # campaign-equity trail densely instead of relying only on the old 0.60R
    # activation, which is effectively dormant for this sleeve.
    for activation in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        for giveback in (0.04, 0.06, 0.08, 0.10, 0.15, 0.20):
            rows.append(
                replace(
                    source,
                    profit_lock_activation_r=activation,
                    profit_giveback_r=giveback,
                    campaign_loss_limit_r=0.0,
                )
            )
    for loss_limit in (0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90):
        rows.append(
            replace(
                source,
                campaign_loss_limit_r=loss_limit,
                profit_lock_activation_r=0.0,
                profit_giveback_r=0.0,
            )
        )
    for target in (
        1.50,
        1.55,
        1.60,
        1.65,
        1.70,
        1.75,
        1.80,
        1.85,
        1.90,
        1.95,
        2.00,
        2.05,
        2.10,
        2.15,
        2.20,
        2.25,
        2.50,
        2.75,
        3.00,
    ):
        rows.append(replace(source, grid_target_spacing=target))
    rows = list(dict.fromkeys(rows))
    rng = random.Random(seed)
    while len(list(dict.fromkeys(rows))) < budget:
        spacing = rng.choice((0.50, 0.60, 0.70, 0.80))
        levels = rng.choice((2, 3))
        hard_stop = rng.choice((2.5, 3.0, 3.5, 4.0))
        if hard_stop <= spacing * levels:
            continue
        protection = rng.choice(("lock", "lock", "loss", "none"))
        activation = (
            rng.choice((0.35, 0.45, 0.60, 0.75, 0.90))
            if protection == "lock"
            else 0.0
        )
        giveback = (
            rng.choice((0.04, 0.06, 0.08, 0.12, 0.18))
            if protection == "lock"
            else 0.0
        )
        loss_limit = (
            rng.choice((0.40, 0.60, 0.80, 1.0))
            if protection == "loss"
            else 0.0
        )
        rows.append(
            replace(
                source,
                grid_spacing_atr=spacing,
                grid_levels=levels,
                grid_target_spacing=rng.choice((1.5, 1.75, 2.0, 2.25, 2.5, 3.0)),
                deeper_level_size_multiplier=rng.choice((0.70, 0.85, 1.0)),
                hard_stop_atr_multiple=hard_stop,
                hard_stop_slow_ema_buffer_atr=rng.choice((0.20, 0.35, 0.50, 0.75)),
                regime_exit_mode=rng.choice(
                    ("fast_ema", "slow_ema", "ema_cross", "fast_or_cross")
                ),
                regime_exit_confirm_bars=rng.choice((1, 2, 3)),
                max_campaign_minutes=rng.choice((2_160, 2_880, 4_320, 5_760)),
                max_cycles_per_level=rng.choice((1, 2, 3)),
                max_total_entries=rng.choice((2, 3, 4, 5)),
                pause_new_fills_on_fast_breach=rng.choice((False, True)),
                campaign_loss_limit_r=loss_limit,
                campaign_take_profit_r=0.0,
                profit_lock_activation_r=activation,
                profit_giveback_r=giveback,
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _portfolio_variants(
    source: GridPortfolioConfig, seed: int, budget: int
) -> list[GridPortfolioConfig]:
    rows = [
        source,
        replace(source, risk_per_campaign_pct=0.08, short_risk_multiplier=1.0),
        replace(source, risk_per_campaign_pct=0.09, short_risk_multiplier=1.0),
        replace(source, max_notional_multiple=4.0),
        replace(source, max_notional_multiple=6.0),
    ]
    rng = random.Random(seed)
    while len(list(dict.fromkeys(rows))) < budget:
        rows.append(
            replace(
                source,
                risk_per_campaign_pct=rng.choice(
                    (0.06, 0.07, 0.08, 0.09, 0.10)
                ),
                max_campaign_risk_pct=0.10,
                max_open_campaigns=1,
                max_daily_campaigns=rng.choice((6, 8, 10, 12)),
                symbol_cooldown_minutes=rng.choice((240, 360, 480, 720)),
                max_notional_multiple=rng.choice((4.0, 5.0, 6.0, 7.0)),
                hard_drawdown_stop_pct=0.70,
                compound=True,
                long_risk_multiplier=1.0,
                short_risk_multiplier=rng.choice((0.8, 1.0, 1.2)),
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _period_score(result: dict[str, Any], baseline: dict[str, Any]) -> float:
    if result["hard_drawdown_stopped"] or result["trade_count"] < 10:
        return -1e9
    net_scale = max(abs(baseline["net_profit"]), baseline["initial_equity"] * 0.10)
    net_delta = (result["net_profit"] - baseline["net_profit"]) / net_scale
    return (
        8.0 * net_delta
        + 0.75 * (_value(result["profit_factor"]) - _value(baseline["profit_factor"]))
        + 1.5 * (result["win_rate"] - baseline["win_rate"])
        + 6.0 * (baseline["max_drawdown_pct"] - result["max_drawdown_pct"])
        - 0.3 * max(0.0, result["top5_profit_contribution"] - 0.65)
    )


def _pair_score(
    six: dict[str, Any],
    three: dict[str, Any],
    baseline_six: dict[str, Any],
    baseline_three: dict[str, Any],
) -> float:
    return _period_score(six, baseline_six) + 1.35 * _period_score(
        three, baseline_three
    )


def _strict_improvement(
    six: dict[str, Any],
    three: dict[str, Any],
    baseline_six: dict[str, Any],
    baseline_three: dict[str, Any],
) -> bool:
    return (
        six["trade_count"] >= 25
        and three["trade_count"] >= 15
        and six["net_profit"] > baseline_six["net_profit"]
        and three["net_profit"] > baseline_three["net_profit"]
        and _value(six["profit_factor"]) > _value(baseline_six["profit_factor"])
        and _value(three["profit_factor"])
        > _value(baseline_three["profit_factor"])
        and six["max_drawdown_pct"] <= baseline_six["max_drawdown_pct"]
        and three["max_drawdown_pct"] <= baseline_three["max_drawdown_pct"]
        # Grid v3 already has a high hit rate.  Keep it within three percentage
        # points while requiring the user's primary net/PF/DD improvements in
        # both independently warmed windows.
        and six["win_rate"] >= baseline_six["win_rate"] - 0.03
        and three["win_rate"] >= baseline_three["win_rate"] - 0.03
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_six = datetime.fromisoformat(args.start_6m)
    start_three = datetime.fromisoformat(args.start_3m)
    end = datetime.fromisoformat(args.end)
    if not start_six < start_three < end:
        raise ValueError("periods must satisfy start_6m < start_3m < end")

    symbols = tuple(UNIVERSE_50)
    signal_data, execution_data, rules, execution, metadata = load_v4_runtime_inputs(
        symbols,
        args.one_minute_roots,
        args.funding_roots,
        args.cost_config,
        start_six - timedelta(days=args.warmup_days),
        end,
    )
    if metadata["minimum_coverage_ratio"] < 0.999999:
        raise RuntimeError("Grid v4 requires gap-free stitched 1m data")
    if metadata["maximum_missing_minutes"]:
        raise RuntimeError("Grid v4 requires gap-free stitched 1m data")
    if metadata["funding_missing_symbols"]:
        raise RuntimeError("Grid v4 requires complete funding data")
    frozen = build_frozen_configs(args.breakout_config, args.grid_config)
    base_signal: TrendGridConfig = frozen["grid_signal"]
    base_overlay: GridMarketOverlay = frozen["grid_overlay"]
    base_portfolio: GridPortfolioConfig = frozen["grid_portfolio"]
    grid_payload = json.loads(Path(args.grid_config).read_text(encoding="utf-8"))
    base_gate = GridV4EntryGate(**grid_payload.get("entry_gate", {}))
    base_gate.validate()

    def build_period(name: str, start: datetime, finish: datetime) -> dict[str, Any]:
        local_signal_data = _slice_signal_data(
            signal_data, start - timedelta(days=args.warmup_days), finish
        )
        context = build_v4_market_context(symbols, local_signal_data)
        raw, snapshots = build_grid_research_timeline(
            symbols,
            local_signal_data,
            execution_data,
            base_signal,
            start,
            finish,
        )
        baseline_candidates = apply_market_overlay(raw, context, base_overlay)
        print(
            f"{name}: raw={_candidate_count(raw)} "
            f"baseline={_candidate_count(baseline_candidates)}",
            flush=True,
        )
        return {
            "start": start,
            "end": finish,
            "context": context,
            "raw": raw,
            "snapshots": snapshots,
            "baseline": baseline_candidates,
        }

    periods = {
        "six": build_period("6m", start_six, end),
        "three": build_period("3m", start_three, end),
        "early": build_period("early3m", start_six, start_three),
    }
    candidate_cache: dict[Any, Any] = {}

    def candidates_for(
        period: str, gate: GridV4EntryGate, overlay: GridMarketOverlay
    ) -> dict[int, list[GridCandidate]]:
        key = (period, gate, overlay)
        if key not in candidate_cache:
            values = periods[period]
            overlaid = (
                values["baseline"]
                if overlay == base_overlay
                else apply_market_overlay(values["raw"], values["context"], overlay)
            )
            candidate_cache[key] = filter_grid_v4_candidates(
                overlaid, values["context"], gate
            )
        return candidate_cache[key]

    def simulate(
        period: str,
        gate: GridV4EntryGate,
        overlay: GridMarketOverlay,
        signal: TrendGridConfig,
        portfolio: GridPortfolioConfig,
        selected_execution: Any = execution,
    ) -> dict[str, Any]:
        values = periods[period]
        return simulate_grid_portfolio(
            candidates_for(period, gate, overlay),
            values["snapshots"],
            symbols,
            execution_data,
            rules,
            signal,
            portfolio,
            selected_execution,
            values["start"],
            values["end"],
            args.initial_equity,
        )

    def simulate_direct(
        period: str,
        selected_candidates: dict[int, list[GridCandidate]],
        signal: TrendGridConfig,
        portfolio: GridPortfolioConfig,
        selected_execution: Any = execution,
        skip_symbols: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        values = periods[period]
        return simulate_grid_portfolio(
            selected_candidates,
            values["snapshots"],
            symbols,
            execution_data,
            rules,
            signal,
            portfolio,
            selected_execution,
            values["start"],
            values["end"],
            args.initial_equity,
            skip_symbols=skip_symbols,
        )

    baseline_six = simulate(
        "six", base_gate, base_overlay, base_signal, base_portfolio
    )
    baseline_three = simulate(
        "three", base_gate, base_overlay, base_signal, base_portfolio
    )
    baseline_report = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    combined_six = baseline_report.get("periods", {}).get("six_month")
    if isinstance(combined_six, dict) and "standalone" in combined_six:
        expected_six = baseline_report["periods"]["six_month"]["standalone"][
            "dynamic_trend_following_grid"
        ]
        expected_three = baseline_report["periods"]["three_month"]["standalone"][
            "dynamic_trend_following_grid"
        ]
    else:
        # A previous independent v4 report can be used as the next frozen
        # optimization anchor without rewriting it into the combined schema.
        expected_six = baseline_report["selected"]["six_month_full"]
        expected_three = baseline_report["selected"]["three_month_full"]
    for label, actual, expected in (
        ("six", baseline_six, expected_six),
        ("three", baseline_three, expected_three),
    ):
        if abs(actual["net_profit"] - expected["net_profit"]) > 1e-6:
            raise RuntimeError(
                f"Grid v4 baseline mismatch {label}: "
                f"{actual['net_profit']} != {expected['net_profit']}"
            )
    print(
        f"baseline 6m={baseline_six['net_profit']:+.2f}/PF{baseline_six['profit_factor']:.3f}/"
        f"win{baseline_six['win_rate']:.1%}/DD{baseline_six['max_drawdown_pct']:.1%}; "
        f"3m={baseline_three['net_profit']:+.2f}/PF{baseline_three['profit_factor']:.3f}/"
        f"win{baseline_three['win_rate']:.1%}/DD{baseline_three['max_drawdown_pct']:.1%}",
        flush=True,
    )

    local_gates = [base_gate]
    local_gates.extend(
        replace(base_gate, max_regime_score=value)
        for value in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    )
    gate_variants = list(
        dict.fromkeys(local_gates + _gate_variants(args.seed, args.gate_budget))
    )[: args.gate_budget]
    gate_rows: list[dict[str, Any]] = []
    for number, gate in enumerate(gate_variants, 1):
        six = simulate("six", gate, base_overlay, base_signal, base_portfolio)
        gate_rows.append(
            {
                "gate": gate,
                "overlay": base_overlay,
                "signal": base_signal,
                "portfolio": base_portfolio,
                "six": six,
                "score6": _period_score(six, baseline_six),
            }
        )
        if number == 1 or number % 10 == 0 or number == args.gate_budget:
            print(
                f"gate {number}/{args.gate_budget}: {six['net_profit']:+.1f}/"
                f"PF{six['profit_factor']:.2f}/win{six['win_rate']:.1%}/"
                f"DD{six['max_drawdown_pct']:.1%}",
                flush=True,
            )
    gate_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in gate_rows[: args.gate_recent_finalists]:
        row["three"] = simulate(
            "three", row["gate"], row["overlay"], row["signal"], row["portfolio"]
        )
        row["pair_score"] = _pair_score(
            row["six"], row["three"], baseline_six, baseline_three
        )
    gate_finalists = sorted(
        gate_rows[: args.gate_recent_finalists],
        key=lambda row: row["pair_score"],
        reverse=True,
    )[: args.gate_finalists]

    overlay_rows: list[dict[str, Any]] = []
    overlays = _overlay_variants(base_overlay, args.seed + 1, args.overlay_budget)
    for gate_number, source in enumerate(gate_finalists, 1):
        for overlay in overlays:
            six = simulate(
                "six", source["gate"], overlay, base_signal, base_portfolio
            )
            overlay_rows.append(
                {
                    "gate": source["gate"],
                    "overlay": overlay,
                    "signal": base_signal,
                    "portfolio": base_portfolio,
                    "six": six,
                    "score6": _period_score(six, baseline_six),
                }
            )
        print(
            f"overlay gate {gate_number}/{len(gate_finalists)} complete", flush=True
        )
    overlay_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in overlay_rows[: args.overlay_recent_finalists]:
        row["three"] = simulate(
            "three", row["gate"], row["overlay"], row["signal"], row["portfolio"]
        )
        row["pair_score"] = _pair_score(
            row["six"], row["three"], baseline_six, baseline_three
        )
    overlay_finalists = sorted(
        overlay_rows[: args.overlay_recent_finalists],
        key=lambda row: row["pair_score"],
        reverse=True,
    )[: args.overlay_finalists]

    exit_rows: list[dict[str, Any]] = []
    exits = _exit_variants(base_signal, args.seed + 2, args.exit_budget)
    for entry_number, source in enumerate(overlay_finalists, 1):
        for signal in exits:
            six = simulate(
                "six", source["gate"], source["overlay"], signal, base_portfolio
            )
            exit_rows.append(
                {
                    "gate": source["gate"],
                    "overlay": source["overlay"],
                    "signal": signal,
                    "portfolio": base_portfolio,
                    "six": six,
                    "score6": _period_score(six, baseline_six),
                }
            )
        print(
            f"exit entry {entry_number}/{len(overlay_finalists)} complete", flush=True
        )
    exit_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in exit_rows[: args.exit_recent_finalists]:
        row["three"] = simulate(
            "three", row["gate"], row["overlay"], row["signal"], row["portfolio"]
        )
        row["pair_score"] = _pair_score(
            row["six"], row["three"], baseline_six, baseline_three
        )
    exit_finalists = sorted(
        exit_rows[: args.exit_recent_finalists],
        key=lambda row: row["pair_score"],
        reverse=True,
    )[: args.exit_finalists]

    portfolio_rows: list[dict[str, Any]] = []
    portfolios = _portfolio_variants(
        base_portfolio, args.seed + 3, args.portfolio_budget
    )
    for exit_number, source in enumerate(exit_finalists, 1):
        for portfolio in portfolios:
            six = simulate(
                "six", source["gate"], source["overlay"], source["signal"], portfolio
            )
            portfolio_rows.append(
                {
                    "gate": source["gate"],
                    "overlay": source["overlay"],
                    "signal": source["signal"],
                    "portfolio": portfolio,
                    "six": six,
                    "score6": _period_score(six, baseline_six),
                }
            )
        print(
            f"portfolio exit {exit_number}/{len(exit_finalists)} complete",
            flush=True,
        )
    portfolio_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in portfolio_rows[: args.portfolio_recent_finalists]:
        row["three"] = simulate(
            "three", row["gate"], row["overlay"], row["signal"], row["portfolio"]
        )
        row["pair_score"] = _pair_score(
            row["six"], row["three"], baseline_six, baseline_three
        )
        row["strict_improvement"] = _strict_improvement(
            row["six"], row["three"], baseline_six, baseline_three
        )
    recent = sorted(
        portfolio_rows[: args.portfolio_recent_finalists],
        key=lambda row: (row["strict_improvement"], row["pair_score"]),
        reverse=True,
    )

    robust_rows: list[dict[str, Any]] = []
    stressed = _scaled_execution(execution, 1.5)
    for number, row in enumerate(recent[: args.robust_finalists], 1):
        early = simulate(
            "early", row["gate"], row["overlay"], row["signal"], row["portfolio"]
        )
        stress_six = simulate(
            "six",
            row["gate"],
            row["overlay"],
            row["signal"],
            row["portfolio"],
            stressed,
        )
        stress_three = simulate(
            "three",
            row["gate"],
            row["overlay"],
            row["signal"],
            row["portfolio"],
            stressed,
        )
        row_candidates_six = candidates_for("six", row["gate"], row["overlay"])
        row_candidates_three = candidates_for("three", row["gate"], row["overlay"])
        delay_six = simulate_direct(
            "six",
            _shift_candidates(row_candidates_six, 1, execution_data),
            row["signal"],
            row["portfolio"],
        )
        delay_three = simulate_direct(
            "three",
            _shift_candidates(row_candidates_three, 1, execution_data),
            row["signal"],
            row["portfolio"],
        )
        fixed_portfolio = replace(row["portfolio"], compound=False)
        fixed_six = simulate(
            "six", row["gate"], row["overlay"], row["signal"], fixed_portfolio
        )
        fixed_three = simulate(
            "three", row["gate"], row["overlay"], row["signal"], fixed_portfolio
        )
        top_symbol_six = max(
            row["six"]["by_symbol"],
            key=lambda symbol: row["six"]["by_symbol"][symbol]["net_pnl"],
            default="",
        )
        top_symbol_three = max(
            row["three"]["by_symbol"],
            key=lambda symbol: row["three"]["by_symbol"][symbol]["net_pnl"],
            default="",
        )
        no_top_six = simulate_direct(
            "six",
            row_candidates_six,
            row["signal"],
            row["portfolio"],
            skip_symbols=frozenset({top_symbol_six}) if top_symbol_six else frozenset(),
        )
        no_top_three = simulate_direct(
            "three",
            row_candidates_three,
            row["signal"],
            row["portfolio"],
            skip_symbols=(
                frozenset({top_symbol_three}) if top_symbol_three else frozenset()
            ),
        )
        anchor_equivalent = (
            abs(row["six"]["net_profit"] - baseline_six["net_profit"]) <= 1e-6
            and abs(row["three"]["net_profit"] - baseline_three["net_profit"])
            <= 1e-6
            and abs(
                row["six"]["max_drawdown_pct"]
                - baseline_six["max_drawdown_pct"]
            )
            <= 1e-9
            and abs(
                row["three"]["max_drawdown_pct"]
                - baseline_three["max_drawdown_pct"]
            )
            <= 1e-9
        )
        robust = (
            (row["strict_improvement"] or anchor_equivalent)
            and early["net_profit"] > 0.0
            and _value(early["profit_factor"]) > 1.0
            and stress_six["net_profit"] > 0.0
            and stress_three["net_profit"] > 0.0
            and _value(stress_six["profit_factor"]) > 1.20
            and _value(stress_three["profit_factor"]) > 1.20
            and delay_six["net_profit"] > 0.0
            and delay_three["net_profit"] > 0.0
            and _value(delay_six["profit_factor"]) > 1.20
            and _value(delay_three["profit_factor"]) > 1.20
            and fixed_six["net_profit"] > 0.0
            and fixed_three["net_profit"] > 0.0
            and _value(fixed_six["profit_factor"]) > 1.20
            and _value(fixed_three["profit_factor"]) > 1.20
            and no_top_six["net_profit"] > 0.0
            and no_top_three["net_profit"] > 0.0
        )
        row.update(
            {
                "early": early,
                "stress_six": stress_six,
                "stress_three": stress_three,
                "delay_six": delay_six,
                "delay_three": delay_three,
                "fixed_six": fixed_six,
                "fixed_three": fixed_three,
                "no_top_six": no_top_six,
                "no_top_three": no_top_three,
                "top_symbol_six": top_symbol_six,
                "top_symbol_three": top_symbol_three,
                "anchor_equivalent": anchor_equivalent,
                "robust": robust,
                "robust_score": row["pair_score"] + (10.0 if robust else 0.0),
            }
        )
        robust_rows.append(row)
        print(
            f"robust {number}/{args.robust_finalists}: "
            f"strict={row['strict_improvement']} early={early['net_profit']:+.1f} "
            f"stress6={stress_six['net_profit']:+.1f} "
            f"delay6={delay_six['net_profit']:+.1f} "
            f"fixed6={fixed_six['net_profit']:+.1f} "
            f"noTop6={no_top_six['net_profit']:+.1f}",
            flush=True,
        )
    selected = max(
        [row for row in robust_rows if row["robust"]]
        or robust_rows
        or recent,
        key=lambda row: row.get("robust_score", row["pair_score"]),
    )

    def public(row: dict[str, Any], full: bool = False) -> dict[str, Any]:
        payload = {
            "pair_score": row["pair_score"],
            "strict_improvement": row.get("strict_improvement", False),
            "anchor_equivalent": row.get("anchor_equivalent", False),
            "robust": row.get("robust", False),
            "entry_gate": row["gate"].as_dict(),
            "market_overlay": asdict(row["overlay"]),
            "signal": row["signal"].as_dict(),
            "portfolio": asdict(row["portfolio"]),
            "six_month": _compact(row["six"]),
            "three_month": _compact(row["three"]),
        }
        for source, target in (
            ("early", "early_three_month"),
            ("stress_six", "stress_six_month"),
            ("stress_three", "stress_three_month"),
            ("delay_six", "entry_delay_one_minute_six_month"),
            ("delay_three", "entry_delay_one_minute_three_month"),
            ("fixed_six", "fixed_risk_six_month"),
            ("fixed_three", "fixed_risk_three_month"),
            ("no_top_six", "without_top_symbol_six_month"),
            ("no_top_three", "without_top_symbol_three_month"),
        ):
            if source in row:
                payload[target] = _compact(row[source])
        if "top_symbol_six" in row:
            payload["removed_top_symbol_six_month"] = row["top_symbol_six"]
            payload["removed_top_symbol_three_month"] = row["top_symbol_three"]
        if full:
            payload["six_month_full"] = row["six"]
            payload["three_month_full"] = row["three"]
        return payload

    selection_status = (
        "strict_robust_improvement"
        if selected.get("robust") and selected.get("strict_improvement")
        else "frozen_anchor_robust"
        if selected.get("robust") and selected.get("anchor_equivalent")
        else "best_available_requires_further_optimization"
    )
    report = {
        "strategy_name": TREND_GRID_V4_NAME,
        "status": "independent_research_gui_grid_v3_apt_grid_unchanged",
        "selection_status": selection_status,
        "periods": {
            "six_month": [start_six.isoformat(), end.isoformat()],
            "three_month": [start_three.isoformat(), end.isoformat()],
            "early_three_month": [start_six.isoformat(), start_three.isoformat()],
        },
        "initial_equity": args.initial_equity,
        "symbols": list(symbols),
        "data_quality": metadata,
        "cost_model": {
            "mode": "conservative_full_cost_point_in_time",
            "market_slippage_bps": execution.market_slippage_bps,
            "stop_slippage_bps": execution.stop_slippage_bps,
            "take_profit_slippage_bps": execution.take_profit_slippage_bps,
            "maker_fee_rate": execution.maker_fee_rate,
            "taker_fee_rate": execution.taker_fee_rate,
            "funding_enabled": execution.funding_enabled,
            "stress_multiplier": 1.5,
        },
        "search": {
            "seed": args.seed,
            "gate_evaluations": len(gate_rows),
            "overlay_evaluations": len(overlay_rows),
            "exit_evaluations": len(exit_rows),
            "portfolio_evaluations": len(portfolio_rows),
            "robust_evaluations": len(robust_rows),
        },
        "baseline": {
            "strategy": grid_payload.get("strategy_name", "Grid frozen anchor"),
            "source_report": args.baseline_report,
            "six_month": _compact(baseline_six),
            "three_month": _compact(baseline_three),
            "six_month_full": baseline_six,
            "three_month_full": baseline_three,
        },
        "selected": public(selected, True),
        "leaderboard": [public(row) for row in recent[:20]],
        "preserved": {
            "active_gui": "unchanged",
            "grid_v3": "unchanged",
            "apt_grid": "unchanged",
            "breakout_versions": "unchanged",
        },
    }
    _write_json(args.output, report)
    config = {
        "strategy_name": TREND_GRID_V4_NAME,
        "status": "frozen_independent_research_candidate_not_live",
        "selection_status": selection_status,
        "entry_gate": selected["gate"].as_dict(),
        "market_overlay": asdict(selected["overlay"]),
        "signal": selected["signal"].as_dict(),
        "portfolio": asdict(selected["portfolio"]),
        "results": {
            "six_month": _compact(selected["six"]),
            "three_month": _compact(selected["three"]),
        },
    }
    _write_json(args.config_output, config)
    lines = [
        "# Grid v4 live-robust optimization",
        "",
        "- Gap-free 1m execution; full fees, slippage and funding; point-in-time context",
        "- Independent 3m/6m warmups plus early-window and 1.5x-cost validation",
        "- GUI, Grid v3, APT Grid and Breakout versions remain unchanged",
        f"- Selection status: `{selection_status}`",
        "",
        "| Period | Version | Trades | Net | PF | Win | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for period, baseline_result, selected_result in (
        ("3 months", baseline_three, selected["three"]),
        ("6 months", baseline_six, selected["six"]),
    ):
        for name, result in (
            ("Grid frozen anchor", baseline_result),
            ("Grid v4 candidate", selected_result),
        ):
            lines.append(
                f"| {period} | {name} | {result['trade_count']} | "
                f"{result['net_profit']:+.2f}U | {result['profit_factor']:.3f} | "
                f"{result['win_rate']:.2%} | {result['max_drawdown_pct']:.2%} |"
            )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "strategy_name": TREND_GRID_V4_NAME,
        "status": "independent_research_artifacts",
        "config": args.config_output,
        "report": args.output,
        "summary": args.summary,
        "hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                "crypto_scalper/trend_grid_v4.py",
                "crypto_scalper/trend_grid_v4_optimize.py",
                "tests/test_trend_grid_v4.py",
                args.config_output,
                args.output,
                args.summary,
            )
        },
        "preserved": report["preserved"],
    }
    _write_json(args.manifest, manifest)
    print(
        f"selected strict={selected.get('strict_improvement', False)} "
        f"robust={selected.get('robust', False)} "
        f"6m={selected['six']['net_profit']:+.2f}/PF{selected['six']['profit_factor']:.3f}/"
        f"win{selected['six']['win_rate']:.1%}/DD{selected['six']['max_drawdown_pct']:.1%}; "
        f"3m={selected['three']['net_profit']:+.2f}/PF{selected['three']['profit_factor']:.3f}/"
        f"win{selected['three']['win_rate']:.1%}/DD{selected['three']['max_drawdown_pct']:.1%}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid v4 dual-window live-realistic optimization"
    )
    parser.add_argument("--start-6m", default="2026-01-19T00:00:00")
    parser.add_argument("--start-3m", default="2026-04-19T00:00:00")
    parser.add_argument("--end", default="2026-07-19T00:00:00")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument(
        "--breakout-config",
        default="config.volatility-breakout.hybrid-v5-balanced-expansion-runner-50.json",
    )
    parser.add_argument(
        "--grid-config", default="config.trend-grid.v3-optimized-50.json"
    )
    parser.add_argument(
        "--baseline-report",
        default="reports/combined_hybrid_v5_grid_v3_max2_3m_6m.json",
    )
    parser.add_argument(
        "--cost-config", default="config.volatility-breakout.v2-balanced-50-shadow.json"
    )
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
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--gate-budget", type=int, default=56)
    parser.add_argument("--gate-recent-finalists", type=int, default=24)
    parser.add_argument("--gate-finalists", type=int, default=4)
    parser.add_argument("--overlay-budget", type=int, default=20)
    parser.add_argument("--overlay-recent-finalists", type=int, default=28)
    parser.add_argument("--overlay-finalists", type=int, default=4)
    parser.add_argument("--exit-budget", type=int, default=24)
    parser.add_argument("--exit-recent-finalists", type=int, default=28)
    parser.add_argument("--exit-finalists", type=int, default=4)
    parser.add_argument("--portfolio-budget", type=int, default=20)
    parser.add_argument("--portfolio-recent-finalists", type=int, default=24)
    parser.add_argument("--robust-finalists", type=int, default=8)
    parser.add_argument(
        "--output", default="reports/trend_grid_v4_live_robust_3m_6m.json"
    )
    parser.add_argument(
        "--summary", default="reports/trend_grid_v4_live_robust_3m_6m.md"
    )
    parser.add_argument(
        "--config-output", default="config.trend-grid.v4-live-robust-50.json"
    )
    parser.add_argument(
        "--manifest", default="config.trend-grid.v4-live-robust-50-manifest.json"
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

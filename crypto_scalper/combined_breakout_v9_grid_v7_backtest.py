from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from .combined_breakout_v7_grid_v5_backtest import (
    _breakout_entry_from_dict,
    _build_breakout_candidates,
    _eligible_grid_for_combined,
)
from .combined_breakout_v8_grid_v6_backtest import _breakout_governor
from .combined_hybrid_v5_grid_v3_backtest import (
    _compact_combined,
    _slice_signal_data,
    _write_json,
)
from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
    CombinedPortfolioConfig,
)
from .combined_volatility_trend_grid_v4_backtest import (
    simulate_combined_v4_portfolio,
)
from .models import Direction
from .risk import BacktestExecutionConfig
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import (
    GridCandidate,
    GridPortfolioConfig,
    build_grid_research_timeline,
    compact_grid_summary,
)
from .trend_grid_v3_optimize import GridMarketOverlay, apply_market_overlay
from .trend_grid_v4 import GridV4EntryGate, filter_grid_v4_candidates
from .trend_grid_v5 import GridV5ConfidencePolicy
from .trend_grid_v5_engine import (
    GridV5ExecutionProfile,
    simulate_grid_v5_portfolio,
)
from .trend_grid_v6 import GridV6CampaignPolicy, grid_v6_entry_decision
from .trend_grid_v7 import TREND_GRID_V7_NAME
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import ExitProtectionConfig
from .volatility_breakout_optimize import (
    Candidate,
    PortfolioSearchConfig,
    UNIVERSE_50,
    _candidate_sort_key,
    sha256_file,
)
from .volatility_breakout_v4_research import (
    V4MarketSnapshot,
    _regime_score,
    build_v4_market_context,
    load_v4_runtime_inputs,
)
from .volatility_breakout_v6_core_runner_optimize import ManagedLaneProfile
from .volatility_breakout_v6_engine import (
    BreakoutV6ExecutionProfile,
    simulate_v6_managed_portfolio,
)
from .volatility_breakout_v6_optimize import compact_v6_metrics
from .volatility_breakout_v7 import (
    BreakoutV7ConfidencePolicy,
    BreakoutV7EntryTiming,
)
from .volatility_breakout_v8 import (
    BreakoutV8ScoreAllocation,
    breakout_v8_risk_multiplier,
)
from .volatility_breakout_v9 import VOLATILITY_BREAKOUT_V9_NAME


COMBINED_V9_GRID_V7_NAME = (
    "dual_thrust_volatility_breakout_v9_profit_ladder_plus_"
    "dynamic_trend_following_grid_v7_cycle_profit_floor_max2"
)
COMBINED_V9_GRID_V7_VERSION = (
    "breakout_v9_grid_v7_shared_max2_backtest_20260728"
)


def _base_exit_config(profile: ManagedLaneProfile) -> ExitProtectionConfig:
    return ExitProtectionConfig(
        breakeven_trigger_r=profile.core_breakeven_trigger_r,
        profit_giveback_activation_r=(
            profile.core_profit_giveback_activation_r
        ),
        profit_giveback_r=profile.core_profit_giveback_r,
        profit_floor_1_activation_r=(
            profile.core_profit_floor_1_activation_r
        ),
        profit_floor_1_lock_r=profile.core_profit_floor_1_lock_r,
        profit_floor_2_activation_r=(
            profile.core_profit_floor_2_activation_r
        ),
        profit_floor_2_lock_r=profile.core_profit_floor_2_lock_r,
        profit_floor_3_activation_r=(
            profile.core_profit_floor_3_activation_r
        ),
        profit_floor_3_lock_r=profile.core_profit_floor_3_lock_r,
        partial_take_profit_r=profile.core_partial_r,
        partial_take_profit_fraction=profile.core_partial_fraction,
        move_stop_to_breakeven_after_partial=(
            profile.core_move_breakeven_after_partial
        ),
    )


def _without_capture(config: ExitProtectionConfig) -> ExitProtectionConfig:
    return replace(
        config,
        profit_floor_1_activation_r=0.0,
        profit_floor_1_lock_r=0.0,
        profit_floor_2_activation_r=0.0,
        profit_floor_2_lock_r=0.0,
        profit_floor_3_activation_r=0.0,
        profit_floor_3_lock_r=0.0,
        partial_take_profit_r=0.0,
        partial_take_profit_fraction=0.0,
        move_stop_to_breakeven_after_partial=False,
    )


def _add_profit_floor(
    config: ExitProtectionConfig,
    activation_r: float,
    lock_r: float,
) -> ExitProtectionConfig:
    levels = {
        float(activation): float(lock)
        for activation, lock in config.profit_floor_levels
    }
    levels[float(activation_r)] = max(
        float(lock_r), levels.get(float(activation_r), 0.0)
    )
    ordered: list[tuple[float, float]] = []
    prior_lock = 0.0
    for activation, lock in sorted(levels.items()):
        prior_lock = max(prior_lock, lock)
        ordered.append((activation, prior_lock))
    ordered = ordered[:3]
    padded = [*ordered, *((0.0, 0.0),) * (3 - len(ordered))]
    return replace(
        config,
        profit_floor_1_activation_r=padded[0][0],
        profit_floor_1_lock_r=padded[0][1],
        profit_floor_2_activation_r=padded[1][0],
        profit_floor_2_lock_r=padded[1][1],
        profit_floor_3_activation_r=padded[2][0],
        profit_floor_3_lock_r=padded[2][1],
    )


def build_v9_v7_configs(
    breakout_path: str | Path,
    grid_path: str | Path,
    shared_path: str | Path,
) -> dict[str, Any]:
    breakout_payload = json.loads(
        Path(breakout_path).read_text(encoding="utf-8")
    )
    grid_payload = json.loads(Path(grid_path).read_text(encoding="utf-8"))
    shared_payload = json.loads(Path(shared_path).read_text(encoding="utf-8"))
    if breakout_payload.get("strategy_name") != VOLATILITY_BREAKOUT_V9_NAME:
        raise RuntimeError("Breakout source is not the selected v9 strategy")
    if grid_payload.get("strategy_name") != TREND_GRID_V7_NAME:
        raise RuntimeError("Grid source is not the selected v7 strategy")
    for name, payload in (
        ("Breakout v9", breakout_payload),
        ("Grid v7", grid_payload),
    ):
        if not str(payload.get("selection_status", "")).startswith(
            "strict_robust_improvement"
        ):
            raise RuntimeError(f"{name} did not clear its frozen anchor")

    breakout_entry = _breakout_entry_from_dict(breakout_payload["entry"])
    breakout_confidence = BreakoutV7ConfidencePolicy(
        **breakout_payload["confidence_policy"]
    )
    breakout_allocation = BreakoutV8ScoreAllocation(
        **breakout_payload["score_allocation"]
    )
    breakout_timing = BreakoutV7EntryTiming(
        **breakout_payload["entry_timing"]
    )
    breakout_managed = ManagedLaneProfile(
        **breakout_payload["managed_profile"]
    )
    breakout_signal = VolatilityBreakoutConfig(
        **breakout_payload["operational_signal"]
    )
    breakout_build_signal = replace(
        breakout_signal, max_signals_per_symbol_day=24
    )
    breakout_portfolio = PortfolioSearchConfig(
        **breakout_payload["operational_portfolio"]
    )
    breakout_operational_portfolio = replace(
        breakout_portfolio, ranking_mode=breakout_managed.ranking_mode
    )
    breakout_managed_signal = replace(
        breakout_signal,
        stop_atr_multiple=breakout_managed.core_stop_atr,
        take_profit_r=60.0,
        max_holding_minutes=breakout_managed.core_max_holding_minutes,
        fail_fast_minutes=breakout_managed.core_fail_fast_minutes,
        fail_fast_min_mfe_r=breakout_managed.core_fail_fast_min_mfe_r,
        fail_fast_max_current_r=(
            breakout_managed.core_fail_fast_max_current_r
        ),
    )
    breakout_exit = _base_exit_config(breakout_managed)

    grid_entry = GridV4EntryGate(**grid_payload["entry_gate"])
    grid_overlay = GridMarketOverlay(**grid_payload["market_overlay"])
    grid_signal = TrendGridConfig(**grid_payload["signal"])
    grid_portfolio = GridPortfolioConfig(**grid_payload["portfolio"])
    grid_confidence = GridV5ConfidencePolicy(
        **grid_payload["confidence_policy"]
    )
    grid_campaign = GridV6CampaignPolicy(
        **grid_payload["campaign_policy"]
    )
    combined = CombinedPortfolioConfig.from_dict(shared_payload["portfolio"])

    breakout_confidence.validate()
    breakout_allocation.validate()
    breakout_timing.validate()
    breakout_exit.validate()
    grid_entry.validate()
    grid_overlay.validate()
    grid_confidence.validate()
    grid_campaign.validate()
    combined.validate()
    if breakout_portfolio.max_open_positions != 1:
        raise RuntimeError("Breakout v9 must remain max one position")
    if grid_portfolio.max_open_campaigns != 1:
        raise RuntimeError("Grid v7 must remain max one campaign")
    if combined.max_open_positions != 2:
        raise RuntimeError("Shared v9/v7 account must remain max two positions")
    if combined.allow_same_symbol_across_strategies:
        raise RuntimeError("Same-symbol cross-strategy overlap must be disabled")

    return {
        "breakout_payload": breakout_payload,
        "breakout_entry": breakout_entry,
        "breakout_confidence": breakout_confidence,
        "breakout_allocation": breakout_allocation,
        "breakout_timing": breakout_timing,
        "breakout_managed": breakout_managed,
        "breakout_signal": breakout_signal,
        "breakout_build_signal": breakout_build_signal,
        "breakout_managed_signal": breakout_managed_signal,
        "breakout_portfolio": breakout_portfolio,
        "breakout_operational_portfolio": breakout_operational_portfolio,
        "breakout_exit": breakout_exit,
        "grid_payload": grid_payload,
        "grid_entry": grid_entry,
        "grid_overlay": grid_overlay,
        "grid_signal": grid_signal,
        "grid_portfolio": grid_portfolio,
        "grid_confidence": grid_confidence,
        "grid_campaign": grid_campaign,
        "combined": combined,
    }


def _breakout_selector(
    configs: dict[str, Any],
    context: dict[int, dict[str, V4MarketSnapshot]],
) -> Callable[
    [Candidate, int, float], Optional[BreakoutV6ExecutionProfile]
]:
    profile: ManagedLaneProfile = configs["breakout_managed"]
    confidence: BreakoutV7ConfidencePolicy = configs[
        "breakout_confidence"
    ]
    allocation: BreakoutV8ScoreAllocation = configs[
        "breakout_allocation"
    ]
    signal: VolatilityBreakoutConfig = configs["breakout_managed_signal"]
    base_portfolio: PortfolioSearchConfig = configs["breakout_portfolio"]
    capture_exit = configs["breakout_exit"]
    no_capture_exit = _without_capture(capture_exit)

    def choose(
        candidate: Candidate, minute: int, _equity: float
    ) -> Optional[BreakoutV6ExecutionProfile]:
        snapshot = context.get(minute - minute % 60, {}).get(
            candidate.signal.symbol
        )
        regime = (
            _regime_score(snapshot, candidate.signal.direction)
            if snapshot is not None
            else 0.0
        )
        alignment = (
            float(candidate.signal.direction.value) * snapshot.symbol_ema55_atr
            if snapshot is not None
            else -999.0
        )
        risk = profile.core_risk_pct
        if (
            regime >= profile.strong_regime_score
            and alignment >= profile.strong_alignment_atr
        ):
            risk = profile.core_strong_risk_pct
        elif regime < profile.weak_regime_score:
            risk *= profile.weak_risk_multiplier
        multiplier, score, lane = breakout_v8_risk_multiplier(
            candidate, snapshot, confidence, allocation
        )
        if multiplier <= 0.0:
            return None
        capture_side = (
            profile.core_profit_capture_long
            if candidate.signal.direction == Direction.LONG
            else profile.core_profit_capture_short
        )
        capture_score = (
            profile.core_profit_capture_min_score
            <= score
            <= profile.core_profit_capture_max_score
        )
        selected_exit = (
            capture_exit
            if capture_side and capture_score
            else no_capture_exit
        )
        if (
            candidate.signal.direction == Direction.LONG
            and profile.core_long_profit_floor_activation_r > 0.0
            and profile.core_long_profit_floor_min_score
            <= score
            <= profile.core_long_profit_floor_max_score
        ):
            selected_exit = _add_profit_floor(
                selected_exit,
                profile.core_long_profit_floor_activation_r,
                profile.core_long_profit_floor_lock_r,
            )
        result = BreakoutV6ExecutionProfile(
            lane=f"v9_{lane}",
            signal=signal,
            portfolio=replace(
                base_portfolio,
                risk_per_trade_pct=risk * multiplier,
                long_risk_multiplier=profile.core_long_multiplier,
                short_risk_multiplier=profile.core_short_multiplier,
                ranking_mode=profile.ranking_mode,
            ),
            exit_protection=selected_exit,
        )
        result.validate()
        return result

    return choose


def _grid_selector(
    configs: dict[str, Any],
    context: dict[int, dict[str, V4MarketSnapshot]],
) -> Callable[
    [GridCandidate, int, float], Optional[GridV5ExecutionProfile]
]:
    confidence: GridV5ConfidencePolicy = configs["grid_confidence"]
    policy: GridV6CampaignPolicy = configs["grid_campaign"]
    base_signal: TrendGridConfig = configs["grid_signal"]
    base_portfolio: GridPortfolioConfig = configs["grid_portfolio"]

    def choose(
        candidate: GridCandidate, minute: int, _equity: float
    ) -> Optional[GridV5ExecutionProfile]:
        snapshot = context.get(minute - minute % 60, {}).get(
            candidate.signal.symbol
        )
        accepted, tier, score, factor = grid_v6_entry_decision(
            candidate, snapshot, confidence, policy
        )
        if not accepted:
            return None
        v5_risk = (
            confidence.strong_risk_multiplier
            if tier == "strong"
            else (
                confidence.weak_risk_multiplier
                if tier == "weak"
                else confidence.standard_risk_multiplier
            )
        )
        return GridV5ExecutionProfile(
            tier=f"v7_{tier}_score_{score}",
            signal=replace(
                base_signal,
                grid_target_spacing=policy.target_spacing,
                campaign_loss_limit_r=policy.campaign_loss_limit_r,
                max_campaign_minutes=policy.max_campaign_minutes,
                campaign_take_profit_r=policy.campaign_take_profit_r,
                profit_lock_activation_r=policy.profit_lock_activation_r,
                profit_giveback_r=policy.profit_giveback_r,
                cycle_profit_floor_min_take_profits=(
                    policy.cycle_profit_floor_min_take_profits
                ),
                cycle_profit_floor_activation_r=(
                    policy.cycle_profit_floor_activation_r
                ),
                cycle_profit_floor_r=policy.cycle_profit_floor_r,
            ),
            portfolio=replace(
                base_portfolio,
                risk_per_campaign_pct=(
                    base_portfolio.risk_per_campaign_pct * v5_risk * factor
                ),
                max_campaign_risk_pct=policy.maximum_campaign_risk_pct,
                max_notional_multiple=(
                    policy.maximum_notional_multiple
                    if policy.maximum_notional_multiple > 0.0
                    else base_portfolio.max_notional_multiple
                ),
            ),
        )

    return choose


def _assert_match(
    name: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if int(actual["trade_count"]) != int(expected["trade_count"]):
        raise RuntimeError(f"{name} trade count mismatch")
    for key in ("net_profit", "max_drawdown_pct"):
        if not math.isclose(
            float(actual[key]),
            float(expected[key]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"{name} {key} mismatch: {actual[key]} != {expected[key]}"
            )
    def normalized_pf(values: dict[str, Any]) -> float:
        value = values.get("profit_factor")
        if value is None and float(values["net_profit"]) > 0.0:
            return math.inf
        return float(value or 0.0)

    actual_pf = normalized_pf(actual)
    expected_pf = normalized_pf(expected)
    if not (
        (math.isinf(actual_pf) and math.isinf(expected_pf))
        or math.isclose(
            actual_pf,
            expected_pf,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise RuntimeError(
            f"{name} profit factor mismatch: "
            f"{actual.get('profit_factor')} != "
            f"{expected.get('profit_factor')}"
        )


def _run_period(
    label: str,
    start: datetime,
    end: datetime,
    warmup_days: int,
    initial_equity: float,
    symbols: tuple[str, ...],
    all_signal_data: dict[str, list[Any]],
    execution_data: dict[str, Any],
    rules: dict[str, Any],
    execution: BacktestExecutionConfig,
    configs: dict[str, Any],
) -> dict[str, Any]:
    signal_data = _slice_signal_data(
        all_signal_data, start - timedelta(days=warmup_days), end
    )
    context = build_v4_market_context(symbols, signal_data)
    breakout_candidates = _build_breakout_candidates(
        symbols,
        signal_data,
        execution_data,
        context,
        configs,
        start,
        end,
    )
    raw_grid, grid_snapshots = build_grid_research_timeline(
        symbols,
        signal_data,
        execution_data,
        configs["grid_signal"],
        start,
        end,
    )
    grid_candidates = filter_grid_v4_candidates(
        apply_market_overlay(raw_grid, context, configs["grid_overlay"]),
        context,
        configs["grid_entry"],
    )
    breakout_selector = _breakout_selector(configs, context)
    grid_selector = _grid_selector(configs, context)
    eligible_grid, grid_signal, grid_portfolio, grid_rejected = (
        _eligible_grid_for_combined(grid_candidates, grid_selector)
    )
    print(
        f"{label}: breakout={sum(map(len, breakout_candidates.values()))} "
        f"grid={sum(map(len, grid_candidates.values()))} "
        f"grid_eligible={sum(map(len, eligible_grid.values()))}",
        flush=True,
    )

    standalone_breakout = simulate_v6_managed_portfolio(
        breakout_candidates,
        symbols,
        execution_data,
        rules,
        configs["breakout_signal"],
        configs["breakout_operational_portfolio"],
        ExitProtectionConfig(),
        execution,
        start,
        end,
        initial_equity,
        profile_selector=breakout_selector,
        priority_selector=lambda candidate: _candidate_sort_key(
            candidate, configs["breakout_managed"].ranking_mode
        ),
        profile_governor=_breakout_governor(configs, initial_equity),
    )
    standalone_grid = simulate_grid_v5_portfolio(
        grid_candidates,
        grid_snapshots,
        symbols,
        execution_data,
        rules,
        configs["grid_signal"],
        configs["grid_portfolio"],
        execution,
        start,
        end,
        initial_equity,
        profile_selector=grid_selector,
    )
    period_key = "three_month" if label == "3m" else "six_month"
    _assert_match(
        f"{label} Breakout v9",
        standalone_breakout,
        configs["breakout_payload"]["results"][period_key],
    )
    _assert_match(
        f"{label} Grid v7",
        standalone_grid,
        configs["grid_payload"]["results"][period_key],
    )

    def combined_for_priority(
        priority: tuple[str, ...],
    ) -> dict[str, Any]:
        peak = initial_equity
        govern = _breakout_governor(configs, initial_equity)

        def select_profile(
            candidate: Candidate, minute: int, equity: float
        ) -> Optional[BreakoutV6ExecutionProfile]:
            nonlocal peak
            selected = breakout_selector(candidate, minute, equity)
            if selected is None:
                return None
            peak = max(peak, equity)
            drawdown = (
                (peak - equity) / peak if peak > 0.0 else 1.0
            )
            return govern(selected, drawdown, minute, equity)

        return simulate_combined_v4_portfolio(
            breakout_candidates,
            eligible_grid,
            grid_snapshots,
            symbols,
            execution_data,
            rules,
            configs["breakout_managed_signal"],
            configs["breakout_operational_portfolio"],
            configs["breakout_exit"],
            grid_signal,
            grid_portfolio,
            replace(configs["combined"], entry_priority=priority),
            execution,
            start,
            end,
            initial_equity,
            breakout_profile_selector=select_profile,
        )

    combined = combined_for_priority(
        tuple(configs["combined"].entry_priority)
    )
    reverse = combined_for_priority((GRID_KEY, BREAKOUT_KEY))
    for name, result in (("primary", combined), ("reverse", reverse)):
        if result["max_concurrent_positions"] > 2:
            raise RuntimeError(f"{name} v9/v7 exceeded max two positions")
        if abs(float(result["pnl_reconciliation_error"])) > 1e-6:
            raise RuntimeError(f"{name} v9/v7 PnL reconciliation failed")
        result["strategy"] = COMBINED_V9_GRID_V7_NAME
    print(
        f"{label}: combined trades={combined['trade_count']} "
        f"net={combined['net_profit']:+.2f}U "
        f"PF={combined['profit_factor']:.3f} "
        f"win={combined['win_rate']:.2%} "
        f"DD={combined['max_drawdown_pct']:.2%}",
        flush=True,
    )
    return {
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "standalone": {
            BREAKOUT_KEY: compact_v6_metrics(standalone_breakout),
            GRID_KEY: compact_grid_summary(standalone_grid),
        },
        "combined": _compact_combined(combined),
        "reverse_priority": _compact_combined(reverse),
        "grid_v7_profile_rejected_before_arbitration": grid_rejected,
        "full_combined_result": combined,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    end = datetime.fromisoformat(args.end)
    start_three = datetime.fromisoformat(args.start_3m)
    start_six = datetime.fromisoformat(args.start_6m)
    if not start_six < start_three < end:
        raise ValueError("periods must satisfy start_6m < start_3m < end")
    symbols = tuple(UNIVERSE_50)
    data_start = start_six - timedelta(days=args.warmup_days)
    signal_data, execution_data, rules, execution, metadata = (
        load_v4_runtime_inputs(
            symbols,
            args.one_minute_roots,
            args.funding_roots,
            args.cost_config,
            data_start,
            end,
        )
    )
    if (
        metadata["minimum_coverage_ratio"] < 0.999999
        or metadata["maximum_missing_minutes"]
        or metadata["funding_missing_symbols"]
    ):
        raise RuntimeError(
            "Combined v9/v7 requires gap-free price and funding data"
        )
    configs = build_v9_v7_configs(
        args.breakout_config, args.grid_config, args.shared_config
    )
    periods = {
        "three_month": _run_period(
            "3m",
            start_three,
            end,
            args.warmup_days,
            args.initial_equity,
            symbols,
            signal_data,
            execution_data,
            rules,
            execution,
            configs,
        ),
        "six_month": _run_period(
            "6m",
            start_six,
            end,
            args.warmup_days,
            args.initial_equity,
            symbols,
            signal_data,
            execution_data,
            rules,
            execution,
            configs,
        ),
    }
    baseline: dict[str, Any] = {}
    if args.baseline_report and Path(args.baseline_report).exists():
        source = json.loads(
            Path(args.baseline_report).read_text(encoding="utf-8")
        )
        baseline = {
            key: source["periods"][key]["combined"]
            for key in ("three_month", "six_month")
        }
    report = {
        "strategy_name": COMBINED_V9_GRID_V7_NAME,
        "version": COMBINED_V9_GRID_V7_VERSION,
        "status": "independent_shared_account_research_gui_unchanged",
        "initial_equity": args.initial_equity,
        "universe_size": len(symbols),
        "symbols": list(symbols),
        "data_period": {
            "warmup_start": data_start.isoformat(),
            "end": end.isoformat(),
        },
        "data_quality": metadata,
        "source_configs": {
            BREAKOUT_KEY: args.breakout_config,
            GRID_KEY: args.grid_config,
            "shared_account_rules": args.shared_config,
        },
        "source_hashes": {
            BREAKOUT_KEY: sha256_file(args.breakout_config),
            GRID_KEY: sha256_file(args.grid_config),
            "shared_account_rules": sha256_file(args.shared_config),
        },
        "cost_model": {
            "mode": "gap_free_1m_conservative_full_cost",
            "market_slippage_bps": execution.market_slippage_bps,
            "stop_slippage_bps": execution.stop_slippage_bps,
            "take_profit_slippage_bps": execution.take_profit_slippage_bps,
            "maker_fee_rate": execution.maker_fee_rate,
            "taker_fee_rate": execution.taker_fee_rate,
            "funding_enabled": execution.funding_enabled,
        },
        "execution_rules": {
            "global_max_open_positions": 2,
            "breakout_max_open_positions": 1,
            "grid_max_open_campaigns": 1,
            "allow_same_symbol_across_strategies": False,
            "old_exits_before_new_entries": True,
            "same_bar_conflict": "adverse stop first",
            "entry_priority": list(configs["combined"].entry_priority),
            "compound": True,
            "breakout_exit_profile": "v9 side/score profile frozen at entry",
            "grid_campaign_profile": "v7 profile frozen at entry",
        },
        "baseline_breakout_v8_grid_v6": baseline,
        "periods": periods,
        "preserved": {
            "active_gui_config": "unchanged",
            "active_gui_state": "unchanged",
            "breakout_v8_source": "unchanged",
            "grid_v6_source": "unchanged",
            "apt_grid": "unchanged",
        },
    }
    _write_json(args.output, report)
    research_config = {
        "strategy_name": COMBINED_V9_GRID_V7_NAME,
        "version": COMBINED_V9_GRID_V7_VERSION,
        "status": "historical_research_only_active_gui_unchanged",
        "initial_equity": args.initial_equity,
        "source_configs": report["source_configs"],
        "source_hashes": report["source_hashes"],
        "portfolio": asdict(configs["combined"]),
        "execution_rules": report["execution_rules"],
        "results": {
            key: value["combined"] for key, value in periods.items()
        },
    }
    _write_json(args.research_config_output, research_config)

    def display_pf(metrics: dict[str, Any]) -> str:
        value = metrics.get("profit_factor")
        return "inf" if value is None else f"{float(value):.3f}"

    lines = [
        "# Breakout v9 + Grid v7 shared max2 backtest",
        "",
        f"- Initial equity: `{args.initial_equity:.2f}U`",
        f"- Universe: `{len(symbols)}` symbols",
        "- Gap-free 1m execution with fees, slippage and funding",
        "- Breakout max1 + Grid max1; shared account max2",
        "",
        "| Period | Mode | Trades | Net | PF | Win | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, period in periods.items():
        label = "3 months" if key == "three_month" else "6 months"
        for mode, metrics in (
            ("Breakout v9 standalone", period["standalone"][BREAKOUT_KEY]),
            ("Grid v7 standalone", period["standalone"][GRID_KEY]),
            ("v9 + v7 shared max2", period["combined"]),
        ):
            lines.append(
                f"| {label} | {mode} | {metrics['trade_count']} | "
                f"{metrics['net_profit']:+.2f}U | {display_pf(metrics)} | "
                f"{metrics['win_rate']:.2%} | "
                f"{metrics['max_drawdown_pct']:.2%} |"
            )
        old = baseline.get(key)
        if old:
            lines.append(
                f"| {label} | v8 + v6 baseline | {old['trade_count']} | "
                f"{old['net_profit']:+.2f}U | {display_pf(old)} | "
                f"{old['win_rate']:.2%} | "
                f"{old['max_drawdown_pct']:.2%} |"
            )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "strategy_name": COMBINED_V9_GRID_V7_NAME,
        "version": COMBINED_V9_GRID_V7_VERSION,
        "status": "independent_combined_research_artifacts",
        "report": args.output,
        "summary": args.summary,
        "research_config": args.research_config_output,
        "hashes": {
            artifact: sha256_file(artifact)
            for artifact in (
                "crypto_scalper/combined_volatility_trend_grid_v4_backtest.py",
                "crypto_scalper/combined_breakout_v9_grid_v7_backtest.py",
                args.output,
                args.summary,
                args.research_config_output,
            )
        },
        "preserved": report["preserved"],
    }
    _write_json(args.manifest, manifest)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Breakout v9 + Grid v7 shared max2 realistic backtest"
    )
    parser.add_argument("--start-6m", default="2026-01-19T00:00:00")
    parser.add_argument("--start-3m", default="2026-04-19T00:00:00")
    parser.add_argument("--end", default="2026-07-19T00:00:00")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument(
        "--breakout-config",
        default="config.volatility-breakout.v9-long-floor-refined-50.json",
    )
    parser.add_argument(
        "--grid-config",
        default="config.trend-grid.v7-drawdown-refined-50.json",
    )
    parser.add_argument(
        "--shared-config",
        default="config.combined-breakout-v7-grid-v5-max2.json",
    )
    parser.add_argument(
        "--baseline-report",
        default="reports/combined_breakout_v8_grid_v6_max2_3m_6m.json",
    )
    parser.add_argument(
        "--cost-config",
        default="config.volatility-breakout.v2-balanced-50-shadow.json",
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
    parser.add_argument(
        "--output",
        default="reports/combined_breakout_v9_grid_v7_max2_3m_6m.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/combined_breakout_v9_grid_v7_max2_3m_6m.md",
    )
    parser.add_argument(
        "--research-config-output",
        default="config.combined-breakout-v9-grid-v7-max2-backtest.json",
    )
    parser.add_argument(
        "--manifest",
        default=(
            "config.combined-breakout-v9-grid-v7-"
            "max2-backtest-manifest.json"
        ),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

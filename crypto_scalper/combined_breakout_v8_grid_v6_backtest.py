from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from .combined_breakout_v7_grid_v5_backtest import (
    UNIVERSE_100,
    _assert_standalone_match,
    _breakout_entry_from_dict,
    _build_breakout_candidates,
    _eligible_grid_for_combined,
)
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
from .trend_grid_v6 import (
    TREND_GRID_V6_NAME,
    GridV6CampaignPolicy,
    grid_v6_entry_decision,
)
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
from .volatility_breakout_v6_core_runner_optimize import (
    ManagedLaneProfile,
)
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
    VOLATILITY_BREAKOUT_V8_NAME,
    BreakoutV8ScoreAllocation,
    breakout_v8_risk_multiplier,
)


COMBINED_V8_GRID_V6_NAME = (
    "dual_thrust_volatility_breakout_v8_score_convex_plus_"
    "dynamic_trend_following_grid_v6_profit_protected_max2"
)
COMBINED_V8_GRID_V6_VERSION = (
    "breakout_v8_grid_v6_shared_max2_backtest_20260724"
)


def build_v8_v6_configs(
    breakout_path: str | Path,
    grid_path: str | Path,
    shared_path: str | Path,
) -> dict[str, Any]:
    breakout_payload = json.loads(
        Path(breakout_path).read_text(encoding="utf-8")
    )
    grid_payload = json.loads(Path(grid_path).read_text(encoding="utf-8"))
    shared_payload = json.loads(Path(shared_path).read_text(encoding="utf-8"))
    if breakout_payload.get("strategy_name") != VOLATILITY_BREAKOUT_V8_NAME:
        raise RuntimeError("Breakout source is not the selected v8 strategy")
    if grid_payload.get("strategy_name") != TREND_GRID_V6_NAME:
        raise RuntimeError("Grid source is not the selected v6 strategy")
    if not str(breakout_payload.get("selection_status", "")).startswith(
        "strict_robust_improvement"
    ):
        raise RuntimeError("Breakout v8 did not clear its frozen anchor")
    if not str(grid_payload.get("selection_status", "")).startswith(
        "strict_robust_improvement"
    ):
        raise RuntimeError("Grid v6 did not clear its frozen anchor")

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
    breakout_exit = ExitProtectionConfig(
        breakeven_trigger_r=breakout_managed.core_breakeven_trigger_r,
        profit_giveback_activation_r=(
            breakout_managed.core_profit_giveback_activation_r
        ),
        profit_giveback_r=breakout_managed.core_profit_giveback_r,
        partial_take_profit_r=breakout_managed.core_partial_r,
        partial_take_profit_fraction=breakout_managed.core_partial_fraction,
        move_stop_to_breakeven_after_partial=(
            breakout_managed.core_move_breakeven_after_partial
        ),
    )

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
        raise RuntimeError("Breakout v8 must remain max one position")
    if grid_portfolio.max_open_campaigns != 1:
        raise RuntimeError("Grid v6 must remain max one campaign")
    if combined.max_open_positions != 2:
        raise RuntimeError("Shared v8/v6 account must remain max two positions")
    if combined.allow_same_symbol_across_strategies:
        raise RuntimeError("Same-symbol cross-strategy overlap must stay disabled")

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
    managed: ManagedLaneProfile = configs["breakout_managed"]
    confidence: BreakoutV7ConfidencePolicy = configs[
        "breakout_confidence"
    ]
    allocation: BreakoutV8ScoreAllocation = configs[
        "breakout_allocation"
    ]
    signal: VolatilityBreakoutConfig = configs[
        "breakout_managed_signal"
    ]
    exit_config: ExitProtectionConfig = configs["breakout_exit"]
    base_portfolio: PortfolioSearchConfig = configs["breakout_portfolio"]

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
        risk = managed.core_risk_pct
        if (
            regime >= managed.strong_regime_score
            and alignment >= managed.strong_alignment_atr
        ):
            risk = managed.core_strong_risk_pct
        elif regime < managed.weak_regime_score:
            risk *= managed.weak_risk_multiplier
        multiplier, score, lane = breakout_v8_risk_multiplier(
            candidate, snapshot, confidence, allocation
        )
        if multiplier <= 0.0:
            return None
        profile = BreakoutV6ExecutionProfile(
            lane=f"v8_{lane}",
            signal=signal,
            portfolio=replace(
                base_portfolio,
                risk_per_trade_pct=risk * multiplier,
                long_risk_multiplier=managed.core_long_multiplier,
                short_risk_multiplier=managed.core_short_multiplier,
                ranking_mode=managed.ranking_mode,
            ),
            exit_protection=exit_config,
        )
        profile.validate()
        return profile

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
        if tier == "strong":
            v5_risk = confidence.strong_risk_multiplier
        elif tier == "weak":
            v5_risk = confidence.weak_risk_multiplier
        else:
            v5_risk = confidence.standard_risk_multiplier
        return GridV5ExecutionProfile(
            tier=f"v6_{tier}_score_{score}",
            signal=replace(
                base_signal,
                grid_target_spacing=policy.target_spacing,
                campaign_loss_limit_r=policy.campaign_loss_limit_r,
                max_campaign_minutes=policy.max_campaign_minutes,
                campaign_take_profit_r=policy.campaign_take_profit_r,
                profit_lock_activation_r=policy.profit_lock_activation_r,
                profit_giveback_r=policy.profit_giveback_r,
            ),
            portfolio=replace(
                base_portfolio,
                risk_per_campaign_pct=(
                    base_portfolio.risk_per_campaign_pct * v5_risk * factor
                ),
                max_campaign_risk_pct=policy.maximum_campaign_risk_pct,
            ),
        )

    return choose


def _breakout_governor(
    configs: dict[str, Any], initial_equity: float
) -> Callable[
    [BreakoutV6ExecutionProfile, float, int, float],
    BreakoutV6ExecutionProfile,
]:
    managed: ManagedLaneProfile = configs["breakout_managed"]
    monthly_key = ""
    monthly_peak = initial_equity

    def govern(
        profile: BreakoutV6ExecutionProfile,
        drawdown: float,
        minute: int,
        equity: float,
    ) -> BreakoutV6ExecutionProfile:
        nonlocal monthly_key, monthly_peak
        if managed.drawdown_scope == "monthly":
            from .volatility_breakout_optimize import minute_datetime

            key = minute_datetime(minute).strftime("%Y-%m")
            if key != monthly_key:
                monthly_key = key
                monthly_peak = equity
            else:
                monthly_peak = max(monthly_peak, equity)
            drawdown = (
                (monthly_peak - equity) / monthly_peak
                if monthly_peak > 0.0
                else 1.0
            )
        if drawdown >= managed.drawdown_deep_start:
            multiplier = managed.drawdown_deep_multiplier
        elif drawdown >= managed.drawdown_reduce_start:
            multiplier = managed.drawdown_reduce_multiplier
        else:
            multiplier = 1.0
        if multiplier >= 1.0:
            return profile
        return replace(
            profile,
            portfolio=replace(
                profile.portfolio,
                risk_per_trade_pct=(
                    profile.portfolio.risk_per_trade_pct * multiplier
                ),
            ),
        )

    return govern


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
    if symbols == tuple(UNIVERSE_50):
        _assert_standalone_match(
            f"{label} Breakout v8",
            standalone_breakout,
            configs["breakout_payload"]["results"][period_key],
        )
        _assert_standalone_match(
            f"{label} Grid v6",
            standalone_grid,
            configs["grid_payload"]["results"][period_key],
        )

    global_peak = initial_equity
    govern = _breakout_governor(configs, initial_equity)

    def combined_breakout_portfolio(
        candidate: Candidate, minute: int, equity: float
    ) -> PortfolioSearchConfig:
        nonlocal global_peak
        profile = breakout_selector(candidate, minute, equity)
        if profile is None:
            raise RuntimeError(
                "Selected Breakout v8 unexpectedly rejected a built candidate"
            )
        if (
            profile.signal != configs["breakout_managed_signal"]
            or profile.exit_protection != configs["breakout_exit"]
        ):
            raise RuntimeError("Breakout v8 entries must share one exit profile")
        global_peak = max(global_peak, equity)
        drawdown = (
            (global_peak - equity) / global_peak if global_peak > 0.0 else 1.0
        )
        return govern(profile, drawdown, minute, equity).portfolio

    combined = simulate_combined_v4_portfolio(
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
        configs["combined"],
        execution,
        start,
        end,
        initial_equity,
        breakout_portfolio_selector=combined_breakout_portfolio,
    )
    reverse_peak = initial_equity
    reverse_govern = _breakout_governor(configs, initial_equity)

    def reverse_breakout_portfolio(
        candidate: Candidate, minute: int, equity: float
    ) -> PortfolioSearchConfig:
        nonlocal reverse_peak
        profile = breakout_selector(candidate, minute, equity)
        if profile is None:
            raise RuntimeError("Reverse-priority Breakout v8 rejection")
        reverse_peak = max(reverse_peak, equity)
        drawdown = (
            (reverse_peak - equity) / reverse_peak
            if reverse_peak > 0.0
            else 1.0
        )
        return reverse_govern(profile, drawdown, minute, equity).portfolio

    reverse = simulate_combined_v4_portfolio(
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
        replace(configs["combined"], entry_priority=(GRID_KEY, BREAKOUT_KEY)),
        execution,
        start,
        end,
        initial_equity,
        breakout_portfolio_selector=reverse_breakout_portfolio,
    )
    if combined["max_concurrent_positions"] > 2:
        raise RuntimeError("Combined v8/v6 exceeded max two positions")
    if abs(float(combined["pnl_reconciliation_error"])) > 1e-6:
        raise RuntimeError("Combined v8/v6 PnL reconciliation failed")
    combined["strategy"] = COMBINED_V8_GRID_V6_NAME
    reverse["strategy"] = COMBINED_V8_GRID_V6_NAME
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
        "grid_v6_profile_rejected_before_arbitration": grid_rejected,
        "full_combined_result": combined,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    end = datetime.fromisoformat(args.end)
    start_three = datetime.fromisoformat(args.start_3m)
    start_six = datetime.fromisoformat(args.start_6m)
    if not start_six < start_three < end:
        raise ValueError("periods must satisfy start_6m < start_3m < end")
    symbols = (
        tuple(UNIVERSE_50)
        if args.universe_size == 50
        else tuple(UNIVERSE_100)
    )
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
            "Combined v8/v6 requires gap-free price and funding data"
        )
    configs = build_v8_v6_configs(
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
        "strategy_name": COMBINED_V8_GRID_V6_NAME,
        "version": COMBINED_V8_GRID_V6_VERSION,
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
        },
        "baseline_breakout_v7_grid_v5": baseline,
        "periods": periods,
        "preserved": {
            "active_gui_config": "unchanged",
            "active_gui_state": "unchanged",
            "breakout_v7_source": "unchanged",
            "grid_v5_source": "unchanged",
            "apt_grid": "unchanged",
        },
    }
    _write_json(args.output, report)
    research_config = {
        "strategy_name": COMBINED_V8_GRID_V6_NAME,
        "version": COMBINED_V8_GRID_V6_VERSION,
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

    lines = [
        "# Breakout v8 + Grid v6 shared max2 backtest",
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
            ("Breakout v8 standalone", period["standalone"][BREAKOUT_KEY]),
            ("Grid v6 standalone", period["standalone"][GRID_KEY]),
            ("v8 + v6 shared max2", period["combined"]),
        ):
            lines.append(
                f"| {label} | {mode} | {metrics['trade_count']} | "
                f"{metrics['net_profit']:+.2f}U | "
                f"{metrics['profit_factor']:.3f} | "
                f"{metrics['win_rate']:.2%} | "
                f"{metrics['max_drawdown_pct']:.2%} |"
            )
        old = baseline.get(key)
        if old:
            lines.append(
                f"| {label} | v7 + v5 baseline | {old['trade_count']} | "
                f"{old['net_profit']:+.2f}U | {old['profit_factor']:.3f} | "
                f"{old['win_rate']:.2%} | {old['max_drawdown_pct']:.2%} |"
            )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "strategy_name": COMBINED_V8_GRID_V6_NAME,
        "version": COMBINED_V8_GRID_V6_VERSION,
        "status": "independent_combined_research_artifacts",
        "report": args.output,
        "summary": args.summary,
        "research_config": args.research_config_output,
        "hashes": {
            artifact: sha256_file(artifact)
            for artifact in (
                "crypto_scalper/combined_breakout_v8_grid_v6_backtest.py",
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
        description="Breakout v8 + Grid v6 shared max2 realistic backtest"
    )
    parser.add_argument("--start-6m", default="2026-01-19T00:00:00")
    parser.add_argument("--start-3m", default="2026-04-19T00:00:00")
    parser.add_argument("--end", default="2026-07-19T00:00:00")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument("--universe-size", type=int, choices=(50, 100), default=50)
    parser.add_argument(
        "--breakout-config",
        default="config.volatility-breakout.v8-score-convex-refined-50.json",
    )
    parser.add_argument(
        "--grid-config",
        default="config.trend-grid.v6-profit-protected-50.json",
    )
    parser.add_argument(
        "--shared-config",
        default="config.combined-breakout-v7-grid-v5-max2.json",
    )
    parser.add_argument(
        "--baseline-report",
        default="reports/combined_breakout_v7_grid_v5_max2_3m_6m.json",
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
        default="reports/combined_breakout_v8_grid_v6_max2_3m_6m.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/combined_breakout_v8_grid_v6_max2_3m_6m.md",
    )
    parser.add_argument(
        "--research-config-output",
        default="config.combined-breakout-v8-grid-v6-max2-backtest.json",
    )
    parser.add_argument(
        "--manifest",
        default="config.combined-breakout-v8-grid-v6-max2-backtest-manifest.json",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from .combined_breakout_v7_grid_v5_shadow import COMBINED_V7_GRID_V5_NAME
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
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import (
    GridCandidate,
    GridPortfolioConfig,
    build_grid_research_timeline,
    compact_grid_summary,
)
from .trend_grid_v3_optimize import GridMarketOverlay, apply_market_overlay
from .trend_grid_v4 import GridV4EntryGate, filter_grid_v4_candidates
from .trend_grid_v5 import (
    TREND_GRID_V5_NAME,
    GridV5ConfidencePolicy,
    grid_v5_tier,
)
from .trend_grid_v5_engine import (
    GridV5ExecutionProfile,
    simulate_grid_v5_portfolio,
)
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import ExitProtectionConfig
from .volatility_breakout_optimize import (
    Candidate,
    PortfolioSearchConfig,
    UNIVERSE_50,
    _candidate_sort_key,
    build_candidates,
    minute_datetime,
    sha256_file,
)
from .volatility_breakout_v4_research import (
    V4MarketSnapshot,
    _regime_score,
    build_v4_market_context,
    enrich_candidates_v4,
    load_v4_runtime_inputs,
)
from .volatility_breakout_v6 import (
    BreakoutV6EntryConfig,
    BreakoutV6SideGate,
    filter_breakout_v6_candidates,
)
from .volatility_breakout_v6_core_runner_optimize import (
    ManagedLaneProfile,
    tiered_drawdown_risk_multiplier,
)
from .volatility_breakout_v6_engine import (
    BreakoutV6ExecutionProfile,
    simulate_v6_managed_portfolio,
)
from .volatility_breakout_v6_optimize import compact_v6_metrics
from .volatility_breakout_v7 import (
    VOLATILITY_BREAKOUT_V7_NAME,
    BreakoutV7ConfidencePolicy,
    BreakoutV7EntryTiming,
    apply_breakout_v7_timing,
    breakout_v7_risk_multiplier,
)


COMBINED_V7_GRID_V5_BACKTEST_VERSION = (
    "breakout_v7_grid_v5_shared_max2_backtest_20260723"
)

UNIVERSE_100: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "TRXUSDT", "HYPEUSDT", "DOGEUSDT", "XLMUSDT", "ZECUSDT",
    "ADAUSDT", "XMRUSDT", "LINKUSDT", "BCHUSDT", "TONUSDT",
    "HBARUSDT", "LTCUSDT", "AVAXUSDT", "SUIUSDT", "1000SHIBUSDT",
    "NEARUSDT", "TAOUSDT", "WLDUSDT", "ONDOUSDT", "DOTUSDT",
    "UNIUSDT", "ICPUSDT", "1000PEPEUSDT", "MORPHOUSDT", "ETCUSDT",
    "QNTUSDT", "AAVEUSDT", "DEXEUSDT", "ENAUSDT", "RENDERUSDT",
    "IMXUSDT", "ATOMUSDT", "KASUSDT", "ALGOUSDT", "POLUSDT",
    "JSTUSDT", "FILUSDT", "APTUSDT", "SIRENUSDT", "JUPUSDT",
    "INJUSDT", "ARBUSDT", "FETUSDT", "VETUSDT", "DASHUSDT",
    "PENGUUSDT", "CAKEUSDT", "TRUMPUSDT", "1000BONKUSDT",
    "VIRTUALUSDT", "SUNUSDT", "STXUSDT", "1000LUNCUSDT", "SEIUSDT",
    "AEROUSDT", "TIAUSDT", "CRVUSDT", "SPXUSDT", "XTZUSDT",
    "CHZUSDT", "ETHFIUSDT", "PYTHUSDT", "ZROUSDT", "JTOUSDT",
    "BSVUSDT", "CFXUSDT", "1000FLOKIUSDT", "JASMYUSDT", "BUSDT",
    "KAIAUSDT", "LDOUSDT", "GRTUSDT", "OPUSDT", "PENDLEUSDT",
    "STRKUSDT", "HOMEUSDT", "GRASSUSDT", "IOTAUSDT", "ENSUSDT",
    "LITUSDT", "AKTUSDT", "AXSUSDT", "TWTUSDT", "COMPUSDT",
    "NEOUSDT", "WIFUSDT", "THETAUSDT", "SYRUPUSDT", "SANDUSDT",
    "MANAUSDT", "ARUSDT", "BATUSDT", "APEUSDT", "GALAUSDT",
    "EIGENUSDT",
)


def _breakout_entry_from_dict(
    payload: dict[str, Any],
) -> BreakoutV6EntryConfig:
    result = BreakoutV6EntryConfig(
        long=BreakoutV6SideGate(**payload["long"]),
        short=BreakoutV6SideGate(**payload["short"]),
        max_signals_per_symbol_day=int(payload["max_signals_per_symbol_day"]),
        require_market_context=bool(payload["require_market_context"]),
    )
    result.validate()
    return result


def build_frozen_v7_v5_configs(
    breakout_path: str | Path,
    grid_path: str | Path,
    combined_path: str | Path,
) -> dict[str, Any]:
    breakout_payload = json.loads(
        Path(breakout_path).read_text(encoding="utf-8")
    )
    grid_payload = json.loads(Path(grid_path).read_text(encoding="utf-8"))
    combined_payload = json.loads(
        Path(combined_path).read_text(encoding="utf-8")
    )
    if breakout_payload.get("strategy_name") != VOLATILITY_BREAKOUT_V7_NAME:
        raise RuntimeError("Breakout source is not the frozen v7 strategy")
    if grid_payload.get("strategy_name") != TREND_GRID_V5_NAME:
        raise RuntimeError("Grid source is not the frozen v5 strategy")
    if combined_payload.get("strategy_name") != COMBINED_V7_GRID_V5_NAME:
        raise RuntimeError("combined source is not the v7/v5 GUI strategy")
    if breakout_payload.get("selection_status") != "strict_robust_improvement":
        raise RuntimeError("Breakout v7 source is not the selected candidate")
    if grid_payload.get("selection_status") != "strict_robust_improvement":
        raise RuntimeError("Grid v5 source is not the selected candidate")

    breakout_entry = _breakout_entry_from_dict(breakout_payload["entry"])
    breakout_policy = BreakoutV7ConfidencePolicy(
        **breakout_payload["confidence_policy"]
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
        breakout_portfolio,
        ranking_mode=breakout_managed.ranking_mode,
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

    grid_entry = GridV4EntryGate(**grid_payload["entry_gate"])
    grid_overlay = GridMarketOverlay(**grid_payload["market_overlay"])
    grid_signal = TrendGridConfig(**grid_payload["signal"])
    grid_portfolio = GridPortfolioConfig(**grid_payload["portfolio"])
    grid_policy = GridV5ConfidencePolicy(
        **grid_payload["confidence_policy"]
    )
    combined = CombinedPortfolioConfig.from_dict(
        combined_payload["portfolio"]
    )

    breakout_policy.validate()
    breakout_timing.validate()
    breakout_exit.validate()
    grid_entry.validate()
    grid_overlay.validate()
    grid_policy.validate()
    combined.validate()
    if breakout_portfolio.max_open_positions != 1:
        raise RuntimeError("Breakout v7 must remain max one position")
    if grid_portfolio.max_open_campaigns != 1:
        raise RuntimeError("Grid v5 must remain max one campaign")
    if combined.max_open_positions != 2:
        raise RuntimeError("combined v7/v5 must remain max two positions")
    if combined.allow_same_symbol_across_strategies:
        raise RuntimeError("combined v7/v5 must keep same-symbol exclusion")
    if combined.entry_priority != (BREAKOUT_KEY, GRID_KEY):
        raise RuntimeError("combined v7/v5 entry priority changed")
    if breakout_timing.confirmation_minutes != 0:
        raise RuntimeError("selected Breakout v7 timing is not immediate")
    if not grid_policy.reject_weak_tier:
        raise RuntimeError("selected Grid v5 must reject weak candidates")

    return {
        "breakout_payload": breakout_payload,
        "grid_payload": grid_payload,
        "combined_payload": combined_payload,
        "breakout_entry": breakout_entry,
        "breakout_policy": breakout_policy,
        "breakout_timing": breakout_timing,
        "breakout_managed": breakout_managed,
        "breakout_build_signal": breakout_build_signal,
        "breakout_signal": breakout_signal,
        "breakout_managed_signal": breakout_managed_signal,
        "breakout_portfolio": breakout_portfolio,
        "breakout_operational_portfolio": breakout_operational_portfolio,
        "breakout_exit": breakout_exit,
        "grid_entry": grid_entry,
        "grid_overlay": grid_overlay,
        "grid_signal": grid_signal,
        "grid_portfolio": grid_portfolio,
        "grid_policy": grid_policy,
        "combined": combined,
    }


def _breakout_profile_selector(
    configs: dict[str, Any],
    context: dict[int, dict[str, V4MarketSnapshot]],
) -> Callable[
    [Candidate, int, float], Optional[BreakoutV6ExecutionProfile]
]:
    managed: ManagedLaneProfile = configs["breakout_managed"]
    policy: BreakoutV7ConfidencePolicy = configs["breakout_policy"]
    signal: VolatilityBreakoutConfig = configs["breakout_managed_signal"]
    exit_config: ExitProtectionConfig = configs["breakout_exit"]
    base_portfolio: PortfolioSearchConfig = configs["breakout_portfolio"]

    def choose(
        candidate: Candidate,
        minute: int,
        _equity: float,
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
            float(candidate.signal.direction.value)
            * snapshot.symbol_ema55_atr
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
        multiplier, score, tier = breakout_v7_risk_multiplier(
            candidate, snapshot, policy
        )
        if multiplier <= 0.0:
            return None
        profile = BreakoutV6ExecutionProfile(
            lane=f"v7_{tier}_score_{score}",
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


def _breakout_profile_governor(
    configs: dict[str, Any],
    initial_equity: float,
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
            period_key = minute_datetime(minute).strftime("%Y-%m")
            if period_key != monthly_key:
                monthly_key = period_key
                monthly_peak = equity
            else:
                monthly_peak = max(monthly_peak, equity)
            drawdown = (
                (monthly_peak - equity) / monthly_peak
                if monthly_peak > 0.0
                else 1.0
            )
        multiplier = tiered_drawdown_risk_multiplier(
            drawdown,
            managed.drawdown_reduce_start,
            managed.drawdown_reduce_multiplier,
            managed.drawdown_deep_start,
            managed.drawdown_deep_multiplier,
        )
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


def _grid_profile_selector(
    configs: dict[str, Any],
    context: dict[int, dict[str, V4MarketSnapshot]],
) -> Callable[
    [GridCandidate, int, float], Optional[GridV5ExecutionProfile]
]:
    policy: GridV5ConfidencePolicy = configs["grid_policy"]
    base_signal: TrendGridConfig = configs["grid_signal"]
    base_portfolio: GridPortfolioConfig = configs["grid_portfolio"]

    def choose(
        candidate: GridCandidate,
        minute: int,
        _equity: float,
    ) -> Optional[GridV5ExecutionProfile]:
        snapshot = context.get(minute - minute % 60, {}).get(
            candidate.signal.symbol
        )
        tier, score = grid_v5_tier(candidate, snapshot, policy)
        if tier == "weak" and policy.reject_weak_tier:
            return None
        if tier == "strong":
            risk_multiplier = policy.strong_risk_multiplier
            target = policy.strong_target_spacing
            loss_limit = policy.strong_loss_limit_r
            duration = policy.strong_max_campaign_minutes
        elif tier == "weak":
            risk_multiplier = policy.weak_risk_multiplier
            target = policy.weak_target_spacing
            loss_limit = policy.weak_loss_limit_r
            duration = policy.weak_max_campaign_minutes
        else:
            risk_multiplier = policy.standard_risk_multiplier
            target = policy.standard_target_spacing
            loss_limit = policy.standard_loss_limit_r
            duration = policy.standard_max_campaign_minutes
        return GridV5ExecutionProfile(
            tier=f"{tier}_score_{score}",
            signal=replace(
                base_signal,
                grid_target_spacing=target,
                campaign_loss_limit_r=loss_limit,
                max_campaign_minutes=duration,
            ),
            portfolio=replace(
                base_portfolio,
                risk_per_campaign_pct=(
                    base_portfolio.risk_per_campaign_pct * risk_multiplier
                ),
                max_campaign_risk_pct=policy.maximum_campaign_risk_pct,
            ),
        )

    return choose


def _build_breakout_candidates(
    symbols: tuple[str, ...],
    signal_data: dict[str, list[Any]],
    execution_data: dict[str, Any],
    context: dict[int, dict[str, V4MarketSnapshot]],
    configs: dict[str, Any],
    start: datetime,
    end: datetime,
) -> dict[int, list[Candidate]]:
    raw = enrich_candidates_v4(
        build_candidates(
            symbols,
            signal_data,
            execution_data,
            configs["breakout_build_signal"],
            start,
            end,
        ),
        context,
    )
    filtered = filter_breakout_v6_candidates(
        raw, context, configs["breakout_entry"]
    )
    return apply_breakout_v7_timing(
        filtered, execution_data, configs["breakout_timing"]
    )


def _eligible_grid_for_combined(
    candidates: dict[int, list[GridCandidate]],
    selector: Callable[
        [GridCandidate, int, float], Optional[GridV5ExecutionProfile]
    ],
) -> tuple[
    dict[int, list[GridCandidate]],
    TrendGridConfig,
    GridPortfolioConfig,
    int,
]:
    """Collapse selected v5 tiers only when their execution is path-identical.

    The frozen Grid-v5 candidate uses the same target, loss limit and duration
    for its strong and standard tiers. Both tier risk paths cap at exactly 11%
    for its short-only sleeve. This invariant lets the existing audited shared
    account simulator reproduce the managed Grid-v5 fills without modifying
    either source strategy.
    """

    output: dict[int, list[GridCandidate]] = {}
    reference: GridV5ExecutionProfile | None = None
    reference_effective_risk: float | None = None
    rejected = 0
    for minute in sorted(candidates):
        selected: list[GridCandidate] = []
        for candidate in candidates[minute]:
            profile = selector(candidate, minute, 200.0)
            if profile is None:
                rejected += 1
                continue
            side_multiplier = (
                profile.portfolio.long_risk_multiplier
                if candidate.signal.direction.value > 0
                else profile.portfolio.short_risk_multiplier
            )
            effective_risk = min(
                profile.portfolio.max_campaign_risk_pct,
                profile.portfolio.risk_per_campaign_pct * side_multiplier,
            )
            if reference is None:
                reference = profile
                reference_effective_risk = effective_risk
            else:
                if profile.signal != reference.signal:
                    raise RuntimeError(
                        "Grid v5 accepted tiers no longer share one exit profile"
                    )
                left = asdict(profile.portfolio)
                right = asdict(reference.portfolio)
                for key in (
                    "risk_per_campaign_pct",
                    "long_risk_multiplier",
                    "short_risk_multiplier",
                ):
                    left.pop(key)
                    right.pop(key)
                if left != right or not math.isclose(
                    effective_risk,
                    float(reference_effective_risk),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        "Grid v5 accepted tiers need a managed combined engine"
                    )
            selected.append(candidate)
        if selected:
            output[minute] = selected
    if reference is None or reference_effective_risk is None:
        raise RuntimeError("Grid v5 period has no eligible combined candidate")
    portfolio = replace(
        reference.portfolio,
        risk_per_campaign_pct=reference_effective_risk,
        long_risk_multiplier=1.0,
        short_risk_multiplier=1.0,
    )
    return output, reference.signal, portfolio, rejected


def _assert_standalone_match(
    name: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if int(actual["trade_count"]) != int(expected["trade_count"]):
        raise RuntimeError(
            f"{name} trade count mismatch: "
            f"{actual['trade_count']} != {expected['trade_count']}"
        )
    for key in ("net_profit", "profit_factor", "max_drawdown_pct"):
        if not math.isclose(
            float(actual[key]),
            float(expected[key]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"{name} {key} mismatch: {actual[key]} != {expected[key]}"
            )


def _run_period(
    name: str,
    start: datetime,
    end: datetime,
    warmup_days: int,
    initial_equity: float,
    symbols: tuple[str, ...],
    all_signal_data: dict[str, list[Any]],
    execution_data: dict[str, Any],
    rules: dict[str, Any],
    execution: Any,
    configs: dict[str, Any],
) -> dict[str, Any]:
    period_signal_data = _slice_signal_data(
        all_signal_data, start - timedelta(days=warmup_days), end
    )
    context = build_v4_market_context(symbols, period_signal_data)
    breakout_candidates = _build_breakout_candidates(
        symbols,
        period_signal_data,
        execution_data,
        context,
        configs,
        start,
        end,
    )
    raw_grid, grid_snapshots = build_grid_research_timeline(
        symbols,
        period_signal_data,
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
    breakout_selector = _breakout_profile_selector(configs, context)
    grid_selector = _grid_profile_selector(configs, context)
    eligible_grid, combined_grid_signal, combined_grid_portfolio, weak_count = (
        _eligible_grid_for_combined(grid_candidates, grid_selector)
    )
    print(
        f"{name}: breakout candidates="
        f"{sum(map(len, breakout_candidates.values()))} "
        f"grid candidates={sum(map(len, grid_candidates.values()))} "
        f"grid eligible={sum(map(len, eligible_grid.values()))}",
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
        profile_governor=_breakout_profile_governor(
            configs, initial_equity
        ),
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
    period_key = "three_month" if name == "3m" else "six_month"
    if symbols == tuple(UNIVERSE_50):
        _assert_standalone_match(
            f"{name} Breakout v7",
            standalone_breakout,
            configs["breakout_payload"]["results"][period_key],
        )
        _assert_standalone_match(
            f"{name} Grid v5",
            standalone_grid,
            configs["grid_payload"]["results"][period_key],
        )

    managed: ManagedLaneProfile = configs["breakout_managed"]
    if (
        managed.drawdown_reduce_multiplier != 1.0
        or managed.drawdown_deep_multiplier != 1.0
    ):
        raise RuntimeError(
            "Breakout v7 combined engine requires active governor support"
        )

    def combined_breakout_portfolio(
        candidate: Candidate,
        minute: int,
        equity: float,
    ) -> PortfolioSearchConfig:
        profile = breakout_selector(candidate, minute, equity)
        if profile is None:
            raise RuntimeError(
                "selected Breakout v7 unexpectedly rejected a candidate"
            )
        if (
            profile.signal != configs["breakout_managed_signal"]
            or profile.exit_protection != configs["breakout_exit"]
        ):
            raise RuntimeError(
                "Breakout v7 profiles no longer share one managed exit"
            )
        return profile.portfolio

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
        combined_grid_signal,
        combined_grid_portfolio,
        configs["combined"],
        execution,
        start,
        end,
        initial_equity,
        breakout_portfolio_selector=combined_breakout_portfolio,
    )
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
        combined_grid_signal,
        combined_grid_portfolio,
        replace(
            configs["combined"], entry_priority=(GRID_KEY, BREAKOUT_KEY)
        ),
        execution,
        start,
        end,
        initial_equity,
        breakout_portfolio_selector=combined_breakout_portfolio,
    )
    if combined["max_concurrent_positions"] > 2:
        raise RuntimeError("combined simulation exceeded max two positions")
    if abs(float(combined["pnl_reconciliation_error"])) > 1e-6:
        raise RuntimeError("combined PnL reconciliation failed")
    combined["strategy"] = COMBINED_V7_GRID_V5_NAME
    combined["source_candidate_count_by_strategy"] = {
        BREAKOUT_KEY: sum(map(len, breakout_candidates.values())),
        GRID_KEY: sum(map(len, grid_candidates.values())),
    }
    combined["grid_v5_profile_rejected_before_arbitration"] = weak_count
    reverse["strategy"] = COMBINED_V7_GRID_V5_NAME
    print(
        f"{name}: combined trades={combined['trade_count']} "
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
        "combined_source_candidate_count_by_strategy": combined[
            "source_candidate_count_by_strategy"
        ],
        "grid_v5_profile_rejected_before_arbitration": weak_count,
        "reverse_priority": _compact_combined(reverse),
        "full_combined_result": combined,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    end = datetime.fromisoformat(args.end)
    start_3m = datetime.fromisoformat(args.start_3m)
    start_6m = datetime.fromisoformat(args.start_6m)
    if not start_6m < start_3m < end:
        raise ValueError("periods must satisfy start_6m < start_3m < end")
    symbols = (
        tuple(UNIVERSE_50)
        if args.universe_size == 50
        else tuple(UNIVERSE_100)
    )
    data_start = start_6m - timedelta(days=args.warmup_days)
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
    ):
        raise RuntimeError("combined backtest requires gap-free stitched data")
    if metadata["funding_missing_symbols"]:
        raise RuntimeError("combined backtest requires complete funding data")
    configs = build_frozen_v7_v5_configs(
        args.breakout_config,
        args.grid_config,
        args.combined_config,
    )
    periods = {
        "three_month": _run_period(
            "3m",
            start_3m,
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
            start_6m,
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
    report = {
        "strategy_name": COMBINED_V7_GRID_V5_NAME,
        "version": COMBINED_V7_GRID_V5_BACKTEST_VERSION,
        "status": "independent_shared_account_research",
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
            "shared_account": args.combined_config,
        },
        "source_hashes": {
            BREAKOUT_KEY: sha256_file(args.breakout_config),
            GRID_KEY: sha256_file(args.grid_config),
            "shared_account": sha256_file(args.combined_config),
        },
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

    lines = [
        "# Breakout v7 + Grid v5 Shared Max2 Backtest",
        "",
        f"- Initial equity: `{args.initial_equity:.2f}U`",
        "- Windows: 3m `2026-04-19..2026-07-19`; "
        "6m `2026-01-19..2026-07-19`",
        f"- {len(symbols)} symbols; each strategy max one position; "
        "shared account max two",
        "- Gap-free 1m execution; full fee, slippage and funding; "
        "adverse stop first",
        "- Standalone rows each start from 200U separately; their profits "
        "must not be added as one 200U portfolio",
        "",
        "| Period | Mode | Trades | Net | PF | Win rate | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for period_name, period in periods.items():
        label = "3 months" if period_name == "three_month" else "6 months"
        for mode, metrics in (
            ("Breakout v7 standalone", period["standalone"][BREAKOUT_KEY]),
            ("Grid v5 standalone", period["standalone"][GRID_KEY]),
            ("Shared account max2", period["combined"]),
        ):
            lines.append(
                f"| {label} | {mode} | {metrics['trade_count']} | "
                f"{metrics['net_profit']:+.2f}U | "
                f"{metrics['profit_factor']:.3f} | "
                f"{metrics['win_rate']:.2%} | "
                f"{metrics['max_drawdown_pct']:.2%} |"
            )
    lines.extend(["", "## Shared-account contribution", ""])
    for period_name, period in periods.items():
        label = "3 months" if period_name == "three_month" else "6 months"
        combined = period["combined"]
        breakout = combined["by_strategy"][BREAKOUT_KEY]
        grid = combined["by_strategy"][GRID_KEY]
        reverse = period["reverse_priority"]
        two_share = combined["concurrency_share"].get(
            "2", combined["concurrency_share"].get(2, 0.0)
        )
        lines.extend(
            [
                f"- {label}: Breakout `{breakout['trade_count']}` trades, "
                f"`{breakout['net_pnl']:+.2f}U`, PF "
                f"`{breakout['profit_factor']:.3f}`.",
                f"- {label}: Grid `{grid['trade_count']}` trades, "
                f"`{grid['net_pnl']:+.2f}U`, PF "
                f"`{grid['profit_factor']:.3f}`.",
                f"- {label}: two-position time share `{two_share:.2%}`; "
                f"reverse-priority net `{reverse['net_profit']:+.2f}U`, "
                f"PF `{reverse['profit_factor']:.3f}`, DD "
                f"`{reverse['max_drawdown_pct']:.2%}`.",
            ]
        )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    research_config = {
        "strategy_name": COMBINED_V7_GRID_V5_NAME,
        "version": COMBINED_V7_GRID_V5_BACKTEST_VERSION,
        "status": "historical_research_only_active_gui_unchanged",
        "initial_equity": args.initial_equity,
        "source_configs": report["source_configs"],
        "source_hashes": report["source_hashes"],
        "portfolio": asdict(configs["combined"]),
        "execution_rules": report["execution_rules"],
        "results": {
            name: period["combined"] for name, period in periods.items()
        },
    }
    _write_json(args.research_config_output, research_config)
    manifest = {
        "strategy_name": COMBINED_V7_GRID_V5_NAME,
        "version": COMBINED_V7_GRID_V5_BACKTEST_VERSION,
        "status": "independent_combined_research_artifacts",
        "report": args.output,
        "summary": args.summary,
        "research_config": args.research_config_output,
        "hashes": {
            artifact: sha256_file(artifact)
            for artifact in (
                "crypto_scalper/combined_breakout_v7_grid_v5_backtest.py",
                "tests/test_combined_breakout_v7_grid_v5_backtest.py",
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
        description="Breakout v7 plus Grid v5 shared max-two backtest"
    )
    parser.add_argument("--start-6m", default="2026-01-19T00:00:00")
    parser.add_argument("--start-3m", default="2026-04-19T00:00:00")
    parser.add_argument("--end", default="2026-07-19T00:00:00")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument(
        "--universe-size",
        type=int,
        choices=(50, 100),
        default=50,
    )
    parser.add_argument(
        "--breakout-config",
        default="config.volatility-breakout.v7-confidence-refined-50.json",
    )
    parser.add_argument(
        "--grid-config",
        default="config.trend-grid.v5-confidence-final-50.json",
    )
    parser.add_argument(
        "--combined-config",
        default="config.combined-breakout-v7-grid-v5-max2.json",
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
        default="reports/combined_breakout_v7_grid_v5_max2_3m_6m.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/combined_breakout_v7_grid_v5_max2_3m_6m.md",
    )
    parser.add_argument(
        "--research-config-output",
        default="config.combined-breakout-v7-grid-v5-max2-backtest.json",
    )
    parser.add_argument(
        "--manifest",
        default=(
            "config.combined-breakout-v7-grid-v5-max2-backtest-manifest.json"
        ),
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

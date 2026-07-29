from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any, Optional

from .combined_hybrid_v5_grid_v3_backtest import (
    _slice_signal_data,
    _write_json,
    build_frozen_configs,
)
from .models import Direction
from .risk import BacktestExecutionConfig
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import ExitProtectionConfig
from .volatility_breakout_optimize import (
    Candidate,
    PortfolioSearchConfig,
    UNIVERSE_50,
    _candidate_sort_key,
    _shift_candidates,
    _without_symbols,
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
    BreakoutV7ConfidencePolicy,
    BreakoutV7EntryTiming,
    apply_breakout_v7_timing,
)
from .volatility_breakout_v8 import (
    VOLATILITY_BREAKOUT_V8_NAME,
    BreakoutV8ScoreAllocation,
    breakout_v8_risk_multiplier,
)


def _entry_from_dict(payload: dict[str, Any]) -> BreakoutV6EntryConfig:
    return BreakoutV6EntryConfig(
        long=BreakoutV6SideGate(**payload["long"]),
        short=BreakoutV6SideGate(**payload["short"]),
        max_signals_per_symbol_day=int(payload["max_signals_per_symbol_day"]),
        require_market_context=bool(payload["require_market_context"]),
    )


def _pf(result: dict[str, Any]) -> float:
    value = result.get("profit_factor")
    if value is None:
        return math.inf if result.get("net_profit", 0.0) > 0.0 else 0.0
    number = float(value)
    return number if not math.isnan(number) else 0.0


def _scaled_execution(
    execution: BacktestExecutionConfig, multiplier: float
) -> BacktestExecutionConfig:
    return replace(
        execution,
        market_slippage_bps=execution.market_slippage_bps * multiplier,
        stop_slippage_bps=execution.stop_slippage_bps * multiplier,
        take_profit_slippage_bps=execution.take_profit_slippage_bps * multiplier,
        maker_fee_rate=execution.maker_fee_rate * multiplier,
        taker_fee_rate=execution.taker_fee_rate * multiplier,
    )


def _strict_improvement(
    six: dict[str, Any],
    three: dict[str, Any],
    base_six: dict[str, Any],
    base_three: dict[str, Any],
) -> bool:
    return (
        six["trade_count"] >= 70
        and three["trade_count"] >= 38
        and six["net_profit"] > base_six["net_profit"]
        and three["net_profit"] > base_three["net_profit"]
        and _pf(six) > _pf(base_six)
        and _pf(three) > _pf(base_three)
        and six["max_drawdown_pct"] <= base_six["max_drawdown_pct"]
        and three["max_drawdown_pct"] <= base_three["max_drawdown_pct"]
    )


def _period_score(
    result: dict[str, Any], baseline: dict[str, Any]
) -> float:
    if result["hard_drawdown_stopped"] or result["trade_count"] < 0.80 * baseline["trade_count"]:
        return -1e12
    growth = math.log(max(result["final_equity"] / result["initial_equity"], 0.01))
    base_growth = math.log(
        max(baseline["final_equity"] / baseline["initial_equity"], 0.01)
    )
    pf_gain = min(_pf(result), 20.0) - min(_pf(baseline), 20.0)
    return (
        7.0 * (growth - base_growth)
        + 0.75 * pf_gain
        + 14.0
        * (baseline["max_drawdown_pct"] - result["max_drawdown_pct"])
        + 2.0 * (result["win_rate"] - baseline["win_rate"])
        - 1.5 * max(0.0, result["top5_profit_contribution"] - 0.78)
    )


def _pair_score(
    six: dict[str, Any],
    three: dict[str, Any],
    base_six: dict[str, Any],
    base_three: dict[str, Any],
) -> float:
    return _period_score(six, base_six) + 1.25 * _period_score(
        three, base_three
    )


def _allocation_variants(
    seed: int, budget: int
) -> list[BreakoutV8ScoreAllocation]:
    anchor = BreakoutV8ScoreAllocation()
    rows = [
        anchor,
        replace(anchor, minimum_score=2),
        replace(anchor, score_1_short_factor=0.0),
    ]
    coordinates = {
        "score_1_short_factor": (0.0, 0.20, 0.40, 0.60, 0.80),
        "score_2_short_factor": (0.70, 0.85, 1.00, 1.10, 1.20),
        "score_3_long_factor": (0.20, 0.40, 0.60, 0.80, 1.00),
        "score_3_short_factor": (0.60, 0.75, 0.90, 1.00, 1.10),
        "score_4_long_factor": (0.90, 1.00, 1.10, 1.20, 1.30, 1.40),
        "score_4_short_factor": (0.80, 0.95, 1.05, 1.15, 1.25),
        "score_5_long_factor": (1.00, 1.15, 1.30, 1.45, 1.60, 1.80),
        "score_5_short_factor": (0.90, 1.05, 1.20, 1.40, 1.60),
    }
    for field, values in coordinates.items():
        rows.extend(replace(anchor, **{field: value}) for value in values)
    # Dense, deliberately moderate convex allocations.  These reduce the
    # historically weak score-1/score-3 lanes while only incrementally adding
    # risk to score-4/score-5 entries; they are evaluated before random tails.
    for low_factor in (0.35, 0.50, 0.65, 0.80):
        for high_factor in (1.10, 1.20, 1.30, 1.40):
            rows.append(
                replace(
                    anchor,
                    minimum_score=2,
                    score_3_long_factor=low_factor,
                    score_3_short_factor=min(0.95, low_factor + 0.20),
                    score_4_long_factor=min(1.25, high_factor),
                    score_4_short_factor=min(1.15, high_factor),
                    score_5_long_factor=high_factor,
                    score_5_short_factor=min(1.30, high_factor),
                )
            )
    rng = random.Random(seed)
    fields = tuple(coordinates)
    while len(dict.fromkeys(rows)) < budget:
        selected = rng.sample(fields, rng.choice((3, 4, 5, 6, 7, 8)))
        changes = {
            field: rng.choice(coordinates[field]) for field in selected
        }
        changes["minimum_score"] = rng.choice((1, 2, 2, 2))
        rows.append(replace(anchor, **changes))
    return list(dict.fromkeys(rows))[:budget]


def _local_allocation_variants(
    source: BreakoutV8ScoreAllocation,
    seed: int,
    budget: int,
) -> list[BreakoutV8ScoreAllocation]:
    rows = [source]
    values_by_field = {
        "score_1_short_factor": (0.0, 0.25, 0.50, 0.75, 1.0),
        "score_3_long_factor": (0.60, 0.75, 0.90, 1.0),
        "score_3_short_factor": (0.70, 0.85, 0.95, 1.0),
        "score_4_long_factor": (0.90, 1.0, 1.05, 1.10, 1.15),
        "score_4_short_factor": (0.90, 1.0, 1.05, 1.10, 1.15),
        "score_5_long_factor": (0.90, 1.0, 1.05, 1.10, 1.20),
        "score_5_short_factor": (
            1.30,
            1.40,
            1.50,
            1.55,
            1.60,
            1.65,
            1.70,
            1.80,
            2.0,
            2.2,
            2.4,
            2.6,
        ),
    }
    for field, values in values_by_field.items():
        rows.extend(replace(source, **{field: value}) for value in values)
    for score5_short in values_by_field["score_5_short_factor"]:
        for score3 in (0.75, 0.90, 1.0):
            rows.append(
                replace(
                    source,
                    score_3_long_factor=score3,
                    score_3_short_factor=min(1.0, score3 + 0.10),
                    score_5_short_factor=score5_short,
                )
            )
    rng = random.Random(seed)
    fields = tuple(values_by_field)
    while len(dict.fromkeys(rows)) < budget:
        chosen = rng.sample(fields, rng.choice((2, 3, 4)))
        changes = {
            field: rng.choice(values_by_field[field]) for field in chosen
        }
        rows.append(replace(source, **changes))
    return list(dict.fromkeys(rows))[:budget]


def _profile_variants(
    source: ManagedLaneProfile, seed: int, budget: int
) -> list[ManagedLaneProfile]:
    rows = [source]
    for stop in (0.74, 0.77, 0.80, 0.83, 0.86):
        rows.append(replace(source, core_stop_atr=stop))
    for duration in (1_080, 1_200, 1_320, 1_440, 1_560):
        rows.append(replace(source, core_max_holding_minutes=duration))
    for trigger in (0.0, 4.0, 5.0, 6.0, 7.0):
        rows.append(replace(source, core_breakeven_trigger_r=trigger))
    for activation, giveback in (
        (12.0, 9.0),
        (14.0, 10.0),
        (15.0, 12.0),
        (17.0, 13.0),
        (20.0, 15.0),
    ):
        rows.append(
            replace(
                source,
                core_profit_giveback_activation_r=activation,
                core_profit_giveback_r=giveback,
            )
        )
    # Causal global drawdown governors are part of the executable v6 engine.
    # They let the search test whether convex winner allocation can retain its
    # upside while respecting the frozen v7 drawdown ceiling.
    for reduce_start, reduce_multiplier, deep_start, deep_multiplier in (
        (0.15, 0.70, 0.24, 0.40),
        (0.18, 0.70, 0.26, 0.40),
        (0.20, 0.75, 0.28, 0.45),
        (0.22, 0.75, 0.29, 0.45),
        (0.24, 0.80, 0.30, 0.50),
        (0.25, 0.70, 0.29, 0.35),
        (0.20, 0.60, 0.26, 0.30),
        (0.18, 0.55, 0.24, 0.25),
    ):
        rows.append(
            replace(
                source,
                drawdown_reduce_start=reduce_start,
                drawdown_reduce_multiplier=reduce_multiplier,
                drawdown_deep_start=deep_start,
                drawdown_deep_multiplier=deep_multiplier,
                drawdown_scope="global",
            )
        )
    rng = random.Random(seed)
    while len(dict.fromkeys(rows)) < budget:
        activation, giveback = rng.choice(
            (
                (12.0, 9.0),
                (14.0, 10.0),
                (15.0, 12.0),
                (17.0, 13.0),
                (20.0, 15.0),
            )
        )
        (
            reduce_start,
            reduce_multiplier,
            deep_start,
            deep_multiplier,
        ) = rng.choice(
            (
                (0.18, 0.55, 0.24, 0.25),
                (0.20, 0.60, 0.26, 0.30),
                (0.22, 0.75, 0.29, 0.45),
                (0.24, 0.80, 0.30, 0.50),
                (1.0, 1.0, 1.0, 1.0),
            )
        )
        rows.append(
            replace(
                source,
                core_stop_atr=rng.choice((0.74, 0.77, 0.80, 0.83, 0.86)),
                core_max_holding_minutes=rng.choice(
                    (1_080, 1_200, 1_320, 1_440, 1_560)
                ),
                core_breakeven_trigger_r=rng.choice((0.0, 4.0, 5.0, 6.0)),
                core_profit_giveback_activation_r=activation,
                core_profit_giveback_r=giveback,
                drawdown_reduce_start=reduce_start,
                drawdown_reduce_multiplier=reduce_multiplier,
                drawdown_deep_start=deep_start,
                drawdown_deep_multiplier=deep_multiplier,
            )
        )
    return list(dict.fromkeys(rows))[:budget]


ProfileVariantBuilder = Callable[
    [ManagedLaneProfile, int, int], list[ManagedLaneProfile]
]
StrictImprovementPredicate = Callable[
    [
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
    bool,
]


def run(
    args: argparse.Namespace,
    *,
    strategy_name: str = VOLATILITY_BREAKOUT_V8_NAME,
    strategy_label: str = "Breakout v8",
    result_version: str = "v8_score_convex",
    profile_variant_builder: ProfileVariantBuilder | None = None,
    strict_improvement_predicate: StrictImprovementPredicate = (
        _strict_improvement
    ),
    baseline_strategy_label: str | None = None,
    improvement_reference_slug: str = "prior_breakout_v8",
    manifest_source_files: Iterable[str] = (
        "crypto_scalper/volatility_breakout_v8.py",
        "crypto_scalper/volatility_breakout_v8_optimize.py",
    ),
) -> dict[str, Any]:
    start_six = datetime.fromisoformat(args.start_6m)
    start_three = datetime.fromisoformat(args.start_3m)
    end = datetime.fromisoformat(args.end)
    if not start_six < start_three < end:
        raise ValueError("periods must satisfy start_6m < start_3m < end")

    payload = json.loads(Path(args.anchor_config).read_text(encoding="utf-8"))
    gate = _entry_from_dict(payload["entry"])
    confidence = BreakoutV7ConfidencePolicy(**payload["confidence_policy"])
    timing = BreakoutV7EntryTiming(**payload["entry_timing"])
    base_profile = ManagedLaneProfile(**payload["managed_profile"])
    base_signal = VolatilityBreakoutConfig(**payload["operational_signal"])
    base_portfolio = PortfolioSearchConfig(**payload["operational_portfolio"])
    frozen = build_frozen_configs(
        args.source_breakout_config, args.source_grid_config
    )
    build_signal = frozen["breakout_build_signal"]
    symbols = tuple(UNIVERSE_50)
    signal_data, execution_data, rules, execution, metadata = load_v4_runtime_inputs(
        symbols,
        args.one_minute_roots,
        args.funding_roots,
        args.cost_config,
        start_six - timedelta(days=args.warmup_days),
        end,
    )
    if (
        metadata["minimum_coverage_ratio"] < 0.999999
        or metadata["maximum_missing_minutes"]
        or metadata["funding_missing_symbols"]
    ):
        raise RuntimeError("Breakout v8 requires gap-free price and funding data")

    def build_period(name: str, start: datetime, finish: datetime) -> dict[str, Any]:
        local_signal = _slice_signal_data(
            signal_data, start - timedelta(days=args.warmup_days), finish
        )
        context = build_v4_market_context(symbols, local_signal)
        raw = enrich_candidates_v4(
            build_candidates(
                symbols,
                local_signal,
                execution_data,
                build_signal,
                start,
                finish,
            ),
            context,
        )
        filtered = filter_breakout_v6_candidates(raw, context, gate)
        selected = apply_breakout_v7_timing(filtered, execution_data, timing)
        print(
            f"{name}: raw={sum(map(len, raw.values()))} "
            f"v7={sum(map(len, selected.values()))}",
            flush=True,
        )
        return {
            "start": start,
            "end": finish,
            "context": context,
            "candidates": selected,
        }

    periods = {
        "six": build_period("6m", start_six, end),
        "three": build_period("3m", start_three, end),
        "early": build_period("early3m", start_six, start_three),
    }

    def simulate(
        period: str,
        allocation: BreakoutV8ScoreAllocation,
        profile: ManagedLaneProfile,
        selected_execution: BacktestExecutionConfig = execution,
        extra_delay_minutes: int = 0,
        force_compound: Optional[bool] = None,
        excluded_symbols: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        values = periods[period]
        candidates: dict[int, list[Candidate]] = values["candidates"]
        if extra_delay_minutes:
            candidates = _shift_candidates(
                candidates, extra_delay_minutes, execution_data
            )
        if excluded_symbols:
            candidates = _without_symbols(candidates, excluded_symbols)
        context: dict[int, dict[str, V4MarketSnapshot]] = values["context"]
        signal = replace(
            base_signal,
            stop_atr_multiple=profile.core_stop_atr,
            take_profit_r=60.0,
            max_holding_minutes=profile.core_max_holding_minutes,
            fail_fast_minutes=profile.core_fail_fast_minutes,
            fail_fast_min_mfe_r=profile.core_fail_fast_min_mfe_r,
            fail_fast_max_current_r=profile.core_fail_fast_max_current_r,
        )
        exit_config = ExitProtectionConfig(
            breakeven_trigger_r=profile.core_breakeven_trigger_r,
            profit_giveback_activation_r=profile.core_profit_giveback_activation_r,
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
        non_capture_exit_config = replace(
            exit_config,
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

        def add_profit_floor(
            selected_exit: ExitProtectionConfig,
            activation_r: float,
            lock_r: float,
        ) -> ExitProtectionConfig:
            levels = {
                float(activation): float(lock)
                for activation, lock in selected_exit.profit_floor_levels
            }
            levels[float(activation_r)] = max(
                float(lock_r), levels.get(float(activation_r), 0.0)
            )
            ordered: list[tuple[float, float]] = []
            previous_lock = 0.0
            for activation, lock in sorted(levels.items()):
                previous_lock = max(previous_lock, lock)
                ordered.append((activation, previous_lock))
            ordered = ordered[:3]
            padded = [*ordered, *((0.0, 0.0),) * (3 - len(ordered))]
            return replace(
                selected_exit,
                profit_floor_1_activation_r=padded[0][0],
                profit_floor_1_lock_r=padded[0][1],
                profit_floor_2_activation_r=padded[1][0],
                profit_floor_2_lock_r=padded[1][1],
                profit_floor_3_activation_r=padded[2][0],
                profit_floor_3_lock_r=padded[2][1],
            )

        def choose(
            candidate: Candidate, minute: int, _equity: float
        ) -> Optional[BreakoutV6ExecutionProfile]:
            snapshot = context.get(minute - minute % 60, {}).get(
                candidate.signal.symbol
            )
            regime = (
                _regime_score(snapshot, candidate.signal.direction)
                if snapshot
                else 0.0
            )
            alignment = (
                float(candidate.signal.direction.value) * snapshot.symbol_ema55_atr
                if snapshot
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
                exit_config
                if capture_side and capture_score
                else non_capture_exit_config
            )
            if (
                candidate.signal.direction == Direction.LONG
                and profile.core_long_profit_floor_activation_r > 0.0
                and profile.core_long_profit_floor_min_score
                <= score
                <= profile.core_long_profit_floor_max_score
            ):
                selected_exit = add_profit_floor(
                    selected_exit,
                    profile.core_long_profit_floor_activation_r,
                    profile.core_long_profit_floor_lock_r,
                )
            portfolio = replace(
                base_portfolio,
                risk_per_trade_pct=risk * multiplier,
                long_risk_multiplier=profile.core_long_multiplier,
                short_risk_multiplier=profile.core_short_multiplier,
                ranking_mode=profile.ranking_mode,
                compound=(
                    base_portfolio.compound
                    if force_compound is None
                    else force_compound
                ),
            )
            return BreakoutV6ExecutionProfile(
                f"v8_{lane}", signal, portfolio, selected_exit
            )

        def priority(candidate: Candidate):
            return _candidate_sort_key(candidate, profile.ranking_mode)

        governor_period = ""
        governor_peak = args.initial_equity

        def govern(
            selected: BreakoutV6ExecutionProfile,
            drawdown: float,
            minute: int,
            equity: float,
        ) -> BreakoutV6ExecutionProfile:
            nonlocal governor_period, governor_peak
            if profile.drawdown_scope == "monthly":
                period_key = minute_datetime(minute).strftime("%Y-%m")
                if period_key != governor_period:
                    governor_period = period_key
                    governor_peak = equity
                else:
                    governor_peak = max(governor_peak, equity)
                drawdown = (
                    (governor_peak - equity) / governor_peak
                    if governor_peak > 0.0
                    else 1.0
                )
            multiplier = tiered_drawdown_risk_multiplier(
                drawdown,
                profile.drawdown_reduce_start,
                profile.drawdown_reduce_multiplier,
                profile.drawdown_deep_start,
                profile.drawdown_deep_multiplier,
            )
            if multiplier >= 1.0:
                return selected
            return replace(
                selected,
                portfolio=replace(
                    selected.portfolio,
                    risk_per_trade_pct=(
                        selected.portfolio.risk_per_trade_pct * multiplier
                    ),
                ),
            )

        result = simulate_v6_managed_portfolio(
            candidates,
            symbols,
            execution_data,
            rules,
            base_signal,
            replace(
                base_portfolio,
                ranking_mode=profile.ranking_mode,
                compound=(
                    base_portfolio.compound
                    if force_compound is None
                    else force_compound
                ),
            ),
            ExitProtectionConfig(),
            selected_execution,
            values["start"],
            values["end"],
            args.initial_equity,
            profile_selector=choose,
            priority_selector=priority,
            profile_governor=govern,
        )
        result["strategy_version"] = result_version
        return result

    anchor_allocation = BreakoutV8ScoreAllocation()
    anchor_profile = base_profile
    expected_payload = payload
    baseline_name = "Breakout v7 frozen"
    if args.refine_from_config:
        refine_payload = json.loads(
            Path(args.refine_from_config).read_text(encoding="utf-8")
        )
        anchor_allocation = BreakoutV8ScoreAllocation(
            **refine_payload["score_allocation"]
        )
        anchor_profile = ManagedLaneProfile(
            **refine_payload["managed_profile"]
        )
        expected_payload = refine_payload
        baseline_name = (
            baseline_strategy_label
            or "Breakout v8 prior strict candidate"
        )
    base_six = simulate("six", anchor_allocation, anchor_profile)
    base_three = simulate("three", anchor_allocation, anchor_profile)
    expected_six = expected_payload["results"]["six_month"]
    expected_three = expected_payload["results"]["three_month"]
    for name, actual, expected in (
        ("6m", base_six, expected_six),
        ("3m", base_three, expected_three),
    ):
        if abs(actual["net_profit"] - expected["net_profit"]) > 1e-6:
            raise RuntimeError(
                f"Breakout v8 anchor mismatch {name}: "
                f"{actual['net_profit']} != {expected['net_profit']}"
            )
    print(
        f"v7 anchor 6m={base_six['net_profit']:+.2f}/PF{_pf(base_six):.3f}/"
        f"DD{base_six['max_drawdown_pct']:.2%}; "
        f"3m={base_three['net_profit']:+.2f}/PF{_pf(base_three):.3f}/"
        f"DD{base_three['max_drawdown_pct']:.2%}",
        flush=True,
    )

    allocation_rows: list[dict[str, Any]] = []
    allocations = (
        _local_allocation_variants(
            anchor_allocation, args.seed, args.allocation_budget
        )
        if args.refine_from_config
        else _allocation_variants(args.seed, args.allocation_budget)
    )
    for number, allocation in enumerate(allocations, 1):
        six = simulate("six", allocation, anchor_profile)
        allocation_rows.append(
            {
                "allocation": allocation,
                "profile": anchor_profile,
                "six": six,
                "score6": _period_score(six, base_six),
            }
        )
        if number == 1 or number % 25 == 0 or number == len(allocations):
            print(
                f"allocation {number}/{len(allocations)} "
                f"{six['net_profit']:+.0f}/PF{_pf(six):.2f}/"
                f"DD{six['max_drawdown_pct']:.1%}",
                flush=True,
            )
    allocation_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in allocation_rows[: args.recent_finalists]:
        row["three"] = simulate("three", row["allocation"], row["profile"])
        row["pair_score"] = _pair_score(
            row["six"], row["three"], base_six, base_three
        )
        row["strict"] = strict_improvement_predicate(
            row["six"], row["three"], base_six, base_three
        )
    allocation_finalists = sorted(
        allocation_rows[: args.recent_finalists],
        key=lambda row: (row.get("strict", False), row.get("pair_score", -1e12)),
        reverse=True,
    )[: args.allocation_finalists]

    profile_rows: list[dict[str, Any]] = []
    build_profile_variants = (
        profile_variant_builder or _profile_variants
    )
    profiles = build_profile_variants(
        anchor_profile, args.seed + 1, args.profile_budget
    )
    for source_number, source in enumerate(allocation_finalists, 1):
        for profile in profiles:
            six = simulate("six", source["allocation"], profile)
            profile_rows.append(
                {
                    "allocation": source["allocation"],
                    "profile": profile,
                    "six": six,
                    "score6": _period_score(six, base_six),
                }
            )
        print(
            f"profile source {source_number}/{len(allocation_finalists)} complete",
            flush=True,
        )
    profile_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in profile_rows[: args.profile_recent_finalists]:
        row["three"] = simulate("three", row["allocation"], row["profile"])
        row["pair_score"] = _pair_score(
            row["six"], row["three"], base_six, base_three
        )
        row["strict"] = strict_improvement_predicate(
            row["six"], row["three"], base_six, base_three
        )
    recent = sorted(
        profile_rows[: args.profile_recent_finalists],
        key=lambda row: (row.get("strict", False), row.get("pair_score", -1e12)),
        reverse=True,
    )

    stressed = _scaled_execution(execution, 1.5)
    robust_rows: list[dict[str, Any]] = []
    for number, row in enumerate(recent[: args.robust_finalists], 1):
        allocation = row["allocation"]
        profile = row["profile"]
        early = simulate("early", allocation, profile)
        stress_six = simulate(
            "six", allocation, profile, selected_execution=stressed
        )
        stress_three = simulate(
            "three", allocation, profile, selected_execution=stressed
        )
        delay_six = simulate(
            "six", allocation, profile, extra_delay_minutes=1
        )
        delay_three = simulate(
            "three", allocation, profile, extra_delay_minutes=1
        )
        fixed_six = simulate(
            "six", allocation, profile, force_compound=False
        )
        fixed_three = simulate(
            "three", allocation, profile, force_compound=False
        )
        top_six = max(
            row["six"]["by_symbol"],
            key=lambda symbol: row["six"]["by_symbol"][symbol]["net_pnl"],
            default="",
        )
        top_three = max(
            row["three"]["by_symbol"],
            key=lambda symbol: row["three"]["by_symbol"][symbol]["net_pnl"],
            default="",
        )
        no_top_six = simulate(
            "six",
            allocation,
            profile,
            excluded_symbols=frozenset({top_six}) if top_six else frozenset(),
        )
        no_top_three = simulate(
            "three",
            allocation,
            profile,
            excluded_symbols=(
                frozenset({top_three}) if top_three else frozenset()
            ),
        )
        robust = (
            row["strict"]
            and early["net_profit"] > 0.0
            and _pf(early) > 1.0
            and stress_six["net_profit"] > 0.0
            and stress_three["net_profit"] > 0.0
            and _pf(stress_six) > 1.5
            and _pf(stress_three) > 1.5
            and delay_six["net_profit"] > 0.0
            and delay_three["net_profit"] > 0.0
            and fixed_six["net_profit"] > 0.0
            and fixed_three["net_profit"] > 0.0
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
                "top_six": top_six,
                "top_three": top_three,
                "robust": robust,
                "robust_score": row["pair_score"] + (20.0 if robust else 0.0),
            }
        )
        robust_rows.append(row)
        print(
            f"robust {number}/{args.robust_finalists} strict={row['strict']} "
            f"robust={robust} early={early['net_profit']:+.0f} "
            f"stress6={stress_six['net_profit']:+.0f} "
            f"delay6={delay_six['net_profit']:+.0f}",
            flush=True,
        )

    selectable = [row for row in robust_rows if row["robust"]]
    selected = max(
        selectable or robust_rows or recent or allocation_finalists,
        key=lambda row: row.get("robust_score", row.get("pair_score", -1e12)),
    )
    selection_status = (
        (
            f"strict_robust_improvement_over_{improvement_reference_slug}"
            if args.refine_from_config
            else "strict_robust_improvement_over_breakout_v7"
        )
        if selected.get("robust")
        else (
            f"best_available_did_not_clear_{improvement_reference_slug}"
            if args.refine_from_config
            else "best_available_did_not_clear_breakout_v7"
        )
    )

    def public(row: dict[str, Any], full: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "pair_score": row.get("pair_score"),
            "strict_improvement": row.get("strict", False),
            "robust": row.get("robust", False),
            "score_allocation": row["allocation"].as_dict(),
            "managed_profile": asdict(row["profile"]),
            "six_month": compact_v6_metrics(row["six"]),
            "three_month": compact_v6_metrics(row["three"]),
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
                output[target] = compact_v6_metrics(row[source])
        if "top_six" in row:
            output["removed_top_symbol_six_month"] = row["top_six"]
            output["removed_top_symbol_three_month"] = row["top_three"]
        if full:
            output["six_month_full"] = row["six"]
            output["three_month_full"] = row["three"]
        return output

    report = {
        "strategy_name": strategy_name,
        "status": "independent_research_v7_and_gui_unchanged",
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
            "mode": "gap_free_1m_full_cost_point_in_time",
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
            "allocation_evaluations": len(allocation_rows),
            "profile_evaluations": len(profile_rows),
            "robust_evaluations": len(robust_rows),
        },
        "baseline": {
            "strategy": baseline_name,
            "source_config": (
                args.refine_from_config or args.anchor_config
            ),
            "six_month": compact_v6_metrics(base_six),
            "three_month": compact_v6_metrics(base_three),
        },
        "selected": public(selected, True),
        "leaderboard": [public(row) for row in recent[:20]],
        "preserved": {
            "breakout_v7": "unchanged",
            "grid_v5": "unchanged",
            "active_gui": "unchanged",
            "apt_grid": "unchanged",
        },
    }
    _write_json(args.output, report)
    config = {
        "strategy_name": strategy_name,
        "status": "independent_research_candidate_not_live",
        "selection_status": selection_status,
        "entry": gate.as_dict(),
        "confidence_policy": confidence.as_dict(),
        "entry_timing": timing.as_dict(),
        "score_allocation": selected["allocation"].as_dict(),
        "managed_profile": asdict(selected["profile"]),
        "operational_signal": base_signal.as_dict(),
        "operational_portfolio": asdict(base_portfolio),
        "results": {
            "six_month": compact_v6_metrics(selected["six"]),
            "three_month": compact_v6_metrics(selected["three"]),
        },
    }
    _write_json(args.config_output, config)
    lines = [
        f"# {strategy_label} optimization",
        "",
        "- Breakout v7 and the active GUI remain unchanged",
        "- Gap-free 1m execution with fees, slippage and funding",
        f"- Selection status: `{selection_status}`",
        "",
        "| Period | Version | Trades | Net | PF | Win | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for period, baseline, result in (
        ("3 months", base_three, selected["three"]),
        ("6 months", base_six, selected["six"]),
    ):
        baseline_display = (
            baseline_strategy_label or "Breakout v8 prior"
            if args.refine_from_config
            else "Breakout v7"
        )
        for name, values in (
            (baseline_display, baseline),
            (strategy_label, result),
        ):
            lines.append(
                f"| {period} | {name} | {values['trade_count']} | "
                f"{values['net_profit']:+.2f}U | {_pf(values):.3f} | "
                f"{values['win_rate']:.2%} | "
                f"{values['max_drawdown_pct']:.2%} |"
            )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "strategy_name": strategy_name,
        "status": "independent_research_artifacts",
        "config": args.config_output,
        "report": args.output,
        "summary": args.summary,
        "hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                *manifest_source_files,
                args.config_output,
                args.output,
                args.summary,
            )
        },
        "preserved": report["preserved"],
    }
    _write_json(args.manifest, manifest)
    print(
        f"selected strict={selected.get('strict', False)} "
        f"robust={selected.get('robust', False)} "
        f"6m={selected['six']['net_profit']:+.2f}/PF{_pf(selected['six']):.3f}/"
        f"DD{selected['six']['max_drawdown_pct']:.2%}; "
        f"3m={selected['three']['net_profit']:+.2f}/PF{_pf(selected['three']):.3f}/"
        f"DD{selected['three']['max_drawdown_pct']:.2%}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Breakout v8 score-convex live-realistic optimization"
    )
    parser.add_argument("--start-6m", default="2026-01-19T00:00:00")
    parser.add_argument("--start-3m", default="2026-04-19T00:00:00")
    parser.add_argument("--end", default="2026-07-19T00:00:00")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument(
        "--anchor-config",
        default="config.volatility-breakout.v7-confidence-refined-50.json",
    )
    parser.add_argument(
        "--refine-from-config",
        default="",
        help=(
            "Optional prior strict v8 config used as the exact local-search "
            "anchor; the frozen v7 source remains unchanged"
        ),
    )
    parser.add_argument(
        "--source-breakout-config",
        default=(
            "config.volatility-breakout."
            "hybrid-v5-balanced-expansion-runner-50.json"
        ),
    )
    parser.add_argument(
        "--source-grid-config",
        default="config.trend-grid.v3-optimized-50.json",
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
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--allocation-budget", type=int, default=320)
    parser.add_argument("--recent-finalists", type=int, default=80)
    parser.add_argument("--allocation-finalists", type=int, default=12)
    parser.add_argument("--profile-budget", type=int, default=36)
    parser.add_argument("--profile-recent-finalists", type=int, default=80)
    parser.add_argument("--robust-finalists", type=int, default=12)
    parser.add_argument(
        "--output",
        default="reports/volatility_breakout_v8_score_convex_3m_6m.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/volatility_breakout_v8_score_convex_3m_6m.md",
    )
    parser.add_argument(
        "--config-output",
        default="config.volatility-breakout.v8-score-convex-50.json",
    )
    parser.add_argument(
        "--manifest",
        default="config.volatility-breakout.v8-score-convex-50-manifest.json",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

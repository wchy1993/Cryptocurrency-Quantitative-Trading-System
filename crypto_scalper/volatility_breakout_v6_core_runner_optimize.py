from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from .combined_hybrid_v5_grid_v3_backtest import (
    _daily_cap_candidates,
    _slice_signal_data,
    _write_json,
    build_frozen_configs,
)
from .volatility_breakout_exit_protection import ExitProtectionConfig
from .volatility_breakout_optimize import (
    Candidate,
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
    filter_candidates_v4,
    load_v4_runtime_inputs,
)
from .volatility_breakout_v6 import (
    BreakoutV6EntryConfig,
    BreakoutV6SideGate,
    build_breakout_v6_lane_candidates,
)
from .volatility_breakout_v6_engine import (
    BreakoutV6ExecutionProfile,
    simulate_v6_managed_portfolio,
)
from .volatility_breakout_v6_optimize import compact_v6_metrics


V6_CORE_RUNNER_NAME = "dual_thrust_volatility_breakout_v6_core_runner"


@dataclass(frozen=True)
class ManagedLaneProfile:
    core_risk_pct: float = 0.0275
    runner_risk_pct: float = 0.0075
    core_strong_risk_pct: float = 0.0275
    runner_strong_risk_pct: float = 0.0075
    weak_risk_multiplier: float = 1.0
    strong_regime_score: float = 999.0
    weak_regime_score: float = -999.0
    strong_alignment_atr: float = 999.0
    core_long_multiplier: float = 0.9
    core_short_multiplier: float = 0.9
    runner_long_multiplier: float = 1.0
    runner_short_multiplier: float = 1.0
    core_stop_atr: float = 0.9
    runner_stop_atr: float = 1.0
    core_max_holding_minutes: int = 1200
    runner_max_holding_minutes: int = 960
    core_fail_fast_minutes: int = 240
    core_fail_fast_min_mfe_r: float = 0.3
    core_fail_fast_max_current_r: float = -0.25
    runner_fail_fast_minutes: int = 120
    runner_fail_fast_min_mfe_r: float = 0.1
    runner_fail_fast_max_current_r: float = -0.5
    core_breakeven_trigger_r: float = 3.5
    core_profit_giveback_activation_r: float = 0.0
    core_profit_giveback_r: float = 0.0
    core_profit_floor_1_activation_r: float = 0.0
    core_profit_floor_1_lock_r: float = 0.0
    core_profit_floor_2_activation_r: float = 0.0
    core_profit_floor_2_lock_r: float = 0.0
    core_profit_floor_3_activation_r: float = 0.0
    core_profit_floor_3_lock_r: float = 0.0
    core_profit_capture_min_score: int = 0
    core_profit_capture_max_score: int = 5
    core_profit_capture_long: bool = True
    core_profit_capture_short: bool = True
    core_long_profit_floor_activation_r: float = 0.0
    core_long_profit_floor_lock_r: float = 0.0
    core_long_profit_floor_min_score: int = 0
    core_long_profit_floor_max_score: int = 5
    core_partial_r: float = 4.0
    core_partial_fraction: float = 0.10
    core_move_breakeven_after_partial: bool = True
    runner_breakeven_trigger_r: float = 0.0
    runner_profit_giveback_activation_r: float = 0.0
    runner_profit_giveback_r: float = 0.0
    runner_partial_r: float = 8.0
    runner_partial_fraction: float = 0.05
    runner_move_breakeven_after_partial: bool = False
    ranking_mode: str = "range_asc"
    drawdown_reduce_start: float = 1.0
    drawdown_reduce_multiplier: float = 1.0
    drawdown_deep_start: float = 1.0
    drawdown_deep_multiplier: float = 1.0
    drawdown_scope: str = "global"


def tiered_drawdown_risk_multiplier(
    drawdown: float,
    reduce_start: float,
    reduce_multiplier: float,
    deep_start: float,
    deep_multiplier: float,
) -> float:
    """Live-reproducible risk governor based on current equity drawdown."""

    if not 0.0 <= reduce_start <= deep_start <= 1.0:
        raise ValueError("drawdown tiers must satisfy 0 <= reduce <= deep <= 1")
    if not 0.0 <= deep_multiplier <= reduce_multiplier <= 1.0:
        raise ValueError("drawdown multipliers must satisfy 0 <= deep <= reduce <= 1")
    if drawdown >= deep_start:
        return deep_multiplier
    if drawdown >= reduce_start:
        return reduce_multiplier
    return 1.0


def _entry_from_dict(payload: dict[str, Any]) -> BreakoutV6EntryConfig:
    return BreakoutV6EntryConfig(
        long=BreakoutV6SideGate(**payload["long"]),
        short=BreakoutV6SideGate(**payload["short"]),
        max_signals_per_symbol_day=int(
            payload.get("max_signals_per_symbol_day", 2)
        ),
        require_market_context=bool(payload.get("require_market_context", True)),
    )


def _reject_side() -> BreakoutV6SideGate:
    return BreakoutV6SideGate(min_quality_score=999.0, max_quality_score=999.0)


def _runner_variants() -> list[BreakoutV6EntryConfig]:
    reject = _reject_side()
    default = BreakoutV6SideGate()
    long_moderate = BreakoutV6SideGate(
        min_quality_score=0.70,
        min_breakout_extension_atr=0.05,
        max_range_atr=7.5,
    )
    long_compact = BreakoutV6SideGate(
        min_quality_score=1.0,
        min_breakout_extension_atr=0.10,
        max_range_atr=6.5,
    )
    long_breadth = BreakoutV6SideGate(
        min_quality_score=0.60,
        min_body_atr=0.40,
        max_range_atr=7.5,
        min_directional_breadth=0.55,
    )
    long_impulse = BreakoutV6SideGate(
        min_quality_score=1.20,
        min_body_atr=0.70,
        min_breakout_extension_atr=0.15,
        max_range_atr=8.0,
    )
    short_moderate = BreakoutV6SideGate(
        max_body_atr=2.2,
        min_breakout_extension_atr=0.05,
        max_range_atr=8.0,
        min_directional_breadth=0.55,
    )
    short_compact = BreakoutV6SideGate(
        max_body_atr=1.6,
        min_breakout_extension_atr=0.05,
        max_range_atr=6.5,
        min_directional_breadth=0.65,
    )
    rows = [
        BreakoutV6EntryConfig(long=reject, short=reject),
        BreakoutV6EntryConfig(long=default, short=default),
        BreakoutV6EntryConfig(long=default, short=reject),
        BreakoutV6EntryConfig(long=long_moderate, short=reject),
        BreakoutV6EntryConfig(long=long_compact, short=reject),
        BreakoutV6EntryConfig(long=long_breadth, short=reject),
        BreakoutV6EntryConfig(long=long_impulse, short=reject),
        BreakoutV6EntryConfig(long=reject, short=default),
        BreakoutV6EntryConfig(long=reject, short=short_moderate),
        BreakoutV6EntryConfig(long=reject, short=short_compact),
        BreakoutV6EntryConfig(long=long_moderate, short=short_moderate),
        BreakoutV6EntryConfig(long=long_compact, short=short_compact),
        BreakoutV6EntryConfig(long=default, short=short_moderate),
        BreakoutV6EntryConfig(long=long_moderate, short=default),
    ]
    return list(dict.fromkeys(rows))


def _core_variants(path: str, budget: int) -> list[BreakoutV6EntryConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [_entry_from_dict(payload["selected"]["entry"])]
    rows.extend(_entry_from_dict(row["entry"]) for row in payload["leaderboard"])
    rows.append(BreakoutV6EntryConfig())
    return list(dict.fromkeys(rows))[:budget]


def _risk_variants(seed: int, budget: int) -> list[ManagedLaneProfile]:
    rows = [
        ManagedLaneProfile(),
        ManagedLaneProfile(core_risk_pct=0.030, runner_risk_pct=0.010),
        ManagedLaneProfile(core_risk_pct=0.0325, runner_risk_pct=0.0125),
        ManagedLaneProfile(core_risk_pct=0.035, runner_risk_pct=0.015),
        ManagedLaneProfile(
            core_risk_pct=0.025,
            runner_risk_pct=0.005,
            core_strong_risk_pct=0.040,
            runner_strong_risk_pct=0.015,
            weak_risk_multiplier=0.50,
            strong_regime_score=0.50,
            weak_regime_score=0.0,
            strong_alignment_atr=0.50,
        ),
        ManagedLaneProfile(
            core_risk_pct=0.0275,
            runner_risk_pct=0.0075,
            core_strong_risk_pct=0.045,
            runner_strong_risk_pct=0.020,
            weak_risk_multiplier=0.40,
            strong_regime_score=0.75,
            weak_regime_score=0.15,
            strong_alignment_atr=1.0,
        ),
    ]
    rng = random.Random(seed)
    while len(list(dict.fromkeys(rows))) < budget:
        core = rng.choice((0.0225, 0.025, 0.0275, 0.030, 0.0325, 0.035))
        runner = rng.choice((0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020))
        dynamic = rng.choice((False, True, True))
        rows.append(
            ManagedLaneProfile(
                core_risk_pct=core,
                runner_risk_pct=runner,
                core_strong_risk_pct=(
                    rng.choice((0.035, 0.040, 0.045, 0.050))
                    if dynamic
                    else core
                ),
                runner_strong_risk_pct=(
                    rng.choice((0.010, 0.015, 0.020, 0.025))
                    if dynamic
                    else runner
                ),
                weak_risk_multiplier=(
                    rng.choice((0.35, 0.50, 0.65, 0.80)) if dynamic else 1.0
                ),
                strong_regime_score=(
                    rng.choice((0.25, 0.50, 0.75, 1.0)) if dynamic else 999.0
                ),
                weak_regime_score=(
                    rng.choice((-0.25, 0.0, 0.15, 0.25))
                    if dynamic
                    else -999.0
                ),
                strong_alignment_atr=(
                    rng.choice((0.0, 0.50, 1.0, 1.50)) if dynamic else 999.0
                ),
                core_long_multiplier=rng.choice((0.9, 1.0, 1.1)),
                core_short_multiplier=rng.choice((0.8, 0.9, 1.0)),
                runner_long_multiplier=rng.choice((0.8, 1.0, 1.2)),
                runner_short_multiplier=rng.choice((0.6, 0.8, 1.0)),
                ranking_mode=rng.choice(
                    ("range_asc", "quality_desc", "breakout_extension_desc")
                ),
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _exit_variants(
    source: ManagedLaneProfile, seed: int, budget: int
) -> list[ManagedLaneProfile]:
    rows = [source]
    rng = random.Random(seed)
    fail_presets = (
        (0, 0.0, 0.0),
        (90, 0.10, -0.35),
        (120, 0.10, -0.50),
        (180, 0.20, -0.35),
        (240, 0.30, -0.25),
    )
    partials = ((0.0, 0.0, False), (4.0, 0.10, True), (6.0, 0.10, False), (8.0, 0.05, False))
    while len(list(dict.fromkeys(rows))) < budget:
        c_fail = rng.choice(fail_presets)
        r_fail = rng.choice(fail_presets)
        c_partial = rng.choice(partials)
        r_partial = rng.choice(partials)
        rows.append(
            replace(
                source,
                core_stop_atr=rng.choice((0.80, 0.90, 1.0, 1.10)),
                runner_stop_atr=rng.choice((0.85, 1.0, 1.15)),
                core_max_holding_minutes=rng.choice((960, 1200, 1440, 1920)),
                runner_max_holding_minutes=rng.choice((480, 720, 960, 1200, 1440)),
                core_fail_fast_minutes=c_fail[0],
                core_fail_fast_min_mfe_r=c_fail[1],
                core_fail_fast_max_current_r=c_fail[2],
                runner_fail_fast_minutes=r_fail[0],
                runner_fail_fast_min_mfe_r=r_fail[1],
                runner_fail_fast_max_current_r=r_fail[2],
                core_breakeven_trigger_r=rng.choice((0.0, 2.5, 3.5, 5.0)),
                core_partial_r=c_partial[0],
                core_partial_fraction=c_partial[1],
                core_move_breakeven_after_partial=c_partial[2],
                runner_breakeven_trigger_r=rng.choice((0.0, 2.5, 3.5)),
                runner_partial_r=r_partial[0],
                runner_partial_fraction=r_partial[1],
                runner_move_breakeven_after_partial=r_partial[2],
                ranking_mode=rng.choice(
                    ("range_asc", "quality_desc", "breakout_extension_desc")
                ),
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _drawdown_governor_variants(
    source: ManagedLaneProfile, seed: int, budget: int, high_global: bool = False
) -> list[ManagedLaneProfile]:
    if high_global:
        rows = [
            source,
            replace(
                source,
                drawdown_reduce_start=0.20,
                drawdown_reduce_multiplier=0.80,
                drawdown_deep_start=0.32,
                drawdown_deep_multiplier=0.45,
                drawdown_scope="global",
            ),
            replace(
                source,
                drawdown_reduce_start=0.25,
                drawdown_reduce_multiplier=0.80,
                drawdown_deep_start=0.35,
                drawdown_deep_multiplier=0.40,
                drawdown_scope="global",
            ),
            replace(
                source,
                drawdown_reduce_start=0.25,
                drawdown_reduce_multiplier=0.70,
                drawdown_deep_start=0.35,
                drawdown_deep_multiplier=0.30,
                drawdown_scope="global",
            ),
            replace(
                source,
                drawdown_reduce_start=0.30,
                drawdown_reduce_multiplier=0.80,
                drawdown_deep_start=0.38,
                drawdown_deep_multiplier=0.40,
                drawdown_scope="global",
            ),
        ]
        rng = random.Random(seed)
        while len(list(dict.fromkeys(rows))) < budget:
            reduce_start = rng.choice(
                (0.18, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325)
            )
            deep_start = rng.choice(
                (0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45)
            )
            if deep_start < reduce_start:
                continue
            reduce_multiplier = rng.choice((0.60, 0.70, 0.80, 0.90))
            deep_multiplier = rng.choice((0.20, 0.30, 0.40, 0.50, 0.60, 0.70))
            if deep_multiplier > reduce_multiplier:
                continue
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
        return list(dict.fromkeys(rows))[:budget]

    rows = [
        source,
        replace(
            source,
            drawdown_reduce_start=0.10,
            drawdown_reduce_multiplier=0.50,
            drawdown_deep_start=0.20,
            drawdown_deep_multiplier=0.25,
            drawdown_scope="monthly",
        ),
        replace(
            source,
            drawdown_reduce_start=0.08,
            drawdown_reduce_multiplier=0.50,
            drawdown_deep_start=0.18,
            drawdown_deep_multiplier=0.20,
            drawdown_scope="monthly",
        ),
        replace(
            source,
            drawdown_reduce_start=0.12,
            drawdown_reduce_multiplier=0.60,
            drawdown_deep_start=0.24,
            drawdown_deep_multiplier=0.30,
            drawdown_scope="monthly",
        ),
        replace(
            source,
            drawdown_reduce_start=0.15,
            drawdown_reduce_multiplier=0.50,
            drawdown_deep_start=0.25,
            drawdown_deep_multiplier=0.20,
            drawdown_scope="monthly",
        ),
    ]
    rng = random.Random(seed)
    while len(list(dict.fromkeys(rows))) < budget:
        reduce_start = rng.choice((0.05, 0.075, 0.10, 0.125, 0.15, 0.175))
        deep_start = rng.choice((0.15, 0.175, 0.20, 0.225, 0.25, 0.30))
        if deep_start < reduce_start:
            continue
        reduce_multiplier = rng.choice((0.35, 0.45, 0.55, 0.65, 0.75, 0.85))
        deep_multiplier = rng.choice((0.10, 0.20, 0.30, 0.40, 0.50))
        if deep_multiplier > reduce_multiplier:
            continue
        rows.append(
            replace(
                source,
                drawdown_reduce_start=reduce_start,
                drawdown_reduce_multiplier=reduce_multiplier,
                drawdown_deep_start=deep_start,
                drawdown_deep_multiplier=deep_multiplier,
                drawdown_scope=rng.choice(("monthly", "monthly", "global")),
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _value(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _period_score(result: dict[str, Any], baseline: dict[str, Any]) -> float:
    if result["hard_drawdown_stopped"] or result["trade_count"] < 20:
        return -1e9
    growth = math.log(max(result["final_equity"] / result["initial_equity"], 0.01))
    baseline_growth = math.log(max(baseline["final_equity"] / baseline["initial_equity"], 0.01))
    return (
        4.0 * (growth - baseline_growth)
        + 1.6 * (_value(result["profit_factor"]) - _value(baseline["profit_factor"]))
        + 3.0 * (result["win_rate"] - baseline["win_rate"])
        + 7.0 * (baseline["max_drawdown_pct"] - result["max_drawdown_pct"])
        - 0.5 * max(0.0, result["top5_profit_contribution"] - 0.80)
    )


def _pair_score(
    six: dict[str, Any],
    three: dict[str, Any],
    baseline_six: dict[str, Any],
    baseline_three: dict[str, Any],
) -> float:
    return _period_score(six, baseline_six) + 1.25 * _period_score(
        three, baseline_three
    )


def _target_improvement(
    six: dict[str, Any],
    three: dict[str, Any],
    baseline_six: dict[str, Any],
    baseline_three: dict[str, Any],
) -> bool:
    return (
        six["trade_count"] >= 50
        and three["trade_count"] >= 25
        and six["net_profit"] > baseline_six["net_profit"]
        and three["net_profit"] > baseline_three["net_profit"]
        and _value(six["profit_factor"]) > _value(baseline_six["profit_factor"])
        and _value(three["profit_factor"])
        > _value(baseline_three["profit_factor"])
        and six["max_drawdown_pct"] <= baseline_six["max_drawdown_pct"]
        and three["max_drawdown_pct"] <= baseline_three["max_drawdown_pct"]
        and six["win_rate"] > baseline_six["win_rate"]
        and three["win_rate"] > baseline_three["win_rate"]
    )


def _stress_execution(execution: Any) -> Any:
    return replace(
        execution,
        market_slippage_bps=execution.market_slippage_bps * 1.5,
        stop_slippage_bps=execution.stop_slippage_bps * 1.5,
        take_profit_slippage_bps=execution.take_profit_slippage_bps * 1.5,
        maker_fee_rate=execution.maker_fee_rate * 1.5,
        taker_fee_rate=execution.taker_fee_rate * 1.5,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_six = datetime.fromisoformat(args.start_6m)
    start_three = datetime.fromisoformat(args.start_3m)
    end = datetime.fromisoformat(args.end)
    if not start_six < start_three < end:
        raise ValueError("periods must satisfy start_6m < start_3m < end")

    symbols = tuple(UNIVERSE_50)
    data_start = start_six - timedelta(days=args.warmup_days)
    signal_data, execution_data, rules, execution, metadata = load_v4_runtime_inputs(
        symbols,
        args.one_minute_roots,
        args.funding_roots,
        args.cost_config,
        data_start,
        end,
    )
    if metadata["minimum_coverage_ratio"] < 0.999999:
        raise RuntimeError("v6 core/runner requires gap-free stitched 1m data")
    if metadata["maximum_missing_minutes"]:
        raise RuntimeError("v6 core/runner requires gap-free stitched 1m data")
    if metadata["funding_missing_symbols"]:
        raise RuntimeError("v6 core/runner requires complete funding data")

    frozen = build_frozen_configs(args.breakout_config, args.grid_config)
    default_entry = BreakoutV6EntryConfig()

    def build_period(name: str, start: datetime, finish: datetime) -> dict[str, Any]:
        period_signal = _slice_signal_data(
            signal_data, start - timedelta(days=args.warmup_days), finish
        )
        context = build_v4_market_context(symbols, period_signal)
        raw = enrich_candidates_v4(
            build_candidates(
                symbols,
                period_signal,
                execution_data,
                frozen["breakout_build_signal"],
                start,
                finish,
            ),
            context,
        )
        baseline = _daily_cap_candidates(
            filter_candidates_v4(
                raw,
                frozen["breakout_build_signal"],
                frozen["breakout_regime"],
                context,
            ),
            2,
        )
        lane_baseline, _, _ = build_breakout_v6_lane_candidates(
            raw, context, default_entry, default_entry
        )
        expected = {
            row.signal.event_id for rows in baseline.values() for row in rows
        }
        actual = {
            row.signal.event_id for rows in lane_baseline.values() for row in rows
        }
        if actual != expected:
            raise RuntimeError(f"lane union does not reproduce frozen candidates: {name}")
        print(
            f"{name}: raw={sum(map(len, raw.values()))} baseline={len(expected)}",
            flush=True,
        )
        return {
            "start": start,
            "end": finish,
            "context": context,
            "raw": raw,
            "baseline": baseline,
        }

    periods = {
        "six": build_period("6m", start_six, end),
        "three": build_period("3m", start_three, end),
        "early": build_period("early3m", start_six, start_three),
    }

    base_signal = frozen["breakout_signal"]
    base_portfolio = frozen["breakout_portfolio"]
    base_exit = frozen["breakout_exit"]

    def baseline(period: str, selected_execution: Any = execution) -> dict[str, Any]:
        values = periods[period]
        return simulate_v6_managed_portfolio(
            values["baseline"],
            symbols,
            execution_data,
            rules,
            base_signal,
            base_portfolio,
            base_exit,
            selected_execution,
            values["start"],
            values["end"],
            args.initial_equity,
        )

    baseline_six = baseline("six")
    baseline_three = baseline("three")
    baseline_report = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    for label, actual, source in (
        (
            "six",
            baseline_six,
            baseline_report["periods"]["six_month"]["standalone"][
                "volatility_breakout"
            ],
        ),
        (
            "three",
            baseline_three,
            baseline_report["periods"]["three_month"]["standalone"][
                "volatility_breakout"
            ],
        ),
    ):
        if abs(actual["net_profit"] - source["net_profit"]) > 1e-6:
            raise RuntimeError(
                f"managed baseline mismatch {label}: "
                f"{actual['net_profit']} != {source['net_profit']}"
            )
    print(
        f"baseline 6m={baseline_six['net_profit']:+.2f}/PF{baseline_six['profit_factor']:.3f}/"
        f"win{baseline_six['win_rate']:.1%}/DD{baseline_six['max_drawdown_pct']:.1%}; "
        f"3m={baseline_three['net_profit']:+.2f}/PF{baseline_three['profit_factor']:.3f}/"
        f"win{baseline_three['win_rate']:.1%}/DD{baseline_three['max_drawdown_pct']:.1%}",
        flush=True,
    )

    candidate_cache: dict[Any, Any] = {}

    def lane_candidates(
        period: str,
        core: BreakoutV6EntryConfig,
        runner: BreakoutV6EntryConfig,
    ):
        key = (period, core, runner)
        if key not in candidate_cache:
            values = periods[period]
            candidate_cache[key] = build_breakout_v6_lane_candidates(
                values["raw"], values["context"], core, runner
            )
        return candidate_cache[key]

    def simulate(
        period: str,
        core: BreakoutV6EntryConfig,
        runner: BreakoutV6EntryConfig,
        profile: ManagedLaneProfile,
        selected_execution: Any = execution,
        candidate_delay_minutes: int = 0,
        excluded_symbols: frozenset[str] = frozenset(),
        force_compound: Optional[bool] = None,
    ) -> dict[str, Any]:
        values = periods[period]
        candidates, core_ids, runner_ids = lane_candidates(period, core, runner)
        if candidate_delay_minutes > 0:
            candidates = _shift_candidates(
                candidates, candidate_delay_minutes, execution_data
            )
        if excluded_symbols:
            candidates = _without_symbols(candidates, excluded_symbols)
        context: dict[int, dict[str, V4MarketSnapshot]] = values["context"]

        core_signal = replace(
            base_signal,
            stop_atr_multiple=profile.core_stop_atr,
            take_profit_r=60.0,
            max_holding_minutes=profile.core_max_holding_minutes,
            fail_fast_minutes=profile.core_fail_fast_minutes,
            fail_fast_min_mfe_r=profile.core_fail_fast_min_mfe_r,
            fail_fast_max_current_r=profile.core_fail_fast_max_current_r,
        )
        runner_signal = replace(
            base_signal,
            stop_atr_multiple=profile.runner_stop_atr,
            take_profit_r=60.0,
            max_holding_minutes=profile.runner_max_holding_minutes,
            fail_fast_minutes=profile.runner_fail_fast_minutes,
            fail_fast_min_mfe_r=profile.runner_fail_fast_min_mfe_r,
            fail_fast_max_current_r=profile.runner_fail_fast_max_current_r,
        )
        core_exit = ExitProtectionConfig(
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
        runner_exit = ExitProtectionConfig(
            breakeven_trigger_r=profile.runner_breakeven_trigger_r,
            profit_giveback_activation_r=(
                profile.runner_profit_giveback_activation_r
            ),
            profit_giveback_r=profile.runner_profit_giveback_r,
            partial_take_profit_r=profile.runner_partial_r,
            partial_take_profit_fraction=profile.runner_partial_fraction,
            move_stop_to_breakeven_after_partial=(
                profile.runner_move_breakeven_after_partial
            ),
        )

        def choose(
            candidate: Candidate, minute: int, _equity: float
        ) -> BreakoutV6ExecutionProfile:
            is_core = candidate.signal.event_id in core_ids
            snapshot = context.get(minute - minute % 60, {}).get(
                candidate.signal.symbol
            )
            score = (
                _regime_score(snapshot, candidate.signal.direction)
                if snapshot is not None
                else 0.0
            )
            alignment = (
                float(candidate.signal.direction.value) * snapshot.symbol_ema55_atr
                if snapshot is not None
                else -999.0
            )
            if is_core:
                risk = profile.core_risk_pct
                if (
                    score >= profile.strong_regime_score
                    and alignment >= profile.strong_alignment_atr
                ):
                    risk = profile.core_strong_risk_pct
                elif score < profile.weak_regime_score:
                    risk *= profile.weak_risk_multiplier
                portfolio = replace(
                    base_portfolio,
                    risk_per_trade_pct=risk,
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
                    "core", core_signal, portfolio, core_exit
                )
            risk = profile.runner_risk_pct
            if (
                score >= profile.strong_regime_score
                and alignment >= profile.strong_alignment_atr
            ):
                risk = profile.runner_strong_risk_pct
            elif score < profile.weak_regime_score:
                risk *= profile.weak_risk_multiplier
            portfolio = replace(
                base_portfolio,
                risk_per_trade_pct=risk,
                long_risk_multiplier=profile.runner_long_multiplier,
                short_risk_multiplier=profile.runner_short_multiplier,
                ranking_mode=profile.ranking_mode,
                compound=(
                    base_portfolio.compound
                    if force_compound is None
                    else force_compound
                ),
            )
            return BreakoutV6ExecutionProfile(
                "runner", runner_signal, portfolio, runner_exit
            )

        def priority(candidate: Candidate):
            return (
                0 if candidate.signal.event_id in core_ids else 1,
                *_candidate_sort_key(candidate, profile.ranking_mode),
            )

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
            elif profile.drawdown_scope != "global":
                raise ValueError(
                    f"unsupported drawdown scope: {profile.drawdown_scope}"
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
            base_exit,
            selected_execution,
            values["start"],
            values["end"],
            args.initial_equity,
            profile_selector=choose,
            priority_selector=priority,
            profile_governor=govern,
        )
        result["core_candidate_count"] = len(core_ids)
        result["runner_candidate_count"] = len(runner_ids)
        return result

    fixed = ManagedLaneProfile()
    entry_rows: list[dict[str, Any]] = []
    core_variants = _core_variants(args.v6_report, args.core_budget)
    runner_variants = _runner_variants()[: args.runner_budget]
    total = len(core_variants) * len(runner_variants)
    number = 0
    for core in core_variants:
        for runner in runner_variants:
            number += 1
            six = simulate("six", core, runner, fixed)
            entry_rows.append(
                {
                    "core": core,
                    "runner": runner,
                    "profile": fixed,
                    "six": six,
                    "score6": _period_score(six, baseline_six),
                }
            )
            if number == 1 or number % 12 == 0 or number == total:
                print(
                    f"entry {number}/{total}: {six['net_profit']:+.0f}/"
                    f"PF{six['profit_factor']:.2f}/win{six['win_rate']:.1%}/"
                    f"DD{six['max_drawdown_pct']:.1%}",
                    flush=True,
                )
    entry_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in entry_rows[: args.entry_recent_finalists]:
        row["three"] = simulate(
            "three", row["core"], row["runner"], row["profile"]
        )
        row["pair_score"] = _pair_score(
            row["six"], row["three"], baseline_six, baseline_three
        )
    entry_finalists = sorted(
        entry_rows[: args.entry_recent_finalists],
        key=lambda row: row["pair_score"],
        reverse=True,
    )[: args.entry_finalists]

    risk_rows: list[dict[str, Any]] = []
    risk_variants = _risk_variants(args.seed + 1, args.risk_budget)
    for entry_number, source in enumerate(entry_finalists, 1):
        for profile in risk_variants:
            six = simulate("six", source["core"], source["runner"], profile)
            risk_rows.append(
                {
                    "core": source["core"],
                    "runner": source["runner"],
                    "profile": profile,
                    "six": six,
                    "score6": _period_score(six, baseline_six),
                }
            )
        print(
            f"risk search entry {entry_number}/{len(entry_finalists)} complete",
            flush=True,
        )
    risk_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in risk_rows[: args.risk_recent_finalists]:
        row["three"] = simulate(
            "three", row["core"], row["runner"], row["profile"]
        )
        row["pair_score"] = _pair_score(
            row["six"], row["three"], baseline_six, baseline_three
        )
    risk_finalists = sorted(
        risk_rows[: args.risk_recent_finalists],
        key=lambda row: row["pair_score"],
        reverse=True,
    )[: args.risk_finalists]

    exit_rows: list[dict[str, Any]] = []
    for risk_number, source in enumerate(risk_finalists, 1):
        for profile in _exit_variants(
            source["profile"], args.seed + 100 + risk_number, args.exit_budget
        ):
            six = simulate("six", source["core"], source["runner"], profile)
            exit_rows.append(
                {
                    "core": source["core"],
                    "runner": source["runner"],
                    "profile": profile,
                    "six": six,
                    "score6": _period_score(six, baseline_six),
                }
            )
        print(
            f"exit search risk {risk_number}/{len(risk_finalists)} complete",
            flush=True,
        )
    exit_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in exit_rows[: args.exit_recent_finalists]:
        row["three"] = simulate(
            "three", row["core"], row["runner"], row["profile"]
        )
        row["pair_score"] = _pair_score(
            row["six"], row["three"], baseline_six, baseline_three
        )
        row["target_improvement"] = _target_improvement(
            row["six"], row["three"], baseline_six, baseline_three
        )
    recent = sorted(
        exit_rows[: args.exit_recent_finalists],
        key=lambda row: (
            row.get("target_improvement", False), row["pair_score"]
        ),
        reverse=True,
    )

    if args.refine_from_report:
        source_payload = json.loads(
            Path(args.refine_from_report).read_text(encoding="utf-8")
        )
        seed_payloads = [source_payload["selected"]]
        seed_payloads.extend(
            source_payload.get("leaderboard", ())[: args.refine_seeds]
        )
        seeds: list[tuple[Any, Any, ManagedLaneProfile]] = []
        for payload in seed_payloads:
            seed = (
                _entry_from_dict(payload["core_entry"]),
                _entry_from_dict(payload["runner_entry"]),
                ManagedLaneProfile(**payload["managed_profile"]),
            )
            if seed not in seeds:
                seeds.append(seed)
        no_runner = BreakoutV6EntryConfig(
            long=_reject_side(), short=_reject_side()
        )
        refinement_rows: list[dict[str, Any]] = []
        risk_points: list[tuple[str, ManagedLaneProfile]] = []
        for _core, _runner, source_profile in seeds:
            if (
                args.refine_core_giveback_activations
                and args.refine_core_giveback_distances
            ):
                risk_points.append(("giveback=off", source_profile))
                for activation in args.refine_core_giveback_activations:
                    for distance in args.refine_core_giveback_distances:
                        if distance >= activation:
                            continue
                        risk_points.append(
                            (
                                f"giveback={activation:.1f}/{distance:.1f}",
                                replace(
                                    source_profile,
                                    core_profit_giveback_activation_r=activation,
                                    core_profit_giveback_r=distance,
                                ),
                            )
                        )
            elif args.refine_governor_budget > 0:
                for governed in _drawdown_governor_variants(
                    source_profile,
                    args.seed + len(risk_points) + 700,
                    args.refine_governor_budget,
                    args.refine_governor_high_global,
                ):
                    risk_points.append(
                        (
                            "govern="
                            f"{governed.drawdown_scope}/"
                            f"{governed.drawdown_reduce_start:.3f}/"
                            f"{governed.drawdown_reduce_multiplier:.2f}/"
                            f"{governed.drawdown_deep_start:.3f}/"
                            f"{governed.drawdown_deep_multiplier:.2f}",
                            governed,
                        )
                    )
            elif args.refine_core_base_risks and args.refine_core_strong_risks:
                for base_risk in args.refine_core_base_risks:
                    for strong_risk in args.refine_core_strong_risks:
                        base_ratio = base_risk / max(
                            source_profile.core_risk_pct, 1e-12
                        )
                        strong_ratio = strong_risk / max(
                            source_profile.core_strong_risk_pct, 1e-12
                        )
                        risk_points.append(
                            (
                                f"base={base_risk:.4f}/strong={strong_risk:.4f}",
                                replace(
                                    source_profile,
                                    core_risk_pct=base_risk,
                                    core_strong_risk_pct=strong_risk,
                                    runner_risk_pct=(
                                        source_profile.runner_risk_pct
                                        * base_ratio
                                    ),
                                    runner_strong_risk_pct=(
                                        source_profile.runner_strong_risk_pct
                                        * strong_ratio
                                    ),
                                ),
                            )
                        )
            else:
                for scale in args.refine_scales:
                    risk_points.append(
                        (
                            f"scale={scale:.2f}",
                            replace(
                                source_profile,
                                core_risk_pct=(
                                    source_profile.core_risk_pct * scale
                                ),
                                runner_risk_pct=(
                                    source_profile.runner_risk_pct * scale
                                ),
                                core_strong_risk_pct=(
                                    source_profile.core_strong_risk_pct * scale
                                ),
                                runner_strong_risk_pct=(
                                    source_profile.runner_strong_risk_pct * scale
                                ),
                            ),
                        )
                    )
        # Risk points are generated per seed so profiles keep the seed's other
        # exit and regime fields.  Consume them in matching contiguous blocks.
        points_per_seed = len(risk_points) // max(len(seeds), 1)
        total = len(seeds) * 2 * points_per_seed
        number = 0
        for seed_number, (core, source_runner, source_profile) in enumerate(seeds):
            start_point = seed_number * points_per_seed
            seed_points = risk_points[start_point : start_point + points_per_seed]
            for runner in (source_runner, no_runner):
                for label, profile in seed_points:
                    number += 1
                    six = simulate("six", core, runner, profile)
                    refinement_rows.append(
                        {
                            "core": core,
                            "runner": runner,
                            "profile": profile,
                            "six": six,
                            "score6": _period_score(six, baseline_six),
                        }
                    )
                    if number == 1 or number % 10 == 0 or number == total:
                        print(
                            f"drawdown refine {number}/{total}: {label} "
                            f"6m={six['net_profit']:+.0f}/PF{six['profit_factor']:.2f}/"
                            f"win{six['win_rate']:.1%}/DD{six['max_drawdown_pct']:.1%}",
                            flush=True,
                        )
        refinement_rows.sort(key=lambda row: row["score6"], reverse=True)
        six_viable = [
            row
            for row in refinement_rows
            if row["six"]["net_profit"] > baseline_six["net_profit"]
            and _value(row["six"]["profit_factor"])
            > _value(baseline_six["profit_factor"])
            and row["six"]["max_drawdown_pct"]
            <= baseline_six["max_drawdown_pct"]
            and row["six"]["win_rate"] > baseline_six["win_rate"]
        ]
        to_validate: list[dict[str, Any]] = []
        for row in [
            *six_viable,
            *refinement_rows[: args.refine_recent_finalists],
        ]:
            if row not in to_validate:
                to_validate.append(row)
        for number, row in enumerate(to_validate, 1):
            row["three"] = simulate(
                "three", row["core"], row["runner"], row["profile"]
            )
            row["pair_score"] = _pair_score(
                row["six"], row["three"], baseline_six, baseline_three
            )
            row["target_improvement"] = _target_improvement(
                row["six"], row["three"], baseline_six, baseline_three
            )
            if number % 10 == 0 or number == len(to_validate):
                print(
                    f"drawdown recent validation {number}/{len(to_validate)}",
                    flush=True,
                )
        recent = sorted(
            to_validate,
            key=lambda row: (
                row["target_improvement"], row["pair_score"]
            ),
            reverse=True,
        )
        exit_rows = refinement_rows

    robust_rows: list[dict[str, Any]] = []
    stressed = _stress_execution(execution)
    for number, row in enumerate(recent[: args.robust_finalists], 1):
        early = simulate("early", row["core"], row["runner"], row["profile"])
        stress_six = simulate(
            "six", row["core"], row["runner"], row["profile"], stressed
        )
        stress_three = simulate(
            "three", row["core"], row["runner"], row["profile"], stressed
        )
        delay_six = simulate(
            "six",
            row["core"],
            row["runner"],
            row["profile"],
            candidate_delay_minutes=1,
        )
        delay_three = simulate(
            "three",
            row["core"],
            row["runner"],
            row["profile"],
            candidate_delay_minutes=1,
        )
        fixed_six = simulate(
            "six",
            row["core"],
            row["runner"],
            row["profile"],
            force_compound=False,
        )
        fixed_three = simulate(
            "three",
            row["core"],
            row["runner"],
            row["profile"],
            force_compound=False,
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
        no_top_six = simulate(
            "six",
            row["core"],
            row["runner"],
            row["profile"],
            excluded_symbols=(
                frozenset({top_symbol_six}) if top_symbol_six else frozenset()
            ),
        )
        no_top_three = simulate(
            "three",
            row["core"],
            row["runner"],
            row["profile"],
            excluded_symbols=(
                frozenset({top_symbol_three}) if top_symbol_three else frozenset()
            ),
        )
        active_lane_rows = [
            lane
            for period_result in (row["six"], row["three"])
            for lane in (period_result.get("by_v6_lane") or {}).values()
            if int(lane.get("trade_count", 0)) > 0
        ]
        lane_robust = bool(active_lane_rows) and all(
            _value(lane.get("net_pnl")) > 0.0
            and _value(lane.get("profit_factor")) > 1.0
            for lane in active_lane_rows
        )
        robust = (
            row["target_improvement"]
            and lane_robust
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
                "lane_robust": lane_robust,
                "robust": robust,
                "robust_score": (
                    row["pair_score"]
                    + 0.35 * _period_score(early, baseline_six)
                    + (12.0 if robust else 0.0)
                ),
            }
        )
        robust_rows.append(row)
        print(
            f"robust {number}/{args.robust_finalists}: target={row['target_improvement']} "
            f"early={early['net_profit']:+.0f} stress6={stress_six['net_profit']:+.0f} "
            f"delay6={delay_six['net_profit']:+.0f} "
            f"fixed6={fixed_six['net_profit']:+.0f} "
            f"noTop6={no_top_six['net_profit']:+.0f}",
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
            "target_improvement": row.get("target_improvement", False),
            "lane_robust": row.get("lane_robust", False),
            "robust": row.get("robust", False),
            "core_entry": row["core"].as_dict(),
            "runner_entry": row["runner"].as_dict(),
            "managed_profile": asdict(row["profile"]),
            "six_month": compact_v6_metrics(row["six"]),
            "three_month": compact_v6_metrics(row["three"]),
        }
        payload["six_month"]["by_v6_lane"] = row["six"].get("by_v6_lane")
        payload["three_month"]["by_v6_lane"] = row["three"].get("by_v6_lane")
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
                payload[target] = compact_v6_metrics(row[source])
        if "top_symbol_six" in row:
            payload["removed_top_symbol_six_month"] = row["top_symbol_six"]
            payload["removed_top_symbol_three_month"] = row["top_symbol_three"]
        if full:
            payload["six_month_full"] = row["six"]
            payload["three_month_full"] = row["three"]
        return payload

    selection_status = (
        "strict_robust_improvement"
        if selected.get("robust")
        else "pareto_candidate_requires_further_optimization"
    )
    report = {
        "strategy_name": V6_CORE_RUNNER_NAME,
        "status": "independent_research_gui_v5_grid_v3_unchanged",
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
            "entry_evaluations": len(entry_rows),
            "risk_evaluations": len(risk_rows),
            "exit_evaluations": len(exit_rows),
            "robust_evaluations": len(robust_rows),
        },
        "baseline": {
            "strategy": "Hybrid v5 Breakout sleeve",
            "source_report": args.baseline_report,
            "six_month": compact_v6_metrics(baseline_six),
            "three_month": compact_v6_metrics(baseline_three),
            "six_month_full": baseline_six,
            "three_month_full": baseline_three,
        },
        "selected": public(selected, True),
        "leaderboard": [public(row) for row in recent[:20]],
        "preserved": {
            "active_gui": "unchanged",
            "hybrid_v5": "unchanged",
            "grid_v3": "unchanged",
            "apt_grid": "unchanged",
        },
    }
    _write_json(args.output, report)

    config = {
        "strategy_name": V6_CORE_RUNNER_NAME,
        "status": "frozen_independent_research_candidate_not_live",
        "selection_status": selection_status,
        "core_entry": selected["core"].as_dict(),
        "runner_entry": selected["runner"].as_dict(),
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
        "# Breakout v6 core/runner optimization",
        "",
        "- Gap-free 1m execution, fees, slippage, funding and point-in-time market context",
        "- Core and runner use independent risk/exit profiles; global max position remains one",
        "- GUI, Hybrid v5 and Grid v3 are unchanged",
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
            ("Hybrid v5 baseline", baseline_result),
            ("Breakout v6 core/runner", selected_result),
        ):
            lines.append(
                f"| {period} | {name} | {result['trade_count']} | "
                f"{result['net_profit']:+.2f}U | {result['profit_factor']:.3f} | "
                f"{result['win_rate']:.2%} | {result['max_drawdown_pct']:.2%} |"
            )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "strategy_name": V6_CORE_RUNNER_NAME,
        "status": "independent_research_artifacts",
        "config": args.config_output,
        "report": args.output,
        "summary": args.summary,
        "hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                "crypto_scalper/volatility_breakout_v6.py",
                "crypto_scalper/volatility_breakout_v6_engine.py",
                "crypto_scalper/volatility_breakout_v6_core_runner_optimize.py",
                "tests/test_volatility_breakout_v6.py",
                args.config_output,
                args.output,
                args.summary,
            )
        },
        "preserved": report["preserved"],
    }
    _write_json(args.manifest, manifest)
    print(
        f"selected target={selected.get('target_improvement', False)} "
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
        description="Breakout v6 core/runner live-realistic optimization"
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
        "--v6-report", default="reports/volatility_breakout_v6_robust_3m_6m.json"
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
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--core-budget", type=int, default=8)
    parser.add_argument("--runner-budget", type=int, default=12)
    parser.add_argument("--entry-recent-finalists", type=int, default=24)
    parser.add_argument("--entry-finalists", type=int, default=4)
    parser.add_argument("--risk-budget", type=int, default=32)
    parser.add_argument("--risk-recent-finalists", type=int, default=28)
    parser.add_argument("--risk-finalists", type=int, default=4)
    parser.add_argument("--exit-budget", type=int, default=24)
    parser.add_argument("--exit-recent-finalists", type=int, default=24)
    parser.add_argument("--robust-finalists", type=int, default=8)
    parser.add_argument(
        "--refine-from-report",
        default="",
        help="Optional prior core/runner report used for focused drawdown scaling",
    )
    parser.add_argument("--refine-seeds", type=int, default=5)
    parser.add_argument(
        "--refine-scales",
        type=float,
        nargs="+",
        default=(0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    )
    parser.add_argument("--refine-recent-finalists", type=int, default=36)
    parser.add_argument(
        "--refine-core-base-risks", type=float, nargs="+", default=()
    )
    parser.add_argument(
        "--refine-core-strong-risks", type=float, nargs="+", default=()
    )
    parser.add_argument("--refine-governor-budget", type=int, default=0)
    parser.add_argument(
        "--refine-governor-high-global", action="store_true"
    )
    parser.add_argument(
        "--refine-core-giveback-activations", type=float, nargs="+", default=()
    )
    parser.add_argument(
        "--refine-core-giveback-distances", type=float, nargs="+", default=()
    )
    parser.add_argument(
        "--output",
        default="reports/volatility_breakout_v6_core_runner_3m_6m.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/volatility_breakout_v6_core_runner_3m_6m.md",
    )
    parser.add_argument(
        "--config-output",
        default="config.volatility-breakout.v6-core-runner-50.json",
    )
    parser.add_argument(
        "--manifest",
        default="config.volatility-breakout.v6-core-runner-50-manifest.json",
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

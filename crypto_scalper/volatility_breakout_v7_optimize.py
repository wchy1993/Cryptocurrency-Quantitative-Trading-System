from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from .combined_hybrid_v5_grid_v3_backtest import (
    _slice_signal_data,
    _write_json,
    build_frozen_configs,
)
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
    VOLATILITY_BREAKOUT_V7_NAME,
    BreakoutV7ConfidencePolicy,
    BreakoutV7EntryTiming,
    apply_breakout_v7_timing,
    breakout_v7_risk_multiplier,
)


def _entry_from_dict(payload: dict[str, Any]) -> BreakoutV6EntryConfig:
    return BreakoutV6EntryConfig(
        long=BreakoutV6SideGate(**payload["long"]),
        short=BreakoutV6SideGate(**payload["short"]),
        max_signals_per_symbol_day=int(payload["max_signals_per_symbol_day"]),
        require_market_context=bool(payload["require_market_context"]),
    )


def _load_anchor(path: str) -> tuple[
    BreakoutV6EntryConfig,
    ManagedLaneProfile,
    VolatilityBreakoutConfig,
    PortfolioSearchConfig,
    dict[str, Any],
]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        _entry_from_dict(payload["core_entry"]),
        ManagedLaneProfile(**payload["managed_profile"]),
        VolatilityBreakoutConfig(**payload["operational_signal"]),
        PortfolioSearchConfig(**payload["operational_portfolio"]),
        payload,
    )


def _value(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _period_score(result: dict[str, Any], anchor: dict[str, Any]) -> float:
    coverage = result["trade_count"] / max(anchor["trade_count"], 1)
    if result["hard_drawdown_stopped"] or coverage < 0.65:
        return -1e9
    growth = math.log(max(result["final_equity"] / result["initial_equity"], 0.01))
    anchor_growth = math.log(
        max(anchor["final_equity"] / anchor["initial_equity"], 0.01)
    )
    return (
        5.0 * (growth - anchor_growth)
        + 2.2 * (_value(result["profit_factor"]) - _value(anchor["profit_factor"]))
        + 4.0 * (result["win_rate"] - anchor["win_rate"])
        + 10.0 * (anchor["max_drawdown_pct"] - result["max_drawdown_pct"])
        - 3.0 * max(0.0, 1.0 - coverage)
        - 1.0 * max(0.0, result["top5_profit_contribution"] - 0.72)
    )


def _pair_score(
    six: dict[str, Any],
    three: dict[str, Any],
    anchor_six: dict[str, Any],
    anchor_three: dict[str, Any],
) -> float:
    return _period_score(six, anchor_six) + 1.35 * _period_score(
        three, anchor_three
    )


def _strict_improvement(
    six: dict[str, Any],
    three: dict[str, Any],
    anchor_six: dict[str, Any],
    anchor_three: dict[str, Any],
) -> bool:
    return (
        six["trade_count"] >= 60
        and three["trade_count"] >= 30
        and six["net_profit"] > anchor_six["net_profit"]
        and three["net_profit"] > anchor_three["net_profit"]
        and _value(six["profit_factor"]) > _value(anchor_six["profit_factor"])
        and _value(three["profit_factor"]) > _value(anchor_three["profit_factor"])
        and six["max_drawdown_pct"] <= anchor_six["max_drawdown_pct"]
        and three["max_drawdown_pct"] <= anchor_three["max_drawdown_pct"]
        and six["win_rate"] >= anchor_six["win_rate"]
        and three["win_rate"] >= anchor_three["win_rate"]
    )


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


def _gate_variants(
    anchor: BreakoutV6EntryConfig, seed: int, budget: int
) -> list[BreakoutV6EntryConfig]:
    long_rows = [anchor.long]
    long_rows.extend(
        replace(anchor.long, max_quality_score=value)
        for value in (2.00, 2.10, 2.20, 2.35, 2.50, 2.75, 3.00)
    )
    long_rows.extend(
        replace(anchor.long, min_volume_ratio=value)
        for value in (2.0, 3.5, 5.0, 6.5)
    )
    long_rows.extend(
        replace(anchor.long, max_directional_breadth=value)
        for value in (0.78, 0.82, 0.86, 0.90, 0.95)
    )
    short_rows = [anchor.short]
    short_rows.extend(
        replace(anchor.short, min_quality_score=value)
        for value in (0.75, 0.90, 1.00, 1.10, 1.20, 1.30)
    )
    short_rows.extend(
        replace(anchor.short, max_directional_breadth=value)
        for value in (0.80, 0.84, 0.87, 0.90, 0.94)
    )
    short_rows.extend(
        replace(anchor.short, max_breakout_extension_atr=value)
        for value in (0.35, 0.45, 0.60, 0.80, 1.20)
    )
    rows = [anchor]
    rows.extend(replace(anchor, long=long) for long in long_rows[1:])
    rows.extend(replace(anchor, short=short) for short in short_rows[1:])
    rng = random.Random(seed)
    while len(dict.fromkeys(rows)) < budget:
        long = replace(
            anchor.long,
            max_quality_score=rng.choice((2.05, 2.20, 2.40, 2.75, 999.0)),
            min_volume_ratio=rng.choice((0.0, 2.0, 3.5, 5.0)),
            max_breakout_extension_atr=rng.choice((2.0, 2.5, 3.0, 999.0)),
            max_directional_breadth=rng.choice((0.80, 0.86, 0.92, 1.0)),
        )
        short = replace(
            anchor.short,
            min_quality_score=rng.choice((-999.0, 0.85, 1.0, 1.15, 1.30)),
            min_body_atr=rng.choice((0.0, 0.4, 0.6, 0.8)),
            min_volume_ratio=rng.choice((0.0, 1.0, 1.5, 2.0)),
            max_breakout_extension_atr=rng.choice((0.35, 0.45, 0.60, 1.0, 999.0)),
            max_directional_breadth=rng.choice((0.82, 0.86, 0.90, 0.95, 1.0)),
        )
        rows.append(replace(anchor, long=long, short=short))
    return list(dict.fromkeys(rows))[:budget]


def _policy_variants(seed: int, budget: int) -> list[BreakoutV7ConfidencePolicy]:
    rows = [BreakoutV7ConfidencePolicy()]
    rows.extend(
        [
            BreakoutV7ConfidencePolicy(
                long_max_quality_score=2.20,
                long_min_body_atr=2.40,
                long_min_volume_ratio=4.0,
                long_max_breakout_extension_atr=2.50,
                long_max_directional_breadth=0.90,
                short_min_quality_score=1.00,
                short_min_body_atr=0.55,
                short_min_volume_ratio=1.20,
                short_max_breakout_extension_atr=0.50,
                short_max_directional_breadth=0.90,
                strong_score=4,
                weak_score=1,
                strong_risk_multiplier=1.20,
                weak_risk_multiplier=0.65,
            ),
            BreakoutV7ConfidencePolicy(
                long_max_quality_score=2.35,
                long_min_body_atr=2.60,
                long_min_volume_ratio=5.0,
                long_max_breakout_extension_atr=2.50,
                long_max_directional_breadth=0.90,
                short_min_quality_score=1.10,
                short_min_body_atr=0.60,
                short_min_volume_ratio=1.50,
                short_max_breakout_extension_atr=0.45,
                short_max_directional_breadth=0.88,
                strong_score=4,
                weak_score=2,
                strong_risk_multiplier=1.25,
                weak_risk_multiplier=0.60,
            ),
            BreakoutV7ConfidencePolicy(
                long_max_quality_score=2.50,
                long_min_body_atr=2.20,
                long_min_volume_ratio=3.5,
                long_max_breakout_extension_atr=2.75,
                long_max_directional_breadth=0.92,
                short_min_quality_score=0.95,
                short_min_body_atr=0.50,
                short_min_volume_ratio=1.0,
                short_max_breakout_extension_atr=0.60,
                short_max_directional_breadth=0.90,
                strong_score=4,
                weak_score=1,
                strong_risk_multiplier=1.30,
                weak_risk_multiplier=0.70,
            ),
        ]
    )
    rng = random.Random(seed)
    while len(dict.fromkeys(rows)) < budget:
        weak_score = rng.choice((0, 1, 2))
        strong_score = rng.choice((3, 4, 5))
        if weak_score >= strong_score:
            continue
        rows.append(
            BreakoutV7ConfidencePolicy(
                long_max_quality_score=rng.choice((2.05, 2.20, 2.40, 2.70, 999.0)),
                long_min_body_atr=rng.choice((1.8, 2.2, 2.5, 2.8)),
                long_min_volume_ratio=rng.choice((2.0, 3.5, 5.0, 6.5)),
                long_max_breakout_extension_atr=rng.choice((1.8, 2.2, 2.6, 3.0)),
                long_max_directional_breadth=rng.choice((0.82, 0.86, 0.90, 0.95)),
                short_min_quality_score=rng.choice((0.75, 0.90, 1.05, 1.20, 1.30)),
                short_min_body_atr=rng.choice((0.35, 0.50, 0.65, 0.80)),
                short_min_volume_ratio=rng.choice((0.8, 1.2, 1.6, 2.0)),
                short_max_breakout_extension_atr=rng.choice((0.32, 0.40, 0.50, 0.65)),
                short_max_directional_breadth=rng.choice((0.82, 0.86, 0.90, 0.94)),
                strong_score=strong_score,
                weak_score=weak_score,
                strong_risk_multiplier=rng.choice((1.10, 1.15, 1.20, 1.25, 1.30)),
                weak_risk_multiplier=rng.choice((0.45, 0.55, 0.65, 0.75, 0.85)),
                long_risk_multiplier=rng.choice((0.90, 0.95, 1.0, 1.05)),
                short_risk_multiplier=rng.choice((0.90, 0.95, 1.0, 1.05)),
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _local_gate_variants(
    source: BreakoutV6EntryConfig,
) -> list[BreakoutV6EntryConfig]:
    rows = [source]
    rows.extend(
        replace(
            source,
            short=replace(source.short, min_quality_score=value),
        )
        for value in (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
    )
    rows.extend(
        replace(
            source,
            short=replace(
                source.short,
                min_quality_score=quality,
                max_directional_breadth=breadth,
            ),
        )
        for quality, breadth in (
            (0.95, 0.95),
            (1.00, 0.95),
            (1.05, 0.95),
            (1.00, 0.90),
        )
    )
    return list(dict.fromkeys(rows))


def _local_policy_variants(
    source: BreakoutV7ConfidencePolicy,
    seed: int,
    budget: int,
) -> list[BreakoutV7ConfidencePolicy]:
    rows = [source, BreakoutV7ConfidencePolicy()]
    field_values = {
        "long_max_quality_score": (2.20, 2.30, 2.40, 2.50, 2.60),
        "long_min_body_atr": (2.40, 2.60, 2.70, 2.80, 2.90, 3.00),
        "long_min_volume_ratio": (1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00),
        "long_max_breakout_extension_atr": (2.30, 2.45, 2.60, 2.75, 2.90),
        "long_max_directional_breadth": (0.84, 0.86, 0.88, 0.90, 0.92, 0.94),
        "short_min_quality_score": (0.95, 1.00, 1.05, 1.10, 1.15),
        "short_min_body_atr": (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
        "short_min_volume_ratio": (1.25, 1.50, 1.75, 2.00, 2.25, 2.50),
        "short_max_breakout_extension_atr": (0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.40),
        "short_max_directional_breadth": (0.80, 0.82, 0.84, 0.86, 0.88, 0.90),
        "strong_risk_multiplier": (1.18, 1.22, 1.26, 1.30, 1.34, 1.38),
        "weak_risk_multiplier": (0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
        "long_risk_multiplier": (0.95, 1.00, 1.05, 1.10),
        "short_risk_multiplier": (0.82, 0.86, 0.90, 0.94, 0.98),
    }
    for field, values in field_values.items():
        rows.extend(replace(source, **{field: value}) for value in values)
    rng = random.Random(seed)
    fields = tuple(field_values)
    while len(dict.fromkeys(rows)) < budget:
        selected_fields = rng.sample(fields, rng.choice((2, 3, 4, 5)))
        changes = {
            field: rng.choice(field_values[field]) for field in selected_fields
        }
        rows.append(replace(source, **changes))
    return list(dict.fromkeys(rows))[:budget]


def _profile_variants(anchor: ManagedLaneProfile, budget: int) -> list[ManagedLaneProfile]:
    rows = [anchor]
    rows.extend(replace(anchor, core_stop_atr=value) for value in (0.72, 0.76, 0.84, 0.88))
    rows.extend(
        replace(anchor, core_max_holding_minutes=value)
        for value in (1080, 1140, 1260, 1320, 1440)
    )
    rows.extend(
        replace(anchor, core_breakeven_trigger_r=value)
        for value in (4.0, 4.5, 5.5, 6.0)
    )
    for activation, distance in ((12.0, 9.0), (14.0, 10.0), (16.0, 12.0), (18.0, 14.0)):
        rows.append(
            replace(
                anchor,
                core_profit_giveback_activation_r=activation,
                core_profit_giveback_r=distance,
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _timing_variants() -> list[BreakoutV7EntryTiming]:
    return [
        BreakoutV7EntryTiming(),
        BreakoutV7EntryTiming(1, -0.10, 0.40, 0.35),
        BreakoutV7EntryTiming(1, -0.05, 0.30, 0.25),
        BreakoutV7EntryTiming(1, 0.00, 0.35, 0.25),
        BreakoutV7EntryTiming(1, 0.02, 0.25, 0.20),
        BreakoutV7EntryTiming(1, -0.15, 0.25, 0.20),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_six = datetime.fromisoformat(args.start_6m)
    start_three = datetime.fromisoformat(args.start_3m)
    end = datetime.fromisoformat(args.end)
    if not start_six < start_three < end:
        raise ValueError("periods must satisfy start_6m < start_3m < end")

    anchor_gate, anchor_profile, base_signal, base_portfolio, anchor_config = (
        _load_anchor(args.anchor_config)
    )
    frozen = build_frozen_configs(
        args.source_breakout_config, args.source_grid_config
    )
    build_signal = frozen["breakout_build_signal"]
    anchor_report = json.loads(Path(args.anchor_report).read_text(encoding="utf-8"))
    anchor_selected = anchor_report["selected"]
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
        raise RuntimeError("Breakout v7 requires gap-free stitched 1m data")
    if metadata["maximum_missing_minutes"] or metadata["funding_missing_symbols"]:
        raise RuntimeError("Breakout v7 requires complete price and funding data")

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
        print(f"{name}: raw={sum(map(len, raw.values()))}", flush=True)
        return {"start": start, "end": finish, "raw": raw, "context": context}

    periods = {
        "six": build_period("6m", start_six, end),
        "three": build_period("3m", start_three, end),
        "early": build_period("early3m", start_six, start_three),
    }
    candidate_cache: dict[Any, dict[int, list[Candidate]]] = {}

    def candidates_for(
        period: str,
        gate: BreakoutV6EntryConfig,
        timing: BreakoutV7EntryTiming,
    ) -> dict[int, list[Candidate]]:
        key = (period, gate, timing)
        if key not in candidate_cache:
            values = periods[period]
            filtered = filter_breakout_v6_candidates(
                values["raw"], values["context"], gate
            )
            candidate_cache[key] = apply_breakout_v7_timing(
                filtered, execution_data, timing
            )
        return candidate_cache[key]

    def simulate(
        period: str,
        gate: BreakoutV6EntryConfig,
        policy: BreakoutV7ConfidencePolicy,
        timing: BreakoutV7EntryTiming,
        profile: ManagedLaneProfile,
        selected_execution: BacktestExecutionConfig = execution,
        extra_delay_minutes: int = 0,
        force_compound: Optional[bool] = None,
        excluded_symbols: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        values = periods[period]
        candidates = candidates_for(period, gate, timing)
        if extra_delay_minutes:
            candidates = _shift_candidates(candidates, extra_delay_minutes, execution_data)
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
            partial_take_profit_r=profile.core_partial_r,
            partial_take_profit_fraction=profile.core_partial_fraction,
            move_stop_to_breakeven_after_partial=profile.core_move_breakeven_after_partial,
        )

        def choose(
            candidate: Candidate, minute: int, _equity: float
        ) -> Optional[BreakoutV6ExecutionProfile]:
            snapshot = context.get(minute - minute % 60, {}).get(candidate.signal.symbol)
            regime = _regime_score(snapshot, candidate.signal.direction) if snapshot else 0.0
            alignment = (
                float(candidate.signal.direction.value) * snapshot.symbol_ema55_atr
                if snapshot
                else -999.0
            )
            risk = profile.core_risk_pct
            if regime >= profile.strong_regime_score and alignment >= profile.strong_alignment_atr:
                risk = profile.core_strong_risk_pct
            elif regime < profile.weak_regime_score:
                risk *= profile.weak_risk_multiplier
            multiplier, score, tier = breakout_v7_risk_multiplier(
                candidate, snapshot, policy
            )
            if multiplier <= 0.0:
                return None
            portfolio = replace(
                base_portfolio,
                risk_per_trade_pct=risk * multiplier,
                long_risk_multiplier=profile.core_long_multiplier,
                short_risk_multiplier=profile.core_short_multiplier,
                ranking_mode=profile.ranking_mode,
                compound=base_portfolio.compound if force_compound is None else force_compound,
            )
            return BreakoutV6ExecutionProfile(
                f"v7_{tier}_score_{score}", signal, portfolio, exit_config
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
                    (governor_peak - equity) / governor_peak if governor_peak > 0.0 else 1.0
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
                    risk_per_trade_pct=selected.portfolio.risk_per_trade_pct * multiplier,
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
                compound=base_portfolio.compound if force_compound is None else force_compound,
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
        return result

    anchor_policy = BreakoutV7ConfidencePolicy()
    anchor_timing = BreakoutV7EntryTiming()
    anchor_six = simulate(
        "six", anchor_gate, anchor_policy, anchor_timing, anchor_profile
    )
    anchor_three = simulate(
        "three", anchor_gate, anchor_policy, anchor_timing, anchor_profile
    )
    for label, actual, expected in (
        ("six", anchor_six, anchor_selected["six_month_full"]),
        ("three", anchor_three, anchor_selected["three_month_full"]),
    ):
        if abs(actual["net_profit"] - expected["net_profit"]) > 1e-6:
            raise RuntimeError(
                f"Breakout v7 anchor mismatch {label}: "
                f"{actual['net_profit']} != {expected['net_profit']}"
            )
    print(
        f"v6 anchor 6m={anchor_six['net_profit']:+.2f}/PF{anchor_six['profit_factor']:.3f}/"
        f"DD{anchor_six['max_drawdown_pct']:.1%}; "
        f"3m={anchor_three['net_profit']:+.2f}/PF{anchor_three['profit_factor']:.3f}/"
        f"DD{anchor_three['max_drawdown_pct']:.1%}",
        flush=True,
    )

    search_gate = anchor_gate
    search_policy = anchor_policy
    search_timing = anchor_timing
    search_profile = anchor_profile
    if args.refine_from_report:
        refine_payload = json.loads(
            Path(args.refine_from_report).read_text(encoding="utf-8")
        )["selected"]
        search_gate = _entry_from_dict(refine_payload["entry"])
        search_policy = BreakoutV7ConfidencePolicy(
            **refine_payload["confidence_policy"]
        )
        search_timing = BreakoutV7EntryTiming(**refine_payload["entry_timing"])
        search_profile = ManagedLaneProfile(**refine_payload["managed_profile"])

    gate_rows: list[dict[str, Any]] = []
    gates = (
        _local_gate_variants(search_gate)[: args.gate_budget]
        if args.refine_from_report
        else _gate_variants(anchor_gate, args.seed, args.gate_budget)
    )
    for number, gate in enumerate(gates, 1):
        six = simulate("six", gate, search_policy, search_timing, search_profile)
        gate_rows.append(
            {"gate": gate, "policy": search_policy, "timing": search_timing,
             "profile": search_profile, "six": six,
             "score6": _period_score(six, anchor_six)}
        )
        if number == 1 or number % 10 == 0 or number == len(gates):
            print(
                f"gate {number}/{len(gates)}: {six['net_profit']:+.0f}/"
                f"PF{six['profit_factor']:.2f}/win{six['win_rate']:.1%}/"
                f"DD{six['max_drawdown_pct']:.1%}", flush=True
            )
    gate_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in gate_rows[: args.gate_recent_finalists]:
        row["three"] = simulate(
            "three", row["gate"], row["policy"], row["timing"], row["profile"]
        )
        row["pair_score"] = _pair_score(row["six"], row["three"], anchor_six, anchor_three)
    gate_finalists = sorted(
        gate_rows[: args.gate_recent_finalists],
        key=lambda row: row["pair_score"], reverse=True
    )[: args.gate_finalists]

    policy_rows: list[dict[str, Any]] = []
    policies = (
        _local_policy_variants(search_policy, args.seed + 1, args.policy_budget)
        if args.refine_from_report
        else _policy_variants(args.seed + 1, args.policy_budget)
    )
    for gate_number, source in enumerate(gate_finalists, 1):
        for policy in policies:
            six = simulate(
                "six", source["gate"], policy, search_timing, search_profile
            )
            policy_rows.append(
                {"gate": source["gate"], "policy": policy, "timing": search_timing,
                 "profile": search_profile, "six": six,
                 "score6": _period_score(six, anchor_six)}
            )
        print(f"policy gate {gate_number}/{len(gate_finalists)} complete", flush=True)
    policy_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in policy_rows[: args.policy_recent_finalists]:
        row["three"] = simulate(
            "three", row["gate"], row["policy"], row["timing"], row["profile"]
        )
        row["pair_score"] = _pair_score(row["six"], row["three"], anchor_six, anchor_three)
        row["strict_improvement"] = _strict_improvement(
            row["six"], row["three"], anchor_six, anchor_three
        )
    policy_finalists = sorted(
        policy_rows[: args.policy_recent_finalists],
        key=lambda row: (row["strict_improvement"], row["pair_score"]), reverse=True
    )[: args.policy_finalists]

    profile_rows: list[dict[str, Any]] = []
    profiles = _profile_variants(search_profile, args.profile_budget)
    for source_number, source in enumerate(policy_finalists, 1):
        for profile in profiles:
            six = simulate(
                "six", source["gate"], source["policy"], anchor_timing, profile
            )
            profile_rows.append(
                {"gate": source["gate"], "policy": source["policy"],
                 "timing": anchor_timing, "profile": profile, "six": six,
                 "score6": _period_score(six, anchor_six)}
            )
        print(f"profile {source_number}/{len(policy_finalists)} complete", flush=True)
    profile_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in profile_rows[: args.profile_recent_finalists]:
        row["three"] = simulate(
            "three", row["gate"], row["policy"], row["timing"], row["profile"]
        )
        row["pair_score"] = _pair_score(row["six"], row["three"], anchor_six, anchor_three)
        row["strict_improvement"] = _strict_improvement(
            row["six"], row["three"], anchor_six, anchor_three
        )
    profile_finalists = sorted(
        profile_rows[: args.profile_recent_finalists],
        key=lambda row: (row["strict_improvement"], row["pair_score"]), reverse=True
    )[: args.profile_finalists]

    timing_rows: list[dict[str, Any]] = []
    for source_number, source in enumerate(profile_finalists, 1):
        for timing in _timing_variants():
            six = simulate(
                "six", source["gate"], source["policy"], timing, source["profile"]
            )
            three = simulate(
                "three", source["gate"], source["policy"], timing, source["profile"]
            )
            timing_rows.append(
                {"gate": source["gate"], "policy": source["policy"],
                 "timing": timing, "profile": source["profile"],
                 "six": six, "three": three,
                 "pair_score": _pair_score(six, three, anchor_six, anchor_three),
                 "strict_improvement": _strict_improvement(
                     six, three, anchor_six, anchor_three
                 )}
            )
        print(f"timing {source_number}/{len(profile_finalists)} complete", flush=True)
    recent = sorted(
        timing_rows,
        key=lambda row: (row["strict_improvement"], row["pair_score"]), reverse=True
    )

    robust_rows: list[dict[str, Any]] = []
    stressed = _scaled_execution(execution, 1.5)
    for number, row in enumerate(recent[: args.robust_finalists], 1):
        values = (row["gate"], row["policy"], row["timing"], row["profile"])
        early = simulate("early", *values)
        stress_six = simulate("six", *values, selected_execution=stressed)
        stress_three = simulate("three", *values, selected_execution=stressed)
        delay_six = simulate("six", *values, extra_delay_minutes=1)
        delay_three = simulate("three", *values, extra_delay_minutes=1)
        fixed_six = simulate("six", *values, force_compound=False)
        fixed_three = simulate("three", *values, force_compound=False)
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
            "six", *values,
            excluded_symbols=frozenset({top_six}) if top_six else frozenset()
        )
        no_top_three = simulate(
            "three", *values,
            excluded_symbols=frozenset({top_three}) if top_three else frozenset()
        )
        robust = (
            row["strict_improvement"]
            and early["net_profit"] > 0.0
            and _value(early["profit_factor"]) > 1.0
            and stress_six["net_profit"] > 0.0
            and stress_three["net_profit"] > 0.0
            and _value(stress_six["profit_factor"]) > 1.5
            and _value(stress_three["profit_factor"]) > 1.5
            and delay_six["net_profit"] > 0.0
            and delay_three["net_profit"] > 0.0
            and _value(delay_six["profit_factor"]) > 1.5
            and _value(delay_three["profit_factor"]) > 1.5
            and fixed_six["net_profit"] > 0.0
            and fixed_three["net_profit"] > 0.0
            and no_top_six["net_profit"] > 0.0
            and no_top_three["net_profit"] > 0.0
        )
        stress_better = (
            stress_six["net_profit"] > anchor_selected["stress_six_month"]["net_profit"]
            and stress_three["net_profit"] > anchor_selected["stress_three_month"]["net_profit"]
            and delay_six["net_profit"]
            > anchor_selected["entry_delay_one_minute_six_month"]["net_profit"]
            and delay_three["net_profit"]
            > anchor_selected["entry_delay_one_minute_three_month"]["net_profit"]
            and fixed_six["net_profit"] > anchor_selected["fixed_risk_six_month"]["net_profit"]
            and fixed_three["net_profit"] > anchor_selected["fixed_risk_three_month"]["net_profit"]
        )
        row.update(
            {"early": early, "stress_six": stress_six, "stress_three": stress_three,
             "delay_six": delay_six, "delay_three": delay_three,
             "fixed_six": fixed_six, "fixed_three": fixed_three,
             "no_top_six": no_top_six, "no_top_three": no_top_three,
             "top_six": top_six, "top_three": top_three,
             "robust": robust, "stress_better": stress_better,
             "robust_score": row["pair_score"] + (12.0 if robust else 0.0)
             + (4.0 if stress_better else 0.0)}
        )
        robust_rows.append(row)
        print(
            f"robust {number}/{args.robust_finalists}: strict={row['strict_improvement']} "
            f"early={early['net_profit']:+.0f} stress6={stress_six['net_profit']:+.0f} "
            f"delay6={delay_six['net_profit']:+.0f} fixed6={fixed_six['net_profit']:+.0f} "
            f"noTop6={no_top_six['net_profit']:+.0f}", flush=True
        )
    selected = max(
        [row for row in robust_rows if row["robust"]] or robust_rows or recent,
        key=lambda row: row.get("robust_score", row["pair_score"]),
    )

    def public(row: dict[str, Any], full: bool = False) -> dict[str, Any]:
        payload = {
            "pair_score": row["pair_score"],
            "strict_improvement": row.get("strict_improvement", False),
            "robust": row.get("robust", False),
            "stress_better": row.get("stress_better", False),
            "entry": row["gate"].as_dict(),
            "confidence_policy": row["policy"].as_dict(),
            "entry_timing": row["timing"].as_dict(),
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
                payload[target] = compact_v6_metrics(row[source])
        if "top_six" in row:
            payload["removed_top_symbol_six_month"] = row["top_six"]
            payload["removed_top_symbol_three_month"] = row["top_three"]
        if full:
            payload["six_month_full"] = row["six"]
            payload["three_month_full"] = row["three"]
        return payload

    selection_status = (
        "strict_robust_improvement"
        if selected.get("robust")
        else "best_available_did_not_clear_v6_anchor"
    )
    report = {
        "strategy_name": VOLATILITY_BREAKOUT_V7_NAME,
        "status": "independent_research_v6_and_gui_unchanged",
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
            "gate_evaluations": len(gate_rows),
            "policy_evaluations": len(policy_rows),
            "profile_evaluations": len(profile_rows),
            "timing_evaluations": len(timing_rows),
            "robust_evaluations": len(robust_rows),
        },
        "baseline": {
            "strategy": "Breakout v6 frozen live-balanced",
            "source_config": args.anchor_config,
            "source_report": args.anchor_report,
            "six_month": compact_v6_metrics(anchor_six),
            "three_month": compact_v6_metrics(anchor_three),
        },
        "selected": public(selected, True),
        "leaderboard": [public(row) for row in recent[:20]],
        "preserved": {
            "breakout_v6": "unchanged",
            "grid_v4": "unchanged",
            "active_gui": "unchanged",
            "apt_grid": "unchanged",
        },
    }
    _write_json(args.output, report)
    config = {
        "strategy_name": VOLATILITY_BREAKOUT_V7_NAME,
        "status": "independent_research_candidate_not_live",
        "selection_status": selection_status,
        "entry": selected["gate"].as_dict(),
        "confidence_policy": selected["policy"].as_dict(),
        "entry_timing": selected["timing"].as_dict(),
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
        "# Breakout v7 confidence-allocation optimization",
        "",
        "- Frozen Breakout v6 is the strict anchor and remains unchanged",
        "- Gap-free 1m execution, fees, slippage, funding and point-in-time context",
        f"- Selection status: `{selection_status}`",
        "",
        "| Period | Version | Trades | Net | PF | Win | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for period, anchor_result, selected_result in (
        ("3 months", anchor_three, selected["three"]),
        ("6 months", anchor_six, selected["six"]),
    ):
        for name, result in (("Breakout v6 anchor", anchor_result), ("Breakout v7", selected_result)):
            lines.append(
                f"| {period} | {name} | {result['trade_count']} | "
                f"{result['net_profit']:+.2f}U | {result['profit_factor']:.3f} | "
                f"{result['win_rate']:.2%} | {result['max_drawdown_pct']:.2%} |"
            )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "strategy_name": VOLATILITY_BREAKOUT_V7_NAME,
        "status": "independent_research_artifacts",
        "config": args.config_output,
        "report": args.output,
        "summary": args.summary,
        "hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                "crypto_scalper/volatility_breakout_v7.py",
                "crypto_scalper/volatility_breakout_v7_optimize.py",
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
        f"robust={selected.get('robust', False)} stress_better={selected.get('stress_better', False)} "
        f"6m={selected['six']['net_profit']:+.2f}/PF{selected['six']['profit_factor']:.3f}/"
        f"win{selected['six']['win_rate']:.1%}/DD{selected['six']['max_drawdown_pct']:.1%}; "
        f"3m={selected['three']['net_profit']:+.2f}/PF{selected['three']['profit_factor']:.3f}/"
        f"win{selected['three']['win_rate']:.1%}/DD{selected['three']['max_drawdown_pct']:.1%}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Breakout v7 confidence-allocation live-realistic optimization"
    )
    parser.add_argument("--start-6m", default="2026-01-19T00:00:00")
    parser.add_argument("--start-3m", default="2026-04-19T00:00:00")
    parser.add_argument("--end", default="2026-07-19T00:00:00")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument(
        "--anchor-config",
        default="config.volatility-breakout.v6-live-balanced-final-50.json",
    )
    parser.add_argument(
        "--anchor-report",
        default="reports/volatility_breakout_v6_live_balanced_final_3m_6m.json",
    )
    parser.add_argument(
        "--source-breakout-config",
        default="config.volatility-breakout.hybrid-v5-balanced-expansion-runner-50.json",
    )
    parser.add_argument(
        "--source-grid-config", default="config.trend-grid.v3-optimized-50.json"
    )
    parser.add_argument(
        "--cost-config", default="config.volatility-breakout.v2-balanced-50-shadow.json"
    )
    parser.add_argument(
        "--one-minute-roots", nargs="+",
        default=("data/binance_1m_365d_top100", "data/binance_1m_v3_exit_holdout_20260522_20260719"),
    )
    parser.add_argument(
        "--funding-roots", nargs="+",
        default=("data/binance_funding_365d_top100", "data/binance_funding_v3_exit_holdout_20260612_20260719"),
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--refine-from-report",
        default="",
        help="Optional prior v7 report used as the center of a local search",
    )
    parser.add_argument("--gate-budget", type=int, default=56)
    parser.add_argument("--gate-recent-finalists", type=int, default=28)
    parser.add_argument("--gate-finalists", type=int, default=5)
    parser.add_argument("--policy-budget", type=int, default=120)
    parser.add_argument("--policy-recent-finalists", type=int, default=48)
    parser.add_argument("--policy-finalists", type=int, default=8)
    parser.add_argument("--profile-budget", type=int, default=18)
    parser.add_argument("--profile-recent-finalists", type=int, default=48)
    parser.add_argument("--profile-finalists", type=int, default=8)
    parser.add_argument("--robust-finalists", type=int, default=12)
    parser.add_argument(
        "--output", default="reports/volatility_breakout_v7_confidence_3m_6m.json"
    )
    parser.add_argument(
        "--summary", default="reports/volatility_breakout_v7_confidence_3m_6m.md"
    )
    parser.add_argument(
        "--config-output", default="config.volatility-breakout.v7-confidence-50.json"
    )
    parser.add_argument(
        "--manifest", default="config.volatility-breakout.v7-confidence-50-manifest.json"
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

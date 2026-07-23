from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .combined_hybrid_v5_grid_v3_backtest import (
    _daily_cap_candidates,
    _slice_signal_data,
    _write_json,
    build_frozen_configs,
)
from .volatility_breakout_exit_protection import (
    ExitProtectionConfig,
    simulate_exit_protected_portfolio,
)
from .volatility_breakout_optimize import (
    Candidate,
    UNIVERSE_50,
    build_candidates,
    minute_token,
    sha256_file,
)
from .volatility_breakout_v4_research import (
    build_v4_market_context,
    enrich_candidates_v4,
    filter_candidates_v4,
    load_v4_runtime_inputs,
)
from .volatility_breakout_v6 import (
    VOLATILITY_BREAKOUT_V6_NAME,
    BreakoutV6EntryConfig,
    BreakoutV6SideGate,
    filter_breakout_v6_candidates,
)


@dataclass(frozen=True)
class BreakoutV6ExitProfile:
    risk_per_trade_pct: float = 0.025
    long_risk_multiplier: float = 1.0
    short_risk_multiplier: float = 1.0
    stop_atr_multiple: float = 1.0
    max_holding_minutes: int = 960
    fail_fast_minutes: int = 120
    fail_fast_min_mfe_r: float = 0.10
    fail_fast_max_current_r: float = -0.50
    breakeven_trigger_r: float = 0.0
    profit_giveback_activation_r: float = 0.0
    profit_giveback_r: float = 0.0
    partial_take_profit_r: float = 8.0
    partial_take_profit_fraction: float = 0.05
    move_stop_to_breakeven_after_partial: bool = False
    ranking_mode: str = "quality_desc"


def _candidate_count(candidates: dict[int, list[Candidate]]) -> int:
    return sum(len(rows) for rows in candidates.values())


def _slice_candidates(
    candidates: dict[int, list[Candidate]],
    start: datetime,
    end: datetime,
) -> dict[int, list[Candidate]]:
    start_minute = minute_token(start)
    end_minute = minute_token(end)
    return {
        minute: rows
        for minute, rows in candidates.items()
        if start_minute <= minute < end_minute
    }


def _metric_value(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def compact_v6_metrics(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_count",
        "trade_count",
        "initial_equity",
        "final_equity",
        "net_profit",
        "return_pct",
        "win_rate",
        "profit_factor",
        "expectancy_usdt",
        "expectancy_r",
        "average_win_loss_ratio",
        "max_drawdown_pct",
        "max_drawdown_duration_minutes",
        "fee",
        "slippage",
        "funding",
        "full_cost",
        "top5_profit_contribution",
        "hard_drawdown_stopped",
        "positive_months",
        "negative_months",
        "by_month",
        "by_side",
        "by_exit_reason",
    )
    output = {key: result.get(key) for key in keys}
    trades = result.get("trades", ())
    output.update(
        {
            "largest_trade_net_profit": max(
                (_metric_value(row.get("net_pnl")) for row in trades),
                default=0.0,
            ),
            "largest_trade_r": max(
                (_metric_value(row.get("pnl_r")) for row in trades),
                default=0.0,
            ),
            "five_r_winner_count": sum(
                _metric_value(row.get("pnl_r")) >= 5.0 for row in trades
            ),
            "full_stop_count": sum(
                str(row.get("exit_reason", "")).startswith("stop_loss")
                for row in trades
            ),
            "protected_exit_count": sum(
                str(row.get("exit_reason", ""))
                in {
                    "breakeven_stop",
                    "profit_giveback_stop",
                    "partial_breakeven_stop",
                }
                for row in trades
            ),
        }
    )
    return output


def _single_period_score(result: dict[str, Any]) -> float:
    if result["hard_drawdown_stopped"] or result["final_equity"] <= 0.0:
        return -1e9
    if result["trade_count"] < 20:
        return -1e8
    growth = math.log(max(result["final_equity"] / result["initial_equity"], 0.01))
    profit_factor = min(_metric_value(result["profit_factor"], 8.0), 8.0)
    return (
        3.2 * growth
        + 1.25 * min(profit_factor, 4.0)
        + 2.2 * result["win_rate"]
        - 7.0 * result["max_drawdown_pct"]
        - 0.8 * result["top5_profit_contribution"]
        - 0.35 * result["negative_months"]
    )


def _strict_improvement(
    six_month: dict[str, Any],
    three_month: dict[str, Any],
    baseline_six: dict[str, Any],
    baseline_three: dict[str, Any],
) -> bool:
    return (
        six_month["trade_count"] >= 50
        and three_month["trade_count"] >= 25
        and six_month["net_profit"] > baseline_six["net_profit"] * 1.005
        and three_month["net_profit"] > baseline_three["net_profit"] * 1.005
        and _metric_value(six_month["profit_factor"])
        > _metric_value(baseline_six["profit_factor"])
        and _metric_value(three_month["profit_factor"])
        > _metric_value(baseline_three["profit_factor"])
        and six_month["max_drawdown_pct"]
        <= baseline_six["max_drawdown_pct"]
        and three_month["max_drawdown_pct"]
        <= baseline_three["max_drawdown_pct"]
        and six_month["win_rate"] >= baseline_six["win_rate"] + 0.02
        and three_month["win_rate"] >= baseline_three["win_rate"] + 0.02
    )


def _pair_score(
    six_month: dict[str, Any],
    three_month: dict[str, Any],
    baseline_six: dict[str, Any],
    baseline_three: dict[str, Any],
) -> float:
    bonus = 12.0 if _strict_improvement(
        six_month,
        three_month,
        baseline_six,
        baseline_three,
    ) else 0.0
    return (
        _single_period_score(six_month)
        + 1.20 * _single_period_score(three_month)
        + bonus
    )


def _baseline_side_gate() -> BreakoutV6SideGate:
    return BreakoutV6SideGate()


def _unique_side_gates(rows: list[BreakoutV6SideGate]) -> list[BreakoutV6SideGate]:
    return list(dict.fromkeys(rows))


def _long_gate_variants(seed: int, budget: int) -> list[BreakoutV6SideGate]:
    rows = [
        _baseline_side_gate(),
        BreakoutV6SideGate(
            min_quality_score=1.20,
            min_body_atr=0.70,
            min_breakout_extension_atr=0.15,
            max_trend_alignment_atr=6.0,
            max_range_atr=7.5,
        ),
        BreakoutV6SideGate(
            min_quality_score=1.50,
            min_body_atr=1.00,
            min_breakout_extension_atr=0.30,
            max_trend_alignment_atr=6.0,
            max_range_atr=6.5,
        ),
        BreakoutV6SideGate(
            min_quality_score=1.70,
            min_body_atr=1.30,
            min_breakout_extension_atr=0.50,
            max_trend_alignment_atr=8.0,
            max_range_atr=6.0,
        ),
        BreakoutV6SideGate(
            min_quality_score=1.80,
            min_body_atr=1.70,
            min_breakout_extension_atr=0.70,
            max_range_atr=5.5,
            max_directional_eth_return_4h=0.015,
        ),
        BreakoutV6SideGate(
            min_quality_score=1.75,
            min_breakout_extension_atr=0.75,
            max_range_atr=6.0,
            max_directional_eth_return_4h=0.012,
        ),
        BreakoutV6SideGate(max_range_atr=5.5),
        BreakoutV6SideGate(
            min_quality_score=1.50,
            min_breakout_extension_atr=0.50,
            max_range_atr=5.0,
            max_directional_eth_return_4h=0.012,
        ),
    ]
    rng = random.Random(seed)
    while len(_unique_side_gates(rows)) < budget:
        rows.append(
            BreakoutV6SideGate(
                min_quality_score=rng.choice(
                    (-999.0, 1.0, 1.25, 1.45, 1.65, 1.80)
                ),
                min_body_atr=rng.choice((0.0, 0.50, 0.85, 1.10, 1.40, 1.75)),
                min_breakout_extension_atr=rng.choice(
                    (-999.0, 0.05, 0.20, 0.35, 0.55, 0.75)
                ),
                max_trend_alignment_atr=rng.choice((4.0, 5.0, 6.0, 8.0, 999.0)),
                max_range_atr=rng.choice((5.0, 5.5, 6.0, 7.0, 8.0, 999.0)),
                min_volume_ratio=rng.choice((0.0, 0.0, 0.50, 0.80)),
                max_volume_ratio=rng.choice((5.0, 10.0, 25.0, 999.0)),
                min_directional_breadth=rng.choice((0.0, 0.25, 0.40, 0.55)),
                max_directional_btc_return_4h=rng.choice((0.008, 0.012, 0.02)),
                max_directional_eth_return_4h=rng.choice(
                    (0.008, 0.015, 0.025, 999.0)
                ),
                min_market_efficiency_12h=rng.choice((0.0, 0.0, 0.08, 0.12)),
                min_directional_symbol_efficiency_12h=rng.choice(
                    (-999.0, -999.0, -0.10, 0.0)
                ),
                min_regime_score=rng.choice((-999.0, -999.0, -0.20, 0.0)),
            )
        )
    return _unique_side_gates(rows)[:budget]


def _short_gate_variants(seed: int, budget: int) -> list[BreakoutV6SideGate]:
    rows = [
        _baseline_side_gate(),
        BreakoutV6SideGate(
            max_body_atr=2.0,
            min_breakout_extension_atr=0.10,
            max_trend_alignment_atr=5.0,
            max_range_atr=7.5,
        ),
        BreakoutV6SideGate(
            max_body_atr=1.5,
            min_breakout_extension_atr=0.10,
            max_trend_alignment_atr=5.0,
            max_range_atr=6.5,
            min_directional_breadth=0.70,
        ),
        BreakoutV6SideGate(
            max_body_atr=1.4,
            min_breakout_extension_atr=0.20,
            max_trend_alignment_atr=4.5,
            max_range_atr=6.0,
            max_volume_ratio=4.0,
            min_directional_breadth=0.85,
        ),
        BreakoutV6SideGate(
            min_breakout_extension_atr=0.25,
            max_range_atr=7.5,
            min_directional_breadth=0.70,
        ),
        BreakoutV6SideGate(
            max_body_atr=2.0,
            max_range_atr=8.0,
            min_directional_breadth=0.85,
        ),
        BreakoutV6SideGate(
            max_body_atr=1.8,
            max_trend_alignment_atr=4.5,
            max_range_atr=6.5,
        ),
    ]
    rng = random.Random(seed)
    while len(_unique_side_gates(rows)) < budget:
        rows.append(
            BreakoutV6SideGate(
                min_quality_score=rng.choice((-999.0, -999.0, 0.60, 0.90)),
                max_quality_score=rng.choice((1.8, 2.1, 2.5, 999.0)),
                max_body_atr=rng.choice((1.3, 1.5, 1.8, 2.2, 3.0, 999.0)),
                min_breakout_extension_atr=rng.choice(
                    (-999.0, 0.05, 0.10, 0.20, 0.30)
                ),
                max_trend_alignment_atr=rng.choice((4.0, 4.5, 5.0, 6.0, 999.0)),
                max_range_atr=rng.choice((6.0, 6.5, 7.0, 7.5, 9.0, 999.0)),
                max_volume_ratio=rng.choice((3.5, 5.0, 8.0, 999.0)),
                min_directional_breadth=rng.choice((0.0, 0.60, 0.70, 0.80, 0.90)),
                max_directional_btc_return_4h=rng.choice((0.010, 0.015, 0.02)),
                max_directional_eth_return_4h=rng.choice(
                    (0.012, 0.020, 0.030, 999.0)
                ),
                min_market_efficiency_12h=rng.choice((0.0, 0.0, 0.08, 0.12)),
                min_directional_symbol_efficiency_12h=rng.choice(
                    (-999.0, -999.0, -0.10, 0.0)
                ),
                min_regime_score=rng.choice((-999.0, -999.0, -0.20, 0.0)),
            )
        )
    return _unique_side_gates(rows)[:budget]


def _unique_exit_profiles(
    rows: list[BreakoutV6ExitProfile],
) -> list[BreakoutV6ExitProfile]:
    return list(dict.fromkeys(rows))


def _exit_variants(seed: int, budget: int) -> list[BreakoutV6ExitProfile]:
    rows = [
        BreakoutV6ExitProfile(),
        BreakoutV6ExitProfile(
            breakeven_trigger_r=2.5,
            profit_giveback_activation_r=4.0,
            profit_giveback_r=2.5,
        ),
        BreakoutV6ExitProfile(
            breakeven_trigger_r=3.0,
            profit_giveback_activation_r=5.0,
            profit_giveback_r=3.0,
            partial_take_profit_r=0.0,
            partial_take_profit_fraction=0.0,
        ),
        BreakoutV6ExitProfile(
            profit_giveback_activation_r=4.0,
            profit_giveback_r=3.0,
            partial_take_profit_r=6.0,
            partial_take_profit_fraction=0.10,
        ),
        BreakoutV6ExitProfile(
            risk_per_trade_pct=0.030,
            fail_fast_minutes=180,
            fail_fast_min_mfe_r=0.20,
            fail_fast_max_current_r=-0.35,
            breakeven_trigger_r=3.0,
            profit_giveback_activation_r=5.0,
            profit_giveback_r=3.0,
        ),
        BreakoutV6ExitProfile(
            risk_per_trade_pct=0.0225,
            short_risk_multiplier=0.90,
            stop_atr_multiple=1.10,
            max_holding_minutes=1200,
            fail_fast_minutes=180,
            fail_fast_min_mfe_r=0.20,
            fail_fast_max_current_r=-0.35,
            breakeven_trigger_r=2.5,
            profit_giveback_activation_r=5.0,
            profit_giveback_r=3.0,
        ),
    ]
    rng = random.Random(seed)
    while len(_unique_exit_profiles(rows)) < budget:
        giveback_activation, giveback = rng.choice(
            (
                (0.0, 0.0),
                (3.0, 2.0),
                (4.0, 2.5),
                (5.0, 3.0),
                (8.0, 5.0),
            )
        )
        partial_r, partial_fraction = rng.choice(
            ((0.0, 0.0), (4.0, 0.10), (6.0, 0.10), (8.0, 0.05))
        )
        fail_minutes, fail_mfe, fail_current = rng.choice(
            (
                (0, 0.0, 0.0),
                (90, 0.10, -0.35),
                (120, 0.10, -0.50),
                (180, 0.20, -0.35),
                (240, 0.30, -0.25),
            )
        )
        rows.append(
            BreakoutV6ExitProfile(
                risk_per_trade_pct=rng.choice(
                    (0.020, 0.0225, 0.025, 0.0275, 0.030)
                ),
                long_risk_multiplier=rng.choice((0.90, 1.0, 1.10)),
                short_risk_multiplier=rng.choice((0.80, 0.90, 1.0)),
                stop_atr_multiple=rng.choice((0.90, 1.0, 1.10, 1.25)),
                max_holding_minutes=rng.choice((720, 960, 1200, 1440)),
                fail_fast_minutes=fail_minutes,
                fail_fast_min_mfe_r=fail_mfe,
                fail_fast_max_current_r=fail_current,
                breakeven_trigger_r=rng.choice((0.0, 0.0, 2.0, 2.5, 3.5)),
                profit_giveback_activation_r=giveback_activation,
                profit_giveback_r=giveback,
                partial_take_profit_r=partial_r,
                partial_take_profit_fraction=partial_fraction,
                move_stop_to_breakeven_after_partial=rng.choice((False, False, True)),
                ranking_mode=rng.choice(
                    (
                        "quality_desc",
                        "quality_desc",
                        "range_asc",
                        "breakout_extension_desc",
                        "directional_breadth_desc",
                    )
                ),
            )
        )
    return _unique_exit_profiles(rows)[:budget]


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
        raise RuntimeError("Breakout v6 requires gap-free stitched 1m data")
    if metadata["maximum_missing_minutes"]:
        raise RuntimeError("Breakout v6 requires gap-free stitched 1m data")
    if metadata["funding_missing_symbols"]:
        raise RuntimeError("Breakout v6 requires complete funding data")

    frozen = build_frozen_configs(args.breakout_config, args.grid_config)
    default_entry = BreakoutV6EntryConfig()

    def build_period_inputs(
        name: str,
        start: datetime,
        finish: datetime,
    ) -> dict[str, Any]:
        period_signal_data = _slice_signal_data(
            signal_data,
            start - timedelta(days=args.warmup_days),
            finish,
        )
        period_context = build_v4_market_context(symbols, period_signal_data)
        period_raw = enrich_candidates_v4(
            build_candidates(
                symbols,
                period_signal_data,
                execution_data,
                frozen["breakout_build_signal"],
                start,
                finish,
            ),
            period_context,
        )
        baseline_filtered = filter_candidates_v4(
            period_raw,
            frozen["breakout_build_signal"],
            frozen["breakout_regime"],
            period_context,
        )
        baseline_candidates = _daily_cap_candidates(baseline_filtered, 2)
        default_v6_candidates = filter_breakout_v6_candidates(
            period_raw,
            period_context,
            default_entry,
        )
        baseline_ids = {
            row.signal.event_id
            for rows in baseline_candidates.values()
            for row in rows
        }
        default_ids = {
            row.signal.event_id
            for rows in default_v6_candidates.values()
            for row in rows
        }
        if baseline_ids != default_ids:
            raise RuntimeError(
                f"permissive v6 entry gate does not reproduce frozen v5 for {name}"
            )
        print(
            f"{name}: raw candidates={_candidate_count(period_raw)} "
            f"baseline={len(baseline_ids)}",
            flush=True,
        )
        return {
            "name": name,
            "start": start,
            "end": finish,
            "context": period_context,
            "raw": period_raw,
            "baseline": baseline_candidates,
        }

    period_inputs = {
        (start_six, end): build_period_inputs("6m", start_six, end),
        (start_three, end): build_period_inputs("3m", start_three, end),
        (start_six, start_three): build_period_inputs(
            "early3m", start_six, start_three
        ),
    }

    baseline_exit = BreakoutV6ExitProfile()
    candidate_cache: dict[
        tuple[datetime, datetime, BreakoutV6EntryConfig],
        dict[int, list[Candidate]],
    ] = {
        (start, finish, default_entry): values["baseline"]
        for (start, finish), values in period_inputs.items()
    }

    def candidates_for(
        entry: BreakoutV6EntryConfig,
        start: datetime,
        finish: datetime,
    ) -> dict[int, list[Candidate]]:
        cache_key = (start, finish, entry)
        cached = candidate_cache.get(cache_key)
        if cached is None:
            inputs = period_inputs[(start, finish)]
            cached = filter_breakout_v6_candidates(
                inputs["raw"],
                inputs["context"],
                entry,
            )
            candidate_cache[cache_key] = cached
        return cached

    def simulate(
        entry: BreakoutV6EntryConfig,
        exit_profile: BreakoutV6ExitProfile,
        start: datetime,
        finish: datetime,
        selected_execution: Any = execution,
    ) -> dict[str, Any]:
        candidates = candidates_for(entry, start, finish)
        signal = replace(
            frozen["breakout_signal"],
            stop_atr_multiple=exit_profile.stop_atr_multiple,
            take_profit_r=60.0,
            max_holding_minutes=exit_profile.max_holding_minutes,
            fail_fast_minutes=exit_profile.fail_fast_minutes,
            fail_fast_min_mfe_r=exit_profile.fail_fast_min_mfe_r,
            fail_fast_max_current_r=exit_profile.fail_fast_max_current_r,
        )
        portfolio = replace(
            frozen["breakout_portfolio"],
            risk_per_trade_pct=exit_profile.risk_per_trade_pct,
            long_risk_multiplier=exit_profile.long_risk_multiplier,
            short_risk_multiplier=exit_profile.short_risk_multiplier,
            ranking_mode=exit_profile.ranking_mode,
        )
        exit_config = ExitProtectionConfig(
            breakeven_trigger_r=exit_profile.breakeven_trigger_r,
            profit_giveback_activation_r=(
                exit_profile.profit_giveback_activation_r
            ),
            profit_giveback_r=exit_profile.profit_giveback_r,
            partial_take_profit_r=exit_profile.partial_take_profit_r,
            partial_take_profit_fraction=(
                exit_profile.partial_take_profit_fraction
            ),
            move_stop_to_breakeven_after_partial=(
                exit_profile.move_stop_to_breakeven_after_partial
            ),
        )
        return simulate_exit_protected_portfolio(
            candidates,
            symbols,
            execution_data,
            rules,
            signal,
            portfolio,
            exit_config,
            selected_execution,
            start,
            finish,
            args.initial_equity,
        )

    baseline_six = simulate(default_entry, baseline_exit, start_six, end)
    baseline_three = simulate(default_entry, baseline_exit, start_three, end)
    baseline_report = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    expected_six = baseline_report["periods"]["six_month"]["standalone"]
    expected_three = baseline_report["periods"]["three_month"]["standalone"]
    for label, actual, expected in (
        ("six_month", baseline_six, expected_six["volatility_breakout"]),
        ("three_month", baseline_three, expected_three["volatility_breakout"]),
    ):
        if abs(actual["net_profit"] - expected["net_profit"]) > 1e-6:
            raise RuntimeError(
                f"Breakout v6 {label} baseline mismatch: "
                f"actual={actual['net_profit']:+.8f} "
                f"expected={expected['net_profit']:+.8f}"
            )
    print(
        f"baseline 6m={baseline_six['net_profit']:+.2f}/PF{baseline_six['profit_factor']:.3f}/"
        f"DD{baseline_six['max_drawdown_pct']:.2%}; "
        f"3m={baseline_three['net_profit']:+.2f}/PF{baseline_three['profit_factor']:.3f}/"
        f"DD{baseline_three['max_drawdown_pct']:.2%}",
        flush=True,
    )

    long_rows: list[dict[str, Any]] = []
    for number, long_gate in enumerate(
        _long_gate_variants(args.seed, args.long_budget),
        1,
    ):
        entry = BreakoutV6EntryConfig(long=long_gate, short=_baseline_side_gate())
        result = simulate(entry, baseline_exit, start_six, end)
        long_rows.append(
            {
                "entry": entry,
                "exit": baseline_exit,
                "six": result,
                "six_score": _single_period_score(result),
            }
        )
        if number == 1 or number % 10 == 0 or number == args.long_budget:
            print(
                f"long {number}/{args.long_budget}: net={result['net_profit']:+.0f} "
                f"PF={result['profit_factor']:.2f} win={result['win_rate']:.1%} "
                f"DD={result['max_drawdown_pct']:.1%}",
                flush=True,
            )
    long_rows.sort(key=lambda row: row["six_score"], reverse=True)
    for row in long_rows[: args.long_recent_finalists]:
        row["three"] = simulate(row["entry"], baseline_exit, start_three, end)
        row["pair_score"] = _pair_score(
            row["six"], row["three"], baseline_six, baseline_three
        )
    long_finalists = sorted(
        long_rows[: args.long_recent_finalists],
        key=lambda row: row["pair_score"],
        reverse=True,
    )[: args.long_finalists]

    entry_six_rows: list[dict[str, Any]] = []
    short_variants = _short_gate_variants(args.seed + 1, args.short_budget)
    for long_number, source in enumerate(long_finalists, 1):
        for short_gate in short_variants:
            entry = BreakoutV6EntryConfig(
                long=source["entry"].long,
                short=short_gate,
            )
            result = simulate(entry, baseline_exit, start_six, end)
            entry_six_rows.append(
                {
                    "entry": entry,
                    "exit": baseline_exit,
                    "six": result,
                    "six_score": _single_period_score(result),
                }
            )
        print(
            f"short gates for long finalist {long_number}/{len(long_finalists)} complete",
            flush=True,
        )
    entry_six_rows.sort(key=lambda row: row["six_score"], reverse=True)
    for row in entry_six_rows[: args.entry_recent_finalists]:
        row["three"] = simulate(row["entry"], baseline_exit, start_three, end)
        row["pair_score"] = _pair_score(
            row["six"], row["three"], baseline_six, baseline_three
        )
    entry_finalists = sorted(
        entry_six_rows[: args.entry_recent_finalists],
        key=lambda row: row["pair_score"],
        reverse=True,
    )[: args.entry_finalists]

    exit_six_rows: list[dict[str, Any]] = []
    exit_variants = _exit_variants(args.seed + 2, args.exit_budget)
    for entry_number, source in enumerate(entry_finalists, 1):
        for exit_profile in exit_variants:
            result = simulate(source["entry"], exit_profile, start_six, end)
            exit_six_rows.append(
                {
                    "entry": source["entry"],
                    "exit": exit_profile,
                    "six": result,
                    "six_score": _single_period_score(result),
                }
            )
        print(
            f"exit search for entry finalist {entry_number}/{len(entry_finalists)} complete",
            flush=True,
        )
    exit_six_rows.sort(key=lambda row: row["six_score"], reverse=True)
    for row in exit_six_rows[: args.exit_recent_finalists]:
        row["three"] = simulate(row["entry"], row["exit"], start_three, end)
        row["pair_score"] = _pair_score(
            row["six"], row["three"], baseline_six, baseline_three
        )
        row["strict_improvement"] = _strict_improvement(
            row["six"], row["three"], baseline_six, baseline_three
        )
    recent_rows = sorted(
        exit_six_rows[: args.exit_recent_finalists],
        key=lambda row: row["pair_score"],
        reverse=True,
    )

    robust_rows: list[dict[str, Any]] = []
    stress_execution = _stress_execution(execution)
    for number, row in enumerate(recent_rows[: args.robust_finalists], 1):
        early = simulate(row["entry"], row["exit"], start_six, start_three)
        stress_six = simulate(
            row["entry"], row["exit"], start_six, end, stress_execution
        )
        stress_three = simulate(
            row["entry"], row["exit"], start_three, end, stress_execution
        )
        row.update(
            {
                "early": early,
                "stress_six": stress_six,
                "stress_three": stress_three,
                "robust": (
                    row["strict_improvement"]
                    and early["net_profit"] > 0.0
                    and _metric_value(early["profit_factor"]) > 1.0
                    and stress_six["net_profit"] > 0.0
                    and stress_three["net_profit"] > 0.0
                    and _metric_value(stress_six["profit_factor"]) > 1.20
                    and _metric_value(stress_three["profit_factor"]) > 1.20
                    and stress_six["max_drawdown_pct"] <= 0.50
                ),
            }
        )
        row["robust_score"] = (
            row["pair_score"]
            + 0.45 * _single_period_score(early)
            + 0.25 * _single_period_score(stress_six)
            + 0.30 * _single_period_score(stress_three)
            + (10.0 if row["robust"] else 0.0)
        )
        robust_rows.append(row)
        print(
            f"robust {number}/{args.robust_finalists}: strict={row['strict_improvement']} "
            f"early={early['net_profit']:+.0f} stress6={stress_six['net_profit']:+.0f}",
            flush=True,
        )
    robust_pool = [row for row in robust_rows if row["robust"]]
    selected = max(
        robust_pool or robust_rows or recent_rows,
        key=lambda row: row.get("robust_score", row["pair_score"]),
    )

    def public_row(row: dict[str, Any], include_full: bool = False) -> dict[str, Any]:
        payload = {
            "pair_score": row["pair_score"],
            "strict_improvement": row.get("strict_improvement", False),
            "robust": row.get("robust", False),
            "entry": row["entry"].as_dict(),
            "exit": asdict(row["exit"]),
            "six_month": compact_v6_metrics(row["six"]),
            "three_month": compact_v6_metrics(row["three"]),
        }
        for source, target in (
            ("early", "early_three_month"),
            ("stress_six", "stress_six_month"),
            ("stress_three", "stress_three_month"),
        ):
            if source in row:
                payload[target] = compact_v6_metrics(row[source])
        if include_full:
            payload["six_month_full"] = row["six"]
            payload["three_month_full"] = row["three"]
        return payload

    report = {
        "strategy_name": VOLATILITY_BREAKOUT_V6_NAME,
        "status": "independent_v6_research_gui_and_v5_unchanged",
        "selection_status": (
            "strict_robust_improvement"
            if selected.get("robust")
            else "best_available_requires_further_optimization"
        ),
        "periods": {
            "six_month": [start_six.isoformat(), end.isoformat()],
            "three_month": [start_three.isoformat(), end.isoformat()],
            "early_three_month": [start_six.isoformat(), start_three.isoformat()],
        },
        "initial_equity": args.initial_equity,
        "universe_size": len(symbols),
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
            "long_budget": args.long_budget,
            "short_budget": args.short_budget,
            "exit_budget": args.exit_budget,
            "long_evaluations": len(long_rows),
            "entry_evaluations": len(entry_six_rows),
            "exit_evaluations": len(exit_six_rows),
            "robust_evaluations": len(robust_rows),
        },
        "baseline": {
            "strategy": "Hybrid v5 Breakout sleeve",
            "source_report": args.baseline_report,
            "six_month": compact_v6_metrics(baseline_six),
            "three_month": compact_v6_metrics(baseline_three),
        },
        "selected": public_row(selected, True),
        "leaderboard": [public_row(row) for row in recent_rows[:20]],
        "preserved": {
            "active_gui": "unchanged",
            "hybrid_v5": "unchanged",
            "grid_v3": "unchanged",
            "apt_grid": "unchanged",
        },
    }
    _write_json(args.output, report)

    selected_signal = replace(
        frozen["breakout_signal"],
        stop_atr_multiple=selected["exit"].stop_atr_multiple,
        take_profit_r=60.0,
        max_holding_minutes=selected["exit"].max_holding_minutes,
        fail_fast_minutes=selected["exit"].fail_fast_minutes,
        fail_fast_min_mfe_r=selected["exit"].fail_fast_min_mfe_r,
        fail_fast_max_current_r=selected["exit"].fail_fast_max_current_r,
    )
    selected_portfolio = replace(
        frozen["breakout_portfolio"],
        risk_per_trade_pct=selected["exit"].risk_per_trade_pct,
        long_risk_multiplier=selected["exit"].long_risk_multiplier,
        short_risk_multiplier=selected["exit"].short_risk_multiplier,
        ranking_mode=selected["exit"].ranking_mode,
    )
    selected_exit = ExitProtectionConfig(
        breakeven_trigger_r=selected["exit"].breakeven_trigger_r,
        profit_giveback_activation_r=(
            selected["exit"].profit_giveback_activation_r
        ),
        profit_giveback_r=selected["exit"].profit_giveback_r,
        partial_take_profit_r=selected["exit"].partial_take_profit_r,
        partial_take_profit_fraction=(
            selected["exit"].partial_take_profit_fraction
        ),
        move_stop_to_breakeven_after_partial=(
            selected["exit"].move_stop_to_breakeven_after_partial
        ),
    )
    config_payload = {
        "strategy_name": VOLATILITY_BREAKOUT_V6_NAME,
        "status": "frozen_research_candidate_not_live_gui_unchanged",
        "selection_status": report["selection_status"],
        "entry": selected["entry"].as_dict(),
        "signal": selected_signal.as_dict(),
        "portfolio": asdict(selected_portfolio),
        "exit_protection": asdict(selected_exit),
        "results": {
            "six_month": compact_v6_metrics(selected["six"]),
            "three_month": compact_v6_metrics(selected["three"]),
        },
    }
    _write_json(args.config_output, config_payload)

    baseline_six_metrics = compact_v6_metrics(baseline_six)
    baseline_three_metrics = compact_v6_metrics(baseline_three)
    selected_six_metrics = compact_v6_metrics(selected["six"])
    selected_three_metrics = compact_v6_metrics(selected["three"])
    lines = [
        "# Breakout v6 live-robust optimization",
        "",
        "- Gap-free 1m execution; full fees, slippage and funding; point-in-time context only",
        "- Current GUI and Hybrid v5 remain unchanged",
        f"- Selection status: `{report['selection_status']}`",
        "",
        "| Period | Version | Trades | Net | PF | Win rate | Max DD | Top-5 concentration |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for period, baseline_metrics, selected_metrics in (
        ("3 months", baseline_three_metrics, selected_three_metrics),
        ("6 months", baseline_six_metrics, selected_six_metrics),
    ):
        for version, metrics in (
            ("Hybrid v5 baseline", baseline_metrics),
            ("Breakout v6 candidate", selected_metrics),
        ):
            lines.append(
                f"| {period} | {version} | {metrics['trade_count']} | "
                f"{metrics['net_profit']:+.2f}U | {metrics['profit_factor']:.3f} | "
                f"{metrics['win_rate']:.2%} | {metrics['max_drawdown_pct']:.2%} | "
                f"{metrics['top5_profit_contribution']:.2%} |"
            )
    if "early" in selected:
        early_metrics = compact_v6_metrics(selected["early"])
        stress_six_metrics = compact_v6_metrics(selected["stress_six"])
        stress_three_metrics = compact_v6_metrics(selected["stress_three"])
        lines.extend(
            [
                "",
                "## Robustness",
                "",
                f"- Early 3m: `{early_metrics['net_profit']:+.2f}U`, PF `{early_metrics['profit_factor']:.3f}`, DD `{early_metrics['max_drawdown_pct']:.2%}`.",
                f"- 1.5x cost stress 6m: `{stress_six_metrics['net_profit']:+.2f}U`, PF `{stress_six_metrics['profit_factor']:.3f}`, DD `{stress_six_metrics['max_drawdown_pct']:.2%}`.",
                f"- 1.5x cost stress 3m: `{stress_three_metrics['net_profit']:+.2f}U`, PF `{stress_three_metrics['profit_factor']:.3f}`, DD `{stress_three_metrics['max_drawdown_pct']:.2%}`.",
            ]
        )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "strategy_name": VOLATILITY_BREAKOUT_V6_NAME,
        "status": "independent_v6_research_artifacts",
        "config": args.config_output,
        "report": args.output,
        "summary": args.summary,
        "hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                "crypto_scalper/volatility_breakout_v6.py",
                "crypto_scalper/volatility_breakout_v6_optimize.py",
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
        f"selected strict={selected.get('strict_improvement', False)} "
        f"robust={selected.get('robust', False)} "
        f"6m={selected['six']['net_profit']:+.2f}/PF{selected['six']['profit_factor']:.3f}/"
        f"win{selected['six']['win_rate']:.1%}/DD{selected['six']['max_drawdown_pct']:.1%} "
        f"3m={selected['three']['net_profit']:+.2f}/PF{selected['three']['profit_factor']:.3f}/"
        f"win{selected['three']['win_rate']:.1%}/DD{selected['three']['max_drawdown_pct']:.1%}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Breakout v6 point-in-time live-robust optimization"
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
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--long-budget", type=int, default=32)
    parser.add_argument("--long-recent-finalists", type=int, default=12)
    parser.add_argument("--long-finalists", type=int, default=5)
    parser.add_argument("--short-budget", type=int, default=28)
    parser.add_argument("--entry-recent-finalists", type=int, default=24)
    parser.add_argument("--entry-finalists", type=int, default=6)
    parser.add_argument("--exit-budget", type=int, default=36)
    parser.add_argument("--exit-recent-finalists", type=int, default=36)
    parser.add_argument("--robust-finalists", type=int, default=8)
    parser.add_argument(
        "--output", default="reports/volatility_breakout_v6_robust_3m_6m.json"
    )
    parser.add_argument(
        "--summary", default="reports/volatility_breakout_v6_robust_3m_6m.md"
    )
    parser.add_argument(
        "--config-output", default="config.volatility-breakout.v6-robust-50.json"
    )
    parser.add_argument(
        "--manifest", default="config.volatility-breakout.v6-robust-50-manifest.json"
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

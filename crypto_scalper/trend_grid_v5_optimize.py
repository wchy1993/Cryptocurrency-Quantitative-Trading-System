from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from .combined_hybrid_v5_grid_v3_backtest import _slice_signal_data, _write_json
from .risk import BacktestExecutionConfig
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import (
    GridCandidate,
    GridPortfolioConfig,
    _scaled_execution,
    _shift_candidates,
    build_grid_research_timeline,
)
from .trend_grid_v3_optimize import GridMarketOverlay, apply_market_overlay
from .trend_grid_v4 import (
    GridV4EntryGate,
    filter_grid_v4_candidates,
)
from .trend_grid_v5 import (
    TREND_GRID_V5_NAME,
    GridV5ConfidencePolicy,
    grid_v5_tier,
)
from .trend_grid_v5_engine import (
    GridV5ExecutionProfile,
    simulate_grid_v5_portfolio,
)
from .volatility_breakout_v4_research import (
    V4MarketSnapshot,
    build_v4_market_context,
    load_v4_runtime_inputs,
)
from .volatility_breakout_optimize import UNIVERSE_50, sha256_file


def _value(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _profitable_tier(tier: dict[str, Any]) -> bool:
    """Treat an all-winning tier as profitable even when JSON stores PF as null."""
    net_pnl = _value(tier.get("net_pnl"))
    gross_profit = _value(tier.get("gross_profit"))
    gross_loss = _value(tier.get("gross_loss"))
    if net_pnl <= 0.0 or gross_profit <= 0.0:
        return False
    return gross_loss <= 0.0 or gross_profit / gross_loss > 1.0


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_count", "trade_count", "grid_entry_count",
        "grid_take_profit_count", "initial_equity", "final_equity",
        "net_profit", "return_pct", "win_rate", "profit_factor",
        "expectancy_usdt", "expectancy_r", "average_win", "average_loss",
        "average_win_loss_ratio", "max_drawdown_pct",
        "max_drawdown_duration_minutes", "fee", "slippage", "funding",
        "full_cost", "cost_to_raw_gross_profit_ratio",
        "top5_profit_contribution", "positive_months", "negative_months",
        "hard_drawdown_stopped", "by_month", "by_side", "by_exit_reason",
    )
    return {key: result[key] for key in keys}


def _period_score(result: dict[str, Any], anchor: dict[str, Any]) -> float:
    coverage = result["trade_count"] / max(anchor["trade_count"], 1)
    if result["hard_drawdown_stopped"] or coverage < 0.75:
        return -1e9
    net_scale = max(anchor["net_profit"], anchor["initial_equity"] * 0.10)
    return (
        9.0 * (result["net_profit"] - anchor["net_profit"]) / net_scale
        + 1.2 * (_value(result["profit_factor"]) - _value(anchor["profit_factor"]))
        + 4.0 * (result["win_rate"] - anchor["win_rate"])
        + 12.0 * (anchor["max_drawdown_pct"] - result["max_drawdown_pct"])
        - 2.0 * max(0.0, 1.0 - coverage)
        - 0.5 * max(0.0, result["top5_profit_contribution"] - 0.55)
    )


def _pair_score(
    six: dict[str, Any],
    three: dict[str, Any],
    anchor_six: dict[str, Any],
    anchor_three: dict[str, Any],
) -> float:
    return _period_score(six, anchor_six) + 1.40 * _period_score(
        three, anchor_three
    )


def _strict_improvement(
    six: dict[str, Any],
    three: dict[str, Any],
    anchor_six: dict[str, Any],
    anchor_three: dict[str, Any],
) -> bool:
    return (
        six["trade_count"] >= 30
        and three["trade_count"] >= 16
        and six["net_profit"] > anchor_six["net_profit"]
        and three["net_profit"] > anchor_three["net_profit"]
        and _value(six["profit_factor"]) > _value(anchor_six["profit_factor"])
        and _value(three["profit_factor"]) > _value(anchor_three["profit_factor"])
        and six["max_drawdown_pct"] <= anchor_six["max_drawdown_pct"]
        and three["max_drawdown_pct"] <= anchor_three["max_drawdown_pct"]
        and six["win_rate"] >= anchor_six["win_rate"]
        and three["win_rate"] >= anchor_three["win_rate"]
    )


def _policy_variants(seed: int, budget: int) -> list[GridV5ConfidencePolicy]:
    rows = [GridV5ConfidencePolicy()]
    rows.extend(
        [
            GridV5ConfidencePolicy(
                min_quality_score=0.47,
                min_alignment_atr=0.50,
                min_extension_atr=0.15,
                max_extension_atr=0.53,
                max_volume_ratio=1.10,
                max_regime_score=0.50,
                strong_score=5,
                weak_score=2,
                strong_risk_multiplier=1.05,
                standard_risk_multiplier=0.90,
                weak_risk_multiplier=0.55,
                maximum_campaign_risk_pct=0.105,
            ),
            GridV5ConfidencePolicy(
                min_quality_score=0.50,
                min_alignment_atr=0.50,
                min_extension_atr=0.12,
                max_extension_atr=0.55,
                max_volume_ratio=1.15,
                max_regime_score=0.40,
                strong_score=5,
                weak_score=2,
                strong_risk_multiplier=1.08,
                standard_risk_multiplier=0.92,
                weak_risk_multiplier=0.50,
                maximum_campaign_risk_pct=0.108,
            ),
            GridV5ConfidencePolicy(
                min_quality_score=0.45,
                min_alignment_atr=0.45,
                min_extension_atr=0.10,
                max_extension_atr=0.55,
                max_volume_ratio=1.20,
                max_regime_score=0.35,
                strong_score=5,
                weak_score=1,
                strong_risk_multiplier=1.10,
                standard_risk_multiplier=0.90,
                weak_risk_multiplier=0.50,
                strong_target_spacing=1.90,
                weak_target_spacing=1.75,
                weak_loss_limit_r=0.55,
                weak_max_campaign_minutes=2_880,
                maximum_campaign_risk_pct=0.11,
            ),
        ]
    )
    rng = random.Random(seed)
    while len(dict.fromkeys(rows)) < budget:
        weak = rng.choice((0, 1, 2, 3))
        strong = rng.choice((4, 5, 6))
        if weak >= strong:
            continue
        adaptive_exit = rng.random() < 0.55
        rows.append(
            GridV5ConfidencePolicy(
                min_quality_score=rng.choice((0.40, 0.45, 0.47, 0.50, 0.52, 0.55)),
                min_alignment_atr=rng.choice((0.40, 0.45, 0.50, 0.55, 0.60)),
                min_extension_atr=rng.choice((0.05, 0.10, 0.12, 0.15, 0.20)),
                max_extension_atr=rng.choice((0.45, 0.50, 0.53, 0.55, 0.60)),
                max_volume_ratio=rng.choice((1.00, 1.05, 1.10, 1.15, 1.25, 1.35)),
                max_regime_score=rng.choice((0.20, 0.30, 0.40, 0.50)),
                strong_score=strong,
                weak_score=weak,
                strong_risk_multiplier=rng.choice((1.00, 1.03, 1.05, 1.08, 1.10, 1.12)),
                standard_risk_multiplier=rng.choice((0.82, 0.86, 0.90, 0.94, 0.98, 1.0)),
                weak_risk_multiplier=rng.choice((0.35, 0.45, 0.55, 0.65, 0.75, 0.85)),
                strong_target_spacing=(
                    rng.choice((1.85, 1.90, 1.95, 2.00))
                    if adaptive_exit else 1.85
                ),
                standard_target_spacing=(
                    rng.choice((1.80, 1.85, 1.90))
                    if adaptive_exit else 1.85
                ),
                weak_target_spacing=(
                    rng.choice((1.65, 1.70, 1.75, 1.80, 1.85))
                    if adaptive_exit else 1.85
                ),
                strong_loss_limit_r=(
                    rng.choice((0.65, 0.70, 0.75, 0.80))
                    if adaptive_exit else 0.70
                ),
                standard_loss_limit_r=(
                    rng.choice((0.60, 0.65, 0.70, 0.75))
                    if adaptive_exit else 0.70
                ),
                weak_loss_limit_r=(
                    rng.choice((0.40, 0.50, 0.55, 0.60, 0.70))
                    if adaptive_exit else 0.70
                ),
                strong_max_campaign_minutes=(
                    rng.choice((4_320, 5_040, 5_760))
                    if adaptive_exit else 4_320
                ),
                standard_max_campaign_minutes=4_320,
                weak_max_campaign_minutes=(
                    rng.choice((2_160, 2_880, 3_600, 4_320))
                    if adaptive_exit else 4_320
                ),
                maximum_campaign_risk_pct=rng.choice((0.10, 0.103, 0.105, 0.108, 0.11, 0.112)),
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _local_policy_variants(
    source: GridV5ConfidencePolicy,
    seed: int,
    budget: int,
) -> list[GridV5ConfidencePolicy]:
    rows = [
        source,
        replace(source, reject_weak_tier=True),
        GridV5ConfidencePolicy(),
    ]
    field_values = {
        "min_quality_score": (0.40, 0.42, 0.45, 0.47, 0.50),
        "min_alignment_atr": (0.48, 0.50, 0.52, 0.55, 0.58, 0.60),
        "min_extension_atr": (0.15, 0.18, 0.20, 0.22, 0.25),
        "max_extension_atr": (0.40, 0.42, 0.45, 0.48, 0.50),
        "max_volume_ratio": (1.00, 1.05, 1.10, 1.15, 1.20),
        "max_regime_score": (0.20, 0.25, 0.30, 0.35, 0.40),
        "strong_score": (5, 6),
        "weak_score": (0, 1, 2, 3),
        "strong_risk_multiplier": (1.06, 1.08, 1.10, 1.12, 1.14),
        "standard_risk_multiplier": (0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92),
        "maximum_campaign_risk_pct": (0.103, 0.105, 0.108, 0.11, 0.112),
    }
    for field, values in field_values.items():
        rows.extend(
            replace(source, reject_weak_tier=True, **{field: value})
            for value in values
        )
    rng = random.Random(seed)
    fields = tuple(field_values)
    while len(dict.fromkeys(rows)) < budget:
        selected_fields = rng.sample(fields, rng.choice((2, 3, 4, 5)))
        changes = {
            field: rng.choice(field_values[field]) for field in selected_fields
        }
        rows.append(
            replace(source, reject_weak_tier=True, **changes)
        )
    return list(dict.fromkeys(rows))[:budget]


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_six = datetime.fromisoformat(args.start_6m)
    start_three = datetime.fromisoformat(args.start_3m)
    end = datetime.fromisoformat(args.end)
    if not start_six < start_three < end:
        raise ValueError("periods must satisfy start_6m < start_3m < end")

    anchor_config = json.loads(Path(args.anchor_config).read_text(encoding="utf-8"))
    anchor_report = json.loads(Path(args.anchor_report).read_text(encoding="utf-8"))
    anchor_selected = anchor_report["selected"]
    gate = GridV4EntryGate(**anchor_config["entry_gate"])
    overlay = GridMarketOverlay(**anchor_config["market_overlay"])
    base_signal = TrendGridConfig(**anchor_config["signal"])
    base_portfolio = GridPortfolioConfig(**anchor_config["portfolio"])
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
        raise RuntimeError("Grid v5 requires gap-free stitched 1m data")
    if metadata["maximum_missing_minutes"] or metadata["funding_missing_symbols"]:
        raise RuntimeError("Grid v5 requires complete price and funding data")

    def build_period(name: str, start: datetime, finish: datetime) -> dict[str, Any]:
        local_signal = _slice_signal_data(
            signal_data, start - timedelta(days=args.warmup_days), finish
        )
        context = build_v4_market_context(symbols, local_signal)
        raw, snapshots = build_grid_research_timeline(
            symbols,
            local_signal,
            execution_data,
            base_signal,
            start,
            finish,
        )
        selected = filter_grid_v4_candidates(
            apply_market_overlay(raw, context, overlay), context, gate
        )
        print(
            f"{name}: raw={sum(map(len, raw.values()))} "
            f"selected={sum(map(len, selected.values()))}",
            flush=True,
        )
        return {
            "start": start,
            "end": finish,
            "context": context,
            "candidates": selected,
            "snapshots": snapshots,
        }

    periods = {
        "six": build_period("6m", start_six, end),
        "three": build_period("3m", start_three, end),
        "early": build_period("early3m", start_six, start_three),
    }

    def simulate(
        period: str,
        policy: GridV5ConfidencePolicy,
        selected_execution: BacktestExecutionConfig = execution,
        candidate_delay_minutes: int = 0,
        force_compound: Optional[bool] = None,
        skip_symbols: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        values = periods[period]
        candidates = values["candidates"]
        if candidate_delay_minutes:
            candidates = _shift_candidates(
                candidates, candidate_delay_minutes, execution_data
            )
        context: dict[int, dict[str, V4MarketSnapshot]] = values["context"]

        def choose(
            candidate: GridCandidate, minute: int, _equity: float
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
            signal = replace(
                base_signal,
                grid_target_spacing=target,
                campaign_loss_limit_r=loss_limit,
                max_campaign_minutes=duration,
            )
            portfolio = replace(
                base_portfolio,
                risk_per_campaign_pct=(
                    base_portfolio.risk_per_campaign_pct * risk_multiplier
                ),
                max_campaign_risk_pct=policy.maximum_campaign_risk_pct,
                compound=(
                    base_portfolio.compound
                    if force_compound is None
                    else force_compound
                ),
            )
            return GridV5ExecutionProfile(
                f"{tier}_score_{score}", signal, portfolio
            )

        return simulate_grid_v5_portfolio(
            candidates,
            values["snapshots"],
            symbols,
            execution_data,
            rules,
            base_signal,
            replace(
                base_portfolio,
                compound=(
                    base_portfolio.compound
                    if force_compound is None
                    else force_compound
                ),
            ),
            selected_execution,
            values["start"],
            values["end"],
            args.initial_equity,
            profile_selector=choose,
            skip_symbols=skip_symbols,
        )

    anchor_policy = GridV5ConfidencePolicy()
    anchor_six = simulate("six", anchor_policy)
    anchor_three = simulate("three", anchor_policy)
    for label, actual, expected in (
        ("six", anchor_six, anchor_selected["six_month_full"]),
        ("three", anchor_three, anchor_selected["three_month_full"]),
    ):
        if abs(actual["net_profit"] - expected["net_profit"]) > 1e-6:
            raise RuntimeError(
                f"Grid v5 anchor mismatch {label}: "
                f"{actual['net_profit']} != {expected['net_profit']}"
            )
    print(
        f"v4 anchor 6m={anchor_six['net_profit']:+.2f}/PF{anchor_six['profit_factor']:.3f}/"
        f"win{anchor_six['win_rate']:.1%}/DD{anchor_six['max_drawdown_pct']:.1%}; "
        f"3m={anchor_three['net_profit']:+.2f}/PF{anchor_three['profit_factor']:.3f}/"
        f"win{anchor_three['win_rate']:.1%}/DD{anchor_three['max_drawdown_pct']:.1%}",
        flush=True,
    )

    six_rows: list[dict[str, Any]] = []
    if args.refine_from_report:
        source_payload = json.loads(
            Path(args.refine_from_report).read_text(encoding="utf-8")
        )["selected"]["confidence_policy"]
        source_policy = GridV5ConfidencePolicy(**source_payload)
        policies = _local_policy_variants(
            source_policy, args.seed, args.policy_budget
        )
    else:
        policies = _policy_variants(args.seed, args.policy_budget)
    for number, policy in enumerate(policies, 1):
        six = simulate("six", policy)
        six_rows.append(
            {"policy": policy, "six": six,
             "score6": _period_score(six, anchor_six)}
        )
        if number == 1 or number % 20 == 0 or number == len(policies):
            print(
                f"policy {number}/{len(policies)}: {six['net_profit']:+.1f}/"
                f"PF{six['profit_factor']:.2f}/win{six['win_rate']:.1%}/"
                f"DD{six['max_drawdown_pct']:.1%}", flush=True
            )
    six_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in six_rows[: args.recent_finalists]:
        row["three"] = simulate("three", row["policy"])
        row["pair_score"] = _pair_score(
            row["six"], row["three"], anchor_six, anchor_three
        )
        row["strict_improvement"] = _strict_improvement(
            row["six"], row["three"], anchor_six, anchor_three
        )
    recent = sorted(
        six_rows[: args.recent_finalists],
        key=lambda row: (row["strict_improvement"], row["pair_score"]),
        reverse=True,
    )

    robust_rows: list[dict[str, Any]] = []
    stressed = _scaled_execution(execution, 1.5)
    for number, row in enumerate(recent[: args.robust_finalists], 1):
        policy = row["policy"]
        early = simulate("early", policy)
        stress_six = simulate("six", policy, stressed)
        stress_three = simulate("three", policy, stressed)
        delay_six = simulate("six", policy, candidate_delay_minutes=1)
        delay_three = simulate("three", policy, candidate_delay_minutes=1)
        fixed_six = simulate("six", policy, force_compound=False)
        fixed_three = simulate("three", policy, force_compound=False)
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
            "six", policy,
            skip_symbols=frozenset({top_six}) if top_six else frozenset()
        )
        no_top_three = simulate(
            "three", policy,
            skip_symbols=frozenset({top_three}) if top_three else frozenset()
        )
        active_tiers = [
            tier
            for result in (row["six"], row["three"])
            for tier in result.get("by_v5_tier", {}).values()
            if int(tier.get("trade_count", 0)) > 0
        ]
        tier_robust = bool(active_tiers) and all(
            _profitable_tier(tier) for tier in active_tiers
        )
        robust = (
            row["strict_improvement"]
            and tier_robust
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
             "tier_robust": tier_robust, "robust": robust,
             "stress_better": stress_better,
             "robust_score": row["pair_score"] + (10.0 if robust else 0.0)
             + (3.0 if stress_better else 0.0)}
        )
        robust_rows.append(row)
        print(
            f"robust {number}/{args.robust_finalists}: strict={row['strict_improvement']} "
            f"tiers={tier_robust} early={early['net_profit']:+.1f} "
            f"stress6={stress_six['net_profit']:+.1f} delay6={delay_six['net_profit']:+.1f} "
            f"fixed6={fixed_six['net_profit']:+.1f} noTop6={no_top_six['net_profit']:+.1f}",
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
            "tier_robust": row.get("tier_robust", False),
            "robust": row.get("robust", False),
            "stress_better": row.get("stress_better", False),
            "confidence_policy": row["policy"].as_dict(),
            "six_month": _compact(row["six"]),
            "three_month": _compact(row["three"]),
            "six_month_tiers": row["six"].get("by_v5_tier"),
            "three_month_tiers": row["three"].get("by_v5_tier"),
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
        else "best_available_did_not_clear_grid_v4_anchor"
    )
    report = {
        "strategy_name": TREND_GRID_V5_NAME,
        "status": "independent_research_grid_v4_and_gui_unchanged",
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
            "policy_evaluations": len(six_rows),
            "recent_evaluations": min(len(six_rows), args.recent_finalists),
            "robust_evaluations": len(robust_rows),
        },
        "baseline": {
            "strategy": "Grid v4 frozen live-robust",
            "source_config": args.anchor_config,
            "source_report": args.anchor_report,
            "six_month": _compact(anchor_six),
            "three_month": _compact(anchor_three),
        },
        "selected": public(selected, True),
        "leaderboard": [public(row) for row in recent[:20]],
        "preserved": {
            "grid_v4": "unchanged",
            "breakout_v6": "unchanged",
            "breakout_v7": "unchanged",
            "active_gui": "unchanged",
            "apt_grid": "unchanged",
        },
    }
    _write_json(args.output, report)
    config = {
        "strategy_name": TREND_GRID_V5_NAME,
        "status": "independent_research_candidate_not_live",
        "selection_status": selection_status,
        "entry_gate": gate.as_dict(),
        "market_overlay": asdict(overlay),
        "signal": base_signal.as_dict(),
        "portfolio": asdict(base_portfolio),
        "confidence_policy": selected["policy"].as_dict(),
        "results": {
            "six_month": _compact(selected["six"]),
            "three_month": _compact(selected["three"]),
        },
    }
    _write_json(args.config_output, config)
    lines = [
        "# Grid v5 confidence-managed optimization",
        "",
        "- Frozen Grid v4 is the strict anchor and remains unchanged",
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
        for name, result in (("Grid v4 anchor", anchor_result), ("Grid v5", selected_result)):
            lines.append(
                f"| {period} | {name} | {result['trade_count']} | "
                f"{result['net_profit']:+.2f}U | {result['profit_factor']:.3f} | "
                f"{result['win_rate']:.2%} | {result['max_drawdown_pct']:.2%} |"
            )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "strategy_name": TREND_GRID_V5_NAME,
        "status": "independent_research_artifacts",
        "config": args.config_output,
        "report": args.output,
        "summary": args.summary,
        "hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                "crypto_scalper/trend_grid_v5.py",
                "crypto_scalper/trend_grid_v5_engine.py",
                "crypto_scalper/trend_grid_v5_optimize.py",
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
        description="Grid v5 confidence-managed live-realistic optimization"
    )
    parser.add_argument("--start-6m", default="2026-01-19T00:00:00")
    parser.add_argument("--start-3m", default="2026-04-19T00:00:00")
    parser.add_argument("--end", default="2026-07-19T00:00:00")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument(
        "--anchor-config", default="config.trend-grid.v4-live-robust-final-50.json"
    )
    parser.add_argument(
        "--anchor-report", default="reports/trend_grid_v4_live_robust_final_3m_6m.json"
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
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--refine-from-report",
        default="",
        help="Optional prior Grid v5 report used as a local-search center",
    )
    parser.add_argument("--policy-budget", type=int, default=180)
    parser.add_argument("--recent-finalists", type=int, default=48)
    parser.add_argument("--robust-finalists", type=int, default=12)
    parser.add_argument(
        "--output", default="reports/trend_grid_v5_confidence_3m_6m.json"
    )
    parser.add_argument(
        "--summary", default="reports/trend_grid_v5_confidence_3m_6m.md"
    )
    parser.add_argument(
        "--config-output", default="config.trend-grid.v5-confidence-50.json"
    )
    parser.add_argument(
        "--manifest", default="config.trend-grid.v5-confidence-50-manifest.json"
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

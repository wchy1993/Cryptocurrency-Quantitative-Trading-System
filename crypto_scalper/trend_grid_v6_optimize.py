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
from .volatility_breakout_optimize import UNIVERSE_50, sha256_file
from .volatility_breakout_v4_research import (
    V4MarketSnapshot,
    build_v4_market_context,
    load_v4_runtime_inputs,
)


def _pf(result: dict[str, Any]) -> float:
    value = result.get("profit_factor")
    if value is None:
        return math.inf if result.get("net_profit", 0.0) > 0.0 else 0.0
    number = float(value)
    return number if not math.isnan(number) else 0.0


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_count",
        "trade_count",
        "grid_entry_count",
        "grid_take_profit_count",
        "initial_equity",
        "final_equity",
        "net_profit",
        "return_pct",
        "win_rate",
        "profit_factor",
        "expectancy_usdt",
        "expectancy_r",
        "average_win",
        "average_loss",
        "average_win_loss_ratio",
        "max_drawdown_pct",
        "max_drawdown_duration_minutes",
        "fee",
        "slippage",
        "funding",
        "full_cost",
        "cost_to_raw_gross_profit_ratio",
        "top5_profit_contribution",
        "positive_months",
        "negative_months",
        "hard_drawdown_stopped",
        "by_month",
        "by_side",
        "by_exit_reason",
    )
    return {key: result[key] for key in keys}


def _strict_improvement(
    six: dict[str, Any],
    three: dict[str, Any],
    base_six: dict[str, Any],
    base_three: dict[str, Any],
) -> bool:
    return (
        six["trade_count"] >= 32
        and three["trade_count"] >= 18
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
    if (
        result["hard_drawdown_stopped"]
        or result["trade_count"] < 0.78 * baseline["trade_count"]
    ):
        return -1e12
    net_scale = max(baseline["net_profit"], 1.0)
    pf_gain = min(_pf(result), 100.0) - min(_pf(baseline), 100.0)
    return (
        8.0 * (result["net_profit"] - baseline["net_profit"]) / net_scale
        + 0.22 * pf_gain
        + 18.0
        * (baseline["max_drawdown_pct"] - result["max_drawdown_pct"])
        + 4.0 * (result["win_rate"] - baseline["win_rate"])
        - 0.8 * max(0.0, result["top5_profit_contribution"] - 0.50)
    )


def _pair_score(
    six: dict[str, Any],
    three: dict[str, Any],
    base_six: dict[str, Any],
    base_three: dict[str, Any],
) -> float:
    return _period_score(six, base_six) + 1.35 * _period_score(
        three, base_three
    )


def _structural_variants(
    seed: int, budget: int
) -> list[GridV6CampaignPolicy]:
    anchor = GridV6CampaignPolicy()
    rows = [anchor]
    extension_values = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15)
    for value in extension_values:
        rows.append(replace(anchor, minimum_actual_extension_atr=value))
    for value in (1.05, 1.10, 1.15, 1.20):
        rows.append(replace(anchor, maximum_actual_volume_ratio=value))
    locks = tuple(
        (activation, giveback)
        for activation in (0.15, 0.20, 0.25, 0.30, 0.35)
        for giveback in (0.05, 0.10, 0.15, 0.20)
        if giveback <= activation
    )
    for activation, giveback in locks:
        rows.append(
            replace(
                anchor,
                profit_lock_activation_r=activation,
                profit_giveback_r=giveback,
            )
        )
    for value in (0.30, 0.35, 0.40, 0.45, 0.50):
        rows.append(replace(anchor, campaign_take_profit_r=value))
    for value in (0.50, 0.55, 0.60, 0.65, 0.70):
        rows.append(replace(anchor, campaign_loss_limit_r=value))
    for value in (1.65, 1.75, 1.85, 1.95, 2.05):
        rows.append(replace(anchor, target_spacing=value))
    for value in (2_880, 3_600, 4_320, 5_040, 5_760):
        rows.append(replace(anchor, max_campaign_minutes=value))
    for extension in (0.01, 0.02, 0.05):
        for activation, giveback in locks:
            rows.append(
                replace(
                    anchor,
                    minimum_actual_extension_atr=extension,
                    profit_lock_activation_r=activation,
                    profit_giveback_r=giveback,
                )
            )
    rng = random.Random(seed)
    while len(dict.fromkeys(rows)) < budget:
        activation, giveback = rng.choice(locks + ((0.0, 0.0),))
        take_profit = rng.choice((0.0, 0.0, 0.35, 0.40, 0.45, 0.50))
        if activation > 0.0:
            take_profit = 0.0
        rows.append(
            replace(
                anchor,
                minimum_actual_extension_atr=rng.choice(extension_values),
                maximum_actual_volume_ratio=rng.choice(
                    (1.05, 1.10, 1.15, 1.20, 999.0)
                ),
                target_spacing=rng.choice((1.70, 1.80, 1.85, 1.90, 2.00)),
                campaign_loss_limit_r=rng.choice(
                    (0.55, 0.60, 0.65, 0.70)
                ),
                max_campaign_minutes=rng.choice(
                    (3_600, 4_320, 5_040)
                ),
                campaign_take_profit_r=take_profit,
                profit_lock_activation_r=activation,
                profit_giveback_r=giveback,
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def _allocation_variants(
    source: GridV6CampaignPolicy, seed: int, budget: int
) -> list[GridV6CampaignPolicy]:
    rows = [source]
    for value in (0.10, 0.105, 0.11, 0.115, 0.12, 0.125):
        rows.append(replace(source, maximum_campaign_risk_pct=value))
    # Reallocate a small amount of risk away from score-6 campaigns, whose
    # intra-campaign excursion sets the current drawdown ceiling, and toward
    # the more numerous standard score-3..5 campaigns.  All inputs remain
    # frozen at campaign entry.
    for cap in (0.112, 0.114, 0.115, 0.118, 0.12):
        for score6 in (0.75, 0.78, 0.80, 0.82, 0.85):
            for standard in (1.02, 1.04, 1.06, 1.10):
                rows.append(
                    replace(
                        source,
                        maximum_campaign_risk_pct=cap,
                        score_3_risk_factor=standard,
                        score_4_risk_factor=standard,
                        score_5_risk_factor=standard,
                        score_6_risk_factor=score6,
                    )
                )
    fields = (
        "score_3_risk_factor",
        "score_4_risk_factor",
        "score_5_risk_factor",
        "score_6_risk_factor",
    )
    values = (0.80, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20)
    for field in fields:
        rows.extend(replace(source, **{field: value}) for value in values)
    rng = random.Random(seed)
    while len(dict.fromkeys(rows)) < budget:
        changes = {
            field: rng.choice(values)
            for field in rng.sample(fields, rng.choice((2, 3, 4)))
        }
        changes["maximum_campaign_risk_pct"] = rng.choice(
            (0.105, 0.11, 0.115, 0.12, 0.125)
        )
        rows.append(replace(source, **changes))
    return list(dict.fromkeys(rows))[:budget]


def _local_structural_variants(
    source: GridV6CampaignPolicy,
    seed: int,
    budget: int,
) -> list[GridV6CampaignPolicy]:
    rows = [source]
    extensions = tuple(round(value * 0.005, 6) for value in range(2, 31))
    rows.extend(
        replace(source, minimum_actual_extension_atr=value)
        for value in extensions
    )
    volumes = (1.075, 1.09, 1.10, 1.105, 1.11, 1.125, 1.15, 999.0)
    rows.extend(
        replace(source, maximum_actual_volume_ratio=value)
        for value in volumes
    )
    for extension in (0.025, 0.035, 0.045, 0.05, 0.055, 0.065, 0.075):
        for volume in (1.10, 1.105, 1.11, 1.15, 999.0):
            rows.append(
                replace(
                    source,
                    minimum_actual_extension_atr=extension,
                    maximum_actual_volume_ratio=volume,
                )
            )
    for value in (1.78, 1.82, 1.85, 1.88, 1.92):
        rows.append(replace(source, target_spacing=value))
    rng = random.Random(seed)
    while len(dict.fromkeys(rows)) < budget:
        rows.append(
            replace(
                source,
                minimum_actual_extension_atr=rng.choice(extensions),
                maximum_actual_volume_ratio=rng.choice(volumes),
                target_spacing=rng.choice((1.80, 1.85, 1.90)),
                campaign_loss_limit_r=rng.choice((0.65, 0.70)),
                max_campaign_minutes=rng.choice((4_080, 4_320, 4_560)),
            )
        )
    return list(dict.fromkeys(rows))[:budget]


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_six = datetime.fromisoformat(args.start_6m)
    start_three = datetime.fromisoformat(args.start_3m)
    end = datetime.fromisoformat(args.end)
    if not start_six < start_three < end:
        raise ValueError("periods must satisfy start_6m < start_3m < end")

    payload = json.loads(Path(args.anchor_config).read_text(encoding="utf-8"))
    gate = GridV4EntryGate(**payload["entry_gate"])
    overlay = GridMarketOverlay(**payload["market_overlay"])
    base_signal = TrendGridConfig(**payload["signal"])
    base_portfolio = GridPortfolioConfig(**payload["portfolio"])
    confidence = GridV5ConfidencePolicy(**payload["confidence_policy"])
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
        raise RuntimeError("Grid v6 requires gap-free price and funding data")

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
            apply_market_overlay(raw, context, overlay),
            context,
            gate,
        )
        print(
            f"{name}: raw={sum(map(len, raw.values()))} "
            f"v5={sum(map(len, selected.values()))}",
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
        policy: GridV6CampaignPolicy,
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
            accepted, tier, score, v6_factor = grid_v6_entry_decision(
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
            signal = replace(
                base_signal,
                grid_target_spacing=policy.target_spacing,
                campaign_loss_limit_r=policy.campaign_loss_limit_r,
                max_campaign_minutes=policy.max_campaign_minutes,
                campaign_take_profit_r=policy.campaign_take_profit_r,
                profit_lock_activation_r=policy.profit_lock_activation_r,
                profit_giveback_r=policy.profit_giveback_r,
            )
            portfolio = replace(
                base_portfolio,
                risk_per_campaign_pct=(
                    base_portfolio.risk_per_campaign_pct
                    * v5_risk
                    * v6_factor
                ),
                max_campaign_risk_pct=policy.maximum_campaign_risk_pct,
                compound=(
                    base_portfolio.compound
                    if force_compound is None
                    else force_compound
                ),
            )
            return GridV5ExecutionProfile(
                f"v6_{tier}_score_{score}", signal, portfolio
            )

        result = simulate_grid_v5_portfolio(
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
        result["strategy_version"] = "v6_profit_protected"
        return result

    anchor_policy = GridV6CampaignPolicy()
    expected_payload = payload
    baseline_name = "Grid v5 frozen"
    if args.refine_from_config:
        refine_payload = json.loads(
            Path(args.refine_from_config).read_text(encoding="utf-8")
        )
        anchor_policy = GridV6CampaignPolicy(
            **refine_payload["campaign_policy"]
        )
        expected_payload = refine_payload
        baseline_name = "Grid v6 prior strict candidate"
    base_six = simulate("six", anchor_policy)
    base_three = simulate("three", anchor_policy)
    for name, actual, expected in (
        ("6m", base_six, expected_payload["results"]["six_month"]),
        ("3m", base_three, expected_payload["results"]["three_month"]),
    ):
        if abs(actual["net_profit"] - expected["net_profit"]) > 1e-6:
            raise RuntimeError(
                f"Grid v6 anchor mismatch {name}: "
                f"{actual['net_profit']} != {expected['net_profit']}"
            )
    print(
        f"v5 anchor 6m={base_six['net_profit']:+.2f}/PF{_pf(base_six):.3f}/"
        f"DD{base_six['max_drawdown_pct']:.2%}; "
        f"3m={base_three['net_profit']:+.2f}/PF{_pf(base_three):.3f}/"
        f"DD{base_three['max_drawdown_pct']:.2%}",
        flush=True,
    )

    structural_rows: list[dict[str, Any]] = []
    policies = (
        _local_structural_variants(
            anchor_policy, args.seed, args.structural_budget
        )
        if args.refine_from_config
        else _structural_variants(args.seed, args.structural_budget)
    )
    for number, policy in enumerate(policies, 1):
        six = simulate("six", policy)
        structural_rows.append(
            {
                "policy": policy,
                "six": six,
                "score6": _period_score(six, base_six),
            }
        )
        if number == 1 or number % 20 == 0 or number == len(policies):
            print(
                f"structure {number}/{len(policies)} "
                f"{six['net_profit']:+.1f}/PF{_pf(six):.2f}/"
                f"DD{six['max_drawdown_pct']:.1%}",
                flush=True,
            )
    structural_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in structural_rows[: args.recent_finalists]:
        row["three"] = simulate("three", row["policy"])
        row["pair_score"] = _pair_score(
            row["six"], row["three"], base_six, base_three
        )
        row["strict"] = _strict_improvement(
            row["six"], row["three"], base_six, base_three
        )
    structural_finalists = sorted(
        structural_rows[: args.recent_finalists],
        key=lambda row: (row.get("strict", False), row.get("pair_score", -1e12)),
        reverse=True,
    )[: args.structural_finalists]

    allocation_rows: list[dict[str, Any]] = []
    for source_number, source in enumerate(structural_finalists, 1):
        variants = _allocation_variants(
            source["policy"],
            args.seed + source_number,
            args.allocation_budget,
        )
        for policy in variants:
            six = simulate("six", policy)
            allocation_rows.append(
                {
                    "policy": policy,
                    "six": six,
                    "score6": _period_score(six, base_six),
                }
            )
        print(
            f"allocation source {source_number}/{len(structural_finalists)} complete",
            flush=True,
        )
    allocation_rows.sort(key=lambda row: row["score6"], reverse=True)
    for row in allocation_rows[: args.allocation_recent_finalists]:
        row["three"] = simulate("three", row["policy"])
        row["pair_score"] = _pair_score(
            row["six"], row["three"], base_six, base_three
        )
        row["strict"] = _strict_improvement(
            row["six"], row["three"], base_six, base_three
        )
    recent = sorted(
        allocation_rows[: args.allocation_recent_finalists],
        key=lambda row: (row.get("strict", False), row.get("pair_score", -1e12)),
        reverse=True,
    )

    stressed = _scaled_execution(execution, 1.5)
    robust_rows: list[dict[str, Any]] = []
    for number, row in enumerate(recent[: args.robust_finalists], 1):
        policy = row["policy"]
        early = simulate("early", policy)
        stress_six = simulate("six", policy, selected_execution=stressed)
        stress_three = simulate("three", policy, selected_execution=stressed)
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
            "six",
            policy,
            skip_symbols=frozenset({top_six}) if top_six else frozenset(),
        )
        no_top_three = simulate(
            "three",
            policy,
            skip_symbols=(
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
            f"robust={robust} early={early['net_profit']:+.1f} "
            f"stress6={stress_six['net_profit']:+.1f} "
            f"delay6={delay_six['net_profit']:+.1f}",
            flush=True,
        )

    selectable = [row for row in robust_rows if row["robust"]]
    selected = max(
        selectable or robust_rows or recent or structural_finalists,
        key=lambda row: row.get("robust_score", row.get("pair_score", -1e12)),
    )
    selection_status = (
        (
            "strict_robust_improvement_over_prior_grid_v6"
            if args.refine_from_config
            else "strict_robust_improvement_over_grid_v5"
        )
        if selected.get("robust")
        else (
            "best_available_did_not_clear_prior_grid_v6"
            if args.refine_from_config
            else "best_available_did_not_clear_grid_v5"
        )
    )

    def public(row: dict[str, Any], full: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "pair_score": row.get("pair_score"),
            "strict_improvement": row.get("strict", False),
            "robust": row.get("robust", False),
            "campaign_policy": row["policy"].as_dict(),
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
                output[target] = _compact(row[source])
        if "top_six" in row:
            output["removed_top_symbol_six_month"] = row["top_six"]
            output["removed_top_symbol_three_month"] = row["top_three"]
        if full:
            output["six_month_full"] = row["six"]
            output["three_month_full"] = row["three"]
        return output

    report = {
        "strategy_name": TREND_GRID_V6_NAME,
        "status": "independent_research_v5_and_gui_unchanged",
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
            "structural_evaluations": len(structural_rows),
            "allocation_evaluations": len(allocation_rows),
            "robust_evaluations": len(robust_rows),
        },
        "baseline": {
            "strategy": baseline_name,
            "source_config": args.refine_from_config or args.anchor_config,
            "six_month": _compact(base_six),
            "three_month": _compact(base_three),
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
        "strategy_name": TREND_GRID_V6_NAME,
        "status": "independent_research_candidate_not_live",
        "selection_status": selection_status,
        "entry_gate": gate.as_dict(),
        "market_overlay": asdict(overlay),
        "signal": base_signal.as_dict(),
        "portfolio": asdict(base_portfolio),
        "confidence_policy": confidence.as_dict(),
        "campaign_policy": selected["policy"].as_dict(),
        "results": {
            "six_month": _compact(selected["six"]),
            "three_month": _compact(selected["three"]),
        },
    }
    _write_json(args.config_output, config)
    lines = [
        "# Grid v6 campaign-protection optimization",
        "",
        "- Grid v5 and the active GUI remain unchanged",
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
        for name, values in (("Grid v5", baseline), ("Grid v6", result)):
            lines.append(
                f"| {period} | {name} | {values['trade_count']} | "
                f"{values['net_profit']:+.2f}U | {_pf(values):.3f} | "
                f"{values['win_rate']:.2%} | "
                f"{values['max_drawdown_pct']:.2%} |"
            )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "strategy_name": TREND_GRID_V6_NAME,
        "status": "independent_research_artifacts",
        "config": args.config_output,
        "report": args.output,
        "summary": args.summary,
        "hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                "crypto_scalper/trend_grid_v6.py",
                "crypto_scalper/trend_grid_v6_optimize.py",
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
        description="Grid v6 profit-protection live-realistic optimization"
    )
    parser.add_argument("--start-6m", default="2026-01-19T00:00:00")
    parser.add_argument("--start-3m", default="2026-04-19T00:00:00")
    parser.add_argument("--end", default="2026-07-19T00:00:00")
    parser.add_argument("--warmup-days", type=int, default=7)
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument(
        "--anchor-config",
        default="config.trend-grid.v5-confidence-final-50.json",
    )
    parser.add_argument(
        "--refine-from-config",
        default="",
        help=(
            "Optional prior strict v6 config used as the exact local-search "
            "anchor; the frozen v5 source remains unchanged"
        ),
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
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--structural-budget", type=int, default=140)
    parser.add_argument("--recent-finalists", type=int, default=60)
    parser.add_argument("--structural-finalists", type=int, default=6)
    parser.add_argument("--allocation-budget", type=int, default=24)
    parser.add_argument("--allocation-recent-finalists", type=int, default=60)
    parser.add_argument("--robust-finalists", type=int, default=10)
    parser.add_argument(
        "--output",
        default="reports/trend_grid_v6_profit_protected_3m_6m.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/trend_grid_v6_profit_protected_3m_6m.md",
    )
    parser.add_argument(
        "--config-output",
        default="config.trend-grid.v6-profit-protected-50.json",
    )
    parser.add_argument(
        "--manifest",
        default="config.trend-grid.v6-profit-protected-50-manifest.json",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REPORT_FILES = {
    "stage0": "cmipr_stage25_stage0_invariance_3m.json",
    "stage2": "cmipr_stage25_stage2_event_diagnostics_3m.json",
    "stage4": "cmipr_stage25_stage4_event_structure_validation.json",
    "stage5": "cmipr_stage25_stage5_r_basis_validation.json",
    "stage5_r3_capped": "cmipr_stage25_stage5_r3_full_initial_risk_capped_validation.json",
    "stage6": "cmipr_stage25_stage6_fail_fast_validation.json",
    "stage7": "cmipr_stage25_stage7_entry_revalidation_validation.json",
    "backcast_1": "cmipr_stage25_stage9_frozen_train_backcast_202507_202509.json",
    "backcast_2": "cmipr_stage25_stage9_frozen_train_backcast_202510_202512.json",
    "historical_test": "cmipr_stage25_stage9_frozen_historical_test_202604_202606.json",
    "campaign_stage2": "cmipr_convex_campaign_stage2_3m.json",
}


def _load(report_dir: Path, name: str) -> dict[str, Any]:
    return json.loads((report_dir / REPORT_FILES[name]).read_text(encoding="utf-8"))


def _result(report: dict[str, Any], name: str) -> dict[str, Any]:
    return report["results"][name]


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "trade_count",
        "net_pnl",
        "profit_factor",
        "expectancy_per_trade",
        "win_rate_pct",
        "average_win",
        "average_loss",
        "average_win_loss_ratio",
        "max_drawdown_pct",
        "fee",
        "slippage_cost",
        "funding",
        "full_cost_total",
        "cost_to_gross_profit_ratio",
    )
    return {key: result.get(key) for key in keys}


def _trimmed_mean(values: list[float], fraction: float = 0.20) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    retained = ordered[trim:len(ordered) - trim] if trim and len(ordered) > trim * 2 else ordered
    return statistics.mean(retained)


def _aggregate_temporal_segments(segments: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    months: list[dict[str, Any]] = []
    segment_metrics: dict[str, Any] = {}
    for label, result in segments:
        trades.extend(result.get("trades", []))
        segment_metrics[label] = _metrics(result)
        for row in result.get("monthly", []):
            months.append(
                {
                    "classification": label,
                    "month": row["month"],
                    "trade_count": row["closed"],
                    "net_pnl": row["pnl"],
                    "profit_factor": row["long"]["profit_factor"],
                }
            )
    pnls = [float(trade.get("net_pnl", 0.0) or 0.0) for trade in trades]
    wins = [value for value in pnls if value > 0.0]
    losses = [value for value in pnls if value <= 0.0]
    initial_rs = [float(trade["net_initial_leg_r"]) for trade in trades if trade.get("net_initial_leg_r") is not None]
    campaign_rs = [float(trade["net_campaign_r"]) for trade in trades if trade.get("net_campaign_r") is not None]
    symbol_pnl: dict[str, float] = defaultdict(float)
    for trade in trades:
        symbol_pnl[str(trade.get("symbol", ""))] += float(trade.get("net_pnl", 0.0) or 0.0)
    full_cost = sum(float(result.get("full_cost_total", 0.0) or 0.0) for _, result in segments)
    gross_profit = sum(wins)
    return {
        "classification": "fixed_config_chronological_evidence_not_untouched_holdout",
        "effective_months": "2025-07 through 2026-06",
        "trade_count": len(trades),
        "net_pnl": sum(pnls),
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
        "expectancy_per_trade": statistics.mean(pnls) if pnls else None,
        "win_rate_pct": len(wins) / len(pnls) * 100.0 if pnls else None,
        "average_win": statistics.mean(wins) if wins else None,
        "median_win": statistics.median(wins) if wins else None,
        "trimmed_mean_win_20pct": _trimmed_mean(wins),
        "average_loss": statistics.mean(losses) if losses else None,
        "median_loss": statistics.median(losses) if losses else None,
        "average_win_loss_ratio": statistics.mean(wins) / abs(statistics.mean(losses)) if wins and losses else None,
        "average_initial_leg_r": statistics.mean(initial_rs) if initial_rs else None,
        "average_campaign_r": statistics.mean(campaign_rs) if campaign_rs else None,
        "full_cost_total": full_cost,
        "cost_to_gross_profit_ratio": full_cost / gross_profit if gross_profit else None,
        "positive_month_count": sum(row["net_pnl"] > 0.0 for row in months),
        "negative_month_count": sum(row["net_pnl"] <= 0.0 for row in months),
        "month_count": len(months),
        "worst_segment_path_aware_drawdown_pct": max(
            float(result.get("max_drawdown_pct", 0.0) or 0.0) for _, result in segments
        ),
        "drawdown_duration": None,
        "drawdown_duration_note": "not aggregated across independently reset path-aware segments",
        "by_month": months,
        "by_symbol_net_pnl": dict(sorted(symbol_pnl.items(), key=lambda item: (-item[1], item[0]))),
        "segments": segment_metrics,
    }


def build_report(report_dir: Path) -> dict[str, Any]:
    stage0 = _load(report_dir, "stage0")
    stage2_result = _result(_load(report_dir, "stage2"), "stage25_event_diagnostics")
    diagnostics = stage2_result["convex_campaign_diagnostics"]
    stage4 = _load(report_dir, "stage4")
    stage5 = _load(report_dir, "stage5")
    stage5_r3 = _result(_load(report_dir, "stage5_r3_capped"), "r3_full_initial_diagnostic")
    stage6 = _load(report_dir, "stage6")
    stage7 = _load(report_dir, "stage7")
    validation = _result(stage7, "revalidation_basic")
    historical_test = _result(_load(report_dir, "historical_test"), "revalidation_basic")
    backcast_1 = _result(_load(report_dir, "backcast_1"), "revalidation_basic")
    backcast_2 = _result(_load(report_dir, "backcast_2"), "revalidation_basic")
    campaign_stage2 = _load(report_dir, "campaign_stage2")
    early = _result(campaign_stage2, "campaign2_1_early_regime")
    early_zec = float(early.get("by_symbol_net_pnl", {}).get("ZECUSDT", 0.0) or 0.0)
    temporal = _aggregate_temporal_segments(
        [
            ("backcast", backcast_1),
            ("backcast", backcast_2),
            ("validation", validation),
            ("historical_test", historical_test),
        ]
    )
    audit = stage2_result["r_basis_audit"]
    distributions = audit["distributions"]
    old_fail_fast = diagnostics["fail_fast_counterfactual"]
    conditional_validation = _result(stage6, "ff_conditional_20m")
    conditional_cf = conditional_validation["convex_campaign_diagnostics"]["fail_fast_counterfactual"]
    historical_cf = historical_test["convex_campaign_diagnostics"]["fail_fast_counterfactual"]
    score = diagnostics["score_decile_analysis"]
    immediate_45 = _result(stage4, "event_b_immediate_45m")
    delayed = _result(stage4, "event_d_recompression_60m")
    combined_delayed = _result(stage4, "event_e_immediate_plus_recompression_60m")
    r3_fraction = stage5_r3["r_basis_audit"]["distributions"]["initial_leg_actual_risk_fraction"]

    return {
        "research_name": "CMIPR Stage 2.5 - R Basis Audit, Event Diagnostics, Stale Signal Control, Conditional Fail-Fast",
        "strategy_name": "cross_sectional_momentum_ignition_pyramid",
        "research_status": "archived_no_proven_alpha",
        "final_acceptance_source": "post_freeze_shadow_or_dry_run",
        "final_acceptance_reached": False,
        "baseline_invariance_report": stage0["baseline_invariance_report"],
        "stage_status": {
            "stage_0": "passed_exact_reproduction",
            "stage_1": "passed_dual_r_audit",
            "stage_2": "passed_event_diagnostics_no_trade_change",
            "stage_3": "passed_isolated_fail_fast_counterfactual",
            "stage_4": "passed_stale_event_safety; delayed_recompression_edge_not_proven",
            "stage_5": "passed_r_basis_comparison; initial_leg_selected",
            "stage_6": "conditional_20m_selected_on_validation_only",
            "stage_7": "basic_revalidation_selected_on_validation_only",
            "stage_8": "not_run_gate_failed_before_maker_entry",
            "stage_9": "stop_condition_met_and_strategy_archived",
        },
        "frozen_research_candidate": {
            "direction": "long_only",
            "entry_mode": "first_pullback",
            "event_structure": "immediate_first_pullback",
            "maximum_ignition_age_minutes": 45,
            "take_profit_r": 1.5,
            "take_profit_r_basis": "initial_leg",
            "fail_fast_r_basis": "initial_leg",
            "fail_fast_mode": "hard_invalidation_plus_conditional_non_extension",
            "conditional_fail_fast_minutes": 20,
            "entry_revalidation": "basic",
            "bull_flag": False,
            "addon": False,
            "runner": False,
            "short": False,
        },
        "r_basis_audit": {
            "definition_version": audit.get("r_definition_version", "cmipr_dual_r_v1"),
            "campaign_risk_definition": "campaign_start_equity * campaign_risk_pct",
            "initial_leg_risk_definition": "actual fill-to-estimated-stop loss plus entry fee and estimated stop exit fee/slippage",
            "actual_initial_risk_fraction_distribution": distributions["initial_leg_actual_risk_fraction"],
            "capacity_clipped_planned_fraction_distribution": distributions["capacity_clipped_initial_risk_fraction"],
            "old_1p5_campaign_r_in_initial_leg_r": distributions["fixed_tp_initial_leg_r_equivalent"],
            "old_0p2_campaign_r_in_initial_leg_r": distributions["fail_fast_initial_leg_r_equivalent"],
            "capacity_clipped_trade_count": audit["capacity_clipped_trade_count"],
            "legacy_max_executable_r_basis": audit["legacy_max_executable_r_basis"],
            "full_initial_risk_cap_validation": {
                "trade_count": stage5_r3["trade_count"],
                "capped_quantity_count": stage5_r3["execution_stats"].get(
                    "initial_quantity_capped_by_full_cost_campaign_risk", 0
                ),
                "rejected_by_cap_count": stage5_r3["execution_stats"].get(
                    "initial_reject_full_cost_campaign_risk_cap", 0
                ),
                "actual_fraction_distribution": r3_fraction,
                "over_budget_trade_count": sum(
                    row["initial_leg_actual_risk_fraction"] > 1.0 + 1e-9
                    for row in stage5_r3["r_basis_audit"]["rows"]
                ),
            },
        },
        "event_diagnostics": {
            "ignition_observations": diagnostics["ignition_observation_count"],
            "unique_ignition_events": diagnostics["unique_ignition_event_count"],
            "pullback_confirmed": stage2_result["funnel"]["pullback_confirmed"],
            "portfolio_fills": stage2_result["funnel"]["initial_fill_count"],
            "shadow_executable_pullbacks": diagnostics["shadow_executable_pullback_count"],
            "ignition_shadow_summary": diagnostics["ignition_summary"],
            "pullback_shadow_summary": diagnostics["pullback_summary"],
            "shadow_isolation": diagnostics["shadow_portfolio_isolation"],
            "score_summary": {
                key: {
                    "evaluated_count": value["evaluated_count"],
                    "spearman_rank_correlation": value["spearman_rank_correlation"],
                    "monotonic": value["expectancy_monotonic_non_decreasing"],
                    "predictive_status": value["predictive_status"],
                    "eligible_as_hard_filter": value["eligible_as_hard_filter"],
                }
                for key, value in score.items()
            },
        },
        "signal_age_and_stale_control": {
            "age_report": diagnostics["signal_age_report"],
            "validation_immediate_45m": _metrics(immediate_45),
            "stale_reject_count_immediate_45m": immediate_45["reject_reasons"].get("stale_ignition_event", 0),
            "delayed_recompression_only": _metrics(delayed),
            "immediate_plus_delayed": _metrics(combined_delayed),
            "conclusion": "predictive decay is visible beyond 60 minutes but sample is insufficient; 45 minutes was only a validation choice, and delayed recompression has no proven edge",
        },
        "fail_fast_counterfactual": {
            "frozen_30_trade_baseline": old_fail_fast,
            "conditional_20m_validation": {
                "portfolio_metrics": _metrics(conditional_validation),
                "counterfactual": conditional_cf,
            },
            "conditional_20m_historical_test": {
                "portfolio_metrics": _metrics(historical_test),
                "counterfactual": historical_cf,
            },
            "conclusion": "conditional fail-fast reduces false exits, but it cannot repair the negative historical-test entry edge",
        },
        "staged_experiments": {
            "stage_4_event_structure": {name: _metrics(result) for name, result in stage4["results"].items()},
            "stage_5_r_basis": {name: _metrics(result) for name, result in stage5["results"].items()},
            "stage_6_fail_fast": {name: _metrics(result) for name, result in stage6["results"].items()},
            "stage_7_revalidation": {name: _metrics(result) for name, result in stage7["results"].items()},
        },
        "early_regime_audit": {
            "metrics": _metrics(early),
            "zec_net_pnl": early_zec,
            "net_pnl_excluding_zec": float(early["net_pnl"]) - early_zec,
            "conclusion": "sample was insufficient and aggregate profit depended on ZEC; EARLY regime is not accepted as independent alpha",
        },
        "time_split_evidence": temporal,
        "historical_test_gate": {
            "classification": "historical_test_not_untouched_holdout",
            "metrics": _metrics(historical_test),
            "monthly": historical_test["monthly"],
            "passed": False,
            "failure_reasons": [
                "profit_factor_below_1",
                "negative_expectancy",
                "all_historical_test_months_negative",
                "cost_and_delay_resilience_not_worth_testing_after_base_gate_failure",
            ],
        },
        "walk_forward_status": {
            "requested_method": "expanding train-validation-test with purge and embargo",
            "completed_evidence": "chronological backcast, validation selection, and frozen historical test",
            "adaptive_monthly_walk_forward_completed": False,
            "reason": "the frozen candidate failed the first historical test gate; continuing parameter selection would use test feedback and violate the protocol",
            "untouched_final_holdout": False,
        },
        "stress_test_status": {
            "final_candidate_exists": False,
            "maker_entry": "not_run_base_edge_failed",
            "fixed_risk": "not_run_no_final_candidate",
            "fixed_initial_equity": "not_run_no_final_candidate",
            "extra_delay": "not_run_no_final_candidate",
            "higher_cost": "not_run_no_final_candidate",
            "path_aware_winner_exclusion": "not_run_no_final_candidate",
            "symbol_exclusion": "not_run_no_final_candidate",
            "month_exclusion": "not_run_no_final_candidate",
            "note": "stress tests are mandatory for a final candidate; no candidate survived the base historical-test gate",
        },
        "final_answers": {
            "1_old_1p5_campaign_r_equivalent": distributions["fixed_tp_initial_leg_r_equivalent"]["median"],
            "2_old_0p2_campaign_r_equivalent": distributions["fail_fast_initial_leg_r_equivalent"]["median"],
            "3_capacity_clipping_impact": "18 of 30 baseline trades were capacity clipped; planned fraction median fell to 0.295 and actual full-cost fraction ranged from 0.071 to 0.549",
            "4_correct_fail_fast_count": old_fail_fast["correct_early_exit_count"],
            "5_premature_fail_fast_count": old_fail_fast["premature_exit_count"],
            "6_post_exit_recovery_counts": {
                "0p5r": old_fail_fast["later_reached_0p5r_count"],
                "1r": old_fail_fast["later_reached_1r_count"],
                "2r": old_fail_fast["later_reached_2r_count"],
            },
            "7_signal_age_distribution": diagnostics["signal_age_report"],
            "8_age_decay_threshold": "performance visibly deteriorated above 60 minutes, but only six events were available and the threshold is not statistically proven",
            "9_delayed_recompression": "not proven; delayed-only validation PF was below 1 and the combined path did not improve immediate-only",
            "10_rank_predictive": False,
            "11_quality_predictive": False,
            "12_extreme_volume_ratio_worse": "not established; the relationship was non-monotonic and each decile was below the 30-event minimum",
            "13_breakout_atr_stable_range": "not established; profitable middle buckets were non-monotonic and under-sampled",
            "14_early_regime_alpha": "not proven; excluding ZEC changed the 16-trade EARLY sample from positive to negative",
            "15_first_pullback_independent_edge": False,
            "16_fail_fast_policy": "retain hard invalidation plus conditional 20-minute non-extension; do not claim it creates alpha",
            "17_continue_cmipr": False,
            "18_archive_cmipr": True,
        },
        "final_decision": {
            "decision": "archive_cmipr_as_unproven_alpha",
            "dry_run_recommended": False,
            "addon_runner_or_risk_increase_allowed": False,
            "reason": "173 chronological trades produced PF 0.777 and negative expectancy; only 3 of 12 months were positive, and the frozen historical test PF was 0.555",
        },
        "source_reports": REPORT_FILES,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the CMIPR Stage 2.5 gate and archival report")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--output", default="reports/cmipr_stage25_final_report.json")
    args = parser.parse_args()
    report = build_report(Path(args.report_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": report["final_decision"]["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

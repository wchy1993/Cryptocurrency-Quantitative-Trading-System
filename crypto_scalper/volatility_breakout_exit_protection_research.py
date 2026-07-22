from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .risk import BacktestExecutionConfig
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import (
    EXIT_PROTECTION_STRATEGY_NAME,
    EXIT_PROTECTION_VERSION,
    ExitProtectionConfig,
    simulate_exit_protected_portfolio,
)
from .volatility_breakout_optimize import (
    Candidate,
    PortfolioSearchConfig,
    _load_runtime_inputs,
    build_candidates,
    compact_summary,
    minute_token,
    sha256_file,
    simulate_portfolio,
)
from .volatility_breakout_v2_optimize import (
    build_market_context,
    enrich_candidates,
    filter_candidates,
)


BASELINE_NAME = "baseline_v3_max2_no_exit_protection"


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    config: ExitProtectionConfig


def _variant_key(config: ExitProtectionConfig) -> tuple[Any, ...]:
    return tuple(asdict(config).values())


def build_variants(matrix: dict[str, Any]) -> list[Variant]:
    variants = [Variant(BASELINE_NAME, "baseline", ExitProtectionConfig())]
    for trigger in matrix["breakeven_trigger_r"]:
        variants.append(
            Variant(
                f"breakeven_{trigger:g}r",
                "breakeven",
                ExitProtectionConfig(breakeven_trigger_r=float(trigger)),
            )
        )
    for activation in matrix["profit_giveback_activation_r"]:
        for giveback in matrix["profit_giveback_r"]:
            if float(giveback) >= float(activation):
                continue
            variants.append(
                Variant(
                    f"giveback_activate_{activation:g}r_allow_{giveback:g}r",
                    "profit_giveback",
                    ExitProtectionConfig(
                        profit_giveback_activation_r=float(activation),
                        profit_giveback_r=float(giveback),
                    ),
                )
            )
    for target in matrix["partial_take_profit_r"]:
        for fraction in matrix["partial_take_profit_fraction"]:
            for move_to_be in matrix["partial_move_to_breakeven"]:
                suffix = "move_be" if move_to_be else "runner_original_stop"
                variants.append(
                    Variant(
                        f"partial_{fraction:g}_at_{target:g}r_{suffix}",
                        "partial_take_profit",
                        ExitProtectionConfig(
                            partial_take_profit_r=float(target),
                            partial_take_profit_fraction=float(fraction),
                            move_stop_to_breakeven_after_partial=bool(move_to_be),
                        ),
                    )
                )
    for target in matrix["combined_partial_targets"]:
        for fraction in matrix["combined_partial_fractions"]:
            for activation, giveback in matrix["combined_giveback_pairs"]:
                variants.append(
                    Variant(
                        f"partial_{fraction:g}_at_{target:g}r_giveback_{activation:g}_{giveback:g}",
                        "partial_plus_giveback",
                        ExitProtectionConfig(
                            profit_giveback_activation_r=float(activation),
                            profit_giveback_r=float(giveback),
                            partial_take_profit_r=float(target),
                            partial_take_profit_fraction=float(fraction),
                        ),
                    )
                )

    unique: list[Variant] = []
    seen: set[tuple[Any, ...]] = set()
    for variant in variants:
        variant.config.validate()
        key = _variant_key(variant.config)
        if key in seen:
            continue
        seen.add(key)
        unique.append(variant)
    return unique


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_count",
        "trade_count",
        "partial_exit_count",
        "trades_with_partial_exit",
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
        "cash_reconciliation_error",
        "by_month",
        "by_side",
        "by_exit_reason",
    )
    return {key: result[key] for key in keys}


def _record(variant: Variant, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": variant.name,
        "family": variant.family,
        "exit_protection_config": asdict(variant.config),
        "result": _compact(result),
    }


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


def _shortlist(rows: list[dict[str, Any]], settings: dict[str, Any]) -> list[str]:
    selected = {BASELINE_NAME}
    families = sorted({row["family"] for row in rows if row["family"] != "baseline"})
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        selected.update(
            row["name"]
            for row in sorted(
                family_rows,
                key=lambda row: row["result"]["net_profit"],
                reverse=True,
            )[: int(settings["top_net_per_family"])]
        )
        selected.update(
            row["name"]
            for row in sorted(
                family_rows,
                key=lambda row: row["result"]["profit_factor"],
                reverse=True,
            )[: int(settings["top_profit_factor_per_family"])]
        )
        profitable = [row for row in family_rows if row["result"]["net_profit"] > 0.0]
        selected.update(
            row["name"]
            for row in sorted(
                profitable,
                key=lambda row: row["result"]["max_drawdown_pct"],
            )[: int(settings["lowest_drawdown_per_family"])]
        )
    return sorted(selected)


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[str]:
    protected = [row for row in rows if row["family"] != "baseline" and row["result"]["net_profit"] > 0.0]
    frontier: list[str] = []
    for row in protected:
        net = row["result"]["net_profit"]
        drawdown = row["result"]["max_drawdown_pct"]
        dominated = any(
            other["result"]["net_profit"] >= net
            and other["result"]["max_drawdown_pct"] <= drawdown
            and (
                other["result"]["net_profit"] > net
                or other["result"]["max_drawdown_pct"] < drawdown
            )
            for other in protected
            if other is not row
        )
        if not dominated:
            frontier.append(row["name"])
    return sorted(frontier)


def _split_score(split_rows: dict[str, dict[str, Any]], full: dict[str, Any]) -> tuple[Any, ...]:
    returns = [row["return_pct"] for row in split_rows.values()]
    positive = sum(value > 0.0 for value in returns)
    return (
        positive,
        min(returns),
        statistics.median(returns),
        full["profit_factor"],
        -full["max_drawdown_pct"],
        full["net_profit"],
    )


def _scale_execution(
    execution: BacktestExecutionConfig,
    multiplier: float,
) -> BacktestExecutionConfig:
    return replace(
        execution,
        maker_fee_rate=execution.maker_fee_rate * multiplier,
        taker_fee_rate=execution.taker_fee_rate * multiplier,
        market_slippage_bps=execution.market_slippage_bps * multiplier,
        stop_slippage_bps=execution.stop_slippage_bps * multiplier,
        take_profit_slippage_bps=execution.take_profit_slippage_bps * multiplier,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _format_pf(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.3f}"


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    baseline = report["baseline_result"]
    rows_by_name = {row["name"]: row for row in report["full_matrix"]}
    chosen_names = [
        BASELINE_NAME,
        report["selection"]["best_protected_net"],
        *report["selection"]["best_net_by_family"].values(),
        report["selection"]["split_stability_candidate"],
        report["selection"]["lowest_drawdown_profitable"],
    ]
    chosen_names = list(dict.fromkeys(chosen_names))
    lines = [
        "# Volatility Breakout v3 Exit-Protection Research",
        "",
        f"- Period: `{report['period']['start']}` to `{report['period']['end']}`",
        f"- Initial equity: `{report['initial_equity']:.2f}U`",
        f"- Variants: `{report['variant_count']}` independent exit overlays",
        f"- Same frozen 50-symbol entries with at most `{report['portfolio']['max_open_positions']}` open positions, "
        "next-1m execution, conservative stop-first path and full costs",
        "- Existing v2 code/config/shadow process was not modified",
        "- Historical research only; the monthly splits are diagnostics, not untouched holdout data",
        "",
        "| Variant | Family | Net | Return | PF | Win rate | Max DD | Top-5 contribution | Partials |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in chosen_names:
        row = rows_by_name[name]
        result = row["result"]
        lines.append(
            f"| {name} | {row['family']} | {result['net_profit']:+.2f}U | "
            f"{result['return_pct']:.2%} | {_format_pf(result['profit_factor'])} | "
            f"{result['win_rate']:.2%} | {result['max_drawdown_pct']:.2%} | "
            f"{result['top5_profit_contribution']:.2%} | {result['partial_exit_count']} |"
        )
    stable_name = report["selection"]["split_stability_candidate"]
    stable = rows_by_name[stable_name]["result"]
    lines.extend(
        [
            "",
            "## Key findings",
            "",
            f"- Baseline invariance: `{report['baseline_invariance']['trade_path_equal']}`; "
            f"net delta `{report['baseline_invariance']['net_profit_delta']:.12f}U`.",
            f"- Best protected net result: `{report['selection']['best_protected_net']}`.",
            f"- Split-stability candidate: `{stable_name}` with `{stable['net_profit']:+.2f}U`, "
            f"PF `{_format_pf(stable['profit_factor'])}`, DD `{stable['max_drawdown_pct']:.2%}`.",
            f"- Protected variant beats the no-protection baseline on net profit: "
            f"`{report['selection']['any_protected_beats_baseline_net']}`.",
            f"- Profitable net-vs-drawdown Pareto variants: `{len(report['selection']['pareto_frontier'])}`.",
            "",
            "The selected configuration is a research candidate only. It must be frozen as a separate "
            "shadow version and collect new data before any live consideration.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    period = config["period"]
    data = config["data"]
    args = argparse.Namespace(
        signal_data_dir=data["signal_data_dir"],
        execution_data_dir=data["execution_data_dir"],
        funding_data_dir=data["funding_data_dir"],
        cost_config=data["cost_config"],
        start=period["start"],
        end=period["end"],
    )
    start, end, signal_data, execution_data, rules, execution, metadata = _load_runtime_inputs(args)

    source_path = Path(config["source"]["optimized_config"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    symbols = tuple(source["symbols"])
    signal_config = VolatilityBreakoutConfig(**source["balanced_signal"])
    portfolio_values = dict(source["balanced_portfolio"])
    portfolio_values.update(config.get("portfolio_overrides", {}))
    portfolio_config = PortfolioSearchConfig(**portfolio_values)
    initial_equity = float(config["initial_equity"])

    print("building frozen v2 balanced candidates", flush=True)
    raw_candidates = build_candidates(
        symbols,
        signal_data,
        execution_data,
        signal_config,
        start,
        end,
    )
    candidates = filter_candidates(
        enrich_candidates(raw_candidates, build_market_context(symbols, signal_data)),
        signal_config,
    )

    print("checking disabled-overlay invariance", flush=True)
    original_baseline = simulate_portfolio(
        candidates,
        symbols,
        execution_data,
        rules,
        signal_config,
        portfolio_config,
        execution,
        start,
        end,
        initial_equity,
    )
    protected_baseline = simulate_exit_protected_portfolio(
        candidates,
        symbols,
        execution_data,
        rules,
        signal_config,
        portfolio_config,
        ExitProtectionConfig(),
        execution,
        start,
        end,
        initial_equity,
    )
    original_path = [
        (trade["event_id"], trade["exit_time"], trade["exit_reason"], trade["net_pnl"])
        for trade in original_baseline["trades"]
    ]
    protected_path = [
        (trade["event_id"], trade["exit_time"], trade["exit_reason"], trade["net_pnl"])
        for trade in protected_baseline["trades"]
    ]
    invariance = {
        "trade_path_equal": protected_path == original_path,
        "trade_count_equal": protected_baseline["trade_count"] == original_baseline["trade_count"],
        "net_profit_delta": protected_baseline["net_profit"] - original_baseline["net_profit"],
        "final_equity_delta": protected_baseline["final_equity"] - original_baseline["final_equity"],
    }
    if not invariance["trade_path_equal"] or abs(invariance["net_profit_delta"]) > 1e-8:
        raise RuntimeError(f"disabled v3 overlay does not reproduce frozen v2: {invariance}")

    variants = build_variants(config["matrix"])
    results_by_name: dict[str, dict[str, Any]] = {
        BASELINE_NAME: protected_baseline
    }
    full_rows = [_record(variants[0], protected_baseline)]
    print(f"running full-period matrix variants={len(variants)}", flush=True)
    for index, variant in enumerate(variants[1:], start=2):
        result = simulate_exit_protected_portfolio(
            candidates,
            symbols,
            execution_data,
            rules,
            signal_config,
            portfolio_config,
            variant.config,
            execution,
            start,
            end,
            initial_equity,
        )
        results_by_name[variant.name] = result
        full_rows.append(_record(variant, result))
        if index % 10 == 0 or index == len(variants):
            print(f"completed matrix {index}/{len(variants)}", flush=True)

    variant_by_name = {variant.name: variant for variant in variants}
    shortlist_names = _shortlist(full_rows, config["shortlist"])
    splits = {
        name: (datetime.fromisoformat(bounds[0]), datetime.fromisoformat(bounds[1]))
        for name, bounds in config["splits"].items()
    }
    split_results: dict[str, dict[str, Any]] = {}
    print(f"running monthly split diagnostics shortlisted={len(shortlist_names)}", flush=True)
    for index, name in enumerate(shortlist_names, start=1):
        variant = variant_by_name[name]
        split_results[name] = {}
        for split_name, (split_start, split_end) in splits.items():
            result = simulate_exit_protected_portfolio(
                _slice_candidates(candidates, split_start, split_end),
                symbols,
                execution_data,
                rules,
                signal_config,
                portfolio_config,
                variant.config,
                execution,
                split_start,
                split_end,
                initial_equity,
            )
            split_results[name][split_name] = _compact(result)
        if index % 5 == 0 or index == len(shortlist_names):
            print(f"completed split diagnostics {index}/{len(shortlist_names)}", flush=True)

    protected_rows = [row for row in full_rows if row["family"] != "baseline"]
    completed_profitable_rows = [
        row
        for row in protected_rows
        if row["result"]["net_profit"] > 0.0
        and not row["result"]["hard_drawdown_stopped"]
    ]
    best_protected_net = max(
        completed_profitable_rows,
        key=lambda row: row["result"]["net_profit"],
    )["name"]
    lowest_drawdown = min(
        completed_profitable_rows,
        key=lambda row: row["result"]["max_drawdown_pct"],
    )["name"]
    stable_pool = [
        name
        for name in shortlist_names
        if name != BASELINE_NAME
        and results_by_name[name]["net_profit"] > 0.0
        and not results_by_name[name]["hard_drawdown_stopped"]
    ]
    stable_name = max(
        stable_pool,
        key=lambda name: _split_score(split_results[name], results_by_name[name]),
    )
    best_net_by_family = {
        family: max(
            (row for row in protected_rows if row["family"] == family),
            key=lambda row: row["result"]["net_profit"],
        )["name"]
        for family in sorted({row["family"] for row in protected_rows})
    }
    selection = {
        "best_protected_net": best_protected_net,
        "recommended_shadow_candidate": best_protected_net,
        "split_stability_candidate": stable_name,
        "lowest_drawdown_profitable": lowest_drawdown,
        "best_net_by_family": best_net_by_family,
        "any_protected_beats_baseline_net": (
            results_by_name[best_protected_net]["net_profit"]
            > protected_baseline["net_profit"]
        ),
        "best_protected_net_delta_vs_baseline": (
            results_by_name[best_protected_net]["net_profit"]
            - protected_baseline["net_profit"]
        ),
        "pareto_frontier": _pareto_frontier(full_rows),
        "split_stability_rule": (
            "among profitable variants that complete the full period without the hard drawdown stop, "
            "maximize positive monthly splits, then worst split return, median split return, "
            "full-period PF, lower drawdown, and full-period net"
        ),
        "recommended_shadow_rule": (
            "highest full-period net profit among protected variants that complete the full period "
            "without the hard drawdown stop; research-only, not approved for live trading"
        ),
    }

    stress_names = list(dict.fromkeys([best_protected_net, stable_name, lowest_drawdown]))
    stress_results: dict[str, dict[str, Any]] = {}
    cost_multiplier = float(config["stress"]["cost_multiplier"])
    print(f"running finalist stress checks finalists={len(stress_names)}", flush=True)
    for name in stress_names:
        variant = variant_by_name[name]
        fixed = simulate_exit_protected_portfolio(
            candidates,
            symbols,
            execution_data,
            rules,
            signal_config,
            replace(portfolio_config, compound=False),
            variant.config,
            execution,
            start,
            end,
            initial_equity,
        )
        higher_cost = simulate_exit_protected_portfolio(
            candidates,
            symbols,
            execution_data,
            rules,
            signal_config,
            portfolio_config,
            variant.config,
            _scale_execution(execution, cost_multiplier),
            start,
            end,
            initial_equity,
        )
        stress_results[name] = {
            "fixed_risk_no_compounding": _compact(fixed),
            f"cost_{cost_multiplier:g}x": _compact(higher_cost),
        }

    report = {
        "strategy_name": EXIT_PROTECTION_STRATEGY_NAME,
        "strategy_version": EXIT_PROTECTION_VERSION,
        "research_status": "historical_matrix_not_untouched_oos_not_live",
        "source_strategy_modified": False,
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "splits": {
            name: {"start": bounds[0].isoformat(), "end": bounds[1].isoformat()}
            for name, bounds in splits.items()
        },
        "initial_equity": initial_equity,
        "symbols": list(symbols),
        "portfolio": asdict(portfolio_config),
        "portfolio_overrides": config.get("portfolio_overrides", {}),
        "source": {
            **config["source"],
            "optimized_config_sha256": sha256_file(source_path),
            "frozen_shadow_config_sha256": sha256_file(
                config["source"]["frozen_shadow_config"]
            ),
        },
        "cost_model": {
            "source_config": data["cost_config"],
            "source_config_sha256": sha256_file(data["cost_config"]),
            "mode": execution.mode,
            "market_slippage_bps": execution.market_slippage_bps,
            "stop_slippage_bps": execution.stop_slippage_bps,
            "take_profit_slippage_bps": execution.take_profit_slippage_bps,
            "taker_fee_rate": execution.taker_fee_rate,
            "funding_enabled": execution.funding_enabled,
            "funding_missing_symbols": sorted(set(symbols) - set(metadata["funding"])),
        },
        "execution_rules": {
            "entry": "next_1m_open_after_closed_60m_signal",
            "same_bar_conflict": "stop_before_full_target_before_partial_target",
            "new_protection_stop_effective": "following_1m_bar",
            "partial_exit": "single_take_profit_market_leg_then_runner",
            "source_entry_signal_parameters_unchanged": True,
            "portfolio_override": config.get("portfolio_overrides", {}),
        },
        "matrix_definition": config["matrix"],
        "variant_count": len(variants),
        "baseline_invariance": invariance,
        "baseline_result": compact_summary(original_baseline),
        "full_matrix": full_rows,
        "shortlisted_names": shortlist_names,
        "split_results": split_results,
        "selection": selection,
        "stress_results": stress_results,
        "selected_full_result": results_by_name[best_protected_net],
    }

    output = Path(config["output"]["report"])
    summary = Path(config["output"]["summary"])
    selected_path = Path(config["output"]["selected_config"])
    manifest = Path(config["output"]["manifest"])
    _write_json(output, report)
    _write_summary(summary, report)
    _write_json(
        selected_path,
        {
            "strategy_name": EXIT_PROTECTION_STRATEGY_NAME,
            "status": "historical_research_candidate_not_live",
            "source_entry_config": str(source_path),
            "source_entry_config_sha256": sha256_file(source_path),
            "symbols": list(symbols),
            "signal": signal_config.as_dict(),
            "portfolio": asdict(portfolio_config),
            "exit_protection": asdict(variant_by_name[best_protected_net].config),
            "selection_name": best_protected_net,
            "selection_rule": selection["recommended_shadow_rule"],
            "report": str(output),
        },
    )
    artifacts = (
        path,
        output,
        summary,
        selected_path,
        source_path,
        Path(config["source"]["frozen_shadow_config"]),
        Path("crypto_scalper/volatility_breakout_exit_protection.py"),
        Path("crypto_scalper/volatility_breakout_exit_protection_research.py"),
        Path("tests/test_volatility_breakout_exit_protection.py"),
    )
    _write_json(
        manifest,
        {
            "strategy_name": EXIT_PROTECTION_STRATEGY_NAME,
            "status": "historical_research_frozen_separately_from_v2",
            "source_strategy_modified": False,
            "report": str(output),
            "summary": str(summary),
            "selected_config": str(selected_path),
            "hashes": {
                str(artifact): sha256_file(artifact)
                for artifact in artifacts
                if artifact.exists()
            },
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research independent v3 exit protection for frozen v2 balanced entries"
    )
    parser.add_argument(
        "--config",
        default="config.volatility-breakout.v3-exit-protection-research.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_research(args.config)
    rows = {row["name"]: row["result"] for row in report["full_matrix"]}
    names = list(
        dict.fromkeys(
            [
                BASELINE_NAME,
                report["selection"]["best_protected_net"],
                report["selection"]["split_stability_candidate"],
                report["selection"]["lowest_drawdown_profitable"],
            ]
        )
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "variant_count": report["variant_count"],
                    "baseline_invariance": report["baseline_invariance"],
                    "selection": report["selection"],
                    "results": {name: rows[name] for name in names},
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

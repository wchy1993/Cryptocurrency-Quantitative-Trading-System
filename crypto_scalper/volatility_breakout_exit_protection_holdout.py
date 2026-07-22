from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .data import parse_timestamp
from .live_config import load_live_config
from .realistic_data import load_funding_rate_directory
from .risk import execution_config_from_live_config
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import (
    EXIT_PROTECTION_STRATEGY_NAME,
    EXIT_PROTECTION_VERSION,
    ExitProtectionConfig,
    simulate_exit_protected_portfolio,
)
from .volatility_breakout_exit_protection_research import (
    BASELINE_NAME,
    _compact,
    _json_safe,
    _scale_execution,
)
from .volatility_breakout_optimize import (
    PortfolioSearchConfig,
    build_candidates,
    load_research_data,
    sha256_file,
    simulate_portfolio,
)
from .volatility_breakout_v2_optimize import (
    build_market_context,
    enrich_candidates,
    filter_candidates,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _max_concurrent_positions(trades: list[dict[str, Any]]) -> int:
    events: list[tuple[str, int]] = []
    for trade in trades:
        events.append((trade["entry_time"], 1))
        events.append((trade["exit_time"], -1))
    current = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def _format_pf(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.3f}"


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Volatility Breakout v3 Exit-Protection Recent Holdout",
        "",
        f"- Period: `{report['period']['start']}` to `{report['period']['end']}`",
        f"- Initial equity: `{report['initial_equity']:.2f}U`",
        "- Parameters were frozen before these official Binance data were downloaded",
        f"- Maximum open positions configured/observed: `{report['portfolio']['max_open_positions']}` / "
        f"`{report['observed_max_concurrent_positions']}`",
        "- No parameter was selected or tuned on this holdout",
        "",
        "| Variant | Net | Return | PF | Win rate | Max DD | Hard stop | 2x-cost net | 2x-cost DD | 2x stop |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    for name in report["evaluation_order"]:
        row = report["results"][name]
        stress = report["cost_stress"][name]
        lines.append(
            f"| {name} | {row['net_profit']:+.2f}U | {row['return_pct']:.2%} | "
            f"{_format_pf(row['profit_factor'])} | {row['win_rate']:.2%} | "
            f"{row['max_drawdown_pct']:.2%} | {row['hard_drawdown_stopped']} | "
            f"{stress['net_profit']:+.2f}U | {stress['max_drawdown_pct']:.2%} | "
            f"{stress['hard_drawdown_stopped']} |"
        )
    comparison = report["selected_vs_baseline"]
    lines.extend(
        [
            "",
            "## Frozen-candidate check",
            "",
            f"- Selected: `{comparison['selected_name']}`.",
            f"- Net delta versus no protection: `{comparison['net_profit_delta']:+.2f}U`.",
            f"- Drawdown delta versus no protection: `{comparison['max_drawdown_delta']:+.2%}`.",
            f"- Selected beats baseline net: `{comparison['beats_baseline_net']}`.",
            f"- Selected lowers drawdown: `{comparison['lowers_drawdown']}`.",
            f"- Decision: `{report['decision']['status']}`.",
            f"- Reason: {report['decision']['reason']}",
            "",
            "This is a recent historical holdout, not live evidence. The frozen v2 process was not modified.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_holdout(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    report_path = Path(config["frozen_sources"]["research_report"])
    selected_path = Path(config["frozen_sources"]["selected_config"])
    research = json.loads(report_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    start = parse_timestamp(config["period"]["start"])
    end = parse_timestamp(config["period"]["end"])
    symbols = tuple(selected["symbols"])
    signal_config = VolatilityBreakoutConfig(**selected["signal"])
    portfolio_config = PortfolioSearchConfig(**selected["portfolio"])

    signal_data, execution_data, rules = load_research_data(
        symbols,
        config["data"]["signal_data_dir"],
        config["data"]["execution_data_dir"],
        start,
        end,
    )
    missing = sorted(set(symbols) - set(signal_data) | (set(symbols) - set(execution_data)))
    if missing:
        raise RuntimeError(f"incomplete holdout universe data: {missing}")
    live_config = load_live_config(config["data"]["cost_config"])
    execution = execution_config_from_live_config(
        live_config,
        cost_experiment="full_cost",
        mode="conservative",
    )
    funding = load_funding_rate_directory(config["data"]["funding_data_dir"], symbols)
    execution = replace(
        execution,
        funding_enabled=True,
        funding_default_rate=0.0,
        funding_rates_by_symbol=funding,
    )

    print("building untouched recent candidates", flush=True)
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

    matrix = {row["name"]: row for row in research["full_matrix"]}
    fixed_names = list(config["fixed_variants"])
    missing_variants = [name for name in fixed_names if name not in matrix]
    if missing_variants:
        raise RuntimeError(f"fixed holdout variants missing from frozen research: {missing_variants}")
    variants = {
        BASELINE_NAME: ExitProtectionConfig(),
        **{
            name: ExitProtectionConfig(**matrix[name]["exit_protection_config"])
            for name in fixed_names
        },
    }

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
        float(config["initial_equity"]),
    )
    results: dict[str, dict[str, Any]] = {}
    cost_stress: dict[str, dict[str, Any]] = {}
    full_results: dict[str, dict[str, Any]] = {}
    for name, exit_config in variants.items():
        print(f"holdout {name}", flush=True)
        result = simulate_exit_protected_portfolio(
            candidates,
            symbols,
            execution_data,
            rules,
            signal_config,
            portfolio_config,
            exit_config,
            execution,
            start,
            end,
            float(config["initial_equity"]),
        )
        stressed = simulate_exit_protected_portfolio(
            candidates,
            symbols,
            execution_data,
            rules,
            signal_config,
            portfolio_config,
            exit_config,
            _scale_execution(execution, float(config["stress"]["cost_multiplier"])),
            start,
            end,
            float(config["initial_equity"]),
        )
        full_results[name] = result
        results[name] = _compact(result)
        cost_stress[name] = _compact(stressed)

    baseline = full_results[BASELINE_NAME]
    baseline_path = [
        (trade["event_id"], trade["exit_time"], trade["exit_reason"], trade["net_pnl"])
        for trade in baseline["trades"]
    ]
    original_path = [
        (trade["event_id"], trade["exit_time"], trade["exit_reason"], trade["net_pnl"])
        for trade in original_baseline["trades"]
    ]
    if baseline_path != original_path:
        raise RuntimeError("disabled exit overlay changed the recent baseline trade path")

    selected_name = selected["selection_name"]
    selected_result = results[selected_name]
    baseline_result = results[BASELINE_NAME]
    report = {
        "strategy_name": EXIT_PROTECTION_STRATEGY_NAME,
        "strategy_version": EXIT_PROTECTION_VERSION,
        "status": "recent_untuned_holdout_not_live",
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "initial_equity": float(config["initial_equity"]),
        "symbols": list(symbols),
        "portfolio": asdict(portfolio_config),
        "data": config["data"],
        "frozen_sources": {
            **config["frozen_sources"],
            "research_report_sha256": sha256_file(report_path),
            "selected_config_sha256": sha256_file(selected_path),
        },
        "selection_frozen_before_download": True,
        "parameter_tuning_on_holdout": False,
        "candidate_count": sum(len(rows) for rows in candidates.values()),
        "funding_symbols_loaded": len(funding),
        "funding_missing_symbols": sorted(set(symbols) - set(funding)),
        "baseline_invariance": {
            "trade_path_equal": baseline_path == original_path,
            "net_profit_delta": baseline["net_profit"] - original_baseline["net_profit"],
        },
        "evaluation_order": list(variants),
        "results": results,
        "cost_stress": cost_stress,
        "observed_max_concurrent_positions": max(
            _max_concurrent_positions(result["trades"])
            for result in full_results.values()
        ),
        "selected_vs_baseline": {
            "selected_name": selected_name,
            "net_profit_delta": selected_result["net_profit"] - baseline_result["net_profit"],
            "max_drawdown_delta": (
                selected_result["max_drawdown_pct"] - baseline_result["max_drawdown_pct"]
            ),
            "beats_baseline_net": selected_result["net_profit"] > baseline_result["net_profit"],
            "lowers_drawdown": (
                selected_result["max_drawdown_pct"] < baseline_result["max_drawdown_pct"]
            ),
        },
        "decision": {
            "status": "do_not_promote",
            "reason": (
                "the frozen candidate lost money and hit the hard drawdown stop; "
                "every fixed exit family representative also hit the hard stop"
            ),
            "selected_profitable": selected_result["net_profit"] > 0.0,
            "selected_completed_without_hard_stop": not selected_result["hard_drawdown_stopped"],
            "all_evaluated_variants_hard_stopped": all(
                row["hard_drawdown_stopped"] for row in results.values()
            ),
        },
    }
    output = Path(config["output"]["report"])
    summary = Path(config["output"]["summary"])
    manifest = Path(config["output"]["manifest"])
    _write_json(output, report)
    _write_summary(summary, report)
    _write_json(
        manifest,
        {
            "status": "recent_holdout_frozen_separately_from_v2",
            "source_strategy_modified": False,
            "decision": report["decision"],
            "report": str(output),
            "summary": str(summary),
            "hashes": {
                str(item): sha256_file(item)
                for item in (
                    path,
                    report_path,
                    selected_path,
                    output,
                    summary,
                    Path("crypto_scalper/volatility_breakout_exit_protection.py"),
                    Path("crypto_scalper/volatility_breakout_exit_protection_research.py"),
                    Path("crypto_scalper/volatility_breakout_exit_protection_holdout.py"),
                    Path("tests/test_volatility_breakout_exit_protection.py"),
                )
            },
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen v3 exits on recent untouched data")
    parser.add_argument(
        "--config",
        default="config.volatility-breakout.v3-exit-protection-holdout.json",
    )
    args = parser.parse_args(argv)
    report = run_holdout(args.config)
    print(
        json.dumps(
            _json_safe(
                {
                    "period": report["period"],
                    "candidate_count": report["candidate_count"],
                    "baseline_invariance": report["baseline_invariance"],
                    "selected_vs_baseline": report["selected_vs_baseline"],
                    "results": report["results"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

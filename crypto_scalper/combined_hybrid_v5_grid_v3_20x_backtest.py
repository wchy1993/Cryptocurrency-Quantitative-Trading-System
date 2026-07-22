from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .combined_hybrid_v5_grid_v3_backtest import (
    _run_period,
    _write_json,
    build_frozen_configs,
)
from .combined_volatility_trend_grid_backtest import BREAKOUT_KEY, GRID_KEY
from .volatility_breakout_optimize import UNIVERSE_50, sha256_file
from .volatility_breakout_v4_research import load_v4_runtime_inputs


COMBINED_V5_GRID_V3_20X_NAME = (
    "hybrid_v5_balanced_expansion_runner_plus_dynamic_trend_grid_v3_"
    "max2_20x_capacity"
)
NOTIONAL_CAP_MULTIPLE = 20.0


def build_20x_configs(
    breakout_path: str | Path,
    grid_path: str | Path,
) -> dict[str, Any]:
    """Clone the frozen strategies and change capacity limits only.

    Risk budgets, signals, exits, compounding, strategy concurrency and the
    shared max-two rule are deliberately inherited unchanged.
    """

    configs = build_frozen_configs(breakout_path, grid_path)
    configs["breakout_portfolio"] = replace(
        configs["breakout_portfolio"],
        max_notional_multiple=NOTIONAL_CAP_MULTIPLE,
    )
    configs["grid_portfolio"] = replace(
        configs["grid_portfolio"],
        max_notional_multiple=NOTIONAL_CAP_MULTIPLE,
    )
    configs["combined"] = replace(
        configs["combined"],
        max_gross_notional_multiple=NOTIONAL_CAP_MULTIPLE,
    )
    configs["combined"].validate()
    return configs


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "trade_count",
        "final_equity",
        "net_profit",
        "return_pct",
        "profit_factor",
        "win_rate",
        "max_drawdown_pct",
        "max_entry_committed_notional_multiple",
        "max_concurrent_positions",
        "full_cost",
        "hard_drawdown_stopped",
    )
    return {key: metrics.get(key) for key in keys}


def _comparison(
    baseline: dict[str, Any],
    periods: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for period_name, period in periods.items():
        old = baseline["periods"][period_name]["combined"]
        new = period["combined"]
        output[period_name] = {
            "baseline_9x": _metric_subset(old),
            "capacity_20x": _metric_subset(new),
            "delta": {
                "trade_count": new["trade_count"] - old["trade_count"],
                "net_profit": new["net_profit"] - old["net_profit"],
                "return_pct_points": new["return_pct"] - old["return_pct"],
                "profit_factor": new["profit_factor"] - old["profit_factor"],
                "win_rate_points": new["win_rate"] - old["win_rate"],
                "max_drawdown_pct_points": (
                    new["max_drawdown_pct"] - old["max_drawdown_pct"]
                ),
            },
        }
    return output


def _load_matching_baseline(
    path: str | Path,
    start_3m: datetime,
    start_6m: datetime,
    end: datetime,
) -> dict[str, Any]:
    source = Path(path)
    baseline = json.loads(source.read_text(encoding="utf-8"))
    expected = {
        "three_month": (start_3m.isoformat(), end.isoformat()),
        "six_month": (start_6m.isoformat(), end.isoformat()),
    }
    for period_name, (expected_start, expected_end) in expected.items():
        actual = baseline["periods"][period_name]["period"]
        if actual != {"start": expected_start, "end": expected_end}:
            raise RuntimeError(
                f"baseline period mismatch for {period_name}: {actual!r}"
            )
    return baseline


def run(args: argparse.Namespace) -> dict[str, Any]:
    end = datetime.fromisoformat(args.end)
    start_3m = datetime.fromisoformat(args.start_3m)
    start_6m = datetime.fromisoformat(args.start_6m)
    if not start_6m < start_3m < end:
        raise ValueError("periods must satisfy start_6m < start_3m < end")

    symbols = tuple(UNIVERSE_50)
    data_start = start_6m - timedelta(days=args.warmup_days)
    signal_data, execution_data, rules, execution, metadata = load_v4_runtime_inputs(
        symbols,
        args.one_minute_roots,
        args.funding_roots,
        args.cost_config,
        data_start,
        end,
    )
    if metadata["minimum_coverage_ratio"] < 0.999999:
        raise RuntimeError("20x backtest requires gap-free stitched 1m data")
    if metadata["maximum_missing_minutes"]:
        raise RuntimeError("20x backtest requires gap-free stitched 1m data")
    if metadata["funding_missing_symbols"]:
        raise RuntimeError("20x backtest requires complete funding data")

    configs = build_20x_configs(args.breakout_config, args.grid_config)
    periods = {
        "three_month": _run_period(
            "3m 20x",
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
            "6m 20x",
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

    baseline = _load_matching_baseline(
        args.baseline_report,
        start_3m,
        start_6m,
        end,
    )
    comparisons = _comparison(baseline, periods)
    report = {
        "strategy_name": COMBINED_V5_GRID_V3_20X_NAME,
        "status": "independent_20x_capacity_research_gui_unchanged",
        "initial_equity": args.initial_equity,
        "universe_size": len(symbols),
        "symbols": list(symbols),
        "leverage_interpretation": {
            "tested_change": "maximum notional-to-equity capacity",
            "breakout_max_notional_multiple": NOTIONAL_CAP_MULTIPLE,
            "grid_max_notional_multiple": NOTIONAL_CAP_MULTIPLE,
            "combined_max_gross_notional_multiple": NOTIONAL_CAP_MULTIPLE,
            "risk_budgets": "unchanged",
            "note": (
                "Changing exchange leverage alone would not change PnL when "
                "notional size is unchanged; this run tests usable 20x capacity."
            ),
        },
        "data_period": {
            "warmup_start": data_start.isoformat(),
            "end": end.isoformat(),
        },
        "data_quality": metadata,
        "source_configs": {
            BREAKOUT_KEY: args.breakout_config,
            GRID_KEY: args.grid_config,
        },
        "frozen_configs": {
            "breakout_signal": configs["breakout_signal"].as_dict(),
            "breakout_portfolio": asdict(configs["breakout_portfolio"]),
            "breakout_exit": asdict(configs["breakout_exit"]),
            "grid_signal": configs["grid_signal"].as_dict(),
            "grid_market_overlay": asdict(configs["grid_overlay"]),
            "grid_portfolio": asdict(configs["grid_portfolio"]),
            "combined_portfolio": asdict(configs["combined"]),
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
        },
        "periods": periods,
        "baseline": {
            "report": args.baseline_report,
            "sha256": sha256_file(Path(args.baseline_report)),
            "breakout_max_notional_multiple": baseline["frozen_configs"]
            ["breakout_portfolio"]["max_notional_multiple"],
            "grid_max_notional_multiple": baseline["frozen_configs"]
            ["grid_portfolio"]["max_notional_multiple"],
            "combined_max_gross_notional_multiple": baseline["frozen_configs"]
            ["combined_portfolio"]["max_gross_notional_multiple"],
        },
        "comparison": comparisons,
        "preserved": {
            "active_gui": "unchanged",
            "hybrid_v5_source": "unchanged",
            "grid_v3_source": "unchanged",
            "existing_combined_backtest": "unchanged",
            "apt_grid": "unchanged",
        },
    }
    _write_json(args.output, report)

    config_payload = {
        "strategy_name": COMBINED_V5_GRID_V3_20X_NAME,
        "status": "historical_research_not_live_gui_unchanged",
        "initial_equity": args.initial_equity,
        "notional_capacity": report["leverage_interpretation"],
        "source_configs": report["source_configs"],
        "breakout_portfolio": asdict(configs["breakout_portfolio"]),
        "grid_portfolio": asdict(configs["grid_portfolio"]),
        "combined_portfolio": asdict(configs["combined"]),
        "execution_rules": report["execution_rules"],
        "results": {
            period_name: period["combined"]
            for period_name, period in periods.items()
        },
    }
    _write_json(args.config_output, config_payload)

    lines = [
        "# Hybrid v5 Breakout + Grid v3 Max2 — 20x Capacity Backtest",
        "",
        f"- Initial equity: `{args.initial_equity:.2f}U`",
        "- 20x means maximum usable notional/equity capacity, not a forced fixed-size position",
        "- Signal rules, stop distances and risk budgets are unchanged",
        "- 50 symbols; each strategy max one position, shared account max two",
        "- Gap-free 1m execution; full fees, slippage and funding; adverse stop first",
        "- Existing GUI and frozen strategies remain unchanged",
        "",
        "| Period | Version | Trades | Net | PF | Win rate | Max DD | Peak entry exposure |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for period_name, comparison in comparisons.items():
        label = "3 months" if period_name == "three_month" else "6 months"
        for version, metrics in (
            ("Current caps (9x/5x/global 9x)", comparison["baseline_9x"]),
            ("20x capacity", comparison["capacity_20x"]),
        ):
            lines.append(
                f"| {label} | {version} | {metrics['trade_count']} | "
                f"{metrics['net_profit']:+.2f}U | {metrics['profit_factor']:.3f} | "
                f"{metrics['win_rate']:.2%} | {metrics['max_drawdown_pct']:.2%} | "
                f"{metrics['max_entry_committed_notional_multiple']:.3f}x |"
            )
    lines.extend(["", "## 20x standalone sleeves", ""])
    for period_name, period in periods.items():
        label = "3 months" if period_name == "three_month" else "6 months"
        for sleeve, metrics in (
            ("Hybrid v5", period["standalone"][BREAKOUT_KEY]),
            ("Grid v3", period["standalone"][GRID_KEY]),
        ):
            lines.append(
                f"- {label} {sleeve}: `{metrics['trade_count']}` trades, "
                f"`{metrics['net_profit']:+.2f}U`, PF `{metrics['profit_factor']:.3f}`, "
                f"win `{metrics['win_rate']:.2%}`, DD `{metrics['max_drawdown_pct']:.2%}`."
            )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "strategy_name": COMBINED_V5_GRID_V3_20X_NAME,
        "status": "independent_20x_capacity_research_artifacts",
        "config": args.config_output,
        "report": args.output,
        "summary": args.summary,
        "hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                "crypto_scalper/combined_hybrid_v5_grid_v3_20x_backtest.py",
                "tests/test_combined_hybrid_v5_grid_v3_20x_backtest.py",
                args.config_output,
                args.output,
                args.summary,
            )
        },
        "preserved": report["preserved"],
    }
    _write_json(args.manifest, manifest)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent Hybrid v5 plus Grid v3 max-two backtest with 20x "
            "notional capacity"
        )
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
        "--baseline-report",
        default="reports/combined_hybrid_v5_grid_v3_max2_3m_6m.json",
    )
    parser.add_argument(
        "--output",
        default="reports/combined_hybrid_v5_grid_v3_max2_20x_3m_6m.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/combined_hybrid_v5_grid_v3_max2_20x_3m_6m.md",
    )
    parser.add_argument(
        "--config-output",
        default="config.combined-hybrid-v5-grid-v3-max2-20x.json",
    )
    parser.add_argument(
        "--manifest",
        default="config.combined-hybrid-v5-grid-v3-max2-20x-manifest.json",
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .models import Candle
from .risk import BacktestExecutionConfig
from .trend_grid import TREND_GRID_STRATEGY_NAME, TREND_GRID_VERSION, TrendGridConfig
from .trend_grid_optimize import (
    GridCandidate,
    GridPortfolioConfig,
    TrendGridSnapshot,
    _attach_split_results,
    _load_runtime_inputs,
    _public_row,
    _run_stage,
    _scaled_execution,
    _selection_score,
    _shift_candidates,
    _signal_cache_key,
    _write_json,
    compact_grid_summary,
    sha256_file,
    simulate_grid_portfolio,
)
from .volatility_breakout_optimize import UNIVERSE_30, UNIVERSE_50, CompactSeries


TREND_GRID_V2_RESEARCH_NAME = "dynamic_trend_following_grid_v2"


def _historical_select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: _selection_score(row, historical_max=True))


def refine_grid_universe(
    universe: tuple[str, ...],
    frozen_signal: TrendGridConfig,
    frozen_portfolio: GridPortfolioConfig,
    signal_data: dict[str, list[Candle]],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, Any],
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
) -> dict[str, Any]:
    periods = {
        "train": (start, start + timedelta(days=31)),
        "validation": (start + timedelta(days=31), start + timedelta(days=61)),
        "test": (start + timedelta(days=61), end),
        "full": (start, end),
    }
    timeline_cache: dict[
        str,
        tuple[dict[int, list[GridCandidate]], dict[str, dict[int, TrendGridSnapshot]]],
    ] = {}
    stages: dict[str, list[dict[str, Any]]] = {}
    alpha_portfolio = replace(
        frozen_portfolio,
        risk_per_campaign_pct=0.02,
        max_campaign_risk_pct=0.10,
        max_open_campaigns=1,
        compound=True,
    )

    volume_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for minimum, maximum in (
        (0.0, 999.0),
        (0.4, 999.0),
        (0.6, 999.0),
        (0.8, 999.0),
        (1.0, 999.0),
        (1.2, 999.0),
        (1.5, 999.0),
        (0.6, 1.2),
        (0.6, 1.5),
        (0.8, 1.2),
        (0.8, 1.5),
        (1.5, 3.0),
    ):
        signal = replace(frozen_signal, min_volume_ratio=minimum, max_volume_ratio=maximum)
        volume_variants.append((f"volume_{minimum:.1f}_{maximum:g}", signal, alpha_portfolio))
    stage6a = _run_stage(
        "stage6a_volume",
        volume_variants,
        universe,
        signal_data,
        execution_data,
        rules,
        execution,
        periods,
        initial_equity,
        timeline_cache,
    )
    stages["stage6a_volume"] = [_public_row(row) for row in stage6a]
    selected_signal = TrendGridConfig(**_historical_select(stage6a)["signal_config"])

    trend_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for alignment in (0.0, 0.25, 0.50, 0.75, 1.0, 1.25):
        for slope in (0.0, 0.02, 0.05, 0.10):
            signal = replace(
                selected_signal,
                min_alignment_atr=alignment,
                min_fast_slope_atr=slope,
            )
            trend_variants.append((f"alignment{alignment:.2f}_slope{slope:.2f}", signal, alpha_portfolio))
    stage6b = _run_stage(
        "stage6b_trend_quality",
        trend_variants,
        universe,
        signal_data,
        execution_data,
        rules,
        execution,
        periods,
        initial_equity,
        timeline_cache,
    )
    stages["stage6b_trend_quality"] = [_public_row(row) for row in stage6b]
    selected_signal = TrendGridConfig(**_historical_select(stage6b)["signal_config"])

    inventory_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for cycles in (1, 2, 3):
        for total_entries in (2, 3, 4, 5, 7, 0):
            for pause in (False, True):
                signal = replace(
                    selected_signal,
                    max_cycles_per_level=cycles,
                    max_total_entries=total_entries,
                    pause_new_fills_on_fast_breach=pause,
                )
                inventory_variants.append(
                    (f"cycles{cycles}_entries{total_entries}_pause{int(pause)}", signal, alpha_portfolio)
                )
    stage7 = _run_stage(
        "stage7_inventory",
        inventory_variants,
        universe,
        signal_data,
        execution_data,
        rules,
        execution,
        periods,
        initial_equity,
        timeline_cache,
    )
    stages["stage7_inventory"] = [_public_row(row) for row in stage7]
    selected_signal = TrendGridConfig(**_historical_select(stage7)["signal_config"])

    protection_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for loss_limit in (0.0, 0.20, 0.35, 0.50, 0.75):
        for take_profit in (0.0, 0.15, 0.25, 0.40):
            signal = replace(
                selected_signal,
                campaign_loss_limit_r=loss_limit,
                campaign_take_profit_r=take_profit,
                profit_lock_activation_r=0.0,
                profit_giveback_r=0.0,
            )
            protection_variants.append((f"loss{loss_limit:.2f}_tp{take_profit:.2f}", signal, alpha_portfolio))
    for activation, giveback in ((0.15, 0.10), (0.25, 0.15), (0.40, 0.20), (0.50, 0.25)):
        signal = replace(
            selected_signal,
            campaign_take_profit_r=0.0,
            profit_lock_activation_r=activation,
            profit_giveback_r=giveback,
        )
        protection_variants.append((f"lock{activation:.2f}_giveback{giveback:.2f}", signal, alpha_portfolio))
    stage8 = _run_stage(
        "stage8_campaign_protection",
        protection_variants,
        universe,
        signal_data,
        execution_data,
        rules,
        execution,
        periods,
        initial_equity,
        timeline_cache,
    )
    stages["stage8_campaign_protection"] = [_public_row(row) for row in stage8]
    selected_signal = TrendGridConfig(**_historical_select(stage8)["signal_config"])

    geometry_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for spacing in (0.55, 0.70, 0.85, 1.0):
        for levels in (2, 3, 4):
            for target in (0.75, 1.0, 1.25, 1.50):
                minimum_stop = spacing * levels + 0.25
                signal = replace(
                    selected_signal,
                    grid_spacing_atr=spacing,
                    grid_levels=levels,
                    grid_target_spacing=target,
                    hard_stop_atr_multiple=max(selected_signal.hard_stop_atr_multiple, minimum_stop),
                )
                geometry_variants.append((f"space{spacing:.2f}_levels{levels}_tp{target:.2f}", signal, alpha_portfolio))
    stage9 = _run_stage(
        "stage9_local_geometry",
        geometry_variants,
        universe,
        signal_data,
        execution_data,
        rules,
        execution,
        periods,
        initial_equity,
        timeline_cache,
    )
    stages["stage9_local_geometry"] = [_public_row(row) for row in stage9]
    selected_signal = TrendGridConfig(**_historical_select(stage9)["signal_config"])

    direction_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for side in ("both", "long", "short"):
        for extension in (0.80, 1.20, 1.60, 2.0):
            signal = replace(
                selected_signal,
                allow_long=side != "short",
                allow_short=side != "long",
                max_entry_extension_atr=extension,
            )
            direction_variants.append((f"{side}_extension{extension:.2f}", signal, alpha_portfolio))
    stage10 = _run_stage(
        "stage10_direction",
        direction_variants,
        universe,
        signal_data,
        execution_data,
        rules,
        execution,
        periods,
        initial_equity,
        timeline_cache,
    )
    stages["stage10_direction"] = [_public_row(row) for row in stage10]
    selected_signal = TrendGridConfig(**_historical_select(stage10)["signal_config"])

    risk_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for risk in (0.02, 0.04, 0.06, 0.08, 0.10):
        for campaigns in (1, 2, 3):
            portfolio = replace(
                frozen_portfolio,
                risk_per_campaign_pct=risk,
                max_campaign_risk_pct=0.10,
                max_open_campaigns=campaigns,
                max_daily_campaigns=12,
                hard_drawdown_stop_pct=0.70,
            )
            risk_variants.append((f"risk{risk:.2f}_campaigns{campaigns}", selected_signal, portfolio))
    stage11 = _run_stage(
        "stage11_risk",
        risk_variants,
        universe,
        signal_data,
        execution_data,
        rules,
        execution,
        periods,
        initial_equity,
        timeline_cache,
    )
    split_candidates = sorted(
        stage11,
        key=lambda row: _selection_score(row, historical_max=True),
        reverse=True,
    )[:5]
    _attach_split_results(
        split_candidates,
        universe,
        execution_data,
        rules,
        execution,
        periods,
        initial_equity,
        timeline_cache,
    )
    stages["stage11_risk"] = [_public_row(row) for row in stage11]
    historical_selected = _historical_select(stage11)
    validation_selected = max(split_candidates, key=lambda row: _selection_score(row, historical_max=False))

    selected_signal = TrendGridConfig(**historical_selected["signal_config"])
    selected_portfolio = GridPortfolioConfig(**historical_selected["portfolio_config"])
    candidates, snapshots = timeline_cache[_signal_cache_key(selected_signal)]
    final_result = historical_selected["_full_result"]
    top_symbol = max(
        final_result["by_symbol"],
        key=lambda symbol: final_result["by_symbol"][symbol]["net_pnl"],
        default="",
    )
    stress = {
        "fixed_risk_no_compounding": compact_grid_summary(
            simulate_grid_portfolio(
                candidates, snapshots, universe, execution_data, rules, selected_signal,
                replace(selected_portfolio, compound=False), execution, start, end, initial_equity,
            )
        ),
        "entry_delay_1m": compact_grid_summary(
            simulate_grid_portfolio(
                _shift_candidates(candidates, 1, execution_data), snapshots, universe,
                execution_data, rules, selected_signal, selected_portfolio, execution,
                start, end, initial_equity,
            )
        ),
        "cost_1.5x": compact_grid_summary(
            simulate_grid_portfolio(
                candidates, snapshots, universe, execution_data, rules, selected_signal,
                selected_portfolio, _scaled_execution(execution, 1.5), start, end, initial_equity,
            )
        ),
    }
    if top_symbol:
        stress["exclude_top_symbol"] = compact_grid_summary(
            simulate_grid_portfolio(
                candidates, snapshots, universe, execution_data, rules, selected_signal,
                selected_portfolio, execution, start, end, initial_equity,
                skip_symbols=frozenset({top_symbol}),
            )
        )
        stress["exclude_top_symbol"]["excluded_symbol"] = top_symbol

    return {
        "universe_size": len(universe),
        "symbols": list(universe),
        "periods": {key: {"start": value[0].isoformat(), "end": value[1].isoformat()} for key, value in periods.items()},
        "selected_signal_config": selected_signal.as_dict(),
        "selected_portfolio_config": asdict(selected_portfolio),
        "validation_selected_signal_config": validation_selected["signal_config"],
        "validation_selected_portfolio_config": validation_selected["portfolio_config"],
        "stages": stages,
        "final_result": final_result,
        "historical_test_result": historical_selected["test"],
        "validation_selected_historical_test": validation_selected["test"],
        "stress_tests": stress,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine the independent full-cost trend grid")
    parser.add_argument("--signal-data-dir", default="data/binance_15m_365d_top100")
    parser.add_argument("--execution-data-dir", default="data/binance_1m_365d_top100")
    parser.add_argument("--funding-data-dir", default="data/binance_funding_365d_top100")
    parser.add_argument("--cost-config", default="config.gui.mtf-momentum-reset-stage21.json")
    parser.add_argument("--start", default="2026-03-06T00:00:00")
    parser.add_argument("--end", default="2026-06-06T00:00:00")
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument("--baseline-report", default="reports/trend_grid_3m_30_vs_50.json")
    parser.add_argument("--output", default="reports/trend_grid_v2_3m_30_vs_50.json")
    parser.add_argument("--summary", default="reports/trend_grid_v2_3m_30_vs_50.md")
    parser.add_argument("--config30", default="config.trend-grid.v2-optimized-30.json")
    parser.add_argument("--config50", default="config.trend-grid.v2-optimized-50.json")
    parser.add_argument("--manifest", default="config.trend-grid.v2-3m-manifest.json")
    return parser


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Dynamic Trend-Following Grid v2 - 3 Month Full-Cost Research",
        "",
        f"- Period: `{report['period']['start']}` to `{report['period']['end']}`",
        f"- Initial equity: `{report['initial_equity']:.2f}U`",
        "- Closed 60m trend state, next 1m open, conservative stop-first path",
        "- Maker grid fills require one-tick trade-through; forced exits use full taker cost and actual funding",
        "- Historical maximum is in-sample research, not a live guarantee or untouched holdout",
        "",
        "| Universe | Campaigns | Grid fills | Net | Return | PF | Win rate | Max DD | Historical test |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in ("30", "50"):
        universe = report["universes"][size]
        result = universe["final_result"]
        test = universe["historical_test_result"]
        lines.append(
            f"| {size} | {result['trade_count']} | {result['grid_entry_count']} | "
            f"{result['net_profit']:+.2f}U | {result['return_pct']:.2%} | "
            f"{result['profit_factor']:.3f} | {result['win_rate']:.2%} | "
            f"{result['max_drawdown_pct']:.2%} | {test['net_profit']:+.2f}U / PF {test['profit_factor']:.3f} |"
        )
    for size in ("30", "50"):
        universe = report["universes"][size]
        signal = universe["selected_signal_config"]
        portfolio = universe["selected_portfolio_config"]
        lines.extend(
            [
                "",
                f"## {size} Symbols",
                "",
                f"- Direction: long=`{signal['allow_long']}`, short=`{signal['allow_short']}`",
                f"- Trend: `{signal['timeframe_minutes']}m` EMA `{signal['fast_ema_period']}/{signal['slow_ema_period']}`, entry `{signal['entry_mode']}`",
                f"- Grid: spacing `{signal['grid_spacing_atr']} ATR`, levels `{signal['grid_levels']}`, target `{signal['grid_target_spacing']} spacing`, cycles `{signal['max_cycles_per_level']}`",
                f"- Protection: loss limit `{signal['campaign_loss_limit_r']}R`, campaign TP `{signal['campaign_take_profit_r']}R`, EMA exit `{signal['regime_exit_mode']}`",
                f"- Risk: `{portfolio['risk_per_campaign_pct']:.2%}` per campaign, max campaigns `{portfolio['max_open_campaigns']}`",
            ]
        )
        for stress_name, stress in universe["stress_tests"].items():
            lines.append(
                f"- Stress `{stress_name}`: `{stress['net_profit']:+.2f}U`, PF `{(stress['profit_factor'] or 0.0):.3f}`, DD `{stress['max_drawdown_pct']:.2%}`"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_refinement(args: argparse.Namespace) -> dict[str, Any]:
    start, end, signal_data, execution_data, rules, execution, metadata = _load_runtime_inputs(args)
    baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    universes: dict[str, Any] = {}
    for size, universe in (("30", UNIVERSE_30), ("50", UNIVERSE_50)):
        frozen = baseline["universes"][size]
        print(f"starting trend-grid v2 refinement for {size} symbols", flush=True)
        universes[size] = refine_grid_universe(
            universe,
            TrendGridConfig(**frozen["selected_signal_config"]),
            GridPortfolioConfig(**frozen["selected_portfolio_config"]),
            signal_data,
            execution_data,
            rules,
            execution,
            start,
            end,
            args.initial_equity,
        )
        result = universes[size]["final_result"]
        print(
            f"completed v2 {size}: campaigns={result['trade_count']} net={result['net_profit']:.2f} "
            f"pf={result['profit_factor']:.3f} dd={result['max_drawdown_pct']:.2%}",
            flush=True,
        )
    report = {
        "strategy_name": TREND_GRID_V2_RESEARCH_NAME,
        "strategy_version": TREND_GRID_VERSION,
        "research_status": "historical_refinement_not_untouched_oos",
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "initial_equity": args.initial_equity,
        "baseline_report": args.baseline_report,
        "preserved_gui_strategy": "config.volatility-breakout.v2-balanced-50-shadow.json",
        "cost_model": baseline["cost_model"],
        "execution_rules": baseline["execution_rules"],
        "funding_missing_symbols": metadata["funding_missing"],
        "universes": universes,
    }
    _write_json(Path(args.output), report)
    write_summary(Path(args.summary), report)
    for size, target in (("30", args.config30), ("50", args.config50)):
        result = universes[size]
        _write_json(
            Path(target),
            {
                "strategy_name": TREND_GRID_STRATEGY_NAME,
                "status": "historical_maximum_not_live",
                "period": report["period"],
                "symbols": result["symbols"],
                "signal": result["selected_signal_config"],
                "portfolio": result["selected_portfolio_config"],
                "validation_selected_signal": result["validation_selected_signal_config"],
                "validation_selected_portfolio": result["validation_selected_portfolio_config"],
                "cost_model": report["cost_model"],
                "execution_rules": report["execution_rules"],
            },
        )
    artifacts = (
        args.output,
        args.summary,
        args.config30,
        args.config50,
        "crypto_scalper/trend_grid.py",
        "crypto_scalper/trend_grid_optimize.py",
        "crypto_scalper/trend_grid_v2_optimize.py",
        "tests/test_trend_grid.py",
        "config.volatility-breakout.v2-balanced-50-shadow-manifest.json",
    )
    _write_json(
        Path(args.manifest),
        {
            "strategy_name": TREND_GRID_V2_RESEARCH_NAME,
            "status": "historical_research_frozen_separately_from_gui_strategy",
            "report": args.output,
            "configs": [args.config30, args.config50],
            "preserved_gui_strategy": "config.volatility-breakout.v2-balanced-50-shadow.json",
            "hashes": {path: sha256_file(path) for path in artifacts if Path(path).exists()},
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    report = run_refinement(build_parser().parse_args(argv))
    print(
        json.dumps(
            {size: compact_grid_summary(report["universes"][size]["final_result"]) for size in ("30", "50")},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

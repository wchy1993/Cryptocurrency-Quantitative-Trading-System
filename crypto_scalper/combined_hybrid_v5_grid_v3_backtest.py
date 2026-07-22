from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
    CombinedPortfolioConfig,
)
from .combined_volatility_trend_grid_v4_backtest import (
    simulate_combined_v4_portfolio,
)
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import (
    GridPortfolioConfig,
    build_grid_research_timeline,
    compact_grid_summary,
    simulate_grid_portfolio,
)
from .trend_grid_v3_optimize import GridMarketOverlay, apply_market_overlay
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import (
    ExitProtectionConfig,
    simulate_exit_protected_portfolio,
)
from .volatility_breakout_optimize import (
    Candidate,
    PortfolioSearchConfig,
    UNIVERSE_50,
    build_candidates,
    compact_summary,
    minute_datetime,
    sha256_file,
)
from .volatility_breakout_v4_research import (
    V4RegimeConfig,
    build_v4_families,
    build_v4_market_context,
    enrich_candidates_v4,
    filter_candidates_v4,
    load_v4_runtime_inputs,
)


COMBINED_V5_GRID_V3_NAME = (
    "hybrid_v5_balanced_expansion_runner_plus_dynamic_trend_grid_v3_max2"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _slice_signal_data(
    signal_data: dict[str, list[Any]],
    start: datetime,
    end: datetime,
) -> dict[str, list[Any]]:
    return {
        symbol: [row for row in rows if start <= row.timestamp < end]
        for symbol, rows in signal_data.items()
    }


def _daily_cap_candidates(
    candidates: dict[int, list[Candidate]],
    cap: int,
) -> dict[int, list[Candidate]]:
    counts: dict[tuple[str, str], int] = {}
    output: dict[int, list[Candidate]] = {}
    for minute in sorted(candidates):
        selected: list[Candidate] = []
        for candidate in candidates[minute]:
            key = (
                candidate.signal.symbol,
                minute_datetime(minute).date().isoformat(),
            )
            if counts.get(key, 0) >= cap:
                continue
            counts[key] = counts.get(key, 0) + 1
            selected.append(candidate)
        if selected:
            output[minute] = selected
    return output


def build_frozen_configs(
    breakout_path: str | Path,
    grid_path: str | Path,
) -> dict[str, Any]:
    breakout_payload = json.loads(Path(breakout_path).read_text(encoding="utf-8"))
    grid_payload = json.loads(Path(grid_path).read_text(encoding="utf-8"))
    entry = breakout_payload["entry"]
    runner = breakout_payload["runner"]
    family = build_v4_families()[entry["family_key"]]
    breakout_build_signal = replace(
        family,
        min_trend_alignment_atr=entry["min_alignment"],
        max_trend_alignment_atr=entry["max_alignment"],
        min_volume_ratio=0.0,
        max_volume_ratio=entry["max_volume"],
        min_body_atr=0.0,
        max_body_atr=entry["max_body"],
        min_directional_close_position=entry["min_close_position"],
        min_range_atr=0.0,
        max_range_atr=entry["max_range"],
        min_breakout_extension_atr=-999.0,
        max_breakout_extension_atr=entry["max_extension"],
        max_signals_per_symbol_day=24,
    )
    breakout_signal = replace(
        family,
        max_signals_per_symbol_day=2,
        stop_atr_multiple=runner["stop_atr_multiple"],
        take_profit_r=runner["take_profit_r"],
        fail_fast_minutes=runner["fail_fast_minutes"],
        fail_fast_min_mfe_r=runner["fail_fast_min_mfe_r"],
        fail_fast_max_current_r=runner["fail_fast_max_current_r"],
        max_holding_minutes=runner["max_holding_minutes"],
    )
    breakout_regime = V4RegimeConfig(
        max_directional_btc_return_4h=entry["max_directional_btc_return_4h"],
        min_market_efficiency_12h=entry["min_market_efficiency"],
        min_directional_symbol_ema55_atr=entry["min_ema55_alignment"],
        max_directional_symbol_ema55_atr=entry["max_ema55_alignment"],
        min_regime_score=entry["min_regime_score"],
    )
    breakout_portfolio = PortfolioSearchConfig(
        risk_per_trade_pct=runner["risk_per_trade_pct"],
        max_trade_risk_pct=0.10,
        max_open_positions=runner["max_open_positions"],
        max_daily_trades=5,
        symbol_cooldown_minutes=120,
        max_notional_multiple=9.0,
        hard_drawdown_stop_pct=0.60,
        compound=runner["compound"],
        ranking_mode=runner["ranking_mode"],
        long_risk_multiplier=runner["long_risk_multiplier"],
        short_risk_multiplier=runner["short_risk_multiplier"],
    )
    breakout_exit = ExitProtectionConfig(
        partial_take_profit_r=runner["partial_take_profit_r"],
        partial_take_profit_fraction=runner["partial_take_profit_fraction"],
    )
    grid_signal = TrendGridConfig(**grid_payload["signal"])
    grid_overlay = GridMarketOverlay(**grid_payload["market_overlay"])
    grid_portfolio = GridPortfolioConfig(**grid_payload["portfolio"])
    combined = CombinedPortfolioConfig(
        max_open_positions=2,
        max_gross_notional_multiple=9.0,
        hard_drawdown_stop_pct=0.60,
        allow_same_symbol_across_strategies=False,
        entry_priority=(BREAKOUT_KEY, GRID_KEY),
    )
    if breakout_portfolio.max_open_positions != 1:
        raise RuntimeError("frozen Hybrid v5 sleeve must remain max one position")
    if grid_portfolio.max_open_campaigns != 1:
        raise RuntimeError("frozen Grid v3 sleeve must remain max one campaign")
    combined.validate()
    return {
        "breakout_build_signal": breakout_build_signal,
        "breakout_signal": breakout_signal,
        "breakout_regime": breakout_regime,
        "breakout_portfolio": breakout_portfolio,
        "breakout_exit": breakout_exit,
        "grid_signal": grid_signal,
        "grid_overlay": grid_overlay,
        "grid_portfolio": grid_portfolio,
        "combined": combined,
    }


def _build_breakout_candidates(
    symbols: tuple[str, ...],
    signal_data: dict[str, list[Any]],
    execution_data: dict[str, Any],
    context: dict[int, dict[str, Any]],
    configs: dict[str, Any],
    start: datetime,
    end: datetime,
) -> dict[int, list[Candidate]]:
    raw = build_candidates(
        symbols,
        signal_data,
        execution_data,
        configs["breakout_build_signal"],
        start,
        end,
    )
    enriched = enrich_candidates_v4(raw, context)
    filtered = filter_candidates_v4(
        enriched,
        configs["breakout_build_signal"],
        configs["breakout_regime"],
        context,
    )
    return _daily_cap_candidates(filtered, 2)


def _compact_combined(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_count",
        "candidate_count_by_strategy",
        "trade_count",
        "initial_equity",
        "final_equity",
        "net_profit",
        "return_pct",
        "win_rate",
        "profit_factor",
        "expectancy_usdt",
        "expectancy_r",
        "max_drawdown_pct",
        "max_drawdown_duration_minutes",
        "fee",
        "slippage",
        "funding",
        "full_cost",
        "top5_profit_contribution",
        "hard_drawdown_stopped",
        "max_concurrent_positions",
        "max_entry_committed_notional_multiple",
        "concurrency_minutes",
        "concurrency_share",
        "position_minutes_by_strategy",
        "pnl_reconciliation_error",
        "rejected",
        "by_strategy",
        "by_month",
        "breakout_partial_exit_count",
    )
    return {key: result[key] for key in keys}


def _run_period(
    name: str,
    start: datetime,
    end: datetime,
    warmup_days: int,
    initial_equity: float,
    symbols: tuple[str, ...],
    all_signal_data: dict[str, list[Any]],
    execution_data: dict[str, Any],
    rules: dict[str, Any],
    execution: Any,
    configs: dict[str, Any],
) -> dict[str, Any]:
    period_signal_data = _slice_signal_data(
        all_signal_data, start - timedelta(days=warmup_days), end
    )
    context = build_v4_market_context(symbols, period_signal_data)
    breakout_candidates = _build_breakout_candidates(
        symbols,
        period_signal_data,
        execution_data,
        context,
        configs,
        start,
        end,
    )
    raw_grid_candidates, grid_snapshots = build_grid_research_timeline(
        symbols,
        period_signal_data,
        execution_data,
        configs["grid_signal"],
        start,
        end,
    )
    grid_candidates = apply_market_overlay(
        raw_grid_candidates, context, configs["grid_overlay"]
    )
    print(
        f"{name}: breakout candidates={sum(map(len, breakout_candidates.values()))} "
        f"grid candidates={sum(map(len, grid_candidates.values()))}",
        flush=True,
    )
    standalone_breakout = simulate_exit_protected_portfolio(
        breakout_candidates,
        symbols,
        execution_data,
        rules,
        configs["breakout_signal"],
        configs["breakout_portfolio"],
        configs["breakout_exit"],
        execution,
        start,
        end,
        initial_equity,
    )
    standalone_grid = simulate_grid_portfolio(
        grid_candidates,
        grid_snapshots,
        symbols,
        execution_data,
        rules,
        configs["grid_signal"],
        configs["grid_portfolio"],
        execution,
        start,
        end,
        initial_equity,
    )
    combined = simulate_combined_v4_portfolio(
        breakout_candidates,
        grid_candidates,
        grid_snapshots,
        symbols,
        execution_data,
        rules,
        configs["breakout_signal"],
        configs["breakout_portfolio"],
        configs["breakout_exit"],
        configs["grid_signal"],
        configs["grid_portfolio"],
        configs["combined"],
        execution,
        start,
        end,
        initial_equity,
    )
    reverse_combined_config = replace(
        configs["combined"], entry_priority=(GRID_KEY, BREAKOUT_KEY)
    )
    reverse_priority = simulate_combined_v4_portfolio(
        breakout_candidates,
        grid_candidates,
        grid_snapshots,
        symbols,
        execution_data,
        rules,
        configs["breakout_signal"],
        configs["breakout_portfolio"],
        configs["breakout_exit"],
        configs["grid_signal"],
        configs["grid_portfolio"],
        reverse_combined_config,
        execution,
        start,
        end,
        initial_equity,
    )
    if combined["max_concurrent_positions"] > 2:
        raise RuntimeError("combined simulation exceeded the global max-two limit")
    if abs(float(combined["pnl_reconciliation_error"])) > 1e-6:
        raise RuntimeError("combined PnL reconciliation failed")
    print(
        f"{name}: combined trades={combined['trade_count']} "
        f"net={combined['net_profit']:+.2f} PF={combined['profit_factor']:.3f} "
        f"DD={combined['max_drawdown_pct']:.2%} max_open={combined['max_concurrent_positions']}",
        flush=True,
    )
    return {
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "standalone": {
            BREAKOUT_KEY: compact_summary(standalone_breakout),
            GRID_KEY: compact_grid_summary(standalone_grid),
        },
        "combined": _compact_combined(combined),
        "reverse_priority": _compact_combined(reverse_priority),
        "full_combined_result": combined,
    }


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
    if metadata["minimum_coverage_ratio"] < 0.999999 or metadata["maximum_missing_minutes"]:
        raise RuntimeError("combined backtest requires gap-free stitched 1m data")
    if metadata["funding_missing_symbols"]:
        raise RuntimeError("combined backtest requires complete funding data")
    configs = build_frozen_configs(args.breakout_config, args.grid_config)
    periods = {
        "three_month": _run_period(
            "3m",
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
            "6m",
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
    report = {
        "strategy_name": COMBINED_V5_GRID_V3_NAME,
        "status": "independent_combined_research_gui_unchanged",
        "initial_equity": args.initial_equity,
        "universe_size": len(symbols),
        "symbols": list(symbols),
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
        "preserved": {
            "active_gui": "unchanged",
            "hybrid_v5_source": "unchanged",
            "grid_v3_source": "unchanged",
            "apt_grid": "unchanged",
        },
    }
    _write_json(args.output, report)
    config_payload = {
        "strategy_name": COMBINED_V5_GRID_V3_NAME,
        "status": "historical_research_not_live_gui_unchanged",
        "initial_equity": args.initial_equity,
        "source_configs": report["source_configs"],
        "portfolio": asdict(configs["combined"]),
        "execution_rules": report["execution_rules"],
        "results": {
            name: period["combined"] for name, period in periods.items()
        },
    }
    _write_json(args.config_output, config_payload)
    lines = [
        "# Hybrid v5 Breakout + Grid v3 Max2 Backtest",
        "",
        f"- Initial equity: `{args.initial_equity:.2f}U`",
        "- 50 symbols; each strategy max one position, shared account max two",
        "- Gap-free 1m execution; full fees, slippage and funding; adverse stop first",
        "- Existing GUI and both frozen source strategies remain unchanged",
        "",
        "| Period | Sleeve | Trades | Net | PF | Win rate | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for period_name, period in periods.items():
        label = "3 months" if period_name == "three_month" else "6 months"
        for sleeve, metrics in (
            ("Hybrid v5 standalone", period["standalone"][BREAKOUT_KEY]),
            ("Grid v3 standalone", period["standalone"][GRID_KEY]),
            ("Combined max2", period["combined"]),
        ):
            lines.append(
                f"| {label} | {sleeve} | {metrics['trade_count']} | "
                f"{metrics['net_profit']:+.2f}U | {metrics['profit_factor']:.3f} | "
                f"{metrics['win_rate']:.2%} | {metrics['max_drawdown_pct']:.2%} |"
            )
    lines.extend(["", "## Concurrency and strategy contribution", ""])
    for period_name, period in periods.items():
        label = "3 months" if period_name == "three_month" else "6 months"
        combined = period["combined"]
        breakout = combined["by_strategy"][BREAKOUT_KEY]
        grid = combined["by_strategy"][GRID_KEY]
        reverse = period["reverse_priority"]
        lines.extend(
            [
                f"- {label}: max concurrent positions `{combined['max_concurrent_positions']}`; "
                f"two-position time share `{combined['concurrency_share'].get('2', combined['concurrency_share'].get(2, 0.0)):.2%}`.",
                f"- {label} Breakout contribution: `{breakout['trade_count']}` trades, "
                f"`{breakout['net_pnl']:+.2f}U`, PF `{breakout['profit_factor']:.3f}`.",
                f"- {label} Grid contribution: `{grid['trade_count']}` trades, "
                f"`{grid['net_pnl']:+.2f}U`, PF `{grid['profit_factor']:.3f}`.",
                f"- {label} reverse priority: `{reverse['net_profit']:+.2f}U`, "
                f"PF `{reverse['profit_factor']:.3f}`, DD `{reverse['max_drawdown_pct']:.2%}`.",
            ]
        )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "strategy_name": COMBINED_V5_GRID_V3_NAME,
        "status": "independent_combined_research_artifacts",
        "config": args.config_output,
        "report": args.output,
        "summary": args.summary,
        "hashes": {
            str(path): sha256_file(Path(path))
            for path in (
                "crypto_scalper/combined_hybrid_v5_grid_v3_backtest.py",
                "tests/test_combined_hybrid_v5_grid_v3_backtest.py",
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
        description="Hybrid v5 Breakout plus Grid v3 shared max-two backtest"
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
        "--output", default="reports/combined_hybrid_v5_grid_v3_max2_3m_6m.json"
    )
    parser.add_argument(
        "--summary", default="reports/combined_hybrid_v5_grid_v3_max2_3m_6m.md"
    )
    parser.add_argument(
        "--config-output", default="config.combined-hybrid-v5-grid-v3-max2.json"
    )
    parser.add_argument(
        "--manifest", default="config.combined-hybrid-v5-grid-v3-max2-manifest.json"
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

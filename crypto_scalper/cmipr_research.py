from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .data import parse_timestamp
from .live_config import load_live_config
from .live_execution_backtest import (
    _align_execution_candles_by_utc_timestamp,
    _load_symbol_data,
    _resample_to_timeframe,
    run_execution_backtest_config,
)


STRESS_VARIANTS = (
    "baseline",
    "fixed_risk",
    "no_compounding",
    "extra_entry_addon_delay",
    "higher_cost",
    "no_addon",
    "one_addon",
    "two_addons",
    "no_runner",
)


def run_stress_suite(
    config_path: str,
    data_dir: str,
    variants: tuple[str, ...] = STRESS_VARIANTS,
    trade_start: Any = None,
    trade_end: Any = None,
    progress: bool = False,
) -> dict[str, Any]:
    base = load_live_config(config_path)
    unknown = sorted(set(variants) - set(STRESS_VARIANTS))
    if unknown:
        raise ValueError(f"unknown stress variants: {', '.join(unknown)}")
    budget = max(1, int(base.cmipr.research.max_experiments_per_stage))
    if len(variants) > budget:
        raise ValueError(f"stress suite has {len(variants)} variants, above stage budget {budget}")
    execution = _load_symbol_data(data_dir, tuple(base.trading.symbols), "1m")
    execution, alignment = _align_execution_candles_by_utc_timestamp(execution)
    symbols = tuple(symbol for symbol in base.trading.symbols if symbol in execution)
    base = replace(
        base,
        trading=replace(
            base.trading,
            symbols=symbols,
            entry_symbols=tuple(symbol for symbol in base.trading.entry_symbols if symbol in symbols),
        ),
    )
    signal = {
        symbol: _resample_to_timeframe(candles, "1m", base.trading.timeframe)
        for symbol, candles in execution.items()
    }
    mtf = {
        timeframe: {
            symbol: _resample_to_timeframe(candles, "1m", timeframe)
            for symbol, candles in execution.items()
        }
        for timeframe in ("5m", "15m", "30m", "1h", "4h")
    }
    output: dict[str, Any] = {
        "strategy_name": "cross_sectional_momentum_ignition_pyramid",
        "experiment_budget": budget,
        "variants_requested": list(variants),
        "time_alignment": alignment,
        "results": {},
    }
    baseline_trades: list[dict[str, Any]] = []
    shared_feature_cache: dict[tuple[Any, ...], Any] = {}
    for name in variants:
        config, cost_experiment = _variant_config(base, name)
        result = run_execution_backtest_config(
            config,
            execution,
            signal,
            mtf_candles_by_timeframe=mtf,
            initial_equity=base.risk.starting_capital_usdt,
            include_trades=True,
            compact=True,
            progress=progress,
            trade_start=trade_start,
            trade_end=trade_end,
            cost_experiment=cost_experiment,
            backtest_mode="conservative",
            cmipr_feature_cache=shared_feature_cache,
        )
        trades = [trade for trade in result.get("trades", []) if trade.get("strategy") == "cross_sectional_momentum_ignition_pyramid"]
        if name == "baseline":
            baseline_trades = trades
        output["results"][name] = _result_summary(result, trades)
    output["top_winner_exclusion"] = _top_winner_exclusion(baseline_trades)
    output["top_symbol_exclusion"] = _top_symbol_exclusion(baseline_trades)
    output["bounded_selection"] = _bounded_selection(base, output["results"])
    output["acceptance_note"] = "2026-04 through 2026-06 is historical test, never untouched final holdout; final acceptance requires post-freeze shadow/dry-run."
    return output


def _bounded_selection(config: Any, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    minimum = max(1, int(config.cmipr.research.min_research_trades))
    eligible = {
        name: row
        for name, row in results.items()
        if int(row.get("trade_count") or 0) >= minimum and row.get("profit_factor") is not None
    }
    if not eligible:
        return {"selected": None, "reason": f"insufficient_sample_min_trades={minimum}"}
    best_pf = max(float(row.get("profit_factor") or 0.0) for row in eligible.values())
    tolerance = max(0.0, float(config.cmipr.research.near_optimal_pf_tolerance))
    near = {
        name: row
        for name, row in eligible.items()
        if float(row.get("profit_factor") or 0.0) >= best_pf - tolerance
    }
    complexity = {
        "no_addon": 0,
        "no_runner": 0,
        "one_addon": 1,
        "baseline": 2,
        "two_addons": 2,
        "fixed_risk": 2,
        "no_compounding": 2,
        "extra_entry_addon_delay": 2,
        "higher_cost": 2,
    }
    selected = min(
        near,
        key=lambda name: (
            complexity.get(name, 3),
            float(near[name].get("max_drawdown_pct") or 0.0),
            -float(near[name].get("profit_factor") or 0.0),
        ),
    )
    return {
        "selected": selected,
        "best_pf": best_pf,
        "pf_tolerance": tolerance,
        "near_optimal_candidates": sorted(near),
        "rule": "prefer simpler variant first, then lower drawdown, within PF tolerance",
    }


def _variant_config(config: Any, name: str) -> tuple[Any, str]:
    if name == "baseline":
        return config, "full_cost"
    if name == "fixed_risk":
        return replace(config, cmipr=replace(config.cmipr, research=replace(config.cmipr.research, sizing_mode="fixed_risk_usdt"))), "full_cost"
    if name == "no_compounding":
        return replace(config, cmipr=replace(config.cmipr, research=replace(config.cmipr.research, sizing_mode="fixed_equity"))), "full_cost"
    if name == "extra_entry_addon_delay":
        return replace(
            config,
            cmipr=replace(
                config.cmipr,
                entry=replace(config.cmipr.entry, extra_execution_delay_minutes=2),
                pyramid=replace(config.cmipr.pyramid, extra_execution_delay_minutes=2),
            ),
        ), "full_cost"
    if name == "higher_cost":
        risk = config.risk
        stressed_risk = replace(
            risk,
            maker_fee_rate=float(risk.maker_fee_rate) * 1.5,
            taker_fee_rate=float(risk.taker_fee_rate) * 1.5,
            market_slippage_bps=float(risk.market_slippage_bps) * 2.0,
            stop_slippage_bps=float(risk.stop_slippage_bps) * 2.0,
            take_profit_slippage_bps=float(risk.take_profit_slippage_bps) * 2.0,
            impact_coefficient_bps=float(risk.impact_coefficient_bps) * 1.5,
        )
        return replace(config, risk=stressed_risk), "full_cost"
    if name == "no_addon":
        return replace(config, cmipr=replace(config.cmipr, pyramid=replace(config.cmipr.pyramid, enabled=False, max_addons=0))), "full_cost"
    if name == "one_addon":
        return replace(config, cmipr=replace(config.cmipr, pyramid=replace(config.cmipr.pyramid, enabled=True, max_addons=1))), "full_cost"
    if name == "two_addons":
        return replace(config, cmipr=replace(config.cmipr, pyramid=replace(config.cmipr.pyramid, enabled=True, max_addons=2))), "full_cost"
    if name == "no_runner":
        return replace(config, cmipr=replace(config.cmipr, exit=replace(config.cmipr.exit, runner_enabled=False))), "full_cost"
    raise ValueError(name)


def _result_summary(result: dict[str, Any], trades: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "final_equity": result.get("final_equity"),
        "net_pnl": result.get("net_pnl"),
        "net_return_pct": result.get("net_return_pct"),
        "trade_count": result.get("trade_count"),
        "win_rate_pct": result.get("win_rate_pct"),
        "profit_factor": result.get("profit_factor"),
        "max_drawdown_pct": result.get("max_drawdown_pct"),
        "fee": result.get("fee"),
        "slippage_cost": result.get("slippage_cost"),
        "funding": result.get("funding"),
        "monthly": result.get("monthly"),
        "cmipr_report": result.get("cmipr_report"),
        "top5_profit_contribution_pct": _top_profit_contribution(trades, 5),
        "top_symbol_profit_contribution_pct": _top_symbol_contribution(trades),
    }


def _trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(trade.get("net_pnl", 0.0) or 0.0) for trade in trades]
    gains = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value <= 0))
    return {
        "trade_count": len(values),
        "net_pnl": sum(values),
        "profit_factor": gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0),
    }


def _top_winner_exclusion(trades: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(trades, key=lambda trade: float(trade.get("net_pnl", 0.0) or 0.0), reverse=True)
    return {
        "excluded_count": min(5, len(ranked)),
        "remaining": _trade_metrics(ranked[min(5, len(ranked)):]),
        "note": "Analytical concentration stress; portfolio path is not replayed after deleting trades.",
    }


def _top_symbol_exclusion(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnl: dict[str, float] = {}
    for trade in trades:
        symbol = str(trade.get("symbol", ""))
        pnl[symbol] = pnl.get(symbol, 0.0) + float(trade.get("net_pnl", 0.0) or 0.0)
    top_symbol = max(pnl, key=pnl.get) if pnl else None
    remaining = [trade for trade in trades if trade.get("symbol") != top_symbol]
    return {
        "excluded_symbol": top_symbol,
        "remaining": _trade_metrics(remaining),
        "note": "Analytical concentration stress; portfolio path is not replayed after deleting a symbol.",
    }


def _top_profit_contribution(trades: list[dict[str, Any]], count: int) -> float:
    winners = sorted((float(trade.get("net_pnl", 0.0) or 0.0) for trade in trades if float(trade.get("net_pnl", 0.0) or 0.0) > 0), reverse=True)
    gross_profit = sum(winners)
    return sum(winners[:count]) / gross_profit * 100.0 if gross_profit > 0 else 0.0


def _top_symbol_contribution(trades: list[dict[str, Any]]) -> float:
    pnl: dict[str, float] = {}
    for trade in trades:
        value = float(trade.get("net_pnl", 0.0) or 0.0)
        if value > 0:
            symbol = str(trade.get("symbol", ""))
            pnl[symbol] = pnl.get(symbol, 0.0) + value
    total = sum(pnl.values())
    return max(pnl.values()) / total * 100.0 if total > 0 and pnl else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="CMIPR bounded full-cost stress suite")
    parser.add_argument("--config", default="config.cmipr.stage0.json")
    parser.add_argument("--data-dir", default="data/binance_1m_3m_top100")
    parser.add_argument("--variants", default=",".join(STRESS_VARIANTS))
    parser.add_argument("--trade-start", default=None)
    parser.add_argument("--trade-end", default=None)
    parser.add_argument("--output", default="reports/cmipr_stress_results.json")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    result = run_stress_suite(
        args.config,
        args.data_dir,
        variants,
        parse_timestamp(args.trade_start) if args.trade_start else None,
        parse_timestamp(args.trade_end) if args.trade_end else None,
        args.progress,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "variants": list(result["results"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

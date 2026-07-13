from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
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


CMIPR_NAME = "cross_sectional_momentum_ignition_pyramid"


def staged_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Return a bounded, cumulative alpha funnel search.

    Each row changes exactly one CMIPR module from the preceding row. The last
    row is an entry-mode comparison built from the same Stage 3 alpha config.
    """
    regime = replace(
        base,
        cmipr=replace(
            base.cmipr,
            regime=replace(
                base.cmipr.regime,
                enter_breadth_above_ema21=0.54,
                min_breadth_positive_1h=0.50,
                enter_ema_slope_pct=0.0008,
                exit_breadth_above_ema21=0.46,
                exit_ema_slope_pct=-0.0001,
                max_direction_conflict=0.45,
                min_state_hold_bars_1h=2,
            ),
        ),
    )
    ranking = replace(
        regime,
        cmipr=replace(
            regime.cmipr,
            ranking=replace(
                regime.cmipr.ranking,
                long_top_fraction=0.25,
                short_bottom_fraction=0.18,
                max_candidates_per_scan=12,
                max_extension_atr=3.50,
            ),
        ),
    )
    compression = replace(
        ranking,
        cmipr=replace(
            ranking.cmipr,
            compression=replace(
                ranking.cmipr.compression,
                max_atr_percentile=0.60,
                max_atr_to_average=1.00,
                max_channel_width_atr=6.00,
                max_volume_contraction=0.95,
                max_failed_breakouts=3,
                max_prior_move_atr=3.00,
            ),
        ),
    )
    ignition = replace(
        compression,
        cmipr=replace(
            compression.cmipr,
            ignition=replace(
                compression.cmipr.ignition,
                breakout_lookback_15m=16,
                min_breakout_distance_atr=0.05,
                min_body_atr=0.35,
                min_close_position=0.65,
                max_wick_ratio=0.35,
                min_volume_ratio=1.30,
                macd_hist_expanding_bars=1,
                max_ema21_distance_atr=3.00,
            ),
        ),
    )
    pullback = replace(
        ignition,
        cmipr=replace(
            ignition.cmipr,
            entry=replace(
                ignition.cmipr.entry,
                pending_expiry_minutes=120,
                pullback_min_depth_atr=0.05,
                pullback_max_depth_atr=1.50,
                pullback_max_volume_ratio=1.00,
                confirmation_min_close_position=0.55,
                max_chase_distance_atr=0.70,
                max_stop_atr=3.00,
                min_target_to_cost_ratio=3.50,
            ),
        ),
    )
    confirmation_open = replace(
        ignition,
        cmipr=replace(
            ignition.cmipr,
            entry=replace(
                ignition.cmipr.entry,
                mode="confirmation_open",
                max_chase_distance_atr=0.50,
                min_target_to_cost_ratio=4.00,
            ),
        ),
    )
    long_only = replace(
        pullback,
        cmipr=replace(pullback.cmipr, allow_short=False),
    )
    long_rank_30 = replace(
        long_only,
        cmipr=replace(
            long_only.cmipr,
            ranking=replace(long_only.cmipr.ranking, long_top_fraction=0.30),
        ),
    )
    long_rank_35 = replace(
        long_rank_30,
        cmipr=replace(
            long_rank_30.cmipr,
            ranking=replace(long_rank_30.cmipr.ranking, long_top_fraction=0.35),
        ),
    )
    return [
        ("stage0_baseline", "frozen Stage 0 baseline", base),
        ("stage1_regime", "only widen confirmed expansion regime", regime),
        ("stage2_ranking", "only widen cross-sectional rank eligibility", ranking),
        ("stage3a_compression", "only relax compression without removing it", compression),
        ("stage3b_ignition", "only relax ignition quality thresholds", ignition),
        ("stage4_pullback", "only widen healthy pullback confirmation", pullback),
        ("stage4_confirmation_open", "entry-mode comparison; next 1m open, chase guarded", confirmation_open),
        ("stage5_long_only", "disable historically unprofitable short side", long_only),
        ("stage5_long_rank30", "only widen long ranking eligibility to top 30 percent", long_rank_30),
        ("stage5_long_rank35", "only widen long ranking eligibility to top 35 percent", long_rank_35),
    ]


def run_staged_optimization(
    config_path: str,
    data_dir: str,
    trade_start: Any = None,
    trade_end: Any = None,
    progress: bool = False,
    experiment_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    base = load_live_config(config_path)
    experiments = staged_configs(base)
    if experiment_names:
        known = {name for name, _, _ in experiments}
        unknown = sorted(set(experiment_names) - known)
        if unknown:
            raise ValueError(f"unknown CMIPR experiments: {', '.join(unknown)}")
        experiments = [row for row in experiments if row[0] in experiment_names]
    budget = max(1, int(base.cmipr.research.max_experiments_per_stage))
    if len(experiments) > budget:
        raise ValueError(f"CMIPR staged search has {len(experiments)} variants, above budget {budget}")

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
    experiments = staged_configs(base)
    if experiment_names:
        experiments = [row for row in experiments if row[0] in experiment_names]
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
    shared_feature_cache: dict[tuple[Any, ...], Any] = {}
    output: dict[str, Any] = {
        "strategy_name": CMIPR_NAME,
        "research_status": "historical_research_not_final_holdout",
        "final_acceptance_source": base.cmipr.research.final_acceptance_source,
        "experiment_budget": budget,
        "experiments_run": len(experiments),
        "time_alignment": alignment,
        "results": {},
    }
    previous_config = None
    for name, description, config in experiments:
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
            cost_experiment="full_cost",
            backtest_mode="conservative",
            cmipr_feature_cache=shared_feature_cache,
        )
        output["results"][name] = {
            "description": description,
            "changed_modules": _changed_modules(previous_config, config),
            "config": _cmipr_alpha_config(config),
            **_summary(result),
        }
        previous_config = config
    output["diagnostic_ranking"] = _diagnostic_ranking(output["results"])
    output["selection_policy"] = {
        "automatic_live_selection": False,
        "minimum_research_trades": int(base.cmipr.research.min_research_trades),
        "rule": "A wider funnel is not accepted solely for frequency; prefer the simplest positive-expectancy configuration with adequate sample and lower drawdown.",
    }
    return output


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    trades = [trade for trade in result.get("trades", []) if trade.get("strategy") == CMIPR_NAME]
    pnls = [float(trade.get("net_pnl", 0.0) or 0.0) for trade in trades]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value <= 0]
    by_side: dict[str, list[float]] = defaultdict(list)
    by_exit: Counter[str] = Counter()
    by_symbol: dict[str, float] = defaultdict(float)
    for trade, pnl in zip(trades, pnls):
        by_side[str(trade.get("side", trade.get("direction", "UNKNOWN")))].append(pnl)
        by_exit[str(trade.get("exit_reason", "unknown"))] += 1
        by_symbol[str(trade.get("symbol", ""))] += pnl
    report = result.get("cmipr_report") or {}
    stats = report.get("stats") or {}
    execution_stats = report.get("execution_stats") or {}
    cost = float(result.get("fee") or 0.0) + float(result.get("slippage_cost") or 0.0) + abs(float(result.get("funding") or 0.0))
    gross_profit = sum(wins)
    return {
        "final_equity": result.get("final_equity"),
        "net_pnl": result.get("net_pnl"),
        "net_return_pct": result.get("net_return_pct"),
        "trade_count": len(trades),
        "win_rate_pct": len(wins) / len(trades) * 100.0 if trades else 0.0,
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else (float("inf") if wins else 0.0),
        "expectancy_per_trade": sum(pnls) / len(pnls) if pnls else 0.0,
        "average_win": sum(wins) / len(wins) if wins else 0.0,
        "average_loss": sum(losses) / len(losses) if losses else 0.0,
        "average_win_loss_ratio": (sum(wins) / len(wins)) / abs(sum(losses) / len(losses)) if wins and losses else 0.0,
        "max_drawdown_pct": result.get("max_drawdown_pct"),
        "fee": result.get("fee"),
        "slippage_cost": result.get("slippage_cost"),
        "funding": result.get("funding"),
        "full_cost_total": cost,
        "cost_to_gross_profit_ratio": cost / gross_profit if gross_profit > 0 else None,
        "funnel": {
            "ignition_count": int(stats.get("ignition_count", 0)),
            "pullback_confirmed": int(stats.get("pullback_confirmed", 0)),
            "entry_ready_count": int(stats.get("entry_ready_count", 0)),
            "initial_fill_count": int(execution_stats.get("initial_fill_count", 0)),
        },
        "reject_reasons": report.get("reject_reasons", {}),
        "regime_observations": report.get("regime_observations", {}),
        "execution_stats": execution_stats,
        "monthly": result.get("monthly", []),
        "by_side": {side: _pnl_metrics(values) for side, values in sorted(by_side.items())},
        "by_exit_reason": dict(sorted(by_exit.items())),
        "by_symbol_net_pnl": dict(sorted(by_symbol.items(), key=lambda item: (-item[1], item[0]))),
        "trades": trades,
    }


def _pnl_metrics(values: list[float]) -> dict[str, Any]:
    gains = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value <= 0))
    return {
        "trade_count": len(values),
        "net_pnl": sum(values),
        "win_rate_pct": sum(value > 0 for value in values) / len(values) * 100.0 if values else 0.0,
        "profit_factor": gains / losses if losses else (float("inf") if gains else 0.0),
    }


def _changed_modules(previous: Any, current: Any) -> list[str]:
    if previous is None:
        return ["baseline"]
    names = ("regime", "ranking", "compression", "ignition", "entry", "pyramid", "exit", "risk_control")
    changed = [name for name in names if getattr(previous.cmipr, name) != getattr(current.cmipr, name)]
    if (previous.cmipr.allow_long, previous.cmipr.allow_short) != (current.cmipr.allow_long, current.cmipr.allow_short):
        changed.append("direction_policy")
    return changed


def _cmipr_alpha_config(config: Any) -> dict[str, Any]:
    return {
        "regime": asdict(config.cmipr.regime),
        "ranking": asdict(config.cmipr.ranking),
        "compression": asdict(config.cmipr.compression),
        "ignition": asdict(config.cmipr.ignition),
        "entry": asdict(config.cmipr.entry),
    }


def _diagnostic_ranking(results: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        results,
        key=lambda name: (
            int(results[name].get("trade_count") or 0) < 10,
            -float(results[name].get("profit_factor") or 0.0),
            -float(results[name].get("expectancy_per_trade") or 0.0),
            float(results[name].get("max_drawdown_pct") or 0.0),
            -int(results[name].get("trade_count") or 0),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CMIPR bounded staged full-cost optimization")
    parser.add_argument("--config", default="config.cmipr.stage0.json")
    parser.add_argument("--data-dir", default="data/binance_1m_3m_top100")
    parser.add_argument("--trade-start", default=None)
    parser.add_argument("--trade-end", default=None)
    parser.add_argument("--output", default="reports/cmipr_staged_optimization_3m.json")
    parser.add_argument("--experiments", default="", help="Optional comma-separated bounded experiment names")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    result = run_staged_optimization(
        args.config,
        args.data_dir,
        parse_timestamp(args.trade_start) if args.trade_start else None,
        parse_timestamp(args.trade_end) if args.trade_end else None,
        args.progress,
        tuple(item.strip() for item in args.experiments.split(",") if item.strip()) or None,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "diagnostic_ranking": result["diagnostic_ranking"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

from .data import parse_timestamp
from .live_config import load_live_config
from .live_execution_backtest import (
    _align_execution_candles_by_utc_timestamp,
    _load_symbol_data,
    _resample_to_timeframe,
    run_execution_backtest_config,
)


def stage0_configs(base: Any) -> list[tuple[str, Any]]:
    return [
        (
            name,
            replace(base, mtper=replace(base.mtper, research=replace(base.mtper.research, stage_variant=name))),
        )
        for name in ("formal_cross", "pre_cross", "pre_cross_htf", "pre_cross_htf_15m")
    ]


def stage1_pre_cross_configs(base: Any) -> list[tuple[str, Any]]:
    rows = []
    for fast, slow in ((13, 34), (20, 50), (21, 55)):
        for gap in (0.30, 0.45, 0.60):
            name = f"ema{fast}_{slow}_gap{gap:.2f}"
            rows.append(
                (
                    name,
                    replace(
                        base,
                        mtper=replace(
                            base.mtper,
                            pre_cross=replace(
                                base.mtper.pre_cross,
                                ema_fast_period=fast,
                                ema_slow_period=slow,
                                max_gap_abs_atr=gap,
                            ),
                        ),
                    ),
                )
            )
    return rows


def stage2_extreme_configs(base: Any) -> list[tuple[str, Any]]:
    return [
        (
            f"extreme_{threshold:.2f}",
            replace(
                base,
                mtper=replace(
                    base.mtper,
                    extreme=replace(base.mtper.extreme, score_threshold=threshold),
                ),
            ),
        )
        for threshold in (0.25, 0.30, 0.35, 0.42, 0.50)
    ]


def stage3_entry_configs(base: Any) -> list[tuple[str, Any]]:
    return [
        (
            f"trigger_{mode}",
            replace(
                base,
                mtper=replace(
                    base.mtper,
                    entry=replace(base.mtper.entry, trigger_mode=mode),
                ),
            ),
        )
        for mode in (
            "higher_low_reclaim",
            "false_breakdown_reclaim",
            "local_structure_breakout",
            "combined",
        )
    ]


def stage4_second_entry_configs(base: Any) -> list[tuple[str, Any]]:
    return [
        (
            f"second_{mode}",
            replace(
                base,
                mtper=replace(
                    base.mtper,
                    second_entry=replace(
                        base.mtper.second_entry,
                        enabled=mode != "none",
                        mode=mode,
                    ),
                ),
            ),
        )
        for mode in ("none", "defensive", "winner_addon")
    ]


def stage5_exit_configs(base: Any) -> list[tuple[str, Any]]:
    allocations = {
        "ladder_a": (0.30, 0.40, 0.30),
        "ladder_b": (0.25, 0.35, 0.40),
        "ladder_c": (0.40, 0.40, 0.20),
    }
    return [
        (
            name,
            replace(
                base,
                mtper=replace(
                    base.mtper,
                    exit=replace(
                        base.mtper.exit,
                        target_mode=name,
                        target_1_fraction=fractions[0],
                        target_2_fraction=fractions[1],
                        trend_conversion_fraction=fractions[2],
                    ),
                ),
            ),
        )
        for name, fractions in allocations.items()
    ]


def stage7_direction_configs(base: Any) -> list[tuple[str, Any]]:
    return [
        ("long_only", replace(base, mtper=replace(base.mtper, allow_long=True, allow_short=False))),
        ("short_only", replace(base, mtper=replace(base.mtper, allow_long=False, allow_short=True))),
        ("long_short", replace(base, mtper=replace(base.mtper, allow_long=True, allow_short=True))),
    ]


def configs_for_stage(base: Any, stage: int) -> list[tuple[str, Any]]:
    builders = {
        0: stage0_configs,
        1: stage1_pre_cross_configs,
        2: stage2_extreme_configs,
        3: stage3_entry_configs,
        4: stage4_second_entry_configs,
        5: stage5_exit_configs,
        7: stage7_direction_configs,
    }
    if stage not in builders:
        raise ValueError(f"MTPER stage {stage} is not implemented for execution")
    rows = builders[stage](base)
    budget = max(1, int(base.mtper.research.max_experiments_per_stage))
    if len(rows) > budget:
        raise RuntimeError(f"stage {stage} experiment count {len(rows)} exceeds configured budget {budget}")
    return rows


def run_stage(
    base: Any,
    execution_data_dir: str,
    stage: int,
    trade_start: Any,
    trade_end: Any,
) -> dict[str, Any]:
    execution = _load_symbol_data(execution_data_dir, tuple(base.trading.symbols), "1m")
    if not execution:
        raise RuntimeError(f"no 1m execution data loaded from {execution_data_dir}")
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
        for timeframe in ("15m", "30m", "1h", "2h", "4h")
    }
    experiments = []
    for name, config in configs_for_stage(base, stage):
        result = run_execution_backtest_config(
            config,
            execution,
            signal,
            mtf_candles_by_timeframe=mtf,
            include_trades=False,
            compact=True,
            progress=False,
            trade_start=trade_start,
            trade_end=trade_end,
            cost_experiment="full_cost",
            backtest_mode="conservative",
        )
        report = result.get("mtper_report", {})
        experiments.append(
            {
                "name": name,
                "summary": {
                    key: result.get(key)
                    for key in (
                        "initial_equity",
                        "final_equity",
                        "net_pnl",
                        "trade_count",
                        "win_rate_pct",
                        "profit_factor",
                        "max_drawdown_pct",
                        "fee",
                        "slippage_cost",
                        "funding",
                    )
                },
                "campaign_summary": report.get("strategy_summary", {}),
                "funnel": report.get("stats", {}),
                "execution_stats": report.get("execution_stats", {}),
                "reject_reasons": report.get("reject_reasons", {}),
                "by_month": report.get("by_month", {}),
                "by_symbol": report.get("by_symbol", {}),
                "long_short": report.get("long_short_comparison", {}),
                "extreme_score_report": report.get("extreme_score_report", {}),
                "pre_cross_candidate_count": report.get("pre_cross_candidate_count", 0),
            }
        )
    return {
        "strategy_name": "multi_timeframe_pre_cross_exhaustion_reversal",
        "stage": stage,
        "selection_policy": "validation_only; test_is_evaluation_only",
        "cost_model": "full_cost_path_aware_conservative",
        "trade_start": trade_start.isoformat() if trade_start else None,
        "trade_end": trade_end.isoformat() if trade_end else None,
        "time_alignment": alignment,
        "experiments": experiments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded staged full-cost MTPER experiments")
    parser.add_argument("--config", default="config.mtper.stage0.json")
    parser.add_argument("--execution-data-dir", default="data/binance_1m_365d_top100")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--trade-start", required=True)
    parser.add_argument("--trade-end", required=True)
    args = parser.parse_args()
    payload = run_stage(
        load_live_config(args.config),
        args.execution_data_dir,
        args.stage,
        parse_timestamp(args.trade_start),
        parse_timestamp(args.trade_end),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

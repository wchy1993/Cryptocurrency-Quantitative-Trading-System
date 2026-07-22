from __future__ import annotations

import argparse
import json
import sys
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
from .mtpc import MTPC_REASON_TOKEN


def stage0_configs(base: Any) -> list[tuple[str, Any]]:
    return [
        (
            "trend_impulse_ranked",
            replace(
                base,
                mtpc=replace(
                    base.mtpc,
                    research=replace(base.mtpc.research, stage_variant="trend_impulse"),
                ),
            ),
        ),
        (
            "first_pullback_ranked",
            replace(
                base,
                mtpc=replace(
                    base.mtpc,
                    research=replace(base.mtpc.research, stage_variant="first_pullback"),
                ),
            ),
        ),
        (
            "first_pullback_no_rank",
            replace(
                base,
                mtpc=replace(
                    base.mtpc,
                    ranking=replace(base.mtpc.ranking, enabled=False),
                    research=replace(base.mtpc.research, stage_variant="first_pullback"),
                ),
            ),
        ),
    ]


def stage1_regime_configs(base: Any) -> list[tuple[str, Any]]:
    variants = {
        "regime_loose": (0.00, 0.00, 0.00, 0.45, 0.45, False, 1),
        "regime_medium": (0.01, 0.02, 0.04, 0.48, 0.47, True, 1),
        "regime_balanced": (0.01, 0.03, 0.06, 0.50, 0.48, True, 2),
        "regime_strict": (0.02, 0.04, 0.08, 0.52, 0.50, True, 2),
    }
    rows = []
    for name, values in variants.items():
        slope4, slope1, gap1, breadth, positive, align, confirmation = values
        rows.append(
            (
                name,
                replace(
                    base,
                    mtpc=replace(
                        base.mtpc,
                        # Stage 0 showed the baseline rank threshold removed every
                        # executable pullback. Keep ranking diagnostic-only while
                        # isolating the regime gate in Stage 1.
                        ranking=replace(base.mtpc.ranking, enabled=False),
                        regime=replace(
                            base.mtpc.regime,
                            min_4h_fast_slope_atr=slope4,
                            min_1h_fast_slope_atr=slope1,
                            min_1h_ema_gap_atr=gap1,
                            min_breadth_above_ema21=breadth,
                            min_breadth_positive_1h=positive,
                            require_btc_eth_alignment=align,
                            enter_confirmation_bars_1h=confirmation,
                        ),
                    ),
                ),
            )
        )
    return rows


def stage2_impulse_configs(base: Any) -> list[tuple[str, Any]]:
    # Freeze the Stage 1 diagnostic selection. This remains a research context,
    # not a live recommendation; later OOS checks may reject the loose gate.
    base = replace(
        base,
        mtpc=replace(
            base.mtpc,
            ranking=replace(base.mtpc.ranking, enabled=False),
            regime=replace(
                base.mtpc.regime,
                min_4h_fast_slope_atr=0.00,
                min_1h_fast_slope_atr=0.00,
                min_1h_ema_gap_atr=0.00,
                min_breadth_above_ema21=0.45,
                min_breadth_positive_1h=0.45,
                require_btc_eth_alignment=False,
                enter_confirmation_bars_1h=1,
            ),
        ),
    )
    variants = {
        "impulse_loose": (0.00, 1.50, 0.12, 0.54, 0.55, 0.70),
        "impulse_medium": (0.00, 1.35, 0.18, 0.57, 0.45, 0.85),
        "impulse_balanced": (0.02, 1.25, 0.22, 0.60, 0.40, 0.90),
        "impulse_quality": (0.03, 1.20, 0.25, 0.62, 0.35, 1.00),
    }
    return [
        (
            name,
            replace(
                base,
                mtpc=replace(
                    base.mtpc,
                    impulse=replace(
                        base.mtpc.impulse,
                        min_breakout_distance_atr=values[0],
                        max_breakout_distance_atr=values[1],
                        min_body_atr=values[2],
                        min_close_position=values[3],
                        max_upper_wick_ratio=values[4],
                        min_volume_ratio=values[5],
                    ),
                ),
            ),
        )
        for name, values in variants.items()
    ]


def stage3_pullback_configs(base: Any) -> list[tuple[str, Any]]:
    # Freeze the diagnostic regime and the middle impulse variant. All Stage 3
    # experiments therefore differ only in pullback/confirmation quality.
    base = replace(
        base,
        mtpc=replace(
            base.mtpc,
            ranking=replace(base.mtpc.ranking, enabled=False),
            regime=replace(
                base.mtpc.regime,
                min_4h_fast_slope_atr=0.00,
                min_1h_fast_slope_atr=0.00,
                min_1h_ema_gap_atr=0.00,
                min_breadth_above_ema21=0.45,
                min_breadth_positive_1h=0.45,
                require_btc_eth_alignment=False,
                enter_confirmation_bars_1h=1,
            ),
            impulse=replace(
                base.mtpc.impulse,
                min_breakout_distance_atr=0.00,
                max_breakout_distance_atr=1.35,
                min_body_atr=0.18,
                min_close_position=0.57,
                max_upper_wick_ratio=0.45,
                min_volume_ratio=0.85,
            ),
        ),
    )
    variants = {
        "pullback_loose": (1.50, 0.85, 1.10, 0.80, 0.52, 0.55, False, False),
        "pullback_medium": (1.35, 0.80, 1.00, 0.65, 0.55, 0.65, False, False),
        "pullback_balanced": (1.20, 0.75, 0.95, 0.55, 0.58, 0.70, True, False),
        "pullback_strict": (1.20, 0.75, 0.90, 0.45, 0.60, 0.75, True, True),
    }
    rows = []
    for name, values in variants.items():
        depth, retrace, volume, proximity, close_position, confirm_volume, macd_required, break_high = values
        rows.append(
            (
                name,
                replace(
                    base,
                    mtpc=replace(
                        base.mtpc,
                        pullback=replace(
                            base.mtpc.pullback,
                            max_depth_atr=depth,
                            max_retrace_fraction=retrace,
                            max_volume_to_impulse=volume,
                            ema_proximity_atr=proximity,
                            confirmation_min_close_position=close_position,
                            confirmation_min_volume_ratio=confirm_volume,
                            require_confirmation_macd_improvement=macd_required,
                            confirmation_break_previous_high=break_high,
                        ),
                    ),
                ),
            )
        )
    return rows


def stage4_exit_configs(base: Any) -> list[tuple[str, Any]]:
    rows = []
    for target_r in (0.8, 1.0, 1.2, 1.5):
        rows.append(
            (
                f"fixed_tp_{target_r:.1f}r",
                replace(
                    base,
                    mtpc=replace(
                        base.mtpc,
                        exit=replace(
                            base.mtpc.exit,
                            take_profit_1_r=target_r,
                            take_profit_1_fraction=1.0,
                            take_profit_2_r=max(target_r, 1.8),
                            runner_enabled=False,
                        ),
                    ),
                ),
            )
        )
    rows.extend(
        [
            (
                "tp1_70_runner",
                replace(
                    base,
                    mtpc=replace(
                        base.mtpc,
                        exit=replace(
                            base.mtpc.exit,
                            take_profit_1_r=1.0,
                            take_profit_1_fraction=0.70,
                            runner_enabled=True,
                        ),
                    ),
                ),
            ),
            (
                "tp1_70_fixed_1_8r",
                replace(
                    base,
                    mtpc=replace(
                        base.mtpc,
                        exit=replace(
                            base.mtpc.exit,
                            take_profit_1_r=1.0,
                            take_profit_1_fraction=0.70,
                            take_profit_2_r=1.8,
                            runner_enabled=False,
                        ),
                    ),
                ),
            ),
        ]
    )
    return rows


def stage5_cooldown_configs(base: Any) -> list[tuple[str, Any]]:
    # Stage 4 validation selected the simple fixed 1.5R exit. Freeze it while
    # isolating candidate-state cooldowns; runner and partial exits stay off.
    base = replace(
        base,
        mtpc=replace(
            base.mtpc,
            exit=replace(
                base.mtpc.exit,
                take_profit_1_r=1.5,
                take_profit_1_fraction=1.0,
                runner_enabled=False,
            ),
        ),
    )
    variants = {
        "cooldown_baseline_12h": (720, 720),
        "setup_60m_cancel_12h": (60, 720),
        "setup_12h_cancel_60m": (720, 60),
        "setup_60m_cancel_60m": (60, 60),
        "setup_15m_cancel_30m": (15, 30),
    }
    return [
        (
            name,
            replace(
                base,
                mtpc=replace(
                    base.mtpc,
                    risk_control=replace(
                        base.mtpc.risk_control,
                        setup_invalidation_cooldown_minutes=setup_minutes,
                        entry_cancel_cooldown_minutes=cancel_minutes,
                    ),
                ),
            ),
        )
        for name, (setup_minutes, cancel_minutes) in variants.items()
    ]


def stage6_cost_guard_configs(base: Any) -> list[tuple[str, Any]]:
    # Cooldown isolation did not add fills. Restore the frozen baseline
    # cooldowns and vary only the minimum full-cost target/cost multiple.
    base = replace(
        base,
        mtpc=replace(
            base.mtpc,
            exit=replace(
                base.mtpc.exit,
                take_profit_1_r=1.5,
                take_profit_1_fraction=1.0,
                runner_enabled=False,
            ),
            risk_control=replace(
                base.mtpc.risk_control,
                setup_invalidation_cooldown_minutes=720,
                entry_cancel_cooldown_minutes=720,
            ),
        ),
    )
    return [
        (
            f"cost_guard_{minimum_multiple:.1f}x",
            replace(
                base,
                mtpc=replace(
                    base.mtpc,
                    pullback=replace(
                        base.mtpc.pullback,
                        min_target_to_cost_ratio=minimum_multiple,
                    ),
                ),
            ),
        )
        for minimum_multiple in (4.0, 3.5, 3.0)
    ]


def stage7_risk_budget_configs(base: Any) -> list[tuple[str, Any]]:
    # Stage 6 selected 3.5x: it admitted one additional validation winner,
    # while 3.0x admitted nothing else. Vary only the full-cost trade budget
    # to measure whether exchange quantity granularity is suppressing fills.
    base = replace(
        base,
        mtpc=replace(
            base.mtpc,
            pullback=replace(
                base.mtpc.pullback,
                min_target_to_cost_ratio=3.5,
            ),
            exit=replace(
                base.mtpc.exit,
                take_profit_1_r=1.5,
                take_profit_1_fraction=1.0,
                runner_enabled=False,
            ),
            risk_control=replace(
                base.mtpc.risk_control,
                setup_invalidation_cooldown_minutes=720,
                entry_cancel_cooldown_minutes=720,
                max_trade_risk_pct=0.02,
                max_total_open_risk_pct=0.02,
            ),
        ),
    )
    return [
        (
            f"trade_risk_{risk_pct * 100:.2f}pct",
            replace(
                base,
                mtpc=replace(
                    base.mtpc,
                    risk_control=replace(
                        base.mtpc.risk_control,
                        trade_risk_pct=risk_pct,
                    ),
                ),
            ),
        )
        for risk_pct in (0.01, 0.0125, 0.015, 0.02)
    ]


def stage8_selected_diagnostic_configs(base: Any) -> list[tuple[str, Any]]:
    return [
        (
            "selected_1_5pct_diagnostic",
            replace(
                base,
                mtpc=replace(
                    base.mtpc,
                    pullback=replace(
                        base.mtpc.pullback,
                        min_target_to_cost_ratio=3.5,
                    ),
                    exit=replace(
                        base.mtpc.exit,
                        take_profit_1_r=1.5,
                        take_profit_1_fraction=1.0,
                        runner_enabled=False,
                    ),
                    risk_control=replace(
                        base.mtpc.risk_control,
                        trade_risk_pct=0.015,
                        max_trade_risk_pct=0.02,
                        max_total_open_risk_pct=0.02,
                        setup_invalidation_cooldown_minutes=720,
                        entry_cancel_cooldown_minutes=720,
                    ),
                ),
            ),
        )
    ]


def stage9_pullback_volume_configs(base: Any) -> list[tuple[str, Any]]:
    # Stage 8 diagnostics identified pullback volume contraction as the
    # dominant pending blocker. Freeze every selected setting and vary only
    # the pullback-to-impulse volume ceiling.
    [(_, selected)] = stage8_selected_diagnostic_configs(base)
    return [
        (
            f"pullback_volume_{volume_ratio:.2f}x",
            replace(
                selected,
                mtpc=replace(
                    selected.mtpc,
                    pullback=replace(
                        selected.mtpc.pullback,
                        max_volume_to_impulse=volume_ratio,
                    ),
                ),
            ),
        )
        for volume_ratio in (1.0, 1.05, 1.10, 1.20)
    ]


def stage10_ranking_floor_configs(base: Any) -> list[tuple[str, Any]]:
    # Pullback-volume relaxation increased pullback observations but did not
    # add executable trades. Restore the 1.0x contraction ceiling and vary
    # only the lower cross-sectional rank bound, retaining the overheated tail
    # rejection at the configured upper bound.
    [(_, selected)] = stage8_selected_diagnostic_configs(base)
    return [
        (
            f"ranking_floor_{minimum_percentile:.2f}",
            replace(
                selected,
                mtpc=replace(
                    selected.mtpc,
                    ranking=replace(
                        selected.mtpc.ranking,
                        min_percentile=minimum_percentile,
                    ),
                    pullback=replace(
                        selected.mtpc.pullback,
                        max_volume_to_impulse=1.0,
                    ),
                ),
            ),
        )
        for minimum_percentile in (0.55, 0.50, 0.45, 0.40)
    ]


def stage11_ranking_ceiling_configs(base: Any) -> list[tuple[str, Any]]:
    # Lower-ranked setups occupied symbol state without adding fills. Restore
    # the 0.55 floor and isolate whether the strongest five-percent tail adds
    # useful continuation candidates or merely late, overextended entries.
    [(_, selected)] = stage8_selected_diagnostic_configs(base)
    return [
        (
            f"ranking_ceiling_{maximum_percentile:.2f}",
            replace(
                selected,
                mtpc=replace(
                    selected.mtpc,
                    ranking=replace(
                        selected.mtpc.ranking,
                        min_percentile=0.55,
                        max_percentile=maximum_percentile,
                    ),
                ),
            ),
        )
        for maximum_percentile in (0.95, 0.98, 1.0)
    ]


def stage12_position_limit_configs(base: Any) -> list[tuple[str, Any]]:
    # Ranking changes did not add fills. Test whether the global single-slot
    # guard is suppressing independent setups while a campaign is open. Each
    # variant keeps 1.5% per campaign and scales the enforced aggregate risk
    # cap linearly with the number of permitted positions.
    [(_, selected)] = stage8_selected_diagnostic_configs(base)
    return [
        (
            f"max_positions_{position_limit}",
            replace(
                selected,
                trading=replace(
                    selected.trading,
                    max_open_positions=position_limit,
                ),
                mtpc=replace(
                    selected.mtpc,
                    risk_control=replace(
                        selected.mtpc.risk_control,
                        max_open_positions=position_limit,
                        max_same_direction_positions=position_limit,
                        max_total_open_risk_pct=0.015 * position_limit,
                    ),
                ),
            ),
        )
        for position_limit in (1, 2, 3)
    ]


def stage13_trend_maturity_configs(base: Any) -> list[tuple[str, Any]]:
    # Concurrent slots did not add trades. Restore the single-position risk
    # envelope and vary only how mature the 1h EMA21/55 expansion may be when
    # a new impulse is accepted.
    [(_, selected)] = stage8_selected_diagnostic_configs(base)
    return [
        (
            f"max_1h_ema_gap_{maximum_gap_atr:.1f}atr",
            replace(
                selected,
                trading=replace(selected.trading, max_open_positions=1),
                mtpc=replace(
                    selected.mtpc,
                    regime=replace(
                        selected.mtpc.regime,
                        max_1h_ema_gap_atr=maximum_gap_atr,
                    ),
                    risk_control=replace(
                        selected.mtpc.risk_control,
                        max_open_positions=1,
                        max_same_direction_positions=1,
                        max_total_open_risk_pct=0.015,
                    ),
                ),
            ),
        )
        for maximum_gap_atr in (1.2, 1.5, 1.8, 2.4)
    ]


def stage14_pullback_timeframe_configs(base: Any) -> list[tuple[str, Any]]:
    # Freeze the selected 15m impulse and 1.5 ATR trend-maturity cap. Compare
    # the original 15m pullback with closed 5m pullbacks over bounded windows;
    # confirmation and execution timing remain unchanged.
    [(_, selected)] = stage8_selected_diagnostic_configs(base)
    selected = replace(
        selected,
        trading=replace(selected.trading, max_open_positions=1),
        mtpc=replace(
            selected.mtpc,
            regime=replace(selected.mtpc.regime, max_1h_ema_gap_atr=1.5),
            risk_control=replace(
                selected.mtpc.risk_control,
                trade_risk_pct=0.015,
                max_open_positions=1,
                max_same_direction_positions=1,
                max_total_open_risk_pct=0.015,
            ),
        ),
    )
    variants = (
        ("pullback_15m_150m", "15m", 10),
        ("pullback_5m_90m", "5m", 18),
        ("pullback_5m_150m", "5m", 30),
    )
    return [
        (
            name,
            replace(
                selected,
                mtpc=replace(
                    selected.mtpc,
                    pullback=replace(
                        selected.mtpc.pullback,
                        pullback_timeframe=timeframe,
                        max_bars_after_impulse=max_bars,
                    ),
                ),
            ),
        )
        for name, timeframe, max_bars in variants
    ]


def stage15_five_minute_confirmation_configs(base: Any) -> list[tuple[str, Any]]:
    # The 5m pullback increased fills but admitted a full-stop loser. Keep its
    # 90-minute window and isolate one confirmation-quality requirement per
    # experiment before considering any combination.
    [(_, selected)] = stage8_selected_diagnostic_configs(base)
    selected = replace(
        selected,
        trading=replace(selected.trading, max_open_positions=1),
        mtpc=replace(
            selected.mtpc,
            regime=replace(selected.mtpc.regime, max_1h_ema_gap_atr=1.5),
            pullback=replace(
                selected.mtpc.pullback,
                pullback_timeframe="5m",
                max_bars_after_impulse=18,
            ),
            risk_control=replace(
                selected.mtpc.risk_control,
                trade_risk_pct=0.015,
                max_open_positions=1,
                max_same_direction_positions=1,
                max_total_open_risk_pct=0.015,
            ),
        ),
    )
    return [
        ("five_minute_current", selected),
        (
            "five_minute_macd_improving",
            replace(
                selected,
                mtpc=replace(
                    selected.mtpc,
                    pullback=replace(selected.mtpc.pullback, require_confirmation_macd_improvement=True),
                ),
            ),
        ),
        (
            "five_minute_break_previous_high",
            replace(
                selected,
                mtpc=replace(
                    selected.mtpc,
                    pullback=replace(selected.mtpc.pullback, confirmation_break_previous_high=True),
                ),
            ),
        ),
        (
            "five_minute_close_position_0_60",
            replace(
                selected,
                mtpc=replace(
                    selected.mtpc,
                    pullback=replace(selected.mtpc.pullback, confirmation_min_close_position=0.60),
                ),
            ),
        ),
        (
            "five_minute_volume_0_80",
            replace(
                selected,
                mtpc=replace(
                    selected.mtpc,
                    pullback=replace(selected.mtpc.pullback, confirmation_min_volume_ratio=0.80),
                ),
            ),
        ),
    ]


def configs_for_stage(base: Any, stage: int) -> list[tuple[str, Any]]:
    builders = {
        0: stage0_configs,
        1: stage1_regime_configs,
        2: stage2_impulse_configs,
        3: stage3_pullback_configs,
        4: stage4_exit_configs,
        5: stage5_cooldown_configs,
        6: stage6_cost_guard_configs,
        7: stage7_risk_budget_configs,
        8: stage8_selected_diagnostic_configs,
        9: stage9_pullback_volume_configs,
        10: stage10_ranking_floor_configs,
        11: stage11_ranking_ceiling_configs,
        12: stage12_position_limit_configs,
        13: stage13_trend_maturity_configs,
        14: stage14_pullback_timeframe_configs,
        15: stage15_five_minute_confirmation_configs,
    }
    if stage not in builders:
        raise ValueError(f"MTPC stage {stage} is not implemented")
    rows = builders[stage](base)
    budget = max(1, int(base.mtpc.research.max_experiments_per_stage))
    if len(rows) > budget:
        raise RuntimeError(f"stage {stage} experiment count {len(rows)} exceeds budget {budget}")
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
        for timeframe in ("5m", "15m", "1h", "4h")
    }
    experiments = []
    stage_configs = configs_for_stage(base, stage)
    for experiment_index, (name, config) in enumerate(stage_configs, start=1):
        print(
            f"[mtpc-optimize] stage={stage} experiment={experiment_index}/{len(stage_configs)} name={name}",
            file=sys.stderr,
            flush=True,
        )
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
        report = result.get("mtpc_report", {})
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
                "strategy_summary": report.get("strategy_summary", {}),
                "funnel": report.get("stats", {}),
                "execution_stats": report.get("execution_stats", {}),
                "reject_reasons": report.get("reject_reasons", {}),
                "by_month": report.get("by_month", {}),
                "by_symbol": report.get("by_symbol", {}),
                "by_exit_reason": report.get("by_exit_reason", {}),
                "event_count": report.get("event_count", 0),
            }
        )
    return {
        "strategy_name": MTPC_REASON_TOKEN,
        "stage": stage,
        "selection_policy": "validation_only; historical test is evaluation only",
        "cost_model": "full_cost_path_aware_conservative",
        "trade_start": trade_start.isoformat() if trade_start else None,
        "trade_end": trade_end.isoformat() if trade_end else None,
        "time_alignment": alignment,
        "experiments": experiments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded staged full-cost MTPC experiments")
    parser.add_argument("--config", default="config.mtpc.stage0.json")
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

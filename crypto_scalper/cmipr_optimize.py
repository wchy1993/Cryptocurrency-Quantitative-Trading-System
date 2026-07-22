from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import timedelta
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


def shortline_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Bounded short-horizon CMIPR experiments.

    The alpha funnel remains cumulative and changes one module at a time.
    Runner rows branch from the same Stage 3 alpha so exit comparisons do not
    accidentally change the entry population.
    """
    compression_15m = replace(
        base,
        cmipr=replace(
            base.cmipr,
            compression=replace(
                base.cmipr.compression,
                timeframe="15m",
                lookback_bars_30m=16,
                atr_percentile_lookback=96,
                max_atr_percentile=0.55,
                max_atr_to_average=1.00,
                max_channel_width_atr=5.50,
                max_volume_contraction=0.95,
                max_prior_move_atr=2.75,
            ),
        ),
    )
    ignition_5m = replace(
        compression_15m,
        cmipr=replace(
            compression_15m.cmipr,
            ignition=replace(
                compression_15m.cmipr.ignition,
                timeframe="5m",
                breakout_lookback_15m=18,
                min_breakout_distance_atr=0.02,
                max_breakout_distance_atr=0.90,
                min_body_atr=0.18,
                min_close_position=0.68,
                max_wick_ratio=0.30,
                min_volume_ratio=1.40,
                macd_hist_expanding_bars=1,
                max_ema21_distance_atr=2.50,
            ),
        ),
    )
    quick_pullback = replace(
        ignition_5m,
        cmipr=replace(
            ignition_5m.cmipr,
            entry=replace(
                ignition_5m.cmipr.entry,
                pending_expiry_minutes=45,
                pullback_min_depth_atr=0.02,
                pullback_max_depth_atr=0.90,
                pullback_max_volume_ratio=0.95,
                confirmation_min_close_position=0.60,
                max_chase_distance_atr=0.45,
                min_stop_atr=0.35,
                max_stop_atr=2.00,
                min_target_to_cost_ratio=4.00,
            ),
        ),
    )

    def runner(trailing_type: str) -> Any:
        return replace(
            quick_pullback,
            cmipr=replace(
                quick_pullback.cmipr,
                exit=replace(
                    quick_pullback.cmipr.exit,
                    runner_enabled=True,
                    runner_activation_r=1.50,
                    trailing_type=trailing_type,
                    giveback_r_low=0.50,
                    giveback_r_mid=0.70,
                    giveback_r_high=0.90,
                    giveback_mid_mfe_r=2.00,
                    giveback_high_mfe_r=3.50,
                    max_holding_minutes=1440,
                ),
            ),
        )

    return [
        ("short0_current", "frozen long-only CMIPR configuration", base),
        ("short1_compression_15m", "only move compression setup from 30m to 15m", compression_15m),
        ("short2_ignition_5m", "only move ignition confirmation from 15m to 5m", ignition_5m),
        ("short3_fixed_1p5r", "only shorten and tighten the 5m pullback entry; keep fixed 1.5R exit", quick_pullback),
        ("short4_runner_ema9_5m", "shortline runner with 5m EMA9 trailing", runner("ema9_5m")),
        ("short4_runner_ema21_5m", "shortline runner with 5m EMA21 trailing", runner("ema21_5m")),
        ("short4_runner_chandelier_5m", "shortline runner with 5m Chandelier trailing", runner("chandelier_5m")),
    ]


def convex_exit_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Compare convex winner exits without changing the proven entry funnel."""

    frozen_entry = replace(
        base,
        cmipr=replace(
            base.cmipr,
            entry=replace(base.cmipr.entry, cost_guard_target_r=1.20),
        ),
    )

    def fixed(target_r: float) -> Any:
        return replace(
            frozen_entry,
            cmipr=replace(
                frozen_entry.cmipr,
                exit=replace(
                    frozen_entry.cmipr.exit,
                    runner_enabled=False,
                    fixed_take_profit_r=target_r,
                    runner_activation_r=target_r,
                ),
            ),
        )

    def runner(trailing_type: str, low: float, mid: float, high: float) -> Any:
        return replace(
            frozen_entry,
            cmipr=replace(
                frozen_entry.cmipr,
                exit=replace(
                    frozen_entry.cmipr.exit,
                    runner_enabled=True,
                    runner_activation_r=2.00,
                    trailing_type=trailing_type,
                    giveback_r_low=low,
                    giveback_r_mid=mid,
                    giveback_r_high=high,
                    giveback_mid_mfe_r=2.00,
                    giveback_high_mfe_r=3.50,
                    max_holding_minutes=1440,
                ),
            ),
        )

    return [
        ("convex0_fixed_1p5r", "frozen entry funnel and fixed 1.5R winner exit", fixed(1.50)),
        ("convex1_fixed_2p0r", "only increase fixed winner exit to 2.0R", fixed(2.00)),
        ("convex2_fixed_2p5r", "only increase fixed winner exit to 2.5R", fixed(2.50)),
        ("convex3_runner_ema9_5m", "activate runner with 5m EMA9 and tight segmented giveback", runner("ema9_5m", 0.45, 0.65, 0.85)),
        ("convex4_runner_ema21_5m", "activate runner with 5m EMA21 and balanced segmented giveback", runner("ema21_5m", 0.55, 0.75, 0.95)),
        ("convex5_runner_chandelier_5m", "activate runner with 5m Chandelier and balanced segmented giveback", runner("chandelier_5m", 0.55, 0.75, 0.95)),
        ("convex6_runner_ema9_15m", "activate runner with 15m EMA9 and wider segmented giveback", runner("ema9_15m", 0.60, 0.80, 1.00)),
    ]


def winner_pyramid_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Allocate more of the original risk budget only after a winner confirms."""
    frozen = replace(
        base,
        cmipr=replace(
            base.cmipr,
            entry=replace(base.cmipr.entry, cost_guard_target_r=1.20),
            exit=replace(
                base.cmipr.exit,
                runner_enabled=False,
                fixed_take_profit_r=2.00,
                runner_activation_r=2.00,
            ),
            pyramid=replace(base.cmipr.pyramid, enabled=True, max_addons=1),
        ),
    )

    def addon(fraction: float, trigger_r: float = 0.60) -> Any:
        return replace(
            frozen,
            cmipr=replace(
                frozen.cmipr,
                pyramid=replace(
                    frozen.cmipr.pyramid,
                    addon_1_risk_fraction=fraction,
                    addon_1_trigger_r=trigger_r,
                ),
            ),
        )

    return [
        ("pyramid0_addon_30", "frozen 40/30 initial and winner add-on allocation", addon(0.30)),
        ("pyramid1_addon_40", "only raise winner add-on allocation to 40 percent", addon(0.40)),
        ("pyramid2_addon_50", "only raise winner add-on allocation to 50 percent", addon(0.50)),
        ("pyramid3_addon_60", "only raise winner add-on allocation to 60 percent", addon(0.60)),
        ("pyramid4_trigger_45_addon_50", "earlier 0.45R confirmation with 50 percent add-on allocation", addon(0.50, 0.45)),
        ("pyramid5_trigger_80_addon_60", "later 0.80R confirmation with 60 percent add-on allocation", addon(0.60, 0.80)),
    ]


def risk_scale_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Scale only the original full-cost trade risk budget."""

    def risk_budget(risk_pct: float) -> Any:
        return replace(base, risk=replace(base.risk, risk_per_trade_pct=risk_pct))

    return [
        ("risk_0p50", "frozen sizing baseline at 0.50 percent total trade risk", risk_budget(0.0050)),
        ("risk_0p75", "only raise total trade risk to 0.75 percent", risk_budget(0.0075)),
        ("risk_1p00", "only raise total trade risk to 1.00 percent", risk_budget(0.0100)),
        ("risk_1p50", "only raise total trade risk to 1.50 percent", risk_budget(0.0150)),
        ("risk_2p00", "only raise total trade risk to 2.00 percent", risk_budget(0.0200)),
    ]


def high_risk_scale_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Stress higher risk budgets under an explicit maintenance-margin cap."""

    def risk_budget(risk_pct: float) -> Any:
        return replace(
            base,
            risk=replace(
                base.risk,
                risk_per_trade_pct=risk_pct,
                max_symbol_margin_pct=0.20,
                max_maintenance_margin_ratio_pct=0.05,
                estimated_maintenance_margin_rate=0.005,
            ),
        )

    return [
        ("risk_2p00_guarded", "2.00 percent risk with five percent maintenance-margin cap", risk_budget(0.0200)),
        ("risk_3p00_guarded", "3.00 percent risk with five percent maintenance-margin cap", risk_budget(0.0300)),
        ("risk_5p00_guarded", "5.00 percent risk with five percent maintenance-margin cap", risk_budget(0.0500)),
        ("risk_7p50_guarded", "7.50 percent risk with five percent maintenance-margin cap", risk_budget(0.0750)),
        ("risk_10p00_guarded", "10.00 percent risk with five percent maintenance-margin cap", risk_budget(0.1000)),
    ]


def campaign_stage1_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Compare the frozen 1.5R baseline with diagnostics off and on."""
    control = replace(
        base,
        cmipr=replace(
            base.cmipr,
            research=replace(base.cmipr.research, event_diagnostics_enabled=False),
        ),
    )
    diagnostics = replace(
        base,
        cmipr=replace(
            base.cmipr,
            research=replace(base.cmipr.research, event_diagnostics_enabled=True),
        ),
    )
    return [
        ("stage0_control", "frozen 1.5R baseline with event diagnostics disabled", control),
        ("stage1_diagnostics", "same baseline with research-only event diagnostics enabled", diagnostics),
    ]


def campaign_stage2_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Bounded initial-entry experiments with the 1.5R exit held fixed."""
    initial_only = replace(
        base,
        cmipr=replace(
            base.cmipr,
            pyramid=replace(base.cmipr.pyramid, enabled=False, max_addons=0),
            exit=replace(base.cmipr.exit, runner_enabled=False, fixed_take_profit_r=1.50),
            research=replace(base.cmipr.research, event_diagnostics_enabled=False),
        ),
    )
    early_regime = replace(
        initial_only,
        cmipr=replace(
            initial_only.cmipr,
            regime=replace(initial_only.cmipr.regime, phase_model_enabled=True),
        ),
    )
    acceleration = replace(
        early_regime,
        cmipr=replace(
            early_regime.cmipr,
            ranking=replace(
                early_regime.cmipr.ranking,
                strength_acceleration_enabled=True,
            ),
        ),
    )
    bull_flag = replace(
        acceleration,
        cmipr=replace(
            acceleration.cmipr,
            entry=replace(
                acceleration.cmipr.entry,
                mode="bull_flag",
                pending_expiry_minutes=180,
            ),
        ),
    )
    combined_entries = replace(
        acceleration,
        cmipr=replace(
            acceleration.cmipr,
            entry=replace(
                acceleration.cmipr.entry,
                mode="first_pullback_or_bull_flag",
                pending_expiry_minutes=180,
            ),
        ),
    )
    return [
        ("campaign2_0_initial_baseline", "1.5R initial entry only; addon and runner disabled", initial_only),
        ("campaign2_1_early_regime", "only admit new longs during EARLY_BULL_EXPANSION", early_regime),
        ("campaign2_2_strength_acceleration", "also require improving cross-sectional rank", acceleration),
        ("campaign2_3_bull_flag_only", "replace first pullback with a confirmed 5m bull flag", bull_flag),
        ("campaign2_4_first_pullback_or_bull_flag", "allow either independent confirmed entry structure", combined_entries),
    ]


def stage25_invariance_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Run the frozen first-pullback baseline through cold and warm caches."""
    baseline = replace(
        base,
        cmipr=replace(
            base.cmipr,
            mode="convex_campaign_stage2p5",
            allow_long=True,
            allow_short=False,
            regime=replace(base.cmipr.regime, phase_model_enabled=False),
            ranking=replace(base.cmipr.ranking, strength_acceleration_enabled=False),
            entry=replace(
                base.cmipr.entry,
                mode="first_pullback",
                event_structure_mode="current",
                delayed_recompression_enabled=False,
                entry_revalidation_mode="none",
            ),
            pyramid=replace(base.cmipr.pyramid, enabled=False, max_addons=0),
            exit=replace(
                base.cmipr.exit,
                runner_enabled=False,
                fixed_take_profit_r=1.50,
                take_profit_r_basis="campaign",
                fail_fast_r_basis="campaign",
                breakeven_r_basis="campaign",
                runner_activation_r_basis="campaign",
                giveback_r_basis="campaign",
                fail_fast_mode="current",
            ),
            research=replace(base.cmipr.research, event_diagnostics_enabled=False),
        ),
    )
    return [
        ("stage25_baseline_cold_cache", "frozen first-pullback baseline, cold feature cache", baseline),
        ("stage25_baseline_warm_cache", "identical baseline, warm feature cache", baseline),
    ]


def stage25_diagnostics_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Enable only research diagnostics on the frozen Stage 2.5 baseline."""
    baseline = stage25_invariance_configs(base)[0][2]
    diagnostics = replace(
        baseline,
        cmipr=replace(
            baseline.cmipr,
            research=replace(baseline.cmipr.research, event_diagnostics_enabled=True),
        ),
    )
    return [
        (
            "stage25_event_diagnostics",
            "frozen first-pullback trades with isolated full-cost event and fail-fast shadow diagnostics",
            diagnostics,
        )
    ]


def stage25_event_structure_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Bounded stale-event/recompression experiments; exits and risk stay frozen."""
    baseline = stage25_invariance_configs(base)[0][2]

    def event_variant(name: str, age: int, delayed: bool) -> Any:
        return replace(
            baseline,
            cmipr=replace(
                baseline.cmipr,
                entry=replace(
                    baseline.cmipr.entry,
                    event_structure_mode=name,
                    immediate_max_age_minutes=age,
                    delayed_recompression_enabled=delayed,
                ),
                research=replace(baseline.cmipr.research, event_diagnostics_enabled=True),
            ),
        )

    return [
        ("event_a_current", "current first-pullback pending semantics", baseline),
        ("event_b_immediate_30m", "immediate first pullback, stale after 30 minutes", event_variant("immediate", 30, False)),
        ("event_b_immediate_45m", "immediate first pullback, stale after 45 minutes", event_variant("immediate", 45, False)),
        ("event_b_immediate_60m", "immediate first pullback, stale after 60 minutes", event_variant("immediate", 60, False)),
        ("event_b_immediate_90m", "immediate first pullback, stale after 90 minutes", event_variant("immediate", 90, False)),
        ("event_d_recompression_60m", "only a new recompression event may trade after a 60 minute stale seed", event_variant("delayed_recompression_only", 60, True)),
        ("event_e_immediate_plus_recompression_60m", "immediate first pullback plus independently rebuilt recompression", event_variant("immediate_or_delayed_recompression", 60, True)),
    ]


def stage25_r_basis_configs(base: Any) -> list[tuple[str, str, Any]]:
    """R0-R3 portfolio experiments; R4 is emitted by the shadow target study."""
    baseline = stage25_invariance_configs(base)[0][2]
    baseline = replace(
        baseline,
        cmipr=replace(
            baseline.cmipr,
            entry=replace(
                baseline.cmipr.entry,
                event_structure_mode="immediate",
                immediate_max_age_minutes=45,
                delayed_recompression_enabled=False,
            ),
        ),
    )

    def exits(tp: float, basis: str) -> Any:
        return replace(
            baseline,
            cmipr=replace(
                baseline.cmipr,
                exit=replace(
                    baseline.cmipr.exit,
                    fixed_take_profit_r=tp,
                    take_profit_r_basis=basis,
                    fail_fast_r_basis=basis,
                ),
                research=replace(baseline.cmipr.research, event_diagnostics_enabled=True),
            ),
        )

    full_initial = exits(1.50, "initial_leg")
    full_initial = replace(
        full_initial,
        cmipr=replace(
            full_initial.cmipr,
            entry=replace(full_initial.cmipr.entry, initial_risk_fraction=1.0),
        ),
    )
    return [
        ("r0_campaign_baseline", "0.4 initial fraction; 1.5 campaign-R TP and 0.2 campaign-R fail-fast", replace(baseline, cmipr=replace(baseline.cmipr, research=replace(baseline.cmipr.research, event_diagnostics_enabled=True)))),
        ("r1_initial_leg_1p5", "0.4 initial fraction; TP and fail-fast use initial-leg R", exits(1.50, "initial_leg")),
        ("r2_initial_leg_2p0", "0.4 initial fraction; 2.0 initial-leg R TP and initial-leg fail-fast", exits(2.00, "initial_leg")),
        ("r3_full_initial_diagnostic", "1.0 initial fraction under unchanged campaign risk cap; 1.5 initial-leg R TP", full_initial),
    ]


def stage25_fail_fast_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Bounded structural/conditional fail-fast comparison on initial-leg R."""
    baseline = stage25_invariance_configs(base)[0][2]
    baseline = replace(
        baseline,
        cmipr=replace(
            baseline.cmipr,
            entry=replace(
                baseline.cmipr.entry,
                event_structure_mode="immediate",
                immediate_max_age_minutes=45,
                delayed_recompression_enabled=False,
            ),
            exit=replace(
                baseline.cmipr.exit,
                take_profit_r_basis="initial_leg",
                fail_fast_r_basis="initial_leg",
                fixed_take_profit_r=1.50,
            ),
            research=replace(baseline.cmipr.research, event_diagnostics_enabled=True),
        ),
    )

    def mode(name: str, minutes: int = 15) -> Any:
        return replace(
            baseline,
            cmipr=replace(
                baseline.cmipr,
                exit=replace(
                    baseline.cmipr.exit,
                    fail_fast_mode=name,
                    conditional_fail_fast_minutes=minutes,
                    conditional_fail_fast_min_mfe_r=0.20,
                    conditional_min_failed_checks=3,
                ),
            ),
        )

    return [
        ("ff_current_initial_r", "current time/lost-level logic using initial-leg R", mode("current")),
        ("ff_structural_only", "closed-5m hard invalidation only", mode("structural_only")),
        ("ff_conditional_15m", "hard invalidation plus conditional non-extension after 15 minutes", mode("conditional", 15)),
        ("ff_conditional_20m", "hard invalidation plus conditional non-extension after 20 minutes", mode("conditional", 20)),
        ("ff_conditional_30m", "hard invalidation plus conditional non-extension after 30 minutes", mode("conditional", 30)),
        ("ff_no_time", "no time-based exit; retain only hard invalidation", mode("no_time")),
    ]


def stage25_entry_revalidation_configs(base: Any) -> list[tuple[str, str, Any]]:
    """Compare entry-time checks after freezing event age, R basis, and fail-fast."""
    baseline = next(
        config
        for name, _, config in stage25_fail_fast_configs(base)
        if name == "ff_conditional_20m"
    )

    def mode(name: str) -> Any:
        return replace(
            baseline,
            cmipr=replace(
                baseline.cmipr,
                entry=replace(baseline.cmipr.entry, entry_revalidation_mode=name),
            ),
        )

    return [
        ("revalidation_none", "frozen conditional fail-fast baseline without entry-time revalidation", mode("none")),
        ("revalidation_basic", "recheck signal freshness, regime, structure, cost, and capacity", mode("basic")),
        ("revalidation_strict", "basic checks plus breadth, rank, relative strength, and extension", mode("strict")),
    ]


def run_staged_optimization(
    config_path: str,
    data_dir: str,
    trade_start: Any = None,
    trade_end: Any = None,
    progress: bool = False,
    experiment_names: tuple[str, ...] | None = None,
    profile: str = "staged",
) -> dict[str, Any]:
    base = load_live_config(config_path)
    factories = {
        "staged": staged_configs,
        "shortline": shortline_configs,
        "convex_exit": convex_exit_configs,
        "winner_pyramid": winner_pyramid_configs,
        "risk_scale": risk_scale_configs,
        "high_risk_scale": high_risk_scale_configs,
        "campaign_stage1": campaign_stage1_configs,
        "campaign_stage2": campaign_stage2_configs,
        "stage25_invariance": stage25_invariance_configs,
        "stage25_diagnostics": stage25_diagnostics_configs,
        "stage25_event_structure": stage25_event_structure_configs,
        "stage25_r_basis": stage25_r_basis_configs,
        "stage25_fail_fast": stage25_fail_fast_configs,
        "stage25_entry_revalidation": stage25_entry_revalidation_configs,
    }
    config_factory = factories[profile]
    experiments = config_factory(base)
    if experiment_names:
        known = {name for name, _, _ in experiments}
        unknown = sorted(set(experiment_names) - known)
        if unknown:
            raise ValueError(f"unknown CMIPR experiments: {', '.join(unknown)}")
        experiments = [row for row in experiments if row[0] in experiment_names]
    budget = max(1, int(base.cmipr.research.max_experiments_per_stage))
    if len(experiments) > budget:
        raise ValueError(f"CMIPR staged search has {len(experiments)} variants, above budget {budget}")

    load_start = trade_start - timedelta(days=45) if trade_start is not None else None
    load_end = trade_end + timedelta(days=3) if trade_end is not None else None
    execution = _load_symbol_data(
        data_dir,
        tuple(base.trading.symbols),
        "1m",
        start=load_start,
        end=load_end,
    )
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
    experiments = config_factory(base)
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
        "profile": profile,
        "research_status": "historical_research_not_final_holdout",
        "final_acceptance_source": base.cmipr.research.final_acceptance_source,
        "experiment_budget": budget,
        "experiments_run": len(experiments),
        "time_alignment": alignment,
        "research_data_window": {
            "requested_trade_start": trade_start.isoformat() if trade_start else None,
            "requested_trade_end": trade_end.isoformat() if trade_end else None,
            "loaded_start": load_start.isoformat() if load_start else None,
            "loaded_end": load_end.isoformat() if load_end else None,
            "warmup_days": 45 if trade_start else None,
            "post_window_embargo_days": 3 if trade_end else None,
        },
        "data_manifest": _data_manifest(data_dir, symbols),
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
    if profile == "stage25_invariance":
        output["baseline_invariance_report"] = _baseline_invariance_report(output["results"])
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
            "bull_flag_confirmed": int(stats.get("bull_flag_confirmed", 0)),
            "entry_ready_count": int(stats.get("entry_ready_count", 0)),
            "initial_fill_count": int(execution_stats.get("initial_fill_count", 0)),
        },
        "reject_reasons": report.get("reject_reasons", {}),
        "regime_observations": report.get("regime_observations", {}),
        "execution_stats": execution_stats,
        "reproducibility": {
            key: report.get(key)
            for key in (
                "strategy_config_hash",
                "cost_model_version",
                "cost_model_hash",
                "universe_version",
                "data_range",
                "feature_version",
                "event_definition_version",
                "r_definition_version",
                "random_seed",
                "cache_namespace",
            )
        },
        "r_basis_audit": report.get("r_basis_audit"),
        "convex_campaign_diagnostics": report.get("convex_campaign_diagnostics"),
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
    if previous.risk.risk_per_trade_pct != current.risk.risk_per_trade_pct:
        changed.append("trade_risk_budget")
    if previous.cmipr.research.event_diagnostics_enabled != current.cmipr.research.event_diagnostics_enabled:
        changed.append("research_diagnostics")
    return changed


def _cmipr_alpha_config(config: Any) -> dict[str, Any]:
    return {
        "mode": config.cmipr.mode,
        "allow_long": config.cmipr.allow_long,
        "allow_short": config.cmipr.allow_short,
        "risk_per_trade_pct": config.risk.risk_per_trade_pct,
        "regime": asdict(config.cmipr.regime),
        "ranking": asdict(config.cmipr.ranking),
        "compression": asdict(config.cmipr.compression),
        "ignition": asdict(config.cmipr.ignition),
        "entry": asdict(config.cmipr.entry),
        "pyramid": asdict(config.cmipr.pyramid),
        "exit": asdict(config.cmipr.exit),
        "risk_control": asdict(config.cmipr.risk_control),
        "research": asdict(config.cmipr.research),
    }


def _data_manifest(data_dir: str, symbols: tuple[str, ...]) -> dict[str, Any]:
    root = Path(data_dir)
    symbol_set = set(symbols)
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(root.glob("*_1m_*.csv")):
        symbol = path.name.split("_1m_", 1)[0]
        if symbol not in symbol_set:
            continue
        size = path.stat().st_size
        with path.open("rb") as handle:
            first = handle.read(min(size, 4096))
            if size > 4096:
                handle.seek(max(0, size - 4096))
                last = handle.read(4096)
            else:
                last = b""
        file_digest = hashlib.sha256(first + last).hexdigest()
        row = {"file": path.name, "size": size, "edge_hash": file_digest}
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "file_count": len(rows),
        "manifest_hash": digest.hexdigest(),
        "hash_method": "sha256(file_name,size,first_4k,last_4k)",
        "files": rows,
    }


def _baseline_invariance_report(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cold = results.get("stage25_baseline_cold_cache", {})
    warm = results.get("stage25_baseline_warm_cache", {})
    trade_fields = (
        "symbol",
        "side",
        "signal_time",
        "signal_available_time",
        "entry_time",
        "exit_time",
        "exit_reason",
        "quantity",
        "net_pnl",
    )
    cold_rows = [tuple(trade.get(field) for field in trade_fields) for trade in cold.get("trades", [])]
    warm_rows = [tuple(trade.get(field) for field in trade_fields) for trade in warm.get("trades", [])]
    metric_fields = (
        "final_equity",
        "net_pnl",
        "trade_count",
        "profit_factor",
        "max_drawdown_pct",
        "fee",
        "slippage_cost",
        "funding",
    )
    metric_differences = {
        field: _numeric_difference(cold.get(field), warm.get(field))
        for field in metric_fields
    }
    funnel_equal = cold.get("funnel") == warm.get("funnel")
    hashes_equal = cold.get("reproducibility") == warm.get("reproducibility")
    exact_trade_match = cold_rows == warm_rows
    tolerance = 1e-9
    metrics_within_tolerance = all(
        difference is None or abs(difference) <= tolerance
        for difference in metric_differences.values()
    )
    return {
        "passed": exact_trade_match and funnel_equal and hashes_equal and metrics_within_tolerance,
        "float_tolerance": tolerance,
        "cold_trade_count": len(cold_rows),
        "warm_trade_count": len(warm_rows),
        "exact_trade_match": exact_trade_match,
        "candidate_funnel_match": funnel_equal,
        "reproducibility_manifest_match": hashes_equal,
        "metric_differences": metric_differences,
        "cold_funnel": cold.get("funnel"),
        "warm_funnel": warm.get("funnel"),
    }


def _numeric_difference(left: Any, right: Any) -> float | None:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return None
    return float(left) - float(right)


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
    parser.add_argument(
        "--profile",
        choices=(
            "staged",
            "shortline",
            "convex_exit",
            "winner_pyramid",
            "risk_scale",
            "high_risk_scale",
            "campaign_stage1",
            "campaign_stage2",
            "stage25_invariance",
            "stage25_diagnostics",
            "stage25_event_structure",
            "stage25_r_basis",
            "stage25_fail_fast",
            "stage25_entry_revalidation",
        ),
        default="staged",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    result = run_staged_optimization(
        args.config,
        args.data_dir,
        parse_timestamp(args.trade_start) if args.trade_start else None,
        parse_timestamp(args.trade_end) if args.trade_end else None,
        args.progress,
        tuple(item.strip() for item in args.experiments.split(",") if item.strip()) or None,
        args.profile,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "diagnostic_ranking": result["diagnostic_ranking"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

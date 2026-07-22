from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.cmipr import (
    CmiprCompressionSnapshot,
    CmiprEngine,
    CmiprIgnitionSnapshot,
    CmiprMarketRegime,
    CmiprOrderLifecycle,
    CmiprState,
    audit_derivative_coverage,
)
from crypto_scalper.cmipr_campaign_diagnostics import CmiprCampaignDiagnostics
from crypto_scalper.live_config import default_live_config
from crypto_scalper.live_execution_backtest import (
    _cmipr_hard_invalidation_reason,
    _cmipr_executable_campaign_r,
    _cmipr_executable_current_r,
    _cmipr_executable_initial_leg_r,
    _cmipr_initial_quantity_within_campaign_budget,
    _cmipr_quantity_within_original_risk_budget,
    _numeric_distribution,
)
from crypto_scalper.cmipr_r_basis import (
    campaign_threshold_in_initial_leg_r,
    initial_leg_risk,
    net_pnl_r,
)
from crypto_scalper.live_portfolio_backtest import PortfolioPosition
from crypto_scalper.models import Candle, Direction, Signal
from crypto_scalper.risk import BacktestExecutionConfig, market_entry_fill, market_exit_fill
from crypto_scalper.cmipr_research import _variant_config
from crypto_scalper.cmipr_optimize import (
    campaign_stage1_configs,
    campaign_stage2_configs,
    convex_exit_configs,
    high_risk_scale_configs,
    risk_scale_configs,
    shortline_configs,
    stage25_invariance_configs,
    stage25_entry_revalidation_configs,
    staged_configs,
    winner_pyramid_configs,
)


def _candles(start: datetime, count: int, minutes: int, slope: float = 0.002) -> list[Candle]:
    output = []
    for index in range(count):
        price = 100.0 * (1.0 + slope) ** index
        output.append(Candle(start + timedelta(minutes=minutes * index), price, price * 1.002, price * 0.998, price * 1.001, 1000.0))
    return output


def _engine_config():
    base = default_live_config()
    cmipr = replace(
        base.cmipr,
        enabled=True,
        enabled_symbols=("BTCUSDT", "ETHUSDT", "ALTUSDT"),
        regime=replace(
            base.cmipr.regime,
            enter_confirmation_bars_1h=2,
            exit_confirmation_bars_1h=2,
            min_state_hold_bars_1h=0,
            enter_ema_slope_pct=0.001,
            enter_breadth_above_ema21=0.55,
            min_breadth_positive_1h=0.50,
            btc_shock_1h_pct=0.10,
        ),
    )
    return replace(base, trading=replace(base.trading, symbols=cmipr.enabled_symbols, entry_symbols=cmipr.enabled_symbols), cmipr=cmipr)


def test_regime_hysteresis_requires_distinct_closed_1h_bars() -> None:
    config = _engine_config()
    start = datetime(2026, 1, 1)
    candles = {
        "5m": {symbol: _candles(start, 400, 5) for symbol in config.trading.symbols},
        "15m": {symbol: _candles(start, 140, 15) for symbol in config.trading.symbols},
        "30m": {symbol: _candles(start, 80, 30) for symbol in config.trading.symbols},
        "1h": {symbol: _candles(start, 40, 60) for symbol in config.trading.symbols},
        "4h": {symbol: _candles(start, 12, 240) for symbol in config.trading.symbols},
    }
    engine = CmiprEngine(config, candles)

    first = engine.regime(start + timedelta(hours=31))
    repeated_same_bar = engine.regime(start + timedelta(hours=31, minutes=30))
    second = engine.regime(start + timedelta(hours=32))

    assert first.state == CmiprMarketRegime.NEUTRAL
    assert repeated_same_bar.state == CmiprMarketRegime.NEUTRAL
    assert second.state == CmiprMarketRegime.BULL_EXPANSION


def test_phase_model_confirms_early_bull_with_same_hysteresis() -> None:
    config = _engine_config()
    config = replace(
        config,
        cmipr=replace(
            config.cmipr,
            regime=replace(
                config.cmipr.regime,
                phase_model_enabled=True,
                early_min_breadth_acceleration_1h=0.0,
                early_max_breadth_above_ema21=1.0,
            ),
        ),
    )
    start = datetime(2026, 1, 1)
    candles = {
        "5m": {symbol: _candles(start, 400, 5) for symbol in config.trading.symbols},
        "15m": {symbol: _candles(start, 140, 15) for symbol in config.trading.symbols},
        "30m": {symbol: _candles(start, 80, 30) for symbol in config.trading.symbols},
        "1h": {symbol: _candles(start, 40, 60) for symbol in config.trading.symbols},
        "4h": {symbol: _candles(start, 12, 240) for symbol in config.trading.symbols},
    }
    engine = CmiprEngine(config, candles)

    first = engine.regime(start + timedelta(hours=31))
    second = engine.regime(start + timedelta(hours=32))

    assert first.state == CmiprMarketRegime.NEUTRAL
    assert second.state == CmiprMarketRegime.EARLY_BULL_EXPANSION


def test_core_model_does_not_treat_missing_derivatives_as_zero(tmp_path) -> None:
    config = _engine_config()
    cmipr = replace(
        config.cmipr,
        auxiliary_oi_data_dir=str(tmp_path / "oi"),
        auxiliary_funding_data_dir=str(tmp_path / "funding"),
        research=replace(config.cmipr.research, model_variant="core"),
    )
    config = replace(config, cmipr=cmipr)
    core = audit_derivative_coverage(config, config.trading.symbols, datetime(2025, 6, 1), datetime(2026, 6, 1))
    assert core.eligible
    enhanced_config = replace(config, cmipr=replace(cmipr, research=replace(cmipr.research, model_variant="derivatives_enhanced")))
    enhanced = audit_derivative_coverage(enhanced_config, config.trading.symbols, datetime(2025, 6, 1), datetime(2026, 6, 1))
    assert not enhanced.eligible
    assert "derivatives_history_unavailable" in enhanced.reason


def test_full_cost_current_r_is_negative_at_unchanged_price() -> None:
    config = _engine_config()
    position = PortfolioPosition(
        "BTCUSDT",
        Direction.LONG,
        1.0,
        100.0,
        98.0,
        150.0,
        0.05,
        0,
        100,
        100.0,
        entry_time=datetime(2026, 1, 1),
        raw_entry_price=100.0,
        entry_slippage_cost=0.02,
        initial_stop_price=98.0,
        risk_budget_usdt=2.0,
    )
    current_r = _cmipr_executable_current_r(
        config,
        position,
        100.0,
        datetime(2026, 1, 1, 0, 5),
        BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0),
        SymbolRules("BTCUSDT", "0.1", "0.001", "0.001", "5"),
    )
    assert current_r < 0


def test_initial_leg_risk_uses_actual_quantity_and_stop_execution_costs() -> None:
    config = _engine_config()
    position = PortfolioPosition(
        "BTCUSDT",
        Direction.LONG,
        1.0,
        100.0,
        98.0,
        150.0,
        0.05,
        0,
        100,
        100.0,
        entry_time=datetime(2026, 1, 1),
        raw_entry_price=99.98,
        entry_slippage_cost=0.02,
        initial_stop_price=98.0,
        capacity_fill_ratio=0.50,
    )
    execution = BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0, stop_slippage_bps=5.0)
    rules = SymbolRules("BTCUSDT", "0.1", "0.001", "0.001", "5")

    audit = initial_leg_risk(position, 5.0, 0.40, execution, rules)

    assert audit.initial_leg_price_risk_usdt > 2.0
    assert audit.initial_leg_full_cost_risk_usdt > audit.initial_leg_price_risk_usdt
    assert audit.initial_leg_actual_risk_fraction == audit.initial_leg_full_cost_risk_usdt / 5.0
    assert audit.capacity_clipped_initial_risk_fraction == 0.20
    assert audit.stop_execution_price_estimate < position.initial_stop_price


def test_campaign_and_initial_leg_r_are_explicit_and_not_interchangeable() -> None:
    config = _engine_config()
    position = PortfolioPosition(
        "BTCUSDT",
        Direction.LONG,
        1.0,
        100.0,
        98.0,
        150.0,
        0.05,
        0,
        100,
        100.0,
        entry_time=datetime(2026, 1, 1),
        raw_entry_price=100.0,
        entry_slippage_cost=0.02,
        initial_stop_price=98.0,
        risk_budget_usdt=5.0,
        campaign_risk_budget_usdt=5.0,
        initial_leg_full_cost_risk_usdt=2.0,
        initial_leg_actual_risk_fraction=0.40,
    )
    execution = BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0)
    rules = SymbolRules("BTCUSDT", "0.1", "0.001", "0.001", "5")

    campaign_r = _cmipr_executable_campaign_r(config, position, 101.0, datetime(2026, 1, 1, 0, 5), execution, rules)
    initial_leg_r = _cmipr_executable_initial_leg_r(config, position, 101.0, datetime(2026, 1, 1, 0, 5), execution, rules)
    legacy_alias = _cmipr_executable_current_r(config, position, 101.0, datetime(2026, 1, 1, 0, 5), execution, rules)

    assert campaign_r == legacy_alias
    assert initial_leg_r == pytest.approx(campaign_r / 0.40)
    assert campaign_threshold_in_initial_leg_r(position, 1.5) == 3.75
    assert net_pnl_r(position, 1.0, "campaign") == 0.20
    assert net_pnl_r(position, 1.0, "initial_leg") == 0.50


def test_initial_quantity_is_capped_by_full_cost_campaign_budget() -> None:
    candle = Candle(datetime(2026, 1, 1), 100.0, 100.0, 100.0, 100.0, 10_000.0)
    signal = Signal(Direction.LONG, 0.55, "cmipr", 0.02, 0.03)
    execution = BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0, stop_slippage_bps=5.0)
    rules = SymbolRules("BTCUSDT", "0.1", "0.001", "0.001", "5")
    budget = 1.0

    quantity = _cmipr_initial_quantity_within_campaign_budget(
        signal,
        candle,
        1.0,
        budget,
        execution,
        rules,
    )

    assert 0.0 < quantity < 1.0
    entry_fill = market_entry_fill(execution, rules, Direction.LONG, quantity, candle.open, candle.volume * candle.open)
    raw_stop = entry_fill.price * (1.0 - signal.stop_loss_pct)
    stop_fill = market_exit_fill(
        execution,
        rules,
        Direction.LONG,
        quantity,
        raw_stop,
        "stop_market",
        candle.volume * candle.open,
    )
    price_risk = quantity * (entry_fill.price - stop_fill.price)
    assert price_risk + entry_fill.fee + stop_fill.fee <= budget + 1e-9


def test_r_basis_distribution_reports_required_percentiles() -> None:
    distribution = _numeric_distribution([1.0, 2.0, 3.0, 4.0, None])

    assert distribution == {
        "count": 4,
        "min": 1.0,
        "p10": pytest.approx(1.3),
        "p25": pytest.approx(1.75),
        "median": 2.5,
        "p75": pytest.approx(3.25),
        "p90": pytest.approx(3.7),
        "max": 4.0,
    }


def test_addon_cannot_exceed_original_full_cost_risk_budget() -> None:
    config = _engine_config()
    position = PortfolioPosition(
        "BTCUSDT",
        Direction.LONG,
        1.0,
        100.0,
        101.0,
        150.0,
        0.05,
        0,
        100,
        105.0,
        entry_time=datetime(2026, 1, 1),
        raw_entry_price=100.0,
        entry_slippage_cost=0.02,
        initial_stop_price=98.0,
        risk_budget_usdt=0.01,
    )
    signal = Signal(Direction.LONG, 0.55, "cmipr addon", 0.02, 0.25)
    candle = Candle(datetime(2026, 1, 1, 0, 10), 105.0, 105.0, 105.0, 105.0, 1000.0)
    quantity = _cmipr_quantity_within_original_risk_budget(
        config,
        position,
        signal,
        candle,
        1.0,
        101.0,
        candle.timestamp,
        BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0, stop_slippage_bps=5.0),
        SymbolRules("BTCUSDT", "0.1", "0.001", "0.001", "5"),
    )
    assert quantity == 0.0


def test_operational_state_machine_contains_protection_and_restart_states() -> None:
    assert CmiprState.ORDER_PENDING.value == "ORDER_PENDING"
    assert CmiprState.PARTIAL_FILL.value == "PARTIAL_FILL"
    assert CmiprState.PROTECTION_PENDING.value == "PROTECTION_PENDING"
    assert CmiprState.PROTECTED.value == "PROTECTED"
    assert CmiprState.CANCEL_PENDING.value == "CANCEL_PENDING"
    assert CmiprState.RECOVERY_AFTER_RESTART.value == "RECOVERY_AFTER_RESTART"


def test_protection_failure_requires_emergency_reduce_or_close() -> None:
    lifecycle = CmiprOrderLifecycle("BTCUSDT")
    lifecycle.submit_entry(1.0)
    lifecycle.record_fill(0.4)
    assert lifecycle.state == CmiprState.PARTIAL_FILL
    lifecycle.request_protection()
    action = lifecycle.protection_result(False, "exchange_rejected_stop")
    assert action == "emergency_reduce_or_close"
    assert lifecycle.state == CmiprState.EXITING


def test_restart_without_protection_requires_replacement_or_flatten() -> None:
    lifecycle = CmiprOrderLifecycle("BTCUSDT")
    action = lifecycle.recover_after_restart(1.0, protective_stop_present=False)
    assert action == "replace_protection_or_emergency_flatten"
    assert lifecycle.state == CmiprState.PROTECTION_PENDING


def test_higher_cost_stress_really_increases_cost_parameters() -> None:
    config = _engine_config()
    stressed, experiment = _variant_config(config, "higher_cost")
    assert experiment == "full_cost"
    assert stressed.risk.taker_fee_rate > config.risk.taker_fee_rate
    assert stressed.risk.market_slippage_bps > config.risk.market_slippage_bps
    assert stressed.risk.stop_slippage_bps > config.risk.stop_slippage_bps
    assert stressed.risk.impact_coefficient_bps > config.risk.impact_coefficient_bps


def test_staged_optimization_changes_one_module_at_a_time() -> None:
    rows = staged_configs(_engine_config())
    expected = {
        "stage1_regime": ["regime"],
        "stage2_ranking": ["ranking"],
        "stage3a_compression": ["compression"],
        "stage3b_ignition": ["ignition"],
        "stage4_pullback": ["entry"],
    }
    previous = rows[0][2]
    for name, _, config in rows[1:6]:
        changed = [
            module
            for module in ("regime", "ranking", "compression", "ignition", "entry", "pyramid", "exit", "risk_control")
            if getattr(previous.cmipr, module) != getattr(config.cmipr, module)
        ]
        assert changed == expected[name]
        previous = config


def test_feature_cache_keys_include_module_config() -> None:
    config = _engine_config()
    changed = replace(
        config,
        cmipr=replace(
            config.cmipr,
            ranking=replace(config.cmipr.ranking, long_top_fraction=0.25),
            regime=replace(config.cmipr.regime, enter_ema_slope_pct=0.0005),
            compression=replace(config.cmipr.compression, max_atr_percentile=0.60),
        ),
    )
    start = datetime(2026, 1, 1)
    candles = {
        "5m": {symbol: _candles(start, 1200, 5) for symbol in config.trading.symbols},
        "15m": {symbol: _candles(start, 500, 15) for symbol in config.trading.symbols},
        "30m": {symbol: _candles(start, 300, 30) for symbol in config.trading.symbols},
        "1h": {symbol: _candles(start, 150, 60) for symbol in config.trading.symbols},
        "4h": {symbol: _candles(start, 50, 240) for symbol in config.trading.symbols},
    }
    cache = {}
    decision_time = start + timedelta(hours=140)
    base_engine = CmiprEngine(config, candles, shared_feature_cache=cache)
    changed_engine = CmiprEngine(changed, candles, shared_feature_cache=cache)

    base_engine.rankings(decision_time)
    changed_engine.rankings(decision_time)
    base_engine._raw_regime_snapshot(decision_time)
    changed_engine._raw_regime_snapshot(decision_time)
    base_engine._compression("ALTUSDT", decision_time)
    changed_engine._compression("ALTUSDT", decision_time)

    assert base_engine.cache_namespace != changed_engine.cache_namespace
    assert sum(key[1] == "rankings" for key in cache) == 2
    assert sum(key[1] == "raw_regime" for key in cache) == 2
    assert sum(key[1] == "compression" for key in cache) == 2
    assert {key[0] for key in cache} == {base_engine.cache_namespace, changed_engine.cache_namespace}


def test_cmipr_shortline_timeframes_are_configurable_without_changing_defaults() -> None:
    config = _engine_config()
    assert config.cmipr.compression.timeframe == "30m"
    assert config.cmipr.ignition.timeframe == "15m"

    shortline = replace(
        config,
        cmipr=replace(
            config.cmipr,
            compression=replace(config.cmipr.compression, timeframe="15m"),
            ignition=replace(config.cmipr.ignition, timeframe="5m"),
        ),
    )

    assert shortline.cmipr.compression.timeframe == "15m"
    assert shortline.cmipr.ignition.timeframe == "5m"


def test_shortline_search_changes_one_alpha_module_per_stage() -> None:
    rows = shortline_configs(_engine_config())
    expected = {
        "short1_compression_15m": ["compression"],
        "short2_ignition_5m": ["ignition"],
        "short3_fixed_1p5r": ["entry"],
    }
    previous = rows[0][2]
    for name, _, config in rows[1:4]:
        changed = [
            module
            for module in ("regime", "ranking", "compression", "ignition", "entry", "pyramid", "exit", "risk_control")
            if getattr(previous.cmipr, module) != getattr(config.cmipr, module)
        ]
        assert changed == expected[name]
        previous = config


def test_convex_exit_search_keeps_entry_funnel_frozen() -> None:
    config = _engine_config()
    rows = convex_exit_configs(config)
    frozen_entry = rows[0][2].cmipr.entry
    for _, _, candidate in rows:
        assert candidate.cmipr.regime == config.cmipr.regime
        assert candidate.cmipr.ranking == config.cmipr.ranking
        assert candidate.cmipr.compression == config.cmipr.compression
        assert candidate.cmipr.ignition == config.cmipr.ignition
        assert candidate.cmipr.entry == frozen_entry
        assert candidate.cmipr.entry.cost_guard_target_r == 1.20
        assert candidate.cmipr.pyramid == config.cmipr.pyramid


def test_winner_pyramid_search_never_raises_original_trade_risk_budget() -> None:
    config = _engine_config()
    rows = winner_pyramid_configs(config)
    for _, _, candidate in rows:
        assert candidate.risk == config.risk
        assert candidate.cmipr.entry.initial_risk_fraction == config.cmipr.entry.initial_risk_fraction
        assert candidate.cmipr.pyramid.max_addons == 1
        assert candidate.cmipr.pyramid.require_full_cost_current_r
        assert candidate.cmipr.exit.runner_activation_r == candidate.cmipr.exit.fixed_take_profit_r == 2.0


def test_risk_scale_search_changes_only_trade_risk_budget() -> None:
    config = _engine_config()

    rows = risk_scale_configs(config)

    assert [candidate.risk.risk_per_trade_pct for _, _, candidate in rows] == [0.005, 0.0075, 0.01, 0.015, 0.02]
    assert all(candidate.cmipr == config.cmipr for _, _, candidate in rows)
    assert all(replace(candidate, risk=config.risk) == config for _, _, candidate in rows)


def test_high_risk_scale_search_caps_maintenance_margin_ratio() -> None:
    config = _engine_config()

    rows = high_risk_scale_configs(config)

    assert [candidate.risk.risk_per_trade_pct for _, _, candidate in rows] == [0.02, 0.03, 0.05, 0.075, 0.10]
    assert all(candidate.risk.max_maintenance_margin_ratio_pct == 0.05 for _, _, candidate in rows)
    assert all(candidate.risk.estimated_maintenance_margin_rate == 0.005 for _, _, candidate in rows)
    assert all(candidate.cmipr == config.cmipr for _, _, candidate in rows)


def test_campaign_event_diagnostics_are_research_only_and_path_aware() -> None:
    start = datetime(2026, 1, 1, 0, 15)
    execution = []
    for index in range(300):
        price = 100.0 + index * 0.02
        execution.append(
            Candle(
                start + timedelta(minutes=index),
                price,
                price + 0.05,
                price - 0.03,
                price + 0.02,
                1000.0,
            )
        )
    mtf_15m = [execution[index] for index in range(0, len(execution), 15)]
    diagnostics = CmiprCampaignDiagnostics(
        enabled=True,
        minimum_bucket_size=30,
        full_cost_pct=0.0017,
        ignition_timeframe="15m",
    )
    regime_state = SimpleNamespace(value="BULL_EXPANSION")
    regime = SimpleNamespace(
        state=regime_state,
        raw_state=regime_state,
        btc_1h_return=0.01,
        btc_4h_return=0.02,
        eth_1h_return=0.008,
        breadth_above_ema21=0.60,
        breadth_positive_1h=0.58,
    )
    previous_regime = SimpleNamespace(**{**vars(regime), "breadth_above_ema21": 0.52})
    ignition = SimpleNamespace(
        direction=Direction.LONG,
        candle=Candle(start - timedelta(minutes=15), 99.5, 100.1, 99.4, 100.0, 2000.0),
        ranking_percentile=0.92,
        ranking_score=0.05,
        breakout_level=99.8,
        breakout_distance_atr=0.4,
        body_atr=0.8,
        volume_ratio=2.0,
        wick_ratio=0.1,
        close_position=0.9,
        stop_price=98.0,
        stop_loss_pct=0.02,
    )
    compression = SimpleNamespace(
        atr_value=1.0,
        atr_percentile=0.30,
        atr_to_average=0.80,
        channel_width_atr=3.0,
        volume_contraction=0.70,
        prior_move_atr=1.0,
        failed_breakouts=0,
    )

    diagnostics.record_ignition(
        "event-1",
        "BTCUSDT",
        start,
        regime,
        previous_regime,
        ignition,
        compression,
        (0.03, 0.80, 1.0),
        0.95,
        12,
    )
    diagnostics.record_ignition(
        "event-1",
        "BTCUSDT",
        start + timedelta(minutes=5),
        regime,
        previous_regime,
        ignition,
        compression,
        (0.03, 0.80, 1.0),
        0.95,
        12,
    )
    diagnostics.update_pullback("event-1", execution[5], 0.4, 0.6, 0.8, 0.2, True)
    diagnostics.mark("event-1", "entry_ready", entry_mode="first_pullback")
    report = diagnostics.finalize(
        {"BTCUSDT": execution},
        {"15m": {"BTCUSDT": mtf_15m}},
        [],
    )

    assert report["research_only"]
    assert report["does_not_affect_execution"]
    assert report["ignition_observation_count"] == 2
    assert report["unique_ignition_event_count"] == 1
    row = report["ignition_events"][0]
    assert row["future_mfe_240m_r"] > row["future_mfe_15m_r"]
    assert row["pullback_shadow_entry_time"] == execution[11].timestamp.isoformat()
    assert row["pullback_shadow_initial_leg_risk_usdt"] > 0
    assert row["pullback_shadow_campaign_risk_usdt"] == pytest.approx(0.8)
    assert row["first_pullback_quality"] is not None
    assert report["bucket_report"]["regime"]["eligible"] == {}
    assert report["bucket_report"]["regime"]["excluded_small_sample"]["BULL_EXPANSION"] == 1


def test_stale_pending_is_consumed_and_cannot_reenter() -> None:
    config = _engine_config()
    config = replace(
        config,
        cmipr=replace(
            config.cmipr,
            entry=replace(
                config.cmipr.entry,
                event_structure_mode="immediate",
                immediate_max_age_minutes=30,
            ),
        ),
    )
    engine = CmiprEngine(config, {timeframe: {} for timeframe in ("5m", "15m", "30m", "1h", "4h")})
    runtime = SimpleNamespace(
        state=CmiprState.LONG_IGNITION_PENDING,
        event_id="old-event",
        direction=Direction.LONG,
        ignition=object(),
        pending_time=datetime(2026, 1, 1),
        expire_time=datetime(2026, 1, 1, 0, 30),
        last_processed_5m=datetime(2026, 1, 1, 0, 25),
        consumed_event_ids=set(),
        event_type="IGNITION",
        stale_parent_event_id=None,
        stale_at=None,
    )

    engine._expire_stale_pending(runtime, datetime(2026, 1, 1, 0, 31))

    assert "old-event" in runtime.consumed_event_ids
    assert runtime.event_id is None
    assert runtime.ignition is None
    assert runtime.last_processed_5m is None
    assert runtime.stale_parent_event_id == "old-event"


def test_delayed_recompression_creates_new_event_with_new_structure() -> None:
    config = _engine_config()
    config = replace(
        config,
        cmipr=replace(
            config.cmipr,
            entry=replace(
                config.cmipr.entry,
                event_structure_mode="immediate_or_delayed_recompression",
                delayed_recompression_enabled=True,
                immediate_max_age_minutes=30,
            ),
        ),
    )
    start = datetime(2026, 1, 1)
    candles = {
        timeframe: {symbol: _candles(start, 80, minutes) for symbol in config.trading.symbols}
        for timeframe, minutes in (("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60), ("4h", 240))
    }
    engine = CmiprEngine(config, candles)
    symbol = "ALTUSDT"
    runtime = engine.runtime[symbol]
    runtime.stale_parent_event_id = "ALTUSDT:LONG:old"
    runtime.stale_at = start
    regime = SimpleNamespace(
        state=CmiprMarketRegime.BULL_EXPANSION,
        raw_state=CmiprMarketRegime.BULL_EXPANSION,
        btc_1h_return=0.01,
        btc_4h_return=0.02,
        eth_1h_return=0.01,
        breadth_above_ema21=0.7,
        breadth_positive_1h=0.7,
    )
    new_candle = Candle(start + timedelta(minutes=30), 100, 103, 99, 102, 2000)
    new_ignition = CmiprIgnitionSnapshot(
        Direction.LONG,
        new_candle,
        101.0,
        0.5,
        0.8,
        0.8,
        0.1,
        2.0,
        0.05,
        0.9,
        0.8,
        2.0,
        98.0,
        0.04,
        0.25,
    )
    new_compression = CmiprCompressionSnapshot(2.0, 0.2, 0.7, 2.0, 0.6, 0.5, 0, 101.0, 97.0)
    engine.regime = lambda _: regime
    engine.rankings = lambda _: {candidate: (0.05, 0.9, 1.0) for candidate in config.trading.symbols}
    engine._ignition = lambda candidate, *_: ((new_ignition, "") if candidate == symbol else (None, "test_skip"))
    engine._compression = lambda *_: (new_compression, "")
    engine._raw_regime_snapshot = lambda _: regime
    engine._liquidity_percentile = lambda *_: 0.9

    engine.scan(start + timedelta(minutes=60), set())

    assert runtime.event_id is not None and runtime.event_id.endswith(":RECOMP1")
    assert runtime.event_id != runtime.stale_parent_event_id
    assert runtime.ignition is new_ignition
    assert runtime.ignition.setup_atr_value == 2.0
    assert runtime.ignition.breakout_level == 101.0
    assert engine.stats["delayed_recompression_count"] == 1


def test_strict_entry_revalidation_rejects_stale_visible_signal() -> None:
    config = _engine_config()
    config = replace(
        config,
        cmipr=replace(
            config.cmipr,
            entry=replace(
                config.cmipr.entry,
                entry_revalidation_mode="strict",
                immediate_max_age_minutes=30,
            ),
        ),
    )
    engine = CmiprEngine(config, {timeframe: {} for timeframe in ("5m", "15m", "30m", "1h", "4h")})
    ignition = CmiprIgnitionSnapshot(
        Direction.LONG,
        Candle(datetime(2026, 1, 1), 100, 102, 99, 101, 1000),
        100.5,
        0.5,
        0.8,
        0.8,
        0.1,
        2.0,
        0.05,
        0.9,
        0.8,
        2.0,
        98.0,
        0.03,
        0.25,
    )
    runtime = SimpleNamespace(
        pending_time=datetime(2026, 1, 1),
        ignition_breadth=0.7,
        ignition_rank_percentile=0.9,
        ignition_relative_strength=0.05,
    )
    regime = SimpleNamespace(state=CmiprMarketRegime.BULL_EXPANSION, breadth_above_ema21=0.7)

    reason = engine._entry_revalidation_reason(
        runtime,
        regime,
        (0.06, 0.9, 1.0),
        Candle(datetime(2026, 1, 1, 0, 35), 101, 102, 100, 101.5, 900),
        ignition,
        5.0,
        datetime(2026, 1, 1, 0, 35),
    )

    assert reason == "stale_ignition_event"


def test_hard_invalidation_uses_only_closed_5m_and_precedes_time_logic() -> None:
    start = datetime(2026, 1, 1)
    candles = [
        Candle(start, 101, 102, 100, 101, 1000),
        Candle(start + timedelta(minutes=5), 101, 101.5, 98.5, 99, 1000),
        Candle(start + timedelta(minutes=10), 99, 103, 98, 102, 1000),
    ]
    by_timeframe = {"5m": {"BTCUSDT": candles}}
    timestamps = {"5m": {"BTCUSDT": [candle.timestamp for candle in candles]}}
    position = SimpleNamespace(symbol="BTCUSDT", direction=Direction.LONG)

    before_second_bar_close = _cmipr_hard_invalidation_reason(
        position,
        Candle(start + timedelta(minutes=9), 101, 101, 101, 101, 1),
        100.0,
        by_timeframe,
        timestamps,
    )
    after_second_bar_close = _cmipr_hard_invalidation_reason(
        position,
        Candle(start + timedelta(minutes=10), 99, 99, 99, 99, 1),
        100.0,
        by_timeframe,
        timestamps,
    )

    assert before_second_bar_close == ""
    assert after_second_bar_close == "cmipr_hard_invalidation_lost_breakout"


def test_campaign_stage1_changes_only_research_diagnostics() -> None:
    config = _engine_config()

    control, diagnostics = [candidate for _, _, candidate in campaign_stage1_configs(config)]

    assert not control.cmipr.research.event_diagnostics_enabled
    assert diagnostics.cmipr.research.event_diagnostics_enabled
    assert replace(
        diagnostics,
        cmipr=replace(diagnostics.cmipr, research=control.cmipr.research),
    ) == control


def test_campaign_stage2_holds_exit_and_risk_fixed_while_changing_one_alpha_module() -> None:
    config = _engine_config()
    rows = campaign_stage2_configs(config)

    assert len(rows) == 5
    assert all(candidate.risk == config.risk for _, _, candidate in rows)
    assert all(not candidate.cmipr.pyramid.enabled for _, _, candidate in rows)
    assert all(candidate.cmipr.pyramid.max_addons == 0 for _, _, candidate in rows)
    assert all(not candidate.cmipr.exit.runner_enabled for _, _, candidate in rows)
    assert all(candidate.cmipr.exit.fixed_take_profit_r == 1.5 for _, _, candidate in rows)
    expected_changes = ["regime", "ranking", "entry", "entry"]
    for expected, previous, current in zip(expected_changes, rows, rows[1:]):
        changed = [
            module
            for module in ("regime", "ranking", "compression", "ignition", "entry")
            if getattr(previous[2].cmipr, module) != getattr(current[2].cmipr, module)
        ]
        assert changed == [expected]


def test_stage25_invariance_profile_is_identical_and_disables_deferred_modules() -> None:
    rows = stage25_invariance_configs(_engine_config())
    cold = rows[0][2]
    warm = rows[1][2]

    assert cold == warm
    assert cold.cmipr.entry.mode == "first_pullback"
    assert cold.cmipr.entry.event_structure_mode == "current"
    assert not cold.cmipr.allow_short
    assert not cold.cmipr.ranking.strength_acceleration_enabled
    assert not cold.cmipr.pyramid.enabled
    assert cold.cmipr.pyramid.max_addons == 0
    assert not cold.cmipr.exit.runner_enabled
    assert cold.cmipr.exit.fixed_take_profit_r == 1.5
    assert cold.cmipr.exit.take_profit_r_basis == "campaign"
    assert cold.cmipr.exit.fail_fast_r_basis == "campaign"


def test_stage25_entry_revalidation_changes_only_entry_checks() -> None:
    rows = stage25_entry_revalidation_configs(_engine_config())
    assert [candidate.cmipr.entry.entry_revalidation_mode for _, _, candidate in rows] == [
        "none",
        "basic",
        "strict",
    ]
    baseline = rows[0][2]
    for _, _, candidate in rows:
        assert candidate.cmipr.entry.event_structure_mode == "immediate"
        assert candidate.cmipr.entry.immediate_max_age_minutes == 45
        assert candidate.cmipr.exit.take_profit_r_basis == "initial_leg"
        assert candidate.cmipr.exit.fail_fast_mode == "conditional"
        assert candidate.cmipr.exit.conditional_fail_fast_minutes == 20
        assert not candidate.cmipr.allow_short
        assert not candidate.cmipr.pyramid.enabled
        assert candidate.cmipr.pyramid.max_addons == 0
        assert not candidate.cmipr.exit.runner_enabled
        assert replace(
            candidate,
            cmipr=replace(candidate.cmipr, entry=baseline.cmipr.entry),
        ) == baseline

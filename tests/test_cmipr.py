from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.cmipr import CmiprEngine, CmiprMarketRegime, CmiprOrderLifecycle, CmiprState, audit_derivative_coverage
from crypto_scalper.live_config import default_live_config
from crypto_scalper.live_execution_backtest import _cmipr_executable_current_r, _cmipr_quantity_within_original_risk_budget
from crypto_scalper.live_portfolio_backtest import PortfolioPosition
from crypto_scalper.models import Candle, Direction, Signal
from crypto_scalper.risk import BacktestExecutionConfig
from crypto_scalper.cmipr_research import _variant_config


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

from __future__ import annotations

from array import array
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.models import Candle, Direction
from crypto_scalper.mtf_structure_derivatives import (
    FeatureBundle,
    RawSetup,
    SignalFilterConfig,
    StructureFeatureConfig,
    TrendState,
    _load_derivatives,
    _resample_1m_to_15m,
    build_frame_features,
    resample_candles,
    setup_passes,
)
from crypto_scalper.mtf_structure_derivatives_backtest import (
    AtrTimeline,
    ExitConfig,
    PortfolioConfig,
    run_backtest,
    simulate_unit_outcome,
)
from crypto_scalper.risk import BacktestExecutionConfig
from crypto_scalper.volatility_breakout_optimize import CompactSeries, minute_token


START = datetime(2026, 1, 1)
RULES = SymbolRules("TESTUSDT", Decimal("0.01"), Decimal("0.01"), Decimal("0.01"), Decimal("5"))
NO_COST = BacktestExecutionConfig(
    market_slippage_bps=0.0,
    stop_slippage_bps=0.0,
    take_profit_slippage_bps=0.0,
    maker_fee_rate=0.0,
    taker_fee_rate=0.0,
    funding_enabled=False,
)


def candle(offset: int, open_: float, high: float, low: float, close: float) -> Candle:
    return Candle(START + timedelta(minutes=offset), open_, high, low, close, 100.0)


def series(rows: list[Candle]) -> CompactSeries:
    return CompactSeries(
        array("q", (minute_token(item.timestamp) for item in rows)),
        array("d", (item.open for item in rows)),
        array("d", (item.high for item in rows)),
        array("d", (item.low for item in rows)),
        array("d", (item.close for item in rows)),
        array("d", (item.volume for item in rows)),
    )


def setup(direction: Direction = Direction.LONG) -> RawSetup:
    return RawSetup(
        event_id=f"event-{direction.name}",
        symbol="TESTUSDT",
        direction=direction,
        signal_bar_time=START - timedelta(minutes=15),
        signal_available_time=START,
        raw_signal_price=100.0,
        atr_15m=1.0,
        sweep_price=99.0 if direction == Direction.LONG else 101.0,
        sweep_age_bars=0,
        trigger_kind="bos",
        four_h_spread_atr=1.0,
        four_h_efficiency=0.5,
        four_h_slope_atr=0.2,
        four_h_bos_age=1,
        four_h_structure_aligned=True,
        one_h_spread_atr=1.0,
        one_h_efficiency=0.5,
        one_h_slope_atr=0.2,
        one_h_structure_age=1,
        one_h_structure_aligned=True,
        entry_extension_atr=0.0,
        volume_ratio=1.5,
        oi_change_30m=0.01,
        directional_taker_imbalance=0.1,
        directional_cvd=0.1,
        directional_funding_rate=0.0,
        quality_score=5.0,
    )


def exit_config(**changes: object) -> ExitConfig:
    values = dict(
        min_stop_atr=1.0,
        max_stop_atr=1.0,
        stop_buffer_atr=0.0,
        max_stop_pct=0.10,
        tp1_r=1.0,
        tp1_fraction=0.50,
        tp2_r=2.0,
        tp2_fraction=0.25,
        breakeven_after_tp1=True,
        trailing_activation_r=10.0,
        trend_exit_mode="hybrid",
        trend_exit_confirm_bars=2,
        max_holding_minutes=1_000,
    )
    values.update(changes)
    return ExitConfig(**values)


def timeline() -> AtrTimeline:
    return AtrTimeline((minute_token(START),), (1.0,))


def test_resampled_frames_are_available_only_after_the_full_bar_closes() -> None:
    rows = [candle(index * 15, 100 + index, 101 + index, 99 + index, 100.5 + index) for index in range(8)]
    hourly = resample_candles(rows, 60)
    assert [item.timestamp for item in hourly] == [START, START + timedelta(hours=1)]
    features = build_frame_features(hourly, 60, 2, 3, StructureFeatureConfig())
    assert features.available_times == (
        START + timedelta(hours=1),
        START + timedelta(hours=2),
    )


def test_confirmed_swing_does_not_use_candles_after_confirmation_bar() -> None:
    rows = [
        candle(0, 10, 11, 9, 10),
        candle(15, 10, 12, 9.5, 11),
        candle(30, 11, 15, 10, 12),
        candle(45, 12, 13, 10.5, 11),
        candle(60, 11, 12, 10, 11),
        candle(75, 11, 16, 10, 15.5),
    ]
    config = StructureFeatureConfig(swing_left=2, swing_right=2)
    features = build_frame_features(rows, 15, 2, 3, config)
    assert features.break_direction[3] == 0
    assert features.break_direction[4] == 0
    assert features.break_direction[5] == 1


def test_derivatives_observation_has_five_minute_publication_lag(tmp_path) -> None:
    oi = tmp_path / "oi.csv"
    taker = tmp_path / "taker.csv"
    timestamps = [START + timedelta(minutes=5 * index) for index in range(7)]
    oi.write_text(
        "timestamp,sumOpenInterestValue\n"
        + "".join(f"{stamp.isoformat()},100\n" for stamp in timestamps),
        encoding="utf-8",
    )
    taker.write_text(
        "timestamp,buySellRatio\n"
        + "".join(f"{stamp.isoformat()},2\n" for stamp in timestamps),
        encoding="utf-8",
    )
    rows = _load_derivatives(oi, taker)
    assert rows[0].available_time == START + timedelta(minutes=5)
    assert rows[-1].available_time == timestamps[-1] + timedelta(minutes=5)
    assert rows[-1].oi_change_30m == pytest.approx(0.0)
    assert rows[-1].taker_imbalance == pytest.approx(1.0 / 3.0)


def test_one_minute_bridge_drops_incomplete_15m_bucket() -> None:
    rows = [candle(index, 100, 101, 99, 100) for index in range(16)]
    aggregated = _resample_1m_to_15m(rows)
    assert len(aggregated) == 1
    assert aggregated[0].timestamp == START
    assert aggregated[0].volume == pytest.approx(1_500.0)


def test_new_chop_and_trigger_filters_are_enforced() -> None:
    candidate = replace(
        setup(),
        trigger_kind="micro_bos",
        four_h_structure_aligned=False,
        entry_extension_atr=-2.0,
        volume_ratio=5.0,
    )
    assert not setup_passes(
        candidate,
        SignalFilterConfig(require_four_h_structure_alignment=True),
    )
    assert not setup_passes(
        candidate,
        SignalFilterConfig(allow_micro_bos_trigger=False),
    )
    assert not setup_passes(
        candidate,
        SignalFilterConfig(min_entry_extension_atr=-1.0),
    )
    assert not setup_passes(
        candidate,
        SignalFilterConfig(max_volume_ratio=4.0),
    )


def test_same_bar_stop_is_resolved_before_take_profit() -> None:
    rows = [candle(0, 100, 102, 98, 101)]
    outcome = simulate_unit_outcome(
        setup(), series(rows), {}, timeline(), (), RULES, exit_config(), NO_COST
    )
    assert outcome is not None
    assert len(outcome.legs) == 1
    assert outcome.legs[0].reason == "atr_stop"
    assert outcome.legs[0].fraction == pytest.approx(1.0)
    assert outcome.net_pnl_per_unit == pytest.approx(-1.0)


def test_partial_take_profit_then_breakeven_stop_reconciles_cash() -> None:
    rows = [
        candle(0, 100, 101.5, 99.5, 101),
        candle(15, 101, 101.2, 100, 100.5),
    ]
    outcome = simulate_unit_outcome(
        setup(), series(rows), {}, timeline(), (), RULES, exit_config(), NO_COST
    )
    assert outcome is not None
    assert [(item.reason, item.fraction) for item in outcome.legs] == [
        ("tp1", pytest.approx(0.5)),
        ("atr_stop", pytest.approx(0.5)),
    ]
    assert outcome.net_pnl_per_unit == pytest.approx(0.5)


def test_confirmed_trend_end_exits_at_next_bar_open() -> None:
    rows = [
        candle(0, 100, 100.5, 99.5, 100.2),
        candle(15, 100.3, 100.4, 100.1, 100.2),
    ]
    minute = minute_token(START + timedelta(minutes=15))
    states = {
        minute: TrendState(
            available_time=START + timedelta(minutes=15),
            fast_invalid_long=True,
            fast_invalid_short=False,
            slow_invalid_long=True,
            slow_invalid_short=False,
            structure_invalid_long=True,
            structure_invalid_short=False,
        )
    }
    outcome = simulate_unit_outcome(
        setup(),
        series(rows),
        states,
        timeline(),
        (),
        RULES,
        exit_config(trend_exit_confirm_bars=1),
        NO_COST,
    )
    assert outcome is not None
    assert outcome.legs[-1].reason == "trend_end"
    assert outcome.legs[-1].phase == 0
    assert outcome.legs[-1].fill_price == pytest.approx(100.3)


def test_portfolio_rejects_third_simultaneous_entry() -> None:
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
    price_rows = [candle(0, 100, 102, 98, 101)]
    setups = tuple(
        replace(setup(), event_id=f"event-{symbol}", symbol=symbol)
        for symbol in symbols
    )
    bundle = FeatureBundle(
        symbols=symbols,
        candles_15m={symbol: tuple(price_rows) for symbol in symbols},
        raw_setups=setups,
        trend_states={symbol: {} for symbol in symbols},
        funding={symbol: () for symbol in symbols},
        source_files={symbol: {} for symbol in symbols},
        cvd_definition="test proxy",
    )
    result = run_backtest(
        bundle,
        SignalFilterConfig(),
        exit_config(),
        PortfolioConfig(risk_per_trade_pct=0.01, max_open_positions=2),
        START,
        START + timedelta(minutes=15),
        execution_series={symbol: series(price_rows) for symbol in symbols},
        execution=NO_COST,
        atr_timelines={symbol: timeline() for symbol in symbols},
        rules={symbol: replace(RULES, symbol=symbol) for symbol in symbols},
    )
    assert result.trade_count == 2
    assert result.rejected["max_open_positions"] == 1

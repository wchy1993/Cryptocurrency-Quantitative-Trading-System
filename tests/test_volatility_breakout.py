from __future__ import annotations

from array import array
from dataclasses import replace
from datetime import datetime, timedelta

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.models import Candle, Direction
from crypto_scalper.risk import BacktestExecutionConfig
from crypto_scalper.volatility_breakout import (
    DualThrustSignal,
    VolatilityBreakoutConfig,
    build_dual_thrust_signals,
    dual_thrust_event_id,
)
from crypto_scalper.volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    PortfolioSearchConfig,
    _market_filter_reject_reason,
    minute_token,
    simulate_portfolio,
)
from crypto_scalper.volatility_breakout_v2_optimize import build_market_context


def _candle(timestamp: datetime, open_: float, high: float, low: float, close: float, volume: float = 10.0) -> Candle:
    return Candle(timestamp, open_, high, low, close, volume)


def test_dual_thrust_uses_only_prior_closed_days_and_next_bar_availability() -> None:
    candles = [
        _candle(datetime(2026, 1, 1, 0, 0), 100, 110, 90, 105),
        _candle(datetime(2026, 1, 1, 0, 30), 105, 108, 100, 104),
        _candle(datetime(2026, 1, 2, 0, 0), 104, 112, 98, 100),
        _candle(datetime(2026, 1, 2, 0, 30), 100, 105, 99, 101),
        _candle(datetime(2026, 1, 3, 0, 0), 100, 106, 99, 106),
        _candle(datetime(2026, 1, 3, 0, 30), 106, 120, 105, 109),
    ]
    config = VolatilityBreakoutConfig(
        timeframe_minutes=30,
        lookback_days=2,
        long_k=0.5,
        short_k=0.5,
        allow_short=False,
    )
    signals = build_dual_thrust_signals("BTCUSDT", candles, config)

    assert len(signals) == 1
    signal = signals[0]
    # Previous two completed days produce max(112 - 101, 104 - 90) = 14.
    assert signal.dual_thrust_range == 14.0
    assert signal.upper_band == 107.0
    assert signal.signal_bar_time == datetime(2026, 1, 3, 0, 30)
    assert signal.signal_available_time == datetime(2026, 1, 3, 1, 0)

    future = candles + [_candle(datetime(2026, 1, 4), 109, 200, 50, 150)]
    repeated = build_dual_thrust_signals("BTCUSDT", future, config)[0]
    assert repeated.event_id == signal.event_id
    assert repeated.upper_band == signal.upper_band


def test_dual_thrust_event_id_changes_with_range_parameters() -> None:
    timestamp = datetime(2026, 1, 3)
    first = VolatilityBreakoutConfig(lookback_days=3, long_k=0.4)
    second = VolatilityBreakoutConfig(lookback_days=5, long_k=0.4)
    assert dual_thrust_event_id("ETHUSDT", Direction.LONG, timestamp, first) != dual_thrust_event_id(
        "ETHUSDT", Direction.LONG, timestamp, second
    )


def test_path_aware_same_bar_conflict_exits_at_stop_first() -> None:
    start = datetime(2026, 3, 6)
    minute = minute_token(start)
    signal = DualThrustSignal(
        event_id="event-1",
        symbol="TESTUSDT",
        direction=Direction.LONG,
        signal_bar_time=start - timedelta(minutes=30),
        signal_available_time=start,
        raw_signal_price=100.0,
        session_open=100.0,
        upper_band=99.0,
        lower_band=90.0,
        dual_thrust_range=10.0,
        atr_value=5.0,
        trend_alignment_atr=1.0,
        volume_ratio=2.0,
        body_atr=1.0,
        directional_close_position=0.9,
        range_atr=2.0,
        breakout_extension_atr=0.2,
        quality_score=2.0,
    )
    series = CompactSeries(
        minutes=array("q", [minute, minute + 1]),
        opens=array("d", [100.0, 100.0]),
        highs=array("d", [112.0, 101.0]),
        lows=array("d", [90.0, 99.0]),
        closes=array("d", [100.0, 100.0]),
        volumes=array("d", [10_000.0, 10_000.0]),
    )
    signal_config = VolatilityBreakoutConfig(
        stop_atr_multiple=1.0,
        take_profit_r=1.0,
        max_holding_minutes=60,
    )
    result = simulate_portfolio(
        {minute: [Candidate(signal, minute)]},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": SymbolRules("TESTUSDT", "0.001", "0.001", "0.01", "5")},
        signal_config,
        PortfolioSearchConfig(risk_per_trade_pct=0.02),
        BacktestExecutionConfig(
            mode="conservative",
            market_slippage_bps=2.0,
            stop_slippage_bps=5.0,
            take_profit_slippage_bps=2.0,
            taker_fee_rate=0.0005,
        ),
        start,
        start + timedelta(minutes=2),
        200.0,
    )

    assert result["trade_count"] == 1
    assert result["trades"][0]["exit_reason"] == "stop_loss"
    assert result["trades"][0]["net_pnl"] < 0.0


def test_conditional_extension_only_extends_profitable_position() -> None:
    start = datetime(2026, 3, 6)
    minute = minute_token(start)
    signal = DualThrustSignal(
        event_id="event-extension",
        symbol="TESTUSDT",
        direction=Direction.LONG,
        signal_bar_time=start - timedelta(minutes=30),
        signal_available_time=start,
        raw_signal_price=100.0,
        session_open=100.0,
        upper_band=99.0,
        lower_band=90.0,
        dual_thrust_range=10.0,
        atr_value=5.0,
        trend_alignment_atr=1.0,
        volume_ratio=2.0,
        body_atr=1.0,
        directional_close_position=0.9,
        range_atr=2.0,
        breakout_extension_atr=0.2,
        quality_score=2.0,
    )
    series = CompactSeries(
        minutes=array("q", [minute + offset for offset in range(5)]),
        opens=array("d", [100.0, 102.0, 103.0, 104.0, 104.0]),
        highs=array("d", [102.0, 103.0, 104.0, 105.0, 105.0]),
        lows=array("d", [99.0, 101.0, 102.0, 103.0, 103.0]),
        closes=array("d", [101.0, 102.0, 103.0, 104.0, 104.0]),
        volumes=array("d", [10_000.0] * 5),
    )
    result = simulate_portfolio(
        {minute: [Candidate(signal, minute)]},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": SymbolRules("TESTUSDT", "0.001", "0.001", "0.01", "5")},
        VolatilityBreakoutConfig(
            stop_atr_multiple=1.0,
            take_profit_r=20.0,
            max_holding_minutes=2,
            extended_holding_minutes=4,
            extension_min_current_r=0.1,
            extension_min_mfe_r=0.1,
        ),
        PortfolioSearchConfig(risk_per_trade_pct=0.02),
        BacktestExecutionConfig(mode="conservative", taker_fee_rate=0.0),
        start,
        start + timedelta(minutes=5),
        200.0,
    )

    assert result["trade_count"] == 1
    assert result["trades"][0]["exit_reason"] == "extended_time_stop"
    assert result["trades"][0]["holding_minutes"] == 4
    assert result["trades"][0]["extension_qualified"] is True


def test_profit_giveback_floor_applies_on_following_bar() -> None:
    start = datetime(2026, 3, 6)
    minute = minute_token(start)
    signal = DualThrustSignal(
        event_id="event-giveback",
        symbol="TESTUSDT",
        direction=Direction.LONG,
        signal_bar_time=start - timedelta(minutes=30),
        signal_available_time=start,
        raw_signal_price=100.0,
        session_open=100.0,
        upper_band=99.0,
        lower_band=90.0,
        dual_thrust_range=10.0,
        atr_value=5.0,
        trend_alignment_atr=1.0,
        volume_ratio=2.0,
        body_atr=1.0,
        directional_close_position=0.9,
        range_atr=2.0,
        breakout_extension_atr=0.2,
        quality_score=2.0,
    )
    series = CompactSeries(
        minutes=array("q", [minute, minute + 1, minute + 2]),
        opens=array("d", [100.0, 108.0, 105.0]),
        highs=array("d", [110.0, 109.0, 106.0]),
        lows=array("d", [99.0, 104.0, 104.0]),
        closes=array("d", [108.0, 105.0, 105.0]),
        volumes=array("d", [10_000.0] * 3),
    )
    result = simulate_portfolio(
        {minute: [Candidate(signal, minute)]},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": SymbolRules("TESTUSDT", "0.001", "0.001", "0.01", "5")},
        VolatilityBreakoutConfig(
            stop_atr_multiple=1.0,
            take_profit_r=20.0,
            profit_giveback_activation_r=1.0,
            profit_giveback_r=0.5,
            max_holding_minutes=60,
        ),
        PortfolioSearchConfig(risk_per_trade_pct=0.02),
        BacktestExecutionConfig(mode="conservative", taker_fee_rate=0.0),
        start,
        start + timedelta(minutes=3),
        200.0,
    )

    assert result["trade_count"] == 1
    assert result["trades"][0]["exit_reason"] == "stop_loss"
    assert result["trades"][0]["net_pnl"] > 0.0


def test_directional_market_filter_is_symmetric() -> None:
    start = datetime(2026, 3, 6)
    signal = DualThrustSignal(
        event_id="event-market-filter",
        symbol="TESTUSDT",
        direction=Direction.LONG,
        signal_bar_time=start - timedelta(minutes=60),
        signal_available_time=start,
        raw_signal_price=100.0,
        session_open=100.0,
        upper_band=99.0,
        lower_band=90.0,
        dual_thrust_range=10.0,
        atr_value=5.0,
        trend_alignment_atr=1.0,
        volume_ratio=2.0,
        body_atr=1.0,
        directional_close_position=0.9,
        range_atr=2.0,
        breakout_extension_atr=0.2,
        quality_score=2.0,
    )
    config = PortfolioSearchConfig(max_directional_btc_return_4h=0.02)

    long_candidate = Candidate(signal, minute_token(start), btc_return_4h=0.03)
    assert _market_filter_reject_reason(long_candidate, config) == "btc_4h_overextended"

    short_candidate = replace(long_candidate, signal=replace(signal, direction=Direction.SHORT))
    assert _market_filter_reject_reason(short_candidate, config) is None


def test_market_context_uses_only_closed_hourly_bars() -> None:
    start = datetime(2026, 1, 1)
    symbols = ("BTCUSDT", "ETHUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT")
    data = {
        symbol: [
            _candle(
                start + timedelta(minutes=15 * index),
                100.0 + index / 4.0,
                101.0 + index / 4.0,
                99.0 + index / 4.0,
                100.0 + index / 4.0,
            )
            for index in range(100)
        ]
        for symbol in symbols
    }
    decision_minute = minute_token(start + timedelta(hours=25))
    before = build_market_context(symbols, data)[decision_minute]

    future = {
        symbol: candles
        + [
            _candle(
                start + timedelta(hours=25, minutes=15 * index),
                125.0,
                250.0,
                50.0,
                200.0,
            )
            for index in range(4)
        ]
        for symbol, candles in data.items()
    }
    after = build_market_context(symbols, future)[decision_minute]

    assert before == after
    assert before["btc_return_4h"] > 0.0


def test_effective_trade_risk_respects_hard_cap() -> None:
    start = datetime(2026, 3, 6)
    minute = minute_token(start)
    signal = DualThrustSignal(
        event_id="event-risk-cap",
        symbol="TESTUSDT",
        direction=Direction.LONG,
        signal_bar_time=start - timedelta(minutes=30),
        signal_available_time=start,
        raw_signal_price=100.0,
        session_open=100.0,
        upper_band=99.0,
        lower_band=90.0,
        dual_thrust_range=10.0,
        atr_value=5.0,
        trend_alignment_atr=1.0,
        volume_ratio=2.0,
        body_atr=1.0,
        directional_close_position=0.9,
        range_atr=2.0,
        breakout_extension_atr=0.2,
        quality_score=2.0,
    )
    series = CompactSeries(
        minutes=array("q", [minute, minute + 1]),
        opens=array("d", [100.0, 100.0]),
        highs=array("d", [101.0, 101.0]),
        lows=array("d", [94.0, 94.0]),
        closes=array("d", [95.0, 95.0]),
        volumes=array("d", [100_000.0, 100_000.0]),
    )
    result = simulate_portfolio(
        {minute: [Candidate(signal, minute)]},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": SymbolRules("TESTUSDT", "0.001", "0.001", "0.01", "5")},
        VolatilityBreakoutConfig(stop_atr_multiple=1.0, take_profit_r=20.0),
        PortfolioSearchConfig(
            risk_per_trade_pct=0.20,
            max_trade_risk_pct=0.10,
            long_risk_multiplier=2.0,
        ),
        BacktestExecutionConfig(mode="conservative", taker_fee_rate=0.0),
        start,
        start + timedelta(minutes=2),
        200.0,
    )

    assert result["trades"][0]["risk_usdt"] <= 20.0 + 1e-9

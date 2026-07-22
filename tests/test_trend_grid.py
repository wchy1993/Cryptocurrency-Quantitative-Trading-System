from __future__ import annotations

from array import array
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.models import Candle, Direction
from crypto_scalper.risk import BacktestExecutionConfig
from crypto_scalper.trend_grid import (
    TrendGridConfig,
    TrendGridSignal,
    build_trend_grid_timeline,
)
from crypto_scalper.trend_grid_optimize import (
    CompactSeries,
    GridCandidate,
    GridPortfolioConfig,
    _create_campaign,
    minute_token,
    simulate_grid_portfolio,
)


def _rules() -> SymbolRules:
    return SymbolRules("TESTUSDT", Decimal("0.001"), Decimal("0.001"), Decimal("0.1"), Decimal("5"))


def _series(start: datetime, rows: list[tuple[float, float, float, float]]) -> CompactSeries:
    return CompactSeries(
        minutes=array("q", [minute_token(start + timedelta(minutes=index)) for index in range(len(rows))]),
        opens=array("d", [row[0] for row in rows]),
        highs=array("d", [row[1] for row in rows]),
        lows=array("d", [row[2] for row in rows]),
        closes=array("d", [row[3] for row in rows]),
        volumes=array("d", [1_000.0 for _ in rows]),
    )


def _signal(start: datetime) -> TrendGridSignal:
    return TrendGridSignal(
        event_id="event-1",
        symbol="TESTUSDT",
        direction=Direction.LONG,
        signal_bar_time=start - timedelta(minutes=15),
        signal_available_time=start,
        raw_signal_price=100.0,
        atr_value=1.0,
        fast_ema=99.0,
        slow_ema=98.0,
        fast_slope_atr=0.2,
        slow_slope_atr=0.1,
        alignment_atr=1.0,
        extension_atr=1.0,
        directional_close_position=0.8,
        volume_ratio=1.0,
        quality_score=1.0,
    )


def test_trend_grid_signal_uses_closed_bar_and_next_open() -> None:
    start = datetime(2026, 1, 1)
    candles: list[Candle] = []
    price = 100.0
    for index in range(90):
        timestamp = start + timedelta(minutes=15 * index)
        open_price = price
        price += 0.20
        candles.append(Candle(timestamp, open_price, price + 0.10, open_price - 0.10, price, 100.0))
    config = TrendGridConfig(
        entry_mode="continuous",
        allow_short=False,
        min_fast_slope_atr=0.0,
        min_slow_slope_atr=-1.0,
        max_alignment_atr=100.0,
        max_entry_extension_atr=10.0,
        reentry_interval_bars=1,
    )

    _, signals = build_trend_grid_timeline("TESTUSDT", candles, config)

    assert signals
    assert all(
        signal.signal_available_time == signal.signal_bar_time + timedelta(minutes=15)
        for signal in signals
    )


def test_hard_stop_must_be_beyond_deepest_grid_level() -> None:
    with pytest.raises(ValueError, match="beyond"):
        TrendGridConfig(grid_spacing_atr=0.5, grid_levels=4, hard_stop_atr_multiple=2.0).validate()


def test_campaign_sizing_caps_full_grid_worst_case_risk() -> None:
    start = datetime(2026, 1, 1)
    series = _series(start, [(100.0, 100.2, 99.8, 100.0)])
    signal_config = TrendGridConfig(grid_levels=2, hard_stop_atr_multiple=2.0)
    portfolio = GridPortfolioConfig(risk_per_campaign_pct=0.05)
    opened = _create_campaign(
        GridCandidate(_signal(start), minute_token(start)),
        series,
        _rules(),
        signal_config,
        portfolio,
        BacktestExecutionConfig(),
        200.0,
    )

    assert opened is not None
    campaign, _ = opened
    assert campaign.risk_budget <= 10.05
    assert len(campaign.levels) == 3


def test_same_minute_stop_and_grid_target_uses_stop_first() -> None:
    start = datetime(2026, 1, 1)
    series = _series(
        start,
        [
            (100.0, 102.0, 97.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ],
    )
    config = TrendGridConfig(
        grid_levels=2,
        grid_spacing_atr=0.5,
        grid_target_spacing=1.0,
        hard_stop_atr_multiple=2.0,
    )
    candidate = GridCandidate(_signal(start), minute_token(start))

    result = simulate_grid_portfolio(
        {minute_token(start): [candidate]},
        {"TESTUSDT": {}},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": _rules()},
        config,
        GridPortfolioConfig(risk_per_campaign_pct=0.05),
        BacktestExecutionConfig(),
        start,
        start + timedelta(minutes=2),
        200.0,
    )

    assert result["trade_count"] == 1
    assert result["trades"][0]["exit_reason"] == "hard_stop"
    assert result["grid_take_profit_count"] == 0
    assert result["net_profit"] < 0.0


def test_limit_touch_without_trade_through_does_not_fill_grid_level() -> None:
    start = datetime(2026, 1, 1)
    series = _series(
        start,
        [
            (100.0, 100.2, 99.5, 100.0),
            (100.0, 100.1, 99.9, 100.0),
        ],
    )
    config = TrendGridConfig(
        grid_levels=2,
        grid_spacing_atr=0.5,
        hard_stop_atr_multiple=2.0,
    )
    candidate = GridCandidate(_signal(start), minute_token(start))

    result = simulate_grid_portfolio(
        {minute_token(start): [candidate]},
        {"TESTUSDT": {}},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": _rules()},
        config,
        GridPortfolioConfig(risk_per_campaign_pct=0.05),
        BacktestExecutionConfig(),
        start,
        start + timedelta(minutes=2),
        200.0,
    )

    assert result["trade_count"] == 1
    assert result["grid_entry_count"] == 1


def test_max_total_entries_prevents_additional_grid_inventory() -> None:
    start = datetime(2026, 1, 1)
    series = _series(
        start,
        [
            (100.0, 100.2, 98.0, 99.0),
            (99.0, 100.0, 98.8, 99.5),
        ],
    )
    config = TrendGridConfig(
        grid_levels=2,
        grid_spacing_atr=0.5,
        hard_stop_atr_multiple=2.0,
        max_total_entries=1,
    )

    result = simulate_grid_portfolio(
        {minute_token(start): [GridCandidate(_signal(start), minute_token(start))]},
        {"TESTUSDT": {}},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": _rules()},
        config,
        GridPortfolioConfig(risk_per_campaign_pct=0.05),
        BacktestExecutionConfig(),
        start,
        start + timedelta(minutes=2),
        200.0,
    )

    assert result["grid_entry_count"] == 1


def test_campaign_loss_limit_exits_before_distant_hard_stop() -> None:
    start = datetime(2026, 1, 1)
    series = _series(
        start,
        [
            (100.0, 100.1, 99.9, 100.0),
            (99.0, 99.1, 98.9, 99.0),
            (99.0, 99.0, 99.0, 99.0),
        ],
    )
    config = TrendGridConfig(
        grid_levels=2,
        grid_spacing_atr=0.5,
        hard_stop_atr_multiple=2.0,
        max_total_entries=1,
        campaign_loss_limit_r=0.05,
    )

    result = simulate_grid_portfolio(
        {minute_token(start): [GridCandidate(_signal(start), minute_token(start))]},
        {"TESTUSDT": {}},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": _rules()},
        config,
        GridPortfolioConfig(risk_per_campaign_pct=0.05),
        BacktestExecutionConfig(),
        start,
        start + timedelta(minutes=3),
        200.0,
    )

    assert result["trades"][0]["exit_reason"] == "campaign_loss_limit"
    assert result["trades"][0]["anchor_price"] > result["trades"][0]["hard_stop"]

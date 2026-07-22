from __future__ import annotations

from datetime import datetime

from crypto_scalper.models import Direction
from crypto_scalper.trend_grid import TrendGridSignal
from crypto_scalper.trend_grid_optimize import GridCandidate
from crypto_scalper.trend_grid_v3_optimize import (
    GridMarketOverlay,
    apply_market_overlay,
)
from crypto_scalper.volatility_breakout_optimize import minute_token
from crypto_scalper.volatility_breakout_v4_research import V4MarketSnapshot


def _candidate(direction: Direction, symbol: str = "BTCUSDT") -> GridCandidate:
    timestamp = datetime(2026, 7, 1, 12, 0)
    signal = TrendGridSignal(
        event_id=f"{symbol}-{direction.name}",
        symbol=symbol,
        direction=direction,
        signal_bar_time=timestamp,
        signal_available_time=timestamp,
        raw_signal_price=100.0,
        atr_value=2.0,
        fast_ema=99.0,
        slow_ema=98.0,
        fast_slope_atr=0.20 * direction.value,
        slow_slope_atr=0.10 * direction.value,
        alignment_atr=0.75,
        extension_atr=0.25,
        directional_close_position=0.70,
        volume_ratio=1.10,
        quality_score=1.25,
    )
    return GridCandidate(signal=signal, entry_minute=minute_token(timestamp))


def _snapshot(minute: int, breadth: float = 0.80) -> V4MarketSnapshot:
    return V4MarketSnapshot(
        available_minute=minute,
        symbol="BTCUSDT",
        btc_return_4h=0.01,
        eth_return_4h=0.01,
        breadth_above_ema21=breadth,
        breadth_change_4h=0.05,
        symbol_return_4h=0.015,
        symbol_efficiency_12h=0.20,
        market_efficiency_12h=0.25,
        symbol_ema55_atr=1.0,
    )


def test_default_overlay_is_identity_and_does_not_reapply_daily_cap() -> None:
    long_candidate = _candidate(Direction.LONG)
    short_candidate = _candidate(Direction.SHORT, "ETHUSDT")
    minute = long_candidate.entry_minute
    candidates = {minute: [long_candidate, short_candidate]}

    filtered = apply_market_overlay(candidates, {}, GridMarketOverlay())

    assert filtered == candidates


def test_overlay_uses_directional_market_breadth_without_future_data() -> None:
    short_candidate = _candidate(Direction.SHORT)
    minute = short_candidate.entry_minute
    context = {minute: {"BTCUSDT": _snapshot(minute, breadth=0.80)}}
    candidates = {minute: [short_candidate]}

    rejected = apply_market_overlay(
        candidates,
        context,
        GridMarketOverlay(min_directional_breadth=0.30),
    )
    accepted = apply_market_overlay(
        candidates,
        context,
        GridMarketOverlay(min_directional_breadth=0.19),
    )

    assert rejected == {}
    assert accepted == candidates

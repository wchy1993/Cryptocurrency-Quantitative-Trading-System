from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from crypto_scalper.models import Direction
from crypto_scalper.trend_grid import TrendGridSignal
from crypto_scalper.trend_grid_optimize import GridCandidate
from crypto_scalper.trend_grid_v4 import (
    GridV4EntryGate,
    filter_grid_v4_candidates,
    passes_grid_v4_entry,
)
from crypto_scalper.volatility_breakout_v4_research import V4MarketSnapshot


def _candidate(direction: Direction = Direction.SHORT) -> GridCandidate:
    signal = TrendGridSignal(
        event_id=f"grid-{direction.name}",
        symbol="BTCUSDT",
        direction=direction,
        signal_bar_time=datetime(2026, 1, 1, 9),
        signal_available_time=datetime(2026, 1, 1, 10),
        raw_signal_price=100.0,
        atr_value=2.0,
        fast_ema=99.0,
        slow_ema=101.0,
        fast_slope_atr=-0.3 if direction == Direction.SHORT else 0.3,
        slow_slope_atr=-0.1 if direction == Direction.SHORT else 0.1,
        alignment_atr=1.0,
        extension_atr=0.3,
        directional_close_position=0.7,
        volume_ratio=1.2,
        quality_score=0.8,
    )
    return GridCandidate(signal=signal, entry_minute=29_455_800)


def _snapshot() -> V4MarketSnapshot:
    return V4MarketSnapshot(
        available_minute=29_455_800,
        symbol="BTCUSDT",
        btc_return_4h=-0.004,
        eth_return_4h=-0.003,
        breadth_above_ema21=0.30,
        breadth_change_4h=-0.05,
        symbol_return_4h=-0.01,
        symbol_efficiency_12h=-0.2,
        market_efficiency_12h=0.3,
        symbol_ema55_atr=-1.0,
    )


def test_grid_v4_gate_uses_directional_closed_bar_features() -> None:
    candidate = _candidate()
    gate = GridV4EntryGate(
        min_quality_score=0.7,
        min_directional_fast_slope_atr=0.2,
        min_directional_breadth=0.6,
    )

    assert passes_grid_v4_entry(candidate, _snapshot(), gate)
    assert not passes_grid_v4_entry(
        replace(candidate, signal=replace(candidate.signal, quality_score=0.5)),
        _snapshot(),
        gate,
    )


def test_grid_v4_rejects_missing_market_context_by_default() -> None:
    candidate = _candidate()

    assert not passes_grid_v4_entry(candidate, None, GridV4EntryGate())
    assert passes_grid_v4_entry(
        candidate,
        None,
        GridV4EntryGate(require_market_context=False),
    )


def test_grid_v4_filter_preserves_candidate_order() -> None:
    accepted = _candidate()
    rejected = replace(
        accepted,
        signal=replace(accepted.signal, event_id="rejected", quality_score=0.1),
    )
    result = filter_grid_v4_candidates(
        {accepted.entry_minute: [accepted, rejected]},
        {accepted.entry_minute: {"BTCUSDT": _snapshot()}},
        GridV4EntryGate(min_quality_score=0.7),
    )

    assert result == {accepted.entry_minute: [accepted]}

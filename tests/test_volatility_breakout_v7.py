from __future__ import annotations

from array import array
from dataclasses import replace
from datetime import datetime

import pytest

from crypto_scalper.models import Direction
from crypto_scalper.volatility_breakout import DualThrustSignal
from crypto_scalper.volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    minute_token,
)
from crypto_scalper.volatility_breakout_v4_research import V4MarketSnapshot
from crypto_scalper.volatility_breakout_v7 import (
    BreakoutV7ConfidencePolicy,
    BreakoutV7EntryTiming,
    apply_breakout_v7_timing,
    breakout_v7_confidence_score,
    breakout_v7_risk_multiplier,
)


def _candidate(direction: Direction = Direction.LONG) -> Candidate:
    signal = DualThrustSignal(
        event_id=f"v7-{direction.name}",
        symbol="TESTUSDT",
        direction=direction,
        signal_bar_time=datetime(2026, 1, 1, 9),
        signal_available_time=datetime(2026, 1, 1, 10),
        raw_signal_price=100.0,
        session_open=99.0,
        upper_band=100.0,
        lower_band=98.0,
        dual_thrust_range=5.0,
        atr_value=1.0,
        trend_alignment_atr=3.0,
        volume_ratio=2.0,
        body_atr=1.5,
        directional_close_position=0.8,
        range_atr=5.0,
        breakout_extension_atr=0.6,
        quality_score=1.7,
    )
    return Candidate(signal, minute_token(signal.signal_available_time))


def _snapshot() -> V4MarketSnapshot:
    return V4MarketSnapshot(
        available_minute=minute_token(datetime(2026, 1, 1, 10)),
        symbol="TESTUSDT",
        btc_return_4h=0.005,
        eth_return_4h=0.004,
        breadth_above_ema21=0.75,
        breadth_change_4h=0.05,
        symbol_return_4h=0.01,
        symbol_efficiency_12h=0.2,
        market_efficiency_12h=0.3,
        symbol_ema55_atr=1.0,
    )


def test_v7_default_policy_is_risk_path_equivalent_to_v6() -> None:
    multiplier, score, tier = breakout_v7_risk_multiplier(
        _candidate(), _snapshot(), BreakoutV7ConfidencePolicy()
    )

    assert score == 5
    assert tier == "standard"
    assert multiplier == 1.0


def test_v7_refined_policy_allocates_only_from_entry_time_features() -> None:
    policy = BreakoutV7ConfidencePolicy(
        long_max_quality_score=2.4,
        long_min_body_atr=2.8,
        long_min_volume_ratio=2.0,
        long_max_breakout_extension_atr=2.6,
        long_max_directional_breadth=0.84,
        strong_score=4,
        weak_score=1,
        strong_risk_multiplier=1.3,
        weak_risk_multiplier=0.45,
        long_risk_multiplier=1.05,
        short_risk_multiplier=0.9,
    )

    candidate = _candidate()
    assert breakout_v7_confidence_score(candidate, _snapshot(), policy) == 4
    multiplier, score, tier = breakout_v7_risk_multiplier(
        candidate, _snapshot(), policy
    )
    assert (score, tier) == (4, "strong")
    assert multiplier == pytest.approx(1.365)


def test_v7_one_minute_confirmation_uses_completed_decision_bar_only() -> None:
    candidate = _candidate()
    entry_minute = candidate.entry_minute
    series = CompactSeries(
        minutes=array("q", [entry_minute, entry_minute + 1]),
        opens=array("d", [100.0, 50.0]),
        highs=array("d", [100.6, 500.0]),
        lows=array("d", [99.9, 1.0]),
        closes=array("d", [100.5, 400.0]),
        volumes=array("d", [1_000.0, 1_000.0]),
    )
    timing = BreakoutV7EntryTiming(
        confirmation_minutes=1,
        min_directional_close_move_atr=0.4,
        max_directional_close_move_atr=0.6,
        max_adverse_excursion_atr=0.2,
    )

    result = apply_breakout_v7_timing(
        {entry_minute: [candidate]}, {"TESTUSDT": series}, timing
    )

    assert list(result) == [entry_minute + 1]
    assert result[entry_minute + 1][0] == replace(
        candidate, entry_minute=entry_minute + 1
    )
    assert not apply_breakout_v7_timing(
        {entry_minute: [candidate]},
        {"TESTUSDT": series},
        replace(timing, min_directional_close_move_atr=0.6),
    )

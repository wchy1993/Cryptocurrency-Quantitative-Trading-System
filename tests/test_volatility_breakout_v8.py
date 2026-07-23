from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from crypto_scalper.models import Direction
from crypto_scalper.volatility_breakout import DualThrustSignal
from crypto_scalper.volatility_breakout_optimize import Candidate, minute_token
from crypto_scalper.volatility_breakout_v4_research import V4MarketSnapshot
from crypto_scalper.volatility_breakout_v7 import (
    BreakoutV7ConfidencePolicy,
    breakout_v7_risk_multiplier,
)
from crypto_scalper.volatility_breakout_v8 import (
    BreakoutV8ScoreAllocation,
    breakout_v8_risk_multiplier,
)


def _candidate(direction: Direction = Direction.SHORT) -> Candidate:
    now = datetime(2026, 7, 1, 12)
    signal = DualThrustSignal(
        event_id=f"v8-{direction.name}",
        symbol="TESTUSDT",
        direction=direction,
        signal_bar_time=now,
        signal_available_time=now,
        raw_signal_price=100.0,
        session_open=99.0,
        upper_band=100.0,
        lower_band=98.0,
        dual_thrust_range=5.0,
        atr_value=1.0,
        trend_alignment_atr=2.0,
        volume_ratio=3.0,
        body_atr=3.0,
        directional_close_position=0.8,
        range_atr=3.0,
        breakout_extension_atr=0.2,
        quality_score=1.5,
    )
    return Candidate(signal, minute_token(now))


def _snapshot(direction: Direction = Direction.SHORT) -> V4MarketSnapshot:
    now = datetime(2026, 7, 1, 12)
    return V4MarketSnapshot(
        available_minute=minute_token(now),
        symbol="TESTUSDT",
        btc_return_4h=-0.01,
        eth_return_4h=-0.01,
        breadth_above_ema21=0.8 if direction == Direction.SHORT else 0.2,
        breadth_change_4h=0.0,
        symbol_return_4h=-0.01,
        symbol_efficiency_12h=0.2,
        market_efficiency_12h=0.3,
        symbol_ema55_atr=-1.0,
    )


def _confidence() -> BreakoutV7ConfidencePolicy:
    return BreakoutV7ConfidencePolicy(
        long_max_quality_score=2.4,
        long_min_body_atr=2.8,
        long_min_volume_ratio=2.0,
        long_max_breakout_extension_atr=2.6,
        long_max_directional_breadth=0.84,
        short_min_quality_score=1.05,
        short_min_body_atr=0.5,
        short_min_volume_ratio=2.0,
        short_max_breakout_extension_atr=0.32,
        short_max_directional_breadth=0.86,
        strong_score=4,
        weak_score=1,
        strong_risk_multiplier=1.3,
        weak_risk_multiplier=0.45,
        long_risk_multiplier=1.05,
        short_risk_multiplier=0.9,
        maximum_risk_multiplier=2.0,
    )


def test_v8_default_allocation_is_exactly_v7_equivalent() -> None:
    candidate = _candidate()
    snapshot = _snapshot()
    confidence = _confidence()
    v7_multiplier, v7_score, _ = breakout_v7_risk_multiplier(
        candidate, snapshot, confidence
    )

    v8_multiplier, v8_score, lane = breakout_v8_risk_multiplier(
        candidate,
        snapshot,
        confidence,
        BreakoutV8ScoreAllocation(),
    )

    assert v8_score == v7_score == 5
    assert v8_multiplier == v7_multiplier
    assert lane == "score_5_short"


def test_v8_score_factor_is_causal_bounded_and_rejectable() -> None:
    candidate = _candidate()
    snapshot = _snapshot()
    confidence = _confidence()
    allocation = BreakoutV8ScoreAllocation(
        score_5_short_factor=2.6,
        maximum_adjusted_multiplier=3.0,
    )

    multiplier, score, lane = breakout_v8_risk_multiplier(
        candidate, snapshot, confidence, allocation
    )

    assert score == 5
    assert lane == "score_5_short"
    assert multiplier == pytest.approx(3.0)
    rejected = replace(allocation, minimum_score=5)
    lower_score_candidate = replace(
        candidate,
        signal=replace(candidate.signal, volume_ratio=0.5),
    )
    rejected_multiplier, rejected_score, rejected_lane = (
        breakout_v8_risk_multiplier(
            lower_score_candidate, snapshot, confidence, rejected
        )
    )
    assert rejected_score < 5
    assert rejected_multiplier == 0.0
    assert rejected_lane == "rejected"

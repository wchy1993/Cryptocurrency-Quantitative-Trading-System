from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from crypto_scalper.models import Direction
from crypto_scalper.trend_grid import TrendGridSignal
from crypto_scalper.trend_grid_optimize import GridCandidate, minute_token
from crypto_scalper.trend_grid_v5 import GridV5ConfidencePolicy, grid_v5_tier
from crypto_scalper.trend_grid_v6 import (
    GridV6CampaignPolicy,
    grid_v6_entry_decision,
)
from crypto_scalper.volatility_breakout_v4_research import V4MarketSnapshot


def _candidate(extension: float = 0.30) -> GridCandidate:
    now = datetime(2026, 7, 1, 12)
    signal = TrendGridSignal(
        event_id="grid-v6-test",
        symbol="TESTUSDT",
        direction=Direction.SHORT,
        signal_bar_time=now,
        signal_available_time=now,
        raw_signal_price=100.0,
        atr_value=1.0,
        fast_ema=99.0,
        slow_ema=101.0,
        fast_slope_atr=-0.2,
        slow_slope_atr=-0.1,
        alignment_atr=0.8,
        extension_atr=extension,
        directional_close_position=0.8,
        volume_ratio=1.0,
        quality_score=0.6,
    )
    return GridCandidate(signal, minute_token(now))


def _snapshot() -> V4MarketSnapshot:
    now = datetime(2026, 7, 1, 12)
    return V4MarketSnapshot(
        available_minute=minute_token(now),
        symbol="TESTUSDT",
        btc_return_4h=0.0,
        eth_return_4h=0.0,
        breadth_above_ema21=0.6,
        breadth_change_4h=0.0,
        symbol_return_4h=0.0,
        symbol_efficiency_12h=0.0,
        market_efficiency_12h=0.0,
        symbol_ema55_atr=0.0,
    )


def _confidence() -> GridV5ConfidencePolicy:
    return GridV5ConfidencePolicy(
        min_quality_score=0.45,
        min_alignment_atr=0.52,
        min_extension_atr=0.2,
        max_extension_atr=0.45,
        max_volume_ratio=1.1,
        max_regime_score=0.3,
        strong_score=6,
        weak_score=2,
        strong_risk_multiplier=1.12,
        standard_risk_multiplier=0.92,
        weak_risk_multiplier=0.35,
        maximum_campaign_risk_pct=0.11,
        reject_weak_tier=True,
    )


def test_grid_v6_default_decision_is_grid_v5_equivalent() -> None:
    candidate = _candidate()
    snapshot = _snapshot()
    confidence = _confidence()

    accepted, tier, score, factor = grid_v6_entry_decision(
        candidate,
        snapshot,
        confidence,
        GridV6CampaignPolicy(),
    )

    assert (tier, score) == grid_v5_tier(candidate, snapshot, confidence)
    assert accepted is True
    assert factor == 1.0


def test_grid_v6_extension_guard_uses_entry_time_signal_only() -> None:
    candidate = _candidate(extension=0.04)
    accepted, tier, score, factor = grid_v6_entry_decision(
        candidate,
        _snapshot(),
        _confidence(),
        GridV6CampaignPolicy(
            minimum_actual_extension_atr=0.05,
            score_4_risk_factor=0.8,
        ),
    )

    assert accepted is False
    assert tier in {"standard", "strong"}
    assert 0 <= score <= 6
    assert factor in {0.8, 1.0}


def test_grid_v6_profit_lock_requires_a_complete_pair() -> None:
    with pytest.raises(ValueError):
        GridV6CampaignPolicy(profit_lock_activation_r=0.2).validate()
    GridV6CampaignPolicy(
        profit_lock_activation_r=0.2,
        profit_giveback_r=0.1,
    ).validate()


def test_grid_v6_cycle_floor_requires_complete_causal_controls() -> None:
    with pytest.raises(ValueError):
        GridV6CampaignPolicy(
            cycle_profit_floor_min_take_profits=2,
        ).validate()
    with pytest.raises(ValueError):
        GridV6CampaignPolicy(
            cycle_profit_floor_min_take_profits=2,
            cycle_profit_floor_activation_r=0.2,
            cycle_profit_floor_r=0.2,
        ).validate()
    GridV6CampaignPolicy(
        cycle_profit_floor_min_take_profits=2,
        cycle_profit_floor_activation_r=0.2,
        cycle_profit_floor_r=0.03,
    ).validate()

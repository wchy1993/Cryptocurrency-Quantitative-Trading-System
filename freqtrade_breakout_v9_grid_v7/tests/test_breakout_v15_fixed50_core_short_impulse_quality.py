from __future__ import annotations

import sys
from pathlib import Path

import pytest

STRATEGY_DIR = Path(__file__).resolve().parents[1] / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from BreakoutV15Fixed50CoreShortImpulseQualityResearchFreqtrade import (  # noqa: E402
    BreakoutV15Fixed50CoreShortImpulse20ResearchFreqtrade,
)


@pytest.fixture
def strategy() -> BreakoutV15Fixed50CoreShortImpulse20ResearchFreqtrade:
    return BreakoutV15Fixed50CoreShortImpulse20ResearchFreqtrade(
        config={"stake_currency": "USDT"}
    )


def test_weak_local_and_nonexceptional_range_gets_minimum_scale(strategy) -> None:
    row = {"symbol_return_4h": 0.0, "bo_range_atr": 5.0}

    assert strategy._v15_breakout_short_impulse_scale(row) == pytest.approx(0.20)


def test_local_downside_confirmation_keeps_full_risk(strategy) -> None:
    row = {"symbol_return_4h": -0.01, "bo_range_atr": 5.0}

    assert strategy._v15_breakout_short_impulse_scale(row) == pytest.approx(1.0)


def test_exceptional_range_keeps_full_risk(strategy) -> None:
    row = {"symbol_return_4h": 0.0, "bo_range_atr": 7.0}

    assert strategy._v15_breakout_short_impulse_scale(row) == pytest.approx(1.0)


def test_missing_signal_features_fail_open(strategy) -> None:
    row = {}

    assert strategy._v15_breakout_short_impulse_scale(row) == pytest.approx(1.0)

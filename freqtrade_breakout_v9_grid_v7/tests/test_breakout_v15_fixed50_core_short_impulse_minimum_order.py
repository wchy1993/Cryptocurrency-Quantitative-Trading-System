from __future__ import annotations

import sys
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from typing import Any

STRATEGY_DIR = Path(__file__).resolve().parents[1] / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from BreakoutV15Fixed50CoreShortImpulseMinimumOrderResearchFreqtrade import (  # noqa: E402
    _V15BreakoutShortImpulseMinimumOrderRiskMixin,
)
from BreakoutV15Fixed50CoreShortImpulseMinimumOrderNeighborhoodResearchFreqtrade import (  # noqa: E402
    BreakoutV15Fixed50CoreShortImpulseFloor25ResearchFreqtrade,
)
from BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade import (  # noqa: E402
    BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade,
)


class _StakeBase:
    row: dict[str, float]

    def custom_stake_amount(self, *args: Any, **kwargs: Any) -> float:
        return 10.0

    @staticmethod
    def _component(entry_tag: str | None) -> str:
        return str(entry_tag)

    def _latest_signal_row(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        return self.row


class _Subject(_V15BreakoutShortImpulseMinimumOrderRiskMixin, _StakeBase):
    pass


def _stake(subject: _Subject, min_stake: float | None) -> float:
    return subject.custom_stake_amount(
        pair="AVAX/USDT:USDT",
        current_time=datetime(2026, 8, 10, tzinfo=timezone.utc),
        current_rate=1.0,
        proposed_stake=10.0,
        min_stake=min_stake,
        max_stake=100.0,
        leverage=10.0,
        entry_tag="breakout",
        side="short",
    )


def test_weak_signal_uses_exchange_minimum_instead_of_being_dropped() -> None:
    subject = _Subject()
    subject.row = {"symbol_return_4h": 0.0, "bo_range_atr": 5.0}

    assert isclose(_stake(subject, min_stake=3.0), 3.0)


def test_weak_signal_keeps_scaled_stake_when_above_minimum() -> None:
    subject = _Subject()
    subject.row = {"symbol_return_4h": 0.0, "bo_range_atr": 5.0}

    assert isclose(_stake(subject, min_stake=1.0), 2.0)


def test_confirmed_impulse_keeps_original_stake() -> None:
    subject = _Subject()
    subject.row = {"symbol_return_4h": -0.01, "bo_range_atr": 5.0}

    assert isclose(_stake(subject, min_stake=3.0), 10.0)


def test_floor25_selection_pins_validated_minimum_scale() -> None:
    assert isclose(
        BreakoutV15Fixed50CoreShortImpulseFloor25ResearchFreqtrade.V15_BO_SHORT_WEAK_IMPULSE_MIN_SCALE,
        0.25,
    )
    assert isclose(
        BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade.V15_BO_SHORT_WEAK_IMPULSE_MIN_SCALE,
        0.25,
    )


def test_floor25_interpolates_weak_impulse_risk_smoothly() -> None:
    subject = BreakoutV15Fixed50CoreShortImpulseFloor25ResearchFreqtrade(
        config={"stake_currency": "USDT"}
    )
    full = subject._v15_breakout_short_impulse_scale(
        {"symbol_return_4h": -0.0045, "bo_range_atr": 5.0}
    )
    middle = subject._v15_breakout_short_impulse_scale(
        {"symbol_return_4h": -0.0035, "bo_range_atr": 5.0}
    )
    floor = subject._v15_breakout_short_impulse_scale(
        {"symbol_return_4h": -0.0025, "bo_range_atr": 5.0}
    )

    assert full == 1.0
    assert floor == 0.25
    assert floor < middle < full

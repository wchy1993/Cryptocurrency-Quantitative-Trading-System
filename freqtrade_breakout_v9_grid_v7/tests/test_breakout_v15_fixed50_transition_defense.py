from __future__ import annotations

import sys
from math import isclose
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STRATEGY_DIR = Path(__file__).resolve().parents[1] / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from BreakoutV15Fixed50RobustQualityResearchFreqtrade import (  # noqa: E402
    _V15RobustBreakoutCountermoveShortRiskMixin,
)
from BreakoutV15Fixed50TransitionDefenseResearchFreqtrade import (  # noqa: E402
    _V15GridShortTransitionDefenseMixin,
)
from BreakoutV15Fixed50TransitionScaleBoundaryResearchFreqtrade import (  # noqa: E402
    BreakoutV15Fixed50TransitionDefense20ResearchFreqtrade,
)


class _StakeBase:
    row: dict[str, float]

    def custom_stake_amount(self, *args: Any, **kwargs: Any) -> float:
        return 100.0

    @staticmethod
    def _component(entry_tag: str | None) -> str:
        return str(entry_tag)

    def _latest_signal_row(
        self,
        pair: str,
        component: str,
        current_time: datetime,
    ) -> dict[str, float]:
        return self.row


class _TransitionSubject(_V15GridShortTransitionDefenseMixin, _StakeBase):
    V15_GRID_TRANSITION_SCALE = 0.20


class _ShortSubject(
    _V15RobustBreakoutCountermoveShortRiskMixin,
    _StakeBase,
):
    V15_BO_SHORT_COUNTERMOVE_SCALE = 0.20


def _stake(strategy: Any, component: str, side: str) -> float:
    return strategy.custom_stake_amount(
        pair="DOT/USDT:USDT",
        current_time=datetime(2026, 7, 20, tzinfo=timezone.utc),
        current_rate=1.0,
        proposed_stake=100.0,
        min_stake=1.0,
        max_stake=1000.0,
        leverage=10.0,
        entry_tag=component,
        side=side,
    )


def test_transition_defense_requires_the_complete_state() -> None:
    strategy = _TransitionSubject()
    strategy.row = {
        "symbol_return_4h": 0.012,
        "market_regime_score": 0.08,
        "low_opportunity_fraction_72h": 0.28,
        "breadth_travel_24h": 3.82,
    }
    assert isclose(_stake(strategy, "grid", "short"), 20.0)

    for key, safe_value in (
        ("symbol_return_4h", 0.009),
        ("market_regime_score", -0.35),
        ("low_opportunity_fraction_72h", 0.41),
        ("breadth_travel_24h", 3.39),
    ):
        original = strategy.row[key]
        strategy.row[key] = safe_value
        assert isclose(_stake(strategy, "grid", "short"), 100.0)
        strategy.row[key] = original


def test_transition_defense_never_changes_other_paths() -> None:
    strategy = _TransitionSubject()
    strategy.row = {
        "symbol_return_4h": 0.012,
        "market_regime_score": 0.08,
        "low_opportunity_fraction_72h": 0.28,
        "breadth_travel_24h": 3.82,
    }
    assert isclose(_stake(strategy, "grid", "long"), 100.0)
    assert isclose(_stake(strategy, "breakout", "short"), 100.0)


def test_robust_breakout_short_guard_preserves_extreme_bear_runner() -> None:
    strategy = _ShortSubject()
    strategy.row = {
        "symbol_return_4h": -0.005,
        "market_regime_score": -0.70,
    }
    assert isclose(_stake(strategy, "breakout", "short"), 20.0)

    strategy.row["market_regime_score"] = -1.0
    assert isclose(_stake(strategy, "breakout", "short"), 100.0)


def test_transition20_candidate_contract() -> None:
    strategy = BreakoutV15Fixed50TransitionDefense20ResearchFreqtrade
    assert isclose(strategy.V15_SCORE5_MAX_BODY_ATR, 3.50)
    assert isclose(strategy.V15_GRID_LONG_LOSS_LIMIT_R, 0.65)
    assert isclose(strategy.V15_BO_LONG_MAX_HEALTHY_EFFICIENCY_12H, 0.82)
    assert isclose(strategy.V15_BO_LONG_LINEAR_EXHAUSTION_SCALE, 0.20)
    assert isclose(strategy.V15_BO_SHORT_COUNTERMOVE_SCALE, 0.20)
    assert isclose(strategy.V15_GRID_TRANSITION_SCALE, 0.20)

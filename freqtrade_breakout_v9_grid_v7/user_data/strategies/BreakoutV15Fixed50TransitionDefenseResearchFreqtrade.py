from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from BreakoutV15Fixed50RobustGridMomentumResearchFreqtrade import (
    BreakoutV15Fixed50RobustQualityL20S20Eff82ResearchFreqtrade,
)


class _V15GridShortTransitionDefenseMixin:
    """De-risk Grid shorts during a fragile breadth-rotation transition.

    The state uses only completed candles.  A target that is still rising is
    dangerous when opportunity has recently become intermittent while market
    leadership continues to rotate quickly.  Strong trends and persistent
    low-opportunity ranges are intentionally left on the inherited path.
    """

    V15_GRID_TRANSITION_MIN_SYMBOL_RETURN_4H = 0.01
    V15_GRID_TRANSITION_MIN_MARKET_REGIME = -0.35
    V15_GRID_TRANSITION_MIN_LOW_OPPORTUNITY_72H = 0.25
    V15_GRID_TRANSITION_MAX_LOW_OPPORTUNITY_72H = 0.40
    V15_GRID_TRANSITION_MIN_BREADTH_TRAVEL_24H = 3.40
    V15_GRID_TRANSITION_SCALE = 0.50

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        stake = super().custom_stake_amount(
            pair,
            current_time,
            current_rate,
            proposed_stake,
            min_stake,
            max_stake,
            leverage,
            entry_tag,
            side,
            **kwargs,
        )
        if (
            stake <= 0.0
            or self._component(entry_tag) != "grid"
            or side != "short"
        ):
            return stake
        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None:
            return stake
        low_opportunity = float(
            row.get("low_opportunity_fraction_72h", np.inf)
        )
        fragile_transition = (
            float(row.get("symbol_return_4h", -np.inf))
            >= self.V15_GRID_TRANSITION_MIN_SYMBOL_RETURN_4H
            and float(row.get("market_regime_score", -np.inf))
            > self.V15_GRID_TRANSITION_MIN_MARKET_REGIME
            and low_opportunity
            >= self.V15_GRID_TRANSITION_MIN_LOW_OPPORTUNITY_72H
            and low_opportunity
            <= self.V15_GRID_TRANSITION_MAX_LOW_OPPORTUNITY_72H
            and float(row.get("breadth_travel_24h", -np.inf))
            >= self.V15_GRID_TRANSITION_MIN_BREADTH_TRAVEL_24H
        )
        if fragile_transition:
            stake *= self.V15_GRID_TRANSITION_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV15Fixed50TransitionDefense35ResearchFreqtrade(
    _V15GridShortTransitionDefenseMixin,
    BreakoutV15Fixed50RobustQualityL20S20Eff82ResearchFreqtrade,
):
    V15_GRID_TRANSITION_SCALE = 0.35
    STRATEGY_VERSION = "breakout_v15_fixed50_transition_defense35_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_transition_defense35"


class BreakoutV15Fixed50TransitionDefense50ResearchFreqtrade(
    BreakoutV15Fixed50TransitionDefense35ResearchFreqtrade,
):
    V15_GRID_TRANSITION_SCALE = 0.50
    STRATEGY_VERSION = "breakout_v15_fixed50_transition_defense50_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_transition_defense50"


class BreakoutV15Fixed50TransitionDefense65ResearchFreqtrade(
    BreakoutV15Fixed50TransitionDefense35ResearchFreqtrade,
):
    V15_GRID_TRANSITION_SCALE = 0.65
    STRATEGY_VERSION = "breakout_v15_fixed50_transition_defense65_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_transition_defense65"


class BreakoutV15Fixed50TransitionDefense50Travel32ResearchFreqtrade(
    BreakoutV15Fixed50TransitionDefense50ResearchFreqtrade,
):
    V15_GRID_TRANSITION_MIN_BREADTH_TRAVEL_24H = 3.20
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_transition_defense50_travel32_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_transition_defense50_travel32"
    )


class BreakoutV15Fixed50TransitionDefense50Travel36ResearchFreqtrade(
    BreakoutV15Fixed50TransitionDefense50ResearchFreqtrade,
):
    V15_GRID_TRANSITION_MIN_BREADTH_TRAVEL_24H = 3.60
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_transition_defense50_travel36_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_transition_defense50_travel36"
    )

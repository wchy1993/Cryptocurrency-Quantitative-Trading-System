from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade import (
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
)
from BreakoutV15Fixed50GridCampaignLossBudgetResearchFreqtrade import (
    _V15GridLongCampaignLossBudgetMixin,
)
from BreakoutV15Fixed50WeakImpulseScore5FailureExitResearchFreqtrade import (
    BreakoutV15Fixed50WeakImpulseScore5Failure350ResearchFreqtrade,
)


class _V15BreakoutCountermoveShortRiskMixin:
    """De-risk breakdown shorts when the symbol has not actually fallen.

    The entry signal is unchanged.  Only completed-candle risk is compressed
    when the target's four-hour return is flat or positive, a setup with
    negative aggregate expectancy in each fixed-50 yearly audit.
    """

    V15_BO_SHORT_MIN_DIRECTIONAL_RETURN_4H = -0.0045
    V15_BO_SHORT_COUNTERMOVE_SCALE = 0.35

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
            or self._component(entry_tag) != "breakout"
            or side != "short"
        ):
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return stake
        if (
            float(row.get("symbol_return_4h", -np.inf))
            >= self.V15_BO_SHORT_MIN_DIRECTIONAL_RETURN_4H
        ):
            stake *= self.V15_BO_SHORT_COUNTERMOVE_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class _V15BreakoutLinearExhaustionRiskMixin:
    """De-risk late Breakout longs after an unusually linear 12-hour run."""

    V15_BO_LONG_MAX_HEALTHY_EFFICIENCY_12H = 0.81
    V15_BO_LONG_LINEAR_EXHAUSTION_SCALE = 0.35

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
            or self._component(entry_tag) != "breakout"
            or side != "long"
        ):
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return stake
        if (
            float(row.get("symbol_efficiency_12h", -np.inf))
            >= self.V15_BO_LONG_MAX_HEALTHY_EFFICIENCY_12H
        ):
            stake *= self.V15_BO_LONG_LINEAR_EXHAUSTION_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV15Fixed50CrossYearQuality35ResearchFreqtrade(
    _V15BreakoutLinearExhaustionRiskMixin,
    _V15BreakoutCountermoveShortRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    STRATEGY_VERSION = "breakout_v15_fixed50_cross_year_quality35_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_cross_year_quality35"


class BreakoutV15Fixed50CrossYearQuality20ResearchFreqtrade(
    BreakoutV15Fixed50CrossYearQuality35ResearchFreqtrade,
):
    V15_BO_SHORT_COUNTERMOVE_SCALE = 0.20
    V15_BO_LONG_LINEAR_EXHAUSTION_SCALE = 0.20
    STRATEGY_VERSION = "breakout_v15_fixed50_cross_year_quality20_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_cross_year_quality20"


class BreakoutV15Fixed50CrossYearQuality50ResearchFreqtrade(
    BreakoutV15Fixed50CrossYearQuality35ResearchFreqtrade,
):
    V15_BO_SHORT_COUNTERMOVE_SCALE = 0.50
    V15_BO_LONG_LINEAR_EXHAUSTION_SCALE = 0.50
    STRATEGY_VERSION = "breakout_v15_fixed50_cross_year_quality50_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_cross_year_quality50"


class BreakoutV15Fixed50Quality35Weak350GridLong65ResearchFreqtrade(
    _V15BreakoutLinearExhaustionRiskMixin,
    _V15BreakoutCountermoveShortRiskMixin,
    _V15GridLongCampaignLossBudgetMixin,
    BreakoutV15Fixed50WeakImpulseScore5Failure350ResearchFreqtrade,
):
    """Cross-year risk layer plus the strongest 2024 structural defenses."""

    V15_GRID_LONG_LOSS_LIMIT_R = 0.65
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_quality35_weak350_grid_long65_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_quality35_weak350_grid_long65"
    )

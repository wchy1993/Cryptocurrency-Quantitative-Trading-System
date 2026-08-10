from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from BreakoutV15Fixed50CrossYearQualityResearchFreqtrade import (
    _V15BreakoutLinearExhaustionRiskMixin,
)
from BreakoutV15Fixed50GridCampaignLossBudgetResearchFreqtrade import (
    _V15GridLongCampaignLossBudgetMixin,
)
from BreakoutV15Fixed50WeakImpulseScore5FailureExitResearchFreqtrade import (
    BreakoutV15Fixed50WeakImpulseScore5Failure350ResearchFreqtrade,
)


class _V15RobustBreakoutCountermoveShortRiskMixin:
    """Compress unconfirmed shorts without clipping extreme-bear runners.

    A slightly wider four-hour return boundary is paired with a completed-
    candle market-regime guard.  This avoids relying on a narrow threshold
    around a single 2026 winner while retaining the cross-year loss cluster.
    """

    V15_BO_SHORT_MIN_DIRECTIONAL_RETURN_4H = -0.006
    V15_BO_SHORT_MIN_MARKET_REGIME = -0.90
    V15_BO_SHORT_COUNTERMOVE_SCALE = 0.50

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
        unconfirmed = (
            float(row.get("symbol_return_4h", -np.inf))
            >= self.V15_BO_SHORT_MIN_DIRECTIONAL_RETURN_4H
            and float(row.get("market_regime_score", -np.inf))
            > self.V15_BO_SHORT_MIN_MARKET_REGIME
        )
        if unconfirmed:
            stake *= self.V15_BO_SHORT_COUNTERMOVE_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV15Fixed50RobustQualityL20S50ResearchFreqtrade(
    _V15BreakoutLinearExhaustionRiskMixin,
    _V15RobustBreakoutCountermoveShortRiskMixin,
    _V15GridLongCampaignLossBudgetMixin,
    BreakoutV15Fixed50WeakImpulseScore5Failure350ResearchFreqtrade,
):
    V15_BO_LONG_LINEAR_EXHAUSTION_SCALE = 0.20
    V15_BO_SHORT_COUNTERMOVE_SCALE = 0.50
    V15_GRID_LONG_LOSS_LIMIT_R = 0.65
    STRATEGY_VERSION = "breakout_v15_fixed50_robust_quality_l20_s50_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_robust_quality_l20_s50"


class BreakoutV15Fixed50RobustQualityL20S35ResearchFreqtrade(
    BreakoutV15Fixed50RobustQualityL20S50ResearchFreqtrade,
):
    V15_BO_SHORT_COUNTERMOVE_SCALE = 0.35
    STRATEGY_VERSION = "breakout_v15_fixed50_robust_quality_l20_s35_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_robust_quality_l20_s35"


class BreakoutV15Fixed50RobustQualityL35S50ResearchFreqtrade(
    BreakoutV15Fixed50RobustQualityL20S50ResearchFreqtrade,
):
    V15_BO_LONG_LINEAR_EXHAUSTION_SCALE = 0.35
    STRATEGY_VERSION = "breakout_v15_fixed50_robust_quality_l35_s50_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_robust_quality_l35_s50"


class BreakoutV15Fixed50RobustQualityL50S50ResearchFreqtrade(
    BreakoutV15Fixed50RobustQualityL20S50ResearchFreqtrade,
):
    V15_BO_LONG_LINEAR_EXHAUSTION_SCALE = 0.50
    STRATEGY_VERSION = "breakout_v15_fixed50_robust_quality_l50_s50_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_robust_quality_l50_s50"


class BreakoutV15Fixed50RobustQualityL20S50Eff80ResearchFreqtrade(
    BreakoutV15Fixed50RobustQualityL20S50ResearchFreqtrade,
):
    V15_BO_LONG_MAX_HEALTHY_EFFICIENCY_12H = 0.80
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_robust_quality_l20_s50_eff80_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_robust_quality_l20_s50_eff80"
    )


class BreakoutV15Fixed50RobustQualityL20S50Eff82ResearchFreqtrade(
    BreakoutV15Fixed50RobustQualityL20S50ResearchFreqtrade,
):
    V15_BO_LONG_MAX_HEALTHY_EFFICIENCY_12H = 0.82
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_robust_quality_l20_s50_eff82_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_robust_quality_l20_s50_eff82"
    )

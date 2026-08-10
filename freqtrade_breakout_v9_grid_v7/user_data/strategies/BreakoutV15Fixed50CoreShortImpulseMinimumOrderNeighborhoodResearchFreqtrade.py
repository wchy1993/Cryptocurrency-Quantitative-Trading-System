from __future__ import annotations

from BreakoutV15Fixed50CoreShortImpulseMinimumOrderResearchFreqtrade import (
    BreakoutV15Fixed50CoreShortImpulseFloor20ResearchFreqtrade,
)


class BreakoutV15Fixed50CoreShortImpulseFloor15ResearchFreqtrade(
    BreakoutV15Fixed50CoreShortImpulseFloor20ResearchFreqtrade,
):
    V15_BO_SHORT_WEAK_IMPULSE_MIN_SCALE = 0.15
    STRATEGY_VERSION = "breakout_v15_fixed50_core_short_impulse_floor15_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_core_short_impulse_floor15"


class BreakoutV15Fixed50CoreShortImpulseFloor25ResearchFreqtrade(
    BreakoutV15Fixed50CoreShortImpulseFloor20ResearchFreqtrade,
):
    V15_BO_SHORT_WEAK_IMPULSE_MIN_SCALE = 0.25
    STRATEGY_VERSION = "breakout_v15_fixed50_core_short_impulse_floor25_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_core_short_impulse_floor25"

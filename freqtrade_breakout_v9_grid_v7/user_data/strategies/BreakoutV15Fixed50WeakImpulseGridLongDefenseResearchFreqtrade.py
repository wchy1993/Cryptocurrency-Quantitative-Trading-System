from __future__ import annotations

from BreakoutV14AdaptiveAllocationRangeResearchFreqtrade import (
    _V14ContinuousEntryQualityRiskMixin,
)
from BreakoutV15Fixed50WeakImpulseScore5FailureExitResearchFreqtrade import (
    BreakoutV15Fixed50WeakImpulseScore5Failure425ResearchFreqtrade,
)


class BreakoutV15Fixed50WeakImpulseGridLong80ResearchFreqtrade(
    _V14ContinuousEntryQualityRiskMixin,
    BreakoutV15Fixed50WeakImpulseScore5Failure425ResearchFreqtrade,
):
    """Combine the weak-impulse failure exit with Grid-long defense.

    Breakout stake, Grid-short stake, signals, ranking, leverage, DCA and Max2
    remain frozen.  Only Grid-long risk is continuously reduced as completed-
    candle market efficiency rises from 0.24 to 0.32.  The three subclasses
    test a broad and ordered minimum-risk neighborhood.
    """

    V14_BO_LONG_RISK_MIN_SCALE = 1.0
    V14_GRID_SHORT_RISK_MIN_SCALE = 1.0
    V14_GRID_LONG_RISK_MIN_SCALE = 0.80
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_grid_long80_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_grid_long80"
    )


class BreakoutV15Fixed50WeakImpulseGridLong65ResearchFreqtrade(
    BreakoutV15Fixed50WeakImpulseGridLong80ResearchFreqtrade,
):
    V14_GRID_LONG_RISK_MIN_SCALE = 0.65
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_grid_long65_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_grid_long65"
    )


class BreakoutV15Fixed50WeakImpulseGridLong50ResearchFreqtrade(
    BreakoutV15Fixed50WeakImpulseGridLong80ResearchFreqtrade,
):
    V14_GRID_LONG_RISK_MIN_SCALE = 0.50
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_grid_long50_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_grid_long50"
    )

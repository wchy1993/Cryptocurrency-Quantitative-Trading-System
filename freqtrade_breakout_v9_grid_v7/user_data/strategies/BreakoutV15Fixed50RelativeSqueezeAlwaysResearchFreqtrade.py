from __future__ import annotations

from BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade import (
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
)
from BreakoutV14AdaptiveAllocationRangeResearchFreqtrade import (
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
)
from BreakoutV15Fixed50WeakImpulseGridLongDefenseResearchFreqtrade import (
    BreakoutV15Fixed50WeakImpulseGridLong50ResearchFreqtrade,
)
from BreakoutV15Fixed50WeakImpulseScore5FailureExitResearchFreqtrade import (
    BreakoutV15Fixed50WeakImpulseScore5Failure425ResearchFreqtrade,
)


class _V15RelativeSqueezeFromFirstTradeMixin:
    """Apply the selected Grid-short relative-squeeze defense immediately.

    V13 waits for a 3% realized-equity drawdown before compressing this
    structurally adverse Grid-short setup.  That leaves the first squeeze at
    full risk when the account is near a high-water mark.  The signal,
    ranking, Max2 occupancy, leverage, exits and the selected 20% risk floor
    are unchanged; only the drawdown prerequisite is removed.
    """

    V13_GRID_SQUEEZE_MIN_DRAWDOWN = 0.0


class BreakoutV15Fixed50RelativeSqueezeAlwaysV13ResearchFreqtrade(
    _V15RelativeSqueezeFromFirstTradeMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_relative_squeeze_always_v13_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_relative_squeeze_always_v13"
    )


class BreakoutV15Fixed50RelativeSqueezeAlwaysWeakImpulseResearchFreqtrade(
    _V15RelativeSqueezeFromFirstTradeMixin,
    BreakoutV15Fixed50WeakImpulseScore5Failure425ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_relative_squeeze_always_weak_impulse_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_relative_squeeze_always_weak_impulse"
    )


class BreakoutV15Fixed50RelativeSqueezeAlwaysGridLong50ResearchFreqtrade(
    _V15RelativeSqueezeFromFirstTradeMixin,
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_relative_squeeze_always_grid_long50_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_relative_squeeze_always_grid_long50"
    )


class BreakoutV15Fixed50RelativeSqueezeAlwaysCombinedResearchFreqtrade(
    _V15RelativeSqueezeFromFirstTradeMixin,
    BreakoutV15Fixed50WeakImpulseGridLong50ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_relative_squeeze_always_combined_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_relative_squeeze_always_combined"
    )

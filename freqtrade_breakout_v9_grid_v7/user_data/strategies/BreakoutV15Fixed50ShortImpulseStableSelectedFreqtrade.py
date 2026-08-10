from __future__ import annotations

from BreakoutV15Fixed50CoreShortImpulseMinimumOrderNeighborhoodResearchFreqtrade import (
    BreakoutV15Fixed50CoreShortImpulseFloor25ResearchFreqtrade,
)


class BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade(
    BreakoutV15Fixed50CoreShortImpulseFloor25ResearchFreqtrade,
):
    """Frozen fixed-50 research selection after cross-regime validation.

    This entry point intentionally inherits the validated Floor25 behavior
    without changing signals, exits, leverage, Max2 allocation, or the Grid
    sleeve.  Keeping a small wrapper gives later LIVE integration a stable
    class name while preserving the complete research lineage.
    """

    STRATEGY_VERSION = (
        "breakout_v15_fixed50_short_impulse_stable_selected_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_short_impulse_stable_selected"
    )

from __future__ import annotations

from BreakoutV15Fixed50GridCampaignLossBudgetResearchFreqtrade import (
    BreakoutV15Fixed50ConfirmedWeakGridLongLoss65ResearchFreqtrade,
)
from BreakoutV15Fixed50TransitionDefenseResearchFreqtrade import (
    _V15GridShortTransitionDefenseMixin,
)


class BreakoutV15Fixed50CoreTransition20ResearchFreqtrade(
    _V15GridShortTransitionDefenseMixin,
    BreakoutV15Fixed50ConfirmedWeakGridLongLoss65ResearchFreqtrade,
):
    """Cross-year core plus the isolated 2026 transition defense.

    Unlike the earlier transition candidate, this class deliberately omits
    the broad Breakout long/short stake overlays that changed the 2025 exact
    compounding path.  The added Grid-short rule has no signal-row matches in
    the fixed-50 2024 or 2025 audits and remains fully causal for LIVE.
    """

    V15_GRID_TRANSITION_SCALE = 0.20
    STRATEGY_VERSION = "breakout_v15_fixed50_core_transition20_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_core_transition20"


class BreakoutV15Fixed50CoreTransition35ResearchFreqtrade(
    BreakoutV15Fixed50CoreTransition20ResearchFreqtrade,
):
    """Neighborhood check for transition sizing stability."""

    V15_GRID_TRANSITION_SCALE = 0.35
    STRATEGY_VERSION = "breakout_v15_fixed50_core_transition35_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_core_transition35"

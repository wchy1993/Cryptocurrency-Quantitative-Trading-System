from __future__ import annotations

from BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade import (
    BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade,
)
from BreakoutV16Fixed50MtfAdaptiveBreakoutMax2ResearchFreqtrade import (
    _V16IntrahourPathMixin,
)
from GridV15Fixed50QualityProtectedResearchFreqtrade import (
    _GridV15QualityProtectedRiskMixin,
)


class BreakoutV16GridV15QualityPfCombinedResearchFreqtrade(
    _GridV15QualityProtectedRiskMixin,
    _V16IntrahourPathMixin,
    BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade,
):
    """Shared-account composition of selected V16 and Grid V15 PF control.

    The frozen combined parent keeps the established portfolio contract: two
    total slots, at most one Breakout campaign and one Grid campaign, with
    Breakout priority when only one slot remains.  The V16 overlay applies
    only its causal 15m path management to Breakout trades.  The Grid overlay
    applies only the frozen PF-control initial-risk scale to Grid campaigns.

    This research adapter intentionally does not inherit either sleeve-only
    filter.  Signals, ranks, leverage, DCA, exits and portfolio constraints
    therefore continue through the original cooperative method chain.
    """

    MAX_OPEN_TRADES = 2

    # Exact selected V16 behavior: keep every V15 entry and apply only the
    # bounded no-follow exit plus watched-runner profit floor.
    V16_REJECT_SHORT_EXHAUSTION = False

    # Exact frozen Grid V15 PF-control allocation contract.
    GRID_V15_SCORE4_SCALE = 0.60
    GRID_V15_SCORE5_SCALE = 0.25
    GRID_V15_LONG_SCALE = 0.35

    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_quality_pf_combined_research_20260812"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v16_grid_v15_quality_pf_combined_research"
    )

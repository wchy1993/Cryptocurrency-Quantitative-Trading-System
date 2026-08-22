from __future__ import annotations

from BreakoutV16GridV15SplitTrailRunnerResearchFreqtrade import (
    BreakoutV16GridV15SplitTrailV3ConservativeResearchFreqtrade,
)


class BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade(
    BreakoutV16GridV15SplitTrailV3ConservativeResearchFreqtrade
):
    """Frozen V3 production entry; all trading methods come from the selected path."""

    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_split_trail_v3_live_parity_20260818"
    )

from __future__ import annotations

from BreakoutV16Score2StructuralDecayResearchFreqtrade import (
    BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade,
)
from BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade import (
    BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade,
)


class BreakoutV16GridV15PrecisionGuardScore2DecayLiveParityFreqtrade(
    BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade,
):
    """Production adapter for the selected Score-2 structural-decay exit.

    Entry signals, sizing, leverage, Grid behavior, precision stops and all
    inherited exits stay owned by the frozen V16 + Grid V15 path.  This class
    only promotes the separately researched eight-hour Score-2 exit to the
    GUI/live entry point without editing the frozen strategy source.
    """

    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_precision_s2_decay_live_parity_20260819"
    )
    # Keep the frozen live state namespace so promoting the exit overlay does
    # not reset portfolio-governor state for an already-running account.
    ADAPTIVE_STATE_BASENAME = (
        BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade
        .ADAPTIVE_STATE_BASENAME
    )

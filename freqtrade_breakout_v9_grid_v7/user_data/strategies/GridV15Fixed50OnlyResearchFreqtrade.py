from __future__ import annotations

from BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade import (
    BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade,
)
from _GridOnlyResearchMixin import GridOnlyResearchMixin


class GridV15Fixed50OnlyResearchFreqtrade(
    GridOnlyResearchMixin,
    BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade,
):
    """Standalone research adapter for the selected v15 Grid sleeve."""

    STRATEGY_VERSION = "grid_v15_fixed50_only_research_20260810"
    ADAPTIVE_STATE_BASENAME = "grid_v15_fixed50_only_research"

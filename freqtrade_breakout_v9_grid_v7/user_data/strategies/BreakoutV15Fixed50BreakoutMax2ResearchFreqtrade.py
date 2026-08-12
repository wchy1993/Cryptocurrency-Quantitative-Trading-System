from __future__ import annotations

from _BreakoutMax2ResearchMixin import BreakoutMax2ResearchMixin
from BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade import (
    BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade,
)


class BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade(
    BreakoutMax2ResearchMixin,
    BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade,
):
    """Research-only v15 Breakout sleeve with two portfolio slots."""

    STRATEGY_VERSION = "breakout_v15_fixed50_breakout_max2_research_20260810"

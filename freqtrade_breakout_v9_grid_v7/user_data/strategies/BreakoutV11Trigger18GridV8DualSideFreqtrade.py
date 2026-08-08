from __future__ import annotations

from BreakoutV11GridV8DualSideFreqtrade import (
    BreakoutV11GridV8DualSideFreqtrade,
)


class BreakoutV11Trigger18GridV8DualSideFreqtrade(
    BreakoutV11GridV8DualSideFreqtrade
):
    """Ablation candidate with an 18% realized-equity risk boundary."""

    PORTFOLIO_GOVERNOR_TRIGGER = 0.18

from __future__ import annotations

from BreakoutV11PortfolioRecovery100ExhaustionQ112GridV8DualSideFreqtrade import (
    BreakoutV11PortfolioRecovery100ExhaustionQ112GridV8DualSideFreqtrade,
)


class BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade(
    BreakoutV11PortfolioRecovery100ExhaustionQ112GridV8DualSideFreqtrade
):
    """Portfolio guard with a 60% intermediate recovery allocation."""

    PORTFOLIO_RECOVERY_SCALE = 0.60

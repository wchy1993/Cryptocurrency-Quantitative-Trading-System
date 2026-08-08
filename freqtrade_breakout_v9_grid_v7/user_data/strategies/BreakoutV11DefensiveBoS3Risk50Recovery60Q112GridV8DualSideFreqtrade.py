from __future__ import annotations

from datetime import datetime
from typing import Any

from BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade import (
    BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade,
)


class BreakoutV11DefensiveBoS3Risk50Recovery60Q112GridV8DualSideFreqtrade(
    BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade
):
    """Reduce score-three Breakout risk only below the portfolio boundary."""

    DEFENSIVE_BO_SCORE3_SCALE = 0.50

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        stake = super().custom_stake_amount(
            pair,
            current_time,
            current_rate,
            proposed_stake,
            min_stake,
            max_stake,
            leverage,
            entry_tag,
            side,
            **kwargs,
        )
        tag = str(entry_tag or "")
        if stake <= 0.0 or not tag.startswith("bo_v9_s3_"):
            return stake
        equity = float(self.wallets.get_total_stake_amount())
        if self._portfolio_risk_scale(equity, current_time) >= 1.0:
            return stake
        scaled = stake * self.DEFENSIVE_BO_SCORE3_SCALE
        if min_stake is not None:
            scaled = max(float(min_stake), scaled)
        return min(max_stake, max(0.0, scaled))

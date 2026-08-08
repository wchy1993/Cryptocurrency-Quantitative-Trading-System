from __future__ import annotations

from pandas import DataFrame

from BreakoutV11Trigger18GridV8DualSideFreqtrade import (
    BreakoutV11Trigger18GridV8DualSideFreqtrade,
)


class BreakoutV11PortfolioRecovery100ExhaustionQ112GridV8DualSideFreqtrade(
    BreakoutV11Trigger18GridV8DualSideFreqtrade
):
    """Narrow-quality neighbor for the portfolio/exhaustion candidate.

    The inherited 1.12 late-short threshold still rejects the ONDO incident.
    Keeping it narrower than the 1.16 candidate tests whether broad filtering
    creates a worse replacement trade after the original signal is removed.
    """

    BO_EXTREME_SHORT_MIN_SESSION_MOVE = 0.15
    PORTFOLIO_RECOVERY_SCALE = 1.0

    def _apply_breakout_quality_guard(
        self, dataframe: DataFrame
    ) -> DataFrame:
        dataframe = super()._apply_breakout_quality_guard(dataframe)
        extreme_score_two_short = (
            dataframe["bo_entry_short"].astype(bool)
            & (dataframe["bo_score"] == 2)
            & (
                dataframe["bo_directional_session_move"]
                >= self.BO_EXTREME_SHORT_MIN_SESSION_MOVE
            )
        )
        dataframe["bo_v11_extreme_short_rejected"] = (
            extreme_score_two_short.astype(int)
        )
        dataframe.loc[extreme_score_two_short, "bo_entry_short"] = 0
        dataframe.loc[extreme_score_two_short, "bo_entry"] = 0
        return dataframe

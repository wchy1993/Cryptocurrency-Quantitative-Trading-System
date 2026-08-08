from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import Trade

from BreakoutV10FGridV8DualSideFreqtrade import (
    BreakoutV10FGridV8DualSideFreqtrade,
)


class BreakoutV11GridV8DualSideFreqtrade(
    BreakoutV10FGridV8DualSideFreqtrade
):
    """Independent v11 research candidate built on frozen v10F/Grid-v8.

    The two changes are deliberately orthogonal:

    * reject only low-conviction score-two shorts that arrive late in an
      already extended intraday move without volume confirmation;
    * reduce portfolio risk after a 20% *realized-equity* drawdown, then
      causally re-arm it when the last eight completed campaigns demonstrate
      recovery.  The 20% boundary sits above both the closed-trade and
      minute-wallet drawdowns of the frozen three- and six-month baselines.

    Grid-v8 entry, campaign management and exits remain frozen.
    """

    BO_LATE_SHORT_MIN_TREND_AGE = 20
    BO_LATE_SHORT_MIN_SESSION_MOVE = 0.060
    BO_LATE_SHORT_MAX_VOLUME_RATIO = 1.10
    BO_LATE_SHORT_MAX_QUALITY = 1.12

    PORTFOLIO_GOVERNOR_TRIGGER = 0.20
    PORTFOLIO_RECENT_WINDOW = 8
    PORTFOLIO_DEFENSIVE_SCALE = 0.15
    PORTFOLIO_RECOVERY_SCALE = 0.50
    PORTFOLIO_RECOVERY_MIN_WINS = 4
    PORTFOLIO_RECOVERY_MIN_RETURN = 0.0
    PORTFOLIO_FULL_RISK_MIN_WINS = 5
    PORTFOLIO_FULL_RISK_MIN_RETURN = 0.60

    @staticmethod
    def _consecutive_bars(condition: pd.Series) -> pd.Series:
        enabled = condition.fillna(False).astype(bool)
        groups = (~enabled).cumsum()
        return enabled.astype(int).groupby(groups).cumsum().astype(int)

    def _apply_breakout_quality_guard(self, dataframe: DataFrame) -> DataFrame:
        dataframe["bo_short_trend_age"] = self._consecutive_bars(
            dataframe["close"] < dataframe["fast_ema"]
        )
        dataframe["bo_directional_session_move"] = (
            dataframe["session_open"] - dataframe["close"]
        ) / dataframe["session_open"].clip(lower=1e-12)
        late_thin_score_two_short = (
            dataframe["bo_entry_short"].astype(bool)
            & (dataframe["bo_score"] == 2)
            & (
                dataframe["bo_short_trend_age"]
                >= self.BO_LATE_SHORT_MIN_TREND_AGE
            )
            & (
                dataframe["bo_directional_session_move"]
                >= self.BO_LATE_SHORT_MIN_SESSION_MOVE
            )
            & (
                dataframe["bo_volume_ratio"]
                <= self.BO_LATE_SHORT_MAX_VOLUME_RATIO
            )
            & (
                dataframe["bo_short_quality"]
                <= self.BO_LATE_SHORT_MAX_QUALITY
            )
        )
        dataframe["bo_v11_late_short_rejected"] = (
            late_thin_score_two_short.astype(int)
        )
        dataframe.loc[late_thin_score_two_short, "bo_entry_short"] = 0
        dataframe.loc[late_thin_score_two_short, "bo_entry"] = 0
        return dataframe

    def _populate_breakout(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_breakout(dataframe)
        return self._apply_breakout_quality_guard(dataframe)

    @staticmethod
    def _closed_trade_return(trade: Trade) -> float | None:
        value = getattr(trade, "close_profit", None)
        if value is None:
            return None
        result = float(value)
        return result if np.isfinite(result) else None

    def _recent_portfolio_returns(
        self, current_time: datetime
    ) -> tuple[float, ...]:
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        closed: list[tuple[datetime, float]] = []
        for trade in Trade.get_trades_proxy(is_open=False):
            close_time = self._trade_close_time(trade)
            value = self._closed_trade_return(trade)
            if close_time is None or close_time > boundary or value is None:
                continue
            closed.append((close_time, value))
        closed.sort(key=lambda item: item[0])
        return tuple(
            value
            for _close_time, value in closed[-self.PORTFOLIO_RECENT_WINDOW :]
        )

    def _portfolio_risk_scale(
        self, equity: float, current_time: datetime
    ) -> float:
        if equity <= 0.0 or self._peak_equity <= 0.0:
            return 0.0
        drawdown = max(0.0, 1.0 - equity / self._peak_equity)
        if drawdown < self.PORTFOLIO_GOVERNOR_TRIGGER:
            return 1.0

        recent = self._recent_portfolio_returns(current_time)
        if len(recent) < self.PORTFOLIO_RECENT_WINDOW:
            return self.PORTFOLIO_DEFENSIVE_SCALE
        wins = sum(value > 0.0 for value in recent)
        aggregate_return = float(sum(recent))
        if (
            wins >= self.PORTFOLIO_FULL_RISK_MIN_WINS
            and aggregate_return >= self.PORTFOLIO_FULL_RISK_MIN_RETURN
        ):
            return 1.0
        if (
            wins >= self.PORTFOLIO_RECOVERY_MIN_WINS
            and aggregate_return >= self.PORTFOLIO_RECOVERY_MIN_RETURN
        ):
            return self.PORTFOLIO_RECOVERY_SCALE
        return self.PORTFOLIO_DEFENSIVE_SCALE

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
        if stake <= 0.0:
            return 0.0
        equity = float(self.wallets.get_total_stake_amount())
        # The frozen combined adapter delegates Grid sizing directly to the
        # Grid-v8 sleeve, whose standalone implementation has no portfolio
        # high-water mark.  Initialize/update it here for both components.
        self._peak_equity = max(self._peak_equity, equity)
        scale = self._portfolio_risk_scale(equity, current_time)
        return min(max_stake, max(0.0, stake * scale))


class BreakoutV11EntryOnlyGridV8DualSideFreqtrade(
    BreakoutV11GridV8DualSideFreqtrade
):
    """Ablation candidate: v11 entry guard without drawdown resizing."""

    PORTFOLIO_GOVERNOR_TRIGGER = 1.0


class BreakoutV11ConservativeGridV8DualSideFreqtrade(
    BreakoutV11GridV8DualSideFreqtrade
):
    """Ablation candidate with a more defensive underwater allocation."""

    PORTFOLIO_DEFENSIVE_SCALE = 0.10
    PORTFOLIO_RECOVERY_SCALE = 0.40
    PORTFOLIO_FULL_RISK_MIN_RETURN = 0.80

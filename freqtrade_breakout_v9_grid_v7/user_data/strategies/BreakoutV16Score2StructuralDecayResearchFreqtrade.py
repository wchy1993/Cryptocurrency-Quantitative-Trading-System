from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from pandas import DataFrame, Series
from freqtrade.strategy import Trade

from BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade import (
    BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade,
)


class _V16Score2StructuralDecayExitMixin:
    """Causally retire a stalled Score-2 short after confirmed reversal.

    The frozen V16 + Grid V15 production class remains untouched.  This
    research-only overlay changes only ``custom_exit`` and requires a fully
    completed hourly rejection plus absent four-hour downside progress.  It
    deliberately ignores an unfinished green 4h candle, so a normal pause in
    a fresh downside breakout cannot trigger the rule.
    """

    # A Score-2 breakout gets a full eight-hour opportunity to develop before
    # this failure overlay may retire it.  The six-hour neighbour produced the
    # same frozen-path events, while five hours admitted the early DASH bounce;
    # eight hours remains the selected, directly cross-year-tested boundary.
    V16_S2_MIN_HOLD_MINUTES = 480.0
    V16_S2_MAX_MFE_R = 1.25
    V16_S2_MAX_CURRENT_R = 0.00

    V16_S2_MIN_RANGE_ATR = 0.65
    V16_S2_MIN_LOWER_WICK_RATIO = 0.30
    V16_S2_MIN_CLOSE_POSITION = 0.80
    V16_S2_MIN_BULL_BODY_ATR = 0.15
    V16_S2_MIN_RECLAIM_ATR = 0.10
    V16_S2_MAX_SHORT_MOMENTUM_4H_ATR = 0.00

    V16_S2_DECISION_WINDOW_MINUTES = 5.0
    V16_S2_EXIT_REASON = "bo_v16_s2_structure_decay"

    @staticmethod
    def _v16_s2_utc(value: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _v16_s2_completed_history(
        self,
        pair: str,
        current_time: datetime,
        trade: Trade,
    ) -> tuple[Series, Series, Series, DataFrame] | None:
        """Return only structure that was knowable at ``current_time``."""

        if self.dp is None:
            return None
        try:
            frame, _updated = self.dp.get_analyzed_dataframe(
                pair,
                self.timeframe,
            )
        except Exception:
            return None
        required = {"date", "open", "high", "low", "close", "atr"}
        if frame is None or frame.empty or not required.issubset(frame.columns):
            return None

        now = self._v16_s2_utc(current_time)
        boundary = now.floor(self.timeframe)
        dates = pd.DatetimeIndex(
            pd.to_datetime(frame["date"], utc=True)
        ).as_unit("ns")
        completed_count = int(dates.searchsorted(boundary, side="left"))
        if completed_count < 5:
            return None

        completed = frame.iloc[:completed_count]
        latest = completed.iloc[-1]
        previous = completed.iloc[-2]
        four_hours_ago = completed.iloc[-5]
        latest_date = self._v16_s2_utc(latest["date"])
        previous_date = self._v16_s2_utc(previous["date"])
        four_hours_ago_date = self._v16_s2_utc(four_hours_ago["date"])
        if (
            latest_date - previous_date != pd.Timedelta(hours=1)
            or latest_date - four_hours_ago_date != pd.Timedelta(hours=4)
        ):
            return None

        available_at = latest_date + pd.Timedelta(hours=1)
        decision_age = (now - available_at).total_seconds() / 60.0
        if not (
            0.0
            <= decision_age
            <= float(self.V16_S2_DECISION_WINDOW_MINUTES)
        ):
            return None

        opened_at = self._v16_s2_utc(self._trade_open_time(trade)).floor(
            self.timeframe
        )
        campaign_dates = dates[:completed_count]
        campaign = completed.loc[campaign_dates >= opened_at]
        if campaign.empty:
            return None
        return previous, latest, four_hours_ago, campaign

    def _v16_s2_structure_features(
        self,
        trade: Trade,
        previous: Series,
        latest: Series,
        four_hours_ago: Series,
        campaign: DataFrame,
    ) -> dict[str, float] | None:
        if not trade.is_short:
            return None
        atr = float(latest.get("atr", 0.0) or 0.0)
        if not pd.notna(atr) or atr <= 0.0:
            return None

        open_rate = float(latest["open"])
        high_rate = float(latest["high"])
        low_rate = float(latest["low"])
        close_rate = float(latest["close"])
        candle_range = max(high_rate - low_rate, 1e-12)

        lower_wick = max(0.0, min(open_rate, close_rate) - low_rate)
        completed_low = min(
            float(campaign["low"].min()),
            float(getattr(trade, "min_rate", trade.open_rate) or trade.open_rate),
        )
        return {
            "range_atr": candle_range / atr,
            "lower_wick_ratio": lower_wick / candle_range,
            "close_position": (close_rate - low_rate) / candle_range,
            "bull_body_atr": (close_rate - open_rate) / atr,
            "reclaim_atr": (
                close_rate - float(previous["close"])
            )
            / atr,
            "short_momentum_4h_atr": -(
                close_rate - float(four_hours_ago["close"])
            )
            / atr,
            "maximum_r": self._trade_r(trade, completed_low),
            "current_r": self._trade_r(trade, close_rate),
        }

    def _v16_s2_decay_reason(
        self,
        *,
        score: int,
        is_short: bool,
        holding_minutes: float,
        features: dict[str, float],
    ) -> str | None:
        if int(score) != 2 or not is_short:
            return None
        if float(holding_minutes) < float(self.V16_S2_MIN_HOLD_MINUTES):
            return None
        if float(features["maximum_r"]) >= float(self.V16_S2_MAX_MFE_R):
            return None
        if float(features["current_r"]) > float(self.V16_S2_MAX_CURRENT_R):
            return None
        if float(features["range_atr"]) < float(self.V16_S2_MIN_RANGE_ATR):
            return None
        if float(features["lower_wick_ratio"]) < float(
            self.V16_S2_MIN_LOWER_WICK_RATIO
        ):
            return None
        if float(features["close_position"]) < float(
            self.V16_S2_MIN_CLOSE_POSITION
        ):
            return None
        if float(features["bull_body_atr"]) < float(
            self.V16_S2_MIN_BULL_BODY_ATR
        ):
            return None
        if float(features["reclaim_atr"]) < float(
            self.V16_S2_MIN_RECLAIM_ATR
        ):
            return None
        if float(features["short_momentum_4h_atr"]) > float(
            self.V16_S2_MAX_SHORT_MOMENTUM_4H_ATR
        ):
            return None
        return self.V16_S2_EXIT_REASON

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        inherited = super().custom_exit(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
            **kwargs,
        )
        if inherited is not None:
            return inherited
        if (
            self._component(getattr(trade, "enter_tag", None)) != "breakout"
            or not bool(getattr(trade, "is_short", False))
            or int(trade.get_custom_data("bo_score", 0) or 0) != 2
        ):
            return None

        holding_minutes = (
            self._v16_s2_utc(current_time)
            - self._v16_s2_utc(self._trade_open_time(trade))
        ).total_seconds() / 60.0
        if holding_minutes < float(self.V16_S2_MIN_HOLD_MINUTES):
            return None

        # A new completed 1h candle can only become available at the top of
        # the hour.  Avoid repeatedly asking the provider for the same hourly
        # frame during the other 54 detail candles in an exact 1m backtest.
        now = self._v16_s2_utc(current_time)
        minute_age = now.minute + now.second / 60.0
        if minute_age > float(self.V16_S2_DECISION_WINDOW_MINUTES):
            return None

        history = self._v16_s2_completed_history(
            pair,
            current_time,
            trade,
        )
        if history is None:
            return None
        previous, latest, four_hours_ago, campaign = history
        features = self._v16_s2_structure_features(
            trade,
            previous,
            latest,
            four_hours_ago,
            campaign,
        )
        if features is None:
            return None
        return self._v16_s2_decay_reason(
            score=int(trade.get_custom_data("bo_score", 0) or 0),
            is_short=bool(trade.is_short),
            holding_minutes=holding_minutes,
            features=features,
        )


class BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade(
    _V16Score2StructuralDecayExitMixin,
    BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade,
):
    """Central research candidate; production and GUI classes stay frozen."""

    STRATEGY_VERSION = "v16_grid_v15_precision_s2_decay_20260819"
    ADAPTIVE_STATE_BASENAME = "v16_grid_v15_precision_s2_decay"


class BreakoutV16GridV15PrecisionGuardScore2DecayLateResearchFreqtrade(
    BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade,
):
    """Neighbour requiring ten elapsed hours before structural retirement."""

    V16_S2_MIN_HOLD_MINUTES = 600.0
    STRATEGY_VERSION = "v16_grid_v15_precision_s2_decay_late_20260819"
    ADAPTIVE_STATE_BASENAME = "v16_grid_v15_precision_s2_decay_late"


class BreakoutV16GridV15PrecisionGuardScore2DecayEarly6ResearchFreqtrade(
    BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade,
):
    """Sensitivity neighbour allowing structural retirement after six hours."""

    V16_S2_MIN_HOLD_MINUTES = 360.0
    STRATEGY_VERSION = "v16_grid_v15_precision_s2_decay_early6_20260819"
    ADAPTIVE_STATE_BASENAME = "v16_grid_v15_precision_s2_decay_early6"


class BreakoutV16GridV15PrecisionGuardScore2DecayStrict100ResearchFreqtrade(
    BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade,
):
    """Sensitivity neighbour requiring an even lower 1.00R campaign MFE."""

    V16_S2_MAX_MFE_R = 1.00
    STRATEGY_VERSION = "v16_grid_v15_precision_s2_decay_strict100_20260819"
    ADAPTIVE_STATE_BASENAME = "v16_grid_v15_precision_s2_decay_strict100"


class BreakoutV16GridV15PrecisionGuardScore2DecayStrictShapeResearchFreqtrade(
    BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade,
):
    """Conservative shape neighbour for the clearest DASH-like rejection."""

    V16_S2_MIN_RANGE_ATR = 0.80
    V16_S2_MIN_LOWER_WICK_RATIO = 0.50
    STRATEGY_VERSION = "v16_grid_v15_precision_s2_decay_strict_shape_20260819"
    ADAPTIVE_STATE_BASENAME = "v16_grid_v15_precision_s2_decay_strict_shape"


class BreakoutV16GridV15PrecisionGuardScore2DecayBroadResearchFreqtrade(
    BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade,
):
    """Rejected mixed-boundary sensitivity neighbour kept for auditability."""

    V16_S2_MAX_MFE_R = 1.25
    V16_S2_MAX_CURRENT_R = 0.25
    V16_S2_MIN_RANGE_ATR = 0.65
    V16_S2_MIN_LOWER_WICK_RATIO = 0.40
    V16_S2_MIN_CLOSE_POSITION = 0.75
    V16_S2_MIN_BULL_BODY_ATR = 0.00
    V16_S2_MIN_RECLAIM_ATR = 0.00
    V16_S2_MAX_SHORT_MOMENTUM_4H_ATR = 0.25
    STRATEGY_VERSION = "v16_grid_v15_precision_s2_decay_broad_20260819"
    ADAPTIVE_STATE_BASENAME = "v16_grid_v15_precision_s2_decay_broad"

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import Order, Trade, stoploss_from_absolute

from BreakoutV9GridV7Freqtrade import BreakoutV9GridV7Freqtrade


class BreakoutV10GridV7Freqtrade(BreakoutV9GridV7Freqtrade):
    """Structural Breakout v10 research candidate with frozen Grid v7.

    Breakout changes are deliberately isolated from the Grid implementation:

    - discard the weakest score-one signals;
    - avoid late/crowded short breakouts in highly efficient broad markets;
    - use liquidation-aware leverage on contracts capped at 10x;
    - arm selective profit floors from completed 1h closes, not transient
      one-minute wicks, so large trend runners retain room to re-expand.
    """

    BO_MIN_SCORE = 2
    BO_SHORT_MARKET_EFFICIENCY_MAX = 0.35
    BO_LOW_MAX_LEVERAGE_THRESHOLD = 10.0
    BO_LOW_MAX_LEVERAGE = 5.0

    BO_CAPTURE_CONFIRMED_TRIGGER_1 = 3.0
    BO_CAPTURE_CONFIRMED_FLOOR_1 = 0.25
    BO_CAPTURE_CONFIRMED_TRIGGER_2 = 4.0
    BO_CAPTURE_CONFIRMED_FLOOR_2 = 1.0
    BO_LONG_CONFIRMED_TRIGGER_1 = 5.0
    BO_LONG_CONFIRMED_FLOOR_1 = 0.50
    BO_LONG_CONFIRMED_TRIGGER_2 = 6.5
    BO_LONG_CONFIRMED_FLOOR_2 = 1.0

    def _populate_breakout(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_breakout(dataframe)
        weak = dataframe["bo_entry"].astype(bool) & (
            dataframe["bo_score"] < self.BO_MIN_SCORE
        )
        crowded_short = dataframe["bo_entry_short"].astype(bool) & (
            dataframe["market_efficiency"]
            > self.BO_SHORT_MARKET_EFFICIENCY_MAX
        )
        rejected = weak | crowded_short
        dataframe.loc[rejected, "bo_entry_long"] = 0
        dataframe.loc[rejected, "bo_entry_short"] = 0
        dataframe.loc[rejected, "bo_entry"] = 0

        # Lower is better. Keep the frozen range term dominant while using
        # causal quality/regime information to break close cross-pair ties.
        side_quality = np.where(
            dataframe["bo_entry_long"].astype(bool),
            dataframe["bo_long_quality"],
            dataframe["bo_short_quality"],
        )
        dataframe["bo_v10_rank"] = (
            dataframe["bo_range_atr"]
            - 0.08 * dataframe["bo_score"].astype(float)
            - 0.03 * pd.Series(side_quality, index=dataframe.index).astype(float)
            - 0.02 * dataframe["bo_regime"].clip(lower=-2.0, upper=2.0)
        )
        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        pair = str(metadata.get("pair") or "")
        if not pair:
            return dataframe
        available_times = (
            pd.to_datetime(dataframe["date"], utc=True)
            + pd.Timedelta(self.timeframe)
        )
        breakout = (
            dataframe["bo_entry_long"].astype(bool)
            | dataframe["bo_entry_short"].astype(bool)
        )
        ranks = self._candidate_ranks.setdefault(pair, {})
        for index in dataframe.index[breakout]:
            ranks[
                (
                    "breakout",
                    int(available_times.at[index].timestamp()),
                )
            ] = float(dataframe.at[index, "bo_v10_rank"])
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        if (
            self._component(entry_tag) == "breakout"
            and max_leverage
            <= self.BO_LOW_MAX_LEVERAGE_THRESHOLD + 1e-12
        ):
            return min(self.BO_LOW_MAX_LEVERAGE, max_leverage)
        return super().leverage(
            pair,
            current_time,
            current_rate,
            proposed_leverage,
            max_leverage,
            entry_tag,
            side,
            **kwargs,
        )

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        super().order_filled(pair, trade, order, current_time, **kwargs)
        if (
            self._component(trade.enter_tag) == "breakout"
            and order.ft_order_side == trade.entry_side
            and trade.nr_of_successful_entries == 1
        ):
            trade.set_custom_data("bo_v10_confirmed_peak_r", -1000.0)
            trade.set_custom_data("bo_v10_last_confirmed_date", "")

    def _confirmed_peak_r(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
    ) -> float:
        peak = float(
            trade.get_custom_data("bo_v10_confirmed_peak_r", -1000.0)
        )
        row = self._latest_row(pair, current_time)
        if row is None:
            return peak
        candle_date = pd.Timestamp(row["date"]).isoformat()
        if candle_date == str(
            trade.get_custom_data("bo_v10_last_confirmed_date", "")
        ):
            return peak
        confirmed_r = self._trade_r(trade, float(row["close"]))
        peak = max(peak, confirmed_r)
        trade.set_custom_data("bo_v10_confirmed_peak_r", peak)
        trade.set_custom_data("bo_v10_last_confirmed_date", candle_date)
        return peak

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        component = self._component(trade.enter_tag)
        if component != "breakout":
            return super().custom_stoploss(
                pair,
                trade,
                current_time,
                current_rate,
                current_profit,
                after_fill,
                **kwargs,
            )

        atr_value = max(
            float(trade.get_custom_data("initial_atr", 0.0)), 1e-12
        )
        side = -1.0 if trade.is_short else 1.0
        stop_rate = trade.open_rate - side * self.BO_STOP_ATR * atr_value
        confirmed_peak = self._confirmed_peak_r(pair, trade, current_time)
        locked_r: float | None = None

        if bool(trade.get_custom_data("bo_capture", False)):
            if confirmed_peak >= self.BO_CAPTURE_CONFIRMED_TRIGGER_1:
                locked_r = self.BO_CAPTURE_CONFIRMED_FLOOR_1
            if confirmed_peak >= self.BO_CAPTURE_CONFIRMED_TRIGGER_2:
                locked_r = max(
                    locked_r or 0.0,
                    self.BO_CAPTURE_CONFIRMED_FLOOR_2,
                )

        if bool(trade.get_custom_data("bo_long_floor", False)):
            if confirmed_peak >= self.BO_LONG_CONFIRMED_TRIGGER_1:
                locked_r = max(
                    locked_r or 0.0,
                    self.BO_LONG_CONFIRMED_FLOOR_1,
                )
            if confirmed_peak >= self.BO_LONG_CONFIRMED_TRIGGER_2:
                locked_r = max(
                    locked_r or 0.0,
                    self.BO_LONG_CONFIRMED_FLOOR_2,
                )

        favorable_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate if favorable_value is None else favorable_value
        )
        maximum_r = self._trade_r(trade, favorable_rate)
        if maximum_r >= 15.0:
            locked_r = max(locked_r or 0.0, maximum_r - 12.0)

        if locked_r is not None:
            unit_risk = max(
                float(trade.get_custom_data("initial_unit_risk", 0.0)),
                1e-12,
            )
            stop_rate = trade.open_rate + side * (
                locked_r * unit_risk
                + 2.0 * self.SIDE_COST * trade.open_rate
            )

        value = stoploss_from_absolute(
            stop_rate,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )
        return abs(float(value))


class BreakoutV10BGridV7Freqtrade(BreakoutV10GridV7Freqtrade):
    """v10 with a less restrictive short-market maturity boundary."""

    BO_SHORT_MARKET_EFFICIENCY_MAX = 0.40


class BreakoutV10CGridV7Freqtrade(BreakoutV10BGridV7Freqtrade):
    """v10B entry/risk/exit logic using the frozen v9 range ranking."""

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        return BreakoutV9GridV7Freqtrade.populate_entry_trend(
            self, dataframe, metadata
        )


class BreakoutV10DGridV7Freqtrade(BreakoutV10GridV7Freqtrade):
    """v10 with a causal, protected extension for confirmed trend runners.

    The frozen v9 20-hour time stop is retained for ordinary breakouts.  A
    trade can become a runner only when it is already materially profitable
    and the latest completed 1h candle still confirms both symbol structure
    and a supportive cross-market state.  Extended trades receive a hard
    profit floor and a wide R-based trail, preserving convex winners without
    turning every small gain into an open-ended hold.
    """

    BO_RUNNER_MIN_SCORE = 4
    BO_RUNNER_MIN_R = 2.0
    BO_RUNNER_MAX_HOLD_MINUTES = 2880
    BO_RUNNER_MIN_EMA55_ATR = 1.50
    BO_RUNNER_MIN_EMA_SPREAD_ATR = 0.50
    BO_RUNNER_MIN_RETURN_8H = -0.03
    BO_RUNNER_MIN_DIRECTIONAL_BREADTH = 0.0
    BO_RUNNER_MOMENTUM_RETURN_4H = 0.02
    BO_RUNNER_MOMENTUM_MARKET_EFFICIENCY = 0.40
    BO_RUNNER_FLOOR_R = 1.0
    BO_RUNNER_TRAIL_GAP_R = 8.0

    @staticmethod
    def _directional_value(trade: Trade, value: float) -> float:
        return -value if trade.is_short else value

    def _runner_environment_valid(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
    ) -> bool:
        row = self._latest_row(pair, current_time)
        if row is None:
            return False

        directional_ema55 = self._directional_value(
            trade, float(row["symbol_ema55_atr"])
        )
        ema_spread_atr = (
            float(row["fast_ema"]) - float(row["slow_ema"])
        ) / max(float(row["atr"]), 1e-12)
        directional_spread = self._directional_value(
            trade, ema_spread_atr
        )
        directional_return_4h = self._directional_value(
            trade, float(row["symbol_return_4h"])
        )
        close_8h_ago = float(row.get("close_8h_ago", np.nan))
        if not np.isfinite(close_8h_ago) or close_8h_ago <= 0.0:
            return False
        return_8h = float(row["close"]) / close_8h_ago - 1.0
        directional_return_8h = self._directional_value(trade, return_8h)
        breadth = float(row["breadth"])
        directional_breadth = (
            breadth - 0.5 if not trade.is_short else 0.5 - breadth
        )
        broad_support = (
            directional_breadth
            >= self.BO_RUNNER_MIN_DIRECTIONAL_BREADTH
        )
        momentum_support = (
            directional_return_4h
            >= self.BO_RUNNER_MOMENTUM_RETURN_4H
            and float(row["market_efficiency"])
            >= self.BO_RUNNER_MOMENTUM_MARKET_EFFICIENCY
        )
        return bool(
            directional_ema55 >= self.BO_RUNNER_MIN_EMA55_ATR
            and directional_spread >= self.BO_RUNNER_MIN_EMA_SPREAD_ATR
            and directional_return_8h >= self.BO_RUNNER_MIN_RETURN_8H
            and (broad_support or momentum_support)
        )

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe["close_8h_ago"] = dataframe["close"].shift(8)
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        base_value = super().custom_stoploss(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
            after_fill,
            **kwargs,
        )
        if (
            self._component(trade.enter_tag) != "breakout"
            or not bool(
                trade.get_custom_data("bo_v10_runner_extended", False)
            )
        ):
            return base_value

        favorable_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate if favorable_value is None else favorable_value
        )
        maximum_r = self._trade_r(trade, favorable_rate)
        locked_r = max(
            self.BO_RUNNER_FLOOR_R,
            maximum_r - self.BO_RUNNER_TRAIL_GAP_R,
        )
        unit_risk = max(
            float(trade.get_custom_data("initial_unit_risk", 0.0)),
            1e-12,
        )
        side = -1.0 if trade.is_short else 1.0
        stop_rate = trade.open_rate + side * (
            locked_r * unit_risk
            + 2.0 * self.SIDE_COST * trade.open_rate
        )
        runner_value = abs(
            float(
                stoploss_from_absolute(
                    stop_rate,
                    current_rate=current_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage,
                )
            )
        )
        if base_value is None:
            return runner_value
        return min(abs(float(base_value)), runner_value)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        if self._component(trade.enter_tag) != "breakout":
            return super().custom_exit(
                pair,
                trade,
                current_time,
                current_rate,
                current_profit,
                **kwargs,
            )

        current_r = self._trade_r(trade, current_rate)
        if current_r >= self.BO_TAKE_PROFIT_R:
            return "bo_v10_take_profit_60r"

        holding_minutes = int(
            (current_time - self._trade_open_time(trade)).total_seconds()
            / 60
        )
        extended = bool(
            trade.get_custom_data("bo_v10_runner_extended", False)
        )
        if not extended and holding_minutes >= self.BO_MAX_HOLD_MINUTES:
            score = int(trade.get_custom_data("bo_score", 0))
            if (
                score >= self.BO_RUNNER_MIN_SCORE
                and current_r >= self.BO_RUNNER_MIN_R
                and self._runner_environment_valid(
                    pair, trade, current_time
                )
            ):
                trade.set_custom_data("bo_v10_runner_extended", True)
                trade.set_custom_data(
                    "bo_v10_runner_armed_at", current_time.isoformat()
                )
                return None
            return "bo_v10_time_stop"

        if extended and holding_minutes >= self.BO_RUNNER_MAX_HOLD_MINUTES:
            return "bo_v10_runner_time_stop"
        return None


class BreakoutV10EGridV7Freqtrade(BreakoutV10GridV7Freqtrade):
    """v10 with a bounded continuation exception for score-two shorts.

    Broad-market efficiency above the normal boundary remains a hard rejection
    for crowded score-three shorts.  The exception admits only a narrow,
    moderate-volume continuation profile where symbol efficiency and EMA
    displacement have not reached exhaustion.  This is a distinct market-state
    rule rather than a global relaxation of the boundary.
    """

    BO_CONTINUATION_MARKET_EFFICIENCY_MAX = 0.40
    BO_CONTINUATION_VOLUME_MAX = 1.50
    BO_CONTINUATION_EFFICIENCY_MIN = 0.25
    BO_CONTINUATION_EFFICIENCY_MAX = 0.55
    BO_CONTINUATION_EMA55_ATR_MIN = 1.50
    BO_CONTINUATION_EMA55_ATR_MAX = 3.00

    def _populate_breakout(self, dataframe: DataFrame) -> DataFrame:
        dataframe = BreakoutV9GridV7Freqtrade._populate_breakout(
            self, dataframe
        )
        weak = dataframe["bo_entry"].astype(bool) & (
            dataframe["bo_score"] < self.BO_MIN_SCORE
        )
        controlled_continuation = (
            dataframe["bo_entry_short"].astype(bool)
            & (dataframe["bo_score"] == 2)
            & (
                dataframe["market_efficiency"]
                <= self.BO_CONTINUATION_MARKET_EFFICIENCY_MAX
            )
            & (
                dataframe["bo_volume_ratio"]
                <= self.BO_CONTINUATION_VOLUME_MAX
            )
            & (
                -dataframe["symbol_efficiency_12h"]
            ).between(
                self.BO_CONTINUATION_EFFICIENCY_MIN,
                self.BO_CONTINUATION_EFFICIENCY_MAX,
            )
            & (
                -dataframe["symbol_ema55_atr"]
            ).between(
                self.BO_CONTINUATION_EMA55_ATR_MIN,
                self.BO_CONTINUATION_EMA55_ATR_MAX,
            )
        )
        crowded_short = (
            dataframe["bo_entry_short"].astype(bool)
            & (
                dataframe["market_efficiency"]
                > self.BO_SHORT_MARKET_EFFICIENCY_MAX
            )
            & ~controlled_continuation
        )
        rejected = weak | crowded_short
        dataframe.loc[rejected, "bo_entry_long"] = 0
        dataframe.loc[rejected, "bo_entry_short"] = 0
        dataframe.loc[rejected, "bo_entry"] = 0

        side_quality = np.where(
            dataframe["bo_entry_long"].astype(bool),
            dataframe["bo_long_quality"],
            dataframe["bo_short_quality"],
        )
        dataframe["bo_v10_rank"] = (
            dataframe["bo_range_atr"]
            - 0.08 * dataframe["bo_score"].astype(float)
            - 0.03 * pd.Series(side_quality, index=dataframe.index).astype(float)
            - 0.02 * dataframe["bo_regime"].clip(lower=-2.0, upper=2.0)
        )
        return dataframe


class BreakoutV10FGridV7Freqtrade(BreakoutV10EGridV7Freqtrade):
    """v10E with a cross-sectional exhaustion guard for score-four longs.

    A sharp four-hour expansion in market breadth means the breakout is
    arriving after most of the universe has already moved above its fast EMA.
    Score-five convex setups remain eligible, while score-four long entries are
    rejected in this late, correlated chase state.
    """

    BO_LONG_BREADTH_ACCELERATION_MAX = 0.10

    def _populate_breakout(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_breakout(dataframe)
        late_correlated_long = (
            dataframe["bo_entry_long"].astype(bool)
            & (dataframe["bo_score"] == 4)
            & (
                dataframe["breadth_change_4h"]
                > self.BO_LONG_BREADTH_ACCELERATION_MAX
            )
        )
        dataframe.loc[late_correlated_long, "bo_entry_long"] = 0
        dataframe.loc[late_correlated_long, "bo_entry"] = 0
        return dataframe

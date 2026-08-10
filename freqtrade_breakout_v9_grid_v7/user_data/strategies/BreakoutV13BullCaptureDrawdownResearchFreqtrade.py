from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from BreakoutV12RegimeAdaptiveGridV9SelectedFreqtrade import (
    BreakoutV12RegimeAdaptiveGridV9SelectedFreqtrade,
)


class _V13BullRotationBoostMixin:
    """Increase only proven Breakout-long risk in an active bull rotation.

    A high 24-hour breadth travel means leadership is rotating through the
    Active50 instead of narrowing to one isolated pump.  Capping the BTC 4h
    trend avoids increasing risk after the benchmark itself is already
    parabolic.  Both inputs are available from completed 1h candles in LIVE.
    """

    V13_BULL_ROTATION_SCALE = 1.50
    V13_BULL_ROTATION_MIN_BREADTH_TRAVEL_24H = 2.40
    V13_BULL_ROTATION_MAX_BTC_TREND_4H = 0.65

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
        if (
            stake <= 0.0
            or self._component(entry_tag) != "breakout"
            or side != "long"
        ):
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return stake
        active_rotation = (
            float(row.get("breadth_travel_24h", -np.inf))
            >= self.V13_BULL_ROTATION_MIN_BREADTH_TRAVEL_24H
            and float(row.get("btc_trend_4h", np.inf))
            <= self.V13_BULL_ROTATION_MAX_BTC_TREND_4H
        )
        if active_rotation:
            stake *= self.V13_BULL_ROTATION_SCALE
        return min(float(max_stake), max(0.0, float(stake)))


class _V13BreakoutShortExhaustionRiskMixin:
    """De-risk late shorts after almost all of Active50 has already broken.

    Extremely low breadth is not automatically a fresh short edge.  When the
    target remains within a bounded distance of its EMA, the setup has the
    asymmetric rebound profile repeatedly observed in the 2025 drawdown.
    """

    V13_BO_SHORT_EXHAUSTION_SCALE = 0.35
    V13_BO_SHORT_EXHAUSTION_MAX_BREADTH = 0.08
    V13_BO_SHORT_EXHAUSTION_MIN_EMA55_ATR = -4.38

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
        if (
            stake <= 0.0
            or self._component(entry_tag) != "breakout"
            or side != "short"
        ):
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return stake
        exhausted = (
            float(row.get("breadth", np.inf))
            <= self.V13_BO_SHORT_EXHAUSTION_MAX_BREADTH
            and float(row.get("symbol_ema55_atr", -np.inf))
            >= self.V13_BO_SHORT_EXHAUSTION_MIN_EMA55_ATR
        )
        if exhausted:
            stake *= self.V13_BO_SHORT_EXHAUSTION_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class _V13GridRotationRiskMixin:
    """Reduce Grid shorts during broadening-but-negative market rotation."""

    V13_GRID_ROTATION_SCALE = 0.25
    V13_GRID_ROTATION_MIN_BREADTH_CHANGE_24H = 0.02
    V13_GRID_ROTATION_MAX_MEDIAN_RETURN_24H = 0.00

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
        if (
            stake <= 0.0
            or self._component(entry_tag) != "grid"
            or side != "short"
        ):
            return stake
        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None:
            return stake
        rotation_chop = (
            float(row.get("breadth_change_24h", -np.inf))
            >= self.V13_GRID_ROTATION_MIN_BREADTH_CHANGE_24H
            and float(row.get("market_median_return_24h", np.inf))
            <= self.V13_GRID_ROTATION_MAX_MEDIAN_RETURN_24H
        )
        if rotation_chop:
            stake *= self.V13_GRID_ROTATION_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class _V13BreakoutLongChopRiskMixin:
    """Reduce Breakout longs in quiet, low-efficiency breadth rotation."""

    V13_BO_LONG_CHOP_SCALE = 0.35
    V13_BO_LONG_CHOP_MAX_BREADTH_TRAVEL_24H = 2.63
    V13_BO_LONG_CHOP_MAX_MARKET_EFFICIENCY = 0.262

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
        if (
            stake <= 0.0
            or self._component(entry_tag) != "breakout"
            or side != "long"
        ):
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return stake
        low_edge = (
            float(row.get("breadth_travel_24h", np.inf))
            <= self.V13_BO_LONG_CHOP_MAX_BREADTH_TRAVEL_24H
            and float(row.get("market_efficiency", np.inf))
            <= self.V13_BO_LONG_CHOP_MAX_MARKET_EFFICIENCY
        )
        if low_edge:
            stake *= self.V13_BO_LONG_CHOP_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV13Bull125GridV9Freqtrade(
    _V13BullRotationBoostMixin,
    BreakoutV12RegimeAdaptiveGridV9SelectedFreqtrade,
):
    V13_BULL_ROTATION_SCALE = 1.25


class BreakoutV13Bull150GridV9Freqtrade(
    BreakoutV13Bull125GridV9Freqtrade,
):
    V13_BULL_ROTATION_SCALE = 1.50


class BreakoutV13DefenseGridV9Freqtrade(
    _V13BreakoutLongChopRiskMixin,
    _V13GridRotationRiskMixin,
    _V13BreakoutShortExhaustionRiskMixin,
    BreakoutV12RegimeAdaptiveGridV9SelectedFreqtrade,
):
    pass


class BreakoutV13Balanced125GridV9Freqtrade(
    _V13BullRotationBoostMixin,
    BreakoutV13DefenseGridV9Freqtrade,
):
    V13_BULL_ROTATION_SCALE = 1.25


class BreakoutV13Convex150GridV9Freqtrade(
    BreakoutV13Balanced125GridV9Freqtrade,
):
    V13_BULL_ROTATION_SCALE = 1.50


class BreakoutV13Convex175GridV9Freqtrade(
    BreakoutV13Balanced125GridV9Freqtrade,
):
    V13_BULL_ROTATION_SCALE = 1.75


class BreakoutV13Convex150MildDefenseGridV9Freqtrade(
    BreakoutV13Convex150GridV9Freqtrade,
):
    V13_BO_SHORT_EXHAUSTION_SCALE = 0.50
    V13_GRID_ROTATION_SCALE = 0.35
    V13_BO_LONG_CHOP_SCALE = 0.50

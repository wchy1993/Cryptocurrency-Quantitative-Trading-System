from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from freqtrade.strategy import stoploss_from_absolute

from BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade import (
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
)


class _V15Fixed50Score5FailureExitMixin:
    """Protect low-efficiency score-five longs without changing entry size."""

    V15_SCORE5_MAX_ENTRY_EFFICIENCY = 0.70
    V15_SCORE5_INITIAL_FLOOR_R = -0.65
    V15_SCORE5_LOCK_1_ACTIVATION_R = 0.75
    V15_SCORE5_LOCK_1_R = -0.10
    V15_SCORE5_LOCK_2_ACTIVATION_R = 1.25
    V15_SCORE5_LOCK_2_R = 0.20
    V15_SCORE5_LOCK_3_ACTIVATION_R = 2.00
    V15_SCORE5_LOCK_3_R = 0.60
    V15_SCORE5_LOCK_4_ACTIVATION_R = 3.00
    V15_SCORE5_LOCK_4_R = 1.25

    def _v15_is_low_efficiency_score5(
        self,
        component: str,
        side: str,
        row: Any,
    ) -> bool:
        return bool(
            component == "breakout"
            and side == "long"
            and int(row.get("bo_score", 0)) == 5
            and float(row.get("symbol_efficiency_12h", np.inf))
            <= self.V15_SCORE5_MAX_ENTRY_EFFICIENCY
        )

    def _v15_score5_locked_r(self, maximum_r: float) -> float:
        locked_r = float(self.V15_SCORE5_INITIAL_FLOOR_R)
        for activation, floor in (
            (
                self.V15_SCORE5_LOCK_1_ACTIVATION_R,
                self.V15_SCORE5_LOCK_1_R,
            ),
            (
                self.V15_SCORE5_LOCK_2_ACTIVATION_R,
                self.V15_SCORE5_LOCK_2_R,
            ),
            (
                self.V15_SCORE5_LOCK_3_ACTIVATION_R,
                self.V15_SCORE5_LOCK_3_R,
            ),
            (
                self.V15_SCORE5_LOCK_4_ACTIVATION_R,
                self.V15_SCORE5_LOCK_4_R,
            ),
        ):
            if maximum_r >= activation:
                locked_r = max(locked_r, float(floor))
        if maximum_r >= 8.0:
            locked_r = max(locked_r, maximum_r - 6.0)
        return locked_r

    def order_filled(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        super().order_filled(
            pair,
            trade,
            order,
            current_time,
            **kwargs,
        )
        if (
            order.ft_order_side != trade.entry_side
            or trade.nr_of_successful_entries != 1
            or self._component(trade.enter_tag) != "breakout"
        ):
            return
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return
        trade.set_custom_data(
            "v15_low_efficiency_score5",
            self._v15_is_low_efficiency_score5(
                "breakout",
                "short" if trade.is_short else "long",
                row,
            ),
        )

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        if not bool(
            trade.get_custom_data("v15_low_efficiency_score5", False)
        ):
            return super().custom_stoploss(
                pair,
                trade,
                current_time,
                current_rate,
                current_profit,
                after_fill,
                **kwargs,
            )

        favorable_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate if favorable_value is None else favorable_value
        )
        maximum_r = self._trade_r(trade, favorable_rate)
        locked_r = self._v15_score5_locked_r(maximum_r)
        side = -1.0 if trade.is_short else 1.0
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


class BreakoutV15Fixed50Score5FailureExitResearchFreqtrade(
    _V15Fixed50Score5FailureExitMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    STRATEGY_VERSION = "breakout_v15_fixed50_score5_failure_exit_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_score5_failure_exit"


class BreakoutV15Fixed50Score5FailureExitMildResearchFreqtrade(
    BreakoutV15Fixed50Score5FailureExitResearchFreqtrade,
):
    V15_SCORE5_INITIAL_FLOOR_R = -0.80
    STRATEGY_VERSION = "breakout_v15_fixed50_score5_failure_exit_mild_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_score5_failure_exit_mild"


class BreakoutV15Fixed50Score5FailureExitTightResearchFreqtrade(
    BreakoutV15Fixed50Score5FailureExitResearchFreqtrade,
):
    V15_SCORE5_INITIAL_FLOOR_R = -0.50
    STRATEGY_VERSION = "breakout_v15_fixed50_score5_failure_exit_tight_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_score5_failure_exit_tight"

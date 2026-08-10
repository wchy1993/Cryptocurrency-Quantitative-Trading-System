from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from BreakoutV15Fixed50CoreTransitionResearchFreqtrade import (
    BreakoutV15Fixed50CoreTransition20ResearchFreqtrade,
)


class _V15BreakoutShortImpulseMinimumOrderRiskMixin:
    """De-risk weak short impulses without dropping a valid signal.

    The exchange minimum order is used as a floor after risk scaling.  This
    keeps Max2 occupancy stable across nearby sizing parameters and removes
    the discontinuity where a tiny sizing change would otherwise skip the
    trade and alter every later compounding decision.
    """

    V15_BO_SHORT_LOCAL_RETURN_FULL = -0.0045
    V15_BO_SHORT_LOCAL_RETURN_WEAK = -0.0025
    V15_BO_SHORT_RANGE_ACTIVATION_FULL = 5.75
    V15_BO_SHORT_RANGE_ACTIVATION_OFF = 6.25
    V15_BO_SHORT_WEAK_IMPULSE_MIN_SCALE = 0.20

    @staticmethod
    def _v15_smooth_axis(value: float) -> float:
        bounded = float(np.clip(value, 0.0, 1.0))
        return bounded * bounded * (3.0 - 2.0 * bounded)

    def _v15_breakout_short_impulse_scale(self, row: Any) -> float:
        local_return = float(row.get("symbol_return_4h", np.nan))
        breakout_range = float(row.get("bo_range_atr", np.nan))
        if not np.isfinite(local_return) or not np.isfinite(breakout_range):
            return 1.0

        weak_local_impulse = self._v15_smooth_axis(
            (local_return - self.V15_BO_SHORT_LOCAL_RETURN_FULL)
            / max(
                self.V15_BO_SHORT_LOCAL_RETURN_WEAK
                - self.V15_BO_SHORT_LOCAL_RETURN_FULL,
                1e-12,
            )
        )
        nonexceptional_range = 1.0 - self._v15_smooth_axis(
            (
                breakout_range
                - self.V15_BO_SHORT_RANGE_ACTIVATION_FULL
            )
            / max(
                self.V15_BO_SHORT_RANGE_ACTIVATION_OFF
                - self.V15_BO_SHORT_RANGE_ACTIVATION_FULL,
                1e-12,
            )
        )
        activation = weak_local_impulse * nonexceptional_range
        minimum = float(self.V15_BO_SHORT_WEAK_IMPULSE_MIN_SCALE)
        return float(np.clip(1.0 - activation * (1.0 - minimum), 0.0, 1.0))

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

        scaled = min(
            float(max_stake),
            max(0.0, float(stake))
            * self._v15_breakout_short_impulse_scale(row),
        )
        if min_stake is not None and scaled < float(min_stake):
            if float(min_stake) > float(max_stake):
                return 0.0
            return float(min_stake)
        return scaled


class BreakoutV15Fixed50CoreShortImpulseFloor20ResearchFreqtrade(
    _V15BreakoutShortImpulseMinimumOrderRiskMixin,
    BreakoutV15Fixed50CoreTransition20ResearchFreqtrade,
):
    STRATEGY_VERSION = "breakout_v15_fixed50_core_short_impulse_floor20_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_core_short_impulse_floor20"


class BreakoutV15Fixed50CoreShortImpulseFloor35ResearchFreqtrade(
    BreakoutV15Fixed50CoreShortImpulseFloor20ResearchFreqtrade,
):
    V15_BO_SHORT_WEAK_IMPULSE_MIN_SCALE = 0.35
    STRATEGY_VERSION = "breakout_v15_fixed50_core_short_impulse_floor35_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_core_short_impulse_floor35"

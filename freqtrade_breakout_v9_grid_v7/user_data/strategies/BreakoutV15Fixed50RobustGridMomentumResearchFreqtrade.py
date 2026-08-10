from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from BreakoutV15Fixed50RobustQualityResearchFreqtrade import (
    BreakoutV15Fixed50RobustQualityL20S50ResearchFreqtrade,
)


class _V15RobustGridShortPositiveMomentumRiskMixin:
    """De-risk relative-strength Grid shorts outside a strong bear regime."""

    V15_GRID_SHORT_MIN_POSITIVE_RETURN_4H = 0.01
    V15_GRID_SHORT_MIN_MARKET_REGIME = -0.35
    V15_GRID_SHORT_POSITIVE_MOMENTUM_SCALE = 0.50

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
        conflict = (
            float(row.get("symbol_return_4h", -np.inf))
            >= self.V15_GRID_SHORT_MIN_POSITIVE_RETURN_4H
            and float(row.get("market_regime_score", -np.inf))
            > self.V15_GRID_SHORT_MIN_MARKET_REGIME
        )
        if conflict:
            stake *= self.V15_GRID_SHORT_POSITIVE_MOMENTUM_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV15Fixed50RobustQualityL20S20ResearchFreqtrade(
    BreakoutV15Fixed50RobustQualityL20S50ResearchFreqtrade,
):
    V15_BO_SHORT_COUNTERMOVE_SCALE = 0.20
    STRATEGY_VERSION = "breakout_v15_fixed50_robust_quality_l20_s20_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_robust_quality_l20_s20"


class BreakoutV15Fixed50RobustQualityL20S20Eff82ResearchFreqtrade(
    BreakoutV15Fixed50RobustQualityL20S20ResearchFreqtrade,
):
    V15_BO_LONG_MAX_HEALTHY_EFFICIENCY_12H = 0.82
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_robust_quality_l20_s20_eff82_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_robust_quality_l20_s20_eff82"
    )


class BreakoutV15Fixed50RobustL20S20GridMomentum35ResearchFreqtrade(
    _V15RobustGridShortPositiveMomentumRiskMixin,
    BreakoutV15Fixed50RobustQualityL20S20ResearchFreqtrade,
):
    V15_GRID_SHORT_POSITIVE_MOMENTUM_SCALE = 0.35
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_robust_l20_s20_grid_momentum35_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_robust_l20_s20_grid_momentum35"
    )


class BreakoutV15Fixed50RobustL20S20GridMomentum50ResearchFreqtrade(
    BreakoutV15Fixed50RobustL20S20GridMomentum35ResearchFreqtrade,
):
    V15_GRID_SHORT_POSITIVE_MOMENTUM_SCALE = 0.50
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_robust_l20_s20_grid_momentum50_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_robust_l20_s20_grid_momentum50"
    )


class BreakoutV15Fixed50RobustL20S20GridMomentum65ResearchFreqtrade(
    BreakoutV15Fixed50RobustL20S20GridMomentum35ResearchFreqtrade,
):
    V15_GRID_SHORT_POSITIVE_MOMENTUM_SCALE = 0.65
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_robust_l20_s20_grid_momentum65_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_robust_l20_s20_grid_momentum65"
    )


class BreakoutV15Fixed50RobustL20S50GridMomentum35ResearchFreqtrade(
    _V15RobustGridShortPositiveMomentumRiskMixin,
    BreakoutV15Fixed50RobustQualityL20S50ResearchFreqtrade,
):
    V15_GRID_SHORT_POSITIVE_MOMENTUM_SCALE = 0.35
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_robust_l20_s50_grid_momentum35_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_robust_l20_s50_grid_momentum35"
    )


class BreakoutV15Fixed50RobustL20S50GridMomentum50ResearchFreqtrade(
    BreakoutV15Fixed50RobustL20S50GridMomentum35ResearchFreqtrade,
):
    V15_GRID_SHORT_POSITIVE_MOMENTUM_SCALE = 0.50
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_robust_l20_s50_grid_momentum50_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_robust_l20_s50_grid_momentum50"
    )


class BreakoutV15Fixed50RobustL20S50GridMomentum65ResearchFreqtrade(
    BreakoutV15Fixed50RobustL20S50GridMomentum35ResearchFreqtrade,
):
    V15_GRID_SHORT_POSITIVE_MOMENTUM_SCALE = 0.65
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_robust_l20_s50_grid_momentum65_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_robust_l20_s50_grid_momentum65"
    )

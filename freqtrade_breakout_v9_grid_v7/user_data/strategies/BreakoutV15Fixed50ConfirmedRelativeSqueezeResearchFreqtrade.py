from __future__ import annotations

from datetime import datetime
from typing import Any

from BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade import (
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
)
from BreakoutV14AdaptiveAllocationRangeResearchFreqtrade import (
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
)
from BreakoutV15Fixed50WeakImpulseGridLongDefenseResearchFreqtrade import (
    BreakoutV15Fixed50WeakImpulseGridLong50ResearchFreqtrade,
)
from BreakoutV15Fixed50WeakImpulseScore5FailureExitResearchFreqtrade import (
    BreakoutV15Fixed50WeakImpulseScore5Failure335ResearchFreqtrade,
    BreakoutV15Fixed50WeakImpulseScore5Failure340ResearchFreqtrade,
    BreakoutV15Fixed50WeakImpulseScore5Failure350ResearchFreqtrade,
    BreakoutV15Fixed50WeakImpulseScore5Failure360ResearchFreqtrade,
    BreakoutV15Fixed50WeakImpulseScore5Failure365ResearchFreqtrade,
    BreakoutV15Fixed50WeakImpulseScore5Failure425ResearchFreqtrade,
)


class _V15ConfirmedRelativeSqueezeFromHighWaterMixin:
    """Preempt only a high-volume Grid-short squeeze near high water.

    The selected V13 relative-squeeze defense remains unchanged once realized
    equity drawdown reaches 3%.  Before that point, the same structural setup
    is compressed only when completed-candle volume confirms participation.
    This avoids changing low-volume Max2 paths while protecting the first
    crowded squeeze of a new drawdown cycle.
    """

    V15_EARLY_SQUEEZE_MIN_VOLUME_RATIO = 1.0

    def _v15_relative_squeeze_conflict(self, row: Any) -> bool:
        base_conflict = (
            float(row.get("symbol_return_4h", float("-inf")))
            >= self.V13_GRID_SQUEEZE_MIN_SYMBOL_RETURN_4H
            and float(row.get("market_regime_score", float("inf")))
            <= self.V13_GRID_SQUEEZE_MAX_MARKET_REGIME
            and float(row.get("breadth", float("-inf")))
            >= self.V13_GRID_SQUEEZE_MIN_BREADTH
        )
        broad_contraction = (
            float(row.get("breadth_change_24h", float("inf")))
            <= self.V13_GRID_SQUEEZE_MAX_BREADTH_CHANGE_24H
        )
        crowded_rebound = (
            float(row.get("grid_volume_ratio", float("-inf")))
            >= self.V13_GRID_SQUEEZE_MIN_VOLUME_RATIO
        )
        low_score_conflict = (
            float(row.get("grid_score", float("inf")))
            <= self.V13_GRID_SQUEEZE_MAX_SCORE
            and (broad_contraction or crowded_rebound)
        )
        weak_structure_conflict = (
            self.V13_GRID_SQUEEZE_ENABLE_WEAK_STRUCTURE
            and float(row.get("grid_alignment", float("inf")))
            <= self.V13_GRID_SQUEEZE_MAX_ALIGNMENT
            and float(row.get("grid_extension", float("inf")))
            <= self.V13_GRID_SQUEEZE_MAX_EXTENSION
        )
        return bool(
            base_conflict
            and (low_score_conflict or weak_structure_conflict)
        )

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

        equity = float(self.wallets.get_total_stake_amount())
        peak = max(float(getattr(self, "_peak_equity", 0.0)), equity)
        drawdown = 0.0 if peak <= 0.0 else max(0.0, 1.0 - equity / peak)
        if drawdown >= self.V13_GRID_SQUEEZE_MIN_DRAWDOWN:
            return stake

        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None:
            return stake
        if (
            float(row.get("grid_volume_ratio", float("-inf")))
            < self.V15_EARLY_SQUEEZE_MIN_VOLUME_RATIO
            or not self._v15_relative_squeeze_conflict(row)
        ):
            return stake

        scaled = min(
            float(max_stake),
            max(0.0, float(stake) * self.V13_GRID_SQUEEZE_SCALE),
        )
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV15Fixed50ConfirmedSqueezeV13ResearchFreqtrade(
    _V15ConfirmedRelativeSqueezeFromHighWaterMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    STRATEGY_VERSION = "breakout_v15_fixed50_confirmed_squeeze_v13_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_confirmed_squeeze_v13"


class BreakoutV15Fixed50ConfirmedSqueezeWeakImpulseResearchFreqtrade(
    _V15ConfirmedRelativeSqueezeFromHighWaterMixin,
    BreakoutV15Fixed50WeakImpulseScore5Failure425ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_squeeze_weak_impulse_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_squeeze_weak_impulse"
    )


class BreakoutV15Fixed50ConfirmedSqueezeGridLong50ResearchFreqtrade(
    _V15ConfirmedRelativeSqueezeFromHighWaterMixin,
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_squeeze_grid_long50_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_squeeze_grid_long50"
    )


class BreakoutV15Fixed50ConfirmedSqueezeCombinedResearchFreqtrade(
    _V15ConfirmedRelativeSqueezeFromHighWaterMixin,
    BreakoutV15Fixed50WeakImpulseGridLong50ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_squeeze_combined_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_squeeze_combined"
    )


class BreakoutV15Fixed50ConfirmedSqueezeWeak335ResearchFreqtrade(
    _V15ConfirmedRelativeSqueezeFromHighWaterMixin,
    BreakoutV15Fixed50WeakImpulseScore5Failure335ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_squeeze_weak335_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_squeeze_weak335"
    )


class BreakoutV15Fixed50ConfirmedSqueezeWeak340ResearchFreqtrade(
    _V15ConfirmedRelativeSqueezeFromHighWaterMixin,
    BreakoutV15Fixed50WeakImpulseScore5Failure340ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_squeeze_weak340_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_squeeze_weak340"
    )


class BreakoutV15Fixed50ConfirmedSqueezeWeak350ResearchFreqtrade(
    _V15ConfirmedRelativeSqueezeFromHighWaterMixin,
    BreakoutV15Fixed50WeakImpulseScore5Failure350ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_squeeze_weak350_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_squeeze_weak350"
    )


class BreakoutV15Fixed50ConfirmedSqueezeWeak360ResearchFreqtrade(
    _V15ConfirmedRelativeSqueezeFromHighWaterMixin,
    BreakoutV15Fixed50WeakImpulseScore5Failure360ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_squeeze_weak360_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_squeeze_weak360"
    )


class BreakoutV15Fixed50ConfirmedSqueezeWeak365ResearchFreqtrade(
    _V15ConfirmedRelativeSqueezeFromHighWaterMixin,
    BreakoutV15Fixed50WeakImpulseScore5Failure365ResearchFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_squeeze_weak365_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_squeeze_weak365"
    )

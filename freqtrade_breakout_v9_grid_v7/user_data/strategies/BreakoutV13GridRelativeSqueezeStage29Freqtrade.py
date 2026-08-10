from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from BreakoutV13BullCaptureDrawdownResearchFreqtrade import (
    BreakoutV13Convex150GridV9Freqtrade,
)


class _V13GridRelativeSqueezeRiskMixin:
    """De-risk low-score Grid shorts whose target resists a bearish tape."""

    V13_GRID_SQUEEZE_MIN_DRAWDOWN = 0.03
    V13_GRID_SQUEEZE_SCALE = 0.25
    V13_GRID_SQUEEZE_MIN_SYMBOL_RETURN_4H = 0.0075
    V13_GRID_SQUEEZE_MAX_MARKET_REGIME = 0.00
    V13_GRID_SQUEEZE_MAX_SCORE = 5.0
    V13_GRID_SQUEEZE_MAX_BREADTH_CHANGE_24H = -0.30
    V13_GRID_SQUEEZE_MIN_VOLUME_RATIO = 1.50
    V13_GRID_SQUEEZE_MIN_BREADTH = 0.00
    V13_GRID_SQUEEZE_ENABLE_WEAK_STRUCTURE = False
    V13_GRID_SQUEEZE_MAX_ALIGNMENT = 0.70
    V13_GRID_SQUEEZE_MAX_EXTENSION = 0.25

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
        if drawdown < self.V13_GRID_SQUEEZE_MIN_DRAWDOWN:
            return stake

        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None:
            return stake
        base_conflict = (
            float(row.get("symbol_return_4h", -np.inf))
            >= self.V13_GRID_SQUEEZE_MIN_SYMBOL_RETURN_4H
            and float(row.get("market_regime_score", np.inf))
            <= self.V13_GRID_SQUEEZE_MAX_MARKET_REGIME
            and float(row.get("breadth", -np.inf))
            >= self.V13_GRID_SQUEEZE_MIN_BREADTH
        )
        broad_contraction = (
            float(row.get("breadth_change_24h", np.inf))
            <= self.V13_GRID_SQUEEZE_MAX_BREADTH_CHANGE_24H
        )
        crowded_rebound = (
            float(row.get("grid_volume_ratio", -np.inf))
            >= self.V13_GRID_SQUEEZE_MIN_VOLUME_RATIO
        )
        low_score_conflict = (
            float(row.get("grid_score", np.inf))
            <= self.V13_GRID_SQUEEZE_MAX_SCORE
            and (broad_contraction or crowded_rebound)
        )
        weak_structure_conflict = (
            self.V13_GRID_SQUEEZE_ENABLE_WEAK_STRUCTURE
            and float(row.get("grid_alignment", np.inf))
            <= self.V13_GRID_SQUEEZE_MAX_ALIGNMENT
            and float(row.get("grid_extension", np.inf))
            <= self.V13_GRID_SQUEEZE_MAX_EXTENSION
        )
        if not (
            base_conflict
            and (low_score_conflict or weak_structure_conflict)
        ):
            return stake

        scaled = min(
            float(max_stake),
            max(0.0, float(stake) * self.V13_GRID_SQUEEZE_SCALE),
        )
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV13GridRelativeSqueeze25GridV9Freqtrade(
    _V13GridRelativeSqueezeRiskMixin,
    BreakoutV13Convex150GridV9Freqtrade,
):
    pass


class BreakoutV13GridRelativeSqueeze15GridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25GridV9Freqtrade,
):
    V13_GRID_SQUEEZE_SCALE = 0.15


class BreakoutV13GridRelativeSqueeze35GridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25GridV9Freqtrade,
):
    V13_GRID_SQUEEZE_SCALE = 0.35


class BreakoutV13GridRelativeSqueeze25Breadth025GridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25GridV9Freqtrade,
):
    V13_GRID_SQUEEZE_MAX_BREADTH_CHANGE_24H = -0.25


class BreakoutV13GridRelativeSqueeze25WeakStructureGridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25GridV9Freqtrade,
):
    V13_GRID_SQUEEZE_ENABLE_WEAK_STRUCTURE = True


class BreakoutV13GridRelativeSqueeze25WeakStructureBreadth45GridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25WeakStructureGridV9Freqtrade,
):
    V13_GRID_SQUEEZE_MIN_BREADTH = 0.45


class BreakoutV13GridRelativeSqueeze15WeakStructureBreadth45GridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25WeakStructureBreadth45GridV9Freqtrade,
):
    V13_GRID_SQUEEZE_SCALE = 0.15


class BreakoutV13GridRelativeSqueeze20WeakStructureBreadth45GridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25WeakStructureBreadth45GridV9Freqtrade,
):
    V13_GRID_SQUEEZE_SCALE = 0.20


class BreakoutV13GridRelativeSqueeze30WeakStructureBreadth45GridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25WeakStructureBreadth45GridV9Freqtrade,
):
    V13_GRID_SQUEEZE_SCALE = 0.30


class BreakoutV13GridRelativeSqueeze25WeakStructureBreadth44GridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25WeakStructureBreadth45GridV9Freqtrade,
):
    V13_GRID_SQUEEZE_MIN_BREADTH = 0.44


class BreakoutV13GridRelativeSqueeze25WeakStructureBreadth46GridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25WeakStructureBreadth45GridV9Freqtrade,
):
    V13_GRID_SQUEEZE_MIN_BREADTH = 0.46


class BreakoutV13GridRelativeSqueeze25WeakStructureBreadth48GridV9Freqtrade(
    BreakoutV13GridRelativeSqueeze25WeakStructureBreadth45GridV9Freqtrade,
):
    V13_GRID_SQUEEZE_MIN_BREADTH = 0.48

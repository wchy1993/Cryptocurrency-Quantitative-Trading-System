from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from GridV15Fixed50OnlyResearchFreqtrade import (
    GridV15Fixed50OnlyResearchFreqtrade,
)


class _GridV15QualityProtectedRiskMixin:
    """Allocate Grid risk by completed-candle setup quality.

    The base Grid signal, cross-pair rank, leverage, DCA ladder and exits stay
    frozen.  This overlay only changes the initial campaign size.  It keeps
    full risk for the empirically stable score-six and specialist score-three
    setups, while reducing middling score-four/five campaigns.  Two smooth
    causal axes add protection when a short is already far below its fast EMA
    or its entry candle has unusually weak participation.
    """

    GRID_V15_SCORE3_SCALE = 1.00
    GRID_V15_SCORE4_SCALE = 0.80
    GRID_V15_SCORE5_SCALE = 0.40
    GRID_V15_SCORE6_SCALE = 1.00
    GRID_V15_LONG_SCALE = 0.50
    GRID_V15_MAX_SCALE = 1.00

    GRID_V15_EXTENSION_FULL = 0.42
    GRID_V15_EXTENSION_DEFENSIVE = 0.52
    GRID_V15_EXTENSION_MIN_SCALE = 0.40

    GRID_V15_VOLUME_DEFENSIVE = 0.74
    GRID_V15_VOLUME_FULL = 0.82
    GRID_V15_VOLUME_MIN_SCALE = 0.50

    @staticmethod
    def _grid_v15_smoothstep(value: float) -> float:
        bounded = float(np.clip(value, 0.0, 1.0))
        return bounded * bounded * (3.0 - 2.0 * bounded)

    def _grid_v15_quality_scale(self, row: Any, side: str) -> float:
        if side != "short":
            return float(self.GRID_V15_LONG_SCALE)

        score = int(row.get("grid_score", 0) or 0)
        score_scale = {
            3: self.GRID_V15_SCORE3_SCALE,
            4: self.GRID_V15_SCORE4_SCALE,
            5: self.GRID_V15_SCORE5_SCALE,
            6: self.GRID_V15_SCORE6_SCALE,
        }.get(score, self.GRID_V15_SCORE5_SCALE)

        extension = float(row.get("grid_extension", np.nan))
        extension_scale = 1.0
        if np.isfinite(extension):
            progress = self._grid_v15_smoothstep(
                (extension - self.GRID_V15_EXTENSION_FULL)
                / max(
                    self.GRID_V15_EXTENSION_DEFENSIVE
                    - self.GRID_V15_EXTENSION_FULL,
                    1e-12,
                )
            )
            extension_scale = 1.0 - progress * (
                1.0 - self.GRID_V15_EXTENSION_MIN_SCALE
            )

        volume = float(row.get("grid_volume_ratio", np.nan))
        volume_scale = 1.0
        if np.isfinite(volume):
            participation = self._grid_v15_smoothstep(
                (volume - self.GRID_V15_VOLUME_DEFENSIVE)
                / max(
                    self.GRID_V15_VOLUME_FULL
                    - self.GRID_V15_VOLUME_DEFENSIVE,
                    1e-12,
                )
            )
            volume_scale = self.GRID_V15_VOLUME_MIN_SCALE + participation * (
                1.0 - self.GRID_V15_VOLUME_MIN_SCALE
            )

        return float(
            np.clip(
                score_scale * extension_scale * volume_scale,
                0.0,
                self.GRID_V15_MAX_SCALE,
            )
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
        if stake <= 0.0 or self._component(entry_tag) != "grid":
            return stake
        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None:
            return stake
        scaled = min(
            float(max_stake),
            max(0.0, float(stake))
            * self._grid_v15_quality_scale(row, side),
        )
        if min_stake is not None and scaled < float(min_stake):
            if float(min_stake) > float(max_stake):
                return 0.0
            return float(min_stake)
        return scaled


class GridV15Fixed50QualityBalancedResearchFreqtrade(
    _GridV15QualityProtectedRiskMixin,
    GridV15Fixed50OnlyResearchFreqtrade,
):
    """Central quality-risk candidate."""

    STRATEGY_VERSION = "grid_v15_fixed50_quality_balanced_20260811"
    ADAPTIVE_STATE_BASENAME = "grid_v15_fixed50_quality_balanced"


class GridV15Fixed50QualityPfResearchFreqtrade(
    GridV15Fixed50QualityBalancedResearchFreqtrade,
):
    """More defensive score-four/five neighborhood."""

    GRID_V15_SCORE4_SCALE = 0.60
    GRID_V15_SCORE5_SCALE = 0.25
    GRID_V15_LONG_SCALE = 0.35
    STRATEGY_VERSION = "grid_v15_fixed50_quality_pf_20260811"
    ADAPTIVE_STATE_BASENAME = "grid_v15_fixed50_quality_pf"


class GridV15Fixed50QualityProfitResearchFreqtrade(
    GridV15Fixed50QualityBalancedResearchFreqtrade,
):
    """Retain score-four convexity while strongly defending score five."""

    GRID_V15_SCORE4_SCALE = 1.00
    GRID_V15_SCORE5_SCALE = 0.25
    GRID_V15_LONG_SCALE = 0.35
    STRATEGY_VERSION = "grid_v15_fixed50_quality_profit_20260811"
    ADAPTIVE_STATE_BASENAME = "grid_v15_fixed50_quality_profit"


class GridV15Fixed50QualityProfitLooseResearchFreqtrade(
    GridV15Fixed50QualityProfitResearchFreqtrade,
):
    """One-step looser smooth-axis neighbor for stability screening."""

    GRID_V15_SCORE5_SCALE = 0.35
    GRID_V15_LONG_SCALE = 0.45
    GRID_V15_EXTENSION_MIN_SCALE = 0.50
    GRID_V15_VOLUME_MIN_SCALE = 0.60
    STRATEGY_VERSION = "grid_v15_fixed50_quality_profit_loose_20260811"
    ADAPTIVE_STATE_BASENAME = "grid_v15_fixed50_quality_profit_loose"


class GridV15Fixed50QualityConviction115ResearchFreqtrade(
    GridV15Fixed50QualityBalancedResearchFreqtrade,
):
    """Reallocate a small part of the saved budget to full-quality shorts."""

    GRID_V15_SCORE6_SCALE = 1.15
    GRID_V15_MAX_SCALE = 1.15
    STRATEGY_VERSION = "grid_v15_fixed50_quality_conviction115_20260811"
    ADAPTIVE_STATE_BASENAME = "grid_v15_fixed50_quality_conviction115"


class GridV15Fixed50QualityConviction125ResearchFreqtrade(
    GridV15Fixed50QualityConviction115ResearchFreqtrade,
):
    """Upper robustness neighbor for the score-six reallocation."""

    GRID_V15_SCORE6_SCALE = 1.25
    GRID_V15_MAX_SCALE = 1.25
    STRATEGY_VERSION = "grid_v15_fixed50_quality_conviction125_20260811"
    ADAPTIVE_STATE_BASENAME = "grid_v15_fixed50_quality_conviction125"


class GridV15Fixed50QualityPfConviction105ResearchFreqtrade(
    GridV15Fixed50QualityPfResearchFreqtrade,
):
    """Narrow score-six reallocation around the exact PF winner."""

    GRID_V15_SCORE6_SCALE = 1.05
    GRID_V15_MAX_SCALE = 1.05
    STRATEGY_VERSION = "grid_v15_fixed50_quality_pf_conviction105_20260811"
    ADAPTIVE_STATE_BASENAME = "grid_v15_fixed50_quality_pf_conviction105"


class GridV15Fixed50QualityPfConviction110ResearchFreqtrade(
    GridV15Fixed50QualityPfConviction105ResearchFreqtrade,
):
    """Upper neighbor for the PF winner's score-six reallocation."""

    GRID_V15_SCORE6_SCALE = 1.10
    GRID_V15_MAX_SCALE = 1.10
    STRATEGY_VERSION = "grid_v15_fixed50_quality_pf_conviction110_20260811"
    ADAPTIVE_STATE_BASENAME = "grid_v15_fixed50_quality_pf_conviction110"


class _GridV15ShortFocusMixin:
    """Disable the immature Grid-long sleeve and its rank candidates."""

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        tags = dataframe["enter_tag"].fillna("").astype(str)
        rejected_long = tags.str.startswith("grid_v8_long_")
        dataframe.loc[rejected_long, "enter_long"] = 0
        dataframe.loc[rejected_long, "enter_short"] = 0
        dataframe.loc[rejected_long, "enter_tag"] = None

        pair = str(metadata.get("pair") or "")
        ranks = getattr(self, "_candidate_ranks", {}).get(pair)
        if pair and ranks is not None and rejected_long.any():
            available_times = (
                pd.to_datetime(dataframe["date"], utc=True)
                + pd.Timedelta(self.timeframe)
            )
            rejected_timestamps = {
                int(available_times.at[index].timestamp())
                for index in dataframe.index[rejected_long]
            }
            self._candidate_ranks[pair] = {
                key: value
                for key, value in ranks.items()
                if not (key[0] == "grid" and key[1] in rejected_timestamps)
            }
        return dataframe


class GridV15Fixed50QualityPfShortFocusResearchFreqtrade(
    _GridV15ShortFocusMixin,
    GridV15Fixed50QualityPfResearchFreqtrade,
):
    """PF candidate with the low-evidence Grid-long sleeve disabled."""

    STRATEGY_VERSION = "grid_v15_fixed50_quality_pf_short_focus_20260811"
    ADAPTIVE_STATE_BASENAME = "grid_v15_fixed50_quality_pf_short_focus"

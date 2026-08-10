from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pandas import DataFrame
import numpy as np
import pandas as pd

from freqtrade.strategy import Trade, stoploss_from_absolute

from BreakoutV10FGridV8DualSideFreqtrade import LIVE_CONTEXT_COLUMNS
from BreakoutV11AdaptiveGridV8DualSideFreqtrade import (
    BreakoutV11AdaptiveGridV8DualSideFreqtrade,
)
from BreakoutV12MultiRegimeGridV9Freqtrade import (
    V12_CONTEXT_COLUMNS,
    build_v12_market_context,
)
from BreakoutV9GridV7Freqtrade import _causal_signal_cap


V12_EXTRA_CONTEXT_COLUMNS = tuple(
    column
    for column in V12_CONTEXT_COLUMNS
    if column not in LIVE_CONTEXT_COLUMNS
)


class BreakoutV12ExactParitySidecarFreqtrade(
    BreakoutV11AdaptiveGridV8DualSideFreqtrade
):
    """Frozen v11/Grid-v8 behavior plus read-only regime columns.

    The five context fields consumed by the frozen strategy are created by
    its original implementation.  New fields are merged only afterward, so
    the Max2 candidate ranking and every frozen decision remain unchanged.
    """

    def _build_market_context(self) -> DataFrame:
        frozen_context = super()._build_market_context()
        pairs = tuple(self.dp.current_whitelist())
        snapshots: dict[str, DataFrame] = {}
        for pair in pairs:
            raw = self.dp.get_pair_dataframe(
                pair=pair,
                timeframe=self.timeframe,
            )
            if raw is not None and not raw.empty:
                snapshots[pair] = raw[["date", "close"]].copy()
        expanded, _target, _unavailable = build_v12_market_context(
            snapshots,
            pairs,
            require_aligned_latest=False,
        )
        if frozen_context.empty or expanded.empty:
            output = frozen_context.copy()
            for column in V12_EXTRA_CONTEXT_COLUMNS:
                output[column] = np.nan
            return output
        return frozen_context.merge(
            expanded[["date", *V12_EXTRA_CONTEXT_COLUMNS]],
            on="date",
            how="left",
            validate="1:1",
        )


class BreakoutV12GridV9PersistentOneTpFloorFreqtrade(
    BreakoutV12ExactParitySidecarFreqtrade
):
    """Freeze entries and protect one completed Grid cycle in compression."""

    GRID_V9_LOW_OPPORTUNITY_PERSISTENCE = 0.90
    GRID_V9_ONE_TP_MIN_REGIME = -1.00
    GRID_V9_ONE_TP_ACTIVATION_R = 0.10
    GRID_V9_ONE_TP_FLOOR_R = 0.02

    def order_filled(
        self,
        pair: str,
        trade: Trade,
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
            self._component(getattr(trade, "enter_tag", None)) != "grid"
            or not bool(getattr(trade, "is_short", False))
            or getattr(order, "ft_order_side", None)
            != getattr(trade, "entry_side", None)
            or int(getattr(trade, "nr_of_successful_entries", 0)) != 1
        ):
            return
        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None:
            return
        enabled = (
            float(row.get("low_opportunity_fraction_72h", -1.0))
            >= self.GRID_V9_LOW_OPPORTUNITY_PERSISTENCE
            and float(row.get("market_regime_score", -2.0))
            >= self.GRID_V9_ONE_TP_MIN_REGIME
        )
        trade.set_custom_data("grid_v9_one_tp_floor", enabled)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        reason = super().custom_exit(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
            **kwargs,
        )
        if reason or not bool(
            trade.get_custom_data("grid_v9_one_tp_floor", False)
        ):
            return reason
        if int(trade.get_custom_data("grid_tp_count", 0)) < 1:
            return reason

        risk_budget = max(
            float(trade.get_custom_data("grid_risk_budget", 0.0)),
            1e-12,
        )
        favorable_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate if favorable_value is None else favorable_value
        )
        best_profit = float(
            trade.calculate_profit(favorable_rate).total_profit
        )
        current_value = float(
            trade.calculate_profit(current_rate).total_profit
        )
        if (
            best_profit >= risk_budget * self.GRID_V9_ONE_TP_ACTIVATION_R
            and current_value <= risk_budget * self.GRID_V9_ONE_TP_FLOOR_R
        ):
            return "grid_v9_persistent_one_tp_floor"
        return reason


class BreakoutV12GridV9PersistentOneTpFloor000Freqtrade(
    BreakoutV12GridV9PersistentOneTpFloorFreqtrade
):
    GRID_V9_ONE_TP_FLOOR_R = 0.00


class BreakoutV12GridV9PersistentOneTpFloor004Freqtrade(
    BreakoutV12GridV9PersistentOneTpFloorFreqtrade
):
    GRID_V9_ONE_TP_FLOOR_R = 0.04


class BreakoutV12GridV9AdaptiveOneTpFloorM015Freqtrade(
    BreakoutV12GridV9PersistentOneTpFloorFreqtrade
):
    """Protect Grid-short profit only after bearish support has faded."""

    GRID_V9_ONE_TP_MIN_REGIME = -0.15


class BreakoutV12GridV9AdaptiveOneTpFloorM010Freqtrade(
    BreakoutV12GridV9PersistentOneTpFloorFreqtrade
):
    """Balanced neutral-regime Grid-short profit protection."""

    GRID_V9_ONE_TP_MIN_REGIME = -0.10


class BreakoutV12GridV9AdaptiveOneTpFloorM005Freqtrade(
    BreakoutV12GridV9PersistentOneTpFloorFreqtrade
):
    """Conservative neutral-regime neighbor."""

    GRID_V9_ONE_TP_MIN_REGIME = -0.05


class _GridV9NeutralShortRiskMixin:
    """Reduce only persistent Grid shorts with no directional regime edge."""

    GRID_V9_NEUTRAL_SHORT_SCALE = 0.65
    GRID_V9_NEUTRAL_SHORT_MIN_PERSISTENCE = 0.90
    GRID_V9_NEUTRAL_SHORT_MIN_REGIME = -0.15
    GRID_V9_NEUTRAL_SHORT_MAX_REGIME = 0.25

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
            return 0.0
        persistence = float(row.get("low_opportunity_fraction_72h", -1.0))
        regime = float(row.get("market_regime_score", -2.0))
        if (
            persistence >= self.GRID_V9_NEUTRAL_SHORT_MIN_PERSISTENCE
            and self.GRID_V9_NEUTRAL_SHORT_MIN_REGIME
            <= regime
            <= self.GRID_V9_NEUTRAL_SHORT_MAX_REGIME
        ):
            stake *= self.GRID_V9_NEUTRAL_SHORT_SCALE

        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV12GridV9NeutralShortRisk035Freqtrade(
    _GridV9NeutralShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_NEUTRAL_SHORT_SCALE = 0.35


class BreakoutV12GridV9NeutralShortRisk050Freqtrade(
    _GridV9NeutralShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_NEUTRAL_SHORT_SCALE = 0.50


class BreakoutV12GridV9NeutralShortRisk065Freqtrade(
    _GridV9NeutralShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_NEUTRAL_SHORT_SCALE = 0.65


class BreakoutV12GridV9NeutralShortRisk080Freqtrade(
    _GridV9NeutralShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_NEUTRAL_SHORT_SCALE = 0.80


class BreakoutV12GridV9AdaptiveFloorM010Risk065Freqtrade(
    _GridV9NeutralShortRiskMixin,
    BreakoutV12GridV9AdaptiveOneTpFloorM010Freqtrade,
):
    """Candidate combining the orthogonal exit and allocation defenses."""

    GRID_V9_NEUTRAL_SHORT_SCALE = 0.65


class BreakoutV12GridV9WeakBullShortRisk000Freqtrade(
    _GridV9NeutralShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    """Skip persistent Grid shorts in a weak-bull, directionless market."""

    GRID_V9_NEUTRAL_SHORT_SCALE = 0.00
    GRID_V9_NEUTRAL_SHORT_MIN_REGIME = 0.05


class BreakoutV12GridV9WeakBullShortRisk025Freqtrade(
    _GridV9NeutralShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_NEUTRAL_SHORT_SCALE = 0.25
    GRID_V9_NEUTRAL_SHORT_MIN_REGIME = 0.05


class BreakoutV12GridV9WeakBullShortRisk050Freqtrade(
    _GridV9NeutralShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_NEUTRAL_SHORT_SCALE = 0.50
    GRID_V9_NEUTRAL_SHORT_MIN_REGIME = 0.05


class BreakoutV12GridV9WeakBullShortRisk075Freqtrade(
    _GridV9NeutralShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_NEUTRAL_SHORT_SCALE = 0.75
    GRID_V9_NEUTRAL_SHORT_MIN_REGIME = 0.05


class _GridV9PersistentReboundRiskMixin:
    """Avoid fading a rebound while the market has stayed opportunity-poor.

    This is intentionally an allocation guard instead of a signal rewrite:
    the frozen Grid-v8 signal, ranking, exits, and order behavior remain intact.
    """

    GRID_V9_REBOUND_SHORT_SCALE = 0.00
    GRID_V9_REBOUND_MIN_RETURN_4H = 0.0075
    GRID_V9_REBOUND_MIN_PERSISTENCE = 0.75
    GRID_V9_REBOUND_REQUIRE_MARKET_CONFIRMATION = False
    GRID_V9_REBOUND_MIN_ETH_RETURN_4H = 0.005
    GRID_V9_REBOUND_MIN_BREADTH = 0.50
    GRID_V9_REBOUND_MIN_REGIME = 0.00

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
        market_confirmation = (
            float(row.get("eth_return_4h", -np.inf))
            >= self.GRID_V9_REBOUND_MIN_ETH_RETURN_4H
            or float(row.get("breadth", -np.inf))
            >= self.GRID_V9_REBOUND_MIN_BREADTH
            or float(row.get("market_regime_score", -np.inf))
            >= self.GRID_V9_REBOUND_MIN_REGIME
        )
        if (
            float(row.get("symbol_return_4h", -1.0))
            >= self.GRID_V9_REBOUND_MIN_RETURN_4H
            and float(row.get("low_opportunity_fraction_72h", -1.0))
            >= self.GRID_V9_REBOUND_MIN_PERSISTENCE
            and (
                not self.GRID_V9_REBOUND_REQUIRE_MARKET_CONFIRMATION
                or market_confirmation
            )
        ):
            stake *= self.GRID_V9_REBOUND_SHORT_SCALE

        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV12GridV9PersistentReboundGuard005Freqtrade(
    _GridV9PersistentReboundRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_REBOUND_MIN_RETURN_4H = 0.005


class BreakoutV12GridV9PersistentReboundGuard0075Freqtrade(
    _GridV9PersistentReboundRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_REBOUND_MIN_RETURN_4H = 0.0075


class BreakoutV12GridV9PersistentReboundGuard010Freqtrade(
    _GridV9PersistentReboundRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_REBOUND_MIN_RETURN_4H = 0.010
    GRID_V9_REBOUND_MIN_PERSISTENCE = 0.90


class BreakoutV12GridV9ConfirmedPersistentReboundGuardFreqtrade(
    _GridV9PersistentReboundRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_REBOUND_MIN_RETURN_4H = 0.0075
    GRID_V9_REBOUND_MIN_PERSISTENCE = 0.75
    GRID_V9_REBOUND_REQUIRE_MARKET_CONFIRMATION = True


class _GridV9BenchmarkReboundRiskMixin:
    """Block a Grid short when both the symbol and ETH confirm a rebound."""

    GRID_V9_BENCHMARK_REBOUND_SCALE = 0.00
    GRID_V9_BENCHMARK_REBOUND_MIN_SYMBOL_RETURN_4H = 0.005
    GRID_V9_BENCHMARK_REBOUND_MIN_ETH_RETURN_4H = 0.00
    GRID_V9_BENCHMARK_REBOUND_PORTFOLIO_CAP: float | None = None

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
        if (
            float(row.get("symbol_return_4h", -np.inf))
            >= self.GRID_V9_BENCHMARK_REBOUND_MIN_SYMBOL_RETURN_4H
            and float(row.get("eth_return_4h", -np.inf))
            >= self.GRID_V9_BENCHMARK_REBOUND_MIN_ETH_RETURN_4H
        ):
            factor = self.GRID_V9_BENCHMARK_REBOUND_SCALE
            if self.GRID_V9_BENCHMARK_REBOUND_PORTFOLIO_CAP is not None:
                equity = float(self.wallets.get_total_stake_amount())
                portfolio_scale = self._portfolio_risk_scale(
                    equity,
                    current_time,
                )
                factor = min(
                    1.0,
                    self.GRID_V9_BENCHMARK_REBOUND_PORTFOLIO_CAP
                    / max(float(portfolio_scale), 1e-12),
                )
            stake *= factor

        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV12GridV9BenchmarkReboundGuard005Freqtrade(
    _GridV9BenchmarkReboundRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_BENCHMARK_REBOUND_MIN_SYMBOL_RETURN_4H = 0.005


class BreakoutV12GridV9BenchmarkReboundGuard0075Freqtrade(
    _GridV9BenchmarkReboundRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_BENCHMARK_REBOUND_MIN_SYMBOL_RETURN_4H = 0.0075


class BreakoutV12GridV9BenchmarkReboundRisk050Freqtrade(
    _GridV9BenchmarkReboundRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_BENCHMARK_REBOUND_MIN_SYMBOL_RETURN_4H = 0.0075
    GRID_V9_BENCHMARK_REBOUND_SCALE = 0.50


class _GridV9LongOverextensionRiskMixin:
    """Avoid late Grid longs after an efficient trend is already stretched."""

    GRID_V9_LONG_OVEREXTENSION_SCALE = 0.00
    GRID_V9_LONG_MAX_ALIGNMENT_ATR = 1.25
    GRID_V9_LONG_MIN_MARKET_EFFICIENCY = 0.23
    GRID_V9_LONG_MAX_BREADTH_CHANGE_24H: float | None = None

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
            or side != "long"
        ):
            return stake

        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None:
            return stake
        weak_structure = (
            self.GRID_V9_LONG_MAX_BREADTH_CHANGE_24H is None
            or float(row.get("breadth_change_24h", np.inf))
            <= self.GRID_V9_LONG_MAX_BREADTH_CHANGE_24H
        )
        if (
            float(row.get("grid_long_alignment", -np.inf))
            >= self.GRID_V9_LONG_MAX_ALIGNMENT_ATR
            and float(row.get("market_efficiency", -np.inf))
            >= self.GRID_V9_LONG_MIN_MARKET_EFFICIENCY
            and weak_structure
        ):
            stake *= self.GRID_V9_LONG_OVEREXTENSION_SCALE

        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV12GridV9LongOverextension120Freqtrade(
    _GridV9LongOverextensionRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_LONG_MAX_ALIGNMENT_ATR = 1.20


class BreakoutV12GridV9LongOverextension125Freqtrade(
    _GridV9LongOverextensionRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_LONG_MAX_ALIGNMENT_ATR = 1.25


class BreakoutV12GridV9LongOverextension130Freqtrade(
    _GridV9LongOverextensionRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_LONG_MAX_ALIGNMENT_ATR = 1.30


class _BreakoutV12NarrowShortRiskMixin:
    """Reject low-expansion short breakouts without changing frozen signals."""

    BO_V12_NARROW_SHORT_SCALE = 0.00
    BO_V12_NARROW_SHORT_MAX_RANGE_ATR = 4.50
    BO_V12_NARROW_SHORT_MIN_REGIME: float | None = None
    BO_V12_NARROW_SHORT_MIN_ETH_RETURN_4H: float | None = None

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
        regime = float(row.get("market_regime_score", -2.0))
        eth_return_4h = float(row.get("eth_return_4h", -1.0))
        blocked = (
            float(row.get("bo_range_atr", np.inf))
            < self.BO_V12_NARROW_SHORT_MAX_RANGE_ATR
            and (
                self.BO_V12_NARROW_SHORT_MIN_REGIME is None
                or regime >= self.BO_V12_NARROW_SHORT_MIN_REGIME
            )
            and (
                self.BO_V12_NARROW_SHORT_MIN_ETH_RETURN_4H is None
                or eth_return_4h
                >= self.BO_V12_NARROW_SHORT_MIN_ETH_RETURN_4H
            )
        )
        if blocked:
            stake *= self.BO_V12_NARROW_SHORT_SCALE

        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV12NarrowShortGuard420GridV8Freqtrade(
    _BreakoutV12NarrowShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_NARROW_SHORT_MAX_RANGE_ATR = 4.20


class BreakoutV12NarrowShortGuard450GridV8Freqtrade(
    _BreakoutV12NarrowShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_NARROW_SHORT_MAX_RANGE_ATR = 4.50


class BreakoutV12NarrowShortGuard480GridV8Freqtrade(
    _BreakoutV12NarrowShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_NARROW_SHORT_MAX_RANGE_ATR = 4.80


class BreakoutV12NarrowShortNeutralGuardGridV8Freqtrade(
    _BreakoutV12NarrowShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_NARROW_SHORT_MAX_RANGE_ATR = 4.50
    BO_V12_NARROW_SHORT_MIN_REGIME = -0.20


class BreakoutV12NarrowShortBenchmarkGuardGridV8Freqtrade(
    _BreakoutV12NarrowShortRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    """Require benchmark downside confirmation for a narrow short break."""

    BO_V12_NARROW_SHORT_MAX_RANGE_ATR = 4.50
    BO_V12_NARROW_SHORT_MIN_ETH_RETURN_4H = -0.005


class _BreakoutV12ModerateBullLongBoostMixin:
    """Reallocate risk toward proven Breakout longs in a broad, orderly bull."""

    BO_V12_MODERATE_BULL_LONG_SCALE = 1.20
    BO_V12_MODERATE_BULL_MIN_SLOW_BREADTH = 0.55
    BO_V12_MODERATE_BULL_MIN_SCORE = 0
    BO_V12_MODERATE_BULL_MAX_SCORE = 4
    BO_V12_MODERATE_BULL_MAX_LOW_OPPORTUNITY = np.inf
    BO_V12_MODERATE_BULL_MAX_RANGE_ATR = np.inf

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
        score = int(row.get("bo_score", 99))
        moderate_bull = (
            0.0 <= float(row.get("btc_trend_1d", -np.inf)) <= 1.0
            and float(row.get("breadth_slow", -np.inf))
            >= self.BO_V12_MODERATE_BULL_MIN_SLOW_BREADTH
            and self.BO_V12_MODERATE_BULL_MIN_SCORE
            <= score
            <= self.BO_V12_MODERATE_BULL_MAX_SCORE
            and float(row.get("low_opportunity_fraction_72h", -np.inf))
            < self.BO_V12_MODERATE_BULL_MAX_LOW_OPPORTUNITY
            and float(row.get("bo_range_atr", np.inf))
            <= self.BO_V12_MODERATE_BULL_MAX_RANGE_ATR
        )
        if moderate_bull:
            stake *= self.BO_V12_MODERATE_BULL_LONG_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV12ModerateBullLongBoost110GridV8Freqtrade(
    _BreakoutV12ModerateBullLongBoostMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_MODERATE_BULL_LONG_SCALE = 1.10


class BreakoutV12ModerateBullLongBoost120GridV8Freqtrade(
    _BreakoutV12ModerateBullLongBoostMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_MODERATE_BULL_LONG_SCALE = 1.20


class BreakoutV12ModerateBullLongBoost130GridV8Freqtrade(
    _BreakoutV12ModerateBullLongBoostMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_MODERATE_BULL_LONG_SCALE = 1.30


class BreakoutV12SelectiveBullRunnerBoost110GridV8Freqtrade(
    _BreakoutV12ModerateBullLongBoostMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    """Boost only score-four runners in a liquid, non-compressed bull state."""

    BO_V12_MODERATE_BULL_LONG_SCALE = 1.10
    BO_V12_MODERATE_BULL_MIN_SCORE = 4
    BO_V12_MODERATE_BULL_MAX_SCORE = 4
    BO_V12_MODERATE_BULL_MAX_LOW_OPPORTUNITY = 0.50
    BO_V12_MODERATE_BULL_MAX_RANGE_ATR = 5.25


class BreakoutV12SelectiveBullRunnerBoost120GridV8Freqtrade(
    BreakoutV12SelectiveBullRunnerBoost110GridV8Freqtrade
):
    BO_V12_MODERATE_BULL_LONG_SCALE = 1.20


class BreakoutV12MultiRegimeCoreAFreqtrade(
    _GridV9PersistentReboundRiskMixin,
    BreakoutV12GridV9WeakBullShortRisk000Freqtrade,
):
    """Two orthogonal Grid-short guards proven on separate market states."""

    GRID_V9_REBOUND_MIN_RETURN_4H = 0.0075
    GRID_V9_REBOUND_MIN_PERSISTENCE = 0.75


class BreakoutV12MultiRegimeCoreAConfirmedFreqtrade(
    _GridV9PersistentReboundRiskMixin,
    BreakoutV12GridV9WeakBullShortRisk000Freqtrade,
):
    """Core-A with broad-market confirmation before suppressing a rebound."""

    GRID_V9_REBOUND_MIN_RETURN_4H = 0.0075
    GRID_V9_REBOUND_MIN_PERSISTENCE = 0.75
    GRID_V9_REBOUND_REQUIRE_MARKET_CONFIRMATION = True


class BreakoutV12MultiRegimeCoreAConfirmedLongGuardFreqtrade(
    _GridV9LongOverextensionRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedFreqtrade,
):
    GRID_V9_LONG_MAX_ALIGNMENT_ATR = 1.25
    GRID_V9_LONG_MIN_MARKET_EFFICIENCY = 0.23


class BreakoutV12MultiRegimeCoreAConfirmedWeakStructureLongGuardFreqtrade(
    _GridV9LongOverextensionRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedFreqtrade,
):
    """Suppress late Grid longs only while Active50 breadth is collapsing."""

    GRID_V9_LONG_MAX_ALIGNMENT_ATR = 1.25
    GRID_V9_LONG_MIN_MARKET_EFFICIENCY = 0.23
    GRID_V9_LONG_MAX_BREADTH_CHANGE_24H = -0.25


class BreakoutV12MultiRegimeCoreABenchmarkFreqtrade(
    _BreakoutV12NarrowShortRiskMixin,
    BreakoutV12MultiRegimeCoreAFreqtrade,
):
    BO_V12_NARROW_SHORT_MAX_RANGE_ATR = 4.50
    BO_V12_NARROW_SHORT_MIN_ETH_RETURN_4H = -0.005


class BreakoutV12MultiRegimeCoreABenchmarkRisk050Freqtrade(
    BreakoutV12MultiRegimeCoreABenchmarkFreqtrade
):
    BO_V12_NARROW_SHORT_SCALE = 0.50


class BreakoutV12MultiRegimeCoreALongGuardFreqtrade(
    _GridV9LongOverextensionRiskMixin,
    BreakoutV12MultiRegimeCoreAFreqtrade,
):
    GRID_V9_LONG_MAX_ALIGNMENT_ATR = 1.25
    GRID_V9_LONG_MIN_MARKET_EFFICIENCY = 0.23


class BreakoutV12MultiRegimeCoreALongRisk050Freqtrade(
    BreakoutV12MultiRegimeCoreALongGuardFreqtrade
):
    GRID_V9_LONG_OVEREXTENSION_SCALE = 0.50


class BreakoutV12MultiRegimeCoreBScaledFreqtrade(
    _BreakoutV12NarrowShortRiskMixin,
    _GridV9LongOverextensionRiskMixin,
    BreakoutV12MultiRegimeCoreAFreqtrade,
):
    BO_V12_NARROW_SHORT_SCALE = 0.50
    BO_V12_NARROW_SHORT_MAX_RANGE_ATR = 4.50
    BO_V12_NARROW_SHORT_MIN_ETH_RETURN_4H = -0.005
    GRID_V9_LONG_OVEREXTENSION_SCALE = 0.50
    GRID_V9_LONG_MAX_ALIGNMENT_ATR = 1.25
    GRID_V9_LONG_MIN_MARKET_EFFICIENCY = 0.23


class BreakoutV12MultiRegimeCoreBFreqtrade(
    _BreakoutV12NarrowShortRiskMixin,
    _GridV9LongOverextensionRiskMixin,
    _GridV9PersistentReboundRiskMixin,
    BreakoutV12GridV9WeakBullShortRisk000Freqtrade,
):
    """Side-specific market-state defenses on the exact frozen execution path."""

    GRID_V9_REBOUND_MIN_RETURN_4H = 0.0075
    GRID_V9_REBOUND_MIN_PERSISTENCE = 0.75
    GRID_V9_LONG_MAX_ALIGNMENT_ATR = 1.25
    GRID_V9_LONG_MIN_MARKET_EFFICIENCY = 0.23
    BO_V12_NARROW_SHORT_MAX_RANGE_ATR = 4.50
    BO_V12_NARROW_SHORT_MIN_ETH_RETURN_4H = -0.005


class _BreakoutV12LongSleeveRiskMixin:
    """Pre-emptively de-risk unsupported pumps after the long sleeve fails."""

    BO_V12_LONG_SLEEVE_SCALE = 0.35
    BO_V12_LONG_SLEEVE_WINDOW = 6
    BO_V12_LONG_SLEEVE_MIN_LOSSES = 4
    BO_V12_LONG_SLEEVE_MAX_RETURN = -0.10
    BO_V12_LONG_SLEEVE_MIN_SYMBOL_RETURN_4H = 0.05
    BO_V12_LONG_SLEEVE_MAX_MARKET_RETURN_24H = 0.01
    BO_V12_LONG_SLEEVE_PORTFOLIO_CAP: float | None = None

    def _bo_v12_recent_long_returns(
        self,
        current_time: datetime,
    ) -> tuple[float, ...]:
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        closed: list[tuple[datetime, float]] = []
        for trade in Trade.get_trades_proxy(is_open=False):
            if (
                self._component(getattr(trade, "enter_tag", None))
                != "breakout"
                or bool(getattr(trade, "is_short", False))
            ):
                continue
            close_time = self._trade_close_time(trade)
            value = self._closed_trade_return(trade)
            if (
                close_time is None
                or close_time > boundary
                or value is None
                or not np.isfinite(value)
            ):
                continue
            closed.append((close_time, float(value)))
        closed.sort(key=lambda item: item[0])
        return tuple(
            value
            for _close_time, value in closed[
                -self.BO_V12_LONG_SLEEVE_WINDOW :
            ]
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
            or self._component(entry_tag) != "breakout"
            or side != "long"
        ):
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return stake
        unsupported_pump = (
            float(row.get("symbol_return_4h", -np.inf))
            >= self.BO_V12_LONG_SLEEVE_MIN_SYMBOL_RETURN_4H
            and float(row.get("market_median_return_24h", np.inf))
            <= self.BO_V12_LONG_SLEEVE_MAX_MARKET_RETURN_24H
        )
        recent = self._bo_v12_recent_long_returns(current_time)
        sleeve_failed = (
            len(recent) >= self.BO_V12_LONG_SLEEVE_WINDOW
            and sum(value < 0.0 for value in recent)
            >= self.BO_V12_LONG_SLEEVE_MIN_LOSSES
            and float(sum(recent))
            <= self.BO_V12_LONG_SLEEVE_MAX_RETURN
        )
        if unsupported_pump and sleeve_failed:
            factor = self.BO_V12_LONG_SLEEVE_SCALE
            if self.BO_V12_LONG_SLEEVE_PORTFOLIO_CAP is not None:
                equity = float(self.wallets.get_total_stake_amount())
                portfolio_scale = self._portfolio_risk_scale(
                    equity,
                    current_time,
                )
                factor = min(
                    1.0,
                    self.BO_V12_LONG_SLEEVE_PORTFOLIO_CAP
                    / max(float(portfolio_scale), 1e-12),
                )
            stake *= factor
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV12LongSleeveRisk000GridV8Freqtrade(
    _BreakoutV12LongSleeveRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_LONG_SLEEVE_SCALE = 0.00


class BreakoutV12LongSleeveRisk035GridV8Freqtrade(
    _BreakoutV12LongSleeveRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_LONG_SLEEVE_SCALE = 0.35


class BreakoutV12MultiRegimeCoreALongSleeve035Freqtrade(
    _BreakoutV12LongSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreAFreqtrade,
):
    BO_V12_LONG_SLEEVE_SCALE = 0.35


class BreakoutV12MultiRegimeCoreALongGuardLongSleeve035Freqtrade(
    _BreakoutV12LongSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreALongGuardFreqtrade,
):
    BO_V12_LONG_SLEEVE_SCALE = 0.35


class _BreakoutV12ShortSleeveRiskMixin:
    """De-risk shorts after sleeve failure in a persistent low-edge regime."""

    BO_V12_SHORT_SLEEVE_SCALE = 0.00
    BO_V12_SHORT_SLEEVE_WINDOW = 3
    BO_V12_SHORT_SLEEVE_MIN_LOSSES = 3
    BO_V12_SHORT_SLEEVE_MAX_RETURN = -0.20
    BO_V12_SHORT_SLEEVE_MIN_LOW_OPPORTUNITY = 0.50
    BO_V12_SHORT_SLEEVE_PORTFOLIO_CAP: float | None = None

    def _bo_v12_recent_short_returns(
        self,
        current_time: datetime,
    ) -> tuple[float, ...]:
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        closed: list[tuple[datetime, float]] = []
        for trade in Trade.get_trades_proxy(is_open=False):
            if (
                self._component(getattr(trade, "enter_tag", None))
                != "breakout"
                or not bool(getattr(trade, "is_short", False))
            ):
                continue
            close_time = self._trade_close_time(trade)
            value = self._closed_trade_return(trade)
            if (
                close_time is None
                or close_time > boundary
                or value is None
                or not np.isfinite(value)
            ):
                continue
            closed.append((close_time, float(value)))
        closed.sort(key=lambda item: item[0])
        return tuple(
            value
            for _close_time, value in closed[
                -self.BO_V12_SHORT_SLEEVE_WINDOW :
            ]
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
            or self._component(entry_tag) != "breakout"
            or side != "short"
        ):
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return stake
        recent = self._bo_v12_recent_short_returns(current_time)
        sleeve_failed = (
            len(recent) >= self.BO_V12_SHORT_SLEEVE_WINDOW
            and sum(value < 0.0 for value in recent)
            >= self.BO_V12_SHORT_SLEEVE_MIN_LOSSES
            and float(sum(recent))
            <= self.BO_V12_SHORT_SLEEVE_MAX_RETURN
        )
        persistent_low_edge = (
            float(row.get("low_opportunity_fraction_72h", -np.inf))
            >= self.BO_V12_SHORT_SLEEVE_MIN_LOW_OPPORTUNITY
        )
        if sleeve_failed and persistent_low_edge:
            factor = self.BO_V12_SHORT_SLEEVE_SCALE
            if self.BO_V12_SHORT_SLEEVE_PORTFOLIO_CAP is not None:
                equity = float(self.wallets.get_total_stake_amount())
                portfolio_scale = self._portfolio_risk_scale(
                    equity,
                    current_time,
                )
                factor = min(
                    1.0,
                    self.BO_V12_SHORT_SLEEVE_PORTFOLIO_CAP
                    / max(float(portfolio_scale), 1e-12),
                )
            stake *= factor
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV12ShortSleeveRisk000GridV8Freqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_SHORT_SLEEVE_SCALE = 0.00


class BreakoutV12ShortSleeveRisk035GridV8Freqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_SHORT_SLEEVE_SCALE = 0.35


class BreakoutV12MultiRegimeCoreALongGuardShortSleeveFreqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreALongGuardFreqtrade,
):
    BO_V12_SHORT_SLEEVE_SCALE = 0.00


class BreakoutV12MultiRegimeCoreALongGuardShortSleeve035Freqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreALongGuardFreqtrade,
):
    BO_V12_SHORT_SLEEVE_SCALE = 0.35


class BreakoutV12MultiRegimeCoreCFreqtrade(
    _GridV9BenchmarkReboundRiskMixin,
    BreakoutV12MultiRegimeCoreALongGuardFreqtrade,
):
    """Core-A plus broad ETH-confirmed protection against rebound shorts."""

    GRID_V9_BENCHMARK_REBOUND_MIN_SYMBOL_RETURN_4H = 0.0075
    GRID_V9_BENCHMARK_REBOUND_MIN_ETH_RETURN_4H = 0.00
    GRID_V9_BENCHMARK_REBOUND_SCALE = 0.00


class BreakoutV12MultiRegimeCoreCRisk050Freqtrade(
    _GridV9BenchmarkReboundRiskMixin,
    BreakoutV12MultiRegimeCoreALongGuardFreqtrade,
):
    GRID_V9_BENCHMARK_REBOUND_MIN_SYMBOL_RETURN_4H = 0.0075
    GRID_V9_BENCHMARK_REBOUND_MIN_ETH_RETURN_4H = 0.00
    GRID_V9_BENCHMARK_REBOUND_SCALE = 0.50


class BreakoutV12MultiRegimeCoreCShortSleeveFreqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreCFreqtrade,
):
    BO_V12_SHORT_SLEEVE_SCALE = 0.00


class BreakoutV12MultiRegimeConfirmedShortSleeveFreqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedLongGuardFreqtrade,
):
    BO_V12_SHORT_SLEEVE_SCALE = 0.00


class BreakoutV12MultiRegimeConfirmedShortSleeve035Freqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedLongGuardFreqtrade,
):
    """Confirmed Grid guards plus a non-zero Breakout-short recovery sleeve."""

    BO_V12_SHORT_SLEEVE_SCALE = 0.35


class BreakoutV12MultiRegimeCoreAConfirmedShortSleeve035Freqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedFreqtrade,
):
    """Confirmed Grid-short guards with no Grid-long intervention."""

    BO_V12_SHORT_SLEEVE_SCALE = 0.35


class BreakoutV12MultiRegimeCoreALongGuardDualSleeve035Freqtrade(
    _BreakoutV12LongSleeveRiskMixin,
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreALongGuardFreqtrade,
):
    """Scale only the failing Breakout direction while keeping its slot."""

    BO_V12_LONG_SLEEVE_SCALE = 0.35
    BO_V12_SHORT_SLEEVE_SCALE = 0.35


class BreakoutV12MultiRegimeConfirmedDualSleeve035Freqtrade(
    _BreakoutV12LongSleeveRiskMixin,
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedLongGuardFreqtrade,
):
    """Market-confirmed Grid guards with independent long/short sleeves."""

    BO_V12_LONG_SLEEVE_SCALE = 0.35
    BO_V12_SHORT_SLEEVE_SCALE = 0.35


class BreakoutV12MultiRegimeConfirmedWeakLongShortSleeve035Freqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedWeakStructureLongGuardFreqtrade,
):
    """Stage-balanced candidate without suppressing healthy bull-market Grid longs."""

    BO_V12_SHORT_SLEEVE_SCALE = 0.35


class BreakoutV12MultiRegimeConfirmedWeakLongShortSleeve020Freqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedWeakStructureLongGuardFreqtrade,
):
    BO_V12_SHORT_SLEEVE_SCALE = 0.20


class BreakoutV12MultiRegimeConfirmedWeakLongShortSleeve050Freqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedWeakStructureLongGuardFreqtrade,
):
    BO_V12_SHORT_SLEEVE_SCALE = 0.50


class BreakoutV12MultiRegimeConfirmedWeakLongShortCap035Freqtrade(
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedWeakStructureLongGuardFreqtrade,
):
    BO_V12_SHORT_SLEEVE_PORTFOLIO_CAP = 0.35


class BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve035Freqtrade(
    _BreakoutV12LongSleeveRiskMixin,
    _BreakoutV12ShortSleeveRiskMixin,
    BreakoutV12MultiRegimeCoreAConfirmedWeakStructureLongGuardFreqtrade,
):
    """Finalist with independent recovery budgets for both Breakout sides."""

    BO_V12_LONG_SLEEVE_SCALE = 0.35
    BO_V12_SHORT_SLEEVE_SCALE = 0.35


class BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve060Freqtrade(
    BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve035Freqtrade,
):
    """Robust plateau: de-risk only unsupported pumps above a 6% four-hour move."""

    BO_V12_LONG_SLEEVE_MIN_SYMBOL_RETURN_4H = 0.06


class BreakoutV12MultiRegimeConfirmedWeakLongRisk020Freqtrade(
    BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve060Freqtrade,
):
    BO_V12_LONG_SLEEVE_SCALE = 0.20


class BreakoutV12MultiRegimeConfirmedWeakLongRisk050Freqtrade(
    BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve060Freqtrade,
):
    BO_V12_LONG_SLEEVE_SCALE = 0.50


class BreakoutV12MultiRegimeConfirmedWeakLongShortRisk020Freqtrade(
    BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve060Freqtrade,
):
    """Keep long recovery risk at 0.35 and reduce the failed short sleeve."""

    BO_V12_SHORT_SLEEVE_SCALE = 0.20


class BreakoutV12MultiRegimeConfirmedWeakLongShortRisk050Freqtrade(
    BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve060Freqtrade,
):
    """Upper short-risk neighbor for walk-forward sensitivity testing."""

    BO_V12_SHORT_SLEEVE_SCALE = 0.50


class _GridV9RelativeStrengthShortRiskMixin:
    """De-risk a narrow Grid-short squeeze setup.

    A symbol that refuses to fall while the broad regime is bearish has
    positive relative strength.  In a moderately persistent low-opportunity
    state this is not a healthy trend-following short; it is a squeeze risk.
    The bounded persistence band deliberately leaves both ordinary trends
    and the separately handled extreme-compression regime unchanged.
    """

    GRID_V9_RS_SHORT_SCALE = 0.35
    GRID_V9_RS_SHORT_MIN_PERSISTENCE = 0.50
    GRID_V9_RS_SHORT_MAX_PERSISTENCE = 0.75
    GRID_V9_RS_SHORT_MAX_REGIME = -0.15
    GRID_V9_RS_SHORT_MIN_SYMBOL_RETURN_4H = 0.00

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
        persistence = float(
            row.get("low_opportunity_fraction_72h", -np.inf)
        )
        squeeze_risk = (
            self.GRID_V9_RS_SHORT_MIN_PERSISTENCE
            <= persistence
            <= self.GRID_V9_RS_SHORT_MAX_PERSISTENCE
            and float(row.get("market_regime_score", np.inf))
            <= self.GRID_V9_RS_SHORT_MAX_REGIME
            and float(row.get("symbol_return_4h", -np.inf))
            >= self.GRID_V9_RS_SHORT_MIN_SYMBOL_RETURN_4H
        )
        if squeeze_risk:
            stake *= self.GRID_V9_RS_SHORT_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV12MultiRegimeRelativeStrengthGuard035Freqtrade(
    _GridV9RelativeStrengthShortRiskMixin,
    BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve060Freqtrade,
):
    """Walk-forward candidate with a non-zero relative-strength guard."""

    GRID_V9_RS_SHORT_SCALE = 0.35


class BreakoutV12MultiRegimeRelativeStrengthGuard000Freqtrade(
    _GridV9RelativeStrengthShortRiskMixin,
    BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve060Freqtrade,
):
    """Hard-skip neighbor used only to test guard sensitivity."""

    GRID_V9_RS_SHORT_SCALE = 0.00


class BreakoutV12MultiRegimeRelativeStrengthGuard035Short020Freqtrade(
    BreakoutV12MultiRegimeRelativeStrengthGuard035Freqtrade,
):
    """Relative-strength guard combined with the conservative short sleeve."""

    BO_V12_SHORT_SLEEVE_SCALE = 0.20


class _BreakoutV12MatureBullChaseRiskMixin:
    """Temper late-cycle Breakout longs when fast market breadth is fading."""

    BO_V12_MATURE_BULL_CHASE_SCALE = 0.75
    BO_V12_MATURE_BULL_CHASE_MIN_PERSISTENCE = 0.75
    BO_V12_MATURE_BULL_CHASE_MIN_REGIME = 0.20
    BO_V12_MATURE_BULL_CHASE_MIN_SYMBOL_RETURN_4H = 0.04
    BO_V12_MATURE_BULL_CHASE_MAX_BREADTH_CHANGE_4H = 0.05

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
        unscaled_stake = float(stake)
        chase_risk = (
            float(row.get("low_opportunity_fraction_72h", -np.inf))
            >= self.BO_V12_MATURE_BULL_CHASE_MIN_PERSISTENCE
            and float(row.get("market_regime_score", -np.inf))
            >= self.BO_V12_MATURE_BULL_CHASE_MIN_REGIME
            and float(row.get("symbol_return_4h", -np.inf))
            >= self.BO_V12_MATURE_BULL_CHASE_MIN_SYMBOL_RETURN_4H
            and float(row.get("breadth_change_4h", np.inf))
            <= self.BO_V12_MATURE_BULL_CHASE_MAX_BREADTH_CHANGE_4H
        )
        if chase_risk:
            stake *= self.BO_V12_MATURE_BULL_CHASE_SCALE
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            # This layer is a sizing guard, not an entry filter.  Keeping the
            # original already-small stake avoids freeing a Max2 slot and
            # changing which unrelated candidate is admitted next.
            return min(float(max_stake), unscaled_stake)
        return scaled


class BreakoutV12MultiRegimeMatureBull075RelativeGuardFreqtrade(
    _BreakoutV12MatureBullChaseRiskMixin,
    BreakoutV12MultiRegimeRelativeStrengthGuard035Freqtrade,
):
    """Research neighbor combining both causal, non-zero risk guards."""

    BO_V12_MATURE_BULL_CHASE_SCALE = 0.75


class BreakoutV12MultiRegimeMatureBull060RelativeGuardFreqtrade(
    _BreakoutV12MatureBullChaseRiskMixin,
    BreakoutV12MultiRegimeRelativeStrengthGuard035Freqtrade,
):
    """Lower-risk sensitivity neighbor; not selected without all-stage proof."""

    BO_V12_MATURE_BULL_CHASE_SCALE = 0.60


class BreakoutV12MultiRegimeMatureBull035RelativeGuardFreqtrade(
    _BreakoutV12MatureBullChaseRiskMixin,
    BreakoutV12MultiRegimeRelativeStrengthGuard035Freqtrade,
):
    """Lower fading-breadth risk while retaining the Max2 position slot."""

    BO_V12_MATURE_BULL_CHASE_SCALE = 0.35


class BreakoutV12MultiRegimeBenchmark050WeakDualSleeve060Freqtrade(
    _GridV9BenchmarkReboundRiskMixin,
    BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve060Freqtrade,
):
    """Keep rebound signals in-slot at half risk unless Core-A confirms danger."""

    GRID_V9_BENCHMARK_REBOUND_MIN_SYMBOL_RETURN_4H = 0.0075
    GRID_V9_BENCHMARK_REBOUND_MIN_ETH_RETURN_4H = 0.00
    GRID_V9_BENCHMARK_REBOUND_SCALE = 0.50


class BreakoutV12MultiRegimeLossStreak3DualSleeve035Freqtrade(
    _BreakoutV12LongSleeveRiskMixin,
    BreakoutV12MultiRegimeConfirmedWeakLongShortSleeve035Freqtrade,
):
    """De-risk idiosyncratic pumps after three completed long-sleeve losses."""

    BO_V12_LONG_SLEEVE_SCALE = 0.35
    BO_V12_LONG_SLEEVE_WINDOW = 3
    BO_V12_LONG_SLEEVE_MIN_LOSSES = 3
    BO_V12_LONG_SLEEVE_MAX_RETURN = -0.30
    BO_V12_LONG_SLEEVE_MIN_SYMBOL_RETURN_4H = 0.065
    BO_V12_LONG_SLEEVE_MAX_MARKET_RETURN_24H = 0.01


class BreakoutV12MultiRegimeLossStreak3LongCap015Freqtrade(
    BreakoutV12MultiRegimeLossStreak3DualSleeve035Freqtrade,
):
    """Use a 15% absolute portfolio-risk cap instead of a second multiplier."""

    BO_V12_LONG_SLEEVE_PORTFOLIO_CAP = 0.15


class BreakoutV12MultiRegimeBenchmark050LossStreak3Freqtrade(
    _GridV9BenchmarkReboundRiskMixin,
    BreakoutV12MultiRegimeLossStreak3DualSleeve035Freqtrade,
):
    GRID_V9_BENCHMARK_REBOUND_MIN_SYMBOL_RETURN_4H = 0.0075
    GRID_V9_BENCHMARK_REBOUND_MIN_ETH_RETURN_4H = 0.00
    GRID_V9_BENCHMARK_REBOUND_SCALE = 0.50


class BreakoutV12MultiRegimeBenchmarkCap050WeakDualSleeve060Freqtrade(
    _GridV9BenchmarkReboundRiskMixin,
    BreakoutV12MultiRegimeConfirmedWeakLongDualSleeve060Freqtrade,
):
    """Cap rebound risk at 50% without double-scaling an active governor."""

    GRID_V9_BENCHMARK_REBOUND_MIN_SYMBOL_RETURN_4H = 0.0075
    GRID_V9_BENCHMARK_REBOUND_MIN_ETH_RETURN_4H = 0.00
    GRID_V9_BENCHMARK_REBOUND_PORTFOLIO_CAP = 0.50


class _BreakoutV12ConfirmedProfitLockMixin:
    """Lock a small gain after a completed 1h close proves a breakout."""

    BO_V12_CONFIRMED_LOCK_TRIGGER_R = 2.00
    BO_V12_CONFIRMED_LOCK_FLOOR_R = 0.10
    BO_V12_CONFIRMED_LOCK_TRAIL_GAP_R: float | None = None

    def _bo_v12_confirmed_lock_enabled(self, trade: Trade) -> bool:
        return True

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
            or not self._bo_v12_confirmed_lock_enabled(trade)
        ):
            return base_value
        confirmed_peak = self._confirmed_peak_r(pair, trade, current_time)
        if confirmed_peak < self.BO_V12_CONFIRMED_LOCK_TRIGGER_R:
            return base_value

        locked_r = self.BO_V12_CONFIRMED_LOCK_FLOOR_R
        if self.BO_V12_CONFIRMED_LOCK_TRAIL_GAP_R is not None:
            locked_r = max(
                locked_r,
                confirmed_peak - self.BO_V12_CONFIRMED_LOCK_TRAIL_GAP_R,
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
        locked_value = abs(
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
            return locked_value
        return min(abs(float(base_value)), locked_value)


class BreakoutV12ConfirmedLock150Floor000GridV8Freqtrade(
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_CONFIRMED_LOCK_TRIGGER_R = 1.50
    BO_V12_CONFIRMED_LOCK_FLOOR_R = 0.00


class BreakoutV12ConfirmedLock200Floor000GridV8Freqtrade(
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_CONFIRMED_LOCK_TRIGGER_R = 2.00
    BO_V12_CONFIRMED_LOCK_FLOOR_R = 0.00


class BreakoutV12ConfirmedLock200Floor010GridV8Freqtrade(
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_CONFIRMED_LOCK_TRIGGER_R = 2.00
    BO_V12_CONFIRMED_LOCK_FLOOR_R = 0.10


class BreakoutV12ConfirmedLock250Floor010GridV8Freqtrade(
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_CONFIRMED_LOCK_TRIGGER_R = 2.50
    BO_V12_CONFIRMED_LOCK_FLOOR_R = 0.10


class _BreakoutV12SelectiveConfirmedProfitLockMixin:
    """Arm the 2R floor only for entry states prone to full reversals."""

    BO_V12_LONG_LOCK_MIN_PERSISTENCE = 0.75
    BO_V12_LONG_LOCK_MIN_BREADTH_ACCELERATION = 0.50
    BO_V12_SHORT_LOCK_MIN_PERSISTENCE = 0.90
    BO_V12_SHORT_LOCK_MAX_TREND_AGE = 10
    BO_V12_SHORT_LOCK_MAX_EMA_SPREAD_ATR = 0.50

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        super().order_filled(pair, trade, order, current_time, **kwargs)
        if (
            self._component(getattr(trade, "enter_tag", None))
            != "breakout"
            or getattr(order, "ft_order_side", None)
            != getattr(trade, "entry_side", None)
            or int(getattr(trade, "nr_of_successful_entries", 0)) != 1
        ):
            return
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return
        persistence = float(row.get("low_opportunity_fraction_72h", -1.0))
        if trade.is_short:
            atr_value = max(float(row.get("atr", 0.0)), 1e-12)
            directional_spread = (
                float(row.get("slow_ema", 0.0))
                - float(row.get("fast_ema", 0.0))
            ) / atr_value
            enabled = (
                persistence >= self.BO_V12_SHORT_LOCK_MIN_PERSISTENCE
                or int(row.get("bo_short_trend_age", 10_000))
                <= self.BO_V12_SHORT_LOCK_MAX_TREND_AGE
                or directional_spread
                <= self.BO_V12_SHORT_LOCK_MAX_EMA_SPREAD_ATR
            )
        else:
            enabled = (
                persistence >= self.BO_V12_LONG_LOCK_MIN_PERSISTENCE
                or float(row.get("breadth_change_24h", -2.0))
                >= self.BO_V12_LONG_LOCK_MIN_BREADTH_ACCELERATION
            )
        trade.set_custom_data("bo_v12_selective_confirmed_lock", enabled)

    def _bo_v12_confirmed_lock_enabled(self, trade: Trade) -> bool:
        return bool(
            trade.get_custom_data("bo_v12_selective_confirmed_lock", False)
        )


class BreakoutV12SelectiveLockConservativeGridV8Freqtrade(
    _BreakoutV12SelectiveConfirmedProfitLockMixin,
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_SHORT_LOCK_MAX_TREND_AGE = 6
    BO_V12_SHORT_LOCK_MAX_EMA_SPREAD_ATR = 0.35


class BreakoutV12SelectiveLockBalancedGridV8Freqtrade(
    _BreakoutV12SelectiveConfirmedProfitLockMixin,
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    pass


class BreakoutV12SelectiveLockLongOnlyGridV8Freqtrade(
    _BreakoutV12SelectiveConfirmedProfitLockMixin,
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_SHORT_LOCK_MIN_PERSISTENCE = 2.0
    BO_V12_SHORT_LOCK_MAX_TREND_AGE = -1
    BO_V12_SHORT_LOCK_MAX_EMA_SPREAD_ATR = -1.0


class BreakoutV12SelectivePersistentLongLockGridV8Freqtrade(
    _BreakoutV12SelectiveConfirmedProfitLockMixin,
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_LONG_LOCK_MIN_PERSISTENCE = 0.75
    BO_V12_LONG_LOCK_MIN_BREADTH_ACCELERATION = 2.0
    BO_V12_SHORT_LOCK_MIN_PERSISTENCE = 2.0
    BO_V12_SHORT_LOCK_MAX_TREND_AGE = -1
    BO_V12_SHORT_LOCK_MAX_EMA_SPREAD_ATR = -1.0


class BreakoutV12ConfirmedTrail200Gap400FloorM050GridV8Freqtrade(
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_CONFIRMED_LOCK_TRIGGER_R = 2.00
    BO_V12_CONFIRMED_LOCK_FLOOR_R = -0.50
    BO_V12_CONFIRMED_LOCK_TRAIL_GAP_R = 4.00


class BreakoutV12ConfirmedTrail200Gap600FloorM050GridV8Freqtrade(
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_CONFIRMED_LOCK_TRIGGER_R = 2.00
    BO_V12_CONFIRMED_LOCK_FLOOR_R = -0.50
    BO_V12_CONFIRMED_LOCK_TRAIL_GAP_R = 6.00


class BreakoutV12ConfirmedTrail300Gap600Floor000GridV8Freqtrade(
    _BreakoutV12ConfirmedProfitLockMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_CONFIRMED_LOCK_TRIGGER_R = 3.00
    BO_V12_CONFIRMED_LOCK_FLOOR_R = 0.00
    BO_V12_CONFIRMED_LOCK_TRAIL_GAP_R = 6.00


class _BreakoutV12ConfirmedPartialMixin:
    """Bank a small slice while retaining most of a confirmed trend runner."""

    BO_V12_PARTIAL_TRIGGER_R = 2.00
    BO_V12_PARTIAL_FRACTION = 0.10
    BO_V12_PARTIAL_EXCLUDE_CAPTURE = True
    BO_V12_PARTIAL_MIN_LOW_OPPORTUNITY_PERSISTENCE = -1.0

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        super().order_filled(pair, trade, order, current_time, **kwargs)
        component = self._component(getattr(trade, "enter_tag", None))
        tag = str(getattr(order, "ft_order_tag", "") or "")
        if component != "breakout":
            return
        if tag == "bo_v12_confirmed_partial":
            trade.set_custom_data("bo_v12_confirmed_partial_done", True)
            return
        if (
            getattr(order, "ft_order_side", None)
            != getattr(trade, "entry_side", None)
            or int(getattr(trade, "nr_of_successful_entries", 0)) != 1
        ):
            return
        row = self._latest_signal_row(pair, "breakout", current_time)
        persistence = (
            -1.0
            if row is None
            else float(row.get("low_opportunity_fraction_72h", -1.0))
        )
        eligible = (
            not self.BO_V12_PARTIAL_EXCLUDE_CAPTURE
            or not bool(trade.get_custom_data("bo_capture", False))
        ) and (
            persistence
            >= self.BO_V12_PARTIAL_MIN_LOW_OPPORTUNITY_PERSISTENCE
        )
        trade.set_custom_data("bo_v12_confirmed_partial_eligible", eligible)
        trade.set_custom_data("bo_v12_confirmed_partial_done", False)

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> Any:
        adjustment = super().adjust_trade_position(
            trade,
            current_time,
            current_rate,
            current_profit,
            min_stake,
            max_stake,
            current_entry_rate,
            current_exit_rate,
            current_entry_profit,
            current_exit_profit,
            **kwargs,
        )
        if adjustment is not None:
            return adjustment
        if (
            self._component(trade.enter_tag) != "breakout"
            or not bool(
                trade.get_custom_data(
                    "bo_v12_confirmed_partial_eligible", False
                )
            )
            or bool(
                trade.get_custom_data("bo_v12_confirmed_partial_done", False)
            )
            or self._confirmed_peak_r(
                trade.pair,
                trade,
                current_time,
            )
            < self.BO_V12_PARTIAL_TRIGGER_R
        ):
            return None
        return (
            -float(trade.stake_amount) * self.BO_V12_PARTIAL_FRACTION,
            "bo_v12_confirmed_partial",
        )


class BreakoutV12ConfirmedPartialAll005GridV8Freqtrade(
    _BreakoutV12ConfirmedPartialMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_PARTIAL_FRACTION = 0.05
    BO_V12_PARTIAL_EXCLUDE_CAPTURE = False


class BreakoutV12ConfirmedPartialNonCapture010GridV8Freqtrade(
    _BreakoutV12ConfirmedPartialMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_PARTIAL_FRACTION = 0.10
    BO_V12_PARTIAL_EXCLUDE_CAPTURE = True


class BreakoutV12ConfirmedPartialPersistent010GridV8Freqtrade(
    _BreakoutV12ConfirmedPartialMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_PARTIAL_FRACTION = 0.10
    BO_V12_PARTIAL_EXCLUDE_CAPTURE = False
    BO_V12_PARTIAL_MIN_LOW_OPPORTUNITY_PERSISTENCE = 0.75


class BreakoutV12ConfirmedPartialPersistent020GridV8Freqtrade(
    BreakoutV12ConfirmedPartialPersistent010GridV8Freqtrade
):
    BO_V12_PARTIAL_FRACTION = 0.20


class _BreakoutV12BullReentryMixin:
    """Add a capped pullback-and-second-breakout entry in confirmed bull cycles."""

    BO_V12_REENTRY_REQUIRE_RECENT_PRIMARY = True
    BO_V12_REENTRY_MIN_REGIME = 0.20
    BO_V12_REENTRY_MIN_BREADTH = 0.55
    BO_V12_REENTRY_MAX_BREADTH = 0.88
    BO_V12_REENTRY_MIN_SLOW_BREADTH = 0.48
    BO_V12_REENTRY_MAX_SLOW_BREADTH = 0.94
    BO_V12_REENTRY_MIN_BREADTH_CHANGE_24H = -0.05
    BO_V12_REENTRY_MIN_BTC_DAILY_TREND = 0.10
    BO_V12_REENTRY_MIN_ETH_DAILY_TREND = 0.05
    BO_V12_REENTRY_MIN_VOLUME = 0.75
    BO_V12_REENTRY_MAX_VOLUME = 2.80
    BO_V12_REENTRY_MAX_EXTENSION_ATR = 1.00
    BO_V12_REENTRY_MIN_SYMBOL_EFFICIENCY = 0.25
    BO_V12_REENTRY_BASE_RISK = 0.018
    BO_V12_REENTRY_STRONG_RISK = 0.030

    def _populate_breakout(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_breakout(dataframe)
        frozen_long_impulse = dataframe["bo_entry_long"].astype(bool).copy()
        dataframe["bo_v12_long_reentry"] = 0

        atr = dataframe["atr"].clip(lower=1e-12)
        slow_slope = (
            dataframe["slow_ema"] - dataframe["slow_ema"].shift(6)
        ) / atr
        recent_primary = (
            frozen_long_impulse.shift(1)
            .rolling(168, min_periods=1)
            .max()
            .fillna(0.0)
            .astype(bool)
        )
        primary_ready = (
            recent_primary
            if self.BO_V12_REENTRY_REQUIRE_RECENT_PRIMARY
            else pd.Series(True, index=dataframe.index)
        )
        pullback = (
            (dataframe["low"] <= dataframe["fast_ema"] + 0.25 * atr)
            & (dataframe["close"] >= dataframe["slow_ema"] - 0.20 * atr)
            & (dataframe["close"] <= dataframe["fast_ema"] + 0.65 * atr)
        )
        recent_pullback = (
            pullback.shift(1)
            .rolling(5, min_periods=1)
            .max()
            .fillna(0.0)
            .astype(bool)
        )
        prior_high = dataframe["high"].shift(1).rolling(8, min_periods=3).max()
        extension = (dataframe["close"] - dataframe["fast_ema"]) / atr
        trend_restarted = (
            (dataframe["close"] > prior_high)
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["fast_ema"])
            & (dataframe["bo_long_close_position"] >= 0.62)
        )
        regime_ready = (
            (dataframe["market_regime_score"] >= self.BO_V12_REENTRY_MIN_REGIME)
            & dataframe["breadth"].between(
                self.BO_V12_REENTRY_MIN_BREADTH,
                self.BO_V12_REENTRY_MAX_BREADTH,
            )
            & dataframe["breadth_slow"].between(
                self.BO_V12_REENTRY_MIN_SLOW_BREADTH,
                self.BO_V12_REENTRY_MAX_SLOW_BREADTH,
            )
            & (
                dataframe["breadth_change_24h"]
                >= self.BO_V12_REENTRY_MIN_BREADTH_CHANGE_24H
            )
            & (
                dataframe["btc_trend_1d"]
                >= self.BO_V12_REENTRY_MIN_BTC_DAILY_TREND
            )
            & (
                dataframe["eth_trend_1d"]
                >= self.BO_V12_REENTRY_MIN_ETH_DAILY_TREND
            )
        )
        pair_ready = (
            (dataframe["fast_ema"] > dataframe["slow_ema"])
            & (slow_slope >= 0.02)
            & dataframe["symbol_ema55_atr"].between(0.10, 4.0)
            & (
                dataframe["symbol_efficiency_12h"]
                >= self.BO_V12_REENTRY_MIN_SYMBOL_EFFICIENCY
            )
            & extension.between(0.0, self.BO_V12_REENTRY_MAX_EXTENSION_ATR)
            & dataframe["bo_volume_ratio"].between(
                self.BO_V12_REENTRY_MIN_VOLUME,
                self.BO_V12_REENTRY_MAX_VOLUME,
            )
            & (dataframe["atr"] / dataframe["close"]).between(0.003, 0.10)
        )
        raw_reentry = (
            primary_ready
            & recent_pullback
            & trend_restarted
            & regime_ready
            & pair_ready
            & ~dataframe["bo_entry"].astype(bool)
        )
        reentry = _causal_signal_cap(
            raw_reentry,
            dataframe["date"],
            daily_limit=1,
            minimum_bar_gap=12,
        ).astype(bool)
        reentry_score = (
            3
            + (dataframe["bo_volume_ratio"] >= 1.10).astype(int)
            + (dataframe["market_regime_score"] >= 0.45).astype(int)
        ).clip(upper=5)
        reentry_risk = pd.Series(
            np.where(
                dataframe["market_regime_score"] >= 0.45,
                self.BO_V12_REENTRY_STRONG_RISK,
                self.BO_V12_REENTRY_BASE_RISK,
            ),
            index=dataframe.index,
        )
        dataframe.loc[reentry, "bo_score"] = reentry_score.loc[reentry]
        dataframe.loc[reentry, "bo_regime"] = dataframe.loc[
            reentry, "bo_regime_long"
        ]
        dataframe.loc[reentry, "bo_risk_pct"] = np.minimum(
            reentry_risk.loc[reentry],
            self.BO_MAX_RISK,
        )
        dataframe.loc[reentry, "bo_capture"] = 0
        dataframe.loc[reentry, "bo_long_floor"] = 1
        dataframe.loc[reentry, "bo_entry_long"] = 1
        dataframe.loc[reentry, "bo_entry_short"] = 0
        dataframe.loc[reentry, "bo_entry"] = 1
        dataframe.loc[reentry, "bo_v12_long_reentry"] = 1
        dataframe.loc[reentry, "bo_v10_rank"] = (
            dataframe.loc[reentry, "bo_range_atr"]
            - 0.10 * dataframe.loc[reentry, "bo_score"].astype(float)
            - 0.08 * dataframe.loc[reentry, "market_regime_score"]
            - 0.03 * dataframe.loc[reentry, "bo_long_quality"]
        )
        return dataframe


class BreakoutV12BullReentryBaseGridV8Freqtrade(
    _BreakoutV12BullReentryMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    pass


class BreakoutV12BullReentryStrongGridV8Freqtrade(
    _BreakoutV12BullReentryMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    BO_V12_REENTRY_MIN_REGIME = 0.35
    BO_V12_REENTRY_MIN_BREADTH = 0.60
    BO_V12_REENTRY_MIN_SLOW_BREADTH = 0.55
    BO_V12_REENTRY_MIN_BTC_DAILY_TREND = 0.15
    BO_V12_REENTRY_MIN_ETH_DAILY_TREND = 0.10
    BO_V12_REENTRY_MAX_EXTENSION_ATR = 0.85


class BreakoutV12BullReentryQualityGridV8Freqtrade(
    BreakoutV12BullReentryStrongGridV8Freqtrade
):
    BO_V12_REENTRY_MIN_SYMBOL_EFFICIENCY = 0.35
    BO_V12_REENTRY_MAX_VOLUME = 2.10


class BreakoutV12BullReentryStrongLowRiskGridV8Freqtrade(
    BreakoutV12BullReentryStrongGridV8Freqtrade
):
    BO_V12_REENTRY_BASE_RISK = 0.010
    BO_V12_REENTRY_STRONG_RISK = 0.018


class BreakoutV12BullPullbackExpansionLowRiskGridV8Freqtrade(
    _BreakoutV12BullReentryMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    """Independent bull pullback entry; does not require a prior BO signal."""

    BO_V12_REENTRY_REQUIRE_RECENT_PRIMARY = False
    BO_V12_REENTRY_MIN_REGIME = 0.30
    BO_V12_REENTRY_MIN_BREADTH = 0.60
    BO_V12_REENTRY_MIN_SLOW_BREADTH = 0.55
    BO_V12_REENTRY_MIN_BREADTH_CHANGE_24H = -0.02
    BO_V12_REENTRY_MIN_BTC_DAILY_TREND = 0.10
    BO_V12_REENTRY_MIN_ETH_DAILY_TREND = 0.05
    BO_V12_REENTRY_MIN_SYMBOL_EFFICIENCY = 0.30
    BO_V12_REENTRY_MAX_EXTENSION_ATR = 0.85
    BO_V12_REENTRY_MAX_VOLUME = 2.20
    BO_V12_REENTRY_BASE_RISK = 0.010
    BO_V12_REENTRY_STRONG_RISK = 0.018


class BreakoutV12BullPullbackExpansionBalancedGridV8Freqtrade(
    BreakoutV12BullPullbackExpansionLowRiskGridV8Freqtrade
):
    BO_V12_REENTRY_BASE_RISK = 0.015
    BO_V12_REENTRY_STRONG_RISK = 0.025


class _GridV9BullPullbackMixin:
    """Replace the symmetric Grid long with a bull-cycle trend pullback."""

    GRID_V9_LONG_MIN_REGIME = 0.12
    GRID_V9_LONG_MIN_BREADTH = 0.53
    GRID_V9_LONG_MAX_BREADTH = 0.91
    GRID_V9_LONG_MIN_SLOW_BREADTH = 0.46
    GRID_V9_LONG_MIN_MARKET_EFFICIENCY = 0.15
    GRID_V9_LONG_MIN_SYMBOL_EFFICIENCY = 0.30
    GRID_V9_LONG_MIN_BTC_DAILY_TREND = -0.10
    GRID_V9_LONG_MIN_ETH_DAILY_TREND = -0.20
    GRID_V9_REPLACE_INHERITED_LONG = True

    def _populate_grid(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_grid(dataframe)
        inherited_long = dataframe["grid_entry_long"].astype(bool).copy()
        dataframe["grid_v9_long_pullback"] = 0
        atr = dataframe["atr"].clip(lower=1e-12)
        long_alignment = (
            dataframe["fast_ema"] - dataframe["slow_ema"]
        ) / atr
        long_extension = (
            dataframe["close"] - dataframe["fast_ema"]
        ) / atr
        pullback = (
            (dataframe["low"] <= dataframe["fast_ema"] + 0.20 * atr)
            & (dataframe["low"] >= dataframe["slow_ema"] - 0.65 * atr)
            & (dataframe["close"] >= dataframe["slow_ema"])
            & (dataframe["close"] <= dataframe["fast_ema"] + 0.55 * atr)
        )
        recent_pullback = (
            pullback.shift(1)
            .rolling(4, min_periods=1)
            .max()
            .fillna(0.0)
            .astype(bool)
        )
        restart = (
            (dataframe["close"] > dataframe["fast_ema"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["high"].shift(1))
            & (dataframe["grid_long_close_position"] >= 0.64)
        )
        trend = (
            (dataframe["fast_ema"] > dataframe["slow_ema"])
            & (dataframe["grid_slow_slope"] >= 0.04)
            & (dataframe["grid_fast_slope"] >= -0.10)
            & long_alignment.between(0.20, 3.50)
            & long_extension.between(0.0, 0.90)
        )
        regime = (
            (dataframe["market_regime_score"] >= self.GRID_V9_LONG_MIN_REGIME)
            & dataframe["breadth"].between(
                self.GRID_V9_LONG_MIN_BREADTH,
                self.GRID_V9_LONG_MAX_BREADTH,
            )
            & (
                dataframe["breadth_slow"]
                >= self.GRID_V9_LONG_MIN_SLOW_BREADTH
            )
            & (
                dataframe["market_efficiency"]
                >= self.GRID_V9_LONG_MIN_MARKET_EFFICIENCY
            )
            & (
                dataframe["btc_trend_1d"]
                > self.GRID_V9_LONG_MIN_BTC_DAILY_TREND
            )
            & (
                dataframe["eth_trend_1d"]
                > self.GRID_V9_LONG_MIN_ETH_DAILY_TREND
            )
        )
        quality = (
            dataframe["grid_volume_ratio"].between(0.65, 2.10)
            & dataframe["symbol_return_4h"].between(-0.02, 0.08)
            & (
                dataframe["symbol_efficiency_12h"]
                >= self.GRID_V9_LONG_MIN_SYMBOL_EFFICIENCY
            )
            & (dataframe["atr"] / dataframe["close"]).between(0.003, 0.08)
        )
        raw = recent_pullback & restart & trend & regime & quality
        eligible = _causal_signal_cap(
            raw,
            dataframe["date"],
            daily_limit=2,
            minimum_bar_gap=12,
        ).astype(bool)
        score = (
            3
            + (dataframe["grid_long_close_position"] >= 0.72).astype(int)
            + (dataframe["grid_volume_ratio"] >= 0.90).astype(int)
            + (dataframe["market_regime_score"] >= 0.35).astype(int)
        ).clip(upper=6)
        selected_long = (
            eligible
            if self.GRID_V9_REPLACE_INHERITED_LONG
            else (inherited_long | eligible)
        )
        dataframe["grid_entry_long"] = selected_long.astype(int)
        dataframe.loc[eligible, "grid_long_score"] = score.loc[eligible]
        dataframe.loc[eligible, "grid_long_quality"] = (
            0.45
            + 0.15 * dataframe.loc[eligible, "market_bull_strength"]
            + 0.10
            * dataframe.loc[eligible, "grid_long_close_position"]
        )
        dataframe.loc[eligible, "grid_v9_long_pullback"] = 1
        dataframe.loc[eligible, "grid_rank"] = (
            long_extension.loc[eligible]
            - 0.08 * dataframe.loc[eligible, "market_regime_score"]
            - 0.03 * score.loc[eligible].astype(float)
        )
        dataframe["grid_entry"] = (
            dataframe["grid_entry_long"].astype(bool)
            | dataframe["grid_entry_short"].astype(bool)
        ).astype(int)
        return dataframe


class BreakoutV12GridV9BullPullbackBaseFreqtrade(
    _GridV9BullPullbackMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    pass


class BreakoutV12GridV9BullPullbackStrongFreqtrade(
    _GridV9BullPullbackMixin,
    BreakoutV12ExactParitySidecarFreqtrade,
):
    GRID_V9_LONG_MIN_REGIME = 0.25
    GRID_V9_LONG_MIN_BREADTH = 0.60
    GRID_V9_LONG_MAX_BREADTH = 0.88
    GRID_V9_LONG_MIN_SLOW_BREADTH = 0.55
    GRID_V9_LONG_MIN_SYMBOL_EFFICIENCY = 0.35
    GRID_V9_LONG_MIN_BTC_DAILY_TREND = 0.05
    GRID_V9_LONG_MIN_ETH_DAILY_TREND = 0.00


class BreakoutV12GridV9BullPullbackLowRiskFreqtrade(
    BreakoutV12GridV9BullPullbackBaseFreqtrade
):
    GRID_LONG_RISK_PCT = 0.040


class BreakoutV12GridV9BullPullbackStrongLowRiskFreqtrade(
    BreakoutV12GridV9BullPullbackStrongFreqtrade
):
    GRID_LONG_RISK_PCT = 0.040


class BreakoutV12GridV9BullPullbackExtendFreqtrade(
    BreakoutV12GridV9BullPullbackBaseFreqtrade
):
    GRID_V9_REPLACE_INHERITED_LONG = False


class BreakoutV12GridV9BullPullbackStrongExtendFreqtrade(
    BreakoutV12GridV9BullPullbackStrongFreqtrade
):
    GRID_V9_REPLACE_INHERITED_LONG = False

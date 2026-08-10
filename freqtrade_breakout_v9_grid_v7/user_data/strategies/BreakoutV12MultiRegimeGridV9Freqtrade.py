from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import Trade, stoploss_from_absolute

from BreakoutV9GridV7Freqtrade import (
    _causal_signal_cap,
    _efficiency_ratio,
    _ema,
)
from BreakoutV10FGridV8DualSideFreqtrade import (
    LIVE_CONTEXT_COLUMNS,
    build_synchronized_market_context,
)
from BreakoutV11AdaptiveGridV8DualSideFreqtrade import (
    BreakoutV11AdaptiveGridV8DualSideFreqtrade,
)


logger = logging.getLogger(__name__)


V12_CONTEXT_COLUMNS = (
    *LIVE_CONTEXT_COLUMNS,
    "breadth_slow",
    "breadth_change_24h",
    "btc_return_24h",
    "eth_return_24h",
    "btc_trend_4h",
    "eth_trend_4h",
    "btc_trend_1d",
    "eth_trend_1d",
    "market_dispersion_24h",
    "market_dispersion_7d",
    "low_opportunity_fraction_72h",
    "market_median_return_24h",
    "breadth_travel_24h",
    "market_regime_score",
    "market_bull_strength",
    "market_bear_strength",
)


def _empty_context() -> DataFrame:
    return DataFrame(columns=("date", *V12_CONTEXT_COLUMNS))


def _normalized_trends(close: Series) -> tuple[Series, Series]:
    """Return causal 4h-structure and daily-cycle trend scores.

    The source data remains the strategy's completed 1h candles.  The first
    score compares one-day and four-day EMAs (the stable part of a 4h trend),
    while the second compares price with a 20-day EMA and includes its
    one-day slope.  Both are continuous rather than binary regime switches.
    """

    ema_24 = _ema(close, 24).clip(lower=1e-12)
    ema_96 = _ema(close, 96).clip(lower=1e-12)
    ema_480 = _ema(close, 480).clip(lower=1e-12)
    trend_4h = ((ema_24 / ema_96 - 1.0) / 0.035).clip(-1.5, 1.5)
    daily_distance = ((close / ema_480 - 1.0) / 0.18).clip(-1.5, 1.5)
    daily_slope = ((ema_480 / ema_480.shift(24) - 1.0) / 0.018).clip(
        -1.5, 1.5
    )
    trend_1d = (0.72 * daily_distance + 0.28 * daily_slope).clip(-1.5, 1.5)
    return trend_4h, trend_1d


def build_v12_market_context(
    snapshots: dict[str, DataFrame],
    pairs: tuple[str, ...],
    *,
    require_aligned_latest: bool,
) -> tuple[DataFrame, pd.Timestamp | None, tuple[str, ...]]:
    """Build one causal BTC/ETH + Active50 regime context.

    In LIVE/DRY-RUN every active pair must have the same newest closed bar.
    Backtesting permits contracts which were not listed for the full window;
    the cross-sectional values still require at least half of Active50.
    """

    if not pairs:
        return _empty_context(), None, ()

    normalized: dict[str, DataFrame] = {}
    missing: list[str] = []
    latest_by_pair: dict[str, pd.Timestamp] = {}
    for pair in pairs:
        frame = snapshots.get(pair)
        if frame is None or frame.empty or not {"date", "close"} <= set(
            frame.columns
        ):
            missing.append(pair)
            continue
        local = frame[["date", "close"]].copy()
        local["date"] = pd.to_datetime(local["date"], utc=True)
        local["close"] = pd.to_numeric(local["close"], errors="coerce")
        local = (
            local.dropna(subset=["date", "close"])
            .drop_duplicates("date", keep="last")
            .sort_values("date")
        )
        if local.empty:
            missing.append(pair)
            continue
        normalized[pair] = local
        latest_by_pair[pair] = pd.Timestamp(local["date"].iloc[-1])

    if not normalized:
        return _empty_context(), None, tuple(sorted(set(missing)))

    target = max(latest_by_pair.values())
    if require_aligned_latest:
        stale = [
            pair
            for pair in pairs
            if latest_by_pair.get(pair) != target
        ]
        unavailable = tuple(sorted(set(missing + stale)))
        if unavailable:
            return _empty_context(), target, unavailable

    above_fast: dict[str, Series] = {}
    above_slow: dict[str, Series] = {}
    efficiencies: dict[str, Series] = {}
    returns_4h: dict[str, Series] = {}
    returns_24h: dict[str, Series] = {}
    trend_4h: dict[str, Series] = {}
    trend_1d: dict[str, Series] = {}
    for pair, local_frame in normalized.items():
        local = local_frame.loc[local_frame["date"] <= target].set_index(
            "date"
        )
        close = local["close"].astype(float)
        symbol = BreakoutV11AdaptiveGridV8DualSideFreqtrade._symbol(pair)
        above_fast[symbol] = (close > _ema(close, 21)).astype(float)
        above_slow[symbol] = (close > _ema(close, 96)).astype(float)
        efficiencies[symbol] = _efficiency_ratio(close, 12).abs()
        returns_4h[symbol] = close / close.shift(4) - 1.0
        returns_24h[symbol] = close / close.shift(24) - 1.0
        local_4h, local_1d = _normalized_trends(close)
        trend_4h[symbol] = local_4h
        trend_1d[symbol] = local_1d

    above_fast_frame = pd.concat(above_fast, axis=1).sort_index()
    above_slow_frame = pd.concat(above_slow, axis=1).reindex(
        above_fast_frame.index
    )
    efficiency_frame = pd.concat(efficiencies, axis=1).reindex(
        above_fast_frame.index
    )
    return_24h_frame = pd.concat(returns_24h, axis=1).reindex(
        above_fast_frame.index
    )
    minimum_count = max(5, len(pairs) // 2)
    breadth = above_fast_frame.mean(axis=1).where(
        above_fast_frame.count(axis=1) >= minimum_count
    )
    breadth_slow = above_slow_frame.mean(axis=1).where(
        above_slow_frame.count(axis=1) >= minimum_count
    )
    market_efficiency = efficiency_frame.mean(axis=1).where(
        efficiency_frame.count(axis=1) >= minimum_count
    )
    market_dispersion_24h = return_24h_frame.std(axis=1).where(
        return_24h_frame.count(axis=1) >= minimum_count
    )
    market_median_return_24h = return_24h_frame.median(axis=1).where(
        return_24h_frame.count(axis=1) >= minimum_count
    )
    market_dispersion_7d = market_dispersion_24h.ewm(
        span=168,
        adjust=False,
        min_periods=72,
    ).mean()
    low_opportunity_fraction_72h = (
        (market_dispersion_7d <= 0.030)
        .astype(float)
        .rolling(72, min_periods=48)
        .mean()
    )

    def series_for(values: dict[str, Series], symbol: str) -> Series:
        return values.get(symbol, Series(dtype=float)).reindex(
            above_fast_frame.index
        )

    btc_return_4h = series_for(returns_4h, "BTCUSDT")
    eth_return_4h = series_for(returns_4h, "ETHUSDT")
    btc_return_24h = series_for(returns_24h, "BTCUSDT")
    eth_return_24h = series_for(returns_24h, "ETHUSDT")
    btc_trend_4h = series_for(trend_4h, "BTCUSDT")
    eth_trend_4h = series_for(trend_4h, "ETHUSDT")
    btc_trend_1d = series_for(trend_1d, "BTCUSDT")
    eth_trend_1d = series_for(trend_1d, "ETHUSDT")
    breadth_change_24h = breadth - breadth.shift(24)
    breadth_travel_24h = (
        breadth.diff().abs().rolling(24, min_periods=12).sum()
    )

    raw_regime = (
        0.22 * ((breadth - 0.50) / 0.25).clip(-1.5, 1.5)
        + 0.12 * ((breadth_slow - 0.50) / 0.25).clip(-1.5, 1.5)
        + 0.08 * (breadth_change_24h / 0.15).clip(-1.5, 1.5)
        + 0.15 * btc_trend_4h.clip(-1.5, 1.5)
        + 0.11 * eth_trend_4h.clip(-1.5, 1.5)
        + 0.17 * btc_trend_1d.clip(-1.5, 1.5)
        + 0.15 * eth_trend_1d.clip(-1.5, 1.5)
    )
    regime_score = raw_regime.ewm(span=6, adjust=False).mean().clip(-1.0, 1.0)
    bull_strength = ((regime_score - 0.10) / 0.60).clip(0.0, 1.0)
    bear_strength = ((-regime_score - 0.10) / 0.60).clip(0.0, 1.0)

    context = DataFrame(
        {
            "date": above_fast_frame.index,
            "breadth": breadth,
            "breadth_change_4h": breadth - breadth.shift(4),
            "market_efficiency": market_efficiency,
            "btc_return_4h": btc_return_4h,
            "eth_return_4h": eth_return_4h,
            "breadth_slow": breadth_slow,
            "breadth_change_24h": breadth_change_24h,
            "btc_return_24h": btc_return_24h,
            "eth_return_24h": eth_return_24h,
            "btc_trend_4h": btc_trend_4h,
            "eth_trend_4h": eth_trend_4h,
            "btc_trend_1d": btc_trend_1d,
            "eth_trend_1d": eth_trend_1d,
            "market_dispersion_24h": market_dispersion_24h,
            "market_dispersion_7d": market_dispersion_7d,
            "low_opportunity_fraction_72h": (
                low_opportunity_fraction_72h
            ),
            "market_median_return_24h": market_median_return_24h,
            "breadth_travel_24h": breadth_travel_24h,
            "market_regime_score": regime_score,
            "market_bull_strength": bull_strength,
            "market_bear_strength": bear_strength,
        }
    ).reset_index(drop=True)
    if require_aligned_latest:
        latest = context.loc[context["date"] == target]
        if latest.empty or latest[list(V12_CONTEXT_COLUMNS)].isna().any(
            axis=None
        ):
            return _empty_context(), target, pairs
    return context, target, ()


class BreakoutV12MultiRegimeGridV9Freqtrade(
    BreakoutV11AdaptiveGridV8DualSideFreqtrade
):
    """Independent multi-regime successor to frozen v11/Grid-v8.

    The existing execution and campaign logic is inherited unchanged.  New
    behavior is split into independently testable layers: directional regime
    conflict filters, side-aware allocation, a true Grid long pullback, and a
    Breakout long secondary-expansion entry.
    """

    ENABLE_REGIME_FILTERS = True
    ENABLE_REGIME_RISK = True
    ENABLE_SLEEVE_PERFORMANCE_RISK = True
    ENABLE_GRID_V9_LONG = True
    ENABLE_BO_V12_LONG_REENTRY = True
    GRID_V9_REPLACE_LONG = True
    HARD_FILTER_BULL_SHORTS = False
    HARD_FILTER_EXHAUSTED_SHORTS = False
    ENABLE_GRID_SHORT_REBOUND_GUARD = False
    ENABLE_GRID_SHORT_LOW_DISPERSION_GUARD = False
    ENABLE_BO_SHORT_NARROW_RANGE_GUARD = False
    ENABLE_BO_SHORT_BREADTH_CHURN_GUARD = False
    ENABLE_BO_LONG_OVERHEAT_GUARD = False
    ENABLE_TARGETED_SETUP_RISK = False
    GUARDS_ONLY_LOW_OPPORTUNITY = False
    TARGETED_RISK_ONLY_LOW_OPPORTUNITY = False
    GUARDS_ONLY_PERSISTENT_LOW_OPPORTUNITY = False
    ENABLE_LOW_OPPORTUNITY_DRAWDOWN_GOVERNOR = False
    ENABLE_LOW_OPPORTUNITY_RECENT_GOVERNOR = False
    ENABLE_LOW_OPPORTUNITY_BASE_SCALE = False
    ENABLE_PERSISTENT_DEFENSIVE_STOPS = False
    ENABLE_PERSISTENT_GRID_ONE_TP_FLOOR = False

    BULL_CONFLICT_SCORE = 0.30
    BEAR_CONFLICT_SCORE = -0.38
    BULL_MIN_FAST_BREADTH = 0.62
    BULL_MIN_SLOW_BREADTH = 0.54
    SHORT_EXHAUSTION_MAX_REGIME = -0.38
    SHORT_EXHAUSTION_MAX_BREADTH = 0.20
    SHORT_EXHAUSTION_MAX_SLOW_BREADTH = 0.24

    BO_REENTRY_MIN_REGIME = 0.20
    BO_REENTRY_MIN_BREADTH = 0.55
    BO_REENTRY_MAX_BREADTH = 0.88
    BO_REENTRY_MIN_SLOW_BREADTH = 0.48
    BO_REENTRY_MAX_SLOW_BREADTH = 0.94
    BO_REENTRY_MIN_BREADTH_CHANGE_24H = -0.05
    BO_REENTRY_MIN_BTC_DAILY_TREND = 0.10
    BO_REENTRY_MIN_ETH_DAILY_TREND = 0.05
    BO_REENTRY_MIN_VOLUME = 0.75
    BO_REENTRY_MAX_VOLUME = 2.80
    BO_REENTRY_MAX_EXTENSION_ATR = 1.00
    BO_REENTRY_MIN_SYMBOL_EFFICIENCY = 0.25
    BO_REENTRY_BASE_RISK = 0.018
    BO_REENTRY_STRONG_RISK = 0.030

    GRID_V9_LONG_MIN_REGIME = 0.12
    GRID_V9_LONG_MIN_BREADTH = 0.53
    GRID_V9_LONG_MAX_BREADTH = 0.91
    GRID_V9_LONG_MIN_SLOW_BREADTH = 0.46
    GRID_V9_LONG_MIN_MARKET_EFFICIENCY = 0.15
    GRID_V9_LONG_MIN_SYMBOL_EFFICIENCY = 0.30
    GRID_V9_LONG_MIN_BTC_DAILY_TREND = -0.10
    GRID_V9_LONG_MIN_ETH_DAILY_TREND = -0.20
    GRID_V9_LONG_RISK_PCT = 0.060

    GRID_SHORT_MAX_REBOUND_4H = 0.0075
    GRID_SHORT_MIN_DISPERSION_24H = 0.0190
    BO_SHORT_MIN_RANGE_ATR = 4.50
    BO_SHORT_MAX_BREADTH_TRAVEL_24H = 3.12
    BO_LONG_MAX_ETH_TREND_4H = 0.48
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.035
    LOW_OPPORTUNITY_MIN_PERSISTENCE_72H = 0.90
    TARGET_GRID_REBOUND_SCALE = 1.0
    TARGET_GRID_LOW_DISPERSION_SCALE = 1.0
    TARGET_BO_SHORT_RANGE_SCALE = 1.0
    TARGET_BO_SHORT_CHURN_SCALE = 1.0
    TARGET_BO_LONG_OVERHEAT_SCALE = 1.0
    LOW_OPPORTUNITY_DRAWDOWN_TRIGGER = 0.10
    LOW_OPPORTUNITY_DRAWDOWN_SCALE = 0.50
    LOW_OPPORTUNITY_SEVERE_DRAWDOWN = 1.0
    LOW_OPPORTUNITY_SEVERE_SCALE = 0.50
    LOW_OPPORTUNITY_DRAWDOWN_SHORT_ONLY = False
    LOW_OPPORTUNITY_RECENT_WINDOW = 4
    LOW_OPPORTUNITY_RECENT_MIN_OBSERVATIONS = 3
    LOW_OPPORTUNITY_RECENT_CAUTION_SCALE = 0.75
    LOW_OPPORTUNITY_RECENT_DEFENSIVE_SCALE = 0.50
    LOW_OPPORTUNITY_RECENT_SHORT_ONLY = False
    LOW_OPPORTUNITY_RECENT_COMPONENT_SPECIFIC = False
    LOW_OPPORTUNITY_BASE_SCALE = 0.50
    DEFENSIVE_MIN_PERSISTENCE_72H = 0.90
    DEFENSIVE_BO_STOP_ATR = 0.60
    DEFENSIVE_GRID_STOP_ATR = 1.80
    DEFENSIVE_REQUIRE_PERSISTENCE = True
    DEFENSIVE_MAX_DISPERSION_7D: float | None = None
    DEFENSIVE_GRID_SHORT_REBOUND = True
    DEFENSIVE_GRID_ALL_SHORTS = False
    DEFENSIVE_BO_SHORT_RANGE = True
    DEFENSIVE_BO_SHORT_CHURN = True
    DEFENSIVE_BO_LONG_OVERHEAT = True
    GRID_V12_ONE_TP_ACTIVATION_R = 0.10
    GRID_V12_ONE_TP_FLOOR_R = 0.02

    LONG_STRONG_BULL_SCALE = 1.12
    LONG_BEAR_SCALE = 0.35
    SHORT_STRONG_BEAR_SCALE = 1.00
    SHORT_BULL_SCALE = 0.20
    NEUTRAL_SCALE = 1.00
    SLEEVE_RECENT_WINDOW = 4
    SLEEVE_MIN_OBSERVATIONS = 3
    GRID_SLEEVE_DEFENSIVE_SCALE = 0.12
    GRID_SLEEVE_CAUTION_SCALE = 0.45
    BO_SLEEVE_DEFENSIVE_SCALE = 0.65
    BO_SLEEVE_CAUTION_SCALE = 0.85

    def _build_market_context(self) -> DataFrame:
        # Preserve the frozen v11/v8 context byte-for-byte.  Even tiny changes
        # in one of these five columns can alter cross-pair ranking under
        # Max2.  New regime columns are therefore attached as a sidecar only.
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
        extra_columns = tuple(
            column
            for column in V12_CONTEXT_COLUMNS
            if column not in LIVE_CONTEXT_COLUMNS
        )
        if frozen_context.empty or expanded.empty:
            output = frozen_context.copy()
            for column in extra_columns:
                output[column] = np.nan
            return output
        return frozen_context.merge(
            expanded[["date", *extra_columns]],
            on="date",
            how="left",
            validate="1:1",
        )

    def _prepare_live_context(self) -> pd.Timestamp | None:
        pairs = tuple(self.dp.current_whitelist())
        frozen_context, frozen_target, frozen_unavailable = (
            build_synchronized_market_context(
                self._live_pair_snapshots,
                pairs,
            )
        )
        expanded, target, unavailable = build_v12_market_context(
            self._live_pair_snapshots,
            pairs,
            require_aligned_latest=True,
        )
        unavailable = tuple(
            sorted(set(unavailable + frozen_unavailable))
        )
        if frozen_target != target:
            unavailable = pairs
        if unavailable:
            signature = (target, unavailable)
            if signature != self._live_context_wait_signature:
                logger.info(
                    "v12 LIVE context waiting: target=%s ready=%d/%d "
                    "missing_or_stale=%s",
                    target,
                    len(pairs) - len(unavailable),
                    len(pairs),
                    ",".join(unavailable),
                )
                self._live_context_wait_signature = signature
            return None
        if target is None or frozen_context.empty or expanded.empty:
            return None
        extra_columns = tuple(
            column
            for column in V12_CONTEXT_COLUMNS
            if column not in LIVE_CONTEXT_COLUMNS
        )
        context = frozen_context.merge(
            expanded[["date", *extra_columns]],
            on="date",
            how="left",
            validate="1:1",
        )
        self._market_context_cache = context
        self._market_context_end = target
        self._live_context_pending_at = target
        self._live_context_wait_signature = None
        return target

    def _attach_market_context(self, dataframe: DataFrame) -> DataFrame:
        if not self._live_context_mode():
            return super()._attach_market_context(dataframe)
        context = getattr(self, "_market_context_cache", None)
        if context is None or context.empty:
            output = dataframe.copy()
            for column in V12_CONTEXT_COLUMNS:
                output[column] = np.nan
            return output
        return dataframe.merge(context, on="date", how="left", validate="m:1")

    def _latest_live_context_failures(
        self,
        pairs: tuple[str, ...],
        target: pd.Timestamp,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        for pair in pairs:
            frame, _updated = self.dp.get_analyzed_dataframe(
                pair,
                self.timeframe,
            )
            if frame is None or frame.empty:
                failures.append(pair)
                continue
            dates = pd.to_datetime(frame["date"], utc=True)
            row = frame.loc[dates == target]
            if (
                row.empty
                or not set(V12_CONTEXT_COLUMNS) <= set(row.columns)
                or row[list(V12_CONTEXT_COLUMNS)].isna().any(axis=None)
            ):
                failures.append(pair)
        return tuple(failures)

    def _bull_conflict(self, dataframe: DataFrame) -> Series:
        return (
            (dataframe["market_regime_score"] >= self.BULL_CONFLICT_SCORE)
            & (dataframe["breadth"] >= self.BULL_MIN_FAST_BREADTH)
            & (dataframe["breadth_slow"] >= self.BULL_MIN_SLOW_BREADTH)
            & (dataframe["breadth_change_24h"] >= -0.06)
            & (dataframe["btc_trend_4h"] > 0.0)
            & (dataframe["eth_trend_4h"] > -0.15)
        )

    def _bear_conflict(self, dataframe: DataFrame) -> Series:
        return (
            (dataframe["market_regime_score"] <= self.BEAR_CONFLICT_SCORE)
            & (dataframe["breadth"] <= 0.38)
            & (dataframe["breadth_slow"] <= 0.44)
            & (dataframe["breadth_change_24h"] <= 0.06)
        )

    def _short_exhaustion(self, dataframe: DataFrame) -> Series:
        """Flag mature selloffs where fresh shorts have convex rebound risk."""

        return (
            (
                dataframe["market_regime_score"]
                <= self.SHORT_EXHAUSTION_MAX_REGIME
            )
            & (
                dataframe["breadth"]
                <= self.SHORT_EXHAUSTION_MAX_BREADTH
            )
            & (
                dataframe["breadth_slow"]
                <= self.SHORT_EXHAUSTION_MAX_SLOW_BREADTH
            )
            & (dataframe["btc_trend_1d"] < 0.05)
            & (dataframe["eth_trend_1d"] < 0.00)
        )

    def _guard_regime(self, dataframe: DataFrame) -> Series:
        if not (
            self.GUARDS_ONLY_LOW_OPPORTUNITY
            or self.GUARDS_ONLY_PERSISTENT_LOW_OPPORTUNITY
        ):
            return Series(True, index=dataframe.index)
        allowed = Series(True, index=dataframe.index)
        if self.GUARDS_ONLY_LOW_OPPORTUNITY:
            allowed &= (
                dataframe["market_dispersion_7d"]
                <= self.LOW_OPPORTUNITY_MAX_DISPERSION_7D
            ).fillna(False)
        if self.GUARDS_ONLY_PERSISTENT_LOW_OPPORTUNITY:
            allowed &= (
                dataframe["low_opportunity_fraction_72h"]
                >= self.LOW_OPPORTUNITY_MIN_PERSISTENCE_72H
            ).fillna(False)
        return allowed

    def _populate_breakout(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_breakout(dataframe)
        frozen_long_impulse = dataframe["bo_entry_long"].astype(bool).copy()
        dataframe["bo_v12_long_reentry"] = 0
        dataframe["bo_v12_regime_rejected"] = 0
        dataframe["bo_v12_quality_rejected"] = 0
        dataframe["bo_v12_defensive_stop"] = 0
        guard_regime = self._guard_regime(dataframe)

        if self.ENABLE_PERSISTENT_DEFENSIVE_STOPS:
            defensive_regime = Series(True, index=dataframe.index)
            if self.DEFENSIVE_REQUIRE_PERSISTENCE:
                defensive_regime &= (
                    dataframe["low_opportunity_fraction_72h"]
                    >= self.DEFENSIVE_MIN_PERSISTENCE_72H
                ).fillna(False)
            if self.DEFENSIVE_MAX_DISPERSION_7D is not None:
                defensive_regime &= (
                    dataframe["market_dispersion_7d"]
                    <= self.DEFENSIVE_MAX_DISPERSION_7D
                ).fillna(False)
            short_conflict = Series(False, index=dataframe.index)
            if self.DEFENSIVE_BO_SHORT_RANGE:
                short_conflict |= (
                    dataframe["bo_range_atr"] <= self.BO_SHORT_MIN_RANGE_ATR
                )
            if self.DEFENSIVE_BO_SHORT_CHURN:
                short_conflict |= (
                    dataframe["breadth_travel_24h"]
                    >= self.BO_SHORT_MAX_BREADTH_TRAVEL_24H
                )
            defensive_breakout = defensive_regime & (
                (
                    dataframe["bo_entry_short"].astype(bool)
                    & short_conflict
                )
                | (
                    dataframe["bo_entry_long"].astype(bool)
                    & self.DEFENSIVE_BO_LONG_OVERHEAT
                    & (
                        dataframe["eth_trend_4h"]
                        >= self.BO_LONG_MAX_ETH_TREND_4H
                    )
                )
            )
            dataframe.loc[
                defensive_breakout, "bo_v12_defensive_stop"
            ] = 1

        if self.ENABLE_REGIME_FILTERS:
            bull_conflict = self._bull_conflict(dataframe)
            bear_conflict = self._bear_conflict(dataframe)
            short_conflict = Series(False, index=dataframe.index)
            if self.HARD_FILTER_BULL_SHORTS:
                short_conflict |= bull_conflict
            if self.HARD_FILTER_EXHAUSTED_SHORTS:
                short_conflict |= self._short_exhaustion(dataframe)
            rejected = (
                dataframe["bo_entry_short"].astype(bool)
                & short_conflict
            ) | (
                dataframe["bo_entry_long"].astype(bool) & bear_conflict
            )
            dataframe.loc[rejected, "bo_v12_regime_rejected"] = 1
            dataframe.loc[rejected, "bo_entry_long"] = 0
            dataframe.loc[rejected, "bo_entry_short"] = 0
            dataframe.loc[rejected, "bo_entry"] = 0

        quality_rejected = Series(False, index=dataframe.index)
        if self.ENABLE_BO_SHORT_NARROW_RANGE_GUARD:
            quality_rejected |= (
                dataframe["bo_entry_short"].astype(bool)
                & guard_regime
                & (dataframe["bo_range_atr"] <= self.BO_SHORT_MIN_RANGE_ATR)
            )
        if self.ENABLE_BO_SHORT_BREADTH_CHURN_GUARD:
            quality_rejected |= (
                dataframe["bo_entry_short"].astype(bool)
                & guard_regime
                & (
                    dataframe["breadth_travel_24h"]
                    >= self.BO_SHORT_MAX_BREADTH_TRAVEL_24H
                )
            )
        if self.ENABLE_BO_LONG_OVERHEAT_GUARD:
            quality_rejected |= (
                dataframe["bo_entry_long"].astype(bool)
                & guard_regime
                & (
                    dataframe["eth_trend_4h"]
                    >= self.BO_LONG_MAX_ETH_TREND_4H
                )
            )
        dataframe.loc[quality_rejected, "bo_v12_quality_rejected"] = 1
        dataframe.loc[quality_rejected, "bo_entry_long"] = 0
        dataframe.loc[quality_rejected, "bo_entry_short"] = 0
        dataframe.loc[quality_rejected, "bo_entry"] = 0

        if not self.ENABLE_BO_V12_LONG_REENTRY:
            return dataframe

        atr = dataframe["atr"].clip(lower=1e-12)
        slow_slope = (
            dataframe["slow_ema"] - dataframe["slow_ema"].shift(6)
        ) / atr
        recent_primary = (
            frozen_long_impulse
            .shift(1)
            .rolling(168, min_periods=1)
            .max()
            .fillna(0.0)
            .astype(bool)
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
        close_position = dataframe["bo_long_close_position"]
        extension = (
            dataframe["close"] - dataframe["fast_ema"]
        ) / atr
        trend_restarted = (
            (dataframe["close"] > prior_high)
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["fast_ema"])
            & (close_position >= 0.62)
        )
        regime_ready = (
            (
                dataframe["market_regime_score"]
                >= self.BO_REENTRY_MIN_REGIME
            )
            & dataframe["breadth"].between(
                self.BO_REENTRY_MIN_BREADTH,
                self.BO_REENTRY_MAX_BREADTH,
            )
            & (
                dataframe["breadth_slow"].between(
                    self.BO_REENTRY_MIN_SLOW_BREADTH,
                    self.BO_REENTRY_MAX_SLOW_BREADTH,
                )
            )
            & (
                dataframe["breadth_change_24h"]
                >= self.BO_REENTRY_MIN_BREADTH_CHANGE_24H
            )
            & (
                dataframe["btc_trend_1d"]
                >= self.BO_REENTRY_MIN_BTC_DAILY_TREND
            )
            & (
                dataframe["eth_trend_1d"]
                >= self.BO_REENTRY_MIN_ETH_DAILY_TREND
            )
        )
        pair_ready = (
            (dataframe["fast_ema"] > dataframe["slow_ema"])
            & (slow_slope >= 0.02)
            & dataframe["symbol_ema55_atr"].between(0.10, 4.0)
            & (
                dataframe["symbol_efficiency_12h"]
                >= self.BO_REENTRY_MIN_SYMBOL_EFFICIENCY
            )
            & extension.between(0.0, self.BO_REENTRY_MAX_EXTENSION_ATR)
            & dataframe["bo_volume_ratio"].between(
                self.BO_REENTRY_MIN_VOLUME,
                self.BO_REENTRY_MAX_VOLUME,
            )
            & (dataframe["atr"] / dataframe["close"]).between(0.003, 0.10)
        )
        raw_reentry = (
            recent_primary
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
        reentry_risk = np.where(
            dataframe["market_regime_score"] >= 0.45,
            self.BO_REENTRY_STRONG_RISK,
            self.BO_REENTRY_BASE_RISK,
        )
        dataframe.loc[reentry, "bo_score"] = reentry_score.loc[reentry]
        dataframe.loc[reentry, "bo_regime"] = dataframe.loc[
            reentry, "bo_regime_long"
        ]
        dataframe.loc[reentry, "bo_risk_pct"] = np.minimum(
            reentry_risk[reentry],
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

    def _populate_grid(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_grid(dataframe)
        dataframe["grid_v9_long_pullback"] = 0
        dataframe["grid_v9_regime_rejected"] = 0
        dataframe["grid_v9_short_guard_rejected"] = 0
        dataframe["grid_v12_defensive_stop"] = 0
        dataframe["grid_v12_one_tp_floor"] = 0
        guard_regime = self._guard_regime(dataframe)

        if self.ENABLE_PERSISTENT_GRID_ONE_TP_FLOOR:
            one_tp_floor = (
                dataframe["grid_entry_short"].astype(bool)
                & (
                    dataframe["low_opportunity_fraction_72h"]
                    >= self.DEFENSIVE_MIN_PERSISTENCE_72H
                ).fillna(False)
            )
            dataframe.loc[one_tp_floor, "grid_v12_one_tp_floor"] = 1

        if self.ENABLE_PERSISTENT_DEFENSIVE_STOPS:
            defensive_regime = Series(True, index=dataframe.index)
            if self.DEFENSIVE_REQUIRE_PERSISTENCE:
                defensive_regime &= (
                    dataframe["low_opportunity_fraction_72h"]
                    >= self.DEFENSIVE_MIN_PERSISTENCE_72H
                ).fillna(False)
            if self.DEFENSIVE_MAX_DISPERSION_7D is not None:
                defensive_regime &= (
                    dataframe["market_dispersion_7d"]
                    <= self.DEFENSIVE_MAX_DISPERSION_7D
                ).fillna(False)
            defensive_grid = (
                dataframe["grid_entry_short"].astype(bool)
                & defensive_regime
                & (
                    self.DEFENSIVE_GRID_ALL_SHORTS
                    | (
                        self.DEFENSIVE_GRID_SHORT_REBOUND
                        & (
                            dataframe["symbol_return_4h"]
                            >= self.GRID_SHORT_MAX_REBOUND_4H
                        )
                    )
                )
            )
            dataframe.loc[
                defensive_grid, "grid_v12_defensive_stop"
            ] = 1

        if self.ENABLE_REGIME_FILTERS:
            short_conflict = Series(False, index=dataframe.index)
            if self.HARD_FILTER_BULL_SHORTS:
                short_conflict |= self._bull_conflict(dataframe)
            if self.HARD_FILTER_EXHAUSTED_SHORTS:
                short_conflict |= self._short_exhaustion(dataframe)
            rejected_short = (
                dataframe["grid_entry_short"].astype(bool)
                & short_conflict
            )
            dataframe.loc[rejected_short, "grid_v9_regime_rejected"] = 1
            dataframe.loc[rejected_short, "grid_entry_short"] = 0

        short_guard_rejected = Series(False, index=dataframe.index)
        if self.ENABLE_GRID_SHORT_REBOUND_GUARD:
            short_guard_rejected |= (
                dataframe["grid_entry_short"].astype(bool)
                & guard_regime
                & (
                    dataframe["symbol_return_4h"]
                    >= self.GRID_SHORT_MAX_REBOUND_4H
                )
            )
        if self.ENABLE_GRID_SHORT_LOW_DISPERSION_GUARD:
            short_guard_rejected |= (
                dataframe["grid_entry_short"].astype(bool)
                & guard_regime
                & (
                    dataframe["market_dispersion_24h"]
                    <= self.GRID_SHORT_MIN_DISPERSION_24H
                )
            )
        dataframe.loc[
            short_guard_rejected, "grid_v9_short_guard_rejected"
        ] = 1
        dataframe.loc[short_guard_rejected, "grid_entry_short"] = 0

        if self.ENABLE_GRID_V9_LONG:
            inherited_long = dataframe["grid_entry_long"].astype(bool).copy()
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
                (
                    dataframe["market_regime_score"]
                    >= self.GRID_V9_LONG_MIN_REGIME
                )
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
                if self.GRID_V9_REPLACE_LONG
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

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        bo_reentry = dataframe["bo_v12_long_reentry"].astype(bool)
        grid_pullback = dataframe["grid_v9_long_pullback"].astype(bool)
        for index in dataframe.index[bo_reentry]:
            tag = str(dataframe.at[index, "enter_tag"] or "")
            dataframe.at[index, "enter_tag"] = f"{tag}_v12reentry"
        for index in dataframe.index[grid_pullback]:
            tag = str(dataframe.at[index, "enter_tag"] or "")
            if tag.startswith("grid_v8"):
                dataframe.at[index, "enter_tag"] = f"{tag}_v9pullback"
        if self.ENABLE_PERSISTENT_DEFENSIVE_STOPS:
            breakout_defensive = (
                dataframe["bo_v12_defensive_stop"].astype(bool)
                & dataframe["enter_tag"].fillna("").str.startswith("bo_v9")
            )
            grid_defensive = (
                dataframe["grid_v12_defensive_stop"].astype(bool)
                & dataframe["enter_tag"].fillna("").str.startswith("grid_v8")
            )
            dataframe.loc[
                breakout_defensive | grid_defensive, "enter_tag"
            ] = (
                dataframe.loc[
                    breakout_defensive | grid_defensive, "enter_tag"
                ].astype(str)
                + "_v12def"
            )
        if self.ENABLE_PERSISTENT_GRID_ONE_TP_FLOOR:
            one_tp_floor = (
                dataframe["grid_v12_one_tp_floor"].astype(bool)
                & dataframe["enter_tag"].fillna("").str.startswith("grid_v8")
            )
            dataframe.loc[one_tp_floor, "enter_tag"] = (
                dataframe.loc[one_tp_floor, "enter_tag"].astype(str)
                + "_v12g1"
            )
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
        tag = str(getattr(trade, "enter_tag", "") or "")
        if (
            not self.ENABLE_PERSISTENT_DEFENSIVE_STOPS
            or "_v12def" not in tag
        ):
            return base_value

        component = self._component(tag)
        stop_atr = (
            self.DEFENSIVE_BO_STOP_ATR
            if component == "breakout"
            else self.DEFENSIVE_GRID_STOP_ATR
        )
        if component not in {"breakout", "grid"}:
            return base_value
        atr_value = max(
            float(trade.get_custom_data("initial_atr", 0.0)),
            1e-12,
        )
        side = -1.0 if trade.is_short else 1.0
        stop_rate = trade.open_rate - side * stop_atr * atr_value
        defensive_value = abs(
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
            return defensive_value
        return min(abs(float(base_value)), defensive_value)

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
        if reason:
            return reason
        tag = str(getattr(trade, "enter_tag", "") or "")
        if (
            not self.ENABLE_PERSISTENT_GRID_ONE_TP_FLOOR
            or "_v12g1" not in tag
            or self._component(tag) != "grid"
        ):
            return reason

        tp_count = int(trade.get_custom_data("grid_tp_count", 0))
        if tp_count < 1:
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
        profit = float(trade.calculate_profit(current_rate).total_profit)
        if (
            best_profit
            >= risk_budget * self.GRID_V12_ONE_TP_ACTIVATION_R
            and profit <= risk_budget * self.GRID_V12_ONE_TP_FLOOR_R
        ):
            return "grid_v12_persistent_one_tp_floor"
        return reason

    def _recent_side_returns(
        self,
        current_time: datetime,
        side: str,
        limit: int = 3,
    ) -> tuple[float, ...]:
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        target_short = side == "short"
        closed: list[tuple[datetime, float]] = []
        for trade in Trade.get_trades_proxy(is_open=False):
            if bool(getattr(trade, "is_short", False)) != target_short:
                continue
            close_time = self._trade_close_time(trade)
            value = self._closed_trade_return(trade)
            if close_time is None or close_time > boundary or value is None:
                continue
            closed.append((close_time, value))
        closed.sort(key=lambda item: item[0])
        return tuple(value for _time, value in closed[-limit:])

    def _recent_sleeve_returns(
        self,
        current_time: datetime,
        side: str,
        component: str,
    ) -> tuple[float, ...]:
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        target_short = side == "short"
        closed: list[tuple[datetime, float]] = []
        for trade in Trade.get_trades_proxy(is_open=False):
            if bool(getattr(trade, "is_short", False)) != target_short:
                continue
            if self._component(getattr(trade, "enter_tag", None)) != component:
                continue
            close_time = self._trade_close_time(trade)
            value = self._closed_trade_return(trade)
            if close_time is None or close_time > boundary or value is None:
                continue
            closed.append((close_time, value))
        closed.sort(key=lambda item: item[0])
        return tuple(
            value
            for _time, value in closed[-self.SLEEVE_RECENT_WINDOW :]
        )

    def _sleeve_performance_scale(
        self,
        current_time: datetime,
        side: str,
        component: str,
    ) -> float:
        if not self.ENABLE_SLEEVE_PERFORMANCE_RISK:
            return 1.0
        recent = self._recent_sleeve_returns(
            current_time,
            side,
            component,
        )
        if len(recent) < self.SLEEVE_MIN_OBSERVATIONS:
            return 1.0
        wins = sum(value > 0.0 for value in recent)
        aggregate = float(sum(recent))
        if aggregate >= 0.35:
            return 1.0
        defensive = wins <= 1 or aggregate <= -0.10
        caution = wins <= 2 or aggregate < 0.0
        if component == "grid":
            if defensive:
                return self.GRID_SLEEVE_DEFENSIVE_SCALE
            if caution:
                return self.GRID_SLEEVE_CAUTION_SCALE
            return 1.0
        if defensive:
            return self.BO_SLEEVE_DEFENSIVE_SCALE
        if caution:
            return self.BO_SLEEVE_CAUTION_SCALE
        return 1.0

    def _regime_risk_scale(
        self,
        row: Series,
        side: str,
        current_time: datetime,
    ) -> float:
        regime = float(row.get("market_regime_score", 0.0))
        breadth_change = float(row.get("breadth_change_24h", 0.0))
        if side == "long":
            if regime >= 0.35:
                scale = self.LONG_STRONG_BULL_SCALE
            elif regime <= -0.35:
                scale = self.LONG_BEAR_SCALE
            elif abs(regime) < 0.12:
                scale = self.NEUTRAL_SCALE
            else:
                scale = 1.0
        else:
            if regime <= -0.35:
                scale = self.SHORT_STRONG_BEAR_SCALE
            elif regime >= 0.30:
                scale = self.SHORT_BULL_SCALE
            elif abs(regime) < 0.12:
                scale = self.NEUTRAL_SCALE
            else:
                scale = 1.0

            recent = self._recent_side_returns(current_time, side, limit=3)
            exhausted = (
                regime <= self.SHORT_EXHAUSTION_MAX_REGIME
                and float(row.get("breadth", 1.0))
                <= self.SHORT_EXHAUSTION_MAX_BREADTH
                and float(row.get("breadth_slow", 1.0))
                <= self.SHORT_EXHAUSTION_MAX_SLOW_BREADTH
            )
            if (
                len(recent) >= 2
                and recent[-1] < 0.0
                and recent[-2] < 0.0
                and (breadth_change > 0.04 or exhausted)
            ):
                scale *= 0.75
        return scale

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
        if stake <= 0.0:
            return stake
        component = self._component(entry_tag)
        needs_row = any(
            (
                self.ENABLE_TARGETED_SETUP_RISK,
                self.ENABLE_REGIME_RISK,
                self.ENABLE_LOW_OPPORTUNITY_DRAWDOWN_GOVERNOR,
                self.ENABLE_LOW_OPPORTUNITY_RECENT_GOVERNOR,
                self.ENABLE_LOW_OPPORTUNITY_BASE_SCALE,
            )
        )
        row = (
            self._latest_signal_row(pair, component, current_time)
            if needs_row
            else None
        )
        if needs_row and row is None:
            return 0.0

        scale = 1.0
        targeted_regime_ready = (
            row is not None
            and (
                not self.TARGETED_RISK_ONLY_LOW_OPPORTUNITY
                or float(row.get("market_dispersion_7d", float("inf")))
                <= self.LOW_OPPORTUNITY_MAX_DISPERSION_7D
            )
        )
        if self.ENABLE_TARGETED_SETUP_RISK and targeted_regime_ready:
            if component == "grid" and side == "short":
                if (
                    float(row.get("symbol_return_4h", 0.0))
                    >= self.GRID_SHORT_MAX_REBOUND_4H
                ):
                    scale *= self.TARGET_GRID_REBOUND_SCALE
                if (
                    float(row.get("market_dispersion_24h", float("inf")))
                    <= self.GRID_SHORT_MIN_DISPERSION_24H
                ):
                    scale *= self.TARGET_GRID_LOW_DISPERSION_SCALE
            elif component == "breakout" and side == "short":
                if (
                    float(row.get("bo_range_atr", float("inf")))
                    <= self.BO_SHORT_MIN_RANGE_ATR
                ):
                    scale *= self.TARGET_BO_SHORT_RANGE_SCALE
                if (
                    float(row.get("breadth_travel_24h", 0.0))
                    >= self.BO_SHORT_MAX_BREADTH_TRAVEL_24H
                ):
                    scale *= self.TARGET_BO_SHORT_CHURN_SCALE
            elif component == "breakout" and side == "long":
                if (
                    float(row.get("eth_trend_4h", float("-inf")))
                    >= self.BO_LONG_MAX_ETH_TREND_4H
                ):
                    scale *= self.TARGET_BO_LONG_OVERHEAT_SCALE

        if self.ENABLE_REGIME_RISK and row is not None:
            scale *= self._regime_risk_scale(row, side, current_time)
            scale *= self._sleeve_performance_scale(
                current_time,
                side,
                component,
            )

        if (
            self.ENABLE_LOW_OPPORTUNITY_DRAWDOWN_GOVERNOR
            and row is not None
            and (
                not self.LOW_OPPORTUNITY_DRAWDOWN_SHORT_ONLY
                or side == "short"
            )
            and float(row.get("market_dispersion_7d", float("inf")))
            <= self.LOW_OPPORTUNITY_MAX_DISPERSION_7D
        ):
            equity = float(self.wallets.get_total_stake_amount())
            drawdown = (
                max(0.0, 1.0 - equity / self._peak_equity)
                if equity > 0.0 and self._peak_equity > 0.0
                else 0.0
            )
            if drawdown >= self.LOW_OPPORTUNITY_SEVERE_DRAWDOWN:
                scale *= self.LOW_OPPORTUNITY_SEVERE_SCALE
            elif drawdown >= self.LOW_OPPORTUNITY_DRAWDOWN_TRIGGER:
                scale *= self.LOW_OPPORTUNITY_DRAWDOWN_SCALE

        if (
            self.ENABLE_LOW_OPPORTUNITY_BASE_SCALE
            and row is not None
            and float(row.get("market_dispersion_7d", float("inf")))
            <= self.LOW_OPPORTUNITY_MAX_DISPERSION_7D
        ):
            scale *= self.LOW_OPPORTUNITY_BASE_SCALE

        if (
            self.ENABLE_LOW_OPPORTUNITY_RECENT_GOVERNOR
            and row is not None
            and (
                not self.LOW_OPPORTUNITY_RECENT_SHORT_ONLY
                or side == "short"
            )
            and float(row.get("market_dispersion_7d", float("inf")))
            <= self.LOW_OPPORTUNITY_MAX_DISPERSION_7D
        ):
            if self.LOW_OPPORTUNITY_RECENT_COMPONENT_SPECIFIC:
                recent = self._recent_sleeve_returns(
                    current_time,
                    side,
                    component,
                )[-self.LOW_OPPORTUNITY_RECENT_WINDOW :]
            else:
                recent = self._recent_portfolio_returns(current_time)[
                    -self.LOW_OPPORTUNITY_RECENT_WINDOW :
                ]
            if len(recent) >= self.LOW_OPPORTUNITY_RECENT_MIN_OBSERVATIONS:
                wins = sum(value > 0.0 for value in recent)
                aggregate = float(sum(recent))
                last_two_lost = recent[-1] < 0.0 and recent[-2] < 0.0
                if last_two_lost or aggregate <= -0.10:
                    scale *= self.LOW_OPPORTUNITY_RECENT_DEFENSIVE_SCALE
                elif aggregate < 0.0 or wins <= 1:
                    scale *= self.LOW_OPPORTUNITY_RECENT_CAUTION_SCALE
        scaled = min(max_stake, max(0.0, stake * scale))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class BreakoutV12ARegimeFilterGridV8Freqtrade(
    BreakoutV12MultiRegimeGridV9Freqtrade
):
    """Ablation A: directional conflict filters only."""

    ENABLE_REGIME_RISK = False
    ENABLE_SLEEVE_PERFORMANCE_RISK = False
    ENABLE_GRID_V9_LONG = False
    ENABLE_BO_V12_LONG_REENTRY = False
    HARD_FILTER_BULL_SHORTS = True


class BreakoutV12BRegimeRiskGridV8Freqtrade(
    BreakoutV12MultiRegimeGridV9Freqtrade
):
    """Ablation B: conflict filters plus side-aware allocation."""

    ENABLE_GRID_V9_LONG = False
    ENABLE_BO_V12_LONG_REENTRY = False


class BreakoutV12CRegimeRiskGridV9Freqtrade(
    BreakoutV12MultiRegimeGridV9Freqtrade
):
    """Ablation C: replace Grid long, but keep Breakout entries frozen."""

    ENABLE_BO_V12_LONG_REENTRY = False


class BreakoutV12DRegimeBreakoutGridV8Freqtrade(
    BreakoutV12MultiRegimeGridV9Freqtrade
):
    """Ablation D: add Breakout re-entry, but keep Grid-v8 long."""

    ENABLE_GRID_V9_LONG = False


class BreakoutV12EGridV9SignalsOnlyFreqtrade(
    BreakoutV12MultiRegimeGridV9Freqtrade
):
    """Grid-v9 long replacement with every frozen base stake unchanged."""

    ENABLE_REGIME_FILTERS = False
    ENABLE_REGIME_RISK = False
    ENABLE_SLEEVE_PERFORMANCE_RISK = False
    ENABLE_BO_V12_LONG_REENTRY = False


class BreakoutV12FGridV9AdditiveSignalsOnlyFreqtrade(
    BreakoutV12EGridV9SignalsOnlyFreqtrade
):
    """Add Grid-v9 pullbacks without removing inherited Grid-v8 longs."""

    GRID_V9_REPLACE_LONG = False


class BreakoutV12GBreakoutSignalsOnlyGridV8Freqtrade(
    BreakoutV12MultiRegimeGridV9Freqtrade
):
    """Breakout re-entry only, with frozen Grid-v8 and base risk."""

    ENABLE_REGIME_FILTERS = False
    ENABLE_REGIME_RISK = False
    ENABLE_SLEEVE_PERFORMANCE_RISK = False
    ENABLE_GRID_V9_LONG = False


class BreakoutV12HSignalsOnlyGridV9Freqtrade(
    BreakoutV12MultiRegimeGridV9Freqtrade
):
    """Both new long structures with every frozen base stake unchanged."""

    ENABLE_REGIME_FILTERS = False
    ENABLE_REGIME_RISK = False
    ENABLE_SLEEVE_PERFORMANCE_RISK = False


class BreakoutV12ResearchFrozenSignalsBase(
    BreakoutV12MultiRegimeGridV9Freqtrade
):
    """Research base which changes no frozen v11/Grid-v8 trade decision."""

    ENABLE_REGIME_FILTERS = False
    ENABLE_REGIME_RISK = False
    ENABLE_SLEEVE_PERFORMANCE_RISK = False
    ENABLE_GRID_V9_LONG = False
    ENABLE_BO_V12_LONG_REENTRY = False


class BreakoutV12IGridShortRebound005Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Reject Grid shorts while the symbol is rebounding at least 0.50%."""

    ENABLE_GRID_SHORT_REBOUND_GUARD = True
    GRID_SHORT_MAX_REBOUND_4H = 0.0050


class BreakoutV12JGridShortRebound075Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Reject Grid shorts while the symbol is rebounding at least 0.75%."""

    ENABLE_GRID_SHORT_REBOUND_GUARD = True
    GRID_SHORT_MAX_REBOUND_4H = 0.0075


class BreakoutV12KGridShortRebound100Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Reject Grid shorts while the symbol is rebounding at least 1.00%."""

    ENABLE_GRID_SHORT_REBOUND_GUARD = True
    GRID_SHORT_MAX_REBOUND_4H = 0.0100


class BreakoutV12LGridShortReboundDispersionFreqtrade(
    BreakoutV12JGridShortRebound075Freqtrade
):
    """Also avoid Grid shorts in exceptionally compressed markets."""

    ENABLE_GRID_SHORT_LOW_DISPERSION_GUARD = True


class BreakoutV12MBreakoutShortRangeGuardFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Require enough prior daily range before accepting a short break."""

    ENABLE_BO_SHORT_NARROW_RANGE_GUARD = True


class BreakoutV12NBreakoutShortChurnGuardFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Reject short breaks while cross-market breadth is whipsawing."""

    ENABLE_BO_SHORT_BREADTH_CHURN_GUARD = True


class BreakoutV12OBreakoutLongOverheatGuardFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Avoid chasing long breakouts after an already extreme ETH impulse."""

    ENABLE_BO_LONG_OVERHEAT_GUARD = True


class BreakoutV12PStructuralGuardComboFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Combined structural guards for the first multi-stage screen."""

    ENABLE_GRID_SHORT_REBOUND_GUARD = True
    ENABLE_BO_SHORT_NARROW_RANGE_GUARD = True
    ENABLE_BO_SHORT_BREADTH_CHURN_GUARD = True
    ENABLE_BO_LONG_OVERHEAT_GUARD = True


class BreakoutV12QStrongBullGridPullbackFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Add Grid trend pullbacks only in a persistent broad bull regime."""

    ENABLE_GRID_V9_LONG = True
    GRID_V9_REPLACE_LONG = False
    GRID_V9_LONG_MIN_REGIME = 0.70
    GRID_V9_LONG_MIN_BTC_DAILY_TREND = 0.25
    GRID_V9_LONG_MIN_ETH_DAILY_TREND = 0.15


class BreakoutV12RGuardedStrongBullGridPullbackFreqtrade(
    BreakoutV12QStrongBullGridPullbackFreqtrade
):
    """Strong-bull pullbacks plus the robust Grid short rebound guard."""

    ENABLE_GRID_SHORT_REBOUND_GUARD = True


class BreakoutV12STargetGridRisk50Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Keep every signal, but halve risk on a Grid short into a rebound."""

    ENABLE_TARGETED_SETUP_RISK = True
    TARGET_GRID_REBOUND_SCALE = 0.50


class BreakoutV12TTargetGridRisk35Disp60Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Defensive Grid sizing for rebound and compressed-market setups."""

    ENABLE_TARGETED_SETUP_RISK = True
    TARGET_GRID_REBOUND_SCALE = 0.35
    TARGET_GRID_LOW_DISPERSION_SCALE = 0.60


class BreakoutV12UTargetBreakoutRisk50Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Targeted Breakout sizing without suppressing any entry."""

    ENABLE_TARGETED_SETUP_RISK = True
    TARGET_BO_SHORT_RANGE_SCALE = 0.50
    TARGET_BO_SHORT_CHURN_SCALE = 0.50
    TARGET_BO_LONG_OVERHEAT_SCALE = 0.50


class BreakoutV12VTargetedRiskModerateFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Moderate setup-aware allocation while preserving trade order."""

    ENABLE_TARGETED_SETUP_RISK = True
    TARGET_GRID_REBOUND_SCALE = 0.50
    TARGET_GRID_LOW_DISPERSION_SCALE = 0.70
    TARGET_BO_SHORT_RANGE_SCALE = 0.60
    TARGET_BO_SHORT_CHURN_SCALE = 0.60
    TARGET_BO_LONG_OVERHEAT_SCALE = 0.70


class BreakoutV12WTargetedRiskDefensiveFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Stronger neighbor for setup-aware allocation robustness."""

    ENABLE_TARGETED_SETUP_RISK = True
    TARGET_GRID_REBOUND_SCALE = 0.35
    TARGET_GRID_LOW_DISPERSION_SCALE = 0.50
    TARGET_BO_SHORT_RANGE_SCALE = 0.50
    TARGET_BO_SHORT_CHURN_SCALE = 0.50
    TARGET_BO_LONG_OVERHEAT_SCALE = 0.60


class BreakoutV12XTargetedRiskStrongBullPullbackFreqtrade(
    BreakoutV12QStrongBullGridPullbackFreqtrade
):
    """Moderate targeted allocation plus rare strong-bull pullbacks."""

    ENABLE_TARGETED_SETUP_RISK = True
    TARGET_GRID_REBOUND_SCALE = 0.50
    TARGET_GRID_LOW_DISPERSION_SCALE = 0.70
    TARGET_BO_SHORT_RANGE_SCALE = 0.60
    TARGET_BO_SHORT_CHURN_SCALE = 0.60
    TARGET_BO_LONG_OVERHEAT_SCALE = 0.70


class BreakoutV12YChurnGuardStrongBullPullbackFreqtrade(
    BreakoutV12QStrongBullGridPullbackFreqtrade
):
    """Hard breadth-churn guard plus rare strong-bull pullbacks."""

    ENABLE_BO_SHORT_BREADTH_CHURN_GUARD = True


class BreakoutV12AALowOpportunityGuards032Freqtrade(
    BreakoutV12PStructuralGuardComboFreqtrade
):
    """Use structural rejection only in very low-opportunity regimes."""

    GUARDS_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.032


class BreakoutV12ABLowOpportunityGuards035Freqtrade(
    BreakoutV12PStructuralGuardComboFreqtrade
):
    """Balanced seven-day dispersion gate for structural rejection."""

    GUARDS_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.035


class BreakoutV12ACLowOpportunityGuards038Freqtrade(
    BreakoutV12PStructuralGuardComboFreqtrade
):
    """Broader neighboring dispersion gate for robustness testing."""

    GUARDS_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.038


class BreakoutV12ADAdaptiveGuardsBullPullbackFreqtrade(
    BreakoutV12ABLowOpportunityGuards035Freqtrade
):
    """Defend compressed regimes and add pullbacks only in strong bulls."""

    ENABLE_GRID_V9_LONG = True
    GRID_V9_REPLACE_LONG = False
    GRID_V9_LONG_MIN_REGIME = 0.70
    GRID_V9_LONG_MIN_BTC_DAILY_TREND = 0.25
    GRID_V9_LONG_MIN_ETH_DAILY_TREND = 0.15


class BreakoutV12AESelectiveLowOpportunityGuardsFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Narrow adaptive guard excluding the weaker range-only veto."""

    GUARDS_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.035
    ENABLE_GRID_SHORT_REBOUND_GUARD = True
    ENABLE_BO_SHORT_BREADTH_CHURN_GUARD = True
    ENABLE_BO_LONG_OVERHEAT_GUARD = True


class BreakoutV12AFLowOpportunityGuards027Freqtrade(
    BreakoutV12PStructuralGuardComboFreqtrade
):
    """Conservative low-opportunity boundary with 2026 signal parity."""

    GUARDS_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.027


class BreakoutV12AGLowOpportunityGuards028Freqtrade(
    BreakoutV12PStructuralGuardComboFreqtrade
):
    """Middle neighbor for the low-opportunity boundary."""

    GUARDS_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.028


class BreakoutV12AHLowOpportunityGuards029Freqtrade(
    BreakoutV12PStructuralGuardComboFreqtrade
):
    """Upper robust neighbor before 2026 profitable signals are touched."""

    GUARDS_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.029


class BreakoutV12AITargetedLowOpportunity030Freqtrade(
    BreakoutV12VTargetedRiskModerateFreqtrade
):
    """Moderate sizing only when seven-day opportunity is compressed."""

    TARGETED_RISK_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.030


class BreakoutV12AJTargetedLowOpportunity032Freqtrade(
    BreakoutV12VTargetedRiskModerateFreqtrade
):
    """Middle low-opportunity sizing neighbor."""

    TARGETED_RISK_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.032


class BreakoutV12AKTargetedLowOpportunity035Freqtrade(
    BreakoutV12VTargetedRiskModerateFreqtrade
):
    """Broad low-opportunity sizing neighbor."""

    TARGETED_RISK_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.035


class BreakoutV12ALDefensiveLowOpportunity030Freqtrade(
    BreakoutV12WTargetedRiskDefensiveFreqtrade
):
    """Stronger sizing reduction at the conservative opportunity boundary."""

    TARGETED_RISK_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.030


class BreakoutV12AMSelectiveRiskLowOpportunity029Freqtrade(
    BreakoutV12VTargetedRiskModerateFreqtrade
):
    """Scale only structurally conflicted setups in compressed regimes."""

    TARGETED_RISK_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.029
    TARGET_GRID_LOW_DISPERSION_SCALE = 1.0


class BreakoutV12ANSelectiveRiskDefensive029Freqtrade(
    BreakoutV12WTargetedRiskDefensiveFreqtrade
):
    """Defensive neighbor without a standalone dispersion penalty."""

    TARGETED_RISK_ONLY_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.029
    TARGET_GRID_LOW_DISPERSION_SCALE = 1.0


class BreakoutV12AOLowOpportunityDrawdown10Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Halve new risk after a 10% drawdown in compressed markets."""

    ENABLE_LOW_OPPORTUNITY_DRAWDOWN_GOVERNOR = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.035
    LOW_OPPORTUNITY_DRAWDOWN_TRIGGER = 0.10
    LOW_OPPORTUNITY_DRAWDOWN_SCALE = 0.50


class BreakoutV12APLowOpportunityDrawdown12Freqtrade(
    BreakoutV12AOLowOpportunityDrawdown10Freqtrade
):
    """Looser drawdown neighbor for the low-opportunity governor."""

    LOW_OPPORTUNITY_DRAWDOWN_TRIGGER = 0.12


class BreakoutV12AQLowOpportunityDrawdownTwoTierFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Gradual portfolio de-risking at 8% and 14% drawdown."""

    ENABLE_LOW_OPPORTUNITY_DRAWDOWN_GOVERNOR = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.035
    LOW_OPPORTUNITY_DRAWDOWN_TRIGGER = 0.08
    LOW_OPPORTUNITY_DRAWDOWN_SCALE = 0.70
    LOW_OPPORTUNITY_SEVERE_DRAWDOWN = 0.14
    LOW_OPPORTUNITY_SEVERE_SCALE = 0.35


class BreakoutV12ARLowOpportunityShortDrawdownFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """De-risk only new shorts after low-opportunity drawdown."""

    ENABLE_LOW_OPPORTUNITY_DRAWDOWN_GOVERNOR = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.035
    LOW_OPPORTUNITY_DRAWDOWN_TRIGGER = 0.08
    LOW_OPPORTUNITY_DRAWDOWN_SCALE = 0.50
    LOW_OPPORTUNITY_DRAWDOWN_SHORT_ONLY = True


class BreakoutV12ASLowOpportunityRecentPortfolioFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Causal recent-campaign governor in low-opportunity markets."""

    ENABLE_LOW_OPPORTUNITY_RECENT_GOVERNOR = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.035


class BreakoutV12ATLowOpportunityRecentShortFreqtrade(
    BreakoutV12ASLowOpportunityRecentPortfolioFreqtrade
):
    """Apply the recent-campaign governor only to new shorts."""

    LOW_OPPORTUNITY_RECENT_SHORT_ONLY = True


class BreakoutV12AULowOpportunityRecentSleeveShortFreqtrade(
    BreakoutV12ATLowOpportunityRecentShortFreqtrade
):
    """Use each short sleeve's own recent campaigns for allocation."""

    LOW_OPPORTUNITY_RECENT_COMPONENT_SPECIFIC = True


class BreakoutV12AVLowOpportunityRecentPortfolio030Freqtrade(
    BreakoutV12ASLowOpportunityRecentPortfolioFreqtrade
):
    """Narrow opportunity neighbor for the recent-campaign governor."""

    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.030


class BreakoutV12AWLowOpportunityRecentPortfolioDefensiveFreqtrade(
    BreakoutV12ASLowOpportunityRecentPortfolioFreqtrade
):
    """Stronger recent-campaign allocation neighbor."""

    LOW_OPPORTUNITY_RECENT_CAUTION_SCALE = 0.65
    LOW_OPPORTUNITY_RECENT_DEFENSIVE_SCALE = 0.35


class BreakoutV12AXLowOpportunityBaseScale50Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Half-size every setup during persistent low-opportunity regimes."""

    ENABLE_LOW_OPPORTUNITY_BASE_SCALE = True
    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.029
    LOW_OPPORTUNITY_BASE_SCALE = 0.50


class BreakoutV12AYLowOpportunityBaseScale70Freqtrade(
    BreakoutV12AXLowOpportunityBaseScale50Freqtrade
):
    """Looser allocation neighbor for low-opportunity regimes."""

    LOW_OPPORTUNITY_BASE_SCALE = 0.70


class BreakoutV12AZLowOpportunityBaseScale35Freqtrade(
    BreakoutV12AXLowOpportunityBaseScale50Freqtrade
):
    """Defensive allocation neighbor for low-opportunity regimes."""

    LOW_OPPORTUNITY_BASE_SCALE = 0.35


class BreakoutV12BALowOpportunity028Scale50Freqtrade(
    BreakoutV12AXLowOpportunityBaseScale50Freqtrade
):
    """Narrow opportunity-boundary neighbor."""

    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.028


class BreakoutV12BBLowOpportunity030Scale50Freqtrade(
    BreakoutV12AXLowOpportunityBaseScale50Freqtrade
):
    """Broad opportunity-boundary neighbor."""

    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.030


class BreakoutV12BCLowOpportunityBaseScale25Freqtrade(
    BreakoutV12AXLowOpportunityBaseScale50Freqtrade
):
    """Lower allocation neighbor for low-opportunity regimes."""

    LOW_OPPORTUNITY_BASE_SCALE = 0.25


class BreakoutV12BDLowOpportunityBaseScale40Freqtrade(
    BreakoutV12AXLowOpportunityBaseScale50Freqtrade
):
    """Middle allocation neighbor for low-opportunity regimes."""

    LOW_OPPORTUNITY_BASE_SCALE = 0.40


class BreakoutV12BELowOpportunity027Scale35Freqtrade(
    BreakoutV12AZLowOpportunityBaseScale35Freqtrade
):
    """Narrow boundary neighbor for the selected allocation level."""

    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.027


class BreakoutV12BFLowOpportunity028Scale35Freqtrade(
    BreakoutV12AZLowOpportunityBaseScale35Freqtrade
):
    """Middle-low boundary neighbor for the selected allocation level."""

    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.028


class BreakoutV12BGLowOpportunity030Scale35Freqtrade(
    BreakoutV12AZLowOpportunityBaseScale35Freqtrade
):
    """Upper boundary neighbor for the selected allocation level."""

    LOW_OPPORTUNITY_MAX_DISPERSION_7D = 0.030


class BreakoutV12BHTargetedConflictRisk80Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Mildly de-risk structurally conflicted setups without deleting them."""

    ENABLE_TARGETED_SETUP_RISK = True
    TARGET_GRID_REBOUND_SCALE = 0.80
    TARGET_GRID_LOW_DISPERSION_SCALE = 1.0
    TARGET_BO_SHORT_RANGE_SCALE = 0.80
    TARGET_BO_SHORT_CHURN_SCALE = 0.80
    TARGET_BO_LONG_OVERHEAT_SCALE = 0.80


class BreakoutV12BITargetedConflictRisk90Freqtrade(
    BreakoutV12BHTargetedConflictRisk80Freqtrade
):
    """Loose conflict-risk neighbor."""

    TARGET_GRID_REBOUND_SCALE = 0.90
    TARGET_BO_SHORT_RANGE_SCALE = 0.90
    TARGET_BO_SHORT_CHURN_SCALE = 0.90
    TARGET_BO_LONG_OVERHEAT_SCALE = 0.90


class BreakoutV12BJTargetedConflictRisk70Freqtrade(
    BreakoutV12BHTargetedConflictRisk80Freqtrade
):
    """Stronger conflict-risk neighbor."""

    TARGET_GRID_REBOUND_SCALE = 0.70
    TARGET_BO_SHORT_RANGE_SCALE = 0.70
    TARGET_BO_SHORT_CHURN_SCALE = 0.70
    TARGET_BO_LONG_OVERHEAT_SCALE = 0.70


class BreakoutV12BKTargetedConflictRisk95Freqtrade(
    BreakoutV12BHTargetedConflictRisk80Freqtrade
):
    """Very mild conflict-risk neighbor."""

    TARGET_GRID_REBOUND_SCALE = 0.95
    TARGET_BO_SHORT_RANGE_SCALE = 0.95
    TARGET_BO_SHORT_CHURN_SCALE = 0.95
    TARGET_BO_LONG_OVERHEAT_SCALE = 0.95


class BreakoutV12BLTargetedConflictRisk97Freqtrade(
    BreakoutV12BHTargetedConflictRisk80Freqtrade
):
    """Near-parity conflict-risk neighbor."""

    TARGET_GRID_REBOUND_SCALE = 0.97
    TARGET_BO_SHORT_RANGE_SCALE = 0.97
    TARGET_BO_SHORT_CHURN_SCALE = 0.97
    TARGET_BO_LONG_OVERHEAT_SCALE = 0.97


class BreakoutV12BMPersistentLowOpportunityGuards080Freqtrade(
    BreakoutV12PStructuralGuardComboFreqtrade
):
    """Structural guards after three days of mostly compressed opportunity."""

    GUARDS_ONLY_PERSISTENT_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MIN_PERSISTENCE_72H = 0.80


class BreakoutV12BNPersistentLowOpportunityGuards090Freqtrade(
    BreakoutV12PStructuralGuardComboFreqtrade
):
    """Balanced persistent low-opportunity structural guards."""

    GUARDS_ONLY_PERSISTENT_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MIN_PERSISTENCE_72H = 0.90


class BreakoutV12BOPersistentLowOpportunityGuards100Freqtrade(
    BreakoutV12PStructuralGuardComboFreqtrade
):
    """Strict persistence neighbor for structural guards."""

    GUARDS_ONLY_PERSISTENT_LOW_OPPORTUNITY = True
    LOW_OPPORTUNITY_MIN_PERSISTENCE_72H = 1.00


class BreakoutV12BPPersistentGuardsBullPullbackFreqtrade(
    BreakoutV12BNPersistentLowOpportunityGuards090Freqtrade
):
    """Persistent defense plus rare strong-bull trend pullbacks."""

    ENABLE_GRID_V9_LONG = True
    GRID_V9_REPLACE_LONG = False
    GRID_V9_LONG_MIN_REGIME = 0.70
    GRID_V9_LONG_MIN_BTC_DAILY_TREND = 0.25
    GRID_V9_LONG_MIN_ETH_DAILY_TREND = 0.15


class BreakoutV12BQPersistentDefensiveStopsBalancedFreqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Keep frozen entries but tighten risk on persistent conflict setups."""

    ENABLE_PERSISTENT_DEFENSIVE_STOPS = True
    DEFENSIVE_MIN_PERSISTENCE_72H = 0.90
    DEFENSIVE_BO_STOP_ATR = 0.60
    DEFENSIVE_GRID_STOP_ATR = 1.80


class BreakoutV12BRPersistentDefensiveStopsTightFreqtrade(
    BreakoutV12BQPersistentDefensiveStopsBalancedFreqtrade
):
    """Tighter neighboring stop geometry for robustness screening."""

    DEFENSIVE_BO_STOP_ATR = 0.55
    DEFENSIVE_GRID_STOP_ATR = 1.50


class BreakoutV12BSPersistentDefensiveStopsLooseFreqtrade(
    BreakoutV12BQPersistentDefensiveStopsBalancedFreqtrade
):
    """Looser neighboring stop geometry for robustness screening."""

    DEFENSIVE_BO_STOP_ATR = 0.70
    DEFENSIVE_GRID_STOP_ATR = 2.10


class BreakoutV12BTChurnShortStop055Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Tighten only Breakout shorts entered during breadth whipsaw."""

    ENABLE_PERSISTENT_DEFENSIVE_STOPS = True
    DEFENSIVE_REQUIRE_PERSISTENCE = False
    DEFENSIVE_GRID_SHORT_REBOUND = False
    DEFENSIVE_BO_SHORT_RANGE = False
    DEFENSIVE_BO_SHORT_CHURN = True
    DEFENSIVE_BO_LONG_OVERHEAT = False
    DEFENSIVE_BO_STOP_ATR = 0.55


class BreakoutV12BUChurnShortStop065Freqtrade(
    BreakoutV12BTChurnShortStop055Freqtrade
):
    """Balanced neighboring stop for breadth-whipsaw shorts."""

    DEFENSIVE_BO_STOP_ATR = 0.65


class BreakoutV12BVChurnShortStop070Freqtrade(
    BreakoutV12BTChurnShortStop055Freqtrade
):
    """Loose neighboring stop for breadth-whipsaw shorts."""

    DEFENSIVE_BO_STOP_ATR = 0.70


class BreakoutV12BWChurnShortStop060Freqtrade(
    BreakoutV12BTChurnShortStop055Freqtrade
):
    """Lower-middle neighbor for the breadth-whipsaw stop."""

    DEFENSIVE_BO_STOP_ATR = 0.60


class BreakoutV12BXChurnShortStop067Freqtrade(
    BreakoutV12BTChurnShortStop055Freqtrade
):
    """Upper-middle neighbor for the breadth-whipsaw stop."""

    DEFENSIVE_BO_STOP_ATR = 0.67


class BreakoutV12BYCompressedChurnShort035Freqtrade(
    BreakoutV12BUChurnShortStop065Freqtrade
):
    """Defensive short stop only in compressed breadth whipsaw."""

    DEFENSIVE_MAX_DISPERSION_7D = 0.035


class BreakoutV12BZCompressedChurnShort038Freqtrade(
    BreakoutV12BUChurnShortStop065Freqtrade
):
    """Middle compression neighbor for breadth-whipsaw shorts."""

    DEFENSIVE_MAX_DISPERSION_7D = 0.038


class BreakoutV12CACompressedChurnShort040Freqtrade(
    BreakoutV12BUChurnShortStop065Freqtrade
):
    """Broad compression neighbor for breadth-whipsaw shorts."""

    DEFENSIVE_MAX_DISPERSION_7D = 0.040


class BreakoutV12CBPersistentGridShortStop150Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Tight Grid-short campaign boundary in persistent compression."""

    ENABLE_PERSISTENT_DEFENSIVE_STOPS = True
    DEFENSIVE_REQUIRE_PERSISTENCE = True
    DEFENSIVE_MIN_PERSISTENCE_72H = 0.90
    DEFENSIVE_GRID_ALL_SHORTS = True
    DEFENSIVE_GRID_SHORT_REBOUND = False
    DEFENSIVE_BO_SHORT_RANGE = False
    DEFENSIVE_BO_SHORT_CHURN = False
    DEFENSIVE_BO_LONG_OVERHEAT = False
    DEFENSIVE_GRID_STOP_ATR = 1.50


class BreakoutV12CCPersistentGridShortStop180Freqtrade(
    BreakoutV12CBPersistentGridShortStop150Freqtrade
):
    """Balanced Grid-short compression boundary."""

    DEFENSIVE_GRID_STOP_ATR = 1.80


class BreakoutV12CDPersistentGridShortStop210Freqtrade(
    BreakoutV12CBPersistentGridShortStop150Freqtrade
):
    """Loose Grid-short compression boundary."""

    DEFENSIVE_GRID_STOP_ATR = 2.10


class BreakoutV12CEPersistentGridOneTpFloor000Freqtrade(
    BreakoutV12ResearchFrozenSignalsBase
):
    """Protect a completed Grid cycle at net breakeven in compression."""

    ENABLE_PERSISTENT_GRID_ONE_TP_FLOOR = True
    DEFENSIVE_MIN_PERSISTENCE_72H = 0.90
    GRID_V12_ONE_TP_ACTIVATION_R = 0.10
    GRID_V12_ONE_TP_FLOOR_R = 0.00


class BreakoutV12CFPersistentGridOneTpFloor002Freqtrade(
    BreakoutV12CEPersistentGridOneTpFloor000Freqtrade
):
    """Protect two percent of campaign risk after the first Grid cycle."""

    GRID_V12_ONE_TP_FLOOR_R = 0.02


class BreakoutV12CGPersistentGridOneTpFloor004Freqtrade(
    BreakoutV12CEPersistentGridOneTpFloor000Freqtrade
):
    """Tighter profit floor neighbor after the first Grid cycle."""

    GRID_V12_ONE_TP_FLOOR_R = 0.04

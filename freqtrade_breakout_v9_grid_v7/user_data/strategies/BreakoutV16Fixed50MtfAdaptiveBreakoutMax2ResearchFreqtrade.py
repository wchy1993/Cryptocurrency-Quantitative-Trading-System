from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import Trade, stoploss_from_absolute

from BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade import (
    BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade,
)


class _V16IntrahourPathMixin:
    """Causal 1h -> 30m -> 15m confirmation for the frozen V15 signals.

    The primary clock remains one hour, exactly as in V15.  A row stamped at
    hour ``T`` may only use the four 15-minute candles inside ``[T, T+1h)``;
    the resulting order is placed at ``T+1h``.  The two 30-minute phases are
    derived from quarters 1-2 and 3-4, so no incomplete or future candle can
    enter a decision.
    """

    V16_CONFIRM_TIMEFRAME = "15m"

    # Cross-year short-exhaustion neighbourhood.  These are deliberately
    # continuous path measurements rather than a second momentum indicator.
    V16_REJECT_SHORT_EXHAUSTION = True
    V16_SHORT_EXHAUSTION_RISK_SCALE = 1.0
    V16_SHORT_REJECTION_MAX = 0.16
    V16_SHORT_LAST15_SHARE_MIN = 0.15
    V16_SHORT_CLOSE_FROM_EXTREME_MAX = 0.20

    # A slow, orderly grind at an hourly extreme is not rejected outright:
    # it can be the start of a large waterfall.  It is marked for a strictly
    # bounded, post-entry no-follow check instead.
    V16_WATCH_HOUR_DIRECTIONAL_RETURN_MAX = 0.010
    V16_WATCH_CLOSE_FROM_EXTREME_MAX = 0.25
    V16_WATCH_MIN_DIRECTIONAL_QUARTERS = 4

    V16_ENABLE_NO_FOLLOW_EXIT = True
    V16_NO_FOLLOW_STAGE1_MINUTES = 15
    V16_NO_FOLLOW_STAGE1_CURRENT_R = -0.45
    V16_NO_FOLLOW_STAGE1_MAX_R = 0.00
    V16_NO_FOLLOW_STAGE2_MINUTES = 30
    V16_NO_FOLLOW_STAGE2_CURRENT_R = -0.60
    V16_NO_FOLLOW_STAGE2_MAX_R = 0.10
    V16_NO_FOLLOW_WINDOW_MINUTES = 60

    V16_ENABLE_WATCH_PROFIT_FLOOR = True
    V16_WATCH_PROFIT_FLOOR_TRIGGER_R = 1.25
    V16_WATCH_PROFIT_FLOOR_R = 0.10

    @staticmethod
    def _v16_empty_columns(dataframe: DataFrame) -> DataFrame:
        defaults = {
            "v16_mtf_ready": 0,
            "v16_hour_return": np.nan,
            "v16_short_hour_directional_return": np.nan,
            "v16_long_hour_directional_return": np.nan,
            "v16_short_last15_share": np.nan,
            "v16_long_last15_share": np.nan,
            "v16_last15_volume_share": np.nan,
            "v16_short_last15_rejection": np.nan,
            "v16_long_last15_rejection": np.nan,
            "v16_short_close_from_extreme": np.nan,
            "v16_long_close_from_extreme": np.nan,
            "v16_short_directional_quarters": np.nan,
            "v16_long_directional_quarters": np.nan,
            "v16_short_first30_return": np.nan,
            "v16_short_last30_return": np.nan,
            "v16_long_first30_return": np.nan,
            "v16_long_last30_return": np.nan,
            "v16_short_exhaustion": 0,
            "v16_no_follow_watch": 0,
        }
        for column, value in defaults.items():
            if column not in dataframe.columns:
                dataframe[column] = value
        return dataframe

    def informative_pairs(self) -> list[tuple[str, str]]:
        inherited: list[tuple[str, str]] = []
        parent = getattr(super(), "informative_pairs", None)
        if callable(parent):
            inherited = list(parent())
        if self.dp is None:
            return inherited
        pairs = list(self.dp.current_whitelist())
        additions = [(pair, self.V16_CONFIRM_TIMEFRAME) for pair in pairs]
        return list(dict.fromkeys(inherited + additions))

    @staticmethod
    def _v16_quarter_frame(raw: DataFrame) -> DataFrame:
        if raw is None or raw.empty:
            return DataFrame()
        source = raw[["date", "open", "high", "low", "close", "volume"]].copy()
        source["date"] = pd.to_datetime(source["date"], utc=True)
        source = source.sort_values("date").drop_duplicates("date", keep="last")
        source = source.loc[
            source["date"].dt.minute.isin((0, 15, 30, 45))
            & source["date"].dt.second.eq(0)
        ].copy()
        source["v16_hour"] = source["date"].dt.floor("h")
        source["v16_quarter"] = source["date"].dt.minute.floordiv(15) + 1

        pieces: list[DataFrame] = []
        for quarter in range(1, 5):
            piece = source.loc[source["v16_quarter"] == quarter].copy()
            piece = piece.rename(
                columns={
                    value: f"v16_q{quarter}_{value}"
                    for value in ("open", "high", "low", "close", "volume")
                }
            )
            pieces.append(
                piece[
                    ["v16_hour"]
                    + [
                        f"v16_q{quarter}_{value}"
                        for value in ("open", "high", "low", "close", "volume")
                    ]
                ]
            )

        merged = pieces[0]
        for piece in pieces[1:]:
            merged = merged.merge(piece, on="v16_hour", how="inner", validate="one_to_one")
        if merged.empty:
            return merged

        q_close = [merged[f"v16_q{q}_close"] for q in range(1, 5)]
        q_open = [merged[f"v16_q{q}_open"] for q in range(1, 5)]
        quarter_returns = [close / open_.clip(lower=1e-12) - 1.0 for close, open_ in zip(q_close, q_open)]
        progress = [
            q_close[0] / q_open[0].clip(lower=1e-12) - 1.0,
            q_close[1] / q_close[0].clip(lower=1e-12) - 1.0,
            q_close[2] / q_close[1].clip(lower=1e-12) - 1.0,
            q_close[3] / q_close[2].clip(lower=1e-12) - 1.0,
        ]
        absolute_quarter_return = sum(value.abs() for value in quarter_returns).clip(
            lower=1e-12
        )
        total_volume = sum(
            merged[f"v16_q{q}_volume"] for q in range(1, 5)
        ).clip(lower=1e-12)
        hour_high = pd.concat(
            [merged[f"v16_q{q}_high"] for q in range(1, 5)], axis=1
        ).max(axis=1)
        hour_low = pd.concat(
            [merged[f"v16_q{q}_low"] for q in range(1, 5)], axis=1
        ).min(axis=1)
        hour_range = (hour_high - hour_low).clip(lower=1e-12)
        q4_range = (
            merged["v16_q4_high"] - merged["v16_q4_low"]
        ).clip(lower=1e-12)
        hour_return = (
            merged["v16_q4_close"]
            / merged["v16_q1_open"].clip(lower=1e-12)
            - 1.0
        )

        result = DataFrame({"date": merged["v16_hour"]})
        result["v16_mtf_ready"] = 1
        result["v16_hour_return"] = hour_return
        result["v16_short_hour_directional_return"] = -hour_return
        result["v16_long_hour_directional_return"] = hour_return
        result["v16_short_last15_share"] = (
            -quarter_returns[3] / absolute_quarter_return
        )
        result["v16_long_last15_share"] = (
            quarter_returns[3] / absolute_quarter_return
        )
        result["v16_last15_volume_share"] = (
            merged["v16_q4_volume"] / total_volume
        )
        result["v16_short_last15_rejection"] = (
            merged["v16_q4_close"] - merged["v16_q4_low"]
        ) / q4_range
        result["v16_long_last15_rejection"] = (
            merged["v16_q4_high"] - merged["v16_q4_close"]
        ) / q4_range
        result["v16_short_close_from_extreme"] = (
            merged["v16_q4_close"] - hour_low
        ) / hour_range
        result["v16_long_close_from_extreme"] = (
            hour_high - merged["v16_q4_close"]
        ) / hour_range
        result["v16_short_directional_quarters"] = sum(
            (value < 0.0).astype(int) for value in progress
        )
        result["v16_long_directional_quarters"] = sum(
            (value > 0.0).astype(int) for value in progress
        )
        result["v16_short_first30_return"] = -(
            merged["v16_q2_close"]
            / merged["v16_q1_open"].clip(lower=1e-12)
            - 1.0
        )
        result["v16_short_last30_return"] = -(
            merged["v16_q4_close"]
            / merged["v16_q3_open"].clip(lower=1e-12)
            - 1.0
        )
        result["v16_long_first30_return"] = -result[
            "v16_short_first30_return"
        ]
        result["v16_long_last30_return"] = -result[
            "v16_short_last30_return"
        ]
        return result

    def _v16_attach_intrahour_path(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe = self._v16_empty_columns(dataframe)
        if self.dp is None:
            return dataframe
        pair = str(metadata.get("pair") or "")
        if not pair:
            return dataframe
        try:
            raw = self.dp.get_pair_dataframe(
                pair=pair,
                timeframe=self.V16_CONFIRM_TIMEFRAME,
            )
        except Exception:
            return dataframe
        quarters = self._v16_quarter_frame(raw)
        if quarters.empty:
            return dataframe

        base = dataframe.drop(
            columns=[column for column in quarters.columns if column != "date"],
            errors="ignore",
        ).copy()
        base["date"] = pd.to_datetime(base["date"], utc=True)
        return base.merge(quarters, on="date", how="left", validate="one_to_one")

    def _v16_mark_path_states(self, dataframe: DataFrame) -> DataFrame:
        ready = dataframe["v16_mtf_ready"].fillna(0).astype(int).eq(1)
        short = dataframe["bo_entry_short"].astype(bool)
        exhaustion = (
            short
            & ready
            & (
                dataframe["v16_short_last15_rejection"]
                <= float(self.V16_SHORT_REJECTION_MAX)
            )
            & (
                dataframe["v16_short_last15_share"]
                >= float(self.V16_SHORT_LAST15_SHARE_MIN)
            )
            & (
                dataframe["v16_short_close_from_extreme"]
                <= float(self.V16_SHORT_CLOSE_FROM_EXTREME_MAX)
            )
        )
        watch = (
            short
            & ready
            & (
                dataframe["v16_short_hour_directional_return"]
                <= float(self.V16_WATCH_HOUR_DIRECTIONAL_RETURN_MAX)
            )
            & (
                dataframe["v16_short_close_from_extreme"]
                <= float(self.V16_WATCH_CLOSE_FROM_EXTREME_MAX)
            )
            & (
                dataframe["v16_short_directional_quarters"]
                >= int(self.V16_WATCH_MIN_DIRECTIONAL_QUARTERS)
            )
        )
        dataframe["v16_short_exhaustion"] = exhaustion.astype(int)
        dataframe["v16_no_follow_watch"] = watch.astype(int)
        if self.V16_REJECT_SHORT_EXHAUSTION:
            dataframe.loc[exhaustion, "bo_entry_short"] = 0
            dataframe["bo_entry"] = (
                dataframe["bo_entry_long"].astype(bool)
                | dataframe["bo_entry_short"].astype(bool)
            ).astype(int)
        return dataframe

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe = self._v16_attach_intrahour_path(dataframe, metadata)
        return self._v16_mark_path_states(dataframe)

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
        scale = float(self.V16_SHORT_EXHAUSTION_RISK_SCALE)
        if (
            stake <= 0.0
            or scale >= 1.0
            or self._component(entry_tag) != "breakout"
            or side != "short"
        ):
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None or int(row.get("v16_short_exhaustion", 0)) != 1:
            return stake
        scaled = min(float(max_stake), max(0.0, float(stake) * scale))
        if min_stake is not None and scaled < float(min_stake):
            if float(min_stake) > float(max_stake):
                return 0.0
            return float(min_stake)
        return scaled

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
            order.ft_order_side != trade.entry_side
            or trade.nr_of_successful_entries != 1
            or self._component(trade.enter_tag) != "breakout"
        ):
            return
        row = self._latest_signal_row(pair, "breakout", current_time)
        watched = bool(
            row is not None
            and int(row.get("v16_no_follow_watch", 0)) == 1
        )
        trade.set_custom_data("v16_no_follow_watch", watched)

    def _v16_no_follow_reason(
        self,
        holding_minutes: int,
        current_r: float,
        maximum_r: float,
    ) -> str | None:
        """Return a bounded failure reason without consulting future bars."""
        if holding_minutes > int(self.V16_NO_FOLLOW_WINDOW_MINUTES):
            return None
        if (
            holding_minutes >= int(self.V16_NO_FOLLOW_STAGE1_MINUTES)
            and current_r <= float(self.V16_NO_FOLLOW_STAGE1_CURRENT_R)
            and maximum_r <= float(self.V16_NO_FOLLOW_STAGE1_MAX_R)
        ):
            return "bo_v16_no_follow_15m"
        if (
            holding_minutes >= int(self.V16_NO_FOLLOW_STAGE2_MINUTES)
            and current_r <= float(self.V16_NO_FOLLOW_STAGE2_CURRENT_R)
            and maximum_r <= float(self.V16_NO_FOLLOW_STAGE2_MAX_R)
        ):
            return "bo_v16_no_follow_30m"
        return None

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        inherited = super().custom_exit(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
            **kwargs,
        )
        if inherited is not None:
            return inherited
        if (
            not self.V16_ENABLE_NO_FOLLOW_EXIT
            or self._component(trade.enter_tag) != "breakout"
            or not bool(trade.get_custom_data("v16_no_follow_watch", False))
        ):
            return None

        holding_minutes = int(
            (current_time - self._trade_open_time(trade)).total_seconds() / 60
        )
        favorable_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate if favorable_value is None else favorable_value
        )
        current_r = self._trade_r(trade, current_rate)
        maximum_r = self._trade_r(trade, favorable_rate)
        return self._v16_no_follow_reason(
            holding_minutes,
            current_r,
            maximum_r,
        )

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
        inherited = super().custom_stoploss(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
            after_fill,
            **kwargs,
        )
        if (
            not self.V16_ENABLE_WATCH_PROFIT_FLOOR
            or self._component(trade.enter_tag) != "breakout"
            or not bool(trade.get_custom_data("v16_no_follow_watch", False))
        ):
            return inherited
        favorable_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate if favorable_value is None else favorable_value
        )
        maximum_r = self._trade_r(trade, favorable_rate)
        if maximum_r < float(self.V16_WATCH_PROFIT_FLOOR_TRIGGER_R):
            return inherited

        unit_risk = max(
            float(trade.get_custom_data("initial_unit_risk", 0.0)),
            1e-12,
        )
        side = -1.0 if trade.is_short else 1.0
        stop_rate = trade.open_rate + side * (
            float(self.V16_WATCH_PROFIT_FLOOR_R) * unit_risk
            + 2.0 * self.SIDE_COST * trade.open_rate
        )
        protected = abs(
            float(
                stoploss_from_absolute(
                    stop_rate,
                    current_rate=current_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage,
                )
            )
        )
        if inherited is None:
            return protected
        return min(abs(float(inherited)), protected)


class BreakoutV16Fixed50MtfObserveBreakoutMax2ResearchFreqtrade(
    _V16IntrahourPathMixin,
    BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade,
):
    """Ablation control: compute MTF state but retain every V15 decision."""

    V16_REJECT_SHORT_EXHAUSTION = False
    V16_ENABLE_NO_FOLLOW_EXIT = False
    V16_ENABLE_WATCH_PROFIT_FLOOR = False
    STRATEGY_VERSION = "breakout_v16_fixed50_mtf_observe_max2_20260811"
    ADAPTIVE_STATE_BASENAME = "breakout_v16_fixed50_mtf_observe_max2"


class BreakoutV16Fixed50MtfEntryBreakoutMax2ResearchFreqtrade(
    _V16IntrahourPathMixin,
    BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade,
):
    """Entry-filter ablation without the new post-entry management."""

    V16_ENABLE_NO_FOLLOW_EXIT = False
    V16_ENABLE_WATCH_PROFIT_FLOOR = False
    STRATEGY_VERSION = "breakout_v16_fixed50_mtf_entry_max2_20260811"
    ADAPTIVE_STATE_BASENAME = "breakout_v16_fixed50_mtf_entry_max2"


class BreakoutV16Fixed50MtfFailureBreakoutMax2ResearchFreqtrade(
    _V16IntrahourPathMixin,
    BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade,
):
    """Post-entry ablation without rejecting any V15 signal."""

    V16_REJECT_SHORT_EXHAUSTION = False
    V16_ENABLE_WATCH_PROFIT_FLOOR = False
    STRATEGY_VERSION = "breakout_v16_fixed50_mtf_failure_max2_20260811"
    ADAPTIVE_STATE_BASENAME = "breakout_v16_fixed50_mtf_failure_max2"


class BreakoutV16Fixed50MtfProtectedBreakoutMax2ResearchFreqtrade(
    _V16IntrahourPathMixin,
    BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade,
):
    """Full central V16 candidate: entry, no-follow and protected runner."""

    STRATEGY_VERSION = "breakout_v16_fixed50_mtf_protected_max2_20260811"
    ADAPTIVE_STATE_BASENAME = "breakout_v16_fixed50_mtf_protected_max2"


class BreakoutV16Fixed50MtfFailureProtectedBreakoutMax2ResearchFreqtrade(
    _V16IntrahourPathMixin,
    BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade,
):
    """Keep all V15 entries; add only bounded failure and profit protection."""

    V16_REJECT_SHORT_EXHAUSTION = False
    STRATEGY_VERSION = (
        "breakout_v16_fixed50_mtf_failure_protected_max2_20260811"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v16_fixed50_mtf_failure_protected_max2"
    )


class BreakoutV16Fixed50MtfFailureProtectedFast10BreakoutMax2ResearchFreqtrade(
    BreakoutV16Fixed50MtfFailureProtectedBreakoutMax2ResearchFreqtrade,
):
    """Check the unchanged -0.45R failure boundary after ten minutes."""

    V16_NO_FOLLOW_STAGE1_MINUTES = 10
    STRATEGY_VERSION = (
        "breakout_v16_fixed50_mtf_failure_protected_fast10_max2_20260811"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v16_fixed50_mtf_failure_protected_fast10_max2"
    )


class BreakoutV16Fixed50MtfRisk25BreakoutMax2ResearchFreqtrade(
    _V16IntrahourPathMixin,
    BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade,
):
    """Preserve slot occupancy while reducing the exhausted-path risk."""

    V16_REJECT_SHORT_EXHAUSTION = False
    V16_SHORT_EXHAUSTION_RISK_SCALE = 0.25
    V16_ENABLE_NO_FOLLOW_EXIT = False
    V16_ENABLE_WATCH_PROFIT_FLOOR = False
    STRATEGY_VERSION = "breakout_v16_fixed50_mtf_risk25_max2_20260811"
    ADAPTIVE_STATE_BASENAME = "breakout_v16_fixed50_mtf_risk25_max2"


class BreakoutV16Fixed50MtfRisk50BreakoutMax2ResearchFreqtrade(
    BreakoutV16Fixed50MtfRisk25BreakoutMax2ResearchFreqtrade,
):
    V16_SHORT_EXHAUSTION_RISK_SCALE = 0.50
    STRATEGY_VERSION = "breakout_v16_fixed50_mtf_risk50_max2_20260811"
    ADAPTIVE_STATE_BASENAME = "breakout_v16_fixed50_mtf_risk50_max2"


class BreakoutV16Fixed50MtfRisk75BreakoutMax2ResearchFreqtrade(
    BreakoutV16Fixed50MtfRisk25BreakoutMax2ResearchFreqtrade,
):
    V16_SHORT_EXHAUSTION_RISK_SCALE = 0.75
    STRATEGY_VERSION = "breakout_v16_fixed50_mtf_risk75_max2_20260811"
    ADAPTIVE_STATE_BASENAME = "breakout_v16_fixed50_mtf_risk75_max2"


class BreakoutV16Fixed50MtfRisk25ProtectedBreakoutMax2ResearchFreqtrade(
    _V16IntrahourPathMixin,
    BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade,
):
    V16_REJECT_SHORT_EXHAUSTION = False
    V16_SHORT_EXHAUSTION_RISK_SCALE = 0.25
    STRATEGY_VERSION = (
        "breakout_v16_fixed50_mtf_risk25_protected_max2_20260811"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v16_fixed50_mtf_risk25_protected_max2"
    )


class BreakoutV16Fixed50MtfRisk75ProtectedBreakoutMax2ResearchFreqtrade(
    _V16IntrahourPathMixin,
    BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade,
):
    """Stage-2 survivor: mild risk reduction plus bounded protection."""

    V16_REJECT_SHORT_EXHAUSTION = False
    V16_SHORT_EXHAUSTION_RISK_SCALE = 0.75
    STRATEGY_VERSION = (
        "breakout_v16_fixed50_mtf_risk75_protected_max2_20260811"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v16_fixed50_mtf_risk75_protected_max2"
    )


class BreakoutV16Fixed50MtfAdaptiveBreakoutMax2ResearchFreqtrade(
    BreakoutV16Fixed50MtfFailureProtectedBreakoutMax2ResearchFreqtrade,
):
    """Walk-forward selection: keep V15 entries, manage confirmed path failure."""

    STRATEGY_VERSION = "breakout_v16_fixed50_mtf_adaptive_max2_20260811"
    ADAPTIVE_STATE_BASENAME = "breakout_v16_fixed50_mtf_adaptive_max2"

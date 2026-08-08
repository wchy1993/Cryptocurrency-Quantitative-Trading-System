from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import Order, Trade


FROZEN_STRATEGY_DIR = (
    Path(__file__).resolve().parents[3]
    / "freqtrade_breakout_v9_grid_v7"
    / "user_data"
    / "strategies"
)
if str(FROZEN_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(FROZEN_STRATEGY_DIR))

from BreakoutV9GridV7Freqtrade import (  # noqa: E402
    BreakoutV9GridV7Freqtrade,
    _causal_signal_cap,
)


class GridV8DualSideFreqtrade(BreakoutV9GridV7Freqtrade):
    """Independent long/short extension of the frozen Grid v7 campaign.

    The short signal starts from the frozen v7 implementation.  The long
    signal is a directionally symmetric implementation with independently
    configurable gates and risk.  Breakout is disabled so every result is a
    standalone Grid campaign.
    """

    MAX_OPEN_TRADES = 1

    # Diagnostic starting point.  These are deliberately explicit so every
    # candidate archived by Freqtrade records the tested risk surface.
    GRID_SHORT_RISK_PCT = 0.1095
    GRID_LONG_RISK_PCT = 0.0600
    GRID_SHORT_RANK_OFFSET = 0.0
    GRID_LONG_RANK_OFFSET = 0.0

    GRID_SHORT_MIN_SCORE = 3
    GRID_LONG_MIN_SCORE = 4
    GRID_LONG_MIN_QUALITY = 0.52
    GRID_LONG_MIN_ALIGNMENT = 0.52
    GRID_LONG_MIN_EXTENSION = 0.10
    GRID_LONG_MAX_EXTENSION = 0.42
    GRID_LONG_MIN_CLOSE_POSITION = 0.62
    GRID_LONG_MIN_VOLUME_RATIO = 0.72
    GRID_LONG_MAX_VOLUME_RATIO = 1.35
    GRID_LONG_MIN_BREADTH = 0.30
    GRID_LONG_MAX_BREADTH = 0.72
    GRID_LONG_MAX_REGIME = 0.30
    GRID_LONG_MIN_MARKET_EFFICIENCY = 0.18
    GRID_LONG_MIN_SYMBOL_EFFICIENCY = -0.05

    @staticmethod
    def _component(tag: str | None) -> str:
        value = str(tag or "")
        if value.startswith("grid_v8"):
            return "grid"
        return BreakoutV9GridV7Freqtrade._component(tag)

    def _populate_breakout(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_breakout(dataframe)
        dataframe["bo_entry_long"] = 0
        dataframe["bo_entry_short"] = 0
        dataframe["bo_entry"] = 0
        return dataframe

    def _short_refinement_mask(self, dataframe: DataFrame) -> pd.Series:
        return pd.Series(True, index=dataframe.index)

    def _populate_grid(self, dataframe: DataFrame) -> DataFrame:
        # Preserve the complete frozen short feature/signal path first.
        dataframe = super()._populate_grid(dataframe)
        dataframe["grid_entry_short"] = (
            dataframe["grid_entry_short"].astype(bool)
            & (dataframe["grid_score"] >= self.GRID_SHORT_MIN_SCORE)
            & self._short_refinement_mask(dataframe)
        ).astype(int)

        atr = dataframe["atr"].clip(lower=1e-12)
        dataframe["grid_long_alignment"] = (
            dataframe["fast_ema"] - dataframe["slow_ema"]
        ) / atr
        dataframe["grid_long_extension"] = (
            dataframe["close"] - dataframe["fast_ema"]
        ) / atr
        candle_range = (dataframe["high"] - dataframe["low"]).clip(
            lower=1e-12
        )
        dataframe["grid_long_close_position"] = (
            dataframe["close"] - dataframe["low"]
        ) / candle_range

        long_trend = (
            (dataframe["fast_ema"] > dataframe["slow_ema"])
            & (dataframe["grid_fast_slope"] >= 0.0)
            & (dataframe["grid_slow_slope"] >= 0.05)
            & dataframe["grid_long_alignment"].between(0.3, 4.0)
            & (dataframe["atr"] / dataframe["close"]).between(0.001, 0.08)
        )
        pullback_touch = (
            (
                dataframe["low"]
                <= dataframe["fast_ema"] + 0.30 * dataframe["atr"]
            )
            & (dataframe["close"] > dataframe["fast_ema"])
        )
        raw_signal = (
            long_trend
            & dataframe["grid_long_extension"].between(
                self.GRID_LONG_MIN_EXTENSION,
                0.65,
            )
            & (
                dataframe["grid_long_close_position"]
                >= self.GRID_LONG_MIN_CLOSE_POSITION
            )
            & dataframe["grid_volume_ratio"].between(
                self.GRID_LONG_MIN_VOLUME_RATIO,
                2.0,
            )
            & pullback_touch
        )
        raw_signal = _causal_signal_cap(
            raw_signal,
            dataframe["date"],
            daily_limit=2,
            minimum_bar_gap=12,
        ).astype(bool)

        dataframe["grid_long_regime"] = self._regime_score(dataframe, 1.0)
        context_ready = dataframe[
            [
                "breadth",
                "market_efficiency",
                "btc_return_4h",
                "eth_return_4h",
                "symbol_efficiency_12h",
                "symbol_ema55_atr",
            ]
        ].notna().all(axis=1)
        overlay = (
            dataframe["btc_return_4h"].between(-0.01, 0.01)
            & (dataframe["eth_return_4h"] <= 0.015)
            & dataframe["breadth"].between(
                self.GRID_LONG_MIN_BREADTH,
                self.GRID_LONG_MAX_BREADTH,
            )
            & dataframe["symbol_return_4h"].between(-0.015, 0.08)
            & (
                dataframe["symbol_efficiency_12h"]
                >= self.GRID_LONG_MIN_SYMBOL_EFFICIENCY
            )
            & (
                dataframe["market_efficiency"]
                >= self.GRID_LONG_MIN_MARKET_EFFICIENCY
            )
            & dataframe["symbol_ema55_atr"].between(-0.5, 2.0)
            & (
                dataframe["grid_long_regime"]
                <= self.GRID_LONG_MAX_REGIME
            )
        )
        dataframe["grid_long_quality"] = (
            0.30
            * (dataframe["grid_fast_slope"] + 0.25).clip(
                lower=0.0, upper=2.0
            )
            + 0.20
            * (dataframe["grid_slow_slope"] + 0.25).clip(
                lower=0.0, upper=2.0
            )
            + 0.20
            * dataframe["grid_long_alignment"].clip(
                lower=0.0, upper=2.0
            )
            + 0.15
            * (
                1.0
                - dataframe["grid_long_extension"] / 0.65
            ).clip(lower=0.0, upper=1.0)
            + 0.10
            * dataframe["grid_volume_ratio"].clip(
                lower=0.0, upper=1.5
            )
            + 0.05 * dataframe["grid_long_close_position"]
        )
        dataframe["grid_long_score"] = (
            (
                dataframe["grid_long_quality"]
                >= self.GRID_LONG_MIN_QUALITY
            ).astype(int)
            + (
                dataframe["grid_long_alignment"]
                >= self.GRID_LONG_MIN_ALIGNMENT
            ).astype(int)
            + (
                dataframe["grid_long_extension"]
                >= self.GRID_LONG_MIN_EXTENSION
            ).astype(int)
            + (
                dataframe["grid_long_extension"]
                <= self.GRID_LONG_MAX_EXTENSION
            ).astype(int)
            + (
                dataframe["grid_volume_ratio"]
                <= self.GRID_LONG_MAX_VOLUME_RATIO
            ).astype(int)
            + (
                dataframe["grid_long_regime"]
                <= self.GRID_LONG_MAX_REGIME
            ).astype(int)
        )
        dataframe["grid_entry_long"] = (
            raw_signal
            & overlay
            & context_ready
            & (
                dataframe["grid_long_score"]
                >= self.GRID_LONG_MIN_SCORE
            )
        ).astype(int)
        dataframe["grid_entry"] = (
            dataframe["grid_entry_long"].astype(bool)
            | dataframe["grid_entry_short"].astype(bool)
        ).astype(int)
        dataframe["grid_rank"] = np.where(
            dataframe["grid_entry_long"].astype(bool),
            dataframe["grid_long_extension"] + self.GRID_LONG_RANK_OFFSET,
            dataframe["grid_extension"] + self.GRID_SHORT_RANK_OFFSET,
        )
        return dataframe

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        grid_long = dataframe["grid_entry_long"].astype(bool)
        grid_short = dataframe["grid_entry_short"].astype(bool)
        dataframe.loc[grid_long, "enter_long"] = 1
        dataframe.loc[grid_short, "enter_short"] = 1
        for index in dataframe.index[grid_long]:
            score = int(dataframe.at[index, "grid_long_score"])
            dataframe.at[index, "enter_tag"] = f"grid_v8_long_s{score}"
        for index in dataframe.index[grid_short]:
            score = int(dataframe.at[index, "grid_score"])
            dataframe.at[index, "enter_tag"] = f"grid_v8_short_s{score}"

        pair = str(metadata.get("pair") or "")
        if pair:
            ranks: dict[tuple[str, int], float] = {}
            available_times = (
                pd.to_datetime(dataframe["date"], utc=True)
                + pd.Timedelta(self.timeframe)
            )
            for index in dataframe.index[grid_long | grid_short]:
                ranks[
                    (
                        "grid",
                        int(available_times.at[index].timestamp()),
                    )
                ] = float(dataframe.at[index, "grid_rank"])
            self._candidate_ranks[pair] = ranks
        return dataframe

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
        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None or current_rate <= 0.0:
            return 0.0

        equity = float(self.wallets.get_total_stake_amount())
        if equity <= 0.0:
            return 0.0
        leverage = max(1.0, leverage)
        atr_value = max(float(row.get("atr", 0.0)), 1e-12)
        is_short = side == "short"
        direction = -1.0 if is_short else 1.0
        slow = float(row.get("slow_ema", current_rate))
        if is_short:
            hard_stop = max(
                current_rate + self.GRID_HARD_STOP_ATR * atr_value,
                slow + self.GRID_SLOW_BUFFER_ATR * atr_value,
            )
        else:
            hard_stop = min(
                current_rate - self.GRID_HARD_STOP_ATR * atr_value,
                slow - self.GRID_SLOW_BUFFER_ATR * atr_value,
            )
        levels = [
            current_rate
            - direction * level * self.GRID_SPACING_ATR * atr_value
            for level in range(self.GRID_LEVELS + 1)
        ]
        unit_losses = [
            max(0.0, direction * (price - hard_stop))
            + 2.0 * self.SIDE_COST * price
            for price in levels
        ]
        risk_pct = (
            self.GRID_SHORT_RISK_PCT
            if is_short
            else self.GRID_LONG_RISK_PCT
        )
        base_quantity = (
            equity * risk_pct / max(sum(unit_losses), 1e-12)
        )
        total_notional = base_quantity * sum(levels)
        if total_notional > equity * self.GRID_MAX_NOTIONAL:
            base_quantity *= (
                equity * self.GRID_MAX_NOTIONAL / total_notional
            )
        initial_notional = base_quantity * levels[0]
        return min(max_stake, max(0.0, initial_notional / leverage))

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        component = self._component(trade.enter_tag)
        if (
            component == "grid"
            and order.ft_order_side == trade.entry_side
            and trade.nr_of_successful_entries == 1
        ):
            row = self._latest_signal_row(pair, "grid", current_time)
            if row is None:
                return
            atr_value = max(float(row.get("atr", 0.0)), 1e-12)
            entry_rate = self._safe_order_price(order, trade.open_rate)
            direction = -1.0 if trade.is_short else 1.0
            slow = float(row.get("slow_ema", entry_rate))
            spacing = self.GRID_SPACING_ATR * atr_value
            if trade.is_short:
                hard_stop = max(
                    entry_rate + self.GRID_HARD_STOP_ATR * atr_value,
                    slow + self.GRID_SLOW_BUFFER_ATR * atr_value,
                )
            else:
                hard_stop = min(
                    entry_rate - self.GRID_HARD_STOP_ATR * atr_value,
                    slow - self.GRID_SLOW_BUFFER_ATR * atr_value,
                )
            level_prices = [
                entry_rate - direction * level * spacing
                for level in range(self.GRID_LEVELS + 1)
            ]
            initial_stake = self._safe_order_stake(
                order, trade.stake_amount
            )
            quantity = max(float(trade.amount), 0.0)
            risk_budget = quantity * sum(
                max(0.0, direction * (price - hard_stop))
                + 2.0 * self.SIDE_COST * price
                for price in level_prices
            )
            trade.set_custom_data("component", "grid")
            trade.set_custom_data("initial_atr", atr_value)
            trade.set_custom_data("grid_side", trade.trade_direction)
            trade.set_custom_data("grid_anchor", entry_rate)
            trade.set_custom_data("grid_spacing", spacing)
            trade.set_custom_data("grid_hard_stop", hard_stop)
            trade.set_custom_data("grid_level_prices", level_prices)
            trade.set_custom_data("grid_base_stake", initial_stake)
            trade.set_custom_data("grid_risk_budget", risk_budget)
            trade.set_custom_data("grid_entry_count", 1)
            trade.set_custom_data("grid_tp_count", 0)
            trade.set_custom_data(
                "grid_cycles",
                {
                    str(level): 0
                    for level in range(self.GRID_LEVELS + 1)
                },
            )
            trade.set_custom_data(
                "grid_lots",
                {
                    "0": {
                        "entry": entry_rate,
                        "stake": initial_stake,
                    }
                },
            )
            trade.set_custom_data("grid_invalid_bars", 0)
            trade.set_custom_data("grid_last_regime_date", "")
            return
        super().order_filled(
            pair,
            trade,
            order,
            current_time,
            **kwargs,
        )

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
        if isinstance(reason, str) and reason.startswith("grid_v7_"):
            return reason.replace("grid_v7_", "grid_v8_", 1)
        return reason


class GridV8DualSideRefinedA(GridV8DualSideFreqtrade):
    """First structural refinement from the diagnostic attribution."""

    GRID_LONG_MIN_SCORE = 6
    GRID_LONG_MIN_MARKET_EFFICIENCY = 0.20
    GRID_LONG_MIN_SYMBOL_EFFICIENCY = 0.10

    def _short_refinement_mask(self, dataframe: DataFrame) -> pd.Series:
        atr_pct = dataframe["atr"] / dataframe["close"].clip(lower=1e-12)
        score = dataframe["grid_score"].astype(int)

        # Reject grid entries after an already-exhausted sell impulse and
        # contracts whose volatility is too small for the modeled costs.
        general = (
            (atr_pct >= 0.005)
            & (dataframe["grid_fast_slope"] >= -0.25)
        )

        # Low-score setups need an established decline and must not enter at
        # the far edge of the permitted pullback extension.
        score_three = (
            (score != 3)
            | (
                (dataframe["grid_fast_slope"] <= -0.075)
                & (dataframe["grid_extension"] <= 0.60)
            )
        )

        # Score-four trades can qualify through symbol quality or through a
        # sufficiently efficient broad market, avoiding weak choppy entries.
        score_four = (
            (score != 4)
            | (dataframe["grid_quality"] >= 0.44)
            | (dataframe["market_efficiency"] >= 0.25)
        )

        # A high-volatility score-six signal is accepted only when the fast
        # trend is still decisively falling instead of already flattening.
        score_six = (
            (score != 6)
            | (atr_pct <= 0.018)
            | (dataframe["grid_fast_slope"] <= -0.08)
        )
        return general & score_three & score_four & score_six


class GridV8DualSideRefinedB(GridV8DualSideRefinedA):
    """Crowding and weak-close refinement without changing risk."""

    GRID_LONG_MIN_CLOSE_POSITION = 0.68
    GRID_SHORT_SCORE4_MIN_QUALITY = 0.44
    GRID_SHORT_SCORE4_MIN_MARKET_EFFICIENCY = 0.20
    GRID_SHORT_SCORE4_MARKET_OVERRIDE = 0.30
    GRID_SHORT_SCORE5_MAX_CROWDED_EXTENSION = 0.60
    GRID_SHORT_SCORE5_MIN_CROWDED_REGIME = -0.25
    GRID_SHORT_SCORE5_MAX_VOLUME = 1.50
    GRID_SHORT_SCORE5_VOLUME_QUALITY_OVERRIDE = 0.60

    def _short_refinement_mask(self, dataframe: DataFrame) -> pd.Series:
        inherited = super()._short_refinement_mask(dataframe)
        score = dataframe["grid_score"].astype(int)
        quality = dataframe["grid_quality"]
        market_efficiency = dataframe["market_efficiency"]

        # Four-point entries need both acceptable symbol quality and market
        # efficiency, unless the broad market is exceptionally efficient.
        score_four = (
            (score != 4)
            | (
                (
                    quality
                    >= self.GRID_SHORT_SCORE4_MIN_QUALITY
                )
                & (
                    market_efficiency
                    >= self.GRID_SHORT_SCORE4_MIN_MARKET_EFFICIENCY
                )
            )
            | (
                market_efficiency
                >= self.GRID_SHORT_SCORE4_MARKET_OVERRIDE
            )
        )

        # Avoid a deep pullback entry when the directional regime is already
        # crowded.  A high-volume score-five entry is retained only when its
        # full quality score confirms the move.
        score_five = (
            (score != 5)
            | ~(
                (
                    dataframe["grid_extension"]
                    > self.GRID_SHORT_SCORE5_MAX_CROWDED_EXTENSION
                )
                & (
                    dataframe["grid_regime"]
                    < self.GRID_SHORT_SCORE5_MIN_CROWDED_REGIME
                )
            )
        ) & (
            (score != 5)
            | (
                dataframe["grid_volume_ratio"]
                <= self.GRID_SHORT_SCORE5_MAX_VOLUME
            )
            | (
                quality
                >= self.GRID_SHORT_SCORE5_VOLUME_QUALITY_OVERRIDE
            )
        )
        return inherited & score_four & score_five


class GridV8DualSideNeighborLoose(GridV8DualSideRefinedB):
    """One-step looser neighborhood used only for robustness validation."""

    GRID_LONG_MIN_CLOSE_POSITION = 0.67
    GRID_LONG_MIN_MARKET_EFFICIENCY = 0.195
    GRID_LONG_MIN_SYMBOL_EFFICIENCY = 0.09
    GRID_SHORT_SCORE4_MIN_QUALITY = 0.435
    GRID_SHORT_SCORE4_MIN_MARKET_EFFICIENCY = 0.195
    GRID_SHORT_SCORE4_MARKET_OVERRIDE = 0.295
    GRID_SHORT_SCORE5_MAX_CROWDED_EXTENSION = 0.61
    GRID_SHORT_SCORE5_MIN_CROWDED_REGIME = -0.275
    GRID_SHORT_SCORE5_MAX_VOLUME = 1.52
    GRID_SHORT_SCORE5_VOLUME_QUALITY_OVERRIDE = 0.59


class GridV8DualSideSelected(GridV8DualSideRefinedB):
    """Frozen selected long/short Grid v8 candidate."""

    pass

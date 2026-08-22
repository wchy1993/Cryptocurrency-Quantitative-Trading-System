from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from pandas import DataFrame
from freqtrade.strategy import Trade, stoploss_from_absolute

from BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade import (
    BreakoutV16GridV15PrecisionGuardGlobalResearchFreqtrade,
)


class _SplitTrailRunnerMixin:
    """Separate loss paths by sleeve and retain confirmed Breakout runners.

    The overlay deliberately leaves entries, leverage, stake sizing, the
    precision initial stop, Breakout-long stops, Grid-long stops and Grid DCA
    untouched.  It changes only:

    * non-capture Breakout shorts receive their own completed-hour profit
      floors;
    * Grid shorts can protect a small campaign loss after first proving a
      meaningful campaign profit;
    * score-four/five Breakouts can outlive the frozen 20-hour time stop while
      completed 1h structure still confirms the trend.
    """

    SPLIT_RUNNER_ENABLED = False

    # Breakout-short protection is intentionally based on completed 1h closes
    # (`_confirmed_peak_r`) rather than transient 1m extremes.  Capture shorts
    # retain their existing, tighter 3R/4R floors from the parent strategy.
    BO_SHORT_SPLIT_TRIGGER_1_R = 3.0
    BO_SHORT_SPLIT_FLOOR_1_R = -0.25
    BO_SHORT_SPLIT_TRIGGER_2_R = 5.0
    BO_SHORT_SPLIT_FLOOR_2_R = 0.25

    # Grid uses campaign PnL, including realized partial take-profits.  Once a
    # short campaign has made +0.25 campaign-R, a later reversal is bounded at
    # -0.10R.  The original -0.70R hard stop remains armed underneath it.
    GRID_SHORT_SPLIT_TRIGGER_R = 0.25
    GRID_SHORT_SPLIT_FLOOR_R = -0.10
    GRID_SHORT_SPLIT_MIN_TP_COUNT = 0
    GRID_SHORT_SPLIT_EXIT_REASON = "grid_v16_short_profit_floor"

    # These are the already-audited V10D structural runner conditions.  Only
    # the bounded holding window is varied by the concrete research classes.
    BO_RUNNER_MIN_SCORE = 4
    BO_RUNNER_MIN_R = 2.0
    BO_RUNNER_MAX_HOLD_MINUTES = 2160
    BO_RUNNER_MIN_EMA55_ATR = 1.50
    BO_RUNNER_MIN_EMA_SPREAD_ATR = 0.50
    BO_RUNNER_MIN_RETURN_8H = -0.03
    BO_RUNNER_MIN_DIRECTIONAL_BREADTH = 0.0
    BO_RUNNER_MOMENTUM_RETURN_4H = 0.02
    BO_RUNNER_MOMENTUM_MARKET_EFFICIENCY = 0.40
    BO_RUNNER_PROTECT_ARMED_PROFIT = False
    BO_RUNNER_ARMED_FLOOR_FRACTION = 0.50
    BO_RUNNER_MIN_FLOOR_R = 1.0

    @staticmethod
    def _split_directional_value(trade: Trade, value: float) -> float:
        return -value if trade.is_short else value

    @staticmethod
    def _split_tighter_stop(
        inherited: float | None,
        protected: float | None,
    ) -> float | None:
        if protected is None or protected <= 0.0:
            return inherited
        if inherited is None:
            return protected
        return min(abs(float(inherited)), abs(float(protected)))

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        # Research-only runner validation.  The shifted value is causal and is
        # not referenced by any entry or allocation method.
        dataframe["split_close_8h_ago"] = dataframe["close"].shift(8)
        return dataframe

    def _split_runner_environment_valid(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
    ) -> bool:
        row = self._latest_row(pair, current_time)
        if row is None:
            return False

        atr = max(float(row["atr"]), 1e-12)
        directional_ema55 = self._split_directional_value(
            trade,
            float(row["symbol_ema55_atr"]),
        )
        directional_spread = self._split_directional_value(
            trade,
            (float(row["fast_ema"]) - float(row["slow_ema"])) / atr,
        )
        directional_return_4h = self._split_directional_value(
            trade,
            float(row["symbol_return_4h"]),
        )
        close_8h_ago = float(row.get("split_close_8h_ago", np.nan))
        if not np.isfinite(close_8h_ago) or close_8h_ago <= 0.0:
            return False
        directional_return_8h = self._split_directional_value(
            trade,
            float(row["close"]) / close_8h_ago - 1.0,
        )
        breadth = float(row["breadth"])
        directional_breadth = (
            0.5 - breadth if trade.is_short else breadth - 0.5
        )
        broad_support = (
            directional_breadth
            >= float(self.BO_RUNNER_MIN_DIRECTIONAL_BREADTH)
        )
        momentum_support = (
            directional_return_4h
            >= float(self.BO_RUNNER_MOMENTUM_RETURN_4H)
            and float(row["market_efficiency"])
            >= float(self.BO_RUNNER_MOMENTUM_MARKET_EFFICIENCY)
        )
        return bool(
            directional_ema55 >= float(self.BO_RUNNER_MIN_EMA55_ATR)
            and directional_spread
            >= float(self.BO_RUNNER_MIN_EMA_SPREAD_ATR)
            and directional_return_8h
            >= float(self.BO_RUNNER_MIN_RETURN_8H)
            and (broad_support or momentum_support)
        )

    def _split_breakout_short_stop(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
    ) -> float | None:
        # Capture entries already have a stronger parent schedule.  Applying a
        # second floor would only duplicate it and obscure attribution.
        if bool(trade.get_custom_data("bo_capture", False)):
            return None
        confirmed_peak = self._confirmed_peak_r(pair, trade, current_time)
        locked_r: float | None = None
        if confirmed_peak >= float(self.BO_SHORT_SPLIT_TRIGGER_1_R):
            locked_r = float(self.BO_SHORT_SPLIT_FLOOR_1_R)
        if confirmed_peak >= float(self.BO_SHORT_SPLIT_TRIGGER_2_R):
            locked_r = max(
                locked_r if locked_r is not None else -np.inf,
                float(self.BO_SHORT_SPLIT_FLOOR_2_R),
            )
        if locked_r is None:
            return None

        unit_risk = max(
            float(trade.get_custom_data("initial_unit_risk", 0.0)),
            1e-12,
        )
        stop_rate = trade.open_rate - (
            locked_r * unit_risk
            + 2.0 * float(self.SIDE_COST) * trade.open_rate
        )
        return abs(
            float(
                stoploss_from_absolute(
                    stop_rate,
                    current_rate=current_rate,
                    is_short=True,
                    leverage=trade.leverage,
                )
            )
        )

    def _split_grid_short_state(
        self,
        trade: Trade,
        current_rate: float,
    ) -> tuple[bool, float, float]:
        risk_budget = max(
            float(trade.get_custom_data("grid_risk_budget", 0.0)),
            1e-12,
        )
        favorable_rate = float(
            current_rate if trade.min_rate is None else trade.min_rate
        )
        best_profit = float(
            trade.calculate_profit(favorable_rate).total_profit
        )
        target_profit = risk_budget * float(self.GRID_SHORT_SPLIT_FLOOR_R)
        armed = (
            int(trade.get_custom_data("grid_tp_count", 0))
            >= int(self.GRID_SHORT_SPLIT_MIN_TP_COUNT)
            and best_profit
            >= risk_budget * float(self.GRID_SHORT_SPLIT_TRIGGER_R)
        )
        current_total = float(
            trade.calculate_profit(current_rate).total_profit
        )
        return armed, target_profit, current_total

    def _split_grid_short_floor_rate(
        self,
        trade: Trade,
        current_rate: float,
        target_profit: float,
    ) -> float | None:
        """Solve the fee-aware short price matching campaign target PnL."""
        # The target price changes only after a DCA/partial exit changes the
        # position structure.  Backtests call custom_stoploss on every detail
        # candle, so cache the solved root between fills instead of repeating
        # 36 fee-aware profit calculations every minute.
        signature = (
            float(trade.open_rate),
            float(getattr(trade, "amount", 0.0) or 0.0),
            float(getattr(trade, "realized_profit", 0.0) or 0.0),
            float(getattr(trade, "max_stake_amount", 0.0) or 0.0),
            float(target_profit),
        )
        cache = getattr(self, "_split_grid_floor_rate_cache", None)
        if cache is None:
            cache = {}
            self._split_grid_floor_rate_cache = cache
        cache_key = id(trade)
        cached = cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return float(cached[1])

        hard_stop = float(
            trade.get_custom_data("grid_hard_stop", 0.0) or 0.0
        )
        low = max(
            min(
                float(current_rate),
                float(trade.open_rate),
                float(trade.min_rate or current_rate),
            )
            * 0.50,
            1e-12,
        )
        high = max(
            float(current_rate),
            float(trade.open_rate),
            hard_stop,
        ) * 1.50
        low_profit = float(trade.calculate_profit(low).total_profit)
        high_profit = float(trade.calculate_profit(high).total_profit)
        if low_profit < target_profit or high_profit > target_profit:
            return None
        for _ in range(36):
            middle = (low + high) * 0.50
            profit = float(trade.calculate_profit(middle).total_profit)
            if profit > target_profit:
                low = middle
            else:
                high = middle
        stop_rate = (low + high) * 0.50
        cache[cache_key] = (signature, stop_rate)
        return stop_rate

    def _split_runner_protected_stop(
        self,
        trade: Trade,
        current_rate: float,
    ) -> float | None:
        if (
            not self.BO_RUNNER_PROTECT_ARMED_PROFIT
            or not bool(
                trade.get_custom_data(
                    "bo_v16_split_runner_extended",
                    False,
                )
            )
        ):
            return None
        armed_r = float(
            trade.get_custom_data("bo_v16_split_runner_armed_r", 0.0)
        )
        locked_r = max(
            float(self.BO_RUNNER_MIN_FLOOR_R),
            armed_r * float(self.BO_RUNNER_ARMED_FLOOR_FRACTION),
        )
        unit_risk = max(
            float(trade.get_custom_data("initial_unit_risk", 0.0)),
            1e-12,
        )
        side = -1.0 if trade.is_short else 1.0
        stop_rate = trade.open_rate + side * (
            locked_r * unit_risk
            + 2.0 * float(self.SIDE_COST) * trade.open_rate
        )
        return abs(
            float(
                stoploss_from_absolute(
                    stop_rate,
                    current_rate=current_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage,
                )
            )
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
        component = self._component(trade.enter_tag)
        if component == "breakout" and trade.is_short:
            protected = self._split_breakout_short_stop(
                pair,
                trade,
                current_time,
                current_rate,
            )
            inherited = self._split_tighter_stop(inherited, protected)
        if component == "breakout":
            runner_protected = self._split_runner_protected_stop(
                trade,
                current_rate,
            )
            return self._split_tighter_stop(inherited, runner_protected)
        if component != "grid" or not trade.is_short:
            # Breakout longs and Grid longs retain exact parent behavior.
            return inherited

        armed, target_profit, current_total = self._split_grid_short_state(
            trade,
            current_rate,
        )
        if not armed or current_total <= target_profit:
            return inherited
        stop_rate = self._split_grid_short_floor_rate(
            trade,
            current_rate,
            target_profit,
        )
        if stop_rate is None:
            return inherited
        protected = abs(
            float(
                stoploss_from_absolute(
                    stop_rate,
                    current_rate=current_rate,
                    is_short=True,
                    leverage=trade.leverage,
                )
            )
        )
        return self._split_tighter_stop(inherited, protected)

    def _split_runner_eligible(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
    ) -> bool:
        return bool(
            self.SPLIT_RUNNER_ENABLED
            and int(trade.get_custom_data("bo_score", 0))
            >= int(self.BO_RUNNER_MIN_SCORE)
            and self._trade_r(trade, current_rate)
            >= float(self.BO_RUNNER_MIN_R)
            and self._split_runner_environment_valid(
                pair,
                trade,
                current_time,
            )
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
        component = self._component(trade.enter_tag)
        if component == "grid" and trade.is_short:
            armed, target_profit, current_total = (
                self._split_grid_short_state(trade, current_rate)
            )
            if armed and current_total <= target_profit:
                return self.GRID_SHORT_SPLIT_EXIT_REASON

        inherited = super().custom_exit(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
            **kwargs,
        )
        if component != "breakout" or not self.SPLIT_RUNNER_ENABLED:
            return inherited

        holding_minutes = int(
            (current_time - self._trade_open_time(trade)).total_seconds()
            / 60
        )
        extended = bool(
            trade.get_custom_data("bo_v16_split_runner_extended", False)
        )
        if extended:
            if holding_minutes >= int(self.BO_RUNNER_MAX_HOLD_MINUTES):
                return "bo_v16_split_runner_time_stop"
            if inherited in {
                "bo_v9_time_stop",
                "bo_v9_take_profit_60r",
            }:
                return None
            return inherited

        if inherited not in {
            "bo_v9_time_stop",
            "bo_v9_take_profit_60r",
        }:
            return inherited
        if not self._split_runner_eligible(
            pair,
            trade,
            current_time,
            current_rate,
        ):
            return inherited
        trade.set_custom_data("bo_v16_split_runner_extended", True)
        trade.set_custom_data(
            "bo_v16_split_runner_armed_at",
            current_time.isoformat(),
        )
        trade.set_custom_data(
            "bo_v16_split_runner_armed_r",
            self._trade_r(trade, current_rate),
        )
        return None


class BreakoutV16GridV15SplitTrailOnlyResearchFreqtrade(
    _SplitTrailRunnerMixin,
    BreakoutV16GridV15PrecisionGuardGlobalResearchFreqtrade,
):
    """Ablation: separated short stop paths without runner extension."""

    STRATEGY_VERSION = "breakout_v16_grid_v15_split_trail_only_20260816"


class BreakoutV16GridV15SplitTrailRunner36ResearchFreqtrade(
    BreakoutV16GridV15SplitTrailOnlyResearchFreqtrade
):
    """Separated stops plus a maximum 36-hour confirmed runner window."""

    SPLIT_RUNNER_ENABLED = True
    BO_RUNNER_MAX_HOLD_MINUTES = 2160
    STRATEGY_VERSION = "breakout_v16_grid_v15_split_trail_runner36_20260816"


class BreakoutV16GridV15SplitTrailRunner48ResearchFreqtrade(
    BreakoutV16GridV15SplitTrailOnlyResearchFreqtrade
):
    """Separated stops plus a maximum 48-hour confirmed runner window."""

    SPLIT_RUNNER_ENABLED = True
    BO_RUNNER_MAX_HOLD_MINUTES = 2880
    STRATEGY_VERSION = "breakout_v16_grid_v15_split_trail_runner48_20260816"


class BreakoutV16GridV15SplitTrailV2OnlyResearchFreqtrade(
    BreakoutV16GridV15SplitTrailOnlyResearchFreqtrade
):
    """Conservative split: confirmed short floor and post-TP Grid floor."""

    BO_SHORT_SPLIT_TRIGGER_1_R = 1.5
    BO_SHORT_SPLIT_FLOOR_1_R = -0.55
    BO_SHORT_SPLIT_TRIGGER_2_R = 3.0
    BO_SHORT_SPLIT_FLOOR_2_R = 0.0
    GRID_SHORT_SPLIT_MIN_TP_COUNT = 1
    GRID_SHORT_SPLIT_FLOOR_R = -0.25
    STRATEGY_VERSION = "breakout_v16_grid_v15_split_trail_v2_only_20260816"


class BreakoutV16GridV15SplitTrailRunnerProtected36V2ResearchFreqtrade(
    BreakoutV16GridV15SplitTrailV2OnlyResearchFreqtrade
):
    """V2 split plus a 36-hour runner retaining half its 20h R-profit."""

    SPLIT_RUNNER_ENABLED = True
    BO_RUNNER_MAX_HOLD_MINUTES = 2160
    BO_RUNNER_PROTECT_ARMED_PROFIT = True
    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_split_trail_runner_protected36_v2_20260816"
    )


class BreakoutV16GridV15SplitTrailV3ConservativeResearchFreqtrade(
    BreakoutV16GridV15SplitTrailV2OnlyResearchFreqtrade
):
    """Wider post-TP Grid floor requiring stronger campaign progress."""

    GRID_SHORT_SPLIT_TRIGGER_R = 0.50
    GRID_SHORT_SPLIT_FLOOR_R = -0.40
    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_split_trail_v3_conservative_20260816"
    )


class BreakoutV16GridV15SplitTrailV3Floor30ResearchFreqtrade(
    BreakoutV16GridV15SplitTrailV3ConservativeResearchFreqtrade
):
    """Tighter post-TP loss floor around the selected V3 trigger."""

    GRID_SHORT_SPLIT_FLOOR_R = -0.30
    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_split_trail_v3_floor30_20260818"
    )


class BreakoutV16GridV15SplitTrailV3Floor50ResearchFreqtrade(
    BreakoutV16GridV15SplitTrailV3ConservativeResearchFreqtrade
):
    """Looser post-TP loss floor around the selected V3 trigger."""

    GRID_SHORT_SPLIT_FLOOR_R = -0.50
    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_split_trail_v3_floor50_20260818"
    )


class BreakoutV16GridV15SplitTrailV3Trigger60ResearchFreqtrade(
    BreakoutV16GridV15SplitTrailV3ConservativeResearchFreqtrade
):
    """Require more confirmed campaign profit before arming the V3 floor."""

    GRID_SHORT_SPLIT_TRIGGER_R = 0.60
    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_split_trail_v3_trigger60_20260818"
    )

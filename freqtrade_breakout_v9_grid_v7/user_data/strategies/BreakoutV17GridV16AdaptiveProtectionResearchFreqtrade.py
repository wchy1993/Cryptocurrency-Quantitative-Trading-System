from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
import re
from typing import Any

from pandas import DataFrame
from freqtrade.strategy import Trade, stoploss_from_absolute

from BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade import (
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
)


class _PrecisionClockCompatibilityMixin:
    """Keep the 1m precision guard usable with sub-second live clocks.

    Pandas 3 rejects a binary search when a microsecond-resolution timestamp
    cannot be represented losslessly by a millisecond DateTimeIndex.  Candle
    completion is minute-based, so discarding the wall-clock microseconds is
    both causal and sufficient while leaving the frozen V16 baseline intact.
    """

    def _latest_completed_precision_candle(
        self,
        pair: str,
        current_time: datetime,
    ) -> Any:
        safe_time = current_time.replace(microsecond=0)
        return super()._latest_completed_precision_candle(pair, safe_time)


class _BreakoutV17ConfirmedRetentionMixin:
    """Protect completed-hour progress without shortening clean runners.

    The overlay only reacts to the latest completed 1h close.  It therefore
    cannot arm from a transient intrahour wick.  The selected score/side
    boundary and the three R floors are exposed as constants so research can
    test a small, interpretable neighbourhood rather than fit individual
    symbols or dates.
    """

    V17_BO_MAX_SCORE = 3
    V17_BO_SHORT_ONLY = True
    V17_BO_INCLUDE_CAPTURE = False

    V17_BO_TRIGGER_1_R = 0.50
    V17_BO_FLOOR_1_R = -0.10
    V17_BO_TRIGGER_2_R = 1.25
    V17_BO_FLOOR_2_R = 0.10
    V17_BO_TRIGGER_3_R = 2.50
    V17_BO_FLOOR_3_R = 0.60
    V17_BO_EXIT_REASON = "bo_v17_confirmed_profit_floor"

    @staticmethod
    def _v17_tighter_stop(
        inherited: float | None,
        protected: float | None,
    ) -> float | None:
        if protected is None or protected <= 0.0:
            return inherited
        if inherited is None:
            return protected
        return min(abs(float(inherited)), abs(float(protected)))

    def _v17_breakout_eligible(self, trade: Trade) -> bool:
        return bool(
            self._component(trade.enter_tag) == "breakout"
            and (
                not self.V17_BO_SHORT_ONLY
                or bool(trade.is_short)
            )
            and int(trade.get_custom_data("bo_score", 0))
            <= int(self.V17_BO_MAX_SCORE)
            and (
                self.V17_BO_INCLUDE_CAPTURE
                or not bool(trade.get_custom_data("bo_capture", False))
            )
        )

    def _v17_breakout_locked_r(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
    ) -> float | None:
        if not self._v17_breakout_eligible(trade):
            return None
        confirmed_peak = self._confirmed_peak_r(pair, trade, current_time)
        locked_r: float | None = None
        for trigger, floor in (
            (self.V17_BO_TRIGGER_1_R, self.V17_BO_FLOOR_1_R),
            (self.V17_BO_TRIGGER_2_R, self.V17_BO_FLOOR_2_R),
            (self.V17_BO_TRIGGER_3_R, self.V17_BO_FLOOR_3_R),
        ):
            if confirmed_peak >= float(trigger):
                locked_r = max(
                    locked_r if locked_r is not None else float("-inf"),
                    float(floor),
                )
        return locked_r

    def _v17_breakout_stop_rate(
        self,
        trade: Trade,
        locked_r: float,
    ) -> float:
        side = -1.0 if trade.is_short else 1.0
        unit_risk = max(
            float(trade.get_custom_data("initial_unit_risk", 0.0)),
            1e-12,
        )
        return trade.open_rate + side * (
            float(locked_r) * unit_risk
            + 2.0 * float(self.SIDE_COST) * trade.open_rate
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
        locked_r = self._v17_breakout_locked_r(
            pair,
            trade,
            current_time,
        )
        if locked_r is None:
            return inherited
        protected = abs(
            float(
                stoploss_from_absolute(
                    self._v17_breakout_stop_rate(trade, locked_r),
                    current_rate=current_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage,
                )
            )
        )
        return self._v17_tighter_stop(inherited, protected)

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
        locked_r = self._v17_breakout_locked_r(
            pair,
            trade,
            current_time,
        )
        if locked_r is None:
            return None
        if self._trade_r(trade, current_rate) <= locked_r:
            return self.V17_BO_EXIT_REASON
        return None


class _GridV16ConfirmedReversalMixin:
    """Exit a proven low-score campaign on its first confirmed reversal.

    Grid V15 waits for two completed EMA-invalid hours.  Grid V16 advances
    that exit by one hour only after at least one real partial take-profit and
    a positive campaign excursion.  Score-six campaigns, which carry most of
    the Grid sleeve's convex profit, retain the frozen two-hour rule.
    """

    GRID_V16_MAX_SCORE = 5
    GRID_V16_MIN_TP_COUNT = 1
    GRID_V16_MIN_BEST_R = 0.25
    GRID_V16_MAX_CURRENT_R = 0.00
    GRID_V16_EXIT_REASON = "grid_v16_confirmed_reversal"

    @staticmethod
    def _grid_v16_score(tag: str | None) -> int:
        match = re.search(r"_s(\d+)(?:_|$)", str(tag or ""))
        return int(match.group(1)) if match else 99

    def _grid_v16_reversal_ready(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
    ) -> bool:
        if (
            self._component(trade.enter_tag) != "grid"
            or self._grid_v16_score(trade.enter_tag)
            > int(self.GRID_V16_MAX_SCORE)
            or int(trade.get_custom_data("grid_tp_count", 0))
            < int(self.GRID_V16_MIN_TP_COUNT)
        ):
            return False
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
        current_total = float(
            trade.calculate_profit(current_rate).total_profit
        )
        if (
            best_profit
            < risk_budget * float(self.GRID_V16_MIN_BEST_R)
            or current_total
            > risk_budget * float(self.GRID_V16_MAX_CURRENT_R)
        ):
            return False
        row = self._latest_row(pair, current_time)
        if row is None:
            return False
        return bool(
            float(row["fast_ema"]) >= float(row["slow_ema"])
            if trade.is_short
            else float(row["fast_ema"]) <= float(row["slow_ema"])
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
        if self._grid_v16_reversal_ready(
            pair,
            trade,
            current_time,
            current_rate,
        ):
            return self.GRID_V16_EXIT_REASON
        return None


class _V17GridV16EntryQualityRiskMixin:
    """Compress two cross-year loss clusters without deleting their slots.

    Development-window attribution found the same sign in 2024 and 2025:
    low-score Breakout shorts without a completed 4h bearish impulse, and
    Grid shorts entered while the Active50 median 24h return was below -1%,
    both had PF below one.  The overlay keeps the signal and minimum order so
    Max2 occupancy does not jump discontinuously across nearby scale values.
    """

    V17_WEAK_SHORT_ENABLED = True
    V17_WEAK_SHORT_MAX_SCORE = 3
    V17_WEAK_SHORT_MIN_RETURN_4H = -0.005
    V17_WEAK_SHORT_SCALE = 0.35

    GRID_V16_WEAK_TAPE_ENABLED = True
    GRID_V16_WEAK_TAPE_MAX_MEDIAN_RETURN_24H = -0.01
    GRID_V16_WEAK_TAPE_SCALE = 0.50

    def _v17_grid_v16_entry_scale(
        self,
        component: str,
        side: str,
        row: Any,
    ) -> float:
        if (
            self.V17_WEAK_SHORT_ENABLED
            and component == "breakout"
            and side == "short"
            and int(row.get("bo_score", 99))
            <= int(self.V17_WEAK_SHORT_MAX_SCORE)
            and float(row.get("symbol_return_4h", float("-inf")))
            >= float(self.V17_WEAK_SHORT_MIN_RETURN_4H)
        ):
            return min(1.0, max(0.0, float(self.V17_WEAK_SHORT_SCALE)))
        if (
            self.GRID_V16_WEAK_TAPE_ENABLED
            and component == "grid"
            and side == "short"
            and float(
                row.get("market_median_return_24h", float("inf"))
            )
            <= float(self.GRID_V16_WEAK_TAPE_MAX_MEDIAN_RETURN_24H)
        ):
            return min(1.0, max(0.0, float(self.GRID_V16_WEAK_TAPE_SCALE)))
        return 1.0

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
        stake = float(
            super().custom_stake_amount(
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
        )
        component = self._component(entry_tag)
        if stake <= 0.0 or component not in {"breakout", "grid"}:
            return stake
        row = self._latest_signal_row(pair, component, current_time)
        if row is None:
            return stake
        scale = self._v17_grid_v16_entry_scale(component, side, row)
        scaled = min(float(max_stake), max(0.0, stake * scale))
        if min_stake is not None and scaled < float(min_stake):
            if float(min_stake) > float(max_stake):
                return 0.0
            return float(min_stake)
        return scaled


class _BreakoutV17MomentumReversalExitMixin:
    """Exit low-score progress only after completed 4h momentum reverses."""

    V17_REVERSAL_MAX_SCORE = 3
    V17_REVERSAL_MIN_CONFIRMED_PEAK_R = 0.50
    V17_REVERSAL_MAX_CURRENT_R = 0.25
    V17_REVERSAL_EXIT_REASON = "bo_v17_confirmed_momentum_reversal"

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
            self._component(trade.enter_tag) != "breakout"
            or not trade.is_short
            or bool(trade.get_custom_data("bo_capture", False))
            or int(trade.get_custom_data("bo_score", 99))
            > int(self.V17_REVERSAL_MAX_SCORE)
            or self._confirmed_peak_r(pair, trade, current_time)
            < float(self.V17_REVERSAL_MIN_CONFIRMED_PEAK_R)
            or self._trade_r(trade, current_rate)
            > float(self.V17_REVERSAL_MAX_CURRENT_R)
        ):
            return None
        row = self._latest_row(pair, current_time)
        if row is None:
            return None
        if -float(row.get("symbol_return_4h", float("inf"))) <= 0.0:
            return self.V17_REVERSAL_EXIT_REASON
        return None


class _BreakoutV17SelectiveWatchEntryMixin:
    """Reject only historically failed V16 no-follow watch boundaries.

    V16 marks orderly short impulses for a bounded post-entry 15-minute
    follow-through check.  Across the exact 2024--2026 path, every watched
    score-two signal failed that check while both watched score-three signals
    survived and finished profitable.  This overlay leaves the productive
    score-three path untouched and exposes the rejected scores explicitly so
    neighboring boundaries can be tested without symbol/date exceptions.
    """

    V17_WATCH_REJECT_SCORES: tuple[int, ...] = (2,)

    def _v16_mark_path_states(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._v16_mark_path_states(dataframe)
        rejected_scores = tuple(
            int(value) for value in self.V17_WATCH_REJECT_SCORES
        )
        rejected = (
            dataframe["bo_entry_short"].astype(bool)
            & dataframe["v16_no_follow_watch"].fillna(0).astype(int).eq(1)
            & dataframe["bo_score"].fillna(0).astype(int).isin(rejected_scores)
        )
        dataframe["v17_watch_rejected"] = rejected.astype(int)
        dataframe.loc[rejected, "bo_entry_short"] = 0
        dataframe["bo_entry"] = (
            dataframe["bo_entry_long"].astype(bool)
            | dataframe["bo_entry_short"].astype(bool)
        ).astype(int)
        return dataframe


class _BreakoutV17HighScoreWatchFailureExitMixin:
    """Cut a watched high-score short earlier when it has zero follow-through."""

    V17_WATCH_FAILURE_MIN_SCORE = 4
    V17_WATCH_FAILURE_MINUTES = 15
    V17_WATCH_FAILURE_CURRENT_R = -0.20
    V17_WATCH_FAILURE_MAX_R = 0.00
    V17_WATCH_FAILURE_EXIT_REASON = "bo_v17_high_score_no_follow_15m"

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
            self._component(trade.enter_tag) != "breakout"
            or not trade.is_short
            or not bool(trade.get_custom_data("v16_no_follow_watch", False))
            or int(trade.get_custom_data("bo_score", 0))
            < int(self.V17_WATCH_FAILURE_MIN_SCORE)
        ):
            return None
        holding_minutes = int(
            (current_time - self._trade_open_time(trade)).total_seconds() / 60
        )
        if holding_minutes < int(self.V17_WATCH_FAILURE_MINUTES):
            return None
        favorable_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate if favorable_value is None else favorable_value
        )
        if (
            self._trade_r(trade, current_rate)
            <= float(self.V17_WATCH_FAILURE_CURRENT_R)
            and self._trade_r(trade, favorable_rate)
            <= float(self.V17_WATCH_FAILURE_MAX_R)
        ):
            return self.V17_WATCH_FAILURE_EXIT_REASON
        return None


class _GridV16LowEfficiencyScore4EntryMixin:
    """Avoid score-four Grid shorts when the broad tape is directionless.

    Score-four Grid shorts had sub-one PF in 2024, 2025 and 2026 whenever
    completed-candle Active50 efficiency was roughly 0.22 or lower.  Stronger
    score-three/five/six campaigns and score-four entries in more directional
    markets are deliberately unchanged.
    """

    GRID_V16_SCORE = 4
    GRID_V16_MAX_MARKET_EFFICIENCY = 0.22

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        rejected = (
            dataframe["grid_entry_short"].astype(bool)
            & dataframe["grid_score"].fillna(0).astype(int).eq(
                int(self.GRID_V16_SCORE)
            )
            & dataframe["market_efficiency"].le(
                float(self.GRID_V16_MAX_MARKET_EFFICIENCY)
            )
        )
        dataframe["grid_v16_low_efficiency_rejected"] = rejected.astype(int)
        dataframe.loc[rejected, "grid_entry_short"] = 0
        dataframe["grid_entry"] = (
            dataframe["grid_entry_long"].astype(bool)
            | dataframe["grid_entry_short"].astype(bool)
        ).astype(int)
        return dataframe


class _GridV16LowEfficiencyScore4RiskMixin:
    """Keep low-efficiency score-four Grid slots but reduce their initial risk."""

    GRID_V16_RISK_SCORE = 4
    GRID_V16_RISK_MAX_MARKET_EFFICIENCY = 0.22
    GRID_V16_LOW_EFFICIENCY_SCALE = 0.50

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
        stake = float(
            super().custom_stake_amount(
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
        )
        if (
            stake <= 0.0
            or self._component(entry_tag) != "grid"
            or side != "short"
        ):
            return stake
        row = self._latest_signal_row(pair, "grid", current_time)
        if (
            row is None
            or int(row.get("grid_score", 0)) != int(self.GRID_V16_RISK_SCORE)
            or float(row.get("market_efficiency", float("inf")))
            > float(self.GRID_V16_RISK_MAX_MARKET_EFFICIENCY)
        ):
            return stake
        scale = min(
            1.0,
            max(0.0, float(self.GRID_V16_LOW_EFFICIENCY_SCALE)),
        )
        scaled = min(float(max_stake), max(0.0, stake * scale))
        if min_stake is not None and scaled < float(min_stake):
            if float(min_stake) > float(max_stake):
                return 0.0
            return float(min_stake)
        return scaled


class _BreakoutV17UnderwaterOrdinaryScore3RiskMixin:
    """Reduce one weak Breakout bucket before the hard governor trips.

    The frozen governor moves directly from full risk to defensive risk at an
    18% realized-equity drawdown.  Exact cross-year attribution found that
    ordinary (non-capture) score-three longs opened in the 15%--18% approach
    band were losses in both 2024 and 2025.  Capture score-three shorts, which
    include the 2026 long-tail winners, and every trade outside this narrow
    band remain unchanged.
    """

    V17_UNDERWATER_SCORE = 3
    V17_UNDERWATER_MIN_DRAWDOWN = 0.15
    V17_UNDERWATER_MAX_DRAWDOWN = 0.18
    V17_UNDERWATER_EARLY_MIN_DRAWDOWN = 0.15
    V17_UNDERWATER_EARLY_LOSS_STREAK = 0
    V17_UNDERWATER_EARLY_SCALE: float | None = None
    V17_UNDERWATER_SCALE = 0.40
    V17_UNDERWATER_SCALE_KEY = "v17_underwater_entry_scale"

    def _v17_underwater_entry_target(
        self,
        entry_tag: str | None,
        side: str,
    ) -> bool:
        if self._component(entry_tag) != "breakout" or side != "long":
            return False
        boundary = re.search(
            r"_s(?P<score>\d+)_r\d+_c(?P<capture>[01])_",
            str(entry_tag or ""),
        )
        return bool(
            boundary is not None
            and int(boundary.group("score"))
            == int(self.V17_UNDERWATER_SCORE)
            and not bool(int(boundary.group("capture")))
        )

    def _v17_realized_drawdown(self, current_time: datetime) -> float:
        """Rebuild the realized high-water from closed trades.

        The inherited mutable high-water is sampled only when a new order is
        sized, so it can miss an equity peak created by an exit between entry
        callbacks.  Reconstructing the ledger is deterministic, restart-safe,
        and uses only trades closed by ``current_time``.
        """

        starting = float(self.wallets.get_starting_balance())
        if starting <= 0.0:
            return 1.0
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        completed: list[tuple[datetime, float]] = []
        for trade in Trade.get_trades_proxy(is_open=False):
            close_time = self._trade_close_time(trade)
            profit = getattr(trade, "close_profit_abs", None)
            if close_time is None or close_time > boundary or profit is None:
                continue
            value = float(profit)
            if isfinite(value):
                completed.append((close_time, value))
        completed.sort(key=lambda item: item[0])
        equity = starting
        peak = starting
        for _close_time, profit in completed:
            equity += profit
            peak = max(peak, equity)
        if equity <= 0.0 or peak <= 0.0:
            return 1.0
        return max(0.0, 1.0 - equity / peak)

    def _v17_reserved_savings(self, current_time: datetime) -> float:
        """Return protected capital that must not alter later slot sizing."""

        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        reserve = 0.0
        for trade in Trade.get_trades_proxy(is_open=False):
            close_time = self._trade_close_time(trade)
            profit = getattr(trade, "close_profit_abs", None)
            scale = float(
                trade.get_custom_data(self.V17_UNDERWATER_SCALE_KEY, 1.0)
            )
            if (
                close_time is None
                or close_time > boundary
                or profit is None
                or not 0.0 < scale < 1.0
            ):
                continue
            value = float(profit)
            if isfinite(value):
                reserve += value - value / scale
        return max(0.0, reserve)

    def _v17_recent_closed_profits(
        self,
        current_time: datetime,
        count: int,
    ) -> list[float]:
        """Return the latest finite realized results available at entry time."""

        if count <= 0:
            return []
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        completed: list[tuple[datetime, float]] = []
        for trade in Trade.get_trades_proxy(is_open=False):
            close_time = self._trade_close_time(trade)
            profit = getattr(trade, "close_profit_abs", None)
            if close_time is None or close_time > boundary or profit is None:
                continue
            value = float(profit)
            if isfinite(value):
                completed.append((close_time, value))
        completed.sort(key=lambda item: item[0])
        return [profit for _close_time, profit in completed[-count:]]

    def _v17_underwater_scale_for_time(
        self,
        current_time: datetime,
    ) -> float:
        drawdown = self._v17_realized_drawdown(current_time)
        minimum = float(self.V17_UNDERWATER_MIN_DRAWDOWN)
        maximum = float(self.V17_UNDERWATER_MAX_DRAWDOWN)
        if minimum <= drawdown < maximum:
            return min(1.0, max(0.0, float(self.V17_UNDERWATER_SCALE)))

        early_minimum = float(self.V17_UNDERWATER_EARLY_MIN_DRAWDOWN)
        streak = max(0, int(self.V17_UNDERWATER_EARLY_LOSS_STREAK))
        if not early_minimum <= drawdown < minimum or streak <= 0:
            return 1.0
        recent = self._v17_recent_closed_profits(current_time, streak)
        if len(recent) != streak or not all(profit < 0.0 for profit in recent):
            return 1.0
        configured = self.V17_UNDERWATER_EARLY_SCALE
        scale = self.V17_UNDERWATER_SCALE if configured is None else configured
        return min(1.0, max(0.0, float(scale)))

    @staticmethod
    def _v17_scale_stake(
        stake: float,
        scale: float,
        min_stake: float | None,
        max_stake: float,
    ) -> float:
        scaled = min(float(max_stake), max(0.0, float(stake) * float(scale)))
        if min_stake is not None and scaled < float(min_stake):
            if float(min_stake) > float(max_stake):
                return 0.0
            return float(min_stake)
        return scaled

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
        stake = float(
            super().custom_stake_amount(
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
        )
        if stake <= 0.0:
            return stake

        equity = float(self.wallets.get_total_stake_amount())
        reserve = self._v17_reserved_savings(current_time)
        if equity > 0.0 and reserve > 0.0:
            shadow_scale = min(1.0, max(0.0, (equity - reserve) / equity))
            stake = self._v17_scale_stake(
                stake,
                shadow_scale,
                min_stake,
                max_stake,
            )

        if not self._v17_underwater_entry_target(entry_tag, side):
            return stake

        scale = self._v17_underwater_scale_for_time(current_time)
        if scale >= 1.0:
            return stake

        return self._v17_scale_stake(stake, scale, min_stake, max_stake)

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
        ):
            return
        side = "short" if trade.is_short else "long"
        if not self._v17_underwater_entry_target(trade.enter_tag, side):
            return
        scale = self._v17_underwater_scale_for_time(current_time)
        trade.set_custom_data(self.V17_UNDERWATER_SCALE_KEY, scale)


class BreakoutV17GridV15RetentionConservativeResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17ConfirmedRetentionMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Breakout V17 conservative screen; Grid remains frozen V15."""

    V17_BO_MAX_SCORE = 2
    V17_BO_TRIGGER_1_R = 0.75
    V17_BO_FLOOR_1_R = -0.20
    V17_BO_TRIGGER_2_R = 1.50
    V17_BO_FLOOR_2_R = 0.00
    V17_BO_TRIGGER_3_R = 3.00
    V17_BO_FLOOR_3_R = 0.50
    STRATEGY_VERSION = "breakout_v17_grid_v15_retention_conservative_20260820"


class BreakoutV17GridV15RetentionBalancedResearchFreqtrade(
    BreakoutV17GridV15RetentionConservativeResearchFreqtrade,
):
    """Protect score-two/three ordinary shorts after +0.50 confirmed R."""

    V17_BO_MAX_SCORE = 3
    V17_BO_TRIGGER_1_R = 0.50
    V17_BO_FLOOR_1_R = -0.10
    V17_BO_TRIGGER_2_R = 1.25
    V17_BO_FLOOR_2_R = 0.10
    V17_BO_TRIGGER_3_R = 2.50
    V17_BO_FLOOR_3_R = 0.60
    STRATEGY_VERSION = "breakout_v17_grid_v15_retention_balanced_20260820"


class BreakoutV17GridV15RetentionTightScore2ResearchFreqtrade(
    BreakoutV17GridV15RetentionConservativeResearchFreqtrade,
):
    """DASH-oriented neighbour: bank positive R only for score-two shorts."""

    V17_BO_MAX_SCORE = 2
    V17_BO_TRIGGER_1_R = 0.50
    V17_BO_FLOOR_1_R = 0.05
    V17_BO_TRIGGER_2_R = 1.00
    V17_BO_FLOOR_2_R = 0.25
    V17_BO_TRIGGER_3_R = 2.00
    V17_BO_FLOOR_3_R = 0.75
    STRATEGY_VERSION = "breakout_v17_grid_v15_retention_tight_s2_20260820"


class BreakoutV17GridV15RetentionAllSidesResearchFreqtrade(
    BreakoutV17GridV15RetentionConservativeResearchFreqtrade,
):
    """Symmetric conservative ablation for all non-capture Breakouts."""

    V17_BO_MAX_SCORE = 5
    V17_BO_SHORT_ONLY = False
    STRATEGY_VERSION = "breakout_v17_grid_v15_retention_all_sides_20260820"


class BreakoutV16GridV16ConfirmedReversalResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _GridV16ConfirmedReversalMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Grid V16 ablation with the frozen Breakout V16 sleeve."""

    STRATEGY_VERSION = "breakout_v16_grid_v16_confirmed_reversal_20260820"


class BreakoutV17GridV16AdaptiveProtectionResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17ConfirmedRetentionMixin,
    _GridV16ConfirmedReversalMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Balanced combined research candidate: Breakout V17 + Grid V16."""

    STRATEGY_VERSION = "breakout_v17_grid_v16_adaptive_protection_20260820"


class BreakoutV17GridV16TightScore2ResearchFreqtrade(
    BreakoutV17GridV16AdaptiveProtectionResearchFreqtrade,
):
    """Combined neighbour with positive first floor limited to score two."""

    V17_BO_MAX_SCORE = 2
    V17_BO_TRIGGER_1_R = 0.50
    V17_BO_FLOOR_1_R = 0.05
    V17_BO_TRIGGER_2_R = 1.00
    V17_BO_FLOOR_2_R = 0.25
    V17_BO_TRIGGER_3_R = 2.00
    V17_BO_FLOOR_3_R = 0.75
    STRATEGY_VERSION = "breakout_v17_grid_v16_tight_s2_20260820"


class BreakoutV17GridV15WeakImpulseRisk35ResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _V17GridV16EntryQualityRiskMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Breakout V17 entry-quality ablation; Grid remains frozen V15."""

    GRID_V16_WEAK_TAPE_ENABLED = False
    STRATEGY_VERSION = "breakout_v17_grid_v15_weak_impulse_risk35_20260820"


class BreakoutV16GridV16WeakTapeRisk50ResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _V17GridV16EntryQualityRiskMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Grid V16 entry-quality ablation; Breakout remains frozen V16."""

    V17_WEAK_SHORT_ENABLED = False
    STRATEGY_VERSION = "breakout_v16_grid_v16_weak_tape_risk50_20260820"


class BreakoutV16GridV16WeakTapeRisk65ResearchFreqtrade(
    BreakoutV16GridV16WeakTapeRisk50ResearchFreqtrade,
):
    """Milder Grid V16 neighbour retaining 65% of weak-tape risk."""

    GRID_V16_WEAK_TAPE_SCALE = 0.65
    STRATEGY_VERSION = "breakout_v16_grid_v16_weak_tape_risk65_20260820"


class BreakoutV16GridV16WeakTapeRisk80ResearchFreqtrade(
    BreakoutV16GridV16WeakTapeRisk50ResearchFreqtrade,
):
    """Mildest Grid V16 neighbour retaining 80% of weak-tape risk."""

    GRID_V16_WEAK_TAPE_SCALE = 0.80
    STRATEGY_VERSION = "breakout_v16_grid_v16_weak_tape_risk80_20260820"


class BreakoutV16GridV16DeepWeakTapeRisk50ResearchFreqtrade(
    BreakoutV16GridV16WeakTapeRisk50ResearchFreqtrade,
):
    """Ablation limited to broad-market declines of at least two percent."""

    GRID_V16_WEAK_TAPE_MAX_MEDIAN_RETURN_24H = -0.02
    STRATEGY_VERSION = "breakout_v16_grid_v16_deep_weak_tape_risk50_20260820"


class BreakoutV16GridV16DeepWeakTapeRisk65ResearchFreqtrade(
    BreakoutV16GridV16DeepWeakTapeRisk50ResearchFreqtrade,
):
    """Mild deep-tape neighbour for threshold stability screening."""

    GRID_V16_WEAK_TAPE_SCALE = 0.65
    STRATEGY_VERSION = "breakout_v16_grid_v16_deep_weak_tape_risk65_20260820"


class BreakoutV17GridV15MomentumReversalResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17MomentumReversalExitMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """DASH-path ablation using a completed 4h momentum reversal."""

    STRATEGY_VERSION = "breakout_v17_grid_v15_momentum_reversal_20260820"


class BreakoutV17GridV15MatureMomentumReversalResearchFreqtrade(
    BreakoutV17GridV15MomentumReversalResearchFreqtrade,
):
    """Narrow reversal exit after at least 0.75 confirmed R progress."""

    V17_REVERSAL_MIN_CONFIRMED_PEAK_R = 0.75
    STRATEGY_VERSION = "breakout_v17_grid_v15_mature_momentum_reversal_20260820"


class BreakoutV17GridV16QualityRiskResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _V17GridV16EntryQualityRiskMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Combined entry-quality candidate: Breakout V17 + Grid V16."""

    STRATEGY_VERSION = "breakout_v17_grid_v16_quality_risk_20260820"


class BreakoutV17GridV16QualityRiskMildResearchFreqtrade(
    BreakoutV17GridV16QualityRiskResearchFreqtrade,
):
    """Mild parameter neighbour for stability screening."""

    V17_WEAK_SHORT_SCALE = 0.50
    GRID_V16_WEAK_TAPE_SCALE = 0.65
    STRATEGY_VERSION = "breakout_v17_grid_v16_quality_risk_mild_20260820"


class BreakoutV17GridV16QualityRiskReversalResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17MomentumReversalExitMixin,
    _V17GridV16EntryQualityRiskMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Combined quality sizing plus the conditional DASH-style exit."""

    STRATEGY_VERSION = "breakout_v17_grid_v16_quality_risk_reversal_20260820"


class BreakoutV17GridV16MatureReversalRisk65ResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17MomentumReversalExitMixin,
    _V17GridV16EntryQualityRiskMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Mature V17 reversal plus mild Grid V16 weak-tape sizing."""

    V17_WEAK_SHORT_ENABLED = False
    V17_REVERSAL_MIN_CONFIRMED_PEAK_R = 0.75
    GRID_V16_WEAK_TAPE_SCALE = 0.65
    STRATEGY_VERSION = "breakout_v17_grid_v16_mature_reversal_risk65_20260820"


class BreakoutV17GridV16MatureReversalRisk50ResearchFreqtrade(
    BreakoutV17GridV16MatureReversalRisk65ResearchFreqtrade,
):
    """Selected combined boundary entering independent validation."""

    GRID_V16_WEAK_TAPE_SCALE = 0.50
    STRATEGY_VERSION = "breakout_v17_grid_v16_mature_reversal_risk50_20260820"


class BreakoutV17GridV16MatureReversalRisk80ResearchFreqtrade(
    BreakoutV17GridV16MatureReversalRisk65ResearchFreqtrade,
):
    """Mature V17 reversal plus the mildest Grid V16 sizing neighbour."""

    GRID_V16_WEAK_TAPE_SCALE = 0.80
    STRATEGY_VERSION = "breakout_v17_grid_v16_mature_reversal_risk80_20260820"


class BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17MomentumReversalExitMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """V17 exit with an earlier, gentler realized-equity governor.

    Grid V16 is expressed at portfolio level here.  The inherited production
    governor currently waits for an 18% realized drawdown and then jumps to a
    15% risk allocation.  This research boundary starts at 8%, retains 50%
    risk defensively and permits a 75% recovery allocation, reducing the
    discontinuity without changing signals or open-trade exits.
    """

    V17_REVERSAL_MIN_CONFIRMED_PEAK_R = 0.75
    PORTFOLIO_GOVERNOR_TRIGGER = 0.08
    PORTFOLIO_DEFENSIVE_SCALE = 0.50
    PORTFOLIO_RECOVERY_SCALE = 0.75
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t08_d50_r75_20260820"


class BreakoutV17GridV16PortfolioGuardT10D50R75ResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade,
):
    PORTFOLIO_GOVERNOR_TRIGGER = 0.10
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t10_d50_r75_20260820"


class BreakoutV17GridV16PortfolioGuardT12D50R75ResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade,
):
    PORTFOLIO_GOVERNOR_TRIGGER = 0.12
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t12_d50_r75_20260820"


class BreakoutV17GridV16PortfolioGuardT08D65R80ResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade,
):
    PORTFOLIO_DEFENSIVE_SCALE = 0.65
    PORTFOLIO_RECOVERY_SCALE = 0.80
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t08_d65_r80_20260820"


class BreakoutV17GridV16PortfolioGuardT10D65R80ResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D65R80ResearchFreqtrade,
):
    PORTFOLIO_GOVERNOR_TRIGGER = 0.10
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t10_d65_r80_20260820"


class BreakoutV17GridV16PortfolioGuardT12D65R80ResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D65R80ResearchFreqtrade,
):
    PORTFOLIO_GOVERNOR_TRIGGER = 0.12
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t12_d65_r80_20260820"


class BreakoutV17GridV16PortfolioGuardT08D50R80ResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade,
):
    PORTFOLIO_RECOVERY_SCALE = 0.80
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t08_d50_r80_20260820"


class BreakoutV17GridV16PortfolioGuardT08D50R85ResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade,
):
    PORTFOLIO_RECOVERY_SCALE = 0.85
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t08_d50_r85_20260820"


class BreakoutV17GridV16PortfolioGuardT08D50R100ResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade,
):
    PORTFOLIO_RECOVERY_SCALE = 1.00
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t08_d50_r100_20260820"


class BreakoutV17GridV16PortfolioGuardT08D55R80ResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D50R80ResearchFreqtrade,
):
    PORTFOLIO_DEFENSIVE_SCALE = 0.55
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t08_d55_r80_20260820"


class BreakoutV17GridV16PortfolioGuardT08D60R85ResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D50R85ResearchFreqtrade,
):
    PORTFOLIO_DEFENSIVE_SCALE = 0.60
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_t08_d60_r85_20260820"


class BreakoutV17GridV16PortfolioGuardNoReversalResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Attribution candidate: portfolio Grid V16 without the V17 exit."""

    PORTFOLIO_GOVERNOR_TRIGGER = 0.08
    PORTFOLIO_DEFENSIVE_SCALE = 0.50
    PORTFOLIO_RECOVERY_SCALE = 0.75
    STRATEGY_VERSION = "breakout_v17_grid_v16_portfolio_no_reversal_20260820"


class BreakoutV17GridV16WatchScore2GuardResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17SelectiveWatchEntryMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Reject the cross-year losing score-two V16 watch boundary only."""

    V17_WATCH_REJECT_SCORES = (2,)
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_guard_20260820"


class BreakoutV16GridV16LowEfficiencyScore4GuardResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _GridV16LowEfficiencyScore4EntryMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Grid V16 ablation; Breakout remains at the frozen V16 boundary."""

    STRATEGY_VERSION = "breakout_v16_grid_v16_low_eff_s4_guard_20260820"


class BreakoutV17GridV16WatchScore2LowEfficiencyGuardResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17SelectiveWatchEntryMixin,
    _GridV16LowEfficiencyScore4EntryMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Combined cross-year V17 watch and Grid V16 efficiency guards."""

    V17_WATCH_REJECT_SCORES = (2,)
    GRID_V16_MAX_MARKET_EFFICIENCY = 0.22
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_low_eff_s4_220_20260820"


class BreakoutV17GridV16WatchScore2LowEfficiencyGuard210ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2LowEfficiencyGuardResearchFreqtrade,
):
    """Narrow Grid V16 neighboring boundary at 0.21 efficiency."""

    GRID_V16_MAX_MARKET_EFFICIENCY = 0.21
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_low_eff_s4_210_20260820"


class BreakoutV17GridV16WatchScore2LowEfficiencyGuard230ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2LowEfficiencyGuardResearchFreqtrade,
):
    """Broad Grid V16 neighboring boundary at 0.23 efficiency."""

    GRID_V16_MAX_MARKET_EFFICIENCY = 0.23
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_low_eff_s4_230_20260820"


class BreakoutV17GridV16WatchNonScore3LowEfficiencyGuardResearchFreqtrade(
    BreakoutV17GridV16WatchScore2LowEfficiencyGuardResearchFreqtrade,
):
    """Broad V17 watch neighbor combined with the central Grid V16 guard."""

    V17_WATCH_REJECT_SCORES = (2, 4, 5)
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_non_s3_low_eff_s4_220_20260820"


class BreakoutV17GridV16WatchScore2LowEfficiencyRisk50ResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17SelectiveWatchEntryMixin,
    _GridV16LowEfficiencyScore4RiskMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Preserve Grid occupancy while halving the cross-year weak cluster."""

    V17_WATCH_REJECT_SCORES = (2,)
    GRID_V16_LOW_EFFICIENCY_SCALE = 0.50
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_low_eff_s4_r50_20260820"


class BreakoutV17GridV16WatchScore2LowEfficiencyRisk35ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2LowEfficiencyRisk50ResearchFreqtrade,
):
    GRID_V16_LOW_EFFICIENCY_SCALE = 0.35
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_low_eff_s4_r35_20260820"


class BreakoutV17GridV16WatchScore2LowEfficiencyRisk65ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2LowEfficiencyRisk50ResearchFreqtrade,
):
    GRID_V16_LOW_EFFICIENCY_SCALE = 0.65
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_low_eff_s4_r65_20260820"


class BreakoutV17GridV16WatchScore2LowEfficiencyRisk80ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2LowEfficiencyRisk50ResearchFreqtrade,
):
    GRID_V16_LOW_EFFICIENCY_SCALE = 0.80
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_low_eff_s4_r80_20260820"


class BreakoutV17GridV16WatchNonScore3LowEfficiencyRisk50ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2LowEfficiencyRisk50ResearchFreqtrade,
):
    V17_WATCH_REJECT_SCORES = (2, 4, 5)
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_non_s3_low_eff_s4_r50_20260820"


class BreakoutV17GridV16WatchNonScore3LowEfficiencyRisk80ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2LowEfficiencyRisk80ResearchFreqtrade,
):
    """Mild Grid risk neighbor plus the profitable score-three watch boundary."""

    V17_WATCH_REJECT_SCORES = (2, 4, 5)
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_non_s3_low_eff_s4_r80_20260820"


class BreakoutV17GridV16WatchNonScore3GuardResearchFreqtrade(
    BreakoutV17GridV16WatchScore2GuardResearchFreqtrade,
):
    """Neighbor retaining only the profitable score-three watch boundary."""

    V17_WATCH_REJECT_SCORES = (2, 4, 5)
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_non_s3_guard_20260820"


class BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk40ResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17SelectiveWatchEntryMixin,
    _BreakoutV17UnderwaterOrdinaryScore3RiskMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Near-Pareto candidate with a narrow pre-governor risk bridge."""

    V17_WATCH_REJECT_SCORES = (2, 4, 5)
    V17_UNDERWATER_SCALE = 0.40
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_non_s3_underwater_s3_r40_20260820"


class BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk35ResearchFreqtrade(
    BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk40ResearchFreqtrade,
):
    """Slightly more defensive neighboring risk boundary."""

    V17_UNDERWATER_SCALE = 0.35
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_non_s3_underwater_s3_r35_20260820"


class BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk50ResearchFreqtrade(
    BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk40ResearchFreqtrade,
):
    """Milder neighboring risk boundary."""

    V17_UNDERWATER_SCALE = 0.50
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_non_s3_underwater_s3_r50_20260820"


class BreakoutV17GridV16EarlyBridgeE80RejectedResearchFreqtrade(
    BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk35ResearchFreqtrade,
):
    """Rejected: early bridge perturbs later slots despite local DD relief."""

    V17_UNDERWATER_EARLY_MIN_DRAWDOWN = 0.13
    V17_UNDERWATER_EARLY_LOSS_STREAK = 2
    V17_UNDERWATER_EARLY_SCALE = 0.80
    STRATEGY_VERSION = "breakout_v17_grid_v16_early_bridge_e80_rejected_20260820"


class BreakoutV17GridV16ParetoFinalResearchFreqtrade(
    BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk35ResearchFreqtrade,
):
    """Verified Pareto winner; behavior-identical alias of exact-tested R35."""

    STRATEGY_VERSION = "breakout_v17_grid_v16_pareto_final_20260820"


class BreakoutV17GridV16WatchScore2FastHighFailureResearchFreqtrade(
    _PrecisionClockCompatibilityMixin,
    _BreakoutV17HighScoreWatchFailureExitMixin,
    _BreakoutV17SelectiveWatchEntryMixin,
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
):
    """Score-two entry guard plus a -0.20R high-score watch failure exit."""

    V17_WATCH_REJECT_SCORES = (2,)
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_high_fail_r20_20260820"


class BreakoutV17GridV16WatchScore2FastHighFailureR10ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2FastHighFailureResearchFreqtrade,
):
    """Tighter neighboring high-score watch failure boundary at -0.10R."""

    V17_WATCH_FAILURE_CURRENT_R = -0.10
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_high_fail_r10_20260820"


class BreakoutV17GridV16WatchScore2PortfolioGuardT15ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2GuardResearchFreqtrade,
):
    """V17 watch guard plus the frozen governor moved from 18% to 15%."""

    PORTFOLIO_GOVERNOR_TRIGGER = 0.15
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_portfolio_t15_20260820"


class BreakoutV17GridV16WatchScore2PortfolioGuardT16ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2PortfolioGuardT15ResearchFreqtrade,
):
    """One-point neighboring realized-equity trigger at 16%."""

    PORTFOLIO_GOVERNOR_TRIGGER = 0.16
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_portfolio_t16_20260820"


class BreakoutV17GridV16WatchScore2PortfolioGuardT17ResearchFreqtrade(
    BreakoutV17GridV16WatchScore2PortfolioGuardT15ResearchFreqtrade,
):
    """Narrowest neighboring realized-equity trigger at 17%."""

    PORTFOLIO_GOVERNOR_TRIGGER = 0.17
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_portfolio_t17_20260820"


class BreakoutV17GridV16WatchScore2PortfolioGuardT16GentleResearchFreqtrade(
    BreakoutV17GridV16WatchScore2PortfolioGuardT16ResearchFreqtrade,
):
    """Gentler 16% response retaining half risk and 75% in recovery."""

    PORTFOLIO_DEFENSIVE_SCALE = 0.50
    PORTFOLIO_RECOVERY_SCALE = 0.75
    STRATEGY_VERSION = "breakout_v17_grid_v16_watch_s2_portfolio_t16_gentle_20260820"


class BreakoutV17GridV16ParetoCandidateResearchFreqtrade(
    BreakoutV17GridV16WatchScore2PortfolioGuardT16GentleResearchFreqtrade,
):
    """Pareto candidate: retain score-three watch signals and smooth risk.

    This remains a research alias until it passes exact 1m and continuous
    cross-year validation.  It does not replace or mutate the frozen V16 /
    Grid V15 production baseline.
    """

    V17_WATCH_REJECT_SCORES = (2, 4, 5)
    STRATEGY_VERSION = "breakout_v17_grid_v16_pareto_candidate_20260820"


class BreakoutV17GridV16CurrentRegimeResearchFreqtrade(
    BreakoutV17GridV16MatureReversalRisk50ResearchFreqtrade,
):
    """Frozen V17/Grid V16 finalist for current-regime paper testing.

    This alias deliberately does not imply a production promotion.  It is the
    higher-return research branch: a completed-candle momentum reversal for
    mature low-score Breakout shorts plus half risk for Grid shorts opened
    after the broad market is already down at least one percent over 24h.
    """

    STRATEGY_VERSION = "breakout_v17_grid_v16_current_regime_research_20260820"


class BreakoutV17GridV16DefensiveResearchFreqtrade(
    BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade,
):
    """Frozen defensive comparator; not a production promotion."""

    STRATEGY_VERSION = "breakout_v17_grid_v16_defensive_research_20260820"

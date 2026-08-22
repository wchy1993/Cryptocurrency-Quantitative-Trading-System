from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any

import pandas as pd
from freqtrade.strategy import Trade, stoploss_from_absolute

from BreakoutV17GridV16AdaptiveProtectionResearchFreqtrade import (
    BreakoutV17GridV16ParetoFinalResearchFreqtrade,
)


class _V18DynamicShadowWalletProxy:
    """Read-only wallet view excluding protected dynamic-exit savings."""

    def __init__(self, wallets: Any, reserve: float) -> None:
        self._wallets = wallets
        self._reserve = max(0.0, float(reserve))

    def get_total_stake_amount(self) -> float:
        return max(
            0.0,
            float(self._wallets.get_total_stake_amount()) - self._reserve,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wallets, name)


class _V18DynamicPositionManagementMixin:
    """Manage open positions from completed-candle state changes.

    Losses are cut early only when the original directional thesis has failed
    on both completed 4h momentum and the completed 1h EMA structure.  Profit
    floors arm progressively after confirmed progress.  Strong high-score
    trends use a wider runner schedule so ordinary pullbacks do not liquidate
    the long-tail trades that drive portfolio profit.
    """

    V18_BO_MIN_HOLD_MINUTES = 120
    V18_BO_EARLY_MAX_PEAK_R = 0.25
    V18_BO_EARLY_MAX_CURRENT_R = -0.35
    V18_BO_REVERSAL_MAX_DIRECTIONAL_RETURN_4H = -0.005
    V18_BO_STRONG_MIN_DIRECTIONAL_RETURN_4H = 0.005
    V18_BO_STRONG_MIN_EMA_SPREAD_ATR = 0.25
    V18_BO_STRONG_MIN_MARKET_EFFICIENCY = 0.35
    V18_BO_RUNNER_MIN_SCORE = 4
    V18_BO_RUNNER_MIN_PEAK_R = 3.0

    # A trade which cannot produce even 0.10R of favorable excursion during
    # its first 15 minutes has not confirmed the breakout thesis.  Arm a
    # shallow loss floor for that narrow path only.  Once a trade has shown
    # real follow-through it is permanently excluded, preserving runners.
    V18_ENABLE_BO_NO_FOLLOW_STOP = True
    V18_BO_NO_FOLLOW_MINUTES = 15
    V18_BO_NO_FOLLOW_MAX_BEST_R = 0.10
    V18_BO_NO_FOLLOW_FLOOR_R = -0.10
    V18_BO_NO_FOLLOW_MAX_SCORE = 4
    V18_BO_NO_FOLLOW_INCLUDE_CAPTURE = False
    V18_BO_NO_FOLLOW_BEST_R_KEY = "v18_no_follow_best_r"
    V18_BO_NO_FOLLOW_LAST_CANDLE_KEY = "v18_no_follow_last_1m_candle"
    V18_BO_NO_FOLLOW_DISARMED_KEY = "v18_no_follow_disarmed"
    V18_BO_NO_FOLLOW_RESERVE_KEY = "v18_no_follow_protected_reserve"
    V18_BO_NO_FOLLOW_SHADOW_PROFIT_ABS_KEY = (
        "v18_no_follow_shadow_profit_abs"
    )
    V18_BO_NO_FOLLOW_SHADOW_RETURN_KEY = "v18_no_follow_shadow_return"
    V18_BO_NO_FOLLOW_RESERVE_SCALE = 1.0
    V18_BO_NO_FOLLOW_RECONCILE_BALANCE_MULTIPLE = 0.0
    V18_BO_NO_FOLLOW_RECONCILE_START_BALANCE_MULTIPLE = 1.0
    V18_BO_NO_FOLLOW_RECONCILE_FULL_BALANCE_MULTIPLE = 0.0
    V18_BO_NO_FOLLOW_RECONCILE_SCALE = 1.0
    V18_BO_NO_FOLLOW_RECONCILE_MIN_PROPOSED_STAKE = 0.0
    V18_BO_NO_FOLLOW_USE_NATIVE_RESERVE_LEDGER = False
    V18_BO_NO_FOLLOW_PARTIAL_FRACTION = 0.0
    V18_BO_NO_FOLLOW_PARTIAL_REQUESTED_KEY = (
        "v18_no_follow_partial_requested"
    )
    V18_BO_NO_FOLLOW_PARTIAL_TAG = "bo_v18_no_follow_partial"

    V18_BO_REVERSAL_FLOORS = (
        (0.75, -0.15),
        (1.50, 0.15),
        (3.00, 0.75),
        (6.00, 1.75),
    )
    V18_BO_RUNNER_FLOORS = (
        (2.00, -0.25),
        (4.00, 0.25),
        (8.00, 1.50),
    )
    V18_BO_LONG_TAIL_TRIGGER_R = 10.0
    V18_BO_LONG_TAIL_GIVEBACK_R = 7.0
    V18_BO_RUNNER_LONG_TAIL_TRIGGER_R = 12.0
    V18_BO_RUNNER_LONG_TAIL_GIVEBACK_R = 8.0

    V18_GRID_MIN_HOLD_MINUTES = 180
    V18_GRID_EARLY_MAX_BEST_R = 0.15
    V18_GRID_EARLY_MAX_CURRENT_R = -0.45
    V18_GRID_PROTECT_MIN_BEST_R = 0.50
    V18_GRID_PROTECT_MAX_CURRENT_R = 0.05

    V18_ENABLE_BO_EARLY_FAILURE = True
    V18_ENABLE_BO_REVERSAL_FLOOR = True
    V18_ENABLE_GRID_EARLY_FAILURE = True
    V18_ENABLE_GRID_REVERSAL_FLOOR = True

    V18_BO_EARLY_EXIT_REASON = "bo_v18_dynamic_thesis_failure"
    V18_BO_NO_FOLLOW_EXIT_REASON = "bo_v18_no_follow_protection"
    V18_BO_REVERSAL_EXIT_REASON = "bo_v18_dynamic_reversal_floor"
    V18_GRID_EARLY_EXIT_REASON = "grid_v18_dynamic_thesis_failure"
    V18_GRID_REVERSAL_EXIT_REASON = "grid_v18_dynamic_reversal_floor"

    @staticmethod
    def _v18_tighter_stop(
        inherited: float | None,
        protected: float | None,
    ) -> float | None:
        if protected is None or protected <= 0.0:
            return inherited
        if inherited is None:
            return protected
        return min(abs(float(inherited)), abs(float(protected)))

    @staticmethod
    def _v18_floor_for_peak(
        peak_r: float,
        schedule: tuple[tuple[float, float], ...],
    ) -> float | None:
        floor: float | None = None
        for trigger, candidate in schedule:
            if float(peak_r) >= float(trigger):
                floor = max(
                    float(candidate),
                    floor if floor is not None else float("-inf"),
                )
        return floor

    def _v18_directional_context(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
    ) -> tuple[bool, bool]:
        """Return (confirmed_reversal, strong_trend) from completed data."""

        row = self._latest_row(pair, current_time)
        if row is None:
            return False, False
        direction = -1.0 if trade.is_short else 1.0
        atr = max(float(row.get("atr", 0.0) or 0.0), 1e-12)
        directional_return = direction * float(
            row.get("symbol_return_4h", 0.0) or 0.0
        )
        directional_spread = direction * (
            float(row.get("fast_ema", 0.0) or 0.0)
            - float(row.get("slow_ema", 0.0) or 0.0)
        ) / atr
        reversal = bool(
            directional_return
            <= float(self.V18_BO_REVERSAL_MAX_DIRECTIONAL_RETURN_4H)
            and directional_spread <= 0.0
        )
        strong = bool(
            directional_return
            >= float(self.V18_BO_STRONG_MIN_DIRECTIONAL_RETURN_4H)
            and directional_spread
            >= float(self.V18_BO_STRONG_MIN_EMA_SPREAD_ATR)
            and float(row.get("market_efficiency", 0.0) or 0.0)
            >= float(self.V18_BO_STRONG_MIN_MARKET_EFFICIENCY)
        )
        return reversal, strong

    def _v18_breakout_locked_r(
        self,
        trade: Trade,
        peak_r: float,
        reversal: bool,
        strong: bool,
    ) -> float | None:
        score = int(trade.get_custom_data("bo_score", 0))
        runner = bool(
            strong
            and score >= int(self.V18_BO_RUNNER_MIN_SCORE)
            and peak_r >= float(self.V18_BO_RUNNER_MIN_PEAK_R)
        )
        if runner:
            # A confirmed strong trend keeps the inherited wide runner path.
            # Dynamic protection is armed only after that support disappears.
            return None
        if not reversal:
            return None
        locked = self._v18_floor_for_peak(
            peak_r,
            tuple(self.V18_BO_REVERSAL_FLOORS),
        )
        if peak_r >= float(self.V18_BO_LONG_TAIL_TRIGGER_R):
            locked = max(
                locked if locked is not None else float("-inf"),
                peak_r - float(self.V18_BO_LONG_TAIL_GIVEBACK_R),
            )
        return locked

    def _v18_breakout_stop(
        self,
        trade: Trade,
        current_rate: float,
        locked_r: float,
    ) -> float:
        side = -1.0 if trade.is_short else 1.0
        unit_risk = max(
            float(trade.get_custom_data("initial_unit_risk", 0.0)),
            1e-12,
        )
        stop_rate = trade.open_rate + side * (
            float(locked_r) * unit_risk
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

    def _v18_breakout_best_r(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
    ) -> float:
        """Persist favorable progress from completed 1m candles.

        Freqtrade may update ``trade.max_rate``/``min_rate`` after strategy
        callbacks for the current detail candle.  Reading the explicitly
        completed precision candle avoids that callback-order ambiguity and
        keeps live and backtest decisions causal.
        """

        best_r = float(
            trade.get_custom_data(
                self.V18_BO_NO_FOLLOW_BEST_R_KEY,
                float("-inf"),
            )
        )
        candle = self._latest_completed_precision_candle(pair, current_time)
        if candle is not None and candle.get("date") is not None:
            candle_date = pd.Timestamp(candle["date"])
            if candle_date.tzinfo is None:
                candle_date = candle_date.tz_localize("UTC")
            else:
                candle_date = candle_date.tz_convert("UTC")
            open_date = pd.Timestamp(self._trade_open_time(trade))
            if open_date.tzinfo is None:
                open_date = open_date.tz_localize("UTC")
            else:
                open_date = open_date.tz_convert("UTC")
            candle_key = candle_date.isoformat()
            if (
                candle_date >= open_date
                and candle_key
                != str(
                    trade.get_custom_data(
                        self.V18_BO_NO_FOLLOW_LAST_CANDLE_KEY,
                        "",
                    )
                )
            ):
                favorable_rate = float(
                    candle["low"] if trade.is_short else candle["high"]
                )
                best_r = max(
                    best_r,
                    float(self._trade_r(trade, favorable_rate)),
                )
                trade.set_custom_data(
                    self.V18_BO_NO_FOLLOW_BEST_R_KEY,
                    best_r,
                )
                trade.set_custom_data(
                    self.V18_BO_NO_FOLLOW_LAST_CANDLE_KEY,
                    candle_key,
                )
        elif best_r == float("-inf"):
            # Safe fallback for environments without the optional 1m feed.
            favorable_value = (
                trade.min_rate if trade.is_short else trade.max_rate
            )
            favorable_rate = float(
                current_rate if favorable_value is None else favorable_value
            )
            best_r = float(self._trade_r(trade, favorable_rate))
        return best_r

    def _v18_no_follow_locked_r(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
    ) -> float | None:
        if not bool(self.V18_ENABLE_BO_NO_FOLLOW_STOP):
            return None
        if (
            int(trade.get_custom_data("bo_score", 99))
            > int(self.V18_BO_NO_FOLLOW_MAX_SCORE)
            or (
                not bool(self.V18_BO_NO_FOLLOW_INCLUDE_CAPTURE)
                and bool(trade.get_custom_data("bo_capture", False))
            )
        ):
            return None
        if bool(
            trade.get_custom_data(
                self.V18_BO_NO_FOLLOW_DISARMED_KEY,
                False,
            )
        ):
            return None
        holding_minutes = int(
            (current_time - self._trade_open_time(trade)).total_seconds() / 60
        )
        best_r = self._v18_breakout_best_r(
            pair,
            trade,
            current_time,
            current_rate,
        )
        if holding_minutes < int(self.V18_BO_NO_FOLLOW_MINUTES):
            return None
        if best_r > float(self.V18_BO_NO_FOLLOW_MAX_BEST_R):
            trade.set_custom_data(
                self.V18_BO_NO_FOLLOW_DISARMED_KEY,
                True,
            )
            return None
        return float(self.V18_BO_NO_FOLLOW_FLOOR_R)

    def _v18_store_no_follow_reserve(
        self,
        trade: Trade,
        current_rate: float,
        protected_fraction: float = 1.0,
    ) -> None:
        """Reserve savings relative to the inherited precision hard stop."""

        if trade.get_custom_data(
            self.V18_BO_NO_FOLLOW_SHADOW_PROFIT_ABS_KEY,
            None,
        ) is not None:
            return
        plan = self._trade_stop_plan(trade)
        if plan is None:
            return
        current_profit = trade.calculate_profit(current_rate)
        counterfactual_profit = trade.calculate_profit(float(plan.hard_stop))
        current_total = float(current_profit.total_profit)
        counterfactual_total = float(counterfactual_profit.total_profit)
        counterfactual_return = float(
            counterfactual_profit.total_profit_ratio
        )
        if isfinite(counterfactual_total):
            trade.set_custom_data(
                self.V18_BO_NO_FOLLOW_SHADOW_PROFIT_ABS_KEY,
                counterfactual_total,
            )
        if isfinite(counterfactual_return):
            trade.set_custom_data(
                self.V18_BO_NO_FOLLOW_SHADOW_RETURN_KEY,
                counterfactual_return,
            )
        fraction = min(1.0, max(0.0, float(protected_fraction)))
        reserve = (current_total - counterfactual_total) * fraction
        if isfinite(reserve) and reserve > 0.0:
            trade.set_custom_data(
                self.V18_BO_NO_FOLLOW_RESERVE_KEY,
                reserve,
            )

    def _v18_shadow_profit_abs(self, trade: Trade) -> float | None:
        shadow = trade.get_custom_data(
            self.V18_BO_NO_FOLLOW_SHADOW_PROFIT_ABS_KEY,
            None,
        )
        value = (
            getattr(trade, "close_profit_abs", None)
            if shadow is None
            else shadow
        )
        if value is None:
            return None
        result = float(value)
        return result if isfinite(result) else None

    def _v18_shadow_close_return(self, trade: Trade) -> float | None:
        shadow = trade.get_custom_data(
            self.V18_BO_NO_FOLLOW_SHADOW_RETURN_KEY,
            None,
        )
        if shadow is None:
            return self._closed_trade_return(trade)
        result = float(shadow)
        return result if isfinite(result) else None

    def _recent_portfolio_returns(
        self,
        current_time: datetime,
    ) -> tuple[float, ...]:
        """Expose Final-equivalent returns to the inherited governor."""

        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        closed: list[tuple[datetime, float]] = []
        for closed_trade in Trade.get_trades_proxy(is_open=False):
            close_time = self._trade_close_time(closed_trade)
            value = self._v18_shadow_close_return(closed_trade)
            if close_time is None or close_time > boundary or value is None:
                continue
            closed.append((close_time, value))
        closed.sort(key=lambda item: item[0])
        return tuple(
            value
            for _close_time, value in closed[-self.PORTFOLIO_RECENT_WINDOW :]
        )

    def _v17_realized_drawdown(self, current_time: datetime) -> float:
        """Rebuild the inherited drawdown from the counterfactual ledger."""

        starting = float(self.wallets.get_starting_balance())
        if starting <= 0.0:
            return 1.0
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        completed: list[tuple[datetime, float]] = []
        for closed_trade in Trade.get_trades_proxy(is_open=False):
            close_time = self._trade_close_time(closed_trade)
            profit = self._v18_shadow_profit_abs(closed_trade)
            if close_time is None or close_time > boundary or profit is None:
                continue
            completed.append((close_time, profit))
        completed.sort(key=lambda item: item[0])
        equity = starting
        peak = starting
        for _close_time, profit in completed:
            equity += profit
            peak = max(peak, equity)
        if equity <= 0.0 or peak <= 0.0:
            return 1.0
        return max(0.0, 1.0 - equity / peak)

    def _v18_no_follow_protected_reserve(
        self,
        current_time: datetime,
        force_reconcile: bool = False,
    ) -> float:
        reserve = 0.0
        scale = min(
            2.0,
            max(0.0, float(self.V18_BO_NO_FOLLOW_RESERVE_SCALE)),
        )
        reconcile_multiple = max(
            0.0,
            float(self.V18_BO_NO_FOLLOW_RECONCILE_BALANCE_MULTIPLE),
        )
        reconcile = bool(force_reconcile)
        reconcile_fraction = 0.0
        starting = float(self.wallets.get_starting_balance())
        total = float(self.wallets.get_total_stake_amount())
        if reconcile_multiple > 0.0:
            reconcile = bool(
                isfinite(starting)
                and isfinite(total)
                and starting > 0.0
                and total >= starting * reconcile_multiple
            )
        full_multiple = max(
            0.0,
            float(self.V18_BO_NO_FOLLOW_RECONCILE_FULL_BALANCE_MULTIPLE),
        )
        if (
            full_multiple > 1.0
            and isfinite(starting)
            and isfinite(total)
            and starting > 0.0
        ):
            wallet_multiple = total / starting
            start_multiple = min(
                full_multiple,
                max(
                    1.0,
                    float(
                        self.V18_BO_NO_FOLLOW_RECONCILE_START_BALANCE_MULTIPLE
                    ),
                ),
            )
            reconcile_fraction = min(
                1.0,
                max(
                    0.0,
                    (wallet_multiple - start_multiple)
                    / max(full_multiple - start_multiple, 1e-12),
                ),
            )
        for closed_trade in Trade.get_trades_proxy(is_open=False):
            close_time = self._trade_close_time(closed_trade)
            if close_time is None or close_time > current_time:
                continue
            stored = float(
                closed_trade.get_custom_data(
                    self.V18_BO_NO_FOLLOW_RESERVE_KEY, 0.0
                )
            )
            value = stored * scale
            if reconcile or reconcile_fraction > 0.0:
                shadow = closed_trade.get_custom_data(
                    self.V18_BO_NO_FOLLOW_SHADOW_PROFIT_ABS_KEY,
                    None,
                )
                actual = getattr(closed_trade, "close_profit_abs", None)
                if shadow is not None and actual is not None:
                    derived = float(actual) - float(shadow)
                    if isfinite(derived) and derived > 0.0:
                        realized_value = derived * min(
                            1.0,
                            max(
                                0.0,
                                float(
                                    self.V18_BO_NO_FOLLOW_RECONCILE_SCALE
                                ),
                            ),
                        )
                        if reconcile:
                            value = realized_value
                        else:
                            value += reconcile_fraction * (
                                realized_value - value
                            )
            if isfinite(value) and value > 0.0:
                reserve += value
        return reserve

    @staticmethod
    def _v18_shadow_stake_scale(equity: float, reserve: float) -> float:
        if equity <= 0.0:
            return 0.0
        return min(1.0, max(0.0, (equity - reserve) / equity))

    def _v17_reserved_savings(self, current_time: datetime) -> float:
        inherited = float(super()._v17_reserved_savings(current_time))
        if not bool(self.V18_BO_NO_FOLLOW_USE_NATIVE_RESERVE_LEDGER):
            return inherited
        return inherited + self._v18_no_follow_protected_reserve(current_time)

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
        if bool(self.V18_BO_NO_FOLLOW_USE_NATIVE_RESERVE_LEDGER):
            return float(
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
        reconcile_min_stake = max(
            0.0,
            float(self.V18_BO_NO_FOLLOW_RECONCILE_MIN_PROPOSED_STAKE),
        )
        reserve = self._v18_no_follow_protected_reserve(
            current_time,
            force_reconcile=bool(
                reconcile_min_stake > 0.0
                and float(proposed_stake) >= reconcile_min_stake
            ),
        )
        real_wallets = self.wallets
        if reserve > 0.0:
            self.wallets = _V18DynamicShadowWalletProxy(
                real_wallets,
                reserve,
            )
        try:
            return float(
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
        finally:
            self.wallets = real_wallets

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
        inherited = super().adjust_trade_position(
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
        if inherited is not None:
            return inherited
        fraction = min(
            0.95,
            max(0.0, float(self.V18_BO_NO_FOLLOW_PARTIAL_FRACTION)),
        )
        if (
            fraction <= 0.0
            or self._component(trade.enter_tag) != "breakout"
            or bool(
                trade.get_custom_data(
                    self.V18_BO_NO_FOLLOW_PARTIAL_REQUESTED_KEY,
                    False,
                )
            )
        ):
            return None
        locked_r = self._v18_no_follow_locked_r(
            trade.pair,
            trade,
            current_time,
            current_exit_rate,
        )
        if (
            locked_r is None
            or self._trade_r(trade, current_exit_rate) > locked_r
        ):
            return None
        self._v18_store_no_follow_reserve(
            trade,
            current_exit_rate,
            protected_fraction=fraction,
        )
        trade.set_custom_data(
            self.V18_BO_NO_FOLLOW_PARTIAL_REQUESTED_KEY,
            True,
        )
        return (
            -float(trade.stake_amount) * fraction,
            self.V18_BO_NO_FOLLOW_PARTIAL_TAG,
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
        if self._component(trade.enter_tag) != "breakout":
            return inherited
        if (
            float(self.V18_BO_NO_FOLLOW_PARTIAL_FRACTION) > 0.0
            and bool(
                trade.get_custom_data(
                    self.V18_BO_NO_FOLLOW_PARTIAL_REQUESTED_KEY,
                    False,
                )
            )
        ):
            plan = self._trade_stop_plan(trade)
            if plan is not None:
                return abs(
                    float(
                        stoploss_from_absolute(
                            float(plan.hard_stop),
                            current_rate=current_rate,
                            is_short=trade.is_short,
                            leverage=trade.leverage,
                        )
                    )
                )
        if float(self.V18_BO_NO_FOLLOW_PARTIAL_FRACTION) <= 0.0:
            no_follow_r = self._v18_no_follow_locked_r(
                pair,
                trade,
                current_time,
                current_rate,
            )
            if no_follow_r is not None:
                inherited = self._v18_tighter_stop(
                    inherited,
                    self._v18_breakout_stop(
                        trade,
                        current_rate,
                        no_follow_r,
                    ),
                )
        if not bool(self.V18_ENABLE_BO_REVERSAL_FLOOR):
            return inherited
        peak_r = float(self._confirmed_peak_r(pair, trade, current_time))
        reversal, strong = self._v18_directional_context(
            pair,
            trade,
            current_time,
        )
        locked_r = self._v18_breakout_locked_r(
            trade,
            peak_r,
            reversal,
            strong,
        )
        if locked_r is None:
            return inherited
        return self._v18_tighter_stop(
            inherited,
            self._v18_breakout_stop(trade, current_rate, locked_r),
        )

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
            getattr(order, "ft_order_side", None) != trade.exit_side
            or getattr(order, "ft_order_tag", None)
            != self.V18_BO_NO_FOLLOW_PARTIAL_TAG
        ):
            return
        plan = self._trade_stop_plan(trade)
        adjust_stop = getattr(trade, "adjust_stop_loss", None)
        if plan is None or not callable(adjust_stop):
            return
        fill_rate = float(
            getattr(order, "safe_price", None)
            or getattr(order, "average", None)
            or trade.open_rate
        )
        ratio = abs(
            float(
                stoploss_from_absolute(
                    float(plan.hard_stop),
                    current_rate=fill_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage,
                )
            )
        )
        adjust_stop(fill_rate, ratio, allow_refresh=True)

    def _v18_grid_campaign_r(
        self,
        trade: Trade,
        current_rate: float,
    ) -> tuple[float, float]:
        risk_budget = max(
            float(trade.get_custom_data("grid_risk_budget", 0.0)),
            1e-12,
        )
        favorable_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate if favorable_value is None else favorable_value
        )
        best_r = float(
            trade.calculate_profit(favorable_rate).total_profit
        ) / risk_budget
        current_r = float(
            trade.calculate_profit(current_rate).total_profit
        ) / risk_budget
        return best_r, current_r

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
        component = self._component(trade.enter_tag)
        if component not in {"breakout", "grid"}:
            return None
        if component == "breakout":
            no_follow_r = self._v18_no_follow_locked_r(
                pair,
                trade,
                current_time,
                current_rate,
            )
            if (
                no_follow_r is not None
                and self._trade_r(trade, current_rate) <= no_follow_r
            ):
                if float(self.V18_BO_NO_FOLLOW_PARTIAL_FRACTION) > 0.0:
                    return None
                self._v18_store_no_follow_reserve(trade, current_rate)
                return self.V18_BO_NO_FOLLOW_EXIT_REASON
        reversal, strong = self._v18_directional_context(
            pair,
            trade,
            current_time,
        )
        if not reversal or strong:
            return None
        holding_minutes = int(
            (current_time - self._trade_open_time(trade)).total_seconds() / 60
        )
        if component == "breakout":
            peak_r = float(self._confirmed_peak_r(pair, trade, current_time))
            current_r = float(self._trade_r(trade, current_rate))
            if (
                bool(self.V18_ENABLE_BO_EARLY_FAILURE)
                and
                holding_minutes >= int(self.V18_BO_MIN_HOLD_MINUTES)
                and peak_r <= float(self.V18_BO_EARLY_MAX_PEAK_R)
                and current_r <= float(self.V18_BO_EARLY_MAX_CURRENT_R)
            ):
                return self.V18_BO_EARLY_EXIT_REASON
            locked_r = self._v18_breakout_locked_r(
                trade,
                peak_r,
                reversal=True,
                strong=False,
            )
            if (
                bool(self.V18_ENABLE_BO_REVERSAL_FLOOR)
                and locked_r is not None
                and current_r <= locked_r
            ):
                return self.V18_BO_REVERSAL_EXIT_REASON
            return None

        best_r, current_r = self._v18_grid_campaign_r(trade, current_rate)
        tp_count = int(trade.get_custom_data("grid_tp_count", 0))
        if (
            bool(self.V18_ENABLE_GRID_EARLY_FAILURE)
            and
            holding_minutes >= int(self.V18_GRID_MIN_HOLD_MINUTES)
            and tp_count == 0
            and best_r <= float(self.V18_GRID_EARLY_MAX_BEST_R)
            and current_r <= float(self.V18_GRID_EARLY_MAX_CURRENT_R)
        ):
            return self.V18_GRID_EARLY_EXIT_REASON
        if (
            bool(self.V18_ENABLE_GRID_REVERSAL_FLOOR)
            and
            tp_count >= 1
            and best_r >= float(self.V18_GRID_PROTECT_MIN_BEST_R)
            and current_r <= float(self.V18_GRID_PROTECT_MAX_CURRENT_R)
        ):
            return self.V18_GRID_REVERSAL_EXIT_REASON
        return None


class BreakoutV17GridV16DynamicPositionResearchFreqtrade(
    _V18DynamicPositionManagementMixin,
    BreakoutV17GridV16ParetoFinalResearchFreqtrade,
):
    """Balanced completed-candle dynamic exit candidate."""

    STRATEGY_VERSION = "breakout_v17_grid_v16_dynamic_position_20260821"
    ADAPTIVE_STATE_BASENAME = "breakout_v17_grid_v16_dynamic_position"


class BreakoutV17GridV16DynamicPositionWideRunnerResearchFreqtrade(
    BreakoutV17GridV16DynamicPositionResearchFreqtrade,
):
    """Wider neighbour prioritizing long-tail runner retention."""

    V18_BO_EARLY_MAX_PEAK_R = 0.15
    V18_BO_EARLY_MAX_CURRENT_R = -0.50
    V18_BO_REVERSAL_MAX_DIRECTIONAL_RETURN_4H = -0.01
    V18_BO_REVERSAL_FLOORS = (
        (1.00, -0.30),
        (2.00, 0.00),
        (4.00, 0.75),
        (8.00, 2.00),
    )
    V18_BO_LONG_TAIL_TRIGGER_R = 12.0
    V18_BO_LONG_TAIL_GIVEBACK_R = 8.0
    V18_GRID_EARLY_MAX_BEST_R = 0.10
    V18_GRID_EARLY_MAX_CURRENT_R = -0.55
    V18_GRID_PROTECT_MIN_BEST_R = 0.75
    V18_GRID_PROTECT_MAX_CURRENT_R = 0.00
    STRATEGY_VERSION = "breakout_v17_grid_v16_dynamic_position_wide_20260821"
    ADAPTIVE_STATE_BASENAME = "breakout_v17_grid_v16_dynamic_position_wide"


class BreakoutV17GridV16DynamicEarlyLossOnlyResearchFreqtrade(
    BreakoutV17GridV16DynamicPositionResearchFreqtrade,
):
    """Ablation: dynamic thesis-failure exits without profit floors."""

    V18_ENABLE_BO_REVERSAL_FLOOR = False
    V18_ENABLE_GRID_REVERSAL_FLOOR = False
    STRATEGY_VERSION = "breakout_v17_grid_v16_dynamic_early_loss_only_20260821"
    ADAPTIVE_STATE_BASENAME = "breakout_v17_grid_v16_dynamic_early_loss_only"


class BreakoutV17GridV16DynamicProfitReversalOnlyResearchFreqtrade(
    BreakoutV17GridV16DynamicPositionResearchFreqtrade,
):
    """Ablation: reversal profit floors without early loss exits."""

    V18_ENABLE_BO_EARLY_FAILURE = False
    V18_ENABLE_GRID_EARLY_FAILURE = False
    STRATEGY_VERSION = "breakout_v17_grid_v16_dynamic_profit_reversal_only_20260821"
    ADAPTIVE_STATE_BASENAME = "breakout_v17_grid_v16_dynamic_profit_reversal_only"


class BreakoutV17GridV16DynamicNoFollowConservativeResearchFreqtrade(
    BreakoutV17GridV16DynamicEarlyLossOnlyResearchFreqtrade,
):
    """Narrow neighbour: protect only trades with at most 0.05R progress."""

    V18_BO_NO_FOLLOW_MAX_BEST_R = 0.05
    STRATEGY_VERSION = "breakout_v17_grid_v16_dynamic_no_follow_p05_20260821"
    ADAPTIVE_STATE_BASENAME = "breakout_v17_grid_v16_dynamic_no_follow_p05"


class BreakoutV17GridV16DynamicNoFollowModerateResearchFreqtrade(
    BreakoutV17GridV16DynamicEarlyLossOnlyResearchFreqtrade,
):
    """Wider loss floor neighbour at -0.15R after weak early progress."""

    V18_BO_NO_FOLLOW_FLOOR_R = -0.15
    STRATEGY_VERSION = "breakout_v17_grid_v16_dynamic_no_follow_r15_20260821"
    ADAPTIVE_STATE_BASENAME = "breakout_v17_grid_v16_dynamic_no_follow_r15"


class BreakoutV17GridV16DynamicNoFollowLongOnlyResearchFreqtrade(
    BreakoutV17GridV16DynamicEarlyLossOnlyResearchFreqtrade,
):
    """Protect weak ordinary longs without perturbing delayed short paths."""

    STRATEGY_VERSION = "breakout_v17_grid_v16_dynamic_no_follow_long_20260821"
    ADAPTIVE_STATE_BASENAME = "breakout_v17_grid_v16_dynamic_no_follow_long"

    def _v18_no_follow_locked_r(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
    ) -> float | None:
        if trade.is_short:
            return None
        return super()._v18_no_follow_locked_r(
            pair,
            trade,
            current_time,
            current_rate,
        )


class BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowLongOnlyResearchFreqtrade,
):
    """Protect weak longs only inside a narrow realized-DD stress band.

    Normal/high-water entries retain the exact Pareto Final path, while the
    rule can still intervene before the inherited 18% portfolio governor is
    reached.  The upper boundary avoids changing deeply underwater recovery
    sequencing, where slot timing and the recent-eight-trade governor are
    especially sensitive to an earlier close.
    """

    V18_BO_NO_FOLLOW_MIN_REALIZED_DD = 0.08
    V18_BO_NO_FOLLOW_MAX_REALIZED_DD = 0.10
    V18_BO_NO_FOLLOW_RESERVE_SCALE = 1.0
    V18_BO_NO_FOLLOW_PARTIAL_FRACTION = 0.90
    V18_BO_NO_FOLLOW_STRESS_DECISION_KEY = (
        "v18_no_follow_stress_band_eligible"
    )
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_08_10_long_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_08_10_long"
    )

    def _v18_no_follow_locked_r(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
    ) -> float | None:
        locked_r = super()._v18_no_follow_locked_r(
            pair,
            trade,
            current_time,
            current_rate,
        )
        if locked_r is None:
            return None
        eligible = trade.get_custom_data(
            self.V18_BO_NO_FOLLOW_STRESS_DECISION_KEY,
            None,
        )
        if eligible is None:
            drawdown = float(self._v17_realized_drawdown(current_time))
            eligible = bool(
                float(self.V18_BO_NO_FOLLOW_MIN_REALIZED_DD)
                <= drawdown
                < float(self.V18_BO_NO_FOLLOW_MAX_REALIZED_DD)
            )
            trade.set_custom_data(
                self.V18_BO_NO_FOLLOW_STRESS_DECISION_KEY,
                eligible,
            )
        if not bool(eligible):
            return None
        return locked_r


class BreakoutV17GridV16DynamicNoFollowStressBandLongReserve10005ResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Calibration neighbour: offset partial-tail settlement granularity."""

    V18_BO_NO_FOLLOW_RESERVE_SCALE = 1.0005
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_reserve_10005_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_reserve_10005"
    )


class BreakoutV17GridV16DynamicNoFollowStressBandLongReserve1005ResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Calibration neighbour at +0.5% projected reserve."""

    V18_BO_NO_FOLLOW_RESERVE_SCALE = 1.005
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_reserve_1005_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_reserve_1005"
    )


class BreakoutV17GridV16DynamicNoFollowStressBandLongReserve101ResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Calibration neighbour at +1.0% projected reserve."""

    V18_BO_NO_FOLLOW_RESERVE_SCALE = 1.01
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_reserve_101_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_reserve_101"
    )


class BreakoutV17GridV16DynamicNoFollowStressBandLongStagedReserveResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Reconcile realized savings after the wallet leaves its tight regime."""

    V18_BO_NO_FOLLOW_RESERVE_SCALE = 1.0005
    V18_BO_NO_FOLLOW_RECONCILE_BALANCE_MULTIPLE = 20.0
    V18_BO_NO_FOLLOW_RECONCILE_SCALE = 1.0
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_staged_reserve_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_staged_reserve"
    )


class BreakoutV17GridV16DynamicNoFollowStressBandLongStagedReserve999ResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongStagedReserveResearchFreqtrade,
):
    """Settlement neighbour retaining a 0.1% reserve reconciliation buffer."""

    V18_BO_NO_FOLLOW_RECONCILE_SCALE = 0.999
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_staged_999_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_stress_staged_999"
    )


class BreakoutV17GridV16DynamicNoFollowStressBandLongConsistentShadowResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Use one counterfactual equity scale for wallet and stake constraints."""

    V18_BO_NO_FOLLOW_RESERVE_SCALE = 1.0
    V18_BO_NO_FOLLOW_RECONCILE_BALANCE_MULTIPLE = 1e-9
    V18_BO_NO_FOLLOW_RECONCILE_SCALE = 1.0
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_consistent_shadow_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_consistent_shadow"
    )


class BreakoutV17GridV16DynamicNoFollowStressBandLongReinvestAllResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Allow all realized dynamic protection savings to compound."""

    V18_BO_NO_FOLLOW_RESERVE_SCALE = 0.0
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_reinvest_all_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_reinvest_all"
    )


class BreakoutV17GridV16DynamicNoFollowStressBandLongReserveHalfResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Reinvest half of dynamic protection savings."""

    V18_BO_NO_FOLLOW_RESERVE_SCALE = 0.5
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_reserve_half_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_reserve_half"
    )


class BreakoutV17GridV16DynamicNoFollowStressBandLongNativeLedgerResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Route realized protection savings through Final's native ledger."""

    V18_BO_NO_FOLLOW_USE_NATIVE_RESERVE_LEDGER = True
    V18_BO_NO_FOLLOW_RECONCILE_BALANCE_MULTIPLE = 0.0
    V18_BO_NO_FOLLOW_RECONCILE_SCALE = 1.0
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_native_ledger_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_native_ledger"
    )


class BreakoutV17GridV16DynamicNoFollowStressBandLongSmoothReserveResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Smoothly reconcile settlement dust as wallet constraints disappear."""

    V18_BO_NO_FOLLOW_RESERVE_SCALE = 1.0005
    V18_BO_NO_FOLLOW_RECONCILE_BALANCE_MULTIPLE = 0.0
    V18_BO_NO_FOLLOW_RECONCILE_START_BALANCE_MULTIPLE = 50.0
    V18_BO_NO_FOLLOW_RECONCILE_FULL_BALANCE_MULTIPLE = 100.0
    V18_BO_NO_FOLLOW_RECONCILE_SCALE = 1.0
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_smooth_reserve_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_smooth_reserve"
    )


class BreakoutV17GridV16DynamicNoFollowStressBandLongLargeStakeLedgerResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Reconcile settlement dust only when stake size can amplify it."""

    V18_BO_NO_FOLLOW_RESERVE_SCALE = 1.0005
    V18_BO_NO_FOLLOW_RECONCILE_BALANCE_MULTIPLE = 0.0
    V18_BO_NO_FOLLOW_RECONCILE_FULL_BALANCE_MULTIPLE = 0.0
    V18_BO_NO_FOLLOW_RECONCILE_MIN_PROPOSED_STAKE = 10_000.0
    V18_BO_NO_FOLLOW_RECONCILE_SCALE = 1.0
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_no_follow_large_stake_ledger_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_no_follow_large_stake_ledger"
    )


class BreakoutV17GridV16DynamicNoFollowGridPostTpReversalGuardResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Do not rebuild a score-three Grid tail into confirmed reversal."""

    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_grid_post_tp_reversal_guard_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_grid_post_tp_reversal_guard"
    )

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
        if adjustment is None:
            return None
        stake_delta = float(
            adjustment[0] if isinstance(adjustment, tuple) else adjustment
        )
        if (
            stake_delta <= 0.0
            or trade.enter_tag != "grid_v8_short_s3"
            or int(getattr(trade, "nr_of_successful_exits", 0)) <= 0
        ):
            return adjustment
        reversal, _strong = self._v18_directional_context(
            trade.pair,
            trade,
            current_time,
        )
        return None if reversal else adjustment


class BreakoutV17GridV16DynamicNoFollowGridScore3NoRebuildResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Keep a de-risked score-three Grid campaign de-risked after its first TP.

    Once a campaign has successfully reduced exposure, adding that exposure
    back during the same campaign converts banked progress into fresh reversal
    risk.  The initial position and every pre-TP DCA remain unchanged, while
    the surviving tail can still capture a large continuation move.
    """

    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_grid_s3_no_rebuild_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_grid_s3_no_rebuild"
    )

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
        if adjustment is None:
            return None
        stake_delta = float(
            adjustment[0] if isinstance(adjustment, tuple) else adjustment
        )
        if (
            stake_delta > 0.0
            and trade.enter_tag == "grid_v8_short_s3"
            and int(getattr(trade, "nr_of_successful_exits", 0)) > 0
        ):
            return None
        return adjustment


class BreakoutV17GridV16DynamicNoFollowGridAllNoRebuildResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowGridScore3NoRebuildResearchFreqtrade,
):
    """Broader ablation: never rebuild any Grid campaign after a first TP."""

    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_grid_all_no_rebuild_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_grid_all_no_rebuild"
    )

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
        if adjustment is None:
            return None
        stake_delta = float(
            adjustment[0] if isinstance(adjustment, tuple) else adjustment
        )
        if (
            stake_delta > 0.0
            and self._component(trade.enter_tag) == "grid"
            and int(getattr(trade, "nr_of_successful_exits", 0)) > 0
        ):
            return None
        return adjustment


class BreakoutV17GridV16DynamicNoFollowGridScore3FlatTailExitResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
):
    """Release a de-risked score-three tail when its trend has gone flat.

    A post-TP DCA is allowed when completed 4h momentum still has a material
    direction.  If momentum is effectively flat, rebuilding the campaign has
    no directional edge; close the small residual tail instead so banked
    profit and the portfolio slot are both protected.  This does not touch
    pre-TP DCA or post-TP campaigns with renewed momentum.
    """

    V18_GRID_POST_TP_FLAT_RETURN_4H = 0.002
    V18_GRID_POST_TP_FLAT_EXIT_TAG = "grid_v18_flat_tail_exit"
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_grid_s3_flat_tail_exit_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_grid_s3_flat_tail_exit"
    )

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
        if adjustment is None:
            return None
        stake_delta = float(
            adjustment[0] if isinstance(adjustment, tuple) else adjustment
        )
        if (
            stake_delta <= 0.0
            or trade.enter_tag != "grid_v8_short_s3"
            or int(getattr(trade, "nr_of_successful_exits", 0)) <= 0
            or int(getattr(trade, "nr_of_successful_entries", 0)) != 1
        ):
            return adjustment
        row = self._latest_row(trade.pair, current_time)
        if row is None:
            return adjustment
        return_4h = float(row.get("symbol_return_4h", float("inf")))
        if (
            not isfinite(return_4h)
            or abs(return_4h)
            > float(self.V18_GRID_POST_TP_FLAT_RETURN_4H)
        ):
            return adjustment
        return (-float(trade.stake_amount), self.V18_GRID_POST_TP_FLAT_EXIT_TAG)


class BreakoutV17GridV16DynamicRiskBalancedResearchFreqtrade(
    BreakoutV17GridV16DynamicNoFollowGridScore3FlatTailExitResearchFreqtrade,
):
    """Dynamic Final candidate with a causal saturated-Grid failure exit.

    A fully loaded Grid campaign has spent its planned averaging capacity.  If
    it still has no TP, is already below -0.33R, and the completed 4h candle
    confirms at least 1.5 percent of adverse momentum, retaining it until the
    inherited trailing stop only magnifies a failed thesis.  Partially loaded
    campaigns, post-TP rebuilds, and all directional runners remain untouched.
    """

    V18_GRID_SATURATED_MIN_HOLD_MINUTES = 120
    V18_GRID_SATURATED_MAX_CURRENT_R = -0.33
    V18_GRID_SATURATED_MIN_ADVERSE_RETURN_4H = 0.015
    V18_GRID_SATURATED_MIN_CURRENT_PROFIT = -0.06
    V18_GRID_SATURATED_PARTIAL_FRACTION = 0.90
    V18_GRID_SATURATED_PARTIAL_KEY = "grid_v18_saturated_partial_done"
    V18_GRID_SATURATED_EXIT_TAG = "grid_v18_saturated_partial"
    V18_BO_NO_FOLLOW_ALLOWED_SCORES = (4,)
    STRATEGY_VERSION = (
        "breakout_v17_grid_v16_dynamic_risk_balanced_20260821"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v17_grid_v16_dynamic_risk_balanced"
    )

    def _v18_no_follow_locked_r(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
    ) -> float | None:
        if int(trade.get_custom_data("bo_score", 99)) not in tuple(
            int(score) for score in self.V18_BO_NO_FOLLOW_ALLOWED_SCORES
        ):
            return None
        return super()._v18_no_follow_locked_r(
            pair,
            trade,
            current_time,
            current_rate,
        )

    def _v18_saturated_grid_armed(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
    ) -> bool:
        if (
            trade.enter_tag != "grid_v8_short_s4"
            or int(getattr(trade, "nr_of_successful_entries", 0)) < 3
            or int(getattr(trade, "nr_of_successful_exits", 0)) > 0
            or float(current_profit)
            < float(self.V18_GRID_SATURATED_MIN_CURRENT_PROFIT)
        ):
            return False
        holding_minutes = int(
            (current_time - self._trade_open_time(trade)).total_seconds() / 60
        )
        if holding_minutes < int(self.V18_GRID_SATURATED_MIN_HOLD_MINUTES):
            return False
        row = self._latest_row(pair, current_time)
        if row is None:
            return False
        adverse_return_4h = float(
            row.get("symbol_return_4h", float("-inf"))
        )
        if (
            not isfinite(adverse_return_4h)
            or adverse_return_4h
            < float(self.V18_GRID_SATURATED_MIN_ADVERSE_RETURN_4H)
        ):
            return False
        _best_r, current_r = self._v18_grid_campaign_r(trade, current_rate)
        return bool(
            current_r <= float(self.V18_GRID_SATURATED_MAX_CURRENT_R)
        )

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
        if bool(
            trade.get_custom_data(
                self.V18_GRID_SATURATED_PARTIAL_KEY,
                False,
            )
        ):
            return None
        inherited = super().adjust_trade_position(
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
        if inherited is not None:
            return inherited
        if not self._v18_saturated_grid_armed(
            trade.pair,
            trade,
            current_time,
            current_exit_rate,
            current_exit_profit,
        ):
            return None
        fraction = min(
            0.95,
            max(0.0, float(self.V18_GRID_SATURATED_PARTIAL_FRACTION)),
        )
        if fraction <= 0.0:
            return None
        trade.set_custom_data(self.V18_GRID_SATURATED_PARTIAL_KEY, True)
        return (
            -float(trade.stake_amount) * fraction,
            self.V18_GRID_SATURATED_EXIT_TAG,
        )

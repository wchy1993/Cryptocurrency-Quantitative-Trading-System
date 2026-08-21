from __future__ import annotations

from datetime import datetime
import logging
import math
from typing import Any

from ccxt import DECIMAL_PLACES, ROUND_DOWN, ROUND_UP, TICK_SIZE
import pandas as pd

from freqtrade.exchange.exchange_utils import price_to_precision
from freqtrade.strategy import Trade, stoploss_from_absolute

from BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade import (
    BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade,
)


logger = logging.getLogger(__name__)


class _PrecisionStopPlan:
    """Executable initial-stop geometry for one Breakout campaign."""

    __slots__ = (
        "soft_stop",
        "inward_stop",
        "outward_stop",
        "hard_stop",
        "tick_size",
        "tick_atr",
        "confirmation_enabled",
        "stake_scale",
    )

    def __init__(
        self,
        *,
        soft_stop: float,
        inward_stop: float,
        outward_stop: float,
        hard_stop: float,
        tick_size: float,
        tick_atr: float,
        confirmation_enabled: bool,
        stake_scale: float,
    ) -> None:
        self.soft_stop = float(soft_stop)
        self.inward_stop = float(inward_stop)
        self.outward_stop = float(outward_stop)
        self.hard_stop = float(hard_stop)
        self.tick_size = float(tick_size)
        self.tick_atr = float(tick_atr)
        self.confirmation_enabled = bool(confirmation_enabled)
        self.stake_scale = float(stake_scale)


class _PrecisionConfirmedInitialStopMixin:
    """Keep structural risk stable across exchange tick-size boundaries.

    The frozen parent remains responsible for entries, leverage, Grid, runner
    protection and every non-initial exit.  This layer changes only the
    executable initial Breakout stop:

    * the mathematical stop is rounded away from the position instead of
      toward it;
    * coarse-tick markets receive a one-minute close-confirmed soft stop and
      a bounded emergency hard stop;
    * entry stake is reduced by the exact increase in executable unit risk.

    The confirmation path is deliberately restricted to markets where one
    exchange tick is material relative to entry ATR.  Fine-tick markets keep
    an immediate stop and only receive directionally safe price rounding.
    """

    PRECISION_GUARD_TIMEFRAME = "1m"
    PRECISION_GUARD_MIN_TICK_ATR = 0.05
    PRECISION_GUARD_HARD_BUFFER_ATR = 0.10
    PRECISION_GUARD_MAX_CANDLE_AGE_SECONDS = 75.0
    PRECISION_GUARD_EXIT_REASON = "bo_precision_soft_1m"
    PRECISION_GUARD_KEY_PREFIX = "bo_precision_guard_"
    order_types = {
        **BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade.order_types,
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 30,
        "stoploss_price_type": "last",
    }

    @staticmethod
    def _precision_tick_size(
        price_precision: float | None,
        precision_mode: int | None,
    ) -> float:
        if price_precision is None or precision_mode is None:
            return 0.0
        precision = float(price_precision)
        if not math.isfinite(precision) or precision < 0.0:
            return 0.0
        if precision_mode == TICK_SIZE:
            return precision
        if precision_mode == DECIMAL_PLACES:
            return 10.0 ** (-int(round(precision)))
        return 0.0

    def _precision_stop_plan(
        self,
        entry_rate: float,
        atr_value: float,
        is_short: bool,
        price_precision: float | None,
        precision_mode: int | None,
    ) -> _PrecisionStopPlan | None:
        entry = float(entry_rate)
        atr = float(atr_value)
        if (
            not math.isfinite(entry)
            or not math.isfinite(atr)
            or entry <= 0.0
            or atr <= 0.0
            or price_precision is None
            or precision_mode is None
        ):
            return None

        loss_side = 1.0 if is_short else -1.0
        soft_stop = entry + loss_side * float(self.BO_STOP_ATR) * atr
        inward_mode = ROUND_DOWN if is_short else ROUND_UP
        outward_mode = ROUND_UP if is_short else ROUND_DOWN
        inward_stop = price_to_precision(
            soft_stop,
            price_precision,
            precision_mode,
            rounding_mode=inward_mode,
        )
        outward_stop = price_to_precision(
            soft_stop,
            price_precision,
            precision_mode,
            rounding_mode=outward_mode,
        )
        tick_size = self._precision_tick_size(
            price_precision,
            precision_mode,
        )
        tick_atr = tick_size / atr if tick_size > 0.0 else 0.0
        confirmation_enabled = (
            tick_atr >= float(self.PRECISION_GUARD_MIN_TICK_ATR)
        )

        hard_raw = soft_stop
        if confirmation_enabled:
            hard_raw += (
                loss_side
                * float(self.PRECISION_GUARD_HARD_BUFFER_ATR)
                * atr
            )
        hard_stop = price_to_precision(
            hard_raw,
            price_precision,
            precision_mode,
            rounding_mode=outward_mode,
        )

        cost = 2.0 * float(self.SIDE_COST) * entry
        original_unit_risk = abs(soft_stop - entry) + cost
        executable_unit_risk = abs(hard_stop - entry) + cost
        if executable_unit_risk <= 0.0:
            return None
        stake_scale = min(
            1.0,
            max(0.0, original_unit_risk / executable_unit_risk),
        )
        return _PrecisionStopPlan(
            soft_stop=float(soft_stop),
            inward_stop=float(inward_stop),
            outward_stop=float(outward_stop),
            hard_stop=float(hard_stop),
            tick_size=float(tick_size),
            tick_atr=float(tick_atr),
            confirmation_enabled=bool(confirmation_enabled),
            stake_scale=float(stake_scale),
        )

    def _market_stop_plan(
        self,
        pair: str,
        entry_rate: float,
        atr_value: float,
        is_short: bool,
    ) -> _PrecisionStopPlan | None:
        try:
            market = self.dp.market(pair) if self.dp is not None else None
        except Exception:
            market = None
        if not market:
            return None
        precision = (market.get("precision") or {}).get("price")
        exchange = getattr(self.dp, "_exchange", None)
        precision_mode = getattr(exchange, "precision_mode_price", TICK_SIZE)
        return self._precision_stop_plan(
            entry_rate,
            atr_value,
            is_short,
            precision,
            precision_mode,
        )

    def _store_precision_stop_plan(
        self,
        trade: Trade,
        plan: _PrecisionStopPlan,
    ) -> None:
        values = {
            "soft_stop": plan.soft_stop,
            "inward_stop": plan.inward_stop,
            "outward_stop": plan.outward_stop,
            "hard_stop": plan.hard_stop,
            "tick_size": plan.tick_size,
            "tick_atr": plan.tick_atr,
            "confirmation_enabled": plan.confirmation_enabled,
            "stake_scale": plan.stake_scale,
        }
        for suffix, value in values.items():
            trade.set_custom_data(
                f"{self.PRECISION_GUARD_KEY_PREFIX}{suffix}",
                value,
            )

    @staticmethod
    def _arm_absolute_stop_on_filled_trade(
        trade: Trade,
        stop_rate: float,
    ) -> None:
        """Persist the executable hard stop before the next engine cycle.

        LIVE places the exchange-side STOP_MARKET from ``trade.stop_loss``.
        Arming it in the fill callback avoids first creating the inherited
        emergency -99% stop and replacing it one cycle later.
        """

        adjust_stop = getattr(trade, "adjust_stop_loss", None)
        if not callable(adjust_stop):
            return
        ratio = abs(
            float(
                stoploss_from_absolute(
                    float(stop_rate),
                    current_rate=float(trade.open_rate),
                    is_short=bool(trade.is_short),
                    leverage=float(trade.leverage or 1.0),
                )
            )
        )
        adjust_stop(
            float(trade.open_rate),
            ratio,
            allow_refresh=True,
        )

    @classmethod
    def _arm_hard_stop_on_filled_trade(
        cls,
        trade: Trade,
        plan: _PrecisionStopPlan,
    ) -> None:
        cls._arm_absolute_stop_on_filled_trade(trade, plan.hard_stop)

    def _trade_stop_plan(self, trade: Trade) -> _PrecisionStopPlan | None:
        prefix = self.PRECISION_GUARD_KEY_PREFIX
        hard_stop = float(
            trade.get_custom_data(f"{prefix}hard_stop", 0.0) or 0.0
        )
        if hard_stop > 0.0:
            return _PrecisionStopPlan(
                soft_stop=float(
                    trade.get_custom_data(f"{prefix}soft_stop", 0.0)
                    or 0.0
                ),
                inward_stop=float(
                    trade.get_custom_data(f"{prefix}inward_stop", 0.0)
                    or 0.0
                ),
                outward_stop=float(
                    trade.get_custom_data(f"{prefix}outward_stop", 0.0)
                    or 0.0
                ),
                hard_stop=hard_stop,
                tick_size=float(
                    trade.get_custom_data(f"{prefix}tick_size", 0.0)
                    or 0.0
                ),
                tick_atr=float(
                    trade.get_custom_data(f"{prefix}tick_atr", 0.0)
                    or 0.0
                ),
                confirmation_enabled=bool(
                    trade.get_custom_data(
                        f"{prefix}confirmation_enabled",
                        False,
                    )
                ),
                stake_scale=float(
                    trade.get_custom_data(f"{prefix}stake_scale", 1.0)
                    or 1.0
                ),
            )

        atr_value = float(trade.get_custom_data("initial_atr", 0.0) or 0.0)
        plan = self._precision_stop_plan(
            float(trade.open_rate),
            atr_value,
            bool(trade.is_short),
            getattr(trade, "price_precision", None),
            getattr(trade, "precision_mode_price", None),
        )
        if plan is not None:
            self._store_precision_stop_plan(trade, plan)
        return plan

    @staticmethod
    def _absolute_stop_from_ratio(
        ratio: float,
        current_rate: float,
        trade: Trade,
    ) -> float:
        leverage = max(float(trade.leverage or 1.0), 1e-12)
        distance = abs(float(ratio)) / leverage
        if trade.is_short:
            return float(current_rate) * (1.0 + distance)
        return float(current_rate) * (1.0 - distance)

    def _parent_stop_is_tighter(
        self,
        inherited: float | None,
        current_rate: float,
        trade: Trade,
        plan: _PrecisionStopPlan,
    ) -> bool:
        if inherited is None:
            return False
        parent_stop = self._absolute_stop_from_ratio(
            float(inherited),
            current_rate,
            trade,
        )
        tolerance = max(plan.tick_size * 0.25, 1e-12)
        if trade.is_short:
            return parent_stop < plan.soft_stop - tolerance
        return parent_stop > plan.soft_stop + tolerance

    def informative_pairs(self) -> list[tuple[str, str]]:
        inherited = list(super().informative_pairs())
        if self.dp is None:
            return inherited

        runmode = str(getattr(getattr(self.dp, "runmode", None), "value", ""))
        if runmode in {"live", "dry_run"}:
            try:
                pairs = {
                    trade.pair
                    for trade in Trade.get_trades_proxy(is_open=True)
                    if self._component(getattr(trade, "enter_tag", None))
                    == "breakout"
                }
            except Exception:
                pairs = set()
        else:
            pairs = set(self.dp.current_whitelist())
        additions = [
            (pair, self.PRECISION_GUARD_TIMEFRAME)
            for pair in sorted(pairs)
        ]
        return list(dict.fromkeys(inherited + additions))

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
        if stake <= 0.0 or self._component(entry_tag) != "breakout":
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return stake
        plan = self._market_stop_plan(
            pair,
            current_rate,
            float(row.get("atr", 0.0) or 0.0),
            str(side).lower() == "short",
        )
        if plan is None:
            return stake
        scaled = min(float(max_stake), max(0.0, stake * plan.stake_scale))
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
        ):
            return
        component = self._component(trade.enter_tag)
        if component == "grid":
            grid_stop = float(
                trade.get_custom_data("grid_hard_stop", 0.0) or 0.0
            )
            if grid_stop > 0.0:
                self._arm_absolute_stop_on_filled_trade(trade, grid_stop)
            return
        if component != "breakout":
            return
        plan = self._precision_stop_plan(
            float(trade.open_rate),
            float(trade.get_custom_data("initial_atr", 0.0) or 0.0),
            bool(trade.is_short),
            getattr(trade, "price_precision", None),
            getattr(trade, "precision_mode_price", None),
        )
        if plan is None:
            return
        self._store_precision_stop_plan(trade, plan)
        self._arm_hard_stop_on_filled_trade(trade, plan)
        logger.info(
            "Breakout precision stop initialized: pair=%s side=%s "
            "soft=%.12g inward=%.12g outward=%.12g hard=%.12g "
            "tick_atr=%.4f confirm_1m=%s stake_scale=%.4f",
            pair,
            "short" if trade.is_short else "long",
            plan.soft_stop,
            plan.inward_stop,
            plan.outward_stop,
            plan.hard_stop,
            plan.tick_atr,
            plan.confirmation_enabled,
            plan.stake_scale,
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
        plan = self._trade_stop_plan(trade)
        if plan is None or self._parent_stop_is_tighter(
            inherited,
            current_rate,
            trade,
            plan,
        ):
            return inherited
        return abs(
            float(
                stoploss_from_absolute(
                    plan.hard_stop,
                    current_rate=current_rate,
                    is_short=trade.is_short,
                    leverage=trade.leverage,
                )
            )
        )

    def _latest_completed_precision_candle(
        self,
        pair: str,
        current_time: datetime,
    ) -> pd.Series | None:
        if self.dp is None:
            return None
        runmode = str(getattr(getattr(self.dp, "runmode", None), "value", ""))
        cache_entry: tuple[pd.DataFrame, pd.DatetimeIndex] | None = None
        if runmode not in {"live", "dry_run"} and hasattr(
            self.dp,
            "historic_ohlcv",
        ):
            # `historic_ohlcv()` intentionally returns the whole backtest
            # frame.  Cache that immutable frame once, then causally locate
            # the latest completed minute with a binary search.  Repeatedly
            # calling get_pair_dataframe() here would rescan/copy a full year
            # for every simulated minute and make detail backtests unusable.
            cache = getattr(self, "_precision_guard_history_cache", None)
            if cache is None:
                cache = {}
                self._precision_guard_history_cache = cache
            cache_entry = cache.get(pair)
            if cache_entry is None:
                try:
                    history = self.dp.historic_ohlcv(
                        pair,
                        timeframe=self.PRECISION_GUARD_TIMEFRAME,
                    )
                except Exception:
                    history = None
                if (
                    history is not None
                    and not history.empty
                    and "date" in history
                ):
                    history = history.sort_values("date").reset_index(drop=True)
                    history_dates = pd.DatetimeIndex(
                        pd.to_datetime(history["date"], utc=True)
                    ).as_unit("ns")
                    cache_entry = (history, history_dates)
                    cache[pair] = cache_entry

        if cache_entry is None:
            try:
                frame = self.dp.get_pair_dataframe(
                    pair,
                    timeframe=self.PRECISION_GUARD_TIMEFRAME,
                )
            except Exception:
                return None
            if frame is None or frame.empty or "date" not in frame.columns:
                return None
            # Live OHLCV dates normally use millisecond resolution while the
            # engine clock carries microseconds.  Pandas 3 refuses a lossy
            # searchsorted comparison between those units, so losslessly
            # promote candle timestamps before locating the completed bar.
            dates = pd.DatetimeIndex(
                pd.to_datetime(frame["date"], utc=True)
            ).as_unit("ns")
        else:
            frame, dates = cache_entry
        if frame is None or frame.empty or "date" not in frame.columns:
            return None

        now = pd.Timestamp(current_time)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
        duration = pd.Timedelta(self.PRECISION_GUARD_TIMEFRAME)
        latest_open = now - duration
        position = int(dates.searchsorted(latest_open, side="right")) - 1
        if position < 0:
            return None
        row = frame.iloc[position]
        row_date = dates[position]
        if row_date.tzinfo is None:
            row_date = row_date.tz_localize("UTC")
        else:
            row_date = row_date.tz_convert("UTC")
        age_seconds = (
            now
            - (row_date + duration)
        ).total_seconds()
        if (
            age_seconds < 0.0
            or age_seconds
            > float(self.PRECISION_GUARD_MAX_CANDLE_AGE_SECONDS)
        ):
            return None
        return row

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
        if inherited:
            return inherited
        if self._component(trade.enter_tag) != "breakout":
            return inherited
        plan = self._trade_stop_plan(trade)
        if plan is None or not plan.confirmation_enabled:
            return inherited

        current_stop = float(getattr(trade, "stop_loss", 0.0) or 0.0)
        tolerance = max(plan.tick_size * 0.25, 1e-12)
        if current_stop > 0.0:
            if trade.is_short and current_stop < plan.soft_stop - tolerance:
                return inherited
            if not trade.is_short and current_stop > plan.soft_stop + tolerance:
                return inherited

        candle = self._latest_completed_precision_candle(pair, current_time)
        if candle is None:
            return inherited
        high = float(candle.get("high", math.nan))
        low = float(candle.get("low", math.nan))
        close = float(candle.get("close", math.nan))
        if not all(math.isfinite(value) for value in (high, low, close)):
            return inherited

        if trade.is_short:
            confirmed = high >= plan.soft_stop and close >= plan.soft_stop
        else:
            confirmed = low <= plan.soft_stop and close <= plan.soft_stop
        if confirmed:
            return self.PRECISION_GUARD_EXIT_REASON
        return inherited


class BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade(
    _PrecisionConfirmedInitialStopMixin,
    BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade,
):
    """Selected V16 + Grid V15 path with bounded tick-safe initial stops."""

    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_precision_guard_live_parity_20260816"
    )


class BreakoutV16GridV15PrecisionGuardB08ResearchFreqtrade(
    BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade
):
    """Research neighbor: smaller emergency buffer."""

    PRECISION_GUARD_HARD_BUFFER_ATR = 0.08


class BreakoutV16GridV15PrecisionGuardB12ResearchFreqtrade(
    BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade
):
    """Research neighbor: larger emergency buffer."""

    PRECISION_GUARD_HARD_BUFFER_ATR = 0.12


class BreakoutV16GridV15PrecisionGuardSelectiveResearchFreqtrade(
    BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade
):
    """Research neighbor: enable confirmation only above 0.08 tick/ATR."""

    PRECISION_GUARD_MIN_TICK_ATR = 0.08


class BreakoutV16GridV15PrecisionGuardGlobalResearchFreqtrade(
    BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade
):
    """Research neighbor: apply close confirmation to every market."""

    PRECISION_GUARD_MIN_TICK_ATR = 0.0


class BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade(
    BreakoutV16GridV15PrecisionGuardGlobalResearchFreqtrade
):
    """Production selection: global one-minute confirmed initial stops."""

    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_precision_guard_global_live_parity_20260816"
    )

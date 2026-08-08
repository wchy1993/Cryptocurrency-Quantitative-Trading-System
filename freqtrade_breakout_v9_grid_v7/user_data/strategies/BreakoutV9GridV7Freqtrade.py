from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import (
    IStrategy,
    Order,
    Trade,
    stoploss_from_absolute,
)


logger = logging.getLogger(__name__)


def _ema(values: Series, period: int) -> Series:
    """Match crypto_scalper.indicators.ema (seeded from the first value)."""

    return values.astype(float).ewm(span=period, adjust=False).mean()


def _atr(dataframe: DataFrame, period: int) -> Series:
    """Match crypto_scalper.indicators.atr (Wilder alpha, adjust=False)."""

    previous_close = dataframe["close"].shift(1).fillna(dataframe["close"])
    true_range = pd.concat(
        (
            dataframe["high"] - dataframe["low"],
            (dataframe["high"] - previous_close).abs(),
            (dataframe["low"] - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


def _efficiency_ratio(values: Series, lookback: int) -> Series:
    displacement = values - values.shift(lookback)
    path = values.diff().abs().rolling(lookback, min_periods=lookback).sum()
    return displacement / path.replace(0.0, np.nan)


def _causal_signal_cap(
    signal: Series,
    dates: Series,
    daily_limit: int,
    minimum_bar_gap: int = 0,
) -> Series:
    """Apply the frozen per-symbol cap without looking past the current row."""

    accepted = np.zeros(len(signal), dtype=np.int8)
    counts: dict[Any, int] = {}
    last_index = -10**12
    for position, enabled in enumerate(signal.fillna(False).astype(bool).to_numpy()):
        if not enabled:
            continue
        day = pd.Timestamp(dates.iloc[position]).date()
        if position - last_index < minimum_bar_gap:
            continue
        if counts.get(day, 0) >= daily_limit:
            continue
        accepted[position] = 1
        counts[day] = counts.get(day, 0) + 1
        last_index = position
    return Series(accepted, index=signal.index, dtype=np.int8)


def _clip_ratio(values: Series, scale: float) -> Series:
    return (values / scale).clip(lower=-2.0, upper=2.0)


class BreakoutV9GridV7Freqtrade(IStrategy):
    """Freqtrade execution adapter for frozen Breakout v9 + Grid v7.

    Signal calculations intentionally use only completed 1h candles. Freqtrade
    handles exchange connectivity, futures funding, fees, order lifecycle,
    position adjustments and stop execution. Backtests use 1m detail candles.
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1h"
    process_only_new_candles = True
    startup_candle_count = 600
    ignore_buying_expired_candle_after = 30

    minimal_roi = {"0": 1000.0}
    stoploss = -0.99
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    trailing_stop = False

    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Shared frozen portfolio.
    LEVERAGE = 10.0
    MAX_OPEN_TRADES = 2
    MAX_GROSS_NOTIONAL = 9.0
    HARD_DRAWDOWN = 0.60

    # Conservative fee + market-slippage approximation per side:
    # 5 bps taker fee + 3 bps modeled market slippage.
    SIDE_COST = 0.0008

    # Breakout v9 frozen signal and management.
    BO_LOOKBACK_DAYS = 5
    BO_LONG_K = 0.70
    BO_SHORT_K = 0.60
    BO_STOP_ATR = 0.77
    BO_MAX_HOLD_MINUTES = 1200
    BO_TAKE_PROFIT_R = 60.0
    BO_MAX_NOTIONAL = 9.0
    BO_MAX_RISK = 0.10

    # Grid v7 frozen signal and campaign management.
    GRID_SPACING_ATR = 0.60
    GRID_LEVELS = 2
    GRID_TARGET_SPACING = 1.85
    GRID_HARD_STOP_ATR = 3.0
    GRID_SLOW_BUFFER_ATR = 0.35
    GRID_MAX_ENTRIES = 3
    GRID_MAX_CYCLES_PER_LEVEL = 2
    GRID_MAX_HOLD_MINUTES = 4320
    GRID_RISK_PCT = 0.1095
    GRID_MAX_NOTIONAL = 5.0
    GRID_LOSS_LIMIT_R = 0.70
    GRID_CYCLE_FLOOR_TPS = 2
    GRID_CYCLE_FLOOR_ACTIVATION_R = 0.20
    GRID_CYCLE_FLOOR_R = 0.03

    # Freqtrade can report the same completed exchange order again when its
    # fee details are refreshed.  Keep a small restart-safe history on the
    # trade so stateful Grid counters are advanced once per unique order,
    # rather than once per callback invocation.
    FILL_STATE_ORDER_IDS_KEY = "fill_state_order_ids"
    FILL_STATE_ORDER_IDS_LIMIT = 64

    plot_config = {
        "main_plot": {
            "upper_band": {"color": "#3B82F6"},
            "lower_band": {"color": "#EF4444"},
            "fast_ema": {"color": "#F59E0B"},
            "slow_ema": {"color": "#8B5CF6"},
        },
        "subplots": {
            "Market": {
                "breadth": {"color": "#10B981"},
                "market_efficiency": {"color": "#6B7280"},
            }
        },
    }

    _market_context_cache: DataFrame | None = None
    _market_context_end: pd.Timestamp | None = None
    _candidate_ranks: dict[
        str, dict[tuple[str, int], float]
    ] = {}
    _peak_equity: float = 0.0

    @staticmethod
    def _symbol(pair: str) -> str:
        base = pair.split("/", 1)[0]
        return f"{base}USDT"

    @staticmethod
    def _component(tag: str | None) -> str:
        value = str(tag or "")
        if value.startswith("bo_v9"):
            return "breakout"
        if value.startswith("grid_v7"):
            return "grid"
        return ""

    @staticmethod
    def _trade_open_time(trade: Trade) -> datetime:
        value = getattr(trade, "open_date_utc", None) or trade.open_date
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    @staticmethod
    def _trade_close_time(trade: Trade) -> datetime | None:
        value = getattr(trade, "close_date_utc", None) or getattr(
            trade, "close_date", None
        )
        if value is None:
            return None
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    def bot_start(self, **kwargs: Any) -> None:
        self._market_context_cache = None
        self._market_context_end = None
        self._candidate_ranks = {}
        self._peak_equity = 0.0

    def _build_market_context(self) -> DataFrame:
        """Build causal, cross-sectional 1h context for the static 50-pair set."""

        pairs = tuple(self.dp.current_whitelist())
        above_ema: dict[str, Series] = {}
        absolute_efficiency: dict[str, Series] = {}
        returns: dict[str, Series] = {}
        for pair in pairs:
            raw = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            if raw is None or raw.empty:
                continue
            local = raw[["date", "close"]].copy()
            local["date"] = pd.to_datetime(local["date"], utc=True)
            local = local.drop_duplicates("date").set_index("date").sort_index()
            close = local["close"].astype(float)
            symbol = self._symbol(pair)
            above_ema[symbol] = (close > _ema(close, 21)).astype(float)
            absolute_efficiency[symbol] = _efficiency_ratio(
                close, 12
            ).abs()
            returns[symbol] = close / close.shift(4) - 1.0

        if not above_ema:
            return DataFrame(
                columns=[
                    "date",
                    "breadth",
                    "breadth_change_4h",
                    "market_efficiency",
                    "btc_return_4h",
                    "eth_return_4h",
                ]
            )

        above_frame = pd.concat(above_ema, axis=1).sort_index()
        efficiency_frame = pd.concat(
            absolute_efficiency, axis=1
        ).reindex(above_frame.index)
        minimum_count = max(5, len(pairs) // 2)
        breadth = above_frame.mean(axis=1).where(
            above_frame.count(axis=1) >= minimum_count
        )
        market_efficiency = efficiency_frame.mean(axis=1).where(
            efficiency_frame.count(axis=1) >= minimum_count
        )
        btc_return = returns.get("BTCUSDT", Series(dtype=float)).reindex(
            above_frame.index
        )
        eth_return = returns.get("ETHUSDT", Series(dtype=float)).reindex(
            above_frame.index
        )
        return DataFrame(
            {
                "date": above_frame.index,
                "breadth": breadth,
                "breadth_change_4h": breadth - breadth.shift(4),
                "market_efficiency": market_efficiency,
                "btc_return_4h": btc_return,
                "eth_return_4h": eth_return,
            }
        ).reset_index(drop=True)

    def _attach_market_context(self, dataframe: DataFrame) -> DataFrame:
        last_date = pd.Timestamp(dataframe["date"].iloc[-1])
        if last_date.tzinfo is None:
            last_date = last_date.tz_localize("UTC")
        if (
            self._market_context_cache is None
            or self._market_context_end is None
            or last_date != self._market_context_end
        ):
            self._market_context_cache = self._build_market_context()
            self._market_context_end = last_date
        context = self._market_context_cache
        if context is None or context.empty:
            for column in (
                "breadth",
                "breadth_change_4h",
                "market_efficiency",
                "btc_return_4h",
                "eth_return_4h",
            ):
                dataframe[column] = np.nan
            return dataframe
        return dataframe.merge(context, on="date", how="left", validate="m:1")

    @staticmethod
    def _daily_dual_range(dataframe: DataFrame, lookback: int) -> Series:
        day = dataframe["date"].dt.floor("D")
        daily = (
            dataframe.assign(_day=day)
            .groupby("_day", sort=True)
            .agg(
                day_high=("high", "max"),
                day_low=("low", "min"),
                day_close=("close", "last"),
            )
        )
        prior_high = (
            daily["day_high"].shift(1).rolling(lookback, min_periods=lookback).max()
        )
        prior_high_close = (
            daily["day_close"]
            .shift(1)
            .rolling(lookback, min_periods=lookback)
            .max()
        )
        prior_low_close = (
            daily["day_close"]
            .shift(1)
            .rolling(lookback, min_periods=lookback)
            .min()
        )
        prior_low = (
            daily["day_low"].shift(1).rolling(lookback, min_periods=lookback).min()
        )
        dual_range = pd.concat(
            (
                prior_high - prior_low_close,
                prior_high_close - prior_low,
            ),
            axis=1,
        ).max(axis=1)
        return day.map(dual_range)

    @staticmethod
    def _regime_score(dataframe: DataFrame, side: float) -> Series:
        directional_breadth = (
            dataframe["breadth"]
            if side > 0
            else 1.0 - dataframe["breadth"]
        )
        return (
            0.18 * _clip_ratio(side * dataframe["btc_return_4h"], 0.01)
            + 0.14 * _clip_ratio(side * dataframe["eth_return_4h"], 0.012)
            + 0.18 * _clip_ratio(directional_breadth - 0.5, 0.20)
            + 0.12
            * _clip_ratio(side * dataframe["breadth_change_4h"], 0.10)
            + 0.18 * _clip_ratio(side * dataframe["symbol_return_4h"], 0.02)
            + 0.12
            * _clip_ratio(side * dataframe["symbol_efficiency_12h"], 0.25)
            + 0.08
            * _clip_ratio(side * dataframe["symbol_ema55_atr"], 2.0)
        )

    def _populate_breakout(self, dataframe: DataFrame) -> DataFrame:
        dataframe["session_open"] = dataframe.groupby(
            dataframe["date"].dt.floor("D")
        )["open"].transform("first")
        dataframe["dual_range"] = self._daily_dual_range(
            dataframe, self.BO_LOOKBACK_DAYS
        )
        dataframe["upper_band"] = (
            dataframe["session_open"] + self.BO_LONG_K * dataframe["dual_range"]
        )
        dataframe["lower_band"] = (
            dataframe["session_open"] - self.BO_SHORT_K * dataframe["dual_range"]
        )

        long_cross = (
            (dataframe["close"].shift(1) <= dataframe["upper_band"])
            & (dataframe["close"] > dataframe["upper_band"])
        )
        short_cross = (
            (dataframe["close"].shift(1) >= dataframe["lower_band"])
            & (dataframe["close"] < dataframe["lower_band"])
        )
        candle_range = (dataframe["high"] - dataframe["low"]).clip(
            lower=1e-12
        )
        average_volume = (
            dataframe["volume"]
            .shift(1)
            .rolling(20, min_periods=1)
            .mean()
            .clip(lower=1e-12)
        )
        dataframe["bo_volume_ratio"] = dataframe["volume"] / average_volume
        dataframe["bo_body_atr"] = (
            dataframe["close"] - dataframe["open"]
        ).abs() / dataframe["atr"].clip(lower=1e-12)
        close_position = (
            dataframe["close"] - dataframe["low"]
        ) / candle_range
        dataframe["bo_long_close_position"] = close_position
        dataframe["bo_short_close_position"] = 1.0 - close_position
        dataframe["bo_long_alignment"] = (
            dataframe["close"] - dataframe["ema48"]
        ) / dataframe["atr"].clip(lower=1e-12)
        dataframe["bo_short_alignment"] = -dataframe["bo_long_alignment"]
        dataframe["bo_range_atr"] = dataframe["dual_range"] / dataframe[
            "atr"
        ].clip(lower=1e-12)
        dataframe["bo_long_extension"] = (
            dataframe["close"] - dataframe["upper_band"]
        ) / dataframe["atr"].clip(lower=1e-12)
        dataframe["bo_short_extension"] = (
            dataframe["lower_band"] - dataframe["close"]
        ) / dataframe["atr"].clip(lower=1e-12)

        def quality(
            volume: Series,
            body: Series,
            close_pos: Series,
            alignment: Series,
            extension: Series,
        ) -> Series:
            return (
                0.30 * volume.clip(lower=0.0, upper=3.0)
                + 0.25 * body.clip(lower=0.0, upper=2.0)
                + 0.20 * close_pos.clip(lower=0.0, upper=1.0)
                + 0.15 * (alignment.clip(lower=-1.0) + 1.0).clip(
                    lower=0.0, upper=2.0
                )
                + 0.10 * extension.clip(lower=0.0, upper=1.5)
            )

        dataframe["bo_long_quality"] = quality(
            dataframe["bo_volume_ratio"],
            dataframe["bo_body_atr"],
            dataframe["bo_long_close_position"],
            dataframe["bo_long_alignment"],
            dataframe["bo_long_extension"],
        )
        dataframe["bo_short_quality"] = quality(
            dataframe["bo_volume_ratio"],
            dataframe["bo_body_atr"],
            dataframe["bo_short_close_position"],
            dataframe["bo_short_alignment"],
            dataframe["bo_short_extension"],
        )

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
        long_breadth = dataframe["breadth"]
        short_breadth = 1.0 - dataframe["breadth"]
        dataframe["bo_regime_long"] = self._regime_score(dataframe, 1.0)
        dataframe["bo_regime_short"] = self._regime_score(dataframe, -1.0)

        long_gate = (
            long_cross
            & context_ready
            & (dataframe["bo_long_quality"] >= 1.8)
            & (dataframe["bo_body_atr"] >= 1.7)
            & (dataframe["bo_long_extension"] >= 0.7)
            & (dataframe["bo_range_atr"].between(0.0, 5.5))
            & (long_breadth.between(0.0, 1.0))
            & (dataframe["btc_return_4h"] <= 0.02)
            & (dataframe["eth_return_4h"] <= 0.015)
        )
        short_gate = (
            short_cross
            & context_ready
            & (dataframe["bo_short_quality"] >= 1.0)
            & (dataframe["bo_short_extension"] >= 0.25)
            & (dataframe["bo_range_atr"].between(0.0, 7.5))
            & (short_breadth.between(0.7, 1.0))
            & (-dataframe["btc_return_4h"] <= 0.02)
        )
        combined_gate = long_gate | short_gate
        capped = _causal_signal_cap(
            combined_gate, dataframe["date"], daily_limit=2
        ).astype(bool)
        long_gate &= capped
        short_gate &= capped

        long_score = (
            (dataframe["bo_long_quality"] <= 2.4).astype(int)
            + (dataframe["bo_body_atr"] >= 2.8).astype(int)
            + (dataframe["bo_volume_ratio"] >= 2.0).astype(int)
            + (dataframe["bo_long_extension"] <= 2.6).astype(int)
            + (long_breadth <= 0.84).astype(int)
        )
        short_score = (
            (dataframe["bo_short_quality"] >= 1.05).astype(int)
            + (dataframe["bo_body_atr"] >= 0.5).astype(int)
            + (dataframe["bo_volume_ratio"] >= 2.0).astype(int)
            + (dataframe["bo_short_extension"] <= 0.32).astype(int)
            + (short_breadth <= 0.86).astype(int)
        )
        dataframe["bo_score"] = np.where(long_gate, long_score, short_score)
        dataframe["bo_regime"] = np.where(
            long_gate,
            dataframe["bo_regime_long"],
            dataframe["bo_regime_short"],
        )
        directional_ema55 = np.where(
            long_gate,
            dataframe["symbol_ema55_atr"],
            -dataframe["symbol_ema55_atr"],
        )

        base_risk = np.where(
            (dataframe["bo_regime"] >= 0.25) & (directional_ema55 >= 0.5),
            0.03366,
            np.where(
                dataframe["bo_regime"] < 0.15,
                0.015147 * 0.35,
                0.015147,
            ),
        )
        score = dataframe["bo_score"].astype(int)
        side_base = np.where(long_gate, 1.05, 0.90)
        confidence_factor = np.where(
            score >= 4,
            1.30,
            np.where(score <= 1, 0.45, 1.0),
        )
        short_score_factor = score.map(
            {0: 1.0, 1: 0.5, 2: 1.0, 3: 0.85, 4: 1.1, 5: 2.6}
        ).fillna(1.0)
        allocation = np.where(long_gate, 1.0, short_score_factor)
        v7_multiplier = np.minimum(side_base * confidence_factor, 2.0)
        adjusted_multiplier = np.minimum(v7_multiplier * allocation, 3.0)
        final_side_risk = np.where(long_gate, 0.90, 1.0)
        dataframe["bo_risk_pct"] = np.minimum(
            base_risk * adjusted_multiplier * final_side_risk,
            self.BO_MAX_RISK,
        )
        dataframe["bo_capture"] = (
            short_gate & score.between(3, 4)
        ).astype(int)
        dataframe["bo_long_floor"] = (
            long_gate & (score == 4)
        ).astype(int)
        dataframe["bo_entry_long"] = long_gate.astype(int)
        dataframe["bo_entry_short"] = short_gate.astype(int)
        dataframe["bo_entry"] = (
            dataframe["bo_entry_long"] | dataframe["bo_entry_short"]
        ).astype(int)
        return dataframe

    def _populate_grid(self, dataframe: DataFrame) -> DataFrame:
        atr = dataframe["atr"].clip(lower=1e-12)
        dataframe["grid_fast_slope"] = (
            dataframe["fast_ema"] - dataframe["fast_ema"].shift(3)
        ) / atr
        dataframe["grid_slow_slope"] = (
            dataframe["slow_ema"] - dataframe["slow_ema"].shift(3)
        ) / atr
        dataframe["grid_alignment"] = (
            dataframe["slow_ema"] - dataframe["fast_ema"]
        ) / atr
        dataframe["grid_extension"] = (
            dataframe["fast_ema"] - dataframe["close"]
        ) / atr
        candle_range = (dataframe["high"] - dataframe["low"]).clip(
            lower=1e-12
        )
        dataframe["grid_close_position"] = (
            dataframe["high"] - dataframe["close"]
        ) / candle_range
        average_volume = (
            dataframe["volume"]
            .shift(1)
            .rolling(20, min_periods=1)
            .mean()
            .clip(lower=1e-12)
        )
        dataframe["grid_volume_ratio"] = dataframe["volume"] / average_volume

        short_trend = (
            (dataframe["fast_ema"] < dataframe["slow_ema"])
            & (dataframe["grid_fast_slope"] <= 0.0)
            & (dataframe["grid_slow_slope"] <= -0.05)
            & dataframe["grid_alignment"].between(0.3, 4.0)
            & (dataframe["atr"] / dataframe["close"]).between(0.001, 0.08)
        )
        pullback_touch = (
            (dataframe["high"] >= dataframe["fast_ema"] - 0.30 * dataframe["atr"])
            & (dataframe["close"] < dataframe["fast_ema"])
        )
        raw_signal = (
            short_trend
            & dataframe["grid_extension"].between(0.0, 0.65)
            & (dataframe["grid_close_position"] >= 0.58)
            & dataframe["grid_volume_ratio"].between(0.7, 2.0)
            & pullback_touch
        )
        raw_signal = _causal_signal_cap(
            raw_signal,
            dataframe["date"],
            daily_limit=2,
            minimum_bar_gap=12,
        ).astype(bool)

        directional_breadth = 1.0 - dataframe["breadth"]
        dataframe["grid_regime"] = self._regime_score(dataframe, -1.0)
        overlay = (
            (-dataframe["btc_return_4h"]).between(-0.01, 0.01)
            & (-dataframe["eth_return_4h"] <= 0.015)
            & directional_breadth.between(0.25, 0.70)
            & (-dataframe["symbol_return_4h"]).between(-0.015, 0.08)
            & (-dataframe["symbol_efficiency_12h"] >= -0.05)
            & (dataframe["market_efficiency"] >= 0.16)
            & (-dataframe["symbol_ema55_atr"]).between(-0.5, 2.0)
            & (dataframe["grid_regime"] <= 1.0)
        )
        entry_gate = (
            dataframe[
                [
                    "breadth",
                    "market_efficiency",
                    "btc_return_4h",
                    "eth_return_4h",
                    "symbol_efficiency_12h",
                    "symbol_ema55_atr",
                ]
            ]
            .notna()
            .all(axis=1)
            & (dataframe["grid_regime"] <= 0.5)
        )
        dataframe["grid_quality"] = (
            0.30
            * (-dataframe["grid_fast_slope"] + 0.25).clip(
                lower=0.0, upper=2.0
            )
            + 0.20
            * (-dataframe["grid_slow_slope"] + 0.25).clip(
                lower=0.0, upper=2.0
            )
            + 0.20
            * dataframe["grid_alignment"].clip(lower=0.0, upper=2.0)
            + 0.15
            * (
                1.0
                - dataframe["grid_extension"]
                / max(0.65, 1e-12)
            ).clip(lower=0.0, upper=1.0)
            + 0.10
            * dataframe["grid_volume_ratio"].clip(lower=0.0, upper=1.5)
            + 0.05 * dataframe["grid_close_position"]
        )
        grid_score = (
            (dataframe["grid_quality"] >= 0.45).astype(int)
            + (dataframe["grid_alignment"] >= 0.52).astype(int)
            + (dataframe["grid_extension"] >= 0.20).astype(int)
            + (dataframe["grid_extension"] <= 0.45).astype(int)
            + (dataframe["grid_volume_ratio"] <= 1.10).astype(int)
            + (dataframe["grid_regime"] <= 0.30).astype(int)
        )
        dataframe["grid_score"] = grid_score
        eligible = (
            raw_signal
            & overlay
            & entry_gate
            & (grid_score > 2)
            & (dataframe["grid_extension"] >= 0.05)
        )
        dataframe["grid_entry_short"] = eligible.astype(int)
        dataframe["grid_entry"] = dataframe["grid_entry_short"]
        dataframe["grid_hard_stop"] = np.maximum(
            dataframe["close"] + self.GRID_HARD_STOP_ATR * dataframe["atr"],
            dataframe["slow_ema"]
            + self.GRID_SLOW_BUFFER_ATR * dataframe["atr"],
        )
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True)
        dataframe["atr"] = _atr(dataframe, 14)
        dataframe["ema48"] = _ema(dataframe["close"], 48)
        dataframe["fast_ema"] = _ema(dataframe["close"], 21)
        dataframe["slow_ema"] = _ema(dataframe["close"], 55)
        dataframe["symbol_return_4h"] = (
            dataframe["close"] / dataframe["close"].shift(4) - 1.0
        )
        dataframe["symbol_efficiency_12h"] = _efficiency_ratio(
            dataframe["close"], 12
        )
        dataframe["symbol_ema55_atr"] = (
            dataframe["close"] - dataframe["slow_ema"]
        ) / dataframe["atr"].clip(lower=1e-12)
        dataframe = self._attach_market_context(dataframe)
        dataframe = self._populate_breakout(dataframe)
        dataframe = self._populate_grid(dataframe)
        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        breakout_long = dataframe["bo_entry_long"].astype(bool)
        breakout_short = dataframe["bo_entry_short"].astype(bool)
        grid_short = dataframe["grid_entry_short"].astype(bool) & ~(
            breakout_long | breakout_short
        )

        dataframe.loc[breakout_long, "enter_long"] = 1
        dataframe.loc[breakout_short | grid_short, "enter_short"] = 1
        for index in dataframe.index[breakout_long | breakout_short]:
            score = int(dataframe.at[index, "bo_score"])
            risk_units = int(
                round(float(dataframe.at[index, "bo_risk_pct"]) * 1_000_000)
            )
            capture = int(dataframe.at[index, "bo_capture"])
            long_floor = int(dataframe.at[index, "bo_long_floor"])
            dataframe.at[index, "enter_tag"] = (
                f"bo_v9_s{score}_r{risk_units}_c{capture}_l{long_floor}"
            )
        for index in dataframe.index[grid_short]:
            score = int(dataframe.at[index, "grid_score"])
            dataframe.at[index, "enter_tag"] = f"grid_v7_s{score}"

        pair = str(metadata.get("pair") or "")
        if pair:
            ranks: dict[tuple[str, int], float] = {}
            available_times = (
                pd.to_datetime(dataframe["date"], utc=True)
                + pd.Timedelta(self.timeframe)
            )
            for index in dataframe.index[breakout_long | breakout_short]:
                ranks[
                    (
                        "breakout",
                        int(available_times.at[index].timestamp()),
                    )
                ] = float(dataframe.at[index, "bo_range_atr"])
            for index in dataframe.index[grid_short]:
                ranks[
                    (
                        "grid",
                        int(available_times.at[index].timestamp()),
                    )
                ] = float(dataframe.at[index, "grid_extension"])
            self._candidate_ranks[pair] = ranks
        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        return min(self.LEVERAGE, max_leverage)

    def _latest_row(
        self, pair: str, current_time: datetime
    ) -> Series | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        boundary = pd.Timestamp(current_time)
        if boundary.tzinfo is None:
            boundary = boundary.tz_localize("UTC")
        else:
            boundary = boundary.tz_convert("UTC")
        boundary = boundary.floor(self.timeframe)
        dates = pd.to_datetime(dataframe["date"], utc=True)
        completed = dataframe.loc[dates < boundary]
        return completed.iloc[-1] if not completed.empty else None

    def _latest_signal_row(
        self,
        pair: str,
        component: str,
        current_time: datetime,
    ) -> Series | None:
        row = self._latest_row(pair, current_time)
        if row is None:
            return None
        column = "bo_entry" if component == "breakout" else "grid_entry"
        return row if int(row.get(column, 0) or 0) == 1 else None

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
        component = self._component(entry_tag)
        row = self._latest_signal_row(pair, component, current_time)
        if not component or row is None or current_rate <= 0.0:
            return 0.0

        equity = float(self.wallets.get_total_stake_amount())
        self._peak_equity = max(self._peak_equity, equity)
        if (
            equity <= 0.0
            or (
                self._peak_equity > 0.0
                and equity
                <= self._peak_equity * (1.0 - self.HARD_DRAWDOWN)
            )
        ):
            return 0.0

        leverage = max(1.0, leverage)
        atr_value = max(float(row.get("atr", 0.0)), 1e-12)
        if component == "breakout":
            risk_pct = max(
                0.0, min(self.BO_MAX_RISK, float(row.get("bo_risk_pct", 0.0)))
            )
            unit_loss_ratio = (
                self.BO_STOP_ATR * atr_value / current_rate
                + 2.0 * self.SIDE_COST
            )
            notional = equity * risk_pct / max(unit_loss_ratio, 1e-12)
            notional = min(notional, equity * self.BO_MAX_NOTIONAL)
            return min(max_stake, max(0.0, notional / leverage))

        hard_stop = max(
            current_rate + self.GRID_HARD_STOP_ATR * atr_value,
            float(row.get("slow_ema", current_rate))
            + self.GRID_SLOW_BUFFER_ATR * atr_value,
        )
        levels = [
            current_rate + level * self.GRID_SPACING_ATR * atr_value
            for level in range(self.GRID_LEVELS + 1)
        ]
        unit_losses = [
            max(0.0, hard_stop - level_price)
            + 2.0 * self.SIDE_COST * level_price
            for level_price in levels
        ]
        base_quantity = (
            equity * self.GRID_RISK_PCT / max(sum(unit_losses), 1e-12)
        )
        total_notional = base_quantity * sum(levels)
        if total_notional > equity * self.GRID_MAX_NOTIONAL:
            base_quantity *= (
                equity * self.GRID_MAX_NOTIONAL / total_notional
            )
        initial_notional = base_quantity * levels[0]
        return min(max_stake, max(0.0, initial_notional / leverage))

    def _component_limits_allow(
        self,
        pair: str,
        component: str,
        current_time: datetime,
    ) -> bool:
        trades = Trade.get_trades_proxy()
        open_component = [
            trade
            for trade in trades
            if trade.is_open
            and self._component(getattr(trade, "enter_tag", None)) == component
        ]
        if open_component:
            return False

        cooldown = 120 if component == "breakout" else 480
        for trade in trades:
            if trade.is_open or trade.pair != pair:
                continue
            if self._component(getattr(trade, "enter_tag", None)) != component:
                continue
            close_time = self._trade_close_time(trade)
            if (
                close_time is not None
                and current_time < close_time + timedelta(minutes=cooldown)
            ):
                return False

        day = current_time.astimezone(timezone.utc).date()
        daily_limit = 5 if component == "breakout" else 12
        entries_today = sum(
            1
            for trade in trades
            if self._component(getattr(trade, "enter_tag", None)) == component
            and self._trade_open_time(trade).date() == day
        )
        return entries_today < daily_limit

    def _ranked_candidates(
        self,
        component: str,
        current_time: datetime,
    ) -> list[tuple[float, str]]:
        rows: list[tuple[float, str]] = []
        timestamp = pd.Timestamp(current_time)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        available_second = int(timestamp.floor(self.timeframe).timestamp())
        for pair in self.dp.current_whitelist():
            if not self._component_limits_allow(pair, component, current_time):
                continue
            rank = self._candidate_ranks.get(pair, {}).get(
                (component, available_second), np.inf
            )
            if np.isfinite(rank):
                rows.append((rank, pair))
        return sorted(rows, key=lambda value: (value[0], value[1]))

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> bool:
        component = self._component(entry_tag)
        if not component or not self._component_limits_allow(
            pair, component, current_time
        ):
            return False

        open_trades = Trade.get_trades_proxy(is_open=True)
        remaining_slots = self.MAX_OPEN_TRADES - len(open_trades)
        if remaining_slots <= 0:
            return False
        if component == "grid" and remaining_slots == 1:
            if self._ranked_candidates("breakout", current_time):
                return False

        ranked = self._ranked_candidates(component, current_time)
        return bool(ranked) and ranked[0][1] == pair

    @staticmethod
    def _safe_order_price(order: Order, fallback: float) -> float:
        value = getattr(order, "safe_price", None)
        return float(value if value is not None else fallback)

    @staticmethod
    def _safe_order_stake(order: Order, fallback: float) -> float:
        for name in ("stake_amount_filled", "stake_amount"):
            value = getattr(order, name, None)
            if value is not None and float(value) > 0.0:
                return float(value)
        return float(fallback)

    @staticmethod
    def _filled_order_identity(order: Order) -> str | None:
        """Return a stable identity for one completed order.

        Live orders always expose the exchange ``order_id``.  Freqtrade's
        local database id is a safe fallback for backtests and restored
        ledgers.  If neither id exists, do not guess from mutable fee/fill
        fields because that could merge two legitimate orders.
        """

        exchange_id = str(getattr(order, "order_id", "") or "").strip()
        if exchange_id:
            return f"exchange:{exchange_id}"
        database_id = getattr(order, "id", None)
        if database_id is not None:
            return f"database:{database_id}"
        return None

    def _claim_fill_state_update(self, trade: Trade, order: Order) -> bool:
        """Claim an order before mutating persistent strategy state.

        Returns ``False`` for a repeated callback of the same completed
        order.  The identity list is stored as trade custom data so a process
        restart cannot re-apply an already-accounted fill.
        """

        identity = self._filled_order_identity(order)
        if identity is None:
            # Freqtrade normally supplies one of the stable ids above.  Keep
            # historical compatibility if a synthetic test/order does not.
            return True
        raw_identities = trade.get_custom_data(
            self.FILL_STATE_ORDER_IDS_KEY,
            [],
        )
        identities = [str(value) for value in (raw_identities or [])]
        if identity in identities:
            logger.info(
                "Ignoring duplicate filled-order callback: trade_id=%s "
                "pair=%s order=%s tag=%s",
                getattr(trade, "id", None),
                getattr(trade, "pair", ""),
                identity,
                getattr(order, "ft_order_tag", None),
            )
            return False
        identities.append(identity)
        trade.set_custom_data(
            self.FILL_STATE_ORDER_IDS_KEY,
            identities[-self.FILL_STATE_ORDER_IDS_LIMIT :],
        )
        return True

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        component = self._component(trade.enter_tag)
        tag = str(getattr(order, "ft_order_tag", "") or "")
        fill_state_claimed = bool(kwargs.get("_fill_state_claimed", False))
        if (
            order.ft_order_side == trade.entry_side
            and trade.nr_of_successful_entries == 1
        ):
            row = self._latest_signal_row(pair, component, current_time)
            if row is None:
                return
            if (
                not fill_state_claimed
                and not self._claim_fill_state_update(trade, order)
            ):
                return
            atr_value = max(float(row.get("atr", 0.0)), 1e-12)
            entry_rate = self._safe_order_price(order, trade.open_rate)
            trade.set_custom_data("component", component)
            trade.set_custom_data("initial_atr", atr_value)
            if component == "breakout":
                trade.set_custom_data(
                    "initial_unit_risk",
                    self.BO_STOP_ATR * atr_value
                    + 2.0 * self.SIDE_COST * entry_rate,
                )
                trade.set_custom_data(
                    "bo_score", int(row.get("bo_score", 0))
                )
                trade.set_custom_data(
                    "bo_capture", bool(int(row.get("bo_capture", 0)))
                )
                trade.set_custom_data(
                    "bo_long_floor",
                    bool(int(row.get("bo_long_floor", 0))),
                )
                trade.set_custom_data("bo_partial_done", False)
                return

            spacing = self.GRID_SPACING_ATR * atr_value
            hard_stop = max(
                entry_rate + self.GRID_HARD_STOP_ATR * atr_value,
                float(row.get("slow_ema", entry_rate))
                + self.GRID_SLOW_BUFFER_ATR * atr_value,
            )
            level_prices = [
                entry_rate + level * spacing
                for level in range(self.GRID_LEVELS + 1)
            ]
            initial_stake = self._safe_order_stake(order, trade.stake_amount)
            quantity = max(float(trade.amount), 0.0)
            risk_budget = quantity * sum(
                max(0.0, hard_stop - price)
                + 2.0 * self.SIDE_COST * price
                for price in level_prices
            )
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
                {str(level): 0 for level in range(self.GRID_LEVELS + 1)},
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

        if component == "breakout" and tag == "bo_v9_partial":
            if (
                not fill_state_claimed
                and not self._claim_fill_state_update(trade, order)
            ):
                return
            trade.set_custom_data("bo_partial_done", True)
            return
        if component != "grid":
            return

        if not (
            tag.startswith("grid_dca_") or tag.startswith("grid_tp_")
        ):
            return
        if (
            not fill_state_claimed
            and not self._claim_fill_state_update(trade, order)
        ):
            return

        lots = dict(trade.get_custom_data("grid_lots", {}) or {})
        cycles = dict(trade.get_custom_data("grid_cycles", {}) or {})
        if tag.startswith("grid_dca_"):
            level = tag.rsplit("_", 1)[-1]
            lots[level] = {
                "entry": self._safe_order_price(order, trade.open_rate),
                "stake": self._safe_order_stake(
                    order,
                    float(trade.get_custom_data("grid_base_stake", 0.0)),
                ),
            }
            trade.set_custom_data("grid_lots", lots)
            trade.set_custom_data(
                "grid_entry_count",
                int(trade.get_custom_data("grid_entry_count", 1)) + 1,
            )
        elif tag.startswith("grid_tp_"):
            level = tag.rsplit("_", 1)[-1]
            lots.pop(level, None)
            cycles[level] = int(cycles.get(level, 0)) + 1
            trade.set_custom_data("grid_lots", lots)
            trade.set_custom_data("grid_cycles", cycles)
            trade.set_custom_data(
                "grid_tp_count",
                int(trade.get_custom_data("grid_tp_count", 0)) + 1,
            )

    def _trade_r(self, trade: Trade, rate: float) -> float:
        side = -1.0 if trade.is_short else 1.0
        risk = max(
            float(trade.get_custom_data("initial_unit_risk", 0.0)),
            1e-12,
        )
        net_move = (
            side * (rate - trade.open_rate)
            - 2.0 * self.SIDE_COST * trade.open_rate
        )
        return net_move / risk

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
        component = self._component(trade.enter_tag)
        if component == "grid":
            stop_rate = float(
                trade.get_custom_data("grid_hard_stop", 0.0)
            )
            if stop_rate <= 0.0:
                return None
            value = stoploss_from_absolute(
                stop_rate,
                current_rate=current_rate,
                is_short=trade.is_short,
                leverage=trade.leverage,
            )
            return abs(float(value))

        if component != "breakout":
            return None
        atr_value = max(
            float(trade.get_custom_data("initial_atr", 0.0)), 1e-12
        )
        side = -1.0 if trade.is_short else 1.0
        stop_rate = trade.open_rate - side * self.BO_STOP_ATR * atr_value
        favorable_rate_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate
            if favorable_rate_value is None
            else favorable_rate_value
        )
        maximum_r = self._trade_r(trade, favorable_rate)
        locked_r: float | None = None
        if bool(trade.get_custom_data("bo_capture", False)) and maximum_r >= 4.0:
            locked_r = 1.0
        if (
            bool(trade.get_custom_data("bo_long_floor", False))
            and maximum_r >= 6.5
        ):
            locked_r = max(locked_r or 0.0, 0.75)
        if maximum_r >= 15.0:
            locked_r = max(locked_r or 0.0, maximum_r - 12.0)
        if locked_r is not None:
            unit_risk = max(
                float(trade.get_custom_data("initial_unit_risk", 0.0)),
                1e-12,
            )
            stop_rate = trade.open_rate + side * (
                locked_r * unit_risk
                + 2.0 * self.SIDE_COST * trade.open_rate
            )
        value = stoploss_from_absolute(
            stop_rate,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )
        return abs(float(value))

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
    ) -> float | None | tuple[float | None, str | None]:
        if trade.has_open_orders:
            return None
        component = self._component(trade.enter_tag)
        if component == "breakout":
            if (
                bool(trade.get_custom_data("bo_capture", False))
                and not bool(
                    trade.get_custom_data("bo_partial_done", False)
                )
                and self._trade_r(trade, current_exit_rate) >= 2.5
            ):
                return -float(trade.stake_amount) * 0.10, "bo_v9_partial"
            return None
        if component != "grid":
            return None

        lots = dict(trade.get_custom_data("grid_lots", {}) or {})
        cycles = dict(trade.get_custom_data("grid_cycles", {}) or {})
        spacing = float(trade.get_custom_data("grid_spacing", 0.0))
        direction = -1.0 if trade.is_short else 1.0

        # Existing lot targets are handled before any new adverse fill.
        for level_text, lot in sorted(
            lots.items(), key=lambda value: int(value[0])
        ):
            target = float(lot["entry"]) + (
                direction * spacing * self.GRID_TARGET_SPACING
            )
            hit = (
                current_exit_rate <= target
                if trade.is_short
                else current_exit_rate >= target
            )
            if not hit:
                continue
            stake = min(float(lot["stake"]), float(trade.stake_amount))
            entry_count = int(
                trade.get_custom_data("grid_entry_count", 1)
            )
            if stake >= trade.stake_amount and entry_count < self.GRID_MAX_ENTRIES:
                stake = float(trade.stake_amount) * 0.99
            if stake > 0.0:
                return -stake, f"grid_tp_{level_text}"

        entry_count = int(trade.get_custom_data("grid_entry_count", 1))
        if entry_count >= self.GRID_MAX_ENTRIES:
            return None
        prices = list(
            trade.get_custom_data("grid_level_prices", []) or []
        )
        anchor = float(trade.get_custom_data("grid_anchor", trade.open_rate))
        base_stake = float(
            trade.get_custom_data("grid_base_stake", 0.0)
        )
        for level, level_price in enumerate(prices):
            key = str(level)
            if key in lots:
                continue
            if int(cycles.get(key, 0)) >= self.GRID_MAX_CYCLES_PER_LEVEL:
                continue
            if level == 0 and int(cycles.get(key, 0)) == 0:
                continue
            hit = (
                current_entry_rate >= float(level_price)
                if trade.is_short
                else current_entry_rate <= float(level_price)
            )
            if not hit:
                continue
            stake = base_stake * float(level_price) / max(anchor, 1e-12)
            stake = min(stake, max_stake)
            if min_stake is not None and stake < min_stake:
                continue
            if stake > 0.0:
                return stake, f"grid_dca_{level}"
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
        component = self._component(trade.enter_tag)
        holding_minutes = int(
            (current_time - self._trade_open_time(trade)).total_seconds() / 60
        )
        if component == "breakout":
            if self._trade_r(trade, current_rate) >= self.BO_TAKE_PROFIT_R:
                return "bo_v9_take_profit_60r"
            if holding_minutes >= self.BO_MAX_HOLD_MINUTES:
                return "bo_v9_time_stop"
            return None
        if component != "grid":
            return None

        risk_budget = max(
            float(trade.get_custom_data("grid_risk_budget", 0.0)),
            1e-12,
        )
        profit = float(trade.calculate_profit(current_rate).total_profit)
        if profit <= -risk_budget * self.GRID_LOSS_LIMIT_R:
            return "grid_v7_campaign_loss_limit"
        if holding_minutes >= self.GRID_MAX_HOLD_MINUTES:
            return "grid_v7_campaign_time_stop"

        tp_count = int(trade.get_custom_data("grid_tp_count", 0))
        favorable_rate_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate
            if favorable_rate_value is None
            else favorable_rate_value
        )
        best_profit = float(
            trade.calculate_profit(favorable_rate).total_profit
        )
        if (
            tp_count >= self.GRID_CYCLE_FLOOR_TPS
            and best_profit
            >= risk_budget * self.GRID_CYCLE_FLOOR_ACTIVATION_R
            and profit <= risk_budget * self.GRID_CYCLE_FLOOR_R
        ):
            return "grid_v7_cycle_profit_floor"

        row = self._latest_row(pair, current_time)
        if row is not None:
            candle_date = pd.Timestamp(row["date"]).isoformat()
            prior_date = str(
                trade.get_custom_data("grid_last_regime_date", "")
            )
            if candle_date != prior_date:
                invalid = (
                    float(row["fast_ema"]) >= float(row["slow_ema"])
                    if trade.is_short
                    else float(row["fast_ema"]) <= float(row["slow_ema"])
                )
                count = int(
                    trade.get_custom_data("grid_invalid_bars", 0)
                )
                count = count + 1 if invalid else 0
                trade.set_custom_data("grid_invalid_bars", count)
                trade.set_custom_data(
                    "grid_last_regime_date", candle_date
                )
                if count >= 2:
                    return "grid_v7_ema_cross"

        lots = dict(trade.get_custom_data("grid_lots", {}) or {})
        entry_count = int(trade.get_custom_data("grid_entry_count", 1))
        if not lots and entry_count >= self.GRID_MAX_ENTRIES:
            return "grid_v7_campaign_complete"
        return None

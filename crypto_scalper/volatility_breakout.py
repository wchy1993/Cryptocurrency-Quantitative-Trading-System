from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from .indicators import atr, ema
from .models import Candle, Direction


VOLATILITY_BREAKOUT_STRATEGY_NAME = "dual_thrust_volatility_breakout"
VOLATILITY_BREAKOUT_VERSION = "dual_thrust_closed_bar_v1"


@dataclass(frozen=True)
class VolatilityBreakoutConfig:
    timeframe_minutes: int = 30
    lookback_days: int = 4
    long_k: float = 0.50
    short_k: float = 0.50
    allow_long: bool = True
    allow_short: bool = True
    atr_period: int = 14
    trend_ema_period: int = 48
    min_trend_alignment_atr: float = -999.0
    min_volume_ratio: float = 0.0
    min_body_atr: float = 0.0
    min_directional_close_position: float = 0.0
    min_range_atr: float = 0.0
    max_range_atr: float = 999.0
    min_breakout_extension_atr: float = -999.0
    max_breakout_extension_atr: float = 999.0
    max_volume_ratio: float = 999.0
    max_body_atr: float = 999.0
    max_trend_alignment_atr: float = 999.0
    max_signals_per_symbol_day: int = 2
    stop_atr_multiple: float = 1.50
    take_profit_r: float = 2.0
    breakeven_trigger_r: float = 0.0
    trailing_activation_r: float = 0.0
    trailing_atr_multiple: float = 0.0
    fail_fast_minutes: int = 0
    fail_fast_min_mfe_r: float = 0.0
    fail_fast_max_current_r: float = 0.0
    extended_holding_minutes: int = 0
    extension_min_current_r: float = 0.0
    extension_min_mfe_r: float = 0.0
    profit_giveback_activation_r: float = 0.0
    profit_giveback_r: float = 0.0
    max_holding_minutes: int = 720

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DualThrustSignal:
    event_id: str
    symbol: str
    direction: Direction
    signal_bar_time: datetime
    signal_available_time: datetime
    raw_signal_price: float
    session_open: float
    upper_band: float
    lower_band: float
    dual_thrust_range: float
    atr_value: float
    trend_alignment_atr: float
    volume_ratio: float
    body_atr: float
    directional_close_position: float
    range_atr: float
    breakout_extension_atr: float
    quality_score: float


@dataclass
class _DailyBar:
    day: date
    open: float
    high: float
    low: float
    close: float


def build_dual_thrust_signals(
    symbol: str,
    candles: list[Candle],
    config: VolatilityBreakoutConfig,
    trade_start: datetime | None = None,
    trade_end: datetime | None = None,
) -> list[DualThrustSignal]:
    """Build close-confirmed Dual Thrust events without using the active day.

    The range and both bands use only fully closed prior UTC days. A signal is
    available after the decision candle closes and can therefore execute no
    earlier than the following 1m open.
    """

    if config.timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive")
    if config.lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if not candles:
        return []

    atr_values = atr(candles, config.atr_period)
    closes = [candle.close for candle in candles]
    ema_values = ema(closes, config.trend_ema_period)
    daily_bars = _closed_daily_bars(candles)
    daily_index = {bar.day: index for index, bar in enumerate(daily_bars)}
    session_opens: dict[date, float] = {}
    signal_counts: dict[date, int] = {}
    result: list[DualThrustSignal] = []

    for index, candle in enumerate(candles):
        session_opens.setdefault(candle.timestamp.date(), candle.open)
        day_index = daily_index.get(candle.timestamp.date())
        if day_index is None or day_index < config.lookback_days or index == 0:
            continue
        if trade_start is not None and candle.timestamp < trade_start:
            continue
        available_time = candle.timestamp + timedelta(minutes=config.timeframe_minutes)
        if trade_end is not None and available_time >= trade_end:
            continue
        if signal_counts.get(candle.timestamp.date(), 0) >= config.max_signals_per_symbol_day:
            continue

        prior_days = daily_bars[day_index - config.lookback_days:day_index]
        dual_range = dual_thrust_range(prior_days)
        current_atr = atr_values[index]
        if dual_range <= 0.0 or current_atr <= 0.0:
            continue
        session_open = session_opens[candle.timestamp.date()]
        upper = session_open + config.long_k * dual_range
        lower = session_open - config.short_k * dual_range
        previous_close = candles[index - 1].close

        direction = Direction.FLAT
        if config.allow_long and previous_close <= upper < candle.close:
            direction = Direction.LONG
        elif config.allow_short and previous_close >= lower > candle.close:
            direction = Direction.SHORT
        if direction == Direction.FLAT:
            continue

        side = float(direction.value)
        body_atr = abs(candle.close - candle.open) / current_atr
        candle_range = max(candle.high - candle.low, 1e-12)
        close_position = (candle.close - candle.low) / candle_range
        directional_close_position = close_position if direction == Direction.LONG else 1.0 - close_position
        volume_window = candles[max(0, index - 20):index]
        average_volume = sum(item.volume for item in volume_window) / max(1, len(volume_window))
        volume_ratio = candle.volume / max(average_volume, 1e-12)
        trend_alignment = side * (candle.close - ema_values[index]) / current_atr
        range_atr = dual_range / current_atr
        crossed_band = upper if direction == Direction.LONG else lower
        breakout_extension = side * (candle.close - crossed_band) / current_atr

        if trend_alignment < config.min_trend_alignment_atr:
            continue
        if not (config.min_volume_ratio <= volume_ratio <= config.max_volume_ratio):
            continue
        if not (config.min_body_atr <= body_atr <= config.max_body_atr):
            continue
        if directional_close_position < config.min_directional_close_position:
            continue
        if not (config.min_range_atr <= range_atr <= config.max_range_atr):
            continue
        if not (
            config.min_breakout_extension_atr
            <= breakout_extension
            <= config.max_breakout_extension_atr
        ):
            continue
        if trend_alignment > config.max_trend_alignment_atr:
            continue

        quality = (
            0.30 * min(3.0, max(0.0, volume_ratio))
            + 0.25 * min(2.0, max(0.0, body_atr))
            + 0.20 * min(1.0, max(0.0, directional_close_position))
            + 0.15 * min(2.0, max(-1.0, trend_alignment) + 1.0)
            + 0.10 * min(1.5, max(0.0, breakout_extension))
        )
        event_id = dual_thrust_event_id(symbol, direction, candle.timestamp, config)
        result.append(
            DualThrustSignal(
                event_id=event_id,
                symbol=symbol.upper(),
                direction=direction,
                signal_bar_time=candle.timestamp,
                signal_available_time=available_time,
                raw_signal_price=candle.close,
                session_open=session_open,
                upper_band=upper,
                lower_band=lower,
                dual_thrust_range=dual_range,
                atr_value=current_atr,
                trend_alignment_atr=trend_alignment,
                volume_ratio=volume_ratio,
                body_atr=body_atr,
                directional_close_position=directional_close_position,
                range_atr=range_atr,
                breakout_extension_atr=breakout_extension,
                quality_score=quality,
            )
        )
        signal_counts[candle.timestamp.date()] = signal_counts.get(candle.timestamp.date(), 0) + 1

    return result


def dual_thrust_range(prior_days: Iterable[_DailyBar]) -> float:
    rows = list(prior_days)
    if not rows:
        return 0.0
    highest_high = max(row.high for row in rows)
    highest_close = max(row.close for row in rows)
    lowest_close = min(row.close for row in rows)
    lowest_low = min(row.low for row in rows)
    return max(highest_high - lowest_close, highest_close - lowest_low)


def dual_thrust_event_id(
    symbol: str,
    direction: Direction,
    signal_time: datetime,
    config: VolatilityBreakoutConfig,
) -> str:
    raw = "|".join(
        (
            VOLATILITY_BREAKOUT_VERSION,
            symbol.upper(),
            direction.name,
            signal_time.isoformat(),
            str(config.timeframe_minutes),
            str(config.lookback_days),
            f"{config.long_k:.8f}",
            f"{config.short_k:.8f}",
        )
    )
    return hashlib.sha256(raw.encode("ascii")).hexdigest()[:24]


def _closed_daily_bars(candles: list[Candle]) -> list[_DailyBar]:
    output: list[_DailyBar] = []
    current: _DailyBar | None = None
    for candle in candles:
        candle_day = candle.timestamp.date()
        if current is None or current.day != candle_day:
            if current is not None:
                output.append(current)
            current = _DailyBar(
                day=candle_day,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
            )
            continue
        current.high = max(current.high, candle.high)
        current.low = min(current.low, candle.low)
        current.close = candle.close
    if current is not None:
        output.append(current)
    return output

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from .indicators import atr, ema
from .models import Candle, Direction


TREND_GRID_STRATEGY_NAME = "dynamic_trend_following_grid"
TREND_GRID_VERSION = "closed_bar_atr_grid_v1"


@dataclass(frozen=True)
class TrendGridConfig:
    timeframe_minutes: int = 15
    atr_period: int = 14
    fast_ema_period: int = 21
    slow_ema_period: int = 55
    slope_lookback: int = 3
    min_fast_slope_atr: float = 0.02
    min_slow_slope_atr: float = -0.02
    min_alignment_atr: float = 0.0
    max_alignment_atr: float = 2.0
    min_atr_pct: float = 0.001
    max_atr_pct: float = 0.08
    max_entry_extension_atr: float = 1.0
    entry_mode: str = "pullback_touch"
    pullback_touch_atr: float = 0.25
    min_directional_close_position: float = 0.50
    min_volume_ratio: float = 0.0
    max_volume_ratio: float = 999.0
    reentry_interval_bars: int = 4
    max_signals_per_symbol_day: int = 4
    allow_long: bool = True
    allow_short: bool = True
    grid_spacing_atr: float = 0.50
    grid_levels: int = 4
    grid_target_spacing: float = 1.0
    deeper_level_size_multiplier: float = 1.0
    initial_entry_enabled: bool = True
    hard_stop_atr_multiple: float = 3.0
    hard_stop_slow_ema_buffer_atr: float = 0.50
    regime_exit_mode: str = "fast_ema"
    regime_exit_confirm_bars: int = 1
    max_campaign_minutes: int = 1_440
    max_cycles_per_level: int = 3
    max_total_entries: int = 0
    pause_new_fills_on_fast_breach: bool = False
    campaign_loss_limit_r: float = 0.0
    campaign_take_profit_r: float = 0.0
    profit_lock_activation_r: float = 0.0
    profit_giveback_r: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.timeframe_minutes not in {15, 30, 60}:
            raise ValueError("timeframe_minutes must be 15, 30, or 60")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.fast_ema_period <= 0 or self.slow_ema_period <= self.fast_ema_period:
            raise ValueError("slow_ema_period must be greater than fast_ema_period")
        if self.slope_lookback <= 0:
            raise ValueError("slope_lookback must be positive")
        if self.entry_mode not in {"continuous", "regime_transition", "fast_reclaim", "pullback_touch", "hybrid"}:
            raise ValueError(f"unsupported entry_mode: {self.entry_mode}")
        if self.grid_spacing_atr <= 0.0 or self.grid_levels <= 0:
            raise ValueError("grid spacing and levels must be positive")
        if self.grid_target_spacing <= 0.0:
            raise ValueError("grid_target_spacing must be positive")
        if not 0.0 < self.deeper_level_size_multiplier <= 1.0:
            raise ValueError("deeper_level_size_multiplier must be in (0, 1]")
        if self.hard_stop_atr_multiple <= self.grid_spacing_atr * self.grid_levels:
            raise ValueError("hard stop must be beyond the deepest grid level")
        if self.regime_exit_mode not in {"fast_ema", "slow_ema", "ema_cross", "fast_or_cross"}:
            raise ValueError(f"unsupported regime_exit_mode: {self.regime_exit_mode}")
        if self.regime_exit_confirm_bars <= 0:
            raise ValueError("regime_exit_confirm_bars must be positive")
        if self.max_total_entries < 0:
            raise ValueError("max_total_entries cannot be negative")
        if min(
            self.campaign_loss_limit_r,
            self.campaign_take_profit_r,
            self.profit_lock_activation_r,
            self.profit_giveback_r,
        ) < 0.0:
            raise ValueError("campaign R controls cannot be negative")


@dataclass(frozen=True)
class TrendGridSnapshot:
    symbol: str
    bar_time: datetime
    available_time: datetime
    close: float
    atr_value: float
    fast_ema: float
    slow_ema: float
    fast_slope_atr: float
    slow_slope_atr: float
    long_trend_valid: bool
    short_trend_valid: bool

    def exit_invalid(self, direction: Direction, mode: str) -> bool:
        if direction == Direction.LONG:
            fast_breach = self.close < self.fast_ema
            slow_breach = self.close < self.slow_ema
            cross = self.fast_ema <= self.slow_ema
        else:
            fast_breach = self.close > self.fast_ema
            slow_breach = self.close > self.slow_ema
            cross = self.fast_ema >= self.slow_ema
        if mode == "fast_ema":
            return fast_breach
        if mode == "slow_ema":
            return slow_breach
        if mode == "ema_cross":
            return cross
        return fast_breach or cross


@dataclass(frozen=True)
class TrendGridSignal:
    event_id: str
    symbol: str
    direction: Direction
    signal_bar_time: datetime
    signal_available_time: datetime
    raw_signal_price: float
    atr_value: float
    fast_ema: float
    slow_ema: float
    fast_slope_atr: float
    slow_slope_atr: float
    alignment_atr: float
    extension_atr: float
    directional_close_position: float
    volume_ratio: float
    quality_score: float


def build_trend_grid_timeline(
    symbol: str,
    candles: list[Candle],
    config: TrendGridConfig,
    trade_start: datetime | None = None,
    trade_end: datetime | None = None,
) -> tuple[list[TrendGridSnapshot], list[TrendGridSignal]]:
    config.validate()
    if not candles:
        return [], []

    closes = [candle.close for candle in candles]
    fast_values = ema(closes, config.fast_ema_period)
    slow_values = ema(closes, config.slow_ema_period)
    atr_values = atr(candles, config.atr_period)
    warmup = max(config.slow_ema_period, config.atr_period) + config.slope_lookback
    snapshots: list[TrendGridSnapshot] = []
    signals: list[TrendGridSignal] = []
    previous_long_valid = False
    previous_short_valid = False
    last_signal_index = {Direction.LONG: -10**9, Direction.SHORT: -10**9}
    daily_signal_count: dict[tuple[datetime.date, Direction], int] = {}

    for index in range(warmup, len(candles)):
        candle = candles[index]
        current_atr = atr_values[index]
        if current_atr <= 0.0 or candle.close <= 0.0:
            continue
        available = candle.timestamp + timedelta(minutes=config.timeframe_minutes)
        fast_slope = (fast_values[index] - fast_values[index - config.slope_lookback]) / current_atr
        slow_slope = (slow_values[index] - slow_values[index - config.slope_lookback]) / current_atr
        atr_pct = current_atr / candle.close
        long_alignment = (fast_values[index] - slow_values[index]) / current_atr
        short_alignment = -long_alignment
        long_extension = (candle.close - fast_values[index]) / current_atr
        short_extension = -long_extension
        long_valid = (
            config.allow_long
            and fast_values[index] > slow_values[index]
            and fast_slope >= config.min_fast_slope_atr
            and slow_slope >= config.min_slow_slope_atr
            and config.min_alignment_atr <= long_alignment <= config.max_alignment_atr
            and config.min_atr_pct <= atr_pct <= config.max_atr_pct
        )
        short_valid = (
            config.allow_short
            and fast_values[index] < slow_values[index]
            and fast_slope <= -config.min_fast_slope_atr
            and slow_slope <= -config.min_slow_slope_atr
            and config.min_alignment_atr <= short_alignment <= config.max_alignment_atr
            and config.min_atr_pct <= atr_pct <= config.max_atr_pct
        )
        snapshot = TrendGridSnapshot(
            symbol=symbol.upper(),
            bar_time=candle.timestamp,
            available_time=available,
            close=candle.close,
            atr_value=current_atr,
            fast_ema=fast_values[index],
            slow_ema=slow_values[index],
            fast_slope_atr=fast_slope,
            slow_slope_atr=slow_slope,
            long_trend_valid=long_valid,
            short_trend_valid=short_valid,
        )
        snapshots.append(snapshot)

        if trade_start is not None and candle.timestamp < trade_start:
            previous_long_valid = long_valid
            previous_short_valid = short_valid
            continue
        if trade_end is not None and available >= trade_end:
            previous_long_valid = long_valid
            previous_short_valid = short_valid
            continue

        candle_range = max(candle.high - candle.low, 1e-12)
        close_position = (candle.close - candle.low) / candle_range
        volume_window = candles[max(0, index - 20):index]
        average_volume = sum(item.volume for item in volume_window) / max(1, len(volume_window))
        volume_ratio = candle.volume / max(average_volume, 1e-12)
        for direction, valid, previous_valid, alignment, extension in (
            (Direction.LONG, long_valid, previous_long_valid, long_alignment, long_extension),
            (Direction.SHORT, short_valid, previous_short_valid, short_alignment, short_extension),
        ):
            directional_position = close_position if direction == Direction.LONG else 1.0 - close_position
            if not valid or not 0.0 <= extension <= config.max_entry_extension_atr:
                continue
            if directional_position < config.min_directional_close_position:
                continue
            if not config.min_volume_ratio <= volume_ratio <= config.max_volume_ratio:
                continue
            if index - last_signal_index[direction] < config.reentry_interval_bars:
                continue
            day_key = (candle.timestamp.date(), direction)
            if daily_signal_count.get(day_key, 0) >= config.max_signals_per_symbol_day:
                continue
            if not _entry_triggered(
                direction,
                config.entry_mode,
                candle,
                candles[index - 1],
                fast_values[index],
                fast_values[index - 1],
                current_atr,
                config.pullback_touch_atr,
                previous_valid,
            ):
                continue

            directional_fast_slope = direction.value * fast_slope
            directional_slow_slope = direction.value * slow_slope
            quality = (
                0.30 * min(2.0, max(0.0, directional_fast_slope + 0.25))
                + 0.20 * min(2.0, max(0.0, directional_slow_slope + 0.25))
                + 0.20 * min(2.0, max(0.0, alignment))
                + 0.15 * min(1.0, max(0.0, 1.0 - extension / max(config.max_entry_extension_atr, 1e-12)))
                + 0.10 * min(1.5, max(0.0, volume_ratio))
                + 0.05 * directional_position
            )
            signals.append(
                TrendGridSignal(
                    event_id=trend_grid_event_id(symbol, direction, candle.timestamp, config),
                    symbol=symbol.upper(),
                    direction=direction,
                    signal_bar_time=candle.timestamp,
                    signal_available_time=available,
                    raw_signal_price=candle.close,
                    atr_value=current_atr,
                    fast_ema=fast_values[index],
                    slow_ema=slow_values[index],
                    fast_slope_atr=fast_slope,
                    slow_slope_atr=slow_slope,
                    alignment_atr=alignment,
                    extension_atr=extension,
                    directional_close_position=directional_position,
                    volume_ratio=volume_ratio,
                    quality_score=quality,
                )
            )
            last_signal_index[direction] = index
            daily_signal_count[day_key] = daily_signal_count.get(day_key, 0) + 1

        previous_long_valid = long_valid
        previous_short_valid = short_valid

    return snapshots, signals


def _entry_triggered(
    direction: Direction,
    mode: str,
    candle: Candle,
    previous: Candle,
    fast_ema: float,
    previous_fast_ema: float,
    atr_value: float,
    touch_atr: float,
    previous_trend_valid: bool,
) -> bool:
    if mode == "continuous":
        return True
    transition = not previous_trend_valid
    if direction == Direction.LONG:
        reclaim = previous.close <= previous_fast_ema and candle.close > fast_ema
        pullback = candle.low <= fast_ema + touch_atr * atr_value and candle.close > fast_ema
    else:
        reclaim = previous.close >= previous_fast_ema and candle.close < fast_ema
        pullback = candle.high >= fast_ema - touch_atr * atr_value and candle.close < fast_ema
    if mode == "regime_transition":
        return transition
    if mode == "fast_reclaim":
        return reclaim
    if mode == "pullback_touch":
        return pullback
    return transition or reclaim or pullback


def trend_grid_event_id(
    symbol: str,
    direction: Direction,
    signal_time: datetime,
    config: TrendGridConfig,
) -> str:
    raw = "|".join(
        (
            TREND_GRID_VERSION,
            symbol.upper(),
            direction.name,
            signal_time.isoformat(),
            str(config.timeframe_minutes),
            str(config.fast_ema_period),
            str(config.slow_ema_period),
            config.entry_mode,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

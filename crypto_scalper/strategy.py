from __future__ import annotations

from collections.abc import Sequence

from .config import StrategyConfig
from .indicators import atr, ema, rolling_high, rolling_low
from .models import Candle, Direction, Signal


class VolatilityBreakoutScalper:
    """Small-timeframe breakout strategy with volatility and trend filters."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self._fast: list[float] = []
        self._slow: list[float] = []
        self._atr: list[float] = []
        self._timestamps = []
        self._opens: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []
        self._volumes: list[float] = []
        self._avg_volume: list[float] = []

    def prepare(self, candles: Sequence[Candle]) -> None:
        self._timestamps = [candle.timestamp for candle in candles]
        self._opens = [candle.open for candle in candles]
        self._highs = [candle.high for candle in candles]
        self._lows = [candle.low for candle in candles]
        self._closes = [candle.close for candle in candles]
        self._volumes = [candle.volume for candle in candles]
        self._fast = ema(self._closes, self.config.fast_ema)
        self._slow = ema(self._closes, self.config.slow_ema)
        self._atr = atr(candles, self.config.atr_period)
        self._avg_volume = self._rolling_average(self._volumes, self.config.volume_period)

    @property
    def warmup_bars(self) -> int:
        return max(self.config.slow_ema, self.config.atr_period, self.config.channel_period, self.config.volume_period) + 2

    def atr_at(self, index: int) -> float:
        if not self._atr:
            return 0.0
        bounded = min(max(index, 0), len(self._atr) - 1)
        return self._atr[bounded]

    def signal(self, index: int, candles: Sequence[Candle]) -> Signal:
        if not self._fast or not self._slow or not self._atr:
            self.prepare(candles)

        candle = candles[index]
        if index < self.warmup_bars:
            return self._hold("warming_up")

        atr_value = self._atr[index - 1]
        if atr_value <= 0:
            return self._hold("zero_atr")

        atr_pct = atr_value / candle.close
        if atr_pct < self.config.min_atr_pct:
            return self._hold("volatility_too_low")
        if self.config.max_atr_pct > 0 and atr_pct > self.config.max_atr_pct:
            return self._hold("volatility_too_high")

        upper_channel = rolling_high(self._highs, index, self.config.channel_period)
        lower_channel = rolling_low(self._lows, index, self.config.channel_period)
        fast = self._fast[index]
        slow = self._slow[index]
        average_volume = self._avg_volume[index - 1]

        spike_signal = self._spike_signal(index, candle, atr_value, average_volume)
        if spike_signal.direction != Direction.FLAT:
            return spike_signal

        if self.config.spike_guard_enabled and self._recent_spike(index, self.config.spike_block_bars):
            return self._hold("spike_guard_cooldown")

        ema_gap = abs(fast - slow) / atr_value
        if ema_gap < self.config.ema_gap_atr:
            return self._hold("trend_gap_too_small")

        if self.config.min_volume_ratio > 0 and average_volume > 0:
            if candle.volume < average_volume * self.config.min_volume_ratio:
                return self._hold("volume_too_low")

        breakout_buffer = atr_value * self.config.breakout_buffer_atr
        stop_pct = max(atr_pct * self.config.stop_loss_atr, 0.0001)
        take_profit_pct = max(atr_pct * self.config.take_profit_atr, stop_pct * 1.01)

        if candle.close > upper_channel + breakout_buffer and fast > slow:
            return Signal(
                direction=Direction.LONG,
                confidence=min(1.0, atr_pct / max(self.config.min_atr_pct, 1e-12)),
                reason="long_breakout",
                stop_loss_pct=stop_pct,
                take_profit_pct=take_profit_pct,
            )

        if self.config.allow_short and candle.close < lower_channel - breakout_buffer and fast < slow:
            return Signal(
                direction=Direction.SHORT,
                confidence=min(1.0, atr_pct / max(self.config.min_atr_pct, 1e-12)),
                reason="short_breakdown",
                stop_loss_pct=stop_pct,
                take_profit_pct=take_profit_pct,
            )

        if fast > slow:
            return self._hold("trend_up_no_breakout")
        if fast < slow:
            return self._hold("trend_down_no_breakdown")
        return self._hold("no_edge")

    @staticmethod
    def _hold(reason: str) -> Signal:
        return Signal(
            direction=Direction.FLAT,
            confidence=0.0,
            reason=reason,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
        )

    def _spike_signal(self, index: int, candle: Candle, atr_value: float, average_volume: float) -> Signal:
        if not self.config.spike_trade_enabled:
            return self._hold("spike_trade_disabled")

        spike = self._classify_spike(candle, atr_value, average_volume)
        if spike == Direction.FLAT:
            return self._hold("no_spike")

        candle_range = max(candle.high - candle.low, 1e-12)
        if spike == Direction.LONG:
            recovered = (candle.close - candle.low) / candle_range
            if recovered < self.config.spike_recovery_ratio or candle.close <= candle.open:
                return self._hold("lower_spike_not_recovered")
        else:
            recovered = (candle.high - candle.close) / candle_range
            if recovered < self.config.spike_recovery_ratio or candle.close >= candle.open:
                return self._hold("upper_spike_not_recovered")
            if not self.config.allow_short:
                return self._hold("short_disabled")

        atr_pct = atr_value / candle.close
        stop_pct = max(atr_pct * self.config.spike_stop_atr, 0.0001)
        take_profit_pct = max(atr_pct * self.config.spike_take_profit_atr, stop_pct * 1.01)
        return Signal(
            direction=spike,
            confidence=0.5,
            reason="lower_spike_reversal" if spike == Direction.LONG else "upper_spike_reversal",
            stop_loss_pct=stop_pct,
            take_profit_pct=take_profit_pct,
            risk_multiplier=max(0.0, min(1.0, self.config.spike_risk_multiplier)),
            max_holding_bars=max(0, self.config.spike_max_holding_bars),
        )

    def _recent_spike(self, index: int, bars: int) -> bool:
        if bars <= 0:
            return False
        start = max(self.warmup_bars, index - bars + 1)
        for candidate in range(start, index + 1):
            average_volume = self._avg_volume[candidate - 1] if candidate > 0 else self._avg_volume[candidate]
            if self._classify_spike(self._candle_at(candidate), self._atr[candidate - 1], average_volume) != Direction.FLAT:
                return True
        return False

    def _classify_spike(self, candle: Candle, atr_value: float, average_volume: float) -> Direction:
        if atr_value <= 0:
            return Direction.FLAT
        candle_range = candle.high - candle.low
        if candle_range <= 0:
            return Direction.FLAT
        if candle_range < atr_value * self.config.spike_min_range_atr:
            return Direction.FLAT

        volume_ratio = candle.volume / average_volume if average_volume > 0 else 1.0
        if volume_ratio < self.config.spike_min_volume_ratio:
            return Direction.FLAT

        upper_wick = candle.high - max(candle.open, candle.close)
        lower_wick = min(candle.open, candle.close) - candle.low
        if lower_wick >= atr_value * self.config.spike_min_wick_atr and lower_wick / candle_range >= self.config.spike_min_wick_ratio:
            return Direction.LONG
        if upper_wick >= atr_value * self.config.spike_min_wick_atr and upper_wick / candle_range >= self.config.spike_min_wick_ratio:
            return Direction.SHORT
        return Direction.FLAT

    def _candle_at(self, index: int) -> Candle:
        return Candle(
            timestamp=self._timestamps[index],
            open=self._opens[index],
            high=self._highs[index],
            low=self._lows[index],
            close=self._closes[index],
            volume=self._volumes[index],
        )

    @staticmethod
    def _rolling_average(values: Sequence[float], period: int) -> list[float]:
        if period <= 0:
            raise ValueError("period must be positive")
        result: list[float] = []
        running_sum = 0.0
        for index, value in enumerate(values):
            running_sum += value
            if index >= period:
                running_sum -= values[index - period]
            width = min(index + 1, period)
            result.append(running_sum / width)
        return result

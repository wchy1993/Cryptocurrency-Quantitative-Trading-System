from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .config import StrategyConfig
from .indicators import atr, ema, rolling_high, rolling_low, rsi
from .models import Candle, Direction, Signal


class VolatilityBreakoutScalper:
    """Small-timeframe breakout strategy with volatility and trend filters."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self._fast: list[float] = []
        self._slow: list[float] = []
        self._atr: list[float] = []
        self._rsi: list[float] = []
        self._breakout_fast_rsi: list[float] = []
        self._breakout_mid_rsi: list[float] = []
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
        self._rsi = rsi(self._closes, 14)
        self._breakout_fast_rsi = rsi(self._closes, self.config.breakout_rsi_fast_period)
        self._breakout_mid_rsi = rsi(self._closes, self.config.breakout_rsi_mid_period)
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

        breakout_buffer = atr_value * self.config.breakout_buffer_atr
        stop_pct = max(atr_pct * self.config.stop_loss_atr, 0.0001)
        take_profit_pct = max(atr_pct * self.config.take_profit_atr, stop_pct * 1.01)

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
        startup_signal = self._startup_breakout_signal(
            index,
            candle,
            atr_value,
            average_volume,
            upper_channel,
            lower_channel,
            breakout_buffer,
            stop_pct,
            take_profit_pct,
        )
        if startup_signal.direction != Direction.FLAT:
            return startup_signal

        if ema_gap < self.config.ema_gap_atr:
            return self._hold("trend_gap_too_small")

        if self.config.min_volume_ratio > 0 and average_volume > 0:
            if candle.volume < average_volume * self.config.min_volume_ratio:
                return self._hold("volume_too_low")

        current_rsi = self._rsi[index]
        previous_rsi = self._rsi[index - 1]

        if (
            self.config.rsi_reversal_enabled
            and previous_rsi <= 30.0 < current_rsi
            and candle.close > candle.open
            and candle.close >= slow * 0.985
        ):
            quality = max(0.35, min(0.75, 0.35 + (30.0 - min(previous_rsi, 30.0)) / 50.0 + (current_rsi - previous_rsi) / 80.0))
            return self._scored_signal(Signal(
                direction=Direction.LONG,
                confidence=quality,
                reason=f"long_rsi_reversal rsi={current_rsi:.1f}",
                stop_loss_pct=stop_pct,
                take_profit_pct=max(stop_pct * 0.65, min(take_profit_pct, stop_pct * 0.9)),
                risk_multiplier=quality,
            ), index, atr_value)

        if (
            self.config.rsi_reversal_enabled
            and self.config.allow_short
            and previous_rsi >= 70.0 > current_rsi
            and candle.close < candle.open
            and candle.close <= slow * 1.015
        ):
            quality = max(0.35, min(0.75, 0.35 + (max(previous_rsi, 70.0) - 70.0) / 50.0 + (previous_rsi - current_rsi) / 80.0))
            return self._scored_signal(Signal(
                direction=Direction.SHORT,
                confidence=quality,
                reason=f"short_rsi_reversal rsi={current_rsi:.1f}",
                stop_loss_pct=stop_pct,
                take_profit_pct=max(stop_pct * 0.65, min(take_profit_pct, stop_pct * 0.9)),
                risk_multiplier=quality,
            ), index, atr_value)

        if candle.close > upper_channel + breakout_buffer and fast > slow:
            quality = self._signal_quality(
                candle.close - upper_channel - breakout_buffer,
                atr_value,
                ema_gap,
                candle,
                average_volume,
            )
            breakout_distance = candle.close - upper_channel - breakout_buffer
            guard_reason = self._breakout_rsi_guard_reason(Direction.LONG, index, atr_value, breakout_distance, average_volume)
            if guard_reason:
                return self._hold(guard_reason)
            signal = Signal(
                direction=Direction.LONG,
                confidence=quality,
                reason="long_breakout",
                stop_loss_pct=stop_pct,
                take_profit_pct=take_profit_pct,
                risk_multiplier=quality,
            )
            if not self.config.ordinary_breakout_enabled and not self._is_super_volume_breakout_candidate(
                Direction.LONG,
                index,
                atr_value,
                breakout_distance,
                average_volume,
            ):
                return self._hold("ordinary_breakout_disabled")
            return self._scored_signal(signal, index, atr_value, breakout_distance, average_volume)

        if self.config.allow_short and candle.close < lower_channel - breakout_buffer and fast < slow:
            quality = self._signal_quality(
                lower_channel - breakout_buffer - candle.close,
                atr_value,
                ema_gap,
                candle,
                average_volume,
            )
            breakout_distance = lower_channel - breakout_buffer - candle.close
            guard_reason = self._breakout_rsi_guard_reason(Direction.SHORT, index, atr_value, breakout_distance, average_volume)
            if guard_reason:
                return self._hold(guard_reason)
            signal = Signal(
                direction=Direction.SHORT,
                confidence=quality,
                reason="short_breakdown",
                stop_loss_pct=stop_pct,
                take_profit_pct=take_profit_pct,
                risk_multiplier=quality,
            )
            if not self.config.ordinary_breakout_enabled and not self._is_super_volume_breakout_candidate(
                Direction.SHORT,
                index,
                atr_value,
                breakout_distance,
                average_volume,
            ):
                return self._hold("ordinary_breakout_disabled")
            return self._scored_signal(signal, index, atr_value, breakout_distance, average_volume)

        previous_close = self._closes[index - 1]
        previous_fast = self._fast[index - 1]
        if (
            self.config.pullback_reclaim_enabled
            and fast > slow
            and previous_close <= previous_fast
            and candle.close > fast
            and candle.close > candle.open
        ):
            quality = self._signal_quality(
                max(candle.close - fast, atr_value * 0.25),
                atr_value,
                ema_gap,
                candle,
                average_volume,
            )
            return self._scored_signal(Signal(
                direction=Direction.LONG,
                confidence=quality,
                reason="long_pullback_reclaim",
                stop_loss_pct=stop_pct,
                take_profit_pct=take_profit_pct,
                risk_multiplier=max(0.25, quality * 0.85),
            ), index, atr_value)

        if (
            self.config.pullback_reclaim_enabled
            and
            self.config.allow_short
            and fast < slow
            and previous_close >= previous_fast
            and candle.close < fast
            and candle.close < candle.open
        ):
            quality = self._signal_quality(
                max(fast - candle.close, atr_value * 0.25),
                atr_value,
                ema_gap,
                candle,
                average_volume,
            )
            return self._scored_signal(Signal(
                direction=Direction.SHORT,
                confidence=quality,
                reason="short_pullback_reject",
                stop_loss_pct=stop_pct,
                take_profit_pct=take_profit_pct,
                risk_multiplier=max(0.25, quality * 0.85),
        ), index, atr_value)

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

    def _startup_breakout_signal(
        self,
        index: int,
        candle: Candle,
        atr_value: float,
        average_volume: float,
        upper_channel: float,
        lower_channel: float,
        breakout_buffer: float,
        stop_pct: float,
        take_profit_pct: float,
    ) -> Signal:
        if not self.config.startup_breakout_enabled or atr_value <= 0 or average_volume <= 0:
            return self._hold("startup_breakout_disabled")

        volume_ratio = candle.volume / average_volume
        if volume_ratio < self.config.startup_min_volume_ratio:
            return self._hold("startup_volume_too_low")

        min_breakout = atr_value * self.config.startup_min_breakout_atr
        min_body = atr_value * self.config.startup_min_body_atr
        long_breakout = candle.close - upper_channel - breakout_buffer
        short_breakout = lower_channel - breakout_buffer - candle.close
        long_body = candle.close - candle.open
        short_body = candle.open - candle.close
        take_profit_pct = max(take_profit_pct * self.config.startup_take_profit_multiplier, stop_pct * 1.05)

        if long_breakout >= min_breakout and long_body >= min_body:
            reason = f"long_startup_breakout volume={volume_ratio:.2f}x"
            if volume_ratio >= self.config.super_volume_min_ratio:
                reason = f"{reason}_super_volume"
            quality = self._signal_quality(
                max(long_breakout, atr_value * 0.35),
                atr_value,
                self.config.ema_gap_atr,
                candle,
                average_volume,
            )
            risk_multiplier = max(
                0.0,
                min(
                    1.5,
                    quality
                    * self.config.startup_risk_multiplier
                    * self.config.long_risk_bias
                    * self._trend_risk_multiplier(Direction.LONG, reason),
                ),
            )
            if self.config.long_risk_bias <= 0:
                return self._hold("long_direction_disabled")
            return Signal(
                direction=Direction.LONG,
                confidence=max(quality, 0.72),
                reason=reason,
                stop_loss_pct=stop_pct,
                take_profit_pct=take_profit_pct,
                risk_multiplier=risk_multiplier,
            )

        if self.config.allow_short and short_breakout >= min_breakout and short_body >= min_body:
            reason = f"short_startup_breakout volume={volume_ratio:.2f}x"
            if volume_ratio >= self.config.super_volume_min_ratio:
                reason = f"{reason}_super_volume"
            quality = self._signal_quality(
                max(short_breakout, atr_value * 0.35),
                atr_value,
                self.config.ema_gap_atr,
                candle,
                average_volume,
            )
            risk_multiplier = max(
                0.0,
                min(
                    1.5,
                    quality
                    * self.config.startup_risk_multiplier
                    * self.config.short_risk_bias
                    * self._trend_risk_multiplier(Direction.SHORT, reason),
                ),
            )
            if self.config.short_risk_bias <= 0:
                return self._hold("short_direction_disabled")
            return Signal(
                direction=Direction.SHORT,
                confidence=max(quality, 0.72),
                reason=reason,
                stop_loss_pct=stop_pct,
                take_profit_pct=take_profit_pct,
                risk_multiplier=risk_multiplier,
            )

        return self._hold("startup_no_breakout")

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
        return self._scored_signal(Signal(
            direction=spike,
            confidence=0.5,
            reason="lower_spike_reversal" if spike == Direction.LONG else "upper_spike_reversal",
            stop_loss_pct=stop_pct,
            take_profit_pct=take_profit_pct,
            risk_multiplier=max(0.0, min(1.0, self.config.spike_risk_multiplier)),
            max_holding_bars=max(0, self.config.spike_max_holding_bars),
        ), index, atr_value)

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

    def _signal_quality(
        self,
        breakout_distance: float,
        atr_value: float,
        ema_gap: float,
        candle: Candle,
        average_volume: float,
    ) -> float:
        # The score uses one feature from each signal family: price expansion, trend, volume, volatility.
        # This avoids counting highly correlated EMA variants as separate independent confirmations.
        breakout_score = min(1.0, max(0.0, breakout_distance / max(atr_value * 0.75, 1e-12)))
        trend_floor = self.config.ema_gap_atr if self.config.ema_gap_atr > 0 else 0.25
        trend_score = min(1.0, max(0.0, ema_gap / max(trend_floor * 2.0, 1e-12)))

        if self.config.min_volume_ratio > 0 and average_volume > 0:
            volume_score = min(1.0, max(0.0, candle.volume / (average_volume * self.config.min_volume_ratio)))
        else:
            volume_score = 0.75

        atr_pct = atr_value / max(candle.close, 1e-12)
        if self.config.max_atr_pct > self.config.min_atr_pct > 0:
            midpoint = (self.config.min_atr_pct + self.config.max_atr_pct) / 2.0
            half_width = (self.config.max_atr_pct - self.config.min_atr_pct) / 2.0
            volatility_score = 1.0 - min(1.0, abs(atr_pct - midpoint) / max(half_width, 1e-12)) * 0.35
        else:
            volatility_score = 0.85

        weighted = (
            breakout_score * 0.35
            + trend_score * 0.30
            + volume_score * 0.20
            + volatility_score * 0.15
        )
        return max(0.25, min(1.0, 0.25 + weighted * 0.75))

    def _scored_signal(
        self,
        signal: Signal,
        index: int,
        atr_value: float,
        breakout_distance: float = 0.0,
        average_volume: float = 0.0,
    ) -> Signal:
        if signal.direction == Direction.FLAT:
            return signal

        if self.config.regime_filter_enabled:
            allowed, regime_reason = self._regime_allows(signal.direction, index, atr_value)
            if not allowed:
                return self._hold(regime_reason)

        score = max(0.0, min(1.0, signal.confidence))
        if signal.direction == Direction.LONG:
            threshold = self.config.long_score_threshold
            bias = self.config.long_risk_bias
            threshold_reason = "long_score_below_threshold"
            disabled_reason = "long_direction_disabled"
        else:
            threshold = self.config.short_score_threshold
            bias = self.config.short_risk_bias
            threshold_reason = "short_score_below_threshold"
            disabled_reason = "short_direction_disabled"

        if bias <= 0:
            return self._hold(disabled_reason)

        if score < threshold:
            return self._hold(f"{threshold_reason} score={score:.2f}")

        signal = self._super_volume_breakout_signal(signal, index, atr_value, breakout_distance, average_volume)
        final_score = max(score, max(0.0, min(1.0, signal.confidence)))
        trend_multiplier = self._trend_risk_multiplier(signal.direction, signal.reason, signal.max_holding_bars)
        risk_multiplier = max(0.0, min(1.5, signal.risk_multiplier * bias * trend_multiplier))
        return replace(signal, confidence=final_score, risk_multiplier=risk_multiplier, reason=f"{signal.reason} score={final_score:.2f}")

    def _trend_risk_multiplier(self, direction: Direction, reason: str = "", max_holding_bars: int = 0) -> float:
        if not self.config.trend_risk_control_enabled:
            return 1.0
        if not self._trend_risk_applies(reason, max_holding_bars):
            return 1.0
        if direction == Direction.LONG:
            multiplier = max(0.0, min(1.0, self.config.trend_long_risk_multiplier))
        elif direction == Direction.SHORT:
            multiplier = max(0.0, min(1.0, self.config.trend_short_risk_multiplier))
        else:
            return 1.0
        if "super_volume" in reason.lower():
            floor = max(0.0, min(1.0, self.config.trend_super_volume_min_risk_multiplier))
            multiplier = max(multiplier, floor)
        return multiplier

    def _trend_risk_applies(self, reason: str, max_holding_bars: int) -> bool:
        normalized = reason.lower()
        if any(token in normalized for token in ("breakout", "breakdown", "pullback")):
            return True
        min_holding = max(1, self.config.trend_risk_min_holding_bars)
        return max_holding_bars >= min_holding

    def _super_volume_breakout_signal(
        self,
        signal: Signal,
        index: int,
        atr_value: float,
        breakout_distance: float,
        average_volume: float,
    ) -> Signal:
        if not self.config.super_volume_breakout_enabled:
            return signal
        if breakout_distance <= 0 or atr_value <= 0 or average_volume <= 0:
            return signal

        candle = self._candle_at(index)
        directional_body = (candle.close - candle.open) * signal.direction.value
        volume_ratio = candle.volume / average_volume
        if volume_ratio < self.config.super_volume_min_ratio:
            return signal
        if breakout_distance < atr_value * self.config.super_volume_min_breakout_atr:
            return signal
        if directional_body < atr_value * self.config.super_volume_min_body_atr:
            return signal

        boosted_confidence = min(1.0, signal.confidence + self.config.super_volume_confidence_boost)
        boosted_risk = min(1.5, signal.risk_multiplier * self.config.super_volume_risk_multiplier)
        boosted_take_profit = max(
            signal.take_profit_pct * self.config.super_volume_take_profit_multiplier,
            signal.stop_loss_pct * 1.15,
        )
        max_holding_bars = signal.max_holding_bars
        if self.config.super_volume_max_holding_bars > 0:
            max_holding_bars = self.config.super_volume_max_holding_bars
        return replace(
            signal,
            confidence=boosted_confidence,
            take_profit_pct=boosted_take_profit,
            risk_multiplier=boosted_risk,
            max_holding_bars=max_holding_bars,
            reason=f"{signal.reason}_super_volume volume={volume_ratio:.2f}x",
        )

    def _breakout_rsi_guard_reason(
        self,
        direction: Direction,
        index: int,
        atr_value: float,
        breakout_distance: float,
        average_volume: float,
    ) -> str | None:
        if not self.config.breakout_rsi_guard_enabled:
            return None
        if self._is_super_volume_breakout_candidate(direction, index, atr_value, breakout_distance, average_volume):
            return None
        fast_rsi = self._breakout_fast_rsi[index]
        mid_rsi = self._breakout_mid_rsi[index]
        if (
            direction == Direction.LONG
            and fast_rsi >= self.config.breakout_long_rsi_fast_ceiling
            and mid_rsi >= self.config.breakout_long_rsi_mid_ceiling
        ):
            return f"long_breakout_rsi_hot rsi{self.config.breakout_rsi_fast_period}={fast_rsi:.1f} rsi{self.config.breakout_rsi_mid_period}={mid_rsi:.1f}"
        if (
            direction == Direction.SHORT
            and fast_rsi <= self.config.breakout_short_rsi_fast_floor
            and mid_rsi <= self.config.breakout_short_rsi_mid_floor
        ):
            return f"short_breakdown_rsi_cold rsi{self.config.breakout_rsi_fast_period}={fast_rsi:.1f} rsi{self.config.breakout_rsi_mid_period}={mid_rsi:.1f}"
        return None

    def _is_super_volume_breakout_candidate(
        self,
        direction: Direction,
        index: int,
        atr_value: float,
        breakout_distance: float,
        average_volume: float,
    ) -> bool:
        if not self.config.super_volume_breakout_enabled:
            return False
        if breakout_distance <= 0 or atr_value <= 0 or average_volume <= 0:
            return False
        candle = self._candle_at(index)
        directional_body = (candle.close - candle.open) * direction.value
        volume_ratio = candle.volume / average_volume
        return (
            volume_ratio >= self.config.super_volume_min_ratio
            and breakout_distance >= atr_value * self.config.super_volume_min_breakout_atr
            and directional_body >= atr_value * self.config.super_volume_min_body_atr
        )

    def _regime_allows(self, direction: Direction, index: int, atr_value: float) -> tuple[bool, str]:
        lookback = max(1, self.config.regime_lookback)
        if index < lookback or atr_value <= 0:
            return True, "regime_warmup"
        slow_slope_atr = (self._slow[index] - self._slow[index - lookback]) / atr_value
        if direction == Direction.LONG and slow_slope_atr < self.config.long_min_slow_slope_atr:
            return False, f"long_regime_block slope_atr={slow_slope_atr:.2f}"
        if direction == Direction.SHORT and slow_slope_atr > self.config.short_max_slow_slope_atr:
            return False, f"short_regime_block slope_atr={slow_slope_atr:.2f}"
        return True, f"regime_ok slope_atr={slow_slope_atr:.2f}"

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

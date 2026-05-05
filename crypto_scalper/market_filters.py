from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .indicators import kdj, macd, rsi
from .live_config import MultiTimeframeFilterConfig
from .models import Candle, Direction


@dataclass(frozen=True)
class TimeframeSignal:
    timeframe: str
    close: float
    rsi: float
    previous_rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    previous_macd_histogram: float
    k: float
    d: float
    j: float
    previous_k: float
    previous_d: float


class MultiTimeframeFilter:
    def __init__(self, config: MultiTimeframeFilterConfig) -> None:
        self.config = config

    @property
    def warmup_bars(self) -> int:
        return max(
            self.config.rsi_period + 2,
            self.config.macd_slow + self.config.macd_signal + 2,
            self.config.kdj_period + 2,
        )

    def snapshot(self, timeframe: str, candles: Sequence[Candle]) -> TimeframeSignal:
        if len(candles) < self.warmup_bars:
            raise ValueError(f"{timeframe} K线不足")
        closes = [candle.close for candle in candles]
        rsi_values = rsi(closes, self.config.rsi_period)
        macd_values, macd_signal_values, macd_histogram_values = macd(
            closes,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal,
        )
        k_values, d_values, j_values = kdj(candles, self.config.kdj_period)
        return TimeframeSignal(
            timeframe=timeframe,
            close=closes[-1],
            rsi=rsi_values[-1],
            previous_rsi=rsi_values[-2],
            macd=macd_values[-1],
            macd_signal=macd_signal_values[-1],
            macd_histogram=macd_histogram_values[-1],
            previous_macd_histogram=macd_histogram_values[-2],
            k=k_values[-1],
            d=d_values[-1],
            j=j_values[-1],
            previous_k=k_values[-2],
            previous_d=d_values[-2],
        )

    def evaluate(self, direction: Direction, frames: Sequence[TimeframeSignal]) -> tuple[bool, str]:
        if not self.config.enabled:
            return True, "filter_disabled"
        if direction == Direction.FLAT:
            return True, "flat"
        if not frames:
            return False, "no_mtf_frames"

        if direction == Direction.LONG:
            return self._evaluate_long(frames)
        return self._evaluate_short(frames)

    def _evaluate_long(self, frames: Sequence[TimeframeSignal]) -> tuple[bool, str]:
        fast = frames[0]
        if not self._long_fast_timing(fast):
            return False, self._frame_reason("15m_timing_not_ready", fast)

        hostile_frames = [
            frame
            for frame in frames
            if frame.rsi < self.config.rsi_oversold and frame.macd_histogram < frame.previous_macd_histogram
        ]
        if len(hostile_frames) >= 2:
            return False, "higher_tf_still_falling"

        score = 0
        notes: list[str] = []
        for frame in frames:
            if self.config.rsi_long_floor <= frame.rsi <= self.config.rsi_long_ceiling:
                score += 1
                notes.append(f"{frame.timeframe}_rsi_ok")
            if frame.macd_histogram > frame.previous_macd_histogram:
                score += 1
                notes.append(f"{frame.timeframe}_macd_improving")
            if frame.k > frame.d and frame.k < 82.0:
                score += 1
                notes.append(f"{frame.timeframe}_kdj_up")

        if score < self.config.min_score:
            return False, f"mtf_long_score_low({score}/{self.config.min_score})"
        return True, f"mtf_long_score={score} {' '.join(notes[:4])}"

    def _evaluate_short(self, frames: Sequence[TimeframeSignal]) -> tuple[bool, str]:
        fast = frames[0]
        if not self._short_fast_timing(fast):
            return False, self._frame_reason("15m_timing_not_ready", fast)

        hostile_frames = [
            frame
            for frame in frames
            if frame.rsi > self.config.rsi_overbought and frame.macd_histogram > frame.previous_macd_histogram
        ]
        if len(hostile_frames) >= 2:
            return False, "higher_tf_still_rising"

        score = 0
        notes: list[str] = []
        for frame in frames:
            if self.config.rsi_short_floor <= frame.rsi <= self.config.rsi_short_ceiling:
                score += 1
                notes.append(f"{frame.timeframe}_rsi_ok")
            if frame.macd_histogram < frame.previous_macd_histogram:
                score += 1
                notes.append(f"{frame.timeframe}_macd_weakening")
            if frame.k < frame.d and frame.k > 18.0:
                score += 1
                notes.append(f"{frame.timeframe}_kdj_down")

        if score < self.config.min_score:
            return False, f"mtf_short_score_low({score}/{self.config.min_score})"
        return True, f"mtf_short_score={score} {' '.join(notes[:4])}"

    def _long_fast_timing(self, frame: TimeframeSignal) -> bool:
        rsi_recovering = frame.rsi > frame.previous_rsi and frame.previous_rsi <= self.config.rsi_oversold + 8.0
        macd_improving = frame.macd_histogram > frame.previous_macd_histogram
        kdj_turning_up = frame.k > frame.d and (frame.previous_k <= frame.previous_d or frame.k < 70.0)
        not_overheated = frame.rsi <= self.config.rsi_long_ceiling
        return not_overheated and (rsi_recovering or macd_improving or kdj_turning_up)

    def _short_fast_timing(self, frame: TimeframeSignal) -> bool:
        rsi_falling = frame.rsi < frame.previous_rsi and frame.previous_rsi >= self.config.rsi_overbought - 8.0
        macd_weakening = frame.macd_histogram < frame.previous_macd_histogram
        kdj_turning_down = frame.k < frame.d and (frame.previous_k >= frame.previous_d or frame.k > 30.0)
        not_oversold = frame.rsi >= self.config.rsi_short_floor
        return not_oversold and (rsi_falling or macd_weakening or kdj_turning_down)

    @staticmethod
    def _frame_reason(prefix: str, frame: TimeframeSignal) -> str:
        return (
            f"{prefix}("
            f"{frame.timeframe} rsi={frame.rsi:.1f} "
            f"macd_hist={frame.macd_histogram:.6g} "
            f"kdj={frame.k:.1f}/{frame.d:.1f})"
        )

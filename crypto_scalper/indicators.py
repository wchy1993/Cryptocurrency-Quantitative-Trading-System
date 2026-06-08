from __future__ import annotations

from collections.abc import Sequence

from .models import Candle


def ema(values: Sequence[float], period: int) -> list[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if not values:
        return []

    alpha = 2.0 / (period + 1.0)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def atr(candles: Sequence[Candle], period: int) -> list[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if not candles:
        return []

    true_ranges: list[float] = []
    previous_close = candles[0].close
    for candle in candles:
        true_range = max(
            candle.high - candle.low,
            abs(candle.high - previous_close),
            abs(candle.low - previous_close),
        )
        true_ranges.append(true_range)
        previous_close = candle.close

    result = [true_ranges[0]]
    alpha = 1.0 / period
    for value in true_ranges[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def rolling_high(values: Sequence[float], end_index: int, length: int) -> float:
    if length <= 0:
        raise ValueError("length must be positive")
    start = max(0, end_index - length)
    window = values[start:end_index]
    if not window:
        raise ValueError("empty rolling window")
    return max(window)


def rolling_low(values: Sequence[float], end_index: int, length: int) -> float:
    if length <= 0:
        raise ValueError("length must be positive")
    start = max(0, end_index - length)
    window = values[start:end_index]
    if not window:
        raise ValueError("empty rolling window")
    return min(window)


def rsi(values: Sequence[float], period: int = 14) -> list[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if not values:
        return []
    if len(values) == 1:
        return [50.0]

    result = [50.0] * len(values)
    gains: list[float] = []
    losses: list[float] = []
    for index in range(1, min(len(values), period + 1)):
        change = float(values[index]) - float(values[index - 1])
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    if len(gains) < period:
        return result

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    result[period] = _rsi_from_averages(avg_gain, avg_loss)

    for index in range(period + 1, len(values)):
        change = float(values[index]) - float(values[index - 1])
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        result[index] = _rsi_from_averages(avg_gain, avg_loss)
    return result


def macd(values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[list[float], list[float], list[float]]:
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("MACD periods must be positive")
    if fast >= slow:
        raise ValueError("MACD fast period must be smaller than slow period")
    if not values:
        return [], [], []
    fast_line = ema(values, fast)
    slow_line = ema(values, slow)
    dif = [fast_value - slow_value for fast_value, slow_value in zip(fast_line, slow_line)]
    dea = ema(dif, signal)
    histogram = [dif_value - dea_value for dif_value, dea_value in zip(dif, dea)]
    return dif, dea, histogram


def kdj(candles: Sequence[Candle], period: int = 9) -> tuple[list[float], list[float], list[float]]:
    if period <= 0:
        raise ValueError("period must be positive")
    if not candles:
        return [], [], []

    k_values: list[float] = []
    d_values: list[float] = []
    j_values: list[float] = []
    k = 50.0
    d = 50.0
    highs = [candle.high for candle in candles]
    lows = [candle.low for candle in candles]
    for index, candle in enumerate(candles):
        start = max(0, index - period + 1)
        highest = max(highs[start : index + 1])
        lowest = min(lows[start : index + 1])
        if highest <= lowest:
            rsv = 50.0
        else:
            rsv = (candle.close - lowest) / (highest - lowest) * 100.0
        k = k * 2.0 / 3.0 + rsv / 3.0
        d = d * 2.0 / 3.0 + k / 3.0
        j = 3.0 * k - 2.0 * d
        k_values.append(k)
        d_values.append(d)
        j_values.append(j)
    return k_values, d_values, j_values


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss <= 0:
        return 100.0 if avg_gain > 0 else 50.0
    relative_strength = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)

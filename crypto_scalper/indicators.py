from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

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


def adx(candles: Sequence[Candle], period: int = 14) -> list[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if not candles:
        return []
    if len(candles) == 1:
        return [0.0]

    true_ranges = [0.0]
    plus_dm = [0.0]
    minus_dm = [0.0]
    previous = candles[0]
    for candle in candles[1:]:
        up_move = candle.high - previous.high
        down_move = previous.low - candle.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        true_ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous.close),
                abs(candle.low - previous.close),
            )
        )
        previous = candle

    smoothed_tr = _wilder_smooth(true_ranges, period)
    smoothed_plus = _wilder_smooth(plus_dm, period)
    smoothed_minus = _wilder_smooth(minus_dm, period)
    dx_values: list[float] = []
    for tr_value, plus_value, minus_value in zip(smoothed_tr, smoothed_plus, smoothed_minus):
        if tr_value <= 0:
            dx_values.append(0.0)
            continue
        plus_di = 100.0 * plus_value / tr_value
        minus_di = 100.0 * minus_value / tr_value
        denominator = plus_di + minus_di
        dx_values.append(0.0 if denominator <= 0 else 100.0 * abs(plus_di - minus_di) / denominator)
    return _wilder_smooth(dx_values, period)


def bollinger_bands(values: Sequence[float], period: int = 20, stddev: float = 2.0) -> tuple[list[float], list[float], list[float], list[float]]:
    if period <= 0:
        raise ValueError("period must be positive")
    if stddev <= 0:
        raise ValueError("stddev must be positive")
    if not values:
        return [], [], [], []

    mids: list[float] = []
    uppers: list[float] = []
    lowers: list[float] = []
    widths: list[float] = []
    for index, value in enumerate(values):
        start = max(0, index - period + 1)
        window = [float(candidate) for candidate in values[start : index + 1]]
        mid = sum(window) / len(window)
        variance = sum((candidate - mid) ** 2 for candidate in window) / len(window)
        band_width = sqrt(variance) * stddev
        upper = mid + band_width
        lower = mid - band_width
        mids.append(mid)
        uppers.append(upper)
        lowers.append(lower)
        widths.append(0.0 if mid <= 0 else (upper - lower) / mid)
    return mids, uppers, lowers, widths


def vwap(candles: Sequence[Candle], period: int = 20) -> list[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if not candles:
        return []

    result: list[float] = []
    typical_volume_sum = 0.0
    volume_sum = 0.0
    typical_volumes: list[float] = []
    volumes: list[float] = []
    for index, candle in enumerate(candles):
        typical_price = (candle.high + candle.low + candle.close) / 3.0
        typical_volume = typical_price * candle.volume
        typical_volumes.append(typical_volume)
        volumes.append(candle.volume)
        typical_volume_sum += typical_volume
        volume_sum += candle.volume
        if index >= period:
            typical_volume_sum -= typical_volumes[index - period]
            volume_sum -= volumes[index - period]
        result.append(candle.close if volume_sum <= 0 else typical_volume_sum / volume_sum)
    return result


def percentile_rank(values: Sequence[float], lookback: int = 100) -> float:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if not values:
        return 0.0
    window = [float(value) for value in values[-lookback:]]
    if len(window) <= 1:
        return 50.0
    current = window[-1]
    below = sum(1 for value in window if value < current)
    equal = sum(1 for value in window if value == current)
    return (below + 0.5 * equal) / len(window) * 100.0


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


def _wilder_smooth(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    result: list[float] = []
    smoothed = float(values[0])
    alpha = 1.0 / period
    for value in values:
        smoothed = alpha * float(value) + (1.0 - alpha) * smoothed
        result.append(smoothed)
    return result

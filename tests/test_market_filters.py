from datetime import datetime, timedelta
import unittest

from crypto_scalper.indicators import adx, bollinger_bands, kdj, macd, rsi, vwap
from crypto_scalper.live_config import MultiTimeframeFilterConfig
from crypto_scalper.market_filters import MultiTimeframeFilter, TimeframeSignal
from crypto_scalper.models import Candle, Direction


class MarketFilterTests(unittest.TestCase):
    def test_indicators_return_series(self) -> None:
        candles = _trend_candles(80, start=100.0, step=0.2)
        closes = [candle.close for candle in candles]
        self.assertEqual(len(rsi(closes, 14)), len(candles))
        macd_line, signal_line, histogram = macd(closes)
        self.assertEqual(len(macd_line), len(candles))
        self.assertEqual(len(signal_line), len(candles))
        self.assertEqual(len(histogram), len(candles))
        k_values, d_values, j_values = kdj(candles)
        self.assertEqual(len(k_values), len(candles))
        self.assertEqual(len(d_values), len(candles))
        self.assertEqual(len(j_values), len(candles))
        self.assertEqual(len(adx(candles)), len(candles))
        mid, upper, lower, widths = bollinger_bands(closes)
        self.assertEqual(len(mid), len(candles))
        self.assertEqual(len(upper), len(candles))
        self.assertEqual(len(lower), len(candles))
        self.assertEqual(len(widths), len(candles))
        self.assertEqual(len(vwap(candles)), len(candles))

    def test_long_filter_rejects_falling_higher_timeframes(self) -> None:
        config = MultiTimeframeFilterConfig(min_score=5)
        market_filter = MultiTimeframeFilter(config)
        fast = _frame("15m", rsi_value=45.0, previous_rsi=34.0, hist=0.2, previous_hist=0.1, k=55.0, d=45.0)
        falling_1h = _frame("1h", rsi_value=28.0, previous_rsi=31.0, hist=-0.3, previous_hist=-0.2, k=35.0, d=45.0)
        falling_4h = _frame("4h", rsi_value=26.0, previous_rsi=29.0, hist=-0.4, previous_hist=-0.3, k=30.0, d=42.0)
        allowed, reason = market_filter.evaluate(Direction.LONG, [fast, falling_1h, falling_4h])
        self.assertFalse(allowed)
        self.assertIn("falling", reason)

    def test_short_filter_rejects_rising_higher_timeframes(self) -> None:
        config = MultiTimeframeFilterConfig(min_score=5)
        market_filter = MultiTimeframeFilter(config)
        fast = _frame("15m", rsi_value=55.0, previous_rsi=66.0, hist=-0.2, previous_hist=-0.1, k=45.0, d=55.0)
        rising_1h = _frame("1h", rsi_value=72.0, previous_rsi=68.0, hist=0.3, previous_hist=0.2, k=70.0, d=58.0)
        rising_4h = _frame("4h", rsi_value=74.0, previous_rsi=70.0, hist=0.4, previous_hist=0.3, k=75.0, d=60.0)
        allowed, reason = market_filter.evaluate(Direction.SHORT, [fast, rising_1h, rising_4h])
        self.assertFalse(allowed)
        self.assertIn("rising", reason)


def _trend_candles(count: int, start: float, step: float) -> list[Candle]:
    candles: list[Candle] = []
    timestamp = datetime(2025, 1, 1)
    price = start
    for index in range(count):
        close = max(1.0, price + step)
        high = max(price, close) + 0.3
        low = min(price, close) - 0.3
        candles.append(Candle(timestamp + timedelta(minutes=index), price, high, low, close, 1000.0))
        price = close
    return candles


def _reversal_candles() -> list[Candle]:
    candles = _trend_candles(70, start=110.0, step=-0.12)
    timestamp = candles[-1].timestamp
    price = candles[-1].close
    for index in range(20):
        close = price + 0.18
        candles.append(Candle(timestamp + timedelta(minutes=index + 1), price, close + 0.25, price - 0.1, close, 1200.0))
        price = close
    return candles


def _upper_reversal_candles() -> list[Candle]:
    candles = _trend_candles(70, start=90.0, step=0.12)
    timestamp = candles[-1].timestamp
    price = candles[-1].close
    for index in range(20):
        close = price - 0.18
        candles.append(Candle(timestamp + timedelta(minutes=index + 1), price, price + 0.1, close - 0.25, close, 1200.0))
        price = close
    return candles


def _frame(
    timeframe: str,
    rsi_value: float,
    previous_rsi: float,
    hist: float,
    previous_hist: float,
    k: float,
    d: float,
) -> TimeframeSignal:
    return TimeframeSignal(
        timeframe=timeframe,
        close=100.0,
        rsi=rsi_value,
        previous_rsi=previous_rsi,
        macd=hist,
        macd_signal=0.0,
        macd_histogram=hist,
        previous_macd_histogram=previous_hist,
        k=k,
        d=d,
        j=3.0 * k - 2.0 * d,
        previous_k=d,
        previous_d=k,
    )


if __name__ == "__main__":
    unittest.main()

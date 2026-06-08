from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from crypto_scalper.data import write_candles_csv
from crypto_scalper.live_config import default_live_config
from crypto_scalper.live_portfolio_backtest import (
    HistoricalClient,
    _align_candles_by_common_timestamps,
    _entry_scan_cycles_for_bar,
    _load_symbol_data,
    _resample_factor,
    _starting_capital_drawdown_stop_hit,
    _weekly_profit_drawdown_stop_hit,
)
from crypto_scalper.models import Candle


class LivePortfolioBacktestTests(unittest.TestCase):
    def test_historical_client_resamples_15m_to_30m_and_1h(self) -> None:
        candles = _candles(12)
        client = HistoricalClient({"BTCUSDT": candles}, "15m", ("15m", "30m", "1h"))

        self.assertEqual(_resample_factor("15m", "30m"), 2)
        self.assertEqual(_resample_factor("15m", "1h"), 4)
        self.assertEqual(len(client.resampled["30m"]["BTCUSDT"]), 6)
        self.assertEqual(len(client.resampled["1h"]["BTCUSDT"]), 3)

        client.current_index = 7
        thirty_minute = client.klines("BTCUSDT", "30m", 10)
        one_hour = client.klines("BTCUSDT", "1h", 10)

        self.assertEqual(len(thirty_minute), 4)
        self.assertEqual(thirty_minute[-1].close, candles[7].close)
        self.assertEqual(len(one_hour), 2)
        self.assertEqual(one_hour[-1].close, candles[7].close)

    def test_load_symbol_data_uses_configured_base_timeframe(self) -> None:
        candles = _candles(8)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_candles_csv(root / "BTCUSDT_15m_20250101_20250102.csv", candles)
            write_candles_csv(root / "ETHUSDT_1h_20250101_20250102.csv", candles)

            loaded = _load_symbol_data(temp_dir, ("BTCUSDT", "ETHUSDT"), "15m")

        self.assertEqual(set(loaded), {"BTCUSDT"})
        self.assertEqual(len(loaded["BTCUSDT"]), len(candles))

    def test_portfolio_data_aligns_by_common_timestamps(self) -> None:
        btc = _candles(6)
        eth = _candles(4, start=datetime(2025, 1, 1, 0, 30))

        aligned = _align_candles_by_common_timestamps({"BTCUSDT": btc, "ETHUSDT": eth})

        self.assertEqual(len(aligned["BTCUSDT"]), 4)
        self.assertEqual(len(aligned["ETHUSDT"]), 4)
        self.assertEqual(aligned["BTCUSDT"][0].timestamp, datetime(2025, 1, 1, 0, 30))
        self.assertEqual(aligned["ETHUSDT"][0].timestamp, datetime(2025, 1, 1, 0, 30))
        self.assertEqual(aligned["BTCUSDT"][-1].timestamp, aligned["ETHUSDT"][-1].timestamp)

    def test_entry_scan_cycles_follow_scan_seconds(self) -> None:
        bar_seconds = 30 * 60

        self.assertEqual(_scan_cycles(2, bar_seconds, 300), [6, 6])
        self.assertEqual(_scan_cycles(5, bar_seconds, 1000), [2, 2, 2, 2, 1])
        self.assertEqual(_scan_cycles(3, bar_seconds, 1800), [1, 1, 1])
        self.assertEqual(_scan_cycles(4, bar_seconds, 3600), [1, 0, 1, 0])

    def test_starting_capital_drawdown_stop_uses_principal_not_peak(self) -> None:
        config = default_live_config()
        self.assertFalse(_starting_capital_drawdown_stop_hit(config, 10_000.0, 8_100.0))
        self.assertTrue(_starting_capital_drawdown_stop_hit(config, 10_000.0, 8_000.0))

    def test_weekly_profit_drawdown_stop_resets_to_weekly_peak(self) -> None:
        config = default_live_config()
        self.assertFalse(_weekly_profit_drawdown_stop_hit(config, 10_000.0, 9_800.0, 10_000.0, 8_600.0))
        self.assertFalse(_weekly_profit_drawdown_stop_hit(config, 10_000.0, 10_500.0, 12_000.0, 10_300.0))
        self.assertTrue(_weekly_profit_drawdown_stop_hit(config, 10_000.0, 10_500.0, 12_000.0, 10_200.0))


def _candles(count: int, start: datetime | None = None) -> list[Candle]:
    start = start or datetime(2025, 1, 1)
    output: list[Candle] = []
    for index in range(count):
        open_price = 100.0 + index
        close_price = open_price + 0.5
        output.append(
            Candle(
                start + timedelta(minutes=15 * index),
                open_price,
                close_price + 0.2,
                open_price - 0.2,
                close_price,
                1000.0 + index,
            )
        )
    return output


def _scan_cycles(bars: int, bar_seconds: int, scan_seconds: int) -> list[int]:
    next_scan = 0.0
    output: list[int] = []
    for bar in range(bars):
        cycles, next_scan = _entry_scan_cycles_for_bar(
            bar * bar_seconds,
            (bar + 1) * bar_seconds,
            next_scan,
            scan_seconds,
        )
        output.append(cycles)
    return output


if __name__ == "__main__":
    unittest.main()

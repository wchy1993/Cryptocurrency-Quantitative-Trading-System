from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from crypto_scalper.backtest import Backtester
from crypto_scalper.config import RiskConfig, StrategyConfig
from crypto_scalper.data import generate_sample_candles, load_candles_csv, write_candles_csv
from crypto_scalper.models import Candle, Direction
from crypto_scalper.strategy import VolatilityBreakoutScalper


class BacktestSmokeTests(unittest.TestCase):
    def test_sample_data_round_trip(self) -> None:
        candles = generate_sample_candles(bars=50, seed=7)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            write_candles_csv(path, candles)
            loaded = load_candles_csv(path)
        self.assertEqual(len(loaded), 50)
        self.assertEqual(loaded[0].timestamp, candles[0].timestamp)
        self.assertGreater(loaded[-1].close, 0)

    def test_backtest_runs_and_reports_summary(self) -> None:
        candles = generate_sample_candles(bars=600, seed=11)
        strategy = VolatilityBreakoutScalper(
            StrategyConfig(
                min_atr_pct=0.0001,
                channel_period=10,
                fast_ema=5,
                slow_ema=13,
                breakeven_atr=1.0,
                trailing_stop_atr=1.2,
                max_holding_bars=60,
            )
        )
        result = Backtester(candles, strategy, RiskConfig(initial_equity=1_000.0)).run()
        self.assertIn("final_equity", result.summary)
        self.assertIn("win_rate_pct", result.summary)
        self.assertGreater(result.summary["final_equity"], 0)
        self.assertGreater(len(result.equity_curve), 0)

    def test_lower_spike_reversal_signal_uses_reduced_risk(self) -> None:
        start = datetime(2025, 1, 1)
        candles = [
            Candle(start + timedelta(minutes=index), 100.0, 100.2, 99.8, 100.1, 100.0)
            for index in range(30)
        ]
        candles.append(Candle(start + timedelta(minutes=30), 100.0, 100.3, 96.0, 100.2, 240.0))
        strategy = VolatilityBreakoutScalper(
            StrategyConfig(
                fast_ema=3,
                slow_ema=5,
                atr_period=5,
                channel_period=5,
                volume_period=5,
                min_atr_pct=0.0001,
                spike_trade_enabled=True,
                spike_guard_enabled=True,
                spike_risk_multiplier=0.35,
            )
        )
        strategy.prepare(candles)
        signal = strategy.signal(len(candles) - 1, candles)
        self.assertEqual(signal.direction, Direction.LONG)
        self.assertEqual(signal.reason, "lower_spike_reversal")
        self.assertLess(signal.risk_multiplier, 1.0)


if __name__ == "__main__":
    unittest.main()

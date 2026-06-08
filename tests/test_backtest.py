from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from crypto_scalper.backtest import Backtester
from crypto_scalper.config import RiskConfig, StrategyConfig
from crypto_scalper.data import generate_sample_candles, interval_to_milliseconds, load_candles_csv, write_candles_csv
from crypto_scalper.models import Candle, Direction, Position
from crypto_scalper.risk import RiskManager, signal_risk_weight
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
        trade_net = sum(trade.net_pnl for trade in result.trades)
        self.assertAlmostEqual(result.summary["final_equity"], 1_000.0 + trade_net)

    def test_liquidation_price_uses_position_effective_leverage(self) -> None:
        candles = generate_sample_candles(bars=20, seed=3)
        backtester = Backtester(
            candles,
            VolatilityBreakoutScalper(StrategyConfig()),
            RiskConfig(initial_equity=1_000.0, max_leverage=30.0, maintenance_margin_pct=0.005),
        )
        position = Position(
            direction=Direction.LONG,
            qty=1.0,
            entry_price=100.0,
            entry_time=candles[0].timestamp,
            stop_price=95.0,
            take_profit_price=110.0,
            entry_fee=0.0,
            peak_price=100.0,
            trough_price=100.0,
            leverage=2.0,
        )
        self.assertAlmostEqual(backtester._liquidation_price(position), 50.5)

    def test_interval_to_milliseconds_supports_intraday_filters(self) -> None:
        self.assertEqual(interval_to_milliseconds("15m"), 15 * 60_000)
        self.assertEqual(interval_to_milliseconds("30m"), 30 * 60_000)
        self.assertEqual(interval_to_milliseconds("1h"), 60 * 60_000)
        self.assertEqual(interval_to_milliseconds("4h"), 4 * 60 * 60_000)
        self.assertEqual(interval_to_milliseconds("8h"), 8 * 60 * 60_000)

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
        self.assertIn("lower_spike_reversal", signal.reason)
        self.assertIn("score=", signal.reason)
        self.assertLess(signal.risk_multiplier, 1.0)

    def test_position_size_uses_signal_score_and_drawdown_brake(self) -> None:
        risk = RiskManager(RiskConfig(initial_equity=1_000.0, max_drawdown_pct=0.10, risk_per_trade_pct=0.01))
        weak = risk.size_position(
            1_000.0,
            100.0,
            signal=type("SignalLike", (), {"confidence": 0.25, "risk_multiplier": 0.25, "stop_loss_pct": 0.02})(),
        )[0]
        strong = risk.size_position(
            1_000.0,
            100.0,
            signal=type("SignalLike", (), {"confidence": 1.0, "risk_multiplier": 1.0, "stop_loss_pct": 0.02})(),
        )[0]
        self.assertGreater(strong, weak)

        risk.peak_equity = 1_000.0
        normal_multiplier = risk.drawdown_size_multiplier(1_000.0)
        stressed_multiplier = risk.drawdown_size_multiplier(910.0)
        self.assertEqual(normal_multiplier, 1.0)
        self.assertLess(stressed_multiplier, normal_multiplier)

    def test_high_confidence_signal_gets_non_linear_risk_boost(self) -> None:
        weak = signal_risk_weight(0.35, 0.35)
        normal = signal_risk_weight(0.65, 0.65)
        strong = signal_risk_weight(1.0, 1.0)

        self.assertLess(weak, 0.35)
        self.assertGreater(normal, weak)
        self.assertGreater(strong, 1.0)
        self.assertLessEqual(strong, 1.8)

    def test_super_volume_breakout_boosts_signal_size_and_target(self) -> None:
        candles = _super_volume_breakout_candles()
        base_config = StrategyConfig(
            fast_ema=3,
            slow_ema=5,
            atr_period=5,
            channel_period=5,
            volume_period=5,
            min_atr_pct=0.0,
            max_atr_pct=0.2,
            min_volume_ratio=1.0,
            breakout_buffer_atr=0.0,
            ema_gap_atr=0.0,
            stop_loss_atr=1.0,
            take_profit_atr=1.2,
            long_score_threshold=0.25,
            super_volume_min_ratio=2.0,
            super_volume_min_breakout_atr=0.2,
            super_volume_min_body_atr=0.1,
            super_volume_max_holding_bars=6,
        )
        disabled_strategy = VolatilityBreakoutScalper(
            StrategyConfig(**{**base_config.__dict__, "super_volume_breakout_enabled": False})
        )
        enabled_strategy = VolatilityBreakoutScalper(base_config)
        disabled_strategy.prepare(candles)
        enabled_strategy.prepare(candles)

        normal = disabled_strategy.signal(len(candles) - 1, candles)
        boosted = enabled_strategy.signal(len(candles) - 1, candles)

        self.assertEqual(normal.direction, Direction.LONG)
        self.assertEqual(boosted.direction, Direction.LONG)
        self.assertIn("super_volume", boosted.reason)
        self.assertGreater(boosted.confidence, normal.confidence)
        self.assertGreater(boosted.risk_multiplier, normal.risk_multiplier)
        self.assertGreater(boosted.take_profit_pct, normal.take_profit_pct)
        self.assertEqual(boosted.max_holding_bars, 6)

    def test_rsi_reversal_signal_can_be_disabled(self) -> None:
        candles = _rsi_reversal_candles()
        config = StrategyConfig(
            fast_ema=3,
            slow_ema=5,
            atr_period=5,
            channel_period=5,
            volume_period=5,
            min_atr_pct=0.0,
            max_atr_pct=0.0,
            min_volume_ratio=0.0,
            breakout_buffer_atr=0.0,
            ema_gap_atr=0.0,
            spike_guard_enabled=False,
            spike_trade_enabled=False,
            allow_short=True,
            short_score_threshold=0.25,
        )
        enabled_strategy = VolatilityBreakoutScalper(config)
        enabled_strategy.prepare(candles)
        _force_short_rsi_reversal_context(enabled_strategy, len(candles) - 1)

        enabled = enabled_strategy.signal(len(candles) - 1, candles)

        disabled_strategy = VolatilityBreakoutScalper(
            StrategyConfig(**{**config.__dict__, "rsi_reversal_enabled": False})
        )
        disabled_strategy.prepare(candles)
        _force_short_rsi_reversal_context(disabled_strategy, len(candles) - 1)

        disabled = disabled_strategy.signal(len(candles) - 1, candles)

        self.assertEqual(enabled.direction, Direction.SHORT)
        self.assertIn("short_rsi_reversal", enabled.reason)
        self.assertEqual(disabled.direction, Direction.FLAT)
        self.assertNotIn("rsi_reversal", disabled.reason)


if __name__ == "__main__":
    unittest.main()


def _super_volume_breakout_candles() -> list[Candle]:
    start = datetime(2025, 1, 1)
    candles: list[Candle] = []
    price = 100.0
    for index in range(20):
        open_price = price
        close_price = price + 0.08
        candles.append(Candle(start + timedelta(minutes=15 * index), open_price, close_price + 0.15, open_price - 0.15, close_price, 1000.0))
        price = close_price
    candles.append(Candle(start + timedelta(minutes=15 * 20), price, price + 3.3, price - 0.1, price + 3.0, 6000.0))
    return candles


def _rsi_reversal_candles() -> list[Candle]:
    start = datetime(2025, 1, 1)
    candles = [
        Candle(start + timedelta(minutes=15 * index), 100.0, 100.3, 99.7, 100.0, 100.0)
        for index in range(11)
    ]
    candles.append(Candle(start + timedelta(minutes=15 * 11), 100.2, 100.3, 99.8, 100.0, 100.0))
    return candles


def _force_short_rsi_reversal_context(strategy: VolatilityBreakoutScalper, index: int) -> None:
    strategy._atr[index - 1] = 1.0
    strategy._fast[index] = 100.0
    strategy._slow[index] = 100.0
    strategy._avg_volume[index - 1] = 100.0
    strategy._rsi[index - 1] = 75.0
    strategy._rsi[index] = 65.0

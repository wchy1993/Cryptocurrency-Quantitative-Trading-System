from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime
from datetime import timedelta

from crypto_scalper.config import StrategyConfig
from crypto_scalper.live_config import default_live_config
from crypto_scalper.live_execution_backtest import _mtf_exit_reason_for_position
from crypto_scalper.live_portfolio_backtest import PortfolioPosition
from crypto_scalper.models import Candle
from crypto_scalper.models import Direction
from crypto_scalper.mtf_4h_rsi_regime import (
    MTF_REASON_TOKEN,
    Mtf4hRsiRegimePullbackStrategy,
    MtfRegimeResult,
    MtfSetupResult,
    MtfTriggerResult,
    available_candle_end,
    build_oi_features,
    closed_candles_for_decision,
    mtf_report_from_summary,
)


class Mtf4hRsiRegimeTests(unittest.TestCase):
    def test_4h_candle_unclosed_is_not_available_for_regime(self) -> None:
        timestamps = [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 1, 4, 0)]

        self.assertEqual(available_candle_end(timestamps, datetime(2026, 1, 1, 7, 59), "4h"), 1)
        self.assertEqual(available_candle_end(timestamps, datetime(2026, 1, 1, 8, 0), "4h"), 2)

    def test_feature_mapping_to_1m_has_no_future_fill(self) -> None:
        start = datetime(2026, 1, 1, 10, 0)
        candles = [_candle(start + timedelta(minutes=15 * i), 100 + i) for i in range(3)]
        timestamps = [item.timestamp for item in candles]

        self.assertEqual(closed_candles_for_decision(candles, timestamps, start + timedelta(minutes=29), "15m", 10)[-1].timestamp, start)
        self.assertEqual(closed_candles_for_decision(candles, timestamps, start + timedelta(minutes=30), "15m", 10)[-1].timestamp, start + timedelta(minutes=15))

    def test_15m_trigger_fills_next_1m_open_not_same_bar(self) -> None:
        signal_time = datetime(2026, 1, 1, 10, 15)
        signal_available = signal_time + timedelta(minutes=15)

        self.assertEqual(signal_available, datetime(2026, 1, 1, 10, 30))

    def test_long_bias_conditions(self) -> None:
        strategy = _strategy(mtf_4h_require_macd_hist_turn=False, mtf_4h_max_distance_from_ema21_pct=1.0)
        candles = _trend_candles(datetime(2026, 1, 1), "4h", [100 - i * 1.0 for i in range(24)] + [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88])

        regime = strategy.regime_4h(candles)

        self.assertEqual(regime.regime, "LONG_BIAS")

    def test_short_bias_conditions(self) -> None:
        strategy = _strategy(mtf_4h_require_macd_hist_turn=False, mtf_4h_max_distance_from_ema21_pct=1.0)
        candles = _trend_candles(datetime(2026, 1, 1), "4h", [100 + i * 1.0 for i in range(24)] + [123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112])

        regime = strategy.regime_4h(candles)

        self.assertEqual(regime.regime, "SHORT_BIAS")

    def test_long_sweep_reclaim_trigger(self) -> None:
        strategy = _strategy()
        setup = MtfSetupResult(Direction.LONG, 100.0, 102.0, 100.0, 100.0, 99.5, 102.0, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 100.2)
        candles.append(Candle(candles[-1].timestamp + timedelta(minutes=15), 99.8, 100.6, 99.4, 100.5, 1))

        trigger, reason = strategy.trigger_15m(Direction.LONG, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "sweep")
        self.assertEqual(reason, "15m_long_sweep")

    def test_5m_volume_filter_rejects_low_volume_trigger(self) -> None:
        strategy = _strategy(
            mtf_trigger_timeframe="5m",
            mtf_trigger_volume_filter_enabled=True,
            mtf_trigger_min_volume_ratio=1.2,
            mtf_trigger_volume_period=5,
        )
        setup = MtfSetupResult(Direction.LONG, 100.0, 102.0, 100.0, 100.0, 99.5, 102.0, 50.0)
        candles = _flat_tf(datetime(2026, 1, 1), 24, 5, 100.2, 100.0)
        candles.append(Candle(candles[-1].timestamp + timedelta(minutes=5), 99.8, 100.6, 99.4, 100.5, 80.0))

        trigger, reason = strategy.trigger_timeframe(Direction.LONG, candles, setup, "5m")

        self.assertIsNone(trigger)
        self.assertEqual(reason, "5m_volume_too_low")

    def test_5m_volume_sweep_trigger(self) -> None:
        strategy = _strategy(
            mtf_trigger_timeframe="5m",
            mtf_trigger_volume_filter_enabled=True,
            mtf_trigger_min_volume_ratio=1.2,
            mtf_trigger_volume_period=5,
        )
        setup = MtfSetupResult(Direction.LONG, 100.0, 102.0, 100.0, 100.0, 99.5, 102.0, 50.0)
        candles = _flat_tf(datetime(2026, 1, 1), 24, 5, 100.2, 100.0)
        candles.append(Candle(candles[-1].timestamp + timedelta(minutes=5), 99.8, 100.6, 99.4, 100.5, 180.0))

        trigger, reason = strategy.trigger_timeframe(Direction.LONG, candles, setup, "5m")

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "sweep")
        self.assertEqual(reason, "5m_long_sweep")
        self.assertGreater(trigger.volume_ratio, 1.2)

    def test_short_sweep_reject_trigger(self) -> None:
        strategy = _strategy()
        setup = MtfSetupResult(Direction.SHORT, 98.0, 100.0, 100.0, 100.0, 98.0, 100.5, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 99.8)
        candles.append(Candle(candles[-1].timestamp + timedelta(minutes=15), 100.2, 100.7, 99.4, 99.5, 1))

        trigger, reason = strategy.trigger_15m(Direction.SHORT, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "sweep")
        self.assertEqual(reason, "15m_short_sweep")

    def test_long_structure_break_trigger(self) -> None:
        strategy = _strategy(mtf_15m_trigger_mode="structure_break")
        setup = MtfSetupResult(Direction.LONG, 100.0, 102.0, 100.0, 100.0, 99.5, 102.0, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 100.0)
        candles.extend([
            Candle(candles[-1].timestamp + timedelta(minutes=15), 100.0, 100.4, 99.7, 100.1, 1),
            Candle(candles[-1].timestamp + timedelta(minutes=30), 100.2, 100.8, 99.8, 100.7, 1),
        ])

        trigger, _reason = strategy.trigger_15m(Direction.LONG, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "structure_break")

    def test_short_structure_break_trigger(self) -> None:
        strategy = _strategy(mtf_15m_trigger_mode="structure_break")
        setup = MtfSetupResult(Direction.SHORT, 98.0, 100.0, 100.0, 100.0, 98.0, 100.5, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 100.0)
        candles.extend([
            Candle(candles[-1].timestamp + timedelta(minutes=15), 100.1, 100.3, 99.7, 99.9, 1),
            Candle(candles[-1].timestamp + timedelta(minutes=30), 99.8, 100.0, 99.0, 99.1, 1),
        ])

        trigger, _reason = strategy.trigger_15m(Direction.SHORT, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "structure_break")

    def test_stop_too_wide_rejects_trade(self) -> None:
        config = _config(mtf_max_stop_pct=0.001, mtf_use_oi_filter=False, mtf_use_funding_filter=False)
        strategy = Mtf4hRsiRegimePullbackStrategy(config)
        strategy.regime = lambda *_args: MtfRegimeResult("LONG_BIAS", 42.0, "test")  # type: ignore[method-assign]
        strategy.confirm_1h = lambda *_args: (True, "ok", 55.0, 100.0)  # type: ignore[method-assign]
        strategy.setup_30m = lambda *_args: (MtfSetupResult(Direction.LONG, 98.0, 101.0, 100.0, 100.0, 98.0, 101.0, 50.0), "ok")  # type: ignore[method-assign]
        strategy.trigger_timeframe = lambda *_args: (MtfTriggerResult(Direction.LONG, "sweep", Candle(datetime(2026, 1, 1), 100.0, 101.0, 99.0, 100.5, 1), 1.0), "ok")  # type: ignore[method-assign]

        decision = strategy.build_signal("BTCUSDT", _flat_15m(datetime(2026, 1, 1), 30, 100), _flat_15m(datetime(2026, 1, 1), 30, 100), _flat_15m(datetime(2026, 1, 1), 30, 100), _flat_15m(datetime(2026, 1, 1), 40, 100), _flat_15m(datetime(2026, 1, 1), 30, 100), _flat_15m(datetime(2026, 1, 1), 30, 100), 0.0, 0.0)

        self.assertIsNone(decision.signal)
        self.assertEqual(decision.reject_reason, "mtf_stop_too_wide")

    def test_funding_filter_direction(self) -> None:
        strategy = _strategy(mtf_long_min_funding_rate=0.0, mtf_short_max_funding_rate=0.0)

        self.assertEqual(strategy._futures_filter_reject_reason(Direction.LONG, 0.0, -0.0001, 0.0, 0.0), "mtf_rejected_funding")
        self.assertEqual(strategy._futures_filter_reject_reason(Direction.LONG, 0.0, 0.0003, 0.0, 0.0), "mtf_rejected_funding")
        self.assertEqual(strategy._futures_filter_reject_reason(Direction.SHORT, 0.0, -0.0003, 0.0, 0.0), "mtf_rejected_funding")
        self.assertEqual(strategy._futures_filter_reject_reason(Direction.SHORT, 0.0, 0.0001, 0.0, 0.0), "mtf_rejected_funding")

    def test_oi_filter(self) -> None:
        strategy = _strategy()

        self.assertEqual(strategy._futures_filter_reject_reason(Direction.LONG, None, 0.0, 0.0, 0.0), "mtf_rejected_oi_missing")
        self.assertEqual(strategy._futures_filter_reject_reason(Direction.LONG, 0.03, 0.0, 0.0, 0.0), "mtf_rejected_oi")

    def test_btc_opposite_filter(self) -> None:
        strategy = _strategy()

        self.assertEqual(strategy._futures_filter_reject_reason(Direction.LONG, 0.0, 0.0, 0.0, -0.02), "mtf_rejected_btc")
        self.assertEqual(strategy._futures_filter_reject_reason(Direction.SHORT, 0.0, 0.0, 0.0, 0.02), "mtf_rejected_btc")

    def test_fail_fast_exit(self) -> None:
        config = _config()
        position = _position(entry_time=datetime(2026, 1, 1, 0, 0), mfe=0.1)
        candle = Candle(datetime(2026, 1, 1, 2, 0), 100, 100.2, 99.8, 100.0, 1)

        self.assertEqual(_mtf_exit_reason_for_position(config, position, candle, {}, {}), "mtf_fail_fast")

    def test_time_stop_exit(self) -> None:
        config = _config(mtf_fail_fast_min_r=0.0, mtf_max_holding_minutes=60)
        position = _position(entry_time=datetime(2026, 1, 1, 0, 0), mfe=100.0)
        candle = Candle(datetime(2026, 1, 1, 1, 1), 100, 100.2, 99.8, 100.0, 1)

        self.assertEqual(_mtf_exit_reason_for_position(config, position, candle, {}, {}), "mtf_time_stop")

    def test_report_contains_mtf_dimensions(self) -> None:
        trade = _trade("LONG", "long_mtf_4h_rsi_regime_pullback regime=LONG_BIAS regime_tf=2h rsi_bucket=rsi_38_45 trigger=sweep trigger_tf=5m vol=vol_1p5_2x funding=funding_0_2bps oi=oi_flat_or_down btc=btc_neutral")
        report = mtf_report_from_summary({"trades": [trade]}, {"mtf_stop_too_wide": 2})

        self.assertIn("by_side", report)
        self.assertIn("by_trigger_mode", report)
        self.assertIn("by_regime_timeframe", report)
        self.assertIn("by_trigger_timeframe", report)
        self.assertIn("by_trigger_volume_bucket", report)
        self.assertIn("by_4h_regime", report)
        self.assertEqual(report["rejected_stop_too_wide_count"], 2)

    def test_oi_feature_available_after_bucket_end(self) -> None:
        start = datetime(2026, 1, 1)
        features = build_oi_features([
            {"timestamp": (start + timedelta(minutes=5 * index)).isoformat(), "sumOpenInterestValue": str(100 + index)}
            for index in range(7)
        ])

        self.assertEqual(features[0].available_time, start + timedelta(minutes=5))
        self.assertAlmostEqual(features[6].oi_chg_30m, 106 / 100 - 1.0)


def _config(**kwargs: object):
    base = default_live_config()
    strategy = replace(base.strategy, **kwargs)
    return replace(base, strategy=strategy)


def _strategy(**kwargs: object) -> Mtf4hRsiRegimePullbackStrategy:
    return Mtf4hRsiRegimePullbackStrategy(_config(**kwargs))


def _candle(timestamp: datetime, price: float) -> Candle:
    return Candle(timestamp, price, price + 0.5, price - 0.5, price, 1)


def _trend_candles(start: datetime, timeframe: str, closes: list[float]) -> list[Candle]:
    minutes = 240 if timeframe == "4h" else 15
    candles = []
    for index, close in enumerate(closes):
        timestamp = start + timedelta(minutes=minutes * index)
        open_price = closes[index - 1] if index else close
        high = max(open_price, close) + 0.4
        low = min(open_price, close) - 0.4
        candles.append(Candle(timestamp, open_price, high, low, close, 1))
    return candles


def _flat_15m(start: datetime, count: int, price: float) -> list[Candle]:
    return [Candle(start + timedelta(minutes=15 * index), price, price + 0.5, price - 0.5, price, 1) for index in range(count)]


def _flat_tf(start: datetime, count: int, minutes: int, price: float, volume: float) -> list[Candle]:
    return [
        Candle(start + timedelta(minutes=minutes * index), price, price + 0.5, price - 0.5, price, volume)
        for index in range(count)
    ]


def _position(entry_time: datetime, mfe: float) -> PortfolioPosition:
    return PortfolioPosition(
        "BTCUSDT",
        Direction.LONG,
        1.0,
        100.0,
        99.0,
        102.0,
        0.0,
        0,
        24,
        100.0,
        entry_reason=MTF_REASON_TOKEN,
        entry_time=entry_time,
        raw_entry_price=100.0,
        mfe=mfe,
    )


def _trade(side: str, reason: str) -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "strategy_bucket": "mtf_4h_rsi_regime_pullback",
        "side": side,
        "direction": side,
        "entry_time": "2026-05-20T00:00:00",
        "entry_reason": reason,
        "net_pnl": 1.0,
        "gross_pnl": 1.2,
        "fee": 0.1,
        "slippage_cost": 0.1,
        "funding": 0.0,
        "mfe": 2.0,
        "mae": -0.5,
        "hold_minutes": 30.0,
    }


if __name__ == "__main__":
    unittest.main()

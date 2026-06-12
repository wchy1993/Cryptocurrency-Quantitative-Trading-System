from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta

from crypto_scalper.config import StrategyConfig
from crypto_scalper.models import Candle, Direction
from crypto_scalper.oi_flush_reversal import (
    PendingFlush,
    _confirm_oi_flush_entry,
    _feature_at,
    _price_indicator_cache,
    build_oi_features,
    build_taker_features,
)


class OiFlushReversalTests(unittest.TestCase):
    def test_5m_feature_is_available_only_after_bucket_end(self) -> None:
        start = datetime(2026, 1, 1, 10, 0)
        oi = build_oi_features([
            {"timestamp": (start + timedelta(minutes=5 * i)).isoformat(), "sumOpenInterestValue": str(100 - i)}
            for i in range(7)
        ])
        taker = build_taker_features([
            {
                "timestamp": (start + timedelta(minutes=5 * i)).isoformat(),
                "buyVol": "40",
                "sellVol": "60",
                "buySellRatio": "0.67",
            }
            for i in range(7)
        ])
        series = {
            "oi": oi,
            "oi_available_times": [item.available_time for item in oi],
            "taker": taker,
            "taker_available_times": [item.available_time for item in taker],
            "funding": [],
            "funding_times": [],
        }
        self.assertIsNone(_feature_at(series, start + timedelta(minutes=4), 0.0))
        self.assertIsNotNone(_feature_at(series, start + timedelta(minutes=35), 0.0))

    def test_oi_and_taker_feature_calculation(self) -> None:
        start = datetime(2026, 1, 1, 10, 0)
        oi = build_oi_features([
            {"timestamp": (start + timedelta(minutes=5 * i)).isoformat(), "sumOpenInterestValue": str(100 - i * 2)}
            for i in range(7)
        ])
        self.assertAlmostEqual(oi[3].oi_chg_15m, 94 / 100 - 1.0)
        self.assertAlmostEqual(oi[6].oi_chg_30m, 88 / 100 - 1.0)

        taker = build_taker_features([
            {"timestamp": (start + timedelta(minutes=5 * i)).isoformat(), "buyVol": "25", "sellVol": "75", "buySellRatio": "0.33"}
            for i in range(3)
        ])
        self.assertAlmostEqual(taker[-1].sell_imbalance_5m, 0.75)
        self.assertAlmostEqual(taker[-1].sell_imbalance_15m, 0.75)

    def test_confirmed_signal_is_long_only(self) -> None:
        candles = [
            Candle(datetime(2026, 1, 1, 10, i), 100, 101.5, 99, 100 + i * 0.05, 1)
            for i in range(20)
        ]
        cache = _price_indicator_cache(candles)
        strategy = StrategyConfig()
        config = type("Config", (), {"strategy": strategy})
        pending = PendingFlush(
            symbol="BTCUSDT",
            flush_low=99.0,
            flush_time=candles[10].timestamp,
            flush_index=10,
            atr_pct=0.005,
            atr_value=0.5,
            funding_rate=0.0,
            oi_chg_15m=-0.03,
            oi_chg_30m=-0.05,
            sell_imbalance_15m=0.7,
            btc_ret_15m=0.0,
            btc_state="btc_neutral",
        )
        feature = type("Feature", (), {"sell_imbalance_5m": 0.5})
        signal = _confirm_oi_flush_entry(config, pending, candles[-1], cache, feature, len(candles) - 1)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, Direction.LONG)
        self.assertEqual(signal.reason, "long_oi_flush_reversal")


if __name__ == "__main__":
    unittest.main()

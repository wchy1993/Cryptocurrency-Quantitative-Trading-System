from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from crypto_scalper.live_execution_backtest import (
    VbpCompressionMetrics,
    _vbp_compression_allows,
    _vbp_compression_metrics,
)
from crypto_scalper.models import Candle


def _candles(count: int = 150) -> list[Candle]:
    rows = []
    price = 100.0
    for index in range(count):
        close = price + (0.03 if index % 4 else -0.05)
        volume = 500.0 if index < count - 20 else 250.0
        rows.append(
            Candle(
                datetime(2026, 1, 1) + timedelta(minutes=index),
                price,
                max(price, close) + 0.20,
                min(price, close) - 0.20,
                close,
                volume,
            )
        )
        price = close
    return rows


def _config(**overrides: object) -> SimpleNamespace:
    values = {
        "compression_quality_enabled": True,
        "compression_atr_lookback_bars": 80,
        "compression_atr_percentile_max": 1.0,
        "compression_range_atr_max": 12.0,
        "compression_volume_recent_bars": 12,
        "compression_volume_baseline_bars": 48,
        "compression_volume_contraction_max": 1.0,
        "compression_prior_move_lookback_bars": 30,
        "compression_prior_move_max_atr": 999.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_compression_metrics_exclude_breakout_candle() -> None:
    candles = _candles()
    index = 140
    baseline = _vbp_compression_metrics(candles, index, 101.0, 99.0, _config())
    candles[index] = Candle(candles[index].timestamp, 100.0, 150.0, 50.0, 140.0, 1_000_000.0)

    mutated = _vbp_compression_metrics(candles, index, 101.0, 99.0, _config())

    assert mutated == baseline


def test_disabled_compression_filter_preserves_baseline() -> None:
    metrics = VbpCompressionMetrics(1.0, 100.0, 10.0, 50.0)

    assert _vbp_compression_allows(_config(compression_quality_enabled=False), metrics) == (True, "ok")


def test_compression_filter_reports_specific_rejection() -> None:
    allowed = VbpCompressionMetrics(0.30, 8.0, 0.80, 2.0)
    high_volume = VbpCompressionMetrics(0.30, 8.0, 1.20, 2.0)

    assert _vbp_compression_allows(_config(), allowed) == (True, "ok")
    assert _vbp_compression_allows(_config(), high_volume) == (
        False,
        "reject_compression_volume_contraction",
    )

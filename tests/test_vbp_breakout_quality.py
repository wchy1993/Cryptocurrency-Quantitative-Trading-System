from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from crypto_scalper.live_execution_backtest import (
    VbpBreakoutMetrics,
    _vbp_breakout_metrics,
    _vbp_breakout_quality_allows,
)
from crypto_scalper.models import Candle


def _config(**overrides: object) -> SimpleNamespace:
    values = {
        "breakout_quality_enabled": True,
        "breakout_distance_atr_min": 0.25,
        "breakout_distance_atr_max": 2.0,
        "breakout_body_atr_min": 0.50,
        "breakout_close_position_min": 0.70,
        "breakout_upper_wick_ratio_max": 0.25,
        "breakout_volume_ratio_min": 3.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_breakout_metrics_are_normalized_by_pre_breakout_atr() -> None:
    candle = Candle(datetime(2026, 1, 1), 100.0, 103.0, 99.0, 102.0, 1000.0)

    metrics = _vbp_breakout_metrics(candle, 101.0, 4.0, 2.0)

    assert metrics.distance_atr == 0.5
    assert metrics.body_atr == 1.0
    assert metrics.close_position == 0.75
    assert metrics.upper_wick_ratio == 0.25
    assert metrics.volume_ratio == 4.0


def test_disabled_breakout_gate_preserves_baseline() -> None:
    metrics = VbpBreakoutMetrics(0.0, 0.0, 0.0, 1.0, 0.0)

    assert _vbp_breakout_quality_allows(_config(breakout_quality_enabled=False), metrics) == (True, "ok")


def test_breakout_gate_reports_body_rejection() -> None:
    metrics = VbpBreakoutMetrics(0.5, 0.2, 0.8, 0.1, 4.0)

    assert _vbp_breakout_quality_allows(_config(), metrics) == (False, "reject_breakout_body_atr")

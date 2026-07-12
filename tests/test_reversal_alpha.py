from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from crypto_scalper.live_config import ReversalAlphaConfig
from crypto_scalper.models import Candle, Direction
from crypto_scalper.reversal_alpha import ReversalAlphaEngine


START = datetime(2026, 1, 1)


def _candles(minutes: int, symbol: str, future_mutation_time: datetime | None = None) -> list[Candle]:
    count = 100 * 60 // minutes
    price = 100.0
    rows = []
    for index in range(count):
        timestamp = START + timedelta(minutes=index * minutes)
        if future_mutation_time is not None and timestamp >= future_mutation_time:
            change = 0.08
        elif symbol == "ETHUSDT" and timestamp < START + timedelta(hours=78):
            change = -0.001
        elif symbol == "ETHUSDT":
            change = 0.004
        else:
            change = 0.0001
        close = price * (1.0 + change)
        rows.append(Candle(timestamp, price, max(price, close) * 1.001, min(price, close) * 0.999, close, 1000.0))
        price = close
    return rows


def _frames(future_mutation_time: datetime | None = None) -> dict[str, dict[str, list[Candle]]]:
    return {
        timeframe: {
            symbol: _candles(minutes, symbol, future_mutation_time)
            for symbol in ("BTCUSDT", "ETHUSDT")
        }
        for timeframe, minutes in (("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60))
    }


def _lenient_config() -> ReversalAlphaConfig:
    return replace(
        ReversalAlphaConfig(enabled=True),
        setup_extension_atr_min=0.0,
        setup_long_rsi_extreme_max=100.0,
        setup_long_kdj_extreme_max=100.0,
        setup_require_rsi_recovery=False,
        setup_require_kdj_recovery=False,
        setup_macd_improvement_bars=0,
        setup_require_no_new_extreme=False,
        trigger_require_directional_candle=False,
        trigger_close_position_min=0.0,
        trigger_require_macd_improvement=False,
    )


def test_reversal_shadow_uses_closed_multitimeframe_bars() -> None:
    decision_time = START + timedelta(hours=81)
    original = ReversalAlphaEngine(_lenient_config(), _frames()).evaluate(
        "ETHUSDT", decision_time, Direction.LONG
    )
    mutated = ReversalAlphaEngine(_lenient_config(), _frames(decision_time)).evaluate(
        "ETHUSDT", decision_time, Direction.LONG
    )

    assert original is not None
    assert original == mutated
    assert "setup_extension_atr" in original.features
    assert "trigger_reclaim_ema" in original.features


def test_disabled_reversal_shadow_returns_none() -> None:
    engine = ReversalAlphaEngine(ReversalAlphaConfig(enabled=False), _frames())

    assert engine.evaluate("ETHUSDT", START + timedelta(hours=81), Direction.LONG) is None

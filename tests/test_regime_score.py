from __future__ import annotations

from datetime import datetime, timedelta

from crypto_scalper.live_config import RegimeScoreConfig
from crypto_scalper.models import Candle, Direction
from crypto_scalper.regime_score import RegimeScoreEngine


def _trend_candles(timeframe_minutes: int, future_drop_time: datetime | None = None) -> list[Candle]:
    rows = []
    price = 100.0
    count = 100 * 60 // timeframe_minutes
    for index in range(count):
        timestamp = datetime(2026, 1, 1) + timedelta(minutes=timeframe_minutes * index)
        if future_drop_time is not None and timestamp >= future_drop_time:
            close = price * 0.92
        else:
            close = price * (0.999 if index % 5 == 0 else 1.0008)
        rows.append(
            Candle(
                timestamp,
                price,
                max(price, close) * 1.002,
                min(price, close) * 0.998,
                close,
                1000.0 + index * 10.0,
            )
        )
        price = close
    return rows


def _frames(future_drop_time: datetime | None = None) -> dict[str, dict[str, list[Candle]]]:
    return {
        timeframe: {
            symbol: _trend_candles(minutes, future_drop_time=future_drop_time)
            for symbol in ("BTCUSDT", "ETHUSDT")
        }
        for timeframe, minutes in (("15m", 15), ("30m", 30), ("1h", 60))
    }


def test_uptrend_scores_as_trend_dominant() -> None:
    engine = RegimeScoreEngine(RegimeScoreConfig(enabled=True), _frames())
    decision_time = datetime(2026, 1, 1) + timedelta(hours=81)

    snapshot = engine.score("ETHUSDT", decision_time, Direction.LONG)

    assert snapshot is not None
    assert snapshot.trend_score > snapshot.reversal_score
    assert snapshot.score_gap == snapshot.trend_score - snapshot.reversal_score
    assert snapshot.features["ema21_slope_1h_atr"] > 0


def test_future_candles_do_not_change_historical_score() -> None:
    config = RegimeScoreConfig(enabled=True)
    decision_time = datetime(2026, 1, 1) + timedelta(hours=81)
    original = RegimeScoreEngine(config, _frames()).score("ETHUSDT", decision_time, Direction.LONG)
    future_mutated = RegimeScoreEngine(config, _frames(future_drop_time=decision_time)).score(
        "ETHUSDT",
        decision_time,
        Direction.LONG,
    )

    assert original is not None
    assert future_mutated is not None
    assert future_mutated == original


def test_disabled_engine_returns_no_score() -> None:
    engine = RegimeScoreEngine(RegimeScoreConfig(enabled=False), _frames())

    assert engine.score("ETHUSDT", datetime(2026, 1, 5), Direction.LONG) is None

from __future__ import annotations

from datetime import datetime, timedelta

from crypto_scalper.alpha_diagnostics import AlphaCandidateDiagnostics
from crypto_scalper.models import Candle, Direction, Signal
from crypto_scalper.regime_score import RegimeScoreSnapshot


def _candles(count: int = 140) -> list[Candle]:
    rows = []
    price = 100.0
    for index in range(count):
        close = price + (0.2 if index % 3 else -0.1)
        rows.append(
            Candle(
                datetime(2026, 1, 1) + timedelta(minutes=index),
                price,
                max(price, close) + 0.1,
                min(price, close) - 0.1,
                close,
                1000.0 + index,
            )
        )
        price = close
    return rows


def test_disabled_recorder_has_no_output() -> None:
    recorder = AlphaCandidateDiagnostics(False)
    candles = _candles()
    signal = Signal(Direction.LONG, 0.7, "indicator_long_macd_golden_cross", 0.01, 0.02)

    assert recorder.record_reversal("BTCUSDT", signal, candles, 120) is None
    assert recorder.finalize({"BTCUSDT": candles}, []) == []


def test_records_reversal_and_vbp_features_without_mutating_signal() -> None:
    recorder = AlphaCandidateDiagnostics(True, full_round_trip_cost_pct=0.001, stop_round_trip_cost_pct=0.0012)
    candles = _candles()
    signal = Signal(Direction.LONG, 0.7, "indicator_long_macd_golden_cross", 0.01, 0.02)

    reversal_id = recorder.record_reversal("BTCUSDT", signal, candles, 120)
    vbp_id = recorder.record_vbp_breakout("BTCUSDT", candles, 121, candles[120].high, candles[100].low, 3.2)
    recorder.update_vbp_pullback(vbp_id, candles[122], 1, candles[121].close, candles[121].volume)
    rows = recorder.finalize({"BTCUSDT": candles}, [])

    assert reversal_id is not None
    assert vbp_id is not None
    assert len(rows) == 2
    assert recorder.rows[reversal_id]["raw_signal_reason"] == signal.reason
    assert recorder.rows[reversal_id]["target_to_full_cost_ratio"] == 20.0
    assert recorder.rows[reversal_id]["stop_to_full_cost_ratio"] == 0.01 / 0.0012
    assert recorder.rows[vbp_id]["breakout_volume_ratio"] == 3.2
    assert recorder.rows[vbp_id]["full_round_trip_cost_pct"] == 0.001
    assert "mfe_15m_pct" in recorder.rows[vbp_id]


def test_attaches_shadow_regime_score_without_changing_signal() -> None:
    class ScoreEngine:
        def score(self, symbol: str, decision_time: datetime, direction: Direction) -> RegimeScoreSnapshot:
            return RegimeScoreSnapshot(
                decision_time.isoformat(),
                direction.name,
                82.0,
                31.0,
                51.0,
                "TREND",
                {"trend_ema_slope": 20.0},
                {"ema21_slope_1h_atr": 0.1},
            )

    recorder = AlphaCandidateDiagnostics(True, regime_score_engine=ScoreEngine())
    candles = _candles()
    signal = Signal(Direction.LONG, 0.7, "indicator_long_macd_golden_cross", 0.01, 0.02)

    event_id = recorder.record_reversal("BTCUSDT", signal, candles, 120, decision_time=candles[121].timestamp)

    assert event_id is not None
    assert recorder.rows[event_id]["trend_score"] == 82.0
    assert recorder.rows[event_id]["shadow_regime"] == "TREND"
    assert signal.reason == "indicator_long_macd_golden_cross"

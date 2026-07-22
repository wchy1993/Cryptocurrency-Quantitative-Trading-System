from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

from crypto_scalper.live_config import ReversalAlphaConfig
from crypto_scalper.live_execution_backtest import _reversal_v2_adjust_candidate_for_fill
from crypto_scalper.live_trader import EntryCandidate
from crypto_scalper.models import Candle, Direction, Signal
from crypto_scalper.reversal_alpha import (
    REVERSAL_V2_REASON_TOKEN,
    ReversalAlphaEngine,
    ReversalAlphaSnapshot,
)
from crypto_scalper.risk import BacktestExecutionConfig


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


def test_execution_decision_uses_structural_risk_and_event_id() -> None:
    config = replace(ReversalAlphaConfig(enabled=True), shadow_mode=False, take_profit_r=1.8)
    engine = ReversalAlphaEngine(config, _frames())
    candle = Candle(START, 100.0, 102.0, 99.0, 101.0, 1000.0)
    snapshot = ReversalAlphaSnapshot(
        setup_pass=True,
        trigger_pass=True,
        eligible=True,
        reject_reasons=(),
        features={"stop_pct": 0.01},
        event_id="ETHUSDT-long-test",
        trigger_candle=candle,
        structural_stop_price=99.0,
        setup_atr=2.0,
        quality_score=0.8,
    )

    decision = engine.decision_from_snapshot(snapshot, Direction.LONG)

    assert decision is not None
    assert REVERSAL_V2_REASON_TOKEN in decision.signal.reason
    assert "event_id=ETHUSDT-long-test" in decision.signal.reason
    assert decision.signal.stop_loss_pct == 0.01
    assert abs(decision.signal.take_profit_pct - 0.018) < 1e-12


def test_fill_guard_recomputes_stop_and_rejects_insufficient_cost_edge() -> None:
    alpha = replace(
        ReversalAlphaConfig(enabled=True, shadow_mode=False),
        min_stop_atr=0.25,
        max_stop_atr=3.0,
        min_stop_pct=0.001,
        max_stop_pct=0.05,
        take_profit_r=1.5,
        min_target_to_cost_ratio=3.0,
    )
    config = SimpleNamespace(reversal_alpha=alpha)
    candle = Candle(START, 100.0, 101.0, 99.0, 100.0, 1000.0)
    candidate = EntryCandidate(
        "ETHUSDT",
        Signal(Direction.LONG, 0.7, f"{REVERSAL_V2_REASON_TOKEN}_long", 0.01, 0.015),
        candle,
        80.0,
        1.0,
        1.2,
        "test",
        metadata={"setup_atr": 2.0, "trigger_close": 100.0, "structural_stop_price": 98.0},
    )
    costs = BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0, take_profit_slippage_bps=2.0)

    adjusted = _reversal_v2_adjust_candidate_for_fill(config, candidate, 100.5, costs, {})

    assert adjusted is not None
    assert abs(adjusted.signal.stop_loss_pct - (2.5 / 100.5)) < 1e-12
    expensive = SimpleNamespace(reversal_alpha=replace(alpha, min_target_to_cost_ratio=100.0))
    assert _reversal_v2_adjust_candidate_for_fill(expensive, candidate, 100.5, costs, {}) is None

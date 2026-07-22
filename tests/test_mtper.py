from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.live_config import default_live_config, load_live_config, write_live_config
from crypto_scalper.live_execution_backtest import (
    _candidate_signal_times,
    _manage_mtper_position_1m,
    _mtper_adjust_candidate_for_fill,
    _mtper_trade_risk_budget,
)
from crypto_scalper.live_portfolio_backtest import PortfolioPosition
from crypto_scalper.live_trader import EntryCandidate
from crypto_scalper.models import Candle, Direction, Signal
from crypto_scalper.mtper import MTPER_REASON_TOKEN, MtperEngine, MtperOrderLifecycle, MtperState
from crypto_scalper.risk import BacktestExecutionConfig, BacktestExecutionStats


def _candles(start: datetime, count: int, minutes: int, slope: float = 0.0005) -> list[Candle]:
    output = []
    price = 100.0
    for index in range(count):
        open_price = price
        close = open_price * (1.0 + slope)
        output.append(
            Candle(
                start + timedelta(minutes=minutes * index),
                open_price,
                max(open_price, close) * 1.002,
                min(open_price, close) * 0.998,
                close,
                1000.0 + index,
            )
        )
        price = close
    return output


def _config():
    base = default_live_config()
    symbols = ("BTCUSDT", "ETHUSDT")
    mtper = replace(base.mtper, enabled=True, enabled_symbols=symbols)
    return replace(
        base,
        trading=replace(base.trading, symbols=symbols, entry_symbols=symbols, leverage=5),
        risk=replace(base.risk, risk_per_trade_pct=mtper.risk_control.campaign_risk_pct),
        mtper=mtper,
    )


def _engine(config=None) -> MtperEngine:
    config = config or _config()
    start = datetime(2025, 1, 1)
    candles = {
        timeframe: {
            symbol: _candles(start, count, minutes)
            for symbol in config.trading.symbols
        }
        for timeframe, count, minutes in (
            ("15m", 500, 15),
            ("30m", 400, 30),
            ("1h", 300, 60),
            ("2h", 200, 120),
            ("4h", 160, 240),
        )
    }
    return MtperEngine(config, candles)


def test_mtper_config_round_trip(tmp_path) -> None:
    config = _config()
    path = tmp_path / "mtper.json"
    write_live_config(path, config)
    loaded = load_live_config(path)
    assert loaded.mtper.enabled
    assert loaded.mtper.enabled_symbols == config.mtper.enabled_symbols
    assert loaded.mtper.extreme.range_lookbacks == (20, 50, 100)


def test_closed_high_timeframe_uses_only_fully_closed_candles() -> None:
    engine = _engine()
    symbol = "BTCUSDT"
    start = datetime(2025, 1, 1)
    at_close = engine._closed("4h", symbol, start + timedelta(hours=8), 10)
    before_close = engine._closed("4h", symbol, start + timedelta(hours=7, minutes=59), 10)
    assert [row.timestamp for row in at_close] == [start, start + timedelta(hours=4)]
    assert [row.timestamp for row in before_close] == [start]


def test_order_lifecycle_requires_protection_after_fill() -> None:
    lifecycle = MtperOrderLifecycle("BTCUSDT")
    lifecycle.submit_entry(2.0)
    lifecycle.record_fill(1.0)
    assert lifecycle.state == MtperState.PARTIAL_FILL
    lifecycle.record_fill(1.0)
    assert lifecycle.state == MtperState.PROTECTION_PENDING
    assert lifecycle.protection_result(False) == "emergency_reduce_or_close"
    assert lifecycle.state == MtperState.EXITING


def test_restart_recovery_never_leaves_unprotected_position_as_protected() -> None:
    lifecycle = MtperOrderLifecycle("BTCUSDT")
    action = lifecycle.recover_after_restart(1.0, protective_stop_present=False)
    assert action == "replace_protection_or_emergency_flatten"
    assert lifecycle.state == MtperState.PROTECTION_PENDING


def test_cancelled_pending_cannot_remain_stuck_in_order_pending() -> None:
    engine = _engine()
    runtime = engine.runtime["BTCUSDT"]
    runtime.state = MtperState.INITIAL_ENTRY_READY
    runtime.setup_id = "setup:test"
    engine.mark_order_pending("BTCUSDT")
    assert runtime.state == MtperState.ORDER_PENDING
    engine.mark_cancelled("BTCUSDT", datetime(2026, 1, 1), "missed_next_1m_open")
    assert runtime.state == MtperState.COOLDOWN
    assert runtime.setup_id is None
    assert engine.stats["cancel_pending_count"] == 1


def test_mtper_signal_availability_uses_trigger_timeframe_close() -> None:
    config = _config()
    candle = Candle(datetime(2026, 1, 1), 100.0, 101.0, 99.0, 100.5, 1000.0)
    signal = Signal(Direction.LONG, 0.55, MTPER_REASON_TOKEN, 0.02, 0.03, 0.5)
    candidate = EntryCandidate(
        "BTCUSDT",
        signal,
        candle,
        0.7,
        0.0,
        1.0,
        "ok",
        {"trigger_timeframe": "15m"},
    )
    signal_time, available = _candidate_signal_times(config, candidate, candle.timestamp)
    assert signal_time == candle.timestamp
    assert available == candle.timestamp + timedelta(minutes=15)


def test_fill_guard_rejects_when_liquidation_would_precede_hard_stop() -> None:
    config = _config()
    candle = Candle(datetime(2026, 1, 1), 100.0, 101.0, 99.0, 100.0, 1000.0)
    signal = Signal(Direction.LONG, 0.55, MTPER_REASON_TOKEN, 0.30, 0.45, 0.5)
    candidate = EntryCandidate(
        "BTCUSDT",
        signal,
        candle,
        0.7,
        0.0,
        1.0,
        "ok",
        {
            "trigger_atr_15m": 2.0,
            "trigger_close": 100.0,
            "structural_stop_price": 70.0,
            "target_1_price": 115.0,
            "target_2_price": 130.0,
            "target_3_price": 145.0,
        },
    )
    stats = {}
    adjusted = _mtper_adjust_candidate_for_fill(
        config,
        candidate,
        100.0,
        BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0, stop_slippage_bps=5.0),
        stats,
    )
    assert adjusted is None
    assert stats["reject_fill_stop_too_wide"] == 1 or stats["reject_liquidation_before_hard_stop"] == 1


def test_campaign_risk_budget_uses_mtper_hard_cap() -> None:
    config = _config()
    config = replace(
        config,
        mtper=replace(
            config.mtper,
            risk_control=replace(config.mtper.risk_control, campaign_risk_pct=0.03, max_campaign_risk_pct=0.02),
        ),
    )
    assert _mtper_trade_risk_budget(config, 160.0) == 3.2


def test_same_bar_stop_and_target_uses_adverse_stop_first() -> None:
    config = _config()
    position = PortfolioPosition(
        "BTCUSDT",
        Direction.LONG,
        1.0,
        100.0,
        98.0,
        106.0,
        0.05,
        0,
        1000,
        100.0,
        entry_time=datetime(2026, 1, 1),
        raw_entry_price=100.0,
        initial_stop_price=98.0,
        campaign_risk_budget_usdt=3.0,
        initial_leg_full_cost_risk_usdt=2.2,
        strategy_metadata={
            "campaign_id": "campaign:test",
            "target_1_price": 102.0,
            "target_2_price": 104.0,
            "target_3_price": 106.0,
            "initial_quantity": 1.0,
        },
    )
    positions = {"BTCUSDT": position}
    trades = []
    stats = {}
    execution_stats = BacktestExecutionStats()
    trader = SimpleNamespace(
        client=SimpleNamespace(
            symbol_rules=lambda symbol: SymbolRules(symbol, "0.1", "0.001", "0.001", "5")
        )
    )
    candle = Candle(datetime(2026, 1, 1, 0, 1), 100.0, 103.0, 97.0, 101.0, 1000.0)
    _, closed = _manage_mtper_position_1m(
        trader,
        config,
        160.0,
        positions,
        trades,
        position,
        candle,
        1,
        BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0, stop_slippage_bps=5.0),
        execution_stats,
        {},
        {},
        stats,
    )
    assert closed
    assert trades[0]["exit_reason"] == "mtper_hard_stop_1m"
    assert execution_stats.same_bar_tp_sl_conflict_count == 1

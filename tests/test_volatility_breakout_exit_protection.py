from __future__ import annotations

from array import array
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.models import Direction
from crypto_scalper.risk import BacktestExecutionConfig
from crypto_scalper.volatility_breakout import DualThrustSignal, VolatilityBreakoutConfig
from crypto_scalper.volatility_breakout_exit_protection import (
    ExitProtectionConfig,
    simulate_exit_protected_portfolio,
)
from crypto_scalper.volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    PortfolioSearchConfig,
    minute_token,
    simulate_portfolio,
)


def _rules() -> SymbolRules:
    return SymbolRules(
        "TESTUSDT",
        Decimal("0.1"),
        Decimal("0.1"),
        Decimal("0.1"),
        Decimal("5"),
    )


def _signal(start: datetime) -> DualThrustSignal:
    return DualThrustSignal(
        event_id="protected-event",
        symbol="TESTUSDT",
        direction=Direction.LONG,
        signal_bar_time=start - timedelta(hours=1),
        signal_available_time=start,
        raw_signal_price=100.0,
        session_open=100.0,
        upper_band=99.0,
        lower_band=90.0,
        dual_thrust_range=10.0,
        atr_value=5.0,
        trend_alignment_atr=1.0,
        volume_ratio=2.0,
        body_atr=1.0,
        directional_close_position=0.9,
        range_atr=2.0,
        breakout_extension_atr=0.2,
        quality_score=2.0,
    )


def _series(start: datetime, rows: list[tuple[float, float, float, float]]) -> CompactSeries:
    return CompactSeries(
        minutes=array("q", [minute_token(start + timedelta(minutes=i)) for i in range(len(rows))]),
        opens=array("d", [row[0] for row in rows]),
        highs=array("d", [row[1] for row in rows]),
        lows=array("d", [row[2] for row in rows]),
        closes=array("d", [row[3] for row in rows]),
        volumes=array("d", [100_000.0] * len(rows)),
    )


def _run(
    rows: list[tuple[float, float, float, float]],
    exit_config: ExitProtectionConfig,
) -> dict:
    start = datetime(2026, 1, 1)
    minute = minute_token(start)
    signal_config = VolatilityBreakoutConfig(
        stop_atr_multiple=1.0,
        take_profit_r=20.0,
        max_holding_minutes=60,
    )
    return simulate_exit_protected_portfolio(
        {minute: [Candidate(_signal(start), minute)]},
        ("TESTUSDT",),
        {"TESTUSDT": _series(start, rows)},
        {"TESTUSDT": _rules()},
        signal_config,
        PortfolioSearchConfig(risk_per_trade_pct=0.02),
        exit_config,
        BacktestExecutionConfig(
            mode="conservative",
            market_slippage_bps=0.0,
            stop_slippage_bps=0.0,
            take_profit_slippage_bps=0.0,
            taker_fee_rate=0.0,
        ),
        start,
        start + timedelta(minutes=len(rows)),
        200.0,
    )


def test_disabled_exit_overlay_matches_original_v2_path() -> None:
    rows = [
        (100.0, 103.0, 99.0, 102.0),
        (102.0, 104.0, 94.0, 95.0),
        (95.0, 96.0, 95.0, 95.0),
    ]
    start = datetime(2026, 1, 1)
    minute = minute_token(start)
    signal = _signal(start)
    series = _series(start, rows)
    signal_config = VolatilityBreakoutConfig(
        stop_atr_multiple=1.0,
        take_profit_r=20.0,
        max_holding_minutes=60,
    )
    portfolio = PortfolioSearchConfig(risk_per_trade_pct=0.02)
    execution = BacktestExecutionConfig(
        mode="conservative",
        market_slippage_bps=0.0,
        stop_slippage_bps=0.0,
        take_profit_slippage_bps=0.0,
        taker_fee_rate=0.0,
    )
    original = simulate_portfolio(
        {minute: [Candidate(signal, minute)]},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": _rules()},
        signal_config,
        portfolio,
        execution,
        start,
        start + timedelta(minutes=len(rows)),
        200.0,
    )
    protected = simulate_exit_protected_portfolio(
        {minute: [Candidate(signal, minute)]},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": _rules()},
        signal_config,
        portfolio,
        ExitProtectionConfig(),
        execution,
        start,
        start + timedelta(minutes=len(rows)),
        200.0,
    )

    assert protected["final_equity"] == original["final_equity"]
    assert protected["trades"][0]["exit_reason"] == original["trades"][0]["exit_reason"]
    assert protected["trades"][0]["net_pnl"] == original["trades"][0]["net_pnl"]


def test_breakeven_activates_after_profitable_bar_and_applies_next_bar() -> None:
    result = _run(
        [
            (100.0, 106.0, 99.0, 105.0),
            (105.0, 105.0, 99.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ],
        ExitProtectionConfig(breakeven_trigger_r=1.0),
    )

    trade = result["trades"][0]
    assert trade["exit_reason"] == "breakeven_stop"
    assert trade["net_pnl"] == pytest.approx(0.0)


def test_profit_giveback_locks_positive_r_after_activation() -> None:
    result = _run(
        [
            (100.0, 107.0, 99.0, 106.0),
            (106.0, 106.0, 103.0, 104.0),
            (104.0, 104.0, 104.0, 104.0),
        ],
        ExitProtectionConfig(
            profit_giveback_activation_r=1.0,
            profit_giveback_r=0.5,
        ),
    )

    trade = result["trades"][0]
    assert trade["exit_reason"] == "profit_giveback_stop"
    assert trade["pnl_r"] == pytest.approx(0.9)


def test_profit_floor_locks_fixed_r_and_leaves_room_before_activation() -> None:
    result = _run(
        [
            (100.0, 111.0, 99.0, 110.0),
            (110.0, 110.0, 104.0, 105.0),
            (105.0, 105.0, 105.0, 105.0),
        ],
        ExitProtectionConfig(
            profit_floor_1_activation_r=2.0,
            profit_floor_1_lock_r=1.0,
        ),
    )

    trade = result["trades"][0]
    assert trade["exit_reason"] == "profit_floor_1_stop"
    assert trade["pnl_r"] == pytest.approx(1.0)


def test_profit_floor_validation_rejects_gaps_and_falling_locks() -> None:
    with pytest.raises(ValueError):
        ExitProtectionConfig(
            profit_floor_2_activation_r=4.0,
            profit_floor_2_lock_r=1.0,
        ).validate()
    with pytest.raises(ValueError):
        ExitProtectionConfig(
            profit_floor_1_activation_r=2.0,
            profit_floor_1_lock_r=1.0,
            profit_floor_2_activation_r=4.0,
            profit_floor_2_lock_r=0.5,
        ).validate()


def test_partial_take_profit_aggregates_legs_and_reconciles_cash() -> None:
    result = _run(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 106.0, 99.0, 105.0),
            (105.0, 105.0, 94.0, 95.0),
        ],
        ExitProtectionConfig(
            partial_take_profit_r=1.0,
            partial_take_profit_fraction=0.5,
        ),
    )

    trade = result["trades"][0]
    assert trade["partial_exit_count"] == 1
    assert trade["partial_realized_net_pnl"] == pytest.approx(2.0)
    assert trade["net_pnl"] == pytest.approx(0.0)
    assert result["cash_reconciliation_error"] == pytest.approx(0.0)


def test_same_bar_stop_wins_over_partial_take_profit() -> None:
    result = _run(
        [
            (100.0, 106.0, 94.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ],
        ExitProtectionConfig(
            partial_take_profit_r=1.0,
            partial_take_profit_fraction=0.5,
        ),
    )

    trade = result["trades"][0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["partial_exit_count"] == 0
    assert trade["net_pnl"] < 0.0
